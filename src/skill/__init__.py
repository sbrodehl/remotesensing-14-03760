# Copyright (C) 2018-2022  Sebastian Brodehl

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


class Skill:
    """Computes `skill` scores: `CSI`, `FAR` and `POD`.

    Assume channels first.
    Only works on 2D data.
    """

    csi = None
    far = None
    pod = None

    def __init__(
        self,
        threshold: float = 0.5,
        pred_class: int = 1,
        search_region_radius_px: int = 3,
        bt_diff: int = -1,
    ):
        self.bt_diff = bt_diff
        self.threshold = threshold
        self.pred_class = pred_class
        self.search_region_px = 2 * search_region_radius_px + 1
        self.pad_pixels = search_region_radius_px
        self.conv_weights = torch.from_numpy(
            self.generate_disc_kernel(self.search_region_px)
        )

    def __call__(  # noqa: C901 too-complex pylint: disable=too-many-statements,E1136
        self,
        state,
        event,
        *args,
        logits=None,
        predictions=None,
        targets=None,
        weighting=None,
        mask=None,
        quiet: bool = False,
        threshold: Union[None, float] = None,
        keep_in_memory: bool = True,
        search_region_radius_px: int = None,
        reduction=torch.mean,
        reduce_non_nan: bool = False,
        **kwargs,
    ):
        """

        Args:
            mask (object): Will mask classifications after shenanigans, and thus, ghost events might occur due to out of bounds influence.
            weighting (object): Will weigh (raw) predictions, before classification takes place.
        """
        if search_region_radius_px is not None:
            self.search_region_px = 2 * search_region_radius_px + 1
            self.pad_pixels = search_region_radius_px
            self.conv_weights = torch.from_numpy(
                self.generate_disc_kernel(self.search_region_px)
            )
        with torch.no_grad():
            _threshold = (
                state["EMA_threshold"].mean.item()
                if "EMA_threshold" in state and state["EMA_threshold"] is not None
                else self.threshold
            )
            _threshold = threshold if threshold is not None else _threshold
            classes = state[f"num_{state['PIPES']['TARGET'].name}_classes"] + 1
            logits = state["logits"] if logits is None and "logits" in state else logits
            targets = (
                state["sample"][state["PIPES"]["TARGET"].name]
                if targets is None and "sample" in state
                else targets
            )
            weighting = (
                weighting
                if weighting is not None
                else (state["mask"] if "mask" in state else None)
            )
            if weighting is not None and weighting.ndim == 5:
                weighting = torch.squeeze(weighting, -3)
            if logits is None and predictions is None:
                return
            if predictions is None:
                probabilities = torch.nn.Softmax(dim=1)(logits)
                if weighting is not None:
                    probabilities *= weighting.expand(probabilities.shape)
                predictions = probabilities >= _threshold
            predictions = predictions.to(
                probabilities.dtype
                if locals().get("probabilities", None) is not None
                else logits.dtype
                if locals().get("logits", None) is not None
                else torch.float32
            )
            targets = torch.squeeze(event.to_one_hot_vector(targets, classes), -3)
            targets_search_region = self._convolve_tensor(targets).to(torch.bool)
            predictions_search_region = self._convolve_tensor(predictions).to(
                torch.bool
            )
            predictions = predictions.to(torch.bool)
            targets = targets.to(torch.bool)
            tp = targets_search_region & predictions
            fn = targets & ~predictions_search_region
            fp = ~targets_search_region & predictions
            tn = ~predictions & ~targets_search_region
            if mask is not None:
                m_dtype = torch.int8
                tp = torch.where(
                    mask.expand_as(tp),
                    tp.to(m_dtype),
                    torch.tensor(0, dtype=m_dtype).expand_as(tp),
                ).to(torch.bool)
                fn = torch.where(
                    mask.expand_as(fn),
                    fn.to(m_dtype),
                    torch.tensor(0, dtype=m_dtype).expand_as(fn),
                ).to(torch.bool)
                fp = torch.where(
                    mask.expand_as(fp),
                    fp.to(m_dtype),
                    torch.tensor(0, dtype=m_dtype).expand_as(fp),
                ).to(torch.bool)
                tn = torch.where(
                    mask.expand_as(tn),
                    tn.to(m_dtype),
                    torch.tensor(0, dtype=m_dtype).expand_as(tn),
                ).to(torch.bool)
            if keep_in_memory or not quiet:
                det_cd = tp.detach().clone()
                det_md = fn.detach().clone()
                det_fd = fp.detach().clone()
                det_cdn = tn.detach().clone()
            if not quiet and event.plot_every():
                event.optional.plot_imgs(
                    "Skill/Detections",
                    torch.cat(
                        [
                            det_fd[: state["batched_data_no"], -1:],
                            det_cd[: state["batched_data_no"], -1:],
                            det_md[: state["batched_data_no"], -1:],
                        ],
                        dim=1,
                    ).to(torch.float32),
                )
            if keep_in_memory:
                state["cd"] = det_cd
                state["md"] = det_md
                state["fd"] = det_fd
                state["cdn"] = det_cdn
            tp = torch.sum(tp, dim=tuple(range(2, len(tp.shape))))
            fn = torch.sum(fn, dim=tuple(range(2, len(fn.shape))))
            fp = torch.sum(fp, dim=tuple(range(2, len(fp.shape))))
            tn = torch.sum(tn, dim=tuple(range(2, len(tn.shape))))
            _csi_divisor = tp + fn + fp
            _csi_defined = _csi_divisor > 0
            csi = torch.div(tp, _csi_divisor)
            if not reduce_non_nan:
                csi[_csi_defined == 0] = 1.0
            _far_divisor = tp + fp
            _far_defined = _far_divisor > 0
            far = torch.div(fp, (tp + fp))
            if not reduce_non_nan:
                far[_far_defined == 0] = 0.0
            _pod_divisor = tp + fn
            _pod_defined = _pod_divisor > 0
            pod = torch.div(tp, (tp + fn))
            if not reduce_non_nan:
                pod[_pod_defined == 0] = 1.0
            _acc_divisor = tp + fn + fp + tn
            _acc_defined = _acc_divisor > 0
            acc = torch.div((tp + tn), (tp + fn + fp + tn))
            if not reduce_non_nan:
                acc[_acc_defined == 0] = 1.0
            if reduction is not None:
                csi = reduction(csi, 0)
                far = reduction(far, 0)
                pod = reduction(pod, 0)
                acc = reduction(acc, 0)
            if kwargs.get("internal", False):
                self.csi = csi.detach().clone()
                self.far = far.detach().clone()
                self.pod = pod.detach().clone()
            if not quiet and event.plot_every():
                event.optional.plot_scalar(
                    "Skill/mCSI", csi.detach().clone().tolist()[1]
                )
                event.optional.plot_scalar(
                    "Skill/mFAR", far.detach().clone().tolist()[1]
                )
                event.optional.plot_scalar(
                    "Skill/mPOD", pod.detach().clone().tolist()[1]
                )
                if not state["log.slim"]:
                    event.optional.plot_scalar(
                        "Skill/mACC", acc.detach().clone().tolist()[1]
                    )
                    event.optional.plot_scalar(
                        "Skill/undefCSI", torch.sum(~_csi_defined)
                    )
                    event.optional.plot_scalar(
                        "Skill/undefFAR", torch.sum(~_far_defined)
                    )
                    event.optional.plot_scalar(
                        "Skill/undefPOD", torch.sum(~_pod_defined)
                    )
                    event.optional.plot_scalar(
                        "Skill/TP", tp.detach().clone()[:, 1].sum().cpu().item()
                    )
                    event.optional.plot_scalar(
                        "Skill/TN", tn.detach().clone()[:, 1].sum().cpu().item()
                    )
                    event.optional.plot_scalar(
                        "Skill/FP", fp.detach().clone()[:, 1].sum().cpu().item()
                    )
                    event.optional.plot_scalar(
                        "Skill/FN", fn.detach().clone()[:, 1].sum().cpu().item()
                    )

    def evaluate(self, state, event, *args, **kwargs):
        self.__call__(state, event, *args, internal=True, **kwargs)
        return self.csi, self.far, self.pod

    def _convolve_tensor(self, tensor: torch.Tensor, booleanize: bool = True):
        previous_dtype = tensor.dtype
        is_fp_tensor = torch.is_floating_point(tensor)
        if not is_fp_tensor:
            tensor = tensor.to(torch.float32)
        nb_channels = tensor.shape[1]
        kernel = (
            self.conv_weights.view(1, 1, self.search_region_px, self.search_region_px)
            .repeat(
                nb_channels,
                1,
                1,
                1,
            )
            .to(tensor)
        )
        tensor = F.pad(
            tensor,
            tuple([self.pad_pixels] * (2 * (len(tensor.shape) - 2))),
            mode="constant",
            value=0,
        )
        with torch.no_grad():
            conv = torch.nn.Conv2d(
                nb_channels,
                nb_channels,
                self.search_region_px,
                groups=nb_channels,
                bias=False,
            )
            conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
            output = conv(tensor)
        tensor = self._to_single_label_multi_class(output)
        if booleanize:
            tensor = tensor.to(torch.bool).to(tensor.dtype)
        if not is_fp_tensor:
            tensor = tensor.to(previous_dtype)
        return tensor

    @staticmethod
    def _to_single_label_multi_class(tensor):
        ch_idx = 1
        aggr_ = [tensor[:, -1]]
        if tensor.shape[ch_idx] > 1:
            for i_ in reversed(range(tensor.shape[ch_idx] - 1)):
                aggr_.append(
                    torch.mul(
                        tensor[:, i_],
                        (~(tensor[:, i_ + 1].to(torch.bool))).to(tensor.dtype),
                    )
                )
        return torch.stack(aggr_[::-1], dim=1).to(tensor)

    @staticmethod
    def generate_disc_kernel(size: int) -> np.array:
        a, b = int(size / 2), int(size / 2)
        r = int(size / 2)
        y, x = np.ogrid[-a : size - a, -b : size - b]
        mask = x * x + y * y <= r * r
        array = np.zeros((size, size), dtype=np.longlong)
        array[mask] = 1
        return array


def register(mf):
    mf.register_event("Skill", Skill().evaluate)
    mf.register_event("after_step", Skill(), unique=False)
