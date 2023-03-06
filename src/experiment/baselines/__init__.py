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

import pickle
from pathlib import Path
import logging
import copy
import hashlib
import json
import lz4.frame
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

DETAILS_TEMPLATE = {
    "csi": [[]],
    "far": [[]],
    "pod": [[]],
    "tp": [[]],
    "fn": [[]],
    "fp": [[]],
    "linets": [[]],
}


def dummy_step():
    pass


def dump_sample(state, comps, z, name):
    for (
        lat,
        lon,
        year,
        month,
        day,
        hour,
        minute,
        targets,
        in_ch,
        pers_pred,
        pers_skill,
        m_logits,
        m_pred,
        m_skill,
    ) in z:
        year = int(year[year != 9999][0].item())
        month = int(month[month != 9999][0].item())
        day = int(day[day != 9999][0].item())
        hour = int(hour[hour != 9999][0].item())
        minute = int(minute[minute != 9999][0].item())
        tag = f"{year:04}{month:02}{day:02}{hour:02}{minute:02}"
        dt_c = comps.get(tag, [])
        dt_c.append(
            {
                "LAT": lat.to(torch.float32),
                "LON": lon.to(torch.float32),
                "TARGETS": targets,
                "CHANNELS": in_ch,
                "PERSISTENCE_PREDICTIONS": pers_pred.to(torch.float32),
                "PERSISTENCE_SKILL": pers_skill.to(torch.int8),
                "MODEL_LOGITS": m_logits.to(torch.float32)
                if m_logits is not None
                else torch.zeros_like(pers_pred, dtype=torch.float32) * float("nan"),
                "MODEL_PREDICTIONS": m_pred.to(torch.float32)
                if m_pred is not None
                else torch.zeros_like(pers_pred, dtype=torch.float32) * float("nan"),
                "MODEL_SKILL": m_skill.to(torch.int8)
                if m_skill is not None
                else torch.zeros_like(pers_skill, dtype=torch.float32) * float("nan"),
            }
        )
        comps[tag] = dt_c
        if np.array(state["composite_blocks"]).prod() == len(comps[tag]):
            suc = stitch_composite(state, tag, comps[tag], name=name)
            if suc:
                del comps[tag]


def dump_details_json(state):
    compute_details_stats(state, "persistence_details")
    with open(
        Path(state["log.dir"]) / "validation_persistence_details.json",
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(state["persistence_details"], fh)
    if "enable_testing" in state and state["enable_testing"]:
        compute_details_stats(state, "persistence_testing_details")
        with open(
            Path(state["log.dir"]) / "testing_persistence_details.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(state["persistence_testing_details"], fh)
    if (
        "model" in state
        and "network_pt" in state
        and state["network_pt"]
        and len([pt for pt in state["network_pt"] if len(pt) > 0]) > 0
        and len(state["model_csi"]) > 0
    ):
        compute_details_stats(state, "model_details")
        with open(
            Path(state["log.dir"]) / "validation_model_details.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(state["model_details"], fh)
        if "enable_testing" in state and state["enable_testing"]:
            compute_details_stats(state, "model_testing_details")
            with open(
                Path(state["log.dir"]) / "testing_model_details.json",
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(state["model_testing_details"], fh)


def accumulate_models(state):
    if (
        "model" in state
        and "network_pt" in state
        and state["network_pt"]
        and len([pt for pt in state["network_pt"] if len(pt) > 0]) > 0
        and len(state["model_csi"]) > 0
    ):
        LOGGER.info(
            f"Avg. Batch CSI Validation (Multi-Model): {np.nanmean(state['m_model_csi'], axis=0)[1]:.2f} +/- {np.nanstd(state['m_model_csi'], axis=0)[1]:.2f}"
        )
        LOGGER.info(
            f"Complete CSI Validation (Multi-Model): {np.nanmean(state['model_csi']):.2f} +/- {np.nanstd(state['model_csi']):.2f}"
        )
        if "enable_testing" in state and state["enable_testing"]:
            LOGGER.info(
                f"Avg. Batch CSI Testing (Multi-Model): {np.nanmean(state['testing_m_model_csi'], axis=0)[1]:.2f} +/- {np.nanstd(state['testing_m_model_csi'], axis=0)[1]:.2f}"
            )
            LOGGER.info(
                f"Complete CSI Testing (Multi-Model): {np.nanmean(state['testing_model_csi']):.2f} +/- {np.nanstd(state['testing_model_csi']):.2f}"
            )


def main(state, event):  # noqa: C901 too-complex pylint: disable=too-many-statements
    refresh_cache = state["refresh_cache"]
    state["refresh_cache"] = False
    Path(state["log.dir"]).mkdir(parents=True, exist_ok=True)
    with state.temporary(
        {
            "batchsize": 4,
            "batchsize_val": 4,
            "batchsize_test": 4,
            "disable_multithreading": True,
            "drop_last": False,
            "refresh_cache": refresh_cache,
        }
    ):
        cache_loader = event.dataloader(subset="val", use_cache=False)
        ds_hash_fp = Path(state["log.dir"]).parent / cache_loader.dataset.get_hash()
        if not ds_hash_fp.exists():
            LOGGER.info(f"{ds_hash_fp} not available, refreshing cache.")
            with torch.no_grad():
                tqdm_batch = tqdm(
                    desc="Caching Validation Batches", **state["tqdm_kwargs"]
                )
                for _ in cache_loader:
                    tqdm_batch.update()
                tqdm_batch.close()
            ds_hash_fp.touch()
        if "enable_testing" in state and state["enable_testing"]:
            cache_loader = event.dataloader(subset="test", use_cache=False)
            ds_hash_fp = Path(state["log.dir"]).parent / cache_loader.dataset.get_hash()
            if not ds_hash_fp.exists():
                LOGGER.info(f"{ds_hash_fp} not available, refreshing cache.")
                with torch.no_grad():
                    tqdm_batch = tqdm(
                        desc="Caching Test Batches", **state["tqdm_kwargs"]
                    )
                    for _ in cache_loader:
                        tqdm_batch.update()
                    tqdm_batch.close()
                ds_hash_fp.touch()
    (
        state["persistence_csi"],
        state["m_per_csi"],
        state["m_per_far"],
        state["m_per_pod"],
    ) = ([], [], [], [])
    state["persistence_details"], state["persistence_testing_details"] = {}, {}
    (
        state["testing_persistence_csi"],
        state["testing_m_per_csi"],
        state["testing_m_per_far"],
        state["testing_m_per_pod"],
    ) = ([], [], [], [])
    state["is_training_warmup"] = False
    state["network_pt"] = [pt for pt in state["network_pt"] if len(pt) > 0]
    if "network_pt" in state and state["network_pt"] and len(state["network_pt"]) > 0:
        state["model_details"], state["model_testing_details"] = {}, {}
        (
            state["model_csi"],
            state["m_model_csi"],
            state["m_model_far"],
            state["m_model_pod"],
        ) = ([], [], [], [])
        (
            state["testing_model_csi"],
            state["testing_m_model_csi"],
            state["testing_m_model_far"],
            state["testing_m_model_pod"],
        ) = ([], [], [], [])
        pt_idx = len(state["network_pt"])
        for pt in state["network_pt"]:
            pt_idx -= 1
            state["network_pt_hash"] = hashlib.sha512(
                str(pt).encode("utf-8")
            ).hexdigest()
            checkpoint = torch.load(pt)
            state["Model_Threshold"] = checkpoint["Model_Threshold"]
            assert state["Model_Threshold"] is not None, "NN model threshold unknown."
            LOGGER.info(f"Setting model threshold as '{state['Model_Threshold']:.5f}'.")
            state["model"] = event.init_net()
            assert (
                f"{state['model'].__class__.__name__}" in checkpoint
            ), "Model class not found in checkpoint."
            LOGGER.info(f"Loading '{state['model'].__class__.__name__}': {pt}")
            state["model"].load_state_dict(
                checkpoint[f"{state['model'].__class__.__name__}"]
            )
            state["model"] = event.optional.to_device(state["model"], altfn=lambda m: m)
            state["model"].eval()
            if pt_idx != 0:
                event.step()
    event.step()


def compute_skill(
    false_detected: Tensor,
    correct_detected: Tensor,
    miss_detected: Tensor,
    nan_mask: Tensor,
):
    """Computes Skill values, such as CSI, FAR, POD and TP, FN and FP.

    Args:
        false_detected: Tensor
        correct_detected: Tensor
        miss_detected: Tensor
        nan_mask: Tensor

    Returns:
        Mean CSI, Mean FAR, Mean POD, TP, FN, FP

    """
    tp = torch.nansum(
        nan_mask.expand(correct_detected.shape) * correct_detected,
        dim=tuple(range(2, len(correct_detected.shape))),
    )
    fn = torch.nansum(
        nan_mask.expand(miss_detected.shape) * miss_detected,
        dim=tuple(range(2, len(miss_detected.shape))),
    )
    fp = torch.nansum(
        nan_mask.expand(false_detected.shape) * false_detected,
        dim=tuple(range(2, len(false_detected.shape))),
    )
    _csi_divisor = tp + fn + fp
    _csi_defined = _csi_divisor > 0
    csi = torch.div(tp, _csi_divisor)
    csi[_csi_defined == 0] = 1.0
    _far_divisor = tp + fp
    _far_defined = _far_divisor > 0
    far = torch.div(fp, _far_divisor)
    far[_far_defined == 0] = 0.0
    _pod_divisor = tp + fn
    _pod_defined = _pod_divisor > 0
    pod = torch.div(tp, _pod_divisor)
    pod[_pod_defined == 0] = 1.0
    m_csi = np.nanmean(csi.detach().cpu().numpy(), axis=0)
    m_far = np.nanmean(far.detach().cpu().numpy(), axis=0)
    m_pod = np.nanmean(pod.detach().cpu().numpy(), axis=0)
    s_tp = torch.nansum(tp[:, -1]).item()
    s_fn = torch.nansum(fn[:, -1]).item()
    s_fp = torch.nansum(fp[:, -1]).item()
    return (m_csi, m_far, m_pod), (s_tp, s_fn, s_fp)


def compute_details_stats(state, detail_tag: str):
    for h in state[detail_tag]:
        for m in state[detail_tag][h]:
            state[detail_tag][h][m]["mcsi"] = (
                np.array(state[detail_tag][h][m]["csi"]).mean(axis=(0, 1)).tolist()
            )
            state[detail_tag][h][m]["mfar"] = (
                np.array(state[detail_tag][h][m]["far"]).mean(axis=(0, 1)).tolist()
            )
            state[detail_tag][h][m]["mpod"] = (
                np.array(state[detail_tag][h][m]["pod"]).mean(axis=(0, 1)).tolist()
            )
            state[detail_tag][h][m]["csis"] = np.array(
                state[detail_tag][h][m]["csi"]
            ).tolist()
            state[detail_tag][h][m]["fars"] = np.array(
                state[detail_tag][h][m]["far"]
            ).tolist()
            state[detail_tag][h][m]["pods"] = np.array(
                state[detail_tag][h][m]["pod"]
            ).tolist()
            state[detail_tag][h][m]["csi"] = [
                None for _ in range(len(state[detail_tag][h][m]["tp"]))
            ]
            state[detail_tag][h][m]["far"] = [
                None for _ in range(len(state[detail_tag][h][m]["tp"]))
            ]
            state[detail_tag][h][m]["pod"] = [
                None for _ in range(len(state[detail_tag][h][m]["tp"]))
            ]
            for n_idx in range(len(state[detail_tag][h][m]["tp"])):
                state[detail_tag][h][m]["csi"][n_idx] = (
                    100.0
                    * np.divide(
                        np.sum(state[detail_tag][h][m]["tp"][n_idx]),
                        (
                            np.sum(state[detail_tag][h][m]["tp"][n_idx])
                            + np.sum(state[detail_tag][h][m]["fn"][n_idx])
                            + np.sum(state[detail_tag][h][m]["fp"][n_idx])
                        ),
                    )
                ).tolist()
                state[detail_tag][h][m]["far"][n_idx] = (
                    100.0
                    * np.divide(
                        np.sum(state[detail_tag][h][m]["fp"][n_idx]),
                        (
                            np.sum(state[detail_tag][h][m]["tp"][n_idx])
                            + np.sum(state[detail_tag][h][m]["fp"][n_idx])
                        ),
                    )
                ).tolist()
                state[detail_tag][h][m]["pod"][n_idx] = (
                    100.0
                    * np.divide(
                        np.sum(state[detail_tag][h][m]["tp"][n_idx]),
                        (
                            np.sum(state[detail_tag][h][m]["tp"][n_idx])
                            + np.sum(state[detail_tag][h][m]["fn"][n_idx])
                        ),
                    )
                ).tolist()


def stitch_composite(state, tag, composite, name="composite-results"):
    blocks = state["composite_blocks"]
    if len(blocks) != 2:
        raise NotImplementedError
    if "composite_blocks" in state and np.array(
        state["composite_blocks"]
    ).prod() == 1 == len(composite):
        return False
    idx_lut: dict = {
        tag: idx for idx, tag in enumerate(sorted(next(iter(composite)).keys()))
    }
    composite = [
        (s["LAT"][s["LAT"] != 9999][0].item(), s["LON"][s["LON"] != 9999][0].item(), s)
        for s in composite
    ]
    composite.sort(key=lambda largs: -largs[0])
    composite = [
        composite[s:e]
        for (s, e) in [
            (
                ridx * len(composite) // blocks[0],
                len(composite) // blocks[0] + ridx * len(composite) // blocks[0],
            )
            for ridx in range(blocks[0])
        ]
    ]
    for ridx, row in enumerate(composite):
        row.sort(key=lambda largs: largs[1])
        composite[ridx] = [i[-1] for i in row]
    composite = {
        tt: torch.stack(
            [torch.stack([e[tt] for e in row], dim=0) for row in composite], dim=0
        )
        for tt in idx_lut.keys()
    }
    for tt in composite:
        t = composite[tt]
        if "padding_px" in state and state["padding_px"]:
            for idx, px in enumerate(reversed(state["padding_px"])):
                _idx = len(t.shape) - (idx + 1)
                t = torch.narrow(t, _idx, px, t.shape[_idx] - 2 * px)
        t = torch.movedim(torch.movedim(t, 0, -1), 0, -1)
        t = t.transpose(-2, -3).transpose(-1, -2).transpose(-3, -4)
        t = t.reshape(
            *list(t.shape[:-4]), t.shape[-4] * t.shape[-3], t.shape[-2] * t.shape[-1]
        )
        if "composite_padding" in state and state["composite_padding"]:
            for idx, (px, block_px) in enumerate(
                zip(
                    np.array(state["composite_padding"]).reshape((2, -1)).tolist(),
                    reversed(state["padding_px"]),
                )
            ):
                _idx = len(t.shape) - (idx + 1)
                t = torch.narrow(
                    t, _idx, px[0] - block_px, t.shape[_idx] - (sum(px) - 2 * block_px)
                )
        composite[tt] = t.detach().cpu().numpy()
    dd = (
        Path(state["log.dir"])
        / name
        / f"epoch-{state['current_epoch'] if 'current_epoch' in state else 'final'}"
    )
    dd.mkdir(parents=True, exist_ok=True)
    dd = dd / f"{tag}-{state['network_pt_hash']}.pkl.lz4"
    with lz4.frame.open(dd, mode="wb") as fp:
        pickle.dump(composite, fp)
    return True


def validate(  # noqa: C901 too-complex pylint: disable=too-many-statements
    state, event
):
    if "is_training_warmup" in state and state["is_training_warmup"]:
        return
    MODEL_AVAILABLE = "model" in state
    LOGGER.info("Starting BASELINE validation")
    if "validation_dataloader" not in state:
        state["validation_dataloader"] = event.dataloader(subset="val")
    if MODEL_AVAILABLE:
        m_model_csi, m_model_far, m_model_pod = [], [], []
        model_tp, model_fn, model_fp = 0, 0, 0
    m_per_csi, m_per_far, m_per_pod = [], [], []
    persistence_tp, persistence_fn, persistence_fp = 0, 0, 0
    for details_tag in ["model_details", "persistence_details"]:
        if details_tag in state and state[details_tag]:
            for h in state[details_tag]:
                for m in state[details_tag][h]:
                    state[details_tag][h][m]["tp"].append([])
                    state[details_tag][h][m]["fn"].append([])
                    state[details_tag][h][m]["fp"].append([])
                    state[details_tag][h][m]["csi"].append([])
                    state[details_tag][h][m]["far"].append([])
                    state[details_tag][h][m]["pod"].append([])
                    state[details_tag][h][m]["linets"].append([])
    comps = {}
    tqdm_batch = tqdm(desc="Validation Batches")
    with torch.no_grad():
        for sample in state["validation_dataloader"]:
            sample = {
                k: event.optional.to_device(v, altfn=lambda a: a)
                for k, v in sample.items()
            }
            if MODEL_AVAILABLE:
                out = state["model"](sample[state["PIPES"]["SOURCE"].name])
                out = out.detach().clone().cpu()
            dummy_out = (
                torch.squeeze(sample[state["PIPES"]["TARGET"].name], 1)
                .detach()
                .clone()
                .cpu()
            )
            lat_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "LAT"
            ][0]
            lon_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "LON"
            ][0]
            year_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "YEAR"
            ][0]
            month_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "MONTH"
            ][0]
            day_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "DAY"
            ][0]
            hour_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "HOUR"
            ][0]
            minute_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "MINUTE"
            ][0]
            hour = int(
                sample["META"][:, hour_idx, -1][
                    sample["META"][:, hour_idx, -1] != 9999
                ][0].item()
            )
            minute = int(
                sample["META"][:, minute_idx, -1][
                    sample["META"][:, minute_idx, -1] != 9999
                ][0].item()
            )
            repeats = [1] * dummy_out.ndim
            repeats[1] = dummy_out.shape[1]
            mask = (
                (
                    sample[state["PIPES"]["META"].name][:, lon_idx, -1:].repeat(
                        *repeats
                    )
                    < 9999.0
                )[:, -1:]
                .unsqueeze(1)
                .cpu()
                .clone()
            )
            if "padding_px" in state and state["padding_px"]:
                mask |= (
                    torch.zeros_like(dummy_out, dtype=torch.int8)[:, -1:]
                    .unsqueeze(1)
                    .to(torch.bool)
                    .clone()
                )
            nan_mask = mask.clone().to(torch.float).squeeze(1)
            nan_mask[nan_mask == 0] = float("nan")
            targets = sample[state["PIPES"]["TARGET"].name].squeeze(1)
            targets = targets.to(nan_mask.device) * nan_mask
            lights = torch.nansum(targets)
            if MODEL_AVAILABLE:
                model_threshold = state["Model_Threshold"]
                event.Skill(
                    state,
                    event,
                    logits=out,
                    targets=sample[state["PIPES"]["TARGET"].name]
                    .detach()
                    .to(out.device),
                    threshold=model_threshold,
                    weighting=mask,
                    quiet=True,
                    search_region_radius_px=state["search_region_radius_px"],
                    reduction=None,
                    reduce_non_nan=True,
                )
                probabilities = torch.nn.Softmax(dim=1)(out.detach().clone())
                model_predictions = (probabilities >= model_threshold).to(
                    probabilities.dtype
                )
                nn_skill = torch.cat(
                    [
                        state["fd"][:, -1:].detach().cpu().clone(),
                        state["cd"][:, -1:].detach().cpu().clone(),
                        state["md"][:, -1:].detach().cpu().clone(),
                    ],
                    dim=1,
                ).to(torch.float)
                nn_skill *= nan_mask.expand(nn_skill.shape)
                (mcsi, mfar, mpod), (stp, sfn, sfp) = compute_skill(
                    state["fd"], state["cd"], state["md"], nan_mask
                )
                m_model_csi.append(mcsi)
                m_model_far.append(mfar)
                m_model_pod.append(mpod)
                model_tp += stp
                model_fn += sfn
                model_fp += sfp
                if hour not in state["model_details"]:
                    state["model_details"][hour] = {}
                if minute not in state["model_details"][hour]:
                    state["model_details"][hour][minute] = copy.deepcopy(
                        DETAILS_TEMPLATE
                    )
                state["model_details"][hour][minute]["csi"][-1].append(mcsi)
                state["model_details"][hour][minute]["far"][-1].append(mfar)
                state["model_details"][hour][minute]["pod"][-1].append(mpod)
                state["model_details"][hour][minute]["tp"][-1].append(stp)
                state["model_details"][hour][minute]["fn"][-1].append(sfn)
                state["model_details"][hour][minute]["fp"][-1].append(sfp)
                state["model_details"][hour][minute]["linets"][-1].append(lights.item())
            persistence_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["SOURCE"].name}_channel_tensor_order']
                )
                if key == "PERSISTENCE"
            ][0]
            persistence_model = (
                sample[state["PIPES"]["SOURCE"].name][:, persistence_idx, -1:]
                .detach()
                .to(dummy_out.device)
            )
            persistence_model = torch.cat(
                [
                    torch.ones_like(persistence_model) - persistence_model,
                    persistence_model,
                ],
                dim=1,
            )
            persistence_model *= torch.squeeze(mask, -3).expand(persistence_model.shape)
            event.Skill(
                state,
                event,
                predictions=persistence_model,
                targets=sample[state["PIPES"]["TARGET"].name]
                .detach()
                .to(dummy_out.device),
                quiet=True,
                search_region_radius_px=state["search_region_radius_px"],
                reduction=None,
                reduce_non_nan=True,
                threshold=0.5,
            )
            persistence_skill = torch.cat(
                [
                    state["fd"][:, -1:].detach().cpu().clone(),
                    state["cd"][:, -1:].detach().cpu().clone(),
                    state["md"][:, -1:].detach().cpu().clone(),
                ],
                dim=1,
            ).to(torch.float)
            persistence_skill *= nan_mask.expand(persistence_skill.shape)
            (mcsi, mfar, mpod), (stp, sfn, sfp) = compute_skill(
                state["fd"], state["cd"], state["md"], nan_mask
            )
            m_per_csi.append(mcsi)
            m_per_far.append(mfar)
            m_per_pod.append(mpod)
            persistence_tp += stp
            persistence_fn += sfn
            persistence_fp += sfp
            if hour not in state["persistence_details"]:
                state["persistence_details"][hour] = {}
            if minute not in state["persistence_details"][hour]:
                state["persistence_details"][hour][minute] = copy.deepcopy(
                    DETAILS_TEMPLATE
                )
            state["persistence_details"][hour][minute]["csi"][-1].append(mcsi)
            state["persistence_details"][hour][minute]["far"][-1].append(mfar)
            state["persistence_details"][hour][minute]["pod"][-1].append(mpod)
            state["persistence_details"][hour][minute]["tp"][-1].append(stp)
            state["persistence_details"][hour][minute]["fn"][-1].append(sfn)
            state["persistence_details"][hour][minute]["fp"][-1].append(sfp)
            state["persistence_details"][hour][minute]["linets"][-1].append(
                lights.item()
            )
            if state["dump_results"]:
                dump_sample(
                    state,
                    comps,
                    zip(
                        sample["META"][:, lat_idx, -1].detach().cpu().clone(),
                        sample["META"][:, lon_idx, -1].detach().cpu().clone(),
                        sample["META"][:, year_idx, -1].detach().cpu().clone(),
                        sample["META"][:, month_idx, -1].detach().cpu().clone(),
                        sample["META"][:, day_idx, -1].detach().cpu().clone(),
                        sample["META"][:, hour_idx, -1].detach().cpu().clone(),
                        sample["META"][:, minute_idx, -1].detach().cpu().clone(),
                        targets,
                        sample[state["PIPES"]["SOURCE"].name][:, :, -1]
                        .detach()
                        .cpu()
                        .clone(),
                        sample[state["PIPES"]["SOURCE"].name][
                            :, persistence_idx: persistence_idx + 1, -1:
                        ]
                        .detach()
                        .cpu()
                        .clone(),
                        persistence_skill,
                        out.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                        model_predictions.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                        nn_skill.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                    ),
                    "validation-results",
                )
            tqdm_batch.update()
    tqdm_batch.close()
    if MODEL_AVAILABLE:
        m_model_csi = np.array(m_model_csi) * 100.0
        m_model_far = np.array(m_model_far) * 100.0
        m_model_pod = np.array(m_model_pod) * 100.0
    m_per_csi = np.array(m_per_csi) * 100.0
    m_per_far = np.array(m_per_far) * 100.0
    m_per_pod = np.array(m_per_pod) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        persistence_csi = 100.0 * np.divide(
            persistence_tp, (persistence_tp + persistence_fn + persistence_fp)
        )
        if MODEL_AVAILABLE:
            model_csi = 100.0 * np.divide(model_tp, (model_tp + model_fn + model_fp))
    if not np.isnan(persistence_csi):
        LOGGER.info(
            f"Avg. Batch CSI Validation (Persistence): {np.nanmean(m_per_csi, axis=0)[1]:.2f} +/- {np.nanstd(m_per_csi, axis=0)[1]:.2f}"
        )
        LOGGER.info(f"Complete CSI Validation (Persistence): {persistence_csi:.2f}")
    if MODEL_AVAILABLE and not np.isnan(model_csi):
        LOGGER.info(
            f"Avg. Batch CSI Validation (Model): {np.nanmean(m_model_csi, axis=0)[1]:.2f} +/- {np.nanstd(m_model_csi, axis=0)[1]:.2f}"
        )
        LOGGER.info(f"Complete CSI Validation (Model): {model_csi:.2f}")
        event.optional.plot_scalar("Model/Validation/CSI", model_csi)
        event.optional.plot_scalar(
            "Model/Validation/mCSI", np.nanmean(m_model_csi, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Model/Validation/mFAR", np.nanmean(m_model_far, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Model/Validation/mPOD", np.nanmean(m_model_pod, axis=0)[1]
        )
        state["model_csi"].append(model_csi)
        state["m_model_csi"].extend(m_model_csi)
        state["m_model_far"].extend(m_model_far)
        state["m_model_pod"].extend(m_model_pod)
    if not np.isnan(persistence_csi):
        event.optional.plot_scalar("Persistence/Validation/CSI", persistence_csi)
        event.optional.plot_scalar(
            "Persistence/Validation/mCSI", np.nanmean(m_per_csi, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Persistence/Validation/mFAR", np.nanmean(m_per_far, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Persistence/Validation/mPOD", np.nanmean(m_per_pod, axis=0)[1]
        )
        state["persistence_csi"].append(persistence_csi)
        state["m_per_csi"].extend(m_per_csi)
        state["m_per_far"].extend(m_per_far)
        state["m_per_pod"].extend(m_per_pod)
    if state["dump_results"] and len(comps) > 0:
        LOGGER.warning(f"Compositions: {len(comps)} could not be stitched and saved!")


def testing(state, event):  # noqa: C901 too-complex pylint: disable=too-many-statements
    if not ("enable_testing" in state and state["enable_testing"]):
        return
    MODEL_AVAILABLE = "model" in state
    LOGGER.info("Starting BASELINE testing")
    if "testing_dataloader" not in state:
        state["testing_dataloader"] = event.dataloader(subset="test")
    if MODEL_AVAILABLE:
        m_model_csi, m_model_far, m_model_pod = [], [], []
        model_tp, model_fn, model_fp = 0, 0, 0
    m_per_csi, m_per_far, m_per_pod = [], [], []
    persistence_tp, persistence_fn, persistence_fp = 0, 0, 0
    for details_tag in ["model_testing_details", "persistence_testing_details"]:
        if details_tag in state and state[details_tag]:
            for h in state[details_tag]:
                for m in state[details_tag][h]:
                    state[details_tag][h][m]["tp"].append([])
                    state[details_tag][h][m]["fn"].append([])
                    state[details_tag][h][m]["fp"].append([])
                    state[details_tag][h][m]["csi"].append([])
                    state[details_tag][h][m]["far"].append([])
                    state[details_tag][h][m]["pod"].append([])
                    state[details_tag][h][m]["linets"].append([])
    comps = {}
    tqdm_batch = tqdm(desc="Testing Batches")
    with torch.no_grad():
        for sample in state["testing_dataloader"]:
            sample = {
                k: event.optional.to_device(v, altfn=lambda a: a)
                for k, v in sample.items()
            }
            if MODEL_AVAILABLE:
                out = state["model"](sample[state["PIPES"]["SOURCE"].name])
                out = out.detach().clone().cpu()
            dummy_out = (
                torch.squeeze(sample[state["PIPES"]["TARGET"].name], 1)
                .detach()
                .clone()
                .cpu()
            )
            lon_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "LON"
            ][0]
            lat_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "LAT"
            ][0]
            year_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "YEAR"
            ][0]
            month_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "MONTH"
            ][0]
            day_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "DAY"
            ][0]
            hour_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "HOUR"
            ][0]
            minute_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "MINUTE"
            ][0]
            hour = int(
                sample["META"][:, hour_idx, -1][
                    sample["META"][:, hour_idx, -1] != 9999
                ][0].item()
            )
            minute = int(
                sample["META"][:, minute_idx, -1][
                    sample["META"][:, minute_idx, -1] != 9999
                ][0].item()
            )
            repeats = [1] * dummy_out.ndim
            repeats[1] = dummy_out.shape[1]
            mask = (
                (
                    sample[state["PIPES"]["META"].name][:, lon_idx, -1:].repeat(
                        *repeats
                    )
                    < 9999.0
                )[:, -1:]
                .unsqueeze(1)
                .cpu()
                .clone()
            )
            if "padding_px" in state and state["padding_px"]:
                mask |= (
                    torch.zeros_like(dummy_out, dtype=torch.int8)[:, -1:]
                    .unsqueeze(1)
                    .to(torch.bool)
                    .clone()
                )
            nan_mask = mask.clone().to(torch.float).squeeze(1)
            nan_mask[nan_mask == 0] = float("nan")
            targets = sample[state["PIPES"]["TARGET"].name].squeeze(1)
            targets = targets.to(nan_mask.device) * nan_mask
            lights = torch.nansum(targets)
            if MODEL_AVAILABLE:
                model_threshold = (
                    state["Model_Threshold"]
                    if not state["use_time_thresholds"]
                    else state["time_thresholds"][(hour, minute)][0]
                )
                event.Skill(
                    state,
                    event,
                    logits=out,
                    targets=sample[state["PIPES"]["TARGET"].name]
                    .detach()
                    .to(out.device),
                    threshold=model_threshold,
                    weighting=mask,
                    quiet=True,
                    search_region_radius_px=state["search_region_radius_px"],
                    reduction=None,
                    reduce_non_nan=True,
                )
                probabilities = torch.nn.Softmax(dim=1)(out.detach().clone())
                model_predictions = (probabilities >= model_threshold).to(
                    probabilities.dtype
                )
                nn_skill = torch.cat(
                    [
                        state["fd"][:, -1:].detach().cpu().clone(),
                        state["cd"][:, -1:].detach().cpu().clone(),
                        state["md"][:, -1:].detach().cpu().clone(),
                    ],
                    dim=1,
                ).to(torch.float)
                nn_skill *= nan_mask.expand(nn_skill.shape)
                (mcsi, mfar, mpod), (stp, sfn, sfp) = compute_skill(
                    state["fd"], state["cd"], state["md"], nan_mask
                )
                m_model_csi.append(mcsi)
                m_model_far.append(mfar)
                m_model_pod.append(mpod)
                model_tp += stp
                model_fn += sfn
                model_fp += sfp
                if hour not in state["model_testing_details"]:
                    state["model_testing_details"][hour] = {}
                if minute not in state["model_testing_details"][hour]:
                    state["model_testing_details"][hour][minute] = copy.deepcopy(
                        DETAILS_TEMPLATE
                    )
                state["model_testing_details"][hour][minute]["csi"][-1].append(mcsi)
                state["model_testing_details"][hour][minute]["far"][-1].append(mfar)
                state["model_testing_details"][hour][minute]["pod"][-1].append(mpod)
                state["model_testing_details"][hour][minute]["tp"][-1].append(stp)
                state["model_testing_details"][hour][minute]["fn"][-1].append(sfn)
                state["model_testing_details"][hour][minute]["fp"][-1].append(sfp)
                state["model_testing_details"][hour][minute]["linets"][-1].append(
                    lights.item()
                )
            persistence_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["SOURCE"].name}_channel_tensor_order']
                )
                if key == "PERSISTENCE"
            ][0]
            persistence_model = (
                sample[state["PIPES"]["SOURCE"].name][:, persistence_idx, -1:]
                .detach()
                .to(dummy_out.device)
            )
            persistence_model = torch.cat(
                [
                    torch.ones_like(persistence_model) - persistence_model,
                    persistence_model,
                ],
                dim=1,
            )
            persistence_model *= torch.squeeze(mask, -3).expand(persistence_model.shape)
            event.Skill(
                state,
                event,
                predictions=persistence_model,
                targets=sample[state["PIPES"]["TARGET"].name]
                .detach()
                .to(dummy_out.device),
                quiet=True,
                search_region_radius_px=state["search_region_radius_px"],
                reduction=None,
                reduce_non_nan=True,
            )
            persistence_skill = torch.cat(
                [
                    state["fd"][:, -1:].detach().cpu().clone(),
                    state["cd"][:, -1:].detach().cpu().clone(),
                    state["md"][:, -1:].detach().cpu().clone(),
                ],
                dim=1,
            ).to(torch.float)
            persistence_skill *= nan_mask.expand(persistence_skill.shape)
            (mcsi, mfar, mpod), (stp, sfn, sfp) = compute_skill(
                state["fd"], state["cd"], state["md"], nan_mask
            )
            m_per_csi.append(mcsi)
            m_per_far.append(mfar)
            m_per_pod.append(mpod)
            persistence_tp += stp
            persistence_fn += sfn
            persistence_fp += sfp
            if hour not in state["persistence_testing_details"]:
                state["persistence_testing_details"][hour] = {}
            if minute not in state["persistence_testing_details"][hour]:
                state["persistence_testing_details"][hour][minute] = copy.deepcopy(
                    DETAILS_TEMPLATE
                )
            state["persistence_testing_details"][hour][minute]["csi"][-1].append(mcsi)
            state["persistence_testing_details"][hour][minute]["far"][-1].append(mfar)
            state["persistence_testing_details"][hour][minute]["pod"][-1].append(mpod)
            state["persistence_testing_details"][hour][minute]["tp"][-1].append(stp)
            state["persistence_testing_details"][hour][minute]["fn"][-1].append(sfn)
            state["persistence_testing_details"][hour][minute]["fp"][-1].append(sfp)
            state["persistence_testing_details"][hour][minute]["linets"][-1].append(
                lights.item()
            )
            if state["dump_results"]:
                dump_sample(
                    state,
                    comps,
                    zip(
                        sample["META"][:, lat_idx, -1].detach().cpu().clone(),
                        sample["META"][:, lon_idx, -1].detach().cpu().clone(),
                        sample["META"][:, year_idx, -1].detach().cpu().clone(),
                        sample["META"][:, month_idx, -1].detach().cpu().clone(),
                        sample["META"][:, day_idx, -1].detach().cpu().clone(),
                        sample["META"][:, hour_idx, -1].detach().cpu().clone(),
                        sample["META"][:, minute_idx, -1].detach().cpu().clone(),
                        targets,
                        sample[state["PIPES"]["SOURCE"].name][:, :, -1]
                        .detach()
                        .cpu()
                        .clone(),
                        sample[state["PIPES"]["SOURCE"].name][
                            :, persistence_idx: persistence_idx + 1, -1:
                        ]
                        .detach()
                        .cpu()
                        .clone(),
                        persistence_skill,
                        out.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                        model_predictions.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                        nn_skill.detach().cpu().clone()
                        if MODEL_AVAILABLE
                        else [None] * state["batchsize"],
                    ),
                    "testing-results",
                )
            tqdm_batch.update()
    tqdm_batch.close()
    if MODEL_AVAILABLE:
        m_model_csi = np.array(m_model_csi) * 100.0
        m_model_far = np.array(m_model_far) * 100.0
        m_model_pod = np.array(m_model_pod) * 100.0
    m_per_csi = np.array(m_per_csi) * 100.0
    m_per_far = np.array(m_per_far) * 100.0
    m_per_pod = np.array(m_per_pod) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        persistence_csi = 100.0 * np.divide(
            persistence_tp, (persistence_tp + persistence_fn + persistence_fp)
        )
        if MODEL_AVAILABLE:
            model_csi = 100.0 * np.divide(model_tp, (model_tp + model_fn + model_fp))
    LOGGER.info(
        f"Avg. Batch CSI Testing (Persistence): {np.nanmean(m_per_csi, axis=0)[1]:.2f} +/- {np.nanstd(m_per_csi, axis=0)[1]:.2f}"
    )
    LOGGER.info(f"Complete CSI Testing (Persistence): {persistence_csi:.2f}")
    if MODEL_AVAILABLE:
        LOGGER.info(
            f"Avg. Batch CSI Testing (Model): {np.nanmean(m_model_csi, axis=0)[1]:.2f} +/- {np.nanstd(m_model_csi, axis=0)[1]:.2f}"
        )
        LOGGER.info(f"Complete CSI Testing (Model): {model_csi:.2f}")
        event.optional.plot_scalar("Model/Testing/CSI", model_csi)
        event.optional.plot_scalar(
            "Model/Testing/mCSI", np.nanmean(m_model_csi, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Model/Testing/mFAR", np.nanmean(m_model_far, axis=0)[1]
        )
        event.optional.plot_scalar(
            "Model/Testing/mPOD", np.nanmean(m_model_pod, axis=0)[1]
        )
        state["testing_model_csi"].append(model_csi)
        state["testing_m_model_csi"].extend(m_model_csi)
        state["testing_m_model_far"].extend(m_model_far)
        state["testing_m_model_pod"].extend(m_model_pod)
    event.optional.plot_scalar("Persistence/Testing/CSI", persistence_csi)
    event.optional.plot_scalar(
        "Persistence/Testing/mCSI", np.nanmean(m_per_csi, axis=0)[1]
    )
    event.optional.plot_scalar(
        "Persistence/Testing/mFAR", np.nanmean(m_per_far, axis=0)[1]
    )
    event.optional.plot_scalar(
        "Persistence/Testing/mPOD", np.nanmean(m_per_pod, axis=0)[1]
    )
    state["testing_persistence_csi"].append(persistence_csi)
    state["testing_m_per_csi"].extend(m_per_csi)
    state["testing_m_per_far"].extend(m_per_far)
    state["testing_m_per_pod"].extend(m_per_pod)
    if state["dump_results"] and len(comps) > 0:
        LOGGER.warning(f"Compositions: {len(comps)} could not be stitched and saved!")


def register(mf):
    mf.set_scope("main")
    mf.register_defaults(
        {
            "warm_up_workers": 0,
            "caching_loop": False,
            "force": False,
            "dump_results": False,
            "enable_testing": False,
            "network_pt": [""],
        }
    )
    mf.register_globals(
        {
            "step": 0,
            "examples_seen": 0,
        }
    )
    mf.overwrite_globals(
        {
            "loader.batchsize": 1,
            "loader.shuffle": False,
            "loader.val_prop": 0.0,
            "loader.drop_last": False,
            "loader.num_workers": 4,
            "loader.disable_multithreading": False,
        }
    )
    mf.overwrite_event("step", dummy_step, keep_attached_events=True)
    mf.overwrite_event("main", main, keep_attached_events=True)
    mf.register_event("after_main", accumulate_models, unique=False)
    mf.register_event("after_main", dump_details_json, unique=False)
    mf.register_event("after_step", testing, unique=False)
    mf.register_event("after_step", validate, unique=False)
