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

import sys
import pickle
from pathlib import Path
import logging
import hashlib
import json
import enum

import lz4.frame
import numpy as np
from tqdm import tqdm
import torch

from batchedmoments import BatchedMoments

LOGGER = logging.getLogger(__name__)


def check_ds_kw_exists(state, ds_hash_fp):
    if not ds_hash_fp.exists():
        return False
    with open(ds_hash_fp, encoding="utf-8") as handle:
        obj = json.load(handle)
        for kw in state["dataset_hash_kw"]:
            if kw not in obj:
                return False
    return True


def print_helper(tensor, batched_data_no=8, timeidx=-1):
    if tensor.nelement() == 0:
        return None
    tensor = tensor[:batched_data_no]
    if len(tensor.shape) == 5:  # (?,C,T,H,W)
        tensor = tensor[:, :, timeidx]
    channels = tensor.shape[1]
    if channels == 2:
        tensor = torch.cat(
            [tensor, torch.sum(torch.zeros_like(tensor), 1, keepdim=True)], dim=1
        )
    if channels > 3:
        tensor = torch.mean(tensor, 1, keepdim=True)
    if tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float32)
    return tensor


def main(state, event):  # noqa: C901 too-complex pylint: disable=too-many-statements
    refresh_cache = state["refresh_cache"]
    state["refresh_cache"] = False
    train_loader = event.dataloader(subset="train")
    state["validation_dataloader"] = event.dataloader(subset="val")
    state["model"] = event.init_net()
    state["model"] = event.optional.to_device(state["model"], altfn=lambda m: m)
    state["model"] = event.optional.data_parallel(state["model"], altfn=lambda m: m)
    state["criterion"] = torch.nn.CrossEntropyLoss(reduction="none")
    state["criterion"] = event.optional.to_device(state["criterion"], altfn=lambda c: c)
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
        cache_loader = event.dataloader(
            subset="train", use_cache=False, deterministic=True
        )
        ds_hash_fp = (
            Path(state["log.dir"]).parent / f"{cache_loader.dataset.get_hash()}.lock"
        )
        if not ds_hash_fp.exists() and not state["force_overwrite_locks"]:
            LOGGER.info(f"{ds_hash_fp} not available, refreshing cache.")
            with torch.no_grad():
                tqdm_batch = tqdm(desc="Caching Batches", **state["tqdm_kwargs"])
                for _ in cache_loader:
                    tqdm_batch.update()
                tqdm_batch.close()
            ds_hash_fp.touch()
        if state["force_overwrite_locks"]:
            ds_hash_fp.touch()
        cache_loader = event.dataloader(subset="val", use_cache=False)
        ds_hash_fp = (
            Path(state["log.dir"]).parent / f"{cache_loader.dataset.get_hash()}.lock"
        )
        if not ds_hash_fp.exists() and not state["force_overwrite_locks"]:
            LOGGER.info(f"{ds_hash_fp} not available, refreshing cache.")
            with torch.no_grad():
                tqdm_batch = tqdm(
                    desc="Caching Validation Batches", **state["tqdm_kwargs"]
                )
                for _ in cache_loader:
                    tqdm_batch.update()
                tqdm_batch.close()
            ds_hash_fp.touch()
        if state["force_overwrite_locks"]:
            ds_hash_fp.touch()
        cache_loader = event.dataloader(subset="test", use_cache=False)
        ds_hash_fp = (
            Path(state["log.dir"]).parent / f"{cache_loader.dataset.get_hash()}.lock"
        )
        if not ds_hash_fp.exists() and not state["force_overwrite_locks"]:
            LOGGER.info(f"{ds_hash_fp} not available, refreshing cache.")
            with torch.no_grad():
                tqdm_batch = tqdm(desc="Caching Test Batches", **state["tqdm_kwargs"])
                for _ in cache_loader:
                    tqdm_batch.update()
                tqdm_batch.close()
            ds_hash_fp.touch()
        if state["force_overwrite_locks"]:
            ds_hash_fp.touch()
    try:
        total_batches = len(train_loader)
        state["num_batches"] = total_batches
    except TypeError:
        ds_loader_hash_state = {
            k: state[k] if not isinstance(state[k], enum.Enum) else state[k].name
            for k in sorted(
                list(
                    k
                    for k in state.all.keys()
                    if ("data.loader" in k or "dataset.binary" in k)
                    and not (
                        "path" in k
                        or "shuffle" in k
                        or "Class" in k
                        or "cache" in k
                        or "thread" in k
                        or "loader.iterable" in k
                        or "loader.mappable" in k
                        or "search_region" in k
                        or "measure_time" in k
                        or "debug_dict" in k
                        or "window_size" in k
                        or "zero_out" in k
                    )
                )
            )
        }
        pipe_digs = train_loader.dataset.generate_pipe_hashes()
        loader_ds_digest = hashlib.sha512(
            "-".join(
                [str(ds_loader_hash_state), str(train_loader.dataset.samples)]
                + [pipe_digs[pipe] for pipe in sorted(pipe_digs)]
            ).encode("utf-8")
        ).hexdigest()
        ds_hash_fp = Path(state["log.dir"]).parent / f"{loader_ds_digest}.meta"
        if not check_ds_kw_exists(state, ds_hash_fp):
            LOGGER.info(f"Dataset meta file not available. ('{ds_hash_fp}')")
            list_of_batch_sizes = []
            idx = None
            cw_training = BatchedMoments(axis=(0))  # pylint: disable=superfluous-parens
            with torch.no_grad():
                tqdm_batch = tqdm(
                    desc="Warm-Up Training Batches", **state["tqdm_kwargs"]
                )
                for idx, batch in enumerate(train_loader):
                    list_of_batch_sizes.append(next(iter(batch.values())).shape[0])
                    cw = event.optional.get_class_weights(batch, altfn=lambda a: None)
                    if cw is not None:
                        cw_training(cw.detach().cpu().numpy())
                    tqdm_batch.update()
            tqdm_batch.close()
            total_batches = idx + 1 if idx is not None else 0
            if total_batches is not None:
                state["num_batches"] = total_batches
                LOGGER.debug(f"Batch sizes: {list_of_batch_sizes}")
            if cw_training is not None and cw_training.mean is not None:
                state["cw_training"] = cw_training.mean.tolist()
                LOGGER.info(f"Class weights: {state['cw_training']}")
            else:
                state["cw_training"] = None
            cw_validation = BatchedMoments(
                axis=(0)
            )  # pylint: disable=superfluous-parens
            with torch.no_grad():
                tqdm_batch = tqdm(
                    desc="Warm-Up Validation Batches", **state["tqdm_kwargs"]
                )
                for idx, batch in enumerate(state["validation_dataloader"]):
                    cw = event.optional.get_class_weights(batch, altfn=lambda a: None)
                    if cw is not None:
                        cw_validation(cw.detach().cpu().numpy())
                    tqdm_batch.update()
            tqdm_batch.close()
            total_batches = idx + 1 if idx is not None else 0
            if total_batches is not None:
                state["validation_num_batches"] = total_batches
            if cw_validation is not None and cw_validation.mean is not None:
                state["cw_validation"] = cw_validation.mean.tolist()
                LOGGER.info(f"Class weights (validation): {state['cw_validation']}")
            else:
                state["cw_validation"] = None
            with open(ds_hash_fp, "w", encoding="utf-8") as handle:
                json.dump({kw: state[kw] for kw in state["dataset_hash_kw"]}, handle)
        else:
            with open(ds_hash_fp, encoding="utf-8") as handle:
                obj = json.load(handle)
                for kw in state["dataset_hash_kw"]:
                    state[kw] = obj[kw]
    if "num_batches" in state and state["num_batches"] == 0:
        LOGGER.warning("No training samples/batches available!")
        return
    if state["epochs"] < 1:
        return
    state["optimizer"] = event.init_optimizer(state["model"])
    LOGGER.info(f"Training Class Weights: {state['cw_training']}")
    LOGGER.info(f"Validation Class Weights: {state['cw_validation']}")
    LOGGER.info("Starting training")
    event.optional.before_training()
    tqdm_epoch = tqdm(
        total=state["epochs"],
        position=1,
        desc="Epoch",
        initial=state["start_epoch"],
        **state["tqdm_kwargs"],
    )
    tqdm_batch = tqdm(total=None, position=2, desc="Batches", **state["tqdm_kwargs"])
    for state["current_epoch"] in range(state["start_epoch"], state["epochs"]):
        total_batches = state["num_batches"] if "num_batches" in state else None
        tqdm_batch.reset(total=total_batches)
        event.optional.before_epoch()
        list_of_batch_sizes = []
        for state["current_batch"], state["sample"] in enumerate(train_loader):
            list_of_batch_sizes.append(next(iter(state["sample"].values())).shape[0])
            state["sample"] = {
                k: event.optional.to_device(v, altfn=lambda a: a)
                for k, v in state["sample"].items()
            }
            event.step()
            if event.plot_every() and not state["log.slim"]:
                for pipe in state["PIPES"]:
                    if pipe == "META":
                        continue
                    d = state["sample"][pipe].detach().clone()
                    if pipe == state["PIPES"]["TARGET"].name:
                        d = event.to_one_hot_vector(
                            d, state[f"num_{state['PIPES']['TARGET'].name}_classes"] + 1
                        )
                        d = d[:, -1:]
                    if pipe == state["PIPES"]["AUXILIARY"].name:
                        d = d.to(torch.float32)
                        d -= d.min()
                        d /= d.max()
                        d = torch.mean(d, dim=1, keepdim=True)
                    d = print_helper(d, batched_data_no=state["batched_data_no"])
                    if d is not None:
                        event.optional.plot_imgs(f"IO/{pipe}", d)
                probabilities = torch.nn.Softmax(dim=1)(
                    state["logits"].detach().clone()
                )
                threshold = (
                    state["EMA_threshold"].mean.item()
                    if "EMA_threshold" in state
                    else 0.5
                )
                predictions = (probabilities >= threshold).to(probabilities.dtype)
                event.optional.plot_imgs(
                    "Model/Probabilities",
                    print_helper(
                        probabilities[:, -1:], batched_data_no=state["batched_data_no"]
                    ),
                )
                event.optional.plot_imgs(
                    "Model/Predictions",
                    print_helper(
                        predictions[:, -1:], batched_data_no=state["batched_data_no"]
                    ),
                )
            state["step"] += 1
            state["examples_seen"] += next(iter(state["sample"].values())).shape[0]
            state["batches_seen"] += 1
            tqdm_batch.update(1)
            del state["sample"], state["mask"], state["logits"]
        tqdm_epoch.update(1)
        if "num_batches" not in state:
            state["num_batches"] = (
                (state["current_batch"] + 1) if "current_batch" in state else 0
            )
        elif state["current_batch"] + 1 != state["num_batches"]:
            LOGGER.warning(
                f"Unexpected amount of batches (got {state['current_batch'] + 1}, expected {state['num_batches']})."
            )
            LOGGER.warning(f"Overall {state['examples_seen']} samples seen so far.")
            LOGGER.info(f"Batch sizes: {list_of_batch_sizes}")
        state["epochs_seen"] += 1
        event.optional.after_epoch()
        LOGGER.info(f"End of epoch {state['epochs_seen']}")
    event.optional.after_training()
    tqdm_batch.close()
    tqdm_epoch.close()


def validate(
    state, event
):  # noqa: C901 too-complex pylint: disable=too-many-statements
    LOGGER.info("Starting validation")
    if "validation_dataloader" not in state:
        state["validation_dataloader"] = event.dataloader(subset="val")
    assert "model" in state
    state["model"].eval()
    state["model"].zero_grad()
    mcsi = BatchedMoments(axis=())
    mfar = BatchedMoments(axis=())
    mpod = BatchedMoments(axis=())
    model_tp, model_fn, model_fp = 0, 0, 0
    comps = {}
    tqdm_batch = tqdm(
        total=state["validation_num_batches"],
        desc="Validation Batches",
        position=2,
        **state["tqdm_kwargs"],
    )
    early_break_is_on = (
        "validation_batches" in state and state["validation_batches"] > 0
    )
    with torch.no_grad():
        for batch_ix, sample in enumerate(state["validation_dataloader"]):
            sample = {
                k: event.optional.to_device(v, altfn=lambda a: a)
                for k, v in sample.items()
            }
            out = state["model"](sample[state["PIPES"]["SOURCE"].name])
            out = out.detach().clone().cpu()  # transfer to cpu
            lon_idx = [
                idx
                for idx, (key, _) in enumerate(
                    state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
                )
                if key == "LON"
            ][0]
            repeats = [1] * out.ndim
            repeats[1] = out.shape[1]
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
                    torch.zeros_like(out, dtype=torch.int8)[:, -1:]
                    .unsqueeze(1)
                    .to(torch.bool)
                    .clone()
                )
            b_csi, b_far, b_pod = event.Skill(
                state,
                event,
                logits=out.detach(),
                targets=sample[state["PIPES"]["TARGET"].name].detach().to(out.device),
                weighting=mask,
                quiet=True,
                search_region_radius_px=state["search_region_radius_px"],
            )
            nn_skill = torch.cat(
                [
                    state["fd"][:, -1:].detach().cpu().clone(),
                    state["cd"][:, -1:].detach().cpu().clone(),
                    state["md"][:, -1:].detach().cpu().clone(),
                ],
                dim=1,
            ).to(torch.int32)
            mcsi(b_csi.detach().cpu().numpy())
            mfar(b_far.detach().cpu().numpy())
            mpod(b_pod.detach().cpu().numpy())
            model_tp += state["cd"][:, -1].sum().item()
            model_fn += state["md"][:, -1].sum().item()
            model_fp += state["fd"][:, -1].sum().item()
            lat_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "LAT"
            ][0]
            lon_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "LON"
            ][0]
            year_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "YEAR"
            ][0]
            month_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "MONTH"
            ][0]
            day_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "DAY"
            ][0]
            hour_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "HOUR"
            ][0]
            minute_idx = [
                idx
                for idx, (key, enu) in enumerate(state["META_channel_tensor_order"])
                if key == "MINUTE"
            ][0]
            if (
                state["dump_validation_results"]
                and state["epochs_seen"] == state["epochs"]
            ):
                for (
                    lat,
                    lon,
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    logits,
                    nn_sk,
                    in_ch,
                ) in zip(
                    sample["META"][:, lat_idx, -1].detach().cpu().clone(),
                    sample["META"][:, lon_idx, -1].detach().cpu().clone(),
                    sample["META"][:, year_idx, -1].detach().cpu().clone(),
                    sample["META"][:, month_idx, -1].detach().cpu().clone(),
                    sample["META"][:, day_idx, -1].detach().cpu().clone(),
                    sample["META"][:, hour_idx, -1].detach().cpu().clone(),
                    sample["META"][:, minute_idx, -1].detach().cpu().clone(),
                    out.detach().cpu().clone(),
                    nn_skill.detach().cpu().clone(),
                    sample[state["PIPES"]["SOURCE"].name][:, :, -1]
                    .detach()
                    .cpu()
                    .clone(),
                ):
                    year = int(year[year != 9999][0].item())
                    month = int(month[month != 9999][0].item())
                    day = int(day[day != 9999][0].item())
                    hour = int(hour[hour != 9999][0].item())
                    minute = int(minute[minute != 9999][0].item())
                    tag = f"{year:04}{month:02}{day:02}{hour:02}{minute:02}"
                    dt_c = comps.get(tag, [])
                    dt_c.append(
                        {
                            "LAT": lat,
                            "LON": lon,
                            "NN_PREDICTIONS": (
                                torch.nn.Softmax(dim=1)(logits)
                                >= state["EMA_threshold"].mean.item()
                            ).to(torch.int32),
                            "NN_SKILL": nn_sk,
                            "CHANNELS": in_ch,
                        }
                    )
                    comps[tag] = dt_c
            tqdm_batch.update(1)
            if early_break_is_on and batch_ix >= state["validation_batches"]:
                break
    tqdm_batch.close()
    with np.errstate(divide="ignore", invalid="ignore"):
        model_csi = np.nan_to_num(np.divide(model_tp, (model_tp + model_fn + model_fp)))
    LOGGER.info(
        f"Avg. Batch CSI: {mcsi.mean[1] if mcsi.mean is not None else 0.0:.5f} +/- {mcsi.std[1] if mcsi.std is not None else 0.0:.5f}"
    )
    LOGGER.info(f"Complete CSI: {model_csi if model_csi is not None else 0.0:.5f}")
    event.optional.plot_scalar(
        "Skill/Validation/CSI", model_csi if model_csi is not None else 0.0
    )
    event.optional.plot_scalar(
        "Skill/Validation/mCSI", mcsi.mean[1] if mcsi.mean is not None else 0.0
    )
    event.optional.plot_scalar(
        "Skill/Validation/mFAR", mfar.mean[1] if mfar.mean is not None else 0.0
    )
    event.optional.plot_scalar(
        "Skill/Validation/mPOD", mpod.mean[1] if mpod.mean is not None else 0.0
    )
    if state["dump_validation_results"] and state["epochs_seen"] == state["epochs"]:
        for tag, values in comps.items():
            if "composite_blocks" in state and np.array(
                state["composite_blocks"]
            ).prod() == 1 == len(values):
                continue
            blocks = state["composite_blocks"]
            if len(blocks) != 2:
                raise NotImplementedError
            idx_lut: dict = {
                tag: idx for idx, tag in enumerate(sorted(next(iter(values)).keys()))
            }
            values = [
                (
                    s["LAT"][s["LAT"] != 9999][0].item(),
                    s["LON"][s["LON"] != 9999][0].item(),
                    s,
                )
                for s in values
            ]
            values.sort(key=lambda largs: largs[0])
            values = [
                values[s:e]
                for (s, e) in [
                    (
                        ridx * len(values) // blocks[0],
                        len(values) // blocks[0] + ridx * len(values) // blocks[0],
                    )
                    for ridx in range(blocks[0])
                ]
            ]
            for ridx, row in enumerate(values):
                row.sort(key=lambda largs: largs[1])
                values[ridx] = [i[-1] for i in row]
            values = {
                tt: torch.stack(
                    [torch.stack([e[tt] for e in row], dim=0) for row in values], dim=0
                )
                for tt in idx_lut.keys()
            }
            for tt in values:
                t = values[tt]
                if "padding_px" in state and state["padding_px"]:
                    for idx, px in enumerate(reversed(state["padding_px"])):
                        _idx = len(t.shape) - (idx + 1)
                        t = torch.narrow(t, _idx, px, t.shape[_idx] - 2 * px)
                t = torch.movedim(torch.movedim(t, 0, -1), 0, -1)
                t = t.transpose(-2, -3).transpose(-1, -2).transpose(-3, -4)
                t = t.reshape(
                    *list(t.shape[:-4]),
                    t.shape[-4] * t.shape[-3],
                    t.shape[-2] * t.shape[-1],
                )
                if "composite_padding" in state and state["composite_padding"]:
                    for idx, (px, block_px) in enumerate(
                        zip(
                            np.array(state["composite_padding"])
                            .reshape((2, -1))
                            .tolist(),
                            reversed(state["padding_px"]),
                        )
                    ):
                        _idx = len(t.shape) - (idx + 1)
                        t = torch.narrow(
                            t,
                            _idx,
                            px[0] - block_px,
                            t.shape[_idx] - (sum(px) - 2 * block_px),
                        )
                values[tt] = t.detach().cpu().numpy()
            dd = (
                Path(state["log.dir"])
                / "validation-results"
                / f"epoch-{state['current_epoch']}"
            )
            dd.mkdir(parents=True, exist_ok=True)
            dd = dd / f"{tag}.pkl.lz4"
            with lz4.frame.open(dd, mode="wb") as fp:
                pickle.dump(values, fp)
    state["model"].train()


def dump_model(state):
    model_pt = (
        Path(state["log.dir"])
        / f"{state['model'].__class__.__name__}.{state['log.repository_state'][0][-1]}.{str(state['epochs_seen']).zfill(len(str(state['epochs'])))}.torch"
    )
    torch.save(
        {
            f"{state['model'].__class__.__name__}": state["model"].state_dict(),
            "Model_Threshold": state["EMA_threshold"].mean.item()
            if "EMA_threshold" in state and state["EMA_threshold"] is not None
            else None,
        },
        model_pt,
    )
    LOGGER.debug(f"Saved model state as: {model_pt}")


def to_one_hot_vector(labels, num_classes):
    return torch.squeeze(
        torch.movedim(
            torch.eye(num_classes, dtype=labels.dtype, device=labels.device)[labels],
            -1,
            1,
        ),
        -3,
    )


def register(mf):
    mf.set_scope("main")
    mf.register_defaults(
        {
            "force_overwrite_locks": False,
            "epochs": 10,
            "validation_batches": -1,
            "warm_up_workers": 0,
            "dump_validation_results": False,
        }
    )
    mf.register_helpers(
        {
            "start_epoch": 0,
            "tqdm_kwargs": {
                "file": sys.stdout,
                "dynamic_ncols": True,
            },
            "dataset_hash_kw": [
                "num_batches",
                "validation_num_batches",
                "cw_validation",
                "cw_training",
            ],
        }
    )
    mf.register_globals(
        {
            "step": 0,
            "examples_seen": 0,
        }
    )
    mf.register_event("main", main)
    mf.register_event("to_one_hot_vector", to_one_hot_vector, unique=True)
    mf.register_event("after_epoch", validate, unique=False)
    mf.register_event("after_epoch", dump_model, unique=False)
