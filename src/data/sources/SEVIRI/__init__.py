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

import os
import time
import logging
from typing import Callable
from enum import Enum
import tarfile
import codecs
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import shutil
import hashlib
from itertools import cycle, chain, filterfalse

import numpy as np
from fsspec.implementations.memory import MemoryFile, MemoryFileSystem
import satpy

satpy.log.setLevel(logging.CRITICAL)
from satpy import Scene
from satpy.readers import FSFile
import satpy.readers.seviri_l1b_hrit

satpy.readers.seviri_l1b_hrit.logger.setLevel(logging.ERROR)
from pyPublicDecompWT import xRITDecompress

from ...sources import Channel, PIPES
from .. import DatasetSplit

LOGGER = logging.getLogger(__name__)
TIMER_ID = time.CLOCK_THREAD_CPUTIME_ID


def inspect_data(root: Path, suffix: str, *args, meta_suffix=".meta", **kwargs):
    del args, kwargs  # unused
    LOGGER.info(f"Inspecting data at location '{root}'.")
    packages = [
        os.path.join(_root, file)
        for _root, dirs, files in os.walk(root)
        for file in files
        if file.endswith(suffix)
    ]
    all_frames = []
    for pkg in packages:
        frames = []
        meta_pt = Path(pkg + meta_suffix)
        if meta_pt.exists():
            with open(meta_pt, "rb") as handle:
                frames = pickle.load(handle)
            all_frames.extend(frames)
            continue
        with tarfile.open(pkg) as tar:
            for member in tar.getmembers():
                with tarfile.open(fileobj=tar.extractfile(member)) as frame_tar:
                    readme_obj = next(iter(frame_tar.members))
                    assert (
                        readme_obj.name == "README.txt"
                    ), "First member in tar-file must be README.txt!"
                    readme = (
                        codecs.getreader("utf-8")(frame_tar.extractfile(readme_obj))
                        .read()
                        .split("\n")
                    )
                    pro_file = next(
                        iter([ff for ff in readme if "-_________-PRO______-" in ff])
                    )
                    dt = pro_file.split("-")[-2]
                    dt = datetime.strptime(dt, "%Y%m%d%H%M")
                    dt = dt.replace(tzinfo=timezone.utc)
                frames.append((dt, pkg, member.name))
        with open(meta_pt, "wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
        all_frames.extend(frames)
    LOGGER.info(
        f"Inspecting data at location '{root}' - found {len(all_frames)} frames in {len(packages)} packages."
    )
    return all_frames


def xRIT_decompress(tar_frame, f):
    xrit = xRITDecompress()
    xrit.decompress(tar_frame.extractfile(f).read())
    return xrit.getAnnotationText(), xrit.data()


def group_samples_by(samples, fn, split_order):
    pre_idx = None
    pre_arg = None
    grouped_dt = []
    merged_dt = sorted(samples.keys())
    idx = None
    for idx, dt in enumerate(merged_dt):
        if pre_idx is None:
            pre_idx = idx
            pre_arg = fn(dt)
        elif pre_arg != fn(dt):
            grouped_dt.append((pre_arg, merged_dt[pre_idx:idx]))
            pre_idx = idx
            pre_arg = fn(dt)
    if idx is not None and idx != pre_idx:
        grouped_dt.append((pre_arg, merged_dt[pre_idx:]))
    LOGGER.warning(
        f"Grouped {sum(len(g[1]) for g in grouped_dt)} / {len(merged_dt)} samples."
    )
    assert sum(len(g[1]) for g in grouped_dt) == len(
        merged_dt
    ), "Some samples are lost."
    if len(grouped_dt) < len(split_order):
        LOGGER.warning(
            f"Some Splits may not have samples! Found {len(grouped_dt)} but got {len(split_order)} splits in order."
        )
    arg_info = {mode.value: [] for mode in DatasetSplit}
    grouped_samples = {mode.value: {} for mode in DatasetSplit}
    for g_split, (g_arg, g) in zip(cycle(split_order), grouped_dt):
        arg_info[g_split.value].append((str(g_arg), str(len(g))))
        for s in g:
            grouped_samples[g_split.value][s] = samples[s]
    LOGGER.info(arg_info)
    return grouped_samples


def setup(state):
    LOGGER.info(f"Running {state['module_name']} setup.")
    roots = {
        split.value: sorted(
            [
                Path(r).expanduser().absolute().resolve()
                for r in state[f"root.{split.value}"]
                if r and len(r) > 0
            ]
        )
        for split in state["DatasetSplit"]
    }
    assert all(r.exists() for split in roots for r in roots[split])
    LOGGER.info(f"Roots found: {roots}")
    split_order = [
        DatasetSplit.TRAIN,
        DatasetSplit.TEST,
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    ]
    state.all[f"{state['module_id']}.samples"] = group_samples_by(
        {
            obj[0]: obj[1:]
            for mode in roots
            for r in roots[mode]
            for obj in inspect_data(r, state["suffix"])
        },
        lambda dt: f"{dt.date().isocalendar()[0]}-{dt.date().isocalendar()[1]}",
        split_order,
    )
    if state["enable_pkg_cache"]:
        packages = {
            mode: {
                os.path.join(_root, file): {}
                for r in roots[mode]
                for _root, dirs, files in os.walk(r)
                for file in files
                if file.endswith(state["suffix"])
            }
            for mode in roots
        }
        seviri_tags = set(
            ch
            for pipe in iter(PIPES)
            for ch in state[f"{pipe.name}_channel_tags"]
            if ch in set(map(lambda e: getattr(e, "name"), set(HRIT) | set(BT)))
        )
        for mode in roots:
            for r in roots[mode]:
                for obj in inspect_data(r, state["suffix"]):
                    packages[mode][obj[1]][obj[0]] = seviri_tags.copy()
        state.all[f"{state['module_id']}.pkg_cache"]: dict = packages
    Path(state["resample_cache_dir"]).mkdir(parents=True, exist_ok=True)
    state["SOURCES"].append(state["module_name"])
    try:
        dt = next(
            chain.from_iterable(
                state.all[f"{state['module_id']}.samples"][split.value]
                for split in DatasetSplit
            )
        )
        get_data(
            HRIT.VIS006.value,
            dt,
            **{
                "mode": next(
                    filterfalse(
                        lambda item: item is None,
                        [
                            split
                            if dt
                            in state.all[f"{state['module_id']}.samples"][split.value]
                            else None
                            for split in DatasetSplit
                        ],
                    )
                ),
                "SEVIRI.samples": state.all[f"{state['module_id']}.samples"],
                "SEVIRI.resample_cache_dir": state["resample_cache_dir"],
                "SEVIRI.enable_pkg_cache": state["enable_pkg_cache"],
                "area_config": state["area_config"],
            },
        )
    except StopIteration as e:
        LOGGER.warning("Resample cache could not be initialized.")
        raise RuntimeError("No samples available.") from e


def get_data(channel, dt, **kwargs):
    mode = kwargs.get("mode", None)
    pkg, frame = kwargs["SEVIRI.samples"][mode.value if mode is not None else "train"][
        dt
    ]
    pkg_cache = kwargs.get("SEVIRI.pkg_cache", None)
    enable_pkg_cache = kwargs.get("SEVIRI.enable_pkg_cache", False)
    calibration = {"HRIT": "counts", "BT": "brightness_temperature"}[channel.group]
    d = {channel.group: {}}
    tar_pkg = Path(pkg)
    cache_pkg_pt = None
    if enable_pkg_cache and pkg_cache is not None and pkg in pkg_cache[mode.value]:
        cache_pkg_pt = (
            Path(kwargs["SEVIRI.resample_cache_dir"])
            / f"{hashlib.sha512(pkg.encode('utf-8')).hexdigest()}-{tar_pkg.name}"
        )
        if not cache_pkg_pt.exists():
            LOGGER.info(f"Copying new cache file {tar_pkg} to cache.")
            tar_pkg = shutil.copy(tar_pkg, cache_pkg_pt)
        else:
            tar_pkg = cache_pkg_pt
    with tarfile.open(tar_pkg) as tar:
        with tarfile.open(fileobj=tar.extractfile(frame)) as frame_tar:
            mem_fs = MemoryFileSystem(skip_instance_cache=True, cachable=False)
            fs_files = [
                MemoryFile(
                    fs=mem_fs, path=f"{mem_fs.root_marker}{annotationText}", data=bites
                )
                for annotationText, bites in (
                    xRIT_decompress(frame_tar, f)
                    for f in frame_tar
                    if f.name != "README.txt"
                    and f.name.split("-")[4] in {f"{channel.tag}___", "_________"}
                )
            ]
            _ = [cmt_f.commit() for cmt_f in fs_files]
            fs_files = [FSFile(open_file) for open_file in fs_files]
            scn = Scene(filenames=fs_files, reader="seviri_l1b_hrit")
            scn.load([channel.tag], calibration=calibration, upper_right_corner="NE")
            assert (
                len(scn.missing_datasets) == 0 and len(scn.wishlist) != 0
            ), "Not all SEVIRI channels available for this frame."
            scn = scn.resample(
                kwargs["area_config"], cache_dir=kwargs["SEVIRI.resample_cache_dir"]
            )
            for dq in scn.wishlist:
                d[channel.group][dq["name"]] = scn[dq].data.compute().copy()
            del scn
            for _ in range(len(fs_files)):
                _f = fs_files[0]
                mem_fs.rm(str(_f))
                del fs_files[0]
            del mem_fs
    if enable_pkg_cache and pkg_cache is not None and pkg in pkg_cache[mode.value]:
        if channel.name in pkg_cache[mode.value][pkg][dt]:
            pkg_cache[mode.value][pkg][dt].remove(channel.name)
        if len(pkg_cache[mode.value][pkg][dt]) == 0:
            del pkg_cache[mode.value][pkg][dt]
        if len(pkg_cache[mode.value][pkg]) == 0 and cache_pkg_pt is not None:
            LOGGER.info(f"Deleting cache file {tar_pkg} from cache.")
            cache_pkg_pt.unlink()
            del pkg_cache[mode.value][pkg]
    return d[channel.group][channel.tag]


class HRITChannel(Channel):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        tag: str,
        operator: Callable = None,
        mean: float = 0.0,
        stddev: float = 1.0,
        classes: int = 1,
    ):
        super().__init__(
            tag,
            "SEVIRI",
            "HRIT",
            operator,
            lambda tensor: tensor / np.float32(2**16),
            get_data,
            mean,
            stddev,
            classes,
        )


class BTChannel(Channel):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        tag: str,
        normalization: Callable = None,
        mean: float = 0.0,
        stddev: float = 1.0,
        classes: int = 1,
    ):
        super().__init__(
            tag,
            "SEVIRI",
            "BT",
            lambda tensor: np.flipud(tensor).astype(np.float32),
            normalization,
            get_data,
            mean,
            stddev,
            classes,
        )


class HRIT(Enum):
    VIS006 = HRITChannel("VIS006")
    VIS008 = HRITChannel("VIS008")
    IR_016 = HRITChannel("IR_016")
    IR_039 = HRITChannel("IR_039")
    WV_062 = HRITChannel("WV_062")
    WV_073 = HRITChannel("WV_073")
    IR_087 = HRITChannel("IR_087")
    IR_097 = HRITChannel("IR_097")
    IR_108 = HRITChannel("IR_108")
    IR_120 = HRITChannel("IR_120")
    IR_134 = HRITChannel("IR_134")


class BT(Enum):
    BT_062 = BTChannel("WV_062")
    BT_073 = BTChannel("WV_073")
    BT_087 = HRITChannel("IR_087")
    BT_097 = HRITChannel("IR_097")
    BT_108 = HRITChannel("IR_108")
    BT_120 = HRITChannel("IR_120")
    BT_134 = HRITChannel("IR_134")


def register(mf):
    mf.register_defaults(
        {
            "roots.train": [str],
            "roots.val": [str],
            "roots.test": [str],
            "resample_cache_dir": "/tmp/dfcc-cache",
        }
    )
    mf.register_helpers(
        {
            "GROUPS": [HRIT, BT],
            "time_delta": timedelta(minutes=15),
            "module_name": mf.module_name,
            "module_id": mf.module_id,
            "suffix": ".tar",
            "calibration_counts": [
                0.6,
                0.8,
                1.6,
                3.9,
                6.2,
                7.3,
                8.7,
                9.7,
                10.8,
                12.0,
                13.4,
            ],
            "calibration_brightness_temperature": [6.2, 7.3, 10.8],
            "enable_pkg_cache": False,
        }
    )
    mf.register_event("init", setup, unique=False)
    mf.register_event("get_data", get_data, unique=False)
