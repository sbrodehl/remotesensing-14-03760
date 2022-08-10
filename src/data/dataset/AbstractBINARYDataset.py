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

# pylint: disable=C0413
import logging
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
import time
import datetime
import random
import hashlib
from typing import List
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.transforms import Compose

from .utils import DatasetSplit
from .cache import CacheLayer  # pylint: disable=relative-beyond-top-level

LOGGER = logging.getLogger(__name__)


def generate_continuous_patches(samples, time_delta=None):
    if time_delta is None:
        return [samples]
    # iterate over samples and check if "time delta" holds
    ret = []
    _lst = []
    _prev = None
    for s in samples:
        if _prev is None:
            _prev = s
            _lst.append(s)
            continue
        if _prev + time_delta == s:
            _prev = s
            _lst.append(s)
        else:
            ret.append(_lst.copy())
            _lst.clear()
            _prev = s
            _lst.append(s)
    # check remaining list items
    if len(_lst) > 0:
        ret.append(_lst.copy())
        del _lst
    return ret


def apply_window(data: list, window_size: int = 1) -> list:
    # if window_size is 1, nothing is to do
    if window_size == 1:
        # flatten all sequences list(s),
        return [[i] for s in data for i in s]
    _data = []
    for d in data:
        for j in range(len(d) - window_size + 1):
            sample = []
            # TODO: slicing is more efficient
            for i in range(window_size):
                sample.append(d[j + i])
            _data.append(sample)
    return _data


class AbstractBINARYDataset:  # pylint: disable=invalid-name,too-many-instance-attributes
    """The great unified BINARY dataset."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-statements
        self,
        space_transforms=None,
        space_time_transforms=None,
        transform=None,
        window_size=3,
        **kwargs,
    ):
        self._state = kwargs.get("state", None)
        self._event = kwargs.get("event", None)
        self.dataset_split: DatasetSplit = kwargs.get(
            "dataset_split", DatasetSplit.TRAIN
        )
        self.window_size = window_size
        # check CHANNELS
        assert self._state["CHANNELS"] is not None and isinstance(
            self._state["CHANNELS"], dict
        )
        # check TAGS
        # assert not overlapping and at least one tags given
        is_not_empty: bool = False
        _tags: set = set()
        _len_tags: int = 0
        for p in self._state["PIPES"].values():
            tags = kwargs.get(f"{p.name}_channel_tags", [])
            is_not_empty |= bool(len(tags))
            _tags |= set(tags)
            _len_tags += len(tags)
            self.__setattr__(f"{p.name}_channel_tags", tags)

        assert is_not_empty, "At least one (1) channel must be set."
        if _len_tags == len(_tags):
            LOGGER.warning("Overlapping channel tags found, this can be dangerous!")

        self.refresh_cache = kwargs.get("refresh_cache", False)
        self.transform = transform
        self.transforms_space = space_transforms
        self.transforms_space_time = space_time_transforms
        self.cache_path: str = kwargs.get("cache_path", "/tmp")
        self.cache_prefix: str = kwargs.get("cache_prefix", "")
        self.cache_prefixes: dict = kwargs.get("cache_prefixes", {})
        if len(self.cache_prefixes) != 0:
            LOGGER.info(f"Cache Prefixes: '{str(self.cache_prefixes)}'")
        self.enable_cache = kwargs.get("enable_cache", False)
        if self.enable_cache:
            try:  # only caching of space_transforms is support
                cache_transform_idx_s = [
                    i
                    for i, tf in enumerate(self.transforms_space.transforms)
                    if CacheLayer in tf.__class__.__bases__
                ]
                cache_transform_idx = next(
                    iter(cache_transform_idx_s[::-1])
                )  # get latest CacheLayer
                cache_transforms = self.transforms_space.transforms[
                    cache_transform_idx:
                ]
                # enable cache on this cache layer
                cache_transforms[0].enable_cache = True
                cache_transforms[0].refresh_cache = self.refresh_cache
                cache_transforms[0].cache_path = Path(str(self.cache_path))
                self.cache_space_transforms = Compose(cache_transforms)
                # TODO: check if time- or regular transforms contain a cache layer and warn the user about it
            except StopIteration:
                LOGGER.warning(
                    "No CacheLayer found in transforms found! Transform-cache disabled!"
                )
                self.enable_cache = False
        self.measure_time = kwargs.get("measure_time", False)
        self.filter_date_start = kwargs.get("filter_date_start", None)
        self.filter_date_end = kwargs.get("filter_date_end", None)
        self.filter_time_start = kwargs.get("filter_time_start", None)
        self.filter_time_end = kwargs.get("filter_time_end", None)
        self.filter_time_hours = kwargs.get("filter_time_hours", None)
        self.filter_time_minutes = kwargs.get("filter_time_minutes", None)
        # general settings of region of interest (ROI)
        self.deg_per_px = kwargs.get("deg_per_px", None)

        self.shuffle_samples = kwargs.get("shuffle_samples", False)
        self.samples = self.generate_samples()
        LOGGER.info(
            f"{self.__class__.__name__}:: Set {len(self.samples)} samples to work with."
        )
        # performance timer things
        self.timer = time.CLOCK_THREAD_CPUTIME_ID
        self.timer_name = "CLOCK_THREAD_CPU_TIME"

        # generate parameters passed to 'loading' method of channels
        self.channel_parameters = {
            p.__name__: [
                a for a in self.__dir__() if a.startswith(p.__name__.lower() + "_")
            ]
            for p in self._state["sources.GROUPS"].values()
        }
        self.source_parameters = {
            s: [
                k
                for k in self._state.all.keys()
                if "data.sources." in k
                and k.split("data.sources.")[-1].split(".")[0]
                not in set(set(self._state["SOURCES"]) - {s})
            ]
            for s in self._state["SOURCES"]
        }

    def get_hash(self):
        pipe_digs = self.generate_pipe_hashes()
        return hashlib.sha512(
            "-".join(
                [str(self.dataset_split), str(self.samples)]
                + [pipe_digs[pipe] for pipe in sorted(pipe_digs)]
            ).encode("utf-8")
        ).hexdigest()

    def generate_pipe_hashes(self, dt_tag=None):
        cache_digests = {}
        _gen_str = ";".join(
            str(e) for e in sorted(self.cache_prefixes["GENERAL"].items())
        ) + str(dt_tag)
        for p in self._state["PIPES"].values():
            _sp = _gen_str + ";".join(
                str(e) for e in sorted(self.cache_prefixes[p.name].items())
            )
            _hexdig = hashlib.sha512(_sp.encode("utf-8")).hexdigest()
            # prepend hex digest with datetime tag
            cache_digests[
                p.name
            ] = f"{str(dt_tag).replace(':', '-').replace(' ', '-')}-{_hexdig}"
        return cache_digests

    def _get_space_tensor(self, dt_tag: datetime, idx: int = None):
        worker_id = (
            torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() is not None
            else 0
        )
        start = time.clock_gettime(self.timer)
        cache_digests = None
        if self.enable_cache:
            # generate cache digests per PIPE
            cache_digests = self.generate_pipe_hashes(dt_tag)
        # get cached tensor from transform
        tensor = None
        if self.enable_cache and not self.refresh_cache:
            tensor = {"_cache_digests": cache_digests}
            for t in self.cache_space_transforms.transforms:
                tensor = t(tensor)
                LOGGER.debug(
                    f"Worker={worker_id} id={str(idx)} {t.__class__.__name__} shape={self._nested_tensor_shape(tensor)}."
                )
            if self.measure_time:
                LOGGER.info(
                    f"{self.timer_name}::{self.__class__.__name__}::__getitem__::enable_cache({dt_tag}):"
                    f" {time.clock_gettime(self.timer) - start:.5f}s"
                )
            if tensor is not None:
                LOGGER.debug(
                    f"Worker={worker_id} id={str(idx)} fp={str(dt_tag)} (cached) shape={self._nested_tensor_shape(tensor)}."
                )
                return tensor
        assert tensor is None  # just to be sure
        # setup 'result' dict
        tensor = {}
        # generate dict structure
        for p in self._state["PIPES"].values():
            tensor.__setitem__(p.name, {})
        # create empty dicts for all data groups
        for cg in self._state["sources.GROUPS"].values():
            for p in self._state["PIPES"].values():
                tensor[p.name].__setitem__(cg.__name__, {})
        # mandatory tag required by first Load-Transform
        tensor["_dt_tag"] = dt_tag
        tensor["_dataset_mode"] = self.dataset_split
        tensor["_CHANNELS"] = self._state["CHANNELS"]
        tensor["_source_parameters"] = {
            src: {
                k.replace("src.data.sources.", ""): self._state[k]
                for k in self.source_parameters[src]
            }
            for src in self.source_parameters
        }
        tensor["_channel_tags"] = {
            p.name: self.__getattribute__(f"{p.name}_channel_tags")
            for p in self._state["PIPES"].values()
        }

        # optional, add digests to tensor
        if cache_digests and tensor is not None:
            # TODO: what happens, if 'digest' don't get consumed by a transform layer?
            tensor["_cache_digests"] = cache_digests

        if self.measure_time:
            LOGGER.info(
                f"{self.timer_name}::{self.__class__.__name__}::__getitem__::_read_pkl({dt_tag}): "
                f"{time.clock_gettime(self.timer) - start:.5f}s"
            )

        start = time.clock_gettime(self.timer)
        # apply "space" transformations
        if self.transforms_space is not None:
            # manually iterate over transforms
            for t in self.transforms_space.transforms:
                tensor = t(tensor)
                LOGGER.debug(
                    f"Worker={worker_id} id={str(idx)} {t.__class__.__name__} shape={self._nested_tensor_shape(tensor)}."
                )
        if self.measure_time:
            LOGGER.info(
                f"{self.timer_name}::{self.__class__.__name__}::__getitem__::space_transforms({dt_tag}): "
                f"{time.clock_gettime(self.timer) - start:.5f}s"
            )
        return tensor

    def getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        worker_id = (
            torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() is not None
            else 0
        )
        LOGGER.debug(
            f"Worker={worker_id} id={idx} __getitem__ data=[{', '.join([str(dt_tag) for dt_tag in self.samples[idx]])}]."
        )

        tensor = [self._get_space_tensor(dt_tag, idx) for dt_tag in self.samples[idx]]
        LOGGER.debug(
            f"Worker={worker_id} id={idx} [get_space_tensor()'s] shape={self._nested_tensor_shape(tensor)}."
        )

        start = time.clock_gettime(self.timer)
        # apply "space-time" transformations
        if self.transforms_space_time is not None:
            # manually iterate over transforms
            for t in self.transforms_space_time.transforms:
                tensor = t(tensor)
                LOGGER.debug(
                    f"Worker={worker_id} id={idx} {t.__class__.__name__} shape={self._nested_tensor_shape(tensor)}."
                )
        if self.measure_time:
            LOGGER.info(
                f"{self.timer_name}::{self.__class__.__name__}::__getitem__::space_time_transforms({idx}): "
                f"{time.clock_gettime(self.timer) - start:.5f}s"
            )

        start = time.clock_gettime(self.timer)
        # apply transformations
        if self.transform is not None:
            for t in self.transform.transforms:
                tensor = t(tensor)
                LOGGER.debug(
                    f"Worker={worker_id} id={idx} {t.__class__.__name__} shape={self._nested_tensor_shape(tensor)}."
                )
        if self.measure_time:
            LOGGER.info(
                f"{self.timer_name}::{self.__class__.__name__}::__getitem__::transform({idx}): "
                f"{time.clock_gettime(self.timer) - start:.5f}s"
            )

        return tensor

    def _nested_tensor_shape(
        self, tensor
    ) -> str:  # pylint: disable=too-many-return-statements
        if isinstance(tensor, tuple):
            # return f"({', '.join([self._nested_tensor_shape(t) for t in tensor])})"
            return f"({len(tensor)},)"
        if isinstance(tensor, list):
            if len(tensor) > 20:
                return f"[{len(tensor)}...]"
            return f"[{', '.join([self._nested_tensor_shape(t) for t in tensor])}]"
        if isinstance(tensor, dict):
            return f'{{{", ".join([f"{k}: {v}" for k, v in { k: self._nested_tensor_shape(v) for k, v in tensor.items()}.items()])}}}'
        if isinstance(tensor, (torch.Tensor, np.ndarray)):
            return f"{list(tensor.shape)}"
        if isinstance(tensor, (float, int)):
            return "(1,)"
        if isinstance(tensor, datetime.datetime):
            return "obj"
        if tensor is None:
            return "None"
        return "Unknown"

    def generate_samples(self) -> List[List[datetime.datetime]]:
        # get all samples keys, which are viable for all sources
        samples = {
            source: frozenset(
                self._state[f"{source}.samples"][self.dataset_split.value]
                if isinstance(self._state[f"{source}.samples"], dict)
                and self.dataset_split.value in self._state[f"{source}.samples"]
                else self._state[f"{source}.samples"]
            )
            if self._state[f"{source}.samples"] is not None
            else None
            for source in self._state["sources.SOURCES"]
        }
        LOGGER.info(
            f"{self.__class__.__name__}:: Found {str({k: len(v) for k, v in samples.items() if v is not None})}."
        )
        samples = frozenset.intersection(
            *filter(lambda i: i is not None, samples.values())
        )
        LOGGER.info(
            f"{self.__class__.__name__}:: Found {len(samples)} overlapping samples."
        )
        if self.window_size > 1:
            samples = sorted(samples)
            # get greatest time delta of all sources
            sample_time_delta = next(
                iter(
                    sorted(
                        set(
                            self._state[f"{source}.time_delta"]
                            for source in self._state["sources.SOURCES"]
                            if f"{source}.time_delta" in self._state
                        ),
                        reverse=True,
                    )
                )
            )
            # split sample keys in continuous patches
            samples = generate_continuous_patches(samples, sample_time_delta)
            LOGGER.info(
                f"{self.__class__.__name__}:: Found {len(samples)} samples after patch generation (delta={sample_time_delta})."
            )
            # apply window transformation
            samples = apply_window(samples, window_size=self.window_size)
        elif self.window_size == 1:
            samples = [[s] for s in samples]

        LOGGER.info(
            f"{self.__class__.__name__}:: Found {len(samples)} samples after window generation."
        )

        if self.shuffle_samples:  # shuffle samples once, arbitrary constant shuffle
            random.Random(42).shuffle(samples)

        return samples  # [[dt, dt+1...], [dt, dt+1, ...], [dt, dt+1, ...]]
