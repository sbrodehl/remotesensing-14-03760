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

import logging
from typing import Optional, Callable

from .cache import CacheLayer

LOGGER = logging.getLogger(__name__)


def select_channels(tags: list, to_obj: dict, from_obj: dict, meta_obj: dict):
    for tag in tags:
        ch = meta_obj[tag].value
        ch_group_name = ch.group
        # skip group, if not in current file
        # will be loaded later on
        if ch_group_name not in from_obj:
            continue
        if isinstance(from_obj[ch_group_name], dict):
            tmp = from_obj[ch_group_name][tag]
        else:
            tmp = from_obj[ch_group_name]
        if tmp is None:
            continue  # assume channel will be loaded later on
        if ch.operator is not None:
            tmp = ch.operator(tmp)
        if ch.normalization is not None:
            tmp = ch.normalization(tmp)
        to_obj[ch_group_name][tag] = tmp


class Load(CacheLayer):
    """Load the data."""

    def __call__(
        self, sample: Optional[dict]
    ):  # noqa: C901 too-complex pylint: disable=too-many-statements
        # if it's None, nothing to do
        if sample is None:
            return None

        # check if cache is available & check if we can LOAD all pipes successfully
        if (
            self.enable_cache
            and not self.refresh_cache
            and "_cache_digests" in sample
            and len(sample) == 1
        ):
            _PIPES = sample["_cache_digests"].keys()
            # TODO: check 'contains' first, only load all pipes, iff all are available
            cached = {
                pipe: self.get_cache(sample["_cache_digests"][pipe], suffixes=[pipe])
                for pipe in _PIPES
            }
            for (
                pipe
            ) in (
                _PIPES
            ):  # Check if any pipe is None, which results in (partially) rebuild of tensor
                if cached[pipe] is None:
                    return None
            LOGGER.debug(
                f"{''.join(sample['_cache_digests'][next(iter(_PIPES))].split('-')[:-1])} - cached"
            )
            return cached  # looks like all pipes are available, good to go!
        # else:
        # STORE pipes after tensor is computed, possibly using cached pipes, if available
        LOGGER.debug(f"{sample['_dt_tag']}")

        _PIPES = list(
            k for k in sample.keys() if not k.startswith("_")
        )  # PIPES excluding private members
        _GROUPS = sample[next(iter(_PIPES))].keys()
        _CHANNELS = sample.get("_CHANNELS", None)
        _source_parameters = sample.get("_source_parameters", {})
        _channel_tags = sample.get("_channel_tags", {})
        _dt_tag = sample["_dt_tag"]
        _mode = sample["_dataset_mode"]
        del (
            sample["_channel_tags"],
            sample["_source_parameters"],
            sample["_CHANNELS"],
            sample["_dt_tag"],
            sample["_dataset_mode"],
        )

        # load needed raw data
        raw_data = {g: {} for g in _GROUPS}
        _channels_needed = (
            set(
                _CHANNELS[ch]
                for pipe in iter(_PIPES)
                for ch in _channel_tags[pipe]
                if "_cache_digests" in sample
                and not self.contains(sample["_cache_digests"][pipe], suffixes=[pipe])
            )
            if self.enable_cache and not self.refresh_cache
            else set(
                _CHANNELS[ch] for pipe in iter(_PIPES) for ch in _channel_tags[pipe]
            )
        )
        for ch in _channels_needed:
            chan = ch.value
            if chan.loading is not None and isinstance(
                chan.loading, Callable
            ):  # pylint: disable=isinstance-second-argument-not-valid-type
                raw_data[chan.group][ch.name] = chan.loading(
                    chan, _dt_tag, mode=_mode, **_source_parameters[chan.source]
                )

        for pipe in iter(_PIPES):
            _cached_pipe = (
                self.get_cache(sample["_cache_digests"][pipe], suffixes=[pipe])
                if self.enable_cache
                and not self.refresh_cache
                and "_cache_digests" in sample
                else None
            )

            if _cached_pipe is not None:  # Reuse partial cache for given pipe
                sample[pipe] = _cached_pipe
                del sample["_cache_digests"][pipe]
                continue

            select_channels(
                _channel_tags[pipe],
                sample[pipe],
                raw_data,
                _CHANNELS,
            )

        # cache pipes of sample, since (some) are not cached yet
        if self.enable_cache and "_cache_digests" in sample:
            _to_del = set()
            for pipe in sample["_cache_digests"]:
                self.set_cache(
                    sample["_cache_digests"][pipe], sample[pipe], suffixes=[pipe]
                )
                _to_del.add(pipe)  # mark digest for deletion
            for dig in _to_del:
                del sample["_cache_digests"][dig]
            assert (
                len(sample["_cache_digests"]) == 0
            ), "Some cache digests aren't consumed."
            del sample["_cache_digests"]

        return sample
