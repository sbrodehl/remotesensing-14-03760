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
from typing import Union
from enum import Enum

import torch
from torchvision import transforms

from .utils import DatasetSplit
from .IterableBINARYDataset import IterableBINARYDataset
from .spacetobatch import SpaceToBatch
from .load import Load
from .spacetimeunion import SpaceTimeUnion
from .selecttimeframe import SelectTimeframe
from .totensor import ToTensor
from .droptensornames import DropTensorNames
from .filterempty import FilterEmpty
from .brightnesstemperaturefilter import BrightnessTemperatureFilter
from .zeroouttensors import ZeroOutTensors

LOGGER = logging.getLogger(__name__)


class SampleReduction(Enum):
    SPACETOBATCH = SpaceToBatch


def setup_channel_state(state) -> dict:
    keyword_args = {}
    # save tensor order definition of PIPES
    for p in state["PIPES"].values():
        state[f"{p.name}_channel_tensor_order"] = ToTensor.order_def(
            state[f"{p.name}_channel_tags"], state["sources.GROUPS"]
        )
        state[f"num_{p.name}s"] = len(state[f"{p.name}_channel_tensor_order"])
        # overwrite tags, since we might have filtered it
        state[f"{p.name}_channel_tags"] = [
            t[0] for t in state[f"{p.name}_channel_tensor_order"]
        ]
        if state[f"num_{p.name}s"]:
            keyword_args[f"{p.name}_channel_tags"] = state[f"{p.name}_channel_tags"]
        # save amount of classes
        classes = 0
        for _, ch in state[f"{p.name}_channel_tensor_order"]:
            classes += ch.classes
        state[f"num_{p.name}_classes"] = classes
    return keyword_args


def get_data(  # noqa: C901 too-complex
    state,
    event,
    ds_split: Union[DatasetSplit, str],
    with_transform: bool = None,
):  # pylint: disable=too-many-statements
    if "refresh_cache" in state and state["refresh_cache"]:
        state["epochs"] = 0  # set epochs to 1

    if isinstance(ds_split, str):
        ds_split = DatasetSplit(ds_split)
    bt_filter_condition = (
        "enable_bt_filter" in state
        and state["enable_bt_filter"]
        and ds_split != DatasetSplit.TRAIN
    )
    if bt_filter_condition:
        # brightness temperature needs 'BT_062' and 'BT_073' channel in 'AUXILIARY' pipe
        assert (
            "BT_062" in state["AUXILIARY_channel_tags"]
            and "BT_073" in state["AUXILIARY_channel_tags"]
        )
        # somewhere needs to be a 'LINET' channel to filter
        if not any(
            "LINET" in state[f"{p.name}_channel_tags"] for p in state["PIPES"].values()
        ):
            LOGGER.warning(
                "No 'LINET' channel found, which is needed for brightness temperature filtering."
            )
            state["enable_bt_filter"] = False

    # update search region radius based on given resolution
    # using WGS84 reference system (EPSG-Code 4326)
    density_mpd = (
        2 * torch.tensor([20037508.3428, 10018754.1714]) / torch.tensor([360.0, 180.0])
    )  # meter per degree
    mean_density_kmpp = (
        state["deg_per_px"] * density_mpd
    ).mean() * 1000**-1  # mean kilometer per pixel
    # update true kilometers
    state["image_density_km"] = mean_density_kmpp
    state["search_region_radius_km"] = (
        state["search_region_radius_px"] + 0.5
    ) * mean_density_kmpp
    LOGGER.info(
        "Computed Search Region (SR):"
        f" {state['search_region_radius_px']} px --> {state['search_region_radius_km']:.2f} km (max)"
    )
    state["true_regrid_extent"] = state["area_config"].area_extent_ll
    state["composition_shape"] = state["area_config"].shape

    space_transforms = [Load()]

    if (
        "zero_out" in state
        and len([s.strip() for s in state["zero_out"] if s.strip()]) > 0
    ):
        space_transforms += [
            ZeroOutTensors(
                [s.strip().split("::") for s in state["zero_out"] if s.strip()]
            )
        ]

    if bt_filter_condition:
        space_transforms += [BrightnessTemperatureFilter()]

    dtype = (
        torch.float16
        if ("half" in state and state["half"] and state["enable_slim_cache"])
        else None
    )
    space_transforms += [ToTensor(dtype=dtype)]

    # setup channel tags
    keyword_args = setup_channel_state(state)

    space_time_transforms = [
        SpaceTimeUnion(),
        SelectTimeframe(state["PIPES"]["TARGET"]),
    ]

    transform = []
    # get optional transforms
    with_transform_cond = (
        with_transform
        if with_transform is not None
        else (state["with_transform"] if "with_transform" in state else False)
    )
    if with_transform_cond:
        transform += [
            t for tr in event.optional.dataset_transform(ds_split) for t in tr
        ]
        if "sample_reduction" in state and state["sample_reduction"]:
            space_time_transforms += [
                state["sample_reduction"].value(
                    tuple(state["transformed_px"]),
                    padding_shape=tuple(state["padding_px"]),
                    composite_shape=list(state["area_config"].shape),
                )
            ]
            try:
                state["composite_blocks"] = space_time_transforms[-1].composite_blocks
                state["padded_composite_shape"] = space_time_transforms[
                    -1
                ].padded_composite_shape
                state["composite_padding"] = space_time_transforms[-1].composite_padding
            except AttributeError:
                pass

    # aten::stack is not yet supported with named tensors (in collate function)
    transform += [DropTensorNames()]

    if (
        "enable_filter_empty" in state
        and state["enable_filter_empty"]
        and "filter_empty" in state
        and state["filter_empty"]
        and ds_split == DatasetSplit.TRAIN
    ):
        space_time_transforms += [FilterEmpty(state["filter_empty"])]

    cache_prefixes = {
        "GENERAL": {
            "REGRID_EXTENT": list(state["regrid_extent"]),
            "DEG_PER_PX": state["deg_per_px"],
        }
    }
    for p in state["PIPES"].values():
        cache_prefixes.__setitem__(
            p.name,
            {
                "CHANNELS": [
                    c[0] for c in sorted(state[f"{p.name}_channel_tensor_order"])
                ]
            },
        )
        for ch in [c[0] for c in sorted(state[f"{p.name}_channel_tensor_order"])]:
            if ch == "LINET":
                cache_prefixes[p.name]["LINET"] = list(
                    map(
                        str,
                        [
                            state["LINET.min_amp"],
                            state["LINET.lead_time"],
                            state["LINET.time_range"],
                        ],
                    )
                )

    shffl_str = "enable_shuffle_samples"
    shuffle_samples = (
        state[shffl_str]
        if shffl_str in state and state[shffl_str] and ds_split == DatasetSplit.TRAIN
        else False
    )

    return state["DatasetClass"](
        dataset_split=ds_split,
        space_transforms=transforms.Compose(space_transforms),
        space_time_transforms=transforms.Compose(space_time_transforms),
        transform=transforms.Compose(transform),
        window_size=state["window_size"],
        refresh_cache=state["refresh_cache"],
        **keyword_args,
        measure_time=state["measure_time"],
        enable_cache=state["enable_cache"],
        cache_path=state["cache_path"],
        cache_prefixes=cache_prefixes,
        shuffle_samples=shuffle_samples,
        deg_per_px=state["deg_per_px"],
        state=state,
        event=event,
    )


def register(mf):
    mf.set_scope("..")
    # register required arguments
    mf.register_defaults(
        {
            "regrid_extent": [-10.0, 33.0, 35.0, 75.0],
            "window_size": 2,
            "deg_per_px": 0.05,
            "enable_shuffle_samples": True,
            "search_region_radius_px": 5,
            "refresh_cache": False,
            "enable_cache": True,
            "cache_path": "./cache",
            "enable_slim_cache": False,
            "enable_filter_empty": True,
            "enable_bt_filter": True,
            "zero_out": [""],
        }
    )
    mf.register_helpers(
        {
            "DatasetClass": IterableBINARYDataset,
            "filter_empty": ["TARGET"],  # TARGET must be non-empty
            "transformed_px": [256, 256],
            "padding_px": [16, 16],
            "sample_reduction": SampleReduction.SPACETOBATCH,
            "as_debug_dict": False,
            "measure_time": False,
            "search_region_radius_km": -1,
            "trainset_size": 0,
            "testset_size": 0,
            "valset_size": 0,
        }
    )
    mf.register_event("dataset", get_data, unique=True)
