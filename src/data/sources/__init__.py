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
from typing import Callable
from collections import OrderedDict
from enum import Enum
from pyresample import parse_area_file

LOGGER = logging.getLogger(__name__)
GROUPS = []


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


class PIPES(Enum):
    META = 0
    SOURCE = 1
    AUXILIARY = 2
    TARGET = 3


class InputGroup:  # pylint: disable=too-few-public-methods
    pass


class Channel:  # pylint: disable=too-few-public-methods
    __slots__ = (
        "tag",
        "source",
        "group",
        "operator",
        "normalization",
        "loading",
        "mean",
        "stddev",
        "classes",
        "name",
    )

    def __init__(
        self,
        tag: str,
        source: str,
        group: str,
        operator: Callable = None,
        normalization: Callable = None,
        loading: Callable = None,
        mean: float = 0.0,
        stddev: float = 1.0,
        classes: int = 1,
        name: str = None,
    ):
        self.tag = tag
        self.source = source
        self.group = group
        self.mean = mean
        self.stddev = stddev
        self.classes = classes
        self.name = name
        if operator is None or isinstance(
            operator, Callable
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            self.operator = operator
        else:
            raise RuntimeError(f"operator type ({type(operator)}) not supported.")
        if normalization is None or isinstance(
            normalization, Callable
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            self.normalization = normalization
        else:
            raise RuntimeError(
                f"normalization type ({type(normalization)}) not supported."
            )
        if loading is None or isinstance(
            loading, Callable
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            self.loading = loading
        else:
            raise RuntimeError(f"loading type ({type(loading)}) not supported.")

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.name if self.name is not None else '-'},"
            f"{self.source},"
            f"{self.group},"
            f"{self.tag},"
            f" classes={self.classes},"
            f" mean={self.mean},"
            f" stddev={self.stddev}"
            f")"
        )


def setup(state):
    LOGGER.info("Running SOURCES setup.")
    state["area_config"] = parse_area_file(state["area_yaml"], "dfcc_europe")[0]


def post_setup(state):
    global GROUPS  # pylint: disable=global-statement
    LOGGER.info("Running SOURCES after setup.")
    _members = [
        (grp.__name__, grp)
        for source in state["SOURCES"]
        for grp in state[f"{source}.GROUPS"]
    ]
    GROUPS = state["GROUPS"] = OrderedDict(
        sorted({n: v for n, v in _members}.items())  # pylint: disable=unnecessary-comprehension
    )
    channel_dict = OrderedDict(
        {ch.name: ch for _, v in state["GROUPS"].items() for ch in v}
    )
    ch_list = [ch for _, v in state["GROUPS"].items() for ch in v]
    if len(ch_list) != len(channel_dict):
        raise RuntimeError("Non-unique channel tags found!")
    for ch in channel_dict:
        channel_dict[ch].value.name = ch
    state["CHANNELS"] = channel_dict
    state["PIPES"] = OrderedDict({p.name: p for p in PIPES})


def register(mf):
    mf.set_scope(".")
    mf.register_defaults({f"{p.name}_channel_tags": ([str]) for p in PIPES})
    mf.register_helpers(
        {
            "DatasetSplit": DatasetSplit,
            "area_yaml": "src/data/sources/dfcc_roi.yaml",
            "GROUPS": [],
            "SOURCES": [],
        }
    )
    mf.register_event("init", setup, unique=False)
    mf.register_event("before_main", post_setup, unique=False)
