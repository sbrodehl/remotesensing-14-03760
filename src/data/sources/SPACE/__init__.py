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
from enum import Enum

import numpy as np

from ...sources import Channel

LOGGER = logging.getLogger(__name__)


def setup(state):
    LOGGER.info(f"Running {state['module_name']} setup.")
    state.all[
        f"{state['module_id']}.samples"
    ] = None  # None == every time step is available
    state["SOURCES"].append(state["module_name"])


def get_data(channel, dt, **kwargs):
    del dt  # dt is not used, static content
    lon, lat = kwargs["area_config"].get_lonlats()
    return {"LON": lon, "LAT": lat}[channel.tag.upper()]


class SPACE(Enum):
    LAT = Channel(
        "LAT",
        "SPACE",
        "SPACE",
        loading=get_data,
        normalization=lambda tensor: tensor / np.float32(90.0),
    )
    LON = Channel(
        "LON",
        "SPACE",
        "SPACE",
        loading=get_data,
        normalization=lambda tensor: tensor / np.float32(180.0),
    )


def register(mf):
    mf.register_helpers(
        {
            "GROUPS": [SPACE],
            "module_name": mf.module_name,
            "module_id": mf.module_id,
        }
    )
    mf.register_event("init", setup, unique=False)
    mf.register_event("get_data", get_data, unique=False)
