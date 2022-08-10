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
    # create sample iterator
    state.all[
        f"{state['module_id']}.samples"
    ] = None  # None == every time step is available
    # register
    state["SOURCES"].append(state["module_name"])


def get_data(channel, dt, **kwargs):
    try:
        res = getattr(dt, channel.tag.lower())
    except AttributeError:
        res = {
            "T__MOD": (dt.hour * 60 + dt.minute) / (24.0 * 60.0),
            "T_DOTM": dt.timetuple().tm_mday / 31.0,
            "T_DOTY": dt.timetuple().tm_yday / 366.0,
        }[channel.tag.upper()]
    return np.zeros(kwargs["area_config"].shape) + res


class TIME(Enum):
    YEAR = Channel("YEAR", "TIME", "TIME", loading=get_data)
    MONTH = Channel("MONTH", "TIME", "TIME", loading=get_data)
    DAY = Channel("DAY", "TIME", "TIME", loading=get_data)
    HOUR = Channel("HOUR", "TIME", "TIME", loading=get_data)
    MINUTE = Channel("MINUTE", "TIME", "TIME", loading=get_data)
    T__MOD = Channel("T__MOD", "TIME", "TIME", loading=get_data)
    T_DOTM = Channel("T_DOTM", "TIME", "TIME", loading=get_data)
    T_DOTY = Channel("T_DOTY", "TIME", "TIME", loading=get_data)


def register(mf):
    mf.register_helpers(
        {
            "GROUPS": [TIME],
            "module_name": mf.module_name,
            "module_id": mf.module_id,
        }
    )
    mf.register_event("init", setup, unique=False)
    mf.register_event("get_data", get_data, unique=False)
