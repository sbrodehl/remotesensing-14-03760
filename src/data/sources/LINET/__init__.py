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
from functools import partial
from enum import Enum
import datetime
from datetime import timezone
from typing import Callable
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import dask.array as da
import pyresample.bucket
import pyresample.kd_tree

# set logging level of pyresample
pyresample.bucket.LOG.setLevel(logging.ERROR)
pyresample.kd_tree.logger.setLevel(logging.ERROR)
from pyresample.bucket import BucketResampler

from ...sources import Channel

LOGGER = logging.getLogger(__name__)


def sample_iter(
    start: datetime.datetime, end: datetime.datetime, delta: datetime.timedelta
):
    while start <= end:
        yield start
        start += delta


def setup(state):
    LOGGER.info(f"Running {state['module_name']} setup.")
    state["db_path"]: Path = Path(state["db_path"])
    if not state["db_path"].exists():
        raise RuntimeError("Database does not exist!")
    LOGGER.info(f"DB found: {state['db_path']}")
    state["cnx"] = partial(
        sqlite3.connect,
        f"file:{str(state['db_path'])}?mode=ro",
        uri=True,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    cnx, asc, desc = None, None, None
    try:  # connect to database
        cnx = state["cnx"]()
        c = cnx.cursor()
        c.execute(
            """SELECT "Event-Time" from LINET ORDER BY "Event-Time" ASC LIMIT 1;"""
        )
        asc: datetime.datetime = c.fetchone()[0]
        asc = asc.astimezone(timezone.utc).replace(second=0, microsecond=0)
        c.execute(
            """SELECT "Event-Time" from LINET ORDER BY "Event-Time" DESC LIMIT 1;"""
        )
        desc: datetime.datetime = c.fetchone()[0] - datetime.timedelta(
            minutes=state["time_range"]
        )
        desc = desc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    except sqlite3.Error as error:
        raise RuntimeError("Error while working with SQLite", error) from error
    finally:
        if cnx:
            cnx.close()
    # create sample iterator
    state.all[f"{state['module_id']}.samples"] = list(
        sample_iter(asc, desc, state["time_delta"])
    )
    # register
    state["SOURCES"].append(state["module_name"])


def query_db(
    sql_query: str,
    cnx,
    from_timestamp: datetime.datetime,
    to_timestamp: datetime.datetime,
):
    from_timestamp_str = from_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    to_timestamp_str = to_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    _cnx = None
    try:  # connect to database
        _cnx = cnx()
        c = _cnx.cursor()
        c.execute(sql_query, (from_timestamp_str, to_timestamp_str))
        result = c.fetchall()
    except sqlite3.Error as error:
        raise RuntimeError("Error while working with SQLite", error) from error
    finally:
        if _cnx:
            _cnx.close()

    return result


def regrid_linet_data(data, **kwargs):
    target_def = pyresample.create_area_def(
        "Binned",
        {"proj": "longlat", "over": True},
        area_extent=kwargs["area_config"].area_extent_ll,
        shape=kwargs["area_config"].shape,
    )
    tt_extent = kwargs["area_config"].area_extent_ll
    lightning = data[
        (tt_extent[0] <= data["Lon"])  # noqa: W503
        & (data["Lon"] <= tt_extent[2])  # noqa: W503
        & (tt_extent[1] <= data["Lat"])  # noqa: W503
        & (data["Lat"] <= tt_extent[3])  # noqa: W503
    ]
    lightning = lightning[lightning["Ampli"] >= kwargs["LINET.min_amp"]]
    if lightning.empty:
        return np.zeros(target_def.shape, dtype=np.uint8)

    resampler = BucketResampler(
        target_def,
        da.from_array(lightning["Lon"].to_numpy()),
        da.from_array(lightning["Lat"].to_numpy()),
    )
    ret = np.zeros(target_def.shape).ravel()
    ret[resampler.idxs.compute()] = 1
    ret = ret.reshape(target_def.shape).astype(np.uint8)
    del resampler, lightning, target_def
    return ret


def get_data(channel, dt, **kwargs):
    start_date = dt + (
        datetime.timedelta(minutes=kwargs["LINET.lead_time"])
        if channel.offset is None
        else datetime.timedelta(minutes=channel.offset)
    )
    end_date = start_date + datetime.timedelta(minutes=kwargs["LINET.time_range"])
    events = query_db(
        kwargs["LINET.sql_query"], kwargs["LINET.cnx"], start_date, end_date
    )
    events = pd.DataFrame(events, columns=["EventTime", "Lat", "Lon", "Ampli"])
    events.loc[:, "Ampli"] = np.abs(events["Ampli"])  # we don't care about the sign
    events["EventTime"] = pd.to_datetime(
        events["EventTime"], utc=True
    )  # convert to timestamp
    return regrid_linet_data(events, **kwargs).copy()


class LINETChannel(Channel):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        tag: str,
        operator: Callable = None,
        normalization: Callable = None,
        loading: Callable = None,
        mean: float = 0.0,
        stddev: float = 1.0,
        classes: int = 1,
        offset: int = None,
    ):
        super().__init__(
            tag,
            "LINET",
            "LINET",
            operator,
            normalization,
            loading,
            mean,
            stddev,
            classes,
        )
        self.offset = offset


class LINET(Enum):
    LINET = LINETChannel("LINET", loading=get_data)
    PERSISTENCE = LINETChannel("PERSISTENCE", loading=get_data, offset=0)


def register(mf):
    mf.register_defaults(
        {
            "db_path": str,
            "lead_time": 15,
        }
    )
    mf.register_helpers(
        {
            "sql_query": """SELECT "Event-Time", Lat, Lon, Ampli from LINET where "Event-Time" >= ? AND "Event-Time" < ?;""",
            "cnx": None,
            "GROUPS": [LINET],
            "time_delta": datetime.timedelta(minutes=1),
            "time_range": 15,
            "min_amp": 10,
            "module_name": mf.module_name,  # required
            "module_id": mf.module_id,  # required
        }
    )
    mf.register_event("init", setup, unique=False)
    mf.register_event("get_data", get_data, unique=False)
