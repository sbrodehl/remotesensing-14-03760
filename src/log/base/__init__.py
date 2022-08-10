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
from enum import Enum

import torch

from .util import plot_every, datestr, get_git_revisions  # pylint: disable=E0401
from .util import datestr_sort, training_header, html_summary  # pylint: disable=E0401


class StepType(Enum):
    """Enum with available log step types.
    The value maps to the available counts in the state.
    """

    BATCH = "batches_seen"
    SAMPLE = "examples_seen"
    EPOCH = "epochs_seen"


def setup_counters(state, *args, **kwargs):
    del args, kwargs  # unused
    for t in StepType:
        state[t.value] = 0


def register(mf):
    mf.set_scope("..")  # one up
    modules, shas, diffs = get_git_revisions()
    mf.register_defaults(
        {"every": 2, "dir": "./logs", "tag": "default", "steptype": StepType.BATCH}
    )
    mf.register_helpers(
        {
            "repository_state": list(zip(modules, diffs, shas)),
            "git_diffs": diffs,
            "cli_overwrites": " ".join(sys.argv),
            "date": datestr(),
            "python": sys.version.replace("\n", " "),
            "pytorch": torch.__version__,
            "loaded_modules": "",
            "StepTypeEnum": StepType,
        }
    )
    mf.register_event("plot_every", plot_every, unique=True)
    mf.register_event("before_main", setup_counters, unique=False)
    mf.register_event("before_main", training_header, unique=False)
    mf.register_event("log_html_summary", html_summary, unique=True)
    mf.register_event("log_datestr_sort", datestr_sort, unique=True)
