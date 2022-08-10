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
import torch

LOGGER = logging.getLogger(__name__)


def to_device(state, t):
    return t.to(state["device"], non_blocking=True)


def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(
        f"Using device: {device} (cuda {'available' if torch.cuda.is_available() else 'not available.'})"
    )
    if device.type == "cuda":
        LOGGER.info(f"Found {torch.cuda.device_count()} CUDA devices:")
        for idx in range(torch.cuda.device_count()):
            LOGGER.info(
                f"Device: {idx} :: {torch.cuda.get_device_name(idx)}; CUDA {torch.cuda.get_device_capability(idx)}"
            )
            LOGGER.info(
                f"Memory: {round(torch.cuda.memory_allocated(idx) / 1024**3, 1)} / {round(torch.cuda.memory_reserved(idx) / 1024**3, 1)} GB"
            )
        LOGGER.info(
            f"Current CUDA device is {torch.cuda.current_device()}:'{torch.cuda.get_device_name(torch.cuda.current_device())}'"
        )


def register(mf):
    mf.register_defaults(
        {
            "gpu": [0],
            "device": lambda state, event: "cuda:" + str(state["gpu"][0])
            if state["gpu"][0] >= 0
            else "cpu",  # TODO: fix miniflask, use sane defaults
        }
    )

    mf.register_event("to_device", to_device)
    mf.register_event("init", setup, unique=False)
