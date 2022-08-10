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

from typing import List, Dict

import torch

from ..sources import PIPES


class SpaceTimeUnion:
    """Union space-time tensors"""

    def __init__(self, time_index: int = 1):
        self.time_index = time_index

    def __call__(self, space_samples: List[Dict]):
        # if it's None, nothing to do
        if space_samples is None:
            return None

        return {
            p.name: torch.cat([s[p.name] for s in space_samples], dim=self.time_index)
            for p in PIPES
        }
