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

from typing import List

import numpy as np


class ZeroOutTensors:
    """Transform to 'zero out' given tensor names, that is, overwrite those tensors with zeros."""

    def __init__(self, zero_channels: List[List[str]]):
        self.zero_channels = zero_channels

    def __call__(self, sample):
        # if it's None, nothing to do
        if sample is None:
            return None

        # we expect a sample to be a dict
        if not isinstance(sample, dict):
            raise RuntimeError(f"Type {type(sample)} is not supported.")

        for zc in self.zero_channels:
            pipe, grp, ch = zc
            assert pipe in sample
            assert grp in sample[pipe]
            assert ch in sample[pipe][grp]
            sample[pipe][grp][ch] = np.zeros_like(sample[pipe][grp][ch])

        return sample
