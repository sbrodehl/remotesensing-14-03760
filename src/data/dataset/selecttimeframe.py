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


class SelectTimeframe:
    """Selects the target timeframe"""

    def __init__(self, pipe, select_index: int = -1):
        self.pipe = pipe
        self.select_index = select_index

    def __call__(self, sample):
        # if it's None, nothing to do
        if sample is None:
            return None

        # compute indices to retain dimension
        s_idx = (
            self.select_index + sample[self.pipe.name].shape[1]
            if self.select_index < 0
            else self.select_index
        )
        sample[self.pipe.name] = sample[self.pipe.name][:, s_idx : s_idx + 1, ...]
        return sample
