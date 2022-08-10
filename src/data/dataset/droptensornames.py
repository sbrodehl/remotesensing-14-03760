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

import torch


class DropTensorNames:
    """Drops all tensor names."""

    def __call__(self, sample: dict):
        if sample is None:
            return None
        if isinstance(sample, list):
            # feed all list elements to this method recursively
            return [self.__call__(e) for e in sample]
        if isinstance(sample, dict):
            return {
                k: (v.rename(None) if isinstance(v, torch.Tensor) else v)
                for k, v in sample.items()
            }
        if isinstance(sample, torch.Tensor):
            return sample.rename(None)
        raise RuntimeError(f"Type {type(sample)} is not implemented (yet).")
