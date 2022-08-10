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

from enum import Enum

import torch


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


def assert_shape_dimensions(sample, dimensions) -> list:
    shapes = [
        torch.tensor(sample[p].shape[-dimensions:])
        for p in (p for p in sample if not p.startswith("_"))
        if p != "META" and p in sample and sample[p].nelement() > 0
    ]
    cmpr = [torch.equal(shpa, shpb) for shpa in shapes[:-1] for shpb in shapes[1:]]
    if len(cmpr) > 0:
        assert torch.all(
            torch.tensor(cmpr)
        ), "Unequal tensor sizes discovered across pipes."
    assert len(shapes[0]) == dimensions, "Size of dimensions do not match!"
    return shapes
