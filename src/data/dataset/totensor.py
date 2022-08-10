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

import numpy as np
import torch


class ToTensor:
    """Convert nd.arrays in sample to Tensors."""

    DTYPES = {"SOURCE": torch.float32, "TARGET": torch.long}

    def __init__(self, dtype=None):
        self.dimension_names = ["C", "T", "H", "W"]
        self.dtype = dtype
        # if dtype given, set all float dtypes accordingly
        if self.dtype is not None:
            for k, v in self.DTYPES.items():
                if v.is_floating_point:
                    self.DTYPES[k] = self.dtype

    @staticmethod
    def order_def(tags: list, groups: dict) -> list:
        return [
            (kt.name, kt.value)
            for gtag in groups.keys()
            for kt in groups[gtag]
            if kt.name in tags
        ]

    @staticmethod
    def aggregate_group_tags(source, group: list):
        # combine single image
        _all_keys = [(tag, kt) for tag in group for kt in source[tag].keys()]
        if len(_all_keys) == 0:
            return None

        _dtypes = {source[k][v].dtype for k, v in _all_keys}
        _shapes = {source[k][v].shape for k, v in _all_keys}
        _dtypes_len = len(_dtypes)
        _shapes_len = len(_shapes)
        shape = None
        if _dtypes_len == 1:
            dtype = _dtypes.pop()
        else:
            # TODO: check the dtype (float > int)
            dtype = np.float32
        if _shapes_len == 1:
            shape = _shapes.pop()
        elif _shapes_len > 1:
            raise RuntimeError("Found multiple different shapes.")
        img = np.zeros(list(shape) + [len(_all_keys)], dtype=dtype)
        # put all things together
        for idx, (k, v) in enumerate(_all_keys):
            img[..., idx] = source[k][v]
        return img

    def __call__(self, sample):
        # if it's None, nothing to do
        if sample is None:
            return None

        # torch data layout: C x T x H x W
        s = {}
        cg = list(next(iter(sample.values())).keys())  # keys of groups
        assert isinstance(sample, dict)
        for pipe in (p for p in sample if not p.startswith("_")):
            img = self.aggregate_group_tags(sample[pipe], cg)
            if img is not None:
                s[pipe] = torch.from_numpy(
                    np.moveaxis(
                        np.stack(
                            [img], axis=0
                        ),  # stack 'time' dimension (not used here, but later)
                        -1,
                        0,
                    )  # move channels first
                )
                if pipe in self.DTYPES:
                    s[pipe] = s[pipe].to(self.DTYPES[pipe])
                # check available dimension names
                max_dn = (
                    s[pipe].ndim
                    if s[pipe].ndim <= len(self.dimension_names)
                    else len(self.dimension_names)
                )
                s[pipe] = s[pipe].refine_names(*self.dimension_names[:max_dn])
            else:
                # create an empty tensor
                s[pipe] = torch.tensor([])
        # there could be some things we need to pass on (e.g. hash digest)
        for key in sample.keys():
            if key.startswith("_"):
                s[key] = sample[key]
        return s
