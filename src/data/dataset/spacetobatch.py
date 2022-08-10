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

from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np

from .utils import assert_shape_dimensions


def SDIV(x, y):
    return (x + y - 1) // y


def all_equal(iterator):
    """Check if all elements in the given iterator are equal (`==`)."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def SoA2AoS(sample: dict) -> list:
    """Convert a Struct of Arrays (SOA) to an Array of Structs (AOS)."""
    # check if length is equal, or sample is completely empty
    assert all_equal(len(v) for v in sample.values())
    return [
        {k: v[idx] for k, v in sample.items()}
        for idx in range(len(next(iter(sample.values()))))
    ]


class SpaceToBatch:
    """This operation divides "spatial" dimensions of the input into a grid of blocks of shape `block_shape`,
     and interleaves these blocks with the "batch" dimension (0).

    Args:
        output_shape (np.ndarray): Desired block shape (h,w).
        padding_shape (np.ndarray): Desired padding shape (h,w).
        mode (str): padding mode.
        value (dict): padding values for each `PIPE` (Default: None).
    """

    def __init__(
        self,
        output_shape: Tuple[int],
        padding_shape: Tuple[int],
        *args,
        composite_shape: Tuple[int] = None,
        mode: str = "constant",
        value: dict = None,
        **kwargs
    ):
        del args, kwargs  # unused
        self.mode = mode
        self.value = (
            {
                "META": 9999,
                "SOURCE": 0,
                "AUXILIARY": 0,
                "TARGET": 0,
            }
            if value is None
            else value
        )
        assert isinstance(output_shape, Tuple)
        assert isinstance(padding_shape, Tuple)
        self.padding_shape = np.array(padding_shape)
        self.block_shape = np.array(output_shape) - 2 * self.padding_shape
        self.composite_shape = composite_shape
        assert len(self.block_shape) == len(
            self.padding_shape
        ), "Shape sizes do not match."
        self.dimensions = len(self.block_shape)
        if self.composite_shape is not None:
            assert len(self.composite_shape) == len(
                self.block_shape
            ), "Shape sizes do not match."
            self.composite_blocks = SDIV(
                np.array(self.composite_shape), self.block_shape
            )
            self.padded_composite_shape = (
                self.composite_blocks * self.block_shape + 2 * self.padding_shape
            )
            self.composite_padding = [
                i
                for sl in [
                    ((n - a) // 2, (n - a) - ((n - a) // 2))
                    for a, n in zip(
                        reversed(self.composite_shape),
                        reversed(self.padded_composite_shape),
                    )
                ]
                for i in sl
            ]

    def __call__(self, sample: dict):
        # if it's None, nothing to do
        if sample is None:
            return None

        shape = assert_shape_dimensions(sample, self.dimensions)[0]
        assert shape.tolist() == self.composite_shape, "Unexpected shape."
        blocks = SDIV(np.array(shape), self.block_shape)
        need_shape = blocks * self.block_shape + 2 * self.padding_shape
        pad = [
            i
            for sl in [
                ((n - a) // 2, (n - a) - ((n - a) // 2))
                for a, n in zip(reversed(shape), reversed(need_shape))
            ]
            for i in sl
        ]
        for pipe in (p for p in sample if not p.startswith("_")):
            if sample[pipe].nelement() > 0:
                # aten::constant_pad_nd is not yet supported with named tensors -> drop named tensors for this op
                _names = sample[pipe].names
                sample[pipe] = sample[pipe].rename(None)
                sample[pipe] = F.pad(
                    sample[pipe], pad=pad, mode=self.mode, value=self.value[pipe]
                )
                for size, padding, idx in zip(
                    self.block_shape,
                    self.padding_shape,
                    range(
                        len(sample[pipe].shape) - self.dimensions,
                        len(sample[pipe].shape),
                    ),
                ):
                    sample[pipe] = sample[pipe].unfold(idx, size + 2 * padding, size)
                sample[pipe] = sample[pipe].reshape(
                    list(
                        list(sample[pipe].shape[: -self.dimensions * 2])
                        + [-1]
                        + list(self.block_shape + 2 * self.padding_shape)
                    )
                )
                # move stb dimension to first, and unbind
                sample[pipe] = torch.unbind(
                    torch.movedim(sample[pipe], 2, 0).rename("STB", *_names), 0
                )
        return SoA2AoS(sample)

    @staticmethod
    def _check_has_elements(sample, _slice=lambda x: x):
        for p in (p for p in sample if not p.startswith("_")):
            if p in sample and sample[p].nelement() > 0:
                if torch.is_nonzero(_slice(sample[p]).rename(None).sum()):
                    return True
        return False
