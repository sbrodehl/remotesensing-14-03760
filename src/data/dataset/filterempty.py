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

import torch


class FilterEmpty:
    """Filter samples with empty tensors."""

    def __init__(self, tags: List):
        """Checks whether the tensor of the given pipe is non-zero or not.
        Return None, if tensor is 'zero', and the sample otherwise.

        Args:
            tags: list, which pipes to check
        """
        self.tags = tags

    def __call__(self, sample):
        # if it's None, nothing to do
        if sample is None:
            return None

        if isinstance(sample, list):
            # feed all list elements to this method recursively
            return [self.__call__(e) for e in sample]
        for pipe_tag in [k for k in sample.keys() if k in self.tags]:
            if sample[pipe_tag] is None or len(sample[pipe_tag]) == 0:
                return None
            # aten::nonzero is not yet supported with named tensors -> drop named tensors for this op
            if not torch.numel(torch.nonzero(sample[pipe_tag].rename(None))) > 0:
                return None
        return sample
