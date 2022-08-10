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
import random

import torch
from torch.utils.data import IterableDataset

from .AbstractBINARYDataset import AbstractBINARYDataset

LOGGER = logging.getLogger(__name__)


class IterableBINARYDataset(
    AbstractBINARYDataset, IterableDataset
):  # pylint: disable=too-few-public-methods
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_id_str = worker.id if worker is not None else str(None)
        dataset = worker.dataset if worker is not None else self
        num_workers = worker.num_workers if worker is not None else 1
        split_size = len(dataset.samples) // num_workers
        remainder = len(dataset.samples) - (split_size * num_workers)
        start = worker_id * split_size + min(remainder, worker_id)
        split_size += 1 if remainder > 0 and worker_id < remainder else 0
        end = start + split_size
        LOGGER.info(
            f"Worker={worker_id_str} {split_size}/{len(dataset.samples)} samples (range='[{start},{end})')."
        )
        indices = list(range(start, end))
        seen_samples = 0
        if dataset.shuffle_samples:
            random.shuffle(indices)
        for idx in indices:
            ret = self.getitem(idx)
            if ret is None:
                continue
            if isinstance(ret, list):
                LOGGER.debug(
                    f"Worker={worker_id_str} id={idx} yields {len([s for s in ret if s is not None])} samples."
                )
                for s in ret:
                    if s is not None:
                        seen_samples += 1
                        yield s
            else:
                seen_samples += 1
                yield ret
        LOGGER.info(f"Worker={worker_id_str} EOL samples={seen_samples}")
