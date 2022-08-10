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
import torch.utils.data

logger = logging.getLogger(__name__)


def dataloader(
    state, event, subset, use_cache=True, deterministic=False
):  # noqa: C901 too-complex
    loader_name = f"{subset}_loader"
    if use_cache and loader_name in state:
        return state[loader_name]
    ds_kwargs = {}
    if subset == "train":
        ds_kwargs = {"with_transform": state["with_transform"]}
    data_set = event.dataset(subset, **ds_kwargs)
    # dis-/enable multithreading
    num_workers = state["num_workers"]
    if state["disable_multithreading"]:
        num_workers = 0
    if subset == "train":
        # dis-/enable shuffling in training
        train_shuffle: bool = False if deterministic else state["shuffle"]
        loader = torch.utils.data.DataLoader(
            data_set,
            shuffle=train_shuffle,
            batch_size=state["batchsize"],
            num_workers=num_workers,
            drop_last=state["drop_last"],
            pin_memory=True,
            sampler=None,
        )
    elif subset == "val":
        loader = torch.utils.data.DataLoader(
            data_set,
            shuffle=False,
            batch_size=state["batchsize_val"],
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
        )
    elif subset == "test":
        loader = torch.utils.data.DataLoader(
            data_set,
            shuffle=False,
            batch_size=state["batchsize_test"],
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        raise RuntimeError(f"Data subset '{subset}' unknown.")
    if use_cache:
        state[loader_name] = loader
    logger.info(f"Iterable Dataloader for subset '{subset}' loaded.")
    return loader


def register(mf):
    mf.set_scope(".")
    mf.register_event("dataloader", dataloader, unique=True)
