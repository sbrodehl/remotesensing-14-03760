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


def register(mf):
    mf.set_scope(".")
    mf.register_defaults(
        {
            "batchsize": 32,
            "batchsize_test": lambda state, event: state["batchsize"]
            if "batchsize" in state
            else 32,
            "batchsize_val": lambda state, event: state["batchsize"]
            if "batchsize" in state
            else 32,
            "data_subset": 0,
            "drop_last": True,
            "val_prop": 0.05,
            "shuffle": True,
            "num_workers": torch.get_num_threads(),
            "disable_multithreading": False,
            "with_transform": True,
        }
    )
