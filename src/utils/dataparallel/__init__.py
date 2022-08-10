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

from torch import nn
import torch
from colored import fg, attr


def data_parallel(state, net):
    if state["device"].lower().startswith("cuda") and len(state["gpu"]) > 1:
        if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            print(
                fg("red")
                + "-> [gpu] non-deterministic/benchmark-mode does not work with multiple gpu atm. Setting to non-benchmark mode.\n\tTo disable this Warning, use any seed."
                + attr("reset")
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        net = nn.DataParallel(net, device_ids=state["gpu"], dim=0)
    return net


def register(mf):
    mf.load(["device"])
    mf.register_event("data_parallel", data_parallel, unique=True)
