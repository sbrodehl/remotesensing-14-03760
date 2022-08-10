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
import logging

from miniflask import like

from torch.optim.lr_scheduler import OneCycleLR

logger = logging.getLogger(__name__)


class AnnealStrategy(Enum):
    COS = "cos"
    QUARTER_COS = "quarter_cos"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


def after_init_optimizer(state, event):
    optimizer = event.hook["result"]
    if "num_batches" not in state:
        logger.warning(
            "Number of batches unknown. `num_batches` is not set."
            " OneCycleLR will only update once every epoch!"
        )
    steps_per_epoch = state["num_batches"] if "num_batches" in state else 1
    last_batch_index = (
        state["start_epoch"] * steps_per_epoch if state["start_epoch"] > 0 else -1
    )
    epochs = (
        state["overwrite_epochs"] if state["overwrite_epochs"] > 0 else state["epochs"]
    )
    state["scheduler"] = OneCycleLR(
        optimizer,
        state["optimizer.lr"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=state["pct_up"],
        anneal_strategy=state["anneal_strategy"].name.lower(),
        cycle_momentum=state["cycle_momentum"],
        base_momentum=state["base_momentum"],
        max_momentum=state["max_momentum"],
        div_factor=state["div_factor"],
        final_div_factor=state["final_div_factor"],
        last_epoch=last_batch_index,
    )
    if steps_per_epoch == 1:
        state["mf"].register_event("after_epoch", scheduler_step, unique=False)
    else:
        state["mf"].register_event("after_step", scheduler_step, unique=False)


def scheduler_step(state):
    state["scheduler"].step()


def register(mf):
    mf.register_defaults(
        {
            "pct_up": 0.1,
            "cycle_momentum": lambda state, event: "optimizer.momentum" in state
            and state["optimizer.momentum"] > 0,
            "base_momentum": lambda state, event: 0.85 * state["optimizer.momentum"]
            if "optimizer.momentum" in state
            else 0,
            "max_momentum": lambda state, event: state["optimizer.momentum"]
            if "optimizer.momentum" in state
            else 0,
            "div_factor": 10,
            "final_div_factor": 1e4,
            "overwrite_epochs": like("epochs", alt=0),
        }
    )
    mf.register_helpers(
        {
            "AnnealStrategy": AnnealStrategy,
            "anneal_strategy": AnnealStrategy.COS,
            "mf": mf,
        }
    )
    mf.register_event("after_init_optimizer", after_init_optimizer, unique=False)
    mf.register_event("init_scheduler", after_init_optimizer)
    mf.register_event("scheduler_step", scheduler_step, unique=False)
