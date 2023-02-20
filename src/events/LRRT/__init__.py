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

import matplotlib.pyplot as plt
import torch
from .lr_finder import LRFinder, TrainDataLoaderIter


def main(state, event):
    train_loader = event.dataloader(subset="train", use_cache=False, deterministic=True)
    state["model"] = event.init_net()
    state["model"] = event.optional.to_device(state["model"], altfn=lambda m: m)
    state["model"] = event.optional.data_parallel(state["model"], altfn=lambda m: m)
    model_name = state["model"].__class__.__name__
    state["criterion"] = torch.nn.CrossEntropyLoss()
    state["criterion"] = event.optional.to_device(state["criterion"], altfn=lambda c: c)
    criterion_name = state["criterion"].__class__.__name__
    state["optimizer"] = event.init_optimizer(state["model"])
    optimizer_name = state["optimizer"].__class__.__name__
    event.optional.before_training()

    class TrainIter(TrainDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data):
            return batch_data["SOURCE"], torch.squeeze(
                torch.squeeze(batch_data["TARGET"], -3), -3
            )

    train_data_iter = TrainIter(train_loader)
    lrf = LRFinder(state["model"], state["optimizer"], state["criterion"])
    lrf.range_test(
        train_data_iter,
        end_lr=state["end_lr"],
        num_iter=state["num_iter"],
        smooth_f=0.05,
        diverge_th=150,
    )
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey="row", figsize=(10, 5), dpi=300)
    fig.suptitle(
        f"LRRT {optimizer_name} {criterion_name} {model_name}(w{state['binary.window_size']}c{state['channel_bottleneck_depth']}lt{state['LINET.lead_time']}sr{state['search_region_radius_px']})"
    )
    _, suggested_lr = lrf.plot(ax=ax1)
    ax1.set_title("Overview")
    lrf.reset()
    lrf.range_test(
        train_data_iter,
        start_lr=suggested_lr,
        end_lr=state["end_lr"],
        num_iter=state["num_iter"],
        smooth_f=0.0,
        diverge_th=150,
    )
    lrf.plot(
        skip_start=0, skip_end=0, suggest_lr=False, ax=ax2
    )  # to inspect the loss-learning rate graph
    ax2.set_title("Detail")
    fig.savefig(
        f"LRRT-{optimizer_name}-{criterion_name}-{model_name}-w{state['binary.window_size']}-c{state['channel_bottleneck_depth']}-lt{state['LINET.lead_time']}-sr{state['search_region_radius_px']}.{state['image_format']}"
    )
    lrf.reset()


def register(mf):
    mf.set_scope("main")
    mf.register_defaults(
        {
            "end_lr": 100,
            "num_iter": 256,
            "image_format": "png",
        }
    )
    mf.register_event("main", main, unique=True)
    mf.register_event("LRFinder", LRFinder, unique=True)
