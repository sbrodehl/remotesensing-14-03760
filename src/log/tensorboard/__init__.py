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

import socket
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# pylint: disable=E0401
from .scalars import plot_scalar, plot_scalars
from .images import plot_image, plot_images, plot_image_with_boxes

# pylint: enable=E0401


def text_summary(state, event):
    html_settings, html_diffs = event.log_html_summary()
    with open(
        Path(state["dir"]) / "experiment_settings.html", "w", encoding="utf-8"
    ) as writer:
        writer.write(html_settings)
        writer.flush()
    with open(
        Path(state["dir"]) / "experiment_diffs.html", "w", encoding="utf-8"
    ) as writer:
        writer.write(html_diffs)
        writer.flush()


def init_tensorboard_writer(state, event):
    state.default.update(
        {state.module_id + "." + k: state[k] for k in ["python", "pytorch"]}
    )
    state["loaded_modules"] = list(event._mf.modules_loaded.keys())
    log_dir = Path(state["dir"]) / (
        event.log_datestr_sort() + "_" + socket.gethostname() + "_" + state["tag"]
    )
    state["dir"] = str(log_dir.absolute().resolve())
    if "tensorboard" not in state:
        state["tensorboard"] = SummaryWriter(state["dir"])
    text_summary(state, event)


def flush_log(state, *args, **kwargs):
    """Flushes the event file to disk.
    All pending events will be written to disk.
    """
    del args, kwargs  # unused
    state["tensorboard"].flush()


def close_file(state, *args, **kwargs):
    """Closes the event file."""
    del args, kwargs  # unused
    state["tensorboard"].close()


def register(mf):
    mf.load("..base")
    mf.set_scope("..")  # one up
    mf.register_defaults(
        {
            "batched_data_no": 1,
            "slim": True,
        }
    )
    mf.register_event("init", init_tensorboard_writer, unique=False)
    mf.register_event("after_epoch", flush_log, unique=False)
    mf.register_event("after_training", flush_log, unique=False)
    mf.register_event("after_testing", flush_log, unique=False)
    mf.register_event("after_main", close_file, unique=False)
    mf.register_event("plot_img", plot_image, unique=False)
    mf.register_event("plot_imgs", plot_images, unique=False)
    mf.register_event("plot_img_w_boxes", plot_image_with_boxes, unique=False)
    mf.register_event("plot_scalar", plot_scalar, unique=False)
    mf.register_event("plot_scalars", plot_scalars, unique=False)
