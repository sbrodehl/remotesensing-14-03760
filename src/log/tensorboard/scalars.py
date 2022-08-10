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


def plot_scalar(state, tag: str, value, *args, **kwargs):
    """Add scalar data to summary.

    Args:
        state (dict): State dict
        tag (string): Data identifier
        value (float or string/blobname): Value to save

    Additional kwargs will be passed to tensorboard call.
    """
    del args  # unused
    global_step = kwargs.get("global_step", state[state["log.steptype"].value])
    if "global_step" in kwargs:
        del kwargs["global_step"]  # noqa: E701  pylint: disable=C0321
    state["tensorboard"].add_scalar(tag, value, global_step=global_step, **kwargs)


def plot_scalars(state, tag: str, tag_scalar_dict: dict, *args, **kwargs):
    del args  # unused
    global_step = kwargs.get("global_step", state[state["log.steptype"].value])
    if "global_step" in kwargs:
        del kwargs["global_step"]  # noqa: E701  pylint: disable=C0321
    state["tensorboard"].add_scalars(
        tag, tag_scalar_dict, global_step=global_step, **kwargs
    )
