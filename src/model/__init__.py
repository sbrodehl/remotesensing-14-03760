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

from functools import partial

from torch import nn
from torch.nn import InstanceNorm3d

from miniflask import get_default_args

from .network import UNet as Net, PreActBlock  # pylint: disable=import-error


def activation(*args, **kwargs):
    return nn.ReLU(*args, **kwargs)


def init_filter_layer_fallback(m, altfn=None):
    del altfn
    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


def register_normalization_with_defaults(
    mf_obj, Normalization, overwrite=None, init_normalization=None, additional_args=None
):
    if not init_normalization:

        def init_normalization(state, *args):
            norm = Normalization(*args, **{k: state[k] for k in state["arg_names"]})
            mf_obj.event.optional.init_normalization(norm)
            return norm

    args = get_default_args(Normalization)
    if overwrite:
        args.update(overwrite)
    mf_obj.register_defaults(args)
    mf_obj.register_helpers(
        {
            "arg_names": list(args.keys()) + []
            if additional_args is None
            else additional_args
        }
    )
    mf_obj.register_event("normalization_layer", init_normalization)
    mf_obj.register_event("normalization_layer_cls", lambda: Normalization)


def register(mf):
    mf.register_helpers(
        {
            "kernel": [3, 3, 3],
            "stride": [1, 2, 2],
            "increase_op": "sampling",
            "inplace": True,
        }
    )
    mf.register_defaults(
        {
            "num_hierarchies": 6,
            "num_bottlenecks": 1,
            "channel_lower_factor": 1.0,
            "channel_hierarchy_depth": 256,
            "channel_bottleneck_depth": lambda state, event: state[
                "channel_hierarchy_depth"
            ]
            if "channel_hierarchy_depth" in state
            else 256,
        }
    )
    mf.register_event("basic_block", PreActBlock, unique=True)
    mf.register_event("down_block", partial(PreActBlock, stride=2), unique=True)
    mf.register_event("up_block", partial(PreActBlock, scale_factor=2), unique=True)
    mf.register_event("init_net", Net, unique=True)
    mf.register_event("activation_layer_cls", lambda: nn.ReLU, unique=True)
    mf.register_event("activation_layer", activation, unique=True)
    mf.register_event(
        "init_filter_layer_fallback", init_filter_layer_fallback, unique=True
    )
    mf.register_event(
        "register_normalization_with_defaults", register_normalization_with_defaults
    )
    mf.event.register_normalization_with_defaults(mf, InstanceNorm3d)
