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
import itertools
from typing import Iterable

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class PreActBlock(nn.Module):
    def __init__(
        self,
        state,
        event,
        in_planes,
        out_planes,
        *args,
        stride=1,
        kernel_size=3,
        scale_factor=None,
        final_activation=None,
        **kwargs,
    ):
        del args, kwargs  # unused
        super().__init__()
        self.event = event
        self.state = state
        self.in_planes = in_planes
        self.out_planes = out_planes
        if scale_factor is not None:
            if isinstance(scale_factor, list):
                scale_factor = tuple(scale_factor)
            self.up_sampling = nn.Upsample(scale_factor=scale_factor, mode="trilinear")
            stride = 1
        act_1 = event.activation_layer(inplace=state["inplace"])
        act_2 = event.activation_layer(inplace=state["inplace"])
        self.block_seq = nn.Sequential(
            event.normalization_layer(in_planes),
            act_1,
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            ),
            event.normalization_layer(out_planes),
            act_2,
            nn.Conv3d(
                out_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.activations = [act_1, act_2]
        self.final_activation = None
        if final_activation is not None:
            self.final_activation = final_activation
            self.activations.append(self.final_activation)
        if (
            ([1] * len(stride)) != stride if isinstance(stride, list) else stride != 1
        ) or in_planes != out_planes:
            act_3 = event.activation_layer(inplace=state["inplace"])
            self.activations.append(act_3)
            shortcut_kernel = [1] * 3
            shortcut_stride = stride if isinstance(stride, list) else [stride] * 3
            assert len(shortcut_kernel) == len(
                shortcut_stride
            ), "Kernel and Stride sizes do not match."
            if shortcut_stride != shortcut_kernel:
                if any(k < s for (k, s) in zip(shortcut_kernel, shortcut_stride)):
                    LOGGER.warning(
                        "Kernel size is (at least in one dimensions) smaller than stride. This leads to lost information."
                    )
                shortcut_kernel = [
                    k if k >= s else s
                    for (k, s) in zip(shortcut_kernel, shortcut_stride)
                ]
            self.shortcut = nn.Sequential(
                event.normalization_layer(in_planes),
                act_3,
                nn.Conv3d(
                    in_planes,
                    out_planes,
                    kernel_size=shortcut_kernel,
                    stride=shortcut_stride,
                    bias=False,
                ),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x, skip=None):
        if skip is not None:
            x = self.up_sampling(x)
            x = torch.cat((x, skip), 1)
        out = self.block_seq(x)
        out = out + self.shortcut(x)
        if self.final_activation:
            out = self.final_activation(out)
        return out


class UNet(nn.Module):
    def __init__(self, state, event):
        super().__init__()
        self.event = event
        self.state = state
        self.source_channels = state[f"num_{state['PIPES']['SOURCE'].name}s"]
        self.target_channels = state[f"num_{state['PIPES']['TARGET'].name}_classes"] + 1
        self.num_hierarchies = state["num_hierarchies"]
        self.num_bottlenecks = state["num_bottlenecks"]
        self.channel_hierarchy_depth = state["channel_hierarchy_depth"]
        self.channel_bottleneck_depth = state["channel_bottleneck_depth"]
        self.pre_block = None
        self.post_block = None
        self.classifier_block = None
        self.classifier_activation = None
        self.hierarchies_down = nn.ModuleList([])
        self.hierarchies_up = nn.ModuleList([])
        self.bottlenecks = nn.ModuleList([])
        self.setup()
        event.optional.init_net_finished(self)

    def setup(self):
        state = self.state
        event = self.event
        in_channel = base_ch = self.source_channels
        out_channel = int(
            2 ** (self.source_channels - 1).bit_length()
            * float(state["channel_lower_factor"])
        )
        self.pre_block = event.optional.pre_block(
            in_planes=in_channel, out_planes=out_channel, altfn=lambda *a, **kw: None
        )
        if self.pre_block is not None:
            in_channel = out_channel
        strategy_base_ch = (
            int(self.source_channels * float(state["channel_lower_factor"]))
            if self.pre_block is None
            else in_channel
        )
        hierarchy_channels = [
            (-1 if i == 0 else 0)
            + strategy_base_ch
            + int(
                (
                    (self.channel_hierarchy_depth - strategy_base_ch)
                    ** (1 / float(self.num_hierarchies))
                )
                ** i
            )
            for i in range(self.num_hierarchies + 1)
        ][1:]
        for _, out_channel in zip(range(self.num_hierarchies), hierarchy_channels):
            self.hierarchies_down.append(
                event.down_block(
                    in_planes=in_channel, out_planes=out_channel, stride=state["stride"]
                )
            )
            self.hierarchies_up.append(
                event.up_block(
                    in_planes=out_channel + in_channel,
                    out_planes=in_channel,
                    scale_factor=state["stride"],
                )
            )
            in_channel = out_channel
        for _ in range(self.num_bottlenecks):
            self.bottlenecks.append(
                event.basic_block(in_planes=in_channel, out_planes=in_channel)
            )
        self.post_block = event.optional.post_block(
            in_planes=base_ch,
            out_planes=self.source_channels,
            altfn=lambda *l, **kw: None,
        )
        in_channel = self.source_channels
        if self.post_block is None:
            in_channel = base_ch
        self.classifier_block = event.basic_block(
            in_planes=in_channel,
            out_planes=self.target_channels,
            stride=[self.state["window_size"], 1, 1],
            final_activation=self.classifier_activation,
        )

    def activations(self) -> Iterable[nn.ReLU]:
        return itertools.chain(
            iter(
                (a for a in self.pre_block.activations)
                if self.pre_block is not None
                else []
            ),
            iter(a for d in self.hierarchies_down for a in d.activations),
            iter(a for b in self.bottlenecks for a in b.activations),
            iter(a for u in self.hierarchies_up for a in u.activations),
            iter(
                (a for a in self.post_block.activations)
                if self.post_block is not None
                else []
            ),
        )

    def forward(self, x):
        out = x
        if self.pre_block is not None:
            out = self.pre_block(out)
        skip_con = []
        for _, down in enumerate(self.hierarchies_down):
            skip_con.append(out)
            out = down(out)
        for _, block_op in enumerate(self.bottlenecks):
            out = block_op(out)
        rev_skip = skip_con[::-1]
        for idx, up in enumerate(self.hierarchies_up[::-1]):
            out = up(out, rev_skip[idx])
        if self.post_block is not None:
            out = self.post_block(out)
        out = self.classifier_block(out)
        out = torch.squeeze(out, 2)
        return out
