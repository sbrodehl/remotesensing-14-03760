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

import sys
import logging
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def model_summary(state, event):
    _model = event.hook["result"]
    get_summary(
        _model,
        torch.zeros(
            tuple(
                [
                    state["batchsize"],
                    len(state["SOURCE_channel_tensor_order"]),
                    state["window_size"],
                ]
                + state["transformed_px"]
            )
        ),
    )
    if state["exit"]:
        LOGGER.info("Exiting due to set option '--modelsummary.exit'. Goodbye!")
        sys.exit(0)


def get_summary(
    model, x, *args, **kwargs
):  # noqa: C901 too-complex pylint: disable=too-many-statements
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function

    Code is heavily borrowed from https://github.com/sksq96/pytorch-summary
    """

    def register_hook(module):
        def hook(module, inputs, outputs):
            del inputs  # unused
            cls_name = str(module.__class__).rsplit(".", maxsplit=1)[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = f"{module_idx}_{name}"

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            _ = model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T

    df["FMA"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(
        columns=dict(
            ksize="Kernel Shape",
            out="Output Shape",
        )
    )
    df_sum = df.sum(numeric_only=True)
    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "FMA"]]

    option = pd.option_context(
        "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    )
    with option:
        LOGGER.info(
            "\n" + df.replace(np.nan, "-").to_string()
        )  # pylint: disable=logging-not-lazy

    total_params = int(df_sum["Params"] + df_sum["Non-trainable params"])
    # assume same model and input type
    num_bytes = x.numpy().dtype.itemsize
    mega_byte = 1024**2.0
    total_input_size = np.prod(list(x.shape)) * num_bytes / mega_byte
    total_params_size = total_params * num_bytes / mega_byte
    total_output_size = (
        sum(np.prod(s) for s in df["Output Shape"].to_numpy()) * num_bytes / mega_byte
    )

    df_total = pd.DataFrame(
        {
            "Total params": total_params,
            "Trainable params": int(df_sum["Params"]),
            "Non-trainable params": int(df_sum["Non-trainable params"]),
            "FMA": int(df_sum["FMA"]),
            "Input Size (MB)": int(total_input_size),
            "Params Size (MB)": int(total_params_size),
            "Forward/backward-Pass Size (MB)": int(2.0 * total_output_size),
        },
        index=["Totals"],
    ).T
    LOGGER.info("\n" + df_total.to_string())  # pylint: disable=logging-not-lazy

    return df


def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).rsplit(".", maxsplit=1)[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_" + key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names


def register(mf):
    mf.register_defaults(
        {
            "exit": False,
        }
    )
    mf.register_event("after_init_net", model_summary, unique=False)
