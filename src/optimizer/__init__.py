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

# define optimization techniques
import logging
import torch
from miniflask import get_default_args

LOGGER = logging.getLogger(__name__)


def eval_loss(state, event, call_after_fwd_pass=True):
    net = state["model"]
    criterion = state["criterion"]
    state["logits"] = net(state["sample"][state["PIPES"]["SOURCE"].name])
    if call_after_fwd_pass:
        event.optional.after_fwd_pass(net)
    targets = torch.squeeze(
        torch.squeeze(state["sample"][state["PIPES"]["TARGET"].name], -3), -3
    )
    current_loss = criterion(state["logits"], targets)
    lon_idx = [
        idx
        for idx, (key, enu) in enumerate(
            state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
        )
        if key == "LON"
    ][0]
    lon_cond = state["sample"][state["PIPES"]["META"].name][:, lon_idx, -1] < 9999.0
    current_loss *= lon_cond
    if "padding_px" in state and state["padding_px"]:
        padding_mask = narrow_mask = torch.zeros_like(state["logits"], dtype=torch.int8)
        for idx, px in enumerate(reversed(state["padding_px"])):
            _idx = len(narrow_mask.shape) - (idx + 1)
            narrow_mask = torch.narrow(
                narrow_mask, _idx, px, narrow_mask.shape[_idx] - 2 * px
            )
        narrow_mask += 1
        state["mask"] = padding_mask[:, -1:].unsqueeze(1).clone()
    current_loss = event.optional.loss_weighting(
        current_loss, altfn=torch.nn.Identity()
    )
    if "padding_px" in state and state["padding_px"]:
        for idx, px in enumerate(reversed(state["padding_px"])):
            _idx = len(current_loss.shape) - (idx + 1)
            current_loss = torch.narrow(
                current_loss, _idx, px, current_loss.shape[_idx] - 2 * px
            )
            targets = torch.narrow(targets, _idx, px, current_loss.shape[_idx] - 2 * px)
    with torch.no_grad():
        current_loss_var = torch.var(current_loss)
        event.optional.plot_scalar(
            f"Optimizer/{criterion.__class__.__name__}VAR", current_loss_var
        )
    current_loss = event.optional.loss_reduction(current_loss, altfn=torch.mean)
    if hasattr(event, "regularizer"):
        regularizer = event.optional.regularizer(net)
        with torch.no_grad():
            regularizer_var = torch.var(regularizer)
        regularizer = event.optional.regularizer_reduction(regularizer, altfn=sum)
        if event.plot_every():
            event.optional.plot_scalar("Optimizer/Regularizer", regularizer)
            event.optional.plot_scalar("Optimizer/RegularizerVAR", regularizer_var)
            event.optional.plot_scalar(
                f"Optimizer/{criterion.__class__.__name__}", current_loss
            )
        regularizer *= state["regularizer_factor"] / state["trainable_parameters"]
    else:
        regularizer = 0
    return event.mix_total_loss(current_loss, regularizer)


def optimize(state, event):
    optimizer = state["optimizer"]
    optimizer.zero_grad()
    total_loss = eval_loss(state, event)
    if event.plot_every():
        event.optional.plot_scalar("Optimizer/TotalLoss", total_loss)
    event.optional.backward(total_loss, altfn=lambda x: x.backward())
    optimizer.step(
        lambda: eval_loss(state, event, False)[0]
        if state["step_with_closure"]
        else None
    )
    if "padding_px" in state and state["padding_px"] and "mask" in state:
        state["logits"] *= torch.squeeze(state["mask"], -3).expand(
            state["logits"].shape
        )
        state["sample"][state["PIPES"]["TARGET"].name] *= state["mask"]
    lon_idx = [
        idx
        for idx, (key, enu) in enumerate(
            state[f'{state["PIPES"]["META"].name}_channel_tensor_order']
        )
        if key == "LON"
    ][0]
    repeats = [1] * state["logits"].ndim
    repeats[1] = state["logits"].shape[1]
    lon_cond = (
        state["sample"][state["PIPES"]["META"].name][:, lon_idx, -1:].repeat(*repeats)
        < 9999.0
    )
    state["logits"] *= lon_cond
    if event.plot_every():
        plot_step(state, event)
    return total_loss


def plot_step(state, event):
    if len(state["optimizer"].param_groups) > 1:
        for group_id, parameters in enumerate(state["optimizer"].param_groups):
            event.optional.plot_scalar2d(
                "learning rate per group", parameters["lr"], group_id
            )
    mean_lr = torch.mean(
        torch.tensor(
            [parameters["lr"] for parameters in state["optimizer"].param_groups]
        )
    )
    event.optional.plot_scalar("Optimizer/LearningRate", mean_lr)


def mix_total_loss(current_loss, regularizer):
    return current_loss + regularizer


def register_optimizer_with_defaults(
    event, mf_obj, Optimizer, with_closure=False, overwrite=None, init_optimizer=None
):
    if not init_optimizer:

        def init_optimizer(state, net, parameters=None):
            if not parameters:
                parameters = net.parameters()
            state["trainable_parameters"] = 0
            for param in net.parameters():
                state["trainable_parameters"] += param.nelement() * param.requires_grad
            LOGGER.info(
                f"Trainable parameters in model: {state['trainable_parameters']}"
            )

            def default_parameter_group(net, parameters):
                del net
                return parameters

            parameters = event.optional.make_optimizer_parameter_groups(
                net, parameters, altfn=default_parameter_group
            )
            return Optimizer(parameters, **{k: state[k] for k in state["arg_names"]})

    args = get_default_args(Optimizer)
    if overwrite:
        args.update(overwrite)
    mf_obj.register_defaults(args)
    mf_obj.register_helpers({"arg_names": args.keys()})
    if with_closure:
        mf_obj.overwrite_defaults({"step_with_closure": True})
    mf_obj.register_event("init_optimizer", init_optimizer)


def L1_regularizer(model):
    return torch.tensor([p.abs().sum() for p in model.parameters()])


def register(mf):
    mf.register_default_module("sgdw", required_event="init_optimizer")
    mf.register_defaults(
        {
            "regularizer_factor": 1e-0,
        }
    )
    mf.register_helpers(
        {
            "optimizer": None,
            "step_with_closure": False,
        }
    )
    mf.register_event("step", optimize)
    mf.register_event(
        "register_optimizer_with_defaults", register_optimizer_with_defaults
    )
    mf.register_event("mix_total_loss", mix_total_loss)
    mf.register_event("regularizer", L1_regularizer)
