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

from typing import List

import torch
from batchedmoments import BatchedMoments


def compute_weighting(event, class_labels, num_classes, to_one_hot=True):
    if to_one_hot:
        class_labels = event.to_one_hot_vector(class_labels, num_classes)
    sum_all = torch.sum(
        class_labels, tuple(range(1, len(class_labels.shape))), keepdim=True
    )
    sum_per_class = torch.sum(
        class_labels, tuple(range(2, len(class_labels.shape))), keepdim=True
    )
    total = torch.div(sum_all, 2.0)
    one_over_cw = torch.div(1.0, sum_per_class)
    total = torch.nan_to_num(total, posinf=0, neginf=0)
    one_over_cw = torch.nan_to_num(one_over_cw, posinf=0, neginf=0)
    return one_over_cw * total


def roc_hook(state, event) -> float:
    thresholds_ids = list(range(len(state["thresholds"])))
    pods: List[torch.Tensor] = [-1] * len(thresholds_ids)
    csis: List[torch.Tensor] = [-1] * len(thresholds_ids)
    fars: List[torch.Tensor] = [-1] * len(thresholds_ids)
    for _id in thresholds_ids:
        csis[_id], fars[_id], pods[_id] = event.Skill(
            state,
            event,
            quiet=True,
            keep_in_memory=False,
            search_region_radius_px=state["search_region_radius_px"],
            threshold=state["thresholds"][_id],
        )
    return state["thresholds"][torch.argmax(torch.stack(csis, dim=-1)[1])]


def get_class_weights(state, event, batch):
    num_classes = state[f"num_{state['PIPES']['TARGET'].name}_classes"] + 1
    return torch.softmax(
        torch.log(
            compute_weighting(
                event,
                torch.squeeze(
                    event.to_one_hot_vector(
                        batch[state["PIPES"]["TARGET"].name], num_classes
                    ),
                    -3,
                ),
                num_classes,
                to_one_hot=False,
            )
        ),
        dim=1,
    )


def class_imbalance(state, event, current_loss):
    num_classes = state[f"num_{state['PIPES']['TARGET'].name}_classes"] + 1
    one_hot_target = torch.squeeze(
        event.to_one_hot_vector(
            state["sample"][state["PIPES"]["TARGET"].name], num_classes
        ),
        -3,
    )
    threshold = roc_hook(state, event)
    if state["EMA_threshold"] is None:
        state["EMA_threshold"] = BatchedMoments(axis=())(threshold)
        state["EMA_threshold"]._n += state["EMA_init_weight"]
    else:
        state["EMA_threshold"](threshold)
    if event.plot_every():
        event.optional.plot_scalar(
            "Thresholds/Imbalanced", state["EMA_threshold"].mean.item()
        )
        event.optional.plot_scalar("Thresholds/Threshold", threshold)
    combined_vec = one_hot_target
    if "enable_skill_weighting" in state and state["enable_skill_weighting"]:
        event.Skill(
            state,
            event,
            quiet=True,
            search_region_radius_px=state["search_region_radius_px"],
        )
        combined_vec |= state["skill.fd"]
        combined_vec[:, :1, ...] -= torch.sum(
            combined_vec[:, 1:, ...], dim=1, keepdim=True, dtype=torch.bool
        ).to(combined_vec.dtype)
        combined_vec[:, :1, ...] = torch.clamp(combined_vec[:, :1, ...], min=0)
    if state["cw_training"] is not None and state["combined_weight"] is None:
        state["combined_weight"] = torch.tensor([state["cw_training"]]).to(
            combined_vec.device
        )
    combined_map = torch.sum(torch.mul(combined_vec, state["combined_weight"]), dim=1)
    return (
        current_loss
        * combined_map
        * torch.log(torch.tensor(combined_map.shape[1:]).prod())
    )


def register(mf):
    mf.register_helpers(
        {
            "enable_skill_weighting": True,
            "EMA_threshold": None,
            "EMA_init_weight": 98,
            "thresholds": [
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.875,
                0.9,
                0.9375,
                0.96875,
                0.984375,
                0.9921875,
                0.99609375,
                0.998046875,
                0.9990234375,
                0.99951171875,
                0.999755859375,
                0.9998779296875,
                0.99993896484375,
                0.999969482421875,
            ],
            "cw_training": None,
            "cw_validation": None,
            "combined_weight": None,
        }
    )
    mf.register_event("loss_weighting", class_imbalance, unique=True)
    mf.register_event("get_class_weights", get_class_weights, unique=True)
