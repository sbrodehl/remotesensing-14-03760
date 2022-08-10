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


def register(mf):
    mf.load(
        [
            "src.optimizer",
            "sgdw",
            "model",
            "SEVIRI",
            "LINET",
            "SPACE",
            "TIME",
            "dataset",
            "loader.iterable",
            "dataparallel",
            "imbalanced",
            "skill",
            "LRRT",
        ]
    )
    mf.overwrite_globals(
        {
            "imbalanced.enable_skill_weighting": False,
        }
    )
    mf.overwrite_globals(
        {
            "sgdw.lr": 1e-6,
            "sgdw.weight_decay": 0.0,
        }
    )
    mf.overwrite_globals(
        {
            "sources.SOURCE_channel_tags": [
                "VIS006",
                "VIS008",
                "IR_016",
                "IR_039",
                "WV_062",
                "WV_073",
                "IR_087",
                "IR_097",
                "IR_108",
                "IR_120",
                "IR_134",
                "PERSISTENCE",
            ],
            "sources.META_channel_tags": [
                "LAT",
                "LON",
                "YEAR",
                "MONTH",
                "DAY",
                "HOUR",
                "MINUTE",
            ],
            "sources.AUXILIARY_channel_tags": ["BT_062", "BT_073"],
            "sources.TARGET_channel_tags": [
                "LINET",
            ],
        }
    )
    mf.overwrite_globals(
        {
            "loader.batchsize": 64,
            "loader.shuffle": True,
            "loader.val_prop": 0.0,
            "loader.drop_last": True,
            "loader.num_workers": 4,
            "loader.disable_multithreading": False,
        }
    )
    mf.overwrite_globals(
        {
            "binary.regrid_extent": [-2.0, 21.5, 44.5, 57.5],
            "binary.window_size": 2,
            "binary.enable_cache": True,
            "binary.enable_bt_filter": True,
        }
    )
