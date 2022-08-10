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

import numpy as np


class BrightnessTemperatureFilter:
    """Filter samples according to brightness temperature difference."""

    def __call__(self, sample):
        # if it's None, nothing to do
        if sample is None:
            return None

        if not isinstance(sample, dict):
            raise RuntimeError(f"Type {type(sample)} is not supported.")
        # https://doi.org/10.3390/rs11040443
        # all regions where the brightness temperature differences
        # of the WV channels (WV6.2 - WV7.3) were above −1 degree were excluded
        cond = np.where(
            (sample["AUXILIARY"]["BT"]["BT_062"] - sample["AUXILIARY"]["BT"]["BT_073"])
            > -1,
            False,
            True,
        )
        for pipe in sample:
            if (
                "LINET" in sample[pipe]
                and sample[pipe]["LINET"] is not None
                and len(sample[pipe]["LINET"])
            ):
                for ch in sample[pipe]["LINET"]:
                    if sample[pipe]["LINET"][ch] is not None and len(
                        sample[pipe]["LINET"][ch]
                    ):  # noqa: E501 line-too-long
                        dtype = sample[pipe]["LINET"][ch].dtype
                        sample[pipe]["LINET"][ch] *= cond.astype(dtype)
        return sample
