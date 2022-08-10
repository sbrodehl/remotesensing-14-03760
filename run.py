#!/usr/bin/env python3

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


if __name__ == '__main__':
    import sys
    import miniflask
    from src import setup_logging

    setup_logging(sys.argv)
    mf = miniflask.init(
        module_dirs=["src"],
        debug="--debug" in sys.argv
    )
    mf.run(
        modules=[
            "settings",
            "system",
        ]
    )
