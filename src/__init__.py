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

__version__ = "1.0.0"
__title__ = "remotesensing-14-03760"
__description__ = (
    "End-to-End Prediction of Lightning Events from Geostationary Satellite Images."
)
__url__ = "https://github.com/sbrodehl/remotesensing-1795622"
__uri__ = __url__
__doc__ = __description__ + " <https://doi.org/10.20944/preprints202206.0238.v1>"
__documentation__ = __url__
__source__ = __url__
__tracker__ = __url__ + "/issues"
__author__ = "Sebastian Brodehl"
__license__ = "GPL-3.0"
__year__ = "2018-2022"
__copyright__ = "Copyright (c) " + __year__ + " " + __author__


def setup_logging(argv):
    import logging
    from pathlib import Path
    import socket
    import datetime

    log_tag_index, log_dir_index, log_tag = None, None, ""
    if "--log.tag" in argv:
        log_tag_index = argv.index("--log.tag")
    elif "--tag" in argv:
        log_tag_index = argv.index("--tag")
    if log_tag_index:
        log_tag = argv[log_tag_index + 1]
    if "--log.dir" in argv:
        log_dir_index = argv.index("--log.dir")
    elif "--dir" in argv:
        log_dir_index = argv.index("--dir")
    log_dir = "logs"
    if log_dir_index:
        log_dir = argv[log_dir_index + 1]
    log_pt = Path(log_dir)
    log_pt.mkdir(exist_ok=True, parents=True)
    log_pt = (
        log_pt
        / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')}_{socket.gethostname()}{('_' + log_tag) if log_tag else ''}.log"
    )
    print(f"Log file at '{log_pt}'.")
    logging.basicConfig(
        level=(logging.DEBUG if "--debug" in argv else logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        handlers=[
            logging.FileHandler(log_pt),
        ],
    )
