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

import pickle
import logging
from typing import Any
from pathlib import Path

import torch

PICKLE_PROTOCOL = 2
LOGGER = logging.getLogger(__name__)


class CacheLayer:
    enable_cache: bool = False
    refresh_cache: bool = False
    # TODO: Add cache-path setter
    cache_path: Path = Path("./cache")

    def contains(
        self, digest: str, suffix: str = "cache", suffixes: list = None
    ) -> bool:
        """Return true if data under `digest` is cached."""
        if suffixes is None:
            suffixes = ""
        else:
            suffixes = "." + ".".join(suffixes)
        return (self.cache_path / f"{digest}{suffixes}.{suffix}").is_file()

    def set_cache(
        self, digest: str, data: Any, suffix: str = "cache", suffixes: list = None
    ) -> None:
        """Saves data under `digest` in cache."""
        if suffixes is None:
            suffixes = ""
        else:
            suffixes = "." + ".".join(suffixes)
        torch.save(
            data,
            (self.cache_path / f"{digest}{suffixes}.{suffix}"),
            pickle_module=pickle,
            pickle_protocol=PICKLE_PROTOCOL,
        )
        LOGGER.debug(f"Storing cache '{digest}'.")

    def get_cache(
        self, digest: str, suffix: str = "cache", suffixes: list = None
    ) -> Any:
        """Retrieve sample from cache."""
        if not self.contains(digest, suffix=suffix, suffixes=suffixes):
            LOGGER.debug(f"Retrieving cache '{digest}' -- not found.")
            return None
        LOGGER.debug(f"Retrieving cache '{digest}' from '{self.cache_path}'.")
        if suffixes is None:
            suffixes = ""
        else:
            suffixes = "." + ".".join(suffixes)
        try:
            return torch.load(
                (self.cache_path / f"{digest}{suffixes}.{suffix}"), pickle_module=pickle
            )
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error(f"Cache '{digest}' caused '{e}'.")
            # delete faulty cache file, if possible
            (self.cache_path / f"{digest}{suffixes}.{suffix}").unlink(missing_ok=True)
            return None
