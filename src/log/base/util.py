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
from typing import Tuple
import datetime
from os.path import basename
from html import escape

import git
from colored import fg, attr, colors

LOGGER = logging.getLogger(__name__)


def rainbow(s: str) -> str:
    """Colorizes the input in random colors."""
    _now = datetime.datetime.now
    _clrs = len(colors.names)
    return "".join(
        [
            f"{fg(colors.names[(hash(s[0]) % _now().microsecond) % _clrs].lower())}"
            f"{s}"
            f"{attr('reset')}"
            for s in iter(s)
        ]
    )


def datestr_sort() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")


def datestr() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def get_git_revisions() -> Tuple:
    repo = git.Repo(search_parent_directories=True)
    name = basename(repo.working_dir)
    sha = [repo.head.object.hexsha]
    diffs = [repo.git.diff("HEAD")]
    modules = [name]
    if len(repo.submodules) > 0:
        modules += [s.name for s in repo.submodules]
        sha += [s.hexsha for s in repo.submodules]
        diffs += [s.module().git.diff("HEAD") for s in repo.submodules]
    return modules, sha, diffs


def training_header(state, *args, **kwargs):  # args, kwargs are inputs of `main` event
    device = (
        (
            f"gpu:{state['gpu'][0]}"
            if len(state["gpu"]) == 1
            else f"multi-gpu:({','.join(map(str, state['gpu']))})"
        )
        if (
            "device" in state
            and state["device"].lower().startswith("cuda")
            and "gpu" in state
        )
        else "cpu"
    )
    s = [" ", "Experiment", state["tag"], "on", device, " "]
    seed_mode = f"seed: {state['seed']} " if "seed" in state else rainbow("random-mode")
    bar = "—" * len(" ".join(s))  # pylint: disable=blacklisted-name
    s[1] = s[1]
    s[2] = fg("red") + attr("bold") + s[2] + attr("reset")
    s[3] = attr("dim") + s[3] + attr("reset")
    s[4] = fg("red") + attr("bold") + s[4] + attr("reset")
    print(" ╭" + bar + "╮")
    print(" │" + " ".join(s) + "│", attr("dim") + seed_mode + attr("reset"))
    print(" ╰" + bar + "╯")
    if "record" in state and state["record"]:
        print(fg("red") + "     Recording Log-Calls" + attr("reset"))
    print(f"Logs saved in '{state['log.dir']}'")
    return args, kwargs


def html_summary(state, event) -> Tuple[str, str]:
    html_repo_state = (
        "<ul style='list-style: circle'>"
        + (
            "".join(
                f"<li style='margin:0 3em;'>{name}:{'clean' if len(diff) == 0 else '<b>diverged</b>'}:<code>{sha[:7]}</code></li>"
                for (name, diff, sha) in state["repository_state"]
            )
        )
        + "</ul>"
    )
    html_loaded_modules = (
        "<ul style='list-style: circle'>"
        + (
            "".join(
                f"<li style='margin:0 3em;'>{s}</li>" for s in state["loaded_modules"]
            )
        )
        + "</ul>"
    )
    html_env = (
        "<ul style='list-style: circle'>"
        + (
            "".join(
                f"<li style='margin:0 3em;'>{name}: <code>{ver}</code></li>"
                for (name, ver) in [
                    ("python", state["python"]),
                    ("pytorch", state["pytorch"]),
                ]
            )
        )
        + "</ul>"
    )
    html_prepend = f"""
<h1>Experiment on {state["date"]}</h1>
<h1 style="font-size:120%%; margin-top: -0.25em;">{state["tag"]}</h1>
<b>Repository Status:</b></br> {html_repo_state} </br></br>
<b>CLI-Call:</b></br> <code><pre>{state["cli_overwrites"]}</pre></code> </br></br>
<b>Loaded Modules:</b></br> {html_loaded_modules} </br></br>
<b>Environment:</b></br> {html_env} </br></br>
    """
    html_diffs = "\n".join(
        f"""
<h1>Repository Diffs</h1>
<b><b>{module}</b>:</b></br> <code><pre>{escape(diff)}</pre></code> </br></br>
    """
        for module, diff, sha in state["repository_state"]
    )
    html_settings = html_prepend + "".join(event.settings_html())
    return html_settings, html_diffs


def plot_every(state, steps=None, steptype=None) -> bool:
    if steps is None:
        steps = state["log.every"]
    if steptype is None:
        steptype = state["log.steptype"]
    assert steptype.value in state, f"StepType ({steptype}) not known"
    return steps and state[steptype.value] % steps == 0
