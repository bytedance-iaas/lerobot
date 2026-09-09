#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`--help` has to work for every console script.

Rendering help is not free: draccus turns the comment next to each config field into an argparse
help string, and argparse then runs `%`-expansion over it. A single literal `%` in ordinary prose
(`~84% of updates`, in configuration_dreamzero.py) used to take down `--help` for lerobot-train,
lerobot-eval and lerobot-rollout — three entry points broken by a comment. Nothing caught it,
because nothing asked any entry point for its help.

Every script runs in this process rather than a subprocess: the imports dominate, and paying for
them once takes the whole suite from minutes to seconds.
"""

import contextlib
import importlib
import io
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# These parse no arguments at all, so `--help` is not theirs to answer: lerobot-info prints system
# info and returns, and lerobot-find-port ignores argv and blocks on input(). Giving them argparse
# would be a real improvement, but it is a change to what they do, not a regression guard.
NO_ARGUMENT_PARSING = {"lerobot-info", "lerobot-find-port"}


def console_scripts() -> list[tuple[str, str]]:
    """Every `[project.scripts]` entry, so a new one is covered without touching this file."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    return sorted(scripts.items())


CONSOLE_SCRIPTS = console_scripts()


@pytest.mark.parametrize("name,target", CONSOLE_SCRIPTS, ids=[name for name, _ in CONSOLE_SCRIPTS])
def test_help_renders(name: str, target: str):
    if name in NO_ARGUMENT_PARSING:
        pytest.skip(f"{name} does not parse arguments, so it has no --help to render")

    module_name, _, func_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        # The fast-test workflow installs one optional-dependency tier at a time, so a script
        # gated behind a different extra is out of scope for this run rather than broken. A
        # lerobot module that will not import is neither -- skipping there would let a genuine
        # breakage read exactly like a tier that was never installed.
        if (e.name or "").startswith("lerobot"):
            raise
        pytest.skip(f"{name} needs an extra that is not installed: {e}")

    stdout = io.StringIO()
    argv = sys.argv
    sys.argv = [name, "--help"]
    try:
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as exc_info:
            getattr(module, func_name)()
    finally:
        sys.argv = argv

    assert exc_info.value.code == 0, f"{name} --help exited {exc_info.value.code}"
    assert stdout.getvalue().strip(), f"{name} --help printed nothing"
