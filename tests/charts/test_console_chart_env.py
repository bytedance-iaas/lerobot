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

"""Container env rendering for the lerobot-agent-console chart.

The chart used to render every `env` entry as a fixed `{name, value}` pair, so a `valueFrom`
entry came out as `value: ""` — the pod started, the variable was empty, and nothing said so.
These tests pin the whole Kubernetes EnvVar shape, `valueFrom` included.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

# Resolved from this file, not the cwd: pytest is run from wherever the developer happens to be.
CHART = Path(__file__).resolve().parents[2] / "lerobot-agent-console/charts/lerobot-agent-console"

# The chart refuses to render without these two: `image.tag` is `required` and a missing
# `auth.existingSecret` hits an explicit `fail`. Every case has to supply them.
BASE_ARGS = ["--set", "image.tag=test", "--set", "auth.existingSecret=console-auth"]

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def render_env(*extra_args: str) -> list[dict]:
    """Return the console container's `env` list from a `helm template` render."""
    out = subprocess.run(
        ["helm", "template", "release", str(CHART), *BASE_ARGS, *extra_args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    statefulsets = [doc for doc in yaml.safe_load_all(out) if doc and doc.get("kind") == "StatefulSet"]
    assert len(statefulsets) == 1, f"expected exactly one StatefulSet, got {len(statefulsets)}"
    containers = statefulsets[0]["spec"]["template"]["spec"]["containers"]
    return containers[0]["env"]


def by_name(env: list[dict], name: str) -> dict:
    matches = [entry for entry in env if entry["name"] == name]
    assert len(matches) == 1, f"expected one {name} entry, got {matches}"
    return matches[0]


def test_literal_value_is_a_string():
    # Nothing coerces `value` any more, so the chart's own defaults have to already be strings —
    # the API server rejects a bare number here.
    entry = by_name(render_env(), "PORT")
    assert entry == {"name": "PORT", "value": "8080"}


def test_secret_key_ref_survives():
    # The regression case. Against the old template this came back as {"name": ..., "value": ""}.
    env = render_env(
        "--set",
        "env[0].name=HF_TOKEN",
        "--set",
        "env[0].valueFrom.secretKeyRef.name=hf-token",
        "--set",
        "env[0].valueFrom.secretKeyRef.key=token",
    )
    assert by_name(env, "HF_TOKEN") == {
        "name": "HF_TOKEN",
        "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}},
    }


def test_config_map_key_ref_survives():
    env = render_env(
        "--set",
        "env[0].name=LOG_LEVEL",
        "--set",
        "env[0].valueFrom.configMapKeyRef.name=console-config",
        "--set",
        "env[0].valueFrom.configMapKeyRef.key=log_level",
    )
    assert by_name(env, "LOG_LEVEL") == {
        "name": "LOG_LEVEL",
        "valueFrom": {"configMapKeyRef": {"name": "console-config", "key": "log_level"}},
    }


def test_empty_env_list_still_renders_the_credentials():
    # `toYaml` on an empty list emits a literal `[]`, which would collide with the two hardcoded
    # entries below it; the template guards that with `with`.
    env = render_env("--set-json", "env=[]")
    assert [entry["name"] for entry in env] == ["CONSOLE_USER", "CONSOLE_PASSWORD"]


def test_entry_order_is_preserved():
    # Reordering container env changes the pod-template hash, which would restart the consoles on
    # an upgrade for no reason. The default list must come out in values.yaml order.
    names = [entry["name"] for entry in render_env()]
    assert names[:7] == [
        "PORT",
        "CONSOLE_WORKDIR",
        "CONSOLE_SHELL",
        "HF_LEROBOT_HOME",
        "HOME",
        "HERMES_CHAT_SKILL",
        "HERMES_HOME",
    ]
