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

"""CPU tests for the DreamZero statistics generator.

The generator exists to stop one specific silent failure: computing a new robot's action
quantiles over *absolute* actions when DreamZero normalizes *per-block displacements*. Against
the released DROID checkpoint the two differ by roughly 6x, and nothing about the resulting run
looks wrong.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")  # dreamzero config imports through the policy package

from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig  # noqa: E402
from lerobot.policies.dreamzero.processor_dreamzero import DreamZeroPackInputsStep  # noqa: E402
from lerobot.policies.dreamzero.scripts.compute_statistics import (  # noqa: E402
    _episode_bounds,
    _parse_layout,
    _placeholder_stats,
    _quantiles,
    _relative_actions,
)

_LAYOUT = [("joint_position", 7), ("gripper_position", 1)]


class _FakeTable:
    def __init__(self, episode_index):
        self._data = {"episode_index": episode_index}
        self.column_names = list(self._data)

    def __getitem__(self, key):
        return self._data[key]


class _FakeDataset:
    def __init__(self, episode_index):
        self.hf_dataset = _FakeTable(episode_index)


def test_parse_layout_reads_name_and_dim_in_order():
    assert _parse_layout("joint_position:7, gripper_position:1") == _LAYOUT


@pytest.mark.parametrize("spec", ["joint_position", "joint_position:", "joint_position:seven"])
def test_parse_layout_rejects_a_missing_or_non_numeric_dim(spec):
    """A silently-dropped dim would shift every slice after it."""
    with pytest.raises(ValueError, match="not 'name:dim'"):
        _parse_layout(spec)


def test_episode_bounds_are_offsets_into_the_loaded_subset():
    """`meta.episodes` offsets index the whole dataset, which overruns a `--episodes` subset."""
    dataset = _FakeDataset([0, 0, 0, 5, 5, 9])

    assert _episode_bounds(dataset) == [(0, 3), (3, 5), (5, 6)]


def test_quantiles_are_sliced_per_key():
    values = np.concatenate(
        [np.zeros((100, 7), dtype=np.float32), np.ones((100, 1), dtype=np.float32)], axis=1
    )

    stats = _quantiles(values, _LAYOUT, 0.01, 0.99)

    assert len(stats["joint_position"]["q01"]) == 7
    assert stats["gripper_position"]["q99"] == [1.0]


def test_relative_quantiles_describe_displacement_not_absolute_position():
    """The property the whole script exists for.

    Build actions that sit far from the origin but move only a little: absolute quantiles then
    span the offset, while the per-block displacement quantiles span the motion. Feeding the
    former to DreamZero rescales every predicted delta by their ratio.
    """
    cfg = DreamZeroConfig()
    horizon = cfg.action_horizon
    pack = DreamZeroPackInputsStep(
        config=cfg,
        stats={"state": _placeholder_stats(_LAYOUT), "action": _placeholder_stats(_LAYOUT)},
    )
    rng = np.random.default_rng(0)
    n = 256
    # Anchors scattered over [-2, 2]; within a block the action ramps 0 -> 0.1 away from its anchor.
    anchors = rng.uniform(-2.0, 2.0, size=(n, 8)).astype(np.float32)
    ramp = np.linspace(0.0, 0.1, horizon, dtype=np.float32)[None, :, None]
    actions = anchors[:, None, :] + ramp

    relative = _relative_actions(pack, actions, anchors).reshape(-1, 8)
    rel_stats = _quantiles(relative, _LAYOUT, 0.01, 0.99)
    abs_stats = _quantiles(actions.reshape(-1, 8), _LAYOUT, 0.01, 0.99)

    rel_range = np.array(rel_stats["joint_position"]["q99"]) - np.array(rel_stats["joint_position"]["q01"])
    abs_range = np.array(abs_stats["joint_position"]["q99"]) - np.array(abs_stats["joint_position"]["q01"])
    # The displacement span is the ramp (0.1); the absolute span is the anchor spread (~4).
    assert rel_range == pytest.approx(np.full(7, 0.1), abs=0.01)
    assert (abs_range > 3.5).all()


def test_relative_encode_leaves_non_relative_keys_alone():
    """`gripper_position` is an absolute command; subtracting an anchor would corrupt it."""
    cfg = DreamZeroConfig()
    pack = DreamZeroPackInputsStep(
        config=cfg,
        stats={"state": _placeholder_stats(_LAYOUT), "action": _placeholder_stats(_LAYOUT)},
    )
    anchors = np.full((4, 8), 0.5, dtype=np.float32)
    actions = np.zeros((4, cfg.action_horizon, 8), dtype=np.float32)

    relative = _relative_actions(pack, actions, anchors)

    assert torch.allclose(torch.from_numpy(relative[..., :7]), torch.full((4, cfg.action_horizon, 7), -0.5))
    assert torch.allclose(torch.from_numpy(relative[..., 7:]), torch.zeros(4, cfg.action_horizon, 1))
