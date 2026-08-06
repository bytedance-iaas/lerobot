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

"""CPU tests for the DreamZero config -> action-head config translation.

These cover the values that decide *which parameters train* and *how the checkpoint is wired
up*. Getting one wrong does not crash: the run trains the wrong subset, or loads nothing and
trains from random init, and only the final policy quality reveals it.
"""

import math

import pytest

pytest.importorskip("diffusers")  # dreamzero config imports through the policy package

from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig  # noqa: E402


@pytest.mark.parametrize(("mode", "expected"), [("full", "full"), ("lora", "lora")])
def test_only_lora_mode_asks_the_head_for_adapters(mode, expected):
    head = DreamZeroConfig(training_mode=mode).build_action_head_config()

    assert head.train_architecture == expected


def test_the_removed_projector_only_mode_says_what_replaces_it():
    """An old command line should get an answer, not "unknown training_mode"."""
    with pytest.raises(ValueError, match="has been removed.*training_mode='lora'"):
        DreamZeroConfig(training_mode="projector_only")


@pytest.mark.parametrize("mode", ["adapters", "frozen", ""])
def test_an_unknown_training_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="Unknown training_mode"):
        DreamZeroConfig(training_mode=mode)


@pytest.mark.parametrize("rank", [0, -1])
def test_a_rank_below_one_is_rejected(rank):
    """Rank 0 would wrap nothing, leaving only the projectors trainable by accident."""
    with pytest.raises(ValueError, match="lora_rank must be >= 1"):
        DreamZeroConfig(training_mode="lora", lora_rank=rank)


@pytest.mark.parametrize("defer", [False, True])
def test_lora_injection_can_be_deferred_past_weight_loading(defer):
    """PEFT renames every wrapped parameter, so injection order depends on the checkpoint.

    A released GEAR checkpoint stores unwrapped DiT keys and must be loaded *before* wrapping;
    a LoRA fine-tune's own checkpoint stores wrapped keys and must be wrapped first. Loading in
    the wrong order does not raise — `strict=False` reports missing keys and leaves the
    randomly-initialised weights in place.
    """
    head = DreamZeroConfig(training_mode="lora").build_action_head_config(defer)

    assert head.defer_lora_injection is defer
    assert head.train_architecture == "lora"


@pytest.mark.parametrize(("mode", "expected"), [("lora", True), ("full", False)])
def test_full_always_writes_complete_weights(mode, expected):
    """A `full` run rewrites the whole DiT, so there is no frozen remainder worth deferring.

    `lora` keeps >99% of the model frozen and bit-identical to the base, which is what makes
    storing only the trainable delta lossless.
    """
    from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy

    cfg = DreamZeroConfig(training_mode=mode, save_lora_only=True)

    assert DreamZeroPolicy._saves_adapter_only(cfg) is expected


def test_lora_hyperparameters_match_the_upstream_droid_recipe():
    """Mirrors upstream `scripts/train/droid_training_lora.sh`."""
    head = DreamZeroConfig(training_mode="lora").build_action_head_config()

    assert (head.lora_rank, head.lora_alpha) == (4, 4)
    assert head.lora_target_modules == "q,k,v,o,ffn.0,ffn.2"


def test_relative_action_false_clears_the_relative_keys():
    """`relative_action=false` must actually turn the convention off.

    The encode/decode paths key off `relative_action_keys` alone (as upstream and RLinf gate on
    both flags), so a config that keeps the keys while the flag is off would still subtract and
    re-add anchors — the flag would be read but silently not applied.
    """
    cfg = DreamZeroConfig(relative_action=False)

    assert cfg.relative_action_keys == ()
    # The default convention stays on.
    assert DreamZeroConfig().relative_action_keys == ("joint_position",)


def test_view_descriptions_must_match_video_modality_keys_length():
    """A description list of the wrong length would misname quadrants without any other symptom."""
    with pytest.raises(ValueError, match="must describe the same cameras"):
        DreamZeroConfig(
            embodiment_tag="so100",
            video_modality_keys=("observation.images.front", "observation.images.wrist"),
            view_descriptions=("front camera",),
        )


def test_optimizer_preset_matches_the_upstream_full_finetune_recipe():
    optim = DreamZeroConfig().get_optimizer_preset()

    assert optim.lr == pytest.approx(1e-5)
    assert optim.betas == (0.95, 0.999)
    assert optim.weight_decay == pytest.approx(1e-5)


def test_warmup_is_derived_from_max_steps_not_from_a_fixed_count():
    """A short run must not spend its whole budget in warmup."""
    cfg = DreamZeroConfig(max_steps=200)

    assert cfg.get_scheduler_preset().num_warmup_steps == math.ceil(200 * cfg.warmup_ratio)


def test_training_window_is_the_upstream_block_contract():
    """8 frames/block x 4 blocks + 1 boundary frame, 24 actions per block.

    These four properties are what makes LeRobot's delta indices reproduce upstream's bespoke
    multi-anchor sampler; a mismatch silently feeds the model a differently-shaped window.
    """
    cfg = DreamZeroConfig()

    assert cfg.video_frame_stride == 3
    assert cfg.training_span == cfg.action_horizon * cfg.max_chunk_size == 96
    assert len(cfg.observation_delta_indices) == cfg.num_frames == 33
    assert len(cfg.action_delta_indices) == 96
    # The last `training_span` frames of an episode cannot start a full window.
    assert cfg.drop_n_last_frames == 96
