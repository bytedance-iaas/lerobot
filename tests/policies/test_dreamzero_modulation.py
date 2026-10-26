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

"""CPU tests for DreamZero's per-token modulation.

The DiT's time embedding has one row per *distinct timestep* -- 9 latent frames, plus one per
action and state register -- but every block used to modulate a tensor expanded to one row per
*token*, which at a 15,940-token training window is 934 MiB per block held for the backward
pass. The blocks now broadcast each row over the tokens that share it.

What these pin down: the broadcast arithmetic matches the expanded arithmetic element for
element, the run layout covers exactly the tokens it claims to, and a real training forward
builds rows at frame resolution rather than token resolution.
"""

import pytest
import torch

from lerobot.policies.dreamzero.wan.causal_wan_model import (
    CausalWanModel,
    _join_runs,
    _modulation_run,
    _per_run,
    gate_tokens,
    modulate_tokens,
)

# (runs, total tokens) -- the third is the real DROID training layout in miniature: a
# teacher-forced clean half on one row per frame, the noisy half likewise, then the action and
# state registers at one row each.
LAYOUTS = [
    (((4, 1),), 4),
    (((3, 5),), 15),
    (((3, 5), (3, 5), (7, 1)), 37),
    (((1, 12), (2, 6), (5, 1)), 29),
]


def _expand(rows: torch.Tensor, runs) -> torch.Tensor:
    """The tensor the blocks used to be handed: one row per token."""
    wide = []
    offset = 0
    for count, repeat in runs:
        wide.append(rows[:, offset : offset + count].repeat_interleave(repeat, dim=1))
        offset += count
    return torch.cat(wide, dim=1)


@pytest.mark.parametrize("runs,tokens", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_modulation_matches_the_expanded_form_bit_for_bit(runs, tokens, dtype):
    torch.manual_seed(0)
    rows = sum(count for count, _ in runs)
    x = torch.randn(2, tokens, 8, dtype=dtype)
    scale = torch.randn(2, rows, 8, dtype=dtype)
    shift = torch.randn(2, rows, 8, dtype=dtype)

    wide_scale, wide_shift = _expand(scale, runs), _expand(shift, runs)
    assert torch.equal(modulate_tokens(x, runs, scale, shift), x * (1 + wide_scale) + wide_shift)
    assert torch.equal(gate_tokens(x, runs, scale), x * wide_scale)


@pytest.mark.parametrize("runs,tokens", LAYOUTS)
def test_the_gradient_reaches_every_row(runs, tokens):
    """The rows come off `time_projection`, so the backward pass has to reach them -- and has
    to agree with what the expanded form produced. In float64 the reduction order the
    broadcast changes is far below tolerance; in bf16 it is worth one last bit."""
    rows = sum(count for count, _ in runs)
    x = torch.randn(1, tokens, 4, dtype=torch.float64)
    compact = [torch.randn(1, rows, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    expanded = [row.detach().clone().requires_grad_() for row in compact]

    modulate_tokens(x, runs, *compact).square().sum().backward()
    reference = x * (1 + _expand(expanded[0], runs)) + _expand(expanded[1], runs)
    reference.square().sum().backward()

    for got, want in zip(compact, expanded, strict=True):
        assert want.grad.abs().min() > 0  # no row left out
        assert torch.allclose(got.grad, want.grad)


def test_a_single_one_to_one_run_is_passed_straight_through():
    """The inference fallback keeps one row per token; it must not pay for slicing and cat."""
    x = torch.randn(1, 6, 3)
    rows = torch.randn(1, 6, 3)
    assert torch.equal(gate_tokens(x, ((6, 1),), rows), x * rows)


def test_runs_must_cover_x():
    x = torch.randn(1, 10, 2)
    with pytest.raises(ValueError, match="do not cover"):
        _per_run(x, ((3, 3),), lambda t, r: t * r, torch.randn(1, 3, 2))


def test_a_run_must_divide_evenly():
    assert _modulation_run(9, 7920) == (9, 880)
    with pytest.raises(ValueError, match="do not divide"):
        _modulation_run(7, 7920)


def test_join_runs_merges_only_the_one_to_one_neighbours():
    assert _join_runs((9, 880), (96, 1), (4, 1)) == ((9, 880), (100, 1))
    assert _join_runs((9, 880), (9, 880)) == ((9, 880), (9, 880))
    assert _join_runs((8020, 1), (100, 1)) == ((8120, 1),)


# --- the real forward -------------------------------------------------------------------
# A 2-block, dim-32 stand-in for the 14 B DiT: same code path, same run bookkeeping, 1 MB of
# parameters instead of 16.5 GB.

TINY = {
    "model_type": "i2v",
    "patch_size": (1, 2, 2),
    "frame_seqlen": 6,
    "text_len": 8,
    "in_dim": 4,
    "dim": 32,
    "ffn_dim": 64,
    "freq_dim": 16,
    "text_dim": 24,
    "out_dim": 4,
    "num_heads": 4,
    "num_layers": 2,
    "max_chunk_size": 2,
    "num_frame_per_block": 2,
    "action_dim": 5,
    "max_state_dim": 7,
    "hidden_size": 16,
    "num_action_per_block": 4,
    "num_state_per_block": 1,
    "concat_first_frame_latent": False,
}
LATENT_FRAMES = 3
VIDEO_TOKENS = 18  # 3 frames x 6 tokens
REGISTERS = 5  # 4 action + 1 state


def _tiny_forward(model: CausalWanModel):
    torch.manual_seed(1)
    kwargs = {
        "x": torch.randn(1, 4, LATENT_FRAMES, 4, 6, dtype=torch.bfloat16),
        "timestep": (torch.rand(1, LATENT_FRAMES) * 1000).to(torch.bfloat16),
        "timestep_action": (torch.rand(1, 4) * 1000).to(torch.bfloat16),
        "context": torch.randn(1, 8, 24, dtype=torch.bfloat16),
        "seq_len": VIDEO_TOKENS,
        "clean_x": torch.randn(1, 4, LATENT_FRAMES, 4, 6, dtype=torch.bfloat16),
        "y": torch.randn(1, 4, LATENT_FRAMES, 4, 6, dtype=torch.bfloat16),
        "clip_feature": torch.randn(1, 3, 1280, dtype=torch.bfloat16),
        "action": torch.randn(1, 4, 5, dtype=torch.bfloat16),
        "state": torch.randn(1, 1, 7, dtype=torch.bfloat16),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }
    return model._forward_train(**kwargs)


@pytest.fixture
def tiny() -> CausalWanModel:
    torch.manual_seed(0)
    return CausalWanModel(**TINY).to(torch.bfloat16)


def test_the_training_forward_keeps_rows_at_frame_resolution(tiny, monkeypatch):
    """What the blocks are handed, which is the whole point: rows, not tokens."""
    seen = []
    original = type(tiny.blocks[0]).forward

    def record(self, x, *args, **kwargs):
        seen.append((x.shape[1], kwargs["e"].shape[1], kwargs["e_runs"]))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(type(tiny.blocks[0]), "forward", record)
    _tiny_forward(tiny)

    tokens = 2 * VIDEO_TOKENS + REGISTERS
    rows = 2 * LATENT_FRAMES + REGISTERS
    assert len(seen) == TINY["num_layers"]
    for seen_tokens, seen_rows, runs in seen:
        assert (seen_tokens, seen_rows) == (tokens, rows)
        # clean half, noisy half, then the registers one row each.
        assert runs == ((LATENT_FRAMES, 6), (LATENT_FRAMES, 6), (REGISTERS, 1))
        assert sum(count * repeat for count, repeat in runs) == tokens


def test_the_dial_does_not_change_the_training_forward(tiny):
    """Recompute is still exact with the modulation broadcast rather than expanded."""
    tiny.gradient_checkpointing = False
    plain = [t.detach().clone() for t in _tiny_forward(tiny)]
    tiny.gradient_checkpointing = True
    for before, after in zip(plain, _tiny_forward(tiny), strict=True):
        assert torch.equal(before, after)


def test_an_indivisible_window_is_rejected_rather_than_repeated(tiny):
    """The old code silently `repeat_interleave`d a mismatch into shape. This one raises."""
    with pytest.raises(ValueError, match="do not divide"):
        tiny._forward_train(
            x=torch.randn(1, 4, LATENT_FRAMES, 4, 6, dtype=torch.bfloat16),
            timestep=(torch.rand(1, 4) * 1000).to(torch.bfloat16),  # 4 rows, 18 tokens
            timestep_action=(torch.rand(1, 4) * 1000).to(torch.bfloat16),
            context=torch.randn(1, 8, 24, dtype=torch.bfloat16),
            seq_len=VIDEO_TOKENS,
            y=torch.randn(1, 4, LATENT_FRAMES, 4, 6, dtype=torch.bfloat16),
            clip_feature=torch.randn(1, 3, 1280, dtype=torch.bfloat16),
            embodiment_id=torch.zeros(1, dtype=torch.long),
        )


def test_the_inference_forward_keeps_rows_at_frame_resolution(tiny, monkeypatch):
    """The KV-cache path builds its own `e0`, so it needs its own run layout.

    Video only: the action registers need the real action horizon for their rope buffers, and
    a toy geometry cannot supply it (that is so before this change too).
    """
    seen = []
    original = type(tiny.blocks[0]).forward

    def record(self, x, *args, **kwargs):
        seen.append((x.shape[1], kwargs["e"].shape[1], kwargs["e_runs"]))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(type(tiny.blocks[0]), "forward", record)
    frames, tokens = 2, 12  # 2 latent frames x 6 tokens, the inference window
    with torch.no_grad():
        video, action, _ = tiny._forward_inference(
            x=torch.randn(1, 4, frames, 4, 6, dtype=torch.bfloat16),
            timestep=(torch.rand(1, frames) * 1000).to(torch.bfloat16),
            context=torch.randn(1, 8, 24, dtype=torch.bfloat16),
            seq_len=tokens,
            kv_cache=[
                torch.zeros(
                    [2, 1, 0, TINY["num_heads"], TINY["dim"] // TINY["num_heads"]], dtype=torch.bfloat16
                )
                for _ in range(TINY["num_layers"])
            ],
            crossattn_cache=None,
            current_start_frame=0,
            y=torch.randn(1, 4, frames, 4, 6, dtype=torch.bfloat16),
            clip_feature=torch.randn(1, 3, 1280, dtype=torch.bfloat16),
            embodiment_id=torch.zeros(1, dtype=torch.long),
        )

    assert action is None
    assert video.shape == (1, 4, frames, 4, 6)
    assert len(seen) == TINY["num_layers"]
    for seen_tokens, seen_rows, runs in seen:
        assert (seen_tokens, seen_rows) == (tokens, frames)
        assert runs == ((frames, 6),)


def test_an_empty_shard_of_a_segment_passes_through():
    """Under sequence parallelism a rank can hold none of the segment the head runs on.

    At degree 4 the first rank is entirely inside the teacher-forced clean half, so its share
    of the video tokens is empty and `shard_runs` hands back no runs at all.
    """
    empty = torch.zeros(1, 0, 4)
    assert modulate_tokens(empty, (), torch.randn(1, 0, 4), torch.randn(1, 0, 4)) is empty
    assert gate_tokens(empty, (), torch.randn(1, 0, 4)) is empty
