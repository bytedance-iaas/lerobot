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

"""Guards on the analytic FLOP model: `PI05Pytorch.estimate_flops` and `benchmarks/mfu.py`.

Not accuracy tests -- accuracy is established by agreeing with a profile of a real run,
which needs hardware. These pin what can rot silently: the peak constant that lives in two
modules, the forward/backward factors, and the scaling laws the formula must obey.

Every model here is built on the meta device: parameter *counts* do not need parameter
*values*, so no checkpoint and no allocation.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.utils.profiler_summary import DEFAULT_PEAK_TFLOPS

# benchmarks/ is not a package, so load the file directly. It must be in sys.modules
# BEFORE exec_module: @dataclass resolves its own module via sys.modules[cls.__module__].
_PATH = Path(__file__).parents[2] / "benchmarks" / "mfu.py"
_spec = importlib.util.spec_from_file_location("_mfu", _PATH)
mfu = importlib.util.module_from_spec(_spec)
sys.modules["_mfu"] = mfu
_spec.loader.exec_module(mfu)


def build(**cfg_kwargs) -> PI05Pytorch:
    with torch.device("meta"):
        return PI05Pytorch(PI05Config(**cfg_kwargs))


@pytest.fixture(scope="module")
def model() -> PI05Pytorch:
    return build()


# --- the peak table --------------------------------------------------------------------


def test_peak_matches_profiler_summary():
    """One number, two modules. An MFU that disagrees with itself is worse than none."""
    assert mfu.PEAK_TFLOPS["ascend950pr"]["bfloat16"] == DEFAULT_PEAK_TFLOPS


def test_only_measured_dtypes_are_listed():
    """A guessed peak silently rescales every MFU printed; the table must stay honest."""
    assert set(mfu.PEAK_TFLOPS["ascend950pr"]) == {"bfloat16"}


def test_mm_is_two_flops_per_mac():
    assert mfu.mm(2, 3, 4) == 2 * 2 * 3 * 4


# --- structural parameter counting -----------------------------------------------------


def test_matmul_params_counts_linear_and_conv_only():
    m = nn.Sequential(
        nn.Linear(4, 8, bias=True),  # 32 weights; the 8 biases are adds, not MACs
        nn.Conv2d(3, 6, kernel_size=2),  # 6*3*2*2 = 72
        nn.Embedding(100, 16),  # a lookup: must NOT count
        nn.LayerNorm(8),  # elementwise: must NOT count
    )
    assert PI05Pytorch._matmul_params(m) == 32 + 72


def test_matmul_params_is_nonzero_for_pi05(model):
    assert PI05Pytorch._matmul_params(model.paligemma_with_expert) > 1e9


# --- forward/backward factors, read off requires_grad ----------------------------------


def test_backward_is_twice_forward_when_everything_trains(model):
    """Full fine-tune: every tower pays fwd + dX + dW, so total == 3x forward."""
    parts = model.estimate_flops()
    bare = build()
    # Recompute with every multiplier forced to 1 by freezing nothing: the ratio of the
    # all-trainable total to the forward-only total must be exactly 3.
    for p in bare.parameters():
        p.requires_grad_(False)
    frozen = bare.estimate_flops()
    assert parts["total"] == pytest.approx(3.0 * frozen["total"])


def test_freezing_the_vision_encoder_makes_it_forward_only():
    """Nothing upstream of the tower trains, so even dX is unnecessary."""
    full = build().estimate_flops()
    frozen = build(freeze_vision_encoder=True).estimate_flops()
    key = "vision tower x1"
    assert frozen[key] == pytest.approx(full[key] / 3.0)
    # The VLM still trains, so it is untouched.
    assert frozen["VLM"] == pytest.approx(full["VLM"])


def test_train_expert_only_freezes_the_vlm_too():
    full = build().estimate_flops()
    expert_only = build(train_expert_only=True).estimate_flops()
    assert expert_only["VLM"] == pytest.approx(full["VLM"] / 3.0)
    assert expert_only["action expert"] == pytest.approx(full["action expert"])
    assert expert_only["total"] < full["total"]


# --- scaling laws the formula must obey ------------------------------------------------


def test_each_camera_costs_more_than_the_last(model):
    """An image adds a tower pass AND lengthens the prefix the VLM attends over."""
    totals = [model.estimate_flops(n_images=n)["total"] for n in (1, 2, 3)]
    assert totals[1] - totals[0] < totals[2] - totals[1]


def test_attention_is_quadratic_in_prefix_length(model):
    short, mid, long = (model.estimate_flops(text_tokens=t)["total"] for t in (100, 200, 400))
    assert (long - mid) > 2 * (mid - short)


def test_recompute_is_not_counted():
    """Gradient checkpointing is hardware work; MFU's numerator must not move."""
    m = build()
    before = m.estimate_flops()["total"]
    m.gradient_checkpointing_enable()
    assert m.estimate_flops()["total"] == before


# --- groot -----------------------------------------------------------------------------
# GR00T N1.7's backbone config lives in `nvidia/Cosmos-Reason2-2B`, a GATED repo, so these
# are skipped unless a local copy is pointed at by GROOT_BACKBONE_PATH. What survives
# without it is the refusal to guess, which is the part most likely to rot into a default.


def test_groot_is_registered():
    assert "groot" in mfu.POLICIES


def test_groot_refuses_to_guess_the_prompt_length():
    """text_tokens is prompt-dependent. A default here would silently rescale every MFU."""
    with pytest.raises(SystemExit, match="text-tokens"):
        mfu.POLICIES["groot"](n_images=1)


def _groot(**overrides):
    path = os.environ.get("GROOT_BACKBONE_PATH")
    if not path:
        pytest.skip("set GROOT_BACKBONE_PATH to a local Cosmos-Reason2-2B config directory")
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    cfg = GR00TN17Config(model_name=path, **overrides)
    with torch.device("meta"):
        return GR00TN17(cfg, load_backbone_weights=False)


def test_groot_truncates_the_llm_to_select_layer():
    """`select_layer` pops decoder layers, so the cost must follow the kept ones."""
    short = _groot(select_layer=8).estimate_flops(text_tokens=19)
    long = _groot(select_layer=16).estimate_flops(text_tokens=19)
    assert "LLM (8 of 28 layers)" in short
    assert "LLM (16 of 28 layers)" in long
    assert long["LLM (16 of 28 layers)"] == pytest.approx(2 * short["LLM (8 of 28 layers)"], rel=0.02)


def test_groot_frozen_backbone_is_forward_only():
    """Unfreezing the ViT must triple it: forward, dX and dW instead of forward alone."""
    frozen = _groot().estimate_flops(text_tokens=19)
    tuned = _groot(tune_visual=True).estimate_flops(text_tokens=19)
    assert tuned["vision tower x1"] == pytest.approx(3 * frozen["vision tower x1"], rel=1e-6)


def test_groot_dit_is_half_self_half_cross():
    """`interleave_self_attention` alternates, so the split must come out even."""
    parts = _groot().estimate_flops(text_tokens=19)
    assert any(k.startswith("DiT (16 self + 16 cross)") for k in parts)


def test_groot_category_weights_are_not_billed_per_embodiment():
    """`CategorySpecificLinear` selects one of `max_num_embodiments` slices per sample.

    Counting the whole parameter would multiply the projectors by 32 on the default
    config, which is exactly the kind of error that hides inside a plausible total.
    """
    few = _groot(max_num_embodiments=2).estimate_flops(text_tokens=19)
    many = _groot(max_num_embodiments=64).estimate_flops(text_tokens=19)
    assert few["state/action projectors"] == pytest.approx(many["state/action projectors"], rel=1e-9)


def test_groot_cross_attention_scales_with_the_backbone_sequence():
    """Cross-attention reads K and V off the VL tokens, so more text must cost more."""
    short = _groot().estimate_flops(text_tokens=19)
    long = _groot().estimate_flops(text_tokens=219)
    key = next(k for k in short if k.startswith("DiT ("))
    assert long[key] > short[key]


def test_groot_counts_the_unused_lm_head_separately():
    """It is real work the profile sees, and naming it is how it gets noticed."""
    parts = _groot().estimate_flops(text_tokens=19)
    assert parts["unused LM head"] > 0


# --- dreamzero -------------------------------------------------------------------------
# DreamZero's count is not structural: `estimate_flops` runs the training forward on meta and
# adds up what it calls, because the teacher-forcing attention is 20 hand-sliced calls per
# layer and the VAE is convolutional. These guard the two halves of that -- that the window
# geometry handed to it is the one the model asserts on, and that the run-and-count is
# attributing work to the right side of the MFU/HFU line.

_DZ_CACHE: dict[tuple, dict[str, float]] = {}


def _dreamzero(**kwargs) -> dict[str, float]:
    """~8 s per distinct config (23 B meta parameters), so memoise."""
    key = tuple(sorted(kwargs.items()))
    if key not in _DZ_CACHE:
        _DZ_CACHE[key] = mfu.POLICIES["dreamzero"](**kwargs)
    return _DZ_CACHE[key]


def test_dreamzero_is_registered():
    assert "dreamzero" in mfu.POLICIES


def test_dreamzero_window_matches_the_asserts_in_the_head():
    """`WANPolicyHead.forward` asserts these two ratios; `training_window` must satisfy them.

    Getting either wrong would not raise here -- it would hand `estimate_flops` a sample shape
    the model would reject at training time, and quietly cost the wrong thing.
    """
    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

    cfg = DreamZeroConfig()
    w = cfg.training_window()
    blocks = w["latent_frames"] - 1
    assert w["action_tokens"] / blocks == cfg.num_action_per_block // cfg.num_frame_per_block
    assert blocks / w["state_tokens"] == cfg.num_frame_per_block // cfg.num_state_per_block


def test_dreamzero_window_tokenizes_to_the_dit_frame_seqlen():
    """The canvas, the VAE's 8x and the DiT's 2x2 patch have to land on `frame_seqlen`.

    `_validate_video_geometry` enforces it at config build; this pins that `training_window`
    reports the same latent size that check was run against.
    """
    from lerobot.policies.dreamzero.configuration_dreamzero import _DIT_PRESETS, DreamZeroConfig

    cfg = DreamZeroConfig()
    w = cfg.training_window()
    tokens = (w["latent_height"] // 2) * (w["latent_width"] // 2)
    assert tokens == _DIT_PRESETS[cfg.model_variant]["frame_seqlen"]


def test_dreamzero_counts_every_attention_call():
    """14 self-attention calls per layer plus both halves of i2v cross-attention.

    The number itself is the point: it is what makes a formula the wrong tool here, and if a
    refactor collapses those calls the count should change loudly rather than drift. It has
    changed loudly once: 800 before the single-token state blocks stopped launching a kernel
    to copy v (`CausalWanSelfAttention._process_state_blocks`), 4 per layer of them.
    """
    parts = _dreamzero(training_mode="full")
    key = next(k for k in parts if k.startswith("DiT attention"))
    assert "(640 calls)" in key
    assert parts[key] > 0


def test_dreamzero_recompute_is_reported_but_not_in_the_total():
    """MFU's numerator must not move when gradient checkpointing does. HFU's must."""
    on = _dreamzero(training_mode="full")
    off = _dreamzero(training_mode="full", gradient_checkpointing=False)
    assert off["recompute"] == 0.0
    assert on["recompute"] > 0.0
    assert on["total"] == pytest.approx(off["total"], rel=1e-9)


def test_dreamzero_recompute_is_one_forward_of_the_blocks():
    """Checkpointing wraps the block loop only -- not the embeddings, head, or encoders.

    Billing it as a whole extra forward would overstate HFU; billing the blocks at their
    trained 3x would overstate it three-fold.
    """
    parts = _dreamzero(training_mode="full")
    blocks = next(v for k, v in parts.items() if k.startswith("DiT blocks"))
    attention = next(v for k, v in parts.items() if k.startswith("DiT attention"))
    assert parts["recompute"] == pytest.approx((blocks + attention) / 3.0, rel=1e-9)


def test_dreamzero_frozen_encoders_are_forward_only():
    """umt5/CLIP/VAE are frozen and nothing precedes them, so no gradient ever reaches them.

    Billing them at 3x like the DiT would add ~180 TFLOP of work that never runs.
    """
    full = _dreamzero(training_mode="full")
    lora = _dreamzero(training_mode="lora")
    for key in ("text encoder (umt5)", "image encoder (CLIP)", "VAE encode (tiled)"):
        assert full[key] == pytest.approx(lora[key], rel=1e-9)


def test_dreamzero_lora_pays_dgrad_but_not_wgrad_on_the_base_weights():
    """A frozen weight with trainable projectors at the DiT's input still costs fwd + dgrad.

    So the base DiT is 2x under `lora` against 3x under `full` -- two thirds, plus whatever the
    adapters themselves add.
    """
    full = next(v for k, v in _dreamzero(training_mode="full").items() if k.startswith("DiT blocks"))
    lora = next(v for k, v in _dreamzero(training_mode="lora").items() if k.startswith("DiT blocks"))
    assert lora / full == pytest.approx(2 / 3, rel=0.02)
    assert lora > full * 2 / 3  # the adapters are real work on top


def test_dreamzero_vae_tiling_is_not_free():
    """`tiled=True` re-encodes overlapping tiles, which no parameter count can see.

    Not free, and not wasted either: the tiles are feathered together, so the two paths give
    different latents (67.8% of elements, at the real geometry). `benchmarks/vae_tiling.py`.
    """
    tiled = _dreamzero(training_mode="full")["VAE encode (tiled)"]
    untiled = _dreamzero(training_mode="full", untiled_vae=True)["VAE encode (untiled)"]
    assert tiled > 2 * untiled


def test_dreamzero_time_embedding_is_not_priced_per_token():
    """The time embedding runs on one row per distinct timestep, not one per token.

    `e0` used to be expanded to one row per token before `time_embedding`, which put 15,940
    rows through two 5120-wide linears and a 5120->30720 projection: 17.5 of the 17.9 TFLOP
    this part used to report, for 135 copies of each distinct row. The blocks broadcast from
    118 rows now, so this part collapses to the patch embedding and the head -- which is the
    FLOPs-side shadow of the ~1.22 GiB per block it stopped keeping for the backward pass.
    """
    parts = _dreamzero(training_mode="full")
    embed = parts["DiT embed/head/projectors"] / 1e12
    assert embed < 1.0, f"time embedding looks per-token again: {embed:.1f} TFLOP"
    # Still the smallest real part, and far below the blocks it feeds.
    assert 0 < embed < parts["DiT blocks x40"] / 1000


def test_dreamzero_estimate_flops_requires_a_meta_model():
    """It runs the forward. On a real device that would allocate 23 B parameters of activations."""
    import torch

    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
    from lerobot.policies.dreamzero.wan.action_head import WANPolicyHead

    cfg = DreamZeroConfig()
    head = WANPolicyHead.__new__(WANPolicyHead)
    nn.Module.__init__(head)
    head.probe = nn.Linear(2, 2)  # a single real parameter is enough to fail the check
    with pytest.raises(RuntimeError, match="meta-device model"):
        WANPolicyHead.estimate_flops(head, **cfg.training_window())
    assert next(head.parameters()).device == torch.device("cpu")


# --- the gradient-checkpointing dial ---------------------------------------------------
# `gradient_checkpointing` is True/False or a count, because all-or-nothing is the wrong
# granularity for a 40-block DiT whose activations do not fit but whose half does. These pin
# the counting, and that MFU's numerator stays put while HFU's tracks the dial.


def _mask(flag, depth=40) -> tuple[bool, ...]:
    """Exercise the selection logic without building 23 B meta parameters for it."""
    from lerobot.policies.dreamzero.wan.causal_wan_model import CausalWanModel

    stub = type("Stub", (), {"blocks": [None] * depth, "gradient_checkpointing": flag})()
    return CausalWanModel.checkpointed_blocks(stub)


def test_checkpoint_dial_endpoints():
    assert _mask(True) == (True,) * 40
    assert _mask(False) == (False,) * 40
    assert _mask(40) == (True,) * 40
    assert _mask(0) == (False,) * 40


def test_checkpoint_dial_true_is_not_one():
    """`bool` is a subclass of `int`, so a careless check would read True as "one block"."""
    assert sum(_mask(True)) == 40
    assert sum(_mask(1)) == 1


@pytest.mark.parametrize("count", [1, 5, 12, 20, 25, 39])
def test_checkpoint_dial_honours_the_count(count):
    mask = _mask(count)
    assert len(mask) == 40
    assert sum(mask) == count


def test_checkpoint_dial_spreads_rather_than_clusters():
    """Position is immaterial to memory and time, but a contiguous run reads as if it weren't."""
    chosen = [i for i, on in enumerate(_mask(20)) if on]
    assert chosen != list(range(20))
    assert max(chosen) < 40


def test_config_rejects_a_count_past_the_depth():
    """Clamping would turn a typo into a silent quarter-step tax, which is what the dial avoids."""
    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

    with pytest.raises(ValueError, match=r"count in \[0, 40\]"):
        DreamZeroConfig(gradient_checkpointing=41)
    with pytest.raises(ValueError, match=r"count in \[0, 40\]"):
        DreamZeroConfig(gradient_checkpointing=-1)
    assert DreamZeroConfig(gradient_checkpointing=20).gradient_checkpointing == 20


def test_recompute_is_linear_in_the_dial():
    """Every block costs the same, so half the blocks must be exactly half the recompute."""
    full = _dreamzero(training_mode="full")["recompute"]
    half = _dreamzero(training_mode="full", gradient_checkpointing=20)["recompute"]
    assert half == pytest.approx(full / 2, rel=1e-9)


def test_the_dial_does_not_move_mfus_numerator():
    """Only HFU may respond to it; the model's own arithmetic is unchanged."""
    totals = {n: _dreamzero(training_mode="full", gradient_checkpointing=n)["total"] for n in (0, 20, 40)}
    assert len(set(totals.values())) == 1
