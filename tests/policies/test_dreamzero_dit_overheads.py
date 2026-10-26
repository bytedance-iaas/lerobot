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

"""CPU tests for the DreamZero DiT step's non-matmul overheads.

Once `gradient_checkpointing=False` closes the MFU/HFU gap, what is left between MFU and 100%
is not arithmetic: it is kernel launches, the elementwise chains between the attention calls,
and dtype casts. These cover the three places that were paying for nothing.
"""

import pytest
import torch

from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
from lerobot.policies.dreamzero.wan import submodule
from lerobot.policies.dreamzero.wan.causal_wan_model import CausalWanSelfAttention


# --- single-token state self-attention is an identity -----------------------------------


def _state_attention(num_state_per_block: int, attn):
    """Exercise the block routing without building a 14 B DiT around it."""
    # staticmethod, or a function in the class namespace would be bound and swallow `self`.
    stub = type("Stub", (), {"num_state_per_block": num_state_per_block, "attn": staticmethod(attn)})()
    return lambda q, k, v, horizon: CausalWanSelfAttention._process_state_blocks(stub, q, k, v, horizon)


def _explode(*args, **kwargs):
    raise AssertionError("attention was called on a single-token state block")


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_single_token_state_blocks_return_v_without_an_attention_call(horizon):
    """Softmax over one key is exactly 1, so the attention returned v. 160 launches a forward."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, horizon, 4, 8, dtype=torch.bfloat16) for _ in range(3))
    out = _state_attention(1, _explode)(q, k, v, horizon)
    assert torch.equal(out, v)
    assert out.dtype == q.dtype


def test_multi_token_state_blocks_still_attend():
    """`num_state_per_block > 1` is real attention within each block, so the loop stays."""
    calls = []

    def attn(q, k, v):
        calls.append(q.shape[1])
        return v

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 6, 4, 8, dtype=torch.bfloat16) for _ in range(3))
    out = _state_attention(2, attn)(q, k, v, 6)
    assert calls == [2, 2, 2]  # 6 state tokens, 2 per block
    assert torch.equal(out, v)


def test_the_droid_config_is_the_single_token_case():
    """Which is why this is worth a shortcut rather than a comment."""
    assert DreamZeroConfig().num_state_per_block == 1


# --- the action rope honours `rope_dtypes` ----------------------------------------------


def _rope_inputs(dtype=torch.float32):
    torch.manual_seed(0)
    x = torch.randn(1, 4, 2, 4, dtype=dtype)
    angles = torch.randn(4, 1, 2, dtype=torch.float64)
    return x, torch.polar(torch.ones_like(angles), angles)


def test_the_action_rope_follows_rope_dtypes_rather_than_hardcoding_float64(monkeypatch):
    """`causal_rope_action_apply_polar`'s sibling in `submodule` was left behind.

    The causal self-attention calls both. `_create_freqs` casts the freqs per `rope_dtypes`
    -- complex64 on NPU, where CANN has no complex128 polar/cat/mul -- so a hardcoded float64
    here meant NPU multiplied a complex128 x by complex64 freqs. Faking the dtype has to change
    the result; if it does not, the function is ignoring it again.
    """
    x, freqs = _rope_inputs()
    wide = submodule.rope_action_apply_polar(x, freqs, None, None, None)

    monkeypatch.setattr(submodule, "rope_dtypes", lambda device: (torch.float32, torch.complex64))
    narrow = submodule.rope_action_apply_polar(x, freqs, None, None, None)

    assert narrow.shape == wide.shape
    assert not torch.equal(narrow, wide), "rope_dtypes was ignored"
    # Single precision, not a different rotation: float32's eps is 1.2e-07.
    assert narrow.dtype == torch.float32 and wide.dtype == torch.float64
    assert torch.allclose(narrow.double(), wide, atol=1e-6)


def test_the_action_rope_agrees_with_its_causal_sibling_on_dtype(monkeypatch):
    """Both are reached from `CausalWanSelfAttention.forward`, so they must not disagree."""
    from lerobot.policies.dreamzero.wan import causal_wan_model

    assert "rope_dtypes" in submodule.rope_action_apply_polar.__code__.co_names
    assert "rope_dtypes" in causal_wan_model.causal_rope_action_apply_polar.__code__.co_names


# --- the compile switch -----------------------------------------------------------------


def test_compile_dit_is_off_by_default_and_separate_from_the_encoders():
    """It is a capability, not a recommendation: no run has been made with it on."""
    cfg = DreamZeroConfig()
    assert cfg.compile_dit is False
    assert cfg.compile_encoders is False


def test_compile_dit_wraps_every_block_and_nothing_else():
    """Per block, because ~20 hand-sliced attention calls rule out a whole-model graph."""
    from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy

    blocks = [type("Block", (), {"forward": staticmethod(lambda: None)})() for _ in range(4)]
    dit = type("Dit", (), {"blocks": blocks})()
    policy = type("P", (), {"action_head": type("H", (), {"dit": lambda self: dit})()})()

    before = [block.forward for block in blocks]
    DreamZeroPolicy._compile_dit(policy)

    for block, original in zip(blocks, before, strict=True):
        assert block.forward is not original, "block was left uncompiled"
        # torch.compile returns a wrapper that still points at what it wraps.
        assert getattr(block.forward, "_torchdynamo_orig_callable", None) is original
