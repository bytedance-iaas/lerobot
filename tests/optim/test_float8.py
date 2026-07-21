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

"""Tests for the fp8 layer-eligibility filter.

These test the FILTER LOGIC only — which layers fp8 would touch — on CPU, with no torchao
and no fp8 GPU. That is exactly the model-agnostic contract we care about: a policy that
cannot benefit must yield an empty conversion set rather than an error. The actual fp8 GEMM
correctness is torchao's own concern and needs Hopper/Ada hardware.
"""

import torch
from torch import nn

from lerobot.optim.float8 import _eligible, is_fp8_supported


def _fqn_eligible(module: nn.Module, fqn: str = "layer", *, min_size: int = 256, skip=("lm_head",)) -> bool:
    return _eligible(module, fqn, min_size=min_size, skip_fqn_substrings=skip)


def test_large_trainable_divisible_linear_is_eligible():
    # Transformer-FFN-shaped: both dims divisible by 16, large, trainable → convert.
    assert _fqn_eligible(nn.Linear(2048, 8192))


def test_dims_not_divisible_by_16_are_skipped():
    # fp8 tensor-core tiling needs in/out % 16 == 0. Action heads (small odd out dim) land here.
    assert not _fqn_eligible(nn.Linear(512, 7))  # e.g. 6 motors + gripper
    assert not _fqn_eligible(nn.Linear(500, 512))  # in not divisible by 16


def test_small_linear_is_skipped():
    # Below min_size the fp8 overhead outweighs the GEMM saving (small RL/MLP projections).
    assert not _fqn_eligible(nn.Linear(64, 64))
    assert not _fqn_eligible(nn.Linear(128, 128))


def test_frozen_linear_is_skipped():
    # A frozen pretrained backbone (smolvla/groot freeze the VLM by default) must NOT convert —
    # converting a layer you don't train is pure overhead.
    lin = nn.Linear(2048, 8192)
    lin.requires_grad_(False)
    assert not _fqn_eligible(lin)
    # ...and flips back to eligible once trainable.
    lin.requires_grad_(True)
    assert _fqn_eligible(lin)


def test_non_linear_modules_are_skipped():
    assert not _fqn_eligible(nn.Conv2d(16, 32, 3))
    assert not _fqn_eligible(nn.LayerNorm(2048))
    assert not _fqn_eligible(nn.Embedding(1000, 512))


def test_skip_fqn_substring():
    lin = nn.Linear(2048, 8192)
    assert not _eligible(lin, "model.lm_head", min_size=256, skip_fqn_substrings=("lm_head",))
    assert _eligible(lin, "model.layers.0.mlp.up_proj", min_size=256, skip_fqn_substrings=("lm_head",))


def test_filter_over_a_mixed_model_selects_only_the_right_layers():
    # A miniature "policy": conv vision stem + frozen backbone linear + trainable FFN + tiny head.
    model = nn.Module()
    model.conv = nn.Conv2d(3, 16, 3)
    model.frozen_proj = nn.Linear(2048, 2048)
    model.frozen_proj.requires_grad_(False)
    model.ffn_up = nn.Linear(2048, 8192)  # trainable, large, divisible → the only fp8 target
    model.ffn_down = nn.Linear(8192, 2048)  # trainable, large, divisible → also a target
    model.action_head = nn.Linear(2048, 7)  # not divisible by 16 → skip

    picked = [fqn for fqn, m in model.named_modules() if _fqn_eligible(m, fqn)]
    assert picked == ["ffn_up", "ffn_down"]


def test_hf_style_attention_projections_are_converted():
    # For HF transformers (Gemma/Qwen/...), q/k/v/o are plain nn.Linear and ARE prime fp8
    # targets — the projection GEMMs should convert (the attention softmax math stays bf16).
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert _fqn_eligible(nn.Linear(2048, 2048), f"model.layers.0.self_attn.{name}")


def test_nn_multiheadattention_out_proj_is_skipped():
    # nn.MultiheadAttention (ACT) packs QKV into a raw in_proj_weight parameter (invisible to
    # the filter) but its out_proj is a NonDynamicallyQuantizableLinear (an nn.Linear SUBCLASS).
    # Converting only that would half-convert the attention; it must be skipped so MHA stays
    # fully bf16.
    mha = nn.MultiheadAttention(512, 8)
    assert not _fqn_eligible(mha.out_proj, "encoder.layers.0.self_attn.out_proj")


def test_is_fp8_supported_is_false_without_cuda():
    if not torch.cuda.is_available():
        assert is_fp8_supported() is False
