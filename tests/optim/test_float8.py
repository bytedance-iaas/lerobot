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


def _fqn_eligible(
    module: nn.Module,
    fqn: str = "model.layers.0.mlp.up_proj",
    *,
    min_size: int = 256,
    skip=("lm_head",),
    include_attention: bool = False,
) -> bool:
    return _eligible(
        module,
        fqn,
        min_size=min_size,
        skip_fqn_substrings=skip,
        include_attention=include_attention,
    )


def test_large_trainable_divisible_mlp_linear_is_eligible():
    # MLP/FFN-shaped: both dims divisible by 16, large, trainable, non-attention FQN → convert.
    assert _fqn_eligible(nn.Linear(2048, 8192), "model.layers.0.mlp.up_proj")


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
    assert not _fqn_eligible(lin, "model.lm_head")
    assert _fqn_eligible(lin, "model.layers.0.mlp.up_proj")


def test_attention_projections_are_skipped_by_default():
    # MLP-only default: attention q/k/v/o projections stay bf16 even though they are large,
    # divisible, trainable nn.Linear. (Across Gemma/Qwen/CLIP/BART/DiT naming.)
    for fqn in (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "vision_model.encoder.layers.0.self_attn.out_proj",
        "blocks.0.attn.qkv",
        "blocks.0.attn.proj",
    ):
        assert not _fqn_eligible(nn.Linear(2048, 2048), fqn), fqn


def test_mlp_projections_are_converted():
    # The FFN Linears (various naming conventions) are the fp8 targets.
    for fqn in (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "encoder.layers.0.fc1",  # BART/Florence FFN
        "blocks.0.mlp.fc2",  # DiT FFN
    ):
        assert _fqn_eligible(nn.Linear(2048, 8192), fqn), fqn


def test_include_attention_true_converts_attention_projections():
    # Escape hatch: opt back into converting attention projections.
    assert _fqn_eligible(nn.Linear(2048, 2048), "model.layers.0.self_attn.q_proj", include_attention=True)


def test_filter_over_a_mixed_transformer_selects_only_mlp():
    # A miniature transformer block: attention projections + FFN + tiny head + frozen embed.
    model = nn.Module()
    model.self_attn_q_proj = nn.Linear(2048, 2048)  # attention → skip (MLP-only)
    model.self_attn_o_proj = nn.Linear(2048, 2048)  # attention → skip
    model.mlp_up_proj = nn.Linear(2048, 8192)  # FFN → convert
    model.mlp_down_proj = nn.Linear(8192, 2048)  # FFN → convert
    model.action_head = nn.Linear(2048, 7)  # dim not ÷16 → skip

    picked = [fqn for fqn, m in model.named_modules() if _fqn_eligible(m, fqn)]
    assert picked == ["mlp_up_proj", "mlp_down_proj"]


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
