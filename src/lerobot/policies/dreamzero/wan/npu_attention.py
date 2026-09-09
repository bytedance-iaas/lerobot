# Copyright 2024 The LeRobot Authors.
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
"""Ascend NPU attention for the vendored Wan blocks.

`wan/attention.py` and `wan/causal_attention.py` are two independent vendored copies of
upstream's attention, and both call FlashAttention kernels that only exist on CUDA. On an
Ascend NPU the fused equivalent is `torch_npu.npu_fusion_attention`, which is what
MindIE-SD's `attention_forward(op_type="fused_attn_score")` wraps -- the path vllm-omni
takes for Wan2.2/DreamZero on NPU. Using it directly avoids the MindIE-SD dependency while
still getting a fused kernel rather than plain SDPA, which matters for a 23B DiT.

Both vendored files import from here so the masking logic exists once.
"""

import warnings

import torch

__all__ = ["npu_fusion_attention_available", "npu_fusion_attention"]


def npu_fusion_attention_available(device) -> bool:
    """True when `device` is an Ascend NPU exposing the fused attention op.

    Do not build a `torch.device` from a string here: "npu" is only a registered device
    type once torch_npu has been imported, so `torch.device("npu")` raises on a plain
    CUDA/CPU box.
    """
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type != "npu":
        return False
    try:
        import torch_npu
    except ImportError:
        return False
    return hasattr(torch_npu, "npu_fusion_attention")


def _build_atten_mask(b, lq, lk, causal, k_lens, device):
    """Boolean mask for npu_fusion_attention, shaped [B, 1, Lq, Lk].

    NOTE the polarity: `atten_mask` is True where attention is *forbidden*. That is the
    inverse of the additive/boolean mask SDPA takes, and getting it backwards silently
    produces anti-causal attention rather than an error.
    """
    mask = None
    if causal:
        # Bottom-right alignment, matching FlashAttention's convention when Lq != Lk: the
        # last query attends through the last key.
        mask = torch.triu(torch.ones(lq, lk, dtype=torch.bool, device=device), diagonal=lk - lq + 1).expand(
            b, 1, lq, lk
        )

    if k_lens is not None:
        # Key j is padding for batch element i when j >= k_lens[i].
        cols = torch.arange(lk, device=device).view(1, 1, 1, lk)
        pad = cols >= k_lens.to(device).view(b, 1, 1, 1)
        mask = pad if mask is None else (mask | pad)

    if mask is None:
        return None

    # aclnnFlashAttentionScore accepts only [B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv] or
    # [Sq,Skv] -- it will not broadcast the query axis, so a key-padding-only mask
    # ([B,1,1,Skv]) has to be materialised over Sq before the call.
    mask = mask.expand(b, 1, lq, lk)

    # A query row with every key masked makes softmax divide by zero -> NaN, which then
    # poisons the whole batch through the backward pass. Such a row's output is discarded
    # by the caller anyway (it is itself padding), so let it attend to key 0.
    dead = mask.all(dim=-1, keepdim=True)
    if dead.any():
        mask = mask.clone()
        mask[..., 0:1] = mask[..., 0:1] & ~dead

    return mask.contiguous()


def npu_fusion_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """Drop-in replacement for the vendored `_sdpa_attention_fallback` on Ascend NPU.

    Same signature and same [B, Lq, Nq, C] in / [B, Lq, Nq, C] out contract, so the two
    vendored `flash_attention` functions can dispatch to either without reshaping.

    Unlike the SDPA fallback, `k_lens` is honoured rather than warned about -- the fused
    op takes an explicit mask, so respecting key padding costs nothing here. `q_lens` is
    still ignored: those rows are padding whose outputs the caller drops.
    """
    import torch_npu

    b, lq, nq, c = q.shape
    lk, nk = k.size(1), k.size(2)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is None:
        softmax_scale = c**-0.5

    if q_lens is not None:
        warnings.warn(
            "q_lens is ignored by the Ascend fused attention path; padded query rows "
            "produce arbitrary outputs that the caller is expected to discard.",
            stacklevel=2,
        )

    # npu_fusion_attention has no GQA mode: materialise the key/value heads. In BSND the
    # head axis is dim=2, and repeat_interleave (not repeat) is what pairs each query head
    # group with its own kv head.
    if nq != nk:
        assert nq % nk == 0, f"Nq ({nq}) must be divisible by Nk ({nk})"
        k = k.repeat_interleave(nq // nk, dim=2)
        v = v.repeat_interleave(nq // nk, dim=2)

    atten_mask = _build_atten_mask(b, lq, lk, causal, k_lens, q.device)

    out = torch_npu.npu_fusion_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        nq,
        "BSND",
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        scale=float(softmax_scale),
        keep_prob=1.0 - dropout_p,
        # Keep torch_npu's API spelling ("tockens").
        pre_tockens=2147483647,
        next_tockens=2147483647,
        inner_precise=0,
        prefix=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
    )[0]

    return out
