# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SDPA with the Ada Toolkit's masked forward and backward kernels."""

from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812


@lru_cache(maxsize=1)
def _ada_kernels():
    # The Toolkit's torch.ops wrapper does not expose mask/scale/GQA in every
    # release. Use its public run_* entry points and retain the mask for backward.
    from triton_ada_dnn.models.ada2s.train_kernels.scaled_dot_product_attention.static_sdpa_backward import (
        run_scaled_dot_product_attention_backward,
    )
    from triton_ada_dnn.models.ada2s.train_kernels.scaled_dot_product_attention.static_sdpa_forward import (
        run_scaled_dot_product_attention,
    )

    return run_scaled_dot_product_attention, run_scaled_dot_product_attention_backward


class _AdaSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale):
        forward, _ = _ada_kernels()
        enable_gqa = query.shape[1] != key.shape[1]
        output, logsumexp = forward(
            query, key, value, mask=mask, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
        )
        ctx.save_for_backward(query, key, value, output, logsumexp, mask)
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _, backward = _ada_kernels()
        query, key, value, output, logsumexp, mask = ctx.saved_tensors
        dq, dk, dv = backward(
            output,
            grad_output.contiguous(),
            query,
            key,
            value,
            logsumexp,
            mask=mask,
            dropout_p=0.0,
            scale=ctx.scale,
            enable_gqa=ctx.enable_gqa,
        )
        torch.ada.synchronize()
        return dq, dk, dv, None, None


class _AdaVisionAttention(torch.autograd.Function):
    """Fused vision forward with native Ada backward for this Toolkit release.

    The vendor's equal-head backward corrupts device memory during SigLIP
    checkpoint recomputation. Recompute softmax and its derivative with native
    Ada matmuls instead; the masked GQA training kernel keeps its fused backward.
    """

    @staticmethod
    def forward(ctx, query, key, value, mask, scale):
        forward, _ = _ada_kernels()
        output, _ = forward(query, key, value, mask=mask, dropout_p=0.0, scale=scale)
        ctx.save_for_backward(query, key, value)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        scores = (query @ key.transpose(-2, -1)) * ctx.scale
        probabilities_fp32 = torch.softmax(scores.float(), dim=-1)
        probabilities = probabilities_fp32.to(query.dtype)
        grad_value = probabilities.transpose(-2, -1) @ grad_output
        grad_probabilities = grad_output @ value.transpose(-2, -1)
        grad_scores = (
            torch._softmax_backward_data(
                grad_probabilities.float(), probabilities_fp32, -1, torch.float32
            ).to(query.dtype)
            * ctx.scale
        )
        grad_query = grad_scores @ key
        grad_key = grad_scores.transpose(-2, -1) @ query
        return grad_query, grad_key, grad_value, None, None


def sdpa_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Return attention in Transformers' [batch, sequence, heads, dim] layout."""
    if query.device.type == "ada":
        if dropout != 0.0:
            raise ValueError("Pi0.5 Ada SDPA requires attention dropout=0")
        if attention_mask is not None and attention_mask.requires_grad:
            raise ValueError("Ada SDPA does not support gradients for the attention mask")
        if attention_mask is None:
            # An explicit zero bias selects the vendor's standard training
            # kernel. Its unmasked pipelined variant has a compiler defect
            # for SigLIP's equal Q/KV head count in this Toolkit release.
            attention_mask = torch.zeros(
                (1, 1, query.shape[2], key.shape[2]), device=query.device, dtype=torch.float32
            )
            head_dim = query.shape[-1]
            padded_dim = max(64, 1 << (head_dim - 1).bit_length())
            if padded_dim != head_dim:
                query = F.pad(query, (0, padded_dim - head_dim))
                key = F.pad(key, (0, padded_dim - head_dim))
                value = F.pad(value, (0, padded_dim - head_dim))
            mask = attention_mask.expand(
                query.shape[0], query.shape[1], query.shape[2], key.shape[2]
            ).contiguous()
            output = _AdaVisionAttention.apply(
                query.contiguous(), key.contiguous(), value.contiguous(), mask, scaling
            )
            return output[..., :head_dim].transpose(1, 2).contiguous(), None
        mask = attention_mask
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, float("-inf"))
            mask = mask.expand(query.shape[0], query.shape[1], query.shape[2], key.shape[2]).contiguous()
        output = _AdaSDPA.apply(query.contiguous(), key.contiguous(), value.contiguous(), mask, scaling)
    else:
        # cuDNN's BF16 attention expects an additive bias in the query dtype.
        # Its FP32-bias path in the deployed CUDA stack produces wrong values.
        if query.device.type == "cuda" and attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(query.dtype)
        # Keep grouped-query behavior compatible with all native SDPA backends.
        groups = query.shape[1] // key.shape[1]
        if groups != 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout, is_causal=False, scale=scaling
        )
    return output.transpose(1, 2).contiguous(), None
