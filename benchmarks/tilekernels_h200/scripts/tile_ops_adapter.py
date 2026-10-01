"""Experimental TileKernels operations that semantically match the policies."""

from __future__ import annotations

import weakref

import tile_adapter
import torch
import transformers.models.gemma.modeling_gemma as gemma_modeling
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen_modeling
from tile_kernels.quant import swiglu_backward, swiglu_forward
from tile_kernels.transform import apply_rotary
from torch import nn

QWEN_MLP_ORIGINAL = qwen_modeling.Qwen3VLTextMLP.forward
QWEN_ROPE_ORIGINAL = qwen_modeling.apply_rotary_pos_emb
GEMMA_ROPE_ORIGINAL = gemma_modeling.apply_rotary_pos_emb

_rope_source_refs: tuple[weakref.ReferenceType[torch.Tensor], weakref.ReferenceType[torch.Tensor]] | None = (
    None
)
_rope_cache: torch.Tensor | None = None
_rope_positions: torch.Tensor | None = None


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        ctx.save_for_backward(flat)
        ctx.shape = x.shape
        fmt = "bf16" if x.dtype == torch.bfloat16 else "fp32"
        return swiglu_forward(flat, fmt).view(*x.shape[:-1], x.shape[-1] // 2)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        fmt = "bf16" if x.dtype == torch.bfloat16 else "fp32"
        dx = swiglu_backward(x, grad.reshape(x.shape[0], -1).contiguous(), fmt)[0]
        return (dx.view(ctx.shape),)


def tile_swiglu(x: torch.Tensor) -> torch.Tensor:
    return SwiGLU.apply(x)


def qwen_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if self.config.hidden_act != "silu":
        return QWEN_MLP_ORIGINAL(self, x)
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    return self.down_proj(tile_swiglu(torch.cat((gate, up), dim=-1)))


def _packed_rope_cache(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global _rope_source_refs, _rope_cache, _rope_positions
    same_source = (
        _rope_source_refs is not None and _rope_source_refs[0]() is cos and _rope_source_refs[1]() is sin
    )
    if not same_source:
        if cos.ndim != 3 or sin.shape != cos.shape:
            raise ValueError(
                f"Expected matching [batch, sequence, head_dim] cos/sin, got {cos.shape}, {sin.shape}"
            )
        batch, sequence, head_dim = cos.shape
        half = head_dim // 2
        # HF Gemma/Qwen duplicate each frequency across the two NeoX halves.
        _rope_cache = (
            torch.cat((cos[..., :half], sin[..., :half]), dim=-1).float().reshape(-1, head_dim).contiguous()
        )
        _rope_positions = torch.arange(batch * sequence, dtype=torch.int32, device=cos.device).view(
            batch, sequence
        )
        _rope_source_refs = (weakref.ref(cos), weakref.ref(sin))
    assert _rope_cache is not None and _rope_positions is not None
    return _rope_cache, _rope_positions


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache, positions = _packed_rope_cache(cos, sin)
        query_out = query.transpose(1, 2).contiguous()
        key_out = key.transpose(1, 2).contiguous()
        apply_rotary(query_out, cache, key_out, positions=positions)
        ctx.save_for_backward(cache, positions)
        return query_out.transpose(1, 2), key_out.transpose(1, 2)

    @staticmethod
    def backward(
        ctx, query_grad: torch.Tensor, key_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        cache, positions = ctx.saved_tensors
        query_grad = query_grad.transpose(1, 2).contiguous()
        key_grad = key_grad.transpose(1, 2).contiguous()
        apply_rotary(query_grad, cache, key_grad, positions=positions, conjugate=True)
        return (
            query_grad.transpose(1, 2),
            key_grad.transpose(1, 2),
            None,
            None,
        )


def tile_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unsqueeze_dim != 1:
        raise ValueError(f"Only B,H,S,D attention layout is supported, got unsqueeze_dim={unsqueeze_dim}")
    return Rotary.apply(query, key, cos, sin)


def set_enabled(*, rms: bool, swiglu: bool, rope: bool) -> None:
    tile_adapter.set_enabled(rms)
    qwen_modeling.Qwen3VLTextMLP.forward = qwen_mlp_forward if swiglu else QWEN_MLP_ORIGINAL
    qwen_modeling.apply_rotary_pos_emb = tile_rope if rope else QWEN_ROPE_ORIGINAL
    gemma_modeling.apply_rotary_pos_emb = tile_rope if rope else GEMMA_ROPE_ORIGINAL
