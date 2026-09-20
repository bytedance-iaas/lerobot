"""Optional per-forward RoPE reuse and Ascend rotary fusion."""

import torch
from torch import Tensor, nn

RotaryCache = dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor]]


def cached_rotary_embeddings(
    rotary: nn.Module, query: Tensor, position_ids: Tensor, cache: RotaryCache
) -> tuple[Tensor, Tensor]:
    """Reuse embeddings for fixed positions within one joint-model forward.

    The caller must create a fresh cache for each forward and share it only
    across layers using the same rotary module and position_ids. Keeping it
    local also makes checkpoint recomputation independent of later forwards.
    Gemma's rotary module uses query only for its dtype/device metadata.
    """
    key = (query.device, query.dtype)
    if key not in cache:
        cache[key] = rotary(query, position_ids)
    return cache[key]


def apply_rotary_pos_emb_npu(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Fuse supported NPU rotations; retain the original implementation elsewhere."""
    supported = (
        all(type(x) is Tensor and x.is_contiguous() for x in (q, k, cos, sin))
        and q.device.type == "npu"
        and q.device == k.device == cos.device == sin.device
        and q.dtype == k.dtype == cos.dtype == sin.dtype
        and q.dtype in (torch.float32, torch.bfloat16)
        and q.ndim == k.ndim == 4
        and q.shape[0] == k.shape[0]
        and q.shape[2:] == k.shape[2:]
        and q.shape[-1] in (64, 256)
        and cos.shape == sin.shape == (q.shape[0], q.shape[2], q.shape[3])
        and q.numel() > 0
        and k.numel() > 0
        and not cos.requires_grad
        and not sin.requires_grad
    )
    if not supported:
        from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # The Ascend Triton dependency is never imported on CPU/CUDA fallback paths.
    from ._npu_rope import Rotary

    return Rotary.apply(q, cos, sin), Rotary.apply(k, cos, sin)
