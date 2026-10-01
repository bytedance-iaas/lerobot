"""Check correctness and timing at representative GR00T/PI05 shapes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from tile_ops_adapter import GEMMA_ROPE_ORIGINAL, tile_rope, tile_swiglu
from torch.nn import functional

torch.manual_seed(0)
device = torch.device("cuda")
results: dict[str, object] = {
    "unavailable_in_tilekernels": [
        "attention",
        "gemm_linear",
        "gelu_geglu",
        "adamw",
    ],
    "available": ["swiglu", "rope"],
}


def timing(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e6 / repeats


def check_swiglu(tokens: int, hidden: int) -> dict[str, float]:
    x_ref = torch.randn(tokens, hidden * 2, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_tile = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    ref = functional.silu(x_ref[..., :hidden]) * x_ref[..., hidden:]
    actual = tile_swiglu(x_tile)
    ref.backward(grad)
    actual.backward(grad)
    error = (actual.float() - ref.float()).abs()
    grad_error = (x_tile.grad.float() - x_ref.grad.float()).abs()
    x = x_ref.detach()
    return {
        "forward_max_abs": error.max().item(),
        "forward_relative_l2": (error.double().square().sum() / ref.float().double().square().sum())
        .sqrt()
        .item(),
        "backward_max_abs": grad_error.max().item(),
        "backward_relative_l2": (
            grad_error.double().square().sum() / x_ref.grad.float().double().square().sum()
        )
        .sqrt()
        .item(),
        "baseline_forward_us": timing(lambda: functional.silu(x[..., :hidden]) * x[..., hidden:], 10, 30),
        "tile_forward_us": timing(lambda: tile_swiglu(x), 10, 30),
    }


def check_rope(batch: int, sequence: int, query_heads: int, kv_heads: int, head_dim: int) -> dict[str, float]:
    q_ref = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_ref = torch.randn(
        batch,
        kv_heads,
        sequence,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    q_tile = q_ref.detach().clone().requires_grad_(True)
    k_tile = k_ref.detach().clone().requires_grad_(True)
    angles = torch.randn(batch, sequence, head_dim // 2, device=device, dtype=torch.bfloat16)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    grad_q = torch.randn_like(q_ref)
    grad_k = torch.randn_like(k_ref)
    ref_q, ref_k = GEMMA_ROPE_ORIGINAL(q_ref, k_ref, cos, sin)
    actual_q, actual_k = tile_rope(q_tile, k_tile, cos, sin)
    torch.autograd.backward((ref_q, ref_k), (grad_q, grad_k))
    torch.autograd.backward((actual_q, actual_k), (grad_q, grad_k))
    forward_error = torch.cat(
        (
            (actual_q.float() - ref_q.float()).flatten(),
            (actual_k.float() - ref_k.float()).flatten(),
        )
    ).abs()
    grad_error = torch.cat(
        (
            (q_tile.grad.float() - q_ref.grad.float()).flatten(),
            (k_tile.grad.float() - k_ref.grad.float()).flatten(),
        )
    ).abs()
    q = q_ref.detach()
    k = k_ref.detach()
    return {
        "forward_max_abs": forward_error.max().item(),
        "backward_max_abs": grad_error.max().item(),
        "baseline_forward_us": timing(lambda: GEMMA_ROPE_ORIGINAL(q, k, cos, sin), 10, 30),
        "tile_forward_us": timing(lambda: tile_rope(q, k, cos, sin), 10, 30),
    }


results["swiglu_qwen"] = check_swiglu(tokens=16 * 156, hidden=6144)
results["rope_qwen"] = check_rope(batch=16, sequence=156, query_heads=16, kv_heads=8, head_dim=128)
results["rope_pi05"] = check_rope(batch=16, sequence=506, query_heads=8, kv_heads=1, head_dim=256)

output = Path(os.environ.get("TK_OP_CHECK_OUTPUT", "tile-ops-check.json"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
