"""Ascend Triton kernels for first-order GeGLU training.

Recompute GELU in backward instead of saving its large activation tensor.
Explicit casts preserve eager BF16 rounding at the GELU output and at the
incoming GELU gradient. FP32 arithmetic still has small rounding differences.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cann import math as cmath


@triton.jit
def _forward(G, U, Y, N: tl.constexpr, B: tl.constexpr):  # noqa: N803
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    t = cmath.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
    a = (0.5 * g * (1.0 + t)).to(G.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, a * u, i < N)


@triton.jit
def _backward(G, U, D, DG, DU, N: tl.constexpr, B: tl.constexpr):  # noqa: N803
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    d = tl.load(D + i, i < N, 0).to(tl.float32)
    t = cmath.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
    a = (0.5 * g * (1.0 + t)).to(G.dtype.element_ty).to(tl.float32)
    derivative = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608028654 * (
        1.0 + 0.134145 * g * g
    )
    incoming = (d * u).to(G.dtype.element_ty).to(tl.float32)
    tl.store(DG + i, incoming * derivative, i < N)
    tl.store(DU + i, d * a, i < N)


class GeGLU(torch.autograd.Function):
    """One forward kernel and one backward kernel; no higher-order derivatives."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(gate)
        with torch.npu.device(gate.device):
            _forward[(triton.cdiv(gate.numel(), 4096),)](
                gate, up, output, gate.numel(), 4096, enable_fp_fusion=False
            )
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
        with torch.npu.device(gate.device):
            _backward[(triton.cdiv(gate.numel(), 4096),)](
                gate, up, grad_output, grad_gate, grad_up, gate.numel(), 4096, enable_fp_fusion=False
            )
        return grad_gate, grad_up
