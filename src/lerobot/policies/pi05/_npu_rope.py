"""Ascend rotary kernels preserving eager FP32/BF16 product rounding."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _rotate(X, C, S, Y, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BACKWARD: tl.constexpr, B: tl.constexpr):  # noqa: N803
    row = tl.program_id(0) * B + tl.arange(0, B)
    column = tl.arange(0, D // 2)
    offset = (tl.program_id(1) * T + row[:, None]) * D + column[None, :]
    position = (tl.program_id(1) // H * T + row[:, None]) * D + column[None, :]
    mask = row[:, None] < T
    x0 = tl.load(X + offset, mask, 0).to(tl.float32)
    x1 = tl.load(X + offset + D // 2, mask, 0).to(tl.float32)
    c0 = tl.load(C + position, mask, 0).to(tl.float32)
    c1 = tl.load(C + position + D // 2, mask, 0).to(tl.float32)
    s0 = tl.load(S + position, mask, 0).to(tl.float32)
    s1 = tl.load(S + position + D // 2, mask, 0).to(tl.float32)
    a = (x0 * c0).to(X.dtype.element_ty).to(tl.float32)
    b = (x1 * c1).to(X.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        p = (x1 * s1).to(X.dtype.element_ty).to(tl.float32)
        q = (x0 * s0).to(X.dtype.element_ty).to(tl.float32)
        y0, y1 = a + p, b - q
    else:
        p = (-x1 * s0).to(X.dtype.element_ty).to(tl.float32)
        q = (x0 * s1).to(X.dtype.element_ty).to(tl.float32)
        y0, y1 = a + p, b + q
    tl.store(Y + offset, y0, mask)
    tl.store(Y + offset + D // 2, y1, mask)


class Rotary(torch.autograd.Function):
    """One rotation kernel per input and per backward; first-order gradients only."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        batch, heads, tokens, dim = x.shape
        with torch.npu.device(x.device):
            _rotate[(triton.cdiv(tokens, 8), batch * heads)](
                x, cos, sin, output, heads, tokens, dim, False, 8, enable_fp_fusion=False
            )
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        batch, heads, tokens, dim = grad_output.shape
        with torch.npu.device(grad_output.device):
            _rotate[(triton.cdiv(tokens, 8), batch * heads)](
                grad_output, cos, sin, grad_input, heads, tokens, dim, True, 8, enable_fp_fusion=False
            )
        return grad_input, None, None
