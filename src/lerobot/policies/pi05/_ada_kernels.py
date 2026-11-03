"""Ada2S Triton kernels for Pi0.5 elementwise training operations."""

import torch
import torch.nn.functional as F  # noqa: N812
import triton
import triton.language as tl
import triton.language.extra.ada as ada
from torch.autograd.function import once_differentiable


@triton.jit
def _geglu_forward(g_ptr, u_ptr, y_ptr, n: tl.constexpr, block: tl.constexpr, gated: tl.constexpr = True):
    # Launch exactly 32 programs: each owns disjoint, consecutive block-sized
    # tiles spaced 32 blocks apart. Callers pad n to complete DMA tiles.
    for chunk in range(triton.cdiv(n, 32 * block)):
        i = (chunk * 32 + tl.program_id(0)) * block + tl.arange(0, block)
        g = tl.load(g_ptr + i, i < n, 0).to(tl.float32)
        if gated:  # noqa: SIM108 -- Triton must eliminate the unused pointer branch at compile time.
            u = tl.load(u_ptr + i, i < n, 0).to(tl.float32)
        else:
            u = 1.0
        t = ada.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
        activation = (0.5 * g * (1.0 + t)).to(g_ptr.dtype.element_ty).to(tl.float32)
        tl.store(y_ptr + i, (activation * u).to(y_ptr.dtype.element_ty), i < n)


@triton.jit
def _geglu_backward(
    g_ptr, u_ptr, d_ptr, dg_ptr, du_ptr, n: tl.constexpr, block: tl.constexpr, gated: tl.constexpr = True
):
    for chunk in range(triton.cdiv(n, 32 * block)):
        i = (chunk * 32 + tl.program_id(0)) * block + tl.arange(0, block)
        g = tl.load(g_ptr + i, i < n, 0).to(tl.float32)
        if gated:  # noqa: SIM108 -- Triton must eliminate the unused pointer branch at compile time.
            u = tl.load(u_ptr + i, i < n, 0).to(tl.float32)
        else:
            u = 1.0
        d = tl.load(d_ptr + i, i < n, 0).to(tl.float32)
        t = ada.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))
        activation = (0.5 * g * (1.0 + t)).to(g_ptr.dtype.element_ty).to(tl.float32)
        derivative = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * g * g)
        incoming = (d * u).to(g_ptr.dtype.element_ty).to(tl.float32)
        tl.store(dg_ptr + i, (incoming * derivative).to(dg_ptr.dtype.element_ty), i < n)
        if gated:
            tl.store(du_ptr + i, (d * activation).to(du_ptr.dtype.element_ty), i < n)


class GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # The vision width 4304 leaves a partial DMA tile. Pad the flat buffer
        # so every PE reads and writes complete aligned blocks.
        count = triton.cdiv(x.numel(), 32 * 4096) * 32 * 4096
        flat = F.pad(x.view(-1), (0, count - x.numel()))
        output = torch.empty_like(flat)
        _geglu_forward[(32,)](flat, None, output, count, 4096, False)
        ctx.save_for_backward(flat)
        ctx.shape = x.shape
        ctx.numel = x.numel()
        return output[: x.numel()].view(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = F.pad(grad_output.contiguous().view(-1), (0, x.numel() - ctx.numel))
        grad_x = torch.empty_like(x)
        _geglu_backward[(32,)](x, None, grad, grad_x, None, x.numel(), 4096, False)
        # DDP can start its bucket reduction before loss.backward() returns.
        # Finish custom kernel writes before autograd exposes these gradients.
        torch.ada.synchronize()
        return grad_x[: ctx.numel].view(ctx.shape)


class GeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        count = triton.cdiv(gate.numel(), 32 * 4096) * 32 * 4096
        padding = count - gate.numel()
        g = F.pad(gate.view(-1), (0, padding))
        u = F.pad(up.view(-1), (0, padding))
        output = torch.empty_like(g)
        _geglu_forward[(32,)](g, u, output, count, 4096)
        ctx.save_for_backward(g, u)
        ctx.shape = gate.shape
        ctx.numel = gate.numel()
        return output[: ctx.numel].view(ctx.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
        grad = F.pad(grad_output.contiguous().view(-1), (0, gate.numel() - ctx.numel))
        _geglu_backward[(32,)](gate, up, grad, grad_gate, grad_up, gate.numel(), 4096)
        torch.ada.synchronize()
        return grad_gate[: ctx.numel].view(ctx.shape), grad_up[: ctx.numel].view(ctx.shape)


@triton.jit
def _residual_forward(
    x_ptr, y_ptr, g_ptr, output, width: tl.constexpr, tokens: tl.constexpr, block: tl.constexpr
):
    rows = tl.program_id(1) * block + tl.arange(0, block)
    cols = tl.arange(0, width)
    i = (tl.program_id(0) * tokens + rows[:, None]) * width + cols[None, :]
    mask = rows[:, None] < tokens
    x = tl.load(x_ptr + i, mask, 0).to(tl.float32)
    y = tl.load(y_ptr + i, mask, 0).to(tl.float32)
    g = tl.load(g_ptr + tl.program_id(0) * width + cols).to(tl.float32)
    product = (y * g[None, :]).to(y_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output + i, (x + product).to(output.dtype.element_ty), mask)


@triton.jit
def _residual_backward(
    y_ptr,
    g_ptr,
    d_ptr,
    dy,
    dg,
    width: tl.constexpr,
    tokens: tl.constexpr,
    batch: tl.constexpr,
    bt: tl.constexpr,
    bd: tl.constexpr,
):
    batch_id = tl.program_id(0)
    columns = tl.program_id(1) * bd + tl.arange(0, bd)
    rows = tl.arange(0, bt)
    offset = (batch_id * tokens + rows[:, None]) * width + columns[None, :]
    mask = (rows[:, None] < tokens) & (columns[None, :] < width)
    d = tl.load(d_ptr + offset, mask, 0).to(tl.float32)
    y = tl.load(y_ptr + offset, mask, 0).to(tl.float32)
    g = tl.load(g_ptr + batch_id * width + columns, columns < width, 0).to(tl.float32)
    tl.store(dy + offset, d * g[None, :], mask)
    # Preserve the rounding of eager BF16 multiplication before broadcast reduction.
    product = (d * y).to(y_ptr.dtype.element_ty).to(tl.float32)
    tl.store(dg + batch_id * width + columns, tl.sum(product, 0), columns < width)


class GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gate):
        output = torch.empty_like(x)
        batch, tokens, width = x.shape
        block = triton.next_power_of_2(triton.cdiv(tokens, max(32 // batch, 1)))
        _residual_forward[(batch, triton.cdiv(tokens, block))](x, y, gate, output, width, tokens, block)
        ctx.save_for_backward(y, gate)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        y, gate = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_y, grad_gate = torch.empty_like(y), torch.empty_like(gate)
        batch, tokens, width = y.shape
        _residual_backward[(batch, triton.cdiv(width, 64))](
            y,
            gate,
            grad_output,
            grad_y,
            grad_gate,
            width,
            tokens,
            batch,
            triton.next_power_of_2(tokens),
            64,
        )
        torch.ada.synchronize()
        return grad_output, grad_y, grad_gate
