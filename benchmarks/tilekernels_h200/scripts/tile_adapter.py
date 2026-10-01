"""Experiment-only TileKernels RMSNorm adapters.

The adapters preserve model-specific rounding and affine boundaries. They do
not change model configs, checkpoints, or the production policy code.
"""

from __future__ import annotations

import torch
from tile_kernels.quant import norm_backward, norm_forward
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

from lerobot.policies.pi_gemma import PiGemmaRMSNorm

PI_ORIGINAL = PiGemmaRMSNorm._norm
QWEN_ORIGINAL = Qwen3VLTextRMSNorm.forward


class RMSCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        y, rstd, saved = norm_forward(flat, None, eps)
        ctx.save_for_backward(saved, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        x, rstd = ctx.saved_tensors
        dx, _ = norm_backward(grad.reshape(x.shape).contiguous(), x, None, rstd)
        return dx.view(ctx.shape), None


def rms_core(x: torch.Tensor, eps: float) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        return RMSCore.apply(x, eps)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    return norm_forward(flat, None, eps)[0].view(x.shape)


def pi_norm(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # AdaRMS consumes this FP32 value before scale/shift/gate.
    return rms_core(x.float(), self.eps)


def qwen_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    # Qwen rounds the normalized value to the input dtype before multiplying
    # the weight. Do not fold the weight into the TileKernels operation.
    return self.weight * rms_core(hidden_states, self.variance_epsilon)


def set_enabled(enabled: bool) -> None:
    PiGemmaRMSNorm._norm = pi_norm if enabled else PI_ORIGINAL
    Qwen3VLTextRMSNorm.forward = qwen_forward if enabled else QWEN_ORIGINAL


def is_candidate(module: nn.Module) -> bool:
    return isinstance(module, (PiGemmaRMSNorm, Qwen3VLTextRMSNorm))
