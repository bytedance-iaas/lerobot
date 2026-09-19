"""Opt-in GeGLU activation fusion for Pi0.5's Gemma MLPs on Ascend.

Weights and state-dict keys are unchanged. Only GELU(tanh) and the gating
multiply are fused; the three linear projections still use PyTorch.
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def gelu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Use the NPU kernel for contiguous FP32/BF16 inputs, otherwise eager ops."""
    supported = (
        type(gate) is torch.Tensor
        and type(up) is torch.Tensor
        and gate.device.type == "npu"
        and gate.device == up.device
        and gate.dtype == up.dtype
        and gate.dtype in (torch.float32, torch.bfloat16)
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.numel() > 0
    )
    if not supported:
        return F.gelu(gate, approximate="tanh") * up

    # Keep the optional Ascend Triton dependency out of CPU/CUDA imports.
    from ._npu_geglu import GeGLU

    return GeGLU.apply(gate, up)


class NpuGeGLUMLP(nn.Module):
    """Reuse an existing Gemma MLP's parameters without nesting its state dict."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.config = mlp.config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.train(mlp.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(gelu_mul(gate, up))


def configure_npu_geglu(layers: nn.ModuleList) -> None:
    """Install only on Gemma MLPs configured with the expected tanh GELU."""
    for layer in layers:
        mlp = layer.mlp
        if isinstance(mlp, NpuGeGLUMLP):
            continue
        if mlp.config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("NPU GeGLU fusion requires gelu_pytorch_tanh")
        layer.mlp = NpuGeGLUMLP(mlp)
