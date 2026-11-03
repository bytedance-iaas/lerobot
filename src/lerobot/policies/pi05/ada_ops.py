"""Automatic Ada2S training-operator selection for Pi0.5."""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.device_utils import is_ada2s_device


def rms_norm(x, weight, eps):
    """Use AdaDNN RMSNorm, including its registered input/weight backward."""
    return torch.ops.ada.fused_rms_norm(x.contiguous(), [x.shape[-1]], weight, eps)


def rotary(q, k, cos, sin):
    """Apply differentiable AdaDNN split-half rotation to Q and K."""
    return (
        torch.ops.ada.rope(q.contiguous(), sin.contiguous(), cos.contiguous(), 1.0),
        torch.ops.ada.rope(k.contiguous(), sin.contiguous(), cos.contiguous(), 1.0),
    )


def gelu_mul(gate, up):
    if not is_ada2s_device(gate.device):
        return F.gelu(gate, approximate="tanh") * up
    from ._ada_kernels import GeGLU

    return GeGLU.apply(gate.contiguous(), up.contiguous())


def gated_residual(x, y, gate):
    from ._ada_kernels import GatedResidual

    return GatedResidual.apply(x.contiguous(), y.contiguous(), gate.contiguous())


class AdaGeGLUMLP(nn.Module):
    """Reuse Gemma's parameter names and projections; fuse GELU(tanh) with gating."""

    def __init__(self, mlp):
        super().__init__()
        self.config = mlp.config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn
        self.train(mlp.training)

    def forward(self, x):
        return self.down_proj(gelu_mul(self.gate_proj(x), self.up_proj(x)))


class AdaGELU(nn.Module):
    """GELU(tanh) with a fused Ada2S forward and backward."""

    def forward(self, x):
        if not is_ada2s_device(x.device):
            return F.gelu(x, approximate="tanh")
        from ._ada_kernels import GELU

        return GELU.apply(x.contiguous())


class AdaGemmaAttention(nn.Module):
    """Use Ada RoPE and attention in cached Gemma inference as well as training."""

    def __init__(self, attention):
        super().__init__()
        for name in (
            "config",
            "layer_idx",
            "head_dim",
            "num_key_value_groups",
            "scaling",
            "attention_dropout",
            "is_causal",
        ):
            setattr(self, name, getattr(attention, name))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(attention, name))
        self.train(attention.training)

    def forward(
        self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs
    ):
        from .attention import sdpa_attention_forward

        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        if is_ada2s_device(q.device):
            q, k = rotary(q, k, cos, sin)
        else:
            from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        output, _ = sdpa_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1).contiguous()), None


def configure_ada2s_model(model):
    """Install the Ada2S MLP implementation without changing checkpoint keys."""
    for decoder in (model.paligemma.model.language_model, model.gemma_expert.model):
        for layer in decoder.layers:
            if layer.mlp.config.hidden_act != "gelu_pytorch_tanh":
                raise ValueError("Ada2S GeGLU requires GELU(tanh)")
            layer.mlp = AdaGeGLUMLP(layer.mlp)
            layer.self_attn = AdaGemmaAttention(layer.self_attn)
    # SigLIP's noncausal vision attention uses the same forward/backward
    # interface; its projections and LayerNorm already use native Ada kernels.
    for module in model.paligemma.model.vision_tower.modules():
        if hasattr(module, "config"):
            module.config._attn_implementation = "lerobot_sdpa"
        if type(module).__name__ == "SiglipMLP":
            if module.config.hidden_act != "gelu_pytorch_tanh":
                raise ValueError("Ada2S vision GELU requires GELU(tanh)")
            module.activation_fn = AdaGELU()
