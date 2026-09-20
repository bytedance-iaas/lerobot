"""RoPE cache lifetime, fallback, and fused forward/backward regression checks."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma.modeling_gemma import (
    GemmaConfig,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from lerobot.policies.pi05.rope import apply_rotary_pos_emb_npu, cached_rotary_embeddings


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_cache_is_local_and_checkpoint_safe(dtype, checkpointed):
    torch.manual_seed(92)
    rotary = GemmaRotaryEmbedding(GemmaConfig(head_dim=64))
    for shift in [0, 21]:
        positions = (torch.arange(17) + shift)[None, :].expand(2, -1)
        x = torch.randn(2, 8, 17, 64, dtype=dtype, requires_grad=True)

        def run(cached):
            cache = {}
            calls = []
            handle = rotary.register_forward_hook(lambda *args: calls.append(1))

            def layer(value):
                c, s = (
                    cached_rotary_embeddings(rotary, value, positions, cache)
                    if cached else rotary(value, positions)
                )
                return apply_rotary_pos_emb(value, value[:, :1], c, s)[0]

            try:
                out = x
                for _ in range(3):
                    out = checkpoint(layer, out, use_reentrant=False) if checkpointed else layer(out)
                grad = torch.autograd.grad(out.float().square().sum(), x)[0]
                if cached:
                    assert len(calls) == 1
                return out, grad
            finally:
                handle.remove()

        for expected, actual in zip(run(False), run(True), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_fallback_preserves_position_gradients():
    q = torch.randn(2, 8, 17, 64, requires_grad=True)
    k = torch.randn(2, 1, 17, 64, requires_grad=True)
    cos = torch.randn(2, 17, 64, requires_grad=True)
    sin = torch.randn_like(cos, requires_grad=True)
    expected = apply_rotary_pos_emb(q, k, cos, sin)
    actual = apply_rotary_pos_emb_npu(q, k, cos, sin)
    dy = [torch.randn_like(x) for x in actual]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(actual, (q, k, cos, sin), dy),
        torch.autograd.grad(expected, (q, k, cos, sin), dy), strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 8, 17, 64), (16, 8, 762, 256)])
@pytest.mark.parametrize("strided_grad", [False, True])
def test_npu_rotation_is_exact(dtype, shape, strided_grad):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.manual_seed(741)
    batch, heads, tokens, dim = shape
    q = torch.randn(shape, device="npu:0", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, 1, tokens, dim, device=q.device, dtype=dtype, requires_grad=True)
    cos = torch.randn(batch, tokens, dim, device=q.device, dtype=dtype)
    sin = torch.randn_like(cos)
    expected = apply_rotary_pos_emb(q, k, cos, sin)
    actual = apply_rotary_pos_emb_npu(q, k, cos, sin)
    dy = [
        torch.randn(*x.shape[:-1], dim * 2, device=x.device, dtype=dtype)[..., ::2]
        if strided_grad else torch.randn_like(x)
        for x in actual
    ]
    expected_grads = torch.autograd.grad(expected, (q, k), dy)
    actual_grads = torch.autograd.grad(actual, (q, k), dy)
    for a, b in zip((*actual, *actual_grads), (*expected, *expected_grads), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
