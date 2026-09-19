"""Numerical and checkpoint compatibility of the optional GeGLU path."""

import copy

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint

from lerobot.policies.pi05.npu_geglu import NpuGeGLUMLP, gelu_mul


@pytest.fixture
def npu_device():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    return torch.device("npu:0")


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "broadcast", "empty"])
def test_cpu_fallback(layout):
    gate = torch.randn(3, 5, requires_grad=True)
    up = torch.randn(3, 5, requires_grad=True)
    if layout == "transpose":
        gate, up = gate.T, up.T
    elif layout == "broadcast":
        up = up[:1]
    elif layout == "empty":
        gate, up = gate[:0], up[:0]
    actual = gelu_mul(gate, up)
    expected = F.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, (gate, up), dy),
        torch.autograd.grad(expected, (gate, up), dy),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided_dy", "extremes"])
def test_npu_output_and_gradients(npu_device, dtype, layout):
    torch.manual_seed(123)
    gate = torch.randn(33, 257, device=npu_device, dtype=dtype, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    if layout == "transpose":
        gate, up = gate.T, up.T
    elif layout == "extremes":
        gate = torch.linspace(-20, 20, gate.numel(), device=npu_device).to(dtype).reshape_as(gate)
        gate.requires_grad_()
    dy = torch.randn_like(gate)
    if layout == "strided_dy":
        dy = torch.randn(*gate.shape, 2, device=npu_device, dtype=dtype)[..., 0]
    actual = gelu_mul(gate, up)
    expected = F.gelu(gate, approximate="tanh") * up
    actual_grads = torch.autograd.grad(actual, (gate, up), dy)
    expected_grads = torch.autograd.grad(expected, (gate, up), dy)
    for a, b in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        # BF16: allow up to two relative ULPs plus a small cancellation floor.
        # Also bound aggregate error to prevent a loose elementwise check hiding drift.
        torch.testing.assert_close(a, b, rtol=0.016 if dtype == torch.bfloat16 else 2e-5, atol=2e-5)
        relative_l2 = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
        assert relative_l2.item() < 2e-5


@pytest.mark.parametrize("checkpointed", [False, True])
def test_npu_mlp_weights_and_checkpoint(npu_device, checkpointed):
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaMLP

    torch.manual_seed(321)
    cfg = GemmaConfig(hidden_size=64, intermediate_size=256, hidden_act="gelu_pytorch_tanh")
    reference = GemmaMLP(cfg).to(device=npu_device, dtype=torch.bfloat16)
    candidate = NpuGeGLUMLP(copy.deepcopy(reference))
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 17, 64, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    y = reference(x)
    yy = checkpoint(candidate, xx, use_reentrant=False) if checkpointed else candidate(xx)
    dy = torch.randn_like(y)
    y.backward(dy)
    yy.backward(dy)
    for a, b in zip(
        [yy, xx.grad, *(p.grad for p in candidate.parameters())],
        [y, x.grad, *(p.grad for p in reference.parameters())],
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0.016, atol=2e-4)
