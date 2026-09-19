"""Compatibility checks for the opt-in clipping entry point."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.optim.grad_clip import clip_grad_norm_npu_


@pytest.mark.parametrize("distributed_type", ["NO", "FSDP", "DEEPSPEED"])
def test_preserves_accelerator_fallback(distributed_type):
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([3.0, 4.0, 0.0])
    calls = []

    def clip(params, max_norm, norm_type):
        calls.append((params, max_norm, norm_type))
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    def unscale():
        raise AssertionError("Fallback must let Accelerator handle unscaling")

    accelerator = SimpleNamespace(
        distributed_type=SimpleNamespace(value=distributed_type),
        num_processes=1,
        clip_grad_norm_=clip,
        unscale_gradients=unscale,
    )
    norm = clip_grad_norm_npu_(accelerator, iter([p]), 1.0)
    assert len(calls) == 1
    assert calls[0][0] == [p]
    torch.testing.assert_close(norm, torch.tensor(5.0))
    torch.testing.assert_close(p.grad, torch.tensor([3.0, 4.0, 0.0]) / (5.0 + 1e-6))


def test_empty_parameters_preserve_accelerator_return():
    calls = []
    accelerator = SimpleNamespace(
        distributed_type="DEEPSPEED",
        num_processes=2,
        clip_grad_norm_=lambda params, max_norm, norm_type: calls.append(params),
    )
    assert clip_grad_norm_npu_(accelerator, iter([]), 1.0) is None
    assert calls == [[]]

