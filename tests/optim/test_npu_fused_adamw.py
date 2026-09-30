"""Numerical, AMP, and checkpoint coverage for NPU fused AdamW."""

import copy

import pytest
import torch

from lerobot.optim.npu_fused_adamw import NpuFusedAdamW, is_npu_fused_adamw_available
from lerobot.optim.optimizers import AdamWConfig, load_optimizer_state, save_optimizer_state

pytestmark = pytest.mark.skipif(not is_npu_fused_adamw_available(), reason="requires NPU fused AdamW")


def make_groups(params):
    return [
        {"params": params[:2], "lr": 1e-3, "weight_decay": 0.01},
        {"params": params[2:], "lr": 3e-4, "weight_decay": 0, "betas": (0.8, 0.95)},
    ]


def test_adamw_config_selects_npu_backend_from_parameters():
    parameter = torch.nn.Parameter(torch.ones(8, device="npu"))
    optimizer = AdamWConfig().build([parameter])
    assert isinstance(optimizer, NpuFusedAdamW)


def test_shared_scalars_match_compatible_kernel_exactly():
    generator = torch.Generator().manual_seed(710)
    values = [torch.randn(7, 11, generator=generator) for _ in range(3)]
    shared_params = [torch.nn.Parameter(value.to("npu")) for value in values]
    compatible_params = [torch.nn.Parameter(value.to("npu")) for value in values]
    shared = NpuFusedAdamW(make_groups(shared_params))
    if shared._shared_adamw is None:
        pytest.skip("shared-scalar extension could not be built")
    compatible = NpuFusedAdamW(make_groups(compatible_params))
    compatible._shared_adamw = None

    for step in range(12):
        for index, (parameter, reference) in enumerate(zip(shared_params, compatible_params, strict=True)):
            grad = torch.randn(11, 7, generator=generator).t()
            missing = index == 1 and step % 3 == 0
            parameter.grad = None if missing else grad.to("npu")
            reference.grad = None if missing else parameter.grad.clone()
        shared.step()
        compatible.step()

    for parameter, reference in zip(shared_params, compatible_params, strict=True):
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                shared.state[parameter][key], compatible.state[reference][key], rtol=0, atol=0
            )
    assert [shared.state[p]["step"].item() for p in shared_params] == [12, 8, 12]


def test_matches_fp32_adamw_with_missing_and_noncontiguous_gradients():
    generator = torch.Generator().manual_seed(711)
    values = [torch.randn(7, 11, generator=generator) for _ in range(3)]
    reference = [torch.nn.Parameter(value.clone()) for value in values]
    params = [torch.nn.Parameter(value.to("npu")) for value in values]
    expected = torch.optim.AdamW(make_groups(reference), fused=True)
    optimizer = NpuFusedAdamW(make_groups(params))

    for step in range(12):
        for index, (parameter, expected_parameter) in enumerate(zip(params, reference, strict=True)):
            grad = torch.randn(11, 7, generator=generator).t()
            missing = index == 1 and step % 3 == 0
            parameter.grad = None if missing else grad.to("npu")
            expected_parameter.grad = None if missing else grad.contiguous()
        optimizer.step()
        expected.step()

    for parameter, expected_parameter in zip(params, reference, strict=True):
        torch.testing.assert_close(parameter.cpu(), expected_parameter, rtol=2e-6, atol=2e-6)


def test_safetensors_resume_keeps_cpu_step_state(tmp_path):
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 257, device="npu"))
    optimizer = NpuFusedAdamW([parameter], lr=1e-3)
    parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
    save_optimizer_state(optimizer, tmp_path)

    resumed = load_optimizer_state(NpuFusedAdamW([parameter], lr=1e-3), tmp_path)
    assert resumed.state[parameter]["step"].device.type == "cpu"
    assert resumed.state[parameter]["step"].item() == 1

    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    reference = NpuFusedAdamW([reference_parameter], lr=1e-3)
    reference.load_state_dict(copy.deepcopy(resumed.state_dict()))
    parameter.grad = torch.full_like(parameter, 0.2)
    reference_parameter.grad = parameter.grad.clone()
    resumed.step()
    reference.step()
    torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)


def test_grad_scaler_skips_overflow_and_unscales_before_step():
    parameter = torch.nn.Parameter(torch.ones(8, device="npu"))
    optimizer = NpuFusedAdamW([parameter], lr=1e-3, weight_decay=0)
    scaler = torch.npu.amp.GradScaler(init_scale=128)
    scaler.scale(parameter.sum() * float("inf")).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    assert not optimizer.state

    optimizer.zero_grad()
    scaler.scale(parameter.sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.999), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.full_like(parameter, 0.1))
    assert optimizer.state[parameter]["step"].item() == 1
