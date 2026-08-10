#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.filterwarnings(
    "ignore:In CPU autocast, but the target dtype is not supported:UserWarning"
)

from conftest import (  # noqa: E402
    BATCH_SIZE,
    IMAGE_SIZE,
    QWEN_HIDDEN_SIZE,
    make_config,
    make_monitor,
)

from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402
from lerobot.policies.vla_jepa.monitor import MonitorOutput  # noqa: E402


def _frames(batch_size: int = BATCH_SIZE, num_views: int = 1) -> torch.Tensor:
    """One control step of observations: [B, V, C, H, W] in [0, 1]."""
    return torch.rand(batch_size, num_views, 3, IMAGE_SIZE, IMAGE_SIZE)


def _action_tokens(batch_size: int = BATCH_SIZE, num_tokens: int = 2) -> torch.Tensor:
    return torch.randn(batch_size, num_tokens, QWEN_HIDDEN_SIZE)


def _build():
    """Build a policy and a monitor wired to its frozen submodules.

    Callers must request the `patch_vla_jepa_external_models` fixture so the fake Qwen and
    V-JEPA models are installed before this runs.
    """
    config = make_config(enable_wm_feedback=True)
    policy = VLAJEPAPolicy(config)
    policy.eval()
    return policy, config, make_monitor(policy, config)


def test_monitor_returns_none_while_warming_up(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    # Buffer needs 2 * tubelet_size frames; the fake encoder uses tubelet_size=1, so 2.
    assert monitor.observe(_frames(), _action_tokens()) is None


def test_monitor_emits_once_buffer_is_full(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    monitor.observe(_frames(), _action_tokens())
    out = monitor.observe(_frames(), _action_tokens())

    assert isinstance(out, MonitorOutput)
    assert out.error.shape == (BATCH_SIZE,)
    assert out.residual.shape == out.predicted.shape
    assert out.residual.shape[0] == BATCH_SIZE
    assert torch.isfinite(out.error).all()
    assert (out.error >= 0).all()


def test_monitor_error_is_mean_abs_residual(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    monitor.observe(_frames(), _action_tokens())
    out = monitor.observe(_frames(), _action_tokens())

    expected = out.residual.abs().mean(dim=(1, 2))
    torch.testing.assert_close(out.error, expected)


def test_monitor_emits_every_tubelet_stride(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    emissions = [monitor.observe(_frames(), _action_tokens()) is not None for _ in range(6)]
    # tubelet_size=1 => first emit on step 2, then every step after.
    assert emissions == [False, True, True, True, True, True]


def test_monitor_records_error_history(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    for _ in range(4):
        monitor.observe(_frames(), _action_tokens())

    assert len(monitor.error_history) == BATCH_SIZE
    assert all(len(h) == 3 for h in monitor.error_history)
    assert all(isinstance(v, float) for h in monitor.error_history for v in h)


def test_monitor_reset_clears_buffer_and_history(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    for _ in range(3):
        monitor.observe(_frames(), _action_tokens())
    assert monitor.error_history[0]

    monitor.reset()
    assert monitor.error_history == []
    # After reset the monitor must warm up again rather than emit immediately.
    assert monitor.observe(_frames(), _action_tokens()) is None


def test_monitor_pads_short_action_tokens(patch_vla_jepa_external_models: None) -> None:
    """The predictor needs t_enc_ctx * num_action_tokens_per_timestep tokens; fewer must be padded."""
    _, _, monitor = _build()
    monitor.observe(_frames(), _action_tokens(num_tokens=1))
    out = monitor.observe(_frames(), _action_tokens(num_tokens=1))
    assert out is not None
    assert torch.isfinite(out.error).all()


def test_monitor_truncates_long_action_tokens(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    monitor.observe(_frames(), _action_tokens(num_tokens=16))
    out = monitor.observe(_frames(), _action_tokens(num_tokens=16))
    assert out is not None
    assert torch.isfinite(out.error).all()


def test_monitor_runs_under_no_grad(patch_vla_jepa_external_models: None) -> None:
    _, _, monitor = _build()
    monitor.observe(_frames(), _action_tokens())
    out = monitor.observe(_frames(), _action_tokens())
    assert out.residual.requires_grad is False
    assert out.predicted.requires_grad is False


def test_dump_error_history_writes_jsonl(patch_vla_jepa_external_models: None, tmp_path) -> None:
    import json

    _, _, monitor = _build()
    for _ in range(4):
        monitor.observe(_frames(), _action_tokens())

    out = tmp_path / "errors.jsonl"
    monitor.dump_error_history(out)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == BATCH_SIZE
    assert [r["batch_index"] for r in records] == list(range(BATCH_SIZE))
    assert all(len(r["errors"]) == 3 for r in records)
    assert all(isinstance(v, float) for r in records for v in r["errors"])


def test_dump_error_history_on_empty_monitor(patch_vla_jepa_external_models: None, tmp_path) -> None:
    _, _, monitor = _build()
    out = tmp_path / "empty.jsonl"
    monitor.dump_error_history(out)
    assert out.read_text() == ""
