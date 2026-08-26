"""Unit tests for the LingBot-VA dynamics-context latent (execution feedback, no weight update).

These cover the parts that can be checked without the 20 GB of frozen sub-models: the rule that
turns the residual stream into ``e``, and the token injection that carries ``e`` into the model.
The end-to-end question -- whether injecting ``e`` actually changes the actions -- needs the real
checkpoint and lives in ``tests/scripts/evals/context_pathway_check.py``.
"""

from collections import deque

import pytest
import torch

from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy


class _Stub:
    """A LingBot-VA policy with only the context-latent state, no transformer or VAE."""

    def __init__(self, **kw):
        self.config = LingBotVAConfig(**kw)
        self.config.device = "cpu"
        self._ctx_e = None
        self._ctx_window = deque(maxlen=max(2, self.config.context_window))
        self.context_history = []
        self.last_residual = None
        self.last_residual_per_frame = []
        self.dtype = torch.float32

    # Borrow the real implementations.
    _context_enabled = LingBotVAPolicy._context_enabled
    _update_context = LingBotVAPolicy._update_context
    _ctx_proj = LingBotVAPolicy._ctx_proj
    _cond_emb = LingBotVAPolicy._cond_emb

    def feed(self, r, per_frame=None):
        self.last_residual = r
        self.last_residual_per_frame = per_frame or [r] * 4
        self._update_context()


def test_disabled_by_default_is_a_no_op():
    s = _Stub()
    for r in (0.3, 0.4, 0.5):
        s.feed(r)
    assert s._ctx_e is None
    assert s.context_history == []
    emb = torch.randn(1, 8, 4096)
    assert torch.equal(s._cond_emb(emb), emb)  # not even a shape change


def test_no_signal_until_a_baseline_exists():
    """The first two chunks have nothing to standardize against; e must stay at zero rather than
    reporting a spurious deviation."""
    s = _Stub(context_latent_dim=6)
    s.feed(0.40, [0.40] * 4)
    assert torch.allclose(s._ctx_e, torch.zeros(6))


def test_e_tracks_deviation_from_baseline_not_absolute_residual():
    """The whole design point: a high but *steady* residual (a hard scene) must leave e near zero,
    while a jump away from the episode's own baseline must move it."""
    steady = _Stub(context_latent_dim=6)
    for _ in range(10):
        steady.feed(0.64, [0.64] * 4)  # task 1750's mean residual -- high, but constant

    jumpy = _Stub(context_latent_dim=6)
    for _ in range(8):
        jumpy.feed(0.30 + 0.001 * _, [0.30] * 4)  # low, near-constant baseline
    jumpy.feed(0.45, [0.45] * 4)  # a jump of many sigmas

    assert steady._ctx_e.abs().max() < jumpy._ctx_e.abs().max()


def test_changepoint_resets_the_baseline_window():
    s = _Stub(context_latent_dim=4, context_reset_z=3.0, context_lr=0.05)
    for i in range(8):
        s.feed(0.30 + 0.001 * i)
    assert len(s._ctx_window) > 2
    before = s._ctx_e.clone()
    s.feed(0.9)  # far outside the baseline -> regime change
    assert len(s._ctx_window) == 1  # window dropped, then the new value appended
    # A changepoint jumps e straight to the new context instead of easing there via the EMA.
    assert not torch.allclose(s._ctx_e, before, atol=1e-3)


def test_ema_smooths_when_there_is_no_changepoint():
    s = _Stub(context_latent_dim=4, context_lr=0.05, context_reset_z=100.0)
    for i in range(8):
        s.feed(0.30 + 0.001 * i)
    before = s._ctx_e.clone()
    s.feed(0.34)
    step = (s._ctx_e - before).abs().max()
    assert 0 < step < 1.0  # moved, but only by the EMA rate


def test_e_is_truncated_or_padded_to_the_configured_dim():
    for d in (1, 2, 6, 16):
        s = _Stub(context_latent_dim=d)
        for i in range(5):
            s.feed(0.3 + 0.01 * i)
        assert s._ctx_e.shape == (d,)


def test_injected_token_is_scaled_against_real_tokens_not_padding():
    """``_get_t5_prompt_embeds`` zero-pads to 512; scaling against the padded mean would shrink the
    injected token by more than an order of magnitude."""
    s = _Stub(context_latent_dim=6, context_inject_scale=1.0)
    for i in range(6):
        s.feed(0.3 + 0.02 * i)

    emb = torch.zeros(1, 512, 4096)
    emb[:, :20] = torch.randn(1, 20, 4096)  # 20 real tokens, 492 zero pads
    out = s._cond_emb(emb)

    assert out.shape == (1, 513, 4096)
    assert torch.equal(out[:, :512], emb)
    real_mean = emb[:, :20].norm(dim=-1).mean()
    assert out[:, 512].norm() == pytest.approx(real_mean.item(), rel=1e-2)


def test_inject_scale_zero_gives_a_zero_token_control_arm():
    s = _Stub(context_latent_dim=6, context_inject_scale=0.0)
    for i in range(6):
        s.feed(0.3 + 0.02 * i)
    emb = torch.randn(1, 8, 4096)
    out = s._cond_emb(emb)
    assert out.shape == (1, 9, 4096)
    assert out[:, 8].abs().max() == 0.0


def test_projection_is_deterministic_per_seed_and_differs_across_seeds():
    a = _Stub(context_latent_dim=6, context_proj_seed=0)
    b = _Stub(context_latent_dim=6, context_proj_seed=0)
    c = _Stub(context_latent_dim=6, context_proj_seed=1)
    pa = a._ctx_proj(6, torch.device("cpu"), torch.float32)
    pb = b._ctx_proj(6, torch.device("cpu"), torch.float32)
    pc = c._ctx_proj(6, torch.device("cpu"), torch.float32)
    assert torch.equal(pa, pb)
    assert not torch.equal(pa, pc)


def test_residual_measurement_is_forced_on_when_context_is_enabled():
    """e is driven by the residual stream, so enabling the context path must enable measurement
    even if the user did not set track_prediction_residual."""
    cfg = LingBotVAConfig(context_latent_dim=6)
    assert cfg.track_prediction_residual is False

    class _P:
        config = cfg

    assert LingBotVAPolicy._residual_enabled.fget(_P()) is True
