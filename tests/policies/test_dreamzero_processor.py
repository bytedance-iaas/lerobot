#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU numerics tests for the DreamZero processor.

These lock the deterministic parts of the port — q99 (de)normalization, the oxe_droid
multi-view stitch, per-view crop/resize, and the relative->absolute action decode — against
the upstream DreamZero semantics. The umt5 tokenizer path and the
GPU model are validated separately (they need the released checkpoint + a GPU).
"""

import pytest
import torch

pytest.importorskip("diffusers")  # dreamzero config imports through the policy package

from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig  # noqa: E402
from lerobot.policies.dreamzero.processor_dreamzero import (  # noqa: E402
    DreamZeroActionDecodeStep,
    DreamZeroPackInputsStep,
    _crop_resize_view,
    _crop_size,
    _q99_forward,
    _q99_inverse,
    _QuantileLayout,
    _sample_crop_offset,
    _stitch_oxe_droid,
    _to_bthwc_uint8,
)
from lerobot.processor.relative_action_processor import to_relative_actions  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402


def test_q99_roundtrip_and_range():
    q01 = torch.tensor([-1.0, 0.0, 3.0, 5.0])
    q99 = torch.tensor([1.0, 4.0, 3.0, 9.0])  # dim 2: q01 == q99 (passthrough)
    x = torch.tensor([[0.3, 2.0, 0.5, 7.0], [-0.9, 3.9, 0.5, 5.5]])
    n = _q99_forward(x, q01, q99)
    # normalized stays in [-1, 1]
    assert (n.abs() <= 1 + 1e-6).all()
    # invertible on the non-degenerate dims
    xi = _q99_inverse(n, q01, q99)
    assert torch.allclose(xi[:, [0, 1, 3]], x[:, [0, 1, 3]], atol=1e-5)
    # q01 == q99 dim: upstream passes the raw value through, then clamps to [-1, 1]
    assert torch.allclose(n[:, 2], x[:, 2].clamp(-1, 1))


def test_stitch_oxe_droid_layout():
    h, w = 4, 6
    ext1 = torch.full((1, 1, h, w, 3), 11, dtype=torch.uint8)
    ext2 = torch.full((1, 1, h, w, 3), 22, dtype=torch.uint8)
    wrist = torch.full((1, 1, h, w, 3), 33, dtype=torch.uint8)
    canvas = _stitch_oxe_droid([ext1, ext2, wrist], h, w)
    assert tuple(canvas.shape) == (1, 1, 2 * h, 2 * w, 3)
    assert (canvas[:, :, :h, :, :] == 33).all()  # top row: wrist, repeated across full width
    assert (canvas[:, :, h:, :w, :] == 11).all()  # bottom-left: exterior_1
    assert (canvas[:, :, h:, w:, :] == 22).all()  # bottom-right: exterior_2


def test_stitch_oxe_droid_stretches_the_wrist_view():
    """The wrist view is upscaled 2x across the top row, not tiled twice.

    The layout test above fills each view with a constant, which cannot tell
    ``repeat_interleave`` (``[a,b,c] -> [a,a,b,b,c,c]``, what upstream's
    ``np.repeat(wrist, 2, axis=-1)`` does) apart from ``Tensor.repeat``
    (``[a,b,c] -> [a,b,c,a,b,c]``, two wrist cameras side by side). A horizontal gradient does.
    """
    h, w = 2, 3
    ext = torch.zeros((1, 1, h, w, 3), dtype=torch.uint8)
    ramp = torch.arange(1, w + 1, dtype=torch.uint8)
    wrist = ramp.view(1, 1, 1, w, 1).expand(1, 1, h, w, 3).contiguous()

    canvas = _stitch_oxe_droid([ext, ext, wrist], h, w)

    top_row = canvas[0, 0, 0, :, 0]
    assert top_row.tolist() == [1, 1, 2, 2, 3, 3]  # stretched
    assert top_row.tolist() != [1, 2, 3, 1, 2, 3]  # not tiled


def test_crop_resize_shape_and_dtype():
    img = torch.rand(2, 3, 200, 400)  # (B, C, H, W) float in [0, 1]
    out = _crop_resize_view(_to_bthwc_uint8(img), 0.95, 160, 320)
    assert tuple(out.shape) == (2, 1, 160, 320, 3)
    assert out.dtype == torch.uint8


def test_action_decode_relative_to_absolute():
    cfg = DreamZeroConfig()  # oxe_droid defaults, action keys (joint_position, gripper_position)
    stats = {
        "action": {
            "joint_position": {"q01": [-2.0] * 7, "q99": [2.0] * 7},  # RELATIVE stats
            "gripper_position": {"q01": [0.0], "q99": [1.0]},  # ABSOLUTE stats
        }
    }
    decode = DreamZeroActionDecodeStep(cfg, stats)
    pack = DreamZeroPackInputsStep(cfg, stats)
    last_joint = torch.arange(7, dtype=torch.float32).unsqueeze(0)  # (1, 7)
    pack._last_raw_state = {"joint_position": last_joint}
    decode.pack_step = pack

    b, horizon, d = 1, 3, cfg.max_action_dim  # 32
    action = torch.zeros(b, horizon, d)
    action[..., :8] = 0.5  # normalized; pad dims 8:32 are dropped by decode

    out = decode({TransitionKey.ACTION: action.clone()})[TransitionKey.ACTION]
    # q99 inverse of 0.5: (0.5 + 1) / 2 * (2 - (-2)) + (-2) = 1.0
    exp_joint = 1.0 + last_joint  # relative -> absolute: + last observed joint state
    exp_grip = (0.5 + 1) / 2 * (1.0 - 0.0) + 0.0  # 0.75, gripper stays absolute

    assert tuple(out.shape) == (b, horizon, 8)  # padding dropped
    assert torch.allclose(out[0, :, :7], exp_joint.expand(horizon, 7), atol=1e-5)
    assert torch.allclose(out[0, :, 7], torch.full((horizon,), exp_grip), atol=1e-5)


def test_action_pack_decode_roundtrip():
    """Training-forward (relative-encode + q99) then decode-inverse recovers the raw action."""
    cfg = DreamZeroConfig()
    stats = {
        "action": {
            "joint_position": {"q01": [-2.0] * 7, "q99": [2.0] * 7},  # RELATIVE stats
            "gripper_position": {"q01": [0.0], "q99": [1.0]},  # ABSOLUTE stats
        }
    }
    torch.manual_seed(0)
    raw = torch.randn(1, 3, 8) * 0.3  # small so the relative action stays within [-2, 2] (no clamp)
    raw[..., 7] = 0.4  # gripper in [0, 1]
    ref_joint = torch.randn(1, 7) * 0.3

    # forward pack (mirror of the DreamZeroPackInputsStep action block)
    layout = _QuantileLayout(stats, "action", cfg.action_modality_keys)
    reference = torch.zeros(1, 8)
    reference[:, :7] = ref_joint
    mask = [True] * 7 + [False]
    rel = to_relative_actions(raw.clone(), reference, mask)
    norm = _q99_forward(rel, layout.q01, layout.q99)
    padded = torch.nn.functional.pad(norm, (0, cfg.max_action_dim - 8))

    # decode-inverse
    decode = DreamZeroActionDecodeStep(cfg, stats)
    pack = DreamZeroPackInputsStep(cfg, stats)
    pack._last_raw_state = {"joint_position": ref_joint}
    decode.pack_step = pack
    out = decode({TransitionKey.ACTION: padded.clone()})[TransitionKey.ACTION]

    assert torch.allclose(out, raw, atol=1e-4)


def _decode_pair(n_action_steps=4):
    """A pack/decode pair wired the way make_dreamzero_pre_post_processors wires them."""
    cfg = DreamZeroConfig(n_action_steps=n_action_steps)
    stats = {
        "action": {
            "joint_position": {"q01": [-2.0] * 7, "q99": [2.0] * 7},
            "gripper_position": {"q01": [0.0], "q99": [1.0]},
        },
        "state": {
            "joint_position": {"q01": [-2.0] * 7, "q99": [2.0] * 7},
            "gripper_position": {"q01": [0.0], "q99": [1.0]},
        },
    }
    pack = DreamZeroPackInputsStep(cfg, stats)
    decode = DreamZeroActionDecodeStep(cfg, stats)
    decode.pack_step = pack
    return cfg, pack, decode


def _decode_one(decode, cfg):
    """Decode a single zero action; the joint dims come back as the anchor state."""
    action = torch.zeros(1, cfg.max_action_dim)
    action[..., :8] = -1.0  # q99 inverse of -1 is q01, which is 0 for these stats after +anchor
    return decode({TransitionKey.ACTION: action.clone()})[TransitionKey.ACTION]


def test_decode_holds_one_anchor_for_the_whole_chunk():
    """Every action of a chunk decodes against the frame the chunk was predicted from.

    `select_action` pops one action per control tick, and the preprocessor runs again on each new
    frame — so reading the pack step's latest state would re-anchor mid-chunk and re-add motion
    that already happened.
    """
    cfg, pack, decode = _decode_pair(n_action_steps=4)

    pack._cache_raw_state(torch.full((1, 8), 0.5))  # frame t0 -> the chunk's anchor
    first = _decode_one(decode, cfg)
    for step in range(1, 4):  # frames t1..t3: state moves, chunk is still being consumed
        pack._cache_raw_state(torch.full((1, 8), 0.5 + 0.1 * step))
        later = _decode_one(decode, cfg)
        assert torch.allclose(later, first), f"action {step} re-anchored on the current frame"

    # The 5th action belongs to the next chunk, so it must pick up the newest state.
    pack._cache_raw_state(torch.full((1, 8), 9.0))
    assert not torch.allclose(_decode_one(decode, cfg), first)


def test_decode_reset_drops_the_latched_anchor():
    """A new episode must re-anchor immediately rather than finish the previous chunk's count."""
    cfg, pack, decode = _decode_pair(n_action_steps=4)

    pack._cache_raw_state(torch.full((1, 8), 0.5))
    first = _decode_one(decode, cfg)

    decode.reset()
    pack._cache_raw_state(torch.full((1, 8), 9.0))
    assert not torch.allclose(_decode_one(decode, cfg), first)


def test_decode_whole_chunk_re_anchors_each_call():
    """predict_action_chunk + one postprocessor pass anchors per call, not per n_action_steps."""
    cfg, pack, decode = _decode_pair(n_action_steps=4)
    horizon = 6  # a full chunk is longer than n_action_steps

    pack._cache_raw_state(torch.full((1, 8), 0.5))
    chunk = torch.zeros(1, horizon, cfg.max_action_dim)
    chunk[..., :8] = -1.0
    first = decode({TransitionKey.ACTION: chunk.clone()})[TransitionKey.ACTION]

    pack._cache_raw_state(torch.full((1, 8), 9.0))
    second = decode({TransitionKey.ACTION: chunk.clone()})[TransitionKey.ACTION]
    assert not torch.allclose(second, first)


# ----------------------------------------------------------------------------------------------
# training window packing
# ----------------------------------------------------------------------------------------------
def _training_pack(n_state_dim=8):
    cfg = DreamZeroConfig()
    stats = {
        "state": {
            "joint_position": {"q01": [-4.0] * 7, "q99": [4.0] * 7},
            "gripper_position": {"q01": [0.0], "q99": [1.0]},
        },
        "action": {
            "joint_position": {"q01": [-2.0] * 7, "q99": [2.0] * 7},  # RELATIVE stats
            "gripper_position": {"q01": [0.0], "q99": [1.0]},
        },
    }
    return cfg, DreamZeroPackInputsStep(cfg, stats)


def test_block_anchor_rows_follow_the_window_geometry():
    """Block c starts at raw offset c*action_horizon, i.e. observation row c*horizon/stride."""
    cfg, pack = _training_pack()
    assert cfg.video_frame_stride == 3
    assert pack._block_anchor_rows == [0, 8, 16, 24]
    # Those rows sit at raw offsets 0/24/48/72 of the 33-row observation window.
    obs_offsets = cfg.observation_delta_indices
    assert [obs_offsets[r] for r in pack._block_anchor_rows] == [0, 24, 48, 72]


def test_training_actions_are_relative_to_their_own_block_anchor():
    """Each macro block subtracts the state at its own first row, not one window-wide anchor.

    Anchoring the whole window on block 0 would make block 3's displacement ~4x larger than the
    quantiles it is normalized against, saturating the q99 clamp.
    """
    cfg, pack = _training_pack()
    n_rows, horizon, blocks = cfg.num_frames, cfg.action_horizon, cfg.max_chunk_size

    # A state that ramps by exactly 1.0 per macro block on every joint dim.
    state = torch.zeros(1, n_rows, 8)
    for row in range(n_rows):
        state[0, row, :7] = cfg.observation_delta_indices[row] / horizon
    # Actions equal to their own block's anchor -> every relative action must come out 0.
    action = torch.zeros(1, horizon * blocks, 8)
    for c in range(blocks):
        action[0, c * horizon : (c + 1) * horizon, :7] = float(c)

    pack._cache_raw_state(state[:, 0])
    pack._block_anchor_state = pack._split_by_key(state[:, pack._block_anchor_rows])
    rel = pack._encode_relative_actions(action[..., :8].clone(), training=True)

    assert torch.allclose(rel[..., :7], torch.zeros_like(rel[..., :7]), atol=1e-6)
    # The gripper is not a relative key, so it passes through untouched.
    assert torch.allclose(rel[..., 7], action[..., 7])


def test_single_window_anchor_would_saturate_later_blocks():
    """Guards the reason for per-block anchoring: one anchor blows past the q99 range."""
    cfg, pack = _training_pack()
    horizon, blocks = cfg.action_horizon, cfg.max_chunk_size
    action = torch.zeros(1, horizon * blocks, 8)
    for c in range(blocks):
        action[0, c * horizon : (c + 1) * horizon, :7] = float(c)  # up to +3.0 by block 3

    # Inference-style single anchor at 0.0 leaves block 3 at +3.0, outside the +/-2.0 q99 span.
    pack._last_raw_state = {"joint_position": torch.zeros(1, 7), "gripper_position": torch.zeros(1, 1)}
    single = _q99_forward(
        pack._encode_relative_actions(action[..., :8].clone(), training=False),
        pack._action_layout.q01,
        pack._action_layout.q99,
    )
    assert single[0, -1, 0] == 1.0  # clamped: information destroyed


def test_training_window_rejects_padding():
    """drop_n_last_frames should make short windows unreachable; a pad flag means it did not."""
    cfg, pack = _training_pack()
    with pytest.raises(ValueError, match="must be complete"):
        pack._assert_window_not_padded(
            {TransitionKey.OBSERVATION: {"action_is_pad": torch.tensor([True])}}, {}
        )
    pack._assert_window_not_padded({TransitionKey.OBSERVATION: {"action_is_pad": torch.tensor([False])}}, {})


def test_crop_offset_is_shared_across_views():
    """One crop offset for every camera — independent crops would misalign the stitched canvas.

    Upstream concatenates all views and frames into a single batch and transforms it once
    (`groot/vla/data/transform/video.py` apply), so the offset cannot vary per view.
    """
    h, w, scale = 180, 320, 0.95
    ch, cw = _crop_size(h, w, scale)
    assert (ch, cw) == (171, 304)  # int(h*scale), int(w*scale) — upstream's formula

    # A view whose pixel value encodes its row, so the crop offset is readable from the output.
    ramp = torch.arange(h, dtype=torch.uint8).view(1, 1, h, 1, 1).expand(1, 1, h, w, 3).contiguous()
    offset = (7, 11)
    a = _crop_resize_view(ramp, scale, ch, cw, offset)
    b = _crop_resize_view(ramp, scale, ch, cw, offset)
    assert torch.equal(a, b), "same offset must give the same crop"
    # A different offset must actually move the window (otherwise the arg is being ignored).
    assert not torch.equal(a, _crop_resize_view(ramp, scale, ch, cw, (0, 0)))


def test_crop_offset_defaults_to_centre():
    """Inference centre-crops, matching upstream's eval mode."""
    h, w, scale = 180, 320, 0.95
    ch, cw = _crop_size(h, w, scale)
    ramp = torch.arange(h, dtype=torch.uint8).view(1, 1, h, 1, 1).expand(1, 1, h, w, 3).contiguous()
    centre = ((h - ch) // 2, (w - cw) // 2)
    assert torch.equal(
        _crop_resize_view(ramp, scale, ch, cw, None), _crop_resize_view(ramp, scale, ch, cw, centre)
    )


def test_sampled_crop_offset_stays_in_bounds():
    h, w, scale = 180, 320, 0.95
    ch, cw = _crop_size(h, w, scale)
    for _ in range(200):
        top, left = _sample_crop_offset(h, w, scale)
        assert 0 <= top <= h - ch and 0 <= left <= w - cw


def test_sorted_image_key_fallback_skips_pad_flags():
    """A training batch carries `observation.images.*_is_pad` flags next to the camera keys.

    Without `video_modality_keys` the views come from the sorted `observation.images.*` keys, and a
    pad flag mistaken for a camera would change both the view count and the stitch content.
    """
    pack = DreamZeroPackInputsStep(DreamZeroConfig())
    obs = {
        "observation.images.exterior_1_left": torch.zeros(1, 3, 4, 4),
        "observation.images.exterior_1_left_is_pad": torch.zeros(1, dtype=torch.bool),
        "observation.images.wrist_left": torch.zeros(1, 3, 4, 4),
        "observation.images.wrist_left_is_pad": torch.zeros(1, dtype=torch.bool),
    }

    assert pack._ordered_image_keys(obs) == [
        "observation.images.exterior_1_left",
        "observation.images.wrist_left",
    ]


def test_train_time_standard_overrides_are_dropped_not_raised():
    """`lerobot-train` always passes normalizer overrides; DreamZero has no such step.

    Normalization and the relative<->absolute conversion live inside the pack/decode steps, driven
    by the checkpoint's statistics rather than dataset stats. Dropping these four keys is
    deliberate — but everything else must still raise, so a real override cannot be lost silently.
    """
    from lerobot.policies.dreamzero.processor_dreamzero import _drop_absent_standard_overrides

    overrides = {
        "normalizer_processor": {"stats": {}},
        "unnormalizer_processor": {"stats": {}},
        "relative_actions_processor": {"enabled": True},
        "absolute_actions_processor": {"enabled": True},
        "device_processor": {"device": "cpu"},
        "rename_observations_processor": {"rename_map": {}},
    }
    kept = _drop_absent_standard_overrides(overrides)
    assert set(kept) == {"device_processor", "rename_observations_processor"}
    assert _drop_absent_standard_overrides(None) is None


def test_unknown_override_key_still_raises():
    """A typo'd or genuinely missing step must fail rather than be quietly ignored."""
    from lerobot.policies.dreamzero.processor_dreamzero import (
        _apply_step_overrides,
        make_dreamzero_pre_post_processors,
    )

    cfg = DreamZeroConfig()
    pre, _ = make_dreamzero_pre_post_processors(cfg)
    with pytest.raises(KeyError, match="matches no step"):
        _apply_step_overrides(pre, {"no_such_processor": {"x": 1}})
