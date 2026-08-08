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
the upstream DreamZero semantics (dreamzero_port_plan.md M2). The umt5 tokenizer path and the
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
    _q99_forward,
    _q99_inverse,
    _stitch_oxe_droid,
    _to_bthwc_uint8,
)
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
