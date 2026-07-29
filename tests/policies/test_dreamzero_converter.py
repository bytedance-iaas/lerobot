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

"""Structure test for the DreamZero checkpoint converter (synthetic inputs, no real checkpoint).

Validates the deterministic conversion logic — key remapping (drop ``backbone.*``, keep
``action_head.*``, clean LoRA ``.base_layer.``), the GEAR stats assembly (joint uses RELATIVE
stats, gripper uses ABSOLUTE stats), and that the emitted ``config.json`` round-trips as a
``dreamzero`` config. The real checkpoint's exact ``meta/`` layout is validated separately.
"""

import json

import pytest
import torch

pytest.importorskip("diffusers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.dreamzero.scripts.convert_dreamzero_checkpoint import convert  # noqa: E402


def _make_fake_upstream(src):
    from safetensors.torch import save_file

    (src / "meta").mkdir(parents=True)
    save_file(
        {
            "backbone.identity.weight": torch.zeros(2),
            "action_head.model.blocks.0.self_attn.q.weight": torch.zeros(4, 4),
            "action_head.model.blocks.0.self_attn.q.base_layer.weight": torch.ones(4, 4),
            "action_head.vae.decoder.conv.weight": torch.zeros(3),
        },
        str(src / "model.safetensors"),
    )
    (src / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "state": {
                    "joint_position": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
                    "gripper_position": {"q01": [0.0], "q99": [1.0]},
                },
                "action": {"gripper_position": {"q01": [0.0], "q99": [1.0]}},
            }
        )
    )
    (src / "meta" / "relative_stats_dreamzero.json").write_text(
        json.dumps({"action": {"joint_position": {"q01": [-0.5] * 7, "q99": [0.5] * 7}}})
    )


def test_convert_structure(tmp_path):
    src, dst = tmp_path / "upstream", tmp_path / "lerobot"
    _make_fake_upstream(src)
    convert(src, dst, embodiment_tag="oxe_droid", model_variant="wan22_5b")

    assert {"config.json", "model.safetensors", "statistics.json"} <= {p.name for p in dst.iterdir()}

    from safetensors.torch import load_file

    keys = set(load_file(str(dst / "model.safetensors")))
    assert "action_head.model.blocks.0.self_attn.q.weight" in keys  # kept
    assert "action_head.model.blocks.0.self_attn.q.base_layer.weight" not in keys  # LoRA cleaned
    assert not any(k.startswith("backbone.") for k in keys)  # backbone dropped
    assert "action_head.vae.decoder.conv.weight" in keys

    emb = json.loads((dst / "statistics.json").read_text())["oxe_droid"]
    assert emb["action"]["joint_position"]["q99"] == [0.5] * 7  # RELATIVE stats
    assert emb["action"]["gripper_position"]["q99"] == [1.0]  # ABSOLUTE stats
    assert emb["state"]["joint_position"]["q01"] == [-1.0] * 7

    cfg = PreTrainedConfig.from_pretrained(str(dst))
    assert cfg.type == "dreamzero"
    assert cfg.model_variant == "wan22_5b"
    assert cfg.embodiment_tag == "oxe_droid"
    assert cfg.max_state_dim == 64
    assert cfg.max_action_dim == 32
