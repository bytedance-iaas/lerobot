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

"""Reading a released NVIDIA DreamZero checkpoint directly (synthetic inputs, no real checkpoint).

The synthetic checkpoint mirrors the layout the ``GEAR-Dreams/DreamZero-*`` repos actually ship: an
upstream ``config.json`` (``action_head_cfg.config`` with the DiT dims), an
``experiment_cfg/metadata.json`` holding the per-embodiment q01/q99 statistics, an
``experiment_cfg/conf.yaml`` whose ``ConcatTransform`` defines the run's state/action concat order,
and HF-sharded ``action_head.*`` weights.

Covers the deterministic derivation: variant inference from the DiT dims, geometry read from the
upstream config (rather than defaulted), layout read from the concat order (rather than from the
alphabetical modality listing), statistics slicing, key remapping and sharded weight loading.
"""

import json

import pytest
import torch

pytest.importorskip("diffusers")

from lerobot.policies.dreamzero import gear_checkpoint  # noqa: E402
from lerobot.policies.dreamzero.configuration_dreamzero import _DIT_PRESETS  # noqa: E402

# DROID statistics are already in the run's action space: state.joint_position spans the real joint
# range while action.joint_position is a DELTA quantile (relative_action_keys == [joint_position]).
_STATISTICS = {
    "state": {
        "joint_position": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
        "gripper_position": {"q01": [0.0], "q99": [1.0]},
        "cartesian_position": {"q01": [-2.0] * 6, "q99": [2.0] * 6},
    },
    "action": {
        "joint_position": {"q01": [-0.5] * 7, "q99": [0.5] * 7},
        "gripper_position": {"q01": [0.0], "q99": [1.0]},
        "cartesian_position": {"q01": [-2.0] * 6, "q99": [2.0] * 6},
    },
}

_CONF_YAML = """
train_dataset:
  dataset_kwargs:
    relative_action: true
    relative_action_keys:
    - joint_position
  all_transforms:
    oxe_droid:
      transforms:
      - _target_: groot.vla.data.transform.VideoCrop
        scale: 0.95
      - _target_: groot.vla.data.transform.VideoResize
        height: 176
        width: 320
      - _target_: groot.vla.data.transform.ConcatTransform
        video_concat_order:
        - video.exterior_image_1_left
        - video.exterior_image_2_left
        - video.wrist_image_left
        state_concat_order:
        - state.joint_position
        - state.gripper_position
        action_concat_order:
        - action.joint_position
        - action.gripper_position
"""


def _upstream_config(variant: str = "wan21_14b") -> dict:
    return {
        "architectures": ["VLA"],
        "action_head_cfg": {
            "config": {
                "num_frames": 33,
                "action_horizon": 24,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "hidden_size": 64,
                "input_embedding_dim": 1536,
                "num_inference_timesteps": 4,
                # 0 on the released checkpoints (IdentityBackbone, no backbone projector).
                "backbone_embedding_dim": 0,
                # "full" on the released checkpoints; defaulting to "lora" would wrap the DiT in
                # PEFT modules and mismatch every q/k/v/o/ffn key in the state dict.
                "train_architecture": "full",
                "diffusion_model_cfg": {
                    **_DIT_PRESETS[variant],
                    "max_chunk_size": 4,
                    "num_frame_per_block": 2,
                    "num_action_per_block": 24,
                    "num_state_per_block": 1,
                },
            }
        },
    }


def _make_fake_checkpoint(src, sharded: bool = True, include_droppable_keys: bool = True):
    from safetensors.torch import save_file

    (src / "experiment_cfg").mkdir(parents=True)
    (src / "config.json").write_text(json.dumps(_upstream_config()))
    (src / "experiment_cfg" / "conf.yaml").write_text(_CONF_YAML)
    (src / "experiment_cfg" / "metadata.json").write_text(
        json.dumps(
            {
                "oxe_droid": {
                    "statistics": _STATISTICS,
                    # Deliberately alphabetical and a superset of the run's layout — this must NOT
                    # be used as the concat order.
                    "modalities": {
                        "state": {
                            "cartesian_position": {},
                            "gripper_position": {},
                            "joint_position": {},
                        },
                        "action": {"gripper_position": {}, "joint_position": {}},
                        "video": {"exterior_image_1_left": {}},
                    },
                }
            }
        )
    )

    tensors = {
        "action_head.model.blocks.0.self_attn.q.weight": torch.zeros(4, 4),
        "action_head.vae.decoder.conv.weight": torch.zeros(3),
    }
    if include_droppable_keys:
        tensors["backbone.identity.weight"] = torch.zeros(2)
        tensors["action_head.model.blocks.0.self_attn.k.base_layer.weight"] = torch.ones(4, 4)

    if not sharded:
        save_file(tensors, str(src / "model.safetensors"))
        return

    items = list(tensors.items())
    shards = {
        "model-00001-of-00002.safetensors": dict(items[: len(items) // 2]),
        "model-00002-of-00002.safetensors": dict(items[len(items) // 2 :]),
    }
    weight_map = {}
    for shard, blob in shards.items():
        save_file(blob, str(src / shard))
        weight_map.update(dict.fromkeys(blob, shard))
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_is_gear_checkpoint_discriminates_on_content(tmp_path):
    """GEAR and LeRobot checkpoints share the config.json filename, so content decides."""
    gear, lerobot, empty = tmp_path / "gear", tmp_path / "lerobot", tmp_path / "empty"
    _make_fake_checkpoint(gear)
    lerobot.mkdir()
    (lerobot / "config.json").write_text(json.dumps({"type": "dreamzero", "model_variant": "wan21_14b"}))
    empty.mkdir()

    assert gear_checkpoint.is_gear_checkpoint(gear)
    assert not gear_checkpoint.is_gear_checkpoint(lerobot)
    assert not gear_checkpoint.is_gear_checkpoint(empty)


def test_build_config_reads_geometry_and_layout_from_the_checkpoint(tmp_path):
    src = tmp_path / "gear"
    _make_fake_checkpoint(src)

    cfg = gear_checkpoint.build_config(src, embodiment_tag="oxe_droid")

    assert cfg.type == "dreamzero"
    assert cfg.embodiment_tag == "oxe_droid"
    # Variant inferred from the checkpoint's own DiT dims, not defaulted.
    assert cfg.model_variant == "wan21_14b"
    # Geometry read from the upstream head config.
    assert cfg.training_mode == "full"
    assert cfg.backbone_embedding_dim == 0
    assert cfg.max_chunk_size == 4
    assert (cfg.max_state_dim, cfg.max_action_dim) == (64, 32)
    assert (cfg.num_frames, cfg.action_horizon) == (33, 24)
    # A training parameter, distinct from the inference schedule despite the similar name.
    assert cfg.num_inference_timesteps == 4
    # Layout read from the ConcatTransform, NOT from the alphabetical modality listing.
    assert cfg.state_modality_keys == ("joint_position", "gripper_position")
    assert cfg.action_modality_keys == ("joint_position", "gripper_position")
    assert cfg.video_modality_keys == (
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    )
    assert (cfg.per_view_height, cfg.per_view_width) == (176, 320)
    assert cfg.view_crop_scale == 0.95
    assert cfg.relative_action and cfg.relative_action_keys == ("joint_position",)
    # Features declared so validate_features() passes at policy construction.
    assert len(cfg.input_features) == 4  # 3 cameras + state
    assert cfg.input_features["observation.state"].shape == (8,)
    assert cfg.output_features["action"].shape == (8,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim", 1234),  # not a Wan backbone we know
        ("dim", 3072),  # the Wan2.2-TI2V-5B DiT: deliberately unsupported, must not silently pass
    ],
)
def test_build_config_rejects_a_non_14b_backbone(tmp_path, field, value):
    """Only wan21_14b is supported; anything else must fail by name, not by a later shape error."""
    src = tmp_path / "gear"
    _make_fake_checkpoint(src)
    blob = _upstream_config()
    blob["action_head_cfg"]["config"]["diffusion_model_cfg"][field] = value
    (src / "config.json").write_text(json.dumps(blob))

    with pytest.raises(ValueError, match="matches no known variant"):
        gear_checkpoint.build_config(src)


def test_build_statistics_slices_to_the_run_layout(tmp_path):
    src = tmp_path / "gear"
    _make_fake_checkpoint(src)
    metadata = gear_checkpoint.load_gear_metadata(src, "oxe_droid")

    stats = gear_checkpoint.build_statistics(
        metadata, ("joint_position", "gripper_position"), ("joint_position", "gripper_position")
    )

    assert stats["action"]["joint_position"]["q99"] == [0.5] * 7  # relative-action quantiles
    assert stats["state"]["joint_position"]["q01"] == [-1.0] * 7  # absolute joint range
    # cartesian_position exists in the statistics but not in the run's concat order; including it
    # would shift every downstream state/action slice.
    assert "cartesian_position" not in stats["state"]
    assert "cartesian_position" not in stats["action"]


def test_build_statistics_rejects_a_missing_key(tmp_path):
    """A silently-empty stat block would degrade the processor into a passthrough."""
    src = tmp_path / "gear"
    _make_fake_checkpoint(src)
    metadata = gear_checkpoint.load_gear_metadata(src, "oxe_droid")

    with pytest.raises(KeyError, match="No q01/q99 for state.elbow_angle"):
        gear_checkpoint.build_statistics(metadata, ("elbow_angle",), ("joint_position",))


_LAYOUT = ("joint_position", "gripper_position")


def test_statistics_survive_a_save_reload_roundtrip(tmp_path):
    """A fine-tuned checkpoint must carry the quantiles it was trained with.

    Without the save side, reloading finds no statistics, and the q99 normalization and the
    relative->absolute decode both silently become the identity.
    """
    src, dst = tmp_path / "gear", tmp_path / "finetuned"
    _make_fake_checkpoint(src)
    dst.mkdir()
    original = gear_checkpoint.load_checkpoint_statistics(src, "oxe_droid", _LAYOUT, _LAYOUT)

    gear_checkpoint.save_checkpoint_statistics(dst, "oxe_droid", original)
    reloaded = gear_checkpoint.load_checkpoint_statistics(dst, "oxe_droid", _LAYOUT, _LAYOUT)

    assert reloaded == original
    # Keyed by embodiment tag, so one checkpoint can hold several.
    assert set(json.loads((dst / "statistics.json").read_text())) == {"oxe_droid"}


def test_video_modality_keys_are_resolved_from_the_checkpoint(tmp_path):
    """A fine-tune builds its config from the CLI, so the stitch order has to come from here.

    Order decides which camera lands in which canvas quadrant. Falling back to sorted keys matches
    DROID only alphabetically, and the dataset's own feature order is outright wrong for it
    (`lerobot/droid_1.0.1` lists the wrist camera first; the checkpoint stitches it last).
    """
    from lerobot.policies.dreamzero.processor_dreamzero import _resolve_video_modality_keys

    src = tmp_path / "gear"
    _make_fake_checkpoint(src)
    cfg = gear_checkpoint.build_config(src, embodiment_tag="oxe_droid")
    cfg.video_modality_keys = ()

    _resolve_video_modality_keys(cfg, src)

    assert cfg.video_modality_keys == (
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    )


def test_an_explicit_video_modality_keys_is_never_overwritten(tmp_path):
    from lerobot.policies.dreamzero.processor_dreamzero import _resolve_video_modality_keys

    src = tmp_path / "gear"
    _make_fake_checkpoint(src)
    cfg = gear_checkpoint.build_config(src, embodiment_tag="oxe_droid")
    cfg.video_modality_keys = ("only_this_one",)

    _resolve_video_modality_keys(cfg, src)

    assert cfg.video_modality_keys == ("only_this_one",)


def test_adapter_manifest_round_trips(tmp_path):
    """A `save_lora_only` checkpoint is not self-contained; the manifest is what makes it loadable."""
    dst = tmp_path / "adapter"
    dst.mkdir()

    gear_checkpoint.write_adapter_manifest(
        dst, "/models/DreamZero-DROID", saved_keys=814, trainable_params=108_582_944
    )
    manifest = gear_checkpoint.read_adapter_manifest(dst)

    assert manifest["base_checkpoint"] == "/models/DreamZero-DROID"
    assert manifest["saved_keys"] == 814


def test_a_full_checkpoint_has_no_adapter_manifest(tmp_path):
    """The manifest is the discriminator, so a complete checkpoint must not look like an adapter."""
    full = tmp_path / "full"
    full.mkdir()
    (full / "config.json").write_text("{}")

    assert gear_checkpoint.read_adapter_manifest(full) is None


def test_load_checkpoint_statistics_returns_none_when_there_are_none(tmp_path):
    """Absent statistics must be reported as absent — the caller warns rather than fabricating."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert gear_checkpoint.load_checkpoint_statistics(empty, "oxe_droid", _LAYOUT, _LAYOUT) is None


def test_a_statistics_file_for_a_different_embodiment_raises(tmp_path):
    """A statistics.json keyed only by some other tag must not be returned as-is.

    Passing the whole ``{other_tag: {...}}`` dict through would give the processor an empty
    quantile layout, silently degrading the q99 normalization to the identity.
    """
    src, dst = tmp_path / "gear", tmp_path / "finetuned"
    _make_fake_checkpoint(src)
    dst.mkdir()
    stats = gear_checkpoint.load_checkpoint_statistics(src, "oxe_droid", _LAYOUT, _LAYOUT)
    gear_checkpoint.save_checkpoint_statistics(dst, "oxe_droid", stats)

    with pytest.raises(ValueError, match="no statistics for embodiment 'so100'"):
        gear_checkpoint.load_checkpoint_statistics(dst, "so100", _LAYOUT, _LAYOUT)


def test_load_gear_metadata_rejects_an_absent_embodiment(tmp_path):
    src = tmp_path / "gear"
    _make_fake_checkpoint(src)

    with pytest.raises(FileNotFoundError, match="No statistics for embodiment 'agibot'"):
        gear_checkpoint.load_gear_metadata(src, "agibot")


@pytest.mark.parametrize("sharded", [True, False])
def test_load_state_dict_remaps_keys(tmp_path, sharded):
    src = tmp_path / "gear"
    _make_fake_checkpoint(src, sharded=sharded)

    state_dict = gear_checkpoint.load_state_dict(src)

    assert "action_head.model.blocks.0.self_attn.q.weight" in state_dict
    assert "action_head.vae.decoder.conv.weight" in state_dict
    # backbone.* belongs to the un-ported IdentityBackbone.
    assert not any(k.startswith("backbone.") for k in state_dict)
    # LoRA base_layer is flattened the way upstream VLA.load_lora does.
    assert "action_head.model.blocks.0.self_attn.k.base_layer.weight" not in state_dict
    assert "action_head.model.blocks.0.self_attn.k.weight" in state_dict


def test_load_state_dict_reports_a_missing_shard(tmp_path):
    src = tmp_path / "gear"
    _make_fake_checkpoint(src, sharded=True)
    (src / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="model-00002-of-00002.safetensors"):
        gear_checkpoint.load_state_dict(src)
