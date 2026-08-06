#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Read a released NVIDIA DreamZero checkpoint (GEAR ``VLA`` layout) directly.

``GEAR-Dreams/DreamZero-DROID`` and friends already ship everything LeRobot needs, so there is no
conversion step — :meth:`DreamZeroPolicy.from_pretrained` points straight at the downloaded repo
and this module translates its files into a :class:`DreamZeroConfig` + processor statistics in
memory:

===============================  ====================================================
``config.json``                  ``action_head_cfg.config``: DiT dims (-> model variant),
                                 block/chunk sizes, state/action padding, train_architecture
``experiment_cfg/conf.yaml``     the run's ``ConcatTransform`` (state/action/video concat ORDER),
                                 per-view resize + crop, ``relative_action_keys``
``experiment_cfg/metadata.json`` per-embodiment q01/q99 statistics
``model-*.safetensors`` (+index) bf16 weights, keyed ``action_head.*`` (loaded as shards)
===============================  ====================================================

Two things are deliberately NOT guessed, because getting either wrong corrupts every rollout while
still running to completion:

* **Model geometry** comes from the checkpoint's own config, never from defaults — a mismatch
  silently produces wrong module shapes or spurious LoRA wrapping.
* **The state/action layout** comes from ``conf.yaml``'s concat order, never from
  ``metadata.json``'s ``modalities`` table — the latter is an alphabetical listing of every
  AVAILABLE key (for DROID it also lists ``cartesian_position``), not the subset and order the run
  actually concatenates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The stat names the DreamZero processor consumes (q99 normalization).
_STAT_NAMES = ("q01", "q99")

# Where a LeRobot-saved DreamZero checkpoint keeps its statistics (GR00T uses the same filename).
STATISTICS_FILENAME = "statistics.json"

# Marks a checkpoint that holds only the trainable delta and needs a base checkpoint to be usable.
ADAPTER_MANIFEST_FILENAME = "lora_adapter.json"


def is_gear_checkpoint(path: str | Path) -> bool:
    """True when ``path`` is a released GEAR checkpoint rather than a LeRobot one.

    A LeRobot checkpoint's ``config.json`` is a draccus dump carrying ``type: dreamzero``; GEAR's
    is a HuggingFace ``VLA`` config carrying ``action_head_cfg``. They share a filename, so the
    discriminator is the content.
    """
    config_file = Path(path) / "config.json"
    if not config_file.is_file():
        return False
    try:
        blob = json.loads(config_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return "action_head_cfg" in blob and "type" not in blob


# --------------------------------------------------------------------------------------------
# raw file readers
# --------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_conf_yaml(src: Path) -> dict:
    path = src / "experiment_cfg" / "conf.yaml"
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _dataset_kwargs(src: Path) -> dict:
    return ((_load_conf_yaml(src).get("train_dataset") or {}).get("dataset_kwargs")) or {}


def load_gear_metadata(src: Path, embodiment_tag: str) -> dict:
    """Return the GEAR metadata block for ``embodiment_tag`` (statistics + modalities).

    Released checkpoints ship ``experiment_cfg/metadata.json`` keyed by embodiment; older GEAR
    dumps use a flat ``meta/stats.json``. Raises when neither carries the requested embodiment,
    because an empty stats block would silently degrade the processor into a passthrough.
    """
    src = Path(src)
    blob = _read_json(src / "experiment_cfg" / "metadata.json")
    if embodiment_tag in blob and isinstance(blob[embodiment_tag], dict):
        return blob[embodiment_tag]

    legacy = _read_json(src / "meta" / "stats.json")
    if embodiment_tag in legacy and isinstance(legacy[embodiment_tag], dict):
        legacy = legacy[embodiment_tag]
    if {"state", "action"} & set(legacy):
        return {"statistics": legacy}

    raise FileNotFoundError(
        f"No statistics for embodiment {embodiment_tag!r} in {src}: expected "
        f"experiment_cfg/metadata.json (keys: {sorted(blob)}) or meta/stats.json."
    )


def read_concat_layout(src: Path, embodiment_tag: str) -> dict[str, Any]:
    """Recover the run's state/action/video concat order from ``experiment_cfg/conf.yaml``.

    Upstream keys are namespaced (``state.joint_position``, ``video.wrist_image_left``); LeRobot's
    config stores the bare modality names, so the prefix is stripped. Also picks up the per-view
    crop + resize the transform applies before the multi-view stitch.
    """
    conf = _load_conf_yaml(Path(src))
    transforms = (
        ((conf.get("train_dataset") or {}).get("all_transforms") or {}).get(embodiment_tag) or {}
    ).get("transforms") or []

    out: dict[str, Any] = {}
    for step in transforms:
        name = str(step.get("_target_", "")).rsplit(".", 1)[-1]
        if name == "ConcatTransform":
            for field, order_key in (
                ("state_modality_keys", "state_concat_order"),
                ("action_modality_keys", "action_concat_order"),
                ("video_modality_keys", "video_concat_order"),
            ):
                order = step.get(order_key)
                if order:
                    out[field] = tuple(k.split(".", 1)[-1] for k in order)
        elif name == "VideoResize":
            out["per_view_height"] = int(step["height"])
            out["per_view_width"] = int(step["width"])
        elif name == "VideoCrop":
            out["view_crop_scale"] = float(step["scale"])
    return out


# --------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------
def _extract_q(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    if all(name in entry for name in _STAT_NAMES):
        return {name: list(entry[name]) for name in _STAT_NAMES}
    return None


def build_statistics(
    metadata: dict,
    state_keys: tuple[str, ...],
    action_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble ``{state: {key: {q01, q99}}, action: {key: {q01, q99}}}`` for the processor.

    The GEAR statistics are already expressed in the run's own action space — for a run with
    ``relative_action_keys: [joint_position]``, ``action.joint_position``'s quantiles are DELTA
    quantiles (±0.4-0.7 rad, mean~0) while ``state.joint_position``'s span the real joint range —
    so both groups are read straight from one source with no relative/absolute switching.

    Every requested key must resolve: a missing one means the layout disagrees with the
    checkpoint, which would mis-slice every state and action vector at run time.
    """
    stats = metadata.get("statistics") or {}
    out: dict[str, Any] = {}
    for modality, keys in (("state", state_keys), ("action", action_keys)):
        group = stats.get(modality) or {}
        resolved: dict[str, Any] = {}
        for key in keys:
            q = _extract_q(group.get(key, {}))
            if q is None:
                raise KeyError(
                    f"No q01/q99 for {modality}.{key} in the checkpoint statistics "
                    f"(available {modality} keys: {sorted(group)})."
                )
            resolved[key] = q
        out[modality] = resolved
    return out


def load_checkpoint_statistics(
    src: str | Path,
    embodiment_tag: str,
    state_keys: tuple[str, ...],
    action_keys: tuple[str, ...],
    override_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Read the q01/q99 statistics from either checkpoint layout, or ``None`` if there are none.

    A released GEAR checkpoint keeps them in ``experiment_cfg/metadata.json`` under the embodiment
    tag; one written by :meth:`DreamZeroPolicy._save_pretrained` keeps them in ``statistics.json``
    as ``{embodiment_tag: {state, action}}``. Both the policy (which re-saves them on fine-tune)
    and the processor (which normalizes with them) go through here, so the two cannot drift.

    ``override_path`` wins over the checkpoint, and is how a new robot's statistics reach a run
    whose weights come from a checkpoint trained on something else. It is required to exist: a
    silently ignored override would train against the wrong normalization.
    """
    if override_path is not None:
        return _read_statistics_file(override_path, embodiment_tag)
    path = Path(src)
    if is_gear_checkpoint(path):
        return build_statistics(load_gear_metadata(path, embodiment_tag), state_keys, action_keys)
    stats_file = path / STATISTICS_FILENAME
    if not stats_file.exists():
        return None
    return _select_embodiment(_read_json(stats_file), embodiment_tag, stats_file)


def _read_statistics_file(path: str | Path, embodiment_tag: str) -> dict[str, Any]:
    """Read an explicitly-pointed-at statistics file (or the directory containing one)."""
    path = Path(path)
    if path.is_dir():
        path = path / STATISTICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"statistics_path points at {path}, which does not exist.")
    stats = _select_embodiment(_read_json(path), embodiment_tag, path)
    for modality in ("state", "action"):
        if not isinstance(stats.get(modality), dict):
            raise ValueError(
                f"{path} has no {modality!r} statistics for embodiment {embodiment_tag!r} "
                f"(top-level keys: {sorted(stats)})."
            )
    return stats


def _select_embodiment(all_stats: dict, embodiment_tag: str, source: Path) -> dict[str, Any]:
    # Files written by this policy are keyed by embodiment tag; tolerate a bare {state, action}
    # dict too, since that is what a hand-written statistics file tends to be. A file that is
    # neither — typically one keyed by a *different* embodiment tag — must raise: returning it
    # as-is would give the processor an empty layout and silently degrade the q99 normalization
    # to the identity.
    block = all_stats.get(embodiment_tag)
    if isinstance(block, dict):
        return block
    if {"state", "action"} <= set(all_stats):
        return all_stats
    raise ValueError(
        f"{source} has no statistics for embodiment {embodiment_tag!r} and is not a bare "
        f"{{state, action}} file (top-level keys: {sorted(all_stats)})."
    )


def read_adapter_manifest(src: str | Path) -> dict[str, Any] | None:
    """Return the adapter manifest if `src` holds only a trainable delta, else ``None``."""
    manifest = Path(src) / ADAPTER_MANIFEST_FILENAME
    return _read_json(manifest) if manifest.exists() else None


def write_adapter_manifest(
    dst: str | Path, base_checkpoint: str, saved_keys: int, trainable_params: int
) -> Path:
    """Record what an adapter-only checkpoint is missing and where to find it.

    A LoRA run saves ~0.2 GB of trainable delta instead of ~46 GB of full weights, which means the
    checkpoint is *not* self-contained. Writing the base path down is what keeps that tractable:
    without it, loading would either need the user to remember which checkpoint the run started
    from, or silently succeed against the wrong one.
    """
    out = Path(dst) / ADAPTER_MANIFEST_FILENAME
    out.write_text(
        json.dumps(
            {
                "base_checkpoint": str(base_checkpoint),
                "saved_keys": saved_keys,
                "trainable_params": trainable_params,
                "note": (
                    "Only the parameters that training updated are stored here. Load this "
                    "directory with DreamZeroPolicy.from_pretrained; it reads base_checkpoint "
                    "for the frozen weights."
                ),
            },
            indent=2,
        )
    )
    return out


def save_checkpoint_statistics(dst: str | Path, embodiment_tag: str, stats: dict[str, Any]) -> Path:
    """Write ``statistics.json`` so a fine-tuned checkpoint keeps the quantiles it was trained with.

    Without this, reloading a fine-tuned checkpoint finds no statistics and the q99 normalization
    and the relative->absolute action decode both degrade to the identity, which produces plausible
    but meaningless actions rather than an error.
    """
    out = Path(dst) / STATISTICS_FILENAME
    out.write_text(json.dumps({embodiment_tag: stats}, indent=2))
    return out


# --------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------
def infer_model_variant(dit_cfg: dict) -> str:
    """Pick the LeRobot variant whose DiT preset matches the checkpoint's own DiT dims."""
    from .configuration_dreamzero import _DIT_PRESETS

    discriminators = ("dim", "num_layers", "num_heads", "in_dim", "out_dim", "ffn_dim")
    for variant, preset in _DIT_PRESETS.items():
        if all(dit_cfg.get(field) == preset[field] for field in discriminators):
            return variant
    got = {field: dit_cfg.get(field) for field in discriminators}
    raise ValueError(
        f"Checkpoint DiT config matches no known variant: {got}. Known: "
        + ", ".join(f"{v}={ {f: p[f] for f in discriminators} }" for v, p in _DIT_PRESETS.items())
    )


def build_config(
    src: str | Path,
    embodiment_tag: str = "oxe_droid",
    metadata: dict | None = None,
    model_variant: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    """Build a :class:`DreamZeroConfig` from a GEAR checkpoint's own files."""
    from .configuration_dreamzero import DreamZeroConfig

    src = Path(src)
    metadata = load_gear_metadata(src, embodiment_tag) if metadata is None else metadata

    upstream = _read_json(src / "config.json")
    head = ((upstream.get("action_head_cfg") or {}).get("config")) or {}
    if not head:
        raise FileNotFoundError(f"No action_head_cfg.config in {src / 'config.json'}")
    dit = head.get("diffusion_model_cfg") or {}

    kwargs: dict[str, Any] = {
        "model_variant": model_variant or infer_model_variant(dit),
        "embodiment_tag": embodiment_tag,
        "skip_component_loading": True,
        # --- geometry, straight from the upstream head config -----------------------------
        "num_frames": head["num_frames"],
        "action_horizon": head["action_horizon"],
        "num_frame_per_block": dit["num_frame_per_block"],
        "num_action_per_block": dit["num_action_per_block"],
        "num_state_per_block": dit["num_state_per_block"],
        "max_chunk_size": dit["max_chunk_size"],
        "max_state_dim": head["max_state_dim"],
        "max_action_dim": head["max_action_dim"],
        "hidden_size": head["hidden_size"],
        "input_embedding_dim": head["input_embedding_dim"],
        "num_inference_timesteps": head["num_inference_timesteps"],
        # 0 for the released checkpoints (IdentityBackbone, no backbone projector).
        "backbone_embedding_dim": head["backbone_embedding_dim"],
        # "full" for the released checkpoints — defaulting to "lora" would wrap the DiT in PEFT
        # modules and mismatch every q/k/v/o/ffn key in the state dict.
        # The released checkpoints record how they were TRAINED ("full"); fine-tuning mode is a
        # separate choice, so only "lora" carries over — anything else starts from "full".
        "training_mode": "lora" if head.get("train_architecture") == "lora" else "full",
    }

    layout = read_concat_layout(src, embodiment_tag)
    if not layout.get("state_modality_keys"):
        raise ValueError(
            f"{src / 'experiment_cfg' / 'conf.yaml'} has no ConcatTransform for {embodiment_tag!r}; "
            "the state/action concat order cannot be recovered from metadata.json (it lists all "
            "available keys alphabetically, not the run's layout). Pass it via `overrides`."
        )
    kwargs.update(layout)

    dataset_kwargs = _dataset_kwargs(src)
    if "relative_action_keys" in dataset_kwargs:
        kwargs["relative_action"] = bool(dataset_kwargs.get("relative_action", True))
        kwargs["relative_action_keys"] = tuple(dataset_kwargs["relative_action_keys"])

    kwargs.update(overrides or {})
    config = DreamZeroConfig(**kwargs)
    _populate_features(config, metadata)
    return config


def _populate_features(config, metadata: dict) -> None:
    """Declare the policy's input/output features so ``validate_features`` passes.

    DreamZero's processor does its own crop/resize/stitch, so the per-camera shapes recorded here
    are the post-resize per-view size rather than any particular dataset's raw resolution — they
    describe the policy's interface (N cameras + a state vector), not a dataset constraint.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    stats = metadata.get("statistics") or {}

    def width(modality: str, keys: tuple[str, ...]) -> int:
        group = stats.get(modality) or {}
        return sum(len(group[key]["q01"]) for key in keys if key in group)

    for view in config.video_modality_keys:
        config.input_features[f"{OBS_IMAGES}.{view}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, config.per_view_height, config.per_view_width),
        )
    config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(width("state", config.state_modality_keys),)
    )
    config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION, shape=(width("action", config.action_modality_keys),)
    )


# --------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------
def remap_key(key: str) -> str | None:
    """Map one upstream weight key to its LeRobot name, or None to drop it.

    ``backbone.*`` belongs to the un-ported IdentityBackbone. LoRA ``.base_layer.`` is flattened
    the way upstream ``VLA.load_lora`` does, so a merged PEFT checkpoint loads too. The released
    checkpoints are already pure ``action_head.*``, making this a no-op for them —
    :class:`DreamZeroPolicy` names its submodule ``action_head`` precisely so the keys map 1:1.
    """
    if key.startswith("backbone."):
        return None
    if not key.startswith("action_head."):
        # Unknown top-level prefix — keep it but warn; the policy load is strict=False-tolerant.
        logger.warning("Keeping unexpected checkpoint key with non action_head prefix: %s", key)
    return key.replace(".base_layer.", ".")


def load_state_dict(src: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load the weights of a checkpoint, single-file or HF-sharded, and remap the keys.

    DreamZero is ~23B parameters (46 GB in bf16), so the released checkpoints keep the weights as
    shards described by ``model.safetensors.index.json``. Shards are loaded one at a time rather
    than merged on disk.
    """
    from safetensors.torch import load_file

    src = Path(src)
    index = src / "model.safetensors.index.json"
    single = src / "model.safetensors"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif single.exists():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No model.safetensors[.index.json] in {src}")

    state_dict: dict[str, Any] = {}
    dropped = 0
    for shard in shards:
        shard_path = src / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"{index} references {shard}, which is missing from {src}.")
        for key, value in load_file(str(shard_path), device=device).items():
            new_key = remap_key(key)
            if new_key is None:
                dropped += 1
                continue
            # `DreamZeroPolicy` promotes 0-d parameters to shape [1] at construction because FSDP
            # cannot shard a scalar, so the checkpoint's scalars have to be reshaped to match
            # (the released checkpoint stores `image_encoder.model.log_scale` as shape []).
            state_dict[new_key] = value.reshape(1) if value.dim() == 0 else value
    logger.info(
        "Loaded %d tensors from %d shard(s) in %s (dropped %d backbone.* keys)",
        len(state_dict),
        len(shards),
        src,
        dropped,
    )
    return state_dict
