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

"""Convert an NVIDIA DreamZero checkpoint (GEAR ``VLA`` layout) to a LeRobot checkpoint.

Upstream ``GEAR-Dreams/DreamZero-DROID`` (and friends) save a ``VLA(PreTrainedModel)`` whose
``state_dict`` keys are ``backbone.*`` (the IdentityBackbone — NOT ported) and ``action_head.*``
(the ``WANPolicyHead``). LeRobot's :class:`DreamZeroPolicy` names its submodule ``action_head``
precisely so those keys map 1:1, so conversion is mostly key filtering + writing the LeRobot
sidecar files that the policy/processor read at load time:

    <dst>/config.json          # draccus-serialized DreamZeroConfig (type: dreamzero)
    <dst>/model.safetensors    # action_head.* weights (backbone.* dropped, LoRA base_layer cleaned)
    <dst>/statistics.json      # {embodiment_tag: {state: {...}, action: {...}}} q01/q99 for the processor

Statistics come from the GEAR ``meta/`` dir: ``state.*`` and ``action.gripper_position`` use the
ABSOLUTE stats (``meta/stats.json``); ``action.joint_position`` (a relative key for DROID) uses the
RELATIVE stats (``meta/relative_stats_dreamzero.json``) — mirroring how the upstream dataloader
normalizes relative actions.

The exact per-key dims / stat layout live in the checkpoint's own ``meta/`` files (dataset-side),
so run this against the released checkpoint and verify the emitted ``config.json`` dims match.

Usage:
    python -m lerobot.policies.dreamzero.scripts.convert_dreamzero_checkpoint \
        --src /path/to/DreamZero-DROID --dst /path/to/lerobot-dreamzero-droid \
        --embodiment-tag oxe_droid --model-variant wan22_5b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The stat names the LeRobot DreamZero processor consumes (q99 normalization).
_STAT_NAMES = ("q01", "q99")


# ----------------------------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------------------------
def load_upstream_state_dict(src: Path) -> dict[str, Any]:
    """Load ``model.safetensors`` (single or sharded via ``model.safetensors.index.json``)."""
    from safetensors.torch import load_file

    index = src / "model.safetensors.index.json"
    single = src / "model.safetensors"
    state_dict: dict[str, Any] = {}
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(str(src / shard)))
    elif single.exists():
        state_dict.update(load_file(str(single)))
    else:
        raise FileNotFoundError(f"No model.safetensors[.index.json] in {src}")
    return state_dict


def remap_action_head_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Keep ``action_head.*`` (dropping ``backbone.*``); clean LoRA ``.base_layer.`` -> ``.``.

    Mirrors upstream ``VLA.load_lora`` key hygiene so a merged/PEFT checkpoint loads too.
    """
    out: dict[str, Any] = {}
    dropped_backbone = 0
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            dropped_backbone += 1
            continue
        if not key.startswith("action_head."):
            # Unknown top-level prefix — keep it but warn; the policy load is strict=False-tolerant.
            logger.warning("Keeping unexpected checkpoint key with non action_head prefix: %s", key)
        out[key.replace(".base_layer.", ".")] = value
    logger.info("Remapped %d keys (dropped %d backbone.* keys)", len(out), dropped_backbone)
    return out


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _select_embodiment(blob: dict, embodiment_tag: str) -> dict:
    """GEAR meta files may be flat ({state:...}) or embodiment-keyed ({oxe_droid:{state:...}})."""
    if embodiment_tag in blob and isinstance(blob[embodiment_tag], dict):
        return blob[embodiment_tag]
    return blob


def _extract_q(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    if all(name in entry for name in _STAT_NAMES):
        return {name: list(entry[name]) for name in _STAT_NAMES}
    return None


def build_statistics(
    meta_dir: Path,
    embodiment_tag: str,
    state_keys: tuple[str, ...],
    action_keys: tuple[str, ...],
    relative_action_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble ``{embodiment_tag: {state:{key:{q01,q99}}, action:{key:{q01,q99}}}}``.

    - state.* and absolute action keys -> ``meta/stats.json``.
    - relative action keys -> ``meta/relative_stats_dreamzero.json``.
    """
    abs_stats = _select_embodiment(_read_json(meta_dir / "stats.json"), embodiment_tag)
    rel_stats = _select_embodiment(
        _read_json(meta_dir / "relative_stats_dreamzero.json"), embodiment_tag
    )

    state_out: dict[str, Any] = {}
    for key in state_keys:
        q = _extract_q((abs_stats.get("state") or {}).get(key, {}))
        if q is None:
            logger.warning("No absolute q01/q99 for state.%s in %s", key, meta_dir / "stats.json")
            continue
        state_out[key] = q

    action_out: dict[str, Any] = {}
    for key in action_keys:
        if key in relative_action_keys:
            q = _extract_q((rel_stats.get("action") or {}).get(key, {}))
            src = "relative_stats_dreamzero.json"
        else:
            q = _extract_q((abs_stats.get("action") or {}).get(key, {}))
            src = "stats.json"
        if q is None:
            logger.warning("No q01/q99 for action.%s in %s", key, src)
            continue
        action_out[key] = q

    return {embodiment_tag: {"state": state_out, "action": action_out}}


# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------
def build_config(embodiment_tag: str, model_variant: str, overrides: dict[str, Any] | None = None):
    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

    kwargs: dict[str, Any] = {
        "model_variant": model_variant,
        "embodiment_tag": embodiment_tag,
        # DROID (droid_relative_wan22.sh) values; harmless for other DROID variants.
        "max_chunk_size": 4,
        "skip_component_loading": True,
    }
    kwargs.update(overrides or {})
    return DreamZeroConfig(**kwargs)


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def convert(
    src: Path,
    dst: Path,
    embodiment_tag: str = "oxe_droid",
    model_variant: str = "wan22_5b",
    config_overrides: dict[str, Any] | None = None,
) -> None:
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    config = build_config(embodiment_tag, model_variant, config_overrides)
    state_dict = remap_action_head_keys(load_upstream_state_dict(src))

    meta_dir = src / "meta"
    stats = build_statistics(
        meta_dir,
        embodiment_tag,
        config.state_modality_keys,
        config.action_modality_keys,
        config.relative_action_keys,
    )

    # config.json (draccus, type: dreamzero) — reuse the policy config's own serializer.
    config._save_pretrained(dst)
    # model.safetensors (action_head.* only).
    save_file(state_dict, str(dst / "model.safetensors"))
    # statistics.json for the processor.
    (dst / "statistics.json").write_text(json.dumps(stats, indent=2))

    logger.info(
        "Wrote LeRobot DreamZero checkpoint to %s (%d weight tensors, stats for %s)",
        dst,
        len(state_dict),
        embodiment_tag,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Upstream DreamZero checkpoint dir (VLA layout + meta/)")
    p.add_argument("--dst", required=True, help="Output LeRobot checkpoint dir")
    p.add_argument("--embodiment-tag", default="oxe_droid")
    p.add_argument("--model-variant", default="wan22_5b", choices=["wan21_14b", "wan22_5b"])
    args = p.parse_args()
    convert(Path(args.src), Path(args.dst), args.embodiment_tag, args.model_variant)


if __name__ == "__main__":
    main()
