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

"""Compute the q01/q99 statistics DreamZero needs for a dataset it was not trained on.

DreamZero does not normalize with LeRobot's mean/std: it maps each modality through
``2*(x-q01)/(q99-q01)-1``, and for ``relative_action_keys`` the quantiles describe the
**per-block displacement** (action minus the raw state at that block's first row), not the raw
action. Feeding it absolute-action quantiles is the failure this script exists to prevent: the
model still runs, but every predicted displacement is scaled by roughly the ratio of the two
ranges — for DROID's joints, about 6x.

Usage::

    python -m lerobot.policies.dreamzero.scripts.compute_statistics \\
        --repo_id=<your/dataset> --root=<path> --output=<checkpoint dir>

The result is a ``statistics.json`` in the layout `DreamZeroPolicy.from_pretrained` reads, so
writing it into a checkpoint directory is enough to fine-tune on the new data.

**Scope.** This regenerates the *statistics* only. A robot whose camera layout or language
template differs from DROID's is handled by the processor's generic 2x2 canvas and generated
prompt (any registered non-``oxe_droid`` embodiment tag), driven by ``video_modality_keys`` /
``view_descriptions`` — see the README's "Training a new robot". What this script contributes to
that recipe is the piece the checkpoint cannot supply: q01/q99 quantiles measured on the new
robot's own state and motion scale.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch

from lerobot.policies.dreamzero import gear_checkpoint
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

logger = logging.getLogger(__name__)


@dataclass
class ComputeStatisticsConfig:
    # Dataset to read.
    repo_id: str
    output: str
    root: str | None = None
    episodes: list[int] | None = None

    # Which columns hold the concatenated state/action vectors.
    state_column: str = "observation.state"
    action_column: str = "action"

    # Concat layout, in order, as ``name:dim`` pairs. Must match the order the columns are
    # concatenated in — a wrong order silently mis-slices every vector.
    state_layout: str = "joint_position:7,gripper_position:1"
    action_layout: str = "joint_position:7,gripper_position:1"

    embodiment_tag: str = "oxe_droid"
    # Keys whose quantiles describe the per-block displacement rather than the raw value.
    relative_action_keys: list[str] = field(default_factory=lambda: ["joint_position"])
    # Frame rate of this dataset, when it is not DreamZero's 15 fps. Must match the value training
    # will run with: it sets how many raw frames one macro block spans, and therefore how large
    # the displacements being measured are. Leaving it unset for a 30 fps dataset would describe
    # half-length blocks — quantiles roughly half the size the model is then trained against.
    source_fps: int | None = None

    # Quantile levels. Upstream uses the 1st/99th percentile.
    q_low: float = 0.01
    q_high: float = 0.99


def _parse_layout(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, dim = part.partition(":")
        if not dim.isdigit():
            raise ValueError(f"Layout entry {part!r} is not 'name:dim'.")
        out.append((name.strip(), int(dim)))
    return out


def _placeholder_stats(layout: list[tuple[str, int]]) -> dict:
    """Stats whose only job is to give `_QuantileLayout` the right per-key slices.

    The relative encode needs to know which dims belong to which key; it does not need their
    values. Using the real quantiles here is impossible — computing them is the point.
    """
    return {name: {"q01": [-1.0] * dim, "q99": [1.0] * dim} for name, dim in layout}


def _episode_bounds(dataset) -> list[tuple[int, int]]:
    """Segment boundaries as row offsets into the loaded table.

    Derived from the table's own ``episode_index`` column rather than from
    ``meta.episodes[dataset_from_index]``: those are offsets into the *whole* dataset, so with
    ``--episodes`` they run past the end of the loaded subset.
    """
    if "episode_index" not in dataset.hf_dataset.column_names:
        raise KeyError("The dataset table has no 'episode_index' column to split episodes on.")
    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    changes = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(episode_index)]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _relative_actions(pack_step, actions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Per-block displacements, computed by the processor's own encoder.

    Routed through `DreamZeroPackInputsStep._encode_relative_actions` rather than reimplemented
    so the statistics cannot describe a different convention from the one training applies.
    Shapes: `actions` (N, horizon, D) and `anchors` (N, D) — one block per row, which is what the
    encoder sees when `max_chunk_size` blocks are handed to it one at a time.
    """
    # `_encode_relative_actions` writes the displacements back in place, and `from_numpy` shares
    # memory with the caller's array — so clone, or computing the relative stats would silently
    # overwrite the absolute actions the non-relative keys still need.
    act = torch.from_numpy(actions).to(torch.float32).clone()
    anchor = torch.from_numpy(anchors).to(torch.float32).unsqueeze(1)  # (N, 1, D): one block
    pack_step._block_anchor_state = pack_step._split_by_key(anchor)
    return pack_step._encode_relative_actions(act, training=True).numpy()


def _quantiles(values: np.ndarray, layout: list[tuple[str, int]], lo: float, hi: float) -> dict:
    out, start = {}, 0
    for name, dim in layout:
        block = values[:, start : start + dim]
        out[name] = {
            "q01": np.quantile(block, lo, axis=0).astype(float).tolist(),
            "q99": np.quantile(block, hi, axis=0).astype(float).tolist(),
        }
        start += dim
    return out


@draccus.wrap()
def main(cfg: ComputeStatisticsConfig):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.dreamzero.processor_dreamzero import DreamZeroPackInputsStep

    state_layout = _parse_layout(cfg.state_layout)
    action_layout = _parse_layout(cfg.action_layout)
    state_dim = sum(d for _, d in state_layout)
    action_dim = sum(d for _, d in action_layout)

    policy_config = DreamZeroConfig(
        embodiment_tag=cfg.embodiment_tag,
        relative_action_keys=tuple(cfg.relative_action_keys),
        state_modality_keys=tuple(n for n, _ in state_layout),
        action_modality_keys=tuple(n for n, _ in action_layout),
        source_fps=cfg.source_fps,
    )
    horizon = policy_config.action_horizon
    # One macro block covers `action_horizon` model steps, which is `action_horizon * multiplier`
    # raw frames sampled every `multiplier`-th — the same rows the dataloader will hand training.
    # Measuring over `action_horizon` raw frames instead would describe a shorter block.
    step = policy_config.frame_rate_multiplier
    block_span = horizon * step
    pack_step = DreamZeroPackInputsStep(
        config=policy_config,
        stats={"state": _placeholder_stats(state_layout), "action": _placeholder_stats(action_layout)},
    )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=cfg.episodes)
    # Read the columns straight off the table: the quantiles need no pixels, and decoding video
    # for every frame would turn minutes into hours.
    states = np.asarray(dataset.hf_dataset[cfg.state_column], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset[cfg.action_column], dtype=np.float32)
    for name, arr, expected in (("state", states, state_dim), ("action", actions, action_dim)):
        if arr.shape[1] != expected:
            raise ValueError(
                f"{name} column has {arr.shape[1]} dims but the layout declares {expected}. "
                f"Fix --{name}_layout so it matches the dataset's concat order."
            )

    # Every frame that can start a macro block contributes one block of displacements. Block
    # anchors sit at `t + c*block_span` for window start `t`, so as `t` and `c` range over their
    # valid values the anchors cover exactly the frames with a full block after them.
    rel_blocks, rel_anchors = [], []
    for start, end in _episode_bounds(dataset):
        last_anchor = end - block_span
        if last_anchor <= start:
            logger.warning(
                "Episode [%d, %d) is shorter than one block (%d frames); skipped.", start, end, block_span
            )
            continue
        idx = np.arange(start, last_anchor)
        rel_blocks.append(np.stack([actions[a : a + block_span : step] for a in idx]))
        rel_anchors.append(states[idx])
    if not rel_blocks:
        raise ValueError("No episode was long enough to contribute a single block of actions.")

    blocks = np.concatenate(rel_blocks)
    anchors = np.concatenate(rel_anchors)
    logger.info(
        "Using %d anchors over %d episodes (%d frames); block = %d model steps over %d raw frames.",
        len(anchors),
        dataset.num_episodes,
        len(states),
        horizon,
        block_span,
    )

    relative = _relative_actions(pack_step, blocks, anchors).reshape(-1, action_dim)
    stats = {
        "state": _quantiles(states, state_layout, cfg.q_low, cfg.q_high),
        "action": _quantiles(relative, action_layout, cfg.q_low, cfg.q_high),
    }
    # Keys that are not relative keep their absolute quantiles: subtracting an anchor from a
    # gripper command would describe a distribution the policy never predicts.
    absolute = _quantiles(actions, action_layout, cfg.q_low, cfg.q_high)
    for name, _ in action_layout:
        if name not in cfg.relative_action_keys:
            stats["action"][name] = absolute[name]

    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = gear_checkpoint.save_checkpoint_statistics(out_dir, cfg.embodiment_tag, stats)
    logger.info("Wrote %s", written)
    for modality in ("state", "action"):
        for name, _ in state_layout if modality == "state" else action_layout:
            kind = "relative" if modality == "action" and name in cfg.relative_action_keys else "absolute"
            entry = stats[modality][name]
            logger.info(
                "  %s.%-18s (%s) q01[0]=%+.4f q99[0]=%+.4f",
                modality,
                name,
                kind,
                entry["q01"][0],
                entry["q99"][0],
            )


if __name__ == "__main__":
    main()
