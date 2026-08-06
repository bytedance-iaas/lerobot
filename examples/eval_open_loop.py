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

"""Open-loop evaluation of a policy against recorded episodes.

`lerobot-eval` rolls a policy out in a simulator and reports task success, which needs a gym
environment for the embodiment. Most real-robot data has none. This script instead replays
recorded episodes, asks the policy what it would have done, and scores the prediction against
what was actually recorded — no simulator involved.

Because a policy predicts a *chunk* from one observation, scoring advances a chunk at a time:
predict from frame `t`, score against frames `t .. t+stride-1`, then jump to `t+stride`. Scoring
frame-by-frame instead would be wrong for policies whose actions are relative to the observation
they were predicted from — such a chunk must be decoded against its own anchor, not against
whatever frame happens to be current.

Usage:
    uv run python examples/eval_open_loop.py \
        --policy.path=/data/models/DreamZero-DROID \
        --dataset.repo_id=lerobot/droid_1.0.1 \
        --dataset.root=/data/datasets/droid_1.0.1 \
        --episodes='[1,2,3]' \
        --max_frames_per_episode=96 \
        --device=cuda \
        --rename_map='{"observation.images.exterior_1_left": "observation.images.exterior_image_1_left",
                       "observation.images.exterior_2_left": "observation.images.exterior_image_2_left",
                       "observation.images.wrist_left": "observation.images.wrist_image_left"}'

`--rename_map` is how a dataset whose cameras are named differently from the checkpoint's training
data is adapted; the policy config keeps the names the model was actually trained on.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import torch

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _apply_cli_overrides(config: PreTrainedConfig, cli_overrides: list[str]) -> PreTrainedConfig:
    """Re-parse `config` with `--policy.foo=bar` applied, as a LeRobot checkpoint would get."""
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        with draccus.config_type("json"):
            with open(tmp, "w") as f:
                draccus.dump(config, f)
            blob = json.loads(tmp.read_text())
            blob.pop("type", None)  # the concrete class is already known
            tmp.write_text(json.dumps(blob))
            return draccus.parse(type(config), str(tmp), args=cli_overrides)
    finally:
        tmp.unlink(missing_ok=True)


def load_policy_config(policy_path: str, cli_overrides: list[str]) -> PreTrainedConfig:
    """Read a policy config, accepting both LeRobot and upstream checkpoint layouts.

    A LeRobot checkpoint has a draccus `config.json` with a `type` key. Some policies can also
    read the layout their upstream project publishes (NVIDIA's DreamZero releases ship a
    HuggingFace `VLA` config, for instance); those expose a `from_foreign_checkpoint` classmethod.
    Try the LeRobot path first, then offer the directory to each registered policy.
    """
    try:
        return PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    except Exception as lerobot_error:
        for name, config_cls in PreTrainedConfig.get_known_choices().items():
            adopt = getattr(config_cls, "from_foreign_checkpoint", None)
            if adopt is None:
                continue
            config = adopt(policy_path)
            if config is None:
                continue
            logging.info("Loaded %s as a %r checkpoint (non-LeRobot layout).", policy_path, name)
            if cli_overrides:
                config = _apply_cli_overrides(config, cli_overrides)
            return config
        raise lerobot_error


@dataclass
class OpenLoopEvalConfig:
    # Recorded episodes to replay. `--policy.path=...` supplies the policy.
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="lerobot/pusht"))
    policy: PreTrainedConfig | None = None
    # Episodes to score. Empty means every episode in the dataset.
    episodes: list[int] = field(default_factory=list)
    # Cap the frames scored per episode. Note that windows start where a demonstration is often
    # still nearly static, so a short cap is not representative of the whole episode.
    max_frames_per_episode: int | None = None
    # Actions consumed per prediction. Defaults to the policy's own `n_action_steps`.
    stride: int | None = None
    # Map dataset observation keys onto the ones the policy expects, when a held-out dataset names
    # the same cameras differently from the dataset the checkpoint was trained on.
    rename_map: dict[str, str] = field(default_factory=dict)
    device: str | None = None
    output_dir: Path | None = None
    # Seeds torch/numpy/python RNGs before evaluating, so policies with stochastic sampling
    # reproduce their numbers (DreamZero itself is deterministic — its head fixes its own seed).
    seed: int | None = 1000

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("A policy is required (--policy.path=<checkpoint dir or hub id>)")
        cli_overrides = parser.get_cli_overrides("policy") or []
        self.policy = load_policy_config(policy_path, cli_overrides)
        self.policy.pretrained_path = Path(policy_path)
        if self.device:
            self.policy.device = self.device

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Lets the parser load the policy config from `--policy.path=local/dir`."""
        return ["policy"]


def _build_observation(frame: dict) -> dict:
    """Keep what a policy consumes: camera frames, the state vector, and the task string."""
    obs = {
        key: value
        for key, value in frame.items()
        if isinstance(key, str) and (key.startswith("observation.image") or key == OBS_STATE)
    }
    obs["task"] = frame.get("task", "")
    return obs


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def _slope(a: torch.Tensor, b: torch.Tensor) -> float:
    """Least-squares gain of `a` over `b` — the magnitude that correlation cannot see."""
    a, b = a.reshape(-1), b.reshape(-1)
    denom = b @ b
    return float((a @ b) / denom) if denom > 0 else float("nan")


@torch.no_grad()
def eval_episode(policy, dataset, preprocessor, postprocessor, episode, n, stride) -> dict:
    sq_errors, hold_errors, pred_deltas, true_deltas = [], [], [], []

    for start in range(0, n, stride):
        anchor = dataset[start]
        policy.reset()
        chunk = policy.predict_action_chunk(preprocessor(_build_observation(anchor)))
        chunk = postprocessor(chunk)
        if chunk.ndim == 3:  # (batch, horizon, action_dim); this replay is single-env
            chunk = chunk[0]

        anchor_state = anchor.get(OBS_STATE)
        for offset in range(min(stride, n - start, chunk.shape[0])):
            action = chunk[offset]
            gt = dataset[start + offset][ACTION].to(action.device, action.dtype)
            if gt.ndim == action.ndim + 1:  # a recorded chunk vs a single step
                gt = gt[0]
            k = min(action.shape[-1], gt.shape[-1])
            sq_errors.append(((action[..., :k] - gt[..., :k]) ** 2).mean().cpu())

            # "Hold the anchor state" reference: for an action space commensurate with the state
            # (absolute joint positions, say) most of the target IS the current state, so a raw
            # MSE looks small whether or not the policy contributes anything.
            if anchor_state is None:
                continue
            state = anchor_state.to(gt.device, gt.dtype).reshape(-1)
            if state.shape[-1] < k:
                continue
            hold_errors.append(((state[..., :k] - gt[..., :k]) ** 2).mean().cpu())
            pred_deltas.append((action[..., :k].reshape(-1) - state[..., :k]).cpu())
            true_deltas.append((gt[..., :k] - state[..., :k]).cpu())

    result = {
        "episode": episode,
        "frames": len(sq_errors),
        "action_mse": float(torch.stack(sq_errors).mean()) if sq_errors else float("nan"),
        "hold_state_mse": float(torch.stack(hold_errors).mean()) if hold_errors else float("nan"),
        "delta_corr": float("nan"),
        "delta_slope": float("nan"),
    }
    if len(pred_deltas) > 1:
        pred, true = torch.stack(pred_deltas), torch.stack(true_deltas)
        # Correlation isolates the part the policy actually predicts (the displacement from the
        # anchor); slope says whether its magnitude is calibrated. Correlation is scale-free, so a
        # policy moving in the right direction by the wrong amount scores well on it, badly on MSE.
        result["delta_corr"] = _corr(pred, true)
        result["delta_slope"] = _slope(pred, true)
    logging.info(
        "episode %s: action_mse %.6f | hold_state_mse %.6f | delta_corr %.4f | delta_slope %.4f",
        episode,
        result["action_mse"],
        result["hold_state_mse"],
        result["delta_corr"],
        result["delta_slope"],
    )
    return result


@parser.wrap()
def main(cfg: OpenLoopEvalConfig):
    init_logging()
    logging.info(pformat(cfg.dataset.repo_id))
    if cfg.seed is not None:
        set_seed(cfg.seed)

    def open_dataset(episode: int) -> LeRobotDataset:
        return LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[episode])

    episodes = cfg.episodes or cfg.dataset.episodes
    if not episodes:
        meta = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root).meta
        episodes = list(range(meta.total_episodes))

    policy = make_policy(cfg=cfg.policy, ds_meta=open_dataset(episodes[0]).meta, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    stride = cfg.stride or getattr(policy.config, "n_action_steps", None) or 1
    per_episode = []
    for episode in episodes:
        dataset = open_dataset(episode)
        n = len(dataset)
        if cfg.max_frames_per_episode is not None:
            n = min(n, cfg.max_frames_per_episode)
        logging.info("Open-loop eval: episode %d, %d frames, stride %d", episode, n, stride)
        per_episode.append(eval_episode(policy, dataset, preprocessor, postprocessor, episode, n, stride))

    aggregate = {}
    for key in ("action_mse", "hold_state_mse", "delta_corr", "delta_slope"):
        values = [e[key] for e in per_episode if e[key] == e[key]]  # drop NaNs
        if values:
            aggregate[key] = sum(values) / len(values)
    info = {"per_episode": per_episode, "aggregate": aggregate}
    logging.info(pformat(aggregate))

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "eval_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
