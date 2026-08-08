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

"""Offline open-loop evaluation for a converted DreamZero checkpoint.

Replays a held-out episode one frame at a time, feeding each observation through the DreamZero
preprocessor -> ``select_action`` -> postprocessor, and comparing the predicted action against the
recorded ground-truth action (mean squared error). This is the LeRobot analogue of upstream
``compare_loss.py`` / ``open_loop_*.py`` and the M2 correctness gate in ``dreamzero_port_plan.md``.

The checkpoint is the LeRobot layout produced by ``convert_dreamzero_checkpoint.py`` (``config.json``
+ ``model.safetensors`` + ``statistics.json``). Running the model needs a GPU with enough memory
(DROID/14B ≥ ~74 GB, so an H20-class card); this script is device-agnostic but only meaningful there.

Usage:
    python -m lerobot.policies.dreamzero.scripts.offline_eval \
        --checkpoint /path/to/lerobot-dreamzero-droid \
        --repo-id <held-out LeRobotDataset> --episode 0 [--max-frames 300] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


def _build_observation(frame: dict) -> dict:
    """Extract the raw observation the DreamZero preprocessor expects from a dataset frame.

    Keeps image keys (``observation.images.*`` / ``observation.image``), the state vector, and the
    task string; drops the action and bookkeeping keys.
    """
    obs: dict = {}
    for key, value in frame.items():
        if not isinstance(key, str):
            continue
        if key.startswith("observation.image") or key == OBS_STATE:
            obs[key] = value
    return obs


def offline_eval(
    checkpoint: str,
    repo_id: str,
    episode: int = 0,
    max_frames: int | None = None,
    device: str = "cuda",
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy
    from lerobot.policies.dreamzero.processor_dreamzero import (
        make_dreamzero_pre_post_processors_from_pretrained,
    )

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    policy = DreamZeroPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_dreamzero_pre_post_processors_from_pretrained(config, checkpoint)

    dataset = LeRobotDataset(repo_id, episodes=[episode])
    n = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    logger.info("Evaluating %d frames of episode %d from %s", n, episode, repo_id)

    policy.reset()
    sq_errors: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(n):
            frame = dataset[i]
            observation = _build_observation(frame)
            # The dataset carries the language instruction under "task"; the preprocessor reads it
            # from COMPLEMENTARY_DATA, so pass it alongside the observation.
            observation["task"] = frame.get("task", "")
            processed = preprocessor(observation)
            action = policy.select_action(processed)
            action = postprocessor(action)

            gt = frame[ACTION].to(action.device, action.dtype)
            if gt.ndim == action.ndim + 1:  # (T, D) recorded chunk vs (D,) single step
                gt = gt[0]
            k = min(action.shape[-1], gt.shape[-1])  # compare shared (real) action dims
            sq_errors.append(((action[..., :k] - gt[..., :k]) ** 2).mean().detach().cpu())

    mse = float(torch.stack(sq_errors).mean()) if sq_errors else float("nan")
    logger.info("episode %d: mean action MSE over %d frames = %.6f", episode, len(sq_errors), mse)
    return {"episode": episode, "frames": len(sq_errors), "action_mse": mse}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Converted LeRobot DreamZero checkpoint dir")
    p.add_argument("--repo-id", required=True, help="Held-out LeRobotDataset repo id or local path")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = offline_eval(args.checkpoint, args.repo_id, args.episode, args.max_frames, args.device)
    print(result)


if __name__ == "__main__":
    main()
