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

"""Model FLOPs per training sample, MEASURED off one real forward+backward.

`benchmarks/mfu.py` derives the same number analytically, but only for the three policies
that implement `estimate_flops`. Writing that derivation per architecture does not scale
to the registry, and for a CNN backbone it is easy to get quietly wrong: `params x tokens`
holds only for a patch embedding whose kernel equals its stride, not for a ResNet, where
each layer's cost depends on its own spatial size. A wrong FLOPs model does not announce
itself -- it silently rescales every MFU computed against it.

So this reads the shapes instead of predicting them. `torch.utils.flop_counter` sits under
the dispatcher and counts the matmuls and convolutions the step actually issued:

    python benchmarks/measured_flops.py --policy.type=act --dataset.repo_id=lerobot/pusht \
        --policy.device=npu --batch_size=1 --steps=1 --save_checkpoint=false

It takes the SAME arguments as `lerobot-train`, so a policy's known-good training command
is also its FLOPs command -- no second description of the model to keep in sync.

What it counts, and why it matches the MFU convention:

* matmuls and convolutions only. Norms, softmax, rotary and activations are
  bandwidth-bound and every published MFU omits them.
* forward AND backward, as issued. The 3x / 2x / 1x rule that `estimate_flops` has to
  infer from `requires_grad` is not assumed here: a frozen tower simply never issues its
  weight-gradient matmul, so the count follows whatever the config actually froze.
* recompute IS included when gradient checkpointing is on, because the recomputed
  forwards really are dispatched. That makes the total HFU-flavoured, not MFU. Pass
  `--policy.gradient_checkpointing=false` for the MFU numerator, or read the
  `recompute` line below, which prices the gap by counting a second time with
  checkpointing off.

Set LEROBOT_STEP_S to a step time measured OUTSIDE this script to get MFU. It must come
from a frozen baseline run (50 steps, first 10 discarded) rather than from here: building
the model and pulling the first batch dominates this process, so nothing it times is a
throughput measurement.
"""

import json
import logging
import os

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.collate import lerobot_collate_fn

TFLOP = 1e12
# Same denominator as benchmarks/mfu.py and profiler_summary.DEFAULT_PEAK_TFLOPS. Sustained
# square bf16 GEMM on an idle 950PR, not the burst clock and not a spec sheet.
DEFAULT_PEAK_TFLOPS = 335.3


def _one_batch(cfg: TrainPipelineConfig, dataset) -> dict:
    """One collated batch at the configured batch size, in dataset order.

    Deliberately not the EpisodeAwareSampler the real loop uses: which frames come back
    changes padding and therefore the token count, so a sampler that reorders episodes
    would make this unreproducible between runs for no gain. FLOPs depend on shapes, and
    every batch of a given policy+dataset has the same shapes.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lerobot_collate_fn,
    )
    return next(iter(loader))


def _prepare(batch: dict, camera_keys, rename_map, preprocessor):
    """Byte-for-byte what the training loop does to a raw batch before the policy sees it."""
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    return preprocessor(rename_batch_keys(batch, rename_map))


def _loss(out):
    """`lerobot-train` unpacks `loss, output_dict`; some policies return a dict instead."""
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out["loss"]
    return out


def _count(policy, batch):
    """FLOPs of one forward+backward, total and by aten op."""
    from torch.utils.flop_counter import FlopCounterMode

    policy.zero_grad(set_to_none=True)
    counter = FlopCounterMode(display=False)
    with counter:
        _loss(policy(batch)).backward()
    by_op = {str(op): v for op, v in counter.get_flop_counts()["Global"].items()}
    return float(counter.get_total_flops()), by_op


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg.validate()
    device = torch.device(cfg.policy.device)

    dataset, _ = make_train_eval_datasets(cfg)
    policy = make_policy(cfg=cfg.trainable_config, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.to(device).train()

    kwargs = ProcessorConfigKwargs()
    kwargs["dataset_stats"] = rename_stats(dataset.meta.stats, cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg.trainable_config, **kwargs)

    batch = _prepare(_one_batch(cfg, dataset), dataset.meta.camera_keys, cfg.rename_map, preprocessor)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    total, by_op = _count(policy, batch)

    # Price the recompute separately when it is on, so the same run yields both an MFU
    # numerator (model work) and an HFU one (work the hardware ran). Policies that do not
    # expose the flag just skip this.
    model_total, recompute = total, 0.0
    if getattr(cfg.policy, "gradient_checkpointing", False):
        try:
            policy.gradient_checkpointing_disable()
            model_total, _ = _count(policy, batch)
            recompute = total - model_total
        except AttributeError:
            logging.warning(
                "gradient_checkpointing is on but the policy has no "
                "gradient_checkpointing_disable(); the total below includes recompute, so it "
                "is an HFU numerator, not an MFU one."
            )

    bs = cfg.batch_size
    out = {
        "policy": cfg.policy.type,
        "batch_size": bs,
        "tflop_per_step": model_total / TFLOP,
        "tflop_per_sample": model_total / bs / TFLOP,
        "by_op_tflop": {k: v / TFLOP for k, v in sorted(by_op.items(), key=lambda kv: -kv[1])},
    }
    if recompute:
        out["recompute_tflop_per_step"] = recompute / TFLOP

    step_s = os.environ.get("LEROBOT_STEP_S")
    if step_s:
        peak = float(os.environ.get("LEROBOT_PEAK_TFLOPS", DEFAULT_PEAK_TFLOPS))
        achieved = model_total / float(step_s)
        out["step_s"] = float(step_s)
        out["peak_tflops"] = peak
        out["achieved_tflops"] = achieved / TFLOP
        out["mfu_pct"] = 100.0 * achieved / (peak * TFLOP)
        if recompute:
            out["hfu_pct"] = 100.0 * ((model_total + recompute) / float(step_s)) / (peak * TFLOP)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
