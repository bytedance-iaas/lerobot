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

"""How much of a lingbot_va training step is the UMT5 text encoder, and where does it run.

`benchmarks/aten_ops.py` found 168 `mm` and 49 `rsqrt` calls running ON THE CPU in every
steady-state step -- 168 = 24 layers x 7 projections, 49 = 24 x 2 + 1 final RMSNorm. That is
a whole 24-layer UMT5 encoder executing on the host. `LingBotVAConfig.text_encoder_device`
defaults to `"cpu"`, and the comment explains why: the ~11 GB UMT5-XXL "runs once per
episode", so keeping it off-device lets the 5B transformer and the VAE share one card.

"Once per episode" holds for inference. In TRAINING, `_build_training_streams` calls
`_get_t5_prompt_embeds` on every step, so the encoder runs per step, on the host, for the
whole batch. This times that call against the step to turn the op counts into a share.

Same arguments as `lerobot-train`. Run it twice to A/B the placement:

    python benchmarks/t5_phase.py --policy.type=lingbot_va ... # cpu (default)
    python benchmarks/t5_phase.py --policy.type=lingbot_va ... --policy.text_encoder_device=npu
"""

import logging
import os
import time

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.collate import lerobot_collate_fn

_T5 = {"s": 0.0, "n": 0}


def _loss(out):
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out["loss"]
    return out


def _sync(device):
    # The encoder itself is synchronous when it runs on the host, but the rest of the step is
    # queued -- without a sync the step time is the launch time, not the step.
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


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

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn
    )
    batch = next(iter(loader))
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(rename_batch_keys(batch, cfg.rename_map))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    orig = LingBotVAPolicy._get_t5_prompt_embeds

    def timed(self, prompt, max_sequence_length):
        t0 = time.perf_counter()
        out = orig(self, prompt, max_sequence_length)
        _T5["s"] += time.perf_counter() - t0
        _T5["n"] += 1
        return out

    LingBotVAPolicy._get_t5_prompt_embeds = timed

    te_dev = "?"
    iters = int(os.environ.get("LEROBOT_T5_ITERS", "4"))
    print(f"\n{'pass':<6}{'step_s':>10}{'t5_s':>10}{'t5 share':>10}{'calls':>7}")
    for i in range(iters):
        policy.zero_grad(set_to_none=True)
        _T5["s"], _T5["n"] = 0.0, 0
        _sync(device)
        t0 = time.perf_counter()
        _loss(policy(batch)).backward()
        _sync(device)
        step = time.perf_counter() - t0
        share = 100 * _T5["s"] / step if step else 0.0
        print(f"{i:<6}{step:>10.3f}{_T5['s']:>10.3f}{share:>9.1f}%{_T5['n']:>7}")
        if te_dev == "?":
            te = policy._frozen.get("text_encoder") if getattr(policy, "_frozen", None) else None
            te_dev = str(next(te.parameters()).device) if te is not None else "n/a"

    print(f"\ntext_encoder 实际所在设备: {te_dev}  (config={cfg.policy.text_encoder_device})")
    print("第 0 遍含惰性加载权重,看后面几遍。")


if __name__ == "__main__":
    main()
