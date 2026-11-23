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

"""Numerical gate for moving lingbot_va's UMT5 text encoder off the host.

`benchmarks/t5_phase.py` measured the encoder at 92.1% (bs2) and 94.5% (bs8) of a training
step, and `--policy.text_encoder_device=npu` cut the step 9.0x / 12.2x. Before that counts
as accepted, the candidate has to clear a numerical gate: the encoder is frozen and in eval
mode, so the two placements should agree to within device arithmetic, nothing more.

This runs the SAME call (`_get_t5_prompt_embeds`) over the SAME prompts with the SAME
weights, once per placement, and reports the difference. It is near-exact, not exact: bf16
matmuls accumulate differently on an NPU cube than on a host vector unit, so the pass
criterion is a small relative error, not bit equality.
"""

import logging

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy
from lerobot.utils.collate import lerobot_collate_fn


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg.validate()
    device = torch.device(cfg.policy.device)

    dataset, _ = make_train_eval_datasets(cfg)
    policy = make_policy(cfg=cfg.trainable_config, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.to(device).eval()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lerobot_collate_fn
    )
    batch = next(iter(loader))
    task = batch.get("task")
    prompts = list(task) if task is not None else ["pick up the black bowl"] * cfg.batch_size
    print(f"\nprompts ({len(prompts)}):")
    for p in prompts[: min(4, len(prompts))]:
        print(f"  {p!r}")

    policy._ensure_frozen_modules()
    te = policy._frozen["text_encoder"]
    max_len = policy.config.max_sequence_length

    # Three placements, not two. Comparing npu-bf16 against cpu-bf16 alone cannot say whose
    # arithmetic moved: torch's CPU bf16 gemm accumulates in fp32, so the host path may be
    # the more precise one and the whole difference would be the NPU's bf16 rounding -- which
    # is what ANY device-resident encoder would produce. cpu-fp32 is the reference that tells
    # them apart.
    outs = {}
    with torch.no_grad():
        for tag, dev, dt in (("cpu-bf16", "cpu", policy.dtype), ("cpu-fp32", "cpu", torch.float32),
                             ("npu-bf16", device, policy.dtype)):
            te.to(device=dev, dtype=dt)
            print(f"encoder on {next(te.parameters()).device} / {next(te.parameters()).dtype} ...")
            outs[tag] = policy._get_t5_prompt_embeds(prompts, max_len).float().cpu()

    ref = outs["cpu-fp32"]
    print(f"\nshape {tuple(ref.shape)}   参照 = cpu-fp32,比较一律在 float32 下做\n")
    print(f"{'对比':<22}{'max |Δ|':>12}{'mean |Δ|':>12}{'‖Δ‖/‖ref‖':>13}{'cosine':>14}")
    for a, b in (("cpu-bf16", "cpu-fp32"), ("npu-bf16", "cpu-fp32"), ("npu-bf16", "cpu-bf16")):
        x, y = outs[a], outs[b]
        d = (x - y).abs()
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        print(f"{a + ' vs ' + b:<22}{d.max().item():>12.3e}{d.mean().item():>12.3e}"
              f"{((x - y).norm() / y.norm()).item():>13.3e}{cos.item():>14.9f}")


if __name__ == "__main__":
    main()
