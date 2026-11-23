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
from torch.utils._python_dispatch import TorchDispatchMode

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


_GEMM = {"mm", "addmm", "bmm", "baddbmm"}


def _up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


class _ShapeRecorder(TorchDispatchMode):
    """Record every GEMM's real shape, and what the cube's alignment makes of it.

    The Ascend profiler bills the padded work in `aic_cube_fops`, so a model whose
    hardware FLOPs run far above its model FLOPs is paying for alignment -- but a profile
    taken without `record_shapes` cannot say which operator does it. Reading the shapes
    off the dispatcher answers that without a second profiling run.
    """

    def __init__(self, align: int = 16):
        super().__init__()
        self.align = align
        self.seen: dict[tuple, list] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]
        if name in _GEMM:
            t = [a for a in args if isinstance(a, torch.Tensor)]
            # mm/bmm take (a, b); addmm/baddbmm take (bias, a, b).
            off = 1 if name in ("addmm", "baddbmm") else 0
            if len(t) >= off + 2:
                a, b = t[off], t[off + 1]
                if a.dim() == b.dim() and a.dim() in (2, 3):
                    bb = a.shape[0] if a.dim() == 3 else 1
                    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
                    model = 2.0 * bb * m * k * n
                    pad = 2.0 * bb * _up(m, self.align) * _up(k, self.align) * _up(n, self.align)
                    r = self.seen.setdefault((name, bb, m, k, n), [0, 0.0, 0.0])
                    r[0] += 1
                    r[1] += model
                    r[2] += pad
        return func(*args, **(kwargs or {}))


def _count(policy, batch, shapes: bool = False):
    """FLOPs of one forward+backward, total and by aten op; optionally GEMM shapes too."""
    from torch.utils.flop_counter import FlopCounterMode

    policy.zero_grad(set_to_none=True)
    counter = FlopCounterMode(display=False)
    rec = _ShapeRecorder() if shapes else None
    with counter:
        if rec is not None:
            with rec:
                _loss(policy(batch)).backward()
        else:
            _loss(policy(batch)).backward()
    by_op = {str(op): v for op, v in counter.get_flop_counts()["Global"].items()}
    return float(counter.get_total_flops()), by_op, (rec.seen if rec else None)


def _report_shapes(seen: dict) -> None:
    tm = sum(v[1] for v in seen.values())
    tp = sum(v[2] for v in seen.values())
    if not tm:
        return
    print(f"\nGEMM shapes: {len(seen)} distinct")
    print(f"  model   {tm / TFLOP:8.3f} TFLOP/step")
    print(f"  16-aligned {tp / TFLOP:8.3f} TFLOP/step  = {tp / tm:.2f}x model")
    print(
        f"\n{'op':<9}{'batch':>8}{'M':>9}{'K':>7}{'N':>7}{'calls':>6}"
        f"{'model TF':>10}{'padded TF':>11}{'ratio':>7}"
    )
    def rows(order, label):
        print(f"  -- top by {label} --")
        for (name, bb, m, k, n), v in sorted(seen.items(), key=order)[:10]:
            print(
                f"{name:<9}{bb:>8}{m:>9}{k:>7}{n:>7}{v[0]:>6}"
                f"{v[1] / TFLOP:>10.3f}{v[2] / TFLOP:>11.3f}{v[2] / v[1]:>7.1f}"
            )

    # Where the model's work actually is, and separately where alignment wastes it: a
    # shape can dominate one ranking and be absent from the other.
    rows(lambda kv: -kv[1][1], "model FLOPs")
    rows(lambda kv: -(kv[1][2] - kv[1][1]), "alignment waste")


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

    want_shapes = os.environ.get("LEROBOT_FLOPS_SHAPES") == "1"
    total, by_op, seen = _count(policy, batch, shapes=want_shapes)

    # Price the recompute separately when it is on, so the same run yields both an MFU
    # numerator (model work) and an HFU one (work the hardware ran). Policies that do not
    # expose the flag just skip this.
    model_total, recompute = total, 0.0
    if getattr(cfg.policy, "gradient_checkpointing", False):
        try:
            policy.gradient_checkpointing_disable()
            model_total, _, _ = _count(policy, batch)
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
    if seen:
        _report_shapes(seen)


if __name__ == "__main__":
    main()
