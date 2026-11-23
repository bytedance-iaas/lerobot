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

"""Every aten op one training step dispatches, with counts and shapes.

A profile names KERNELS, not source calls. When a kernel dominates a step and
`grep` finds nothing in the policy that would issue it, the two are separated by the
dispatcher: `aclnnTriangularSolve` taking 54% of g05's device time corresponds to no
`triangular_solve` anywhere in the repo, and an Ascend profile taken at Level1 without
`record_shapes` carries neither call stacks nor input shapes to close the gap.

Reading the dispatcher closes it from the other side: this records the aten op for every
call the step issues, so the kernel can be matched to the op that produced it.

It takes the SAME arguments as `lerobot-train`, like `benchmarks/measured_flops.py`:

    python benchmarks/aten_ops.py --policy.type=act --dataset.repo_id=lerobot/pusht \
        --policy.device=npu --batch_size=2 --steps=1 --save_checkpoint=false

It runs SEVERAL forward+backward passes and reports each separately. One pass is NOT
enough, even though every step dispatches the same ops: a policy that loads a submodule's
weights lazily does that work inside the FIRST forward, and those calls are then counted
as if they happened every step. Measured on lingbot_va: 196 of the first pass's 207
`_local_scalar_dense` calls came from `safetensors.torch.load_file`, and the steady-state
count is 11. Read the last pass, not the first.

Counts are what matter here, not time: use the profile for time.
"""

import collections
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


class _OpRecorder(TorchDispatchMode):
    """Count every dispatched aten op, keeping one example shape signature each."""

    def __init__(self):
        super().__init__()
        self.counts: collections.Counter = collections.Counter()
        self.shapes: dict[str, str] = {}
        # Where each call's tensors live, and how many elements it touched there. A constant
        # call count whose host time grows with batch cannot be dispatch overhead -- dispatch
        # does not care how big a tensor is. It is arithmetic running on the CPU.
        self.by_dev: collections.Counter = collections.Counter()
        self.elems: collections.Counter = collections.Counter()
        self.cpu_ops: collections.Counter = collections.Counter()
        self.it = 0
        # Per-pass tallies, so lazy weight loading in the first forward cannot masquerade
        # as a per-step cost.
        self.it_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.it_dev: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.it_elems: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.it_cpu: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]
        self.counts[name] += 1
        self.it_counts[self.it][name] += 1
        ts = [a for a in args if isinstance(a, torch.Tensor)]
        if ts:
            # Classify by where ALL of a call's tensors live, not by the first one. Attributing
            # on arg[0] files every H2D copy under "cpu" and -- if the elements of every arg are
            # then added to that bucket -- counts the accelerator-side tensor as CPU work. The
            # bucket then grows with batch size for a model that does nothing on the CPU at all.
            devs = {a.device.type for a in ts}
            if len(devs) > 1:
                self.by_dev["mixed (H2D/D2H)"] += 1
            else:
                dev = devs.pop()
                self.by_dev[dev] += 1
                self.elems[dev] += sum(a.numel() for a in ts)
                self.it_dev[self.it][dev] += 1
                self.it_elems[self.it][dev] += sum(a.numel() for a in ts)
                if dev == "cpu":
                    self.cpu_ops[name] += 1
                    self.it_cpu[self.it][name] += 1
        if name not in self.shapes:
            sig = ", ".join(
                "x".join(str(d) for d in a.shape) + f":{a.dtype}".replace("torch.", "")
                for a in args
                if isinstance(a, torch.Tensor)
            )
            self.shapes[name] = sig[:110]
        return func(*args, **(kwargs or {}))


def _loss(out):
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict):
        return out["loss"]
    return out


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

    iters = int(os.environ.get("LEROBOT_ATEN_ITERS", "3"))
    rec = _OpRecorder()
    for i in range(iters):
        policy.zero_grad(set_to_none=True)
        rec.it = i
        with rec:
            _loss(policy(batch)).backward()

    want = os.environ.get("LEROBOT_ATEN_GREP", "")
    last = iters - 1
    hdr = "".join(f"{'pass ' + str(i):>12}" for i in range(iters))
    print(f"\n{sum(rec.counts.values())} dispatched calls over {iters} passes, "
          f"{len(rec.counts)} distinct aten ops")
    print(f"\n{'dispatched calls':<34}{hdr}")
    print(f"{'total':<34}" + "".join(f"{sum(rec.it_counts[i].values()):>12}" for i in range(iters)))

    print(f"\n{'calls by device':<34}{hdr}")
    for dev in ("npu", "cpu", "meta", "mixed (H2D/D2H)"):
        if any(rec.it_dev[i][dev] for i in range(iters)):
            print(f"{dev:<34}" + "".join(f"{rec.it_dev[i][dev]:>12}" for i in range(iters)))
    print(f"\n{'elements touched by device':<34}{hdr}")
    for dev in ("npu", "cpu", "meta"):
        if any(rec.it_elems[i][dev] for i in range(iters)):
            print(f"{dev:<34}" + "".join(f"{rec.it_elems[i][dev]:>12,}" for i in range(iters)))

    if rec.it_cpu[last]:
        print(f"\naten ops running ON CPU in the LAST pass (steady state):")
        print(f"  {'op':<32}{hdr}")
        for k, _ in rec.it_cpu[last].most_common(14):
            print(f"  {k:<32}" + "".join(f"{rec.it_cpu[i][k]:>12}" for i in range(iters)))

    print(f"\n{'aten op':<30}{'last pass':>10}{'pass 0':>9}  example shapes")
    for name, _ in rec.it_counts[last].most_common(40):
        print(f"{name:<30}{rec.it_counts[last][name]:>10}{rec.it_counts[0][name]:>9}  {rec.shapes[name]}")
    if want:
        print(f"\nmatching {want!r}:")
        hit = [(k, v) for k, v in rec.counts.items() if want.lower() in k.lower()]
        for k, v in sorted(hit, key=lambda kv: -kv[1]):
            print(f"  {k:<32}{v:>8}  {rec.shapes[k]}")
        if not hit:
            print("  (none)")


if __name__ == "__main__":
    main()
