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

"""Where a training step's device syncs come from.

A sync drains the queue: the host cannot read the value until every kernel already
launched has finished, so one `.item()` in a hot path costs the whole pipeline's depth
rather than its own microseconds.

lingbot_va dispatches 203 `_local_scalar_dense` calls in one step -- that is `.item()` --
and an Ascend profile of the same step puts more host ACL time in 192 synchronization
calls (1.14 s) than in 49k kernel launches (0.77 s). Those are one population seen from
two sides, and neither side names a line of source: an aten op carries no call site, and
a Level1 profile taken without `record_shapes` has an empty `Call Stack`.

So this records the Python stack at dispatch time for the ops that force a sync and
aggregates by call site, which turns "203 syncs somewhere" into a list of lines to read.

Takes the SAME arguments as `lerobot-train`, like `benchmarks/aten_ops.py`:

    python benchmarks/sync_sites.py --policy.type=lingbot_va --dataset.repo_id=lerobot/libero \
        --policy.device=npu --batch_size=2 --steps=1 --save_checkpoint=false

It runs SEVERAL forward+backward passes and reports each separately, because one pass
cannot tell a per-step cost from a once-per-process one. Measured here: lingbot_va loads
a submodule's weights inside its FIRST forward, and `safetensors.torch.load_file` does
196 `.item()` calls while reading the state dict. A single-pass probe files all 196 under
"per step" and inflates the count by 18x. Read the steady-state column, not the first.
"""

import collections
import logging
import os
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.collate import lerobot_collate_fn

# Ops that cannot return without the device catching up. `_local_scalar_dense` is what
# `.item()`, `float(t)` and an f-string of a 0-d tensor all lower to; the rest need a
# device-dependent output SHAPE, which the host cannot know until the data is read back.
SYNC_OPS = frozenset(
    {
        "_local_scalar_dense",
        "nonzero",
        "masked_select",
        "unique2",
        "_unique2",
        "item",
        "equal",
        "allclose",
        "_assert_async",
        "_local_scalar_dense_",
    }
)

# Frames that are never the answer: reporting `torch/_tensor.py` as the call site is true
# and useless. The deepest frame OUTSIDE these is the line a person can go and change.
_SKIP = ("/torch/", "/torch_npu/", "sync_sites.py", "/_python_dispatch", "/accelerate/")


def _site(limit: int = 3) -> tuple[str, ...]:
    out = []
    for fr in reversed(traceback.extract_stack()):
        if any(s in fr.filename for s in _SKIP):
            continue
        short = "/".join(fr.filename.rsplit("/", 2)[-2:])
        out.append(f"{short}:{fr.lineno} {(fr.line or '').strip()[:72]}")
        if len(out) >= limit:
            break
    return tuple(out)


class _SyncRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.it = 0
        self.by_op: collections.Counter = collections.Counter()
        self.by_site: collections.Counter = collections.Counter()
        self.stacks: dict[str, tuple[str, ...]] = {}
        self.d2h = 0
        # Per-iteration tallies: {iteration: Counter}. The first iteration carries lazy
        # weight loading and one-off cache fills; only the later ones are the step cost.
        self.per_it: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.op_it: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]
        hit = name in SYNC_OPS
        # A copy whose source is on the accelerator and whose target is the host is also a
        # sync, and it is NOT in SYNC_OPS: it is spelled `_to_copy` or `copy_`, same as the
        # H2D direction. Tell them apart by where the tensors are, not by the op name.
        if not hit and name in ("_to_copy", "copy_"):
            devs = [a.device.type for a in args if isinstance(a, torch.Tensor)]
            tgt = (kwargs or {}).get("device")
            if devs and devs[0] != "cpu" and (str(tgt) == "cpu" or (len(devs) > 1 and devs[1] == "cpu")):
                hit = True
                self.d2h += 1
        if hit:
            self.by_op[name] += 1
            st = _site()
            key = st[0] if st else "(no user frame)"
            self.by_site[key] += 1
            self.per_it[self.it][key] += 1
            self.op_it[self.it][name] += 1
            self.stacks.setdefault(key, st)
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

    iters = int(os.environ.get("LEROBOT_SYNC_ITERS", "3"))
    rec = _SyncRecorder()
    for i in range(iters):
        policy.zero_grad(set_to_none=True)
        rec.it = i
        with rec:
            _loss(policy(batch)).backward()

    print(f"\n{sum(rec.by_op.values())} sync-forcing calls over {iters} passes "
          f"({rec.d2h} D2H copies). Per pass:\n")
    hdr = "".join(f"{'pass ' + str(i):>10}" for i in range(iters))
    print(f"{'op':<28}{hdr}")
    for k in sorted(rec.by_op, key=lambda k: -rec.by_op[k]):
        print(f"{k:<28}" + "".join(f"{rec.op_it[i][k]:>10}" for i in range(iters)))
    print(f"\n{'TOTAL':<28}" + "".join(f"{sum(rec.op_it[i].values()):>10}" for i in range(iters)))

    print(f"\n{'call site (deepest non-torch frame)':<80}{hdr}")
    for site, _ in rec.by_site.most_common(25):
        print(f"{site:<80}" + "".join(f"{rec.per_it[i][site]:>10}" for i in range(iters)))
    steady = sum(rec.per_it[iters - 1].values()) if iters > 1 else None
    if steady is not None:
        print(f"\n稳态(最后一遍)每步同步数 = {steady}")

    want = int(os.environ.get("LEROBOT_SYNC_STACKS", "3"))
    print(f"\n--- stacks of the top {want} sites ---")
    for site, n in rec.by_site.most_common(want):
        print(f"\n[{n} calls] {site}")
        for fr in rec.stacks[site][1:]:
            print(f"    called from {fr}")


if __name__ == "__main__":
    main()
