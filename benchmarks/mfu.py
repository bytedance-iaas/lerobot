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

"""MFU for a LeRobot policy: per-card peak, analytic model FLOPs, and the division.

The FLOPs come from the policy's own `estimate_flops()`, which counts matmul parameters
structurally off the built modules (nanochat's approach). This file deliberately holds no
architecture knowledge -- a width or depth written down here would be a second source of
truth that silently goes stale.

The model is built on the **meta** device, so nothing is allocated and no checkpoint is
needed: counting parameters does not require their values. That makes this runnable on a
laptop, before any hardware is booked.

This is the second of two independent paths to the MFU numerator:

* `lerobot.utils.profiler_summary` sums per-kernel shapes out of an Ascend profile. It
  measures what actually ran -- but it needs a run, and it sees only the kernel types it
  has formulas for.
* this one derives the cost from the architecture. No run, no hardware, checkable by hand.

They share no code, which is what makes agreement evidence. Measured on pi05 / pusht /
one 950PR card, batch 32, bf16, one camera: this module says 6.314 TFLOP/sample, the
profile says 6.099 -- 3.4% apart. The residual is known and sits in two places: the
fused attention kernel pads the sequence (profile reads 8% above the analytic attention
term, which is itself only ~2% of the total), and this module bills every Linear in the
tree against its tower's token count, which slightly overcounts conditioning projections
that in fact run once per sample. Neither is worth closing by fudging one side.

Expect the profile to read HIGHER on a gradient-checkpointed run: recompute is real work
that it sees and this does not count, because recompute is HFU's business, not MFU's.

Usage:
    # FLOPs only
    python benchmarks/mfu.py pi05 --images 1

    # MFU from a step time measured in a train log
    python benchmarks/mfu.py pi05 --images 1 --batch 32 --step-s 1.84

    # Re-measure this machine's achievable peak rather than trusting the table
    python benchmarks/mfu.py --measure-peak
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import NamedTuple

TFLOP = 1e12

# ---------------------------------------------------------------------------------------
# Per-card peak. MEASURED, NOT SPEC SHEET.
#
# `--measure-peak` reproduces these: one large square GEMM, the most favourable shape
# there is, so the result is an achievable ceiling and not a theoretical one. MFU against
# a measured ceiling reads HIGHER than MFU against a theoretical peak -- when comparing
# against a published figure, check which denominator they used.
#
# Only dtypes actually measured on the hardware are listed. A guessed entry would
# silently rescale every MFU this tool ever prints, so a missing one is an error
# (`--measure-peak` or `--peak-tflops`), never a fallback.
#
# 950PR bf16 = 368.6 is also `profiler_summary.DEFAULT_PEAK_TFLOPS`; the two must stay
# equal, which `tests/benchmarks/test_mfu.py` asserts.
#
# The value is on the BURST basis, and what that costs is worth stating, because every
# MFU printed here is divided by it. Measured on idle cards, square bf16 GEMM:
#
#                       burst        sustained
#     950PR             363.6            335.3   (-7.8%)
#     H200              734.2            650.2   (-11.4%)
#
# `measure_peak` now reports the sustained figure, so it no longer reproduces 368.6 --
# it reads 9% under it. Sustained is the basis a training step actually runs at; the
# burst clock is one no step can hold. Moving the table to it would raise pi05's
# measured MFU on the 950PR from 61.0% to 67.0%, which is a re-baselining of every
# number this tool has printed, so it is a deliberate decision and not done here.
#
# A third basis exists and sits above both: during real pi05 training the profile shows
# its large GEMMs running at 374-375 TFLOP/s, above even the burst probe. A pure-GEMM
# probe runs the cube unit at 100% duty cycle, while a training step interleaves norms
# and elementwise work that let the clocks recover -- so the probe is pessimistic for
# a mixed workload and the hardware's own in-situ reading is the generous end.
# ---------------------------------------------------------------------------------------
PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "ascend950pr": {"bfloat16": 368.6},
}
DEFAULT_DEVICE = "ascend950pr"


def mm(m: int, k: int, n: int) -> float:
    """FLOPs of an (m,k) @ (k,n) matmul: a multiply and an add per MAC."""
    return 2.0 * m * k * n


def pi05_flops(**kwargs) -> dict[str, float]:
    """Build pi05 on the meta device and ask it what a sample costs.

    `freeze_vision_encoder` / `train_expert_only` go through the config rather than being
    re-implemented here, because `estimate_flops` reads `requires_grad` off the built
    modules -- so whatever the config actually does to the model is what gets costed.
    """
    import torch

    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    n_images = kwargs.pop("n_images", 1)
    text_tokens = kwargs.pop("text_tokens", None)
    cfg = PI05Config(**{k: v for k, v in kwargs.items() if v is not None})
    with torch.device("meta"):
        model = PI05Pytorch(cfg)
    return model.estimate_flops(n_images=n_images, text_tokens=text_tokens)


POLICIES = {"pi05": pi05_flops}


class Peak(NamedTuple):
    """A sustained peak reading, with the spread that produced it."""

    median: float
    lo: float
    hi: float


def measure_peak(
    dtype: str = "bfloat16", n: int = 8192, iters: int = 50, reps: int = 6, warmup_s: float = 3.0
) -> Peak:
    """Sustained TFLOP/s of a large square GEMM on this machine. Needs an accelerator.

    Sustained, not burst. A fresh card runs its first GEMMs at a boost clock it cannot
    hold: on an H200 the first timed block read 708 TFLOP/s and the next seven settled at
    ~672. Timing one block right after a few warmup calls therefore measures a clock the
    training run will never see, and flatters every MFU computed against it. So the
    warmup here is a wall-clock soak rather than a fixed iteration count, and the answer
    is the median of several blocks.

    The spread comes back with the number because it is not always small -- same card,
    same shape, 665 to 734 TFLOP/s depending on how busy its neighbours were. A peak
    quoted without it reads more precise than it is.
    """
    import time

    import torch

    if torch.cuda.is_available():
        dev, sync = "cuda", torch.cuda.synchronize
    else:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise SystemExit("no accelerator: --measure-peak needs a CUDA or NPU device")
        dev, sync = "npu", torch.npu.synchronize

    a = torch.randn(n, n, device=dev, dtype=getattr(torch, dtype))
    b = torch.randn(n, n, device=dev, dtype=getattr(torch, dtype))
    soak = time.perf_counter() + warmup_s  # let the clocks come down to what they can hold
    while time.perf_counter() < soak:
        a @ b
        sync()

    reads = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        sync()
        reads.append((iters * mm(n, n, n) / (time.perf_counter() - t0)) / TFLOP)
    return Peak(statistics.median(reads), min(reads), max(reads))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", nargs="?", choices=sorted(POLICIES))
    ap.add_argument("--images", type=int, default=1, help="cameras per observation")
    ap.add_argument("--text-tokens", type=int, help="prefix language tokens (default: tokenizer_max_length)")
    ap.add_argument("--chunk-size", type=int, help="action horizon")
    ap.add_argument("--freeze-vision-encoder", action="store_true", default=None)
    ap.add_argument("--train-expert-only", action="store_true", default=None)
    ap.add_argument("--batch", type=int, default=1, help="per-rank batch size")
    ap.add_argument("--ranks", type=int, default=1, help="world size, for global throughput")
    ap.add_argument("--step-s", type=float, help="measured seconds per step; omit for FLOPs only")
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=sorted(PEAK_TFLOPS))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--peak-tflops", type=float, help="override the table")
    ap.add_argument("--measure-peak", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    peak = None
    if args.measure_peak:
        read = measure_peak(args.dtype)
        peak = read.median
        print(
            f"measured peak: {peak:.1f} TFLOP/s  ({args.dtype}, square GEMM, sustained; "
            f"{read.lo:.1f}-{read.hi:.1f} across blocks)"
        )
        if not args.policy:
            return
    elif args.peak_tflops:
        peak = args.peak_tflops
    elif args.step_s:
        try:
            peak = PEAK_TFLOPS[args.device][args.dtype]
        except KeyError:
            raise SystemExit(
                f"no measured peak for {args.device}/{args.dtype}; run --measure-peak or pass "
                f"--peak-tflops. Guessing one would rescale every MFU printed here."
            ) from None

    if not args.policy:
        ap.error("a policy is required unless --measure-peak is all that was asked for")

    parts = POLICIES[args.policy](
        n_images=args.images,
        text_tokens=args.text_tokens,
        chunk_size=args.chunk_size,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
    )
    per_sample = parts.pop("total")
    per_step = per_sample * args.batch

    if args.json:
        out = {
            "policy": args.policy,
            "tflop_per_sample": per_sample / TFLOP,
            "tflop_per_step": per_step / TFLOP,
            "parts_tflop": {k: v / TFLOP for k, v in parts.items()},
        }
        if args.step_s:
            out["peak_tflops"] = peak
            out["achieved_tflops"] = per_step / args.step_s / TFLOP
            out["mfu_pct"] = 100.0 * (per_step / args.step_s) / (peak * TFLOP)
        print(json.dumps(out, indent=2))
        return

    print(f"{args.policy}: model FLOPs by architecture (fwd+bwd, matmuls only)")
    for name, v in parts.items():
        print(f"  {name:<28} {v / TFLOP:9.3f} TFLOP")
    print(f"  {'per sample':<28} {per_sample / TFLOP:9.3f} TFLOP")
    print(f"  {'per step (batch ' + str(args.batch) + ')':<28} {per_step / TFLOP:9.3f} TFLOP")
    if args.step_s:
        achieved = per_step / args.step_s
        print(f"  {'step time':<28} {args.step_s:9.3f} s")
        print(f"  {'throughput':<28} {args.batch * args.ranks / args.step_s:9.2f} samples/s global")
        print(f"  {'achieved':<28} {achieved / TFLOP:9.1f} TFLOP/s per card")
        print(f"  {'MFU':<28} {100.0 * achieved / (peak * TFLOP):9.1f} %  of {peak} TFLOP/s")
    else:
        print("  (pass --step-s to turn this into MFU)")


if __name__ == "__main__":
    main()
