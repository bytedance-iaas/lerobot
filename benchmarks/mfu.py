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

The FLOPs come from the policy's own `estimate_flops()`, which for pi05 and GR00T counts
matmul parameters structurally off the built modules (nanochat's approach), and for DreamZero
runs the training forward on meta with hooks and adds up what it calls -- see that method for
why a formula is the wrong tool there. This file deliberately holds no architecture knowledge
either way: a width or depth written down here would be a second source of truth that silently
goes stale.

The model is built on the **meta** device, so nothing is allocated and no checkpoint is
needed: counting parameters does not require their values. That makes this runnable on a
laptop, before any hardware is booked.

This is the second of two independent paths to the MFU numerator:

* `lerobot.utils.profiler_summary` sums per-kernel shapes out of an Ascend profile. It
  measures what actually ran -- but it needs a run, and it sees only the kernel types it
  has formulas for.
* this one derives the cost from the architecture. No run, no hardware, checkable by hand.

They share no code, which is what makes agreement evidence. Both measured on pusht, one
950PR card, batch 32, one camera:

    pi05    6.314 TFLOP/sample here   6.099 from the profile   3.4% apart
    groot   0.720                     0.719                    0.1% apart

pi05's residual is known and sits in two places: the fused attention kernel pads the
sequence (profile reads 8% above the analytic attention term, which is itself only ~2% of
the total), and pi05's own `estimate_flops` bills every Linear in a tower against that
tower's token count, which slightly overcounts conditioning projections that in fact run
once per sample. Neither is worth closing by fudging one side. groot agrees more closely
because its `estimate_flops` attributes those conditioning projections explicitly -- it
has to, since N1.7 runs five different token counts at once.

Expect the profile to read HIGHER on a gradient-checkpointed run: recompute is real work
that it sees and this does not count, because recompute is HFU's business, not MFU's.

MFU and HFU differ only in the numerator, and the difference is the point:

    MFU = model FLOPs / time / peak       the arithmetic the model's math requires
    HFU = hardware FLOPs / time / peak    everything the card actually ran, recompute included

HFU >= MFU, equal when nothing is recomputed. A policy that recomputes returns the extra work
under a `recompute` key that is NOT in its `total`, and both are printed, because either number
alone misleads: with checkpointing on, a high HFU and a low MFU means the kernels are fine and
a quarter of their output is being thrown away to save memory, which is a conclusion about the
memory budget rather than about the kernels. (Definitions: MFU from PaLM, HFU from NVIDIA's
"Reducing Activation Recomputation in Large Transformer Models".)

Usage:
    # FLOPs only
    python benchmarks/mfu.py pi05 --images 1

    # groot needs the prompt's token count, and a local copy of its gated backbone config
    python benchmarks/mfu.py groot --text-tokens 19 --backbone-path /path/to/Cosmos-Reason2-2B

    # MFU from a step time measured in a train log
    python benchmarks/mfu.py pi05 --images 1 --batch 32 --step-s 1.84

    # dreamzero costs the TRAINING window, and reports HFU alongside MFU because the DiT
    # recomputes. 8xH200, training_mode=full, micro-batch 1:
    python benchmarks/mfu.py dreamzero --training-mode=full --ranks 8 --step-s 6.9 \
        --peak-tflops 650.2

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
# 950PR bf16 = 335.3 is also `profiler_summary.DEFAULT_PEAK_TFLOPS`; the two must stay
# equal, which `tests/benchmarks/test_mfu.py` asserts.
#
# The value is on the SUSTAINED basis, and which basis it is matters more than the
# digits, because every MFU printed here is divided by it. Measured on idle cards,
# square bf16 GEMM:
#
#                       burst        sustained
#     950PR             363.6            335.3   (-7.8%)
#     H200              734.2            650.2   (-11.4%)
#
# Sustained is what a training step actually runs at; the burst clock is one no step can
# hold, so dividing by it credits the hardware with a ceiling it never offers. The table
# held the burst figure (368.6) until the 950PR reports were rewritten in MFU terms, and
# moving it here raises every MFU this tool prints by ~10% -- pi05 on the 950PR reads
# 67.0% where it used to read 61.0%. Published numbers from before that switch are on
# the old basis; divide by 1.0993 to compare. `measure_peak` reports sustained, so the
# table is now reproducible from it, which 368.6 never was.
#
# A third basis exists and sits above both: during real pi05 training the profile shows
# its large GEMMs running at 374-375 TFLOP/s, above even the burst probe. A pure-GEMM
# probe runs the cube unit at 100% duty cycle, while a training step interleaves norms
# and elementwise work that let the clocks recover -- so the probe is pessimistic for
# a mixed workload and the hardware's own in-situ reading is the generous end.
# ---------------------------------------------------------------------------------------
PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "ascend950pr": {"bfloat16": 335.3},
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
    fields = ("chunk_size", "freeze_vision_encoder", "train_expert_only")
    cfg = PI05Config(**{k: kwargs[k] for k in fields if kwargs.get(k) is not None})
    with torch.device("meta"):
        model = PI05Pytorch(cfg)
    return model.estimate_flops(n_images=n_images, text_tokens=text_tokens)


def groot_flops(**kwargs) -> dict[str, float]:
    """Build GR00T N1.7 on the meta device from config and ask it what a sample costs.

    `load_backbone_weights=False` keeps this off the 6.5 GB checkpoint -- the backbone is
    built from its config, which is all a parameter count needs. That config does have to
    be reachable, and `nvidia/Cosmos-Reason2-2B` is a gated repo, so off-box you point
    `--backbone-path` at a local copy of it.
    """
    import torch

    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    text_tokens = kwargs.get("text_tokens")
    if text_tokens is None:
        raise SystemExit(
            "groot needs --text-tokens. The backbone's language token count is prompt-\n"
            "dependent and no config field records it, so there is nothing honest to\n"
            "default to. Read it off a profile -- a GEMM's M divided by the batch size is\n"
            "the whole backbone sequence, minus n_images * 64 merged image tokens -- or\n"
            "tokenize the prompt. On pusht it is 19."
        )
    overrides = {
        name: kwargs[name]
        for name in ("action_horizon", "tune_llm", "tune_visual", "select_layer")
        if kwargs.get(name) is not None
    }
    if kwargs.get("backbone_path"):
        overrides["model_name"] = kwargs["backbone_path"]
    cfg = GR00TN17Config(**overrides)
    with torch.device("meta"):
        model = GR00TN17(cfg, load_backbone_weights=False)
    return model.estimate_flops(n_images=kwargs.get("n_images", 1), text_tokens=text_tokens)


def dreamzero_flops(**kwargs) -> dict[str, float]:
    """Build DreamZero's head on the meta device and ask it what a TRAINING sample costs.

    The head rather than the policy, because `DreamZeroPolicy.__init__` validates dataset
    features and so cannot be built from a bare config -- same reason `pi05_flops` builds
    `PI05Pytorch` and not the policy.

    `training_window()` supplies the sample's shapes and `estimate_flops` runs the training
    forward under hooks, so neither this file nor that one holds a width: change the canvas, the
    window length or `max_chunk_size` and the count follows. This costs ~20 s, almost all of it
    constructing 23 B meta parameters.

    Scope: the TRAINING path. Inference is a different shape entirely (a 2-frame block against a
    KV cache, 16 steps, CFG doubling it) and is not what this returns.
    """
    import contextlib
    import io

    import torch

    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
    from lerobot.policies.dreamzero.wan.action_head import WANPolicyHead

    fields = ("training_mode", "gradient_checkpointing", "num_frames", "action_horizon")
    cfg = DreamZeroConfig(**{k: kwargs[k] for k in fields if kwargs.get(k) is not None})
    # The head builds a `Beta` noise distribution in its constructor, and torch validates a
    # distribution's parameters with `.item()`, which a meta tensor cannot answer. The values are
    # config constants, so there is nothing for the check to catch here.
    validate = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    # Both of these print a per-parameter trainability dump, which would land in --json output.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with torch.device("meta"):
                head = WANPolicyHead(cfg.build_action_head_config())
            # `WANPolicyHead.__init__` already applied tune_projector/tune_diffusion_model and,
            # under `lora`, injected the adapters -- calling `set_trainable_parameters` again here
            # would wrap the model with PEFT a second time and double-count the adapters.
            #
            # Gradient checkpointing does need applying: the DiT hardcodes it on and the head's
            # config field of the same name is never read, so `DreamZeroPolicy.__init__` sets it
            # from the LeRobot config and this has to do the same. It goes on the DiT the forward
            # actually reads it off, which under `lora` is wrapped by a PeftModel.
            head.dit().gradient_checkpointing = cfg.gradient_checkpointing
    finally:
        torch.distributions.Distribution.set_default_validate_args(validate)
    if kwargs.get("untiled_vae"):
        head.tiled = False
    return head.estimate_flops(**cfg.training_window())


POLICIES = {"pi05": pi05_flops, "groot": groot_flops, "dreamzero": dreamzero_flops}


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
    ap.add_argument("--action-horizon", type=int, help="groot: action chunk length")
    ap.add_argument("--select-layer", type=int, help="groot: LLM layers kept")
    ap.add_argument("--tune-llm", action="store_true", default=None, help="groot: unfreeze the LLM")
    ap.add_argument("--tune-visual", action="store_true", default=None, help="groot: unfreeze the ViT")
    ap.add_argument("--backbone-path", help="groot: local copy of the (gated) Cosmos-Reason2-2B config")
    ap.add_argument("--training-mode", help="dreamzero: 'full' or 'lora'")
    ap.add_argument("--num-frames", type=int, help="dreamzero: raw frames in the training window")
    ap.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        default=None,
        help="dreamzero: cost the run as if the DiT kept its activations (no recompute)",
    )
    ap.add_argument(
        "--untiled-vae", action="store_true", help="dreamzero: cost the VAE encode without tiling"
    )
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
        action_horizon=args.action_horizon,
        select_layer=args.select_layer,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        backbone_path=args.backbone_path,
        training_mode=args.training_mode,
        num_frames=args.num_frames,
        gradient_checkpointing=args.gradient_checkpointing,
        untiled_vae=args.untiled_vae,
    )
    per_sample = parts.pop("total")
    per_step = per_sample * args.batch
    # Work the hardware does that the model does not require, so it belongs to HFU and not to
    # MFU. Only a policy that recomputes reports it; the others leave the key out entirely.
    recompute_per_step = parts.pop("recompute", 0.0) * args.batch

    if args.json:
        out = {
            "policy": args.policy,
            "tflop_per_sample": per_sample / TFLOP,
            "tflop_per_step": per_step / TFLOP,
            "parts_tflop": {k: v / TFLOP for k, v in parts.items()},
        }
        if recompute_per_step:
            out["recompute_tflop_per_step"] = recompute_per_step / TFLOP
        if args.step_s:
            out["peak_tflops"] = peak
            out["achieved_tflops"] = per_step / args.step_s / TFLOP
            out["mfu_pct"] = 100.0 * (per_step / args.step_s) / (peak * TFLOP)
            if recompute_per_step:
                out["hfu_pct"] = 100.0 * ((per_step + recompute_per_step) / args.step_s) / (peak * TFLOP)
        print(json.dumps(out, indent=2))
        return

    print(f"{args.policy}: model FLOPs by architecture (fwd+bwd, matmuls only)")
    for name, v in parts.items():
        print(f"  {name:<28} {v / TFLOP:9.3f} TFLOP")
    print(f"  {'per sample':<28} {per_sample / TFLOP:9.3f} TFLOP")
    print(f"  {'per step (batch ' + str(args.batch) + ')':<28} {per_step / TFLOP:9.3f} TFLOP")
    if recompute_per_step:
        print(f"  {'+ recompute (HFU only)':<28} {recompute_per_step / TFLOP:9.3f} TFLOP")
    if args.step_s:
        achieved = per_step / args.step_s
        print(f"  {'step time':<28} {args.step_s:9.3f} s")
        print(f"  {'throughput':<28} {args.batch * args.ranks / args.step_s:9.2f} samples/s global")
        print(f"  {'achieved':<28} {achieved / TFLOP:9.1f} TFLOP/s per card")
        print(f"  {'MFU':<28} {100.0 * achieved / (peak * TFLOP):9.1f} %  of {peak} TFLOP/s")
        if recompute_per_step:
            hardware = (per_step + recompute_per_step) / args.step_s
            print(f"  {'HFU':<28} {100.0 * hardware / (peak * TFLOP):9.1f} %  (with recompute)")
            print(
                f"  {'':<28} {'':>9}  the gap is what gradient checkpointing costs: "
                f"{100.0 * recompute_per_step / (per_step + recompute_per_step):.0f}% of the work run"
            )
    else:
        print("  (pass --step-s to turn this into MFU)")


if __name__ == "__main__":
    main()
