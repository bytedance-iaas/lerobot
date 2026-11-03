#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Measure single-card Ada2S or CUDA BF16 GEMM with synchronized wall timing.

FMA counts as two floating-point operations. Each measurement excludes allocation
and warmup, preallocates the output, and synchronizes once per repeated GEMM block.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--device", choices=("ada", "cuda"), default="ada")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(args.repeats, args.warmup, args.trials) <= 0:
        parser.error("repeats, warmup, and trials must be positive")
    import torch

    if args.device == "ada":
        import torch_ada  # noqa: F401

    accelerator = getattr(torch, args.device)
    accelerator.set_device(args.card)
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    device = torch.device(f"{args.device}:{args.card}")
    # Check correctness on a small nontrivial matrix before benchmarking.
    a = torch.randn(128, 128, dtype=torch.bfloat16)
    b = torch.randn(128, 128, dtype=torch.bfloat16)
    expected = a.float() @ b.float()
    actual = (a.to(device) @ b.to(device)).cpu().float()
    error = (actual - expected).norm().item() / expected.norm().item()
    if not math.isfinite(error) or error > 0.02:
        raise RuntimeError(f"BF16 GEMM correctness check failed: relative_l2={error}")
    cases = [
        ("square-2048", 2048, 2048, 2048),
        ("square-4096", 4096, 4096, 4096),
        ("square-8192", 8192, 8192, 8192),
        ("pi05-q-projection", 506, 2048, 2048),
        ("pi05-mlp-up", 506, 16384, 2048),
        ("pi05-mlp-down", 506, 2048, 16384),
        ("pi05-action-mlp-up", 50, 4096, 1024),
    ]
    rows = []
    with torch.inference_mode():
        for label, m, n, k in cases:
            left = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            right = torch.randn(k, n, dtype=torch.bfloat16, device=device)
            output = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            for _ in range(args.warmup):
                torch.mm(left, right, out=output)
            accelerator.synchronize()
            seconds = []
            for _ in range(args.trials):
                start = time.perf_counter()
                for _ in range(args.repeats):
                    torch.mm(left, right, out=output)
                accelerator.synchronize()
                seconds.append((time.perf_counter() - start) / args.repeats)
            median = statistics.median(seconds)
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"Nonfinite output in {label}")
            row = {
                "label": label,
                "m": m,
                "n": n,
                "k": k,
                "median_seconds_per_gemm": median,
                "tflops": 2 * m * n * k / median / 1e12,
                "trial_seconds_per_gemm": seconds,
            }
            rows.append(row)
            print("GEMM_RESULT " + json.dumps(row), flush=True)
            del left, right, output
    report = {
        "device": str(device),
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "warmup_gemms": args.warmup,
        "repeated_gemms_per_trial": args.repeats,
        "trials": args.trials,
        "counting": "2*M*N*K; FMA=2 FLOPs; dense GEMM",
        "correctness_relative_l2": error,
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "ADAX_RT_USE_TSIM_MODE",
                "TRITON_ADA_ARCH",
                "TORCH_ADA_OP_SKIP_RUN",
                "PYTORCH_DISABLE_ADA_FALLBACK",
                "PYTORCH_DISABLE_ADA_FAIL_FALLBACK",
            )
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GEMM_BENCHMARK_PASS", flush=True)


if __name__ == "__main__":
    main()
