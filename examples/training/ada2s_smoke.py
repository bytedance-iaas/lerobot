#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Isolate Ada2S FP32 vector, FP32 non-vector and BF16 compute paths.

Run with the interpreter containing the vendor's torch + torch_ada installation:
    python3 examples/training/ada2s_smoke.py --card 0

Each case runs in its own process with a timeout. CPU mode checks the harness only:
    uv run --no-sync python examples/training/ada2s_smoke.py --backend cpu

Pass --training to additionally check backward and an AdamW update.
"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def mark(message: str) -> None:
    print(message, flush=True)


def probe(backend: str, dtype_name: str, size: int, training: bool) -> None:
    mark("START import")
    import torch

    if backend == "ada":
        import torch_ada  # noqa: F401

    mark(f"PASS import: torch={torch.__version__}, backend={backend}, dtype={dtype_name}")
    dtype = getattr(torch, dtype_name)
    device = "ada:0" if backend == "ada" else "cpu"
    reference = (torch.arange(size * size, dtype=torch.float32).reshape(size, size) % 8 - 4).to(dtype)

    mark("START host_to_device + device_to_host")
    x = reference.clone().to(device).requires_grad_(training)
    torch.testing.assert_close(x.detach().cpu(), reference)
    mark("PASS host_to_device + device_to_host")

    mark("START add + device_to_host")
    torch.testing.assert_close((x + x).detach().cpu(), reference + reference)
    mark("PASS add + device_to_host")

    mark("START multiply + device_to_host")
    torch.testing.assert_close((x * x).detach().cpu(), reference * reference)
    mark("PASS multiply + device_to_host")

    mark("START matmul + device_to_host")
    y = x @ x
    torch.testing.assert_close(y.detach().cpu(), (reference.float() @ reference.float()).to(dtype))
    mark("PASS matmul + device_to_host")

    if not training:
        return

    mark("START backward + device_to_host")
    loss = y.float().square().mean()
    loss.backward()
    expected = reference.float().detach().clone().requires_grad_(True)
    (expected @ expected).square().mean().backward()
    torch.testing.assert_close(x.grad.cpu().float(), expected.grad, rtol=0.03, atol=0.03)
    mark(f"PASS backward: loss={loss.detach().cpu().item()}")

    mark("START AdamW + device_to_host")
    optimizer = torch.optim.AdamW([x], lr=1e-2, foreach=False, fused=False)
    optimizer.step()
    updated = x.detach().cpu()
    assert torch.isfinite(updated).all(), "optimizer produced non-finite parameters"
    assert not torch.equal(updated, reference), "optimizer did not update parameters"
    mark("PASS AdamW")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ada", "cpu"), default="ada")
    parser.add_argument("--card", type=int, default=0, help="Physical card; exposed as ada:0 in each child")
    parser.add_argument(
        "--size", type=int, default=4, help="Square matrix size; 4 matches the reported failure"
    )
    parser.add_argument(
        "--timeout", type=float, default=60, help="Maximum seconds per case, including imports"
    )
    parser.add_argument("--case", choices=("fp32-vector", "fp32-non-vector", "bf16"))
    parser.add_argument(
        "--training", action="store_true", help="Also check backward and AdamW after computation"
    )
    parser.add_argument("--probe-dtype", choices=("float32", "bfloat16"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.card < 0 or args.timeout <= 0 or args.size <= 0:
        parser.error("--card must be non-negative; --timeout and --size must be positive")
    if args.probe_dtype:
        probe(args.backend, args.probe_dtype, args.size, args.training)
        return 0

    cases = [("fp32-vector", "float32", "1"), ("fp32-non-vector", "float32", "0"), ("bf16", "bfloat16", "0")]
    failed = False
    for name, dtype, vector in cases:
        if args.case and args.case != name:
            continue
        env = os.environ.copy()
        if args.backend == "ada":
            env.update(
                ADAX_VISIBLE_DEVICES=str(args.card),
                ADAX_RT_USE_TSIM_MODE="0",
                TRITON_ADA_ARCH="s",
                ADAX_LAUNCH_BLOCKING="1",
                TORCH_ADA_OP_SKIP_RUN="0",
                TORCH_ADA_ALLOW_FP32_BY_VECTOR=vector,
                PYTORCH_DISABLE_ADA_FALLBACK="1",
                PYTORCH_DISABLE_ADA_FAIL_FALLBACK="1",
            )
        mark(
            f"CASE {name}: backend={args.backend}, physical_card={args.card}, size={args.size}, timeout={args.timeout}s"
        )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--backend",
            args.backend,
            "--probe-dtype",
            dtype,
            "--size",
            str(args.size),
        ]
        if args.training:
            command.append("--training")
        proc = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            mark(f"FAIL {name}: timeout; last START without PASS identifies the blocked stage")
            failed = True
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        else:
            mark(f"{'PASS' if code == 0 else 'FAIL'} {name}: exit={code}")
            failed |= code != 0
    if args.backend == "cpu":
        mark("CPU mode validates the harness only; Ada2S has not been tested.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
