#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run sequential H20 DP=1/2/4 benchmarks with the prepared LeRobot image."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/pi05-h20-20261102"))
    parser.add_argument("--cpu-cores", default=",".join(str(cpu) for cpu in range(90, 122, 2)))
    parser.add_argument("--attempt", default="a1")
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    source = root / "lerobot"
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=str(source / "src"),
    )
    for dp in (1, 2, 4):
        name = f"sdpa-dp{dp}-{args.attempt}"
        output = results / name
        if output.exists():
            raise RuntimeError(f"Output already exists: {output}")
        env = dict(environment, CUDA_VISIBLE_DEVICES=",".join(map(str, range(dp))))
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={dp}",
            str(source / "benchmarks/pi05_ada_dp.py"),
            "--assets",
            str(root / "assets"),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--cpu-cores",
            args.cpu_cores,
            "--attention",
            "sdpa",
            "--batch-size",
            "1",
            "--warmup",
            "5",
            "--steps",
            "20",
        ]
        recorded_env = {
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "HF_HUB_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTHONPATH",
            )
        }
        (results / f"{name}-command.json").write_text(
            json.dumps({"argv": command, "env": recorded_env}, indent=2) + "\n"
        )
        print(f"DP {dp} START", flush=True)
        with (results / f"{name}.log").open("w") as log:
            run = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        (results / f"{name}.exit").write_text(str(run.returncode) + "\n")
        print(f"DP {dp} EXIT {run.returncode}", flush=True)
        if run.returncode:
            raise RuntimeError(f"DP {dp} failed; inspect {name}.log")
    subprocess.run([sys.executable, str(source / "benchmarks/ada_dp_summary.py"), str(results)], check=True)
    print("H20_SERIES_PASS", flush=True)


if __name__ == "__main__":
    main()
