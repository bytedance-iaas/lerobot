#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compare validated Ada2S and H20 Pi0.5 DP throughput summaries."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ada", type=Path, required=True)
    parser.add_argument("--h20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ada = {row["dp"]: row for row in json.loads(args.ada.read_text())}
    h20 = {row["dp"]: row for row in json.loads(args.h20.read_text())}
    rows = []
    for dp in (1, 2, 4):
        left, right = ada[dp], h20[dp]
        if not left["all_parameters_synchronized"] or not right["all_parameters_synchronized"]:
            raise RuntimeError(f"DP={dp}: unvalidated result")
        if left["global_batch"] != dp or right["global_batch"] != dp:
            raise RuntimeError(f"DP={dp}: expected batch=1 per rank")
        rows.append(
            {
                "dp": dp,
                "global_batch": dp,
                "ada_step_seconds": left["step_seconds"],
                "h20_step_seconds": right["step_seconds"],
                "ada_samples_per_second": left["samples_per_second"],
                "h20_samples_per_second": right["samples_per_second"],
                "ada_over_h20_throughput": left["samples_per_second"] / right["samples_per_second"],
                "ada_throughput_speedup": left["throughput_speedup"],
                "h20_throughput_speedup": right["throughput_speedup"],
                "ada_parallel_efficiency": left["parallel_efficiency"],
                "h20_parallel_efficiency": right["parallel_efficiency"],
                "ada_peak_allocated_gib": left["peak_allocated_gib"],
                "h20_peak_allocated_gib": right["peak_allocated_gib"],
                "ada_run": left["run"],
                "h20_run": right["run"],
            }
        )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("| DP | Ada step s | H20 step s | Ada samples/s | H20 samples/s | Ada/H20 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row['dp']} | {row['ada_step_seconds']:.3f} | {row['h20_step_seconds']:.3f} | "
            f"{row['ada_samples_per_second']:.3f} | {row['h20_samples_per_second']:.3f} | "
            f"{row['ada_over_h20_throughput']:.2f}x |"
        )


if __name__ == "__main__":
    main()
