#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Aggregate validated per-rank Pi0.5 training benchmark results."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for dp in (1, 2, 4, 8, 32):
        paths = sorted(
            (
                path
                for path in args.root.glob(f"sdpa-dp{dp}-a*/rank0.json")
                if (args.root / (path.parent.name + ".exit")).is_file()
                and (args.root / (path.parent.name + ".exit")).read_text().strip() == "0"
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if not paths:
            continue
        selected = paths[-1].parent
        exit_path = args.root / (selected.name + ".exit")
        if not exit_path.is_file() or exit_path.read_text().strip() != "0":
            continue
        ranks = [json.loads((selected / f"rank{rank}.json").read_text()) for rank in range(dp)]
        assert all(row["all_parameters_synchronized"] for row in ranks)
        assert len({row["parameter_sha256"] for row in ranks}) == 1
        first = ranks[0]
        rows.append(
            {
                "dp": dp,
                "global_batch": first["global_batch"],
                "step_seconds": first["max_rank_median_step_seconds"],
                "samples_per_second": first["samples_per_second"],
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in ranks),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in ranks),
                "all_parameters_synchronized": True,
                "run": selected.name,
            }
        )
    if not rows or rows[0]["dp"] != 1:
        raise RuntimeError("A validated DP=1 baseline is required")
    baseline = rows[0]["samples_per_second"]
    for row in rows:
        row["throughput_speedup"] = row["samples_per_second"] / baseline
        row["parallel_efficiency"] = row["throughput_speedup"] / row["dp"]
    (args.root / "dp-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("| DP | Global batch | Step s | Samples/s | Speedup | Efficiency |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row['dp']} | {row['global_batch']} | {row['step_seconds']:.3f} | "
            f"{row['samples_per_second']:.3f} | {row['throughput_speedup']:.2f}x | "
            f"{row['parallel_efficiency']:.1%} |"
        )


if __name__ == "__main__":
    main()
