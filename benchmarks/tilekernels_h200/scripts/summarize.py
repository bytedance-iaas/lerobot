"""Summarize the interleaved H200 training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    report: dict[str, object] = {
        "batch_per_gpu": 16,
        "world_size": 2,
        "steps": 60,
        "warmup_steps": 20,
        "tilekernels_revision": "66258df6175d2f630ffecb04c5ab66bff8a2ae6a",
        "ops_by_model": {
            "groot": ["rms", "swiglu", "rope"],
            "pi05": ["rms", "rope"],
        },
        "unavailable_in_tilekernels": [
            "attention",
            "gemm_linear",
            "gelu_geglu",
            "adamw",
        ],
        "models": {},
    }

    for model in ("groot", "pi05"):
        runs: dict[str, object] = {}
        for mode in ("baseline", "full"):
            group = []
            for path in sorted((root / "results").glob(f"{model}-{mode}-run[0-9]*")):
                if not (path / "exit-code").exists() or (path / "exit-code").read_text().strip() != "0":
                    continue
                if not all((path / f"summary-rank{rank}.json").exists() for rank in (0, 1)):
                    continue
                summaries = [json.loads((path / f"summary-rank{rank}.json").read_text()) for rank in (0, 1)]
                assert all(
                    summary["steps"] == 60 and summary["measured_steps"] == 40 for summary in summaries
                )
                iteration = max(summary["mean_iteration_s"] for summary in summaries)
                update = max(summary["mean_update_s"] for summary in summaries)
                row = {
                    "run": path.name,
                    "iteration_ms": iteration * 1000,
                    "update_ms": update * 1000,
                    "samples_per_second": 32 / iteration,
                    "peak_allocated_gib": max(summary["peak_allocated_gib"] for summary in summaries),
                    "peak_reserved_gib": max(summary["peak_reserved_gib"] for summary in summaries),
                }
                steps = []
                for rank in (0, 1):
                    with (path / f"steps-rank{rank}.jsonl").open() as stream:
                        steps.append([json.loads(line) for line in stream][20:])
                count = min(map(len, steps))
                iteration_steps = [
                    max(
                        steps[0][index]["iteration_interval_s"],
                        steps[1][index]["iteration_interval_s"],
                    )
                    * 1000
                    for index in range(count)
                ]
                update_steps = [
                    max(
                        steps[0][index]["update_wall_s"],
                        steps[1][index]["update_wall_s"],
                    )
                    * 1000
                    for index in range(count)
                ]
                row.update(
                    {
                        "iteration_median_ms": statistics.median(iteration_steps),
                        "iteration_p90_ms": percentile(iteration_steps, 0.9),
                        "iteration_max_ms": max(iteration_steps),
                        "update_median_ms": statistics.median(update_steps),
                        "update_p90_ms": percentile(update_steps, 0.9),
                        "update_max_ms": max(update_steps),
                    }
                )
                with (path / "gpu.csv").open() as stream:
                    reader = csv.reader(stream)
                    next(reader)
                    memory = [
                        float(csv_row[2].split()[0]) / 1024
                        for csv_row in reader
                        if len(csv_row) >= 5 and int(csv_row[1]) in (0, 1)
                    ]
                row["sampled_device_peak_gib"] = max(memory) if memory else None
                group.append(row)

            mode_report: dict[str, object] = {"runs": group}
            if group:
                for key in (
                    "iteration_ms",
                    "update_ms",
                    "samples_per_second",
                    "iteration_median_ms",
                    "iteration_p90_ms",
                    "update_median_ms",
                    "update_p90_ms",
                ):
                    values = [run[key] for run in group]
                    mode_report["mean_" + key] = statistics.mean(values)
                    mode_report["range_" + key] = [min(values), max(values)]
                mode_report["peak_allocated_gib"] = max(run["peak_allocated_gib"] for run in group)
                mode_report["peak_sampled_device_gib"] = max(run["sampled_device_peak_gib"] for run in group)
            runs[mode] = mode_report

        baseline = runs["baseline"]
        full = runs["full"]
        if baseline["runs"] and full["runs"]:
            runs["throughput_change_percent"] = 100 * (
                full["mean_samples_per_second"] / baseline["mean_samples_per_second"] - 1
            )
            runs["iteration_time_change_percent"] = 100 * (
                full["mean_iteration_ms"] / baseline["mean_iteration_ms"] - 1
            )
            runs["median_iteration_time_change_percent"] = 100 * (
                full["mean_iteration_median_ms"] / baseline["mean_iteration_median_ms"] - 1
            )
            runs["median_based_throughput_change_percent"] = 100 * (
                baseline["mean_iteration_median_ms"] / full["mean_iteration_median_ms"] - 1
            )
        paired_paths = sorted(
            (root / "results" / f"{model}-full-runvalidate").glob("paired-check-rank*.json")
        )
        runs["paired_checks"] = [json.loads(path.read_text()) for path in paired_paths]
        report["models"][model] = runs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
