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

"""Read an MFU / overlap summary out of an Ascend profile directory, after the fact.

`lerobot-train` prints this summary itself when LEROBOT_PROF=1, but only if the profiler
has finished flushing its CSVs by the time training exits -- otherwise it tells you to
come back with this script. It is also the way to re-read a profile you still have from a
run that is long gone.

    python benchmarks/prof_flops.py ./prof --batch-size 32 --step-s 0.858

`--step-s` is what turns FLOPs into MFU, and it has to come from OUTSIDE the profiled
window: the profiler slows down the steps it captures, so a step time measured inside it
understates the real throughput. Take it from the train log's steady-state steps. Without
it, this prints FLOPs and the compute/communication split and says MFU is unavailable
rather than quoting a number it cannot stand behind.

The per-card peak cannot be detected, so it defaults to the 950PR bf16 value recorded in
`profiler_summary.DEFAULT_PEAK_TFLOPS`. On any other card, or any other dtype, pass
`--peak-tflops` -- `benchmarks/mfu.py --measure-peak` will measure it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.utils.profiler_summary import DEFAULT_PEAK_TFLOPS, summarise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prof_dir", type=Path, help="LEROBOT_PROF_DIR of the run")
    ap.add_argument("--batch-size", type=int, default=1, help="per-rank batch size used")
    ap.add_argument("--num-processes", type=int, default=1, help="world size of the run")
    ap.add_argument(
        "--step-s", type=float, help="steady-state seconds per step, measured outside the profiled window"
    )
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    args = ap.parse_args()

    if not args.prof_dir.exists():
        raise SystemExit(f"no such directory: {args.prof_dir}")

    lines = summarise(
        args.prof_dir,
        step_s=args.step_s,
        step_s_label="passed on the command line",
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        peak_tflops=args.peak_tflops,
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
