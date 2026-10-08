#!/usr/bin/env python3
"""Compute sustained training throughput from a lerobot training log.

Usage:
    python3 throughput.py TRAIN.log [--nw 4] [--eff-bs 512] [--window 20]
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics as st
import sys
from dataclasses import dataclass

STEP = re.compile(
    r"step:(\d+).*?(?:epch:([\d.]+).*?)?updt_s:([\d.]+)\s+data_s:([\d.]+)"
    r"(?:\s+step_s:([\d.]+))?\s+smp/s:(\d+)"
)
TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TOTAL_STEPS = re.compile(r"cfg\.steps=(\d+)")
EFF_BS = re.compile(r"effective batch size:\s*(\d+)")
NUM_WORKERS = re.compile(r"'?num_workers'?[:=]\s*(\d+)")


@dataclass(frozen=True)
class Step:
    n: int
    epoch_frac: float | None
    updt_s: float
    data_s: float
    logged_rate: int
    ts: dt.datetime | None = None
    # Whole step timed on one rank before reduction (lerobot >= the step_s fix). This is the
    # only high-resolution quantity that is a true step duration; prefer it over everything.
    step_s: float | None = None

    @property
    def iter_s(self) -> float:
        """updt_s + data_s. NOT a step duration -- see `sustained`."""
        return self.updt_s + self.data_s


def parse(path: str) -> tuple[list[Step], int | None, int | None]:
    steps: list[Step] = []
    eff_bs: int | None = None
    nw: int | None = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if eff_bs is None and (m := EFF_BS.search(line)):
                eff_bs = int(m.group(1))
            if nw is None and (m := NUM_WORKERS.search(line)):
                nw = int(m.group(1))
            if m := STEP.search(line):
                tm = TS.search(line)
                steps.append(
                    Step(
                        n=int(m.group(1)),
                        epoch_frac=float(m.group(2)) if m.group(2) else None,
                        updt_s=float(m.group(3)),
                        data_s=float(m.group(4)),
                        step_s=float(m.group(5)) if m.group(5) else None,
                        logged_rate=int(m.group(6)),
                        ts=dt.datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S") if tm else None,
                    )
                )
    steps.sort(key=lambda s: s.n)
    # Step 1 carries one-off kernel compilation and allocator growth (measured updt_s
    # ~11s vs ~2s steady) and is never representative; drop it from every aggregate.
    steps = [s for s in steps if s.n > 1]
    return steps, eff_bs, nw


def sustained(steps: list[Step], eff_bs: int) -> float:
    """Samples per second from WALL CLOCK between the window's first and last step.

    Do NOT sum updt_s + data_s for this. Both are reduced with max across ranks
    (AverageMeter(reduction="max")), and a straggler's wait lands in BOTH: in its own data_s,
    and again in every other rank's all-reduce wait inside updt_s. The two maxima come from
    different ranks, so the sum double-counts the stall -- by 6% when the pipeline is keeping
    up and by 35% when it is not, which is exactly when the number matters. Wall clock has no
    such artifact. `sum_of_phases` below reports the old figure so the gap stays visible.
    """
    have = [x.step_s for x in steps if x.step_s is not None]
    if len(have) == len(steps) and have:
        # Exact: each value is one rank's whole-step elapsed time, max-reduced.
        return eff_bs * len(have) / sum(have)
    stamped = [x for x in steps if x.ts is not None]
    if len(stamped) >= 2:
        secs = (stamped[-1].ts - stamped[0].ts).total_seconds()
        n = len(stamped) - 1
        if secs > 0 and n > 0:
            return eff_bs * n / secs
    return float("nan")


def step_series(steps: list[Step]) -> tuple[list[float], str]:
    """Per-step durations for the variance checks, and a label saying how trustworthy they are.

    step_s is exact. Falling back to iter_s (updt_s + data_s) inflates every stalled step,
    so variance computed from it overstates -- that must be said, not silently reported.
    Per-step wall-clock deltas are useless here: the log stamps whole seconds, which is half
    a step.
    """
    have = [x.step_s for x in steps if x.step_s is not None]
    if len(have) == len(steps) and have:
        return have, "step_s (exact)"
    return [x.iter_s for x in steps], "updt_s+data_s (INFLATED on stalled steps)"


def sum_of_phases(steps: list[Step], eff_bs: int) -> float:
    """The old, biased estimate: eff_bs / mean(max(updt_s) + max(data_s)). Diagnostic only."""
    total_time = sum(s.iter_s for s in steps)
    return eff_bs * len(steps) / total_time if total_time > 0 else float("nan")


def epoch_bounds(steps: list[Step]) -> list[tuple[int, int, int]]:
    """Group steps into epochs using the log's own epch: field. (epoch, first, last)."""
    if steps[0].epoch_frac is None:
        return []
    out, cur, start = [], 0, steps[0].n
    for s in steps:
        e = int(s.epoch_frac or 0)
        if e > cur:
            out.append((cur + 1, start, s.n - 1))
            cur, start = e, s.n
    out.append((cur + 1, start, steps[-1].n))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--eff-bs", type=int, help="effective batch size (bs x ranks); auto-detected if the log states it")
    ap.add_argument("--nw", type=int, help="num_workers, for the stall-period check")
    ap.add_argument("--window", type=int, default=20, help="block size for the stability check (default 20)")
    args = ap.parse_args()

    steps, eff_bs, nw = parse(args.log)
    if not steps:
        print(f"{args.log}: no step lines found", file=sys.stderr)
        return 1
    eff_bs = args.eff_bs or eff_bs
    nw = args.nw or nw
    if eff_bs is None:
        print("Could not determine effective batch size; pass --eff-bs (batch_size x num_ranks).", file=sys.stderr)
        return 2

    print(f"log            : {args.log}")
    print(f"steps parsed   : {len(steps)}  (step {steps[0].n}..{steps[-1].n})")
    with open(args.log, errors="replace") as _fh:
        want = TOTAL_STEPS.search(_fh.read(400_000))
    if want and steps[-1].n < int(want.group(1)):
        print(f"  WARNING: the config asked for {want.group(1)} steps but the log stops at "
              f"{steps[-1].n} -- this run did not finish, so the window below is whatever it "
              f"reached, not what was intended.")
    if all(x.step_s is None for x in steps):
        print("  note: no step_s field (pre-fix lerobot). The rate comes from 1-second log")
        print("        timestamps, so a window needs >=100 steps for better than 1% resolution,")
        print("        and per-step variance falls back to the inflated updt_s+data_s.")
    print(f"effective bs   : {eff_bs}" + ("  [auto-detected]" if not args.eff_bs else ""))
    if nw:
        print(f"num_workers    : {nw}" + ("  [auto-detected]" if not args.nw else ""))

    epochs = epoch_bounds(steps)
    if epochs:
        print("\nper-epoch sustained throughput (total samples / total time):")
        lengths = [hi - lo + 1 for _, lo, hi in epochs]
        full_len = st.median(lengths) if lengths else 0
        for e, lo, hi in epochs:
            g = [s for s in steps if lo <= s.n <= hi]
            if len(g) < 3:
                continue
            partial = "   <- partial epoch, too few steps to trust" if len(g) < 0.8 * full_len else ""
            print(f"  epoch {e}  steps {lo:>4}-{hi:<4} n={len(g):<4} {sustained(g, eff_bs):7.1f} smp/s{partial}")
    else:
        print("\n(no epch: field in the log -- cannot split by epoch; reporting whole run)")

    # Headline: epoch 2 onward, skipping a few steps after the boundary.
    if len(epochs) >= 2:
        start = epochs[1][1] + 6
        label = f"steady state (epoch 2 onward, from step {start})"
    else:
        start = steps[0].n + max(10, len(steps) // 5)
        label = f"whole run from step {start} -- RUN NEVER REACHED EPOCH 2, so this still includes the prefetch-buffer transient and is an OVERESTIMATE"
        # Tell the caller how long the run needs to be. Measured requirement: the
        # estimate settles to within ~2% once the measurement window is about one
        # full epoch (bs32 needs ~100 steps, bs16 ~200 -- bs16 carries low-frequency
        # drift that short windows do not average out). 2.2 epochs delivers that.
        if steps[-1].epoch_frac:
            epoch_len = round(steps[-1].n / steps[-1].epoch_frac)
            want = round(2.2 * epoch_len)
            label += (f"\n  epoch length here is ~{epoch_len} steps; rerun with"
                      f" STEPS={want} (2.2 epochs) to measure steady state")
    w = [s for s in steps if s.n >= start]
    if len(w) < 10:
        print("\nnot enough steps after warmup to report a steady state", file=sys.stderr)
        return 3

    sus = sustained(w, eff_bs)
    stamped = [x for x in w if x.ts is not None]
    print(f"\n{label}")
    print(f"  n steps            : {len(w)}")
    if stamped and len(stamped) >= 2:
        secs = (stamped[-1].ts - stamped[0].ts).total_seconds()
        print(f"  wall clock window  : steps {stamped[0].n}-{stamped[-1].n}, {secs:.0f} s, "
              f"{secs / (len(stamped) - 1):.3f} s/step")
        print(f"  SUSTAINED          : {sus:.1f} smp/s      <-- report this (wall clock)")
    else:
        print("  SUSTAINED          : UNAVAILABLE -- this log has no per-step timestamps, so the")
        print("                       rate cannot be measured without double-counting stalls.")
    print(f"  updt_s median      : {st.median([s.updt_s for s in w]):.3f}   (max across ranks)")
    print(f"  data_s median      : {st.median([s.data_s for s in w]):.3f}   (max across ranks)")

    sop = sum_of_phases(w, eff_bs)
    med = st.median([s.logged_rate for s in w])
    mean = st.mean([s.logged_rate for s in w])
    print("\n  for contrast, aggregations that are biased and must not be reported:")
    if sus == sus:  # not NaN
        print(f"    eff_bs / mean(updt_s+data_s) : {sop:7.1f}  ({100 * (sop / sus - 1):+.0f}% vs wall clock)")
        print(f"    median of per-step rates     : {med:7.1f}  ({100 * (med / sus - 1):+.0f}%)")
        print(f"    mean   of per-step rates     : {mean:7.1f}  ({100 * (mean / sus - 1):+.0f}%)")
        if sop < sus * 0.95:
            print("    -> updt_s and data_s are BOTH max-reduced, so a straggler's wait is counted")
            print(f"       twice; the {100 * (1 - sop / sus):.0f}% gap is that artifact, not the system.")
    else:
        print(f"    eff_bs / mean(updt_s+data_s) : {sop:7.1f}")
        print(f"    median / mean of per-step rates : {med:7.1f} / {mean:7.1f}")

    it, series_label = step_series(w)
    print(f"\n  variance measured on {series_label}")
    win = args.window
    if nw and win % nw:
        win = nw * max(1, round(args.window / nw))
        print(f"\n  note: block window rounded {args.window} -> {win} so it is a multiple of num_workers={nw};")
        print("        a misaligned window reports window misalignment, not system variance.")
    blocks = [it[i : i + win] for i in range(0, len(it) - win + 1, win)]
    if len(blocks) >= 3:
        rates = [eff_bs * win / sum(b) for b in blocks]
        print(f"\n  per-step cv           : {100 * st.pstdev(it) / st.mean(it):5.1f}%")
        print(f"  {win}-step block cv      : {100 * st.pstdev(rates) / st.mean(rates):5.1f}%  (n={len(blocks)} blocks)")

    if nw:
        buckets: dict[int, list[float]] = {}
        for s_, dur in zip(w, it, strict=True):
            buckets.setdefault(s_.n % nw, []).append(dur)
        meds = {k: st.median(v) for k, v in sorted(buckets.items()) if len(v) >= 3}
        if len(meds) > 1:
            spread = max(meds.values()) / min(meds.values())
            print(f"\n  step duration by (step mod num_workers={nw}): "
                  + "  ".join(f"{k}:{v:.2f}" for k, v in meds.items()))
            verdict = ("periodic dataloader stall present -- the dataloader is the bottleneck"
                       if spread > 1.3 else "no periodic stall -- dataloading is keeping up")
            print(f"  spread {spread:.2f}x  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
