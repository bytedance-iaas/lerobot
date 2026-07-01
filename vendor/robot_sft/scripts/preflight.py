#!/usr/bin/env python3
"""Pre-flight smoke test before a long robot_sft run (lerobot edition).

A full run costs GPU-hours, and the classic failures (gated Hub model, dataset/feature
key mismatch, /dev/shm, bad CLI flag) only surface minutes in. This runs the SAME
`lerobot-train` command for just a few steps in a throwaway output dir, then classifies
the result. ~1-3 minutes to catch a bug that would otherwise waste a multi-hour run.

It MUTATES the plan command for the smoke run only: overrides --steps/--save_freq to tiny
values, --log_freq=1, and redirects --output_dir to a temp dir, so it never touches the
real run. It also samples peak GPU memory during the smoke run and, when there is headroom,
suggests a bigger --batch_size (re-run preflight after applying it to confirm the fit).

Usage:
    python preflight.py --session <session_dir> [--steps 2] [--timeout 900]
    python preflight.py --command "<launch cmd>" --output-dir <real_dir> [--steps 2]

Exit code 0 = looks good to launch; non-zero = problem (see printed classification).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_patterns  # noqa: E402

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")   # tqdm "1/2 ["
CUDA_RE = re.compile(r"CUDA_VISIBLE_DEVICES=([0-9,]+)")


def gpu_ids_from_cmd(cmd: str):
    m = CUDA_RE.search(cmd)
    return m.group(1) if m else None


def sample_gpu_mem(ids: str):
    """Return (max_used_mb, total_mb) across the given GPU ids right now, or (0, 0)."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", ids],
            capture_output=True, text=True, timeout=10)
        used, total = 0, 0
        for line in q.stdout.strip().splitlines():
            u, t = (int(x) for x in line.split(","))
            used, total = max(used, u), max(total, t)
        return used, total
    except Exception:  # noqa: BLE001
        return 0, 0


def suggest_batch(batch, gpus, peak_mb, total_mb, frac):
    """Conservatively scale --batch_size to use ~frac of GPU memory. Assuming memory is
    purely linear in batch UNDER-estimates true capacity (real runs have fixed overhead), so
    this is a safe lower bound — always re-run preflight at the new batch to confirm."""
    if not (batch and peak_mb and total_mb):
        return None
    factor = (frac * total_mb) / peak_mb
    new_b = int(batch * factor)
    if new_b <= batch:
        return None
    return {"batch_size": new_b, "global_batch_size": new_b * max(1, gpus),
            "from_batch_size": batch, "headroom_factor": round(factor, 2),
            "note": "estimate — set this batch, RE-RUN preflight to confirm it fits, then "
                    "consider scaling LR with batch and recomputing steps/save_freq."}


def _override_flag(cmd: str, flag: str, value) -> str:
    """Replace `--flag=X` / `--flag X` (draccus style) or append the flag if absent."""
    pat = re.compile(rf"--{re.escape(flag)}[= ]\S+")
    if pat.search(cmd):
        return pat.sub(f"--{flag}={value}", cmd)
    return cmd + f" --{flag}={value}"


def smoke_command(cmd: str, real_output_dir: str, tmp_dir: str, steps: int) -> str:
    """Force tiny steps + temp output dir for the smoke run only."""
    cmd = _override_flag(cmd, "steps", steps)
    cmd = _override_flag(cmd, "save_freq", steps)   # exercise a checkpoint save too
    cmd = _override_flag(cmd, "log_freq", 1)
    if real_output_dir and real_output_dir in cmd:
        cmd = cmd.replace(real_output_dir, tmp_dir)
    else:
        cmd = _override_flag(cmd, "output_dir", tmp_dir)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--command")
    ap.add_argument("--output-dir")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--mem-frac", type=float, default=0.85,
                    help="target fraction of GPU memory to fill when suggesting a bigger batch")
    args = ap.parse_args()

    plan = None
    if args.session:
        plan = json.load(open(os.path.join(args.session, "training_plan.json")))
        cmd, real_out = plan["launch_command"], plan["output_dir"]
    elif args.command:
        cmd, real_out = args.command, args.output_dir or ""
    else:
        ap.error("provide --session or --command")

    tmp_root = tempfile.mkdtemp(prefix="robot_sft_preflight_")
    # lerobot-train refuses a pre-existing output_dir; give it a fresh subdir.
    tmp_dir = os.path.join(tmp_root, "out")
    log_path = os.path.join(tmp_root, "preflight.log")
    smoke = smoke_command(cmd, real_out, tmp_dir, args.steps)
    print(f"[preflight] smoke run ({args.steps} steps) -> {tmp_root}")

    gpu_ids = gpu_ids_from_cmd(cmd)
    peak_mb, total_mb = 0, 0

    logf = open(log_path, "wb")
    proc = subprocess.Popen(["bash", "-lc", smoke], stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    start = time.time()
    timed_out = False
    while proc.poll() is None:
        if time.time() - start > args.timeout:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except ProcessLookupError:
                pass
            break
        if gpu_ids:  # track peak memory of the training GPUs during the smoke run
            u, t = sample_gpu_mem(gpu_ids)
            peak_mb, total_mb = max(peak_mb, u), max(total_mb, t)
        time.sleep(5)
    logf.close()

    text = open(log_path, "rb").read().decode("utf-8", "replace")
    cls = error_patterns.classify(text)
    steps_seen = [int(m.group(1)) for m in STEP_RE.finditer(text)]
    progressed = bool(steps_seen) and max(steps_seen) >= 1
    # a checkpoint save is part of the smoke test (save_freq == steps)
    saved_ckpt = os.path.isdir(os.path.join(tmp_dir, "checkpoints"))
    rc = proc.poll()

    verdict = {
        "ok": False, "rc": rc, "timed_out": timed_out,
        "steps_seen": max(steps_seen) if steps_seen else 0,
        "checkpoint_saved": saved_ckpt,
        "classification": cls, "smoke_output_dir": tmp_root, "log": log_path,
        "peak_gpu_mem_mb": peak_mb or None, "gpu_total_mb": total_mb or None,
    }
    # memory headroom -> suggest a bigger batch. Only when the smoke run actually
    # progressed and we have a plan to compare against.
    if plan and peak_mb and total_mb:
        sug = suggest_batch(plan.get("batch_size"), plan.get("gpus") or 1,
                            peak_mb, total_mb, args.mem_frac)
        if sug:
            verdict["batch_suggestion"] = sug
    if cls["category"] == "fatal":
        verdict["message"] = f"FATAL config problem before any real cost: {cls['reason']}. {cls['fix']}"
    elif cls["category"] == "oom":
        verdict["message"] = f"OOM at smoke scale: {cls['fix']}"
    elif progressed and (rc == 0 or timed_out):
        verdict["ok"] = True
        msg = f"Training stepped ({verdict['steps_seen']} step(s)) with no fatal signature"
        msg += " and saved a checkpoint" if saved_ckpt else " (no checkpoint dir seen)"
        msg += " — safe to launch."
        if peak_mb and total_mb:
            msg += f" GPU mem {peak_mb}/{total_mb} MB ({100*peak_mb//total_mb}%)."
            if verdict.get("batch_suggestion"):
                s = verdict["batch_suggestion"]
                msg += (f" Headroom: try --batch_size {s['from_batch_size']}→{s['batch_size']} "
                        f"and re-run preflight to confirm.")
        verdict["message"] = msg
    elif not progressed:
        verdict["message"] = ("No training step completed; model/data init likely failed. "
                              + (cls["fix"] or "Inspect the preflight log."))
    else:
        verdict["message"] = f"Exited rc={rc} with: {cls['reason'] or 'unknown'}. {cls['fix']}"

    print(json.dumps(verdict, indent=2))
    sys.exit(0 if verdict["ok"] else 1)


if __name__ == "__main__":
    main()
