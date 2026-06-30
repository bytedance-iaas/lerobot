#!/usr/bin/env python3
"""Preflight hardware check for robot_sft training.

Reports GPUs (and which are idle), /dev/shm size, and free disk on candidate volumes,
plus actionable warnings tied to the lessons in references/lessons_learned.md. Pure
stdlib + nvidia-smi; safe to run anywhere.

Usage:
    python check_hardware.py [--json] [--shm-min-gb 4] [--ckpt-gb 12] [--save-limit 5]
                             [--paths /data /tmp .]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception as e:  # noqa: BLE001
        return [{"error": f"nvidia-smi unavailable: {e}"}]
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mtot, mused, util = parts
        mtot_i, mused_i = int(mtot), int(mused)
        res.append({
            "index": int(idx), "name": name,
            "mem_total_mb": mtot_i, "mem_used_mb": mused_i,
            "util_pct": int(util),
            "idle": mused_i < 0.05 * mtot_i,  # <5% memory used ~ idle
        })
    return res


def shm_gb() -> float:
    try:
        st = shutil.disk_usage("/dev/shm")
        return round(st.total / 1024**3, 2)
    except Exception:  # noqa: BLE001
        return -1.0


def disk(paths: list[str]) -> list[dict]:
    res = []
    for p in paths:
        try:
            st = shutil.disk_usage(p)
            res.append({"path": p, "free_gb": round(st.free / 1024**3, 1),
                        "total_gb": round(st.total / 1024**3, 1),
                        "used_pct": round(100 * st.used / st.total, 1)})
        except Exception:  # noqa: BLE001
            continue
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--shm-min-gb", type=float, default=4.0)
    ap.add_argument("--ckpt-gb", type=float, default=12.0,
                    help="approx size of one checkpoint (GR00T-3B ~11-12 GB)")
    ap.add_argument("--save-limit", type=int, default=5)
    ap.add_argument("--paths", nargs="*", default=["/data", "/tmp", "."])
    args = ap.parse_args()

    g = gpus()
    shm = shm_gb()
    dk = disk(args.paths)
    idle = [x["index"] for x in g if x.get("idle")]
    headroom = args.ckpt_gb * args.save_limit

    warnings = []
    if any("error" in x for x in g):
        warnings.append("nvidia-smi not available — cannot confirm GPUs.")
    elif not idle:
        warnings.append("No idle GPU detected; training will contend with other jobs.")
    if 0 <= shm < args.shm_min_gb:
        warnings.append(
            f"/dev/shm is {shm} GB (< {args.shm_min_gb} GB). num_workers>0 may crash with a "
            f"Bus error. Try `mount -o remount,size=16g /dev/shm` (needs root+CAP_SYS_ADMIN; "
            f"re-check `df -h /dev/shm` after — some containers ignore it), else use "
            f"num_workers=0.")
    good_vols = [d for d in dk if d["free_gb"] >= headroom]
    if not good_vols:
        warnings.append(
            f"No candidate volume has >= {headroom:.0f} GB free for "
            f"{args.save_limit} checkpoints; checkpoint saving may fail. Free space or lower "
            f"save_total_limit.")

    report = {
        "gpus": g,
        "idle_gpus": idle,
        "recommended_cuda_visible_devices": ",".join(str(i) for i in idle) or None,
        "shm_gb": shm,
        "shm_ok_for_workers": shm >= args.shm_min_gb,
        "disk": dk,
        "checkpoint_headroom_gb_needed": headroom,
        "good_checkpoint_volumes": [d["path"] for d in good_vols],
        "warnings": warnings,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"GPUs: {len(g)}  idle: {idle or 'none'}")
        for x in g:
            if "error" in x:
                print("  " + x["error"]); continue
            tag = " (idle)" if x["idle"] else ""
            print(f"  [{x['index']}] {x['name']}  {x['mem_used_mb']}/{x['mem_total_mb']} MB"
                  f"  util {x['util_pct']}%{tag}")
        print(f"/dev/shm: {shm} GB  ({'ok' if report['shm_ok_for_workers'] else 'TOO SMALL'})")
        print("disk:")
        for d in dk:
            print(f"  {d['path']:<10} free {d['free_gb']} GB / {d['total_gb']} GB "
                  f"({d['used_pct']}% used)")
        print(f"checkpoint headroom needed: {headroom:.0f} GB -> "
              f"{report['good_checkpoint_volumes'] or 'NONE big enough'}")
        if warnings:
            print("WARNINGS:")
            for w in warnings:
                print("  ! " + w)
    sys.exit(0 if not warnings else 1)


if __name__ == "__main__":
    main()
