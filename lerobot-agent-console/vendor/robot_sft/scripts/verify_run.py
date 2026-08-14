#!/usr/bin/env python3
"""Post-run self-verification for robot_sft (lerobot edition).

`exit 0` is not proof a run succeeded — real runs have exited cleanly after silently
failing (auth error, truncated checkpoint). This produces a set of INDEPENDENT verdicts
about whether the run genuinely worked, written to VERIFY.md + verify.json in the run dir,
so success is verified rather than assumed.

Checks (each pass/fail/warn, independently):
  1. progress       — training reached a meaningful step (>= --min-step, or its max_step)
  2. loss_decreased — last loss is meaningfully below the early loss
  3. loss_finite    — no NaN/Inf in the loss trace
  4. checkpoint     — a checkpoint exists under <output_dir>/checkpoints/
  5. resumable      — latest checkpoint has full training_state (optimizer/rng/step)
  6. inference_ready— latest checkpoint's pretrained_model/ loads (config + weights)
  7. no_fatal       — the log has no fatal error signature (error_patterns)

Usage:
    python verify_run.py --session <session_dir> --run run-001 [--min-step 100]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_patterns  # noqa: E402
import watchdog as wd  # reuse is_resumable / latest checkpoint helpers  # noqa: E402

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")               # tqdm exact step
LOSS_RE = re.compile(r"\bloss:([0-9.eE+-]+|nan|inf|-inf)")    # lerobot tracker


def read_tail(path: str, max_bytes: int = 400_000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def latest_checkpoint_any(output_dir: str):
    """Highest-step dir under checkpoints/ regardless of completeness."""
    best, step = None, -1
    try:
        for n in os.listdir(os.path.join(output_dir, "checkpoints")):
            d = os.path.join(output_dir, "checkpoints", n)
            if n.isdigit() and os.path.isdir(d) and int(n) > step:
                best, step = d, int(n)
    except OSError:
        pass
    return best


def checkpoint_step(ckpt: str):
    """Exact saved step from training_state/training_step.json (None if unreadable)."""
    try:
        d = json.load(open(os.path.join(ckpt, "training_state", "training_step.json")))
        return int(d["step"])
    except Exception:  # noqa: BLE001
        return None


def inference_ready(ckpt: str) -> bool:
    """pretrained_model/ must be loadable for eval: policy config + weights (+ train cfg)."""
    if not ckpt:
        return False
    pm = os.path.join(ckpt, "pretrained_model")
    try:
        names = set(os.listdir(pm))
    except OSError:
        return False
    return "config.json" in names and "model.safetensors" in names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-step", type=int, default=100)
    args = ap.parse_args()

    run_dir = os.path.join(args.session, "runs", args.run)
    run = json.load(open(os.path.join(run_dir, "run.json")))
    log_path = run.get("log") or os.path.join(run_dir, "train.log")
    output_dir = run.get("output_dir") or json.load(
        open(os.path.join(args.session, "training_plan.json"))).get("output_dir", "")

    text = read_tail(log_path)
    steps = [int(m.group(1)) for m in STEP_RE.finditer(text)]
    maxsteps = [int(m.group(2)) for m in STEP_RE.finditer(text)]
    losses = []
    for raw in LOSS_RE.findall(text):
        try:
            losses.append(float(raw))
        except ValueError:
            losses.append(float("nan"))

    ckpt = latest_checkpoint_any(output_dir)
    resumable_ck = wd.latest_resumable_checkpoint(output_dir)
    # the checkpoint's own recorded step is authoritative (tqdm may be truncated from the tail)
    ck_step = checkpoint_step(resumable_ck or ckpt) if (resumable_ck or ckpt) else None
    last_step = max([s for s in steps] + ([ck_step] if ck_step else []) or [0])
    target = maxsteps[-1] if maxsteps else None
    cls = error_patterns.classify(text)

    def v(name, ok, detail):
        return {"check": name, "verdict": "pass" if ok else "fail", "detail": detail}

    checks = []
    checks.append(v("progress", last_step >= args.min_step or (target and last_step >= target),
                    f"reached step {last_step}" + (f"/{target}" if target else "")))
    if len(losses) >= 2:
        early = sum(losses[:max(1, len(losses)//5)]) / max(1, len(losses)//5)
        late = sum(losses[-max(1, len(losses)//5):]) / max(1, len(losses)//5)
        checks.append(v("loss_decreased", late < early * 0.95,
                        f"early≈{early:.3f} -> late≈{late:.3f}"))
        checks.append(v("loss_finite", all(x == x and abs(x) != float('inf') for x in losses),
                        "no NaN/Inf in loss trace"))
    else:
        checks.append({"check": "loss_decreased", "verdict": "warn", "detail": "too few loss points logged"})
        checks.append({"check": "loss_finite", "verdict": "warn", "detail": "too few loss points logged"})
    checks.append(v("checkpoint", ckpt is not None, ckpt or "no checkpoint found"))
    checks.append(v("resumable", resumable_ck is not None,
                    (os.path.basename(resumable_ck) if resumable_ck else "no resumable checkpoint "
                     "(missing training_state optimizer/rng/step — truncated?)")))
    checks.append(v("inference_ready", inference_ready(ckpt),
                    "pretrained_model/ has config+weights" if inference_ready(ckpt)
                    else "latest checkpoint missing pretrained_model/config.json or "
                         "model.safetensors (repair before eval)"))
    checks.append(v("no_fatal", cls["category"] != "fatal",
                    "no fatal signature" if cls["category"] != "fatal" else f"{cls['reason']}: {cls['fix']}"))

    n_fail = sum(1 for c in checks if c["verdict"] == "fail")
    overall = "PASS" if n_fail == 0 else "FAIL"
    summary = {"run": args.run, "overall": overall, "failures": n_fail,
               "last_step": last_step, "target_step": target,
               "latest_checkpoint": os.path.basename(ckpt) if ckpt else None,
               "latest_resumable": os.path.basename(resumable_ck) if resumable_ck else None,
               "checks": checks}

    json.dump(summary, open(os.path.join(run_dir, "verify.json"), "w"), indent=2)
    with open(os.path.join(run_dir, "VERIFY.md"), "w") as f:
        f.write(f"# VERIFY — {args.run}: {overall}\n\n")
        f.write(f"- last step: {last_step}{'/' + str(target) if target else ''}\n")
        f.write(f"- latest checkpoint: {summary['latest_checkpoint']}  "
                f"(resumable: {summary['latest_resumable']})\n\n")
        f.write("| check | verdict | detail |\n|---|---|---|\n")
        for c in checks:
            f.write(f"| {c['check']} | {c['verdict']} | {c['detail']} |\n")

    print(json.dumps(summary, indent=2))
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
