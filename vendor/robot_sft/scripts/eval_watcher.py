#!/usr/bin/env python3
"""Periodic offline-eval watcher for robot_sft (lerobot edition).

Training on real-robot data has no simulator to eval in, so the eval *curve over
training* comes from scoring each checkpoint as it is saved, OUT OF BAND from training:

  - polls ``<output_dir>/checkpoints/`` for step dirs (lerobot writes
    ``<step>/pretrained_model/`` + ``<step>/training_state/``),
  - waits until a checkpoint is COMPLETE (``training_state/training_step.json`` is part
    of the last-written state dir — its presence means the save finished), then
  - runs ``offline_eval.py`` (open-loop replay) on the HELD-OUT episodes recorded by
    split_train_eval.py, on a SEPARATE GPU so it never contends with training
    (lessons_learned #9), and
  - appends one JSON line per checkpoint to ``<session>/eval/eval_results.jsonl`` with
    MSE/MAE. monitor_server.py reads this file to draw the eval curve.

It is crash-safe/resumable: on restart it reads eval_results.jsonl and skips checkpoints
already evaluated. Dataset repo/root + eval episode ids come from
<session>/training_plan.json (``dataset_repo_id``, ``dataset_root``, ``eval_episodes``,
``repo``). Stops once the run is done/failed AND no complete, un-evaluated checkpoint
remains.

Run (background, from anywhere — it cd's into the lerobot checkout itself):
    python eval_watcher.py --session <dir> --run run-001 --gpu 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal as _signal
import subprocess
import sys
import time

AVG_MSE_RE = re.compile(r"Average MSE across all trajs:\s*([0-9.eE+-]+)")
AVG_MAE_RE = re.compile(r"Average MAE across all trajs:\s*([0-9.eE+-]+)")
TRAJ_MSE_RE = re.compile(r"MSE for trajectory \d+:\s*([0-9.eE+-]+),\s*MAE:\s*([0-9.eE+-]+)")

SKILL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def log(eval_dir, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(eval_dir, "eval_watcher.log"), "a") as f:
        f.write(line + "\n")


def complete_checkpoints(output_dir):
    """lerobot checkpoint step-dirs that are fully written, step-sorted.
    Layout: <output_dir>/checkpoints/<NNNNNN>/{pretrained_model,training_state}."""
    out = []
    for d in glob.glob(os.path.join(output_dir, "checkpoints", "*")):
        name = os.path.basename(d.rstrip("/"))
        if not (os.path.isdir(d) and name.isdigit()):
            continue  # skips the 'last' symlink too
        if os.path.exists(os.path.join(d, "training_state", "training_step.json")):
            out.append((int(name), d))
    return sorted(out)


def eval_one(repo, ckpt_dir, plan, gpu, eval_dir, env, plot_dest, timeout,
             max_frames):
    """Run offline_eval.py for one checkpoint; return (mean_mse, mean_mae, [plot files])."""
    eval_eps = plan.get("eval_episodes") or []
    model_path = os.path.join(ckpt_dir, "pretrained_model")
    cmd = ["uv", "run", "python", os.path.join(SKILL_SCRIPTS, "offline_eval.py"),
           "--model-path", model_path,
           "--dataset-repo-id", plan["dataset_repo_id"],
           "--episodes", *[str(i) for i in eval_eps],
           "--device", "cuda",
           "--max-frames-per-episode", str(max_frames),
           "--plot-dir", plot_dest]
    if plan.get("dataset_root"):
        cmd += ["--dataset-root", plan["dataset_root"]]
    # Own session so we can kill the WHOLE tree on timeout — a hung eval must never
    # block the pipeline.
    p = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), _signal.SIGKILL)
        except Exception:  # noqa: BLE001
            p.kill()
        out, _ = p.communicate()
        rc = -9
        log(eval_dir, f"  ! eval TIMEOUT after {timeout}s — killed, skipping")
    blob = out or ""
    plots = sorted(os.path.basename(f) for f in glob.glob(os.path.join(plot_dest, "*.png")))
    if rc == -9:
        return None, None, plots
    mse, mae = AVG_MSE_RE.search(blob), AVG_MAE_RE.search(blob)
    if mse and mae:
        return float(mse.group(1)), float(mae.group(1)), plots
    pairs = TRAJ_MSE_RE.findall(blob)  # fallback: average the per-trajectory lines
    if pairs:
        return (sum(float(a) for a, _ in pairs) / len(pairs),
                sum(float(b) for _, b in pairs) / len(pairs), plots)
    tail = "\n".join(blob.strip().splitlines()[-8:])
    log(eval_dir, f"  ! no MSE parsed (rc={rc}). tail:\n{tail}")
    return None, None, plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for eval (keep off training GPUs)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=450, help="per-episode replay cap")
    ap.add_argument("--eval-timeout", type=int, default=1800,
                    help="per-checkpoint hard timeout (s); a hung eval is killed and skipped")
    ap.add_argument("--threads", type=int, default=8,
                    help="cap CPU threads per eval so it doesn't thread-storm alongside training")
    args = ap.parse_args()

    session = args.session
    plan = read_json(os.path.join(session, "training_plan.json")) or {}
    output_dir = plan.get("output_dir")
    repo = plan.get("repo") or ("/lerobot" if os.path.isdir("/lerobot") else os.getcwd())
    if not output_dir or not plan.get("dataset_repo_id") or not plan.get("eval_episodes"):
        print("ERROR: training_plan.json needs output_dir + dataset_repo_id + eval_episodes "
              "(run split_train_eval.py and plan_training.py --episodes-file first)",
              file=sys.stderr)
        sys.exit(2)

    eval_dir = os.path.join(session, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "eval_results.jsonl")
    run_json = os.path.join(session, "runs", args.run, "run.json")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Cap CPU threads: torch/blas default to all-cores; an eval running next to training
    # thread-storms into stalls. Keep eval lean.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(args.threads)
    env["TOKENIZERS_PARALLELISM"] = "false"

    done_steps = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    done_steps.add(int(json.loads(line)["step"]))
                except Exception:  # noqa: BLE001
                    pass
    log(eval_dir, f"eval_watcher start: gpu={args.gpu} output_dir={output_dir} "
                  f"eval_episodes={plan.get('eval_episodes')} already_done={sorted(done_steps)}")

    while True:
        for step, ckpt in complete_checkpoints(output_dir):
            if step in done_steps:
                continue
            log(eval_dir, f"== eval checkpoint {step} ==")
            t0 = time.time()
            art_dest = os.path.join(eval_dir, "artifacts", f"ckpt-{step}",
                                    os.path.basename(plan["dataset_repo_id"]))
            mse, mae, arts = eval_one(repo, ckpt, plan, args.gpu, eval_dir, env,
                                      art_dest, args.eval_timeout, args.max_frames)
            rec = {
                "step": step,
                "checkpoint": ckpt,
                "metrics": {"mean_mse": mse, "mean_mae": mae},
                "primary_metric": "mean_mse",
                "mean_mse": mse,
                "mean_mae": mae,
                "artifacts": [os.path.join("artifacts", f"ckpt-{step}",
                                           os.path.basename(plan["dataset_repo_id"]), f)
                              for f in arts],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_steps.add(step)
            log(eval_dir, f"== checkpoint {step} mean_mse="
                          f"{mse if mse is None else round(mse, 6)} "
                          f"({time.time()-t0:.0f}s, {len(arts)} plots) ==")

        run = read_json(run_json) or {}
        status = run.get("status")
        remaining = [s for s, _ in complete_checkpoints(output_dir) if s not in done_steps]
        if status in ("done", "failed", "stopped") and not remaining:
            log(eval_dir, f"run status={status}, no remaining checkpoints -> eval_watcher exiting")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
