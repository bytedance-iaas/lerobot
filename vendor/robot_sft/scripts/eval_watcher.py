#!/usr/bin/env python3
"""Periodic open-loop eval watcher for robot_sft (GR00T has no in-loop eval).

GR00T's sharded finetune path asserts ``eval_strategy == "no"`` (see lessons_learned #13),
so the only way to get an eval *curve over training* is to evaluate each checkpoint as it is
saved. This watcher does exactly that, OUT OF BAND from training:

  - polls the run's ``output_dir`` for ``checkpoint-N`` directories,
  - waits until a checkpoint is COMPLETE (``trainer_state.json`` is written last by HF
    Trainer, so its presence means the model + processor + experiment_cfg are all there —
    lessons_learned #4), then
  - runs ``gr00t/eval/open_loop_eval.py`` on every held-out eval dataset (the ``eval/`` dirs
    produced by split_train_eval.py), on a SEPARATE GPU so it never contends with training
    (lessons_learned #9), and
  - appends one JSON line per checkpoint to ``<session>/eval/eval_results.jsonl`` with the
    per-dataset and mean MSE/MAE. monitor_server.py reads this file to draw the eval curve.

It is crash-safe/resumable: on restart it reads eval_results.jsonl and skips checkpoints it
already evaluated. It does NOT pass --modality-config-path to open_loop_eval (the modality is
read from the checkpoint's experiment_cfg — lessons_learned #11).

Eval dataset paths + embodiment tag are read from <session>/training_plan.json
(``eval_dataset_paths``, ``embodiment_tag``); traj-ids default to every episode in each eval
dir (read from its meta/info.json). Stops once the run is done/failed AND no complete,
un-evaluated checkpoint remains.

Run (background):
    python eval_watcher.py --session <dir> --run run-001 --gpu 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

# open_loop_eval.py hardcodes this dir for per-trajectory plots when --save-plot-path is unset
# (it can only take a single file path, so we let it use the default and harvest from here).
TMP_PLOT_DIR = "/tmp/open_loop_eval"

AVG_MSE_RE = re.compile(r"Average MSE across all trajs:\s*([0-9.eE+-]+)")
AVG_MAE_RE = re.compile(r"Average MAE across all trajs:\s*([0-9.eE+-]+)")
TRAJ_MSE_RE = re.compile(r"MSE for trajectory \d+:\s*([0-9.eE+-]+),\s*MAE:\s*([0-9.eE+-]+)")
CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def log(eval_dir, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(eval_dir, "eval_watcher.log"), "a") as f:
        f.write(line + "\n")


def complete_checkpoints(output_dir):
    """checkpoint-N dirs that are fully written (trainer_state.json present), step-sorted."""
    out = []
    for d in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        m = CKPT_RE.search(d)
        if not m or not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "trainer_state.json")):
            out.append((int(m.group(1)), d))
    return sorted(out)


def num_episodes(eval_path):
    info = read_json(os.path.join(eval_path, "meta", "info.json")) or {}
    return int(info.get("total_episodes", 0))


def _harvest_artifacts(dest_dir):
    """Move whatever artifact files the eval tool wrote to TMP_PLOT_DIR into dest_dir.
    Generic: captures ANY file (images, html, csv, ...), not just GR00T's traj_*.jpeg, so a
    richer eval that drops extra outputs is preserved too. Returns saved filenames, traj-sorted."""
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for src in sorted(glob.glob(os.path.join(TMP_PLOT_DIR, "*"))):
        if not os.path.isfile(src):
            continue
        fn = os.path.basename(src)
        try:
            shutil.move(src, os.path.join(dest_dir, fn))
            saved.append(fn)
        except Exception:  # noqa: BLE001
            pass
    return sorted(saved, key=lambda f: int(re.search(r"(\d+)", f).group(1)) if re.search(r"(\d+)", f) else 0)


def eval_one(py, repo, ckpt, eval_path, emb_tag, steps, horizon, gpu, eval_dir, env, plot_dest, timeout):
    """Run open_loop_eval on one eval dataset; save its trajectory plots into plot_dest.
    Return (mean_mse, mean_mae, [plot filenames]) or (None, None, [])."""
    n = num_episodes(eval_path)
    if n <= 0:
        log(eval_dir, f"  ! {os.path.basename(eval_path)}: no episodes, skipping")
        return None, None, []
    # clear the shared tmp plot dir so we only harvest THIS dataset's plots
    shutil.rmtree(TMP_PLOT_DIR, ignore_errors=True)
    traj_ids = [str(i) for i in range(n)]
    cmd = [
        py, "gr00t/eval/open_loop_eval.py",
        "--dataset-path", eval_path,
        "--embodiment-tag", emb_tag,
        "--model-path", ckpt,
        "--action-horizon", str(horizon),
        "--steps", str(steps),
        "--traj-ids", *traj_ids,
    ]
    # Run in its own session so we can kill the WHOLE tree on timeout. A hung eval (e.g. a
    # CUDA/thread deadlock when running alongside training) must never block the pipeline.
    import signal as _signal
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
        log(eval_dir, f"  ! {os.path.basename(eval_path)}: eval TIMEOUT after {timeout}s — killed, skipping")
    blob = out or ""
    proc = type("R", (), {"returncode": rc})  # keep downstream code unchanged
    plots = _harvest_artifacts(plot_dest)
    if rc == -9:
        return None, None, plots
    mse = AVG_MSE_RE.search(blob)
    mae = AVG_MAE_RE.search(blob)
    if mse and mae:
        return float(mse.group(1)), float(mae.group(1)), plots
    # fallback: average the per-trajectory lines
    pairs = TRAJ_MSE_RE.findall(blob)
    if pairs:
        ms = [float(a) for a, _ in pairs]
        mas = [float(b) for _, b in pairs]
        return sum(ms) / len(ms), sum(mas) / len(mas), plots
    tail = "\n".join(blob.strip().splitlines()[-6:])
    log(eval_dir, f"  ! {os.path.basename(eval_path)}: no MSE parsed (rc={proc.returncode}). tail:\n{tail}")
    return None, None, plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for eval (keep off training GPUs)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--eval-timeout", type=int, default=900,
                    help="per-dataset hard timeout (s); a hung eval is killed and skipped")
    ap.add_argument("--threads", type=int, default=8,
                    help="cap CPU threads per eval so it doesn't thread-storm alongside training")
    ap.add_argument("--embodiment-tag", default=None, help="override; else read from training_plan.json")
    args = ap.parse_args()

    session = args.session
    repo = os.getcwd()
    plan = read_json(os.path.join(session, "training_plan.json")) or {}
    output_dir = plan.get("output_dir")
    eval_paths = plan.get("eval_dataset_paths") or []
    emb_tag = args.embodiment_tag or plan.get("embodiment_tag", "new_embodiment")
    if not output_dir or not eval_paths:
        print("ERROR: training_plan.json missing output_dir or eval_dataset_paths", file=sys.stderr)
        sys.exit(2)

    eval_dir = os.path.join(session, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "eval_results.jsonl")
    run_json = os.path.join(session, "runs", args.run, "run.json")

    # env: dedicate a GPU; preserve HF cache/token discovery from the plan's env
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for k, v in (plan.get("env") or {}).items():
        if k.startswith("HF_") and isinstance(v, str):
            env[k] = v
    # Cap CPU threads: torch/torchcodec/blas each default to all-cores, and a 200+-thread eval
    # running next to training thread-storms into a CUDA-init deadlock. Keep eval lean.
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
                except Exception:
                    pass
    log(eval_dir, f"eval_watcher start: gpu={args.gpu} output_dir={output_dir} "
                  f"eval_sets={len(eval_paths)} already_done={sorted(done_steps)}")

    py = sys.executable
    while True:
        for step, ckpt in complete_checkpoints(output_dir):
            if step in done_steps:
                continue
            log(eval_dir, f"== eval checkpoint-{step} ==")
            per = {}
            mses, maes = [], []
            # standard, tool-agnostic artifact layout the dashboard scans: eval/artifacts/ckpt-N/<group>/
            ckpt_art_root = os.path.join(eval_dir, "artifacts", f"ckpt-{step}")
            for ep in eval_paths:
                name = os.path.basename(os.path.dirname(ep)) if ep.rstrip("/").endswith("eval") else os.path.basename(ep)
                t0 = time.time()
                art_dest = os.path.join(ckpt_art_root, name)
                mse, mae, arts = eval_one(py, repo, ckpt, ep, emb_tag, args.steps,
                                          args.action_horizon, args.gpu, eval_dir, env, art_dest,
                                          args.eval_timeout)
                if mse is not None:
                    per[name] = {
                        "metrics": {"mse": mse, "mae": mae},
                        "mse": mse, "mae": mae,  # kept flat for backward-compat
                        "artifacts": [os.path.join("artifacts", f"ckpt-{step}", name, f) for f in arts],
                    }
                    mses.append(mse)
                    maes.append(mae)
                    log(eval_dir, f"  {name}: mse={mse:.5f} mae={mae:.5f} artifacts={len(arts)} ({time.time()-t0:.0f}s)")
            mean_mse = (sum(mses) / len(mses)) if mses else None
            mean_mae = (sum(maes) / len(maes)) if maes else None
            rec = {
                "step": step,
                "checkpoint": ckpt,
                "per_dataset": per,
                # generic primary-metric block (dashboard plots/displays this without knowing the tool)
                "metrics": {"mean_mse": mean_mse, "mean_mae": mean_mae},
                "primary_metric": "mean_mse",
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_steps.add(step)
            mm = rec["mean_mse"]
            log(eval_dir, f"== checkpoint-{step} mean_mse={mm if mm is None else round(mm,5)} ==")

        run = read_json(run_json) or {}
        status = run.get("status")
        remaining = [s for s, _ in complete_checkpoints(output_dir) if s not in done_steps]
        if status in ("done", "failed") and not remaining:
            log(eval_dir, f"run status={status}, no remaining checkpoints -> eval_watcher exiting")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
