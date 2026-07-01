#!/usr/bin/env python3
"""Compute lerobot-train parameters from dataset size + hardware, not a hardcoded default.

The #1 quiet mistake in robot SFT is keeping a default step count (e.g. lerobot-train's
100k) regardless of how much data exists. This computes steps from epochs and the real
sample count, picks a sane epoch band for the dataset size, sizes the batch for the GPU
and policy family, and prints a ready-to-run `lerobot-train` command.

Environment convention (console pod): the lerobot checkout lives at /lerobot (run commands
from there via `uv run`), while ALL big artifacts — checkpoints, session state, HF caches —
live on the roomy /opt/data volume. Defaults below follow that.

Usage:
    python plan_training.py --samples 18881 --policy-type act \
        --dataset-repo-id user/so101_pick [--dataset-root /opt/data/datasets/so101_pick] \
        [--episodes-file <session>/preprocess.json] \
        [--gpus 1] [--gpu-mem-gb 24] [--epochs 6] [--batch-size 8] \
        [--save-freq N] [--throughput-it-s X] [--max-eval-hours 1] \
        [--num-workers auto] [--shm-gb 16] [--cuda 0] [--output-dir DIR] [--json]

`--samples` is the training-frame count (≈ total_frames from meta/info.json, minus the
held-out eval episodes; from dataset_explore). If you only know episodes, pass --episodes
and --avg-len.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os

# Policy families for batch sizing: small BC heads train comfortably at large batches;
# VLA-style policies (big frozen backbone, heavy vision tower) need far smaller ones.
SMALL_POLICIES = {"act", "diffusion", "tdmpc", "vqbet", "multi_task_dit", "gaussian_actor"}
VLA_POLICIES = {"pi0", "pi05", "pi0_fast", "smolvla", "groot", "molmoact2", "eo1", "wall_x", "sarm"}


def _default_artifact_root() -> str:
    env = os.environ.get("ROBOT_SFT_HOME")
    if env:
        return env
    if os.path.isdir("/opt/data"):
        return "/opt/data/robot_sft"
    return ".robot_sft"


def suggest_epochs(samples: int) -> int:
    """Heuristic epoch band by dataset size. Small sets overfit fast — keep epochs modest
    and let held-out eval pick the best checkpoint; large sets need fewer passes."""
    if samples < 5_000:
        return 10
    if samples < 30_000:
        return 6           # e.g. ~50 episodes / ~19k frames -> ~6 epochs
    if samples < 150_000:
        return 4
    return 3


def suggest_batch(policy_type: str, gpu_mem_gb: float) -> int:
    """Per-process batch by policy family + GPU memory. A starting point only — preflight
    measures real peak memory and suggests scaling up."""
    if policy_type in VLA_POLICIES:
        if gpu_mem_gb >= 120:
            return 32
        if gpu_mem_gb >= 70:
            return 16
        if gpu_mem_gb >= 40:
            return 8
        return 4
    # small BC policies (ACT ~50M params etc.)
    if gpu_mem_gb >= 70:
        return 64
    if gpu_mem_gb >= 40:
        return 32
    if gpu_mem_gb >= 20:
        return 16
    return 8


def load_train_episodes(path: str):
    """Pull the train episode list from a split artifact (preprocess.json or the
    split_train_eval.py output). Accepts {train_episodes:[...]} at top level or under
    the first entry of {datasets:[...]}. Returns (train_eps, eval_eps) (either may be None)."""
    d = json.load(open(path))
    if isinstance(d, dict) and "train_episodes" in d:
        return d.get("train_episodes"), d.get("eval_episodes")
    for ds in (d.get("datasets") or []):
        if "train_episodes" in ds:
            return ds.get("train_episodes"), ds.get("eval_episodes")
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, help="training frames (total_frames minus eval episodes)")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--avg-len", type=int, default=400)
    ap.add_argument("--policy-type", default="act",
                    help="lerobot policy preset: act, diffusion, pi0, pi05, smolvla, ...")
    ap.add_argument("--policy-path", default=None,
                    help="pretrained policy to finetune (--policy.path); overrides --policy-type")
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None,
                    help="local dataset dir (else $HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--episodes-file", default=None,
                    help="split artifact (from split_train_eval.py) with train/eval episode ids")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-mem-gb", type=float, default=24.0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, help="per-process batch (lerobot --batch_size)")
    ap.add_argument("--save-freq", type=int)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--throughput-it-s", type=float, default=None,
                    help="measured training it/s (from preflight/early log); caps save_freq for eval cadence")
    ap.add_argument("--max-eval-hours", type=float, default=1.0,
                    help="guarantee an eval (checkpoint) at least this often in wall-clock")
    ap.add_argument("--num-workers", default="auto")
    ap.add_argument("--shm-gb", type=float, default=-1.0)
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 5,6")
    ap.add_argument("--output-dir", default=None,
                    help="checkpoint dir; default <artifact_root>/runs/<policy>_<ts> (big disk)")
    ap.add_argument("--repo", default="/lerobot" if os.path.isdir("/lerobot") else ".",
                    help="lerobot checkout to run from (uv venv lives there)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    samples = args.samples or ((args.episodes or 0) * args.avg_len)
    if samples <= 0:
        ap.error("provide --samples (preferred) or --episodes")

    batch = args.batch_size or suggest_batch(args.policy_type, args.gpu_mem_gb)
    global_batch = batch * max(1, args.gpus)
    epochs = args.epochs or suggest_epochs(samples)
    steps_per_epoch = math.ceil(samples / global_batch)
    steps = steps_per_epoch * epochs
    save_freq = args.save_freq or max(100, round(steps / 10 / 100) * 100 or 100)

    # Offline eval runs only when a checkpoint is saved, so checkpoint cadence == eval
    # cadence. Guarantee at least one eval per --max-eval-hours by capping save_freq to what
    # throughput covers in that window (needs measured it/s from preflight / early train.log).
    eval_cadence_note = None
    if args.throughput_it_s and args.throughput_it_s > 0:
        hourly_cap = int(args.throughput_it_s * 3600 * args.max_eval_hours)
        hourly_cap = max(100, (hourly_cap // 100) * 100)
        if save_freq > hourly_cap:
            eval_cadence_note = (f"save_freq {save_freq} -> {hourly_cap} to keep eval "
                                 f"<= {args.max_eval_hours}h apart at {args.throughput_it_s:.1f} it/s")
            save_freq = hourly_cap
        est_min = save_freq / args.throughput_it_s / 60
        eval_cadence_note = (eval_cadence_note or "") + f" (~{est_min:.0f} min/eval)"

    # dataloader workers: need adequate /dev/shm for >0
    if args.num_workers == "auto":
        num_workers = 4 if (args.shm_gb < 0 or args.shm_gb >= 4) else 0
        workers_note = ("shm unknown -> assuming ok, using 4; verify with check_hardware"
                        if args.shm_gb < 0 else
                        ("/dev/shm ok -> 4" if num_workers else
                         "/dev/shm too small -> 0 (no async prefetch; expect stalls)"))
    else:
        num_workers = int(args.num_workers)
        workers_note = "user-specified"

    train_eps, eval_eps = (None, None)
    if args.episodes_file:
        train_eps, eval_eps = load_train_episodes(args.episodes_file)

    out_dir = args.output_dir or os.path.join(
        _default_artifact_root(), "runs",
        f"{args.policy_type}_{_dt.datetime.now():%Y%m%d_%H%M%S}")

    # ---- build the lerobot-train command -----------------------------------
    parts = ["uv run lerobot-train"]
    parts.append(f"--dataset.repo_id={args.dataset_repo_id}")
    if args.dataset_root:
        parts.append(f"--dataset.root={args.dataset_root}")
    if train_eps:
        eps_str = "[" + ", ".join(str(i) for i in train_eps) + "]"
        parts.append(f"--dataset.episodes='{eps_str}'")
    if args.policy_path:
        parts.append(f"--policy.path={args.policy_path}")
    else:
        parts.append(f"--policy.type={args.policy_type}")
    parts.append(f"--output_dir={out_dir}")
    parts.append(f"--steps={steps}")
    parts.append(f"--batch_size={batch}")
    parts.append(f"--num_workers={num_workers}")
    parts.append(f"--save_freq={save_freq}")
    parts.append(f"--log_freq={args.log_freq}")
    parts.append("--eval_freq=0")            # eval is out-of-band (eval_watcher on held-out episodes)
    parts.append("--wandb.enable=false")
    cuda = f"CUDA_VISIBLE_DEVICES={args.cuda} " if args.cuda else ""
    cmd = f"cd {args.repo} && {cuda}" + " \\\n  ".join(parts)

    plan = {
        "samples": samples, "epochs": epochs,
        "policy_type": args.policy_type, "policy_path": args.policy_path,
        "batch_size": batch, "global_batch_size": global_batch, "gpus": args.gpus,
        "steps_per_epoch": steps_per_epoch, "max_steps": steps,
        "save_freq": save_freq, "log_freq": args.log_freq,
        "num_workers": num_workers, "num_workers_note": workers_note,
        "eval_cadence_note": eval_cadence_note,
        "output_dir": out_dir,
        "repo": args.repo,
        "cuda_visible_devices": args.cuda,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
        # resume (see references/lerobot_resume.md): checkpoints always carry full training
        # state, so any complete checkpoint is resumable via --resume=true.
        "resume_command": (
            f"cd {args.repo} && {cuda}uv run lerobot-train --resume=true "
            f"--config_path={out_dir}/checkpoints/last/pretrained_model/train_config.json"),
        "launch_command": cmd,
    }
    if args.gpus > 1:
        plan["multi_gpu_note"] = ("for multi-GPU use `uv run accelerate launch --num_processes="
                                  f"{args.gpus} $(which lerobot-train) ...` with the same flags; "
                                  "batch_size is per process")

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"samples={samples}  epochs={epochs}  batch={batch}"
              f"{' x ' + str(args.gpus) + ' gpu = ' + str(global_batch) if args.gpus > 1 else ''}")
        print(f"steps_per_epoch={steps_per_epoch}  -> steps={steps}")
        print(f"save_freq={save_freq}  log_freq={args.log_freq}  (checkpoints keep full "
              f"training state -> always resumable)")
        print(f"num_workers={num_workers}  ({workers_note})")
        if train_eps is not None:
            print(f"train episodes: {len(train_eps)}  held-out eval episodes: "
                  f"{len(eval_eps or [])}")
        print("\n# launch command:\n" + cmd)
        print("\n# resume after any stop:\n" + plan["resume_command"])


if __name__ == "__main__":
    main()
