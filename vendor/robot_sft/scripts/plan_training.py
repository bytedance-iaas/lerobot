#!/usr/bin/env python3
"""Compute training parameters from dataset size + hardware, not a hardcoded default.

The #1 quiet mistake in robot SFT is keeping the entrypoint's default max_steps (e.g.
10000) regardless of how much data exists. This computes steps from epochs and the real
sample count, picks a sane epoch band for the dataset size, splits the global batch across
GPUs, and prints a ready-to-edit launch command.

Usage:
    python plan_training.py --samples 18881 --gpus 1 --gpu-mem-gb 143 \
        [--epochs 6] [--global-batch 32] [--save-total-limit 5] \
        [--output-dir /data/run1] [--base-model ...] [--dataset-path ...] \
        [--modality-config ...] [--embodiment-tag NEW_EMBODIMENT] \
        [--num-workers auto] [--shm-gb 16] [--cuda 0] [--json]

`--samples` is the training-sample count (≈ total_frames; from dataset_explore). If you
only know episodes, pass --episodes and --avg-len.
"""
from __future__ import annotations

import argparse
import json
import math


def suggest_epochs(samples: int) -> int:
    """Heuristic epoch band by dataset size. Small sets overfit fast — keep epochs modest
    and let open-loop eval pick the best checkpoint; large sets need fewer passes."""
    if samples < 5_000:
        return 10
    if samples < 30_000:
        return 6           # e.g. ~50 episodes / ~19k samples -> ~6 epochs
    if samples < 150_000:
        return 4
    return 3


def suggest_global_batch(gpu_mem_gb: float, gpus: int) -> int:
    """Conservative per-GPU batch for a ~3B VLA with a frozen backbone, scaled by GPUs."""
    if gpu_mem_gb >= 120:
        per = 32
    elif gpu_mem_gb >= 70:
        per = 16
    elif gpu_mem_gb >= 40:
        per = 8
    else:
        per = 4
    return per * max(1, gpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--avg-len", type=int, default=400)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-mem-gb", type=float, default=80.0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--global-batch", type=int)
    ap.add_argument("--save-total-limit", type=int, default=5)
    ap.add_argument("--save-steps", type=int)
    ap.add_argument("--throughput-it-s", type=float, default=None,
                    help="measured training it/s (from preflight/early log); caps save_steps for eval cadence")
    ap.add_argument("--max-eval-hours", type=float, default=1.0,
                    help="guarantee an eval (checkpoint) at least this often in wall-clock")
    ap.add_argument("--num-workers", default="auto")
    ap.add_argument("--shm-gb", type=float, default=-1.0)
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 5,6")
    ap.add_argument("--output-dir", default="/data/robot_sft_run")
    ap.add_argument("--base-model", default="<BASE_MODEL_PATH>")
    ap.add_argument("--dataset-path", default="<DATASET_PATH>")
    ap.add_argument("--modality-config", default=None)
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--entrypoint", default="examples/finetune.sh")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    samples = args.samples or ((args.episodes or 0) * args.avg_len)
    if samples <= 0:
        ap.error("provide --samples (preferred) or --episodes")

    gb = args.global_batch or suggest_global_batch(args.gpu_mem_gb, args.gpus)
    if gb % max(1, args.gpus) != 0:
        gb = (gb // args.gpus) * args.gpus or args.gpus
    per_device = gb // max(1, args.gpus)
    epochs = args.epochs or suggest_epochs(samples)
    steps_per_epoch = math.ceil(samples / gb)
    max_steps = steps_per_epoch * epochs
    save_steps = args.save_steps or max(100, round(max_steps / 10 / 100) * 100 or 100)

    # Eval runs only when a checkpoint is saved, so checkpoint cadence == eval cadence. Guarantee
    # at least one eval per --max-eval-hours by capping save_steps to what throughput covers in
    # that wall-clock window (needs measured it/s, best taken from preflight / early train.log).
    eval_cadence_note = None
    if args.throughput_it_s and args.throughput_it_s > 0:
        hourly_cap = int(args.throughput_it_s * 3600 * args.max_eval_hours)
        hourly_cap = max(100, (hourly_cap // 100) * 100)
        if save_steps > hourly_cap:
            eval_cadence_note = (f"save_steps {save_steps} -> {hourly_cap} to keep eval "
                                 f"<= {args.max_eval_hours}h apart at {args.throughput_it_s:.1f} it/s")
            save_steps = hourly_cap
        est_min = save_steps / args.throughput_it_s / 60
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

    plan = {
        "samples": samples, "epochs": epochs,
        "global_batch_size": gb, "per_device_batch": per_device, "gpus": args.gpus,
        "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
        "save_steps": save_steps, "save_total_limit": args.save_total_limit,
        "num_workers": num_workers, "num_workers_note": workers_note,
        "eval_cadence_note": eval_cadence_note,
        "output_dir": args.output_dir,
        "cuda_visible_devices": args.cuda,
        "resumable": True, "save_only_model": False,
    }

    cuda = f"CUDA_VISIBLE_DEVICES={args.cuda} " if args.cuda else ""
    env = (f"{cuda}NUM_GPUS={args.gpus} GLOBAL_BATCH_SIZE={gb} "
           f"DATALOADER_NUM_WORKERS={num_workers} MAX_STEPS={max_steps} "
           f"SAVE_STEPS={save_steps} USE_WANDB=0")
    modality = f" \\\n  --modality-config-path {args.modality_config}" if args.modality_config else ""
    cmd = (f"{env} \\\n"
           f"uv run bash {args.entrypoint} \\\n"
           f"  --base-model-path {args.base_model} \\\n"
           f"  --dataset-path {args.dataset_path} \\\n"
           f"  --embodiment-tag {args.embodiment_tag}{modality} \\\n"
           f"  --output-dir {args.output_dir}")
    plan["launch_command"] = cmd

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"samples={samples}  epochs={epochs}  global_batch={gb} "
              f"(per_device {per_device} x {args.gpus} gpu)")
        print(f"steps_per_epoch={steps_per_epoch}  -> max_steps={max_steps}")
        print(f"save_steps={save_steps}  save_total_limit={args.save_total_limit}")
        print(f"num_workers={num_workers}  ({workers_note})")
        print(f"resumable=True (save_only_model OFF)")
        print("\n# launch command:\n" + cmd)


if __name__ == "__main__":
    main()
