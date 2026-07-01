#!/usr/bin/env python3
"""Deterministic train/eval episode split for lerobot datasets.

lerobot needs NO physical dataset split: `lerobot-train --dataset.episodes='[...]'`
trains on an episode subset, and the offline evaluator (offline_eval.py) scores a
checkpoint on any episode list. So "splitting" = choosing two disjoint, deterministic
episode-id lists and recording them — the dataset itself is never touched.
(The GR00T-era version of this script physically copied + re-indexed a v2.1 dataset;
none of that is needed here.)

Why hold out at all: training loss alone cannot tell memorization from generalization.
Scoring checkpoints on episodes the model NEVER saw is the only honest offline signal
(and how the eval curve picks the best checkpoint). Default holdout ≈ 10% of episodes
(min 1); tiny datasets get a warning.

The episode count comes from (first match wins):
  1. --dataset-root/meta/info.json  (local dataset dir, lerobot v3.0 layout)
  2. $HF_LEROBOT_HOME/<repo_id>/meta/info.json (defaults to ~/.cache/huggingface/lerobot)
  3. --total-episodes N (explicit)

Usage:
    python split_train_eval.py --dataset-repo-id user/so101_pick \
        [--dataset-root /opt/data/datasets/so101_pick] [--total-episodes N] \
        [--holdout-frac 0.1] [--seed 42] [--out <session>/preprocess.json]

Output JSON: {dataset_repo_id, dataset_root, total_episodes, seed, holdout_frac,
              train_episodes: [...], eval_episodes: [...]}
plan_training.py reads it via --episodes-file; eval_watcher.py reads eval_episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def find_info_json(repo_id: str, root: str | None) -> str | None:
    candidates = []
    if root:
        candidates.append(os.path.join(root, "meta", "info.json"))
    hf_lerobot_home = os.environ.get(
        "HF_LEROBOT_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot"))
    candidates.append(os.path.join(hf_lerobot_home, repo_id, "meta", "info.json"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--total-episodes", type=int, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--min-eval", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write the split JSON here (else stdout only)")
    args = ap.parse_args()

    total = args.total_episodes
    info_path = find_info_json(args.dataset_repo_id, args.dataset_root)
    if total is None:
        if not info_path:
            print("ERROR: dataset meta/info.json not found (pass --dataset-root or "
                  "--total-episodes)", file=sys.stderr)
            sys.exit(2)
        info = json.load(open(info_path))
        total = int(info["total_episodes"])
    if total <= 0:
        print("ERROR: dataset has no episodes", file=sys.stderr)
        sys.exit(2)

    n_eval = max(args.min_eval, round(total * args.holdout_frac)) if args.holdout_frac > 0 else 0
    warning = None
    if 0 < total < 5 and n_eval:
        warning = (f"only {total} episodes — holding out {n_eval} leaves very little "
                   f"training data; consider --holdout-frac 0 (eval then only sanity-checks "
                   f"learning, not generalization)")
    if n_eval >= total:
        n_eval = max(0, total - 1)
        warning = f"holdout clamped to {n_eval} so at least 1 episode remains for training"

    rng = random.Random(args.seed)          # deterministic: same seed => same split
    ids = list(range(total))
    rng.shuffle(ids)
    eval_eps = sorted(ids[:n_eval])
    train_eps = sorted(ids[n_eval:])

    out = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "info_json": info_path,
        "total_episodes": total,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
    }
    if warning:
        out["warning"] = warning

    blob = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(blob + "\n")
        print(f"wrote {args.out}  (train={len(train_eps)} eval={len(eval_eps)} of {total})")
        if warning:
            print("WARNING: " + warning)
    else:
        print(blob)


if __name__ == "__main__":
    main()
