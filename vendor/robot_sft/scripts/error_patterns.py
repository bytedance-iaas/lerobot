#!/usr/bin/env python3
"""Shared training-log error classifier for robot_sft (lerobot edition).

Not all non-zero exits should be auto-retried (SkyPilot's principle). Some failures are
*configuration* problems that will recur identically on every restart — retrying them just
burns time and hides the real issue. Others are transient or fixable-on-retry. This module
classifies a training log tail into one of:

    "fatal"      -> do NOT retry; surface to the user with the fix (config/auth/data bug)
    "oom"        -> retryable, but only after reducing batch / freeing memory
    "retryable"  -> transient; resume from the last good checkpoint
    "ok"         -> no known error signature found

Used by preflight.py (classify a smoke-test) and watchdog.py (decide restart strategy).
The patterns target `lerobot-train` failure signatures (see lessons_learned.md).
"""
from __future__ import annotations

import re

# (regex, reason, fix) — checked in order; first match wins within a category.
FATAL = [
    (r"gated repo|Access to model .* is restricted|401 Client Error|403 Forbidden.*huggingface",
     "gated/unauthorized Hub model or dataset",
     "Accept the repo's license on huggingface.co and `hf auth login` (pi0's PaliGemma and "
     "some bases are gated), or point at a local path."),
    (r"Output directory .* already exists and resume is (False|false)",
     "output_dir exists but --resume not set",
     "Re-launch with --resume=true --config_path=<output_dir>/checkpoints/last/"
     "pretrained_model/train_config.json, or pick a fresh --output_dir."),
    (r"A config_path is expected when resuming",
     "--resume=true without --config_path",
     "Pass --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json."),
    (r"KeyError: ['\"]observation\.images|KeyError: ['\"]observation\.state|"
     r"Missing key.*observation\.|does not match.*input_features",
     "feature/camera key mismatch between policy and dataset",
     "The policy's input_features must match the dataset's keys (meta/info.json `features`). "
     "Check camera names (observation.images.<cam>) and state/action dims (lessons #6)."),
    (r"RepositoryNotFoundError|Repository Not Found|"
     r"FileNotFoundError.*(info\.json|meta/|\.parquet)|No such file or directory.*(data/|meta/)",
     "dataset not found / files missing",
     "Re-check --dataset.repo_id / --dataset.root; the dataset needs meta/info.json + data/. "
     "For local datasets pass --dataset.root=<path>."),
    (r"is not a valid EpisodeIndex|episodes.*out of range|Invalid episode",
     "episode index out of range",
     "The --dataset.episodes list references episodes the dataset doesn't have; re-run the "
     "split against this dataset's actual meta/info.json total_episodes."),
    (r"Python\.h: No such file|fatal error:.*\.h: No such file",
     "missing build headers",
     "Install pythonX.Y-dev system headers, then re-sync deps."),
    (r"draccus.*(error|invalid)|unrecognized arguments|invalid choice",
     "bad CLI arguments",
     "A lerobot-train flag is misspelled or has a bad value; re-check the launch command "
     "against `lerobot-train --help`."),
]

OOM = [
    (r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
     "CUDA OOM",
     "Lower --batch_size, then resume from the last checkpoint."),
]

RETRYABLE = [
    (r"Bus error|out of shared memory|unable to write.*No space left.*torch_|"
     r"DataLoader worker.*(killed|exited unexpectedly)",
     "/dev/shm exhausted or dataloader worker died",
     "Enlarge /dev/shm or set --num_workers=0, then resume (lessons #2)."),
    (r"NCCL.*(timeout|error|unhandled)|Socket Timeout|Connection reset",
     "transient distributed/NCCL error",
     "Resume from the last checkpoint; if it recurs, check interconnect."),
    (r"Traceback \(most recent call last\)|RuntimeError|Segmentation fault|Killed",
     "unclassified crash",
     "Resume from the last checkpoint; inspect the traceback if it repeats."),
]


def classify(log_text: str) -> dict:
    """Return {category, reason, fix, pattern} for the most specific signature found."""
    for cat, table in (("fatal", FATAL), ("oom", OOM), ("retryable", RETRYABLE)):
        for pat, reason, fix in table:
            if re.search(pat, log_text, re.IGNORECASE):
                return {"category": cat, "reason": reason, "fix": fix, "pattern": pat}
    return {"category": "ok", "reason": "", "fix": "", "pattern": ""}


def classify_file(path: str, max_bytes: int = 300_000) -> dict:
    import os
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            return classify(f.read().decode("utf-8", "replace"))
    except OSError as e:  # noqa: BLE001
        return {"category": "ok", "reason": f"log unreadable: {e}", "fix": "", "pattern": ""}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import json
        print(json.dumps(classify_file(sys.argv[1]), indent=2))
    else:
        print(__doc__)
