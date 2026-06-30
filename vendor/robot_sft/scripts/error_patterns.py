#!/usr/bin/env python3
"""Shared training-log error classifier for robot_sft.

Not all non-zero exits should be auto-retried (SkyPilot's principle). Some failures are
*configuration* problems that will recur identically on every restart — retrying them just
burns time and hides the real issue. Others are transient or fixable-on-retry. This module
classifies a training log tail into one of:

    "fatal"      -> do NOT retry; surface to the user with the fix (config/auth/data bug)
    "oom"        -> retryable, but only after reducing batch / freeing memory
    "retryable"  -> transient; resume from the last good checkpoint
    "ok"         -> no known error signature found

Used by preflight.py (classify a smoke-test) and watchdog.py (decide restart strategy).
The patterns come straight from lessons_learned.md — real GR00T/LeRobot failure signatures.
"""
from __future__ import annotations

import re

# (regex, reason, fix) — checked in order; first match wins within a category.
FATAL = [
    (r"gated repo|Access to model .* is restricted|401 Client Error",
     "gated backbone / HF auth",
     "Accept the gated repo's license and `hf auth login`, or pass local model paths."),
    (r"Unrecognized (processing|configuration) class|Can't instantiate a processor",
     "checkpoint missing processor/config files",
     "Copy processor/ + experiment_cfg/ from a complete sibling checkpoint (lessons #4)."),
    (r"KeyError: .*observation\.images|modality.*not.*found|original_key",
     "camera/modality key mismatch",
     "Fix modality.json original_key to match meta/info.json features (lessons #6)."),
    (r"FileNotFoundError.*(dataset|parquet|episode_)|No such file or directory.*data/",
     "dataset path / files missing",
     "Re-check the dataset path and that conversion produced data/ + videos/."),
    (r"Python\.h: No such file|fatal error:.*\.h: No such file",
     "missing build headers",
     "Install pythonX.Y-dev system headers, then re-sync deps (lessons #10)."),
    (r"global_batch_size must be divisible by num_gpus|AssertionError.*batch.*gpus",
     "batch/gpu config invalid",
     "Make global_batch_size divisible by num_gpus (lessons #8)."),
]

OOM = [
    (r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
     "CUDA OOM",
     "Lower global_batch_size (or per-device batch), then resume from the last checkpoint."),
]

RETRYABLE = [
    (r"Bus error|out of shared memory|unable to write.*No space left.*torch_",
     "/dev/shm exhausted (dataloader workers)",
     "Enlarge /dev/shm or set num_workers=0, then resume (lessons #2)."),
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
