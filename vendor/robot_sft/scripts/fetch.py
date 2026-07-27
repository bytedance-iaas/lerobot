#!/usr/bin/env python3
"""Resolve a model/dataset download source: oniond (Volcengine TOS) first, HF-mirror fallback.

Policy: for an EXTERNAL model or dataset (an HF-style `org/name`), prefer `oniond` — it pulls
from a Volcengine TOS bucket (default `ai-infra`), fast and inside the cluster, no AK/SK. If the
name isn't in the bucket, fall back to the HF hub (via HF_ENDPOINT / hf-mirror, handled by
lerobot's own huggingface_hub download). `tos://…` datasets are NOT touched here — they keep the
StreamingTOSRobotDataset path.

oniond stores files flat: `oniond download model <name> --dir <D>` -> `<D>/<name>/<files>`, and
`<D>/<name>` is directly usable as lerobot's `--policy.path` (config.json + model.safetensors) or
`--dataset.root`. The HF org is stripped (`lerobot/pi05_base` -> oniond `pi05_base`).

Usage:
    python fetch.py model  <org/name | name>  [--dir <cache>]
    python fetch.py dataset <org/name | tos://… | name> [--dir <cache>]

Prints JSON: {"source": "oniond"|"hf"|"tos", "repo_id": <name>, "local_path": <dir>|null}.
Then use:  model  -> --policy.path=<local_path or repo_id>
           dataset-> --dataset.repo_id=<repo_id> [--dataset.root=<local_path> if set]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec - fixed argv, no shell
import sys

_OBJECT_STORE_PREFIXES = ("tos://", "s3://", "gs://")
DEFAULT_CACHE = os.environ.get("ONIOND_CACHE", "/opt/data/.cache/oniond")


def _oniond_env() -> dict:
    env = dict(os.environ)
    env.setdefault("BUCKET", "ai-infra")
    return env


def _oniond_has(kind: str, name: str) -> bool:
    """True if `name` (no org) is listed in the oniond bucket for this kind."""
    try:
        out = subprocess.run(["oniond", "list", kind], capture_output=True, text=True,
                             env=_oniond_env(), timeout=60)  # nosec
    except Exception:
        return False
    # `oniond list` prints a boxed table, one name per row — match the whole cell.
    names = {ln.strip(" |").strip() for ln in out.stdout.splitlines()}
    return name in names


def resolve(kind: str, ref: str, cache_dir: str) -> dict:
    """kind: 'model'|'dataset'. Returns {source, repo_id, local_path}."""
    if ref.startswith(_OBJECT_STORE_PREFIXES):
        return {"source": "tos", "repo_id": ref, "local_path": None}   # streaming path, untouched

    name = ref.rstrip("/").split("/")[-1]   # oniond is org-less: strip `org/`
    # Announce each phase to stderr: this script runs for minutes (download) and a silent
    # stretch reads as "stuck" — the caller tails the log for progress (see SKILL.md).
    print(f"[fetch] checking oniond bucket for {kind} '{name}' (≤60s)...", file=sys.stderr, flush=True)
    if shutil.which("oniond") and _oniond_has(kind, name):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[fetch] oniond download {kind} '{name}' -> {cache_dir} "
              f"(aria2 progress follows)...", file=sys.stderr, flush=True)
        proc = subprocess.run(["oniond", "download", kind, name, "--dir", cache_dir],  # nosec
                              env=_oniond_env(), stdout=sys.stderr, stderr=sys.stderr)
        local = os.path.join(cache_dir, name)
        if proc.returncode == 0 and os.path.isdir(local) and os.listdir(local):
            print(f"[fetch] oniond download complete: {local}", file=sys.stderr, flush=True)
            return {"source": "oniond", "repo_id": ref, "local_path": local}
        print(f"[fetch] oniond download of {kind} '{name}' failed — falling back to HF.",
              file=sys.stderr, flush=True)
    else:
        print(f"[fetch] '{name}' not in oniond bucket — using HF (hf-mirror); lerobot will "
              f"download on first use (can take minutes, watch its own progress).",
              file=sys.stderr, flush=True)
    return {"source": "hf", "repo_id": ref, "local_path": None}   # lerobot downloads via hf-mirror


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["model", "dataset"])
    ap.add_argument("ref", help="HF-style org/name, a bare name, or tos://… (datasets)")
    ap.add_argument("--dir", default=DEFAULT_CACHE, help=f"oniond download cache (default {DEFAULT_CACHE})")
    args = ap.parse_args()
    print(json.dumps(resolve(args.kind, args.ref, args.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
