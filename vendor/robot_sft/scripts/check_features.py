#!/usr/bin/env python3
"""Fast static pre-check: does the dataset's cameras match the (pretrained) policy's?

The #1 crash when FINETUNING a pretrained VLA (`--policy.path=lerobot/pi05_base`, `pi0_base`,
`smolvla_base`, …) is a camera-key mismatch: the checkpoint was trained on one camera set
(e.g. DROID's `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`) but your dataset names its
cameras differently (`front / wrist`). lerobot then raises deep inside `make_policy`:

    ValueError: Feature mismatch ... Missing: [...] Extra: [...]

...but only AFTER importing torch + resolving the policy, which is slow and — inside the
console agent — tends to blow up a whole turn. This script catches it in a second from two
tiny JSON files (the dataset `meta/info.json` + the policy `config.json`), with NO torch and
NO model download, and prints the concrete fix:

- policy needs the SAME number of cameras (or fewer) → a `--rename_map` maps dataset keys to
  the expected keys (finetuning keeps working);
- policy needs MORE cameras than the dataset has → rename can't invent one; train from scratch
  with `--policy.type=<family>` (adapts to your cameras) instead of finetuning that checkpoint.

Usage:
    python check_features.py --dataset-repo-id <id|tos://…> [--dataset-root <dir>] \
        --policy-path <lerobot/pi05_base | local_dir>
    python check_features.py --dataset-repo-id <id> --policy-type pi05   # no pretrained → always OK

Exit 0 = OK (or --policy-type, which adapts). Exit 2 = mismatch (the fix is printed). Exit 3 =
could not read one of the JSONs (network/creds) — training is NOT gated on an unknown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

_OBJECT_STORE_PREFIXES = ("tos://", "s3://", "gs://")
IMG_PREFIX = "observation.images"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _load_json_local(path: str):
    return json.load(open(path)) if os.path.isfile(path) else None


def _load_json_hf(repo_id: str, filename: str, repo_type: str):
    """Download a small metadata file from the HF hub (or its mirror). Prefer huggingface_hub —
    it honours HF_ENDPOINT, follows the mirror's redirects, sends a proper UA (hf-mirror 403s
    raw urllib), and passes HF_TOKEN for gated repos (e.g. PaliGemma). Fall back to urllib +UA."""
    try:
        from huggingface_hub import hf_hub_download

        return json.load(open(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)))
    except Exception:
        pass
    sub = "datasets/" if repo_type == "dataset" else ""
    url = f"{HF_ENDPOINT}/{sub}{repo_id}/resolve/main/{filename}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "robot_sft-check_features/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec - fixed HF/mirror host
            return json.loads(r.read().decode())
    except Exception:
        return None


def _load_json_fsspec(url: str):
    try:
        import fsspec

        so: dict = {}
        if url.startswith("tos://"):
            so = {"endpoint": os.environ.get("TOS_ENDPOINT", "https://tos-cn-beijing.volces.com"),
                  "region": os.environ.get("TOS_REGION", "cn-beijing")}
            if os.environ.get("TOS_ACCESS_KEY"):
                so["key"] = os.environ["TOS_ACCESS_KEY"]
            if os.environ.get("TOS_SECRET_KEY"):
                so["secret"] = os.environ["TOS_SECRET_KEY"]
        with fsspec.open(url, "r", **so) as f:
            return json.load(f)
    except Exception:
        return None


def dataset_info(repo_id: str, root: str | None) -> dict | None:
    """The dataset's meta/info.json — local cache, object store, or the HF (mirror) hub."""
    cands = []
    if root:
        cands.append(os.path.join(root, "meta", "info.json"))
    hf_home = os.environ.get("HF_LEROBOT_HOME",
                             os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot"))
    cands.append(os.path.join(hf_home, repo_id, "meta", "info.json"))
    for c in cands:
        j = _load_json_local(c)
        if j is not None:
            return j
    if repo_id.startswith(_OBJECT_STORE_PREFIXES):
        return _load_json_fsspec(f"{repo_id.rstrip('/')}/meta/info.json")
    return _load_json_hf(repo_id, "meta/info.json", "dataset")


def policy_config(policy_path: str) -> dict | None:
    """The (pretrained) policy's config.json — a local dir or an HF (mirror) model repo."""
    local = policy_path if policy_path.endswith(".json") else os.path.join(policy_path, "config.json")
    j = _load_json_local(local)
    if j is not None:
        return j
    return _load_json_hf(policy_path, "config.json", "model")


def _image_keys(features: dict | None) -> set[str]:
    """`observation.images.*` keys, excluding pi0/pi05's synthetic `empty_camera_*` padding."""
    if not isinstance(features, dict):
        return set()
    return {k for k in features if k.startswith(IMG_PREFIX) and "empty_camera" not in k}


def _pair(provided: set[str], expected: set[str]) -> dict[str, str]:
    """{dataset_key -> checkpoint_key} rename, by sorted order. pi0/pi05 have NO per-camera slot
    embedding (each image goes through the same SigLIP, torch.cat'd into the prefix), so WHICH
    real camera takes which checkpoint name is arbitrary — a deterministic pairing is enough.
    Any checkpoint camera left over is auto-padded (black, attention-masked) at runtime."""
    return dict(zip(sorted(provided), sorted(expected)))


def check(dataset_repo_id: str, dataset_root: str | None,
          policy_path: str | None, policy_type: str | None = None) -> dict:
    """Compare dataset cameras vs the pretrained policy's. Returns a verdict dict:
    {status: "ok"|"mismatch"|"unknown", ok: bool, fix: str|None, provided/expected/...}.
    `status=="unknown"` means a JSON couldn't be read — callers should NOT gate on it."""
    # Training from scratch (--policy.type) has no pretrained camera set to match — always fine.
    if not policy_path:
        return {"status": "ok", "ok": True, "reason": "from_scratch", "fix": None}

    info = dataset_info(dataset_repo_id, dataset_root)
    cfg = policy_config(policy_path)
    if info is None or cfg is None:
        which = "dataset meta/info.json" if info is None else f"policy config.json ({policy_path})"
        return {"status": "unknown", "ok": True, "detail": f"could not read {which}", "fix": None}

    provided = _image_keys(info.get("features"))
    expected = _image_keys(cfg.get("input_features"))
    missing = sorted(expected - provided)   # cameras the policy needs but the dataset lacks
    extra = sorted(provided - expected)      # dataset cameras the policy doesn't name

    if not missing:  # lerobot's rule: the policy's cameras must be a SUBSET of the dataset's.
        return {"status": "ok", "ok": True, "provided": sorted(provided),
                "expected": sorted(expected), "fix": None, "rename_map": None}

    # ---- mismatch: finetune keeps working with a rename; extra cameras auto-pad ------------
    # A rename_map both (a) aligns the dataset's camera names to the checkpoint's, and (b) makes
    # lerobot SKIP visual-feature validation (factory.py:650). Any checkpoint camera the dataset
    # doesn't cover is auto-fed a black, attention-masked image at runtime (modeling_pi05.py:1207),
    # so the pretrained weights are kept. (Only hard rule: >=1 real camera — satisfied.)
    rename = _pair(provided, expected)                       # {dataset_key: checkpoint_key}
    padded = sorted(set(expected) - set(rename.values()))    # leftover -> auto-padded black
    rename_json = "{" + ", ".join(f'"{d}": "{e}"' for d, e in sorted(rename.items())) + "}"
    fix = (f"--rename_map='{rename_json}'"
           + (f"   (checkpoint cameras {padded} auto-pad black)" if padded else ""))
    return {"status": "mismatch", "ok": False, "provided": sorted(provided),
            "expected": sorted(expected), "missing": missing, "extra": extra,
            "rename_map": rename, "padded_cameras": padded, "fix": fix}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--policy-path", default=None, help="pretrained checkpoint (--policy.path)")
    ap.add_argument("--policy-type", default=None, help="from-scratch family (--policy.type); no camera constraint")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = check(args.dataset_repo_id, args.dataset_root, args.policy_path, args.policy_type)
    if args.json:
        print(json.dumps(r))
    elif r["status"] == "unknown":
        print(f"UNKNOWN: {r['detail']} — skipping the camera pre-check (not gating).", file=sys.stderr)
    elif r["ok"]:
        exp = r.get("expected")
        print(f"OK: dataset cameras cover the policy's {exp}." if exp
              else f"OK: --policy.type={args.policy_type or '?'} trains from scratch; adapts to the dataset's cameras.")
    else:
        print("CAMERA MISMATCH between dataset and pretrained policy:")
        print(f"  dataset has : {r['provided']}")
        print(f"  policy wants: {r['expected']}")
        print(f"  missing (policy needs, dataset lacks): {r['missing']}")
        print(f"  extra   (dataset has, policy ignores): {r['extra']}")
        print("\nFIX:\n" + r["fix"])
    return {"ok": 0, "mismatch": 2, "unknown": 3}[r["status"]]


if __name__ == "__main__":
    sys.exit(main())
