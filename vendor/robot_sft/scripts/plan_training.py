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
        [--num-workers auto] [--shm-gb 16] [--cuda 0] [--output-dir DIR] \
        [--runner uv|python-module] [--out plan.json] [--json]

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


_OBJECT_STORE_PREFIXES = ("tos://", "s3://", "gs://", "gcs://")
# Policies whose config exposes a `compile_model` field (so --policy.compile_model=true is valid).
_COMPILE_POLICIES = {"pi0", "pi05", "pi0_fast", "smolvla", "diffusion"}


def read_meta_counts(repo_id: str, root: str | None) -> tuple[int | None, int | None]:
    """(total_frames, total_episodes) from the dataset `meta/info.json`. Reads it locally
    (--dataset-root or $HF_LEROBOT_HOME/<repo_id>), or — for a `tos://…`/`s3://` repo_id —
    stream-reads it via fsspec (TOS creds from env; a few-KB read, no dataset download)."""
    info = None
    candidates = []
    if root:
        candidates.append(os.path.join(root, "meta", "info.json"))
    hf = os.environ.get("HF_LEROBOT_HOME",
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot"))
    candidates.append(os.path.join(hf, repo_id, "meta", "info.json"))
    for c in candidates:
        if os.path.isfile(c):
            info = json.load(open(c))
            break
    if info is None and repo_id.startswith(_OBJECT_STORE_PREFIXES):
        import fsspec  # lazy: only object-store repo_ids need it

        so: dict = {}
        if repo_id.startswith("tos://"):
            so = {"endpoint": os.environ.get("TOS_ENDPOINT", "https://tos-cn-beijing.volces.com"),
                  "region": os.environ.get("TOS_REGION", "cn-beijing")}
            if os.environ.get("TOS_ACCESS_KEY"):
                so["key"] = os.environ["TOS_ACCESS_KEY"]
            if os.environ.get("TOS_SECRET_KEY"):
                so["secret"] = os.environ["TOS_SECRET_KEY"]
        with fsspec.open(f"{repo_id.rstrip('/')}/meta/info.json", "r", **so) as f:
            info = json.load(f)
    if info is None:
        return None, None
    return (int(info["total_frames"]) if info.get("total_frames") else None,
            int(info["total_episodes"]) if info.get("total_episodes") else None)


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
    ap.add_argument("--runner", default="uv",
                    choices=["uv", "python-module"],
                    help="command runner: 'uv' → 'uv run lerobot-train', "
                         "'python-module' → 'python -u -m lerobot.scripts.lerobot_train'")
    ap.add_argument("--float8", action="store_true",
                    help="fp8 training via NVIDIA TransformerEngine (te.LayerNormMLP) for the VLM "
                         "MLP layers. pi0/pi05 ONLY — appends --policy.vlm_mlp_fp8_enable=true "
                         "--policy.dtype=bfloat16. HOPPER/ADA GPUs ONLY (H20/H100, sm_89/90+); on "
                         "older cards TE errors at runtime, so only pass it when check_hardware "
                         "reports an H20/Hopper. Needs the TE-enabled lerobot image. Errors for "
                         "non-pi0/pi05 policies. See references/policy_selection.md.")
    ap.add_argument("--float8-recipe", default="delayed_scaling",
                    choices=["delayed_scaling", "float8_block_scaling"],
                    help="TE fp8 recipe → --policy.vlm_mlp_fp8_recipe_kind. "
                         "delayed_scaling (default): per-tensor delayed scaling, 16-step amax "
                         "history — the TE analogue of the old torchao 'tensorwise'. "
                         "float8_block_scaling: block-wise (1D row-wise for activations/grads, "
                         "2D tiles for weights) — the TE analogue of the old torchao 'rowwise'.")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                    help="torch.compile the policy (--policy.compile_model=true). Default OFF: "
                         "MEASURED on pi05/H20 it gave NO steady-state speedup (1.61 vs 1.54 s/step) "
                         "while adding a ~5 min first-step warmup — not worth it for these VLA "
                         "models. Opt in for experiments. Only valid for pi0/pi05/pi0_fast/smolvla/"
                         "diffusion.")
    ap.add_argument("--compile-mode", default=None,
                    help="torch.compile mode (default reduce-overhead: fast warm-up, most of the "
                         "win). Use max-autotune for a long run to squeeze the last bit — but on a "
                         "4B VLA its kernel search warms up 10+ min and uses more GPU memory.")
    ap.add_argument("--rename-map", default=None, metavar="JSON",
                    help="camera-key rename passed THROUGH to lerobot-train as "
                         "--rename_map='<json>' (e.g. from check_features.py's fix). "
                         "This is a lerobot-train flag, not a planning input — accepted "
                         "here as a passthrough so the full command can be planned in one "
                         "call. Not needed with --policy.type (config derives from the "
                         "dataset); mainly for finetuning a checkpoint whose camera names "
                         "differ (preflight also auto-adds it when detectable).")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write JSON plan to this file (instead of stdout)")
    args = ap.parse_args()

    # Train/eval split (if provided): sizes the train set AND inlines the episode list below.
    train_eps, eval_eps = (None, None)
    if args.episodes_file:
        train_eps, eval_eps = load_train_episodes(args.episodes_file)

    # Auto-read total_frames / total_episodes from the dataset meta when counts aren't given —
    # local dir or streamed from a tos://…/meta/info.json (so the caller need not pass them).
    total_frames = total_episodes = None
    if args.samples is None or (args.episodes is None and train_eps is None):
        try:
            total_frames, total_episodes = read_meta_counts(args.dataset_repo_id, args.dataset_root)
        except Exception:  # noqa: BLE001 — fall back to explicit args / avg-len estimate
            total_frames = total_episodes = None

    episodes = args.episodes
    if episodes is None:
        episodes = len(train_eps) if train_eps is not None else total_episodes

    samples = args.samples
    if samples is None:
        if total_frames and total_episodes and train_eps is not None:
            samples = round(total_frames * len(train_eps) / total_episodes)  # train subset (≈uniform ep len)
        elif total_frames:
            samples = total_frames
        elif episodes:
            samples = episodes * args.avg_len
    if not samples or samples <= 0:
        ap.error("could not determine training frames — pass --samples (or --episodes), or make the "
                 "dataset meta readable (--dataset-root, or TOS creds for a tos:// repo_id)")

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

    out_dir = args.output_dir or os.path.join(
        _default_artifact_root(), "runs",
        f"{args.policy_type}_{_dt.datetime.now():%Y%m%d_%H%M%S}")

    # ---- build the lerobot-train command -----------------------------------
    if args.runner == "python-module":
        runner_cmd = "python -u -m lerobot.scripts.lerobot_train"
    else:
        runner_cmd = "uv run lerobot-train"
    parts = [runner_cmd]
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
    parts.append("--policy.push_to_hub=false")   # don't require a Hub repo_id / push weights
    parts.append(f"--output_dir={out_dir}")
    parts.append(f"--steps={steps}")
    parts.append(f"--batch_size={batch}")
    parts.append(f"--num_workers={num_workers}")
    parts.append(f"--save_freq={save_freq}")
    parts.append(f"--log_freq={args.log_freq}")
    parts.append("--env_eval_freq=0")        # eval is out-of-band (eval_watcher on held-out episodes); flag is env_eval_freq, NOT eval_freq
    parts.append("--wandb.enable=false")

    # Resolve the policy family (from --policy-type, or the checkpoint config's `type`) once — used
    # to gate both --float8 (pi0/pi05 only) and --compile (compile-capable policies only).
    import sys as _sys
    compile_family = args.policy_type
    if not compile_family and args.policy_path:
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import check_features as _cf
        compile_family = (_cf.policy_config(args.policy_path) or {}).get("type")

    # fp8 via TransformerEngine — pi0/pi05 ONLY (the only policies with the te.LayerNormMLP VLM-MLP
    # config fields). fp8 composes with bf16 autocast (master weights stay bf16), so set dtype too.
    # Needs a Hopper/Ada GPU at runtime + the TE-enabled lerobot image.
    if args.float8:
        if compile_family not in ("pi0", "pi05"):
            print(f"error: --float8 (TransformerEngine fp8) is only supported for pi0/pi05, not "
                  f"'{compile_family or 'unknown'}'. Re-run without --float8 (bf16).", file=_sys.stderr)
            _sys.exit(2)
        parts.append("--policy.dtype=bfloat16")
        parts.append("--policy.vlm_mlp_fp8_enable=true")
        parts.append(f"--policy.vlm_mlp_fp8_recipe_kind={args.float8_recipe}")

    # Camera-key rename: pure passthrough to lerobot-train (validate it's JSON so a typo fails
    # HERE with a clear message instead of deep inside lerobot's config parsing).
    if args.rename_map:
        try:
            json.loads(args.rename_map)
        except json.JSONDecodeError as e:
            print(f"error: --rename-map is not valid JSON ({e}): {args.rename_map}", file=_sys.stderr)
            _sys.exit(2)
        parts.append(f"--rename_map='{args.rename_map}'")

    # torch.compile — OFF by default (MEASURED no steady-state speedup on pi05/H20, ~5 min warmup).
    # Only pi0/pi05/pi0_fast/smolvla/diffusion expose a `compile_model` config field; passing it to
    # others (ACT, …) would be an unknown-flag error, so gate on the policy family.
    compile_supported = compile_family in _COMPILE_POLICIES
    compile_on = args.compile and compile_supported
    if compile_on:
        parts.append("--policy.compile_model=true")
        # Default to reduce-overhead, NOT the policies' own max-autotune default: on a 4B VLA,
        # max-autotune's exhaustive kernel search takes 10+ min to warm up (and more GPU mem),
        # while reduce-overhead compiles in ~a minute for most of the runtime win. Opt into the
        # exhaustive search with --compile-mode max-autotune on a long run.
        parts.append(f"--policy.compile_mode={args.compile_mode or 'reduce-overhead'}")
    elif args.compile and compile_family and not compile_supported:
        print(f"note: --compile skipped — policy '{compile_family}' has no compile_model field.",
              file=_sys.stderr)

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
        "float8": args.float8,  # TE fp8 (te.LayerNormMLP) for pi0/pi05 VLM MLP
        "float8_recipe": args.float8_recipe if args.float8 else None,
        "compile": compile_on,
        "compile_note": ("on" if compile_on
                         else f"off — policy '{compile_family}' can't compile" if args.compile
                         else "off (--no-compile)"),
        "cuda_visible_devices": args.cuda,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
        # resume (see references/lerobot_resume.md): checkpoints always carry full training
        # state, so any complete checkpoint is resumable via --resume=true.
        "resume_command": (
            f"cd {args.repo} && {cuda}{runner_cmd} --resume=true "
            f"--config_path={out_dir}/checkpoints/last/pretrained_model/train_config.json"),
        "launch_command": cmd,
    }
    if args.gpus > 1:
        plan["multi_gpu_note"] = ("for multi-GPU use `uv run accelerate launch --num_processes="
                                  f"{args.gpus} $(which lerobot-train) ...` with the same flags; "
                                  "batch_size is per process")

    # Camera pre-check when finetuning a pretrained VLA. If the dataset's cameras don't match the
    # checkpoint's, AUTO-ADD a --rename_map so the finetune just works (pretrained weights kept;
    # any leftover checkpoint camera auto-pads black at runtime) — instead of handing back a
    # command that crashes in make_policy. preflight re-checks. (Read from two config JSONs.)
    if args.policy_path:
        import sys as _sys

        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import check_features
        fc = check_features.check(args.dataset_repo_id, args.dataset_root, args.policy_path, args.policy_type)
        if fc["status"] == "mismatch" and fc.get("rename_map"):
            rename_flag = f"--rename_map='{json.dumps(fc['rename_map'])}'"
            cmd = cmd + " \\\n  " + rename_flag
            plan["launch_command"] = cmd
            plan["camera_rename_map"] = fc["rename_map"]
            plan["camera_padded"] = fc.get("padded_cameras")
            print(f"\n⚠ camera mismatch (dataset {fc['provided']} vs checkpoint {fc['expected']}) "
                  f"auto-fixed: added {rename_flag}"
                  + (f"; {fc['padded_cameras']} auto-pad black." if fc.get("padded_cameras") else "."),
                  file=_sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"wrote {args.out}", file=__import__("sys").stderr)

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"samples={samples}  epochs={epochs}  batch={batch}"
              f"{' x ' + str(args.gpus) + ' gpu = ' + str(global_batch) if args.gpus > 1 else ''}")
        print(f"steps_per_epoch={steps_per_epoch}  -> steps={steps}")
        print(f"save_freq={save_freq}  log_freq={args.log_freq}  (checkpoints keep full "
              f"training state -> always resumable)")
        print(f"num_workers={num_workers}  ({workers_note})")
        print(f"fp8(TE)={'on (' + args.float8_recipe + ')' if args.float8 else 'off'}  "
              f"compile={plan['compile_note']}")
        if train_eps is not None:
            print(f"train episodes: {len(train_eps)}  held-out eval episodes: "
                  f"{len(eval_eps or [])}")
        print("\n# launch command:\n" + cmd)
        print("\n# resume after any stop:\n" + plan["resume_command"])


if __name__ == "__main__":
    main()
