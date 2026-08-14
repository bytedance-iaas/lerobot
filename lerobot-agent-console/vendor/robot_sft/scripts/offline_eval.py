#!/usr/bin/env python3
"""Open-loop offline eval of a lerobot checkpoint on held-out episodes.

lerobot-eval needs a simulator env; for real-robot datasets there is none. The honest
offline signal is OPEN-LOOP REPLAY on episodes the model never trained on: feed the
recorded observations frame-by-frame (resetting the policy per episode so action-chunking
queues behave like deployment), predict an action per frame, and compare with the recorded
action. Low MSE on *held-out* episodes = generalization; on train episodes it only proves
memorization (lessons_learned #13).

Run it from the lerobot checkout (/lerobot in the console pod) so its venv resolves:
    cd /lerobot && python <skill>/scripts/offline_eval.py \
        --model-path <output_dir>/checkpoints/<N>/pretrained_model \
        --dataset-repo-id user/so101_pick [--dataset-root DIR] \
        --episodes 3 17 41 [--device cuda] [--max-frames-per-episode 450] \
        [--plot-dir <session>/eval/artifacts/ckpt-N/<ds>]

Prints per-trajectory and average MSE/MAE lines (parsed by eval_watcher.py) plus a final
JSON line, and (with --plot-dir + matplotlib) saves a gt-vs-pred plot per episode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# MUST happen before anything imports huggingface_hub / transformers: both read HF_HUB_OFFLINE
# (and TRANSFORMERS_OFFLINE) at IMPORT time into module constants, so setting them inside main()
# is too late — the first attempt at this did exactly that and the eval still spent 8m35s dying
# on "Network is unreachable".
#
# Offline is the right default here: by the time a checkpoint is being evaluated its tokenizer
# and backbone are necessarily already in the HF cache — the training that produced it could not
# have run otherwise. Left online, the Hub is contacted to revalidate files that are already
# local, and this pod cannot reach huggingface.co, so every call waits out a TCP timeout and
# then fails. Parsed off sys.argv because argparse has not run yet.
if "--online" not in sys.argv:
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(_k, "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="a checkpoint's pretrained_model/ dir (or any pretrained policy dir)")
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", required=True,
                    help="held-out episode ids to replay")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames-per-episode", type=int, default=450,
                    help="cap per-episode replay length (whole episode if shorter)")
    # Without this, a run that trained with --rename_map could never be evaluated: the
    # checkpoint expects the RENAMED keys (observation.images.base_0_rgb …) while the dataset
    # still has the originals (observation.images.front …), and make_policy dies with
    # "Feature mismatch … Missing features / Extra features". That is exactly the finetune-a-
    # pretrained-VLA case, i.e. the runs where eval matters most. lerobot-train passes the map
    # in two places; so do we (see below).
    ap.add_argument("--online", action="store_true",
                    help="allow HF Hub network access (default: offline — everything needed is "
                         "already cached, and reaching the Hub from this pod just times out)")
    ap.add_argument("--rename-map", default=None, metavar="JSON",
                    help='dataset->policy observation key map, same JSON as lerobot-train\'s '
                         '--rename_map, e.g. \'{"observation.images.front": '
                         '"observation.images.base_0_rgb"}\'. Take it from the plan\'s '
                         "camera_rename_map.")
    ap.add_argument("--plot-dir", default=None, help="save gt-vs-pred plots here (needs matplotlib)")
    args = ap.parse_args()

    import torch  # deferred so --help works without the venv

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[offline_eval] CUDA unavailable -> {device}", file=sys.stderr)

    rename_map = None
    if args.rename_map:
        try:
            rename_map = json.loads(args.rename_map)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--rename-map is not valid JSON: {e}") from e
        if not isinstance(rename_map, dict):
            raise SystemExit("--rename-map must be a JSON object of dataset_key -> policy_key")
        print(f"[offline_eval] rename_map: {rename_map}")

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    cfg.pretrained_path = args.model_path
    cfg.device = device

    eval_eps = sorted(set(args.episodes))
    if args.dataset_repo_id.startswith(("tos://", "s3://", "gs://", "gcs://")):
        # Stream held-out episodes from object storage (no download). Two things the streaming
        # dataset forces us to handle for correct OPEN-LOOP replay (frames must arrive in
        # temporal order within an episode):
        #   - it ALWAYS buffer-shuffles (shuffle=False only makes it deterministic); buffer_size=1
        #     disables the buffer -> source/frame order.
        #   - multiple episodes interleave in one stream -> iterate ONE episode at a time.
        # Its .meta feeds make_policy; return_uint8 default (False) matches the local path so the
        # checkpoint's preprocessor gets images in the range it expects.
        import tempfile

        from lerobot.datasets import StreamingTOSRobotDataset

        _meta_cache = tempfile.mkdtemp(prefix="offline_eval_meta_")

        def _stream(eps):
            return StreamingTOSRobotDataset(args.dataset_repo_id, episodes=eps, shuffle=False,
                                            buffer_size=1, meta_cache_dir=_meta_cache)

        ds_meta = _stream(eval_eps).meta

        def _tos_items():
            for ep in eval_eps:
                yield from _stream([ep])

        item_iter = _tos_items()
        n_frames = "streaming"
    else:
        ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=eval_eps)
        item_iter = (dataset[i] for i in range(len(dataset)))
        n_frames = dataset.num_frames

    # Same two places lerobot_train.py applies it: make_policy reconciles the feature names,
    # and the preprocessor's rename step remaps each incoming batch.
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename_map)
    policy.eval()
    _pp_kwargs = {}
    if rename_map:
        _pp_kwargs["preprocessor_overrides"] = {
            "rename_observations_processor": {"rename_map": rename_map}}
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=args.model_path, **_pp_kwargs)

    print(f"[offline_eval] model={args.model_path} device={device} "
          f"episodes={eval_eps} frames={n_frames}")

    # Replay each episode sequentially. Items of the selected episodes come back in order;
    # a change of episode_index marks a boundary -> reset the policy (fresh action queue).
    per_ep: dict[int, list[tuple[float, float]]] = {}
    cur_ep, ep_frames = None, 0
    gt_hist: dict[int, list] = {}
    pred_hist: dict[int, list] = {}
    with torch.inference_mode():
        for item in item_iter:
            ep = int(item["episode_index"])
            if ep != cur_ep:
                cur_ep, ep_frames = ep, 0
                policy.reset()
            ep_frames += 1
            if ep_frames > args.max_frames_per_episode:
                continue
            gt = item["action"]
            batch = {}
            for k, v in item.items():
                if k == "action" or not (k.startswith("observation.") or k == "task"):
                    continue
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
                else:
                    batch[k] = [v]           # e.g. the task string
            batch = preprocessor(batch)
            action = postprocessor(policy.select_action(batch))
            pred = action.squeeze(0).detach().float().cpu()
            gt = gt.detach().float().cpu()
            n = min(pred.numel(), gt.numel())  # guard vs shape drift between policy/dataset
            diff = pred.flatten()[:n] - gt.flatten()[:n]
            per_ep.setdefault(ep, []).append((float((diff ** 2).mean()), float(diff.abs().mean())))
            if args.plot_dir:
                gt_hist.setdefault(ep, []).append(gt.flatten()[:n].tolist())
                pred_hist.setdefault(ep, []).append(pred.flatten()[:n].tolist())

    if not per_ep:
        print("ERROR: no frames evaluated (bad episode ids?)", file=sys.stderr)
        sys.exit(2)

    mses, maes = [], []
    for ep in sorted(per_ep):
        pairs = per_ep[ep]
        mse = sum(p[0] for p in pairs) / len(pairs)
        mae = sum(p[1] for p in pairs) / len(pairs)
        mses.append(mse)
        maes.append(mae)
        print(f"MSE for trajectory {ep}: {mse:.6f}, MAE: {mae:.6f}  ({len(pairs)} frames)")

    mean_mse = sum(mses) / len(mses)
    mean_mae = sum(maes) / len(maes)
    print(f"Average MSE across all trajs: {mean_mse:.6f}")
    print(f"Average MAE across all trajs: {mean_mae:.6f}")
    print(json.dumps({"mean_mse": mean_mse, "mean_mae": mean_mae,
                      "episodes": sorted(per_ep), "model_path": args.model_path}))

    if args.plot_dir and gt_hist:
        try:
            import os

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(args.plot_dir, exist_ok=True)
            for ep in sorted(gt_hist):
                g = list(zip(*gt_hist[ep]))    # per-dim series
                p = list(zip(*pred_hist[ep]))
                dims = len(g)
                fig, axes = plt.subplots(dims, 1, figsize=(9, 1.6 * dims), sharex=True)
                axes = [axes] if dims == 1 else list(axes)
                for d in range(dims):
                    axes[d].plot(g[d], label="gt", lw=1)
                    axes[d].plot(p[d], label="pred", lw=1)
                    axes[d].set_ylabel(f"a[{d}]", fontsize=7)
                axes[0].legend(fontsize=7)
                axes[0].set_title(f"episode {ep} — gt vs pred")
                fig.tight_layout()
                fig.savefig(os.path.join(args.plot_dir, f"traj_{ep}.png"), dpi=90)
                plt.close(fig)
            print(f"[offline_eval] plots -> {args.plot_dir}")
        except Exception as e:  # noqa: BLE001  (plots are best-effort, never fail the eval)
            print(f"[offline_eval] plotting skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
