#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
split_train_eval.py -- deterministic train/eval splitter for LeRobot v2.1 datasets.

This is part of the ``robot_sft`` skill. It takes one already-converted LeRobot
v2.1 dataset (the kind GR00T's ``LeRobotEpisodeLoader`` opens) and writes two new,
fully self-contained v2.1 datasets: a ``train/`` (kept episodes) and an ``eval/``
(held-out episodes). The source dataset is never modified.

WHY RE-INDEXING IS REQUIRED
---------------------------
A LeRobot v2.1 dataset addresses each episode by its integer ``episode_index``.
GR00T's loader resolves the parquet/video files for episode i as::

    chunk_idx = episode_index // chunks_size
    data:  data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet
    video: videos/chunk-{chunk_idx:03d}/{video_key}/episode_{episode_index:06d}.mp4

and it lists ``task_index -> task`` from ``meta/tasks.jsonl`` plus per-episode
metadata (``length``) from ``meta/episodes.jsonl``. So a valid split CANNOT just
copy a subset of files keeping their original numbers (that would leave holes like
episode 0,3,7,... -> chunk math + episodes.jsonl mismatch). Every selected episode
must be RENUMBERED to a contiguous 0..M-1 range, and every place that records an
episode id, a task id, or a global frame counter must be updated consistently.

WHAT THIS SCRIPT REWRITES (for BOTH outputs)
--------------------------------------------
For each split we map old_episode_index -> new_episode_index (0..M-1, in the order
the episodes are kept). Then:

* data parquet:  copied to data/chunk-{new//chunks_size:03d}/episode_{new:06d}.parquet
  with columns rewritten:
    - episode_index : set to new_episode_index (constant per file)
    - frame_index   : 0..L-1 (per episode; left as-is, already contiguous)
    - index         : reassigned to a split-global running counter so rows are
                      contiguous 0..total_frames-1 across the whole split
    - task_index    : remapped through old_task_index -> new_task_index
* videos: every camera mp4 present on disk for the episode is copied to
  videos/chunk-{new//chunks_size:03d}/{cam}/episode_{new:06d}.mp4
  (we copy ALL camera dirs found, not only the ones referenced by modality.json,
  so the split stays a faithful v2.1 dataset).
* meta/episodes.jsonl : one line per kept episode, episode_index=new, original
  ``tasks`` list and ``length`` preserved, sorted by new index.
* meta/tasks.jsonl    : only tasks actually referenced by the kept episodes,
  re-indexed to a dense 0..T-1 range (task strings preserved). The same old->new
  task map is applied to the parquet ``task_index`` column above.
* meta/info.json      : total_episodes, total_frames, total_chunks recomputed;
  splits set to {"train": "0:M"} / {"eval": "0:M"}; total_videos recomputed;
  everything else (features, fps, chunks_size, path templates, robot_type) kept.
* meta/modality.json  : copied verbatim (the joint layout is identical).
* meta/stats.json     : copied verbatim. This is the dataset-level aggregate used
  for normalization; the loader REQUIRES it to exist. Recomputing exact per-split
  aggregates would require re-reading every frame; the global distribution is a
  safe, standard normalization basis for a 90/10 split. (Noted as a caveat.)
* meta/episodes_stats.jsonl (if present) : subset to the kept episodes, with each
  line's ``episode_index`` remapped to new. The numeric value stats (state/action/
  image min/max/mean/std/quantiles) are intrinsic to the episode content and are
  copied unchanged; only the episode_index key is remapped.

Dependency-light: pandas, pyarrow (via pandas), shutil, json, random, argparse.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _read_jsonl(path: Path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _aggregate_stats(ep_stats_lines: list[dict]) -> dict:
    """Aggregate per-episode stats (episodes_stats.jsonl entries) into a single
    dataset-level stats.json dict, matching lerobot's aggregation semantics:

    * count : summed
    * min   : element-wise min across episodes
    * max   : element-wise max across episodes
    * mean  : count-weighted average of per-episode means
    * std   : combined std via parallel-variance (within + between episode var)
    * q01..q99 : count-weighted average of per-episode quantiles (approximation,
                 since exact global quantiles can't be recovered from per-episode
                 quantiles -- this mirrors lerobot's own approach)

    Used as a fallback when the source dataset has no meta/stats.json (the GR00T
    loader requires that file). Stats are computed over ONLY the episodes in the
    split, so this is a faithful per-split normalization basis.
    """
    feature_keys = list(ep_stats_lines[0]["stats"].keys())
    out: dict = {}
    for fk in feature_keys:
        counts = np.array([np.asarray(l["stats"][fk]["count"], dtype=np.float64)
                           for l in ep_stats_lines])  # (E, 1)
        total = counts.sum(axis=0)  # (1,)

        def stack(name):
            return np.array([np.asarray(l["stats"][fk][name], dtype=np.float64)
                             for l in ep_stats_lines])

        mins = stack("min")
        maxs = stack("max")
        means = stack("mean")
        stds = stack("std")

        # broadcast counts to the feature's shape
        c = counts.reshape((counts.shape[0],) + (1,) * (means.ndim - 1))
        w = c / total.reshape((1,) + (1,) * (means.ndim - 1))

        agg_mean = (means * w).sum(axis=0)
        # combined variance = E[var] + Var[mean] (count-weighted)
        within = (stds**2 * w).sum(axis=0)
        between = (((means - agg_mean) ** 2) * w).sum(axis=0)
        agg_std = np.sqrt(within + between)

        res = {
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
            "mean": agg_mean.tolist(),
            "std": agg_std.tolist(),
            "count": [int(total.reshape(-1)[0])],
        }
        for q in ("q01", "q10", "q50", "q90", "q99"):
            if q in ep_stats_lines[0]["stats"][fk]:
                vals = stack(q)
                res[q] = (vals * w).sum(axis=0).tolist()
        out[fk] = res
    return out


def _discover_camera_dirs(src: Path) -> list[str]:
    """Return the camera subdir names present under videos/chunk-*/<cam>/."""
    cams = set()
    videos = src / "videos"
    if not videos.is_dir():
        return []
    for chunk_dir in videos.glob("chunk-*"):
        for cam_dir in chunk_dir.iterdir():
            if cam_dir.is_dir():
                cams.add(cam_dir.name)
    return sorted(cams)


def _build_one_split(
    src: Path,
    out: Path,
    selected_old_eps: list[int],
    info: dict,
    episodes_meta: list[dict],
    tasks_rows: list[dict],
    ep_stats: list[dict] | None,
    cams: list[str],
    split_name: str,
):
    """Write a complete v2.1 dataset to `out` containing `selected_old_eps`.

    `selected_old_eps` is the list of ORIGINAL episode indices to keep, in the
    order they should be renumbered (0..M-1)."""
    chunks_size = info["chunks_size"]
    data_tmpl = info["data_path"]
    video_tmpl = info.get("video_path")

    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    # old episode index -> per-episode metadata line
    ep_meta_by_idx = {int(e["episode_index"]): e for e in episodes_meta}
    ep_stats_by_idx = (
        {int(s["episode_index"]): s for s in ep_stats} if ep_stats is not None else None
    )

    old2new_ep = {old: new for new, old in enumerate(selected_old_eps)}

    # First pass over selected episodes' parquet to find referenced task indices.
    referenced_tasks: set[int] = set()
    parquet_cache: dict[int, pd.DataFrame] = {}
    for old in selected_old_eps:
        old_chunk = old // chunks_size
        p = src / data_tmpl.format(episode_chunk=old_chunk, episode_index=old)
        df = pd.read_parquet(p)
        parquet_cache[old] = df
        if "task_index" in df.columns:
            referenced_tasks.update(int(t) for t in df["task_index"].unique())

    # Re-index tasks: dense 0..T-1 over referenced tasks (stable by old index).
    tasks_by_old = {int(t["task_index"]): t["task"] for t in tasks_rows}
    sorted_old_tasks = sorted(referenced_tasks)
    old2new_task = {old: new for new, old in enumerate(sorted_old_tasks)}
    new_tasks_rows = [
        {"task_index": new, "task": tasks_by_old[old]}
        for old, new in sorted((o, n) for o, n in old2new_task.items())
    ]
    _write_jsonl(out / "meta" / "tasks.jsonl", new_tasks_rows)

    # Second pass: copy parquet (re-indexed) + videos, build episodes.jsonl/stats.
    new_episodes_meta = []
    new_ep_stats = [] if ep_stats_by_idx is not None else None
    running_index = 0
    total_frames = 0
    max_new_chunk = 0
    total_videos = 0

    for old in selected_old_eps:
        new = old2new_ep[old]
        new_chunk = new // chunks_size
        max_new_chunk = max(max_new_chunk, new_chunk)

        df = parquet_cache[old].copy()
        n = len(df)
        df["episode_index"] = new
        if "frame_index" in df.columns:
            df["frame_index"] = list(range(n))
        if "index" in df.columns:
            df["index"] = list(range(running_index, running_index + n))
        if "task_index" in df.columns:
            df["task_index"] = df["task_index"].map(lambda x: old2new_task[int(x)]).astype(
                df["task_index"].dtype
            )
        running_index += n
        total_frames += n

        out_parquet = out / data_tmpl.format(episode_chunk=new_chunk, episode_index=new)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

        # videos: copy every camera mp4 that exists for this episode.
        if video_tmpl:
            for cam in cams:
                src_v = src / video_tmpl.format(
                    episode_chunk=old // chunks_size, video_key=cam, episode_index=old
                )
                if src_v.exists():
                    dst_v = out / video_tmpl.format(
                        episode_chunk=new_chunk, video_key=cam, episode_index=new
                    )
                    dst_v.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_v, dst_v)
                    total_videos += 1

        # episodes.jsonl line (preserve original tasks list + length, use df length)
        src_line = ep_meta_by_idx[old]
        new_episodes_meta.append(
            {
                "episode_index": new,
                "tasks": src_line.get("tasks", []),
                "length": int(src_line.get("length", n)),
            }
        )

        # episodes_stats.jsonl line (remap episode_index, keep value stats)
        if new_ep_stats is not None and old in ep_stats_by_idx:
            s = json.loads(json.dumps(ep_stats_by_idx[old]))  # deep copy
            s["episode_index"] = new
            new_ep_stats.append(s)

    new_episodes_meta.sort(key=lambda e: e["episode_index"])
    _write_jsonl(out / "meta" / "episodes.jsonl", new_episodes_meta)
    if new_ep_stats is not None:
        new_ep_stats.sort(key=lambda e: e["episode_index"])
        _write_jsonl(out / "meta" / "episodes_stats.jsonl", new_ep_stats)

    # info.json
    new_info = json.loads(json.dumps(info))
    m = len(selected_old_eps)
    new_info["total_episodes"] = m
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(new_tasks_rows)
    new_info["total_chunks"] = max_new_chunk + 1
    new_info["total_videos"] = total_videos
    new_info["splits"] = {split_name: f"0:{m}"}
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # modality.json verbatim
    shutil.copy2(src / "meta" / "modality.json", out / "meta" / "modality.json")

    # stats.json: the GR00T loader REQUIRES this file.
    # Preferred: copy the source's dataset-level stats.json verbatim (global
    # aggregate is a valid normalization basis for a 90/10 split).
    # Fallback: if the source has none but per-episode stats exist, aggregate a
    # stats.json from exactly the kept episodes (faithful per-split stats).
    stats_src = src / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, out / "meta" / "stats.json")
    elif new_ep_stats:
        agg = _aggregate_stats(new_ep_stats)
        with open(out / "meta" / "stats.json", "w") as f:
            json.dump(agg, f, indent=4)
    else:
        raise FileNotFoundError(
            f"Source {src} has no meta/stats.json and no episodes_stats.jsonl to "
            f"aggregate from; cannot produce a GR00T-loadable split."
        )

    return {
        "out": str(out),
        "n_episodes": m,
        "total_frames": total_frames,
        "total_chunks": max_new_chunk + 1,
        "n_tasks": len(new_tasks_rows),
        "n_videos": total_videos,
        "old2new_ep": old2new_ep,
    }


def split_dataset(
    src: str,
    out_train: str,
    out_eval: str,
    eval_frac: float = 0.1,
    eval_count: int | None = None,
    seed: int = 42,
):
    src = Path(src)
    out_train = Path(out_train)
    out_eval = Path(out_eval)

    meta = src / "meta"
    with open(meta / "info.json") as f:
        info = json.load(f)
    episodes_meta = _read_jsonl(meta / "episodes.jsonl")
    tasks_rows = _read_jsonl(meta / "tasks.jsonl")
    ep_stats = (
        _read_jsonl(meta / "episodes_stats.jsonl")
        if (meta / "episodes_stats.jsonl").exists()
        else None
    )
    cams = _discover_camera_dirs(src)

    all_old_eps = sorted(int(e["episode_index"]) for e in episodes_meta)
    total = len(all_old_eps)

    if eval_count is not None:
        k = int(eval_count)
    else:
        k = max(1, round(eval_frac * total))
    if k >= total:
        raise ValueError(f"eval count {k} >= total episodes {total}")

    # Deterministic seeded shuffle -> first k are eval.
    rng = random.Random(seed)
    shuffled = list(all_old_eps)
    rng.shuffle(shuffled)
    eval_old = sorted(shuffled[:k])
    train_old = sorted(shuffled[k:])

    assert set(eval_old).isdisjoint(train_old)
    assert len(eval_old) + len(train_old) == total

    print(f"[split] src={src}")
    print(f"[split] total={total}  train={len(train_old)}  eval={len(eval_old)}  seed={seed}")
    print(f"[split] eval original episode ids: {eval_old}")
    print(f"[split] cameras on disk: {cams}")

    train_res = _build_one_split(
        src, out_train, train_old, info, episodes_meta, tasks_rows, ep_stats, cams, "train"
    )
    eval_res = _build_one_split(
        src, out_eval, eval_old, info, episodes_meta, tasks_rows, ep_stats, cams, "eval"
    )

    summary = {
        "src": str(src),
        "total_episodes": total,
        "seed": seed,
        "eval_count": k,
        "eval_orig_episode_ids": eval_old,
        "train_orig_episode_ids": train_old,
        "train": train_res,
        "eval": eval_res,
    }
    print(f"[split] DONE  train -> {out_train} ({train_res['n_episodes']} ep, "
          f"{train_res['total_frames']} frames)  eval -> {out_eval} "
          f"({eval_res['n_episodes']} ep, {eval_res['total_frames']} frames)")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Deterministic train/eval splitter for LeRobot v2.1 datasets."
    )
    ap.add_argument("--src", required=True, help="Source LeRobot v2.1 dataset dir.")
    ap.add_argument("--out-train", required=True, help="Output dir for the train split.")
    ap.add_argument("--out-eval", required=True, help="Output dir for the eval split.")
    ap.add_argument("--eval-frac", type=float, default=0.1, help="Holdout fraction (default 0.1).")
    ap.add_argument("--eval-count", type=int, default=None,
                    help="Exact number of held-out episodes (overrides --eval-frac).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42).")
    args = ap.parse_args()

    split_dataset(
        src=args.src,
        out_train=args.out_train,
        out_eval=args.out_eval,
        eval_frac=args.eval_frac,
        eval_count=args.eval_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
