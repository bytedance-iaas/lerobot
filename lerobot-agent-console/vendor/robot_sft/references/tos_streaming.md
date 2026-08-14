# Streaming a dataset from TOS (object storage)

Only relevant when the dataset lives in TOS rather than on local disk / the Hub.

For a dataset too large to download, stream it. lerobot's stock `StreamingLeRobotDataset`
streams only from the **HF Hub** or a **local dir** — not from `tos://`/`s3://` (it
Path-mangles the URL and passes no credentials). So for a dataset on **Volcengine TOS**,
use **`lerobot.datasets.StreamingTOSRobotDataset`** (a `StreamingLeRobotDataset` subclass in
the lerobot package), which adds the three fsspec seams: metadata is mirrored locally (the
tiny `meta/`), low-dim parquet streams via `fsspec`, and video mp4s are decoded **directly off
fsspec** (`fsspec.open` → torchcodec range-reads only the bytes it needs). It reads TOS
credentials from the environment, so you just pass the `tos://` URL. Validated live against a
v3.0 dataset on TOS (metadata mirror + parquet streaming + bit-exact video decode).

**1. Put a dataset on TOS** (LeRobot **v3.0** layout — `meta/ data/ videos/`). Upload with
`tosutil` (already configured on the box):
```bash
# download from the Hub (or record with lerobot-record), then push the tree to TOS:
tosutil cp <local_dataset_dir> tos://<bucket>/<prefix>/<name> -r -flat -f
# verify it kept the LeRobot tree:
tosutil ls tos://<bucket>/<prefix>/<name>/ -s | grep -E 'meta/info.json|data/.*parquet|videos/'
```

**2. Credentials.** Pass **real** TOS Access Key / Secret Key — via env vars, never
hardcoded:
```bash
export TOS_ACCESS_KEY=<AKLT...>   TOS_SECRET_KEY=<...>
export TOS_ENDPOINT=https://tos-cn-beijing.volces.com   TOS_REGION=cn-beijing
```
⚠️ **The `~/.tosutilconfig` ak/sk are OBFUSCATED** (they don't start with `AKLT…` and 403
if used raw) — `tosutil` de-obfuscates them internally, but the Python SDK / `tosfs` do NOT.
Use the plaintext AK/SK from the Volcengine console / IAM, installed with
`multinode.py env set TOS_ACCESS_KEY` (see Credentials) — never by editing an rc file.

**3. The TOS fsspec impl (`tosfs`) is pre-installed** in the console image (registers the
`tos://` protocol + TOS SDK), so `StreamingTOSRobotDataset` works out of the box. Only if you
hit `ImportError: Install tosfs …` (e.g. a non-console env): `cd /lerobot && uv pip install
--native-tls tosfs`.

**4. Open it** — just pass the `tos://` URL; credentials are read from the environment
(`repo_id` optional, auto-derived from the URL):
```python
from lerobot.datasets import StreamingTOSRobotDataset

ds = StreamingTOSRobotDataset(
    "tos://<bucket>/<prefix>/<dataset>",
    episodes=[0, 3, 17],           # held-out subset; omit for the whole dataset
)                                  # storage_options={...} only to override the env creds
print(ds.num_frames, ds.num_episodes, ds.fps, ds.meta.camera_keys)
for item in ds:                    # IterableDataset: iterate, no ds[i]
    item["observation.images.front"]   # (C,H,W); item["observation.state"], item["action"]
    break
```

**Scope:** `StreamingTOSRobotDataset` backs both the standalone readers (dataset exploration,
`offline_eval`) and `lerobot-train` itself — `make_dataset` recognises a `tos://` repo_id, so
training straight off TOS is the default path, not a workaround. Downloading + `--dataset.root`
remains available if you want the data local.

Notes: it's an **`IterableDataset`** (buffer-shuffled, no random index) — same trade-offs as
`StreamingLeRobotDataset` (lessons_learned #13 caveats). Video decode needs **torchcodec**
(present in the lerobot image; missing on a bare Mac). Validated end-to-end against a v3.0
dataset (metadata mirror + parquet streaming + episode filter + bit-exact video-frame alignment
vs the non-streaming reader).

### Training on a TOS dataset

`make_dataset` now recognizes a `tos://` URL, so **`lerobot-train` streams from TOS directly** —
no download, no custom loop. Pass the `tos://` URL as `--dataset.repo_id`; it auto-forces
`--dataset.streaming` and builds `StreamingTOSRobotDataset` (TOS creds from env). The training
loop, checkpoints (`pretrained_model/` + `training_state/`), resume, watchdog, and
`offline_eval` all work unchanged. **This is the default path for a TOS dataset.**
```bash
cd /lerobot && HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=<gpu> python -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=tos://<bucket>/<prefix>/<name> \
  --policy.type=act --policy.push_to_hub=false \
  --dataset.episodes="[<train ids>]" --env_eval_freq=0 \
  --output_dir=<run_dir> --steps=<N> --batch_size=<B> --num_workers=<W> --save_freq=<F> --wandb.enable=false
```
- Creds are read from env (`TOS_ACCESS_KEY`/`TOS_SECRET_KEY`[/`TOS_ENDPOINT`/`TOS_REGION`]),
  installed with `multinode.py env set` (see Credentials) — every shell then has them, on every
  node. `tosfs` is baked into the image.
- Verified end-to-end: a real `lerobot-train` run on a `tos://` dataset spanning **both** video
  files (ep0 in file-000, ep40 in file-001) trained and checkpointed correctly. **`plan_training.py`
  emits the launch/resume commands as usual — just with the `tos://` repo_id.**
- **Streaming caveats (as for any `--dataset.streaming`):** it's an `IterableDataset` →
  buffer-shuffled, **no** `EpisodeAwareSampler` / `drop_n_last_frames` (those need random access),
  and `--num_workers=0` isn't supported for streaming (use `>=1`).

**Alternative — download once + `--dataset.root`** (for a small dataset, or to avoid streaming):
copy it to the PVC and train non-streaming — identical to any local dataset.
```bash
python -c "import fsspec; fsspec.filesystem('tos').get('<bucket>/<prefix>/<name>', '/opt/data/datasets/<name>', recursive=True)"
lerobot-train --dataset.repo_id=<name> --dataset.root=/opt/data/datasets/<name> --policy.type=act ...
```

**Frame alignment** was an upstream `StreamingLeRobotDataset` bug (global vs file-relative video
timestamps) that broke multi-video-file datasets — **fixed** (see lessons_learned #18); streaming
is now bit-exact vs the non-streaming reader. Still spot-check a new/unusual dataset if in doubt.

