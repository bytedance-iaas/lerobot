# lerobot-train entrypoint quirks

Concrete differences between the `lerobot-train` CLI and the GR00T `finetune.sh` entrypoint
that the `plan_training.py` / `preflight.py` scripts were originally built for.

## Command format

| Aspect | `lerobot-train` (lerobot) | `finetune.sh` (GR00T) |
|--------|--------------------------|----------------------|
| Steps | `--steps=N` | `MAX_STEPS=N` (env var) |
| Save cadence | `--save_freq=N` | `SAVE_STEPS=N` (env var) |
| Batch size | `--batch_size=N` | `GLOBAL_BATCH_SIZE=N` (env var) |
| GPU count | `CUDA_VISIBLE_DEVICES=X` | `NUM_GPUS=N` (env var) |
| Workers | `--num_workers=N` | `DATALOADER_NUM_WORKERS=N` (env var) |
| Dataset | `--dataset.repo_id=org/name` | `--dataset-path /path/to/dir` |
| Policy (fresh) | `--policy.type=pi05` | `--base-model-path /path` |
| Policy (finetune) | `--policy.path=org/name` | same |
| Output dir | `--output_dir=/path` | `--output-dir /path` |
| Resume | `--resume=true --config_path=<dir>/checkpoints/last/pretrained_model/train_config.json` | re-run same command (auto-resumes) |

## Fresh-init `--policy.type` requires `--policy.push_to_hub=false`

When using `--policy.type=pi05` (or any policy type) for fresh training — not loading
from a pretrained checkpoint via `--policy.path` — the `PreTrainedConfig.push_to_hub`
defaults to `True`. Validation then fails because `repo_id` is `None`:

```
ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.
```

**Fix:** always add `--policy.push_to_hub=false` when training from scratch with
`--policy.type=<name>`.

## pi05 / pi0 gated backbone

pi05 (and pi0) hardcode their tokenizer to `google/paligemma-3b-pt-224` in
`src/lerobot/policies/pi05/processor_pi05.py:152`. This is a **gated** HuggingFace
model — downloading it requires:

1. A valid HF token (`hf auth login` or `HF_TOKEN` env var)
2. The token's account must have **accepted the license** for
   `google/paligemma-3b-pt-224` on huggingface.co

Without this, training fails immediately with `403 Forbidden` during `AutoTokenizer.from_pretrained()`.
The `paligemma_variant` config option (`gemma_2b` vs `gemma_300m`) only affects the action
expert size — the tokenizer/vision backbone is always the same gated model.

This is a **fatal** blocker (lessons_learned #1). Do not proceed to stage e until the
Paligemma model is accessible (either via token, or pre-downloaded to local cache).

## `/dev/shm` remount can silently fail

Some container runtimes fix `/dev/shm` size at creation. `mount -o remount,size=16g /dev/shm`
may return "write-protected" or succeed but not actually change the size. **Always**
re-check `df -h /dev/shm` after attempting remount. If it's still 64 MB, fall back to
`--num_workers=0` and record why.

## Output dir must not exist

`lerobot-train` refuses to start if `--output_dir` already exists and `--resume` is not
`True`. This is checked early in validation (before any model loading), so it's a cheap
failure. The preflight smoke test must use a fresh temp dir each time.

## Dataset access: LeRobotDataset vs plain HF dataset

`lerobot-train --dataset.repo_id=<name>` expects a **LeRobot-format** dataset (with
`meta/info.json`, `data/chunk-*/file-*.parquet`, `videos/`). A plain HuggingFace
dataset (loaded via `datasets.load_dataset()`) won't work — it must be in LeRobot v3.0
layout.

To check if a HF repo is LeRobot-format: try `huggingface_hub.hf_hub_download(repo_id,
'meta/info.json', repo_type='dataset')`. If it succeeds, the dataset is in LeRobot
format.

## Episode subsetting

lerobot supports `--dataset.episodes='[0,3,7,...]'` to train on a subset of episodes.
This is how the train/eval split is enforced — pass the train episode list at launch
time. No physical dataset copy is needed.
