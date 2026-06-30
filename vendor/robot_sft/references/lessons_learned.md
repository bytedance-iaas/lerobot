# Lessons learned — real robot SFT failure modes

Distilled from real Isaac GR00T N1.7 fine-tuning runs. Each entry: the failure, why it
happens, and the **check** that prevents it. Use these as the backbone of the stage-a
review and as preflight checks in stages d/e. They generalize to most LeRobot-style VLA
SFT, not just GR00T.

## 1. Gated backbone you can't download
**Symptom:** training dies seconds in with `OSError: You are trying to access a gated
repo ... nvidia/Cosmos-Reason2-2B` (401). The main model (`nvidia/GR00T-N1.7-3B`) is
public, but its config hardcodes a **gated** VLM backbone, fetched at model init.
**Why:** no HF token, or the account hasn't accepted the gated repo's license.
**Check (stage a/d):** before any long run, verify `hf auth whoami` succeeds AND the
gated backbone is accessible (token present + license accepted), or that both
`base_model` and its backbone exist **locally** and you pass local paths. Prefer local
paths to avoid re-downloading multi-GB weights every run.

## 2. `/dev/shm` too small → dataloader Bus error
**Symptom:** ~minutes/hours in: `DataLoader worker killed by signal: Bus error ... out of
shared memory` and/or `unable to write ... No space left on device` for `/torch_*` files.
**Why:** containers default `/dev/shm` to **64 MB**. With `num_workers>0`, dataloader
workers pass tensors through `/dev/shm`; 64 MB overflows.
**Check (stage d):** read `/dev/shm` size. For `num_workers>0` you want **several GB+**.
Remediate with `mount -o remount,size=16g /dev/shm` (needs root + CAP_SYS_ADMIN; in some
containers the size is fixed at creation via `--shm-size` and can't be grown — then it
silently no-ops, so **re-check `df -h /dev/shm` after remounting**). If you can't grow it,
fall back to `num_workers=0` (works, but ~no async prefetch → periodic stalls) and say so.

## 3. `num_workers=0` is a fallback, not a default
**Default to 4.** Only drop to 0 when `/dev/shm` can't be enlarged. Multi-worker async
prefetch overlaps video decode with compute; with video-heavy LeRobot data the speedup is
large (a run that took 2h23m at workers=0 can be ~1.5h or less at workers=4 with adequate
shm). Always record *why* if you used 0.

## 4. Checkpoint truncated by an early kill
**Symptom:** a checkpoint dir has the model shards but is missing `processor_config.json`,
`statistics.json`, `experiment_cfg/`, `trainer_state.json` → later eval/resume fails with
`Unrecognized processing class` or "no resumable state".
**Why:** HF Trainer writes a checkpoint in order — **model shards first, then optimizer,
then processor / trainer_state / rng**. Killing the moment the last shard appears truncates
the rest. (This is exactly how a "stop at step N" watchdog corrupts a checkpoint.)
**Check (stage e):** to stop at a target step, **wait for `trainer_state.json`** (written
last) to appear in `checkpoint-N/`, not just the model shards, before sending SIGTERM.
**Repair:** the missing processor/config files are identical across checkpoints — copy
`processor/` + `experiment_cfg/` + `processor_config.json` + `statistics.json` from a
complete sibling checkpoint (or the top-level output dir) into the truncated one.

## 5. Output dir on a full / wrong disk
**Symptom:** checkpoint save fails with `No space left on device`; or the root overlay is
near-full. GR00T checkpoints are ~11 GB each × `save_total_limit`.
**Check (stage d):** put `--output-dir` on a volume with tens of GB free
(`save_total_limit × ~12 GB` headroom). Don't default to `/tmp` if root is tight.

## 6. Camera-key / dimension mismatch in modality.json
**Symptom:** training errors on missing keys, or silently learns garbage, because the
`modality.json` video `original_key` doesn't match the dataset's actual columns (e.g. the
SO100 example maps `front → observation.images.front` but the dataset's camera is
`observation.images.top`), or the state/action slice dims don't match the parquet.
**Check (stage b/c):** read `meta/info.json` `features` and map every modality key to a
real column. Verify state/action `start:end` slices equal the parquet feature shapes.

## 7. Default `max_steps` ignores dataset size
**Symptom:** using the entrypoint's default (e.g. 10000) on 50 episodes = ~17 epochs →
overfitting; or too few steps on a large set → underfitting.
**Check (stage d):** compute from data. `steps_per_epoch = ceil(num_samples /
global_batch_size)`; pick epochs by dataset size (small sets: ~5–8 epochs is a reasonable
starting band, then judge by open-loop eval). For 50 eps / 18.8k samples / batch 32 →
~590 steps/epoch → ~3000–5000 steps, not 10000. Save checkpoints often enough to pick the
best by eval (and remember `save_total_limit` prunes the oldest).

## 8. Effective batch = per_device × grad_accum × num_gpus
**Check (stage d):** `global_batch_size` is split as `per_device = global // num_gpus`
(grad_accum 1 by default). Make sure it divides evenly by `num_gpus`, and that per-device
batch fits GPU memory. Bigger GPUs (e.g. H200 143 GB) can take 32–64 comfortably for a 3B.

## 9. Pick idle GPUs explicitly
**Check (stage d/e):** parse `nvidia-smi`; launch on GPUs with low memory use via
`CUDA_VISIBLE_DEVICES`. Don't assume GPU 0 is free. Leave eval to a different free GPU so
it doesn't contend with training.

## 10. Build deps need system headers
**Symptom:** `uv sync` / dataset-conversion deps fail building a C extension:
`fatal error: Python.h: No such file or directory` (e.g. `evdev` via `lerobot`→`pynput`).
**Check (stage c/d):** ensure `pythonX.Y-dev` headers are installed before building.

## 11. Open-loop ≠ closed-loop; eval doesn't take a modality flag
**Check (stage e):** open-loop eval (predicted vs recorded actions, MSE/MAE) validates
*learning*, not real-robot success — compounding error makes closed-loop harder. Use it to
pick checkpoints, not to claim deployment readiness. Note: GR00T's `open_loop_eval.py` reads
the modality config from the checkpoint's `experiment_cfg`, so do **not** pass
`--modality-config-path` to it (it's an unrecognized arg).

## 12. Resume is automatic — don't fight it
GR00T calls `trainer.train(resume_from_checkpoint=True)` unconditionally → re-running the
**same command with the same `--output-dir`** auto-resumes from the latest checkpoint.
**Never** set `--save_only_model` for a run you might resume (it drops optimizer/scheduler/
RNG → not resumable). See `gr00t_resume.md`.

## 13. No in-loop eval on the sharded path — split train/eval *before* training
**Symptom:** you want a validation curve / generalization number during training, or you run
`open_loop_eval.py` on trajectories the model was trained on and mistake low MSE for
generalization.
**Why:** the `finetune.sh` sharded dataset path hard-asserts `eval_strategy == "no"`
(`gr00t/data/dataset/factory.py`), so HF-Trainer in-loop eval is disabled and the returned
`eval_dataset` is `None`. Setting `val_dataset_path` does **not** help on this path. Eval is
therefore **only** post-hoc via `gr00t/eval/open_loop_eval.py` (per-trajectory MSE/MAE).
**Check (stage c/e):** to get an honest generalization signal, **hold out ≈10% of episodes
per dataset into a separate eval dataset dir before training** (training set must not contain
them), then run `open_loop_eval.py` on those held-out ids to pick the best checkpoint. A
LeRobot v2.1 split requires rebuilding `meta/` (episodes/tasks/info/modality) with re-indexed
episode ids for both the train and eval dirs. Don't pass `--modality-config-path` to
`open_loop_eval.py` (it reads modality from the checkpoint's `experiment_cfg`, #11).

## 14. When to stop — train-loss plateau is NOT "done"; the eval curve decides
**Symptom:** behaviour-cloning train loss drops fast (e.g. 1.1 → 0.05 in the first ~1k steps)
then looks flat, tempting an early stop — or, conversely, blindly running a hardcoded
`max_steps` long after the model stopped improving.
**Why:** the reconstruction/flow loss bottoms out early and is a poor proxy for task quality;
a flat loss says "fitting the data distribution," not "best policy." The real selection
signal is the **open-loop eval MSE curve** over checkpoints (#13), which can keep improving
(or start over-fitting) well after the loss flattens.
**Check (stage e):** don't conclude from loss alone. The watchdog writes a per-poll
`assessment` to `run.json` (shown on the dashboard) and only flags `stop_recommended` once
loss is flat **AND** there are ≥2 eval points whose MSE has stopped improving — pick the
checkpoint with the lowest eval MSE. Need ≥2 eval points before judging; one point can't show
a trend. To stop a run cleanly, `touch <run_dir>/STOP`: the watchdog stops at the **latest
complete checkpoint** (keeps it resumable, never truncates — #4) and does not restart.

## 15. Eval cadence must be time-bounded — derive save_steps from throughput
**Symptom:** a run evaluates only a handful of times (or, early on, just once), so the eval
curve is too sparse to pick a good checkpoint or judge convergence.
**Why:** periodic open-loop eval (#13) only fires when a checkpoint is saved, so the eval
cadence equals the checkpoint cadence (`save_steps`). A `save_steps` chosen as a fraction of
`max_steps` ignores wall-clock: on a slow run (low it/s, or a big model) that can be hours
apart; thread-storm/stall incidents can also drop early evals.
**Check (stage d/e):** measure throughput (it/s) — from preflight's steady steps or the first
~hundred steps of `train.log` — and cap `save_steps <= it/s x 3600 x max_eval_hours` so an
eval lands **at least hourly** (`plan_training.py --throughput-it-s ... --max-eval-hours 1`).
Also start `eval_watcher.py` with a per-dataset `--eval-timeout` and a small `--threads` cap:
an eval running next to training otherwise thread-storms into a CUDA-init deadlock, and with
no timeout one hang silently starves the whole eval curve.

## 16. Right-size the batch from measured memory — the planner flies blind
**Symptom:** training runs at a small batch (e.g. per_device 16) using a fraction of the GPU
(28 GB of an H200's 143 GB), leaving throughput on the table — the run is slower than it needs
to be because the batch was chosen before anyone looked at real memory.
**Why:** `plan_training.py` sets the batch from a rough mem-per-GPU heuristic *before* the
model+data are ever loaded. With a frozen backbone (`tune_llm/tune_visual=False`, only
projector+diffusion trained) optimizer state is tiny and activations dominate, so the true
headroom is large and very model-specific — unknowable without measuring.
**Check (stage d):** `preflight.py` samples **peak GPU memory** during the 2-step smoke run and
emits a `batch_suggestion` (scale `per_device_batch` to ~85% of total, conservative because a
pure-linear assumption under-estimates capacity). Apply it, **re-run preflight to confirm it
fits** (catches a wrong estimate cheaply), then recompute `max_steps`/`save_steps` for the new
global batch and scale LR with batch. Note: changing batch mid-run isn't a clean resume — tune
the batch *before* the long launch, not after, since restarting forfeits all prior progress.

## 17. Resumable-checkpoint check must understand DeepSpeed ZeRO format
**Symptom:** `verify_run.py` reports `resumable: fail` on a perfectly good checkpoint, and the
watchdog logs "NO resumable checkpoint — restart will be from scratch" even though full state
was saved — so a crash mid-run would needlessly retrain from zero.
**Why:** GR00T trains with **DeepSpeed ZeRO**, which does NOT write HF-native `optimizer.pt`.
Its resumable state lives in a `global_step<N>/` dir (`bf16_zero_pp_rank_*_optim_states.pt` +
`mp_rank_00_model_states.pt`) with a top-level `latest` file naming that dir. A predicate that
hard-requires `optimizer.pt`/`scheduler.pt` false-negatives every DeepSpeed checkpoint.
**Check:** `is_resumable` (in `watchdog.py`, reused by `verify_run.py`) must accept EITHER
HF-native (`optimizer.pt`) OR DeepSpeed (`latest` → `global_step*/…optim_states.pt`), plus
`trainer_state.json` (written last ⇒ save finished), RNG, and weights (`.safetensors` or
`*model_states.pt`). Confirm with a real GR00T checkpoint's contents, not assumptions.
