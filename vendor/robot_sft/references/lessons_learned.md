# Lessons learned — real robot SFT failure modes

Distilled from real robot-SFT runs (originally Isaac GR00T fine-tuning; re-grounded here
for **lerobot / `lerobot-train`**). Each entry: the failure, why it happens, and the
**check** that prevents it. Use these as the backbone of the stage-a review and as
preflight checks in stages d/e.

Environment note (console pod): the lerobot checkout is **/lerobot** (run everything from
there with `uv run`); all big artifacts — checkpoints, session state, HF caches — belong on
the roomy **/opt/data** volume, never in the repo.

## 1. Gated / unauthorized Hub model or dataset
**Symptom:** training dies seconds in with `401 Client Error` / `gated repo` /
`RepositoryNotFoundError`.
**Why:** no HF token, or the account hasn't accepted a gated license. In lerobot this bites
when finetuning a pretrained VLA (`--policy.path=lerobot/pi0` pulls a **gated PaliGemma**
backbone; smolvla similarly needs its base), or when the dataset repo is private.
**Check (stage a/d):** before any long run, verify `hf auth whoami` succeeds AND the exact
`--policy.path` / `--dataset.repo_id` is accessible, or that everything exists **locally**
(`--dataset.root`, local policy dir). Prefer local paths to avoid re-downloading multi-GB
weights every run. Fresh `--policy.type=act` training needs no Hub model at all.

## 2. `/dev/shm` too small → dataloader Bus error
**Symptom:** ~minutes/hours in: `DataLoader worker killed by signal: Bus error ... out of
shared memory` and/or `unable to write ... No space left on device` for `/torch_*` files.
**Why:** containers default `/dev/shm` to **64 MB**. With `--num_workers>0`, dataloader
workers pass tensors through `/dev/shm`; 64 MB overflows.
**Check (stage d):** read `/dev/shm` size. For `num_workers>0` you want **several GB+**.
Remediate with `mount -o remount,size=16g /dev/shm` (needs root + CAP_SYS_ADMIN; in some
containers the size is fixed at creation via `--shm-size` and can't be grown — then it
silently no-ops, so **re-check `df -h /dev/shm` after remounting**). If you can't grow it,
fall back to `--num_workers=0` (works, but ~no async prefetch → periodic stalls) and say so.

## 3. `num_workers=0` is a fallback, not a default
**Default to 4.** Only drop to 0 when `/dev/shm` can't be enlarged. Multi-worker async
prefetch overlaps video decode with compute; with video-heavy lerobot data the speedup is
large. Always record *why* if you used 0.

## 4. Checkpoint truncated by an early kill
**Symptom:** a step dir has `pretrained_model/` but `training_state/` is missing or
incomplete → resume fails ("no resumable state").
**Why:** lerobot's `save_checkpoint` writes `pretrained_model/` (config + weights + train
config + processor) first, **then** `training_state/` (optimizer, rng, scheduler,
`training_step.json`). Killing between the two truncates the state. (This is exactly how a
naive "stop at step N" kill corrupts a checkpoint.)
**Check (stage e):** to stop at a step, wait until `training_state/training_step.json`
exists in the newest `checkpoints/<step>/` before sending SIGTERM — that file lands with the
last-written dir. The watchdog's STOP-file flow does this for you.
**Repair:** a truncated checkpoint usually still has a loadable `pretrained_model/` — fine
for eval/deployment, just not resumable. Resume from the previous complete step instead.

## 5. Output dir on a full / wrong disk — and lerobot keeps EVERY checkpoint
**Symptom:** checkpoint save fails with `No space left on device`; or the root overlay fills.
**Why:** lerobot has **no save_total_limit** — every `--save_freq` step dir stays on disk.
**Check (stage d):** put `--output_dir` on the big volume (console pod: `/opt/data/...`,
never `/lerobot` or `/tmp`). Budget `(steps / save_freq) × ckpt_size` (ACT ≈ 0.5–2 GB per
checkpoint incl. optimizer state; VLA policies far more) and prune old step dirs during
long runs if space gets tight — keep the best-eval and latest ones.

## 6. Policy/dataset feature mismatch (camera keys, state/action dims)
**Symptom:** `KeyError: 'observation.images...'` at startup, shape errors, or silently bad
behaviour when deploying.
**Why:** a policy's `input_features`/`output_features` must match the dataset's `features`
(see `meta/info.json`). Training fresh (`--policy.type=...`) derives features FROM the
dataset so it self-matches; the mismatch bites when **finetuning a pretrained policy**
(`--policy.path=...`) whose features were derived from a *different* dataset (camera named
`observation.images.top` vs `.front`, different state/action dims).
**Check (stage b/d):** diff the dataset's `meta/info.json` features against the pretrained
policy's `config.json` input/output features before launching; rename dataset camera keys
at recording time (or use lerobot's rename_map facilities) rather than hoping.

## 7. Default `--steps` ignores dataset size
**Symptom:** `lerobot-train` defaults to `--steps=100000`; on 50 episodes that's massive
overfitting, and on a huge set it may underfit.
**Check (stage d):** compute from data. `steps_per_epoch = ceil(num_frames / batch_size)`;
pick epochs by dataset size (small sets: ~5–8 epochs starting band, then judge by the eval
curve). 50 eps / ~19k frames / batch 16 → ~1.2k steps/epoch → ~6–7k steps, not 100k.
`plan_training.py` does this.

## 8. Know your effective batch
**Check (stage d):** lerobot's `--batch_size` is **per process**. Single-process runs:
effective batch = batch_size. Multi-GPU via `accelerate launch --num_processes=N`:
effective batch = batch_size × N, and `--steps` means optimizer steps regardless. Check
per-device fit with preflight (#16), not guesswork.

## 9. Pick idle GPUs explicitly
**Check (stage d/e):** parse `nvidia-smi`; launch on GPUs with low memory use via
`CUDA_VISIBLE_DEVICES`. Don't assume GPU 0 is free. Give eval_watcher a **different** free
GPU so eval never contends with training. (Single-GPU pod: eval waits its turn — use a
larger `--poll`, or accept eval running on the same GPU between checkpoints; the eval
subprocess exits between scores.)

## 10. Build deps need system headers
**Symptom:** `uv sync` / dataset deps fail building a C extension:
`fatal error: Python.h: No such file or directory`.
**Check (stage c/d):** ensure `pythonX.Y-dev` headers are installed before building.

## 11. Open-loop ≠ closed-loop
**Check (stage e):** offline eval (predicted vs recorded actions on held-out episodes,
MSE/MAE via `offline_eval.py`) validates *learning*, not real-robot success — compounding
error makes closed-loop harder. Use it to pick checkpoints, not to claim deployment
readiness. The real test is `lerobot-record`/teleop replay on the actual robot.

## 12. Resume is explicit — `--resume=true --config_path=...`
lerobot does **not** auto-resume by re-running the same command; an existing
`--output_dir` without `--resume` is a hard error (by design, so you never silently
overwrite a run). Resume =
`lerobot-train --resume=true --config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json`.
Every checkpoint carries full training state (there is no `--save_only_model` to
mis-set), and data order resumes sample-exact. See `lerobot_resume.md`.

## 13. No sim env → hold out episodes for eval *before* training
**Symptom:** you want a validation/generalization signal during training, or you score
checkpoints on trajectories the model trained on and mistake low MSE for generalization.
**Why:** `lerobot-train`'s in-loop eval (`--eval_freq`) needs a **simulator env**; real-robot
datasets have none, so set `--eval_freq=0` and eval out-of-band.
**Check (stage c/e):** hold out ≈10% of episodes (`split_train_eval.py` — it just picks
deterministic episode-id lists; **no physical dataset copy needed**, lerobot subsets with
`--dataset.episodes='[...]'`). Train on the train list; have `eval_watcher.py` score every
checkpoint on the held-out list with `offline_eval.py`. Disjointness is what makes the MSE
curve an honest generalization signal.

## 14. When to stop — train-loss plateau is NOT "done"; the eval curve decides
**Symptom:** behaviour-cloning train loss drops fast (e.g. 1.1 → 0.05 in the first ~1k steps)
then looks flat, tempting an early stop — or, conversely, blindly running a hardcoded
step count long after the model stopped improving.
**Why:** the reconstruction/flow loss bottoms out early and is a poor proxy for task quality;
a flat loss says "fitting the data distribution," not "best policy." The real selection
signal is the **held-out eval MSE curve** over checkpoints (#13), which can keep improving
(or start over-fitting) well after the loss flattens.
**Check (stage e):** don't conclude from loss alone. The watchdog writes a per-poll
`assessment` to `run.json` (shown on the dashboard) and only flags `stop_recommended` once
loss is flat **AND** there are ≥2 eval points whose MSE has stopped improving — pick the
checkpoint with the lowest eval MSE. Need ≥2 eval points before judging; one point can't show
a trend. To stop a run cleanly, `touch <run_dir>/STOP`: the watchdog stops at the **latest
complete checkpoint** (keeps it resumable, never truncates — #4) and does not restart.

## 15. Eval cadence must be time-bounded — derive save_freq from throughput
**Symptom:** a run evaluates only a handful of times (or, early on, just once), so the eval
curve is too sparse to pick a good checkpoint or judge convergence.
**Why:** offline eval (#13) only fires when a checkpoint is saved, so the eval cadence
equals the checkpoint cadence (`--save_freq`). A save_freq chosen as a fraction of steps
ignores wall-clock: on a slow run that can be hours apart.
**Check (stage d/e):** measure throughput (it/s) — from preflight's steady steps or the first
~hundred steps of `train.log` — and cap `save_freq <= it/s × 3600 × max_eval_hours` so an
eval lands **at least hourly** (`plan_training.py --throughput-it-s ... --max-eval-hours 1`).
Also start `eval_watcher.py` with an `--eval-timeout` and a small `--threads` cap: an eval
running next to training otherwise thread-storms into stalls, and with no timeout one hang
silently starves the whole eval curve.

## 16. Right-size the batch from measured memory — the planner flies blind
**Symptom:** training runs at a small batch using a fraction of the GPU, leaving throughput
on the table — the run is slower than it needs to be because the batch was chosen before
anyone looked at real memory.
**Why:** `plan_training.py` sets the batch from a rough policy-family × GPU-memory heuristic
*before* the model+data are ever loaded; the true headroom is model-specific and unknowable
without measuring.
**Check (stage d):** `preflight.py` samples **peak GPU memory** during the 2-step smoke run
and emits a `batch_suggestion` (scale `--batch_size` to ~85% of total; conservative because a
pure-linear assumption under-estimates capacity). Apply it, **re-run preflight to confirm it
fits**, then recompute `--steps`/`--save_freq` for the new batch and consider scaling LR.
Note: changing batch mid-run isn't a clean resume — tune the batch *before* the long launch.

## 17. Verify resumability against the REAL checkpoint anatomy, not assumptions
**Symptom:** a resumability check reports `fail` on a good checkpoint (needless from-scratch
restarts), or `pass` on a truncated one (resume then crashes).
**Why:** every trainer lays out state differently (the GR00T era needed DeepSpeed-ZeRO
special-casing here). lerobot's layout is
`checkpoints/<step>/{pretrained_model/{config.json,model.safetensors,train_config.json,...},
training_state/{optimizer_state.safetensors,optimizer_param_groups.json,rng_state.safetensors,
scheduler_state.json,training_step.json}}` plus a `checkpoints/last` symlink.
**Check:** `is_resumable` (in `watchdog.py`, reused by `verify_run.py`) requires
`pretrained_model/{config.json,model.safetensors,train_config.json}` +
`training_state/{optimizer_state.safetensors,training_step.json,rng_state*}`. Confirm against
a real checkpoint produced by *this* lerobot version if in doubt (preflight saves one).
