---
name: robot_sft
description: >-
  Orchestrate supervised fine-tuning (SFT) of robot imitation-learning / VLA policies with
  lerobot (`lerobot-train`: ACT, Diffusion Policy, pi0, SmolVLA, ...) through a resumable,
  multi-agent pipeline: review the user's setup, explore the dataset, plan training
  parameters from hardware + dataset size, then launch training under a self-healing
  watchdog with a status dashboard and a held-out eval curve. Use this skill WHENEVER the
  user wants to fine-tune, SFT, post-train, or train a robot policy / VLA model, asks to
  "train ACT on my SO-101 data", mentions a robot + a dataset, wants training
  monitored/auto-resumed, or wants a review of their training parameters — even if they
  don't say the word "skill".
---

# robot_sft — resumable multi-agent robot SFT orchestration (lerobot)

## What this skill is

A **conductor**, not a monolith. Robot SFT fails in boring, expensive ways: a gated Hub
backbone you can't download, a 64 MB `/dev/shm` that kills the dataloader 90 minutes in,
a checkpoint truncated by a bad kill, a camera key that doesn't match the policy, or
100000 steps when the data only justifies 5000. This skill turns one training **session**
into a sequence of small, **independently verifiable, file-checkpointed stages**, each
run by a focused sub-agent, so a crash or context reset never loses progress — you
re-read the session state and continue.

Read `references/lessons_learned.md` early and keep it in mind throughout — it is the
distilled list of the failure modes above, each with the concrete check that prevents it.

## Environment (console pod)

- The **lerobot checkout is `/lerobot`** — run every lerobot command from there so the uv
  venv resolves: `cd /lerobot && uv run lerobot-train ...` / `uv run python ...`.
- **All big artifacts live on `/opt/data`** (the roomy persistent volume): session state
  defaults to `/opt/data/robot_sft/` (via `session.py`), checkpoints to
  `/opt/data/robot_sft/runs/...` (via `plan_training.py`), and HF caches land under
  `/opt/data/.cache/...` automatically (`HOME=/opt/data` in the pod). **Never** write
  checkpoints or datasets into `/lerobot` or `/tmp`.
- Outside the pod both default to the current directory (`.robot_sft/`); override with
  `$ROBOT_SFT_HOME` / `--output-dir`.

## Core model: Session → Stages → Runs

- **Session** = one user intent ("fine-tune ACT on my SO-101 pick-place data"). One
  session directory under `<root>/sessions/<session_id>/`.
- **Stages** = the pipeline below (a→e). Each writes a JSON artifact and flips its status
  in `session.json`. Stages are resumable: on re-entry, skip any stage already `done`.
- **Runs** = actual invocations of `lerobot-train` inside the train stage. A session may
  need several runs (crash-resume, early-stop-and-restart). Each run has its own state
  under `runs/<run_id>/`.

All cross-agent communication is **files**, never memory. The session directory is the
single source of truth. This is what makes the whole thing resumable (requirement: "随时
resume"). Use `scripts/session.py` for every state read/write — do not hand-edit JSON.

```
<root>/sessions/<session_id>/          # root = /opt/data/robot_sft in the pod
├── session.json            # master state: stages, status, current_stage, config
├── overview.json           # stage a output
├── dataset_explore.json    # stage b output
├── preprocess.json         # stage c output (the train/eval episode split)
├── training_plan.json      # stage d output (the launch + resume commands, computed steps)
└── runs/
    └── run-001/
        ├── run.json        # run state: status, restarts, last_step, last_loss, checkpoint
        └── train.log       # training stdout/stderr
```

## How to run a session (the orchestrator loop)

This is your top-level algorithm. Follow it; let the sub-agents do the depth.

1. **Resolve the session.** Run `python scripts/session.py status`. If an unfinished
   session exists, ask the user whether to resume it or start fresh. To resume: read
   `session.json`, jump to `current_stage`. To start: `session.py create`.
2. **Run each unfinished stage in order** (a→e). For each, spawn the matching sub-agent
   with the instructions in `references/agents.md`, hand it the session dir, and have it
   write its artifact + mark the stage `done` via `session.py`. **Validate the artifact
   exists and parses before advancing** (plan-validate-execute).
3. **Gate on hard errors.** Stage a stops the whole session if the request needs a
   training path lerobot doesn't provide. Stage d stops if hardware is insufficient and
   can't be remediated. Surface the blocker to the user; do not silently continue.
4. **Hand off to the watchdog** for the train stage and keep the user informed via the
   dashboard, not a wall of CLI text.

Keep a visible checklist in your reply so the user can see where the session is:

```
- [ ] a. overview / review
- [ ] b. dataset explore
- [ ] c. train/eval episode split (conversion if needed)
- [ ] d. training plan + hardware
- [ ] e. train under watchdog
```

## The five stages (spawn one sub-agent each — details in references/agents.md)

### a. Overview & review  (always; gates the session)
Analyze the user's request and intended setup. The entrypoint here is lerobot's own
`lerobot-train` — a dataset (`--dataset.repo_id` / `--dataset.root`) plus a policy (fresh
`--policy.type=act|diffusion|...` or finetune `--policy.path=<pretrained>`) IS a complete
setup; nothing needs scaffolding. Produce `overview.json`: the resolved goal, dataset
reference, policy choice, the parameters found vs missing, and a **review** that combines
`references/lessons_learned.md` with general SFT advice. Only stop the session if the
request genuinely has no lerobot training path.

### b. Dataset explore  (whenever the dataset is specified OR vague)
Use a dedicated explorer sub-agent any time the user names a dataset, points at a path,
or is fuzzy about it. It inspects `meta/info.json` (codebase_version — this lerobot expects
**v3.0**; v2.x needs conversion first), episode/frame counts, camera keys
(`observation.images.*`), state/action dims, fps, and — when finetuning a pretrained
policy — whether the dataset's features match the policy's `config.json` features. Writes
`dataset_explore.json`. This is where camera-key and dimension mismatches get caught
**before** they waste a training run.

### c. Train/eval episode split  (always, unless the user opts out; conversion is conditional)
Two responsibilities:
- **Conversion (conditional):** if stage b found a v2.x dataset, convert it with lerobot's
  tooling, then re-verify `meta/info.json`.
- **Train/eval split (always — unless the user opts out):** hold out a fraction of
  episodes as an eval set *before* training, so the training set does not contain them.
  Real-robot data has no simulator, so in-loop eval is off (`--eval_freq=0`) and post-hoc
  offline eval on **held-out episodes** is the only honest generalization signal
  (lessons_learned #13). lerobot subsets episodes natively (`--dataset.episodes='[...]'`),
  so **no physical dataset copy is needed** — run `python scripts/split_train_eval.py
  --dataset-repo-id <id> [--dataset-root <dir>] --out <session>/preprocess.json`
  (default ≈10% holdout, min 1, seeded/deterministic). Verify the two id lists are
  disjoint and in range.

Writes `preprocess.json` with `train_episodes` + `eval_episodes`. If the user opted out,
record `eval_episodes: []` (eval then only sanity-checks learning, not generalization).

### d. Training plan & hardware  (always; gates the train stage)
Run `python scripts/check_hardware.py` and `python scripts/plan_training.py`. This stage:
- Checks **GPU count + free memory** (pick idle GPUs), **disk space** for checkpoints
  (must be on `/opt/data` in the pod — and lerobot keeps EVERY checkpoint, no rotation,
  so budget `(steps/save_freq) × ckpt_size`), and **`/dev/shm` size**.
- **Ensures multi-process dataloading works**: if `/dev/shm` is too small for
  `--num_workers>0`, either remediate (remount larger — needs the user / sufficient caps)
  or fall back to `--num_workers=0`, and say which and why.
- **Computes real steps from the data**, not lerobot's 100k default:
  `steps_per_epoch = ceil(num_frames / batch_size)`, `steps = epochs × steps_per_epoch`,
  with a sane epoch count for the dataset size. See `plan_training.py`.
- Emits BOTH the **launch command** (with the train episode list inlined) and the
  **resume command** (`--resume=true --config_path=.../checkpoints/last/pretrained_model/
  train_config.json`) into `training_plan.json`.
- **Sets the eval cadence from throughput** (lessons_learned #15): offline eval only fires
  when a checkpoint is saved, so checkpoint cadence == eval cadence. Measure it/s (from
  preflight's steady steps, or the first ~hundred steps of `train.log`) and pass
  `--throughput-it-s` to `plan_training.py`; it caps `save_freq` so eval runs **at least
  once per `--max-eval-hours` (default 1h)** of wall-clock.
Then **smoke-test before committing GPU-hours**: `python scripts/preflight.py --session
<dir>` runs the same command for ~2 steps (including one checkpoint save) in a throwaway
dir and classifies the result — catching auth / `/dev/shm` / feature-mismatch / bad-flag
bugs in ~1–3 min instead of after a 6-hour launch. Do not proceed to stage e until
preflight is green (or the user accepts the risk).

**Right-size the batch from measured memory (don't fly blind):** the planner picks an
initial batch from a policy-family heuristic *before* seeing real usage. preflight samples
**peak GPU memory** during the smoke run and emits a `batch_suggestion` (scale
`--batch_size` to ~85% of memory). Apply it, **re-run preflight to confirm it fits**, then
recompute `--steps`/`--save_freq` for the new batch (and consider scaling LR). A bigger
batch that fits = higher GPU utilization and a faster run (lessons_learned #16).

### e. Train under watchdog + periodic eval  (the long pole)
Launch training and the monitors. Three background processes, all communicating via files:
1. Start the **dashboard**: `python scripts/monitor_server.py --session <dir>`
   (background). It only *reads* the state files and serves an HTML page + JSON — it never
   runs training. It is **TensorBoard-like**: it plots, with plain `<canvas>` (no external
   deps, works offline), the **training loss curve** (log-y, parsed from `train.log`) and
   the **held-out eval curve** (mean MSE per checkpoint, from `eval/eval_results.jsonl`).
   Give the user the URL (in the console UI: viewer "+ 打开" → the port) so they watch
   remotely instead of the CLI.
2. Start the **watchdog**: `python scripts/watchdog.py --session <dir> --run <run_id>`
   (background). It owns the training subprocess and implements the self-healing loop
   below. It checks status on a cadence of **≤5 minutes** and writes everything it observes
   to `run.json` (which the dashboard surfaces).
3. Start the **eval watcher**: `python scripts/eval_watcher.py --session <dir> --run
   <run_id> --gpu <idle_gpu>` (background). It watches `output_dir/checkpoints/` for each
   new *complete* step dir (waits for `training_state/training_step.json`) and runs
   `offline_eval.py` (open-loop replay) on the held-out episodes from stage c — on a
   **separate GPU** when one exists so it never steals from training (lessons_learned #9;
   on a single-GPU pod evals run between checkpoints and exit, keep `--eval-timeout`) —
   appending mean MSE/MAE per checkpoint to `eval/eval_results.jsonl`. It is resumable
   (skips checkpoints already scored).

You may either let `watchdog.py` run autonomously and poll its `run.json`, or drive the
cadence yourself with `/loop` (re-invoking a status check every few minutes). Prefer the
autonomous watchdog for unattended runs; use `/loop` when the user wants you in the loop.

**When the run ends, verify it — don't trust the exit code.** `exit 0` has lied before.
Run `python scripts/verify_run.py --session <dir> --run <run_id>`; it writes `VERIFY.md`
with independent verdicts (did loss actually drop? reach target step? is the latest
checkpoint resumable AND inference-loadable? any fatal signature?). Report the VERIFY
verdict, not just "done".

**Pick the best checkpoint from the eval curve.** The `eval_watcher` (process 3 above) has
been scoring every checkpoint on the held-out split *throughout* training, so by the end
`eval/eval_results.jsonl` already holds the MSE/MAE curve — choose the checkpoint with the
lowest mean MSE (visible on the dashboard's eval chart). To (re)score a checkpoint
manually: `cd /lerobot && uv run python <skill>/scripts/offline_eval.py --model-path
<output_dir>/checkpoints/<N>/pretrained_model --dataset-repo-id <id> [--dataset-root <dir>]
--episodes <held-out ids>`. Because these episodes were excluded from training (stage c),
this is a real generalization signal, not memorization. Open-loop still ≠ closed-loop: it
picks checkpoints, it does not prove real-robot success.

## The self-healing watchdog contract

`watchdog.py` implements — and you must preserve — this contract (full algorithm in
`references/agents.md` and the script itself):

- **Monitor:** parse the train log for `step` (tqdm `N/M`) and `loss` (tracker `loss:x`)
  at least every 5 minutes; record throughput and last checkpoint.
- **Assess every poll (record the conclusion):** each cycle, write a human-readable
  `assessment` to `run.json` (the dashboard shows it) — loss trend, plateau length,
  eval-curve state, and a `stop_recommended` flag. Train-loss plateau alone never sets it;
  a stop is only recommended once loss is flat **AND** ≥2 eval points show the eval MSE
  has stopped improving (lessons_learned #14). This turns "is it done?" into a
  continuously-updated, visible verdict instead of a guess.
- **Graceful manual stop:** if `<run_dir>/STOP` exists, stop at the **latest complete
  checkpoint** (resumable, never truncated — lessons #4) and do **not** restart (status
  `stopped`). This is the safe way to honor "just stop it now."
- **Auto-resume on stop:** if the training process exits unexpectedly, **before
  restarting, check the latest checkpoint is resumable** (`pretrained_model/` complete AND
  `training_state/` has optimizer + rng + `training_step.json`). If yes, relaunch with the
  plan's **resume command** — `lerobot-train --resume=true --config_path=<output_dir>/
  checkpoints/last/pretrained_model/train_config.json` (lerobot does NOT auto-resume from
  a re-run of the original command; that errors on the existing output_dir). Never start
  from scratch when a resumable checkpoint exists. See `references/lerobot_resume.md`.
- **Early-stop on trouble:** if loss goes NaN/Inf or diverges (worse than a threshold for
  a sustained window), or the run stalls (no step progress for a timeout), **stop the run**,
  capture the reason, then **re-run** — again resuming from the last good checkpoint, not
  from zero.
- **Classify before retrying (don't loop on config bugs):** on any exit, classify the log
  via `error_patterns.py`. **Fatal** signatures (Hub auth, feature/camera mismatch, missing
  dataset, bad CLI flag, output-dir-exists) recur identically every restart — mark `failed`
  immediately with the fix, do NOT burn restarts on them. **OOM** → surface "lower the
  batch" before resuming. Only genuinely transient failures get the resume-and-retry path.
- **Backoff & cap:** apply capped exponential backoff to crash-restarts and cap the total
  restart count, so a hard-failing config doesn't loop forever. On exceeding the cap, mark
  the run `failed` and surface it.

## Bundled resources

- `references/lessons_learned.md` — concrete robot-SFT gotchas + the check that prevents
  each, re-grounded for lerobot. **Read this in stage a and consult it in d/e.**
- `references/lerobot_resume.md` — exactly how lerobot checkpointing/resume works, and what
  a checkpoint must contain to be resumable vs inference-only. **Read before writing the
  watchdog's resume logic or trusting any checkpoint.**
- `references/agents.md` — the full per-stage sub-agent briefs (inputs, steps, output
  schema) and the watchdog algorithm in detail.
- `references/prior_art.md` — the GitHub / docs prior art this skill is built on:
  Anthropic skills spec, multi-agent orchestration skills, SkyPilot / MosaicML / Lightning
  watchdog patterns, FastAPI status-dashboard pattern.
- `scripts/session.py` — file-based session/run state (create, status, set-stage, add-run,
  update-run). Use for all state changes. Defaults to `/opt/data/robot_sft` in the pod.
- `scripts/check_hardware.py` — GPUs, free memory, disk, `/dev/shm`; prints JSON + warnings.
- `scripts/plan_training.py` — compute steps/epochs/batch and emit the `lerobot-train`
  launch + resume commands.
- `scripts/split_train_eval.py` — deterministic held-out episode split (id lists only; no
  dataset copying — lerobot subsets via `--dataset.episodes`).
- `scripts/preflight.py` — ~2-step smoke test of the real command (incl. one checkpoint
  save); catches config bugs cheaply and measures GPU-memory headroom.
- `scripts/error_patterns.py` — shared log classifier: fatal (no-retry) vs oom vs retryable.
- `scripts/watchdog.py` — the self-healing training monitor (resume via --resume=true,
  early-stop, backoff).
- `scripts/verify_run.py` — post-run independent verification → `VERIFY.md` (don't trust
  exit 0).
- `scripts/offline_eval.py` — open-loop replay of a checkpoint on held-out episodes
  (per-episode + mean MSE/MAE, optional gt-vs-pred plots). The lerobot answer to "no sim
  env": run it under `cd /lerobot && uv run python ...`.
- `scripts/eval_watcher.py` — periodic offline eval: scores each new checkpoint on the
  held-out episodes (separate GPU when available), saving metrics to
  `eval/eval_results.jsonl` and plots under `eval/artifacts/ckpt-N/<group>/`.
- `scripts/monitor_server.py` — TensorBoard-like FastAPI dashboard (dependency-free
  `<canvas>`): plots the training-loss + eval curves, shows the watchdog's `assessment`
  verdict, and galleries any images found under `eval/artifacts/`.

Run scripts with `python` (or `uv run python` from /lerobot when they import lerobot —
that's only `offline_eval.py`; the rest are stdlib). Scripts are designed to be executed
for their output, not read into context — only open one if you need to adapt it.

## Style

This skill spends real money and GPU-hours per run. Bias toward **catching problems before
launch** (stages a–d are cheap; a wasted 6-hour run is not) and toward **honest status**
(if a run is resuming for the 3rd time, say so on the dashboard). Explain the *why* to the
user — most failure modes here are non-obvious infra issues, and a one-line reason ("only
got 50 episodes → 6400 steps, not 100000") builds the trust that keeps them from
second-guessing the plan.
