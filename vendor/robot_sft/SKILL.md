---
name: robot_sft
description: >-
  Orchestrate supervised fine-tuning (SFT) of robot VLA / imitation-learning models
  (e.g. Isaac GR00T, LeRobot policies) through a resumable, multi-agent pipeline:
  review the user's setup, explore/convert the dataset, plan training parameters from
  hardware + dataset size, then launch training under a self-healing watchdog with a
  FastAPI status dashboard. Use this skill WHENEVER the user wants to fine-tune,
  SFT, post-train, or train a robot policy / VLA model, asks to "train GR00T on my
  data", mentions an embodiment + a dataset, wants training monitored/auto-resumed,
  or wants a review of their training script and parameters — even if they don't say
  the word "skill". Requires an existing training entrypoint (e.g. finetune.sh); if
  none exists, the skill reports that and stops.
---

# robot_sft — resumable multi-agent robot SFT orchestration

## What this skill is

A **conductor**, not a monolith. Robot SFT fails in boring, expensive ways: a gated
backbone you can't download, a 64 MB `/dev/shm` that kills the dataloader 90 minutes
in, a checkpoint truncated by a bad kill, a wrong camera key in `modality.json`, or
10000 steps when the data only justifies 3000. This skill turns one training **session**
into a sequence of small, **independently verifiable, file-checkpointed stages**, each
run by a focused sub-agent, so a crash or context reset never loses progress — you
re-read the session state and continue.

Read `references/lessons_learned.md` early and keep it in mind throughout — it is the
distilled list of the failure modes above, each with the concrete check that prevents it.

## Core model: Session → Stages → Runs

- **Session** = one user intent ("fine-tune GR00T on my SO100 pick-place data"). One
  session directory under `.robot_sft/sessions/<session_id>/`.
- **Stages** = the pipeline below (a→e). Each writes a JSON artifact and flips its status
  in `session.json`. Stages are resumable: on re-entry, skip any stage already `done`.
- **Runs** = actual invocations of the training command inside the train stage. A session
  may need several runs (crash-resume, early-stop-and-restart). Each run has its own
  state under `runs/<run_id>/`.

All cross-agent communication is **files**, never memory. The session directory is the
single source of truth. This is what makes the whole thing resumable (requirement: "随时
resume"). Use `scripts/session.py` for every state read/write — do not hand-edit JSON.

```
.robot_sft/sessions/<session_id>/
├── session.json            # master state: stages, status, current_stage, config
├── overview.json           # stage a output
├── dataset_explore.json    # stage b output
├── preprocess.json         # stage c output
├── training_plan.json      # stage d output (the launch command + computed steps)
└── runs/
    └── run-001/
        ├── run.json        # run state: status, restarts, last_step, last_loss, checkpoint
        └── train.log       # training stdout/stderr
```

## How to run a session (the orchestrator loop)

This is your top-level algorithm. Follow it; let the sub-agents do the depth.

1. **Resolve the session.** Run `python scripts/session.py status` in the repo. If an
   unfinished session exists, ask the user whether to resume it or start fresh. To
   resume: read `session.json`, jump to `current_stage`. To start: `session.py create`.
2. **Run each unfinished stage in order** (a→e). For each, spawn the matching sub-agent
   with the instructions in `references/agents.md`, hand it the session dir, and have it
   write its artifact + mark the stage `done` via `session.py`. **Validate the artifact
   exists and parses before advancing** (plan-validate-execute).
3. **Gate on hard errors.** Stage a stops the whole session if there is no training
   entrypoint. Stage d stops if hardware is insufficient and can't be remediated. Surface
   the blocker to the user; do not silently continue.
4. **Hand off to the watchdog** for the train stage and keep the user informed via the
   dashboard, not a wall of CLI text.

Keep a visible checklist in your reply so the user can see where the session is:

```
- [ ] a. overview / review
- [ ] b. dataset explore
- [ ] c. data preprocess (conditional)
- [ ] d. training plan + hardware
- [ ] e. train under watchdog
```

## The five stages (spawn one sub-agent each — details in references/agents.md)

### a. Overview & review  (always; gates the session)
Analyze the user's request, their training script, and their parameters. **If there is no
basic training entrypoint (e.g. `finetune.sh` / a launch command), STOP and report it** —
do not invent one. Otherwise produce `overview.json`: the resolved goal, the entrypoint,
the parameters found vs missing, the embodiment, the dataset reference, and a **review**
(requirement #1) that combines `references/lessons_learned.md` with general SFT advice.

### b. Dataset explore  (whenever the dataset is specified OR vague)
Use a dedicated explorer sub-agent any time the user names a dataset, points at a path,
or is fuzzy about it. It inspects format (LeRobot v2.1 vs v3.0), `meta/info.json`,
episode/frame counts, camera keys, state/action dims, fps, and whether a `modality.json`
mapping is needed. Writes `dataset_explore.json`. This is where camera-key and dimension
mismatches get caught **before** they waste a training run.

### c. Data preprocess + train/eval split  (the split part is always; conversion is conditional)
Two responsibilities:
- **Conversion / modality (conditional):** if stage b found a conversion is needed (e.g.
  LeRobot v3.0 → v2.1) or a `modality.json` must be authored/placed, run it, then verify the
  resulting directory structure.
- **Train/eval split (always — unless the user opts out):** **hold out a fraction of
  episodes per dataset into a separate eval dataset *before* training**, so the training set
  does not contain them. This is required because GR00T's sharded finetune path **does not
  support in-loop evaluation** — `factory.py` asserts `eval_strategy == "no"`, so the *only*
  way to get an honest generalization signal is post-hoc `open_loop_eval.py` on episodes the
  model never saw (lessons_learned #13). Default holdout ≈ 10% of episodes per dataset (min 1,
  and warn if a dataset is too tiny to spare any). Splitting a LeRobot v2.1 dataset means
  copying the held-out `episode_*.parquet` + their `videos/.../episode_*.mp4` into an `eval/`
  dir and **rebuilding `meta/` for BOTH** dirs (`episodes.jsonl`, `tasks.jsonl`,
  `info.json` `total_episodes`/`total_frames`, and the `modality.json`) with contiguous
  re-indexed episode ids. Verify both dirs load and that episode sets are disjoint.

Writes `preprocess.json` with, per dataset, the **train path** and **eval path** (+ held-out
episode ids), plus the conversion actions. If the user opted out of a split, record that and
set eval path = train path (open-loop eval then only sanity-checks learning, not generalization).

### d. Training plan & hardware  (always; gates the train stage)
Run `python scripts/check_hardware.py` and `python scripts/plan_training.py`. This stage:
- Checks **GPU count + free memory** (pick idle GPUs), **disk space** for checkpoints
  (avoid a near-full root — choose a roomy volume), and **`/dev/shm` size**.
- **Ensures multi-process dataloading works**: if `/dev/shm` is too small for
  `num_workers>0`, either remediate (remount larger — needs the user / sufficient caps)
  or fall back to `num_workers=0`, and say which and why.
- **Computes real steps from the data**, not a hardcoded default:
  `steps_per_epoch = ceil(num_samples / global_batch_size)`, `max_steps = epochs ×
  steps_per_epoch`, with a sane epoch count for the dataset size. See `plan_training.py`.
- Plans **checkpoint storage location** and `save_steps` / `save_total_limit`, and
  guarantees `--save_only_model` is OFF so runs stay resumable.
- **Sets the eval cadence from throughput** (lessons_learned #15): periodic open-loop eval
  only fires when a checkpoint is saved, so checkpoint cadence == eval cadence. Measure it/s
  (from preflight's steady steps, or the first ~hundred steps of `train.log`) and pass
  `--throughput-it-s` to `plan_training.py`; it caps `save_steps` so eval runs **at least once
  per `--max-eval-hours` (default 1h)** of wall-clock — never let a slow run eval only a
  handful of times. (Don't make it absurdly frequent either: each checkpoint is ~12 GB I/O.)
Writes `training_plan.json` containing the exact launch command and the computed values.
Then **smoke-test before committing GPU-hours**: `python scripts/preflight.py --session
<dir>` runs the same command for ~2 steps in a throwaway dir and classifies the result —
catching gated-backbone / `/dev/shm` / camera-key bugs in ~1–3 min instead of after a 6-hour
launch. Do not proceed to stage e until preflight is green (or the user accepts the risk).

**Right-size the batch from measured memory (don't fly blind):** the planner picks an initial
batch *before* seeing real usage, and big GPUs (H200 = 143 GB) are usually far from full —
especially here, where the backbone is frozen (`tune_llm/tune_visual=False`) so activations,
not optimizer state, dominate. preflight now samples **peak GPU memory** during the smoke run
and emits a `batch_suggestion` (scale `per_device_batch` to ~85% of memory). Apply it, **re-run
preflight to confirm it fits**, then recompute `max_steps`/`save_steps` for the new global batch
(and consider scaling LR with batch). A bigger batch that fits = higher GPU utilization and a
faster run (lessons_learned #16).

### e. Train under watchdog + periodic eval  (the long pole)
Launch training and the monitors. Three background processes, all communicating via files:
1. Start the **FastAPI dashboard**: `python scripts/monitor_server.py --session <dir>`
   (background). It only *reads* the state files and serves an HTML page + JSON — it never
   runs training. It is **TensorBoard-like**: besides the stage/run status it plots, with
   plain `<canvas>` (no external/CDN deps, works offline), the **training loss curve** (log-y,
   parsed from `train.log`) and the **open-loop eval curve** (mean MSE per checkpoint, from
   `eval/eval_results.jsonl`). Give the user the URL so they watch remotely instead of the CLI.
2. Start the **watchdog**: `python scripts/watchdog.py --session <dir> --run <run_id>`
   (background). It owns the training subprocess and implements the self-healing loop
   below. It checks status on a cadence of **≤5 minutes** and writes everything it observes
   to `run.json` (which the dashboard surfaces).
3. Start the **eval watcher**: `python scripts/eval_watcher.py --session <dir> --run <run_id>
   --gpu <idle_gpu>` (background). GR00T has no in-loop eval (lessons_learned #13), so this
   gives an **eval curve over training**: it watches `output_dir` for each new *complete*
   `checkpoint-N` (waits for `trainer_state.json`) and runs `open_loop_eval.py` on the held-out
   `eval/` dirs from stage c, on a **separate GPU** so it never steals from training
   (lessons_learned #9), appending mean MSE/MAE per checkpoint to `eval/eval_results.jsonl`.
   It is resumable (skips checkpoints already scored) and does **not** pass
   `--modality-config-path` (read from the checkpoint's `experiment_cfg`, #11).

You may either let `watchdog.py` run autonomously and poll its `run.json`, or drive the
cadence yourself with `/loop` (re-invoking a status check every few minutes). Prefer the
autonomous watchdog for unattended runs; use `/loop` when the user wants you in the loop.

**When the run ends, verify it — don't trust the exit code.** `exit 0` has lied before (a
gated-repo failure and a truncated checkpoint both exited "cleanly"). Run `python
scripts/verify_run.py --session <dir> --run <run_id>`; it writes `VERIFY.md` with independent
verdicts (did loss actually drop? reach target step? is the latest checkpoint resumable AND
inference-loadable? any fatal signature?). Report the VERIFY verdict, not just "done".

**Pick the best checkpoint from the eval curve.** The `eval_watcher` (process 3 above) has
been scoring every checkpoint on the held-out split *throughout* training, so by the end
`eval/eval_results.jsonl` already holds the MSE/MAE curve — choose the checkpoint with the
lowest mean MSE (visible on the dashboard's eval chart). If you need to (re)score a specific
checkpoint manually: `python gr00t/eval/open_loop_eval.py --dataset-path <EVAL_path>
--embodiment-tag <tag> --model-path <output_dir>/checkpoint-N --traj-ids <held-out ids>
--action-horizon 16` (do **not** pass `--modality-config-path` — read from the checkpoint's
`experiment_cfg`, lessons_learned #11). Because these episodes were excluded from training
(stage c), this is a real generalization signal, not memorization. Open-loop still ≠
closed-loop: it picks checkpoints, it does not prove real-robot success.

## The self-healing watchdog contract (requirement 2e)

`watchdog.py` implements — and you must preserve — this contract (full algorithm in
`references/agents.md` and the script itself):

- **Monitor:** parse the train log for `step` and `loss` at least every 5 minutes; record
  throughput and last checkpoint.
- **Assess every poll (record the conclusion):** each cycle, write a human-readable
  `assessment` to `run.json` (the dashboard shows it) — loss trend, plateau length, eval-curve
  state, and a `stop_recommended` flag. Train-loss plateau alone never sets it; a stop is only
  recommended once loss is flat **AND** ≥2 eval points show the eval MSE has stopped improving
  (lessons_learned #14). This turns "is it done?" into a continuously-updated, visible verdict
  instead of a guess.
- **Graceful manual stop:** if `<run_dir>/STOP` exists, stop at the **latest complete
  checkpoint** (resumable, never truncated — #4) and do **not** restart (status `stopped`).
  This is the safe way to honor "just stop it now."
- **Auto-resume on stop:** if the training process exits unexpectedly, **before
  restarting, check whether the latest `checkpoint-N` is resumable** (must contain
  `optimizer.pt` + `trainer_state.json` + `scheduler.pt` + `rng_state*`). If yes, relaunch
  the **same command with the same `--output-dir`** — GR00T then auto-resumes from that
  checkpoint (it calls `resume_from_checkpoint=True` → `get_last_checkpoint`). Never start
  from scratch when a resumable checkpoint exists. See `references/gr00t_resume.md`.
- **Early-stop on trouble:** if loss goes NaN/Inf or diverges (worse than a threshold for
  a sustained window), or the run stalls (no step progress for a timeout), **stop the run**,
  capture the reason, then **re-run** — again resuming from the last good checkpoint, not
  from zero.
- **Classify before retrying (don't loop on config bugs):** on any exit, classify the log
  via `error_patterns.py`. **Fatal** signatures (gated repo, camera-key mismatch, missing
  data, build-header error) recur identically every restart — mark `failed` immediately with
  the fix, do NOT burn restarts on them. **OOM** → surface "lower the batch" before resuming.
  Only genuinely transient failures get the resume-and-retry path.
- **Backoff & cap:** apply capped exponential backoff to crash-restarts and cap the total
  restart count, so a hard-failing config doesn't loop forever. On exceeding the cap, mark
  the run `failed` and surface it.

## Bundled resources

- `references/lessons_learned.md` — concrete gotchas from real GR00T SFT runs + the check
  that prevents each. **Read this in stage a and consult it in d/e.**
- `references/gr00t_resume.md` — exactly how GR00T/HF-Trainer resume works, and what a
  checkpoint must contain to be resumable vs inference-only. **Read before writing the
  watchdog's resume logic or trusting any checkpoint.**
- `references/agents.md` — the full per-stage sub-agent briefs (inputs, steps, output
  schema) and the watchdog algorithm in detail.
- `references/prior_art.md` — the GitHub / docs prior art this skill is built on
  (requirement #4): Anthropic skills spec, multi-agent orchestration skills, SkyPilot /
  MosaicML / Lightning watchdog patterns, FastAPI status-dashboard pattern.
- `scripts/session.py` — file-based session/run state (create, status, set-stage, add-run,
  update-run). Use for all state changes.
- `scripts/check_hardware.py` — GPUs, free memory, disk, `/dev/shm`; prints JSON + warnings.
- `scripts/plan_training.py` — compute steps/epochs/batch and emit the launch command.
- `scripts/preflight.py` — ~2-step smoke test of the real command; catches config bugs cheaply.
- `scripts/error_patterns.py` — shared log classifier: fatal (no-retry) vs oom vs retryable.
- `scripts/watchdog.py` — the self-healing training monitor (auto-resume, early-stop, backoff).
- `scripts/verify_run.py` — post-run independent verification → `VERIFY.md` (don't trust exit 0).
- `scripts/split_train_eval.py` — split a LeRobot v2.1 dataset into disjoint train/eval dirs
  (re-indexes episodes + rebuilds `meta/`); used in stage c to hold out an eval set.
- `scripts/eval_watcher.py` — periodic open-loop eval: scores each new checkpoint on the
  held-out `eval/` dirs (separate GPU), saving scalar metrics to `eval/eval_results.jsonl` and
  any artifact images (e.g. gt-vs-pred trajectory plots) under `eval/artifacts/ckpt-N/<group>/`.
- `scripts/monitor_server.py` — TensorBoard-like FastAPI dashboard (dependency-free `<canvas>`):
  plots the training-loss + open-loop-eval curves, shows the watchdog's `assessment` verdict
  and a per-dataset eval metrics table, and **generically galleries** any images found under
  `eval/artifacts/` (so a richer eval that drops extra outputs is surfaced with no schema).

Run scripts with `python` (or `uv run python` in a uv project). Scripts are designed to be
executed for their output, not read into context — only open one if you need to adapt it.

## Style

This skill spends real money and GPU-hours per run. Bias toward **catching problems before
launch** (stages a–d are cheap; a wasted 6-hour run is not) and toward **honest status**
(if a run is resuming for the 3rd time, say so on the dashboard). Explain the *why* to the
user — most failure modes here are non-obvious infra issues, and a one-line reason ("only
got 50 episodes → 4000 steps, not 10000") builds the trust that keeps them from
second-guessing the plan.
