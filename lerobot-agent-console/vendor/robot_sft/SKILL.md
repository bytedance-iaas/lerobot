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

- The **lerobot checkout is `/lerobot`** — run every lerobot command from there.
  **Never `uv run`.** Its venv is already resolved, and `uv run` triggers multi-process issues
  here. Use `python -u -m lerobot.scripts.lerobot_train …` for training and plain `python` for
  scripts that import lerobot. `plan_training.py` emits this form by default (`--runner
  python-module`); you should not have to think about it.
- **All big artifacts live on `/opt/data`** (the roomy persistent volume): session state
  defaults to `/opt/data/robot_sft/` (via `session.py`), checkpoints to
  `/opt/data/robot_sft/runs/...` (via `plan_training.py`), and HF caches land under
  `/opt/data/.cache/...` automatically (`HOME=/opt/data` in the pod). **Never** write
  checkpoints or datasets into `/lerobot` or `/tmp`.
- **Use the provided `scripts/` — do NOT hand-write ad-hoc scratch scripts.** Each stage has a
  tool: plan with `plan_training.py` (writes `training_plan.json`) — don't write your own
  `save_plan.py`; explore/split/train/eval/verify via the bundled scripts, and **train by
  running `lerobot-train`** (it streams `tos://` directly — no custom training script). If you
  genuinely need a one-off helper, write it under the **session dir**
  (`/opt/data/robot_sft/sessions/<id>/`), **never `/tmp`** (ephemeral + forbidden above) and
  **never** into the skill's own `scripts/` dir (that's the vendored, versioned tooling).
- Outside the pod both default to the current directory (`.robot_sft/`); override with
  `$ROBOT_SFT_HOME` / `--output-dir`.
- **External models/datasets: prefer `oniond`, then HF-mirror.** For an HF-style `org/name`
  model or dataset, resolve the source with `python scripts/fetch.py {model|dataset} <org/name>`
  FIRST — it downloads from the Volcengine TOS bucket via `oniond` (fast, in-cluster, no AK/SK)
  when the name is in the bucket, else falls back to the HF hub. It prints JSON
  `{"source","repo_id","local_path"}`; then use:
  - model  → `--policy.path=<local_path or repo_id>`
  - dataset→ `--dataset.repo_id=<repo_id>` plus `--dataset.root=<local_path>` when `local_path` is set.

  `oniond` stores files flat, so `local_path` (e.g. `/opt/data/.cache/oniond/pi05_base`) is a
  valid `--policy.path` / `--dataset.root` as-is. `tos://…` refs are passed through untouched
  (StreamingTOSRobotDataset).
- **The HF-mirror fallback: the pod is behind a firewall that blocks `huggingface.co`.** Set
  `export HF_ENDPOINT=https://hf-mirror.com` at session start; every Hub pull (`--dataset.repo_id`,
  pretrained backbones via `--policy.path`, the `hf` CLI) then goes through the mirror.
  TOS/`StreamingTOSRobotDataset` datasets don't touch the Hub.
- **Gated / private repos need a token — the pod has none, so PROMPT THE USER for it.** pi0's
  PaliGemma backbone and many bases are gated, and private datasets need auth. If stage a/b
  involves a gated backbone or a private repo, ask the user for their HuggingFace token (`hf_…`,
  from https://huggingface.co/settings/tokens). Surface this **up front**, not after a 6-hour run
  fails on `401/403`. Never hardcode the token — it comes from the user.
- **Creds the user gives in chat go in with one command — never `export`, never an rc file.**
  ```bash
  # value on STDIN with a QUOTED heredoc, so $ ' " | ; in a token are all safe
  python /opt/data/skills/robot_sft/scripts/multinode.py env set HF_TOKEN --worker <addr> <<'EOF'
  hf_xxxxxxxxxxxxxxxxxxxx
  EOF
  python .../multinode.py env show HF_TOKEN --worker <addr>   # verify — prints LENGTHS, not values
  ```
  Same for `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` / `HF_ENDPOINT`. Drop `--worker` on a single node.
  It reaches every shell on every node, including the ones ssh starts, and the value never
  touches a command line. A *wrong* credential is worse than a missing one (`test -n` passes and
  you 403 hours later), so always `env show` afterwards; `env unset` clears a bad one.
  **`references/credentials.md`** — why `export` and rc files fail, which shell reads which hook,
  and what to do when a value is set but invisible.
- **Session** = one user intent ("fine-tune ACT on my SO-101 pick-place data"). One
  session directory under `<root>/sessions/<session_id>/`.
- **Stages** = the pipeline below (a→e). Each writes a JSON artifact and flips its status
  in `session.json`. Stages are resumable: on re-entry, skip any stage already `done`.
- **Runs** = one independent **training lineage** (its own loss + eval + open-loop MSE curves),
  under `runs/<run_id>/`. **A resume from a checkpoint is the SAME run — NOT a new one.** The
  watchdog re-launches with `--resume=true` and **appends** to that run's `train.log`, so its
  curves continue unbroken across crashes/restarts (its `restarts` counter increments; the
  `run_id` does NOT change).
  - **Resume/restart an interrupted training → reuse the SAME `run_id`** (re-run
    `watchdog.py --run <same_run_id>`; it auto-resumes from `checkpoints/last`). Do **NOT**
    `session.py add-run` — a new run starts a fresh curve and, sharing the plan's one
    `output_dir`, hits `output_dir exists but --resume not set`.
  - **`add-run` (a new `run-NNN`) is ONLY for a genuinely fresh attempt** — a different config,
    a new `output_dir`, or an explicit train-from-scratch. Those are legitimately independent runs.

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
4. **Confirm the plan before training (HARD GATE).** After stage d (plan + green preflight)
   and BEFORE launching stage e, summarize the final training parameters and ask the user
   to confirm or adjust — wait for explicit go-ahead. See stage **d★**. Never start a run
   on assumptions.
5. **Hand off to the watchdog** for the train stage and keep the user informed via the
   dashboard, not a wall of CLI text.

Keep a visible checklist in your reply so the user can see where the session is:

```
- [ ] a. overview / review
- [ ] b. dataset explore
- [ ] c. train/eval episode split (conversion if needed)
- [ ] d. training plan + hardware
- [ ] d★. confirm parameters with the user (gate before GPU-hours)
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

**TOS datasets (object storage) — explore without downloading.** When the user points at a
`tos://bucket/prefix` dataset, do NOT download it. First `ls` the tree with fsspec to confirm
the LeRobot layout (`meta/ data/ videos/`), then load metadata with **`StreamingTOSRobotDataset`**
(see "Streaming a dataset from TOS" below) and read `num_frames`, `num_episodes`, `fps`,
`meta.camera_keys`, and the state/action feature shapes — all from the mirrored `meta/` (a few
MB), no bulk download. Write the same `dataset_explore.json` fields as for a local/Hub dataset.
```python
from lerobot.datasets import StreamingTOSRobotDataset   # reads TOS creds from env
ds = StreamingTOSRobotDataset("tos://bucket/prefix/<name>")  # metadata only, no bulk download
print(ds.num_frames, ds.num_episodes, ds.fps, ds.meta.camera_keys)
```
Credentials come from the environment (`TOS_ACCESS_KEY` / `TOS_SECRET_KEY`, + optional
`TOS_ENDPOINT` / `TOS_REGION`), installed with `multinode.py env set` (see Credentials).
No `storage_options` plumbing needed (pass it only to override).

### c. Train/eval episode split  (always, unless the user opts out; conversion is conditional)
Two responsibilities:
- **Conversion (conditional):** if stage b found a v2.x dataset, convert it with lerobot's
  tooling, then re-verify `meta/info.json`.
- **Train/eval split (always — unless the user opts out):** hold out a fraction of
  episodes as an eval set *before* training, so the training set does not contain them.
  Real-robot data has no simulator, so in-loop eval is off (`--env_eval_freq=0`) and post-hoc
  offline eval on **held-out episodes** is the only honest generalization signal
  (lessons_learned #13). lerobot subsets episodes natively (`--dataset.episodes='[...]'`),
  so **no physical dataset copy is needed** — run `python scripts/split_train_eval.py
  --dataset-repo-id <id> [--dataset-root <dir>] --out <session>/preprocess.json`
  (default ≈10% holdout, min 1, seeded/deterministic). Verify the two id lists are
  disjoint and in range.
  - **TOS datasets:** `split_train_eval.py` **stream-reads `total_episodes` from
    `tos://…/meta/info.json` via fsspec** (creds from env) — just pass `--dataset-repo-id
    tos://<bucket>/<prefix>/<name>`, no `--total-episodes` needed (pass it only to override).
    The split itself is the same: it writes disjoint id lists (`train_episodes` /
    `eval_episodes`); `lerobot-train --dataset.episodes='[...]'` subsets on TOS with nothing copied.

Writes `preprocess.json` with `train_episodes` + `eval_episodes`. If the user opted out,
record `eval_episodes: []` (eval then only sanity-checks learning, not generalization).

### d. Training plan & hardware  (always; gates the train stage)
Run `python scripts/check_hardware.py` and `python scripts/plan_training.py`. This stage:
- **Flags (exact names):** `plan_training.py --dataset-repo-id <id|tos://…> --gpus <n>
  --gpu-mem-gb <g> --cuda <idx> --policy-type <act|…> [--policy-path <pretrained>]
  --episodes-file <session>/preprocess.json --output-dir <run_dir> [--out <path>]`.
  Note **`--gpus`** (plural, GPU count) and **`--gpu-mem-gb`** — not `--gpu`; `--cuda` is
  `CUDA_VISIBLE_DEVICES` (e.g. `0`). **Use `--out <file>` to write the JSON plan** — not shell
  `>` redirection.
  It **auto-reads `total_frames`/`total_episodes` from the dataset meta** (local, or stream-read
  from a `tos://…/meta/info.json`), and with `--episodes-file` sizes the **train subset** — so
  you normally don't pass `--samples`/`--episodes` (they're optional overrides). The generated
  command already carries `--env_eval_freq=0` + `--policy.push_to_hub=false`.
  - **torch.compile is OFF by default.** Leave it off unless you measure a win.
  - **pi0/pi05 CANNOT be trained with FSDP — use DDP.** Every wrap policy fails the same way;
    this is settled, do not re-litigate it per run.
  - **FSDP and fp8 do not work together — pick one.**
  - **fp8 changes numerics: NEVER enable it silently, ASK THE USER first.**
  - Reasoning, measurements, and the actual configuration for all four:
    **`references/fsdp_fp8.md`**. Read it before enabling anything here.
- Checks **GPU count + free memory** (pick idle GPUs), **disk space** for checkpoints
  (must be on `/opt/data` in the pod — and lerobot keeps EVERY checkpoint, no rotation,
  so budget `(steps/save_freq) × ckpt_size`), and **`/dev/shm` size**.
- **Ensures multi-process dataloading works**: if `/dev/shm` is too small for
  `--num_workers>0`, either remediate (remount larger — needs the user / sufficient caps)
  or fall back to `--num_workers=0`, and say which and why.
- **Computes real steps from the data**, not lerobot's 100k default:
  `steps_per_epoch = ceil(num_frames / batch_size)`, `steps = epochs × steps_per_epoch`,
  with a sane epoch count for the dataset size. See `plan_training.py`.
- Emits `train_command` (the train episode list inlined) — which **the watchdog runs, never
  you**; follow the plan's `how_to_launch` instead — and the
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

**⚠️ SLOW COMMANDS MUST STREAM PROGRESS — never run them synchronously.** The terminal
tool returns output only when the command *finishes*, so a synchronous `preflight.py`
(1–3+ min), `fetch.py` (dataset/model download, minutes to hours), or any command expected
to take >30 s shows the user NOTHING until it ends — it looks dead/stuck. Instead:
1. Launch in the background with output to a log:
   `python scripts/preflight.py --session <dir> > /tmp/preflight_run.log 2>&1 &`
   (same pattern for `fetch.py` → `/tmp/fetch_run.log`).
2. Poll every ~20–30 s with a QUICK command: `tail -5 /tmp/preflight_run.log`, and relay
   the latest progress line (both scripts print heartbeats: preflight every 30 s with
   elapsed/GPU-mem/last-log-line; fetch prints download progress) to the user each poll.
3. Detect completion by the process exiting (`kill -0 <pid>` fails, or the final JSON
   verdict / summary appears in the log), then report the full result.
Never leave the user >60 s without a visible progress update during any slow stage.

**🚫 Never use the `process` tool's `wait` action for this.** It BLOCKS — up to
`TERMINAL_TIMEOUT`, 180 s by default — and emits nothing while it does: no tokens, no tool
updates, no log tail. Three minutes of that is indistinguishable from a hung agent, and
repeating it (`wait` → timeout → `wait` → …) is how a 2-step smoke test turned into 13 minutes
of a motionless UI when the run itself had already finished. Worse, it keeps blocking after the
process is GONE: the wait ends on its own timer, not on the process dying.
Use the non-blocking actions — `process` supports
`list · poll · log · wait · kill · write · submit · close`:
- **`poll`** — instant status + the tail of new output. This is the polling primitive.
- **`log`** — full output with pagination, when the tail is not enough.
- plain `tail -5 <logfile>` is equally fine and is what the steps above assume.
Between polls, RELAY the newest progress line to the user. `wait` is defensible only for
something known to finish in seconds, and `poll` is better even then.

**Camera pre-check (finetuning a pretrained VLA).** When `--policy.path` is a pretrained
checkpoint (`lerobot/pi05_base`, `pi0_base`, `smolvla_base`, …), both `plan_training.py`
and `preflight.py` run `scripts/check_features.py` — a 1-second static compare of the dataset's
`observation.images.*` keys vs the checkpoint's expected ones (two small JSONs, no torch, no
weights). On a mismatch (e.g. checkpoint wants DROID's `base_0_rgb / left_wrist_0_rgb /
right_wrist_0_rgb` but the dataset has `front / wrist`) they **auto-add a `--rename_map`** so the
finetune just works, instead of crashing deep in `make_policy` after a slow model load:
- the dataset's cameras are renamed onto the checkpoint's (sorted order — pi0/pi05 have no
  per-camera slot embedding, so the pairing is arbitrary; the rename is saved with the
  preprocessor for eval);
- any checkpoint camera the dataset doesn't cover is auto-fed a black, attention-masked image at
  runtime (`modeling_pi05.py`), so the **pretrained weights are kept** — no need to train from
  scratch. A `--rename_map` also makes lerobot skip visual-feature validation (`factory.py:650`).
Run standalone: `python scripts/check_features.py --dataset-repo-id <id> --policy-path <ckpt>`.

**`--rename_map` is a `lerobot-train` flag, not a planning input.** To set one explicitly,
either append `--rename_map='<json>'` to the final training command, or pass
`--rename-map '<json>'` to `plan_training.py` — a pure passthrough that validates the JSON and
appends `--rename_map='<json>'` to the generated command. NOT needed when training from scratch
with `--policy.type`: there the config derives its cameras from the dataset, so nothing to rename.

**Right-size the batch from measured memory (don't fly blind):** the planner picks an
initial batch from a policy-family heuristic *before* seeing real usage. preflight samples
**peak GPU memory** during the smoke run and emits a `batch_suggestion` (scale
`--batch_size` to ~85% of memory). Apply it, **re-run preflight to confirm it fits**, then
recompute `--steps`/`--save_freq` for the new batch (and consider scaling LR). A bigger
batch that fits = higher GPU utilization and a faster run (lessons_learned #16).

**⚠️ PITFALL: preflight memory measurement is noisy on 2-step runs.** Peak GPU memory
sampled over only 2 steps can vary wildly (e.g. batch=50 measured 6015 MB in one run and
23913 MB in another — 4× spread). When the `batch_suggestion` seems implausible (e.g.
suggesting 173 from 50), ignore it — use the **highest** stable reading across 2–3
preflight runs, and never push past ~85% of total memory. When in doubt, prefer a
conservative batch that passed preflight cleanly over an aggressive suggestion.

### d★. Confirm the plan with the user  (HARD GATE — before any real GPU-hours)
Once preflight is green and the batch is right-sized, **do NOT launch training silently.**
This is the last cheap checkpoint before committing GPU-hours, so **always** present a concise
**training summary** and get the user's **explicit go-ahead** first. Summarize the final,
about-to-run values from `training_plan.json` (not the defaults) — at minimum:
- **dataset** (`repo_id` / `root`) + **train / eval episode counts**
- **policy** (fresh `type`, or the finetuned `path`) and the key camera/state features
- **steps** (and the epochs + `steps_per_epoch` they came from), **batch_size**, **num_workers**
- **learning rate** (if set), **save_freq** and the derived **eval cadence** (≈ once per N min/h)
- **precision**: bf16 (default) or **fp8** — state it explicitly and, when proposing fp8, say what it
  costs: changed numerics, and a checkpoint that only loads on a TE-enabled lerobot. Never enable
  fp8 without the user's explicit yes.
- **output_dir**, the **GPU(s)** chosen, and the checkpoint **disk budget** `(steps/save_freq) × ckpt_size`
- the exact `train_command` that the watchdog will run

Then **ask the user to confirm or change it, and WAIT** — use `AskUserQuestion` if your runtime
has it (e.g. options 「开始训练」/「修改参数」/「取消」), otherwise ask in plain chat and block on the
reply. If they want changes, re-run `plan_training.py` with the new args (and re-run
`preflight.py` when a batch/memory-affecting flag changed), then **re-summarize and re-ask**.
Only advance to stage e after an explicit "go". Never start a run on assumptions — this
summary-and-confirm step is mandatory, even if the plan looks obviously fine.

### e. Train under watchdog + periodic eval  (the long pole)
First `session.py add-run` **ONCE** to get the `run_id` for this training lineage. **Reuse that
same `run_id` for every resume/restart** — the watchdog auto-resumes into it and its curves
continue unbroken. Only `add-run` again for a genuinely fresh attempt (new config / from scratch;
see **Runs** above). Do NOT mint a new run per crash — that fragments the loss/eval curves and
hits `output_dir exists but --resume not set`.

Then launch the monitors — three background processes, all communicating via files:
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
3. **Eval — branch on whether a SPARE GPU exists** (`plan_training.py` reports this as
   `eval_mode`: `concurrent` vs `post_training`, plus an `eval_gpu_note`):
   - **A spare idle GPU exists → concurrent eval.** Start the **eval watcher**:
     `python scripts/eval_watcher.py --session <dir> --run <run_id> --gpu <idle_gpu>` (background,
     on a GPU **off** the training ones). It watches `output_dir/checkpoints/` for each new
     *complete* step dir (waits for `training_state/training_step.json`) and runs `offline_eval.py`
     (open-loop replay) on the held-out episodes from stage c, appending mean MSE/MAE per checkpoint
     to `eval/eval_results.jsonl`. Resumable (skips scored checkpoints).
   - **⚠️ SINGLE-GPU (no spare — training owns the only GPU) → DO NOT run the eval watcher during
     training** (it would contend with / OOM the training GPU; `eval_mode: post_training`). **Say so
     UP FRONT in the plan you show the user** ("only 1 GPU → eval can't run during training; we'll
     eval after it finishes"). Then AFTER training finishes, eval the **last few checkpoints** on the
     freed GPU and pick the best by held-out MSE — `plan_training` prints the exact command:
     `python scripts/offline_eval.py --model-path <output_dir>/checkpoints/<STEP>/pretrained_model
     --dataset-repo-id <id> --episodes <held-out eps> --device cuda` (run it for the last 2–3
     checkpoints).

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
manually: `cd /lerobot && python <skill>/scripts/offline_eval.py --model-path
<output_dir>/checkpoints/<N>/pretrained_model --dataset-repo-id <id> [--dataset-root <dir>]
--episodes <held-out ids>`. Because these episodes were excluded from training (stage c),
this is a real generalization signal, not memorization. Open-loop still ≠ closed-loop: it
picks checkpoints, it does not prove real-robot success.

## 跨机训练 (multi-node / cross-machine distributed)
Only when the user **explicitly asks** for cross-node / multi-machine / 跨机 training (2+ GPU
boxes). **`Read references/multinode.md` first — all of it — before running anything.** It holds
the M0 communication gate, the mandatory smoke test, the per-node launch/resume commands, and
what `scripts/multinode.py` (`check · status · rdma · env · resume · clean · launch`) enforces.
Do not reconstruct any of it from memory: the master and the workers fail in different ways, and
the one that reports the error is usually not the one that is wrong.

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
- `references/multinode.md` — 跨机训练 in full: the M0 communication gate, the mandatory smoke
  test, per-node launch/resume, and what multinode.py enforces. Read before any cross-node run.
- `references/fsdp_fp8.md` — FSDP / fp8 / torch.compile: the verdicts, the measurements behind
  them, and the configuration. Read before enabling any of the three.
- `references/credentials.md` — why credentials go in via `env set` and nothing else; which shell
  reads which hook; what to do when a value is set but invisible.
- `references/tos_streaming.md` — training straight off a `tos://` dataset: layout, the
  obfuscated-key trap, the verified end-to-end path.
- `references/policy_selection.md` — choosing a policy/backbone, and the trade-offs behind it.
- `references/lerobot_entrypoint.md` — how the `lerobot-train` entrypoint resolves and what it
  actually executes.
- `references/gr00t_resume.md` — GR00T-specific checkpoint/resume behaviour.
- `references/prior_art.md` — the GitHub / docs prior art this skill is built on:
  Anthropic skills spec, multi-agent orchestration skills, SkyPilot / MosaicML / Lightning
  watchdog patterns, FastAPI status-dashboard pattern.
- `scripts/session.py` — file-based session/run state (create, status, set-stage, add-run,
  update-run). Use for all state changes. Defaults to `/opt/data/robot_sft` in the pod.
- `scripts/check_hardware.py` — GPUs, free memory, disk, `/dev/shm`; prints JSON + warnings.
- `scripts/plan_training.py` — compute steps/epochs/batch and emit the `lerobot-train`
  train + resume commands. Defaults to `python -u -m lerobot.scripts.lerobot_train`;
  `--runner uv` switches to `uv run lerobot-train` (do not — see Environment). Use
  `--out <file>` to write the JSON plan.
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
  env": run it from `/lerobot` with plain `python`.
  **If the run trained with a camera rename, eval needs the same map** — pass
  `--rename-map '<the plan's camera_rename_map as JSON>'`, or the checkpoint's expected keys
  (`observation.images.base_0_rgb` …) will not match the dataset's (`observation.images.front`
  …) and make_policy dies with "Feature mismatch". `eval_watcher.py` passes it automatically.
  Runs HF-offline by default — by eval time everything is cached, and reaching the Hub from this
  pod only times out (measured 8m35s vs 2m0s). `--online` opts back in.
- `scripts/eval_watcher.py` — periodic offline eval: scores each new checkpoint on the
  held-out episodes (separate GPU when available), saving metrics to
  `eval/eval_results.jsonl` and plots under `eval/artifacts/ckpt-N/<group>/`.
- `scripts/monitor_server.py` — TensorBoard-like FastAPI dashboard (dependency-free
  `<canvas>`): plots the training-loss + eval curves, shows the watchdog's `assessment`
  verdict, and galleries any images found under `eval/artifacts/`.
- `scripts/multinode.py` — cross-node helper: `check` (all M0 gates), `status` (per-node
  phase while the launch is silent), `clean` (kill orphaned ranks everywhere + verify the
  GPUs came back), `launch` (start workers with correct remote redirects + recorded PIDs).
  Use it instead of hand-rolled ssh — see 跨机训练.
- `lerobot.datasets.StreamingTOSRobotDataset` (in the lerobot package, not this skill) —
  stream a LeRobot v3.0 dataset from object storage (Volcengine **TOS** via `tosfs`, or S3)
  without downloading it. See "Streaming a dataset from TOS" below.

Run scripts with `python`, from `/lerobot` for the ones that import lerobot (only
`offline_eval.py`; the rest are stdlib). Scripts are designed to be executed
for their output, not read into context — only open one if you need to adapt it.

## Streaming a dataset from TOS (object storage)
Only when the dataset lives in TOS (`tos://…`) instead of local disk or the Hub. See
**`references/tos_streaming.md`** — layout requirements, the obfuscated-credential trap, and the
verified end-to-end path. Credentials go in via `multinode.py env set`, never an rc file.

## Style

This skill spends real money and GPU-hours per run. Bias toward **catching problems before
launch** (stages a–d are cheap; a wasted 6-hour run is not) and toward **honest status**
(if a run is resuming for the 3rd time, say so on the dashboard). Explain the *why* to the
user — most failure modes here are non-obvious infra issues, and a one-line reason ("only
got 50 episodes → 6400 steps, not 100000") builds the trust that keeps them from
second-guessing the plan.
