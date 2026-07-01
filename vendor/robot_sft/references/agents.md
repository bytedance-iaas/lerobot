# Sub-agent briefs & watchdog algorithm

One sub-agent per stage. Each receives the **session directory path** and reads/writes only
through `scripts/session.py` + its own JSON artifact. Keep each agent narrow: it does its
stage, validates, writes, marks the stage `done`, returns a 2–3 line summary. The
orchestrator (main loop in SKILL.md) decides what runs next.

Environment (console pod): the lerobot checkout is **/lerobot** — run all lerobot commands
from there (`cd /lerobot && uv run ...`). All big artifacts (session state, checkpoints,
HF caches) live on **/opt/data** (the roomy persistent volume); `session.py` and
`plan_training.py` default there already.

Artifact convention: every stage writes `<stage>.json` with at least
`{"stage": ..., "status": "done|blocked", "summary": "...", ...stage-specific...}` and then
calls `session.py set-stage <stage> done` (or `blocked`).

---

## a. overview — review & gate

**Goal:** understand intent + the user's current training setup, review it, and gate the
session. **Inputs:** the conversation/request, the lerobot checkout, any training command
the user points to. **Steps:**
1. Resolve the training **entrypoint**. The default here is lerobot's own
   `lerobot-train` — if the user has a dataset and names (or we can infer) a policy type,
   that IS the entrypoint; nothing needs scaffolding. Only mark `blocked` if the request
   needs a training path lerobot doesn't provide (e.g. a custom trainer they haven't
   written) — report that and STOP; do not invent one.
2. Extract the resolved **goal**, **dataset reference** (`repo_id` / local root),
   **policy** (fresh `--policy.type` vs finetune `--policy.path`), and the **parameters
   present vs missing** (batch, steps, output dir, device…).
3. Produce a **review**: walk `references/lessons_learned.md` against this setup and call
   out risks (Hub auth for gated policy bases? shm? steps vs dataset size? output_dir on
   the big disk? camera-key match for --policy.path?), plus general SFT advice (small data
   → modest epochs, held-out eval to pick checkpoints). Be concrete and prioritized.

**Output `overview.json`:** `{goal, entrypoint:"lerobot-train", policy_type|policy_path,
dataset_ref, params_found:{}, params_missing:[], review:[{risk, severity, fix}], status}`.

---

## b. dataset_explore — inspect, never assume

**Trigger:** the user names a dataset / path, OR is vague about it (either way, run this).
**Goal:** know the data cold before planning. **Steps:**
1. Resolve the dataset location: local dir (`--dataset.root`) or HF repo id (cached under
   `$HF_LEROBOT_HOME`, which in the pod resolves onto /opt/data via `HOME`). If HF, note
   whether it needs downloading.
2. Read `meta/info.json`: `codebase_version` (this lerobot expects **v3.0**; v2.x needs
   converting with lerobot's dataset conversion tooling first), `robot_type`, `fps`,
   `total_episodes`, `total_frames`, and `features` (every `observation.images.*` camera
   key, `observation.state` shape, `action` shape).
3. Derive **num training frames** (≈ total_frames minus the frames of episodes that will be
   held out) for the step computation in stage d.
4. If **finetuning a pretrained policy** (`--policy.path`), diff the dataset features
   against the policy's `config.json` input/output features — camera names and state/action
   dims must line up (lessons_learned #6). Fresh `--policy.type` training self-derives
   features from the dataset, so this check is a no-op there.
5. Flag anything off: missing cameras, dim mismatch, weird fps, tiny dataset (warn that a
   handful of episodes won't generalize).

**Output `dataset_explore.json`:** `{path_or_repo, version, needs_conversion:bool,
robot_type, fps, total_episodes, total_frames, est_samples, cameras:[], state_dim,
action_dim, feature_match_ok:bool, warnings:[]}`.

---

## c. data_preprocess — conversion (if needed) + train/eval split (always, unless opted out)

**Trigger:** runs whenever a conversion is needed OR a train/eval split is wanted (the
default). **Steps:**
1. If conversion needed (v2.x dataset → v3.0), run lerobot's converter; ensure build deps
   are present first (lessons_learned #10).
2. **Train/eval split (lessons_learned #13):** real-robot data has no simulator, so in-loop
   eval is off (`--eval_freq=0`) and generalization is judged on **held-out episodes**.
   lerobot subsets episodes natively (`--dataset.episodes='[...]'`), so the split is just
   two deterministic id lists — run
   `python scripts/split_train_eval.py --dataset-repo-id <id> [--dataset-root <dir>]
   --out <session>/preprocess.json` (default ≈10% holdout, min 1, seeded → reproducible).
   **No physical dataset copy is needed.**
3. **Verify:** train∩eval is empty, counts add up, and the ids are < total_episodes.

**Output `preprocess.json`:** the split_train_eval.py output —
`{dataset_repo_id, dataset_root, total_episodes, holdout_frac, seed,
train_episodes:[], eval_episodes:[]}` (+ `warning` when the dataset is tiny). If the user
opted out of a split, set `eval_episodes: []` and note that eval will only sanity-check
learning, not generalization.

---

## d. training_plan — hardware + real parameters (gates train)

**Goal:** a launch command that fits the hardware and the data. **Steps:**
1. Run `python scripts/check_hardware.py --json`. Read GPUs (idle ones), free disk per
   candidate volume, and `/dev/shm` size.
2. **Resolve dataloader workers:** if `/dev/shm` < a few GB and you want workers>0, try to
   remediate (remount; re-check it actually grew) or set `--num_workers=0` and record why.
3. **Pick checkpoint storage:** `--output_dir` on the big volume (pod: `/opt/data/...`).
   lerobot keeps EVERY checkpoint (no rotation), so budget `(steps/save_freq) × ckpt_size`
   (lessons_learned #5).
4. Run `python scripts/plan_training.py --samples <est> --policy-type <t>
   --dataset-repo-id <id> [--dataset-root <dir>] --episodes-file <session>/preprocess.json
   [--gpus N --gpu-mem-gb G] [--epochs E] [--batch-size B] [--cuda IDS]`. It computes
   `steps_per_epoch`, a sane `--steps`, `--batch_size`, `--save_freq`, inlines the train
   episode list, and prints both the launch command AND the resume command. Never silently
   keep lerobot's default `--steps=100000` (lessons_learned #7).
5. **Resumability is built in** — every lerobot checkpoint carries full training state
   (there is no save_only_model to mis-set); just make sure `save_freq` yields enough
   checkpoints for the eval curve to pick from.
6. **Preflight Hub auth** (lessons_learned #1): for `--policy.path` bases or Hub datasets,
   confirm access (`hf auth whoami`) or local paths before committing to a long run.
7. **Smoke-test the plan**: `python scripts/preflight.py --session <dir> --steps 2`. It runs
   the real command for ~2 steps (including one checkpoint save) in a temp output dir and
   classifies the result. If it returns a `fatal` classification (auth, feature mismatch,
   missing data, bad flag…), **fix that before stage e**. It also measures peak GPU memory
   and may emit a `batch_suggestion` (lessons_learned #16) — apply, re-run preflight,
   recompute the plan. Record the verdict in the plan notes.

**Output `training_plan.json`:** plan_training.py's `--json` output (launch_command,
resume_command, output_dir, dataset_repo_id/root, train/eval_episodes, batch_size, steps,
save_freq, num_workers, repo, cuda_visible_devices, ...) + `{shm_ok:bool, notes:[]}`.

---

## e. train — watchdog + dashboard + periodic eval

1. `session.py add-run` → get `run-NNN`. Record the launch command from `training_plan.json`
   into `run.json`.
2. Start dashboard (background): `python scripts/monitor_server.py --session <dir>
   [--port 8770]`. Tell the user the URL (in the console UI it appears in the viewer via
   "+ 打开" → port 8770). It plots the **loss curve** (from `train.log`) and the **held-out
   eval curve** (mean MSE/checkpoint from `eval/eval_results.jsonl`).
3. Start watchdog (background): `python scripts/watchdog.py --session <dir> --run <run-NNN>`.
4. Start the eval watcher (background), on an idle GPU if one exists:
   `python scripts/eval_watcher.py --session <dir> --run <run-NNN> --gpu <idle>`. It scores
   every new complete `checkpoints/<step>` on the held-out episodes with
   `offline_eval.py` and appends to `eval/eval_results.jsonl` — the eval curve *during*
   training (lessons_learned #13). On a single-GPU pod it shares the training GPU; evals
   run between checkpoints and exit, so contention is brief — keep `--eval-timeout` set.
5. Either poll `runs/<run-NNN>/run.json` periodically, or use `/loop` to check on a cadence.
   Report status changes (resuming, early-stopped, failed, done) to the user concisely.
6. **When the run finishes, verify it**: `python scripts/verify_run.py --session <dir>
   --run <run-NNN>`. It writes `VERIFY.md` + `verify.json` with independent pass/fail
   verdicts (progress, loss decreased, loss finite, checkpoint exists, resumable,
   inference-ready, no fatal signature). Surface the overall PASS/FAIL — a clean exit code
   alone is not evidence of success.
7. **Pick the best checkpoint from the eval curve.** By the end, `eval_watcher` has scored
   every checkpoint, so `eval/eval_results.jsonl` holds the held-out MSE/MAE curve — pick
   the lowest mean MSE (also on the dashboard). To (re)score one checkpoint manually:
   `cd /lerobot && uv run python <skill>/scripts/offline_eval.py
   --model-path <output_dir>/checkpoints/<N>/pretrained_model
   --dataset-repo-id <id> [--dataset-root <dir>] --episodes <held-out ids>`.
   Held-out ⇒ real generalization signal; still ≠ closed-loop success (lessons #11).

### Watchdog algorithm (what `watchdog.py` does; preserve this contract)

```
load training_plan + run.json
restarts = 0
launch training subprocess → tee to train.log
  (first launch uses launch_command; if a resumable checkpoint already exists, resume_command)
loop every POLL seconds (POLL ≤ 300):
    parse train.log tail → last_step (tqdm N/M), last_loss ("loss:x"), last_ckpt
    write run.json {status:"running", last_step, last_loss, restarts, assessment, ts}
    # --- trouble detection ---
    if last_loss is NaN/Inf
       or diverged (loss > divergence_threshold)
       or stalled (no step increase for STALL_TIMEOUT):
        record reason; terminate subprocess gracefully  → treat as a stop
    # --- process exit handling ---
    if subprocess exited:
        if exit looked clean AND reached target step → status:"done"; break
        classify log (error_patterns): fatal → status:"failed", DON'T retry
        if restarts >= MAX_RESTARTS: status:"failed"; break
        ck = latest_resumable_checkpoint(output_dir)   # lerobot_resume.md predicate
        if ck: relaunch with resume_command            # --resume=true --config_path=.../last/...
        else if checkpoints/ empty: clear output_dir, relaunch original command
        else: status:"failed" (non-resumable checkpoints present — don't wipe them)
        sleep backoff = min(BACKOFF_CAP, BASE * 2**restarts); restarts += 1
write final run.json status
```

Notes: only crash/early-stop restarts use backoff; a deliberate stop is the STOP-file path,
which waits for the checkpoint's `training_state/` to be complete before SIGTERM
(lessons_learned #4). The watchdog never edits training internals — it only starts/stops
the process and reads files, so it is itself crash-safe and resumable (re-running it
re-reads `run.json`, and it resumes rather than fresh-launches when a checkpoint exists).

Each poll the watchdog also writes a human-readable `assessment` to `run.json` (loss trend,
plateau length, eval-curve state, `stop_recommended`) so the dashboard always shows a current
"keep going / safe to stop" verdict — `stop_recommended` only trips when loss is flat AND ≥2
eval points show eval MSE has stopped improving (lessons_learned #14). A user can stop cleanly
with `touch <run_dir>/STOP` → the watchdog stops at the latest complete checkpoint (status
`stopped`, no restart). Do NOT restart the watchdog while training runs — it would launch a
second (resuming) training process; to change watchdog behaviour mid-run, STOP first.
