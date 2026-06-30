# Sub-agent briefs & watchdog algorithm

One sub-agent per stage. Each receives the **session directory path** and reads/writes only
through `scripts/session.py` + its own JSON artifact. Keep each agent narrow: it does its
stage, validates, writes, marks the stage `done`, returns a 2–3 line summary. The
orchestrator (main loop in SKILL.md) decides what runs next.

Artifact convention: every stage writes `<stage>.json` with at least
`{"stage": ..., "status": "done|blocked", "summary": "...", ...stage-specific...}` and then
calls `session.py set-stage <stage> done` (or `blocked`).

---

## a. overview — review & gate

**Goal:** understand intent + the user's current training setup, review it, and gate the
session. **Inputs:** the conversation/request, the repo, any training script the user
points to. **Steps:**
1. Locate the training **entrypoint** (e.g. `examples/finetune.sh`, a `launch_*.py`, or a
   command the user gave). **If none exists, write `overview.json` with `status:"blocked"`,
   reason "no training entrypoint", and STOP the session** — do not scaffold one.
2. Extract the resolved **goal**, **embodiment tag**, **dataset reference**, **base model**,
   and the **parameters present vs missing** (batch, lr, steps, output dir, modality config…).
3. Produce a **review** (requirement #1): walk `references/lessons_learned.md` against this
   setup and call out risks (gated backbone? shm? steps vs dataset size? save_only_model?
   camera keys?), plus general SFT advice (freeze the VLM for small data, augmentation,
   open-loop eval to pick checkpoints). Be concrete and prioritized.

**Output `overview.json`:** `{goal, entrypoint, base_model, embodiment_tag,
dataset_ref, params_found:{}, params_missing:[], review:[{risk, severity, fix}], status}`.

---

## b. dataset_explore — inspect, never assume

**Trigger:** the user names a dataset / path, OR is vague about it (either way, run this).
**Goal:** know the data cold before planning. **Steps:**
1. Resolve the dataset location (local path or HF repo id). If HF, note whether it needs
   downloading.
2. Read `meta/info.json`: `codebase_version` (v2.1 vs v3.0 → conversion?), `robot_type`,
   `fps`, `total_episodes`, `total_frames`, and `features` (every `observation.images.*`
   camera key, `observation.state` shape, `action` shape).
3. Derive **num training samples** (≈ total_frames, modulo the action-horizon windowing the
   trainer uses) for the step computation in stage d.
4. Determine the **modality mapping** needed: which GR00T modality keys (`front`, `wrist`,
   `single_arm`, `gripper`, …) map to which real columns, and whether the existing
   `modality.json` matches (catch the `front`-vs-`top` class of bug here).
5. Flag anything off: missing cameras, dim mismatch, weird fps, tiny dataset (warn that
   pure-public data won't cross the calibration/viewpoint gap to a real robot).

**Output `dataset_explore.json`:** `{path_or_repo, version, needs_conversion:bool,
robot_type, fps, total_episodes, total_frames, est_samples, cameras:[], state_dim,
action_dim, modality_ok:bool, modality_fixes:[], warnings:[]}`.

---

## c. data_preprocess — conversion (if needed) + train/eval split (always, unless opted out)

**Trigger:** runs whenever a conversion/modality fix is needed OR a train/eval split is wanted
(the default). **Steps:**
1. If conversion needed (e.g. LeRobot v3.0→v2.1), run the project's converter; ensure build
   deps are present first (e.g. `pythonX.Y-dev` headers — lessons_learned #10).
2. Author/place the corrected `modality.json` in the dataset `meta/` so camera keys + state/
   action slices match the real columns (lessons_learned #6).
3. **Train/eval split (lessons_learned #13):** GR00T's sharded finetune path can't do in-loop
   eval, so hold out episodes **before** training. For each dataset, reserve ≈10% of episodes
   (min 1; warn if too tiny) into a sibling `eval/` dataset dir and keep the rest in `train/`.
   A LeRobot v2.1 split = copy the chosen `data/chunk-*/episode_*.parquet` + matching
   `videos/chunk-*/<cam>/episode_*.mp4`, then **rebuild `meta/` for BOTH** dirs with
   contiguous re-indexed episode ids: `episodes.jsonl`, `tasks.jsonl`, `info.json`
   (`total_episodes`, `total_frames`), and copy `modality.json` into each. Use the same RNG
   seed each run so the split is reproducible/resumable. The training launch then points
   `--dataset-path` at the `train/` dirs; eval (stage e) points `open_loop_eval.py` at the
   `eval/` dirs with the held-out ids.
4. **Verify** structure of every resulting dir: `meta/` (info/episodes/tasks/modality),
   `data/chunk-*` parquet per episode, `videos/chunk-*/<camera>/` mp4; confirm train∩eval
   episode sets are empty and counts add up. Don't trust the converter/splitter blindly.

**Output `preprocess.json`:** `{status, actions_taken:[], verified:bool, split_done:bool,
holdout_frac, per_dataset:[{repo, train_path, eval_path, train_eps, eval_eps, eval_traj_ids:[]}]}`.
If the user opted out of a split, set `eval_path = train_path` and `split_done:false`.

---

## d. training_plan — hardware + real parameters (gates train)

**Goal:** a launch command that fits the hardware and the data. **Steps:**
1. Run `python scripts/check_hardware.py --json`. Read GPUs (idle ones), free disk per
   candidate volume, and `/dev/shm` size.
2. **Resolve dataloader workers:** if `/dev/shm` < a few GB and you want workers>0, try to
   remediate (remount; re-check it actually grew) or set `num_workers=0` and record why.
3. **Pick checkpoint storage:** a volume with `save_total_limit × ~12 GB` headroom (GR00T
   checkpoints are big). Not a near-full root.
4. Run `python scripts/plan_training.py --samples <est> --gpus <n> --gpu-mem-gb <g>
   [--epochs E] [--global-batch B]`. It computes `steps_per_epoch`, a sane `max_steps`,
   `per_device_batch`, `save_steps`, and prints a ready launch command. Adjust if the user
   has preferences; never silently keep a hardcoded default that ignores dataset size
   (lessons_learned #7).
5. **Guarantee resumability:** ensure `--save_only_model` is NOT in the command, and that
   `save_steps`/`save_total_limit` keep enough checkpoints to pick the best by eval.
6. **Preflight the gated backbone / HF auth** (lessons_learned #1): confirm weights are
   reachable (local paths preferred) before committing to a long run.

**Output `training_plan.json`:** `{launch_command, output_dir, cuda_visible_devices,
global_batch_size, per_device_batch, num_workers, max_steps, steps_per_epoch, epochs,
save_steps, save_total_limit, shm_ok:bool, notes:[]}`.

7. **Smoke-test the plan** (cheap insurance, idea from ml-intern): run
   `python scripts/preflight.py --session <dir> --steps 2`. It runs the real command for ~2
   steps in a temp output dir and classifies the result. If it returns non-zero with a
   `fatal` classification (gated repo, camera-key, missing data…), **fix that before stage e**
   — it would otherwise recur minutes into the real run. Record the verdict in the plan notes.

---

## e. train — watchdog + dashboard + periodic eval

1. `session.py add-run` → get `run-NNN`. Record the launch command from `training_plan.json`
   into `run.json`.
2. Start dashboard (background): `python scripts/monitor_server.py --session <dir>
   [--port 8770]`. Tell the user the URL. It's TensorBoard-like — it plots the **loss curve**
   (from `train.log`) and the **open-loop eval curve** (mean MSE/checkpoint from
   `eval/eval_results.jsonl`) with dependency-free `<canvas>`.
3. Start watchdog (background): `python scripts/watchdog.py --session <dir> --run <run-NNN>`.
4. Start the eval watcher (background) on an **idle** GPU (not the training GPUs):
   `python scripts/eval_watcher.py --session <dir> --run <run-NNN> --gpu <idle>`. It scores
   every new complete `checkpoint-N` on the held-out `eval/` dirs and appends to
   `eval/eval_results.jsonl` — giving an eval curve *during* training despite GR00T having no
   in-loop eval (lessons_learned #13). Resumable; one eval GPU keeps it off the training path.
5. Either poll `runs/<run-NNN>/run.json` periodically, or use `/loop` to check on a cadence.
   Report status changes (resuming, early-stopped, failed, done) to the user concisely.
6. **When the run finishes, verify it** (idea from ml-intern's VERIFY.md): run
   `python scripts/verify_run.py --session <dir> --run <run-NNN>`. It writes `VERIFY.md` +
   `verify.json` with independent pass/fail verdicts (progress, loss decreased, loss finite,
   checkpoint exists, resumable, inference-ready, no fatal signature). Surface the overall
   PASS/FAIL — a clean exit code alone is not evidence of success.
7. **Pick the best checkpoint from the eval curve.** By the end, `eval_watcher` has scored
   every checkpoint, so `eval/eval_results.jsonl` already holds the held-out MSE/MAE curve —
   pick the lowest mean MSE (also shown on the dashboard eval chart). To (re)score one
   checkpoint manually: `python gr00t/eval/open_loop_eval.py --dataset-path <eval_path>
   --embodiment-tag <tag> --model-path <output_dir>/checkpoint-N --traj-ids <ids>
   --action-horizon 16` (do NOT pass `--modality-config-path` — read from the checkpoint's
   `experiment_cfg`, lessons_learned #11). Held-out ⇒ real generalization signal; still ≠
   closed-loop success.

### Watchdog algorithm (what `watchdog.py` does; preserve this contract)

```
load training_plan + run.json
restarts = 0
launch training subprocess (CUDA_VISIBLE_DEVICES, cmd) → tee to train.log
loop every POLL seconds (POLL ≤ 300):
    parse train.log tail → last_step, last_loss, last_ckpt
    write run.json {status:"running", last_step, last_loss, throughput, restarts, ts}
    # --- trouble detection ---
    if last_loss is NaN/Inf
       or diverged (loss > divergence_threshold sustained over a window)
       or stalled (no step increase for STALL_TIMEOUT):
        record reason; terminate subprocess gracefully  → treat as a stop
    # --- process exit handling ---
    if subprocess exited:
        if exit looked clean AND reached target step → status:"done"; break
        ck = latest_resumable_checkpoint(output_dir)   # see gr00t_resume.md predicate
        if restarts >= MAX_RESTARTS:
            status:"failed"; write reason; break
        if ck is None:
            note "no resumable checkpoint — restart would be from scratch"
        sleep backoff = min(BACKOFF_CAP, BASE * 2**restarts)
        restarts += 1
        relaunch SAME command, SAME --output-dir  # auto-resumes from ck if present
write final run.json status
```

Notes: only crash/early-stop restarts use backoff; a deliberate stop-at-step is a separate
path that waits for `trainer_state.json` before SIGTERM (lessons_learned #4). The watchdog
never edits training internals — it only starts/stops the process and reads files, so it is
itself crash-safe and resumable (re-running it re-reads `run.json`).

Each poll the watchdog also writes a human-readable `assessment` to `run.json` (loss trend,
plateau length, eval-curve state, `stop_recommended`) so the dashboard always shows a current
"keep going / safe to stop" verdict — `stop_recommended` only trips when loss is flat AND ≥2
eval points show eval MSE has stopped improving (lessons_learned #14). A user can stop cleanly
with `touch <run_dir>/STOP` → the watchdog stops at the latest complete checkpoint (status
`stopped`, no restart). Do NOT restart the watchdog while training runs — it would launch a
second (resuming) training process; to change watchdog behaviour mid-run, STOP first.
