# robot_sft — resumable multi-agent robot SFT skill (lerobot)

A Claude Code [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that turns one
supervised fine-tuning (SFT) **session** for a robot imitation-learning / VLA policy trained
with **[lerobot](https://github.com/huggingface/lerobot)** (`lerobot-train` — ACT, Diffusion
Policy, pi0, SmolVLA, ...) into a sequence of small, independently verifiable,
file-checkpointed **stages**, each run by a focused sub-agent — so a crash or context reset
never loses progress (re-read the session state and continue).

It exists because robot SFT fails in boring, expensive ways: a gated Hub backbone you can't
download, a 64 MB `/dev/shm` that kills the dataloader 90 minutes in, a checkpoint truncated by
a bad kill, a camera key that doesn't match the policy, 100000 steps when the data justifies
5000, or an eval that never tells you whether the policy generalizes. Each of those has a
concrete check that prevents it — see [`references/lessons_learned.md`](references/lessons_learned.md).

> **Environment:** in the LeRobot Agent Console pod the lerobot checkout is `/lerobot` (run
> commands from there via `uv run`) and all big artifacts (sessions, checkpoints, caches) live
> on the roomy `/opt/data` volume — the scripts default there. Outside the pod they default to
> `./.robot_sft`.

## Pipeline (one sub-agent per stage; all state in files)

| Stage | What it does |
|-------|--------------|
| **a. overview** | review the user's setup against the known failure modes; the entrypoint is `lerobot-train` |
| **b. dataset explore** | inspect LeRobotDataset `meta/info.json` — version, episodes, camera keys, state/action dims — catch mismatches *before* a run |
| **c. split** | hold out a deterministic **train/eval episode split** (id lists; lerobot subsets via `--dataset.episodes`, no dataset copy); convert v2.x→v3.0 if needed |
| **d. plan + preflight** | compute `--steps`/`--batch_size`/`--save_freq` from data + hardware, emit the `lerobot-train` launch + resume commands, then a ~2-step smoke test catches config bugs in minutes |
| **e. train** | launch under a **self-healing watchdog** + a **periodic offline eval** on held-out episodes + a **TensorBoard-like dashboard** |

## Key components (`scripts/`)

- `session.py` — file-based session/run state (the single source of truth; everything resumable).
- `check_hardware.py` / `plan_training.py` / `preflight.py` — hardware probe, data-driven
  `lerobot-train` plan, cheap smoke test (incl. one checkpoint save + GPU-memory headroom).
- `split_train_eval.py` — deterministic held-out episode split (two disjoint id lists; no
  physical dataset copy — lerobot subsets episodes natively).
- `watchdog.py` — owns training; auto-resumes via `--resume=true` from the last *resumable*
  checkpoint, early-stops on NaN/divergence/stall, classifies fatal-vs-retryable errors, and
  writes a per-poll **assessment** (loss/eval trend + `stop_recommended`). Graceful stop via
  `touch <run_dir>/STOP`.
- `offline_eval.py` — open-loop replay of a lerobot checkpoint on held-out episodes
  (per-episode + mean MSE/MAE, optional gt-vs-pred plots). The answer to "real-robot data has
  no sim env".
- `eval_watcher.py` — scores each new checkpoint on the held-out episodes with `offline_eval.py`;
  saves scalar MSE/MAE + trajectory-plot artifacts.
- `verify_run.py` — post-run independent verification against the real lerobot checkpoint
  anatomy → `VERIFY.md` (don't trust exit 0).
- `monitor_server.py` — dependency-free FastAPI dashboard: loss curve, eval-MSE curve, the
  watchdog's assessment, and a generic gallery of eval artifacts.

## Status

Ported from an Isaac GR00T orchestrator to target lerobot's `lerobot-train` end to end
(launch/resume, checkpoint anatomy, held-out offline eval). The `references/` docs are the
living record of what broke and the check that now prevents it.
