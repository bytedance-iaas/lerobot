# robot_sft — resumable multi-agent robot VLA fine-tuning skill

A Claude Code [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that turns one
supervised fine-tuning (SFT) **session** for a robot vision-language-action (VLA) model
(e.g. NVIDIA Isaac **GR00T N1.7**, or LeRobot-style policies) into a sequence of small,
independently verifiable, file-checkpointed **stages**, each run by a focused sub-agent — so a
crash or context reset never loses progress (re-read the session state and continue).

It exists because robot SFT fails in boring, expensive ways: a gated backbone you can't
download, a 64 MB `/dev/shm` that kills the dataloader 90 minutes in, a checkpoint truncated by
a bad kill, a wrong camera key in `modality.json`, 10000 steps when the data justifies 3000, or
an eval that never tells you whether the policy generalizes. Each of those has a concrete check
that prevents it — see [`references/lessons_learned.md`](references/lessons_learned.md).

## Pipeline (one sub-agent per stage; all state in files)

| Stage | What it does |
|-------|--------------|
| **a. overview** | review the user's setup against the known failure modes; gate the session if there's no training entrypoint |
| **b. dataset explore** | inspect LeRobot format/version, episodes, camera keys, state/action dims — catch mismatches *before* a run |
| **c. preprocess + split** | convert (v3.0→v2.1), author `modality.json`, and **hold out a train/eval split** (GR00T has no in-loop eval) |
| **d. plan + preflight** | compute steps/batch/`save_steps` from data + hardware, then a ~2-step smoke test catches config bugs in minutes |
| **e. train** | launch under a **self-healing watchdog** + a **periodic open-loop eval** + a **TensorBoard-like dashboard** |

## Key components (`scripts/`)

- `session.py` — file-based session/run state (the single source of truth; everything resumable).
- `check_hardware.py` / `plan_training.py` / `preflight.py` — hardware probe, data-driven plan, cheap smoke test.
- `watchdog.py` — owns training; auto-resumes from the last *resumable* checkpoint, early-stops on
  NaN/divergence/stall, classifies fatal-vs-retryable errors, and writes a per-poll **assessment**
  (loss/eval trend + `stop_recommended`). Graceful stop via `touch <run_dir>/STOP`.
- `split_train_eval.py` — split a LeRobot v2.1 dataset into disjoint train/eval dirs (re-indexes
  episodes + rebuilds `meta/`).
- `eval_watcher.py` — scores each new checkpoint on the held-out set with open-loop eval, on a
  separate GPU; saves scalar MSE/MAE + trajectory-plot artifacts.
- `monitor_server.py` — dependency-free FastAPI dashboard: loss curve, eval-MSE curve, the
  watchdog's assessment, a per-dataset metrics table, and a generic gallery of eval artifacts.

## Status

Built and exercised on real multi-dataset GR00T N1.7 / SO101 runs. Continuously optimized — the
`references/` docs are the living record of what broke and the check that now prevents it.
