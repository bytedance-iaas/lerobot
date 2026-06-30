# Prior art this skill is built on

This skill deliberately reuses established patterns rather than inventing its own. Sources
below, grouped by the part of the design they informed.

## Agent Skills format & best practices
- **anthropics/skills** — https://github.com/anthropics/skills — the canonical Agent Skills
  repo: `spec/` (the specification), `template/` (skill template), and example skills
  (document skills are the best `SKILL.md` + `scripts/` + reference bundle examples). This
  skill follows that layout: required `name`/`description` frontmatter, `scripts/` executed
  not loaded, `references/` loaded on demand.
- **Agent Skills best practices** —
  https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices —
  three-tier progressive disclosure (metadata → SKILL.md body <500 lines → bundled
  resources); references one level deep; scripts executed for output, not read into context;
  workflow-with-checklist pattern; plan-validate-execute with verifiable intermediate
  artifacts. The session/stage file-state design is the "validate the plan before executing"
  pattern made durable.

## Closest Claude-skill precedents (ML training as a skill)
These are the nearest existing *Claude skills* to robot_sft — general ML training, not
robot/VLA-specific. robot_sft borrows their best ideas and adds the robot-SFT domain layer
(embodiment tags, modality.json camera mapping, GR00T's auto-resume, open-loop eval).
- **AlexWortega/claude-ml-intern-skill** — https://github.com/AlexWortega/claude-ml-intern-skill
  — an "autonomous ML intern" skill: **smoke-test before training** (instantiate + forward
  pass on tiny input), train with a **NaN guard** + checkpointing, then **self-verify into a
  VERIFY.md** with independent verdicts. Directly inspired robot_sft's `preflight.py`
  (cheap pre-launch smoke test) and `verify_run.py` (don't trust exit 0; independent checks).
- **wanshuiyin/ARIS (Auto-Research-In-Sleep)** —
  https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep — autonomous ML research
  skills with an `/experiment-queue`: **OOM-aware retry with backoff**, an idle-timeout
  watchdog, and a **crash-safe scheduler that resumes from JSON state**. Validates robot_sft's
  watchdog (capped backoff, resume-from-state) and motivated the OOM branch in
  `error_patterns.py` (classify OOM → reduce batch, rather than blindly resuming).
- **hacktivist123/agent-session-resume** — https://github.com/hacktivist123/agent-session-resume
  — a cross-agent **session-resume** skill that writes a handoff checkpoint (done / open /
  next action) before proceeding. Same "file-based, always-resumable" ethos as `session.py`.
- **hoangsonww/Claude-Code-Agent-Monitor** — https://github.com/hoangsonww/Claude-Code-Agent-Monitor
  — a real-time monitoring **dashboard** with a periodic error-scan watchdog. Same shape as
  `monitor_server.py` (it monitors Claude sessions; robot_sft monitors the training run).

## Multi-agent orchestration with file-based state
- **dsifry/metaswarm** — https://github.com/dsifry/metaswarm — agents coordinate via a
  git-native state store as the single source of truth rather than passing messages; quality
  gates via threshold files. Direct precedent for "all cross-agent comms are files; the
  session dir is the source of truth; therefore resumable."
- **am-will/swarms** — https://github.com/am-will/swarms — orchestrator that plans explicit
  dependencies and executes in waves while maintaining/verifying context. Informs the
  orchestrator→sub-agent stage loop.
- **rohitg00/awesome-claude-code-toolkit** —
  https://github.com/rohitg00/awesome-claude-code-toolkit — catalogs multi-agent topologies
  (orchestrator-worker, pipeline, MapReduce). robot_sft is orchestrator-worker + pipeline.
- **VoltAgent/awesome-agent-skills** — https://github.com/VoltAgent/awesome-agent-skills —
  the broad "awesome list" of agent skills across tools, for discovering comparable skills.

## Self-healing training / job watchdog
- **SkyPilot managed jobs** —
  https://docs.skypilot.co/en/stable/examples/managed-jobs.html — the core auto-recovery
  doctrine: "write checkpoints periodically AND always attempt to load checkpoints on
  startup, regardless of first run or restart"; `max_restarts_on_errors` and the idea that
  not all non-zero exits should auto-recover (choose retryable conditions). Mirrored in the
  watchdog's resume + restart-cap logic.
- **MosaicML Watchdog / autoresume** —
  https://docs.mosaicml.com/projects/mcli/en/latest/training/watchdog.html — autoresume from
  the latest checkpoint after a crash; caveat that a naive watchdog "restarts from the
  beginning" — which is *exactly* why robot_sft checks `is_resumable()` before relaunching.
- **PyTorch Lightning EarlyStopping** —
  https://lightning.ai/docs/pytorch/stable/common/early_stopping.html — `check_finite`
  (stop on NaN/Inf) and `divergence_threshold` (stop when loss gets worse than a bound);
  these map to the watchdog's NaN/divergence early-stop.
- **HF Trainer resume** —
  https://huggingface.co/docs/transformers/en/main_classes/trainer — `resume_from_checkpoint`
  semantics and the RNG/optimizer/scheduler state restored; basis for the
  resumable-checkpoint anatomy in `gr00t_resume.md`.
- **SkyPilot issue #2805** — https://github.com/skypilot-org/skypilot/issues/2805 — argues
  against exponential backoff for up-waits; hence robot_sft applies capped backoff only to
  crash-restarts, not to all retries.

## FastAPI file-state status dashboard
- **FastAPI background tasks** — https://fastapi.tiangolo.com/tutorial/background-tasks/ —
  notes that built-in background tasks can't report status; therefore robot_sft keeps
  training OUT of FastAPI and has the dashboard only *read* the state files the watchdog
  writes.
- **Long-running job submit+poll pattern** —
  https://medium.com/@bhagyarana80/serving-long-running-jobs-with-fastapi-using-webhooks-and-task-polling-860bb0d3e0f9
  — one HTML page + a JSON status endpoint that re-reads job state each request; the browser
  polls. `monitor_server.py` is the minimal version: `/` HTML auto-refreshes, `/api/...`
  returns `json.load(state_file)`, no DB, no worker.
