#!/usr/bin/env python3
"""Self-healing training watchdog for robot_sft (lerobot edition).

Owns the `lerobot-train` subprocess and keeps it alive across crashes WITHOUT losing
progress. It only starts/stops the process and reads files, so it is itself crash-safe:
re-running it re-reads run.json. See references/agents.md (watchdog contract) and
references/lerobot_resume.md (checkpoint anatomy + why resume = --resume=true).

It reads the launch/resume commands + output_dir from <session>/training_plan.json,
writes live status to <session>/runs/<run>/run.json (which monitor_server.py serves), and:
  - parses the train log for step/loss at least every --poll seconds (default 60, <=300)
  - early-stops on NaN/Inf loss, divergence, or a stall (no step progress)
  - on any stop, checks the latest checkpoint is RESUMABLE, then relaunches with
    `--resume=true --config_path=.../checkpoints/last/pretrained_model/train_config.json`;
    never restarts from scratch silently
  - applies capped exponential backoff to crash/early-stop restarts and caps total restarts

Usage:
    python watchdog.py --session <session_dir> --run run-001 \
        [--poll 60] [--max-restarts 5] [--stall-timeout 1800] \
        [--divergence 1e6] [--base-backoff 30] [--backoff-cap 600] \
        [--target-step N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_patterns  # noqa: E402

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")          # tqdm "4000/10000 [" (exact step)
LOSS_RE = re.compile(r"\bloss:([0-9.eE+-]+|nan|inf|-inf)")  # lerobot tracker "loss:0.123"


def now() -> float:
    return time.time()


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def is_resumable(ckpt_dir: str) -> bool:
    """lerobot checkpoint = <step>/{pretrained_model,training_state}. Resume needs the FULL
    training_state (optimizer + rng + step; scheduler optional) AND loadable weights.
    training_state is written after pretrained_model, so training_step.json's presence also
    means the save finished (not truncated). See lerobot_resume.md."""
    pm = os.path.join(ckpt_dir, "pretrained_model")
    ts = os.path.join(ckpt_dir, "training_state")
    try:
        pm_names = set(os.listdir(pm))
        ts_names = set(os.listdir(ts))
    except OSError:
        return False
    has_weights = "model.safetensors" in pm_names and "config.json" in pm_names
    has_train_cfg = "train_config.json" in pm_names
    has_state = ("optimizer_state.safetensors" in ts_names
                 and "training_step.json" in ts_names
                 and any(n.startswith("rng_state") for n in ts_names))
    return has_weights and has_train_cfg and has_state


def latest_resumable_checkpoint(output_dir: str):
    """Highest-step resumable checkpoint dir under <output_dir>/checkpoints/, or None."""
    best, best_step = None, -1
    ckpts = os.path.join(output_dir, "checkpoints")
    try:
        entries = os.listdir(ckpts)
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue  # skip the 'last' symlink
        d = os.path.join(ckpts, name)
        if os.path.isdir(d) and is_resumable(d):
            step = int(name)
            if step > best_step:
                best, best_step = d, step
    return best


def read_eval_series(session: str):
    """Load eval_watcher's per-checkpoint results: [(step, mean_mse), ...] sorted by step."""
    path = os.path.join(session, "eval", "eval_results.jsonl")
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("mean_mse") is not None:
                        out.append((int(d["step"]), float(d["mean_mse"])))
                except Exception:  # noqa: BLE001
                    pass
    except OSError:
        return []
    return sorted(out)


# convergence thresholds (a conclusion, not an auto-stop — we surface it, the human decides)
LOSS_IMPROVE_FRAC = 0.03   # a >3% drop in loss counts as "still improving"


def assess(step, max_step, loss, best_loss, best_loss_step, evals):
    """Produce a human-readable training conclusion for the dashboard each poll.

    Train-loss plateau is NOT 'done' — behaviour-cloning loss bottoms out early. The real
    selection signal is the held-out EVAL curve, so a stop is only *recommended* once loss is
    flat AND there are >=2 eval points whose MSE has stopped improving (lessons_learned #14)."""
    plateau_steps = (step - best_loss_step) if (step and best_loss_step is not None) else 0
    flat_thresh = max(1500, int(0.1 * max_step)) if max_step else 1500
    loss_flat = plateau_steps >= flat_thresh
    loss_trend = "plateaued" if loss_flat else "improving"

    eval_txt, best_eval_step, stop = "", None, False
    if evals:
        best_eval_step, best_eval = min(evals, key=lambda x: x[1])
        last_step, last_mse = evals[-1]
        if len(evals) < 2:
            eval_txt = f"only 1 eval point (ckpt-{last_step} mse={last_mse:.4f}); need >=2 to judge"
        else:
            improving = last_mse <= best_eval * 1.001
            eval_txt = (f"eval MSE {'still improving' if improving else 'plateaued'}; "
                        f"best ckpt-{best_eval_step} mse={best_eval:.4f}")
            stop = loss_flat and not improving
    else:
        eval_txt = "no eval points yet"

    if stop:
        verdict = (f"CONVERGED — safe to stop. loss flat {plateau_steps} steps and {eval_txt}. "
                   f"Pick ckpt-{best_eval_step} (lowest eval MSE).")
    elif loss_flat and (not evals or len(evals) < 2):
        verdict = (f"loss plateaued ({plateau_steps} steps since last >3% drop), but {eval_txt}. "
                   f"Let 1-2 more checkpoints score before deciding to stop.")
    elif loss_flat:
        verdict = f"loss plateaued ({plateau_steps} steps) but {eval_txt} — keep going."
    else:
        verdict = f"training healthy — loss {loss_trend}; {eval_txt}."

    return {
        "verdict": verdict,
        "stop_recommended": stop,
        "loss_trend": loss_trend,
        "loss_plateau_steps": plateau_steps,
        "best_loss": best_loss if best_loss != float("inf") else None,
        "best_eval_step": best_eval_step,
        "n_eval_points": len(evals),
        "ts": time.ctime(),
    }


def tail_metrics(log_path: str, max_bytes: int = 200_000):
    """Return (last_step, max_step, last_loss, loss_is_bad) from the log tail.
    Step comes from tqdm's exact "N/M [" (lerobot's tracker line rounds step to K)."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None, None, None, False
    last_step = max_step = last_loss = None
    bad = False
    for m in STEP_RE.finditer(text):
        last_step, max_step = int(m.group(1)), int(m.group(2))
    losses = LOSS_RE.findall(text)
    if losses:
        raw = losses[-1].lower()
        if raw in ("nan", "inf", "-inf"):
            bad = True
            last_loss = raw
        else:
            try:
                last_loss = float(raw)
                if math.isnan(last_loss) or math.isinf(last_loss):
                    bad = True
            except ValueError:
                pass
    return last_step, max_step, last_loss, bad


def launch(cmd: str, log_path: str) -> subprocess.Popen:
    """Run a launch/resume command under bash, teeing stdout+stderr to the log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "ab", buffering=0)
    logf.write(f"\n===== watchdog launch @ {time.ctime()} =====\n{cmd}\n".encode())
    return subprocess.Popen(["bash", "-lc", cmd], stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(proc: subprocess.Popen, grace: int = 20) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(grace):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--stall-timeout", type=int, default=1800)
    ap.add_argument("--divergence", type=float, default=1e6)
    ap.add_argument("--base-backoff", type=int, default=30)
    ap.add_argument("--backoff-cap", type=int, default=600)
    ap.add_argument("--target-step", type=int, default=None)
    args = ap.parse_args()

    poll = min(args.poll, 300)  # cadence must be <= 5 minutes
    session = args.session
    plan = read_json(os.path.join(session, "training_plan.json"))
    cmd = plan["launch_command"]
    output_dir = plan["output_dir"]
    resume_cmd = plan.get("resume_command") or (
        f"cd {plan.get('repo', '/lerobot')} && uv run lerobot-train --resume=true "
        f"--config_path={output_dir}/checkpoints/last/pretrained_model/train_config.json")
    run_dir = os.path.join(session, "runs", args.run)
    run_json = os.path.join(run_dir, "run.json")
    log_path = os.path.join(run_dir, "train.log")
    run = read_json(run_json) if os.path.exists(run_json) else {"run_id": args.run}

    def update(**kw):
        run.update(kw)
        run["updated_at"] = time.ctime()
        write_json(run_json, run)

    restarts = run.get("restarts", 0)
    # If a resumable checkpoint already exists (watchdog itself restarted), resume — the
    # fresh-launch command would die on "output directory already exists".
    first_cmd = resume_cmd if latest_resumable_checkpoint(output_dir) else cmd
    proc = launch(first_cmd, log_path)
    update(status="running", pid=proc.pid, restarts=restarts, log=log_path,
           output_dir=output_dir, started_at=time.ctime())

    last_progress_step, last_progress_time = None, now()
    best_loss, best_loss_step = float("inf"), None
    stop_file = os.path.join(run_dir, "STOP")
    stop_requested = False

    while True:
        time.sleep(poll)
        step, max_step, loss, loss_bad = tail_metrics(log_path)
        ckpt = latest_resumable_checkpoint(output_dir)

        # track the best (lowest) loss so the assessment can measure plateau length
        if isinstance(loss, float) and loss < best_loss * (1 - LOSS_IMPROVE_FRAC):
            best_loss, best_loss_step = loss, step
        assessment = assess(step, max_step, loss, best_loss, best_loss_step,
                            read_eval_series(session))
        update(status="running", last_step=step, max_step=max_step, last_loss=loss,
               checkpoint=os.path.basename(ckpt) if ckpt else None, restarts=restarts,
               assessment=assessment)

        # graceful manual stop: `touch <run_dir>/STOP`. We stop and KEEP the latest complete
        # checkpoint (training_state written last => not truncated); we do NOT restart.
        # Progress since the last save is lost but no checkpoint is corrupted.
        if os.path.exists(stop_file) and proc.poll() is None:
            stop_requested = True
            update(status="stopping", reason="manual STOP file — stopping at last complete checkpoint")
            stop(proc)

        # stall detection
        if step is not None and step != last_progress_step:
            last_progress_step, last_progress_time = step, now()
        stalled = (now() - last_progress_time) > args.stall_timeout and proc.poll() is None

        diverged = isinstance(loss, float) and loss > args.divergence

        trouble = None
        if loss_bad:
            trouble = f"loss is {loss} (NaN/Inf)"
        elif diverged:
            trouble = f"loss diverged ({loss} > {args.divergence})"
        elif stalled:
            trouble = f"stalled (no step progress for {args.stall_timeout}s)"

        if trouble and proc.poll() is None:
            update(status="early_stopping", reason=trouble)
            stop(proc)

        exited = proc.poll() is not None
        if not exited and not trouble:
            continue  # healthy, keep monitoring

        # ---- run ended (clean exit, crash, or we early-stopped it) ----
        rc = proc.poll()

        # manual graceful stop takes precedence over restart logic — do NOT relaunch.
        if stop_requested:
            final_ckpt = latest_resumable_checkpoint(output_dir)
            update(status="stopped", exit_code=rc, last_step=step,
                   checkpoint=os.path.basename(final_ckpt) if final_ckpt else None,
                   note="stopped by user; latest complete checkpoint kept and is resumable")
            print(f"[watchdog] stopped by user at step {step}; kept {final_ckpt}")
            return

        reached_target = (args.target_step is not None and step is not None
                          and step >= args.target_step)
        clean_done = (rc == 0 and not trouble and
                      (max_step is None or step is None or step >= max_step or reached_target))
        if clean_done:
            update(status="done", exit_code=rc, last_step=step)
            print(f"[watchdog] run done at step {step} (rc={rc})")
            return

        # classify the failure: some errors are config bugs that recur identically on every
        # restart — retrying them just burns time and hides the real fix (SkyPilot principle).
        cls = error_patterns.classify_file(log_path)
        if cls["category"] == "fatal":
            update(status="failed", reason=cls["reason"], fix=cls["fix"],
                   note="non-retryable config error — not restarting")
            print(f"[watchdog] FATAL (no retry): {cls['reason']} — {cls['fix']}")
            return
        if cls["category"] == "oom":
            update(oom_hint=cls["fix"])  # surface; resume still helps if a checkpoint exists

        if restarts >= args.max_restarts:
            update(status="failed", reason=trouble or f"exit code {rc}",
                   note=f"hit max_restarts={args.max_restarts}")
            print(f"[watchdog] FAILED after {restarts} restarts: {trouble or rc}")
            return

        ckpt = latest_resumable_checkpoint(output_dir)
        if ckpt:
            next_cmd = resume_cmd  # --resume=true from checkpoints/last (lerobot_resume.md)
            resume_note = f"resuming from {os.path.basename(ckpt)} via --resume=true"
        else:
            # No checkpoint yet. lerobot-train refuses an existing output_dir without
            # --resume, so a from-scratch relaunch must clear it — safe ONLY when there is
            # nothing resumable inside to lose.
            ckpts_dir = os.path.join(output_dir, "checkpoints")
            has_any = os.path.isdir(ckpts_dir) and any(
                n.isdigit() for n in os.listdir(ckpts_dir))
            if has_any:
                update(status="failed", reason="checkpoints exist but none is resumable "
                       "(truncated save?)", fix="inspect the newest step dir; see lerobot_resume.md")
                print("[watchdog] FAILED: non-resumable checkpoints present — not wiping them")
                return
            shutil.rmtree(output_dir, ignore_errors=True)
            next_cmd = cmd
            resume_note = "no checkpoint yet — cleared output_dir, restarting from scratch"
        backoff = min(args.backoff_cap, args.base_backoff * (2 ** restarts))
        restarts += 1
        update(status="restarting", reason=trouble or f"exit code {rc}",
               resume=resume_note, backoff_s=backoff, restarts=restarts)
        print(f"[watchdog] restart {restarts}/{args.max_restarts} in {backoff}s — {resume_note}")
        time.sleep(backoff)
        proc = launch(next_cmd, log_path)
        update(status="running", pid=proc.pid, restarts=restarts)
        last_progress_step, last_progress_time = step, now()


if __name__ == "__main__":
    main()
