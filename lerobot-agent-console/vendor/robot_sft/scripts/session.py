#!/usr/bin/env python3
"""File-based session/run state for robot_sft.

The session directory is the single source of truth, so any stage can be resumed after a
crash or context reset. All sub-agents read/write state ONLY through this script — never
hand-edit the JSON, so the schema stays consistent.

Layout (under --root; default $ROBOT_SFT_HOME, else /opt/data/robot_sft when /opt/data
exists — the console pod's big persistent volume — else ./.robot_sft):
    sessions/<session_id>/session.json     master state
    sessions/<session_id>/<stage>.json     per-stage artifacts (written by the agents)
    sessions/<session_id>/runs/<run_id>/run.json
    current                                 file holding the active session_id

Commands:
    create   [--goal G] [--root R]                      -> new session, prints id + dir
    status   [--root R]                                 -> human summary of active session
    get      [--session DIR | --root R]                 -> dump session.json
    set-stage <stage> <status> [--summary S] [--session DIR]
    set-config <key> <json_value> [--session DIR]
    add-run  [--session DIR]                            -> new run-NNN, prints id + dir
    update-run <run_id> --set k=v ... [--session DIR]   (values JSON-parsed, else string)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

STAGES = ["overview", "dataset_explore", "preprocess", "training_plan", "train"]


def _default_root() -> str:
    """State root. Big artifacts (sessions, checkpoints, logs) must NOT land in the code
    checkout: in the console pod /lerobot is the repo but /opt/data is the roomy PVC."""
    env = os.environ.get("ROBOT_SFT_HOME")
    if env:
        return env
    if os.path.isdir("/opt/data"):
        return "/opt/data/robot_sft"
    return ".robot_sft"


DEFAULT_ROOT = _default_root()


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic, so a crash mid-write never corrupts state


def _sessions_dir(root: str) -> str:
    return os.path.join(root, "sessions")


def _resolve_session(args) -> str:
    """Return the session directory from --session, else the 'current' pointer."""
    if getattr(args, "session", None):
        return args.session
    root = getattr(args, "root", None) or DEFAULT_ROOT
    cur = os.path.join(root, "current")
    if os.path.exists(cur):
        with open(cur) as f:
            sid = f.read().strip()
        d = os.path.join(_sessions_dir(root), sid)
        if os.path.isdir(d):
            return d
    print("ERROR: no active session (run `session.py create` first)", file=sys.stderr)
    sys.exit(2)


def _session_json(session_dir: str) -> str:
    return os.path.join(session_dir, "session.json")


def cmd_create(args) -> None:
    root = args.root or DEFAULT_ROOT
    sid = "sess-" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(_sessions_dir(root), sid)
    data = {
        "session_id": sid,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "in_progress",
        "current_stage": STAGES[0],
        "goal": args.goal or "",
        "config": {},
        "stages": {s: {"status": "pending", "summary": "", "updated_at": None} for s in STAGES},
        "runs": [],
    }
    _write(_session_json(session_dir), data)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "current"), "w") as f:
        f.write(sid)
    print(json.dumps({"session_id": sid, "session_dir": session_dir}))


def cmd_get(args) -> None:
    print(json.dumps(_read(_session_json(_resolve_session(args))), indent=2))


def cmd_status(args) -> None:
    session_dir = _resolve_session(args)
    d = _read(_session_json(session_dir))
    print(f"session   : {d['session_id']}  [{d['status']}]")
    print(f"dir       : {session_dir}")
    print(f"goal      : {d.get('goal','')}")
    print(f"stage now : {d['current_stage']}")
    for s in STAGES:
        st = d["stages"][s]
        mark = {"done": "[x]", "blocked": "[!]", "pending": "[ ]", "in_progress": "[~]"}.get(
            st["status"], "[ ]"
        )
        print(f"  {mark} {s:<15} {st['status']:<12} {st.get('summary','')}")
    if d["runs"]:
        print("runs:")
        for r in d["runs"]:
            print(f"  - {r['run_id']:<10} {r.get('status','?'):<12} "
                  f"step={r.get('last_step','?')} restarts={r.get('restarts',0)}")


def _advance_current_stage(d: dict) -> None:
    for s in STAGES:
        if d["stages"][s]["status"] not in ("done",):
            d["current_stage"] = s
            return
    d["current_stage"] = "done"
    d["status"] = "completed"


def cmd_set_stage(args) -> None:
    session_dir = _resolve_session(args)
    p = _session_json(session_dir)
    d = _read(p)
    if args.stage not in d["stages"]:
        print(f"ERROR: unknown stage {args.stage}", file=sys.stderr)
        sys.exit(2)
    d["stages"][args.stage]["status"] = args.status
    if args.summary is not None:
        d["stages"][args.stage]["summary"] = args.summary
    d["stages"][args.stage]["updated_at"] = _now()
    if args.status == "blocked":
        d["status"] = "blocked"
    _advance_current_stage(d)
    d["updated_at"] = _now()
    _write(p, d)
    print(json.dumps({"stage": args.stage, "status": args.status,
                      "current_stage": d["current_stage"]}))


def cmd_set_config(args) -> None:
    session_dir = _resolve_session(args)
    p = _session_json(session_dir)
    d = _read(p)
    try:
        val = json.loads(args.value)
    except json.JSONDecodeError:
        val = args.value
    d.setdefault("config", {})[args.key] = val
    d["updated_at"] = _now()
    _write(p, d)
    print(json.dumps({"config": {args.key: val}}))


def cmd_add_run(args) -> None:
    session_dir = _resolve_session(args)
    p = _session_json(session_dir)
    d = _read(p)
    run_id = f"run-{len(d['runs']) + 1:03d}"
    run_dir = os.path.join(session_dir, "runs", run_id)
    run = {
        "run_id": run_id,
        "created_at": _now(),
        "status": "created",
        "restarts": 0,
        "last_step": None,
        "last_loss": None,
        "checkpoint": None,
        "log": os.path.join(run_dir, "train.log"),
    }
    _write(os.path.join(run_dir, "run.json"), run)
    d["runs"].append({"run_id": run_id, "status": "created", "restarts": 0, "last_step": None})
    d["updated_at"] = _now()
    _write(p, d)
    print(json.dumps({"run_id": run_id, "run_dir": run_dir}))


def cmd_update_run(args) -> None:
    session_dir = _resolve_session(args)
    run_dir = os.path.join(session_dir, "runs", args.run_id)
    rp = os.path.join(run_dir, "run.json")
    run = _read(rp)
    for kv in args.set or []:
        if "=" not in kv:
            print(f"ERROR: --set expects k=v, got {kv}", file=sys.stderr)
            sys.exit(2)
        k, v = kv.split("=", 1)
        try:
            run[k] = json.loads(v)
        except json.JSONDecodeError:
            run[k] = v
    run["updated_at"] = _now()
    _write(rp, run)
    # mirror a few fields into the session summary list
    p = _session_json(session_dir)
    d = _read(p)
    for r in d["runs"]:
        if r["run_id"] == args.run_id:
            for k in ("status", "restarts", "last_step", "last_loss"):
                if k in run:
                    r[k] = run[k]
    d["updated_at"] = _now()
    _write(p, d)
    print(json.dumps(run))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create"); p.add_argument("--goal"); p.add_argument("--root"); p.set_defaults(func=cmd_create)
    p = sub.add_parser("status"); p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("get"); p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_get)
    p = sub.add_parser("set-stage"); p.add_argument("stage"); p.add_argument("status")
    p.add_argument("--summary"); p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_set_stage)
    p = sub.add_parser("set-config"); p.add_argument("key"); p.add_argument("value")
    p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_set_config)
    p = sub.add_parser("add-run"); p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_add_run)
    p = sub.add_parser("update-run"); p.add_argument("run_id"); p.add_argument("--set", action="append")
    p.add_argument("--root"); p.add_argument("--session"); p.set_defaults(func=cmd_update_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
