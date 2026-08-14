"""JSON bridge to hermes' own session store — runs in HERMES' interpreter, not ours.

The console lives in /lerobot/.venv and hermes in /opt/hermes/.venv, deliberately: the
two share 49 packages and disagree on 11 of them (rich 15 vs 14.3, websockets 16 vs 15,
packaging 25 vs 26, …), so importing `hermes_state` in-process would shadow one venv's
deps with the other's. Hence a subprocess: `/opt/hermes/.venv/bin/python session_api.py`.

Why not parse `hermes sessions list`? It renders a fixed-width table — titles truncated
to 32 chars, times as "3d ago", CJK breaking the column alignment. `SessionDB` is what
that table is rendered FROM, so we call it directly and emit JSON.

Read-only. Writes (delete/rename) stay on the `hermes sessions …` CLI, which is
schema-aware (it also cleans the FTS index and related tables).
"""

import argparse
import json
import sys

from hermes_state import SessionDB  # hermes venv only


def list_sessions(limit: int, include_empty: bool) -> list[dict]:
    # exclude_sources=["tool"] mirrors the CLI's default: hide third-party tool sessions.
    rows = SessionDB().list_sessions_rich(source=None, exclude_sources=["tool"], limit=limit)
    out = []
    for r in rows:
        # A session with no messages is a ghost — the console used to create one on every
        # startup. They carry no title/preview and can't be usefully loaded, so drop them
        # unless explicitly asked for (--include-empty, for cleanup tooling).
        if not include_empty and not r.get("message_count"):
            continue
        out.append(
            {
                "id": r.get("id"),
                "title": r.get("title") or "",
                "preview": r.get("preview") or "",
                "lastActive": r.get("last_active") or r.get("started_at") or 0,
                "messageCount": r.get("message_count") or 0,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="list sessions as JSON")
    p_list.add_argument("--limit", type=int, default=200)
    p_list.add_argument("--include-empty", action="store_true")
    args = ap.parse_args()

    if args.cmd == "list":
        json.dump(list_sessions(args.limit, args.include_empty), sys.stdout, ensure_ascii=False)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
