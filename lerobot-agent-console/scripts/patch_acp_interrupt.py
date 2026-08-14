#!/usr/bin/env python3
"""Fix a hermes ACP bug: an INTERRUPTED turn crashes the whole prompt handler with
`'NoneType' object has no attribute 'startswith'`.

acp_adapter/server.py (prompt handler) does:

    final_response = result.get("final_response", "")           # <-- buggy default
    ...
    suppress_interrupt_response = interrupted and final_response.startswith(PREFIX)

When a turn is interrupted/cancelled, `result` contains the key `"final_response"` set
EXPLICITLY to None. `dict.get(key, "")` only returns the "" default when the key is
ABSENT — a present-but-None value returns None. So `final_response` is None and
`None.startswith(...)` raises, which the ACP layer wraps into an opaque
`RequestError.internal_error` (`{'code': -32603, 'Internal error', 'details':
"'NoneType' object has no attribute 'startswith'"}`) — the web chat "gets stuck".

Fix: coerce missing OR None to "" via `or ""`. `"".startswith(...)` is False and the
downstream `if final_response and ...` stays correctly falsy — the intended behavior for
an interrupted turn that produced no assistant prose.

Idempotent; no-op if hermes changes the line. Baked into the console image (Dockerfile),
runs in the HERMES venv where acp_adapter lives.
"""
from __future__ import annotations

import sys


def find_server_py() -> str | None:
    import importlib.util
    spec = importlib.util.find_spec("acp_adapter.server")
    return spec.origin if spec and spec.origin else None


def main() -> int:
    path = find_server_py()
    if not path:
        print("patch_acp_interrupt: acp_adapter.server not found — skipping")
        return 0
    src = open(path).read()

    buggy = 'final_response = result.get("final_response", "")'
    fixed = ('final_response = result.get("final_response") or ""  '
             '# console patch: interrupted turns set final_response=None -> None.startswith crash')

    if fixed in src:
        print("patch_acp_interrupt: already patched — no-op")
        return 0
    if buggy not in src:
        print("patch_acp_interrupt: expected line not found — hermes changed; skipping")
        return 0

    open(path, "w").write(src.replace(buggy, fixed, 1))
    print(f"patch_acp_interrupt: patched {path} — interrupted turns no longer crash on None.startswith")
    return 0


if __name__ == "__main__":
    sys.exit(main())
