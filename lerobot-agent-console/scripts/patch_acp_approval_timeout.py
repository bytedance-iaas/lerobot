#!/usr/bin/env python3
"""Make hermes' ACP approval timeout configurable, instead of a hardcoded 60s.

When a tool needs approval, hermes BLOCKS its agent thread and waits for the ACP client to
answer. The wait is `make_approval_callback(..., timeout: float = 60.0)`, and the ACP adapter
calls it without a timeout, so 60s is what you get. `approvals.timeout` in config.yaml does NOT
reach this path: `prompt_dangerous_approval` computes it and then invokes the callback without
passing it.

60 seconds is far too short for a human:

  * Step away for a minute and hermes records "BLOCKED: User denied this command. The user has
    NOT consented to this action." The user denied nothing — they never saw a prompt.
  * The console waited 300s for the click, so hermes gave up FIRST. A button pressed at t=90s
    resolved a future nobody was reading, while the tool had already been reported as denied.
  * hermes stays mid-turn for the whole wait, so anything typed in the meantime comes back
    "Queued for the next turn" and the UI looks frozen.

Fix: honour $HERMES_ACP_APPROVAL_TIMEOUT (default 900s). The console's own wait
(CONSOLE_PERM_TIMEOUT, 870s) must stay BELOW it, so the UI is always the side that gives up
first — and only after the person had a realistic chance to answer.

Idempotent; no-op if hermes changes the signature. Baked into the console image (Dockerfile),
runs in the HERMES venv where acp_adapter lives.
"""

from __future__ import annotations

import os
import sys

OLD = "    timeout: float = 60.0,\n"
NEW = (
    "    timeout: float | None = None,\n"
)
# Inserted at the top of the function body, after the docstring's closing quotes.
INJECT_ANCHOR = "    def _callback(\n"
INJECT = (
    "    # PATCHED (patch_acp_approval_timeout.py): 60s is not a human timescale for an\n"
    "    # approval prompt; see the module docstring. Env-overridable, default 900s.\n"
    "    if timeout is None:\n"
    "        try:\n"
    "            timeout = float(os.environ.get('HERMES_ACP_APPROVAL_TIMEOUT', '900'))\n"
    "        except (TypeError, ValueError):\n"
    "            timeout = 900.0\n"
    "\n"
)


def find_permissions_py() -> str | None:
    for root in sys.path:
        cand = os.path.join(root, "acp_adapter", "permissions.py")
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    path = find_permissions_py()
    if not path:
        print("patch_acp_approval_timeout: acp_adapter/permissions.py not found", file=sys.stderr)
        return 1

    src = open(path, encoding="utf-8").read()

    if "HERMES_ACP_APPROVAL_TIMEOUT" in src:
        print("patch_acp_approval_timeout: already applied")
        return 0

    if OLD not in src or INJECT_ANCHOR not in src:
        # Never fail the build on an upstream refactor — but say so loudly, because the console
        # then has to keep its own timeout under hermes' unpatched 60s.
        print("patch_acp_approval_timeout: WARNING anchors not found; hermes keeps its 60s "
              "default (set CONSOLE_PERM_TIMEOUT below 60)", file=sys.stderr)
        return 0

    if "\nimport os\n" not in src:
        src = src.replace("\nimport asyncio\n", "\nimport asyncio\nimport os\n", 1)

    src = src.replace(OLD, NEW, 1)
    src = src.replace(INJECT_ANCHOR, INJECT + INJECT_ANCHOR, 1)

    open(path, "w", encoding="utf-8").write(src)
    print(f"patch_acp_approval_timeout: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
