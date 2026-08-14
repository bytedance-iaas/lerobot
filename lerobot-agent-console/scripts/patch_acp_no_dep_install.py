#!/usr/bin/env python3
"""Stop hermes from installing system dependencies at runtime (node, Chromium, …).

The image deliberately ships no node and no Chromium: the browser toolset never worked here
(no agent-browser CLI, and the bundled Chrome was missing 10 shared objects), so 764 MB of it
was dead weight. Removing it from the image is not enough on its own — hermes puts itself back
together at runtime, and it does so from a place that is easy to miss:

  tools/browser_tool.py resolves the agent-browser binary, and when it cannot find one it calls
  ensure_dependency("browser"), which shells out to install.sh. That resolver is on the path of
  check_browser_requirements(), i.e. the AVAILABILITY check. So merely asking "is the browser
  toolset usable?" installs it. Listing browser/browser-cdp in agent.disabled_toolsets does not
  help: availability is computed for every toolset, and the disabled list filters the result
  afterwards — the download has already happened. Measured on a fresh PVC: /opt/data went from
  220 KB to 897 MB during startup, with node back on disk.

HERMES_DISABLE_LAZY_INSTALLS does not cover this either; it is read only by tools/lazy_deps.py,
for lazy installs of Python backend packages. dep_ensure.py never looks at it.

Fix: when $HERMES_NO_DEP_INSTALL=1, ensure_dependency reports a missing dependency as
unavailable instead of installing it. Deps that are genuinely present still return True, so
nothing that works today starts failing.

Idempotent; no-op if hermes changes the function. Baked into the console image (Dockerfile),
runs in the HERMES venv where hermes_cli lives.
"""

from __future__ import annotations

import os
import sys

ANCHOR = """    if check():
        return True
"""
INJECT = """    if check():
        return True

    # PATCHED (patch_acp_no_dep_install.py): the console image ships no node/Chromium on
    # purpose, and this function is reachable from a mere availability check — see the module
    # docstring. Report "unavailable" rather than pulling ~764 MB onto a fresh PVC at startup.
    if os.environ.get("HERMES_NO_DEP_INSTALL") == "1":
        return False
"""


def find_dep_ensure() -> str | None:
    for root in sys.path:
        cand = os.path.join(root, "hermes_cli", "dep_ensure.py")
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    path = find_dep_ensure()
    if not path:
        print("patch_acp_no_dep_install: hermes_cli/dep_ensure.py not found", file=sys.stderr)
        return 1

    src = open(path, encoding="utf-8").read()

    if "HERMES_NO_DEP_INSTALL" in src:
        print("patch_acp_no_dep_install: already applied")
        return 0

    if src.count(ANCHOR) != 1:
        # Loud, but not fatal: an unpatched hermes still runs, it just re-downloads node and
        # Chromium onto every fresh PVC.
        print(f"patch_acp_no_dep_install: WARNING anchor found {src.count(ANCHOR)} times, "
              "expected 1; runtime dep installs stay ENABLED", file=sys.stderr)
        return 0

    if "\nimport os\n" not in src:
        src = src.replace("\nimport platform\n", "\nimport os\nimport platform\n", 1)

    open(path, "w", encoding="utf-8").write(src.replace(ANCHOR, INJECT, 1))
    print(f"patch_acp_no_dep_install: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
