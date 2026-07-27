#!/usr/bin/env python3
"""Patch hermes' ACP SessionManager so an acp process dying MID-TURN no longer loses the
whole in-flight conversation.

Stock behavior: messages accumulate in-memory in ``state.history`` and are persisted to
state.db ONLY at turn boundaries (``save_session`` — "Called by the server after prompt
completion"). So when the acp process exits mid-turn (crash, OOM, or the console's
restart/Stop path), everything since the last boundary — assistant replies, tool calls —
is gone; the session is left with just the user's opening prompt. That is exactly the
"messages all lost after acp exit" failure.

The patch appends two guards to ``SessionManager.__init__``:
  1. A daemon autosave thread that persists every in-memory session every 10s
     (covers hard crashes / SIGKILL; at most ~10s of tail loss).
  2. A best-effort SIGTERM handler that persists all sessions before re-raising the
     signal (covers the console's graceful-restart path, which SIGTERMs then waits 5s).

``_persist``/``replace_messages`` writes the full history in one atomic transaction
(a mid-rewrite failure rolls back), so concurrent/periodic writes are safe.

Idempotent; no-op if hermes changes the anchor line. Baked into the console image.
"""
from __future__ import annotations

import sys

ANCHOR = "        self._db_instance = db  # None → lazy-init on first use\n"

INJECT = ANCHOR + '''
        # ---- console patch: mid-turn persistence (acp-autosave) ----
        # Stock hermes persists only at turn boundaries, so a mid-turn process death
        # loses the whole in-flight conversation. Autosave every 10s + flush on SIGTERM.
        import threading as _th

        def _autosave_all() -> None:
            for _sid in list(self._sessions.keys()):
                try:
                    self.save_session(_sid)
                except Exception:
                    pass

        def _autosave_loop() -> None:
            import time as _t
            while True:
                _t.sleep(10)
                try:
                    _autosave_all()
                except Exception:
                    pass

        _th.Thread(target=_autosave_loop, daemon=True, name="acp-autosave").start()
        try:  # best-effort SIGTERM flush (only from the main thread)
            import signal as _sig
            _prev = _sig.getsignal(_sig.SIGTERM)

            def _on_term(_signum, _frame):
                try:
                    _autosave_all()
                finally:
                    _sig.signal(_sig.SIGTERM, _prev if callable(_prev) else _sig.SIG_DFL)
                    _sig.raise_signal(_sig.SIGTERM)

            _sig.signal(_sig.SIGTERM, _on_term)
        except Exception:
            pass
        # ---- end console patch ----
'''


def main() -> int:
    import importlib.util
    spec = importlib.util.find_spec("acp_adapter.session")
    if not (spec and spec.origin):
        print("patch_acp_autosave: acp_adapter.session not found — skipping")
        return 0
    path = spec.origin
    src = open(path).read()

    if "acp-autosave" in src:
        print("patch_acp_autosave: already patched — no-op")
        return 0
    if ANCHOR not in src:
        print("patch_acp_autosave: anchor line not found — hermes changed; skipping")
        return 0

    open(path, "w").write(src.replace(ANCHOR, INJECT, 1))
    # sanity: still imports
    import importlib
    importlib.invalidate_caches()
    import acp_adapter.session  # noqa: F401
    print(f"patch_acp_autosave: patched {path} — sessions autosave every 10s + on SIGTERM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
