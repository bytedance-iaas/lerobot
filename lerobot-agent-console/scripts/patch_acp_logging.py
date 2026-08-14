#!/usr/bin/env python3
"""Patch hermes' ACP connection layer to LOG the real traceback before it masks every
handler exception into a generic `RequestError.internal_error(...)` + `raise err from None`.

Without this, any error inside an ACP request handler (e.g. the `process wait` /
detached-session bug) reaches the client as an opaque `{'code': -32603, 'Internal error',
'details': "<str(exc)>"}` with the file:line traceback DISCARDED — which is why the console
web chat "gets stuck" undebuggably. Idempotent; no-op if hermes changes.

Baked into the console image (see Dockerfile) so it survives pod restarts.
"""
from __future__ import annotations

import re
import sys


def find_connection_py() -> str | None:
    import importlib.util
    spec = importlib.util.find_spec("acp.connection")
    return spec.origin if spec and spec.origin else None


def main() -> int:
    path = find_connection_py()
    if not path:
        print("patch_acp_logging: acp.connection not found — skipping")
        return 0
    src = open(path).read()

    if "ACP handler exception (unmasked by console patch)" in src:
        print("patch_acp_logging: already patched — no-op")
        return 0

    # Insert a logging.exception(...) at the top of every `except Exception as exc:` block that
    # precedes a `raise err from None` inside the request runner. Match the block header exactly
    # (indentation preserved) and inject a logger line right after it.
    marker = "            except Exception as exc:\n"
    inject = (
        "            except Exception as exc:\n"
        "                import logging as _acp_logging  # ACP handler exception (unmasked by console patch)\n"
        "                _acp_logging.getLogger(\"acp.connection\").exception(\n"
        "                    \"ACP request handler failed (unmasked): %s\", exc)\n"
    )
    if marker not in src:
        print("patch_acp_logging: expected `except Exception as exc:` block not found — hermes changed; skipping")
        return 0

    patched = src.replace(marker, inject, 1)
    open(path, "w").write(patched)
    print(f"patch_acp_logging: patched {path} — ACP now logs the real traceback before masking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
