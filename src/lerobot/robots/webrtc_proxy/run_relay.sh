#!/usr/bin/env bash
#
# RUN-time launcher for the WebRTC signaling relay.
#
# This NEVER installs anything: on FaaS-style hosts the rootfs is read-only at run time,
# so the environment must already exist (built by build_relay.sh during the build phase).
# It just picks an interpreter that already has aiohttp and execs signaling_server.py
# directly (the relay is self-contained: stdlib + aiohttp, no relative imports, so the
# lerobot package tree is never imported — torch/aiortc/etc. are NOT needed).
#
# Usage (args forwarded verbatim to signaling_server.py):
#   ./run_relay.sh --port 8765
#   ./run_relay.sh --port 8765 --stun-url stun:stun.qq.com:3478 --auth-token "$TOKEN"
#
# Env overrides:
#   RELAY_VENV     venv built by build_relay.sh (default: <this dir>/.venv-relay)
#   RELAY_PYTHON   explicit interpreter to use (skips venv lookup)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RELAY_VENV:-$HERE/.venv-relay}"
SERVER="$HERE/signaling_server.py"

pick_python() {
  # 1) explicit override
  if [ -n "${RELAY_PYTHON:-}" ]; then echo "$RELAY_PYTHON"; return 0; fi
  # 2) the venv built at build time
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; return 0; fi
  # 3) any system interpreter that already has aiohttp (e.g. installed --system at build)
  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import aiohttp' >/dev/null 2>&1; then
      echo "$py"; return 0
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "[run_relay] no interpreter with aiohttp found (looked for $VENV/bin/python and system python)." >&2
  echo "[run_relay] run build_relay.sh during the BUILD phase first (run-time rootfs is read-only)." >&2
  exit 1
fi

exec "$PY" "$SERVER" "$@"
