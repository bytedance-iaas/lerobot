#!/usr/bin/env bash
#
# Minimal launcher for the WebRTC signaling relay.
#
# The relay (signaling_server.py) only needs `aiohttp` + the Python stdlib — none of
# lerobot's heavy deps (torch, aiortc, ...). It has NO relative imports, so we run the
# file directly. Do NOT use `python -m lerobot.robots.webrtc_proxy.signaling_server`:
# the `-m` form imports the whole package tree (lerobot.robots.__init__ ->
# make_robot_from_config -> torch/...), defeating the point of a slim install.
#
# This creates a throwaway venv with aiohttp only (a few MB) instead of running a full
# `uv sync` (which pulls in the entire lerobot dependency set).
#
# Usage:
#   ./run_relay.sh --port 8765
#   ./run_relay.sh --port 8765 --stun-url stun:stun.l.google.com:19302
#   ./run_relay.sh --port 8765 --auth-token "$SIGNALING_AUTH_TOKEN"
# All arguments are forwarded verbatim to signaling_server.py.
#
# Env overrides:
#   RELAY_VENV     venv location (default: <this dir>/.venv-relay)
#   AIOHTTP_SPEC   pip spec for aiohttp (default: aiohttp==3.14.1, matching uv.lock)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RELAY_VENV:-$HERE/.venv-relay}"
AIOHTTP_SPEC="${AIOHTTP_SPEC:-aiohttp==3.14.1}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[run_relay] creating venv at $VENV with $AIOHTTP_SPEC only…" >&2
  uv venv "$VENV" >&2
  # --native-tls: fall back to the OS trust store (some networks/proxies present a cert
  # chain uv's bundled roots reject). Harmless when not needed.
  uv pip install --native-tls --python "$VENV/bin/python" "$AIOHTTP_SPEC" >&2
fi

exec "$VENV/bin/python" "$HERE/signaling_server.py" "$@"
