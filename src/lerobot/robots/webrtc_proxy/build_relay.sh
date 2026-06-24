#!/usr/bin/env bash
#
# BUILD-time setup for the WebRTC signaling relay.
#
# Run this during the image/build phase (filesystem writable). It creates a venv with
# aiohttp only (~5MB) right next to this script, so it is packaged into the image and
# available at run time — when the rootfs is read-only (e.g. ByteFaaS) and run_relay.sh
# must NOT try to install anything.
#
# FaaS config:
#   build command:  ./src/lerobot/robots/webrtc_proxy/build_relay.sh
#   run command:    ./src/lerobot/robots/webrtc_proxy/run_relay.sh --port 8765 ...
#
# Env overrides:
#   RELAY_VENV     venv location (default: <this dir>/.venv-relay) — keep it inside the
#                  packaged code dir so it survives into the read-only run image.
#   AIOHTTP_SPEC   pip spec for aiohttp (default: aiohttp==3.14.1, matching uv.lock)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RELAY_VENV:-$HERE/.venv-relay}"
AIOHTTP_SPEC="${AIOHTTP_SPEC:-aiohttp==3.14.1}"

echo "[build_relay] creating venv at $VENV with $AIOHTTP_SPEC only…" >&2
# No --seed: the run phase only execs python (no pip needed), so skip fetching pip.
uv venv "$VENV" >&2
# --native-tls: fall back to the OS trust store (some build networks/proxies present a
# cert chain uv's bundled roots reject). Harmless when not needed.
uv pip install --native-tls --python "$VENV/bin/python" "$AIOHTTP_SPEC" >&2

# Fail the build loudly if the relay can't import — better here than at cold start.
"$VENV/bin/python" -c 'import aiohttp; print("[build_relay] aiohttp", aiohttp.__version__, "ready")' >&2
