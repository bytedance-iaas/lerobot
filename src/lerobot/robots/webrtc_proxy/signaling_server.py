# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebSocket signaling relay: pairs a Mac daemon with a cloud controller.

A session has two roles — ``robot`` (the Mac daemon) and ``controller`` (the cloud
``WebRTCProxyRobot``). Each connects to ``GET /ws?session=<id>&role=<role>`` and the
relay forwards every message to the *other* role, buffering until that peer joins
(so the daemon may offer before a controller exists). When one side drops, the
relay sends ``{"kind":"bye"}`` to the survivor so the daemon can reset and loop.

This is the M3 control-plane signaler. It carries only SDP — no media, no STUN/TURN.
In production it runs as an ordinary (non-hostNetwork) Deployment behind a normal
Service/Ingress; the media path (UDP, hostNetwork, coturn) is separate (M4).

Run standalone:  ``python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
from collections import defaultdict

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

_OTHER = {"robot": "controller", "controller": "robot"}


class SignalingRelay:
    """Forwards SDP between the two roles of each session, buffering for late joiners."""

    def __init__(self) -> None:
        self._peers: dict[tuple[str, str], web.WebSocketResponse] = {}
        self._inbox: dict[tuple[str, str], list[dict]] = defaultdict(list)

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        session = request.query.get("session", "default")
        role = request.query.get("role", "")
        if role not in _OTHER:
            return web.Response(status=400, text="role must be 'robot' or 'controller'")
        other = _OTHER[role]

        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._peers[(session, role)] = ws
        logger.info("signaling: %s joined session %s", role, session)

        # Flush anything buffered for me before I joined.
        for msg in self._inbox.pop((session, role), []):
            await ws.send_json(msg)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = msg.json()
                    target = self._peers.get((session, other))
                    if target is not None and not target.closed:
                        await target.send_json(data)
                    else:
                        self._inbox[(session, other)].append(data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._peers.pop((session, role), None)
            peer = self._peers.get((session, other))
            if peer is not None and not peer.closed:
                await peer.send_json({"kind": "bye"})
            logger.info("signaling: %s left session %s", role, session)
        return ws


def make_app() -> web.Application:
    relay = SignalingRelay()
    app = web.Application()
    app.router.add_get("/ws", relay.handle)
    return app


async def start_relay(host: str = "127.0.0.1", port: int = 0) -> tuple[web.AppRunner, int]:
    """Start the relay on ``host:port`` (port 0 => ephemeral). Returns (runner, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    actual_port = sock.getsockname()[1]
    runner = web.AppRunner(make_app())
    await runner.setup()
    await web.SockSite(runner, sock).start()
    logger.info("signaling relay listening on ws://%s:%d/ws", host, actual_port)
    return runner, actual_port


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTCProxyRobot WebSocket signaling relay")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 (server binds all ifaces by design)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _run() -> None:
        runner, _ = await start_relay(args.host, args.port)
        try:
            await asyncio.Event().wait()  # run forever
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
