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

"""Signaling abstraction: exchanges SDP offer/answer between the two peers.

M1 uses :class:`LoopbackSignaling`, an in-process pair backed by ``asyncio.Queue``
(no STUN/TURN, no network — both peers share one event loop). M3 will add a
WebSocket implementation against the K8s signaler with the *same* ``send``/``recv``
interface, so neither the capture agent nor the proxy changes.

aiortc gathers ICE candidates into the SDP (non-trickle) by the time
``localDescription`` is read, so exchanging full session descriptions is enough on
loopback; trickle-ICE is only needed once real NAT traversal (STUN/TURN) is in play.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from aiortc import RTCSessionDescription


class SignalingClosed(Exception):
    """Raised by ``recv`` when the peer left or the transport closed."""


class Signaling(Protocol):
    """Minimal duplex SDP exchange used by both endpoints."""

    async def open(self) -> None: ...

    async def send(self, description: RTCSessionDescription) -> None: ...

    async def recv(self) -> RTCSessionDescription: ...

    async def close(self) -> None: ...


class LoopbackSignaling:
    """One end of an in-process signaling pair. See :func:`loopback_signaling_pair`."""

    def __init__(self, outbox: asyncio.Queue, inbox: asyncio.Queue) -> None:
        self._outbox = outbox
        self._inbox = inbox

    async def open(self) -> None:
        pass

    async def send(self, description: RTCSessionDescription) -> None:
        await self._outbox.put(description)

    async def recv(self) -> RTCSessionDescription:
        return await self._inbox.get()

    async def close(self) -> None:
        pass


def loopback_signaling_pair() -> tuple[LoopbackSignaling, LoopbackSignaling]:
    """Return ``(offerer_signaling, answerer_signaling)`` wired back-to-back."""
    a_to_b: asyncio.Queue = asyncio.Queue()
    b_to_a: asyncio.Queue = asyncio.Queue()
    offerer = LoopbackSignaling(outbox=a_to_b, inbox=b_to_a)
    answerer = LoopbackSignaling(outbox=b_to_a, inbox=a_to_b)
    return offerer, answerer


class WebSocketSignaling:
    """SDP exchange over a WebSocket relay (see ``signaling_server.py``).

    Both the Mac daemon (``role="robot"``) and the cloud controller
    (``role="controller"``) connect to the same relay URL with a shared
    ``session`` id; the relay forwards each SDP between them (buffering until the
    peer joins). Non-trickle ICE means just one offer + one answer cross the wire.
    """

    def __init__(self, base_url: str, session: str, role: str) -> None:
        sep = "&" if "?" in base_url else "?"
        self._url = f"{base_url}{sep}session={session}&role={role}"
        self._client = None
        self._ws = None

    async def open(self) -> None:
        import aiohttp

        self._client = aiohttp.ClientSession()
        self._ws = await self._client.ws_connect(self._url)

    async def send(self, description: RTCSessionDescription) -> None:
        await self._ws.send_json({"kind": "sdp", "type": description.type, "sdp": description.sdp})

    async def recv(self) -> RTCSessionDescription:
        import aiohttp

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                if data.get("kind") == "sdp":
                    return RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                if data.get("kind") == "bye":
                    raise SignalingClosed("peer left the signaling session")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
        raise SignalingClosed("signaling websocket closed")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._client is not None:
            await self._client.close()
