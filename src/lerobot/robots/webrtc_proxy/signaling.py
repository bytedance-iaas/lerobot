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


class Signaling(Protocol):
    """Minimal duplex SDP exchange used by both endpoints."""

    async def send(self, description: RTCSessionDescription) -> None: ...

    async def recv(self) -> RTCSessionDescription: ...


class LoopbackSignaling:
    """One end of an in-process signaling pair. See :func:`loopback_signaling_pair`."""

    def __init__(self, outbox: asyncio.Queue, inbox: asyncio.Queue) -> None:
        self._outbox = outbox
        self._inbox = inbox

    async def send(self, description: RTCSessionDescription) -> None:
        await self._outbox.put(description)

    async def recv(self) -> RTCSessionDescription:
        return await self._inbox.get()


def loopback_signaling_pair() -> tuple[LoopbackSignaling, LoopbackSignaling]:
    """Return ``(offerer_signaling, answerer_signaling)`` wired back-to-back."""
    a_to_b: asyncio.Queue = asyncio.Queue()
    b_to_a: asyncio.Queue = asyncio.Queue()
    offerer = LoopbackSignaling(outbox=a_to_b, inbox=b_to_a)
    answerer = LoopbackSignaling(outbox=b_to_a, inbox=a_to_b)
    return offerer, answerer
