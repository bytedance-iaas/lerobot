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

"""Control plane: cloud-driven device onboarding over a reliable RPC channel.

Port + camera IDs are *Mac-local* physical identifiers that are meaningless in the
cloud, so the cloud never stores them. Instead the cloud drives discovery over the
``control`` DataChannel and the Mac answers from its own OS:

    cloud  ──RpcRequest{list_ports}──▶  Mac (ControlServer -> DeviceInventory)
    cloud  ◀──RpcResponse{result}─────  Mac

The crux vs. the local ``lerobot-find-port`` CLI: that tool blocks on ``input()``
while the human unplugs the bus, but here the human is at the Mac and the
orchestrator is in the cloud. So ``find_port`` becomes an *event-driven* two-step —
``find_port_begin`` snapshots the ports, the human unplugs (prompted by the cloud
UI), then ``find_port_result`` diffs to the port that disappeared. No shared stdin.

``DeviceInventory`` is the seam between this transport and the OS: M3 ships a
``SyntheticInventory`` (loopback-testable); a real ``LocalDeviceInventory`` wrapping
``lerobot.scripts.lerobot_find_port`` / ``lerobot_find_cameras`` lands with M2/M4
hardware bring-up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from .protocol import RpcRequest, RpcResponse

logger = logging.getLogger(__name__)


class FindPortError(RuntimeError):
    """Raised when a find_port diff is ambiguous (0 or >1 ports changed)."""


class DeviceInventory(Protocol):
    """OS-facing device enumeration. Implementations run on the Mac."""

    def list_ports(self) -> list[str]: ...

    def list_cameras(self) -> list[dict[str, Any]]: ...


class SyntheticInventory:
    """In-memory inventory for loopback tests / the synthetic capture agent.

    Mirrors the real shapes: ``list_ports`` returns serial-port strings; cameras
    carry a *stable* identifier (``index_or_path`` for opencv, ``serial`` for
    realsense) plus a human-readable name so a person can map them to logical roles.
    ``simulate_unplug`` lets a test reproduce the find_port unplug step.
    """

    def __init__(
        self,
        ports: list[str] | None = None,
        cameras: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ports: list[str] = list(
            ports if ports is not None else ["/dev/tty.usbmodem-FAKE-A", "/dev/tty.usbmodem-FAKE-B"]
        )
        self._cameras: list[dict[str, Any]] = list(
            cameras
            if cameras is not None
            else [
                {"type": "opencv", "index_or_path": 0, "name": "FaceTime HD Camera"},
                {"type": "opencv", "index_or_path": 1, "name": "USB Camera"},
            ]
        )

    def list_ports(self) -> list[str]:
        return list(self._ports)

    def list_cameras(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._cameras]

    def simulate_unplug(self, port: str) -> None:
        if port in self._ports:
            self._ports.remove(port)


class ControlServer:
    """Mac side: receives RpcRequests on the control channel and answers them.

    Stateful only for the find_port begin/result handshake (it stashes the
    pre-unplug port snapshot between the two calls).
    """

    def __init__(self, inventory: DeviceInventory) -> None:
        self.inventory = inventory
        self._channel = None
        self._ports_before: list[str] | None = None

    def attach(self, channel) -> None:  # noqa: ANN001 (aiortc RTCDataChannel)
        self._channel = channel
        channel.on("message", self._on_message)

    def _on_message(self, raw: str) -> None:
        try:
            req = RpcRequest.from_json(raw)
        except Exception:
            logger.exception("ControlServer: malformed RpcRequest")
            return
        try:
            result = self._dispatch(req)
            resp = RpcResponse(id=req.id, ok=True, result=result)
        except Exception as e:  # report failures back to the cloud, don't crash the loop
            resp = RpcResponse(id=req.id, ok=False, error=f"{type(e).__name__}: {e}")
        if self._channel is not None and self._channel.readyState == "open":
            self._channel.send(resp.to_json())

    def _dispatch(self, req: RpcRequest) -> Any:
        if req.method == "list_ports":
            return {"ports": self.inventory.list_ports()}
        if req.method == "list_cameras":
            return {"cameras": self.inventory.list_cameras()}
        if req.method == "find_port_begin":
            self._ports_before = self.inventory.list_ports()
            return {"ports": list(self._ports_before)}
        if req.method == "find_port_result":
            if self._ports_before is None:
                raise FindPortError("find_port_result called before find_port_begin")
            after = self.inventory.list_ports()
            diff = sorted(set(self._ports_before) - set(after))
            self._ports_before = None
            if len(diff) == 1:
                return {"port": diff[0]}
            raise FindPortError(f"expected exactly one disconnected port, found {diff}")
        raise ValueError(f"unknown control method {req.method!r}")


class ControlClient:
    """Cloud side: issues RpcRequests and awaits the matching RpcResponse by id."""

    def __init__(self) -> None:
        self._channel = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}

    def attach(self, channel) -> None:  # noqa: ANN001
        self._channel = channel
        channel.on("message", self._on_message)

    def _on_message(self, raw: str) -> None:
        try:
            resp = RpcResponse.from_json(raw)
        except Exception:
            logger.exception("ControlClient: malformed RpcResponse")
            return
        fut = self._pending.pop(resp.id, None)
        if fut is None or fut.done():
            return
        if resp.ok:
            fut.set_result(resp.result)
        else:
            fut.set_exception(RuntimeError(resp.error or "control RPC failed"))

    async def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        if self._channel is None or self._channel.readyState != "open":
            raise RuntimeError("control channel not open")
        self._next_id += 1
        req = RpcRequest(id=self._next_id, method=method, params=params or {})
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req.id] = fut
        self._channel.send(req.to_json())
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(req.id, None)
