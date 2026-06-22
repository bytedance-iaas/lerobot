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

"""Mac-side capture agent (the *offerer*).

In M1 this produces a *synthetic* robot: it does not touch a serial bus or camera.
It runs the capture clock, pushes proprioceptive state + per-frame metadata over
DataChannels, streams a synthetic camera over the media track, receives actions,
and runs the P0 safety watchdog. Swapping the synthetic source for a real
``so_follower`` + cameras is M2 and only touches ``_capture_sample`` /
``_apply_action`` / ``_safe_stop``.

The agent owns the WebRTC offer: it creates the three DataChannels and adds the
video track, then hands its SDP to the signaling channel.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame

from .protocol import (
    CH_ACTION,
    CH_FRAMEMETA,
    CH_STATE,
    ORDERED_CHANNEL_KWARGS,
    RT_CHANNEL_KWARGS,
    VIDEO_CLOCK_RATE,
    ActionMsg,
    FrameMetaMsg,
    StateMsg,
)
from .signaling import Signaling

logger = logging.getLogger(__name__)


class _SyntheticCameraTrack(MediaStreamTrack):
    """Outbound video track fed by the capture loop via an asyncio queue.

    Pixels only — the capture timestamp travels on the ``framemeta`` channel, so we
    just need a sane monotonically-increasing pts here.
    """

    kind = "video"

    def __init__(self, queue: asyncio.Queue[np.ndarray]) -> None:
        super().__init__()
        self._queue = queue
        self._t0: float | None = None

    async def recv(self) -> VideoFrame:
        img = await self._queue.get()
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = int((now - self._t0) * VIDEO_CLOCK_RATE)
        frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return frame


class CaptureAgent:
    """Synthetic Mac-side endpoint of the WebRTC proxy link."""

    def __init__(
        self,
        signaling: Signaling,
        motors: list[str],
        cam_name: str,
        cam_height: int,
        cam_width: int,
        capture_fps: int = 30,
        action_timeout_s: float = 0.5,
        on_safe_stop: Callable[[], None] | None = None,
    ) -> None:
        self.signaling = signaling
        self.motors = list(motors)
        self.cam_name = cam_name
        self.cam_h = cam_height
        self.cam_w = cam_width
        self.period = 1.0 / capture_fps
        self.action_timeout_s = action_timeout_s
        self._on_safe_stop = on_safe_stop

        # iceServers=[] => host candidates only (no STUN). Loopback needs no NAT
        # traversal; M3 will inject STUN/TURN here for real public-net peers.
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._frame_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2)
        self._ch_state = None
        self._ch_framemeta = None
        self._ch_action = None
        self._seq = 0
        self._action_seq = 0  # last action seq applied (for telemetry)
        self._last_goal: dict[str, float] = {f"{m}.pos": 0.0 for m in self.motors}
        self._last_action_t = time.monotonic()
        self._safed = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ----- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        """Connect (as offerer) and run until ``close()`` / signaling ends."""
        self._ch_state = self.pc.createDataChannel(CH_STATE, **RT_CHANNEL_KWARGS)
        self._ch_framemeta = self.pc.createDataChannel(CH_FRAMEMETA, **ORDERED_CHANNEL_KWARGS)
        self._ch_action = self.pc.createDataChannel(CH_ACTION, **RT_CHANNEL_KWARGS)
        self._ch_action.on("message", self._on_action)

        self.pc.addTrack(_SyntheticCameraTrack(self._frame_q))

        await self.pc.setLocalDescription(await self.pc.createOffer())
        await self.signaling.send(self.pc.localDescription)
        answer = await self.signaling.recv()
        if not isinstance(answer, RTCSessionDescription):
            raise RuntimeError(f"capture agent expected an SDP answer, got {type(answer)!r}")
        await self.pc.setRemoteDescription(answer)

        self._tasks = [
            asyncio.ensure_future(self._capture_loop()),
            asyncio.ensure_future(self._watchdog_loop()),
        ]
        logger.info("CaptureAgent connected; streaming %d motors + camera %r", len(self.motors), self.cam_name)

    async def close(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await self.pc.close()

    # ----- capture (replace these 3 for real hardware in M2) ---------------
    def _capture_sample(self, t: float, seq: int) -> tuple[dict[str, float], np.ndarray]:
        """Synthetic sample: slow sinusoidal joints + a frame whose colour encodes seq."""
        joints = {f"{m}.pos": 30.0 * np.sin(t + i) for i, m in enumerate(self.motors)}
        img = np.empty((self.cam_h, self.cam_w, 3), dtype=np.uint8)
        img[:] = (seq % 256, (seq * 5) % 256, 128)
        return joints, img

    def _apply_action(self, goal: dict[str, float]) -> dict[str, float]:
        """Pretend to drive the arm. Real impl clips + calls robot.send_action (M2)."""
        self._last_goal = dict(goal)
        return self._last_goal

    def _safe_stop(self) -> None:
        """P0: called by the watchdog when actions stop arriving. Real impl cuts torque."""
        logger.warning("WATCHDOG: no action for %.0fms -> SAFE STOP", self.action_timeout_s * 1e3)
        if self._on_safe_stop is not None:
            self._on_safe_stop()

    # ----- loops -----------------------------------------------------------
    async def _capture_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            seq = self._seq
            self._seq += 1
            joints, img = self._capture_sample(t, seq)

            if self._ch_state is not None and self._ch_state.readyState == "open":
                self._ch_state.send(StateMsg(t=t, seq=seq, joints=joints).to_json())
            if self._ch_framemeta is not None and self._ch_framemeta.readyState == "open":
                self._ch_framemeta.send(FrameMetaMsg(t=t, seq=seq).to_json())
            # Drop the oldest pending frame rather than block the capture clock.
            if self._frame_q.full():
                _ = self._frame_q.get_nowait()
            self._frame_q.put_nowait(img)

            next_t += self.period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    def _on_action(self, raw: str) -> None:
        try:
            msg = ActionMsg.from_json(raw)
        except Exception:
            logger.exception("CaptureAgent: bad action message")
            return
        self._last_action_t = time.monotonic()
        self._action_seq = msg.seq
        if self._safed:
            logger.info("WATCHDOG: action resumed (seq=%d) -> clearing safe state", msg.seq)
            self._safed = False
        self._apply_action(msg.goal)

    async def _watchdog_loop(self) -> None:
        # Poll at ~4x the timeout so we catch a stall well within one window.
        tick = max(self.action_timeout_s / 4.0, 0.02)
        while not self._stop.is_set():
            await asyncio.sleep(tick)
            stalled = (time.monotonic() - self._last_action_t) > self.action_timeout_s
            if stalled and not self._safed:
                self._safed = True
                self._safe_stop()

    # ----- introspection (tests) -------------------------------------------
    @property
    def is_safed(self) -> bool:
        return self._safed
