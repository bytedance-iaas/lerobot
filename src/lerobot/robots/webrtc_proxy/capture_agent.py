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

Without a real ``robot`` it produces a *synthetic* source (no serial bus / camera).
It runs the capture clock, pushes proprioceptive state over the state DataChannel,
streams the camera over the media track (each frame's capture seq carried in its
pts), receives actions, answers control-plane RPCs, and runs the P0 safety watchdog.

The agent owns the WebRTC offer: it creates the state/action/control DataChannels and
adds the video track, then hands its SDP to the signaling channel.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .control import ControlServer, DeviceInventory, SyntheticInventory
from .protocol import CH_ACTION, CH_CONTROL, CH_STATE, ActionMsg, StateMsg
from .signaling import Signaling
from .transport import AiortcTransport, Transport

logger = logging.getLogger(__name__)


def _fit_frame(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Coerce a camera frame to the contiguous ``(height, width, 3)`` uint8 RGB the
    media track requires (the cloud declared this shape in ``observation_features``)."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[0] != height or img.shape[1] != width:
        import cv2

        img = cv2.resize(img, (width, height))
    return np.ascontiguousarray(img)


class CaptureAgent:
    """Mac-side endpoint of the proxy link: publishes state + camera, applies actions.

    Transport-agnostic: it talks to a :class:`Transport` (default aiortc). Swap the
    transport for another backend (e.g. LiveKit) without touching this logic.
    """

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
        inventory: DeviceInventory | None = None,
        ice_servers: list[str] | None = None,
        camera=None,  # an opened lerobot Camera (read_latest); None => synthetic frames
        robot=None,  # a connected lerobot Robot (so_follower) — drives joints+action+torque (M2)
        reliable_state: bool = False,  # True for record (no lost obs); False for teleop/eval (fresh)
        reliable_action: bool = False,
        transport: Transport | None = None,  # default: AiortcTransport (WebRTC P2P)
    ) -> None:
        self.signaling = signaling
        self.motors = list(motors)
        self.cam_name = cam_name
        self.cam_h = cam_height
        self.cam_w = cam_width
        self.period = 1.0 / capture_fps
        self.action_timeout_s = action_timeout_s
        self._on_safe_stop = on_safe_stop

        # The transport offers + sends video, exposes the data channels. The publisher
        # (Mac) sets channel reliability; control is always reliable.
        self._transport = transport or AiortcTransport(
            role="publisher",
            channels={CH_STATE: reliable_state, CH_ACTION: reliable_action, CH_CONTROL: True},
            ice_servers=ice_servers,
        )
        self.closed = self._transport.closed  # set when the link drops
        self._transport.channel(CH_ACTION).on_message(self._on_action)
        self._control = ControlServer(
            inventory if inventory is not None else SyntheticInventory(),
            on_camera_plan=self._apply_camera_plan,
        )
        self._control.attach(self._transport.channel(CH_CONTROL))
        self._camera = camera
        self._robot = robot
        self._last_real_frame: np.ndarray | None = None
        # All serial-bus access (read joints, send action, toggle torque) goes through
        # ONE worker thread so the public-net event loop never blocks on serial and the
        # bus is never touched concurrently. None when there is no real robot.
        self._io: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="webrtc-robot-io") if robot is not None else None
        )
        self._seq = 0
        self._action_seq = 0  # last action seq applied (for telemetry)
        self._last_obs_seq_seen = -1  # provenance of the last action (which obs it came from)
        self._last_goal: dict[str, float] = {f"{m}.pos": 0.0 for m in self.motors}
        self._last_action_t = time.monotonic()
        self._safed = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ----- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        """Establish the transport (offer) and start the capture/watchdog loops.

        Returns once the loops are running; use :pymeth:`wait_closed` to block until
        the link drops.
        """
        await self._transport.open(self.signaling)

        if self._robot is not None and self._io is not None:
            # New session: re-enable torque (a previous session's safe-stop may have cut it).
            self._io.submit(self._robot_enable_torque)

        self._tasks = [
            asyncio.ensure_future(self._capture_loop()),
            asyncio.ensure_future(self._watchdog_loop()),
        ]
        logger.info("CaptureAgent connected; streaming %d motors + camera %r", len(self.motors), self.cam_name)

    async def wait_closed(self) -> None:
        """Block until the WebRTC link to the controller drops."""
        await self.closed.wait()

    def force_safe_stop(self) -> None:
        """Idempotently engage the safe state (used when a session ends)."""
        if not self._safed:
            self._safed = True
            self._safe_stop()

    async def close(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await self._transport.close()
        await self.signaling.close()
        if self._io is not None:
            # Let a pending safe-stop (disable_torque) finish, then stop the io thread.
            self._io.shutdown(wait=True)

    # ----- capture / actuation -------------------------------------------
    def _capture_sample(self, t: float, seq: int) -> tuple[dict[str, float], np.ndarray]:
        """One sample: joints + a camera frame.

        With a real ``robot``, both come from a single ``robot.get_observation()`` (so
        they share a capture instant). Else: a real camera if attached, otherwise a
        synthetic seq-coloured frame; joints are a synthetic sinusoid.
        """
        if self._robot is not None:
            obs = self._robot.get_observation()
            joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
            frame = obs.get(self.cam_name)
            if frame is not None:
                self._last_real_frame = _fit_frame(frame, self.cam_h, self.cam_w)
            img = self._last_real_frame
            if img is None:
                img = np.zeros((self.cam_h, self.cam_w, 3), dtype=np.uint8)
            return joints, img

        # Synthetic arm: hold the last commanded pose, so a jog visibly moves the joints
        # (lets you test the full loop without real hardware). Camera stays live below.
        joints = dict(self._last_goal)
        if self._camera is not None:
            try:
                img = _fit_frame(self._camera.read_latest(max_age_ms=1000), self.cam_h, self.cam_w)
                self._last_real_frame = img
            except Exception:
                # Camera warming up / momentarily stale: reuse last good frame (or black).
                img = self._last_real_frame
                if img is None:
                    img = np.zeros((self.cam_h, self.cam_w, 3), dtype=np.uint8)
        else:
            img = np.empty((self.cam_h, self.cam_w, 3), dtype=np.uint8)
            img[:] = (seq % 256, (seq * 5) % 256, 128)
        return joints, img

    def _apply_action(self, goal: dict[str, float]) -> dict[str, float]:
        """Drive the arm (runs on the io thread). Returns the action actually sent."""
        if self._robot is not None:
            try:
                return self._robot.send_action(goal)
            except Exception:
                logger.exception("CaptureAgent: robot.send_action failed")
                return goal
        self._last_goal = dict(goal)
        return self._last_goal

    def _robot_enable_torque(self) -> None:
        try:
            self._robot.bus.enable_torque()
        except Exception:
            logger.exception("CaptureAgent: enable_torque failed")

    def _robot_disable_torque(self) -> None:
        try:
            self._robot.bus.disable_torque()
            logger.warning("CaptureAgent: torque disabled (safe stop)")
        except Exception:
            logger.exception("CaptureAgent: disable_torque failed")

    def _apply_camera_plan(self, plan: dict) -> None:
        """Cloud told us its desired obs size — encode/resize frames to it (bandwidth)."""
        w, h = plan.get("width"), plan.get("height")
        if w and h and (w != self.cam_w or h != self.cam_h):
            logger.info("camera plan: obs size %dx%d -> %dx%d", self.cam_w, self.cam_h, w, h)
            self.cam_w, self.cam_h = int(w), int(h)

    def _safe_stop(self) -> None:
        """P0: watchdog fired (actions stopped). Cut torque so the arm goes limp."""
        logger.warning("WATCHDOG: no action for %.0fms -> SAFE STOP", self.action_timeout_s * 1e3)
        if self._robot is not None and self._io is not None:
            self._io.submit(self._robot_disable_torque)  # never touch the bus off the io thread
        if self._on_safe_stop is not None:
            self._on_safe_stop()

    # ----- loops -----------------------------------------------------------
    async def _capture_loop(self) -> None:
        loop = asyncio.get_event_loop()
        next_t = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            seq = self._seq
            self._seq += 1
            # Real serial reads run off-loop so public-net timing never blocks on the bus.
            if self._io is not None:
                joints, img = await loop.run_in_executor(self._io, self._capture_sample, t, seq)
            else:
                joints, img = self._capture_sample(t, seq)

            # Piggyback the last applied action (seq + time) so the cloud can confirm
            # landing and measure round-trip without an extra channel.
            self._transport.channel(CH_STATE).send(
                StateMsg(
                    t=t,
                    seq=seq,
                    joints=joints,
                    applied_seq=self._action_seq,
                    applied_t=self._last_action_t,
                ).to_json()
            )
            # The frame carries its seq in its pts (set inside the transport).
            self._transport.send_frame(seq, img)

            next_t += self.period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    def _on_action(self, raw: str) -> None:
        try:
            msg = ActionMsg.from_json(raw)
        except Exception:
            logger.exception("CaptureAgent: bad action message")
            return
        self._last_action_t = time.monotonic()  # sync: keeps the watchdog honest
        self._action_seq = msg.seq
        self._last_obs_seq_seen = msg.obs_seq
        resumed = self._safed
        if self._safed:
            logger.info("WATCHDOG: action resumed (seq=%d) -> clearing safe state", msg.seq)
            self._safed = False
        if self._io is not None:
            if resumed:
                self._io.submit(self._robot_enable_torque)  # safe-stop cut torque; bring it back
            self._io.submit(self._apply_action, msg.goal)  # serial write off the event loop
        else:
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
