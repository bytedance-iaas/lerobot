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

"""Cloud-side ``WebRTCProxyRobot`` — a fake robot that proxies a real one on a Mac.

LeRobot's record/teleop/policy code calls ``get_observation`` / ``send_action`` /
``observation_features`` synchronously and assumes they are local and instant. This
class honours that synchronous contract while the work happens over WebRTC: it owns
a background asyncio loop running an :class:`_ProxyEndpoint` (the *answerer*).

- ``get_observation`` reads the thread-safe :class:`AlignmentBuffer` (no loop hop) and
  assembles the LeRobot obs dict by *capture* timestamp (handoff 难点 A).
- ``send_action`` marshals an :class:`ActionMsg` onto the loop's action DataChannel.
- the Mac-side watchdog (not here) handles disconnect safety (handoff 难点 C).

M1: ``signaling_url`` unset => loopback mode, which also spins up a synthetic
:class:`CaptureAgent` *in the same loop* so the whole link is self-contained and
testable on one machine. Real mode (M3) replaces loopback signaling with the K8s
WebSocket signaler and drops the in-process capture agent.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from functools import cached_property

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .alignment import AlignmentBuffer
from .capture_agent import CaptureAgent
from .configuration_webrtc_proxy import WebRTCProxyRobotConfig
from .protocol import CH_ACTION, CH_FRAMEMETA, CH_STATE, ActionMsg, FrameMetaMsg, StateMsg
from .signaling import Signaling, loopback_signaling_pair

logger = logging.getLogger(__name__)


class _ProxyEndpoint:
    """Async answerer: receives state/framemeta/action channels + a video track."""

    def __init__(self, buffer: AlignmentBuffer, cam_name: str) -> None:
        self.buffer = buffer
        self.cam_name = cam_name
        # iceServers=[] => no STUN (loopback). M3 supplies STUN/TURN for real peers.
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._ch_action = None
        # ordered framemeta acts as the frame<->capture-time index (popped 1:1 per frame).
        self._framemeta: deque[FrameMetaMsg] = deque(maxlen=256)
        self.connected = asyncio.Event()
        self._action_seq = 0
        self._register()

    def _register(self) -> None:
        @self.pc.on("datachannel")
        def _on_channel(channel):  # noqa: ANN001
            if channel.label == CH_STATE:
                channel.on("message", self._on_state)
            elif channel.label == CH_FRAMEMETA:
                channel.on("message", self._on_framemeta)
            elif channel.label == CH_ACTION:
                self._ch_action = channel

        @self.pc.on("track")
        def _on_track(track):  # noqa: ANN001
            asyncio.ensure_future(self._consume(track))

        @self.pc.on("connectionstatechange")
        async def _on_state_change():
            logger.info("proxy connectionState=%s", self.pc.connectionState)
            if self.pc.connectionState == "connected":
                self.connected.set()

    def _on_state(self, raw: str) -> None:
        try:
            msg = StateMsg.from_json(raw)
        except Exception:
            logger.exception("proxy: bad state message")
            return
        self.buffer.add_state(msg.t, msg.joints, msg.seq)

    def _on_framemeta(self, raw: str) -> None:
        try:
            self._framemeta.append(FrameMetaMsg.from_json(raw))
        except Exception:
            logger.exception("proxy: bad framemeta message")

    async def _consume(self, track) -> None:  # noqa: ANN001
        while True:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("proxy: video track ended")
                return
            # Pop the matching capture timestamp (ordered, 1:1 on a lossless link).
            if not self._framemeta:
                continue  # early frame before its metadata; drop (startup only)
            meta = self._framemeta.popleft()
            img = frame.to_ndarray(format="rgb24")
            self.buffer.add_frame(meta.t, img)

    async def run(self, signaling: Signaling) -> None:
        offer = await signaling.recv()
        if not isinstance(offer, RTCSessionDescription):
            raise RuntimeError(f"proxy expected an SDP offer, got {type(offer)!r}")
        await self.pc.setRemoteDescription(offer)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        await signaling.send(self.pc.localDescription)

    async def send_action(self, goal: dict[str, float]) -> dict[str, float]:
        if self._ch_action is None or self._ch_action.readyState != "open":
            raise RuntimeError("action channel not open")
        self._action_seq += 1
        self._ch_action.send(ActionMsg(t=time.monotonic(), seq=self._action_seq, goal=goal).to_json())
        return goal

    async def close(self) -> None:
        await self.pc.close()


class _EventLoopThread:
    """Owns an asyncio loop in a daemon thread; bridges sync calls to coroutines."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="webrtc-proxy-loop", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)


class WebRTCProxyRobot(Robot):
    """Cloud-side proxy presenting a real Mac-tethered robot as a local LeRobot Robot."""

    config_class = WebRTCProxyRobotConfig
    name = "webrtc_proxy"

    def __init__(self, config: WebRTCProxyRobotConfig):
        super().__init__(config)
        self.config = config
        if len(config.cameras) != 1:
            # M1 transports a single media track. Multi-camera is M2 (one track each).
            raise NotImplementedError(
                f"WebRTCProxyRobot M1 supports exactly one camera, got {list(config.cameras)}"
            )
        self.cam_name, self.cam_spec = next(iter(config.cameras.items()))
        self.motors = list(config.motors)

        self._buffer = AlignmentBuffer(pair_tolerance_s=config.pair_tolerance_s)
        self._loop: _EventLoopThread | None = None
        self._endpoint: _ProxyEndpoint | None = None
        self._agent: CaptureAgent | None = None  # loopback only
        self._last_frame: np.ndarray | None = None
        self._connected = False

    # ----- schema (callable whether connected or not) ----------------------
    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.motors}

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, self.cam_name: (self.cam_spec.height, self.cam_spec.width, 3)}

    @property
    def action_features(self) -> dict:
        return dict(self._motors_ft)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ----- no-op hardware hooks (calibration lives on the Mac, M2) ----------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ----- lifecycle -------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError("WebRTCProxyRobot already connected")
        loopback = self.config.signaling_url in (None, "loopback")
        if not loopback:
            raise NotImplementedError("real signaling (WebSocket/STUN/TURN) is M3; use loopback for now")

        self._loop = _EventLoopThread()
        self._loop.start()

        # CRITICAL: every aiortc object (RTCPeerConnection), asyncio.Event and
        # asyncio.Queue must be constructed *on the loop thread* so it binds to the
        # right running loop. Building them in the caller thread silently deadlocks.
        async def _bringup() -> None:
            self._endpoint = _ProxyEndpoint(self._buffer, self.cam_name)
            proxy_sig, agent_sig = loopback_signaling_pair()
            self._agent = CaptureAgent(
                signaling=agent_sig,
                motors=self.motors,
                cam_name=self.cam_name,
                cam_height=self.cam_spec.height,
                cam_width=self.cam_spec.width,
                capture_fps=self.config.capture_fps,
                action_timeout_s=self.config.action_timeout_s,
            )
            await asyncio.gather(self._endpoint.run(proxy_sig), self._agent.run())
            await self._endpoint.connected.wait()

        self._loop.run(_bringup(), timeout=self.config.connect_timeout_s)
        self._wait_first_obs(self.config.connect_timeout_s)
        self._connected = True
        logger.info("WebRTCProxyRobot connected (%s)", "loopback" if loopback else self.config.signaling_url)

    def _wait_first_obs(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            aligned = self._buffer.assemble()
            if aligned is not None and aligned.frame is not None:
                self._last_frame = aligned.frame
                return
            time.sleep(0.02)
        raise TimeoutError("WebRTCProxyRobot: no aligned observation within connect_timeout_s")

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("WebRTCProxyRobot not connected")
        aligned = self._buffer.assemble()
        if aligned is None:
            raise RuntimeError("no observation available yet")
        frame = aligned.frame if aligned.frame is not None else self._last_frame
        if frame is not None:
            self._last_frame = frame
        if aligned.skew_ms is not None and aligned.skew_ms > self.config.pair_tolerance_s * 1e3:
            logger.warning("state<->frame skew %.0fms exceeds tolerance", aligned.skew_ms)
        obs: RobotObservation = dict(aligned.joints)
        obs[self.cam_name] = frame
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected or self._endpoint is None or self._loop is None:
            raise RuntimeError("WebRTCProxyRobot not connected")
        goal = {k: float(v) for k, v in action.items() if k.endswith(".pos")}
        return self._loop.run(self._endpoint.send_action(goal), timeout=2.0)

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._loop is not None:
            if self._agent is not None:
                try:
                    self._loop.run(self._agent.close(), timeout=2.0)
                except Exception:
                    logger.exception("error closing capture agent")
            if self._endpoint is not None:
                try:
                    self._loop.run(self._endpoint.close(), timeout=2.0)
                except Exception:
                    logger.exception("error closing proxy endpoint")
            self._loop.stop()
        self._connected = False
