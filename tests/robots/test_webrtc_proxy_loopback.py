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

"""End-to-end loopback smoke tests for WebRTCProxyRobot (skipped without aiortc)."""

import time

import numpy as np
import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")

from lerobot.robots.webrtc_proxy import WebRTCProxyRobotConfig  # noqa: E402
from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import WebRTCCameraSpec  # noqa: E402
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


def _make_config() -> WebRTCProxyRobotConfig:
    # Small frames keep the test fast.
    return WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        action_timeout_s=0.3,
        connect_timeout_s=15.0,
    )


def test_schema_available_before_connect():
    robot = WebRTCProxyRobot(_make_config())
    assert not robot.is_connected
    assert robot.action_features == {f"{m}.pos": float for m in robot.motors}
    obs_ft = robot.observation_features
    assert obs_ft["front"] == (48, 64, 3)
    assert obs_ft["shoulder_pan.pos"] is float


def test_loopback_observation_roundtrip():
    robot = WebRTCProxyRobot(_make_config())
    robot.connect()
    try:
        assert robot.is_connected
        obs = robot.get_observation()
        # Schema matches observation_features.
        assert set(obs) == set(robot.observation_features)
        frame = obs["front"]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        for m in robot.motors:
            assert isinstance(obs[f"{m}.pos"], float)

        # send_action returns the (clipped/echoed) action actually sent.
        sent = robot.send_action({"shoulder_pan.pos": 12.5})
        assert sent == {"shoulder_pan.pos": 12.5}
    finally:
        robot.disconnect()
    assert not robot.is_connected


def test_watchdog_safes_on_action_stall_then_clears():
    robot = WebRTCProxyRobot(_make_config())
    robot.connect()
    try:
        robot.send_action({"shoulder_pan.pos": 0.0})
        # Stop sending; watchdog must safe within ~action_timeout_s.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not robot._agent.is_safed:
            time.sleep(0.02)
        assert robot._agent.is_safed, "watchdog did not engage after action stall"

        # Resuming actions clears the safe state.
        robot.send_action({"shoulder_pan.pos": 1.0})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and robot._agent.is_safed:
            time.sleep(0.02)
        assert not robot._agent.is_safed, "watchdog did not clear after action resumed"
    finally:
        robot.disconnect()
