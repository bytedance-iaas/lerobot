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

"""Real-camera streaming: a frame source feeds the media track end-to-end."""

import numpy as np
import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")

from lerobot.robots.webrtc_proxy.capture_agent import _fit_frame  # noqa: E402
from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


class _FakeCamera:
    """Duck-types a lerobot Camera: read_latest() returns a fixed RGB frame."""

    def __init__(self, height: int, width: int, color: tuple[int, int, int]) -> None:
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame[:] = color

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        return self._frame.copy()


def test_fit_frame_resizes_and_normalizes():
    # wrong size + RGBA + non-contiguous -> coerced to (48, 64, 3) uint8 contiguous
    src = np.zeros((30, 40, 4), dtype=np.uint8)
    out = _fit_frame(src, 48, 64)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_get_observation_enforces_declared_shape():
    """A wrong-sized frame in the buffer is re-fit to the declared obs shape."""
    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        connect_timeout_s=15.0,
    )
    robot = WebRTCProxyRobot(cfg)
    robot.connect()
    try:
        # Inject a frame of the WRONG size straight into the alignment buffer.
        robot._buffer.add_frame(9.0, np.full((100, 200, 3), 77, np.uint8))
        robot._buffer.add_state(9.0, {f"{m}.pos": 0.0 for m in robot.motors}, seq=999)
        obs = robot.get_observation()
        assert obs["front"].shape == (48, 64, 3)  # re-fit to the declared spec
    finally:
        robot.disconnect()


def test_camera_plan_updates_agent_obs_size():
    """The cloud's set_camera_plan reaches the (in-process loopback) agent."""
    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        connect_timeout_s=15.0,
    )
    robot = WebRTCProxyRobot(cfg)
    robot.connect()
    try:
        # The plan is pushed at connect; the agent's resize target matches the spec.
        assert (robot._agent.cam_w, robot._agent.cam_h) == (64, 48)
        # A fresh plan is applied too.
        robot._agent._apply_camera_plan({"width": 32, "height": 24})
        assert (robot._agent.cam_w, robot._agent.cam_h) == (32, 24)
    finally:
        robot.disconnect()


def test_grab_camera_preview_over_loopback():
    """Cloud-driven single-frame grab returns a decoded RGB image of the asked size."""
    from lerobot.robots.webrtc_proxy.control import SyntheticInventory

    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        connect_timeout_s=15.0,
    )
    robot = WebRTCProxyRobot(cfg, inventory=SyntheticInventory())
    robot.connect()
    try:
        img = robot.grab_camera_preview(1, width=64, height=48)
        assert img.shape == (48, 64, 3)
        assert img.dtype == np.uint8
        # SyntheticInventory colours each id distinctly -> id 0 and id 1 differ.
        other = robot.grab_camera_preview(0, width=64, height=48)
        assert not np.array_equal(img, other)
    finally:
        robot.disconnect()


def test_real_camera_frames_reach_the_cloud():
    color = (200, 100, 50)
    cam = _FakeCamera(48, 64, color)
    cfg = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        action_timeout_s=0.5,
        connect_timeout_s=15.0,
    )
    robot = WebRTCProxyRobot(cfg, camera=cam)
    robot.connect()
    try:
        frame = robot.get_observation()["front"]
        assert frame.shape == (48, 64, 3)
        # The encode/decode round-trip (VP8/H264) is lossy, so allow tolerance, but the
        # mean colour must clearly track the camera's frame, not the synthetic generator.
        mean = frame.reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean, color, atol=40), f"got mean {mean}, expected ~{color}"
    finally:
        robot.disconnect()
