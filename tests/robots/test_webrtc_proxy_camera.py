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
