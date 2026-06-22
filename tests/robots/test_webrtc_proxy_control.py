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

"""Control-plane (device discovery) tests over the loopback link."""

import pytest

pytest.importorskip("aiortc", reason="WebRTCProxyRobot needs the lerobot[webrtc] extra (aiortc)")

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import (  # noqa: E402
    WebRTCCameraSpec,
    WebRTCProxyRobotConfig,
)
from lerobot.robots.webrtc_proxy.control import FindPortError, SyntheticInventory  # noqa: E402
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot  # noqa: E402


def _config() -> WebRTCProxyRobotConfig:
    return WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=48, width=64, fps=30)},
        capture_fps=30,
        action_timeout_s=0.5,
        connect_timeout_s=15.0,
    )


# ---- pure server logic (no transport) -------------------------------------
def test_find_port_diff_logic():
    from lerobot.robots.webrtc_proxy.control import ControlServer
    from lerobot.robots.webrtc_proxy.protocol import RpcRequest

    inv = SyntheticInventory(ports=["/dev/a", "/dev/b", "/dev/c"])
    srv = ControlServer(inv)
    srv._dispatch(RpcRequest(1, "find_port_begin", {}))
    inv.simulate_unplug("/dev/b")
    assert srv._dispatch(RpcRequest(2, "find_port_result", {})) == {"port": "/dev/b"}


def test_find_port_ambiguous_raises():
    from lerobot.robots.webrtc_proxy.control import ControlServer
    from lerobot.robots.webrtc_proxy.protocol import RpcRequest

    inv = SyntheticInventory(ports=["/dev/a", "/dev/b"])
    srv = ControlServer(inv)
    srv._dispatch(RpcRequest(1, "find_port_begin", {}))
    # nothing unplugged -> 0 diff -> error
    with pytest.raises(FindPortError):
        srv._dispatch(RpcRequest(2, "find_port_result", {}))


# ---- end-to-end over the loopback control channel -------------------------
def test_discovery_rpc_over_loopback():
    inv = SyntheticInventory(
        ports=["/dev/tty.usbmodem-A", "/dev/tty.usbmodem-B"],
        cameras=[{"type": "opencv", "index_or_path": 0, "name": "front cam"}],
    )
    robot = WebRTCProxyRobot(_config(), inventory=inv)
    robot.connect()
    try:
        assert set(robot.list_ports()) == {"/dev/tty.usbmodem-A", "/dev/tty.usbmodem-B"}
        cams = robot.list_cameras()
        assert cams == [{"type": "opencv", "index_or_path": 0, "name": "front cam"}]

        # Event-driven find_port: begin, (human unplugs == simulate), result.
        before = robot.find_port_begin()
        assert "/dev/tty.usbmodem-B" in before
        inv.simulate_unplug("/dev/tty.usbmodem-B")
        assert robot.find_port_result() == "/dev/tty.usbmodem-B"
    finally:
        robot.disconnect()


def test_control_rpc_error_propagates():
    robot = WebRTCProxyRobot(_config(), inventory=SyntheticInventory())
    robot.connect()
    try:
        # find_port_result before begin -> server raises -> surfaces as RuntimeError on cloud.
        with pytest.raises(RuntimeError):
            robot.find_port_result()
    finally:
        robot.disconnect()
