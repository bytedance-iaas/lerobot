#!/usr/bin/env python

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

"""Cloud side: teleoperate a remote SO-100 over WebRTC.

The whole point of WebRTCProxyRobot: it is a drop-in lerobot ``Robot``, so existing
teleop/record/policy code drives the remote arm *as if it were local*. Here we reuse
the ``web_so100`` jog panel unchanged — its ``get_action()`` flows through the proxy,
across the public net, to the Mac daemon and the real motors; obs (joints + camera)
flow back over the WebRTC media + data channels.

    Run after the relay and the Mac daemon are up (see README):
        uv run python examples/webrtc_remote_so100/cloud_teleop_so100.py
    then open http://localhost:8080 and jog the remote arm.
"""

import time

from lerobot.robots.webrtc_proxy.configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig
from lerobot.robots.webrtc_proxy.proxy_robot import WebRTCProxyRobot
from lerobot.teleoperators.web_so100 import WebSO100Teleop, WebSO100TeleopConfig

SIGNALING_URL = "ws://127.0.0.1:8765/ws"
SESSION_ID = "so100"
FPS, WIDTH, HEIGHT = 30, 640, 480


def main() -> None:
    # The proxy declares the obs schema (must match the Mac robot's): 6 SO-100 motors
    # + one camera "front". The physical port/index live on the Mac, never here.
    robot = WebRTCProxyRobot(
        WebRTCProxyRobotConfig(
            cameras={"front": WebRTCCameraSpec(height=HEIGHT, width=WIDTH, fps=FPS)},
            signaling_url=SIGNALING_URL,
            session_id=SESSION_ID,
            capture_fps=FPS,
            action_timeout_s=0.5,
        )
    )
    robot.connect()  # reaches the Mac daemon over the relay

    teleop = WebSO100Teleop(WebSO100TeleopConfig(host="0.0.0.0", port=8080))  # noqa: S104
    teleop.connect()
    teleop.attach_robot(robot)  # seed jog targets from the remote arm's current pose

    print("\nTeleoperating the REMOTE SO-100. Open http://localhost:8080 to jog it. Ctrl-C to stop.\n")
    try:
        while True:
            robot.send_action(teleop.get_action())  # action -> WebRTC -> Mac motors
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
