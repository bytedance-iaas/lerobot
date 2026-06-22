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

"""Runnable M1 loopback demo for ``WebRTCProxyRobot``.

    uv run python -m lerobot.robots.webrtc_proxy.demo_loopback

Brings up the full WebRTC link (synthetic Mac capture agent <-> cloud proxy) on one
machine, then drives it through the *synchronous* LeRobot Robot API: streams
re-assembled observations for a couple seconds, sends actions, and demonstrates the
P0 watchdog firing once the action stream stops.
"""

from __future__ import annotations

import logging
import time

from .configuration_webrtc_proxy import WebRTCCameraSpec, WebRTCProxyRobotConfig
from .proxy_robot import WebRTCProxyRobot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = WebRTCProxyRobotConfig(
        cameras={"front": WebRTCCameraSpec(height=120, width=160, fps=30)},
        capture_fps=30,
        action_timeout_s=0.4,
    )
    robot = WebRTCProxyRobot(config)

    print("observation_features:", {k: v for k, v in robot.observation_features.items()})
    print("action_features:    ", robot.action_features)

    robot.connect()
    print("\n== connected; streaming re-assembled observations (capture-ts aligned) ==")
    try:
        for _ in range(30):
            obs = robot.get_observation()
            aligned = robot._buffer.assemble()  # for skew telemetry only
            skew = aligned.skew_ms if aligned else None
            frame = obs["front"]
            pan = obs["shoulder_pan.pos"]
            skew_s = f"{skew:.1f}ms" if skew is not None else "n/a"
            print(f"  shoulder_pan.pos={pan:+7.2f}  front={frame.shape}{frame.dtype}  skew={skew_s}")
            robot.send_action({"shoulder_pan.pos": pan + 1.0})
            time.sleep(1 / 15)

        print("\n== stop sending actions; watchdog should SAFE STOP within action_timeout_s ==")
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not robot._agent.is_safed:
            time.sleep(0.05)
        print(f"  watchdog safed = {robot._agent.is_safed}")

        print("\n== resume actions; watchdog should clear ==")
        robot.send_action({"shoulder_pan.pos": 0.0})
        time.sleep(0.1)
        print(f"  watchdog safed = {robot._agent.is_safed}")
    finally:
        robot.disconnect()
        print("\n== disconnected cleanly ==")


if __name__ == "__main__":
    main()
