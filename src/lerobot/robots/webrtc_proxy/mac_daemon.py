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

"""Mac-side daemon: the long-lived process that keeps the real robot on the user's
machine and serves cloud sessions over WebRTC.

It outlives any single cloud session: connect to the signaling relay, offer, serve
one controller until the link drops, **safe the arm**, then loop and wait for the
next session. This is the persistent counterpart to the cloud ``WebRTCProxyRobot``.

M3 ships the synthetic source (``CaptureAgent`` + ``SyntheticInventory``); M2 swaps
in a real ``so_follower`` + cameras + ``LocalDeviceInventory`` behind the same loop.

Run:
    # 1) start the relay (cloud, here localhost for a same-host demo)
    python -m lerobot.robots.webrtc_proxy.signaling_server --port 8765
    # 2) start this daemon on the Mac
    python -m lerobot.robots.webrtc_proxy.mac_daemon --signaling-url ws://127.0.0.1:8765/ws
    # 3) cloud side: WebRTCProxyRobotConfig(signaling_url="ws://127.0.0.1:8765/ws").connect()
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .capture_agent import CaptureAgent
from .configuration_webrtc_proxy import SO100_MOTORS
from .control import DeviceInventory, LocalDeviceInventory, SyntheticInventory
from .signaling import SignalingClosed, WebSocketSignaling

logger = logging.getLogger(__name__)


async def run_daemon(
    signaling_url: str,
    session_id: str = "default",
    motors: list[str] | None = None,
    cam_name: str = "front",
    cam_height: int = 480,
    cam_width: int = 640,
    capture_fps: int = 30,
    action_timeout_s: float = 0.5,
    ice_servers: list[str] | None = None,
    inventory: DeviceInventory | None = None,
    camera=None,
    stop: asyncio.Event | None = None,
) -> None:
    """Serve cloud sessions forever (until ``stop`` is set), one session at a time.

    The camera (if any) is owned by the caller and reused across sessions — only the
    per-session WebRTC peer is rebuilt each loop.
    """
    motors = list(motors or SO100_MOTORS)
    stop = stop or asyncio.Event()
    while not stop.is_set():
        sig = WebSocketSignaling(signaling_url, session_id, role="robot")
        agent = CaptureAgent(
            signaling=sig,
            motors=motors,
            cam_name=cam_name,
            cam_height=cam_height,
            cam_width=cam_width,
            capture_fps=capture_fps,
            action_timeout_s=action_timeout_s,
            inventory=inventory if inventory is not None else SyntheticInventory(),
            ice_servers=ice_servers,
            camera=camera,
        )
        try:
            await agent.run()  # blocks here until a controller answers the offer
            logger.info("daemon: session %s established", session_id)
            await agent.wait_closed()
            logger.info("daemon: session %s closed", session_id)
        except SignalingClosed:
            logger.info("daemon: signaling closed before a session formed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("daemon: session error")
        finally:
            agent.force_safe_stop()  # P0: never leave the arm live across sessions
            await agent.close()
        if not stop.is_set():
            await asyncio.sleep(0.2)  # brief backoff before waiting for the next session


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTCProxyRobot Mac-side daemon")
    parser.add_argument("--signaling-url", required=True, help="ws://host:port/ws")
    parser.add_argument("--session", default="default")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-timeout", type=float, default=0.5)
    parser.add_argument(
        "--ice-server", action="append", default=[], help="STUN/TURN url (repeatable); omit for same-host"
    )
    parser.add_argument(
        "--real-devices",
        action="store_true",
        help="enumerate the Mac's actual serial ports + cameras (find_port/list_cameras return real ids)",
    )
    parser.add_argument(
        "--real-camera",
        default=None,
        help="open this opencv camera (index e.g. 0, or /dev/videoN) and stream it instead of synthetic frames",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    inventory: DeviceInventory = LocalDeviceInventory() if args.real_devices else SyntheticInventory()
    logger.info("daemon device inventory: %s", type(inventory).__name__)

    camera = None
    if args.real_camera is not None:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        index_or_path = int(args.real_camera) if args.real_camera.isdigit() else args.real_camera
        camera = OpenCVCamera(
            OpenCVCameraConfig(index_or_path=index_or_path, fps=args.fps, width=args.width, height=args.height)
        )
        camera.connect()
        logger.info("daemon streaming real camera %r @ %dx%d", index_or_path, args.width, args.height)

    try:
        asyncio.run(
            run_daemon(
                signaling_url=args.signaling_url,
                session_id=args.session,
                cam_name=args.camera_name,
                cam_height=args.height,
                cam_width=args.width,
                capture_fps=args.fps,
                action_timeout_s=args.action_timeout,
                ice_servers=args.ice_server,
                inventory=inventory,
                camera=camera,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.disconnect()


if __name__ == "__main__":
    main()
