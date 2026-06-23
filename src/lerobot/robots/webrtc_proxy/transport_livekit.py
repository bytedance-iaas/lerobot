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

"""LiveKit transport backend — EXPERIMENTAL / UNTESTED SCAFFOLD.

Implements the :class:`Transport` interface on top of the LiveKit Python SDK
(``livekit`` / ``livekit-rtc``), so the proxy can run over a LiveKit SFU (LiveKit Cloud
or self-hosted) instead of aiortc P2P — which gives built-in signaling + NAT
traversal/TURN + scale, at the cost of running/using a LiveKit server (see DESIGN §11).

⚠️ This is a SKELETON. It follows the SDK API but has NOT been run against a real
LiveKit server. Spots that need verification against your SDK version are marked
``# VERIFY``. Map of concepts:

    our channel label  ->  LiveKit data **topic** (publish_data/on data_received)
    reliable bool      ->  publish_data(reliable=...)
    video stream       ->  a published/subscribed video track
    session_id         ->  room name (in the JWT)
    our Signaling      ->  unused (LiveKit does its own signaling; pass url+token)

The biggest open detail (``# VERIFY``) is **carrying the capture seq with each video
frame** — LiveKit frames don't carry our app pts. This scaffold stamps the seq into the
frame ``timestamp_us`` and recovers it on the far side; if your SDK doesn't preserve
that, fall back to sending ``{seq}`` on a reliable data topic alongside each frame.

Install: ``uv pip install livekit`` (or add the ``webrtc-livekit`` extra).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import numpy as np

from .transport import Channel, Transport

logger = logging.getLogger(__name__)


class _LiveKitChannel(Channel):
    """A named data pipe backed by a LiveKit data **topic**."""

    def __init__(self, transport: "LiveKitTransport", topic: str, reliable: bool) -> None:
        self._t = transport
        self._topic = topic
        self._reliable = reliable
        self._cb: Callable[[str], None] | None = None

    def send(self, data: str) -> None:
        room = self._t.room
        if room is None or not self._t.connected.is_set():
            return  # best-effort, like the aiortc backend
        # publish_data is async; fire-and-forget on the room's loop.
        asyncio.ensure_future(
            room.local_participant.publish_data(data.encode("utf-8"), reliable=self._reliable, topic=self._topic)
        )

    def on_message(self, callback: Callable[[str], None]) -> None:
        self._cb = callback

    def _dispatch(self, raw: bytes) -> None:
        if self._cb is not None:
            self._cb(raw.decode("utf-8"))

    @property
    def is_open(self) -> bool:
        return self._t.connected.is_set()


class LiveKitTransport(Transport):
    """Transport over a LiveKit room. EXPERIMENTAL — see module docstring."""

    def __init__(
        self,
        *,
        role: str,  # "publisher" (publishes video) | "subscriber" (subscribes)
        channels: dict[str, bool],  # label -> reliable
        url: str,
        token: str,
    ) -> None:
        super().__init__()
        try:
            from livekit import rtc
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "transport_backend='livekit' needs the LiveKit SDK: `uv pip install livekit` "
                "(or the lerobot[webrtc-livekit] extra)."
            ) from e
        self._rtc = rtc
        if role not in ("publisher", "subscriber"):
            raise ValueError(f"role must be 'publisher' or 'subscriber', got {role!r}")
        self.role = role
        self._url = url
        self._token = token
        self.room = rtc.Room()
        self._channels = {label: _LiveKitChannel(self, label, reliable) for label, reliable in channels.items()}
        self._frame_cb: Callable[[int, np.ndarray], None] | None = None
        self._video_source = None  # created lazily on first send_frame (needs frame dims)
        self._register()

    def _register(self) -> None:
        rtc = self._rtc

        @self.room.on("connected")
        def _on_connected() -> None:
            logger.info("livekit: room connected")
            self.connected.set()

        @self.room.on("disconnected")
        def _on_disconnected(*_args) -> None:
            logger.info("livekit: room disconnected")
            self.closed.set()

        @self.room.on("data_received")
        def _on_data(packet) -> None:  # rtc.DataPacket  # VERIFY: field names below
            ch = self._channels.get(getattr(packet, "topic", ""))
            if ch is not None:
                ch._dispatch(packet.data)

        if self.role == "subscriber":

            @self.room.on("track_subscribed")
            def _on_track(track, publication, participant) -> None:  # noqa: ANN001
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    asyncio.ensure_future(self._consume(track))

    async def _consume(self, track) -> None:  # noqa: ANN001
        rtc = self._rtc
        stream = rtc.VideoStream(track)
        async for event in stream:  # event.frame: rtc.VideoFrame
            frame = event.frame
            # VERIFY: recover the capture seq stamped at publish time.
            seq = int(round(getattr(event, "timestamp_us", 0) / 1000.0))
            rgba = frame.convert(rtc.VideoBufferType.RGBA)
            arr = np.frombuffer(rgba.data, dtype=np.uint8).reshape(frame.height, frame.width, 4)
            if self._frame_cb is not None:
                self._frame_cb(seq, np.ascontiguousarray(arr[:, :, :3]))

    async def open(self, signaling=None) -> None:  # noqa: ANN001 - signaling unused for LiveKit
        await self.room.connect(self._url, self._token)
        # publish_data/track work after connect; the video track is published lazily on
        # the first send_frame so we know the frame dimensions.

    def channel(self, label: str) -> Channel:
        return self._channels[label]

    def send_frame(self, seq: int, img: np.ndarray) -> None:
        if self.role != "publisher":
            return
        rtc = self._rtc
        h, w = img.shape[:2]
        if self._video_source is None:
            self._video_source = rtc.VideoSource(w, h)
            track = rtc.LocalVideoTrack.create_video_track("camera", self._video_source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            asyncio.ensure_future(self.room.local_participant.publish_track(track, options))
        rgba = np.dstack([img, np.full((h, w, 1), 255, np.uint8)])  # RGB -> RGBA
        # VERIFY: stamp the seq so the far side can pair frame<->state by seq.
        frame = rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba.tobytes())
        self._video_source.capture_frame(frame, timestamp_us=seq * 1000)  # VERIFY: kwarg name

    def set_frame_handler(self, callback: Callable[[int, np.ndarray], None]) -> None:
        self._frame_cb = callback

    async def close(self) -> None:
        self.closed.set()
        await self.room.disconnect()
