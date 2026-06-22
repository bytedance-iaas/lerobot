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

"""Wire protocol shared by the Mac-side capture agent and the cloud-side proxy.

Three logical DataChannels plus one video media track connect the two ends:

    label        dir          reliability                payload
    -----------  -----------  -------------------------  ---------------------------
    state        Mac -> cloud unreliable (rt)            StateMsg: joints + capture ts
    framemeta    Mac -> cloud ordered/reliable           FrameMetaMsg: {seq, t} per frame
    action       cloud -> Mac unreliable (rt)            ActionMsg: goal joints + seq
    <video>      Mac -> cloud media track (H.264/VP8)    one camera, pixels only

Why ``framemeta`` is its own *ordered* channel: a decoded ``av.VideoFrame`` carries
no application-level capture timestamp that survives RTP re-stamping, so the Mac
sends a parallel ``{seq, t}`` record per emitted frame. The cloud pops one framemeta
per received frame to tag it with its Mac-side ``time.monotonic()`` capture time.
This 1:1 ordered pop is a *prototype* simplification valid only on a lossless link
(loopback). Production must carry ``seq`` in an RTP header extension or in-pixel —
see ``README.md`` "Known limitations".

All timestamps are the Mac's ``time.monotonic()`` seconds (a single clock — the
cloud only ever *compares* them, never interprets them as wall time)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

# DataChannel labels (must match on both ends).
CH_STATE = "state"
CH_FRAMEMETA = "framemeta"
CH_ACTION = "action"

# Video clock used when stamping sender-side frames. Not relied on for capture-time
# recovery across the wire (see module docstring) — kept standard for sane RTP output.
VIDEO_CLOCK_RATE = 90_000

# Unreliable, real-time DataChannel options for state/action (drop, don't retransmit).
RT_CHANNEL_KWARGS: dict[str, Any] = {"ordered": False, "maxRetransmits": 0}
# framemeta must arrive in order and not be dropped (it is the frame<->time index).
ORDERED_CHANNEL_KWARGS: dict[str, Any] = {"ordered": True}


@dataclass(frozen=True)
class StateMsg:
    """Proprioceptive sample captured on the Mac. Sent on ``CH_STATE``."""

    t: float  # capture time.monotonic() seconds
    seq: int  # capture sequence number (shared with the paired frame)
    joints: dict[str, float]  # {"<motor>.pos": degrees}

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> StateMsg:
        d = json.loads(raw)
        return cls(t=float(d["t"]), seq=int(d["seq"]), joints={k: float(v) for k, v in d["joints"].items()})


@dataclass(frozen=True)
class FrameMetaMsg:
    """Capture metadata for one video frame. Sent on ``CH_FRAMEMETA`` (ordered)."""

    t: float  # capture time.monotonic() seconds of this frame
    seq: int  # capture sequence number

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> FrameMetaMsg:
        d = json.loads(raw)
        return cls(t=float(d["t"]), seq=int(d["seq"]))


@dataclass(frozen=True)
class ActionMsg:
    """Goal joint command issued by the cloud. Sent on ``CH_ACTION``."""

    t: float  # cloud send time.monotonic() seconds (debug/telemetry only)
    seq: int  # monotonically increasing action id
    goal: dict[str, float]  # {"<motor>.pos": degrees}

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> ActionMsg:
        d = json.loads(raw)
        return cls(t=float(d["t"]), seq=int(d["seq"]), goal={k: float(v) for k, v in d["goal"].items()})
