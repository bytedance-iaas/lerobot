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

Two real-time DataChannels + one control DataChannel + one video media track:

    label        dir          reliability                payload
    -----------  -----------  -------------------------  ---------------------------
    state        Mac -> cloud unreliable (rt)            StateMsg: joints + capture ts + seq
    action       cloud -> Mac unreliable (rt)            ActionMsg: goal joints + seq + obs_seq
    control      both         ordered/reliable           onboarding RPC (device discovery, plan)
    <video>      Mac -> cloud media track (H.264/VP8)    one camera; seq carried in pts

The frame's capture **seq** rides its ``pts`` (``pts = seq * VIDEO_PTS_PER_SEQ``); the
cloud recovers ``seq = round(pts / VIDEO_PTS_PER_SEQ)`` and pairs each frame to the
StateMsg with the same seq (both are produced by one ``robot.get_observation()``, so
they share a seq). A dropped frame just leaves a gap in seq — no cascade. See
``DESIGN.md`` §5.1 for the re-basing caveat and the RTP-header-extension end state.

All timestamps are the Mac's ``time.monotonic()`` seconds (a single clock — the
cloud only ever *compares* them, never interprets them as wall time)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

# DataChannel labels (must match on both ends).
CH_STATE = "state"
CH_ACTION = "action"
# Control plane: cloud-driven onboarding RPC (device discovery, calibrate, ...).
# Ordered + reliable — these are one-shot commands, not the realtime control loop.
CH_CONTROL = "control"

# The capture seq is carried in each frame's pts: pts = seq * VIDEO_PTS_PER_SEQ at a
# 90 kHz clock. The cloud recovers seq = round(pts / VIDEO_PTS_PER_SEQ) and pairs the
# frame to the state with the same seq. pts survives VP8/H.264; a dropped frame just
# leaves a gap in seq (no cascade), unlike an ordered 1:1 side channel.
# Caveat: the receiver re-bases the *first received* frame to pts=0, so seq is recovered
# relative to the first received frame. The daemon resets seq to 0 per session, so as long
# as the first frame lands (true at session start) relative == absolute. Production should
# carry an absolute seq in an RTP header extension to be robust to initial-frame loss.
VIDEO_CLOCK_RATE = 90_000
VIDEO_PTS_PER_SEQ = 3_000  # 1/30 s per seq at 90 kHz; round(pts/this) recovers seq

# Unreliable, real-time DataChannel options for state/action (drop, don't retransmit).
RT_CHANNEL_KWARGS: dict[str, Any] = {"ordered": False, "maxRetransmits": 0}
# Control commands must arrive in order and not be dropped.
ORDERED_CHANNEL_KWARGS: dict[str, Any] = {"ordered": True}


@dataclass(frozen=True)
class StateMsg:
    """Proprioceptive sample captured on the Mac. Sent on ``CH_STATE``."""

    t: float  # capture time.monotonic() seconds
    seq: int  # capture sequence number (shared with the paired frame)
    joints: dict[str, float]  # {"<motor>.pos": degrees}
    # Piggybacked closed-loop feedback: the most recent action this Mac applied.
    # Lets the cloud confirm landing + measure round-trip without an extra channel.
    applied_seq: int = -1  # ActionMsg.seq of the last applied action (-1 = none yet)
    applied_t: float = 0.0  # Mac time.monotonic() when it was applied

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> StateMsg:
        d = json.loads(raw)
        return cls(
            t=float(d["t"]),
            seq=int(d["seq"]),
            joints={k: float(v) for k, v in d["joints"].items()},
            applied_seq=int(d.get("applied_seq", -1)),
            applied_t=float(d.get("applied_t", 0.0)),
        )


@dataclass(frozen=True)
class ActionMsg:
    """Goal joint command issued by the cloud. Sent on ``CH_ACTION``."""

    t: float  # cloud send time.monotonic() seconds (debug/telemetry only)
    seq: int  # monotonically increasing action id
    goal: dict[str, float]  # {"<motor>.pos": degrees}
    obs_seq: int = -1  # provenance: the StateMsg.seq of the obs this action was derived from

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> ActionMsg:
        d = json.loads(raw)
        return cls(
            t=float(d["t"]),
            seq=int(d["seq"]),
            goal={k: float(v) for k, v in d["goal"].items()},
            obs_seq=int(d.get("obs_seq", -1)),
        )


@dataclass(frozen=True)
class RpcRequest:
    """Control-plane request: cloud -> Mac. Sent on ``CH_CONTROL``."""

    id: int  # client-assigned, echoed in the matching RpcResponse
    method: str  # e.g. "list_ports", "find_port_begin", "list_cameras"
    params: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"id": self.id, "method": self.method, "params": self.params}, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> RpcRequest:
        d = json.loads(raw)
        return cls(id=int(d["id"]), method=str(d["method"]), params=dict(d.get("params") or {}))


@dataclass(frozen=True)
class RpcResponse:
    """Control-plane response: Mac -> cloud. Sent on ``CH_CONTROL``."""

    id: int
    ok: bool
    result: Any | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "ok": self.ok, "result": self.result, "error": self.error},
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> RpcResponse:
        d = json.loads(raw)
        return cls(id=int(d["id"]), ok=bool(d["ok"]), result=d.get("result"), error=d.get("error"))
