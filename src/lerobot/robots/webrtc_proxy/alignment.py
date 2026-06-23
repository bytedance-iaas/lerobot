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

"""Capture-timestamp alignment buffer (handoff 难点 A).

The cloud receives proprioceptive state and camera frames on *separate* channels
that traverse the public internet with independent jitter. Pairing them by
*arrival* time would let network jitter corrupt the temporal ordering of a
recorded dataset. Instead we pair by the Mac-side *capture* timestamp carried with
each sample, so public-net jitter only adds latency — never reorders.

``AlignmentBuffer`` keeps a short bounded history of state and frame samples and,
on demand, returns the most recent state paired with the camera frame whose
capture timestamp is nearest to that state's. It is deliberately tiny and
thread-safe via a single lock: producers (asyncio receive callbacks) push from one
thread, the consumer (``get_observation``) pulls from another.
"""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedObs:
    """Result of pairing one state sample with its nearest-in-time camera frame."""

    t_state: float
    joints: dict[str, float]
    frame: np.ndarray | None  # HxWx3 uint8 RGB, or None if no frame buffered yet
    t_frame: float | None
    seq_state: int

    @property
    def skew_ms(self) -> float | None:
        """Absolute capture-time gap between the paired state and frame, in ms."""
        if self.t_frame is None:
            return None
        return abs(self.t_state - self.t_frame) * 1e3


class AlignmentBuffer:
    """Bounded, thread-safe nearest-neighbour pairing of state and frames by capture ts."""

    def __init__(self, maxlen: int = 64, pair_tolerance_s: float = 0.1) -> None:
        # Each entry is (capture_t, value); appended in capture order == time order, so
        # the timestamps stay sorted and bisect (keyed on the timestamp) can binary-search.
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._states: deque[tuple[float, dict[str, float], int]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add_state(self, t: float, joints: dict[str, float], seq: int) -> None:
        with self._lock:
            self._states.append((t, dict(joints), seq))

    def add_frame(self, t: float, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append((t, frame))

    def _nearest_frame_locked(self, t: float) -> tuple[float | None, np.ndarray | None]:
        # bisect by the timestamp key only (never compares the ndarray frame).
        i = bisect.bisect_left(self._frames, t, key=lambda pair: pair[0])
        best: tuple[float, np.ndarray] | None = None
        best_dt: float | None = None
        for j in (i - 1, i):  # the two frames straddling t
            if 0 <= j < len(self._frames):
                ft, fv = self._frames[j]
                dt = abs(ft - t)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, (ft, fv)
        return best if best is not None else (None, None)

    def assemble(self) -> AlignedObs | None:
        """Pair the newest state with its nearest frame. None if no state yet."""
        with self._lock:
            if not self._states:
                return None
            t_state, joints, seq = self._states[-1]
            t_frame, frame = self._nearest_frame_locked(t_state)
            return AlignedObs(
                t_state=t_state,
                joints=dict(joints),
                frame=frame,
                t_frame=t_frame,
                seq_state=seq,
            )

    def has_state(self) -> bool:
        with self._lock:
            return bool(self._states)
