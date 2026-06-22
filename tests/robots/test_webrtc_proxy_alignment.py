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

"""Unit tests for the capture-timestamp alignment buffer (no aiortc needed)."""

import numpy as np

from lerobot.robots.webrtc_proxy.alignment import AlignmentBuffer


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_empty_buffer_assembles_to_none():
    buf = AlignmentBuffer()
    assert buf.assemble() is None
    assert not buf.has_state()


def test_state_without_frame_returns_obs_with_none_frame():
    buf = AlignmentBuffer()
    buf.add_state(t=1.0, joints={"shoulder_pan.pos": 10.0}, seq=0)
    aligned = buf.assemble()
    assert aligned is not None
    assert aligned.frame is None
    assert aligned.skew_ms is None
    assert aligned.joints == {"shoulder_pan.pos": 10.0}


def test_pairs_with_nearest_frame_by_capture_timestamp():
    buf = AlignmentBuffer()
    # Three frames at t=1.00, 1.10, 1.20 with distinguishable contents.
    buf.add_frame(1.00, _frame(1))
    buf.add_frame(1.10, _frame(2))
    buf.add_frame(1.20, _frame(3))
    # Newest state at t=1.12 -> nearest frame is the t=1.10 one (frame value 2).
    buf.add_state(t=1.12, joints={"a.pos": 0.0}, seq=7)

    aligned = buf.assemble()
    assert aligned is not None
    assert aligned.seq_state == 7
    assert int(aligned.frame[0, 0, 0]) == 2
    assert aligned.t_frame == 1.10
    # skew = |1.12 - 1.10| = 20ms
    assert abs(aligned.skew_ms - 20.0) < 1e-6


def test_assemble_uses_newest_state():
    buf = AlignmentBuffer()
    buf.add_frame(2.0, _frame(9))
    buf.add_state(t=1.0, joints={"a.pos": 1.0}, seq=0)
    buf.add_state(t=2.0, joints={"a.pos": 2.0}, seq=1)
    aligned = buf.assemble()
    assert aligned.seq_state == 1
    assert aligned.joints == {"a.pos": 2.0}
    assert aligned.skew_ms == 0.0


def test_history_is_bounded():
    buf = AlignmentBuffer(maxlen=2)
    for i in range(5):
        buf.add_frame(float(i), _frame(i))
        buf.add_state(t=float(i), joints={"a.pos": float(i)}, seq=i)
    # Only the last two states/frames survive; assemble pairs newest state (t=4) -> frame t=4.
    aligned = buf.assemble()
    assert aligned.seq_state == 4
    assert int(aligned.frame[0, 0, 0]) == 4
