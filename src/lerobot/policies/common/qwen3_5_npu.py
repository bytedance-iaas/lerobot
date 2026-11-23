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

"""Keep Qwen3.5's gated delta rule off ``linalg.solve_triangular`` on an Ascend device.

``torch_chunk_gated_delta_rule`` condenses a chunk of delta-rule updates through a UT
transform, and solves that unit lower triangular system twice per layer -- once for the new
values, once for the decayed keys. CANN backs those solves with ``aclnnTriangularSolve``,
which is very slow at the shapes a policy actually uses: measured on a 950PR at G0.5's
``[2, 16, 9, 32, 32]`` system with ``[..., 32, 128]`` right-hand sides, one layer's pair of
solves costs 18.7 ms. Across 18 linear-attention layers that is 336 ms, and an Ascend
profile of a G0.5 training step attributes 54.2% of all device time to that one kernel --
time the MFU numerator never sees, since a solve counts no matmul or convolution FLOPs.

Upstream already carries the cure. Export targets that lack a triangular solver take a
second path through the same function, which builds the inverse by forward substitution and
then applies it with two matmuls. That path is also structurally cheaper here: it inverts
once and reuses the result for both right-hand sides, where the solver solves twice. On the
same shapes it runs in 1.05 ms -- 17.8x faster -- and agrees with the solver to a relative
norm of 7.2e-08, which is fp32 round-off.

Choosing it needs no rewritten math, because ``is_torchdynamo_exporting`` is read exactly
once in that module, at this branch. Forcing it true selects upstream's own code and changes
nothing else. The override engages only for Ascend; CUDA and CPU keep the solver, which is
fast there.

Usage::

    patch_qwen3_5_delta_rule(device)
"""

from __future__ import annotations

import torch

_FLAG = "_lerobot_npu_delta_rule"


def patch_qwen3_5_delta_rule(device: torch.device | str) -> None:
    """Route Qwen3.5's gated delta rule through forward substitution on an Ascend device.

    A no-op off NPU, when transformers is not installed, and when already applied. ``device``
    is the device the model will run on, not the device of any particular tensor: the branch
    this selects is chosen once per call to the delta rule, with no tensor in hand to ask.
    """
    if torch.device(device).type != "npu":
        return

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    except ImportError:
        return

    if getattr(mod, _FLAG, False):
        return

    # Module-level rebinding, not a patch of `transformers.utils`: the name is imported into
    # this module's namespace, so every other module keeps the real predicate.
    mod.is_torchdynamo_exporting = lambda: True
    setattr(mod, _FLAG, True)
