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

"""Guards on the analytic FLOP model: `PI05Pytorch.estimate_flops` and `benchmarks/mfu.py`.

Not accuracy tests -- accuracy is established by agreeing with a profile of a real run,
which needs hardware. These pin what can rot silently: the peak constant that lives in two
modules, the forward/backward factors, and the scaling laws the formula must obey.

Every model here is built on the meta device: parameter *counts* do not need parameter
*values*, so no checkpoint and no allocation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.utils.profiler_summary import DEFAULT_PEAK_TFLOPS

# benchmarks/ is not a package, so load the file directly. It must be in sys.modules
# BEFORE exec_module: @dataclass resolves its own module via sys.modules[cls.__module__].
_PATH = Path(__file__).parents[2] / "benchmarks" / "mfu.py"
_spec = importlib.util.spec_from_file_location("_mfu", _PATH)
mfu = importlib.util.module_from_spec(_spec)
sys.modules["_mfu"] = mfu
_spec.loader.exec_module(mfu)


def build(**cfg_kwargs) -> PI05Pytorch:
    with torch.device("meta"):
        return PI05Pytorch(PI05Config(**cfg_kwargs))


@pytest.fixture(scope="module")
def model() -> PI05Pytorch:
    return build()


# --- the peak table --------------------------------------------------------------------


def test_peak_matches_profiler_summary():
    """One number, two modules. An MFU that disagrees with itself is worse than none."""
    assert mfu.PEAK_TFLOPS["ascend950pr"]["bfloat16"] == DEFAULT_PEAK_TFLOPS


def test_only_measured_dtypes_are_listed():
    """A guessed peak silently rescales every MFU printed; the table must stay honest."""
    assert set(mfu.PEAK_TFLOPS["ascend950pr"]) == {"bfloat16"}


def test_mm_is_two_flops_per_mac():
    assert mfu.mm(2, 3, 4) == 2 * 2 * 3 * 4


# --- structural parameter counting -----------------------------------------------------


def test_matmul_params_counts_linear_and_conv_only():
    m = nn.Sequential(
        nn.Linear(4, 8, bias=True),  # 32 weights; the 8 biases are adds, not MACs
        nn.Conv2d(3, 6, kernel_size=2),  # 6*3*2*2 = 72
        nn.Embedding(100, 16),  # a lookup: must NOT count
        nn.LayerNorm(8),  # elementwise: must NOT count
    )
    assert PI05Pytorch._matmul_params(m) == 32 + 72


def test_matmul_params_is_nonzero_for_pi05(model):
    assert PI05Pytorch._matmul_params(model.paligemma_with_expert) > 1e9


# --- forward/backward factors, read off requires_grad ----------------------------------


def test_backward_is_twice_forward_when_everything_trains(model):
    """Full fine-tune: every tower pays fwd + dX + dW, so total == 3x forward."""
    parts = model.estimate_flops()
    bare = build()
    # Recompute with every multiplier forced to 1 by freezing nothing: the ratio of the
    # all-trainable total to the forward-only total must be exactly 3.
    for p in bare.parameters():
        p.requires_grad_(False)
    frozen = bare.estimate_flops()
    assert parts["total"] == pytest.approx(3.0 * frozen["total"])


def test_freezing_the_vision_encoder_makes_it_forward_only():
    """Nothing upstream of the tower trains, so even dX is unnecessary."""
    full = build().estimate_flops()
    frozen = build(freeze_vision_encoder=True).estimate_flops()
    key = "vision tower x1"
    assert frozen[key] == pytest.approx(full[key] / 3.0)
    # The VLM still trains, so it is untouched.
    assert frozen["VLM"] == pytest.approx(full["VLM"])


def test_train_expert_only_freezes_the_vlm_too():
    full = build().estimate_flops()
    expert_only = build(train_expert_only=True).estimate_flops()
    assert expert_only["VLM"] == pytest.approx(full["VLM"] / 3.0)
    assert expert_only["action expert"] == pytest.approx(full["action expert"])
    assert expert_only["total"] < full["total"]


# --- scaling laws the formula must obey ------------------------------------------------


def test_each_camera_costs_more_than_the_last(model):
    """An image adds a tower pass AND lengthens the prefix the VLM attends over."""
    totals = [model.estimate_flops(n_images=n)["total"] for n in (1, 2, 3)]
    assert totals[1] - totals[0] < totals[2] - totals[1]


def test_attention_is_quadratic_in_prefix_length(model):
    short, mid, long = (model.estimate_flops(text_tokens=t)["total"] for t in (100, 200, 400))
    assert (long - mid) > 2 * (mid - short)


def test_recompute_is_not_counted():
    """Gradient checkpointing is hardware work; MFU's numerator must not move."""
    m = build()
    before = m.estimate_flops()["total"]
    m.gradient_checkpointing_enable()
    assert m.estimate_flops()["total"] == before
