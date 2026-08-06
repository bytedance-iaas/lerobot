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

"""Embodiments other than DROID: the generic canvas, its prompt, and the frame-rate stretch.

DROID has a bespoke three-view layout; everything else goes through upstream's generic 2x2 grid
with the spare quadrants blacked out. Two things about that are easy to get silently wrong — the
prompt describing a layout the canvas does not have, and a dataset recorded at a different frame
rate handing the model a window of the wrong duration.
"""

import pytest
import torch

pytest.importorskip("diffusers")  # dreamzero config imports through the policy package

from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig  # noqa: E402
from lerobot.policies.dreamzero.processor_dreamzero import (  # noqa: E402
    DreamZeroPackInputsStep,
    _stitch_generic,
)

_SO100_VIEWS = ("observation.images.front", "observation.images.wrist")


def _so100_config(**overrides) -> DreamZeroConfig:
    return DreamZeroConfig(embodiment_tag="so100", video_modality_keys=_SO100_VIEWS, **overrides)


def _view(cfg, fill: int) -> torch.Tensor:
    return torch.full((1, 1, cfg.per_view_height, cfg.per_view_width, 3), fill, dtype=torch.uint8)


def test_generic_stitch_fills_the_left_column_and_blacks_the_rest():
    """Two cameras occupy top-left and bottom-left; the right column stays black.

    Upstream's slot order is top-left, bottom-left, top-right — not raster order — so a view in
    the wrong quadrant would contradict the prompt without changing the canvas shape.
    """
    cfg = _so100_config()
    h, w = cfg.per_view_height, cfg.per_view_width

    canvas = _stitch_generic([_view(cfg, 11), _view(cfg, 22)], h, w)[0, 0, :, :, 0]

    assert canvas.shape == (2 * h, 2 * w)  # pinned by frame_seqlen; cannot shrink for fewer views
    assert canvas[0, 0] == 11  # top-left
    assert canvas[h, 0] == 22  # bottom-left
    assert canvas[0, w] == 0 and canvas[h, w] == 0  # right column black


def test_generic_stitch_uses_the_third_slot_before_the_fourth():
    cfg = _so100_config()
    h, w = cfg.per_view_height, cfg.per_view_width

    canvas = _stitch_generic([_view(cfg, 1), _view(cfg, 2), _view(cfg, 3)], h, w)[0, 0, :, :, 0]

    assert canvas[0, w] == 3  # top-right
    assert canvas[h, w] == 0  # bottom-right is the one left black


@pytest.mark.parametrize("count", [0, 5])
def test_generic_stitch_rejects_an_unfillable_view_count(count):
    cfg = _so100_config()
    views = [_view(cfg, 1)] * count

    with pytest.raises(ValueError, match="holds 1-4 views"):
        _stitch_generic(views, cfg.per_view_height, cfg.per_view_width)


def test_prompt_describes_the_quadrants_the_canvas_actually_fills():
    """The prompt is the only signal telling the model which quadrant holds which camera."""
    step = DreamZeroPackInputsStep(_so100_config())

    prompt = step._build_language("pick up the red cube", 1)[0]

    assert "The top-left view shows the camera view from the front" in prompt
    assert "the bottom-left view shows the camera view from the wrist" in prompt
    assert "the top-right and bottom-right views are black screens" in prompt
    # Upstream's frame: the instruction opens and closes the prompt.
    assert prompt.startswith("A multi-view video shows that a robot pick up the red cube")
    assert prompt.endswith("The robot pick up the red cube")


def test_prompt_prefers_explicit_view_descriptions():
    step = DreamZeroPackInputsStep(_so100_config(view_descriptions=("overhead camera", "robot's wrist")))

    prompt = step._build_language("stack the blocks", 1)[0]

    assert "from the overhead camera" in prompt
    assert "from the robot's wrist" in prompt


def test_prompt_refuses_to_guess_the_layout():
    """Without view names the prompt would describe quadrants it cannot know."""
    step = DreamZeroPackInputsStep(DreamZeroConfig(embodiment_tag="so100"))

    with pytest.raises(ValueError, match="which camera is in which quadrant"):
        step._build_language("do something", 1)


def test_prompt_rejects_a_name_count_that_disagrees_with_the_canvas():
    """One camera name for a two-view canvas would call an occupied quadrant a black screen."""
    step = DreamZeroPackInputsStep(DreamZeroConfig(embodiment_tag="so100", view_descriptions=("front",)))

    with pytest.raises(ValueError, match="but the canvas holds 2 view"):
        step._build_language("do something", 1, n_views=2)


def test_a_configured_camera_missing_from_the_observation_raises():
    """Silently dropping a configured view would stitch a canvas the prompt no longer describes."""
    step = DreamZeroPackInputsStep(_so100_config())
    obs = {"observation.images.front": torch.zeros(1, 3, 8, 8)}  # wrist is missing

    with pytest.raises(KeyError, match="observation.images.wrist"):
        step._ordered_image_keys(obs)


def test_droid_keeps_its_own_prompt():
    step = DreamZeroPackInputsStep(DreamZeroConfig(embodiment_tag="oxe_droid"))

    prompt = step._build_language("open the drawer", 1)[0]

    assert "split into three views" in prompt


@pytest.mark.parametrize(("fps", "multiplier"), [(None, 1), (15, 1), (30, 2), (60, 4)])
def test_the_window_covers_the_same_wall_clock_at_any_frame_rate(fps, multiplier):
    """DreamZero trained on 6.4 s windows; a 30 fps dataset would otherwise supply 3.2 s ones."""
    cfg = _so100_config(source_fps=fps)

    assert cfg.frame_rate_multiplier == multiplier
    # Same window contract: 33 observation rows, 96 actions — just spread over more raw frames.
    assert len(cfg.observation_delta_indices) == cfg.num_frames == 33
    assert len(cfg.action_delta_indices) == 96
    assert cfg.training_span == 96 * multiplier
    assert cfg.drop_n_last_frames == cfg.training_span
    # 6.4 s at the dataset's own frame rate.
    assert cfg.training_span / (fps or 15) == pytest.approx(6.4)


def test_the_block_anchors_stay_put_when_the_window_is_stretched():
    """The anchors are observation ROWS, so the stretch must cancel out of them."""
    fast = DreamZeroPackInputsStep(_so100_config(source_fps=30))
    slow = DreamZeroPackInputsStep(_so100_config(source_fps=15))

    assert fast._block_anchor_rows == slow._block_anchor_rows == [0, 8, 16, 24]


def test_a_frame_rate_below_the_model_is_rejected():
    """Striding cannot lengthen a window that is already too short in wall-clock terms."""
    cfg = _so100_config(source_fps=5)

    with pytest.raises(ValueError, match="below DreamZero's 15 fps"):
        _ = cfg.frame_rate_multiplier
