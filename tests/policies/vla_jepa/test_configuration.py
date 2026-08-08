#!/usr/bin/env python

from __future__ import annotations

import pytest
from conftest import ACTION_DIM, ACTION_HORIZON, IMAGE_SIZE, NUM_VIDEO_FRAMES, STATE_DIM, make_config

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_delta_indices() -> None:
    config = make_config()
    assert config.observation_delta_indices == list(range(NUM_VIDEO_FRAMES))
    assert config.action_delta_indices == list(range(ACTION_HORIZON))


def test_n_action_steps_exceeds_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="n_action_steps"):
        VLAJEPAConfig(chunk_size=4, n_action_steps=8)


def test_too_few_video_frames_raises() -> None:
    with pytest.raises(ValueError, match="video_horizon"):
        VLAJEPAConfig(
            chunk_size=16,
            n_action_steps=16,
            num_video_frames=2,
            jepa_tubelet_size=2,  # needs >= 4 frames (2 for current, 2 for future) to have a window of size > 0
        )


def test_validate_features_no_image_raises() -> None:
    config = VLAJEPAConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    with pytest.raises(ValueError, match="at least one visual input feature"):
        config.validate_features()


def test_validate_features_no_action_raises() -> None:
    config = VLAJEPAConfig(
        input_features={
            f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
        },
        output_features={},
    )
    with pytest.raises(ValueError, match="action output feature"):
        config.validate_features()


def test_validate_features_sets_action_dim_from_feature() -> None:
    config = make_config(action_dim=6, state_dim=10)
    assert config.action_dim == 6
    assert config.state_dim == 10


# ---------------------------------------------------------------------------
# World-model prediction-error feedback flags
# ---------------------------------------------------------------------------


def test_wm_feedback_defaults_to_disabled() -> None:
    config = make_config()
    assert config.enable_wm_feedback is False
    assert config.wm_error_threshold is None
    assert config.wm_replan_on_surprise is True


def test_wm_feedback_requires_world_model() -> None:
    with pytest.raises(ValueError, match="enable_wm_feedback requires enable_world_model"):
        make_config(enable_wm_feedback=True, enable_world_model=False)


def test_wm_error_threshold_requires_feedback_enabled() -> None:
    with pytest.raises(ValueError, match="wm_error_threshold requires enable_wm_feedback"):
        make_config(enable_wm_feedback=False, wm_error_threshold=0.5)


def test_wm_feedback_enabled_config_is_valid() -> None:
    config = make_config(enable_wm_feedback=True, wm_error_threshold=0.5)
    assert config.enable_wm_feedback is True
    assert config.wm_error_threshold == 0.5
