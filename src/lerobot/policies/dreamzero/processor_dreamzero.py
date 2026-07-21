#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Pre/post processing for the DreamZero policy.

The preprocessor turns a LeRobot observation into DreamZero's action-input dict
(`images` / `text` / `text_attention_mask` / `text_negative` / `text_attention_mask_negative`
/ `state` / `embodiment_id`); the postprocessor unnormalizes the predicted chunk and converts
relative joint deltas back to absolute.

Faithfully reproducing DreamZero's `ComposedModalityTransform` — q99 normalization to [-1, 1],
per-embodiment view stitching + language templates, the umt5 tokenizer, state padding, and the
relative-action decode — is the correctness crux of the port. Because the exact statistics and
tokenizer identity come from the released checkpoint, these two steps are validated against
golden samples from the upstream dataloader (see dreamzero_port_plan.md, milestone M2) before
they are considered done. Until then they raise `NotImplementedError` rather than silently
applying an unvalidated transform.
"""

from typing import Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_dreamzero import DreamZeroConfig

_VALIDATION_MSG = (
    "The DreamZero {step} step is pending golden-sample validation against the upstream "
    "DreamZero dataloader/checkpoint (dreamzero_port_plan.md, milestone M2). It is intentionally "
    "not yet implemented so that an unvalidated normalization/tokenization transform is never "
    "applied silently. Implement and validate it before running DreamZero train/eval end-to-end."
)


@ProcessorStepRegistry.register("dreamzero_pack_inputs")
class DreamZeroPackInputsStep(ProcessorStep):
    """Pack a LeRobot observation into DreamZero's action-input dict.

    Responsibilities (reproduce DreamZero's ComposedModalityTransform inference path):
      - per-embodiment multi-view stitching + resize to the model resolution,
      - language templating + fixed CFG negative prompt, umt5 tokenization,
      - state padding to ``max_state_dim`` and q99 normalization to [-1, 1],
      - inject ``embodiment_id`` (projector index).
    """

    def __init__(self, config: DreamZeroConfig, stats: dict | None = None):
        self.config = config
        self.stats = stats

    def __call__(self, transition):
        raise NotImplementedError(_VALIDATION_MSG.format(step="input pack"))

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("dreamzero_action_decode")
class DreamZeroActionDecodeStep(ProcessorStep):
    """Unnormalize the predicted action chunk and convert relative joints back to absolute.

    Inverts the q99 normalization applied in :class:`DreamZeroPackInputsStep` and, for
    ``relative_action_keys`` (DROID: ``joint_position``), adds back the last observed state —
    mirroring ``GrootSimPolicy.unapply``.
    """

    def __init__(self, config: DreamZeroConfig, stats: dict | None = None):
        self.config = config
        self.stats = stats

    def __call__(self, transition):
        raise NotImplementedError(_VALIDATION_MSG.format(step="action decode"))

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        return features


def make_dreamzero_pre_post_processors(
    config: DreamZeroConfig,
    dataset_stats: dict | None = None,
    dataset_meta=None,
    **kwargs,
):
    """Build the DreamZero (preprocessor, postprocessor) pipeline pair.

    The pipeline skeleton (rename → add batch dim → pack → device; decode → device) matches the
    GR00T policy's structure. The two DreamZero-specific steps carry the normalization/tokenizer
    logic and are pending golden-sample validation (see module docstring).
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DreamZeroPackInputsStep(config=config, stats=dataset_stats),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        DreamZeroActionDecodeStep(config=config, stats=dataset_stats),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
