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

# Ported from NVIDIA DreamZero: groot/vla/model/n1_5/action_head/base_action_head.py

from abc import ABC, abstractmethod

from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class ActionHead(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        pass

    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_action_samples: int = 1,
        inference_batch_size: int = 32,
    ) -> BatchFeature:
        # Used for predicting actions during inference
        # By default, the action head does the same thing as a normal forward pass
        return self.forward(backbone_output, action_input)

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass

    def set_override_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            setattr(self, key, value)
