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

"""DreamZero policy wrapper for LeRobot.

Wraps DreamZero's `WANPolicyHead` (Wan-video joint video+action causal DiT) behind the
LeRobot `PreTrainedPolicy` interface. Inference uses DreamZero's own native KV-cache
autoregressive path (`WANPolicyHead.lazy_joint_video_action`) — NOT any vLLM serving
pipeline. The stateful cache (current_start_frame / kv caches / clip_feas / ys / language)
lives on the action head and is cleared by `reset()`.

First version targets single-environment rollout (num_envs == 1): the KV cache has no batch
semantics, so batched vector-env eval is out of scope until the cache is made batch-aware.
"""

import logging
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.import_utils import require_package

from .configuration_dreamzero import DreamZeroConfig

logger = logging.getLogger(__name__)


class DreamZeroPolicy(PreTrainedPolicy):
    """DreamZero World Action Model policy."""

    config_class = DreamZeroConfig
    name = "dreamzero"

    def __init__(
        self,
        config: DreamZeroConfig,
        dataset_stats: dict | None = None,
        dataset_meta=None,
    ):
        super().__init__(config)
        require_package("diffusers", extra="dreamzero")
        config.validate_features()
        self.config = config

        # Imported lazily so `import lerobot.policies.dreamzero` stays cheap and the heavy
        # torch/diffusers deps only load when a policy is actually constructed.
        from transformers.feature_extraction_utils import BatchFeature

        from .wan.action_head import WANPolicyHead

        self._BatchFeature = BatchFeature
        # Named `action_head` so released DreamZero checkpoint keys (action_head.*) map directly.
        self.action_head = WANPolicyHead(config.build_action_head_config())

        # select_action's rolling chunk buffer.
        self._action_queue: deque[Tensor] = deque([], maxlen=config.n_action_steps)
        self.reset()

    # ------------------------------------------------------------------
    # PreTrainedPolicy surface
    # ------------------------------------------------------------------

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def reset(self):
        """Clear the AR session state and the action-selection queue."""
        self._action_queue.clear()
        head = self.action_head
        head.current_start_frame = 0
        head.language = None
        head.clip_feas = None
        head.ys = None
        head.kv_cache1 = None
        head.kv_cache_neg = None
        head.crossattn_cache = None
        head.crossattn_cache_neg = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Training forward — flow-matching loss over video + action.

        Training is not the focus of the initial port (LoRA post-training is staged separately),
        but the wiring matches upstream: `WANPolicyHead.forward` returns a BatchFeature carrying
        the combined loss.
        """
        action_input = self._BatchFeature(data=dict(batch))
        backbone_output = self._empty_backbone_output(action_input)
        outputs = self.action_head(backbone_output, action_input)
        loss = outputs["loss"]
        loss_dict = {k: float(v) for k, v in outputs.items() if k != "loss" and _is_scalar(v)}
        return loss, loss_dict or None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Run one AR step of the joint video+action model → (B, action_horizon, action_dim).

        Returns the still-normalized action chunk; the postprocessor unnormalizes and converts
        relative joint deltas back to absolute.
        """
        action_input = self._BatchFeature(data=dict(batch))
        backbone_output = self._empty_backbone_output(action_input)
        outputs = self.action_head.lazy_joint_video_action(backbone_output, action_input, latent_video=None)
        return outputs["action_pred"]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a single action, refilling the chunk queue from the model when empty."""
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)  # (B, horizon, action_dim)
            chunk = chunk[:, : self.config.n_action_steps]
            # queue holds per-timestep (B, action_dim) tensors, popped left-to-right.
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_backbone_output(self, action_input):
        """DreamZero's backbone is an identity that contributes no real features.

        The action head only reads `action_input`; it never consumes backbone features, so an
        empty BatchFeature carrying the expected `backbone_features` key is sufficient.
        """
        return self._BatchFeature(data={"backbone_features": torch.empty(0)})


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)
