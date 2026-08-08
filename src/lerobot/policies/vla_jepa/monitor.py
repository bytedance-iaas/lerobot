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

"""Inference-time monitoring of world-model prediction error.

The world model is trained as an auxiliary objective and normally sits idle at inference.
This module runs it forward one temporal position and compares the prediction against the
observations that actually arrived, yielding a residual and a scalar surprise signal.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .configuration_vla_jepa import VLAJEPAConfig


@dataclass
class MonitorOutput:
    """One world-model prediction-error measurement.

    `residual` is ``real - predicted`` in V-JEPA latent space, shaped
    ``[B, tokens_per_position, embed_dim]``. `error` is the per-batch-element mean absolute
    residual, matching the L1 objective the predictor was trained under. `predicted` is kept
    for logging and for the V2 residual-conditioning pathway.
    """

    residual: Tensor
    error: Tensor
    predicted: Tensor


class WorldModelMonitor:
    """Tracks world-model prediction error across a rollout.

    Holds a ring buffer of recent observation frames. Once ``2 * tubelet_size`` frames have
    accumulated it V-JEPA-encodes them, splits the encoding shift-by-one into a context
    position and a ground-truth position, runs the action-conditioned predictor on the
    context, and compares.

    The monitor is inference-only and never accumulates gradients. It holds references to
    the policy's frozen submodules rather than owning them, so it adds no parameters.
    """

    def __init__(
        self,
        video_encoder: torch.nn.Module,
        video_processor: object,
        video_predictor: torch.nn.Module,
        config: VLAJEPAConfig,
    ) -> None:
        self.video_encoder = video_encoder
        self.video_processor = video_processor
        self.video_predictor = video_predictor
        self.config = config

        # `jepa_tubelet_size` is the *view count* in this codebase (see modeling_vla_jepa.py:110
        # and :217); the real tubelet size comes from the encoder. Do not conflate them.
        self.num_views: int = config.jepa_tubelet_size
        self.tubelet_size: int = video_encoder.config.tubelet_size

        # One context position plus one ground-truth position.
        self.window: int = 2 * self.tubelet_size

        self._buffer: deque[Tensor] = deque(maxlen=self.window)
        self._steps_since_emit: int = 0
        self.error_history: list[list[float]] = []

    def reset(self) -> None:
        """Clear the frame buffer, the emit counter, and the error history."""
        self._buffer.clear()
        self._steps_since_emit = 0
        self.error_history = []

    def dump_error_history(self, path: str | Path) -> None:
        """Write the accumulated per-step errors as newline-delimited JSON.

        One record per batch element: ``{"batch_index": int, "errors": list[float]}``.
        Writing an empty history produces an empty file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for index, errors in enumerate(self.error_history):
                handle.write(json.dumps({"batch_index": index, "errors": errors}) + "\n")

    @torch.no_grad()
    def observe(self, frames: Tensor, action_tokens: Tensor) -> MonitorOutput | None:
        """Push one control step's frames and, on a tubelet boundary, measure prediction error.

        `frames`: ``[B, V, C, H, W]`` float in ``[0, 1]`` — one frame per camera view.
        `action_tokens`: ``[B, N, H]`` Qwen action-token hidden states cached from the chunk
        prediction currently being executed. The policy stores them when
        ``predict_action_chunk`` runs and passes the same tensor on every step until the next
        replan.

        Returns ``None`` until the buffer holds both a context position and the ground-truth
        position that follows it; thereafter returns a ``MonitorOutput`` every
        ``tubelet_size`` steps.
        """
        self._buffer.append(frames)
        self._steps_since_emit += 1

        if len(self._buffer) < self.window or self._steps_since_emit < self.tubelet_size:
            return None
        self._steps_since_emit = 0

        embeddings = self._encode_buffer()
        return self._compare(embeddings, action_tokens)

    def _encode_buffer(self) -> Tensor:
        """V-JEPA-encode the buffered frames. Returns ``[B, tokens_total, V * D]``.

        Mirrors ``VLAJEPAModel._world_model_loss`` (modeling_vla_jepa.py:216-237) so the
        monitor's inputs match what the predictor was trained on.
        """
        # [B, V, T, C, H, W]
        videos = torch.stack(list(self._buffer), dim=2)

        # Match the predictor's expected view count exactly: pad with the first view or trim.
        if videos.shape[1] < self.num_views:
            missing = self.num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > self.num_views:
            videos = videos[:, : self.num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        video_pixels = self.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=self.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]

        embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
        # Merge views: [B*V, tokens, D] -> [B, tokens, V*D]
        return torch.cat(torch.chunk(embeddings, chunks=v, dim=0), dim=2)

    def _compare(self, embeddings: Tensor, action_tokens: Tensor) -> MonitorOutput:
        """Split shift-by-one, predict the next position, and diff against reality."""
        t_enc_total = self.window // self.tubelet_size  # == 2 by construction
        tokens_per_position = embeddings.shape[1] // t_enc_total
        t_enc_ctx = t_enc_total - 1  # == 1

        input_states = embeddings[:, : tokens_per_position * t_enc_ctx, :].float()
        gt_states = embeddings[:, tokens_per_position:, :].float()

        expected = t_enc_ctx * self.config.num_action_tokens_per_timestep
        conditioning = action_tokens.float()
        if conditioning.shape[1] < expected:
            pad = conditioning[:, -1:].repeat(1, expected - conditioning.shape[1], 1)
            conditioning = torch.cat([conditioning, pad], dim=1)
        conditioning = conditioning[:, :expected]

        predicted = self.video_predictor(input_states, conditioning)
        residual = gt_states - predicted
        error = residual.abs().mean(dim=(1, 2))

        if not self.error_history:
            self.error_history = [[] for _ in range(error.shape[0])]
        for index, value in enumerate(error.tolist()):
            self.error_history[index].append(value)

        return MonitorOutput(residual=residual, error=error, predicted=predicted)
