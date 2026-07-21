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

"""LeRobot config for the DreamZero policy (NVIDIA GEAR Wan-video World Action Model).

Translates DreamZero's Hydra config tree into a draccus `PreTrainedConfig`. Two model
variants are supported, mirroring the upstream action-head YAMLs:

- ``wan21_14b`` — DreamZero-DROID / DreamZero-AgiBot backbone (Wan2.1-I2V-14B, VAE 16ch)
- ``wan22_5b``  — Wan2.2-TI2V-5B backbone (VAE38 48ch, 160x320)

The concrete field values for a released checkpoint live in that checkpoint's own saved
config; `DreamZeroPolicy.from_pretrained` reads and overrides these defaults from it.
"""

import logging
from dataclasses import dataclass

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

DREAMZERO_WAN21_14B = "wan21_14b"
DREAMZERO_WAN22_5B = "wan22_5b"

# Embodiment projector indices, from groot/vla/configs/model/dreamzero/transform/base.yaml
# (embodiment_tag_to_projector_index). Used to select the per-embodiment category head.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid": 17,
    "agibot": 26,
    "yam": 32,
}

# Fixed CFG negative prompt shared by all embodiments (see DreamTransform.apply_single).
DREAMZERO_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, "
    "mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused "
    "fingers, stagnant image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)


# Per-variant DiT (CausalWanModel) constructor kwargs. Values from the upstream action-head
# YAMLs (wan_flow_matching_action_tf.yaml / _wan22.yaml).
_DIT_PRESETS = {
    DREAMZERO_WAN21_14B: {
        "model_type": "i2v",
        "frame_seqlen": 880,
        "dim": 5120,
        "in_dim": 36,
        "ffn_dim": 13824,
        "out_dim": 16,
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": 40,
        "num_layers": 40,
    },
    DREAMZERO_WAN22_5B: {
        "model_type": "i2v",
        "frame_seqlen": 50,
        "dim": 3072,
        "in_dim": 48,
        "ffn_dim": 14336,
        "out_dim": 48,
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": 24,
        "num_layers": 30,
    },
}

# VAE class + target resolution per variant.
_VAE_PRESETS = {
    DREAMZERO_WAN21_14B: {"vae_class": "WanVideoVAE", "vae_kwargs": {}, "target_video_height": None, "target_video_width": None},
    DREAMZERO_WAN22_5B: {
        "vae_class": "WanVideoVAE38",
        "vae_kwargs": {"z_dim": 48, "dim": 160},
        "target_video_height": 160,
        "target_video_width": 320,
    },
}


@PreTrainedConfig.register_subclass("dreamzero")
@dataclass
class DreamZeroConfig(PreTrainedConfig):
    """Configuration for the DreamZero policy."""

    # Which pretrained backbone / action-head preset to build.
    model_variant: str = DREAMZERO_WAN21_14B

    # Embodiment identity — selects the per-embodiment projector index and action layout.
    embodiment_tag: str = "oxe_droid"

    # Video / action / state windowing (DROID defaults from droid_training_lora.sh).
    num_frames: int = 33
    action_horizon: int = 24
    num_views: int = 3
    num_frame_per_block: int = 2
    num_action_per_block: int = 24
    num_state_per_block: int = 1
    max_chunk_size: int = 5

    # Number of actions consumed per env step before re-querying the policy.
    n_action_steps: int = 8

    # Zero-pad width for state / action vectors (transform/dreamzero_cotrain.yaml + droid override).
    max_state_dim: int = 64
    max_action_dim: int = 32

    # Diffusion / inference.
    num_inference_steps: int = 16
    cfg_scale: float = 5.0
    hidden_size: int = 64
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 1536

    # Relative-action decoding (DROID converts joint_position deltas back to absolute at output).
    relative_action: bool = True
    relative_action_keys: tuple[str, ...] = ("joint_position",)

    # Base Wan component checkpoints. None => auto-download from the base Wan HF repo at build time
    # (matches upstream WANPolicyHead.__init__ ensure_file behaviour).
    text_encoder_pretrained_path: str | None = None
    image_encoder_pretrained_path: str | None = None
    vae_pretrained_path: str | None = None
    # When loading a full DreamZero checkpoint, skip loading the base Wan DiT (checkpoint supplies it).
    skip_component_loading: bool = True

    # LoRA (post-training) — mirrors scripts/train/*_training_lora.sh.
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    train_architecture: str = "lora"
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # Optimizer / scheduler presets (LoRA: AdamW lr 1e-4 + cosine warmup).
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    warmup_ratio: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if self.model_variant not in _DIT_PRESETS:
            raise ValueError(
                f"Unknown model_variant {self.model_variant!r}; expected one of {list(_DIT_PRESETS)}."
            )
        if self.embodiment_tag not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            raise ValueError(
                f"Unknown embodiment_tag {self.embodiment_tag!r}; expected one of "
                f"{list(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)}."
            )
        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed action_horizon ({self.action_horizon})."
            )

    @property
    def embodiment_projector_index(self) -> int:
        return EMBODIMENT_TAG_TO_PROJECTOR_INDEX[self.embodiment_tag]

    def build_action_head_config(self):
        """Assemble the internal `WANPolicyHeadConfig` consumed by `WANPolicyHead`.

        Kept here (rather than in the model) so the Hydra `_target_`/`instantiate` translation
        lives next to the values it mirrors.
        """
        # Imported lazily so importing the config never pulls in torch/diffusers.
        from .wan.action_head import WANPolicyHeadConfig

        dit_cfg = dict(
            _DIT_PRESETS[self.model_variant],
            diffusion_model_pretrained_path=None,
            max_chunk_size=self.max_chunk_size,
            num_frame_per_block=self.num_frame_per_block,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_dim=self.max_action_dim,
        )
        vae_preset = _VAE_PRESETS[self.model_variant]
        vae_cfg = dict(vae_preset["vae_kwargs"], vae_pretrained_path=self.vae_pretrained_path)

        return WANPolicyHeadConfig(
            num_frames=self.num_frames,
            num_frame_per_block=self.num_frame_per_block,
            hidden_size=self.hidden_size,
            input_embedding_dim=self.input_embedding_dim,
            backbone_embedding_dim=self.backbone_embedding_dim,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            action_dim=self.max_action_dim,
            action_horizon=self.action_horizon,
            num_inference_timesteps=self.num_inference_steps,
            train_architecture=self.train_architecture,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_target_modules=self.lora_target_modules,
            tune_projector=self.tune_projector,
            tune_diffusion_model=self.tune_diffusion_model,
            skip_component_loading=self.skip_component_loading,
            target_video_height=vae_preset["target_video_height"],
            target_video_width=vae_preset["target_video_width"],
            diffusion_model_cfg=dit_cfg,
            text_encoder_cfg={"text_encoder_pretrained_path": self.text_encoder_pretrained_path},
            image_encoder_cfg={"image_encoder_pretrained_path": self.image_encoder_pretrained_path},
            vae_cfg=vae_cfg,
            vae_class=vae_preset["vae_class"],
        )

    # ------------------------------------------------------------------
    # PreTrainedConfig abstract surface
    # ------------------------------------------------------------------

    def validate_features(self) -> None:
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "DreamZero requires at least one visual input feature; none of type "
                "FeatureType.VISUAL found in input_features."
            )
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        import math

        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=math.ceil(self.max_steps * self.warmup_ratio),
        )

    @property
    def observation_delta_indices(self) -> None:
        # Video history is assembled by the policy's session state (AR frame accumulation),
        # not by dataset delta windows, so no observation deltas are requested here.
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def drop_n_last_frames(self) -> int:
        return max(0, len(self.action_delta_indices) - 1)
