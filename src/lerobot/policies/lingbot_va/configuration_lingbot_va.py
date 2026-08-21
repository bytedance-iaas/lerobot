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

"""Configuration for the LingBot-VA policy.

LingBot-VA is an autoregressive video-action world-model policy built on the Wan2.2
video-diffusion stack. It interleaves prediction of future video latents and robot
actions in a single dual-stream transformer. See ``docs/source/lingbot_va.mdx`` and the
upstream repository (https://github.com/Robbyant/lingbot-va).

Defaults below match the upstream LIBERO configuration (``wan_va/configs/va_libero_cfg.py``)
and the ``transformer/config.json`` of the released checkpoints.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("lingbot_va")
@dataclass
class LingBotVAConfig(PreTrainedConfig):
    """Configuration for the native LingBot-VA policy integration in LeRobot."""

    # Wan transformer architecture
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    # "flex" = training only (needs recent torch); inference uses "torch" SDPA or "flashattn".
    attn_mode: str = "torch"

    # Frozen sub-models (VAE + UMT5 text encoder + tokenizer)
    # ~20 GB of frozen weights, NOT bundled in the checkpoint; lazily pulled from this HF repo /
    # local dir (must hold diffusers-style ``vae/``, ``text_encoder/``, ``tokenizer/`` sub-folders).
    wan_pretrained_path: str = "robbyant/lingbot-va-base"
    dtype: str = "bfloat16"  # transformer / VAE / text-encoder dtype: "bfloat16", "float16", "float32"
    # Frozen UMT5-XXL encoder device; "cpu" frees ~11 GB VRAM (it runs once per episode).
    text_encoder_device: str = "cpu"

    # Observation cameras (order matters: latents are concatenated on width; LIBERO defaults)
    obs_cam_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    # Undo the LIBERO env processor's extra horizontal flip to match the model's training orientation.
    image_hflip: bool = False
    # Camera latent layout: "width_concat" (cameras concatenated on width; LIBERO) or
    # "robotwin_tshape" (full-res head + half-res wrists in a "T"; RoboTwin).
    camera_layout: str = "width_concat"

    # Inference hyperparameters (LIBERO defaults)
    n_obs_steps: int = 1
    height: int = 128
    width: int = 128
    action_per_frame: int = 4
    frame_chunk_size: int = 4
    attn_window: int = 30
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    max_sequence_length: int = 512  # UMT5 prompt length

    # Subset of the 30-d action space used by the benchmark (LIBERO = 7-DoF). The action
    # (un)normalization quantiles live in the checkpoint's ``policy_postprocessor.json``, not here.
    used_action_channel_ids: list[int] = field(default_factory=lambda: list(range(7)))

    # Opt-in: VAE-decode predicted video latents to ``self.last_predicted_frames`` for saving MP4s.
    save_predicted_video: bool = False

    # Prediction-residual feedback. At every chunk boundary the real observed keyframes are
    # VAE-encoded and written into the KV cache, so the residual against the latents the model
    # *predicted* for those same frames measures how far the world model has drifted from reality.
    # Measurement only (no behaviour change): records ``policy.last_residual`` (per-frame mean of
    # ``||z_real - z_pred|| / ||z_real||``, computed in latent space -- no VAE decode) and appends
    # it to ``policy.residual_history``.
    track_prediction_residual: bool = False
    # Control: when the residual measured at the start of a chunk exceeds this, the chunk is cut
    # short so real observations re-enter the KV cache sooner (MPC-style early re-planning).
    # ``None`` disables it and keeps the default fixed-cadence behaviour bit-for-bit.
    residual_replan_threshold: float | None = None
    # Re-planning may only happen on a latent-frame boundary (a multiple of ``action_per_frame``
    # executed actions), otherwise the observed-frame / executed-action counts fed back into the
    # KV cache no longer line up. This is the smallest chunk we are willing to execute, in latent
    # frames; it is clamped to [1, frame_chunk_size - 1].
    residual_min_exec_frames: int = 2

    # Execution feedback via a low-dimensional dynamics-context latent ``e`` (no weight update).
    # The residual stream measured above is compressed into ``e``, which is injected as an extra
    # conditioning token alongside the prompt embedding, modulating both the video and the action
    # branch (they share ``text_emb`` and the KV cache). Frozen base weights; only ``e`` moves, and
    # only within an episode. ``None`` disables the whole path and keeps behaviour bit-for-bit.
    context_latent_dim: int | None = None
    # Scale of the injected token, in units of the mean prompt-token norm. 0.0 injects a zero token
    # (a useful control arm: exercises the same code path with no signal).
    context_inject_scale: float = 1.0
    # EMA rate at which ``e`` follows the residual features. 1.0 = no smoothing.
    context_lr: float = 0.05
    # Sliding window (in chunks) used as the within-episode residual baseline. The absolute residual
    # is dominated by scene identity, so ``e`` is driven by the deviation from this baseline.
    context_window: int = 8
    # Changepoint detection: a residual this many sigmas above the window baseline is treated as a
    # dynamics regime change -- the window is dropped so ``e`` can move to the new regime fast.
    context_reset_z: float = 3.0
    # Seed for the fixed random projection from ``e`` to the text-embedding space. In v0 this
    # projection is untrained (the point is to test whether the injection path can change behaviour
    # at all); a trained injector would replace it.
    context_proj_seed: int = 0

    # Language ablation: condition on this instruction instead of the one the env supplies.
    # Set it to another task's instruction (or an empty string) to test how much the policy
    # actually relies on language; ``None`` uses the env's instruction as usual.
    prompt_override: str | None = None

    # Normalization: IDENTITY here; images are scaled + VAE-encoded and actions are
    # quantile-(un)normalized inside the policy / dedicated processor steps.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Optimizer / scheduler (training; AdamW + warmup-constant per upstream train.py)
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.attn_mode not in ("torch", "flashattn", "flex"):
            raise ValueError(f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}")

    @property
    def chunk_size(self) -> int:
        """Number of single-step actions produced per autoregressive chunk."""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        """Number of actions executed before refilling (the whole chunk)."""
        return self.chunk_size

    def validate_features(self) -> None:
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VA requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(len(self.used_action_channel_ids),)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        # Upstream uses a linear warmup followed by a constant LR (warmup_constant_lambda).
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        temporal_downsample = 4
        stride = max(1, self.action_per_frame // temporal_downsample)
        return list(range(0, self.frame_chunk_size * temporal_downsample * stride, stride))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
