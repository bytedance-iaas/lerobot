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

Translates DreamZero's Hydra config tree into a draccus `PreTrainedConfig`.

Only the ``wan21_14b`` backbone (Wan2.1-I2V-14B DiT, 16-channel `WanVideoVAE`) is supported,
because it is the only one with released DreamZero weights: `GEAR-Dreams/DreamZero-DROID` is a
14B checkpoint, and upstream's own LIBERO configs also start from it. The Wan2.2-TI2V-5B variant
exists upstream but only as a cold start from base Wan components (`model_path: null` — DiT/VAE/T5
from `Wan-AI/Wan2.2-TI2V-5B`, CLIP from `Wan-AI/Wan2.1-I2V-14B-480P`), with no published DreamZero
checkpoint to validate a port against, so it is deliberately not carried here rather than shipped
untested.

The concrete field values for a released checkpoint live in that checkpoint's own saved config;
`DreamZeroPolicy.from_pretrained` reads and overrides these defaults from it.
"""

import logging
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

DREAMZERO_WAN21_14B = "wan21_14b"

# Frame rate the released checkpoints were trained at (DROID / OXE). The window contract counts
# steps, not seconds, so a dataset recorded faster has to be strided down to this — see
# `DreamZeroConfig.source_fps`.
_DREAMZERO_FPS = 15

# Which parameters fine-tuning updates; see `DreamZeroConfig.training_mode`.
TRAINING_MODES = ("full", "lora")

# Modes that existed and no longer do, with what replaces them. Named explicitly so an old command
# line gets an answer instead of "unknown training_mode".
_REMOVED_TRAINING_MODES = {
    "projector_only": (
        "use training_mode='lora', which trains the same projectors plus rank-`lora_rank` "
        "adapters on the DiT blocks for 19.2 M more parameters"
    ),
}

# Embodiments this port knows how to preprocess. The value is the projector index the model
# would select — and it is 0 for every one of them, because that is the only index that exists.
#
# Upstream's `CategorySpecificLinear` keeps a per-embodiment pool (`W[cat_ids]`) and its config
# names indices 17/26/32 for oxe_droid/agibot/yam, but the released 14B checkpoint has a pool of
# ONE: `causal_wan_model.py` overrides `max_num_embodiments = 1` when building the encoders, and
# both forward paths overwrite the incoming `embodiment_id` with 0 before indexing. So the
# published indices index nothing — passing 17 or 64 would be out of bounds if it were ever used,
# and identical to 0 because it is not.
#
# The tensor still has to be emitted: `_forward_train` reads `.device` off it before discarding
# the value, so `None` would raise there. What actually distinguishes an embodiment is the data
# representation — statistics, canvas layout, prompt, and the state/action masks — not this.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = dict.fromkeys(("oxe_droid", "agibot", "yam", "so100"), 0)

DREAMZERO_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, "
    "mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused "
    "fingers, stagnant image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)


# DiT (CausalWanModel) constructor kwargs, from the upstream action-head YAML
# (wan_flow_matching_action_tf.yaml). Kept as a dict keyed by variant so a checkpoint's DiT dims
# can be matched against it and a non-14B checkpoint rejected by name rather than by shape errors
# deep inside the model.
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
}

# The 14B backbone pairs with Wan2.1's 16-channel VAE, which downsamples 8x spatially. Combined
# with the DiT's 2x2 latent patch this fixes the video canvas size (see _validate_video_geometry).
_VAE_CLASS = "WanVideoVAE"
_VAE_SPATIAL_COMPRESSION = 8
# DiT patch size along (H, W); the video head consumes 2x2 latent patches.
_DIT_PATCH_HW = 2


@PreTrainedConfig.register_subclass("dreamzero")
@dataclass
class DreamZeroConfig(PreTrainedConfig):
    """Configuration for the DreamZero policy."""

    # Which pretrained backbone / action-head preset to build.
    model_variant: str = DREAMZERO_WAN21_14B

    # DreamZero normalizes inside its own processor steps, using the q01/q99 statistics that ship
    # with the checkpoint rather than dataset stats — so the standard normalizer must be a no-op.
    # (`lerobot-train` reads this to build the normalizer overrides that
    # `_drop_absent_standard_overrides` then discards; GrootConfig declares the same IDENTITY map.)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Embodiment identity — selects the per-embodiment projector index and action layout.
    embodiment_tag: str = "oxe_droid"

    # Video / action / state windowing (DROID defaults from droid_training_lora.sh).
    num_frames: int = 33
    action_horizon: int = 24
    num_frame_per_block: int = 2
    num_action_per_block: int = 24
    num_state_per_block: int = 1
    max_chunk_size: int = 4

    # Number of actions consumed per env step before re-querying the policy.
    n_action_steps: int = 8

    # Zero-pad width for state / action vectors (transform/dreamzero_cotrain.yaml + droid override).
    max_state_dim: int = 64
    max_action_dim: int = 32

    # Runtime weight dtype. The released checkpoints are bf16 and upstream's `post_initialize`
    # casts every component (DiT / text / image / VAE) to bf16 before inference, so this is both
    # the faithful setting and the one that fits: 23B params is ~46 GB in bf16 vs ~92 GB in fp32.
    # For a FULL fine-tune set this to "float32": FSDP then keeps fp32 master weights and does the
    # forward/backward in bf16 (`mixed_precision` in the accelerate config), which is what both
    # RLinf (`actor.model.precision: fp32`) and upstream (DeepSpeed ZeRO-2, whose partitioned
    # optimizer state holds the fp32 copies) do. bf16 master weights would silently lose most of
    # the update at the reference lr of 1e-5: bf16's spacing at |w| is about |w| * 2^-8, and with
    # DiT weights at |w| ~ 1.2e-2 that is ~5e-5 against an Adam step of ~1e-5, so ~84% of updates
    # round away (measured on the released checkpoint's own weights).
    compute_dtype: str = "bfloat16"
    # torch.compile the text/image/VAE encoders (upstream `post_initialize` does this too). Off by
    # default: it costs a few minutes of warmup and is only worth it for long rollouts.
    compile_encoders: bool = False

    # Diffusion. `num_inference_timesteps` is a TRAINING parameter (the released DROID checkpoint
    # trained with 4); the inference loop uses `WANPolicyHead.num_inference_steps`, which upstream
    # fixes at 16 in its constructor. They are distinct despite the similar names, so this config
    # carries the training value and does not attempt to drive the inference schedule.
    num_inference_timesteps: int = 4

    # Inference.
    num_inference_steps: int = 16
    cfg_scale: float = 5.0
    hidden_size: int = 64
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 1536

    # Relative-action decoding (DROID converts joint_position deltas back to absolute at output).
    relative_action: bool = True
    relative_action_keys: tuple[str, ...] = ("joint_position",)

    # Per-embodiment I/O layout, from the checkpoint's meta/modality.json concat order. Each key's
    # width is taken from its normalization stats (len of q01). Defaults are the oxe_droid/DROID
    # layout (joint_position: 7, gripper_position: 1); other embodiments override via from_pretrained.
    state_modality_keys: tuple[str, ...] = ("joint_position", "gripper_position")
    action_modality_keys: tuple[str, ...] = ("joint_position", "gripper_position")
    # Camera keys in the view-stitch order the model expects — the third of the three concat
    # orders, alongside state/action above, and read from the same `ConcatTransform` in the
    # checkpoint's `experiment_cfg/conf.yaml` (`video_concat_order`). Order is not cosmetic: it
    # decides which camera lands in which canvas quadrant, and the generated prompt names the
    # quadrants from this same list.
    #
    # Left empty, `make_dreamzero_pre_post_processors_from_pretrained` fills it from the
    # checkpoint, the way GR00T resolves its own `video_modality_keys` — so a fine-tune normally
    # does not pass it. Falling back further (sorted keys) is a last resort: it happens to give
    # DROID's order alphabetically, which is luck rather than a guarantee. Note the dataset's own
    # feature order is NOT usable here — `lerobot/droid_1.0.1` lists the wrist camera first, while
    # the checkpoint stitches it last.
    video_modality_keys: tuple[str, ...] = ()
    # How to name each view in the prompt, in `video_modality_keys` order (e.g. "wrist camera",
    # "front camera"). The prompt is the only thing telling the model which quadrant holds which
    # camera, so a name that does not match the layout is a silent mismatch. Empty => derived from
    # `video_modality_keys` by stripping the LeRobot key prefix. Unused by oxe_droid, whose prompt is
    # fixed upstream text.
    view_descriptions: tuple[str, ...] = ()
    # Frames per second of the dataset being trained on, when it differs from the 15 fps DreamZero
    # was trained at. The window contract is defined in *steps*, not seconds, so a 30 fps dataset
    # would hand the model a 3.2 s window where it learned 6.4 s ones. Setting this stretches the
    # delta indices by `source_fps / 15` so the window covers the same wall-clock span, sampling
    # every n-th frame — sound here because the actions are absolute position targets rather than
    # increments, so dropping intermediate ones changes the rate, not the meaning.
    source_fps: int | None = None
    # A `statistics.json` (or the directory holding one) to normalize with, overriding whatever the
    # checkpoint ships. Needed whenever the weights and the data come from different places — a new
    # robot fine-tuned from the released DROID checkpoint has DROID's quantiles in the checkpoint
    # and its own in the file `scripts/compute_statistics.py` wrote. `lerobot-train` cannot supply
    # them through `dataset_stats`: it passes LeRobot's flat mean/std schema, which DreamZero
    # rejects rather than misinterpret.
    statistics_path: str | None = None
    # Training-time video augmentation, from upstream's oxe_droid transform chain
    # (VideoCrop random -> VideoResize -> VideoColorJitter). Inference always centre-crops and
    # never jitters. This is separate from LeRobot's `--dataset.image_transforms`, which would
    # compose on top.
    video_color_jitter: bool = True
    color_jitter_params: dict = field(
        default_factory=lambda: {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    )

    # Per-view resolution the transform resizes each camera to before stitching, and the
    # center-crop fraction applied first (upstream VideoCrop scale=0.95 -> VideoResize 176x320).
    # The stitched canvas is 2*per_view_height x 2*per_view_width, and its token count per latent
    # frame must equal the DiT's `frame_seqlen` (which sizes the blockwise-causal attention mask):
    #   (2*H/8/2) * (2*W/8/2) == frame_seqlen   ->   176x320 gives 22*40 = 880 == wan21_14b preset.
    # __post_init__ enforces this, so a mismatch fails at config build instead of silently
    # corrupting the attention blocks.
    per_view_height: int = 176
    per_view_width: int = 320
    view_crop_scale: float = 0.95

    # Base Wan component checkpoints. None => auto-download from the base Wan HF repo at build time
    # (matches upstream WANPolicyHead.__init__ ensure_file behaviour).
    text_encoder_pretrained_path: str | None = None
    image_encoder_pretrained_path: str | None = None
    vae_pretrained_path: str | None = None
    # When loading a full DreamZero checkpoint, skip loading the base Wan DiT (checkpoint supplies it).
    skip_component_loading: bool = True

    # Which parameters fine-tuning updates. The released checkpoint is the starting point
    # (continued training), not a cold start from the base Wan weights:
    #   "lora"   — the default. The action/state projectors (89.4 M) plus rank-`lora_rank` adapters
    #              on the DiT blocks (19.2 M at rank 4). Fits on one GPU, and its checkpoints are
    #              ~0.2 GB rather than ~46 GB. This is what a fine-tune should normally use.
    #   "full"   — the whole DiT incl. projectors (16.48 B). This is how upstream and RLinf train
    #              DROID, and it is here for parity with them, not as a recommendation: it needs
    #              fp32 master weights sharded across ~8 GPUs (57 GB each) and writes a 46 GB
    #              checkpoint per save. Reach for it only when LoRA has been shown not to be enough.
    # The text/image/VAE encoders are frozen in both modes (WANPolicyHead does that itself), which
    # is why even "full" leaves 6.44 B parameters untouched.
    training_mode: str = "lora"

    # LoRA — mirrors upstream scripts/train/droid_training_lora.sh.
    lora_rank: int = 4
    lora_alpha: int = 4
    # Matched by suffix, so `q,k,v,o` covers self- AND cross-attention (10 wrapped modules per
    # block, 19.2 M adapters) but NOT `cross_attn.k_img` / `v_img`, the image-conditioning
    # projections — add `k_img,v_img` to include them (12 modules per block, 22.5 M).
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    # Write only the parameters training updated, instead of every weight. Lossless: what is left
    # out is bit-identical to the base checkpoint, which the saved manifest names and the loader
    # reads back. Not low-rank and not an approximation — a `full` run stores its 16,484 M dense
    # DiT weights and merely omits the 6,440 M text/image/VAE encoders that `WANPolicyHead` freezes
    # in every mode.
    #
    #   lora, rank 4    108.6 M    0.4 GB   instead of 46 GB
    #   full         16,484.0 M   66.0 GB   instead of 92 GB (fp32 master under FSDP)
    #
    # The cost is that the checkpoint is not self-contained; loading raises rather than guess if
    # the base has moved. Set false for a standalone checkpoint.
    save_trainable_only: bool = True
    # Deprecated spelling of `save_trainable_only`. The old name described the LoRA case only,
    # which misled once the same mechanism started applying to `full` runs. Set it and it still
    # wins, with a warning.
    save_lora_only: bool | None = None
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # Optimizer / scheduler preset, from upstream's DROID full-finetune recipe
    # (scripts/train/droid_training_full_finetune_wan21.sh; RLinf's droid_sft_dreamzero_14b.yaml
    # agrees). The previous defaults here were the LoRA script's (lr 1e-4, betas 0.9/0.95).
    # Nominal run length, used only to turn `warmup_ratio` into a warmup step count (the same
    # role it plays in GrootConfig). The real training length comes from `lerobot-train --steps`,
    # which `DiffuserSchedulerConfig` reads at runtime for the decay schedule.
    max_steps: int = 100_000
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.training_mode in _REMOVED_TRAINING_MODES:
            raise ValueError(
                f"training_mode={self.training_mode!r} has been removed: "
                f"{_REMOVED_TRAINING_MODES[self.training_mode]}."
            )
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(
                f"Unknown training_mode {self.training_mode!r}; expected one of {list(TRAINING_MODES)}."
            )
        if self.save_lora_only is not None:
            logger.warning(
                "save_lora_only is deprecated (it stores every trainable parameter, not just LoRA "
                "ones, and now applies to `full` too); use save_trainable_only=%s.",
                self.save_lora_only,
            )
            self.save_trainable_only = self.save_lora_only
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.model_variant not in _DIT_PRESETS:
            raise ValueError(
                f"Unknown model_variant {self.model_variant!r}; expected one of {list(_DIT_PRESETS)}."
            )
        if self.embodiment_tag not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            raise ValueError(
                f"Unknown embodiment_tag {self.embodiment_tag!r}; expected one of "
                f"{list(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)}."
            )
        if not self.relative_action:
            # `relative_action_keys` is only meaningful with the relative-action convention on
            # (upstream and RLinf gate on both). Clearing the keys here keeps the training encode,
            # the inference decode and `compute_statistics` consistent with the flag — otherwise
            # `relative_action=false` would be read but silently not applied.
            self.relative_action_keys = ()
        if (
            self.view_descriptions
            and self.video_modality_keys
            and len(self.view_descriptions) != len(self.video_modality_keys)
        ):
            # The prompt is the only thing telling the model which quadrant holds which camera; a
            # description list of the wrong length would either misname quadrants or describe
            # occupied ones as black screens, without changing anything else observable.
            raise ValueError(
                f"view_descriptions has {len(self.view_descriptions)} entries but video_modality_keys "
                f"names {len(self.video_modality_keys)} views; they must describe the same cameras "
                "in the same order."
            )
        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed action_horizon ({self.action_horizon})."
            )
        self._validate_video_geometry()

    def _validate_video_geometry(self) -> None:
        """Fail if the stitched video canvas does not yield the DiT's `frame_seqlen` tokens.

        `frame_seqlen` sizes the blockwise-causal attention mask, so a canvas that tokenizes to a
        different length silently mis-aligns every video block against the action registers rather
        than raising — hence the explicit check here.
        """
        # The 3-view stitch produces a 2H x 2W canvas, which the 14B head consumes unresized.
        canvas_h = 2 * self.per_view_height
        canvas_w = 2 * self.per_view_width
        divisor = _VAE_SPATIAL_COMPRESSION * _DIT_PATCH_HW
        if canvas_h % divisor or canvas_w % divisor:
            raise ValueError(
                f"DreamZero video canvas {canvas_h}x{canvas_w} is not divisible by {divisor} "
                f"(VAE {_VAE_SPATIAL_COMPRESSION}x downsample * {_DIT_PATCH_HW}x{_DIT_PATCH_HW} DiT patch)."
            )
        tokens = (canvas_h // divisor) * (canvas_w // divisor)
        expected = _DIT_PRESETS[self.model_variant]["frame_seqlen"]
        if tokens != expected:
            raise ValueError(
                f"DreamZero video geometry mismatch for {self.model_variant}: per-view "
                f"{self.per_view_height}x{self.per_view_width} -> canvas {canvas_h}x{canvas_w} -> "
                f"{tokens} tokens per latent frame, but the DiT expects frame_seqlen={expected}. "
                f"Set per_view_height/per_view_width so the canvas tokenizes to {expected}."
            )

    @classmethod
    def from_foreign_checkpoint(cls, pretrained_name_or_path) -> "DreamZeroConfig | None":
        """Adopt a released NVIDIA checkpoint (GEAR ``VLA`` layout), or return None if it isn't one.

        LeRobot checkpoints carry a draccus ``config.json`` with a ``type`` key; the released
        DreamZero repos ship a HuggingFace ``VLA`` config instead, which draccus cannot parse.
        Generic tooling that wants to accept both can look for this classmethod on registered
        policy configs — see ``examples/eval_open_loop.py``.
        """
        from . import gear_checkpoint

        if not gear_checkpoint.is_gear_checkpoint(pretrained_name_or_path):
            return None
        return gear_checkpoint.build_config(pretrained_name_or_path)

    @property
    def embodiment_projector_index(self) -> int:
        """Always 0 — see `EMBODIMENT_TAG_TO_PROJECTOR_INDEX`. Kept because the tensor is."""
        return EMBODIMENT_TAG_TO_PROJECTOR_INDEX[self.embodiment_tag]

    def build_action_head_config(self, defer_lora_injection: bool = False):
        """Assemble the internal `WANPolicyHeadConfig` consumed by `WANPolicyHead`.

        Kept here (rather than in the model) so the Hydra `_target_`/`instantiate` translation
        lives next to the values it mirrors.

        `defer_lora_injection` postpones the PEFT wrapping to after the weights are loaded. It is
        not a config field because it is not a property of the run — it is a property of *which
        checkpoint is being loaded*, decided by `DreamZeroPolicy.from_pretrained`. See the comment
        on `DreamZeroPolicy._finalize_lora` for why the two directions differ.
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
            num_inference_timesteps=self.num_inference_timesteps,
            train_architecture=("lora" if self.training_mode == "lora" else "full"),
            defer_lora_injection=defer_lora_injection,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_target_modules=self.lora_target_modules,
            tune_projector=self.tune_projector,
            tune_diffusion_model=self.tune_diffusion_model,
            skip_component_loading=self.skip_component_loading,
            # No post-stitch resize: the 352x320*2 canvas already tokenizes to frame_seqlen.
            target_video_height=None,
            target_video_width=None,
            diffusion_model_cfg=dit_cfg,
            text_encoder_cfg={"text_encoder_pretrained_path": self.text_encoder_pretrained_path},
            image_encoder_cfg={"image_encoder_pretrained_path": self.image_encoder_pretrained_path},
            vae_cfg={"vae_pretrained_path": self.vae_pretrained_path},
            vae_class=_VAE_CLASS,
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
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.max_action_dim,)
            )

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

    # ------------------------------------------------------------------
    # Training window
    #
    # A training sample is one `max_chunk_size`-block window: `num_frames` video frames spanning
    # `action_horizon * max_chunk_size` steps, the dense actions over that span, and one state per
    # block. Upstream samples it with a bespoke multi-anchor sampler, but the set of windows it can
    # produce is just "consecutive frames within one language segment", which LeRobot expresses
    # natively with delta indices plus `drop_n_last_frames` — no custom Dataset needed. The
    # bidirectional expansion in upstream's sampler only re-centres windows (changing sampling
    # weight near segment ends); it does not enlarge the support.
    #
    # Inference does not use these: it feeds a single observation per query.
    # ------------------------------------------------------------------

    @property
    def video_frame_stride(self) -> int:
        """Raw-frame stride between the video frames of one macro block.

        `num_frames` = `frames_per_block * max_chunk_size + 1` (the +1 is the boundary frame), and
        a block spans `action_horizon` raw steps, so the stride follows from the two.
        """
        frames_per_block = (self.num_frames - 1) // self.max_chunk_size
        return self.action_horizon // frames_per_block

    @property
    def frame_rate_multiplier(self) -> int:
        """Dataset frames per model step, from `source_fps` against DreamZero's 15 fps."""
        if not self.source_fps:
            return 1
        multiplier = round(self.source_fps / _DREAMZERO_FPS)
        if multiplier < 1:
            raise ValueError(
                f"source_fps={self.source_fps} is below DreamZero's {_DREAMZERO_FPS} fps. The "
                "window cannot be stretched to cover the span the model was trained on; "
                "re-record or upsample the dataset instead."
            )
        return multiplier

    @property
    def training_span(self) -> int:
        """Raw frames covered by one training window (also the last usable start offset)."""
        return self.action_horizon * self.max_chunk_size * self.frame_rate_multiplier

    @property
    def observation_delta_indices(self) -> list[int]:
        # 33 frames at stride 3 spanning 96 steps, plus the boundary frame at +96. Applies to
        # every `observation.*` key, so `observation.state` arrives with the same 33 rows; the
        # processor picks rows [0, 8, 16, 24] as the per-block anchors.
        return list(range(0, self.training_span + 1, self.video_frame_stride * self.frame_rate_multiplier))

    @property
    def action_delta_indices(self) -> list[int]:
        # Dense actions over the whole window: action_horizon * max_chunk_size.
        return list(range(0, self.training_span, self.frame_rate_multiplier))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def drop_n_last_frames(self) -> int:
        # The window needs the boundary frame at +training_span, so a start index is only usable
        # if `training_span` further frames exist. This is what makes padded windows impossible;
        # the processor still asserts on the `*_is_pad` masks as a tripwire.
        return self.training_span
