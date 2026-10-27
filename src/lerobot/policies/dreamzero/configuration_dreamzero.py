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
# Temporal compression of the same VAE: the first latent frame encodes one raw frame and every
# later one encodes four, so `num_frames` must be 4k+1 (33 -> 9 latent frames).
_VAE_TEMPORAL_COMPRESSION = 4
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
    # Inference only -- `prepare_for_inference` is what applies it.
    compile_encoders: bool = False
    # torch.compile each DiT block, for training as well as inference. Per block and
    # `fullgraph=False` on purpose: the teacher-forcing attention is ~20 hand-sliced calls into
    # an opaque flash-attention kernel, so a whole-model graph is not obtainable. Compiling
    # `block.forward` puts the compiled region INSIDE the FSDP hooks, since `fsdp.yaml` wraps
    # `CausalWanAttentionBlock` -- the order that is documented to work.
    #
    # Off by default and UNVERIFIED on either card. Read the following before turning it on,
    # because the case for it is weaker than it looks:
    #
    # * Those ~20 opaque calls mean ~21 graph breaks per block, so what gets fused is only the
    #   norm/modulation/residual chains BETWEEN the attention calls. 40 blocks sharing one
    #   trace amortises the compile, but it amortises a trace of 21 small subgraphs, not of one
    #   block-sized graph.
    # * That elementwise share is smaller than it was. The 950PR profile that motivated this
    #   read Add 14% and Cast 8.8%, but that was recompute traffic, which
    #   `gradient_checkpointing=False` removes; and the largest remaining chain -- the per-token
    #   time embedding, 934 MiB a block -- was fixed by hand instead (see `_per_run`). Compile's
    #   upside here has been eaten by the two changes above it.
    # * RLinf's 9.0 -> 6.7 s/step on 8xH100 is NOT a forecast for this switch. That is compile
    #   plus CUDA graphs plus their own attention implementation; none of it transfers to
    #   flipping Dynamo on over this code.
    # * Do not turn it on an a 950PR. torch_npu's inductor autotunes serially, which is what
    #   made compiling GR00T there unusable; it is a toolchain defect, pending with Ascend, and
    #   nothing about it is specific to this model.
    #
    # It also wants `gradient_checkpointing=False`: compile under `torch.utils.checkpoint` has
    # to handle the recomputed forward too and the combination is a known source of trouble, so
    # the dial being open is what makes this clean. Compile warmup lands on top of the ~95 s the
    # first step already costs, so judge it on the steady-state step and nothing else.
    compile_dit: bool = False
    # Recompute DiT blocks in the backward pass instead of keeping their activations, which is
    # what the recompute traffic in a profile is: 16.6 k in-place copies and 5.1 k in-place adds
    # per step. Upstream hardcodes it on, and on a 950PR there is no room to turn it off: all 40
    # blocks of activations at batch 1 want ~217 GiB against a 123.2 GiB card, and adding cards
    # does not help because FSDP shards parameters and gradients but not activations. Keeping 5
    # of the 40 unrecomputed does run on a single card there (9.54 vs 9.89 s/step) but still
    # OOMs under FSDP, which has the all-gathered shards to hold as well. The knob is here
    # because the DiT hardcoded it and `WANPolicyHeadConfig` has a field of the same meaning
    # that nothing reads, so neither could be turned off at all; it is wired to the DiT in
    # `DreamZeroPolicy.__init__`.
    #
    # It is a DIAL, not a switch, because all-or-nothing is the wrong granularity for this
    # model: True checkpoints all 40 blocks, False none, and an int checkpoints that many
    # (`CausalWanModel.checkpointed_blocks` spreads them; which ones is immaterial, only how
    # many). The all-or-nothing version left the whole gap between them unreachable, and the
    # gap is large -- 24% of the work an 8xH200 full fine-tune runs, which is the 10 points
    # between its ~33% MFU and ~43% HFU (`benchmarks/mfu.py dreamzero`). Only about a quarter of
    # those ten points is actually reachable, though: see the measured cost per block below.
    #
    # The arithmetic for choosing a value, MEASURED on one 950PR card (lora, batch 1,
    # --log_freq=1): peak is 56.68 GiB at 40 checkpointed, 62.20 at 38, 73.31 at 36, 84.41 at
    # 34, 95.52 at 32. Three consecutive +11.11 GiB steps over two blocks each, so **5.56 GiB
    # per block left unrecomputed**, and all 40 of them want ~217 GiB. The first pair costs
    # half that because at True the peak falls during recompute rather than at the end of the
    # forward, so the first two blocks fit in the slack before the forward becomes the peak.
    #
    # An 8xH200 full fine-tune sits at 57.0 GB of 141 GB, so ~78 GiB of headroom buys 15
    # blocks (5.52 + 13 x 5.555 = 77.7): `gradient_checkpointing=25` is the line and **28 is
    # the value to start from**, leaving ~17 GiB for fragmentation and FSDP's all-gather
    # transients, neither of which `max_memory_allocated` counts. That is worth ~2.5 points of
    # MFU and ~8% of throughput, not the ten points to HFU.
    #
    # Do not carry 5.56 GiB/block straight to CUDA. `rope_dtypes` returns complex64 on NPU and
    # complex128 on CUDA, and each block ropes a q and a k of [1, 15940, 40, 64] -- 326 MB
    # apiece against 653 MB -- so expect ~6.2 GiB there until it is measured.
    #
    # What this replaced, because the mistake is instructive: the earlier figure of 3.04 GiB
    # per block came from dividing 121.7 GiB by 40, and that 121.7 was read off the allocator's
    # OOM message ("121.74 GiB already allocated" of 123.16). That is where a run ran out, not
    # what it needed -- runs wanting 217 GiB and runs wanting 130 both fail at ~122 on that
    # card. The number carried no information about the requirement, and everything derived
    # from it, including a recommendation of `False`, was wrong by a factor of two.
    gradient_checkpointing: bool | int = True
    # Ulysses sequence-parallel degree. 1 is off, and off is a single code path rather than a
    # branch -- the all-to-alls return their input.
    #
    # This is what makes `gradient_checkpointing=False` reachable, which no card does on its
    # own: the activation bill divides by the degree, so 217 GiB at degree 4 is 54 GiB against
    # the ~78 an 8xH200 at micro-batch 1 leaves free. Nothing else here buys those ten points
    # between MFU and HFU.
    #
    # Two divisibility constraints, both checked in `CausalWanModel.set_sequence_parallel`
    # before any data moves. The heads must divide, because the attention runs on `num_heads /
    # degree` of them per rank: 40 gives 1, 2, 4, 5, 8, 10, 20, 40. And the sequence must
    # divide, because the all-to-all splits it evenly: L = 15,940 = 2^2 x 5 x 797 at the DROID
    # window gives 1, 2, 4, 5, 10, 20. **Their intersection excludes 8.** So on eight cards the
    # configuration is 4-way Ulysses against 2-way data parallelism, which is also enough; a
    # degree of 8 would need the sequence padded, which is not implemented.
    #
    # One thing that must be true wherever the training loop lives: a rank holds a slice of the
    # tokens, so every weight's gradient is a PARTIAL SUM over its own tokens and the partials
    # add up to the whole. FSDP averages gradients across ranks rather than summing them, so
    # the loss has to be multiplied by the degree to compensate. Get that wrong and nothing
    # raises -- the forward is exact either way, and the symptom is that multi-card converges
    # as if the learning rate were `degree` times smaller.
    #
    # **The logged loss is therefore `degree` times larger than a degree-1 run's.** Divide by
    # `sp_degree` before comparing curves. The alternative -- correcting the gradients after
    # the backward instead, which would leave the number honest -- costs a pass over every
    # gradient in the model, ~33 GB of traffic a step for 16.5 B parameters, to make a log line
    # read better. Scaling the loss is free, so the cost is paid here in documentation instead.
    sp_degree: int = 1

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
    #              on the DiT blocks (19.2 M at rank 4). One GPU at 42.5 GB, 0.2 GB checkpoints.
    #   "full"   — the whole DiT incl. projectors (16.48 B), which is how upstream and RLinf train
    #              DROID. Needs fp32 master weights sharded across ~8 GPUs (57 GB each) and writes
    #              a 66 GB checkpoint per save.
    #
    # For a change of EMBODIMENT, "full" is the mode that works: fine-tuning the released DROID
    # checkpoint on SO-101 and scoring held-out episodes open-loop, `full` beat the "hold the
    # current state" baseline on 10 of 10 episodes (action MSE 79.7 vs 126.7) while ranks 4, 32 and
    # 128 lost on every one of them — raising the rank improved the predicted direction monotonically
    # but never enough. See the README for the table. Using "lora" on the SAME robot, which is what
    # upstream ships it for, is untested here.
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
        if self.sp_degree < 1:
            raise ValueError(f"sp_degree must be at least 1, got {self.sp_degree}.")
        if self.sp_degree > 1:
            heads = _DIT_PRESETS[self.model_variant]["num_heads"]
            if heads % self.sp_degree:
                raise ValueError(
                    f"sp_degree must divide the {heads} attention heads of {self.model_variant}; "
                    f"got {self.sp_degree}."
                )
        if not isinstance(self.gradient_checkpointing, bool):
            # Clamping instead would hide a typo: a value above the depth reads as "all of them"
            # and would silently cost a quarter of the step, which is the thing the dial exists
            # to avoid paying by accident.
            depth = _DIT_PRESETS[self.model_variant]["num_layers"]
            if not 0 <= self.gradient_checkpointing <= depth:
                raise ValueError(
                    f"gradient_checkpointing must be True, False, or a count in [0, {depth}] "
                    f"for {self.model_variant}; got {self.gradient_checkpointing}."
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

    def training_window(self) -> dict[str, int]:
        """Every shape one training sample presents to the head, derived from this config.

        `WANPolicyHead.estimate_flops` takes exactly these as keyword arguments, so the geometry
        is computed once, here, next to the `_validate_video_geometry` that already enforces it —
        rather than a second time in whatever tool wants a FLOPs count.

        The two token counts come from the asserts in `WANPolicyHead.forward`, which are the
        authority on how a window is laid out: one action register per `num_frame_per_block`
        frames carries `num_action_per_block` action tokens, and the state registers are strided
        against the same blocks. The first latent frame is the i2v conditioning frame and carries
        no registers, hence `latent_frames - 1`.
        """
        latent_frames = (self.num_frames - 1) // _VAE_TEMPORAL_COMPRESSION + 1
        canvas_h = 2 * self.per_view_height
        canvas_w = 2 * self.per_view_width
        blocks = latent_frames - 1
        return {
            "canvas_height": canvas_h,
            "canvas_width": canvas_w,
            "raw_frames": self.num_frames,
            "latent_frames": latent_frames,
            "latent_height": canvas_h // _VAE_SPATIAL_COMPRESSION,
            "latent_width": canvas_w // _VAE_SPATIAL_COMPRESSION,
            "action_tokens": blocks * (self.num_action_per_block // self.num_frame_per_block),
            "state_tokens": blocks // (self.num_frame_per_block // self.num_state_per_block),
        }

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
            # Ascend's fused ApplyAdamW takes beta2 in the parameter dtype, and bfloat16 has no
            # value between 0.99609375 and 1.0, so this recipe's 0.999 arrives as exactly 1.0 and
            # the second-moment bias correction divides by zero. Measured on a 950PR: LoRA loss is
            # 0.066 at step 1 and NaN from step 2 on, where torch.optim.AdamW continues 0.088,
            # 0.058, 0.107. Passing fp32 scalars instead is rejected outright, so the dtype of the
            # parameters is what decides it -- the same bf16 spacing that costs a bf16 master
            # weight ~84% of its updates at lr 1e-5, which is why a full fine-tune keeps fp32.
            #
            # Keyed on the dtype rather than on `training_mode` because `compute_dtype` defaults to
            # bfloat16 and a full fine-tune only gets fp32 by passing the flag; a full run that
            # forgot it would otherwise hand bf16 parameters straight back to the kernel.
            #
            # LoRA loses nothing by sitting this out: it trains 108.6 M of 22.9 B, so the optimizer
            # measured 0.4% of a 10.8 s step, inside the run-to-run noise. A full fine-tune trains
            # 16,484 M and has 150x more to gain.
            allow_npu_fused=(self.compute_dtype == "float32"),
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
