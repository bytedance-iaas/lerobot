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

Faithful port of DreamZero's ``ComposedModalityTransform`` inference path for the
oxe_droid / DROID embodiment (see ``dreamzero_port_plan.md`` M2 and the upstream
sources cited inline). The preprocessor turns a LeRobot observation into DreamZero's
action-input dict (``images`` / ``text`` / ``text_attention_mask`` / ``text_negative``
/ ``text_attention_mask_negative`` / ``state`` / ``embodiment_id``); the postprocessor
unnormalizes the predicted chunk and converts relative joint deltas back to absolute.

Numerics (upstream ``groot/vla/data/transform/state_action.py`` + ``model/dreamzero/
transform/dreamzero_cotrain.py``):
  - **video** is emitted as uint8; ``/255`` + Normalize(0.5, 0.5) → [-1, 1] and the resize
    to ``(target_video_height, target_video_width)`` happen INSIDE the model, not here. This
    step only stitches the per-embodiment multi-view canvas.
  - **state**: per-key q99 normalize to [-1, 1] (``2*(x-q01)/(q99-q01)-1``, clamp; q01==q99 ⇒
    passthrough), concat in modality order, pad to ``max_state_dim`` with a ``state_mask``.
  - **action decode**: per-key q99 inverse (``(x+1)/2*(q99-q01)+q01``); ``joint_position`` uses
    the RELATIVE stats and is converted back to absolute by adding the last observed joint state
    (mirrors ``GrootSimPolicy.unapply``); ``gripper_position`` uses ABSOLUTE stats and is left
    as-is (no binarization).
  - **text**: umt5 tokenizer, ``max_length=512``, the oxe_droid multi-view language template, and
    the fixed CFG negative prompt.

The per-embodiment stitch + language template are implemented for oxe_droid (the released
DreamZero-DROID checkpoint); other embodiments raise ``NotImplementedError`` until ported.
The exact per-key dims + statistics come from the checkpoint (``statistics.json`` /
``meta/modality.json``), supplied through ``dataset_stats`` by the checkpoint converter.
"""

import logging
from typing import Any

import numpy as np
import torch

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
from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_dreamzero import DreamZeroConfig

logger = logging.getLogger(__name__)

# oxe_droid multi-view language template (upstream dreamzero_cotrain.py `collate`, the
# embodiment_id == 17 branch). The raw instruction is lowercased and wrapped twice.
_DROID_LANG_PREFIX = "A multi-view video shows that a robot "
_DROID_LANG_MID = (
    " The video is split into three views: The top view shows the camera view from the robot's "
    "wrist, the bottom-left view shows the camera view from the left exterior camera, and the "
    "bottom-right view shows the camera view from the right exterior camera. During training, one "
    "of the two bottom exterior views may be a black screen (dropped view). The robot "
)
# Fixed default when the observation carries no instruction (transform/dreamzero_cotrain.yaml).
_DEFAULT_INSTRUCTION = "Perform the default behavior."


# ----------------------------------------------------------------------------------------------
# q99 normalization (upstream Normalizer, mode "q99")
# ----------------------------------------------------------------------------------------------
def _q99_forward(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """2*(x-q01)/(q99-q01) - 1, clamped to [-1, 1]; passthrough where q01 == q99."""
    q01 = q01.to(dtype=x.dtype, device=x.device)
    q99 = q99.to(dtype=x.dtype, device=x.device)
    span = q99 - q01
    mask = span != 0
    safe = torch.where(mask, span, torch.ones_like(span))
    normalized = 2 * (x - q01) / safe - 1
    normalized = torch.where(mask, normalized, x)
    return normalized.clamp(-1.0, 1.0)


def _q99_inverse(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """(x+1)/2 * (q99-q01) + q01."""
    q01 = q01.to(dtype=x.dtype, device=x.device)
    q99 = q99.to(dtype=x.dtype, device=x.device)
    return (x + 1) / 2 * (q99 - q01) + q01


def _stat_vec(stats: dict, modality: str, key: str, name: str) -> np.ndarray | None:
    """Pull a 1-D stat (e.g. q01) for ``modality.key`` from the nested stats dict."""
    group = (stats or {}).get(modality)
    if not isinstance(group, dict):
        return None
    entry = group.get(key)
    if not isinstance(entry, dict) or name not in entry:
        return None
    return np.asarray(entry[name], dtype=np.float32).reshape(-1)


class _QuantileLayout:
    """Concatenated q01/q99 tensors + per-key [start, end) slices for one modality group.

    Built from the checkpoint stats so the processor normalizes/denormalizes a single
    concatenated vector while still knowing where each modality key lives.
    """

    def __init__(self, stats: dict, modality: str, keys: tuple[str, ...]):
        self.keys = tuple(keys)
        self.slices: dict[str, tuple[int, int]] = {}
        q01_parts, q99_parts = [], []
        start = 0
        for key in self.keys:
            q01 = _stat_vec(stats, modality, key, "q01")
            q99 = _stat_vec(stats, modality, key, "q99")
            if q01 is None or q99 is None:
                # No stats for this key (e.g. tiny CPU-test config) → skip it in the layout.
                continue
            dim = int(q01.shape[0])
            self.slices[key] = (start, start + dim)
            q01_parts.append(q01)
            q99_parts.append(q99)
            start += dim
        self.dim = start
        self.q01 = torch.from_numpy(np.concatenate(q01_parts)) if q01_parts else torch.empty(0)
        self.q99 = torch.from_numpy(np.concatenate(q99_parts)) if q99_parts else torch.empty(0)


# ----------------------------------------------------------------------------------------------
# video: per-view crop/resize + per-embodiment stitch
# ----------------------------------------------------------------------------------------------
def _to_bthwc_uint8(video: torch.Tensor) -> torch.Tensor:
    """Coerce a LeRobot image tensor to (B, T, H, W, C) uint8 in [0, 255]."""
    v = video
    if v.ndim == 4:  # (B, C, H, W) or (B, H, W, C) → add a length-1 time axis
        v = v.unsqueeze(1)
    if v.ndim != 5:
        raise ValueError(f"DreamZero expects image tensors of rank 4/5, got shape {tuple(video.shape)}")
    # Channels-first (…C,H,W with C in {1,3}) → channels-last.
    if v.shape[2] in (1, 3) and v.shape[-1] not in (1, 3):
        v = v.permute(0, 1, 3, 4, 2)
    if v.dtype != torch.uint8:
        # Float images are assumed in [0, 1] (LeRobot's convention).
        v = (v.float().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return v.contiguous()


def _crop_resize_view(view: torch.Tensor, crop_scale: float, out_h: int, out_w: int) -> torch.Tensor:
    """Center-crop by ``crop_scale`` then bilinear-resize to (out_h, out_w). (B,T,H,W,C) uint8."""
    b, t, h, w, c = view.shape
    ch, cw = int(h * crop_scale), int(w * crop_scale)
    top, left = (h - ch) // 2, (w - cw) // 2
    view = view[:, :, top : top + ch, left : left + cw, :]
    # interpolate wants (N, C, H, W) float.
    x = view.reshape(b * t, ch, cw, c).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True)
    x = x.round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).reshape(b, t, out_h, out_w, c).contiguous()


def _stitch_oxe_droid(views: list[torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """Stitch [exterior_1, exterior_2, wrist] into DreamZero's 2H x 2W DROID canvas.

    Layout (dreamzero_cotrain.py `_prepare_video`, OXE_DROID branch): top row = wrist repeated
    across the full width; bottom-left = exterior_1; bottom-right = exterior_2. Returns
    (B, T, 2*out_h, 2*out_w, 3) uint8.
    """
    ext1, ext2, wrist = views[0], views[1], views[2]
    b, t = ext1.shape[0], ext1.shape[1]
    canvas = torch.zeros((b, t, 2 * out_h, 2 * out_w, 3), dtype=torch.uint8, device=ext1.device)
    canvas[:, :, :out_h, :, :] = wrist.repeat(1, 1, 1, 2, 1)  # top: wrist spans 2W
    canvas[:, :, out_h:, :out_w, :] = ext1  # bottom-left
    canvas[:, :, out_h:, out_w:, :] = ext2  # bottom-right
    return canvas


# ----------------------------------------------------------------------------------------------
# pack step
# ----------------------------------------------------------------------------------------------
@ProcessorStepRegistry.register("dreamzero_pack_inputs")
class DreamZeroPackInputsStep(ProcessorStep):
    """Pack a LeRobot observation into DreamZero's action-input dict (oxe_droid inference path)."""

    def __init__(self, config: DreamZeroConfig, stats: dict | None = None):
        self.config = config
        self.stats = stats or {}
        self._state_layout = _QuantileLayout(self.stats, "state", config.state_modality_keys)
        self._action_layout = _QuantileLayout(self.stats, "action", config.action_modality_keys)
        # Raw (unnormalized) last observed state per modality key — reference for relative decode.
        self._last_raw_state: dict[str, torch.Tensor] = {}
        self._tokenizer = None  # lazy umt5

    # -- helpers ----------------------------------------------------------------------------
    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self.config.text_encoder_pretrained_path or "google/umt5-xxl"
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def _tokenize(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self._get_tokenizer()
        enc = tok(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _ordered_image_keys(self, obs: dict) -> list[str]:
        if self.config.image_view_order:
            keys = []
            for k in self.config.image_view_order:
                cand = k if k in obs else f"{OBS_IMAGES}.{k}"
                if cand in obs:
                    keys.append(cand)
            return keys
        keys = sorted(k for k in obs if isinstance(k, str) and k.startswith(f"{OBS_IMAGES}."))
        if not keys and OBS_IMAGE in obs:
            keys = [OBS_IMAGE]
        return keys

    def _build_language(self, task: Any, bsz: int) -> list[str]:
        if isinstance(task, (list, tuple)):
            raw = [str(t) if t else _DEFAULT_INSTRUCTION for t in task]
        else:
            raw = [str(task) if task else _DEFAULT_INSTRUCTION] * bsz
        if self.config.embodiment_tag != "oxe_droid":
            raise NotImplementedError(
                f"DreamZero language template is only ported for oxe_droid; got {self.config.embodiment_tag!r}."
            )
        out = []
        for instr in raw:
            low = instr.lower()
            out.append(_DROID_LANG_PREFIX + low + _DROID_LANG_MID + low)
        return out

    # -- main -------------------------------------------------------------------------------
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.config.embodiment_tag != "oxe_droid":
            raise NotImplementedError(
                "DreamZero preprocessing is only ported for the oxe_droid/DROID embodiment "
                f"(view stitching + language template); got {self.config.embodiment_tag!r}."
            )
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        device = next((v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))

        # ---- VIDEO: per-view crop+resize then oxe_droid stitch → images (B,T,2h,2w,3) uint8 ----
        img_keys = self._ordered_image_keys(obs)
        if len(img_keys) < 3:
            raise ValueError(
                f"DreamZero (oxe_droid) needs 3 camera views [exterior_1, exterior_2, wrist]; "
                f"found {len(img_keys)} image key(s): {img_keys}. Set config.image_view_order."
            )
        views = [
            _crop_resize_view(
                _to_bthwc_uint8(obs[k]),
                self.config.view_crop_scale,
                self.config.per_view_height,
                self.config.per_view_width,
            )
            for k in img_keys[:3]
        ]
        obs["images"] = _stitch_oxe_droid(views, self.config.per_view_height, self.config.per_view_width)
        for k in [key for key in obs if isinstance(key, str) and key.startswith(OBS_IMAGES)]:
            obs.pop(k, None)
        obs.pop(OBS_IMAGE, None)

        # ---- STATE: q99 normalize (absolute), concat, pad to max_state_dim, + state_mask ----
        if OBS_STATE in obs:
            state = obs[OBS_STATE]
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz = state.shape[0]
            self._cache_raw_state(state)
            norm = state.clone().float()
            for _key, (s, e) in self._state_layout.slices.items():
                if e <= state.shape[1]:
                    q01 = self._state_layout.q01[s:e]
                    q99 = self._state_layout.q99[s:e]
                    norm[:, s:e] = _q99_forward(state[:, s:e].float(), q01, q99)
            real_dim = norm.shape[1]
            pad = self.config.max_state_dim - real_dim
            if pad < 0:
                raise ValueError(f"state dim {real_dim} exceeds max_state_dim {self.config.max_state_dim}")
            state_padded = torch.nn.functional.pad(norm, (0, pad)).unsqueeze(1)  # (B, 1, max_state_dim)
            state_mask = torch.zeros(bsz, 1, self.config.max_state_dim, dtype=torch.bool, device=norm.device)
            state_mask[:, :, :real_dim] = True
            obs["state"] = state_padded
            obs["state_mask"] = state_mask
            obs.pop(OBS_STATE, None)
        else:
            bsz = obs["images"].shape[0]

        # ---- ACTION (training only): relative-encode joints + q99 normalize + pad + masks ----
        # Present only when the transition carries a ground-truth action chunk (lerobot-train).
        # The forward mirror of DreamZeroActionDecodeStep: joint_position is made relative to the
        # last observed state then q99-normalized with RELATIVE stats; gripper stays absolute.
        # Outputs land in the observation dict so they reach DreamZeroPolicy.forward's action_input
        # (action / action_mask / has_real_action) — the exact batch flattening is validated on the
        # first real training run (dreamzero_port_plan.md M3).
        action = transition.get(TransitionKey.ACTION)
        if action is not None and self._action_layout.dim > 0:
            real_dim = self._action_layout.dim
            was_2d = action.dim() == 2
            a = (action.unsqueeze(1) if was_2d else action)[..., :real_dim].clone().float()  # (B,T,D)
            rel_keys = set(self.config.relative_action_keys or ())
            if rel_keys and self._last_raw_state:
                reference = torch.zeros(a.shape[0], real_dim, dtype=a.dtype, device=a.device)
                mask = [False] * real_dim
                for key, (s, e) in self._action_layout.slices.items():
                    if key in rel_keys and key in self._last_raw_state:
                        ref = self._last_raw_state[key].to(a.device, a.dtype)
                        reference[:, s:e] = ref[:, : e - s]
                        for i in range(s, e):
                            mask[i] = True
                a = to_relative_actions(a, reference, mask)
            a = _q99_forward(a, self._action_layout.q01, self._action_layout.q99)
            a_padded = torch.nn.functional.pad(a, (0, self.config.max_action_dim - real_dim))
            action_mask = torch.zeros(*a_padded.shape, dtype=torch.bool, device=a_padded.device)
            action_mask[..., :real_dim] = True
            obs["action"] = a_padded
            obs["action_mask"] = action_mask
            obs["has_real_action"] = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
            transition[TransitionKey.ACTION] = a_padded.squeeze(1) if was_2d else a_padded

        # ---- TEXT: umt5 tokenize (template + fixed negative prompt) ----
        texts = self._build_language(comp.get("task"), bsz)
        text_ids, text_mask = self._tokenize(texts, device)
        neg_ids, neg_mask = self._tokenize([self.config_negative_prompt] * bsz, device)
        obs["text"] = text_ids
        obs["text_attention_mask"] = text_mask
        obs["text_negative"] = neg_ids
        obs["text_attention_mask_negative"] = neg_mask

        # ---- embodiment_id (projector index) ----
        obs["embodiment_id"] = torch.full(
            (bsz,), self.config.embodiment_projector_index, dtype=torch.long, device=device
        )

        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def _cache_raw_state(self, state: torch.Tensor) -> None:
        """Cache the raw (unnormalized) per-key state — the reference for relative→absolute decode."""
        grouped = {}
        for key, (s, e) in self._state_layout.slices.items():
            if e <= state.shape[1]:
                grouped[key] = state[:, s:e].detach().float()
        if grouped:
            self._last_raw_state = grouped

    @property
    def config_negative_prompt(self) -> str:
        # Kept as an attribute lookup so a checkpoint can override the fixed prompt if needed.
        from .configuration_dreamzero import DREAMZERO_NEGATIVE_PROMPT

        return getattr(self.config, "negative_prompt", None) or DREAMZERO_NEGATIVE_PROMPT

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(self, features):
        return features


# ----------------------------------------------------------------------------------------------
# action decode step
# ----------------------------------------------------------------------------------------------
@ProcessorStepRegistry.register("dreamzero_action_decode")
class DreamZeroActionDecodeStep(ProcessorStep):
    """Unnormalize the predicted chunk (q99 inverse) and convert relative joints back to absolute.

    ``joint_position`` uses the RELATIVE stats and is made absolute by adding the last observed
    joint state (cached by the paired :class:`DreamZeroPackInputsStep`); ``gripper_position`` uses
    ABSOLUTE stats and is left unchanged. Mirrors ``GrootSimPolicy.unapply``.
    """

    def __init__(self, config: DreamZeroConfig, stats: dict | None = None):
        self.config = config
        self.stats = stats or {}
        self._action_layout = _QuantileLayout(self.stats, "action", config.action_modality_keys)
        # Set by make_dreamzero_pre_post_processors so decode can read the pack step's cached state.
        self.pack_step: DreamZeroPackInputsStep | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or self._action_layout.dim == 0:
            return transition

        real_dim = self._action_layout.dim
        was_2d = action.dim() == 2
        if was_2d:
            action = action.unsqueeze(1)  # (B, 1, D)
        # Drop the zero-padding beyond the real action layout.
        action = action[..., :real_dim].clone().float()

        # q99 inverse over the concatenated action vector (joint stats are RELATIVE, gripper ABSOLUTE).
        action = _q99_inverse(action, self._action_layout.q01, self._action_layout.q99)

        # relative → absolute: add the last observed state for relative_action_keys only.
        rel_keys = set(self.config.relative_action_keys or ())
        if rel_keys and self.pack_step is not None and self.pack_step._last_raw_state:
            reference = torch.zeros(action.shape[0], real_dim, dtype=action.dtype, device=action.device)
            mask = [False] * real_dim
            for key, (s, e) in self._action_layout.slices.items():
                if key in rel_keys and key in self.pack_step._last_raw_state:
                    ref = self.pack_step._last_raw_state[key].to(action.device, action.dtype)
                    reference[:, s:e] = ref[:, : e - s]
                    for i in range(s, e):
                        mask[i] = True
            action = to_absolute_actions(action, reference, mask)

        if was_2d:
            action = action.squeeze(1)
        transition[TransitionKey.ACTION] = action
        return transition

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

    Structure mirrors GR00T: rename → add batch dim → pack → device; decode → device. The decode
    step is wired to the pack step so it can read the last observed state for the relative→absolute
    action conversion (the same pattern as GR00T's relative/absolute step reconnection).
    """
    pack_step = DreamZeroPackInputsStep(config=config, stats=dataset_stats)
    decode_step = DreamZeroActionDecodeStep(config=config, stats=dataset_stats)
    decode_step.pack_step = pack_step

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        pack_step,
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        decode_step,
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


def make_dreamzero_pre_post_processors_from_pretrained(
    config: DreamZeroConfig,
    pretrained_path,
    dataset_stats: dict | None = None,
    **kwargs,
):
    """Build the DreamZero processors, reading ``statistics.json`` from the checkpoint.

    The converter writes ``statistics.json`` as ``{embodiment_tag: {state, action}}`` (GEAR q01/q99).
    When ``dataset_stats`` isn't supplied, load the block for ``config.embodiment_tag`` so the q99
    normalization + relative decode use the checkpoint's own statistics (mirrors GR00T's
    ``make_groot_pre_post_processors_from_pretrained``).
    """
    import json
    from pathlib import Path

    stats = dataset_stats
    if stats is None:
        stats_file = Path(pretrained_path) / "statistics.json"
        if stats_file.exists():
            all_stats = json.loads(stats_file.read_text())
            block = all_stats.get(config.embodiment_tag)
            stats = block if isinstance(block, dict) else all_stats
        else:
            logger.warning(
                "No statistics.json in %s — DreamZero q99 normalization/decode will have no stats.",
                pretrained_path,
            )
    return make_dreamzero_pre_post_processors(config, dataset_stats=stats, **kwargs)
