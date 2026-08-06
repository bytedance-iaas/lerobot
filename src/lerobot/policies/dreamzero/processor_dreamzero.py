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

oxe_droid keeps its bespoke three-view stitch and fixed prompt (the released DreamZero-DROID
checkpoint); every other embodiment goes through upstream's generic 2x2 canvas, whose spare
quadrants are blacked out, with a prompt generated to describe that layout.
The exact per-key dims + statistics come from the checkpoint (``statistics.json`` /
``experiment_cfg/metadata.json``), or from ``scripts/compute_statistics.py`` for a new dataset.
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

# Wording for the generic (non-DROID) canvas, following upstream's agibot/yam/xdof prompts: the
# same "A multi-view video shows that a robot <instr> The video is split into four views: ...
# The robot <instr>" frame, with the quadrant descriptions swapped for the ones this embodiment
# actually fills. The prompt is the only thing that tells the model which quadrant is which
# camera, so it has to track `_stitch_generic`'s slot order.
_GENERIC_LANG_PREFIX = "A multi-view video shows that a robot "
_GENERIC_LANG_SPLIT = " The video is split into four views: "


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


def _crop_size(h: int, w: int, crop_scale: float) -> tuple[int, int]:
    """Upstream's crop size: ``(int(h*scale), int(w*scale))`` (VideoCrop.get_transform)."""
    return int(h * crop_scale), int(w * crop_scale)


def _sample_crop_offset(h: int, w: int, crop_scale: float, generator=None) -> tuple[int, int]:
    """A random (top, left) for training, matching torchvision ``RandomCrop``.

    Upstream concatenates every view and frame into one batch and transforms it once, so a single
    offset is shared across views and time — cropping views independently would misalign the
    stitched canvas against what the model was trained on.
    """
    ch, cw = _crop_size(h, w, crop_scale)
    top = int(torch.randint(0, h - ch + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - cw + 1, (1,), generator=generator).item())
    return top, left


def _crop_resize_view(
    view: torch.Tensor,
    crop_scale: float,
    out_h: int,
    out_w: int,
    crop_offset: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Crop by ``crop_scale`` then bilinear-resize to (out_h, out_w). (B,T,H,W,C) uint8.

    ``crop_offset`` selects the crop's top-left corner; None means centre (upstream's eval mode).
    """
    b, t, h, w, c = view.shape
    ch, cw = _crop_size(h, w, crop_scale)
    top, left = crop_offset if crop_offset is not None else ((h - ch) // 2, (w - cw) // 2)
    view = view[:, :, top : top + ch, left : left + cw, :]
    # interpolate wants (N, C, H, W) float.
    x = view.reshape(b * t, ch, cw, c).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(
        x, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True
    )
    x = x.round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).reshape(b, t, out_h, out_w, c).contiguous()


def _color_jitter_views(views: list[torch.Tensor], params: dict) -> list[torch.Tensor]:
    """Apply one shared ColorJitter to every view, as upstream's single batched transform does.

    Runs after the resize and before the stitch (upstream order: crop -> resize -> jitter ->
    concat). torchvision samples one factor per call, so the views are jittered identically.
    ``ColorJitter.get_params`` draws from torch's global RNG, so ``--seed`` governs it.
    """
    from torchvision.transforms import ColorJitter, functional as tvf

    jitter = ColorJitter(**params)
    # get_params returns the sampled factors, so the same ones can be replayed on every view.
    idx, brightness, contrast, saturation, hue = ColorJitter.get_params(
        jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
    )

    def apply(view: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = view.shape
        x = view.reshape(b * t, h, w, c).permute(0, 3, 1, 2)  # (N, C, H, W) uint8
        for i in idx:
            if i == 0 and brightness is not None:
                x = tvf.adjust_brightness(x, brightness)
            elif i == 1 and contrast is not None:
                x = tvf.adjust_contrast(x, contrast)
            elif i == 2 and saturation is not None:
                x = tvf.adjust_saturation(x, saturation)
            elif i == 3 and hue is not None:
                x = tvf.adjust_hue(x, hue)
        return x.permute(0, 2, 3, 1).reshape(b, t, h, w, c).contiguous()

    return [apply(v) for v in views]


def _stitch_oxe_droid(views: list[torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """Stitch [exterior_1, exterior_2, wrist] into DreamZero's 2H x 2W DROID canvas.

    Layout: top row = the wrist view stretched across the full width; bottom-left = exterior_1;
    bottom-right = exterior_2. Returns (B, T, 2*out_h, 2*out_w, 3) uint8.

    The wrist view is widened with ``repeat_interleave`` — a nearest-neighbour 2x horizontal
    upscale (``[a,b,c] -> [a,a,b,b,c,c]``), matching upstream's ``np.repeat(wrist, 2, axis=-1)``.
    ``Tensor.repeat`` would TILE it instead (``[a,b,c] -> [a,b,c,a,b,c]``), putting two copies of
    the wrist camera side by side — a plausible-looking canvas that is not what the model was
    trained on.
    """
    ext1, ext2, wrist = views[0], views[1], views[2]
    b, t = ext1.shape[0], ext1.shape[1]
    canvas = torch.zeros((b, t, 2 * out_h, 2 * out_w, 3), dtype=torch.uint8, device=ext1.device)
    canvas[:, :, :out_h, :, :] = torch.repeat_interleave(wrist, 2, dim=3)  # top: wrist spans 2W
    canvas[:, :, out_h:, :out_w, :] = ext1  # bottom-left
    canvas[:, :, out_h:, out_w:, :] = ext2  # bottom-right
    return canvas


# Quadrants of the generic canvas, in the order views are assigned to them. Upstream fills
# top-left, then bottom-left, then top-right, and leaves bottom-right black — so a robot with two
# cameras occupies the left column and the right column stays black. `_GENERIC_QUADRANT_NAMES`
# names them for the language template, which has to describe the layout the model is looking at.
_GENERIC_QUADRANT_NAMES = ("top-left", "bottom-left", "top-right", "bottom-right")


def _stitch_generic(views: list[torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """Stitch up to four views into the 2H x 2W canvas, black-filling the unused quadrants.

    This is upstream's non-DROID layout (`DreamZeroCotrainTransform._prepare_video`, the branch
    after the `OXE_DROID` special case), which every other embodiment it ships — agibot, yam,
    xdof — goes through. Views land top-left, bottom-left, top-right; anything left over stays
    black, exactly as upstream's prompts describe ("the bottom-right view is a black screen").

    Filling the spare quadrants with black rather than, say, tiling two views across the full
    canvas is deliberate: the canvas size is pinned by `frame_seqlen` and cannot shrink, and black
    is the padding the model was trained to see there.
    """
    if not 1 <= len(views) <= 4:
        raise ValueError(f"The DreamZero canvas holds 1-4 views, got {len(views)}.")
    b, t = views[0].shape[0], views[0].shape[1]
    canvas = torch.zeros((b, t, 2 * out_h, 2 * out_w, 3), dtype=torch.uint8, device=views[0].device)
    slots = [
        (slice(None, out_h), slice(None, out_w)),  # top-left
        (slice(out_h, None), slice(None, out_w)),  # bottom-left
        (slice(None, out_h), slice(out_w, None)),  # top-right
        (slice(out_h, None), slice(out_w, None)),  # bottom-right
    ]
    for view, (rows, cols) in zip(views, slots, strict=False):
        canvas[:, :, rows, cols, :] = view
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
        # Per-macro-block raw anchor states, set on a training window (see _block_anchor_rows).
        self._block_anchor_state: dict[str, torch.Tensor] = {}
        # Augmentation RNG; left as the global generator so `--seed` governs it.
        self._generator = None
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
        if self.config.video_modality_keys:
            keys, missing = [], []
            for k in self.config.video_modality_keys:
                cand = k if k in obs else f"{OBS_IMAGES}.{k}"
                (keys if cand in obs else missing).append(cand if cand in obs else k)
            if missing:
                # Silently dropping a configured view would stitch a canvas that no longer matches
                # the prompt built from `video_modality_keys` — same shape, wrong content — so a
                # typo'd or renamed camera key must fail here instead.
                available = sorted(
                    k for k in obs if isinstance(k, str) and k.startswith((f"{OBS_IMAGES}.", OBS_IMAGE))
                )
                raise KeyError(
                    f"config.video_modality_keys names camera view(s) {missing} that are not in the "
                    f"observation. Available image keys: {available}. Fix video_modality_keys or the "
                    "rename_map."
                )
            return keys
        # `_is_pad` window flags share the observation.images prefix in a training batch and must
        # not be mistaken for camera views.
        keys = sorted(
            k
            for k in obs
            if isinstance(k, str) and k.startswith(f"{OBS_IMAGES}.") and not k.endswith("_is_pad")
        )
        if not keys and OBS_IMAGE in obs:
            keys = [OBS_IMAGE]
        return keys

    def _build_language(self, task: Any, bsz: int, n_views: int | None = None) -> list[str]:
        if isinstance(task, (list, tuple)):
            raw = [str(t) if t else _DEFAULT_INSTRUCTION for t in task]
        else:
            raw = [str(task) if task else _DEFAULT_INSTRUCTION] * bsz
        if self.config.embodiment_tag == "oxe_droid":
            return [_DROID_LANG_PREFIX + i.lower() + _DROID_LANG_MID + i.lower() for i in raw]
        middle = _GENERIC_LANG_SPLIT + self._describe_canvas(n_views) + " The robot "
        return [_GENERIC_LANG_PREFIX + i.lower() + middle + i.lower() for i in raw]

    def _describe_canvas(self, n_views: int | None = None) -> str:
        """Sentence naming what sits in each quadrant of the generic canvas.

        The camera names come from `view_descriptions` when set, else from `video_modality_keys` with
        the LeRobot key prefix stripped — so `observation.images.wrist` reads as "the wrist
        camera". Quadrants with no view are described as black screens, matching both
        `_stitch_generic` and upstream's own prompts for embodiments with fewer than four cameras.
        """
        names = list(self.config.view_descriptions or ()) or [
            k.rsplit(".", 1)[-1].replace("_", " ") for k in (self.config.video_modality_keys or ())
        ]
        if not names:
            raise ValueError(
                f"Embodiment {self.config.embodiment_tag!r} needs `video_modality_keys` (or "
                "`view_descriptions`) so the prompt can say which camera is in which quadrant; "
                "the model has no other way to know the layout."
            )
        if n_views is not None and len(names) != n_views:
            # A name count that disagrees with the stitched view count would either misname
            # quadrants or call occupied ones black screens — a prompt/canvas mismatch the model
            # cannot detect. (`__call__` passes the actual view count; the config-level length
            # check cannot see it when the views come from sorted observation keys.)
            raise ValueError(
                f"The prompt has {len(names)} camera name(s) {names} but the canvas holds "
                f"{n_views} view(s); set `video_modality_keys`/`view_descriptions` to match the "
                "observation's cameras."
            )
        # Upstream capitalises only the first clause of the list ("...four views: The top-left
        # view shows..., the top-right view shows...").
        described = [
            f"{'The' if i == 0 else 'the'} {quadrant} view shows the camera view from the {name}"
            for i, (quadrant, name) in enumerate(zip(_GENERIC_QUADRANT_NAMES, names, strict=False))
        ]
        blank = list(_GENERIC_QUADRANT_NAMES[len(names) :])
        if len(blank) == 1:
            described.append(f"and the {blank[0]} view is a black screen (inactive view)")
        elif blank:
            described.append(
                "and the " + " and ".join(f"{q}" for q in blank) + " views are black screens (inactive views)"
            )
        return ", ".join(described) + "."

    # -- main -------------------------------------------------------------------------------
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        device = next((v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))

        # A training sample carries the whole window: `num_frames` observation rows and
        # `action_horizon * max_chunk_size` actions. Inference carries a single observation.
        # Detected up front because it also selects random-vs-centre crop for the video below.
        raw_state = obs.get(OBS_STATE)
        training = raw_state is not None and raw_state.dim() == 3 and raw_state.shape[1] > 1
        if training:
            self._assert_window_not_padded(transition, comp)

        # ---- VIDEO: per-view crop+resize then stitch → images (B,T,2h,2w,3) uint8 ----
        droid = self.config.embodiment_tag == "oxe_droid"
        # DROID's layout is the only one that assigns a specific meaning to each view, so it needs
        # exactly three; every other embodiment goes through the generic grid, which takes 1-4.
        min_views, max_views = (3, 3) if droid else (1, 4)
        img_keys = self._ordered_image_keys(obs)
        if not min_views <= len(img_keys) <= max_views:
            expected = (
                "3 camera views [exterior_1, exterior_2, wrist]"
                if droid
                else f"1-4 camera views (embodiment {self.config.embodiment_tag!r})"
            )
            raise ValueError(
                f"DreamZero needs {expected}; found {len(img_keys)} image key(s): {img_keys}. "
                "Set config.video_modality_keys."
            )
        raw_views = [_to_bthwc_uint8(obs[k]) for k in img_keys[:max_views]]
        # Upstream asserts every camera shares a resolution, then crops them as one batch.
        src_h, src_w = raw_views[0].shape[2], raw_views[0].shape[3]
        if any(v.shape[2:4] != (src_h, src_w) for v in raw_views):
            raise ValueError(
                "DreamZero crops all camera views with one shared offset, so they must share a "
                f"resolution; got {[tuple(v.shape[2:4]) for v in raw_views]}."
            )
        crop_offset = (
            _sample_crop_offset(src_h, src_w, self.config.view_crop_scale, self._generator)
            if training
            else None  # centre crop at inference, mirroring upstream's eval mode
        )
        views = [
            _crop_resize_view(
                v,
                self.config.view_crop_scale,
                self.config.per_view_height,
                self.config.per_view_width,
                crop_offset,
            )
            for v in raw_views
        ]
        if training and self.config.video_color_jitter:
            views = _color_jitter_views(views, dict(self.config.color_jitter_params))
        stitch = _stitch_oxe_droid if droid else _stitch_generic
        obs["images"] = stitch(views, self.config.per_view_height, self.config.per_view_width)
        for k in [key for key in obs if isinstance(key, str) and key.startswith(OBS_IMAGES)]:
            obs.pop(k, None)
        obs.pop(OBS_IMAGE, None)

        # ---- STATE: q99 normalize (absolute), pad to max_state_dim, + state_mask ----
        if raw_state is not None:
            # (B, D) at inference; (B, num_frames, D) in training -> keep one row per macro block.
            anchors = raw_state if training else raw_state.unsqueeze(1)
            if training:
                anchors = anchors[:, self._block_anchor_rows]  # (B, max_chunk_size, D)
            bsz = anchors.shape[0]
            self._cache_raw_state(anchors[:, 0])
            self._block_anchor_state = self._split_by_key(anchors)

            norm = anchors.clone().float()
            for _key, (s, e) in self._state_layout.slices.items():
                if e <= anchors.shape[-1]:
                    norm[..., s:e] = _q99_forward(
                        anchors[..., s:e].float(), self._state_layout.q01[s:e], self._state_layout.q99[s:e]
                    )
            real_dim = norm.shape[-1]
            pad = self.config.max_state_dim - real_dim
            if pad < 0:
                raise ValueError(f"state dim {real_dim} exceeds max_state_dim {self.config.max_state_dim}")
            state_mask = torch.zeros(
                bsz, norm.shape[1], self.config.max_state_dim, dtype=torch.bool, device=norm.device
            )
            state_mask[..., :real_dim] = True
            obs["state"] = torch.nn.functional.pad(norm, (0, pad))
            obs["state_mask"] = state_mask
            obs.pop(OBS_STATE, None)
        else:
            bsz = obs["images"].shape[0]

        # ---- ACTION (training only): per-block relative encode + q99 + pad + masks ----
        # Mirrors upstream's `_convert_to_relative_action`: each macro block of `action_horizon`
        # rows subtracts the RAW state at that block's own first row, and only then is q99
        # normalized (with the relative stats for `relative_action_keys`). Anchoring the whole
        # window on one state instead would make later blocks' displacements several times larger
        # than the quantiles they are normalized against.
        action = transition.get(TransitionKey.ACTION)
        if action is not None and self._action_layout.dim == 0:
            # Silently skipping the action branch here is how a stats-schema mismatch used to
            # surface: the model then failed much later on a missing `has_real_action`.
            raise ValueError(
                "A ground-truth action is present but DreamZero has no action statistics, so it "
                f"cannot be normalized. Expected q01/q99 for {list(self.config.action_modality_keys)} "
                "under an 'action' key — check that the checkpoint's statistics were loaded."
            )
        if action is not None:
            real_dim = self._action_layout.dim
            was_2d = action.dim() == 2
            a = (action.unsqueeze(1) if was_2d else action)[..., :real_dim].clone().float()  # (B,T,D)
            a = self._encode_relative_actions(a, training=training)
            a = _q99_forward(a, self._action_layout.q01, self._action_layout.q99)
            a_padded = torch.nn.functional.pad(a, (0, self.config.max_action_dim - real_dim))
            action_mask = torch.zeros(*a_padded.shape, dtype=torch.bool, device=a_padded.device)
            action_mask[..., :real_dim] = True
            obs["action"] = a_padded
            obs["action_mask"] = action_mask
            obs["has_real_action"] = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
            transition[TransitionKey.ACTION] = a_padded.squeeze(1) if was_2d else a_padded

        # ---- TEXT: umt5 tokenize (template + fixed negative prompt) ----
        texts = self._build_language(comp.get("task"), bsz, n_views=len(img_keys))
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

    @property
    def _block_anchor_rows(self) -> list[int]:
        """Observation rows that start each macro block.

        Observation rows sit at raw offsets ``0, stride, 2*stride, ...``; block ``c`` starts at raw
        offset ``c * action_horizon``, i.e. row ``c * action_horizon // stride``. For DROID that is
        rows [0, 8, 16, 24] of the 33-row window.
        """
        step = self.config.action_horizon // self.config.video_frame_stride
        return [c * step for c in range(self.config.max_chunk_size)]

    def _split_by_key(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice a raw (unnormalized) state tensor into its per-modality-key components."""
        out = {}
        for key, (s, e) in self._state_layout.slices.items():
            if e <= rows.shape[-1]:
                out[key] = rows[..., s:e].detach().float()
        return out

    def _assert_window_not_padded(self, transition: EnvTransition, comp: dict) -> None:
        """Refuse a padded training window.

        `drop_n_last_frames` should make short windows unreachable, so a set `*_is_pad` flag means
        the window contract and the sampler disagree — and padded rows would train the model on
        repeated frames without any mask saying so.
        """
        source = {**(comp or {}), **(transition.get(TransitionKey.OBSERVATION) or {})}
        padded = [
            key
            for key, value in source.items()
            if isinstance(key, str) and key.endswith("_is_pad") and torch.as_tensor(value).any()
        ]
        if padded:
            raise ValueError(
                f"DreamZero training windows must be complete, but these are padded: {padded}. "
                f"Check that the dataset was built with drop_n_last_frames="
                f"{self.config.drop_n_last_frames}."
            )

    def _encode_relative_actions(self, actions: torch.Tensor, *, training: bool) -> torch.Tensor:
        """Subtract each block's own anchor state from the relative action dims. (B, T, real_dim)."""
        rel_keys = set(self.config.relative_action_keys or ())
        anchors = self._block_anchor_state if training else None
        if not rel_keys or not (anchors or self._last_raw_state):
            return actions

        horizon = self.config.action_horizon
        n_blocks = self.config.max_chunk_size if training else 1
        real_dim = actions.shape[-1]
        for block in range(n_blocks):
            lo = block * horizon
            hi = min(lo + horizon, actions.shape[1])
            if lo >= actions.shape[1]:
                break
            reference = torch.zeros(actions.shape[0], real_dim, dtype=actions.dtype, device=actions.device)
            mask = [False] * real_dim
            for key, (s, e) in self._action_layout.slices.items():
                if key not in rel_keys:
                    continue
                if training and key in (anchors or {}):
                    ref = anchors[key][:, block]
                elif not training and key in self._last_raw_state:
                    ref = self._last_raw_state[key]
                else:
                    continue
                reference[:, s:e] = ref.to(actions.device, actions.dtype)[:, : e - s]
                for i in range(s, e):
                    mask[i] = True
            actions[:, lo:hi] = to_relative_actions(actions[:, lo:hi], reference, mask)
        return actions

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

    ``joint_position`` uses the RELATIVE stats and is made absolute by adding the observed joint
    state; ``gripper_position`` uses ABSOLUTE stats and is left unchanged. Mirrors
    ``GrootSimPolicy.unapply``.

    **Which** observed state matters: the model's chunk is relative to the frame the chunk was
    predicted from, so every action in it must be decoded against that one anchor. The paired
    :class:`DreamZeroPackInputsStep` caches the current state on every call, which is right for the
    frame that triggered the prediction and wrong for the frames that follow while the chunk is
    still being consumed. So this step latches the anchor for the life of a chunk instead of
    reading the pack step's latest value each time.

    Chunk length comes from ``config.n_action_steps`` because that is exactly what
    ``DreamZeroPolicy.select_action`` queues per prediction (``deque(maxlen=n_action_steps)``,
    refilled when empty). ``reset()`` drops the latch so a new episode re-anchors immediately.
    """

    def __init__(self, config: DreamZeroConfig, stats: dict | None = None):
        self.config = config
        self.stats = stats or {}
        self._action_layout = _QuantileLayout(self.stats, "action", config.action_modality_keys)
        # Set by make_dreamzero_pre_post_processors so decode can read the pack step's cached state.
        self.pack_step: DreamZeroPackInputsStep | None = None
        self._anchor: dict[str, torch.Tensor] = {}
        self._actions_left_in_chunk = 0

    def reset(self) -> None:
        """Forget the latched anchor so the next decode re-anchors on the current observation."""
        self._anchor = {}
        self._actions_left_in_chunk = 0

    def _anchor_for(self, n_actions: int) -> dict[str, torch.Tensor]:
        """Return the anchor for this decode, latching a fresh one when a chunk starts."""
        if self._actions_left_in_chunk <= 0:
            self._anchor = dict(self.pack_step._last_raw_state) if self.pack_step is not None else {}
            self._actions_left_in_chunk = max(int(self.config.n_action_steps), 1)
        self._actions_left_in_chunk -= n_actions
        return self._anchor

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

        # relative → absolute: add the chunk's anchor state for relative_action_keys only.
        rel_keys = set(self.config.relative_action_keys or ())
        anchor = self._anchor_for(action.shape[1])
        if rel_keys and anchor:
            reference = torch.zeros(action.shape[0], real_dim, dtype=action.dtype, device=action.device)
            mask = [False] * real_dim
            for key, (s, e) in self._action_layout.slices.items():
                if key in rel_keys and key in anchor:
                    ref = anchor[key].to(action.device, action.dtype)
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


# Standard override keys that `lerobot-train` always passes but a DreamZero pipeline has no step
# for: q99 normalization and the relative<->absolute action conversion both happen inside
# DreamZeroPackInputsStep / DreamZeroActionDecodeStep, driven by the checkpoint's own statistics
# rather than by dataset stats. Same situation as GR00T (_GROOT_ABSENT_STANDARD_OVERRIDE_KEYS).
_ABSENT_STANDARD_OVERRIDE_KEYS = frozenset(
    {
        "absolute_actions_processor",
        "normalizer_processor",
        "relative_actions_processor",
        "unnormalizer_processor",
    }
)


def _drop_absent_standard_overrides(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip standard override keys a DreamZero pipeline has no step for.

    Everything else still raises on a miss — silently ignoring an override the caller meant to
    take effect is the failure mode this whole path is guarding against.
    """
    if not overrides:
        return overrides
    filtered = {}
    for key, value in overrides.items():
        if key in _ABSENT_STANDARD_OVERRIDE_KEYS:
            logger.debug(
                "Ignoring override key %r: DreamZero normalizes inside its own processor steps "
                "using the checkpoint's statistics, so it has no matching step.",
                key,
            )
            continue
        filtered[key] = value
    return filtered


def _apply_step_overrides(pipeline, overrides: dict[str, Any] | None) -> None:
    """Apply ``from_pretrained``-style step overrides to a freshly built pipeline.

    DreamZero builds its processors from scratch rather than deserializing a saved pipeline (its
    normalization statistics live in the checkpoint, not in a ``policy_preprocessor.json``), so
    caller overrides such as ``rename_observations_processor`` and ``device_processor`` have to be
    applied to the constructed steps. Mirrors GR00T's equivalent, including raising on keys that
    match nothing — an override that silently does nothing is worse than one that fails.
    """
    if not overrides:
        return

    def step_keys(step) -> set[str]:
        keys = {type(step).__name__}
        registry_name = getattr(type(step), "_registry_name", None)
        if registry_name:
            keys.add(registry_name)
        return keys

    for override_key, step_overrides in overrides.items():
        matched = [step for step in pipeline.steps if override_key in step_keys(step)]
        if not matched:
            available = [getattr(type(s), "_registry_name", None) or type(s).__name__ for s in pipeline.steps]
            raise KeyError(
                f"Override key {override_key!r} matches no step of the DreamZero processor "
                f"pipeline. Available step keys: {available}."
            )
        for step in matched:
            for field_name, value in dict(step_overrides).items():
                if not hasattr(step, field_name):
                    raise TypeError(f"Override field {field_name!r} is not a field of step {override_key!r}.")
                setattr(step, field_name, value)
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                # Re-derive anything computed from the overridden config (DeviceProcessorStep
                # resolves its torch.device there, for instance).
                post_init()


def make_dreamzero_pre_post_processors(
    config: DreamZeroConfig,
    dataset_stats: dict | None = None,
    dataset_meta=None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
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

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _apply_step_overrides(preprocessor, _drop_absent_standard_overrides(preprocessor_overrides))
    _apply_step_overrides(postprocessor, _drop_absent_standard_overrides(postprocessor_overrides))
    return preprocessor, postprocessor


def _has_dreamzero_stats(stats: dict | None, config: DreamZeroConfig) -> bool:
    """True when `stats` is in DreamZero's per-modality-key schema for this config's layout."""
    if not stats:
        return False
    for modality, keys in (("state", config.state_modality_keys), ("action", config.action_modality_keys)):
        group = stats.get(modality)
        if not isinstance(group, dict) or not all(_extract_has_q(group.get(k)) for k in keys):
            return False
    return True


def _extract_has_q(entry) -> bool:
    return isinstance(entry, dict) and "q01" in entry and "q99" in entry


def _resolve_video_modality_keys(config: DreamZeroConfig, pretrained_path) -> None:
    """Fill an empty `video_modality_keys` from the checkpoint's own recorded concat order.

    The order decides which camera occupies which canvas quadrant, and the checkpoint records the
    one its weights were trained with (`experiment_cfg/conf.yaml`'s `video_concat_order`). Only
    the `--policy.path` route reads that automatically, because a training run builds its config
    from the CLI — so without this a fine-tune falls back to sorted camera keys, which matches
    DROID only by alphabetical accident.

    Mirrors how GR00T resolves its `video_modality_keys` from the checkpoint at processor-build
    time. Left alone when the caller set the field explicitly.
    """
    from . import gear_checkpoint

    if config.video_modality_keys or not gear_checkpoint.is_gear_checkpoint(pretrained_path):
        return
    order = gear_checkpoint.read_concat_layout(pretrained_path, config.embodiment_tag).get(
        "video_modality_keys"
    )
    if not order:
        return
    config.video_modality_keys = tuple(order)
    logger.info(
        "video_modality_keys not set; using the order %s recorded in %s. Cameras named differently "
        "in the dataset need --rename_map, or set --policy.video_modality_keys explicitly.",
        list(order),
        pretrained_path,
    )


def make_dreamzero_pre_post_processors_from_pretrained(
    config: DreamZeroConfig,
    pretrained_path,
    dataset_stats: dict | None = None,
    **kwargs,
):
    """Build the DreamZero processors, reading the q01/q99 statistics from the checkpoint.

    A released NVIDIA checkpoint carries them in ``experiment_cfg/metadata.json``; a LeRobot one
    saved by this policy carries them in ``statistics.json`` as ``{embodiment_tag: {state, action}}``.
    Either way the q99 normalization and the relative→absolute decode use the checkpoint's own
    statistics, sliced to ``config.state_modality_keys`` / ``config.action_modality_keys`` (mirrors
    GR00T's ``make_groot_pre_post_processors_from_pretrained``).
    """
    from . import gear_checkpoint

    _resolve_video_modality_keys(config, pretrained_path)

    stats = dataset_stats if _has_dreamzero_stats(dataset_stats, config) else None
    if dataset_stats is not None and stats is None:
        # `lerobot-train` always passes `dataset.meta.stats`, which is LeRobot's FLAT schema
        # ({"observation.state": {...}, "action": {...}}). DreamZero needs per-modality-key
        # quantiles ({state: {joint_position: {q01, q99}}}), and — more importantly — a fine-tune
        # must normalize with the statistics the checkpoint was trained on, not the ones recomputed
        # from whatever subset is being trained on now. Ignore them and say so.
        logger.info(
            "Ignoring the supplied dataset_stats (not in DreamZero's per-modality-key schema); "
            "using the checkpoint's own statistics from %s.",
            pretrained_path,
        )
    if stats is None:
        stats = gear_checkpoint.load_checkpoint_statistics(
            pretrained_path,
            config.embodiment_tag,
            config.state_modality_keys,
            config.action_modality_keys,
            override_path=config.statistics_path,
        )
        if stats is None:
            logger.warning(
                "No statistics found in %s — DreamZero q99 normalization/decode will have no stats.",
                pretrained_path,
            )
    return make_dreamzero_pre_post_processors(config, dataset_stats=stats, **kwargs)
