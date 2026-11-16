#!/usr/bin/env python

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
"""Let the Qwen2-VL image processor patchify on an Ascend device.

Nothing in that processor is accelerator-specific -- it is torchvision plus plain tensor ops,
so whichever device the pixels already live on is where it runs. One op does not survive the
trip. Upstream patchifies by expanding the batch to NINE dimensions and reshaping back::

    patches.unsqueeze(6).expand(..., temporal_patch_size, ...).reshape(B, grid_h * grid_w, -1)

CANN's ``aclnnInplaceCopy`` takes at most eight ("AclNN_Parameter_Error(EZ1001): The self tensor
cannot be larger than 8 dimensions"), and that reshape copies, so on an NPU tensor it raises and
the whole preprocessing step dies::

    RuntimeError: copy_d2d_baseformat_opapi ... call aclnnInplaceCopy failed, error code is 161002

:func:`patch_qwen2vl_patchify` swaps in a rewrite that never exceeds six dimensions. It collapses
the four grid axes into the sequence axis first and only then expands the temporal axis, which
yields the identical element order -- upstream inserts the temporal axis between ``channel`` and
the two patch axes, and so does this. The override engages only for NPU tensors; on CUDA and CPU
the call goes straight to upstream's implementation, so nothing else changes.

Apply it to any processor built on ``Qwen2VLImageProcessor`` (GR00T, LaWAM, G0.5, Lingbot, ...)::

    processor = AutoProcessor.from_pretrained(model_id)
    patch_qwen2vl_patchify(processor)
"""

from __future__ import annotations

from typing import Any

import torch


def _patchify_low_rank(
    images: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Upstream's patchify, in at most six dimensions instead of nine.

    Upstream orders a patch by ``(grid_t, h-block, w-block, h-in-block, w-in-block)`` and its
    features by ``(channel, temporal, y, x)``. Collapsing the four grid axes before the temporal
    axis is expanded preserves both orders exactly, so the output is element-for-element what
    upstream returns -- only the intermediate rank differs.
    """
    batch_size, channel, resized_height, resized_width = images.shape
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = images.reshape(
        batch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch] -- 8 dims, allowed.
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)

    # Collapse the four grid axes into the sequence axis, keeping a length-1 slot where the
    # temporal axis goes. 6 dims, so the copy this reshape performs stays inside CANN's limit.
    merged = patches.reshape(batch_size, grid_h * grid_w, channel, 1, patch_size, patch_size)
    merged = merged.expand(-1, -1, -1, temporal_patch_size, -1, -1)
    flatten_patches = merged.reshape(
        batch_size,
        grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flatten_patches, grid_h, grid_w


def patch_qwen2vl_patchify(processor: Any) -> Any:
    """Install the Ascend-safe ``patchify`` on ``processor``'s Qwen2-VL image processor.

    Accepts either a full processor (``AutoProcessor``, which carries ``.image_processor``) or the
    image processor itself. Returns the argument, so it can wrap a construction expression. It is
    idempotent, and a processor that has no ``patchify`` to override is left untouched.
    """
    image_processor = getattr(processor, "image_processor", processor)
    upstream = getattr(type(image_processor), "patchify", None)
    if upstream is None or getattr(image_processor, "_lerobot_npu_patchify", False):
        return processor

    def patchify(
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        if images.device.type != "npu":
            return upstream(image_processor, images, patch_size, merge_size, temporal_patch_size)
        return _patchify_low_rank(images, patch_size, merge_size, temporal_patch_size)

    # Bind on the instance: `type(image_processor)` is shared with every other processor built
    # from the same class, including ones this policy does not own.
    image_processor.patchify = patchify
    image_processor._lerobot_npu_patchify = True
    return processor
