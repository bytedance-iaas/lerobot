#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Neighborhood attention in plain PyTorch, for devices NATTEN will not run on.

NATTEN picks a backend by hardware: fused CUTLASS kernels on recent NVIDIA parts, else
`flex-fna`, which it calls a universal fallback. It is not universal. `can_run_flex_attention`
accepts cuda, rocm and cpu only, so on an Ascend NPU every backend is refused before any
math happens:

    RuntimeError: Can't run Flex Attention; tensor is not on a CUDA, ROCm, or CPU device: npu

Nothing about the operation needs a special kernel, so this module computes the same thing
with gather + softmax + matmul. It is the path an NPU takes; cuda and cpu keep using NATTEN.

Semantics, checked pointwise against NATTEN 0.21.6 on CPU (max abs diff 3e-07 for na2d,
4e-07 for na3d with a causal axis):

  - Each query attends to exactly `kernel_size` keys per axis. At a border the window slides
    inward to stay in bounds -- it is not zero-padded, so a corner query sees a full window
    that is simply off-centre.
  - On a causal axis the window ends at the query. Positions before the window fills up get
    fewer real keys; the index table repeats index 0 to keep the window rectangular and those
    repeats are masked out.

Shapes follow NATTEN's: (batch, *spatial, heads, head_dim), spatial being (H, W) for 2D and
(T, H, W) for 3D.

The spatial axes are flattened into one before gathering. An unfold-style expansion would
build a tensor with two axes per spatial dim -- ten for the 3D case -- and CANN's
`aclnnInplaceCopy` takes at most eight:

    The self tensor cannot be larger than 8 dimensions

Flattening keeps every intermediate at five. Queries are processed in chunks so that peak
memory is set by `chunk` rather than by the sequence length, since the gathered keys are
`kernel_size` times the size of the input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["na2d_torch", "na3d_torch", "neighborhood_attention_torch"]

# Queries per chunk. The gathered key/value tensors are K times the per-query size, so this
# bounds peak memory at roughly chunk * K * heads * head_dim elements per tensor.
_DEFAULT_CHUNK = 4096


def _axis_window(length: int, kernel: int, causal: bool, device: torch.device) -> Tensor:
    """(length, kernel) key indices covered by each position's window along one axis.

    `kernel` is the effective window from `_effective_kernel`, never wider than the axis.
    """
    pos = torch.arange(length, device=device)
    if causal:
        start = (pos - kernel + 1).clamp(min=0)
    else:
        start = (pos - kernel // 2).clamp(min=0, max=length - kernel)
    return start[:, None] + torch.arange(kernel, device=device)[None, :]


def _effective_kernel(spatial: Sequence[int], kernel_size: Sequence[int]) -> tuple[int, ...]:
    """Clamp each window to its axis.

    The VAE downsamples time, so a stage can hand a 4-frame axis to a window of 5. A window
    wider than its axis degenerates to the whole axis -- full attention along it, and causal
    attention when that axis is causal -- which is the limit of the sliding window, not a
    special case. Raising here instead would make the fallback refuse shapes the model
    legitimately produces.
    """
    return tuple(min(k, s) for k, s in zip(kernel_size, spatial, strict=True))


def _neighborhood_tables(
    spatial: Sequence[int],
    kernel_size: Sequence[int],
    is_causal: Sequence[bool],
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    """Flat key indices (S, K) and, when any axis is causal, a (S, K) validity mask."""
    nd = len(spatial)
    kernel_size = _effective_kernel(spatial, kernel_size)
    strides = [1] * nd
    for a in range(nd - 2, -1, -1):
        strides[a] = strides[a + 1] * spatial[a + 1]

    idx = torch.zeros(tuple(spatial) + tuple(kernel_size), dtype=torch.long, device=device)
    valid: Tensor | None = None
    for a in range(nd):
        table = _axis_window(spatial[a], kernel_size[a], is_causal[a], device)
        view = [1] * (2 * nd)
        view[a] = spatial[a]
        view[nd + a] = kernel_size[a]
        idx = idx + table.reshape(view) * strides[a]

        if is_causal[a]:
            pos = torch.arange(spatial[a], device=device)
            first = (pos - kernel_size[a] + 1).clamp(min=0)
            axis_valid = (table >= first[:, None]) & (table <= pos[:, None])
            axis_valid = axis_valid.reshape(view)
            valid = axis_valid if valid is None else valid & axis_valid

    s = math.prod(spatial)
    k = math.prod(kernel_size)
    idx = idx.reshape(s, k)
    if valid is not None:
        valid = valid.expand(tuple(spatial) + tuple(kernel_size)).reshape(s, k)
    return idx, valid


def neighborhood_attention_torch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    scale: float | None = None,
    chunk: int = _DEFAULT_CHUNK,
) -> Tensor:
    """NATTEN-equivalent neighborhood attention. (b, *spatial, heads, dim) in and out."""
    nd = len(kernel_size)
    if query.ndim != nd + 3:
        raise ValueError(f"expected {nd + 3} dims for a {nd}D neighborhood, got {query.ndim}")
    if is_causal is None:
        is_causal = [False] * nd
    if len(is_causal) != nd:
        raise ValueError(f"is_causal has {len(is_causal)} entries for a {nd}D neighborhood")

    b = query.shape[0]
    spatial = tuple(query.shape[1 : 1 + nd])
    heads, dim = query.shape[-2], query.shape[-1]
    if scale is None:
        scale = dim**-0.5

    idx, valid = _neighborhood_tables(spatial, tuple(kernel_size), tuple(is_causal), query.device)

    s = math.prod(spatial)
    q_flat = query.reshape(b, s, heads, dim)
    k_flat = key.reshape(b, s, heads, dim)
    v_flat = value.reshape(b, s, heads, dim)
    out = torch.empty_like(q_flat)

    for start in range(0, s, chunk):
        stop = min(start + chunk, s)
        window = idx[start:stop]  # (c, K)
        # (b, c, K, heads, dim) -- five dims, whatever the spatial rank is.
        k_win = k_flat.index_select(1, window.reshape(-1)).unflatten(1, window.shape)
        v_win = v_flat.index_select(1, window.reshape(-1)).unflatten(1, window.shape)

        logits = torch.einsum("bchd,bckhd->bchk", q_flat[:, start:stop], k_win) * scale
        if valid is not None:
            logits = logits.masked_fill(~valid[start:stop][None, :, None, :], float("-inf"))
        probs = logits.softmax(-1).to(v_win.dtype)
        out[:, start:stop] = torch.einsum("bchk,bckhd->bchd", probs, v_win)

    return out.reshape(b, *spatial, heads, dim)


def na2d_torch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    **kwargs,
) -> Tensor:
    """Drop-in for `natten.functional.na2d`, minus the backend selection."""
    return neighborhood_attention_torch(query, key, value, kernel_size, is_causal, **kwargs)


def na3d_torch(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    **kwargs,
) -> Tensor:
    """Drop-in for `natten.functional.na3d`, minus the backend selection."""
    return neighborhood_attention_torch(query, key, value, kernel_size, is_causal, **kwargs)
