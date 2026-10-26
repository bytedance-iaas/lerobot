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

"""Ulysses sequence parallelism for the DreamZero DiT.

Why this and not ring/context parallelism. All 40 blocks of activations want ~217 GiB at the
DROID window (measured: 5.56 GiB per block, `configuration_dreamzero.gradient_checkpointing`),
against 131 GiB on the largest card that exists, and FSDP shards parameters and gradients but
not activations -- so adding cards does not help and `gradient_checkpointing=False` is
unreachable. Ulysses divides the activation bill by the degree, which is the only thing that
makes the dial fully openable.

Ulysses is also the only form this model can take without rewriting its attention. It moves
the shard from the sequence axis to the head axis with an all-to-all *inside* the attention:
after that exchange every rank holds **all** tokens for N/P heads, so the teacher-forcing
pattern -- 640 hand-sliced attention calls built by indexing token ranges in Python rather
than with a mask -- is correct with no changes at all. Ring/context parallelism keeps the
shard on tokens, which would mean re-deriving the block-causal pattern per shard, in five
functions. 40 heads divide by 8, 5, 4 and 2.

What lives here: the two all-to-alls as autograd functions (each is the other's backward), and
the pure index arithmetic for taking a rank's slice of a sequence and of the compact
modulation layout. The index arithmetic is the part that is easy to get subtly wrong and needs
no devices to test, so it is separated from anything that touches a process group.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _exchange(x: torch.Tensor, group) -> torch.Tensor:
    """One all-to-all over dim 0, which both directions below reduce to.

    Both callers hand this a permuted view, so the `contiguous()` comes before `empty_like`:
    `empty_like` preserves memory format, and an output with a permuted layout makes
    `all_to_all_single` write its chunks to the wrong elements -- silently, with correct
    shapes.
    """
    x = x.contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out


def _to_head_shard(x: torch.Tensor, world: int, group) -> torch.Tensor:
    """[B, S/P, H, D] -> [B, S, H/P, D]. Sequence becomes whole, heads become a shard."""
    b, s_local, h, d = x.shape
    if h % world:
        raise ValueError(f"{h} heads do not divide over {world} sequence-parallel ranks")
    # Group the heads this rank must send to each peer into dim 0, which is what all_to_all
    # splits on, then put the peers' sequence chunks back together along the sequence axis.
    x = x.reshape(b, s_local, world, h // world, d).permute(2, 0, 1, 3, 4)
    x = _exchange(x, group)
    return x.permute(1, 0, 2, 3, 4).reshape(b, world * s_local, h // world, d)


def _to_seq_shard(x: torch.Tensor, world: int, group) -> torch.Tensor:
    """[B, S, H/P, D] -> [B, S/P, H, D]. The exact inverse of `_to_head_shard`."""
    b, s, h_local, d = x.shape
    if s % world:
        raise ValueError(f"a sequence of {s} does not divide over {world} ranks; pad it first")
    x = x.reshape(b, world, s // world, h_local, d).permute(1, 0, 2, 3, 4)
    x = _exchange(x, group)
    return x.permute(1, 2, 0, 3, 4).reshape(b, s // world, world * h_local, d)


class ToHeadShard(torch.autograd.Function):
    """Sequence shard -> head shard, on the way into the attention."""

    @staticmethod
    def forward(ctx, x, world, group):
        ctx.world, ctx.group = world, group
        return _to_head_shard(x, world, group)

    @staticmethod
    def backward(ctx, grad):
        return _to_seq_shard(grad, ctx.world, ctx.group), None, None


class ToSeqShard(torch.autograd.Function):
    """Head shard -> sequence shard, on the way out of the attention."""

    @staticmethod
    def forward(ctx, x, world, group):
        ctx.world, ctx.group = world, group
        return _to_seq_shard(x, world, group)

    @staticmethod
    def backward(ctx, grad):
        return _to_head_shard(grad, ctx.world, ctx.group), None, None


def to_head_shard(x: torch.Tensor, world: int, group=None) -> torch.Tensor:
    return x if world == 1 else ToHeadShard.apply(x, world, group)


def to_seq_shard(x: torch.Tensor, world: int, group=None) -> torch.Tensor:
    return x if world == 1 else ToSeqShard.apply(x, world, group)


# --- index arithmetic, which needs no devices ------------------------------------------


def shard_bounds(total: int, world: int, rank: int) -> tuple[int, int]:
    """This rank's half-open token range of an evenly divided sequence."""
    if total % world:
        raise ValueError(f"a sequence of {total} does not divide over {world} ranks; pad it first")
    per = total // world
    return rank * per, (rank + 1) * per


def shard_runs(
    runs: tuple[tuple[int, int], ...], start: int, end: int
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """A rank's slice of a compact modulation layout: `(row_start, row_end, local_runs)`.

    `runs` is the global `(rows, tokens_per_row)` layout `causal_wan_model._per_run` consumes.
    Slicing it needs no new machinery, because a shard can only cut its first and last row
    part-way, and a partially covered row is just a run of one row with fewer tokens -- which
    the existing layout already expresses. So the result is still a run list, and the rows
    tensor is sliced to `[row_start:row_end]`.

    Whole rows with the same repeat are merged so the broadcast stays one op per segment.
    """
    if not 0 <= start <= end:
        raise ValueError(f"bad shard range [{start}, {end})")
    covered = sum(rows * repeat for rows, repeat in runs)
    if end > covered:
        raise ValueError(f"shard [{start}, {end}) runs past the {covered} tokens the layout covers")

    local: list[tuple[int, int]] = []
    row_start: int | None = None
    row = 0
    token = 0
    for rows, repeat in runs:
        for _ in range(rows):
            lo, hi = max(token, start), min(token + repeat, end)
            if lo < hi:
                if row_start is None:
                    row_start = row
                take = hi - lo
                # Merge with the previous row only when both are whole and equally wide.
                if local and local[-1][1] == repeat == take:
                    local[-1] = (local[-1][0] + 1, repeat)
                else:
                    local.append((1, take))
            token += repeat
            row += 1
            if token >= end:
                break
        if token >= end:
            break
    if row_start is None:  # an empty shard, which only happens if start == end
        return 0, 0, ()
    return row_start, row_start + sum(rows for rows, _ in local), tuple(local)


def pad_to_multiple(total: int, world: int) -> int:
    """Tokens to append so an all-to-all can split the sequence evenly.

    The padding goes after the action/state registers, past every index the teacher-forcing
    slicing refers to, so padded tokens reach no attention call and are dropped before the
    head. They cost their share of the norms and the FFN and nothing else.
    """
    remainder = total % world
    return 0 if remainder == 0 else world - remainder


def segment_in_shard(start: int, end: int, lo: int, hi: int) -> tuple[int, int]:
    """Where a global token segment `[lo, hi)` lands inside this rank's shard `[start, end)`.

    Returned as indices into the shard, so `x_local[:, a:b]` is this rank's part of it. Empty
    when the two do not overlap, which is normal: at degree 4 the first rank holds nothing but
    the teacher-forced clean half, so its share of the video segment is empty.
    """
    a, b = max(start, lo), min(end, hi)
    return (0, 0) if a >= b else (a - start, b - start)


def segment_sizes(total: int, world: int, lo: int, hi: int) -> list[int]:
    """How many tokens of `[lo, hi)` each rank holds, which an uneven gather needs up front."""
    sizes = []
    for rank in range(world):
        start, end = shard_bounds(total, world, rank)
        a, b = segment_in_shard(start, end, lo, hi)
        sizes.append(b - a)
    return sizes


def _gather_uneven(x: torch.Tensor, sizes: list[int], group) -> torch.Tensor:
    """All-gather along dim 1 where the ranks' pieces differ in length.

    `dist.all_gather` requires every rank to contribute the same shape, and these pieces do
    not: a rank's share of the video segment depends on where its shard falls, and at degree 4
    the first rank's share is empty. So each rank pads to the longest piece, and the receiver
    trims each one back to the length it knows that rank holds. The tensors this runs on are
    the head's output, 64 wide against the block's 5120, so the padding is not worth avoiding.
    """
    widest = max(sizes)
    padded = x.new_zeros(x.shape[0], widest, *x.shape[2:])
    if x.shape[1]:
        padded[:, : x.shape[1]] = x
    received = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(received, padded.contiguous(), group=group)
    return torch.cat([piece[:, :size] for piece, size in zip(received, sizes, strict=True)], dim=1)


class GatherSeq(torch.autograd.Function):
    """Sharded -> whole along the sequence, for the small tensors at the end of the forward.

    The adjoint is a slice, not a sum, and that is worth being explicit about. Every rank
    computes the same thing from the gathered tensor, so each receives the same incoming
    gradient; what rank r contributed was its own piece, so its gradient is that piece of the
    incoming one. Nothing is added, because the gather replicated nothing within a rank -- it
    concatenated disjoint pieces.

    Each rank therefore ends up with a *partial* gradient for every weight, covering only its
    own tokens, and the partials sum to the true gradient. That uniformity is the point: it
    lets one `x degree` on the loss correct for FSDP averaging instead of summing, everywhere
    in the model at once. It is also why the head runs on the shard and only its output is
    gathered -- a head run redundantly on a gathered input would produce whole gradients where
    the blocks produce partial ones, and no single scale factor fixes both.
    """

    @staticmethod
    def forward(ctx, x, sizes, rank, group):
        ctx.offset = sum(sizes[:rank])
        ctx.length = sizes[rank]
        return _gather_uneven(x, sizes, group)

    @staticmethod
    def backward(ctx, grad):
        return grad[:, ctx.offset : ctx.offset + ctx.length], None, None, None


def gather_seq(x: torch.Tensor, sizes: list[int], rank: int, world: int, group=None) -> torch.Tensor:
    return x if world == 1 else GatherSeq.apply(x, sizes, rank, group)


def make_sp_group(degree: int):
    """This rank's Ulysses group, or None at degree 1.

    Members are contiguous ranks, so the all-to-alls -- four per block per forward, the heavy
    traffic here -- stay inside one node's NVLink domain, and the data-parallel axis is the
    strided one. Every rank builds every group in the same order, because `new_group` is itself
    collective and skipping one deadlocks the ranks that do not.
    """
    if degree == 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("a sequence-parallel degree above 1 needs an initialised process group")
    world = dist.get_world_size()
    if world % degree:
        raise ValueError(f"a world size of {world} does not divide into sequence-parallel groups of {degree}")
    rank = dist.get_rank()
    mine = None
    for base in range(0, world, degree):
        members = list(range(base, base + degree))
        group = dist.new_group(members)
        if rank in members:
            mine = group
    return mine
