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

"""CPU tests for DreamZero's Ulysses sequence parallelism.

Two things are covered, and they are deliberately separated. The index arithmetic needs no
devices and is where a subtle error would survive review -- so it is checked against a
brute-force expansion over every shard of every layout. The all-to-alls need ranks, so they
run over gloo in spawned processes: small tensors, two ranks, but a real collective and a real
backward pass.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lerobot.policies.dreamzero.wan.ulysses import (
    pad_to_multiple,
    shard_bounds,
    shard_runs,
    to_head_shard,
    to_seq_shard,
)

# The third is the DROID layout in miniature: a teacher-forced clean half on one row, the noisy
# half one row per frame, then the action and state registers at one row each.
LAYOUTS = [
    ((4, 1),),
    ((3, 5),),
    ((1, 12), (3, 4), (5, 1)),
    ((1, 24), (3, 8), (4, 1)),
]


def _expand(runs) -> list[int]:
    """One row index per token, which is what the layout is shorthand for."""
    out = []
    row = 0
    for rows, repeat in runs:
        for _ in range(rows):
            out += [row] * repeat
            row += 1
    return out


# --- the index arithmetic ---------------------------------------------------------------


@pytest.mark.parametrize("runs", LAYOUTS)
def test_shard_runs_selects_exactly_the_rows_the_tokens_need(runs):
    """Against a brute-force expansion, for every shard of every world size that divides."""
    rows_per_token = _expand(runs)
    total = len(rows_per_token)
    for world in (w for w in range(1, total + 1) if total % w == 0):
        for rank in range(world):
            start, end = shard_bounds(total, world, rank)
            row_start, row_end, local = shard_runs(runs, start, end)

            want = rows_per_token[start:end]
            assert row_start == want[0], f"{runs} w={world} r={rank}: first row"
            assert row_end == want[-1] + 1, f"{runs} w={world} r={rank}: last row"
            # The local layout, read against the sliced rows, must name the same rows in order.
            got = [row_start + r for r in _expand(local)]
            assert got == want, f"{runs} w={world} r={rank}: {got} != {want}"
            assert sum(r * n for r, n in local) == end - start


def test_shard_runs_keeps_whole_equal_rows_merged():
    """A shard that covers whole rows should stay one broadcast, not one op per row."""
    _, _, local = shard_runs(((8, 10),), 20, 60)
    assert local == ((4, 10),)


def test_shard_runs_splits_a_row_it_only_partly_covers():
    """A partial row is a run of one row with fewer tokens, which the layout already says."""
    row_start, row_end, local = shard_runs(((4, 10),), 5, 25)
    assert (row_start, row_end) == (0, 3)
    assert local == ((1, 5), (1, 10), (1, 5))


def test_shard_runs_rejects_a_range_past_the_layout():
    with pytest.raises(ValueError, match="runs past"):
        shard_runs(((2, 3),), 0, 7)


def test_shard_bounds_refuses_an_uneven_split():
    with pytest.raises(ValueError, match="does not divide"):
        shard_bounds(15940, 8, 0)


def test_padding_is_what_makes_the_droid_window_divide():
    """L = 15,940 at the DROID window, which 8 ranks do not divide."""
    assert pad_to_multiple(15940, 8) == 4
    assert (15940 + 4) % 8 == 0
    assert pad_to_multiple(15940, 4) == 0
    assert pad_to_multiple(15940, 1) == 0


# --- the collectives --------------------------------------------------------------------

WORLD = 2
HEADS = 4
SEQ = 6  # global; 3 per rank
DIM = 2


def _worker(rank: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        # Every rank builds the same global tensor, then keeps its own sequence shard.
        torch.manual_seed(0)
        full = torch.randn(1, SEQ, HEADS, DIM, dtype=torch.float64)
        start, end = shard_bounds(SEQ, WORLD, rank)
        mine = full[:, start:end].clone().requires_grad_()

        heads = to_head_shard(mine, WORLD)
        # After the exchange: all tokens, this rank's heads.
        head_lo, head_hi = rank * (HEADS // WORLD), (rank + 1) * (HEADS // WORLD)
        exchanged_ok = torch.equal(heads, full[:, :, head_lo:head_hi])

        back = to_seq_shard(heads, WORLD)
        round_trip_ok = torch.equal(back, mine)

        # A backward through both must land the incoming gradient unchanged on the shard.
        grad = torch.randn_like(mine)
        back.backward(grad)
        grad_ok = torch.equal(mine.grad, grad)

        queue.put((rank, tuple(heads.shape), exchanged_ok, round_trip_ok, grad_ok))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seed", [0])
def test_the_all_to_alls_move_the_shard_from_tokens_to_heads(seed):
    """Over gloo, with two real ranks: the exchange, its inverse, and the backward."""
    import socket

    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(rank, port, queue)) for rank in range(WORLD)]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=120) for _ in range(WORLD)]
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"

    assert len(results) == WORLD
    for rank, shape, exchanged_ok, round_trip_ok, grad_ok in results:
        assert shape == (1, SEQ, HEADS // WORLD, DIM), f"rank {rank} shape {shape}"
        assert exchanged_ok, f"rank {rank} did not receive the heads it should own"
        assert round_trip_ok, f"rank {rank} round trip is not the identity"
        assert grad_ok, f"rank {rank} gradient did not come back unchanged"


def test_a_degree_of_one_is_a_no_op_with_no_process_group():
    """So the code path is the same whether or not the run is sequence-parallel."""
    x = torch.randn(1, 5, 4, 2, requires_grad=True)
    assert to_head_shard(x, 1) is x
    assert to_seq_shard(x, 1) is x


def test_heads_must_divide_over_the_ranks():
    from lerobot.policies.dreamzero.wan.ulysses import _to_head_shard

    with pytest.raises(ValueError, match="heads do not divide"):
        _to_head_shard(torch.randn(1, 2, 5, 2), 2, None)
