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

The whole-block comparison at the bottom costs ~2 minutes of spawn and imports, so it is gated
behind `LEROBOT_TEST_SP=1`. Run it when changing `wan/ulysses.py` or
`CausalWanSelfAttention.forward`: it is the only test here that would catch a layout error that
the collectives themselves are innocent of, and it is how the gradient-scaling confusion
documented in `_run_block` was found -- the forward passed while every gradient was `degree`
times too large.
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


# --- a real block, sharded ---------------------------------------------------------------
# The point of Ulysses here is that it is numerically exact against one rank: no collective in
# either direction reduces anything, and every head's attention is computed start to finish on
# one rank. So this does not check `allclose` -- it checks equality, forward and backward.

BLOCK = {
    "cross_attn_type": "i2v_cross_attn",
    "dim": 16,
    "ffn_dim": 32,
    "num_heads": 4,
    "frame_seqlen": 3,
    "local_attn_size": -1,
    "sink_size": 0,
    "num_frame_per_block": 1,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "num_action_per_block": 2,
    "num_state_per_block": 1,
}
B_SEQ = 8  # global tokens: a clean half and a noisy half of 4, no registers
B_RUNS = ((1, 4), (1, 4))


def _block_inputs(dtype=torch.bfloat16):
    torch.manual_seed(7)
    dim, heads = BLOCK["dim"], BLOCK["num_heads"]
    return {
        "x": torch.randn(1, B_SEQ, dim, dtype=dtype),
        "e": torch.randn(1, 2, 6, dim, dtype=dtype),
        # Per HALF the sequence: the clean and noisy halves are roped with the same freqs,
        # which is what `CausalWanSelfAttention.forward` does and relies on seeing both halves.
        "freqs": torch.polar(
            torch.ones(B_SEQ // 2, 1, dim // heads // 2, dtype=torch.float64),
            torch.randn(B_SEQ // 2, 1, dim // heads // 2, dtype=torch.float64),
        ),
        "context": torch.randn(1, 6, dim, dtype=dtype),
    }


def _run_block(world: int, rank: int, dtype=torch.bfloat16):
    """One forward+backward of a real block at the given Ulysses degree."""
    from lerobot.policies.dreamzero.wan.causal_wan_model import (
        CausalWanAttentionBlock,
        CausalWanSelfAttention,
    )

    torch.manual_seed(0)
    block = CausalWanAttentionBlock(**BLOCK).to(dtype)
    for module in block.modules():
        if isinstance(module, CausalWanSelfAttention):
            module.sp_world = world
            module.sp_group = None if world == 1 else dist.group.WORLD

    data = _block_inputs(dtype)
    start, end = shard_bounds(B_SEQ, world, rank)
    x = data["x"][:, start:end].clone().requires_grad_()
    row_start, row_end, runs = shard_runs(B_RUNS, start, end)

    out, _ = block(
        x,
        e=data["e"][:, row_start:row_end],
        freqs=data["freqs"],
        freqs_action=None,
        freqs_state=None,
        action_register_length=None,
        context=data["context"],
        e_runs=runs,
    )
    # NOT scaled by the degree, and the distinction matters. The identity being checked is
    # `sum_r dL_r/dW == dL/dW`, which holds with no correction at all because the per-rank
    # losses already sum to the whole-sequence loss. The x`degree` the *training* loop needs is
    # a correction for FSDP averaging gradients across ranks rather than summing them -- apply
    # it here and every gradient comes out `degree` times too large, which is how this test
    # first failed.
    out.square().sum().backward()
    grads = {name: param.grad.clone() for name, param in block.named_parameters() if param.grad is not None}
    return out.detach().clone(), x.grad.clone(), grads


def _sp_worker(rank: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        out, x_grad, grads = _run_block(WORLD, rank)
        queue.put((rank, out, x_grad, grads))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("LEROBOT_TEST_SP") != "1",
    reason="~2 min of process spawn and imports; set LEROBOT_TEST_SP=1. REQUIRED when touching "
    "wan/ulysses.py or CausalWanSelfAttention.forward -- the cheaper gloo tests above cover the "
    "collectives, but only this one covers the whole block.",
)
def test_a_sharded_block_matches_one_rank_exactly():
    """Ulysses reduces nothing, so SP is not an approximation -- outputs and gradients match.

    The weight gradients are the interesting half: under sequence parallelism each rank sees a
    subset of the tokens, so every weight's gradient is a *partial sum*. Summing the ranks'
    gradients has to give the single-rank gradient, and if the per-rank loss were not scaled by
    the degree it would come out short by exactly that factor -- the kind of error that reads
    as "multi-card converges slower" rather than as a bug.
    """
    import socket

    single_out, single_x_grad, single_grads = _run_block(1, 0)

    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_sp_worker, args=(rank, port, queue)) for rank in range(WORLD)]
    for proc in procs:
        proc.start()
    # Short on purpose: a mismatched collective sequence between ranks shows up as a hang,
    # and a hang that fails in two minutes is diagnosable where one that fails in five is not.
    results = sorted((queue.get(timeout=120) for _ in range(WORLD)), key=lambda r: r[0])
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"

    # The forward: each rank's shard is the matching slice of the single-rank output.
    for rank, out, _, _ in results:
        start, end = shard_bounds(B_SEQ, WORLD, rank)
        assert torch.equal(out, single_out[:, start:end]), f"rank {rank} forward"

    # dL/dx likewise, since the loss is a sum over tokens.
    for rank, _, x_grad, _ in results:
        start, end = shard_bounds(B_SEQ, WORLD, rank)
        assert torch.equal(x_grad, single_x_grad[:, start:end]), f"rank {rank} dL/dx"

    # The weight gradients are partial sums; added over the ranks they must be the whole.
    # This one gets a tolerance rather than equality, and only this one: the per-rank partials
    # are added here in a different order than the single-rank backward added its tokens, and
    # these run in bfloat16 because `AttentionModule` casts to it regardless. One bf16 ulp is
    # 3.9e-3, so 2e-2 of full scale is loose enough for a reordered sum and far tighter than
    # the factor the degree would introduce if the loss were not scaled.
    for name, want in single_grads.items():
        got = sum(grads[name] for _, _, _, grads in results)
        scale = want.abs().max().double().clamp_min(1e-30)
        error = (got.double() - want.double()).abs().max()
        assert error <= 2e-2 * scale, f"{name}: summed SP gradient off by {error / scale:.3%} of full scale"
