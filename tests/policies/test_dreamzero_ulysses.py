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
    gather_seq,
    pad_to_multiple,
    segment_in_shard,
    segment_sizes,
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

# Overridable so the same tests can be pointed at real devices and a real backend, which is
# the only way to find out whether that backend's all_to_all_single agrees with gloo's:
#   LEROBOT_TEST_SP=1 LEROBOT_SP_BACKEND=hccl LEROBOT_SP_WORLD=4 pytest ...
BACKEND = os.environ.get("LEROBOT_SP_BACKEND", "gloo")
WORLD = int(os.environ.get("LEROBOT_SP_WORLD", "2"))
HEADS = 4 * WORLD // 2  # keep heads divisible by the degree
SEQ = 3 * WORLD  # global; 3 per rank
DIM = 2


def _init(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if BACKEND != "gloo":
        import torch  # noqa: F811

        getattr(torch, BACKEND.replace("hccl", "npu")).set_device(rank)
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD)


# A spawned rank on an accelerator spends most of a minute in device init alone, so the wait
# for its result has to scale with that; 120 s is right for CPU gloo and far too short for NPU.
QUEUE_TIMEOUT = 120 if BACKEND == "gloo" else 900


def _device() -> str:
    return "cpu" if BACKEND == "gloo" else BACKEND.replace("hccl", "npu")


def _worker(rank: int, port: int, queue) -> None:
    _init(rank, port)
    try:
        # Every rank builds the same global tensor, then keeps its own sequence shard.
        torch.manual_seed(0)
        full = torch.randn(1, SEQ, HEADS, DIM, dtype=torch.float32, device=_device())
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

        queue.put((rank, tuple(heads.shape), bool(exchanged_ok), bool(round_trip_ok), bool(grad_ok)))
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
    results = [queue.get(timeout=QUEUE_TIMEOUT) for _ in range(WORLD)]
    for proc in procs:
        proc.join(timeout=QUEUE_TIMEOUT)
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
    _init(rank, port)
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
    # Bounded on purpose: a mismatched collective sequence between ranks shows up as a hang,
    # and a hang that fails is diagnosable where one that waits forever is not.
    results = sorted((queue.get(timeout=QUEUE_TIMEOUT) for _ in range(WORLD)), key=lambda r: r[0])
    for proc in procs:
        proc.join(timeout=QUEUE_TIMEOUT)
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


# --- the segment arithmetic the sharded tail needs ---------------------------------------
# The tail has to pull three global ranges out of a contiguous shard -- the video tokens, the
# action registers, the state registers -- and a rank's share of any of them can be empty or
# straddle a boundary. All of it is static index arithmetic, so none of it needs ranks.

DROID = {"total": 15940, "clean": (0, 7920), "video": (7920, 15840), "action": (15840, 15936)}


@pytest.mark.parametrize("world", [1, 2, 4, 5, 10, 20])
def test_segment_sizes_account_for_every_token_of_a_segment(world):
    """Whatever the degree, the ranks' shares of a segment add up to it and nothing else."""
    for lo, hi in (DROID["clean"], DROID["video"], DROID["action"]):
        sizes = segment_sizes(DROID["total"], world, lo, hi)
        assert len(sizes) == world
        assert sum(sizes) == hi - lo, f"world={world} segment=({lo},{hi})"
        assert all(size >= 0 for size in sizes)


@pytest.mark.parametrize("world", [2, 4, 5])
def test_a_shard_can_hold_none_of_a_segment(world):
    """At degree 4 rank 0 is entirely inside the clean half, so its video share is empty."""
    sizes = segment_sizes(DROID["total"], 4, *DROID["video"])
    assert sizes[0] == 0
    assert sum(sizes) == 7920
    # And the same arithmetic agrees with slicing the shard directly.
    for rank in range(world):
        start, end = shard_bounds(DROID["total"], world, rank)
        a, b = segment_in_shard(start, end, *DROID["video"])
        assert b - a == segment_sizes(DROID["total"], world, *DROID["video"])[rank]


def test_segment_in_shard_is_a_slice_of_the_shard_not_of_the_sequence():
    """It returns shard-local indices, which is what `x_local[:, a:b]` wants."""
    assert segment_in_shard(100, 200, 150, 400) == (50, 100)
    assert segment_in_shard(100, 200, 0, 120) == (0, 20)
    assert segment_in_shard(100, 200, 0, 50) == (0, 0)
    assert segment_in_shard(100, 200, 300, 400) == (0, 0)


def test_the_droid_segments_tile_the_window():
    """A sanity check on the ranges themselves: clean, video, action, state, nothing lost."""
    assert DROID["clean"][1] == DROID["video"][0]
    assert DROID["video"][1] == DROID["action"][0]
    assert DROID["total"] - DROID["action"][1] == 4  # the state registers


def _gather_worker(rank: int, port: int, queue) -> None:
    _init(rank, port)
    try:
        # A 10-token sequence whose "video" segment is tokens 3..9, so the two ranks hold 2 and
        # 4 of it -- uneven on purpose, which is the case `all_gather_into_tensor` cannot do.
        total, lo, hi = 10, 3, 9
        torch.manual_seed(3)
        full = torch.randn(1, hi - lo, 2, dtype=torch.float32, device=_device())
        sizes = segment_sizes(total, WORLD, lo, hi)
        offset = sum(sizes[:rank])
        mine = full[:, offset : offset + sizes[rank]].clone().requires_grad_()

        whole = gather_seq(mine, sizes, rank, WORLD)
        grad = torch.randn_like(whole)
        whole.backward(grad)
        queue.put(
            (
                rank,
                sizes,
                torch.equal(whole.detach(), full),
                torch.equal(mine.grad, grad[:, offset : offset + sizes[rank]]),
            )
        )
    finally:
        dist.destroy_process_group()


def test_the_uneven_gather_reassembles_and_its_backward_is_a_slice():
    """Over gloo: pieces of different lengths, and a backward that slices rather than sums."""
    import socket

    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_gather_worker, args=(rank, port, queue)) for rank in range(WORLD)]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=QUEUE_TIMEOUT) for _ in range(WORLD)]
    for proc in procs:
        proc.join(timeout=QUEUE_TIMEOUT)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"

    for rank, sizes, whole_ok, grad_ok in results:
        assert sum(sizes) == 6 and len(set(sizes)) > 1, f"rank {rank} wanted uneven pieces, got {sizes}"
        assert whole_ok, f"rank {rank} did not reassemble the segment"
        assert grad_ok, f"rank {rank} backward is not a slice of the incoming gradient"


# --- the whole model boundary -------------------------------------------------------------

MODEL = {
    "model_type": "i2v",
    "patch_size": (1, 2, 2),
    "frame_seqlen": 6,
    "text_len": 8,
    "in_dim": 4,
    "dim": 32,
    "ffn_dim": 64,
    "freq_dim": 16,
    "text_dim": 24,
    "out_dim": 4,
    "num_heads": 4,
    "num_layers": 2,
    "max_chunk_size": 2,
    "num_frame_per_block": 2,
    "action_dim": 5,
    "max_state_dim": 7,
    "hidden_size": 16,
    "num_action_per_block": 4,
    "num_state_per_block": 1,
    "concat_first_frame_latent": False,
}
FRAMES = 3
VIDEO = 18  # 3 frames x 6 tokens, so L = 36 with no registers: 2 and 4 both divide it


def _forward_model(world: int):
    """One `_forward_train` at the given degree, video-only so the window divides evenly."""
    from lerobot.policies.dreamzero.wan.causal_wan_model import CausalWanModel

    torch.manual_seed(0)
    model = CausalWanModel(**MODEL).to(torch.bfloat16)
    if world > 1:
        model.set_sequence_parallel(world, dist.group.WORLD)

    torch.manual_seed(1)
    video, _ = model._forward_train(
        x=torch.randn(1, 4, FRAMES, 4, 6, dtype=torch.bfloat16),
        timestep=(torch.rand(1, FRAMES) * 1000).to(torch.bfloat16),
        timestep_action=None,
        context=torch.randn(1, 8, 24, dtype=torch.bfloat16) if MODEL["text_len"] else None,
        seq_len=VIDEO,
        clean_x=torch.randn(1, 4, FRAMES, 4, 6, dtype=torch.bfloat16),
        y=torch.randn(1, 4, FRAMES, 4, 6, dtype=torch.bfloat16),
        clip_feature=torch.randn(1, 3, 1280, dtype=torch.bfloat16),
        embodiment_id=torch.zeros(1, dtype=torch.long),
    )
    video.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return video.detach().clone(), grads


def _model_worker(rank: int, port: int, queue) -> None:
    _init(rank, port)
    try:
        queue.put((rank, *_forward_model(WORLD)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("LEROBOT_TEST_SP") != "1",
    reason="~2 min of process spawn and imports; set LEROBOT_TEST_SP=1. REQUIRED when touching "
    "the sequence sharding in `_forward_train`.",
)
def test_the_whole_forward_matches_one_rank():
    """L = 36 here, so at degree 2 rank 0 holds nothing but the teacher-forced clean half.

    That makes this the test for the empty-share case as well: rank 0 contributes no video
    tokens to the head at all, and the gather still has to reassemble the prediction and route
    the gradient back to the rank that produced each piece.
    """
    import socket

    single_video, single_grads = _forward_model(1)

    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_model_worker, args=(rank, port, queue)) for rank in range(WORLD)]
    for proc in procs:
        proc.start()
    results = sorted((queue.get(timeout=QUEUE_TIMEOUT) for _ in range(WORLD)), key=lambda r: r[0])
    for proc in procs:
        proc.join(timeout=QUEUE_TIMEOUT)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"

    # Every rank reconstructs the same whole prediction, and it is the single-rank one.
    for rank, video, _ in results:
        assert video.shape == single_video.shape, f"rank {rank} shape"
        assert torch.equal(video, single_video), f"rank {rank} prediction"

    # And every weight gradient is a partial sum that adds up to the single-rank gradient.
    for name, want in single_grads.items():
        got = sum(grads[name] for _, _, grads in results if name in grads)
        scale = want.abs().max().double().clamp_min(1e-30)
        error = (got.double() - want.double()).abs().max()
        assert error <= 3e-2 * scale, f"{name}: off by {error / scale:.3%} of full scale"


def test_the_sp_helpers_resolve_where_they_are_called():
    """A guard against the import going missing again, which cost a real training run.

    `sync_in_group` is called from `WANPolicyHead.forward` at `sp_degree=1` as well as above
    it -- it returns its input when there is no group -- so a missing import breaks ordinary
    single-card training, not just the parallel path. It did: `NameError` on the first run
    after the calls were added, because the patch touched two modules and only one was linted.

    Nothing in this suite reaches `WANPolicyHead.forward` (the sharded-forward tests drive
    `CausalWanModel` directly), so ruff's F821 and this are what stand between that call site
    and a run. Cheap enough to be worth the tautology.
    """
    from lerobot.policies.dreamzero.wan import action_head, ulysses

    assert action_head.sync_in_group is ulysses.sync_in_group


def test_sp_requires_recompute_off():
    """SP exists to make GC=0 affordable; asking for both pays the all-to-all three passes."""
    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

    for gc in (True, 40, 1, 25):
        with pytest.raises(ValueError, match="requires gradient_checkpointing=0"):
            DreamZeroConfig(sp_degree=4, gradient_checkpointing=gc)

    # Both spellings of "none" are accepted, and sp_degree=1 is unconstrained.
    for gc in (False, 0):
        DreamZeroConfig(sp_degree=4, gradient_checkpointing=gc)
    for gc in (True, 40, 0, False):
        DreamZeroConfig(sp_degree=1, gradient_checkpointing=gc)


# --- padding, which is what makes a degree that does not divide the sequence reachable -------

def test_pad_to_multiple_on_the_real_sequence_length():
    """15940 = 2^2 x 5 x 797, so only degrees built from {2, 2, 5, 797} divide it."""
    assert pad_to_multiple(15940, 1) == 0
    for degree in (2, 4, 5, 10, 20):
        assert pad_to_multiple(15940, degree) == 0, degree
    assert pad_to_multiple(15940, 8) == 4  # -> 15944 = 8 x 1993
    assert pad_to_multiple(15940, 16) == 12  # -> 15952 = 16 x 997
    for degree in (2, 4, 5, 8, 10, 16, 20):
        assert (15940 + pad_to_multiple(15940, degree)) % degree == 0, degree


@pytest.mark.parametrize("degree", [2, 4, 8, 16])
def test_padding_makes_the_shards_equal_and_covering(degree):
    total = 15940
    padded = total + pad_to_multiple(total, degree)
    bounds = [shard_bounds(padded, degree, r) for r in range(degree)]
    widths = {hi - lo for lo, hi in bounds}
    assert len(widths) == 1, f"shards must be equal for an all-to-all, got {widths}"
    assert bounds[0][0] == 0 and bounds[-1][1] == padded
    for (_, hi), (lo, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert hi == lo, "shards must tile the padded sequence with no gap"


def test_unpadded_length_is_rejected_rather_than_split_unevenly():
    """`shard_bounds` refuses rather than rounding, which is what sends the caller to the pad."""
    with pytest.raises(ValueError, match="pad it first"):
        shard_bounds(15940, 8, 0)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_the_modulation_layout_covers_every_padded_shard(degree):
    """The run list has to grow with the tokens or `shard_runs` refuses the last shard.

    This mirrors what `_forward_train` does: one extra row of `(1, pad)` after the real runs.
    """
    from lerobot.policies.dreamzero.wan.causal_wan_model import _join_runs

    # A stand-in for the real layout: one row per video token group plus the action registers.
    runs = ((1, 15840), (100, 1))
    total = sum(rows * repeat for rows, repeat in runs)
    assert total == 15940
    pad = pad_to_multiple(total, degree)
    padded_runs = _join_runs(*runs, (1, pad)) if pad else runs
    covered = sum(rows * repeat for rows, repeat in padded_runs)
    assert covered == total + pad

    rows_total = sum(rows for rows, _ in padded_runs)
    seen_rows = 0
    for rank in range(degree):
        lo, hi = shard_bounds(total + pad, degree, rank)
        row_start, row_end, local = shard_runs(padded_runs, lo, hi)
        assert sum(rows * repeat for rows, repeat in local) == hi - lo
        assert row_start >= 0 and row_end <= rows_total
        seen_rows = max(seen_rows, row_end)
    assert seen_rows == rows_total, "every modulation row must land in some shard"
