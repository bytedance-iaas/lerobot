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

"""The streaming dataset must split itself across DDP ranks and DataLoader workers.

An IterableDataset has no sampler, so without this every rank iterates the SAME frames —
duplicated gradients. These tests pin the partition maths in `_ddp_shard_plan`: the union of
all consumers must cover the stream exactly once, with no overlap, in both regimes
(shards >= consumers -> split by shard; shards < consumers -> stride the frames).
"""

from unittest.mock import patch

import pytest

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset


class _Plan:
    """Just the sharding maths, without constructing a real dataset.

    Mirrors what __init__ stores: the DDP topology is captured in the main process, so the
    plan reads `_ddp_rank` / `_ddp_world_size` rather than querying torch.distributed from
    inside a DataLoader worker (a spawned worker would see an uninitialized process group).
    """

    def __init__(self, num_shards: int, rank: int, world_size: int):
        self.num_shards = num_shards
        self._ddp_rank = rank
        self._ddp_world_size = world_size

    plan = StreamingLeRobotDataset._ddp_shard_plan


def _plan_for(num_shards: int, rank: int, world_size: int, worker_id: int = 0, num_workers: int = 0):
    obj = _Plan(num_shards, rank, world_size)
    worker_info = None
    if num_workers:
        worker_info = type("WI", (), {"id": worker_id, "num_workers": num_workers})()
    with patch("torch.utils.data.get_worker_info", return_value=worker_info):
        return _Plan.plan(obj)


def test_single_process_reads_everything():
    shard_ids, stride, offset = _plan_for(num_shards=16, rank=0, world_size=1)
    assert shard_ids == list(range(16))
    assert (stride, offset) == (1, 0)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_shards_split_disjointly_and_completely(world_size):
    """shards >= consumers: every shard goes to exactly one rank."""
    num_shards = 16
    seen: list[int] = []
    for rank in range(world_size):
        shard_ids, stride, _ = _plan_for(num_shards, rank=rank, world_size=world_size)
        assert stride == 1, "should split by shard, not by striding frames"
        seen.extend(shard_ids)
    assert sorted(seen) == list(range(num_shards))  # complete, and no shard twice


def test_ranks_and_workers_are_disjoint_together():
    """The split must account for BOTH ranks and DataLoader workers."""
    num_shards, world_size, num_workers = 8, 2, 2
    seen: list[int] = []
    for rank in range(world_size):
        for worker_id in range(num_workers):
            ids, stride, _ = _plan_for(
                num_shards, rank=rank, world_size=world_size,
                worker_id=worker_id, num_workers=num_workers,
            )
            assert stride == 1
            seen.extend(ids)
    assert sorted(seen) == list(range(num_shards))


def test_fewer_shards_than_consumers_falls_back_to_striding():
    """A stream that cannot be split by shard is still partitioned, frame-wise."""
    num_shards, world_size = 2, 4
    offsets = []
    for rank in range(world_size):
        ids, stride, offset = _plan_for(num_shards, rank=rank, world_size=world_size)
        assert ids == list(range(num_shards)), "every consumer must read every shard here"
        assert stride == world_size
        offsets.append(offset)
    # Disjoint and complete: each frame index lands on exactly one rank.
    assert sorted(offsets) == list(range(world_size))
