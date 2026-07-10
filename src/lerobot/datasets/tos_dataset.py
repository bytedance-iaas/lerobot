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
"""Stream a LeRobot v3.0 dataset from Volcengine TOS (object storage) — no download.

:class:`StreamingLeRobotDataset` streams from the HF Hub or a local dir. This subclass points
it at a TOS (or any ``fsspec``) URL instead, with credentials read from the environment and no
per-provider code:

1. **metadata** — the small ``meta/`` tree is mirrored to a local dir via fsspec and handed to
   the parent as ``root``, so the stock metadata path runs unchanged (a few MB; ``data/`` and
   ``videos/`` are never downloaded).
2. **low-dim data** — overrides the parent's ``_load_hf_dataset`` seam:
   ``load_dataset("parquet", data_files="<url>/data/*/*.parquet", storage_options=…,
   streaming=True)`` streams the parquet shards over fsspec, plus the ``episodes`` filter.
3. **video** — overrides the parent's ``_get_video_path`` seam to return the full ``<url>/…mp4``
   fsspec URL; lerobot's decoder opens it with ``fsspec.open(...)`` and lets torchcodec
   range-read it (frame-accurate — verified bit-exact vs the non-streaming reader).

Everything else — shuffle buffer, sharding, delta-timestamp windows, per-item construction — is
inherited from the parent. Credentials come from ``TOS_ACCESS_KEY`` / ``TOS_SECRET_KEY`` (plus
optional ``TOS_ENDPOINT`` / ``TOS_REGION``); pass ``storage_options`` to override. Install
``tosfs`` for the ``tos://`` protocol. It's an ``IterableDataset`` (buffer-shuffled, no random
index) — iterate it, don't index ``ds[i]``.

Example::

    ds = StreamingTOSRobotDataset("tos://bucket/prefix/name", episodes=[0, 3, 17])
    for item in ds:  # no ds[i]
        item["observation.images.front"]  # (C, H, W); also ["observation.state"], ["action"]
        break
"""

from __future__ import annotations

import os
import tempfile

import datasets
import fsspec
from datasets import load_dataset

from .streaming_dataset import StreamingLeRobotDataset


def _tos_env_storage_options() -> dict:
    """TOS fsspec ``storage_options`` from the environment (never hardcode secrets):
    ``TOS_ACCESS_KEY`` / ``TOS_SECRET_KEY`` (+ optional ``TOS_ENDPOINT`` / ``TOS_REGION``)."""
    opts: dict = {
        "endpoint": os.environ.get("TOS_ENDPOINT", "https://tos-cn-beijing.volces.com"),
        "region": os.environ.get("TOS_REGION", "cn-beijing"),
    }
    if os.environ.get("TOS_ACCESS_KEY"):
        opts["key"] = os.environ["TOS_ACCESS_KEY"]
    if os.environ.get("TOS_SECRET_KEY"):
        opts["secret"] = os.environ["TOS_SECRET_KEY"]
    return opts


class StreamingTOSRobotDataset(StreamingLeRobotDataset):
    """A :class:`StreamingLeRobotDataset` that reads a v3.0 dataset from Volcengine TOS."""

    def __init__(
        self,
        url: str,
        repo_id: str | None = None,
        *,
        storage_options: dict | None = None,
        meta_cache_dir: str | None = None,
        **kwargs,
    ):
        """
        Args:
            url: dataset root on TOS, e.g. ``tos://bucket/prefix/name`` (any fsspec URL works too).
            repo_id: optional label only (metadata is read from the mirrored ``meta/``, never the
                Hub). Defaults to the last path segment of ``url``.
            storage_options: fsspec kwargs; merged over the env-derived TOS credentials (explicit
                values win). Also registered as the protocol default so the video decoder's bare
                ``fsspec.open`` authenticates.
            meta_cache_dir: where to mirror ``meta/`` (default: a temp dir).
            **kwargs: forwarded to :class:`StreamingLeRobotDataset` (``episodes``,
                ``delta_timestamps``, ``image_transforms``, ``tolerance_s``, ``buffer_size``,
                ``max_num_shards``, ``seed``, ``shuffle``, ``return_uint8``, …).
        """
        self._url = url.rstrip("/")
        self._protocol, self._rpath = fsspec.core.split_protocol(self._url)
        self._rpath = (self._rpath or "").rstrip("/")

        so = _tos_env_storage_options()
        if storage_options:
            so.update(storage_options)  # explicit values win over the environment
        if self._protocol == "tos" and (not so.get("key") or not so.get("secret")):
            raise ValueError(
                "TOS credentials not found: set TOS_ACCESS_KEY and TOS_SECRET_KEY in the environment "
                "(optionally TOS_ENDPOINT / TOS_REGION), or pass storage_options={'key':…, 'secret':…}."
            )
        self.storage_options = dict(so)
        # instance-cached by fsspec, so this is the same object load_dataset/fsspec.open resolve.
        self._fs = fsspec.filesystem(self._protocol, **self.storage_options)

        # Make the credentials the default for this protocol, so the video decoder's bare
        # ``fsspec.open("<url>/…mp4")`` (which passes no storage_options) can authenticate.
        if self._protocol and self.storage_options:
            fsspec.config.conf.setdefault(self._protocol, {}).update(self.storage_options)

        # repo_id is only a label (metadata comes from the mirrored meta/); derive from the URL.
        repo_id = repo_id or (self._rpath.rsplit("/", 1)[-1] or "dataset")
        # Mirror meta/ locally and hand it to the parent as `root`: the stock metadata,
        # version-check, delta-timestamp and shuffle/shard setup then run unchanged. The
        # parent's data loading goes through the `_load_hf_dataset` seam overridden below.
        meta_root = self._mirror_meta(meta_cache_dir, repo_id)
        super().__init__(repo_id, root=meta_root, **kwargs)

    # ---- metadata mirror -------------------------------------------------
    def _mirror_meta(self, cache_dir: str | None, repo_id: str) -> str:
        local = cache_dir or tempfile.mkdtemp(prefix="tos_lerobot_")
        dst = os.path.join(local, repo_id.replace("/", "__"))
        meta_dst = os.path.join(dst, "meta")
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            os.makedirs(dst, exist_ok=True)
            # copy the small remote meta/ tree (info.json, stats.json, tasks.parquet,
            # episodes/chunk-*/file-*.parquet) — not data/ or videos/.
            self._fs.get(f"{self._rpath}/meta", meta_dst, recursive=True)
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            raise FileNotFoundError(
                f"no meta/info.json under {self._url}/meta — is this a LeRobot v3.0 dataset on TOS?"
            )
        return dst

    # ---- low-dim data: stream the parquet shards off the backend ---------
    def _load_hf_dataset(self) -> datasets.IterableDataset:
        ds = load_dataset(
            "parquet",
            data_files=f"{self._url}/data/*/*.parquet",
            storage_options=self.storage_options,
            split="train",
            streaming=True,
        )
        if self.episodes is not None:
            keep = {int(e) for e in self.episodes}
            # the parent ignores `episodes` when streaming; apply a lazy per-frame filter here.
            ds = ds.filter(lambda x: int(x["episode_index"]) in keep)
        return ds

    # ---- video: decoded straight off fsspec (no download) ----------------
    def _get_video_path(self, ep_idx: int, video_key: str) -> str:
        # lerobot's decoder opens this with fsspec.open(...) and lets torchcodec range-read it.
        return f"{self._url}/{self.meta.get_video_file_path(ep_idx, video_key)}"
