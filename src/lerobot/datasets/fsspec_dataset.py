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
"""Stream a LeRobot v3.0 dataset from an fsspec object store (Volcengine TOS, S3, GCS…).

:class:`StreamingLeRobotDataset` streams only from the HF Hub or a local dir: it wraps
``root`` in ``Path`` (so ``tos://…`` / ``s3://…`` URLs break), passes no ``storage_options``
to ``load_dataset``, and decodes video from ``{root}/…mp4`` — a path torchcodec can only
open when it is local or HTTPS. :class:`FsspecLeRobotDataset` subclasses it and swaps in the
three fsspec-aware seams:

1. **metadata** — the small ``meta/`` tree is mirrored to a local temp dir via fsspec, then
   read by :class:`LeRobotDatasetMetadata` (a few MB; not ``data/`` or ``videos/``).
2. **low-dim data** — ``load_dataset("parquet", data_files="<url>/data/*/*.parquet",
   storage_options=…, streaming=True)`` streams the parquet shards over fsspec.
3. **video** — ``_query_videos`` is overridden to hand torchcodec a short-lived **presigned
   HTTPS URL** (``fs.sign``) so it range-reads only the bytes it needs.

Everything else (the Backtrackable buffer for ``delta_timestamps``, buffer-shuffle, sharding,
per-item construction) is inherited unchanged.

You build the ``tos://`` (or ``s3://``) URL yourself and pass credentials via
``storage_options`` — same connection as the ``tosfs`` fsspec impl
(``fsspec.filesystem("tos", key, secret, region, endpoint)``). Install ``tosfs`` (or
``tosfsspec``) for the ``tos://`` protocol. Never hardcode secrets — read them from the env.

Example::

    import os
    ds = FsspecLeRobotDataset(
        "tos://my-bucket/lerobot-datasets/finish_sandwich",
        storage_options={
            "key": os.environ["TOS_ACCESS_KEY"],
            "secret": os.environ["TOS_SECRET_KEY"],
            "endpoint": "https://tos-cn-beijing.volces.com",
            "region": "cn-beijing",
        },
        episodes=[0, 3, 17],   # optional held-out subset
    )
    for item in ds:  # IterableDataset — iterate, no ds[i]
        item["observation.images.front"]  # (C, H, W); also item["observation.state"], ["action"]
        break
"""

from __future__ import annotations

import os
import tempfile
import time

import fsspec
import numpy as np
import torch
from datasets import load_dataset

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .feature_utils import get_delta_indices
from .streaming_dataset import StreamingLeRobotDataset
from .utils import check_version_compatibility
from .video_utils import decode_video_frames_torchcodec


class FsspecLeRobotDataset(StreamingLeRobotDataset):
    """A :class:`StreamingLeRobotDataset` that reads a v3.0 dataset from any fsspec URL."""

    def __init__(
        self,
        url: str,
        repo_id: str | None = None,
        *,
        storage_options: dict | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
        video_url_resolver=None,
        video_url_expiry: int = 3600,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        return_uint8: bool = False,
        meta_cache_dir: str | None = None,
    ):
        """
        Args:
            url: dataset root on the backend, e.g. ``tos://bucket/prefix`` or ``s3://bucket/prefix``.
            repo_id: optional label only (metadata is read from the mirrored ``meta/``, never the
                Hub). Defaults to the last path segment of ``url``.
            storage_options: fsspec kwargs for the backend (TOS: ``key``/``secret``/``endpoint``/``region``).
            fs: a prebuilt fsspec filesystem (else built from the url protocol + ``storage_options``).
            video_url_resolver: optional ``fn(rel_mp4_path) -> https_url`` overriding the default
                presign (``fs.sign``).
            video_url_expiry: presigned-URL lifetime in seconds (and the memo TTL).
            episodes: restrict to these episode ids (streaming filter) — for train/eval splits.
            (remaining args mirror :class:`StreamingLeRobotDataset`.)
        """
        # NOTE: intentionally does NOT call super().__init__ — it would Path-mangle the URL and
        # load_dataset without storage_options. We replicate its setup with the fsspec seams.
        torch.utils.data.IterableDataset.__init__(self)

        self._url = url.rstrip("/")
        self._protocol, self._rpath = fsspec.core.split_protocol(self._url)
        self._rpath = (self._rpath or "").rstrip("/")
        # repo_id is only a label (metadata comes from the mirrored meta/); derive from the URL.
        self.repo_id = repo_id or (self._rpath.rsplit("/", 1)[-1] or "dataset")
        self.storage_options = dict(storage_options or {})
        self._fs = fs or fsspec.filesystem(self._protocol, **self.storage_options)
        self._video_url_resolver = video_url_resolver
        self._video_expiry = video_url_expiry
        self._video_url_cache: dict[str, tuple[float, str]] = {}

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = CODEBASE_VERSION
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle
        self.streaming = True
        self.streaming_from_local = False  # our _query_videos handles the video path, not self.root
        self.buffer_size = buffer_size
        self._return_uint8 = return_uint8
        self.video_decoder_cache = None

        # 1) metadata: mirror the small meta/ tree locally so LeRobotDatasetMetadata reads it.
        self._meta_root = self._mirror_meta(meta_cache_dir)
        self.meta = LeRobotDatasetMetadata(repo_id, root=self._meta_root, revision=None)
        self.root = self.meta.root
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        # 2) delta-timestamp windows (inherited validator + index math)
        self.delta_timestamps = None
        self.delta_indices = None
        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        # 3) low-dim data: stream the parquet shards straight off the backend via fsspec.
        self.hf_dataset = load_dataset(
            "parquet",
            data_files=f"{self._url}/data/*/*.parquet",
            storage_options=self.storage_options,
            split="train",
            streaming=True,
        )
        if episodes is not None:
            keep = {int(e) for e in episodes}
            # the stock streaming class ignores `episodes`; apply a lazy per-frame filter here.
            self.hf_dataset = self.hf_dataset.filter(lambda x, k=keep: int(x["episode_index"]) in k)
        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

    # ---- metadata mirror -------------------------------------------------
    def _mirror_meta(self, cache_dir: str | None) -> str:
        local = cache_dir or tempfile.mkdtemp(prefix="fsspec_lerobot_")
        dst = os.path.join(local, self.repo_id.replace("/", "__"))
        meta_dst = os.path.join(dst, "meta")
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            os.makedirs(dst, exist_ok=True)
            # copy the small remote meta/ tree (info.json, stats.json, tasks.parquet,
            # episodes/chunk-*/file-*.parquet) — not data/ or videos/.
            self._fs.get(f"{self._rpath}/meta", meta_dst, recursive=True)
        if not os.path.exists(os.path.join(meta_dst, "info.json")):
            raise FileNotFoundError(
                f"no meta/info.json under {self._url}/meta — is this a LeRobot v3.0 dataset on the backend?"
            )
        return dst

    # ---- video: presigned HTTPS URLs torchcodec can range-read ----------
    def _video_url(self, rel: str) -> str:
        if self._video_url_resolver is not None:
            return self._video_url_resolver(rel)
        # local backends (file:// or a FUSE mount): torchcodec reads the path directly.
        if self._protocol in (None, "", "file", "local"):
            return f"{self._rpath}/{rel}"
        now = time.time()
        cached = self._video_url_cache.get(rel)
        if cached and cached[0] > now:
            return cached[1]
        key = f"{self._rpath}/{rel}"  # bucket/prefix/videos/.../file.mp4  (no protocol)
        url = None
        # (a) fsspec-standard signing (s3fs implements it; tosfs inherits the base stub → raises)
        try:
            url = self._fs.sign(key, expiration=self._video_expiry)
        except (NotImplementedError, AttributeError):
            url = None
        # (b) TOS: presign via the `tos` SDK (bundled with tosfs/tosfsspec)
        if url is None and self._protocol.startswith("tos"):
            url = self._tos_presign(key)
        if url is None:
            raise NotImplementedError(
                f"backend '{self._protocol}' cannot presign video URLs; pass video_url_resolver=fn(rel)->https"
            )
        self._video_url_cache[rel] = (now + self._video_expiry - 60, url)
        return url

    def _tos_presign(self, bucket_key: str) -> str:
        import tos

        bucket, _, obj_key = bucket_key.partition("/")
        so = self.storage_options
        client = tos.TosClientV2(so["key"], so["secret"], so["endpoint"], so.get("region"))
        out = client.pre_signed_url(
            tos.HttpMethodType.Http_Method_Get, bucket, obj_key, expires=self._video_expiry
        )
        return out.signed_url

    def _query_videos(self, query_timestamps: dict, ep_idx: int) -> dict:
        item = {}
        for video_key, query_ts in query_timestamps.items():
            rel = str(self.meta.get_video_file_path(ep_idx, video_key))
            frames = decode_video_frames_torchcodec(
                self._video_url(rel),
                query_ts,
                self.tolerance_s,
                decoder_cache=self.video_decoder_cache,
                return_uint8=self._return_uint8,
            )
            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames
        return item
