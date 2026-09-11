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
"""Run the simulator somewhere else.

Closed-loop eval needs a policy and a simulator in the same loop, but they do not need the
same machine, and sometimes they cannot share one: an Ascend host runs the policy and has
no GL stack at all (no ``/dev/dri``, software OSMesa only), while the CUDA box that renders
in hardware cannot run the NPU checkpoint. This splits the loop at ``gym.vector.VectorEnv``:
:class:`RemoteVectorEnv` forwards ``reset``/``step`` to a server built by
``lerobot.envs.remote_server``, so ``lerobot_eval`` keeps calling the same two methods and
does not know the difference.

Payloads are pickled, like the rest of ``lerobot.transport``. Gym hands back whatever an
env decided to put in an observation dict, an ``info`` dict or a ``Space``; pickle carries
all of it without a per-type encoding rule to keep in sync as envs change.

That assumes both ends sit on a trusted network, which is the normal deployment -- the
simulator host is a GPU box in the same cluster. ``pickle.load`` runs what the request body
says, so if this ever gets published through a gateway, the gateway has to authenticate
every call; nothing in this module will.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.envs.configs import EnvConfig

# ---------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------


@EnvConfig.register_subclass("remote")
@dataclass
class RemoteEnvConfig(EnvConfig):
    """Points eval at a simulator running elsewhere.

    ``target`` is the gRPC authority (``host:port``). ``remote`` carries the env config the
    *server* should build -- its ``type`` is the real env type (``libero``, ``pusht``, ...),
    resolved there, not here.
    """

    target: str = ""
    secure: bool = False
    remote: dict[str, Any] = field(default_factory=dict)
    n_envs: int = 1
    timeout_s: float = 120.0
    task: str | None = None
    fps: int = 30
    episode_length: int = 400
    features: dict = field(default_factory=dict)
    features_map: dict = field(default_factory=dict)

    def __post_init__(self):
        """Adopt the real env config's features without building anything.

        make_policy() derives the policy's input/output features from the env config, and
        it does that before any env exists. Those features are static metadata declared on
        the concrete EnvConfig class, so instantiating that class locally is enough -- no
        round trip, and no need for the env's runtime dependencies on this side. The result
        is that --env.type=remote --env.remote='{"type": "pusht"}' presents the policy with
        exactly what --env.type=pusht would.
        """
        if not self.remote or "type" not in self.remote:
            return  # validated in make_remote_envs, which produces the actionable message
        spec = {k: v for k, v in self.remote.items() if k != "type"}
        try:
            inner = EnvConfig.get_choice_class(self.remote["type"])(**spec)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"cannot build the local view of env.remote {self.remote['type']!r}: {e}. "
                "The client needs the env type registered (it is, if it appears in "
                "--env.type's choices), not the env's runtime dependencies."
            ) from e
        self.features = inner.features
        self.features_map = inner.features_map
        self.task = inner.task
        self.fps = inner.fps
        self.episode_length = inner.episode_length

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        """Hand back proxies instead of building anything locally.

        make_env() ends in cfg.create_envs(), so overriding here is the whole integration --
        lerobot_eval, factory.py and the eval loop stay untouched. n_envs from the caller
        wins over the field, since --env.n_envs and eval's own batching are the same number.
        The server decides sync vs async; use_async_envs is meaningless across a wire.
        """
        self.n_envs = n_envs
        return make_remote_envs(self)


# ---------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------


def _reassemble(chunk_iter) -> bytes:
    """Concatenate a Chunk stream. The server always terminates it, so no state machine."""
    return b"".join(c.data for c in chunk_iter)


class RemoteVectorEnv:
    """A ``gym.vector.VectorEnv`` stand-in whose steps happen on another machine.

    Only what ``lerobot_eval`` touches is implemented -- ``reset``, ``step``, ``close``,
    ``num_envs`` and the two spaces. Deliberately not a ``VectorEnv`` subclass: inheriting
    would bring along ``call``/``get_attr``/autoreset plumbing that would each need its own
    round trip, and pretending to support them is worse than not offering them.
    """

    def __init__(self, stub, suite: str, task_id: int, handle: dict, timeout_s: float):
        self._stub = stub
        self._suite = suite
        self._task_id = int(task_id)
        self._timeout = timeout_s
        self.num_envs = int(handle["num_envs"])
        self.observation_space = handle["observation_space"]
        self.action_space = handle["action_space"]

    def _call(self, rpc, payload: dict | None):
        # Imported here, not at module scope: lerobot.transport pulls in protobuf, which
        # lives behind the optional grpcio-dep extra. Registering this env type must not
        # make that extra mandatory for everyone who imports lerobot.envs.
        from lerobot.transport import services_pb2
        from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

        req = services_pb2.EnvCall(
            suite=self._suite,
            task_id=self._task_id,
            payload=python_object_to_bytes(payload) if payload is not None else b"",
        )
        return bytes_to_python_object(_reassemble(rpc(req, timeout=self._timeout)))

    def reset(self, seed=None, options=None):
        out = self._call(self._stub.Reset, {"seed": seed, "options": options})
        return out["obs"], out["info"]

    def step(self, action):
        out = self._call(self._stub.Step, {"action": np.asarray(action)})
        return out["obs"], out["reward"], out["terminated"], out["truncated"], out["info"]

    def call(self, name: str, *args, **kwargs):
        """Mirror of VectorEnv.call(): returns one value per sub-env.

        rollout() probes optional attributes by calling them and catching AttributeError /
        NotImplementedError -- `task_description`, then `task`, then giving up. gRPC turns
        any server-side exception into _InactiveRpcError, which those handlers would not
        catch, so the server reports the condition as NOT_FOUND and we raise the original
        type again here. Without this the first probe aborts the whole eval.
        """
        import grpc

        try:
            out = self._call(self._stub.Call, {"kind": "call", "name": name, "args": args, "kwargs": kwargs})
        except grpc.RpcError as e:
            if e.code() is grpc.StatusCode.NOT_FOUND:
                detail = e.details() or ""
                if detail.startswith("NotImplementedError"):
                    raise NotImplementedError(detail) from None
                raise AttributeError(detail) from None
            raise
        return out["result"]

    @property
    def unwrapped(self):
        """rollout() reads env.unwrapped.metadata for the render fps, nothing else."""
        return _RemoteUnwrapped(self)

    def close(self):
        from lerobot.transport import services_pb2

        # Closing a channel the server already dropped must not fail an eval that finished.
        with contextlib.suppress(Exception):
            self._stub.Close(services_pb2.Empty(), timeout=self._timeout)

    def __repr__(self):
        return f"RemoteVectorEnv(suite={self._suite!r}, task_id={self._task_id}, num_envs={self.num_envs})"


class _RemoteUnwrapped:
    """Just enough of `env.unwrapped` for eval: the metadata dict, fetched once."""

    def __init__(self, env: RemoteVectorEnv):
        self._env = env
        self._metadata = None

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = self._env._call(self._env._stub.Call, {"kind": "attr", "name": "metadata"})[
                "result"
            ]
        return self._metadata


def make_remote_envs(cfg: RemoteEnvConfig) -> dict[str, dict[int, RemoteVectorEnv]]:
    """Ask the server to build ``cfg.remote`` and return proxies in ``make_env``'s shape."""
    import grpc

    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import (
        bytes_to_python_object,
        grpc_channel_options,
        python_object_to_bytes,
    )

    if not cfg.target:
        raise ValueError("env.target is required, e.g. --env.target=host:18990")
    if not cfg.remote or "type" not in cfg.remote:
        raise ValueError(
            "env.remote must carry the env the SERVER should build, including its type, e.g. "
            '--env.remote=\'{"type": "pusht"}\''
        )

    options = grpc_channel_options()
    channel = (
        grpc.secure_channel(cfg.target, grpc.ssl_channel_credentials(), options=options)
        if cfg.secure
        else grpc.insecure_channel(cfg.target, options=options)
    )
    stub = services_pb2_grpc.RemoteEnvStub(channel)

    spec = dict(cfg.remote)
    spec["n_envs"] = cfg.n_envs
    reply = bytes_to_python_object(
        _reassemble(stub.Make(services_pb2.EnvSpec(spec=python_object_to_bytes(spec)), timeout=cfg.timeout_s))
    )

    out: dict[str, dict[int, RemoteVectorEnv]] = {}
    for h in reply["handles"]:
        env = RemoteVectorEnv(stub, h["suite"], h["task_id"], h, cfg.timeout_s)
        out.setdefault(h["suite"], {})[int(h["task_id"])] = env
    return out
