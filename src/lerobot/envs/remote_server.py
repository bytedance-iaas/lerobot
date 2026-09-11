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
"""Serve gym environments over gRPC, so eval can run where the policy is.

    python -m lerobot.envs.remote_server --port 18990

The client is :class:`lerobot.envs.remote.RemoteVectorEnv`; see that module for why the
split exists.

.. warning::
   Requests are unpickled, so anyone who can open a connection here can run code here.
   Bind this to an internal network. It has no authentication of its own and is not meant
   to grow any -- put a gateway in front if it must be reachable from outside.
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent import futures

import grpc

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_python_object,
    grpc_channel_options,
    python_object_to_bytes,
    send_bytes_in_chunks,
)

MAX_WORKERS = 4


class RemoteEnvService(services_pb2_grpc.RemoteEnvServicer):
    """Holds the envs `Make` built, keyed by the (suite, task_id) the client was handed.

    One env set per server process. Two eval runs must not share one: gym envs carry
    episode state, and interleaved resets from two clients would silently corrupt both
    runs' returns rather than fail.
    """

    def __init__(self):
        self._envs: dict[tuple[str, int], object] = {}
        self._lock = threading.Lock()

    # -- helpers ------------------------------------------------------------------------

    def _get(self, req, context):
        key = (req.suite, int(req.task_id))
        env = self._envs.get(key)
        if env is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"no env for suite={req.suite!r} task_id={req.task_id}; call Make first "
                f"(have {sorted(self._envs)})",
            )
        return env

    @staticmethod
    def _reply(payload):
        yield from send_bytes_in_chunks(
            python_object_to_bytes(payload), services_pb2.Chunk, log_prefix="[ENV]"
        )

    # -- rpcs ---------------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def Make(self, request, context):  # noqa: N802
        from lerobot.configs.types import PipelineFeatureType  # noqa: F401  (env import side effects)
        from lerobot.envs.factory import make_env

        spec = bytes_to_python_object(request.spec)
        n_envs = int(spec.pop("n_envs", 1))
        use_async = bool(spec.pop("use_async_envs", False))
        env_type = spec.pop("type")

        cls = type(self)._env_config_class(env_type)
        cfg = cls(**spec)
        logging.info("[ENV] Make type=%s n_envs=%d async=%s", env_type, n_envs, use_async)

        with self._lock:
            self.close_all()
            built = make_env(cfg, n_envs=n_envs, use_async_envs=use_async)
            handles = []
            for suite, by_task in built.items():
                for task_id, env in by_task.items():
                    self._envs[(suite, int(task_id))] = env
                    handles.append(
                        {
                            "suite": suite,
                            "task_id": int(task_id),
                            "num_envs": int(env.num_envs),
                            "observation_space": env.observation_space,
                            "action_space": env.action_space,
                        }
                    )
        logging.info("[ENV] built %d handle(s)", len(handles))
        yield from self._reply({"handles": handles})

    def Reset(self, request, context):  # noqa: N802
        env = self._get(request, context)
        kwargs = bytes_to_python_object(request.payload) if request.payload else {}
        obs, info = env.reset(seed=kwargs.get("seed"), options=kwargs.get("options"))
        yield from self._reply({"obs": obs, "info": info})

    def Step(self, request, context):  # noqa: N802
        env = self._get(request, context)
        action = bytes_to_python_object(request.payload)["action"]
        obs, reward, terminated, truncated, info = env.step(action)
        yield from self._reply(
            {
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )

    def Close(self, request, context):  # noqa: N802
        with self._lock:
            self.close_all()
        return services_pb2.Empty()

    def close_all(self):
        for env in self._envs.values():
            try:
                env.close()
            except Exception:  # noqa: BLE001 - a failed close must not block a new Make
                logging.exception("[ENV] close failed")
        self._envs.clear()

    @staticmethod
    def _env_config_class(env_type: str):
        from lerobot.envs.configs import EnvConfig

        try:
            return EnvConfig.get_choice_class(env_type)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"unknown env type {env_type!r}; the simulator host must have the extra that "
                f"registers it installed"
            ) from e


def serve(port: int, max_workers: int = MAX_WORKERS) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=grpc_channel_options(),
    )
    services_pb2_grpc.add_RemoteEnvServicer_to_server(RemoteEnvService(), server)
    # Plaintext h2c: TLS is terminated by the gateway in front of this, and adding a second
    # layer here would mean shipping a cert into the pod for no gain.
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logging.info("[ENV] serving on :%d", port)
    server.wait_for_termination()


def main() -> None:
    p = argparse.ArgumentParser(description="Serve gym envs over gRPC for remote eval")
    p.add_argument("--port", type=int, default=18990)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    serve(a.port, a.max_workers)


if __name__ == "__main__":
    main()
