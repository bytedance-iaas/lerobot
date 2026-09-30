import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

try:  # optional: Ascend NPU fused kernels
    import torch_npu
except ImportError:
    torch_npu = None

_shared_adamw: Any | None = None
_shared_adamw_load_attempted = False


def _load_shared_adamw() -> Any | None:
    """Build the batched binding once, falling back on unsupported toolchains."""
    global _shared_adamw, _shared_adamw_load_attempted
    if _shared_adamw_load_attempted:
        return _shared_adamw
    _shared_adamw_load_attempted = True
    if torch_npu is None:
        return None

    try:
        from torch.utils.cpp_extension import load

        npu_root = Path(torch_npu.__file__).parent
        source = Path(__file__).with_name("_npu_adamw.cpp")
        _shared_adamw = load(
            name="lerobot_npu_adamw",
            sources=[str(source)],
            extra_include_paths=[
                str(npu_root / "include"),
                str(npu_root / "include" / "third_party" / "op-plugin"),
                str(npu_root / "include" / "third_party" / "acl" / "inc"),
            ],
            extra_cflags=["-O2"],
            extra_ldflags=[
                f"-L{npu_root / 'lib'}",
                "-ltorch_npu",
                f"-Wl,-rpath,{npu_root / 'lib'}",
            ],
            verbose=False,
        )
    except Exception as error:
        logging.warning("Could not build shared-scalar NPU AdamW; using the compatible path: %s", error)
    return _shared_adamw


def is_npu_fused_adamw_available() -> bool:
    """Whether ``npu_apply_adam_w`` can actually be dispatched on this host."""
    if torch_npu is None or not hasattr(torch_npu, "npu_apply_adam_w"):
        return False
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


class NpuFusedAdamW(torch.optim.Optimizer):
    """Ascend AdamW with scalar tensors shared across parameters at each step.

    The installed torch_npu wrapper copies all seven scalar inputs to NPU for
    every parameter. When its C++ headers are available, a small cached binding
    calls the same ApplyAdamW kernel while sharing those tensors per parameter
    group. The original wrapper remains the automatic compatibility fallback.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if torch_npu is None or not hasattr(torch_npu, "npu_apply_adam_w"):
            raise RuntimeError(
                "NpuFusedAdamW requires torch_npu with npu_apply_adam_w; use torch.optim.AdamW instead."
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self._shared_adamw = _load_shared_adamw()
        logging.info(
            "NpuFusedAdamW backend: %s",
            "shared-scalar ApplyAdamW" if self._shared_adamw is not None else "compatible ApplyAdamW",
        )

    def _init_state(self, parameter: torch.Tensor) -> dict[str, Any]:
        state = self.state[parameter]
        if not state:
            # LeRobot serializes optimizer state with safetensors, which requires tensor values.
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        elif state["step"].device.type != "cpu":
            # Preserve the established checkpoint schema if a native fused checkpoint is loaded.
            state["step"] = state["step"].cpu()
        return state

    def _compatible_update(self, group: dict[str, Any], parameter: torch.Tensor, step: int) -> None:
        state = self.state[parameter]
        beta1, beta2 = group["betas"]
        grad = self._contiguous_grad(parameter)
        torch_npu.npu_apply_adam_w(
            beta1 ** (step - 1),
            beta2 ** (step - 1),
            group["lr"],
            group["weight_decay"],
            beta1,
            beta2,
            group["eps"],
            grad,
            None,
            False,
            False,
            out=(parameter, state["exp_avg"], state["exp_avg_sq"]),
        )

    @staticmethod
    def _contiguous_grad(parameter: torch.Tensor) -> torch.Tensor:
        grad = parameter.grad
        if grad is None:
            raise RuntimeError("Expected an initialized gradient")
        return grad if grad.is_contiguous() else grad.contiguous()

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            buckets: dict[tuple[torch.device, torch.dtype, int], list[torch.Tensor]] = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("NpuFusedAdamW does not support sparse gradients")
                state = self._init_state(parameter)
                state["step"] += 1
                step = int(state["step"].item())
                if self._shared_adamw is None:
                    self._compatible_update(group, parameter, step)
                else:
                    buckets[parameter.device, parameter.dtype, step].append(parameter)

            for (device, dtype, step), parameters in buckets.items():
                beta1, beta2 = group["betas"]
                values = (
                    beta1 ** (step - 1),
                    beta2 ** (step - 1),
                    group["lr"],
                    group["weight_decay"],
                    beta1,
                    beta2,
                    group["eps"],
                )
                # Separate zero-dimensional allocations are required by aclnnApplyAdamW;
                # views obtained by unbinding one vector are rejected by its tiling check.
                scalars = [torch.tensor(value, device=device, dtype=dtype) for value in values]
                self._shared_adamw.adamw(
                    parameters,
                    [self._contiguous_grad(p) for p in parameters],
                    [self.state[p]["exp_avg"] for p in parameters],
                    [self.state[p]["exp_avg_sq"] for p in parameters],
                    scalars,
                )

        return loss
