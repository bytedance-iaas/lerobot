"""Opt-in single-NPU gradient scaling using the foreach Scalar kernel."""

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from accelerate import Accelerator


@torch.no_grad()
def clip_grad_norm_npu_(
    accelerator: "Accelerator",
    parameters: Iterable[torch.Tensor] | torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor | None:
    """Clip single-NPU gradients, preserving Accelerator behavior on other paths.

    Keep the original global norm and FP32/BF16 coefficient precision. Reading
    the coefficient once on the host enables NPU's fused foreach Scalar kernel.
    This intentionally adds one synchronization; do not cast the coefficient
    to the gradient dtype, which introduces additional BF16 rounding.
    """
    params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    distributed_type = getattr(accelerator.distributed_type, "value", accelerator.distributed_type)
    supported = (
        distributed_type == "NO"
        and accelerator.num_processes == 1
        and bool(grads)
        and all(
            type(g) is torch.Tensor
            and g.device.type == "npu"
            and g.dtype in (torch.float32, torch.bfloat16)
            and not g.is_sparse
            and g.is_contiguous()
            for g in grads
        )
        and len({g.device for g in grads}) == 1
    )
    if not supported:
        # In particular, preserve FSDP/DeepSpeed/DTensor reduction and AMP logic.
        # Noncontiguous NPU gradients must not use the Scalar fast path.
        return accelerator.clip_grad_norm_(params, max_norm, norm_type=norm_type)

    accelerator.unscale_gradients()
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type)
    coefficient = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0).item()
    grouped: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
    for grad in grads:
        grouped[grad.dtype].append(grad)
    for group in grouped.values():
        torch._foreach_mul_(group, coefficient)
    return total_norm

