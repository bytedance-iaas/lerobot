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

"""fp8 (float8) training via `torchao <https://github.com/pytorch/ao>`_.

Swaps eligible ``nn.Linear`` layers in a policy for torchao's ``Float8Linear`` so their
matmuls run in fp8 on Hopper/Ada tensor cores. Master weights, optimizer state, gradients,
and every non-Linear op stay in bf16/fp32 — it composes with Accelerate's bf16 autocast;
only the Linear GEMMs are quantized. This is fp8 *training*, not post-training quantization.

fp8 does NOT help every model, so this is applied model-agnostically through a filter that
auto-skips layers where it is wrong or useless. Enabling it on a policy that cannot benefit
is a safe no-op (a warning is logged that 0 layers were converted), never a crash. A layer
is converted only when ALL of these hold:

* it is an ``nn.Linear``;
* both ``in_features`` and ``out_features`` are divisible by 16 (fp8 tensor-core tiling) —
  this alone excludes most action heads, whose output dim is the (small, arbitrary) action
  size;
* it is large enough to be worth it (``max(in, out) >= min_size``) — skips small RL/MLP
  projections where fp8 overhead outweighs the GEMM saving;
* at least one of its parameters is trainable — skips frozen pretrained backbones (e.g.
  smolvla/groot freeze the VLM by default), where converting would add cost for no gain.

See ``docs/source/fp8_training.mdx`` for the per-policy applicability analysis.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

# fp8 e4m3/e5m2 tensor-core matmul first appears on Ada Lovelace (sm_89) and Hopper (sm_90);
# Ampere (sm_80/sm_86, e.g. A30/A100) and older have no fp8 GEMM path.
FP8_MIN_COMPUTE_CAPABILITY: tuple[int, int] = (8, 9)


def is_fp8_supported() -> bool:
    """True only if CUDA is available AND every visible device has fp8 tensor cores.

    Requires ALL devices to qualify: a mixed A30+H20 job would silently fall back to a slow
    emulated path on the A30, so we treat the weakest device as the gate.
    """
    if not torch.cuda.is_available():
        return False
    return all(
        torch.cuda.get_device_capability(i) >= FP8_MIN_COMPUTE_CAPABILITY
        for i in range(torch.cuda.device_count())
    )


def assert_fp8_supported() -> None:
    """Raise a clear error when fp8 training is requested on hardware that cannot run it."""
    if torch.cuda.is_available():
        caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
        detail = f"GPU compute capabilities present: {sorted(caps)}"
    else:
        detail = "no CUDA device is visible"
    if not is_fp8_supported():
        raise RuntimeError(
            "fp8 training needs GPU compute capability >= "
            f"{FP8_MIN_COMPUTE_CAPABILITY[0]}.{FP8_MIN_COMPUTE_CAPABILITY[1]} on EVERY device "
            "(Ada sm_89 / Hopper sm_90+, e.g. H20/H100/L40S). "
            f"{detail}. Ampere (A30/A100, sm_80) and older have no fp8 tensor cores — "
            "train those in bf16 instead (drop --use_float8)."
        )


def _eligible(module: nn.Module, fqn: str, *, min_size: int, skip_fqn_substrings: tuple[str, ...]) -> bool:
    """torchao ``module_filter_fn``: return True to convert this module to fp8."""
    if not isinstance(module, nn.Linear):
        return False
    if module.in_features % 16 != 0 or module.out_features % 16 != 0:
        return False
    if max(module.in_features, module.out_features) < min_size:
        return False
    # `recurse=False`: judge THIS Linear's own weight/bias, not children (Linears have none).
    if not any(p.requires_grad for p in module.parameters(recurse=False)):
        return False
    if any(s in fqn for s in skip_fqn_substrings):
        return False
    return True


def apply_float8_training(
    model: nn.Module,
    *,
    recipe: str = "rowwise",
    min_size: int = 256,
    skip_fqn_substrings: tuple[str, ...] = ("lm_head",),
    enforce_hardware: bool = True,
) -> list[str]:
    """Convert eligible ``nn.Linear`` layers of ``model`` in place to fp8 training.

    Must be called AFTER the policy is built (and any freezing / PEFT wrapping applied, so
    ``requires_grad`` is final) and BEFORE the optimizer is created and before
    ``accelerator.prepare()`` — the optimizer must capture the swapped parameters.

    Args:
        model: the policy (an ``nn.Module``); modified in place.
        recipe: torchao float8 recipe — ``"tensorwise"`` (fastest), ``"rowwise"`` (more
            accurate, default), or ``"rowwise_with_gw_hp"``.
        min_size: skip Linear layers whose larger dim is below this (overhead > benefit).
        skip_fqn_substrings: skip any layer whose fully-qualified name contains one of these.
        enforce_hardware: raise if the GPUs lack fp8 tensor cores. Set False only to test the
            filter on CPU.

    Returns:
        The fully-qualified names of the layers that were converted (``[]`` if none — the
        caller should warn, since it means fp8 had no effect on this model).
    """
    if enforce_hardware:
        assert_fp8_supported()

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as e:
        raise ImportError(
            "fp8 training needs torchao. Install it with `pip install lerobot[fp8]` "
            "(or `pip install torchao`)."
        ) from e

    config = Float8LinearConfig.from_recipe_name(recipe)

    converted: list[str] = []

    def module_filter_fn(module: nn.Module, fqn: str) -> bool:
        keep = _eligible(module, fqn, min_size=min_size, skip_fqn_substrings=skip_fqn_substrings)
        if keep:
            converted.append(fqn)
        return keep

    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)

    if converted:
        logging.info(
            "fp8: converted %d nn.Linear layer(s) to torchao Float8Linear (recipe=%s). "
            "For real speedup, run with torch.compile.",
            len(converted),
            recipe,
        )
    else:
        logging.warning(
            "fp8: requested but 0 layers were converted — this model has no eligible trainable "
            "nn.Linear (dims divisible by 16, size >= %d, not frozen). fp8 has NO effect here; "
            "conv/UNet or frozen-backbone policies fall in this case. See docs/source/fp8_training.mdx.",
            min_size,
        )
    return converted
