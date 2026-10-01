"""Measure real LeRobot training; startup/JIT and validation are outside timed steps."""

from __future__ import annotations

import functools
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

from tile_adapter import is_candidate  # noqa: E402
from tile_ops_adapter import set_enabled  # noqa: E402

from lerobot.scripts import lerobot_train  # noqa: E402

rank = int(os.environ.get("RANK", "0"))
root = Path(os.environ["RUN_ROOT"])
root.mkdir(parents=True, exist_ok=True)
enabled_ops = {op for op in os.environ.get("TK_OPS", "").split(",") if op}


def configure_ops(active: bool) -> None:
    ops = enabled_ops if active else set()
    set_enabled(rms="rms" in ops, swiglu="swiglu" in ops, rope="rope" in ops)


configure_ops(True)
records: list[dict[str, object]] = []
handles: list[torch.utils.hooks.RemovableHandle] = []
census: Counter[tuple[object, ...]] = Counter()
previous_end: float | None = None


def rng_state() -> tuple[object, ...]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )


def restore_rng(state: tuple[object, ...]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def check_training_pair(policy, batch, optimizer, accelerator) -> None:
    state = rng_state()
    samples: list[dict[str, torch.Tensor]] = []
    losses: list[float] = []
    policy.train()
    # Repeat reference once to distinguish kernel differences from model noise.
    for use_tile in (False, False, True):
        configure_ops(use_tile)
        restore_rng(state)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss, _ = policy.forward(batch)
        accelerator.backward(loss)
        losses.append(loss.item())
        snapshot = {}
        for name, param in accelerator.unwrap_model(policy).named_parameters():
            if param.grad is not None:
                flat = param.grad.detach().reshape(-1)
                stride = max(1, flat.numel() // 2048)
                snapshot[name] = flat[::stride][:2048].float().cpu()
        samples.append(snapshot)
        del loss
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state)
    configure_ops(True)

    assert samples[0].keys() == samples[1].keys() == samples[2].keys(), "gradient presence changed"
    delta2 = ref2 = repeat_delta2 = max_abs = 0.0
    for name, ref in samples[0].items():
        actual = samples[2][name]
        assert torch.isfinite(actual).all(), name
        delta = (actual - ref).double()
        delta2 += delta.square().sum().item()
        ref2 += ref.double().square().sum().item()
        repeat_delta2 += (samples[1][name] - ref).double().square().sum().item()
        max_abs = max(max_abs, delta.abs().max().item())
    relative_l2 = (delta2 / max(ref2, 1e-30)) ** 0.5
    loss_relative = abs(losses[2] - losses[0]) / max(abs(losses[0]), 1e-12)
    result = {
        "baseline_loss": losses[0],
        "baseline_repeat_loss": losses[1],
        "tile_loss": losses[2],
        "baseline_repeat_gradient_relative_l2": (repeat_delta2 / max(ref2, 1e-30)) ** 0.5,
        "loss_relative_error": loss_relative,
        "sampled_gradient_relative_l2": relative_l2,
        "sampled_gradient_max_abs": max_abs,
        "gradient_tensors": len(samples[0]),
        "sampled_gradient_elements": sum(x.numel() for x in samples[0].values()),
        "note": "Up to 2048 evenly spaced elements per gradient tensor; same weights, batch and RNG.",
    }
    (root / f"paired-check-rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    assert loss_relative < 0.006, result
    assert relative_l2 < 0.05, result
    del samples
    torch.cuda.empty_cache()


def observe(module, args, output) -> None:
    x = args[0]
    y = output[0] if isinstance(output, tuple) else output
    key = (
        type(module).__name__,
        tuple(x.shape),
        str(x.dtype),
        str(y.dtype),
        getattr(module, "cond_dim", None),
        bool(x.requires_grad),
    )
    census[key] += 1


original_update = lerobot_train.update_policy


@functools.wraps(original_update)
def update(*args, **kwargs):
    global previous_end
    policy, batch, optimizer = args[1:4]
    accelerator = kwargs["accelerator"]
    if not records:
        if os.environ.get("TK_VERIFY") == "1":
            check_training_pair(policy, batch, optimizer, accelerator)
            if os.environ.get("TK_VALIDATE_ONLY") == "1":
                raise SystemExit(0)
        unwrapped = accelerator.unwrap_model(policy)
        groups = {}
        for name, module in unwrapped.named_children():
            groups[name] = {
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "parameter_dtypes": dict(Counter(str(p.dtype) for p in module.parameters())),
            }
        candidates = [(n, m) for n, m in unwrapped.named_modules() if is_candidate(m)]
        assert candidates, "No RMSNorm candidates found"
        metadata = {
            "rank": rank,
            "world_size": accelerator.num_processes,
            "tile_ops": sorted(enabled_ops),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "groups": groups,
            "candidate_modules": [n for n, _ in candidates],
            "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
            "policy_type": unwrapped.config.type,
            "dtype": getattr(unwrapped.config, "dtype", None),
            "model_params_fp32": getattr(unwrapped.config, "model_params_fp32", None),
            "ddp": (policy._get_ddp_logging_data() if hasattr(policy, "_get_ddp_logging_data") else None),
        }
        (root / f"metadata-rank{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        handles.extend(m.register_forward_hook(observe) for _, m in candidates)

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_update(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    row = {
        "step": len(records) + 1,
        "rank": rank,
        "update_wall_s": end - start,
        "iteration_interval_s": end - previous_end if previous_end else None,
        "dataloading_s": float(result[0].dataloading_s.val),
        "loss": float(result[0].loss.val),
        "grad_norm": float(result[0].grad_norm.val),
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    previous_end = end
    records.append(row)
    with (root / f"steps-rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    if handles:
        for handle in handles:
            handle.remove()
        handles.clear()
        census_rows = [
            {
                "class": key[0],
                "input_shape": key[1],
                "input_dtype": key[2],
                "output_dtype": key[3],
                "cond_dim": key[4],
                "input_requires_grad": key[5],
                "calls": count,
            }
            for key, count in census.items()
        ]
        (root / f"norm-census-rank{rank}.json").write_text(json.dumps(census_rows, indent=2) + "\n")
    if row["step"] % 10 == 0:
        print("PERF_STEP", json.dumps(row), flush=True)
    return result


lerobot_train.update_policy = update
try:
    lerobot_train.main()
finally:
    if records:
        measured = records[20:]
        summary = {
            "rank": rank,
            "steps": len(records),
            "warmup_steps": 20,
            "measured_steps": len(measured),
            "mean_update_s": (
                float(np.mean([row["update_wall_s"] for row in measured])) if measured else None
            ),
            "mean_iteration_s": (
                float(np.mean([row["iteration_interval_s"] for row in measured])) if measured else None
            ),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
        }
        (root / f"summary-rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
