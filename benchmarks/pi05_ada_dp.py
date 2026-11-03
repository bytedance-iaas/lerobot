#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Measure full Pi0.5 data parallel training on Ada or CUDA, one process per card.

Run with torchrun --standalone --nproc-per-node=N. Each rank loads real PushT
batches and trains the full model. Warmup and final correctness checks are excluded
from timing. Optional model saving happens after timing.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", choices=("ada", "cuda"), default="ada")
    parser.add_argument("--cpu-cores", help="Ordered physical CPU IDs, comma separated; four cores per rank")
    parser.add_argument("--save-model", action="store_true", help="Save rank 0 model after timed steps")
    parser.add_argument(
        "--diagnose-sync", action="store_true", help="Compare parameter/gradient hashes after every stage"
    )
    parser.add_argument("--diagnose-final", action="store_true", help="Compare final hashes before metrics")
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.batch_size) <= 0:
        parser.error("warmup, steps, and batch-size must be positive")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # Ada's original layout is preserved. CUDA hosts can provide their own
    # physical core IDs since SMT sibling numbering differs between machines.
    cores = list(map(int, args.cpu_cores.split(","))) if args.cpu_cores else list(range(world * 4))
    if len(cores) < world * 4 or len(set(cores)) != len(cores):
        parser.error("cpu-cores must contain at least four distinct cores per rank")
    affinity = set(cores[local_rank * 4 : (local_rank + 1) * 4])
    if not affinity.issubset(os.sched_getaffinity(0)):
        raise RuntimeError(f"CPU cores {sorted(affinity)} are unavailable")
    os.sched_setaffinity(0, affinity)
    os.environ["HF_HOME"] = str(args.assets / "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.device == "ada":
        os.environ.setdefault("CXX", "/usr/local/llvm-adax/bin/clang++")

    import torch
    import torch.distributed as dist

    if args.device == "ada":
        import torch_ada  # noqa: F401

    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers.initialization import no_init_weights

    from lerobot.configs import NormalizationMode
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    torch.set_num_threads(4)
    accelerator = getattr(torch, args.device)
    accelerator.set_device(local_rank)
    device = torch.device(f"{args.device}:{local_rank}")
    if world > 1:
        dist.init_process_group("accl" if args.device == "ada" else "nccl", timeout=timedelta(seconds=900))
    control_group = dist.new_group(backend="gloo") if world > 1 else None
    torch.manual_seed(1000)
    config = PI05Config(
        device=str(device),
        dtype="bfloat16",
        attention_implementation=args.attention,
        input_features={},
        pretrained_path=str(args.assets / "pi05-base"),
        gradient_checkpointing=True,
        compile_model=False,
        push_to_hub=False,
        normalization_mapping={
            "ACTION": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        },
    )
    train_config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/pusht", root=str(args.assets / "pusht"), video_backend="pyav"),
        policy=config,
        batch_size=args.batch_size,
        num_workers=0,
    )
    dataset = make_dataset(train_config)
    with no_init_weights():
        policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    policy.train()
    attention = policy.model.paligemma_with_expert.attention_implementation
    initial = policy.model.action_out_proj.weight.detach().cpu().float().clone()
    if world > 1:
        # Every rank loads the same full checkpoint. Check that before skipping
        # DDP's initial coalesced broadcast, which corrupts this mixed-dtype
        # model with the installed AdaCCL release.
        initial_digest = hashlib.sha256()
        for name, parameter in policy.named_parameters():
            cpu = parameter.detach().cpu().contiguous()
            initial_digest.update(name.encode())
            initial_digest.update(cpu.view(torch.uint8).numpy().tobytes())
        initial_hashes = [None for _ in range(world)]
        dist.all_gather_object(initial_hashes, initial_digest.hexdigest(), group=control_group)
        if len(set(initial_hashes)) != 1:
            raise RuntimeError("Loaded checkpoint differs across ranks")
        print(f"RANK {rank} INITIAL_PARAMETERS_MATCH", flush=True)
    if world > 1:
        model = DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=200,
            init_sync=False,
        )
    else:
        model = policy
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    from lerobot.utils.device_utils import is_ada2s_device

    optimizer_fused = all(group["fused"] is True for group in optimizer.param_groups)
    if is_ada2s_device(device) and not optimizer_fused:
        raise RuntimeError("Ada2S parameter groups did not enable fused AdamW")

    def diagnose(stage, gradients=False):
        if world == 1 or not (args.diagnose_sync or (args.diagnose_final and stage == "final")):
            return
        accelerator.synchronize()
        names, hashes = [], []
        for name, parameter in policy.named_parameters():
            value = parameter.grad if gradients else parameter
            if value is None:
                continue
            cpu = value.detach().cpu().contiguous()
            names.append(name)
            hashes.append(list(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).digest()))
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, hashes, group=control_group)
        matching = [left == right for left, right in zip(gathered[0], gathered[rank], strict=True)]
        failures = [name for name, matches in zip(names, matching, strict=True) if not matches]
        print(
            f"RANK {rank} SYNC_DIAG {stage} mismatch_count={len(failures)} names={failures[:12]}", flush=True
        )
        (args.output / f"{stage}-rank{rank}.json").write_text(
            json.dumps({"stage": stage, "mismatches": failures}) + "\n"
        )
        if stage == "initial" and failures:
            raise RuntimeError("Checkpoint weights differ before training")

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=1000)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    iterator = iter(loader)
    args.output.mkdir(parents=True, exist_ok=True)
    diagnose("initial")
    print(f"RANK {rank} READY device={device} attention={attention} dp={world}", flush=True)
    if world > 1:
        dist.barrier()
    times = []
    losses = []
    data_times = []
    start_window = None
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            accelerator.synchronize()
            if world > 1:
                dist.barrier()
            accelerator.reset_peak_memory_stats()
            start_window = time.perf_counter()
        start = time.perf_counter()
        batch = preprocessor(next(iterator))
        data_end = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(args.device, dtype=torch.bfloat16):
            loss, _ = model(batch)
        loss.backward()
        if args.device == "ada":
            accelerator.synchronize()
        if args.diagnose_sync:
            diagnose(f"gradient_step{step + 1}", gradients=True)
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
        if args.device == "ada":
            accelerator.synchronize()
        if step == 0:
            print(f"RANK {rank} GRAD_NORM {norm.item()}", flush=True)
        if args.diagnose_sync:
            diagnose(f"clipped_gradient_step{step + 1}", gradients=True)
        optimizer.step()
        if args.diagnose_sync:
            diagnose(f"updated_step{step + 1}")
        accelerator.synchronize()
        value = loss.item()
        if not math.isfinite(value) or not math.isfinite(norm.item()):
            raise RuntimeError(f"rank {rank} step {step}: nonfinite loss/gradient")
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            times.append(elapsed)
            data_times.append(data_end - start)
            losses.append(value)
        print(f"RANK {rank} STEP {step + 1} loss={value:.6f} seconds={elapsed:.6f}", flush=True)
    accelerator.synchronize()
    window = time.perf_counter() - start_window
    diagnose("final")
    metrics = torch.tensor([window, max(times), statistics.median(times)], dtype=torch.float32, device=device)
    if world > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    # Check every parameter, including BF16 values whose view(dtype) dispatch is
    # not involved: hashing happens only after transferring the tensor to CPU.
    digest = hashlib.sha256()
    parameter_count = 0
    for name, parameter in policy.named_parameters():
        cpu = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
        parameter_count += parameter.numel()
    if world > 1:
        all_hashes = [None for _ in range(world)]
        dist.all_gather_object(all_hashes, digest.hexdigest(), group=control_group)
        if len(set(all_hashes)) != 1:
            raise RuntimeError("DP ranks have different final parameter hashes")
    change = (policy.model.action_out_proj.weight.detach().cpu().float() - initial).abs().max().item()
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("Action projection did not update")
    optimizer_steps = [state["step"].item() for state in optimizer.state.values()]
    if not optimizer_steps or any(value != args.warmup + args.steps for value in optimizer_steps):
        raise RuntimeError("Optimizer steps do not match completed training")
    summary = {
        "rank": rank,
        "cpu_affinity": sorted(affinity),
        "world_size": world,
        "device": str(device),
        "attention": attention,
        "batch_per_rank": args.batch_size,
        "global_batch": world * args.batch_size,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "max_rank_window_seconds": metrics[0].item(),
        "max_rank_median_step_seconds": metrics[2].item(),
        "samples_per_second": args.steps * world * args.batch_size / metrics[0].item(),
        "local_step_seconds": times,
        "local_data_seconds": data_times,
        "local_losses": losses,
        "peak_allocated_gib": accelerator.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": accelerator.max_memory_reserved() / 1024**3,
        "torch_version": torch.__version__,
        "optimizer_fused": optimizer_fused,
        "communication_backend": "accl" if args.device == "ada" else "nccl",
        "device_name": accelerator.get_device_name(local_rank) if args.device == "cuda" else "Ada2S",
        "all_parameters_synchronized": True,
        "parameter_sha256": digest.hexdigest(),
        "parameter_count": parameter_count,
        "action_projection_max_abs_change": change,
        "optimizer_state_count": len(optimizer_steps),
        "optimizer_step": optimizer_steps[0],
    }
    (args.output / f"rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BENCHMARK_PASS " + json.dumps(summary), flush=True)
    if args.save_model and rank == 0:
        policy.save_pretrained(args.output / "pretrained_model")
        preprocessor.save_pretrained(args.output / "pretrained_model")
        postprocessor.save_pretrained(args.output / "pretrained_model")
        print("MODEL_SAVED", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
