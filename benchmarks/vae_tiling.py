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

"""What DreamZero's tiled VAE encode costs, and what turning it off would change.

`WANPolicyHeadConfig.tiled` defaults to True, and at the default tile geometry a 352x640
training canvas is cut into **741 tiles of 34x34**, each of which is a full pass through the
VAE encoder. `benchmarks/mfu.py` prices that at 84.6 TFLOP per sample against 37.0 untiled --
but FLOPs understate it, because 741 passes over a 34x34 patch are launch-bound rather than
arithmetic-bound. This measures the wall clock instead.

It also measures the thing that decides whether `tiled=False` is usable at all. The tiles
overlap and are feathered together with `build_mask`, so the two paths do NOT agree
numerically: turning tiling off is a change to the latents the DiT trains on, which is a recipe
change needing a checkpoint-level check, not a free switch. The error this reports is how big
that change is.

One caveat on the error it reports: it is geometry-dependent, and a canvas small enough that
every tile touches a border comes out exact, because `is_bound` then makes every mask all-ones.
Only the real canvas means anything -- which is what the defaults are.

Random weights are the default and are the right choice for timing -- the shapes are what cost
time, and a 254 MB checkpoint is not needed to measure them. Pass `--vae-path` to use real
weights, which is what the error column needs to mean anything.

    python benchmarks/vae_tiling.py --device cuda
    python benchmarks/vae_tiling.py --device cuda --vae-path /data/models/DreamZero-DROID/vae.pth
    python benchmarks/vae_tiling.py --device cuda --tile-size 64 --tile-stride 48
"""

import argparse
import time

import torch


def tile_count(height: int, width: int, size: tuple[int, int], stride: tuple[int, int]) -> int:
    """The same task enumeration `WanVideoVAE.tiled_encode` does, without running it."""
    size_h, size_w = size
    stride_h, stride_w = stride
    tasks = 0
    for h in range(0, height, stride_h):
        if h - stride_h >= 0 and h - stride_h + size_h >= height:
            continue
        for w in range(0, width, stride_w):
            if w - stride_w >= 0 and w - stride_w + size_w >= width:
                continue
            tasks += 1
    return tasks


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "npu":
        return torch.npu.max_memory_allocated()
    return None


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "npu":
        torch.npu.reset_peak_memory_stats()


def time_encode(
    vae, video, *, tiled, size, stride, repeats, device
) -> tuple[float, torch.Tensor, int | None]:
    kwargs = {"tiled": tiled, "tile_size": size, "tile_stride": stride}
    with torch.no_grad():
        latents = vae.encode(video, **kwargs)  # warmup: allocator, cuDNN plans, lazy init
        synchronize(device)
        reset_peak(device)
        start = time.perf_counter()
        for _ in range(repeats):
            latents = vae.encode(video, **kwargs)
        synchronize(device)
        elapsed = (time.perf_counter() - start) / repeats
    return elapsed, latents, peak_bytes(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--vae-path", default=None, help="real weights; needed for the error column to mean anything"
    )
    # The DROID training window: 33 raw frames of a 2x2 canvas of 176x320 views.
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--tile-size", type=int, nargs=2, default=(34, 34))
    parser.add_argument("--tile-stride", type=int, nargs=2, default=(18, 16))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--step-s",
        type=float,
        default=None,
        help="a policy step on THIS card, to express the delta as a share of it; mixing cards here is "
        "the easy mistake (8xH200 predicts 5.21 s at gradient_checkpointing=false, a 950PR is ~11.9 s)",
    )
    args = parser.parse_args()

    from lerobot.policies.dreamzero.wan.vae import WanVideoVAE

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    size = tuple(args.tile_size)
    stride = tuple(args.tile_stride)
    tiles = tile_count(args.height, args.width, size, stride)

    vae = WanVideoVAE(z_dim=16, vae_pretrained_path=args.vae_path).to(device=device, dtype=dtype).eval()
    torch.manual_seed(0)
    video = torch.randn(1, 3, args.frames, args.height, args.width, device=device, dtype=dtype)

    print(f"canvas {args.height}x{args.width}, {args.frames} frames, {args.dtype} on {device}")
    print(
        f"tile {size[0]}x{size[1]} stride {stride[0]}x{stride[1]} -> {tiles} tiles, "
        f"{tiles * size[0] * size[1] / (args.height * args.width):.2f}x the pixel area"
    )
    print(f"weights: {args.vae_path or 'random (timing only)'}")
    print(f"mean of {args.repeats} after one warmup\n")

    tiled_s, tiled_z, tiled_peak = time_encode(
        vae, video, tiled=True, size=size, stride=stride, repeats=args.repeats, device=device
    )
    plain_s, plain_z, plain_peak = time_encode(
        vae, video, tiled=False, size=size, stride=stride, repeats=args.repeats, device=device
    )

    print(f"{'':10} {'latency':>10} {'peak':>10} {'latent':>22}")
    for name, seconds, peak, latent in (
        ("tiled", tiled_s, tiled_peak, tiled_z),
        ("untiled", plain_s, plain_peak, plain_z),
    ):
        mem = f"{peak / 2**30:.2f} GiB" if peak is not None else "n/a"
        print(f"{name:10} {seconds * 1000:9.1f}ms {mem:>10} {str(tuple(latent.shape)):>22}")

    print(
        f"\n  tiling costs {tiled_s / plain_s:.2f}x the time ({(tiled_s - plain_s) * 1000:.1f} ms per sample)"
    )
    if args.step_s:
        print(
            f"  which at a {args.step_s:.2f} s step is {(tiled_s - plain_s) / args.step_s * 100:.1f}% of it"
        )
    if tiled_peak and plain_peak:
        print(f"  and {(plain_peak - tiled_peak) / 2**30:+.2f} GiB of peak to turn off")

    # The reason this is a recipe change and not a bug fix.
    if tiled_z.shape == plain_z.shape:
        a, b = tiled_z.float(), plain_z.float()
        scale = b.abs().max().clamp_min(1e-30)
        print(
            f"\n  latents differ by {(a - b).abs().max().item() / scale.item():.3%} of full scale "
            f"(max abs {(a - b).abs().max().item():.3e})"
        )
        print("  -> `tiled=False` changes what the DiT trains on. Score it against the released")
        print("     checkpoint before adopting it; the feathered tile overlap is not cosmetic.")
    else:
        print(f"\n  shapes differ: {tuple(tiled_z.shape)} vs {tuple(plain_z.shape)}")


if __name__ == "__main__":
    main()
