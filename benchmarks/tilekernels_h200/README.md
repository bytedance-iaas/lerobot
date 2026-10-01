# TileKernels training benchmark on H200

This directory contains the isolated experiment used to measure TileKernels in
real two-GPU GR00T and PI05 training. It is intentionally not wired into the
production policies: the tested TileKernels revision requires PyTorch 2.13,
while this repository currently supports PyTorch versions below 2.12.

## Result

The test used two H200 GPUs, batch 16 per GPU (global batch 32), 60 steps per
run, and discarded the first 20 steps. Baseline and TileKernels runs used the
same container and were interleaved as baseline 1, full 1, full 2, baseline 2.
Each number below is the mean of two runs.

| Policy | Enabled TileKernels operations | Baseline iteration | Tile iteration | Throughput change | Baseline / tile peak allocated |
|---|---|---:|---:|---:|---:|
| GR00T | RMSNorm, SwiGLU, RoPE | 400.444 ms | 397.434 ms | +0.760% mean, -0.282% median-based | 18.469 / 18.468 GiB |
| PI05 | RMSNorm, RoPE | 435.882 ms | 418.362 ms | +4.185% mean, +4.184% median-based | 66.211 / 65.132 GiB |

PI05 also reduced mean update time from 347.913 ms to 329.612 ms and sampled
device peak memory from 69.384 GiB to 68.489 GiB. Its paired same-weight,
same-batch check had a maximum loss relative error of 0.0000684 and sampled
gradient relative L2 error of 0.001216.

GR00T did not show a stable end-to-end win: its median result regressed slightly,
and the paired check had up to 0.005010 loss relative error and 0.033806 sampled
gradient relative L2 error. The result does not justify production integration
for GR00T.

The representative standalone forward measurements were:

| Operation and shape | PyTorch | TileKernels |
|---|---:|---:|
| Qwen SwiGLU, 2496 x 6144 | 88.576 us | 28.600 us |
| Qwen RoPE, batch 16, sequence 156, head 128 | 115.036 us | 58.396 us |
| PI05 RoPE, batch 16, sequence 506, head 256 | 244.702 us | 72.947 us |

These standalone numbers are kernel timings, not training throughput. The
runner generates per-run means, medians, P90 values, memory peaks, both DDP-rank
paired checks, and standalone operator results under RESULTS_ROOT. Generated
JSON and logs are deliberately not checked into the repository.

At TileKernels revision 66258df6175d2f630ffecb04c5ab66bff8a2ae6a,
Attention, GEMM/Linear, GELU/GeGLU, and AdamW implementations applicable to these
models were not present. PI05 uses GeGLU, so Qwen's SwiGLU kernel must not be
substituted into PI05. The included patch only admits PI05's 256-wide rotary
dimension in the dimension-generic CUDA backend; it keeps the Ascend backend's
64/128 restriction.

## Environment

- NVIDIA H200, two GPUs used by each run
- Ubuntu 24.04 CUDA 13.1.2 development image
- Python 3.12
- PyTorch 2.13.0+cu130
- TileLang 0.1.15
- TileKernels 66258df6175d2f630ffecb04c5ab66bff8a2ae6a
- LeRobot f717dd1a5 (the parent of the benchmark commit)

The Docker image is an experiment environment, not a supported LeRobot runtime.
It first installs the repository's normal training, PI, and GR00T dependencies,
then deliberately replaces PyTorch with 2.13.0 for TileKernels.

## Build

Run from the repository root on H200-1. GitHub is directly reachable there.

    docker build --network host \
      -f benchmarks/tilekernels_h200/Dockerfile \
      -t lerobot-tilekernels:h200 .

Create a persistent container and expose the local model, dataset, and result
storage. This example follows the H200-1 mount layout used for the recorded run.

    docker run -d --name lerobot-tilekernels-h200 --init --network host \
      --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --mount type=bind,source=/data02,target=/data \
      lerobot-tilekernels:h200

## Run

The matrix runner refuses to start if GPU 0 or 1 already has at least 512 MiB
allocated or 5 percent utilization. Run the long job in tmux so the experiment
survives an SSH disconnect.

    tmux new -s tilekernels-h200
    docker exec \
      -e RESULTS_ROOT=/data/tilekernels-h200-results \
      -e GROOT_MODEL_PATH=/data/models/GR00T-N1.7-LIBERO/libero_spatial \
      -e GROOT_DATASET_ROOT=/data/groot-memory-h200-assets/libero_spatial \
      -e PI05_DATASET_ROOT=/data/tilekernels-policy-assets/pusht \
      -e HF_HUB_CACHE=/data/tilekernels-policy-assets/hf-hub \
      lerobot-tilekernels-h200 \
      bash /workspace/lerobot/benchmarks/tilekernels_h200/scripts/run_matrix.sh

The runner performs eight timed jobs, then two paired numerical validation jobs,
then representative standalone operator checks. Raw per-step JSONL, GPU samples,
logs, exit codes, metadata, and the regenerated summary are written below
RESULTS_ROOT. Existing successful jobs are reused; an incomplete directory must
be inspected or moved before retrying.

To run only one configuration:

    docker exec \
      -e RESULTS_ROOT=/data/tilekernels-h200-results \
      -e GROOT_MODEL_PATH=/data/models/GR00T-N1.7-LIBERO/libero_spatial \
      -e GROOT_DATASET_ROOT=/data/groot-memory-h200-assets/libero_spatial \
      lerobot-tilekernels-h200 \
      bash /workspace/lerobot/benchmarks/tilekernels_h200/scripts/run_one.sh \
      groot full 1

Valid modes are baseline, full, or an explicit plus-separated set such as
rms+rope. Full means RMSNorm/SwiGLU/RoPE for GR00T and RMSNorm/RoPE for PI05.
