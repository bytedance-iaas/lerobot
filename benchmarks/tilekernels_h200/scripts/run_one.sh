#!/usr/bin/env bash
set -euo pipefail

policy=${1:?Usage: run_one.sh POLICY MODE REPEAT}
mode=${2:?Usage: run_one.sh POLICY MODE REPEAT}
repeat=${3:?Usage: run_one.sh POLICY MODE REPEAT}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=${LEROBOT_ROOT:-$(cd "$script_dir/../../.." && pwd)}
results_root=${RESULTS_ROOT:?Set RESULTS_ROOT to a writable container path}
run_root="$results_root/results/${policy}-${mode}-run${repeat}"

export RUN_ROOT="$run_root"
export PYTHONPATH="$script_dir:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$mode" == baseline ]]; then
    export TK_OPS=""
elif [[ "$mode" == full ]]; then
    if [[ "$policy" == groot ]]; then
        export TK_OPS="rms,swiglu,rope"
    else
        export TK_OPS="rms,rope"
    fi
else
    export TK_OPS=${mode//+/,}
fi

export TK_VERIFY=${TK_VERIFY:-0}
export TK_VALIDATE_ONLY=${TK_VALIDATE_ONLY:-0}
export CUDA_VISIBLE_DEVICES=${BENCH_GPUS:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export SDL_VIDEODRIVER=dummy

mkdir -p "$run_root"
exec >"$run_root/train.log" 2>&1
trap 'rc=$?; printf "%s\n" "$rc" >"$run_root/exit-code"' EXIT
cd "$repo_root"

args=(
    --policy.type="$policy"
    --policy.device=cuda
    --policy.push_to_hub=false
    --batch_size=16
    --steps=60
    --num_workers=0
    --log_freq=10
    --save_checkpoint=false
    --wandb.enable=false
    --seed=1000
    --dataset.revision=main
    --dataset.video_backend=pyav
    --output_dir="$run_root/train"
)

if [[ "$policy" == groot ]]; then
    : "${GROOT_MODEL_PATH:?Set GROOT_MODEL_PATH}"
    : "${GROOT_DATASET_ROOT:?Set GROOT_DATASET_ROOT}"
    args+=(
        --policy.model_params_fp32=false
        --policy.base_model_path="$GROOT_MODEL_PATH"
        --policy.embodiment_tag=libero_sim
        --policy.use_relative_actions=false
        --dataset.repo_id=IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
        --dataset.root="$GROOT_DATASET_ROOT"
    )
else
    : "${PI05_DATASET_ROOT:?Set PI05_DATASET_ROOT}"
    args+=(
        --policy.dtype=bfloat16
        --policy.gradient_checkpointing=false
        --dataset.repo_id=lerobot/pusht
        --dataset.root="$PI05_DATASET_ROOT"
    )
fi

/opt/venv/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 \
    "$script_dir/train_bench.py" "${args[@]}"
