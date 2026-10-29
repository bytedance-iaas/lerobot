#!/usr/bin/env bash
# Run inside the ada2s_compute container with the ada2s branch installed.
set -euo pipefail

run_root="${ADA2S_RUN_ROOT:-/workspace/ada2s-training-20261030}"
steps="${STEPS:-20}"
if [[ ! "${steps}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'STEPS must be a positive integer\n' >&2
    exit 2
fi
save_freq="${SAVE_FREQ:-$(( (steps + 1) / 2 ))}"
output_dir="${OUTPUT_DIR:-${run_root}/pi05-single}"
status_file="${STATUS_FILE:-${run_root}/train.exit}"
if [[ -n "${LOG_FILE:-}" ]]; then
    exec >"${LOG_FILE}" 2>&1
fi
trap 'printf "%s\n" "$?" >"${status_file}"' EXIT

export HF_HOME="${run_root}/hf"
export HF_HUB_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export ADAX_RT_USE_TSIM_MODE=0
export ADAX_VISIBLE_DEVICES="${ADAX_VISIBLE_DEVICES:-0}"
export TRITON_ADA_ARCH=s
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

test -s "${run_root}/pi05-base/model.safetensors"
test -f "${run_root}/pusht/meta/info.json"

python3 -u -m lerobot.scripts.lerobot_train \
    --policy.type=pi05 \
    --policy.device=ada:0 \
    --policy.pretrained_path="${run_root}/pi05-base" \
    --policy.input_features='{}' \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=2 \
    --policy.scheduler_decay_steps="${steps}" \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --dataset.repo_id=lerobot/pusht \
    --dataset.root="${run_root}/pusht" \
    --dataset.video_backend=pyav \
    --output_dir="${output_dir}" \
    --job_name=pi05_ada2s_single \
    --steps="${steps}" \
    --batch_size=1 \
    --num_workers=0 \
    --log_freq=1 \
    --env_eval_freq=0 \
    --save_freq="${save_freq}" \
    --wandb.enable=false \
    "$@"
