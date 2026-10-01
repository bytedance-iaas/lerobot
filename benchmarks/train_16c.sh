#!/usr/bin/env bash
# Pi0.5 multi-card training throughput on Ascend 950PR.
#
# All seven optimisations are individually switchable and DEFAULT TO OFF, so a bare run
# of this script measures the un-optimised path. Set ALL_OPT=true for the fully optimised
# configuration, or flip individual flags.
#
# Note on smp/s: lerobot computes it as (batch_size * num_processes) / step_time, so the
# value in the log is ALREADY global throughput -- do not multiply by NP. And do not
# aggregate those per-step values by median or mean; use bench/throughput.py, which
# computes total samples / total time over a window in epoch 2 or later.
set -euo pipefail

NP=${NP:-16}
BS=${BS:-16}
GC=${GC:-false}
STEPS=${STEPS:-455}
NW=${NW:-2}
TAG=${TAG:-np${NP}_bs${BS}_gc${GC}}
OUTD=${OUTD:-/workspace/bench/mc16}
PORT=${PORT:-29531}

# Optimisations, all off by default. ALL_OPT=true turns every one on at once; an
# individual variable set explicitly still wins over ALL_OPT.
ALL_OPT=${ALL_OPT:-false}
FUSED_CLIP=${FUSED_CLIP:-$ALL_OPT}     # fused grad-norm clip (single process only; silently
                                       # falls back when num_processes > 1)
FUSED_GEGLU=${FUSED_GEGLU:-$ALL_OPT}   # fused GELU(tanh)+gating in the Gemma MLP
ROPE_REUSE=${ROPE_REUSE:-$ALL_OPT}     # reuse cos/sin across layers of one joint forward
FUSED_ROPE=${FUSED_ROPE:-$ALL_OPT}     # fused NPU rotary embedding
FUSED_ATTN=${FUSED_ATTN:-$ALL_OPT}     # torch_npu npu_fusion_attention
FUSED_RMS=${FUSED_RMS:-$ALL_OPT}       # torch_npu npu_rms_norm

STATIC_GRAPH=${STATIC_GRAPH:-true}
BUCKET_MB=${BUCKET_MB:-200}
GRAD_AS_VIEW=${GRAD_AS_VIEW:-true}
TOK_MAX=${TOK_MAX:-64}

# Profiling. PROF=1 makes rank 0 (PROF_RANKS to change) profile PROF_ACTIVE steps after
# skipping PROF_SKIP, writing ASCEND_PROFILER_OUTPUT/step_trace_time.csv under PROF_DIR.
#
# With PROF=1 the run also prints an MFU + compute/communication-overlap summary at the
# end (src/lerobot/utils/profiler_summary.py). Model FLOPs come from the profiled steps'
# kernel shapes; the step time comes from UNPROFILED steps.
PROF=${PROF:-0}
if [ "$PROF" != "0" ]; then
  export LEROBOT_PROF=1
  export LEROBOT_PROF_DIR=${PROF_DIR:-$OUTD/prof_$TAG}
  export LEROBOT_PROF_SKIP=${PROF_SKIP:-60}
  export LEROBOT_PROF_ACTIVE=${PROF_ACTIVE:-4}
  export LEROBOT_PROF_RANKS=${PROF_RANKS:-0}
  export LEROBOT_PROF_SHAPES=${PROF_SHAPES:-0}
  # PROF_AIC=1 adds the AI Core PMU counters -> aic_cube_fops in kernel_details.csv, i.e.
  # hardware-measured matmul FLOPs including the fused kernels. Read with bench/prof_flops.py.
  export LEROBOT_PROF_AIC=${PROF_AIC:-0}
  if [ -n "${PROF_PEAK_TFLOPS:-}" ]; then
    export LEROBOT_PROF_PEAK_TFLOPS=$PROF_PEAK_TFLOPS
  fi
  mkdir -p "$LEROBOT_PROF_DIR"
  echo "# profiling rank(s) $LEROBOT_PROF_RANKS, $LEROBOT_PROF_ACTIVE steps after" \
       "$LEROBOT_PROF_SKIP, aic_pmu=$LEROBOT_PROF_AIC, into $LEROBOT_PROF_DIR" >&2
  # The profiled window must exist: skip + warmup + active steps.
  MIN_STEPS=$((LEROBOT_PROF_SKIP + LEROBOT_PROF_ACTIVE + 2))
  if [ "$STEPS" -lt "$MIN_STEPS" ]; then
    echo "ERROR: STEPS=$STEPS is below the profiled window ($MIN_STEPS)." >&2
    exit 1
  fi
fi

mkdir -p "$OUTD"

IDS=$(seq -s, 0 $((NP-1)))
export ASCEND_RT_VISIBLE_DEVICES=$IDS
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/hf-hub
export HF_HOME=/workspace/.hf
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}

OMP=${OMP:-8}
export OMP_NUM_THREADS=$OMP
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

echo "# optimisations: adamw=$FUSED_ADAMW clip=$FUSED_CLIP geglu=$FUSED_GEGLU" \
     "rope_reuse=$ROPE_REUSE fused_rope=$FUSED_ROPE attn=$FUSED_ATTN rms=$FUSED_RMS" >&2
echo "# NP=$NP BS=$BS GC=$GC NW=$NW OMP=$OMP STEPS=$STEPS" >&2
echo "# ddp: static_graph=$STATIC_GRAPH bucket_mb=$BUCKET_MB grad_as_view=$GRAD_AS_VIEW" "tokenizer_max_length=$TOK_MAX pyav_threads=${LEROBOT_PYAV_THREADS:-1}" >&2

cd /workspace

LAUNCH_MODE=(--multi_gpu)
if [ "$NP" -eq 1 ]; then LAUNCH_MODE=(); fi
exec accelerate launch \
  "${LAUNCH_MODE[@]}" --num_machines=1 --num_processes="$NP" \
  --mixed_precision=no --main_process_port="$PORT" \
  -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --policy.pretrained_path=/models/pi05_base \
  --policy.device=npu \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing="$GC" \
  --policy.compile_model=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.push_to_hub=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --npu_fused_grad_clip="$FUSED_CLIP" \
  --policy.npu_fused_geglu="$FUSED_GEGLU" \
  --policy.reuse_rope_embeddings="$ROPE_REUSE" \
  --policy.npu_fused_rope="$FUSED_ROPE" \
  --policy.npu_fused_attention="$FUSED_ATTN" \
  --policy.npu_fused_rms_norm="$FUSED_RMS" \
  --ddp_static_graph="$STATIC_GRAPH" \
  --ddp_bucket_cap_mb="$BUCKET_MB" \
  --ddp_gradient_as_bucket_view="$GRAD_AS_VIEW" \
  --policy.tokenizer_max_length="$TOK_MAX" \
  --dataset.repo_id=local/libero_spatial \
  --dataset.root=/datasets/libero_spatial_no_noops_1.0.0_lerobot \
  --dataset.video_backend=pyav \
  --batch_size="$BS" --num_workers="$NW" --steps="$STEPS" \
  --log_freq=1 --save_checkpoint=false \
  --wandb.enable=false --seed=1000 \
  --output_dir="/workspace/outputs/mc_${TAG}_$(date +%s)" \
  --job_name="pi05_mc_${TAG}" "$@"
