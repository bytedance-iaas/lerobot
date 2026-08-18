#!/usr/bin/env bash
# Language ablation on LIBERO-Spatial. All ten tasks share one scene layout and
# differ only in which of two visually identical black bowls the instruction
# names, so the instruction is the only thing that can select the right bowl.
#
# Each task is run with another task's instruction (i -> (i+5) % 10). If success
# rates hold up under a wrong instruction, the policy is not reading language at
# all; if they collapse, it reads language and merely resolves the spatial
# reference badly. Those two conclusions call for very different fixes.
#
# Usage:
#   tests/scripts/evals/ablate_libero_language.sh [n_episodes] [policy_path] [gpus...]
#
# Compare against the unmodified run of the same suite and episode count from
# eval_libero_sharded.sh.
set -u

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

N_EPISODES=${1:-20}
POLICY=${2:-lerobot/lingbot_va_libero_long}
shift 2 2>/dev/null || shift $#
GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1)

OUT=outputs/eval/libero_spatial_language_ablation
mkdir -p "$OUT/logs"
unset CUDA_VISIBLE_DEVICES

# LIBERO-Spatial instructions, indexed by task id.
INSTRUCTIONS=(
  "pick up the black bowl between the plate and the ramekin and place it on the plate"
  "pick up the black bowl next to the ramekin and place it on the plate"
  "pick up the black bowl from table center and place it on the plate"
  "pick up the black bowl on the cookie box and place it on the plate"
  "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
  "pick up the black bowl on the ramekin and place it on the plate"
  "pick up the black bowl next to the cookie box and place it on the plate"
  "pick up the black bowl on the stove and place it on the plate"
  "pick up the black bowl next to the plate and place it on the plate"
  "pick up the black bowl on the wooden cabinet and place it on the plate"
)

run_shard() {
  local gpu=$1; shift
  for tid in "$@"; do
    local wrong=$(( (tid + 5) % 10 ))
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync lerobot-eval \
      --policy.path="$POLICY" --policy.device="cuda:$gpu" \
      --policy.prompt_override="${INSTRUCTIONS[$wrong]}" \
      --env.type=libero --env.task=libero_spatial --env.task_ids="[$tid]" \
      --env.observation_height=128 --env.observation_width=128 \
      --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
      --output_dir="$OUT/task_${tid}_prompt_${wrong}" \
      > "$OUT/logs/task_$tid.log" 2>&1
    echo "task $tid (prompt of task $wrong, cuda:$gpu) exit=$?" >> "$OUT/logs/status.txt"
  done
}

n=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  shard=()
  for tid in $(seq 0 9); do
    [ $((tid % n)) -eq "$i" ] && shard+=("$tid")
  done
  run_shard "${GPUS[$i]}" "${shard[@]}" &
done
wait

echo "DONE $(date)" >> "$OUT/logs/status.txt"
echo "results: $OUT"
