#!/usr/bin/env bash
# LIBERO-plus baseline with residual tracking on. One run produces both things
# the residual-replanning A/B needs: the control-arm success rate, and the
# residual distribution on perturbed scenes (which is where the residual should
# actually carry signal -- viewpoint/lighting drift makes the world model wrong
# in a way that re-planning can correct, unlike the wrong-target failures on
# LIBERO-Spatial).
#
# Usage:
#   tests/scripts/evals/eval_libero_plus_residual.sh [n_episodes] [policy_path] [gpus...]
#
# LIBERO-plus lives outside the venv; see the env vars below. Task ids are
# sampled across the 2519 perturbed variants of the LIBERO-10 suite so the
# sample spans the perturbation dimensions rather than one of them.
set -u

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

N_EPISODES=${1:-10}
POLICY=${2:-lerobot/lingbot_va_libero_long}
shift 2 2>/dev/null || shift $#
GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(2 3)

LIBERO_PLUS_ROOT=${LIBERO_PLUS_ROOT:-$HOME/src/libero-plus}
export PYTHONPATH="$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/src/libero-plus-config}
export MAGICK_HOME=${MAGICK_HOME:-$HOME/opt/imagemagick}
export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Set RESIDUAL_REPLAN_THRESHOLD to also turn on early re-planning (the treatment
# arm); leave it unset for the control arm. Pick the value from the per-chunk
# residual distribution of a control run, not by guessing.
THRESHOLD=${RESIDUAL_REPLAN_THRESHOLD:-}
REPLAN_ARG=()
SUFFIX=""
if [ -n "$THRESHOLD" ]; then
  REPLAN_ARG=(--policy.residual_replan_threshold="$THRESHOLD")
  SUFFIX="_replan${THRESHOLD}"
fi

OUT="outputs/eval/libero_plus_residual${SUFFIX}"
mkdir -p "$OUT/logs"
unset CUDA_VISIBLE_DEVICES

TASK_IDS=(0 250 500 750 1000 1250 1500 1750 2000 2250)

run_shard() {
  local gpu=$1; shift
  for tid in "$@"; do
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync lerobot-eval \
      --policy.path="$POLICY" --policy.device="cuda:$gpu" \
      --policy.track_prediction_residual=true \
      "${REPLAN_ARG[@]}" \
      --env.type=libero_plus --env.task=libero_10 --env.task_ids="[$tid]" \
      --env.observation_height=128 --env.observation_width=128 \
      --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
      --output_dir="$OUT/task_$tid" \
      > "$OUT/logs/task_$tid.log" 2>&1
    echo "task $tid (cuda:$gpu) exit=$?" >> "$OUT/logs/status.txt"
  done
}

n=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  shard=()
  for j in "${!TASK_IDS[@]}"; do
    [ $((j % n)) -eq "$i" ] && shard+=("${TASK_IDS[$j]}")
  done
  run_shard "${GPUS[$i]}" "${shard[@]}" &
done
wait

echo "DONE $(date)" >> "$OUT/logs/status.txt"
echo "results: $OUT"
