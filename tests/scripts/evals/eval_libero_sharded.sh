#!/usr/bin/env bash
# Evaluate a policy on a 10-task LIBERO suite, sharding the tasks across the
# local GPUs. LingBot-VA's streaming inference is single-env, so the only way to
# use more than one GPU is to run one task per process.
#
# Usage:
#   tests/scripts/evals/eval_libero_sharded.sh <suite> [n_episodes] [policy_path] [n_gpus]
#
# Example (the LIBERO-Spatial baseline, 50 episodes x 10 tasks):
#   tests/scripts/evals/eval_libero_sharded.sh libero_spatial 50
#
# Results land in outputs/eval/<policy_name>_<suite>/task_<i>/eval_info.json,
# with per-shard logs and a status file under logs/. Check logs/status.txt right
# after launching: a failed shard exits within seconds and is easy to mistake
# for one that is still queued.
#
# NOTE on GPU selection: do not switch this to CUDA_VISIBLE_DEVICES. robosuite
# passes that value to EGL as a device index, and a host that exposes a single
# EGL device node then rejects every index but 0 ("MUJOCO_EGL_DEVICE_ID must be
# an integer between 0 and 0"). Rendering goes through the one EGL node and the
# policy is placed with --policy.device instead.
set -u

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

SUITE=${1:?usage: eval_libero_sharded.sh <suite> [n_episodes] [policy_path] [n_gpus]}
N_EPISODES=${2:-50}
POLICY=${3:-lerobot/lingbot_va_libero_long}
N_GPUS=${4:-$(nvidia-smi --list-gpus | wc -l)}

OUT="outputs/eval/$(basename "$POLICY")_${SUITE}"
mkdir -p "$OUT/logs"
unset CUDA_VISIBLE_DEVICES

run_shard() {
  local gpu=$1; shift
  for tid in "$@"; do
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync lerobot-eval \
      --policy.path="$POLICY" --policy.device="cuda:$gpu" \
      --env.type=libero --env.task="$SUITE" --env.task_ids="[$tid]" \
      --env.observation_height=128 --env.observation_width=128 \
      --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
      --output_dir="$OUT/task_$tid" \
      > "$OUT/logs/task_$tid.log" 2>&1
    echo "task $tid (cuda:$gpu) exit=$?" >> "$OUT/logs/status.txt"
  done
}

# Deal the 10 tasks round-robin so every GPU gets a comparable amount of work.
for gpu in $(seq 0 $((N_GPUS - 1))); do
  shard=()
  for tid in $(seq 0 9); do
    [ $((tid % N_GPUS)) -eq "$gpu" ] && shard+=("$tid")
  done
  [ ${#shard[@]} -gt 0 ] && run_shard "$gpu" "${shard[@]}" &
done
wait

echo "DONE $(date)" >> "$OUT/logs/status.txt"
echo "results: $OUT"
