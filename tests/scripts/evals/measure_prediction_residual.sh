#!/usr/bin/env bash
# Collect LingBot-VA prediction-residual traces alongside episode outcomes, to
# characterise how far the world model drifts from reality before and while it
# fails. Reproduces the numbers quoted for --policy.track_prediction_residual.
#
# Usage:
#   tests/scripts/evals/measure_prediction_residual.sh [n_episodes] [policy_path]
#
# Four runs in parallel, chosen so the traces cover the full outcome range:
#   libero_10 task 0        ~96% success  -> in-suite reference
#   libero_spatial task 2   ~64% success  -> mixed, residual vs. outcome
#   libero_spatial task 3   ~44% success  -> mixed, residual vs. outcome
#   libero_spatial task 1     0% success  -> pure-failure distribution
#
# Each episode's trace lands in eval_info.json as per_task -> metrics ->
# residual_traces (one list of per-chunk values per episode), paired with
# successes. See eval_libero_sharded.sh for why GPUs are selected with
# --policy.device rather than CUDA_VISIBLE_DEVICES.
set -u

REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

N_EPISODES=${1:-10}
POLICY=${2:-lerobot/lingbot_va_libero_long}

OUT=outputs/eval/prediction_residual
mkdir -p "$OUT/logs"
unset CUDA_VISIBLE_DEVICES

run() {
  local gpu=$1 suite=$2 tid=$3 name=$4
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync lerobot-eval \
    --policy.path="$POLICY" --policy.device="cuda:$gpu" \
    --policy.track_prediction_residual=true \
    --env.type=libero --env.task="$suite" --env.task_ids="[$tid]" \
    --env.observation_height=128 --env.observation_width=128 \
    --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
    --output_dir="$OUT/$name" \
    > "$OUT/logs/$name.log" 2>&1
  echo "$name exit=$?" >> "$OUT/logs/status.txt"
}

run 0 libero_10      0 libero10_t0 &
run 1 libero_spatial 2 spatial_t2 &
run 2 libero_spatial 3 spatial_t3 &
run 3 libero_spatial 1 spatial_t1 &
wait

echo "DONE $(date)" >> "$OUT/logs/status.txt"
python - "$OUT" <<'PY'
import json, statistics as st, sys, pathlib
out = pathlib.Path(sys.argv[1])
def auc(f, s):
    return sum((a > b) + 0.5 * (a == b) for a in f for b in s) / (len(f) * len(s)) if f and s else None
for run in ("libero10_t0", "spatial_t2", "spatial_t3", "spatial_t1"):
    info = out / run / "eval_info.json"
    if not info.exists():
        continue
    ok, bad = [], []
    for task in json.load(open(info))["per_task"]:
        for success, trace in zip(task["metrics"]["successes"], task["metrics"]["residual_traces"]):
            (ok if success else bad).append(st.mean(trace))
    a = auc(bad, ok)
    fmt = lambda g: f"{st.mean(g):.4f}" if g else "  -   "  # noqa: E731
    print(f"{run:12s} success {len(ok):2d} ({fmt(ok)})  failure {len(bad):2d} ({fmt(bad)})"
          f"  within-task AUC {'n/a' if a is None else round(a, 3)}")
PY
