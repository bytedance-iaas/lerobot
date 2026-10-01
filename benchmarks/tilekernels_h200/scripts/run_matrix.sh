#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_root=${RESULTS_ROOT:?Set RESULTS_ROOT to a writable container path}
export BENCH_GPUS=${BENCH_GPUS:-0,1}
mkdir -p "$results_root/results"
trap 'rc=$?; printf "%s\n" "$rc" >"$results_root/matrix-exit-code"' EXIT

for policy in groot pi05; do
    for item in baseline:1 full:1 full:2 baseline:2; do
        mode=${item%:*}
        repeat=${item#*:}
        run="$results_root/results/${policy}-${mode}-run${repeat}"
        if [[ -f "$run/exit-code" && "$(<"$run/exit-code")" == 0 ]]; then
            continue
        fi
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits >"$results_root/gpu-current.csv"
        if ! awk -F, '($1+0==0 || $1+0==1) && ($2+0>=512 || $3+0>=5) {busy=1} END {exit busy}' \
            "$results_root/gpu-current.csv"; then
            printf '%s\n' "Selected GPUs busy before $policy $mode $repeat; matrix stopped."
            exit 3
        fi
        if [[ -d "$run" ]]; then
            printf '%s\n' "Existing incomplete run $run must be inspected before retry."
            exit 4
        fi
        mkdir -p "$run"
        cp "$results_root/gpu-current.csv" "$run/gpu-before.csv"
        printf '%s %s %s %s\n' "$(date -u +%FT%TZ)" "$policy" "$mode" "$repeat"
        nvidia-smi \
            --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
            --format=csv -l 1 >"$run/gpu.csv" &
        monitor_pid=$!
        set +e
        "$script_dir/run_one.sh" "$policy" "$mode" "$repeat"
        run_rc=$?
        kill "$monitor_pid" 2>/dev/null
        wait "$monitor_pid" 2>/dev/null
        set -e
        printf '%s\n' "$run_rc" >"$run/launcher-exit-code"
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader >"$run/gpu-after.csv"
        if [[ "$run_rc" != 0 ]]; then
            exit "$run_rc"
        fi
    done
done

for policy in groot pi05; do
    validation="$results_root/results/${policy}-full-runvalidate"
    if [[ ! -f "$validation/exit-code" || "$(<"$validation/exit-code")" != 0 ]]; then
        if [[ -d "$validation" ]]; then
            printf '%s\n' "Existing incomplete validation $validation must be inspected before retry."
            exit 4
        fi
        TK_VERIFY=1 TK_VALIDATE_ONLY=1 \
            "$script_dir/run_one.sh" "$policy" full validate
    fi
done

CUDA_VISIBLE_DEVICES=${BENCH_GPUS%%,*} \
TK_OP_CHECK_OUTPUT="$results_root/tile-ops-check.json" \
    /opt/venv/bin/python "$script_dir/check_tile_ops.py"

/opt/venv/bin/python "$script_dir/summarize.py" \
    --root "$results_root" --output "$results_root/summary.json"
