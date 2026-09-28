#!/usr/bin/env bash
# Repair-mixture experiment: treatment vs matched control, three training seeds each.
# Strictly sequential, one model job at a time.
#
# Three bugs killed earlier versions of this script. Named so they stay dead:
#   1. gated on `free >= 4.2` via bc. bc is NOT installed here, so the comparison returned
#      0 forever and it waited 90 minutes for a condition it could never observe.
#   2. the PowerShell probe was inlined with nested quotes and broke bash's parser. It now
#      lives in host_state.ps1.
#   3. it treated arm_dose's "REFUSING TO START" as success and fell through all six arms
#      without training anything. A refusal is now detected and the SAME arm is retried.
#
# The gate is 4.15, not 3.6. arm_dose checks `hostbudget.require(3.6)` AFTER importing
# torch, and the interpreter itself costs ~0.5 GiB, so a pre-launch reading of 3.97 becomes
# 3.48 at the check. Gate on what the job will see, not on what is free before it starts.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/c/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
cd "$REPO" || exit 1
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
LOGDIR="$HERE/repair_logs"; mkdir -p "$LOGDIR"
DEADLINE=$(( $(date +%s) + 20700 ))          # ~5h45m, inside the 6h background cap

host_free () {
  powershell -NoProfile -ExecutionPolicy Bypass -File "$HERE/host_state.ps1" 2>/dev/null \
    | tr -d '\r' | tail -1
}

gate_ok () {
  local state jobs free
  state=$(host_free)
  jobs=$(echo "$state" | awk '{print $1+0}')
  free=$(echo "$state" | awk '{print $2+0}')
  [ "$jobs" = "0" ] && [ "$(awk -v f="$free" 'BEGIN{print (f>=4.15)?1:0}')" = "1" ]
}

run_arm () {
  local split="$1" budget="$2" seed="$3" label="$4"
  local tag="B28last20_lr1e-05_b${budget}_seed${seed}_wsd0.2-0.2_b20.95_${label}"
  local art="$REPO/experiments/architecture_benchmark_001/evidence/retention-design-v1/ARM_DOSE_${tag}.json"
  local log="$LOGDIR/${label}_seed${seed}.log"
  if [ -f "$art" ]; then echo "=== $label seed $seed already done, skipping"; return 0; fi
  while true; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      echo "=== DEADLINE reached before $label seed $seed could start"; return 1
    fi
    if gate_ok; then
      echo "=== $label seed $seed starting ($(host_free) )"
      RAB001_SPLIT_DIR="$split" "$PYG" experiments/rab001_exploration/arm_dose.py \
          B28last20 1e-5 "$budget" "$seed" 1.0 wsd:0.2:0.2 0.95 final "$label" \
          > "$log" 2>&1
      if grep -q "REFUSING TO START" "$log"; then
        echo "    refused at the harness guard; host moved under us, retrying in 3 min"
        sleep 180; continue
      fi
      if [ -f "$art" ]; then
        echo "    OK  $(grep -Eo 'correct +[0-9]+/150' "$log" | tail -1)"
        tail -3 "$log"
        return 0
      fi
      echo "    FAILED without producing an artifact -- see $log"; tail -5 "$log"; return 1
    fi
    sleep 120
  done
}

for seed in 17 23 29; do
  run_arm splits_curated_repair_a33_b5233_s17 5302 "$seed" repairA33 || exit 1
  run_arm splits_curated_random_b5233_s17     5233 "$seed" ctrlRandom || exit 1
done
echo "ALL TRAINING RUNS COMPLETE"
