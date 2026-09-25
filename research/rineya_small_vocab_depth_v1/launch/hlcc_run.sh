#!/usr/bin/env bash
# HLCC-001 real run (HLCC001_PREREG.md §9, R2). Launch ONLY via launch_hlcc.ps1 (WMI), and only after
# CLOCKFIX-001's "ended" (order agreed with Rineya Delta). Refuses unless the code and data on disk are
# the registered ones. One retry on the GPU-reset signature only (§9 permits infrastructure reruns;
# HLCC_RESULT.json is created only at the end; every post-boundary failure leaves an attempt record).
set -u
PY=/c/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
PKG=/c/Users/clclc/BusinessProject/ChatBubble-small-vocab-depth/research/rineya_small_vocab_depth_v1
T=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
R=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/smallvd001-20260924
H=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hlcc001-20260925
LOG="$1"
cd "$PKG" || exit 2
check() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "REFUSE: $1 is not the registered file" | tee -a "$LOG"; exit 3; }; }
check hlcc_verdict.py baf465b803c25d26d95d70c90dc138e68b774d617bd4e18932e4bb7f90952531
check run_hlcc.py 7cbfa2391cfa96f1dbfae37e6ddfe5bc2b803aff76c2e97560ee233dd116981d
check linear_compensation.py a8abc72e6f91167f9711fa9270d9d43f014c6a6a0153d2ec60a59872890a637d
check "$H/panel/CONFIRM_PANEL.jsonl" 8eec83da9fccfc7e2a253ebf2ce4fab51be319a4c1def8ce1e196343602c0220
check "$H/HLCC_ALLAXIS_UNIT_MAP.json" b5137cde168a667619fa31edea45b7cd5c15b1791b8268601eafa2baef407a64
[ -e "$H/result" ] && { echo "REFUSE: $H/result exists" | tee -a "$LOG"; exit 4; }
run() {
  "$PY" -B run_hlcc.py --teacher "$T" --panels "$R/calibration/LAYER_PANELS.jsonl" --study "$R/layer_study" \
    --confirm "$H/panel/CONFIRM_PANEL.jsonl" --vocab-primary "$R/vocab/V4_32768" --vocab-secondary "$R/vocab/V2_65536" \
    --primary-kept 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 \
    --secondary-kept 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 \
    --smallvd-lc "$R/lc_arms/LC_ARMS.json" --allaxis-map "$H/HLCC_ALLAXIS_UNIT_MAP.json" --out "$H/result" --device cuda
}
for attempt in 1 2; do
  echo "== $(date -u +%FT%TZ) attempt $attempt" >> "$LOG"
  run >> "$LOG" 2>&1; rc=$?
  echo "== $(date -u +%FT%TZ) attempt $attempt rc=$rc" >> "$LOG"
  [ $rc -eq 0 ] && exit 0
  tail -40 "$LOG" | grep -q "CUDA error: unknown error" || exit $rc
  echo "== GPU-RESET signature: one retry" >> "$LOG"
done
exit $rc
