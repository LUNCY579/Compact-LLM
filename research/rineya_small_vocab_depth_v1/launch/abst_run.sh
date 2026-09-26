#!/usr/bin/env bash
# ABST-001 real run (ABST001_PREREG.md §9). Launch ONLY via launch_abst.ps1 (WMI), and only after the
# agreed queue slot. Refuses (exit 3, nothing written) unless EVERY file in launch/abst_hashes.txt --
# each imported module and each data file -- hashes to its registered value. One retry on the GPU-reset
# signature only, and only if ABST_PRIMARY.json was NOT yet written (§9).
set -u
PY=/c/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
PKG=/c/Users/clclc/BusinessProject/ChatBubble-small-vocab-depth/research/rineya_small_vocab_depth_v1
T=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
R=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/smallvd001-20260924
HL=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hlcc001-20260925
A=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/abst001-20260925
LOG="$1"
HASHES_SHA=f780766b8bdc02059b0e450f0d6950b661cd48681b1c07659516bf9a68b35f35
cd "$PKG" || exit 2
# the hash list verifies ITSELF first (it does not list this script), then every line strictly
[ "$(sha256sum launch/abst_hashes.txt | cut -d' ' -f1)" = "$HASHES_SHA" ] || { echo "REFUSE: launch/abst_hashes.txt is not the registered list" | tee -a "$LOG"; exit 3; }
sha256sum --strict --quiet -c launch/abst_hashes.txt >> "$LOG" 2>&1 || { echo "REFUSE: a gated file is not the registered file" | tee -a "$LOG"; exit 3; }
[ -e "$A/result" ] && { echo "REFUSE: $A/result exists" | tee -a "$LOG"; exit 4; }
run() {
  "$PY" -B run_abst.py --teacher "$T" --panels "$R/calibration/LAYER_PANELS.jsonl" --study "$R/layer_study" \
    --smallvd-lc "$R/lc_arms/LC_ARMS.json" --hlcc-result "$HL/result/HLCC_RESULT.json" \
    --abst-panel "$A/panel/ABST_PANEL.jsonl" --manifest "$A/panel/ABST_MANIFEST.json" \
    --confirm-panel "$HL/panel/CONFIRM_PANEL.jsonl" --tokenizer-corpus "$R/calibration/TOKENIZER_CORPUS.jsonl" \
    --vocab-v4 "$R/vocab/V4_32768" --vocab-v2 "$R/vocab/V2_65536" \
    --s20-kept 0,1,2,3,4,7,9,10,11,15,16,17,18,19,20,21,22,25,26,27 \
    --s18-kept 0,1,2,3,4,9,10,11,15,16,17,18,20,21,22,25,26,27 \
    --out "$A/result" --device cuda
}
for attempt in 1 2; do
  echo "== $(date -u +%FT%TZ) attempt $attempt" >> "$LOG"
  run >> "$LOG" 2>&1; rc=$?
  echo "== $(date -u +%FT%TZ) attempt $attempt rc=$rc" >> "$LOG"
  [ $rc -eq 0 ] && exit 0
  [ -e "$A/result/ABST_PRIMARY.json" ] && { echo "== primary already written: no retry (§9)" >> "$LOG"; exit $rc; }
  tail -40 "$LOG" | grep -q "CUDA error: unknown error" || exit $rc
  echo "== GPU-RESET signature before the primary was written: one retry" >> "$LOG"
done
exit $rc
