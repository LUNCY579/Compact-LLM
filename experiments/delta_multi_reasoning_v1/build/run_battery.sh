#!/usr/bin/env bash
# DMRF-001 battery for one arm. STRICTLY SEQUENTIAL: 15.22 GiB host RAM is the binding
# constraint, so never more than one model job at a time. Identical wrapper, probe SHAs,
# interpreter and conditions as RANK-001/INT-001/SPARSE-001 so the numbers are comparable
# to the existing table.
#
# usage: run_battery.sh <arm-label> <arm-dir>
set -u
LABEL="$1"; ARM="$2"
PY="C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe"
PILOT="C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917"
PROBE="$PILOT/docs/model-quality/audit-2026-09-17/supervised_probe.py"
OUTROOT="C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/dmrf001-20260918/eval"
SHA_PROCESS="d89f86abd522ca4f8155f6e67e3fcb2cd5fd609698ad0eb8e3e04a02588cc2bf"
SHA_RETENTION="34e0c4b2cc17184fe320b1636f7d1fdc690e8885765efe3423a0e03104b6b586"
SHA_LONGCTX="43b96ca7514740ae144007ed0c6db2a55e47c9fa62c43e7ed4a9ccd31a461d7a"
mkdir -p "$OUTROOT"
cd "$PILOT" || exit 1
echo "=================================================================="
echo "ARM $LABEL   dir=$ARM"; date '+start %H:%M:%S'

echo "--- 1/3 process (SELECT-VIEWS-001, selection-admissible) ---"
"$PY" "$PROBE" --kind process --out "$OUTROOT/$LABEL-process" --timeout 900 \
  --expected-script-sha "$SHA_PROCESS" -- \
  --model "$ARM" --label "dmrf001_$LABEL" --device cuda --instrument select-views-001
echo "process exit=$?"; date '+  done %H:%M:%S'

echo "--- 2/3 retention (150-row arithmetic panel + 47 source answers) ---"
"$PY" "$PROBE" --kind retention --out "$OUTROOT/$LABEL-retention" --timeout 900 \
  --expected-script-sha "$SHA_RETENTION" -- \
  "dmrf001_$LABEL" "hf:$ARM" cuda
echo "retention exit=$?"; date '+  done %H:%M:%S'

echo "--- 3/3 long_context (256 rows, SPENT suite, REPORT-ONLY) ---"
"$PY" "$PROBE" --kind long_context --out "$OUTROOT/$LABEL-long_context" --timeout 900 \
  --expected-script-sha "$SHA_LONGCTX" -- \
  --model "$ARM" --label "dmrf001_$LABEL" --device cuda
echo "long_context exit=$?"; date '+  done %H:%M:%S'
echo "ARM $LABEL COMPLETE"
