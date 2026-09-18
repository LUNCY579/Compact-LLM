#!/usr/bin/env bash
# DMRF-001 O/T/REP/MR matrix for ONE system on ONE split.
#
# STRICTLY SEQUENTIAL. 15.22 GiB host RAM is the binding constraint and this programme runs one
# model job at a time. Three generation passes per system, each a single model load:
#   solver      96 tokens  -- direct (= O, REP[0], MR R0), rep1-3, R1-3
#   tokenmatch 392 tokens  -- T, the same prompt at MR's total output allowance
#   selector     8 tokens  -- rep_s2, mr_s2, mr_s1
#
# Rendering happens under transformers 4.51.3 (the probes' pin) and generation under the
# version stage2_generate.py was written for. That split is deliberate: the prompt text must
# match the qualified table, the generator need only be faithful, and QWEN-001's path control
# showed it reproduces all six qualified values exactly.
#
# usage: run_matrix.sh <system-label> <model-dir> <split>
set -u
LAB="$1"; MODEL="$2"; SPLIT="$3"

D="C:/Users/clclc/BusinessProject/ChatBubble-dmrf/experiments/delta_multi_reasoning_v1"
S="C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/dmrf001-20260918/scout"
OUT="C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/dmrf001-20260918/matrix-$SPLIT"
PY_RENDER="C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe"
PY_GEN="C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe"
GEN="C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/qwen001-20260918/stage2_generate.py"
TOK="C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"

mkdir -p "$OUT"
V="$OUT/variants"

wait_gpu(){ for _ in $(seq 1 400); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ -z "$u" ] && break; [ "$u" -lt 1500 ] && return 0; sleep 15; done; }

# Variants are split-level, not system-level: render once, reuse for every system. Reusing the
# same prompts file across systems is what makes the comparison paired at all.
if [ ! -f "$V/solver/prompts.jsonl" ]; then
  echo "--- render variants ($SPLIT) ---"
  rm -rf "$V"
  "$PY_RENDER" "$D/inference/render_variants.py" --scout "$S/SCOUT.jsonl" --split "$SPLIT" \
    --tokenizer "$TOK" --out "$V" | grep -E "items|solver_rows|T_max_new_tokens" || exit 1
fi

gen(){ # gen <pass-name> <prompts-dir> <max-new-tokens>
  local name="$1" pdir="$2" mnt="$3"
  [ -f "$pdir/prompts.jsonl" ] || { echo "  skip $name (no prompts)"; return 0; }
  rm -rf "$OUT/gen-$LAB-$name"; wait_gpu
  "$PY_GEN" "$GEN" --model "$MODEL" --label "${LAB}_${name}" \
    --prompts "$pdir/prompts.jsonl" --out "$OUT/gen-$LAB-$name" \
    --max-new-tokens "$mnt" --max-context 4096 --dtype fp32 2>&1 \
    | grep -E "rows_skipped|wall_seconds|Traceback|Error" | head -4
}

echo "=================================================================="
echo "SYSTEM $LAB   split=$SPLIT   model=$MODEL"; date '+start %H:%M:%S'

echo "--- 1/3 solver pass (96 tokens) ---"
gen solver "$V/solver" 96
[ -f "$OUT/gen-$LAB-solver/predictions.jsonl" ] || { echo "SOLVER FAILED"; exit 1; }

echo "--- 2/3 token-allowance pass (392 tokens) ---"
gen tokenmatch "$V/tokenmatch" 392

echo "--- render selector prompts from recorded candidates (0 model calls) ---"
rm -rf "$OUT/sel-$LAB"
"$PY_RENDER" "$D/inference/render_selector.py" --scout "$S/SCOUT.jsonl" --split "$SPLIT" \
  --solver-gen "$OUT/gen-$LAB-solver" --tokenizer "$TOK" --out "$OUT/sel-$LAB" \
  | grep -E "rep_s2|mr_s2|mr_s1|malformed|findings" || exit 1

echo "--- 3/3 selector passes (8 tokens) ---"
gen rep_s2 "$OUT/sel-$LAB/rep_s2" 8
gen mr_s2  "$OUT/sel-$LAB/mr_s2"  8
gen mr_s1  "$OUT/sel-$LAB/mr_s1"  8

echo "--- score ---"
"$PY_GEN" "$D/analysis/score_matrix.py" --scout "$S/SCOUT.jsonl" --split "$SPLIT" --label "$LAB" \
  --solver "$OUT/gen-$LAB-solver" --tokenmatch "$OUT/gen-$LAB-tokenmatch" \
  --rep-s2 "$OUT/gen-$LAB-rep_s2" --mr-s2 "$OUT/gen-$LAB-mr_s2" --mr-s1 "$OUT/gen-$LAB-mr_s1" \
  --out "$OUT/MATRIX_$LAB.json"

date '+  done %H:%M:%S'
echo "SYSTEM $LAB COMPLETE"
