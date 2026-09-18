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

# Required FREE VRAM before a pass may start. A 596M model in FP32 is 2.38 GiB of weights, and
# the load transiently needs roughly double that plus a KV cache and cuBLAS workspace. The
# first CHECK attempt asked only "is used < 1500 MiB", passed that check once before a
# concurrent battery claimed the GPU, and then OOMed BOTH systems at ~4950 MiB free. So the
# guard now measures FREE memory, requires real headroom, and re-checks before every pass
# rather than once per system.
GPU_FREE_REQUIRED_MIB=${GPU_FREE_REQUIRED_MIB:-6000}
GPU_WAIT_TRIES=${GPU_WAIT_TRIES:-240}      # x15s = up to 60 minutes

wait_gpu(){
  local f
  for _ in $(seq 1 "$GPU_WAIT_TRIES"); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
    [ -z "$f" ] && return 0                 # no GPU introspection: let the job decide
    [ "$f" -ge "$GPU_FREE_REQUIRED_MIB" ] && return 0
    echo "  waiting for VRAM: ${f} MiB free, need ${GPU_FREE_REQUIRED_MIB}"
    sleep 15
  done
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
  echo "GPU_GUARD_TIMEOUT: ${f} MiB free after $((GPU_WAIT_TRIES*15))s, need ${GPU_FREE_REQUIRED_MIB}."
  echo "Refusing to start rather than OOM mid-pass. Nothing was written."
  return 1
}

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
  rm -rf "$OUT/gen-$LAB-$name"
  wait_gpu || { echo "PASS $name NOT STARTED (VRAM guard)"; return 1; }
  "$PY_GEN" "$GEN" --model "$MODEL" --label "${LAB}_${name}" \
    --prompts "$pdir/prompts.jsonl" --out "$OUT/gen-$LAB-$name" \
    --max-new-tokens "$mnt" --max-context 4096 --dtype fp32 2>&1 \
    | grep -E "rows_skipped|wall_seconds|Traceback|Error" | head -4
  # A pass that produced no predictions has failed, whatever the pipeline's exit status was.
  # Checked here so a later pass cannot run against a missing dependency and so the system
  # aborts instead of scoring a hole.
  [ -s "$OUT/gen-$LAB-$name/predictions.jsonl" ] || {
    echo "PASS $name PRODUCED NO PREDICTIONS"; rm -rf "$OUT/gen-$LAB-$name"; return 1; }
  return 0
}

echo "=================================================================="
echo "SYSTEM $LAB   split=$SPLIT   model=$MODEL"; date '+start %H:%M:%S'

echo "--- 1/3 solver pass (96 tokens) ---"
gen solver "$V/solver" 96 || { echo "SYSTEM $LAB ABORTED at solver"; exit 1; }

echo "--- 2/3 token-allowance pass (392 tokens) ---"
gen tokenmatch "$V/tokenmatch" 392 || { echo "SYSTEM $LAB ABORTED at tokenmatch"; exit 1; }

echo "--- render selector prompts from recorded candidates (0 model calls) ---"
rm -rf "$OUT/sel-$LAB"
"$PY_RENDER" "$D/inference/render_selector.py" --scout "$S/SCOUT.jsonl" --split "$SPLIT" \
  --solver-gen "$OUT/gen-$LAB-solver" --tokenizer "$TOK" --out "$OUT/sel-$LAB" \
  | grep -E "rep_s2|mr_s2|mr_s1|malformed|findings" || exit 1

echo "--- 3/3 selector passes (8 tokens) ---"
gen rep_s2 "$OUT/sel-$LAB/rep_s2" 8 || { echo "SYSTEM $LAB ABORTED at rep_s2"; exit 1; }
gen mr_s2  "$OUT/sel-$LAB/mr_s2"  8 || { echo "SYSTEM $LAB ABORTED at mr_s2"; exit 1; }
gen mr_s1  "$OUT/sel-$LAB/mr_s1"  8 || { echo "SYSTEM $LAB ABORTED at mr_s1"; exit 1; }

echo "--- score ---"
"$PY_GEN" "$D/analysis/score_matrix.py" --scout "$S/SCOUT.jsonl" --split "$SPLIT" --label "$LAB" \
  --solver "$OUT/gen-$LAB-solver" --tokenmatch "$OUT/gen-$LAB-tokenmatch" \
  --rep-s2 "$OUT/gen-$LAB-rep_s2" --mr-s2 "$OUT/gen-$LAB-mr_s2" --mr-s1 "$OUT/gen-$LAB-mr_s1" \
  --out "$OUT/MATRIX_$LAB.json"

date '+  done %H:%M:%S'
echo "SYSTEM $LAB COMPLETE"
