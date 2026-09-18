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

# Shared GPU lease and VRAM guard. SOURCED, not reimplemented: two sessions run model jobs on
# this one card, and if each writes its own lease the field names drift and the lease stops
# being decidable, which is the whole point of having one.
#
# The two mechanisms do different jobs and both are needed:
#   the LEASE arbitrates INTENT   -- who gets the card next, so we queue instead of racing
#   vram_wait arbitrates REALITY  -- whether the headroom is actually there when we load
# Holding the lease does not mean the card is free: the other session's CUDA context can still
# be resident after its job logically finishes. So acquire the lease, THEN wait on real VRAM.
source "C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/gpu_lease.sh"

HOLDER="${HOLDER:-chatbubble-slft-4e}"
GPU_FREE_REQUIRED_MIB=${GPU_FREE_REQUIRED_MIB:-6000}
LEASE_MINUTES=${LEASE_MINUTES:-40}

# Release the lease on ANY exit, including an abort or a kill, so a crash cannot leave the
# other session waiting out the full staleness window.
trap 'gpu_lease_release "$HOLDER"' EXIT INT TERM

wait_gpu(){
  gpu_lease_acquire "$HOLDER" "$LEASE_MINUTES" "DMRF-001 $LAB $SPLIT" || {
    echo "LEASE_TIMEOUT: could not acquire the GPU lease."; return 1; }
  vram_wait "$GPU_FREE_REQUIRED_MIB" 60 || {
    echo "VRAM_TIMEOUT: lease held but ${GPU_FREE_REQUIRED_MIB} MiB never became free."
    echo "Refusing to start rather than OOM mid-pass. Nothing was written."; return 1; }
  return 0
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
  local ok=0
  [ -s "$OUT/gen-$LAB-$name/predictions.jsonl" ] || {
    echo "PASS $name PRODUCED NO PREDICTIONS"; rm -rf "$OUT/gen-$LAB-$name"; ok=1; }
  # Hand the card back between passes. The other session's arms take ~25 minutes and my
  # passes take minutes, so holding the lease across a whole system would starve it for no
  # reason. Re-acquired by the next pass's wait_gpu.
  gpu_lease_release "$HOLDER"
  return $ok
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
