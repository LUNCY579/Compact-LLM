#!/usr/bin/env bash
# SMALLVD-001 job queue. Each step waits for host RAM, then runs with its OWN gates intact.
# Retries ONLY on the gates' own RAM refusals/stand-downs ("REFUSING TO START", "standing down");
# any other failure stops the chain, so a real bug is never retried into a pass.
# Polling costs one GlobalMemoryStatusEx call per minute (observability is not free: keep it rare).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
HB="$(cygpath -w "$HERE/../../experiments/rab001_exploration")"   # Windows python needs a Windows path
T=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
R=C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/smallvd001-20260924
S="${SMALLVD_SCRATCH:?set SMALLVD_SCRATCH}"
export PYTHONIOENCODING=utf-8
LOG="$S/queue.log"

free_ge() { python -B -c "import sys; sys.path.insert(0, r'$HB'); import hostbudget; sys.exit(0 if hostbudget.free_gib() >= $1 else 1)"; }

wait_ram() {   # $1 GiB, $2 max minutes
  local i=0
  until free_ge "$1"; do
    i=$((i + 1)); [ "$i" -ge "$2" ] && { echo "$(date -u +%FT%TZ) TIMEOUT waiting for $1 GiB" >> "$LOG"; return 1; }
    sleep 60
  done
}

step() {       # name, wait-GiB, max-tries, command...
  local name=$1 need=$2 tries=$3; shift 3
  local k=0
  while [ "$k" -lt "$tries" ]; do
    k=$((k + 1))
    wait_ram "$need" 720 || return 1
    echo "$(date -u +%FT%TZ) START $name (try $k)" >> "$LOG"
    out=$("$@" 2>&1); rc=$?
    printf '%s\n' "$out" | tail -40 > "$S/$name.try$k.log"
    if [ "$rc" -eq 0 ]; then echo "$(date -u +%FT%TZ) OK $name" >> "$LOG"; return 0; fi
    if printf '%s' "$out" | grep -q -e "REFUSING TO START" -e "standing down"; then
      echo "$(date -u +%FT%TZ) RAM-REFUSAL $name (try $k), backing off 5 min then waiting" >> "$LOG"
      sleep 300; continue
    fi
    echo "$(date -u +%FT%TZ) FAIL $name rc=$rc (not a RAM refusal) -- chain stopped" >> "$LOG"; return 1
  done
  echo "$(date -u +%FT%TZ) GAVE UP $name after $tries RAM refusals" >> "$LOG"; return 1
}

cd "$HERE" || exit 1
# SELF-TEST: the RAM probe must return a real number before anything waits on it. The first launch
# of this queue passed a POSIX path to Windows python, the import failed, and every wait would have
# "waited for RAM" forever -- a vacuous check that looks exactly like patience.
probe=$(python -B -c "import sys; sys.path.insert(0, r'$HB'); import hostbudget; print(round(hostbudget.free_gib(), 2))")   || { echo "$(date -u +%FT%TZ) SELF-TEST FAILED: RAM probe does not run" >> "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) SELF-TEST OK: free ${probe} GiB" >> "$LOG"
step dryrun 1.2 8 $PY -B run_layer_study.py --teacher "$S/tiny_teacher" --panels "$R/calibration/LAYER_PANELS.jsonl" \
  --vocab-dirs "$S/dry_vocab_V2" --out "$S/dry_run_3" --resume --min-layers 2 --depths 4,3,2 --device cpu \
  --max-rows-per-category 3 --gen-tokens 8 --dry-run || exit 1
step dry_surgery 1.2 8 $PY -B run_surgery_arms.py --teacher "$S/tiny_teacher" --vocab-dir "$S/dry_vocab_V2"   --layer-report "$S/dry_report.json" --expect-depth 3 --build-root "$S/dry_arms_build"   --records-out "$S/dry_arms_records" --resume --dry-run || exit 1
# built in the SCRATCH staging dir and moved into the records tree only when complete: a RAM
# stand-down mid-build then leaves nothing partial in records (nothing there may be deleted)
vocab_build() { rm -rf "$S/vocab_build"; $PY -B build_vocab_candidates.py --teacher "$T"   --corpus "$R/calibration/TOKENIZER_CORPUS.jsonl" --out "$S/vocab_build"; }
[ -d "$R/vocab" ] || { step vocab 1.8 8 vocab_build && mv "$S/vocab_build" "$R/vocab"; } || exit 1
V1="$R/vocab/V1_98304"; V2="$R/vocab/V2_65536"; V3="$R/vocab/V3_49152"; V4="$R/vocab/V4_32768"
g1_tests() { SMALLVD_TOKENIZER_DIRS="$(cygpath -w "$V1");$(cygpath -w "$V2");$(cygpath -w "$V3");$(cygpath -w "$V4")"   $PY -B -m pytest -p no:cacheprovider -q tests/test_tokenizer.py tests/test_tokenizer_negative_controls.py   > "$S/G1_TOKENIZER_TESTS.log" 2>&1; rc=$?; cp "$S/G1_TOKENIZER_TESTS.log" "$R/vocab/G1_TOKENIZER_TESTS_$(date -u +%Y%m%dT%H%M%SZ).log"; return $rc; }
step g1 1.5 8 g1_tests || exit 1
[ -f "$R/G0B_BASELINE_REPRODUCTION.json" ] || step g0b 3.8 8 $PY -B g0b_reproduce.py --teacher "$T"   --out "$R/G0B_BASELINE_REPRODUCTION.json" || exit 1
step layerstudy 3.8 10 $PY -B run_layer_study.py --teacher "$T" --panels "$R/calibration/LAYER_PANELS.jsonl"   --vocab-dirs "$V1" "$V2" "$V3" "$V4" --out "$R/layer_study" --resume || exit 1
[ -f "$R/layer_study/LAYER_REPORT.json" ] || step report 1.0 4 $PY -B report_layer_study.py --study "$R/layer_study"   --panels "$R/calibration/LAYER_PANELS.jsonl" --out-md "$R/layer_study/LAYER_REPORT.md"   --out-json "$R/layer_study/LAYER_REPORT.json" || exit 1
[ -d "$R/arch" ] || step arch 0.5 4 $PY -B arch_manifests.py --teacher "$T" --vocab-dir "$V2"   --layer-report "$R/layer_study/LAYER_REPORT.json" --out "$R/arch" || exit 1
step surgery 3.8 8 $PY -B run_surgery_arms.py --teacher "$T" --vocab-dir "$V2" --resume   --layer-report "$R/layer_study/LAYER_REPORT.json" --build-root "$S/arms_build" --records-out "$R/arms" || exit 1
echo "$(date -u +%FT%TZ) QUEUE COMPLETE" >> "$LOG"
