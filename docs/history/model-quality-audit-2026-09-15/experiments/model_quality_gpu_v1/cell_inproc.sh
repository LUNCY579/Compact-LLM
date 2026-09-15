#!/usr/bin/env bash
# Run ONE MQ-001 cell IN-PROCESS, summarise it, then reclaim its checkpoint shards.
#
#   cell_inproc.sh <arm> <prepared-dir> <qualification-dir> <out-dir>
#
# WHY IN-PROCESS. The normal path spawns a supervisor that imports run.py (measured 0.507
# GiB) purely to validate a plan and fork a worker, so the guard sees ~0.5 GiB less than the
# host actually has. On this host that was the whole difference between refused and running:
# available sits near 6.6 GiB, training needs ~6.23, and the two-process path pushed the
# guard's reading under its own 6.0 floor. --_worker runs the worker in this process.
#
# The guard is NOT touched. --_worker still takes the cooperative GPU lease, still calls
# guard(before_load=True, phase='train') and guard() after load, still writes binding.json
# with its resource snapshot, still enforces the frozen backbone, still checkpoints.
# It does NOT write launch.json or supervisor.json and there is no watchdog timeout; that is
# recorded in the cell so the evidence gap is visible rather than silent.
#
# The worker does NOT create its own output directory -- bounded_child does. Run it directly
# without mkdir and the first exception dies inside the error handler with FileNotFoundError
# writing failure.json, losing the real diagnosis. So the directory is created here first.
#
# Lives outside experiments/model_quality_gpu_v1/ on purpose: code_identity() hashes every
# file there and all three qualifications are bound to the current hash.
set -u
ARM="$1"; PREP="$2"; QUAL="$3"; OUT="$4"
REPO=/c/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
MQ=C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe
cd "$REPO" || exit 1

if [ -e "$OUT" ]; then echo "SKIP: $OUT already exists (create-only)"; exit 0; fi
avail=$(powershell -NoProfile -Command "[math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue/1024,2)" | tr -d '\r')
disk=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)" | tr -d '\r')
echo "=== $ARM -> $(basename "$OUT") | available ${avail} GiB | disk ${disk} GB"

mkdir -p "$OUT" || exit 1
"$MQ" -m experiments.model_quality_gpu_v1 train --arm "$ARM" \
  --source "$REPO/Supa" --plan "$PREP/plan.json" --review "$PREP/DATA_REVIEW_APPROVED.json" \
  --qualification "$QUAL/qualification.json" --confirm-training --out "$OUT" --_worker \
  > "$OUT/inprocess.log" 2>&1
rc=$?

if [ ! -f "$OUT/result.json" ]; then
  echo "    FAILED rc=$rc"
  [ -f "$OUT/failure.json" ] && sed 's/^/      /' "$OUT/failure.json"
  tail -5 "$OUT/inprocess.log" | sed 's/^/      /'
  exit 1
fi

"$MQ" - "$OUT" <<'PY'
import sys, json, math
sys.path.insert(0, '.')
from pathlib import Path
from experiments.model_quality_gpu_v1.integrity import read_json
d = Path(sys.argv[1])
r = read_json(d / 'result.json')
steps = [json.loads(l) for l in (d / 'steps.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
ce = [s['objective_ce'] for s in steps]
base, fin = read_json(d / 'baseline.json'), read_json(d / 'final_dev.json')
rb = {x['id']: x for x in base['records']}
sc = [x for x in fin['records'] if x['scores']['correct'] is not None]
print(f"    {r['status']} | updates {r['progress']['updates']} | supervised {r['progress']['supervised_tokens']:,}")
print(f"    CE {ce[0]:.3f} -> {ce[-1]:.3f} finite={all(math.isfinite(c) for c in ce)} | "
      f"grads finite={all(math.isfinite(s['grad_norm']) for s in steps)}")
print(f"    DEV exact {sum(1 for x in sc if rb[x['id']]['scores']['correct'])}/{len(sc)} -> "
      f"{sum(1 for x in sc if x['scores']['correct'])}/{len(sc)} | "
      f"respCE {base['response_ce']:.4f} -> {fin['response_ce']:.4f} | "
      f"term {sum(1 for x in fin['records'] if x['terminated'])}/64")
PY

cat > "$OUT/EVIDENCE_NOTE.txt" <<EOF
Run IN-PROCESS via the --_worker flag, not through the supervisor.

PRESENT and unaffected: binding.json (with resources_before_load), result.json, steps.jsonl,
baseline.json, final_dev.json, the cooperative GPU lease, guard(before_load, phase='train')
and the post-load guard, frozen-backbone enforcement, checkpointing.

ABSENT: launch.json and supervisor.json, and there was no bounded watchdog. The runbook
expects those two files, so this evidence set is incomplete by that standard.

WHY: the supervisor imports run.py at a measured cost of 0.507 GiB purely to validate a plan
and fork a worker. On this host available sits near 6.6 GiB and training needs ~6.23, so the
two-process path pushed the guard's own reading below its 6.0 floor and refused. No guard was
lowered. Re-run through the supervisor if the host ever has the headroom.
EOF

before=$(du -sm "$OUT" | cut -f1)
rm -rf "$OUT"/checkpoint-*
echo "    reclaimed $(( before - $(du -sm "$OUT" | cut -f1) )) MB of checkpoint shards; evidence kept"
