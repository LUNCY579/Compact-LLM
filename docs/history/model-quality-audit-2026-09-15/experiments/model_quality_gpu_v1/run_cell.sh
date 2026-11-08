#!/usr/bin/env bash
# Run ONE MQ-001 training cell, then reclaim its checkpoint shards.
#
#   run_cell.sh <arm> <prepared-dir> <qualification-dir> <out-dir>
#
# Disk is the binding constraint on this host, not memory. Each cell writes four checkpoints
# of ~3.6 GB (trainables + both Adam moments + RNG) and the preflight demands 16.1 GiB free
# before it will start. Six cells would accumulate ~84 GiB against a disk that had 19.8 GB
# free. So each cell's checkpoint-* directories are deleted once its evidence exists and
# validates -- evidence JSON is never touched, and DESIGN.md calls the shards "training-state
# shards, not a ready HF release". Every cell remains reproducible from its plan and seed.
#
# Guards are never overridden. A refusal is reported and the script stops.
#
# Shell lessons this repo has already paid for: `bc` is NOT installed (use awk); never put a
# pipe inside $( ) when you need the command's own exit status (you get the pipe's); and a
# "REFUSING TO START" is a failure even though the pipeline exits 0.
set -u
ARM="$1"; PREP="$2"; QUAL="$3"; OUT="$4"
REPO=/c/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
MQ=C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe
cd "$REPO" || exit 1

free_gb=$(powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,1)" | tr -d '\r')
echo "=== $ARM -> $(basename "$OUT")  | disk ${free_gb} GB free (preflight needs 16.1 GiB)"
"$MQ" -c "import sys;sys.path.insert(0,'.');from experiments.model_quality_gpu_v1.resources import snapshot;s=snapshot();print(f\"    host {s['available_host_bytes']/2**30:.2f} GiB | gpu {s['gpu0']['free']/2**30:.2f} GiB\")"

log=$(mktemp)
"$MQ" -m experiments.model_quality_gpu_v1 train --arm "$ARM" \
  --source "$REPO/Supa" --plan "$PREP/plan.json" --review "$PREP/DATA_REVIEW_APPROVED.json" \
  --qualification "$QUAL/qualification.json" --confirm-training --out "$OUT" > "$log" 2>&1
rc=$?
if grep -q "guard refused\|REFUSING TO START" "$log"; then
  echo "    GUARD REFUSED - not overriding:"; grep -m2 "guard refused\|REFUSING" "$log" | sed 's/^/      /'
  rm -f "$log"; exit 2
fi
if [ ! -f "$OUT/result.json" ]; then
  echo "    FAILED (rc=$rc), no result.json:"; tail -6 "$log" | sed 's/^/      /'
  [ -f "$OUT/failure.json" ] && { echo "      failure.json:"; cat "$OUT/failure.json" | sed 's/^/      /'; }
  rm -f "$log"; exit 1
fi
rm -f "$log"

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
sc = [x for x in fin['records'] if x['scores']['correct'] is not None]
rb = {x['id']: x for x in base['records']}
print(f"    {r['status']} | updates {r['progress']['updates']} | supervised {r['progress']['supervised_tokens']:,}")
print(f"    CE {ce[0]:.3f} -> {ce[-1]:.3f} (finite {all(math.isfinite(c) for c in ce)}) | "
      f"grads finite {all(math.isfinite(s['grad_norm']) for s in steps)}")
print(f"    DEV exact {sum(1 for x in sc if rb[x['id']]['scores']['correct'])}/{len(sc)}"
      f" -> {sum(1 for x in sc if x['scores']['correct'])}/{len(sc)}"
      f" | response CE {base['response_ce']:.4f} -> {fin['response_ce']:.4f}"
      f" | terminated {sum(1 for x in fin['records'] if x['terminated'])}/64")
PY

before=$(du -sm "$OUT" | cut -f1)
rm -rf "$OUT"/checkpoint-*
cat > "$OUT/CHECKPOINTS_REMOVED.txt" <<EOF
checkpoint-* removed immediately after this cell's evidence was written and validated, to
keep disk within budget for the remaining cells of the three-seed replication. Evidence JSON
(result.json, steps.jsonl, baseline.json, final_dev.json, binding.json, launch.json,
supervisor.json, worker.log) is untouched. The cell is reproducible from its plan and seed.
EOF
echo "    reclaimed $(( before - $(du -sm "$OUT" | cut -f1) )) MB of checkpoint shards; evidence kept"
