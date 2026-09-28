"""Q6: is the terminal-period habit Opallo-acquired, or already Supa's? Recorded panel outputs only."""
import json, statistics, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_DOSE, PANEL, PROBE, SUPA_REF, dump, jl  # noqa: E402
probe = jl(PROBE)
gold = {r['id']: r['answer'] for r in probe}
supa = {r['id']: r['source']['raw'] for r in json.loads(SUPA_REF.read_text(encoding='utf-8'))['rows']}
bf = {i: v['text'] for i, v in json.loads(PANEL.read_text(encoding='utf-8'))['predictions'].items()}
arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
fp = {i: v['text'] for i, v in arm['checkpoints'][-1]['outputs'].items()}
s0 = {i: v['text'] for i, v in arm['checkpoints'][0]['outputs'].items()}
res = {}
for name, P in [('gold (positive control)', gold), ('supa_ref', supa), ('armdose_step0 (=supa)', s0), ('opallo_fp32', fp), ('opallo_bf16', bf)]:
    per = sum(P[i].strip().endswith('.') for i in gold)
    ln = statistics.mean(len(P[i].strip()) for i in gold)
    wd = statistics.median(len(P[i].split()) for i in gold)
    res[name] = {'ends_period': per, 'mean_chars': ln, 'median_words': wd}
    print(f'{name:26s} ends with "." {per:3d}/150  mean chars {ln:6.1f}  median words {wd}')
# Negative control: a detector that fires on everything would give 150 for a period-free transform
nop = {i: P.rstrip('.') for i, P in fp.items()}
print('NEG control (opallo_fp32 with trailing periods stripped): ends with "."', sum(v.strip().endswith('.') for v in nop.values()))
print('wrote q6_period_signature.json', dump('q6_period_signature.json', res))
