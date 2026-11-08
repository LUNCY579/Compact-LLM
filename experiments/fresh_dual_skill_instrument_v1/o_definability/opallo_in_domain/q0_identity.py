"""Q0: reproduce the recorded correct sets BY IDENTITY before any downstream number.

Supa reference (permissive 47 / strict 41), Opallo BF16 PANEL (73 / 59), Opallo FP32
ARM_DOSE checkpoints[-1] (74 / 59). Also: ARM_DOSE checkpoints[-1].outputs == the
outputs/arm-runs probe_0005200.json outputs (the copy that physically lives under outputs/).
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARM_DOSE, ARM_RUN, AUDIT, PANEL, PROBE, SUPA_REF, dump, jl, load_scorer,  # noqa
                    sha, typed_targets)

R, rh = load_scorer()
print('retention.py', rh[:16], 'module', R.__name__, '| torch imported:', 'torch' in sys.modules)
hashes = {'retention.py': rh}
for n, p in [('audit', AUDIT), ('probe', PROBE), ('supa_ref', SUPA_REF), ('panel', PANEL),
             ('arm_dose', ARM_DOSE), ('arm_run_probe_0005200', ARM_RUN / 'probe_0005200.json'),
             ('arm_run_probe_0000000', ARM_RUN / 'probe_0000000.json')]:
    hashes[n] = sha(p)
assert hashes['probe'].startswith('5f6df1c1')
probe = jl(PROBE)
assert len(probe) == 150
raw_probe = PROBE.read_bytes()
print('probe CRLF lines:', raw_probe.count(b'\r\n'), 'rows', len(probe))
audit = json.loads(AUDIT.read_text(encoding='utf-8'))
T = typed_targets(R, audit)
print('typed targets built via target_from_audit for', {p: len(t) for p, t in T.items()})

supa = json.loads(SUPA_REF.read_text(encoding='utf-8'))
panel = json.loads(PANEL.read_text(encoding='utf-8'))
arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
p5200 = json.loads((ARM_RUN / 'probe_0005200.json').read_text(encoding='utf-8'))
p0 = json.loads((ARM_RUN / 'probe_0000000.json').read_text(encoding='utf-8'))

arms = {
    'supa': {r['id']: (r['source']['raw'], r['source']['terminated']) for r in supa['rows']},
    'opallo_bf16': {i: (v['text'], v['terminated']) for i, v in panel['predictions'].items()},
    'opallo_fp32': {i: (v['text'], v['terminated'])
                    for i, v in arm['checkpoints'][-1]['outputs'].items()},
}
ids = [r['id'] for r in probe]
for a, d in arms.items():
    assert set(d) == set(ids), a


def correct(preds, pol):
    return {i for i, (t, term) in preds.items() if R.score_prediction(t, term, T[pol][i])['correct']}


C = {(a, p): correct(d, p) for a, d in arms.items() for p in R.BARE_POLICIES}
checks = {}
checks['supa_perm_vs_reference_ids'] = C[('supa', 'permissive')] == set(supa['reference_ids_by_policy']['permissive'])
checks['supa_strict_vs_reference_ids'] = C[('supa', 'strict')] == set(supa['reference_ids_by_policy']['strict'])
checks['bf16_perm_vs_PANEL_correct_ids'] = C[('opallo_bf16', 'permissive')] == set(panel['by_policy']['permissive']['correct_ids'])
checks['bf16_strict_vs_PANEL_correct_ids'] = C[('opallo_bf16', 'strict')] == set(panel['by_policy']['strict']['correct_ids'])
per_row = arm['checkpoints'][-1]['retention']['per_row']
checks['fp32_perm_vs_ARM_DOSE_per_row'] = C[('opallo_fp32', 'permissive')] == {i for i, ok in per_row.items() if ok}
checks['fp32_strict_total_vs_recorded'] = len(C[('opallo_fp32', 'strict')]) == arm['checkpoints'][-1]['retention']['strict']['correct_total']
checks['arm_dose_last_outputs_eq_outputs_dir_probe_0005200'] = arm['checkpoints'][-1]['outputs'] == p5200['outputs']
checks['arm_dose_step0_outputs_eq_probe_0000000'] = arm['checkpoints'][0]['outputs'] == p0['outputs']
counts = {f'{a}|{p}': len(v) for (a, p), v in C.items()}
print('counts:', counts)
for k, v in checks.items():
    print(f'  {k}: {v}')
expected = {'supa|permissive': 47, 'supa|strict': 41, 'opallo_bf16|permissive': 73,
            'opallo_bf16|strict': 59, 'opallo_fp32|permissive': 74, 'opallo_fp32|strict': 59}
ok = all(checks.values()) and counts == expected
print('IDENTITY', 'OK' if ok else 'FAILED')
# Negative control for the identity check itself: shuffle predictions across rows; the
# recomputed set must NOT equal the recorded one (else identity equality is vacuous).
import random
rng = random.Random(0)
keys = list(arms['opallo_bf16'])
vals = [arms['opallo_bf16'][k] for k in keys]
rng.shuffle(vals)
sh = dict(zip(keys, vals))
neg = correct(sh, 'permissive')
print('NEGATIVE control (BF16 preds shuffled across rows): correct', len(neg),
      'equal to recorded?', neg == set(panel['by_policy']['permissive']['correct_ids']))
res = {'identity_ok': ok, 'checks': checks, 'counts': counts, 'hashes': hashes,
       'neg_control_shuffled_bf16_perm_correct': len(neg),
       'correct_sets': {k: sorted(v) for k, v in {f'{a}|{p}': s for (a, p), s in C.items()}.items()}}
print('wrote q0_identity.json', dump('q0_identity.json', res))
