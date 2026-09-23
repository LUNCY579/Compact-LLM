"""Inventory panel artifacts (probe 5f6df1c1): what per-row information does each carry?

Kinds recorded:
  PRED_FULL    150 raw predictions (text + terminated), rescorable
  SCORED_FULL  150 scored rows with raw text (per_row in a RETENTION_PROBE file), rescorable
  BOOL_FULL    150 per-row correctness flags only
  LOST_ONLY    lost_ids per class (+ counts), no gained ids
  SAMPLES_n    n stored sample rows with raw text
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

REC = Path('C:/Users/clclc/BusinessProject/rineya-dataset-retention-records')
PILOT_EV = Path('C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917/experiments/architecture_benchmark_001/evidence/retention-design-v1')
AB_EV = C.AB / 'evidence' / 'retention-design-v1'
AB_OUT = C.AB / 'outputs'
IDSET = set(C.IDS)
SKIP_DIRS = {'.git', 'node_modules'}


def walk_json(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith('.json'):
                yield Path(dp) / f


def kinds_of(d):
    ks = []
    if not isinstance(d, dict):
        return ks
    if isinstance(d.get('predictions'), list):
        ids = {p.get('id') for p in d['predictions'] if isinstance(p, dict)}
        if ids == IDSET and all('text' in p and 'terminated' in p for p in d['predictions']):
            ks.append('PRED_FULL')
    if isinstance(d.get('predictions'), dict) and set(d['predictions']) == IDSET:
        ks.append('PRED_FULL')
    bbp = d.get('by_bare_policy')
    if isinstance(bbp, dict) and 'permissive' in bbp:
        b = bbp['permissive']
        if isinstance(b.get('per_row'), dict) and set(b['per_row']) == IDSET:
            v = next(iter(b['per_row'].values()))
            ks.append('SCORED_FULL' if isinstance(v, dict) and 'raw' in v else 'BOOL_FULL')
        if isinstance(b.get('per_class'), dict) and all('lost_ids' in c for c in b['per_class'].values()):
            ks.append('LOST_ONLY')
    if isinstance(d.get('checkpoints'), list) and d['checkpoints']:
        c = d['checkpoints'][-1]
        if isinstance(c, dict) and isinstance(c.get('outputs'), dict) and set(c['outputs']) == IDSET:
            ks.append('PRED_FULL(ckpt[-1])')
    if isinstance(d.get('rows'), list) and d['rows'] and isinstance(d['rows'][0], dict) and 'source' in d['rows'][0]:
        if {r['id'] for r in d['rows']} == IDSET:
            ks.append('ROWS_FULL(source/arm)')
    s = d.get('samples')
    if isinstance(s, list) and s:
        ks.append(f'SAMPLES_{len(s)}')
    return ks


def probe_sha_of(d):
    if not isinstance(d, dict):
        return ''
    return (d.get('probe_sha256') or '')[:8]


out = []
for root in (REC, PILOT_EV, AB_EV, AB_OUT):
    for p in walk_json(root):
        name = p.name
        if not (name.startswith('RETENTION_PROBE') or name == 'retention_predictions.json' or name.startswith('ARM_DOSE_')
                or name in ('retention.json', 'PANEL.json', 'result.json') or 'G1_RETENTION' in name or 'G2_RETENTION' in name
                or name.endswith('__retention.json')):
            continue
        try:
            b = p.read_bytes()
            d = json.loads(b)
        except Exception as e:  # noqa
            out.append({'path': str(p), 'error': str(e)[:80]})
            continue
        ks = kinds_of(d)
        if not ks:
            continue
        rec = {'path': str(p), 'sha8': hashlib.sha256(b).hexdigest()[:8], 'probe_sha8': probe_sha_of(d),
               'label': d.get('label') if isinstance(d, dict) else None,
               'weight8': ((d.get('hf_candidate_identity') or {}).get('weight_sha256') or '')[:8] if isinstance(d, dict) else '',
               'kinds': ks}
        if name == 'retention_predictions.json':
            for cand in (p.parent / 'probe_binding.json', p.parent.parent / 'probe_binding.json'):
                if cand.is_file():
                    bd = json.loads(cand.read_text(encoding='utf-8'))
                    rec['binding_label'] = bd['arguments'][0]
                    rec['binding_target'] = bd['arguments'][1]
                    rec['binding_result'] = bd['result']
                    break
        out.append(rec)

(HERE / 'inventory.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('artifacts with any per-row structure', len(out))
for r in out:
    lab = r.get('label') or r.get('binding_label') or ''
    print(r.get('sha8', '--'), r.get('probe_sha8', '').ljust(8), (r.get('weight8') or '').ljust(8), str(lab)[:34].ljust(34),
          ','.join(r.get('kinds', []))[:44].ljust(44), r['path'].replace('C:\\Users\\clclc\\BusinessProject\\', '')[-95:])
