"""Scoring core for the Titan/Opallo panel-overlap analysis (attempt 2, fresh run).

Read-only against every repo. CPU only, recorded JSON only. The scorer is loaded privately
by file path; run.py and the package are never imported.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

QUAL = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
AB = QUAL / 'experiments' / 'architecture_benchmark_001'
RET_PY = AB / 'retention.py'
AUDIT = AB / 'evidence' / 'retention-design-v1' / 'PROBE_LABEL_AUDIT.json'
PROBE = AB / 'data' / 'monitor_probe.jsonl'
SUPA = AB / 'outputs' / 'retention-a329eb7b' / 'B28' / 'retention.json'
OPALLO_BF16 = QUAL / 'Rineya_Published_Cores' / 'Rineya-Opallo-1.0.0-devfreeze' / 'PANEL.json'
OPALLO_FP32 = AB / 'evidence' / 'retention-design-v1' / 'ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(RET_PY).startswith('c52b6a48'), sha(RET_PY)
assert sha(PROBE).startswith('5f6df1c1'), sha(PROBE)
_spec = importlib.util.spec_from_file_location('_toov_retention_c52b6a48', str(RET_PY))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
POLS = R.BARE_POLICIES

ROWS = [json.loads(l) for l in PROBE.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
IDS = [r['id'] for r in ROWS]
assert len(IDS) == 150 and len(set(IDS)) == 150
CLASS = {r['id']: r['reply_class'] for r in ROWS}

_audit = json.loads(AUDIT.read_text(encoding='utf-8'))
# Exactly the validator's construction: build_overlay == target_from_audit over every audit row.
TARGETS = {}
for p in POLS:
    t = {row['id']: R.target_from_audit(row, p) for row in _audit['rows']}
    assert t == R.build_overlay(_audit, p)['targets']
    assert set(t) == set(IDS), p
    TARGETS[p] = t


def score(preds, policy):
    """preds: {id: {'text': str, 'terminated': bool}} -> set of correct ids."""
    assert set(preds) == set(IDS), (len(preds), len(set(preds) & set(IDS)))
    return {i for i in IDS
            if R.score_prediction(preds[i]['text'], preds[i]['terminated'], TARGETS[policy][i])['correct']}


_supa = json.loads(SUPA.read_text(encoding='utf-8'))
SUPA_PREDS = {r['id']: {'text': r['source']['raw'], 'terminated': r['source']['terminated']} for r in _supa['rows']}
SUPA_REF = {p: set(v) for p, v in _supa['reference_ids_by_policy'].items()}
SUPA_CORRECT = {p: score(SUPA_PREDS, p) for p in POLS}

_bf = json.loads(OPALLO_BF16.read_text(encoding='utf-8'))
BF16_PREDS = {i: {'text': v['text'], 'terminated': v['terminated']} for i, v in _bf['predictions'].items()}
BF16_CORRECT = {p: score(BF16_PREDS, p) for p in POLS}
BF16_RECORDED = {p: set(_bf['by_policy'][p]['correct_ids']) for p in POLS}

_fp = json.loads(OPALLO_FP32.read_text(encoding='utf-8'))
_fpc = _fp['checkpoints'][-1]
FP32_PREDS = {i: {'text': v['text'], 'terminated': v['terminated']} for i, v in _fpc['outputs'].items()}
FP32_CORRECT = {p: score(FP32_PREDS, p) for p in POLS}
FP32_RECORDED_PERROW = {i for i, ok in _fpc['retention']['per_row'].items() if ok}
FP32_REG_POLICY = _fp['registered_bare_policy']
FP32_RECORDED_TOTAL = {p: _fpc['retention'][p]['correct_total'] for p in POLS}
FP32_RECORDED_LOST = {p: _fpc['retention'][p]['lost_total'] for p in POLS}
FP32_RECORDED_GAINED = {p: _fpc['retention'][p]['gained_total'] for p in POLS}


def identity_report():
    out = {}
    for p in POLS:
        out[p] = {
            'supa_rescored_n': len(SUPA_CORRECT[p]), 'supa_reference_n': len(SUPA_REF[p]),
            'supa_rescored_eq_reference_BY_IDENTITY': SUPA_CORRECT[p] == SUPA_REF[p],
            'bf16_rescored_n': len(BF16_CORRECT[p]), 'bf16_recorded_n': len(BF16_RECORDED[p]),
            'bf16_rescored_eq_recorded_BY_IDENTITY': BF16_CORRECT[p] == BF16_RECORDED[p],
            'fp32_rescored_n': len(FP32_CORRECT[p]), 'fp32_recorded_total': FP32_RECORDED_TOTAL[p],
            'fp32_rescored_lost': len(SUPA_REF[p] - FP32_CORRECT[p]), 'fp32_recorded_lost': FP32_RECORDED_LOST[p],
            'fp32_rescored_gained': len(FP32_CORRECT[p] - SUPA_REF[p]), 'fp32_recorded_gained': FP32_RECORDED_GAINED[p],
        }
    out[FP32_REG_POLICY]['fp32_rescored_eq_recorded_per_row_BY_IDENTITY'] = FP32_CORRECT[FP32_REG_POLICY] == FP32_RECORDED_PERROW
    return out


if __name__ == '__main__':
    print('retention.py', sha(RET_PY)[:12], 'probe', sha(PROBE)[:12], 'audit', sha(AUDIT)[:12])
    print('supa', sha(SUPA)[:12], 'bf16', sha(OPALLO_BF16)[:12], 'fp32', sha(OPALLO_FP32)[:12])
    rep = identity_report()
    print(json.dumps(rep, indent=1))
    Path(__file__).with_name('identity.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
