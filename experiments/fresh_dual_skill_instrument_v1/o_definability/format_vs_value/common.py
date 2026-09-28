"""Shared loaders (attempt 2, written fresh). READ-ONLY over recorded JSON. No model, no GPU."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
AB = ROOT / 'experiments' / 'architecture_benchmark_001'
RETENTION_PY = AB / 'retention.py'
AUDIT = AB / 'evidence' / 'retention-design-v1' / 'PROBE_LABEL_AUDIT.json'
PROBE = AB / 'data' / 'monitor_probe.jsonl'
SUPA = AB / 'outputs' / 'retention-a329eb7b' / 'B28' / 'retention.json'
PANEL = ROOT / 'Rineya_Published_Cores' / 'Rineya-Opallo-1.0.0-devfreeze' / 'PANEL.json'
FP32 = (AB / 'evidence' / 'retention-design-v1'
        / 'ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json')
OUT = Path(__file__).parent
CLASSES = ('percentage_reasoning', 'ratio_proportion', 'time_duration')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_scorer():
    h = sha(RETENTION_PY)
    assert h.startswith('c52b6a48'), h
    spec = importlib.util.spec_from_file_location('_fvv2_retention_private_mod', RETENTION_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, h


def load_all():
    R, rh = load_scorer()
    hashes = {'retention.py': rh}
    for name, p in [('audit', AUDIT), ('probe', PROBE), ('supa', SUPA), ('panel', PANEL),
                    ('fp32', FP32)]:
        hashes[name] = sha(p)
    assert hashes['probe'].startswith('5f6df1c1'), hashes['probe']
    assert hashes['audit'].startswith('4857cf86'), hashes['audit']
    probe = [json.loads(l) for l in PROBE.read_bytes().decode('utf-8').splitlines() if l.strip()]
    assert len(probe) == 150
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    overlays = {n: R.build_overlay(audit, n) for n in R.BARE_POLICIES}
    for n in R.BARE_POLICIES:           # named primitive must agree with the overlay row by row
        for row in audit['rows']:
            assert R.target_from_audit(row, n) == overlays[n]['targets'][row['id']]
    targets = {n: overlays[n]['targets'] for n in R.BARE_POLICIES}
    supa = json.loads(SUPA.read_text(encoding='utf-8'))
    panel = json.loads(PANEL.read_text(encoding='utf-8'))
    fp32 = json.loads(FP32.read_text(encoding='utf-8'))
    assert fp32['completed'] is True
    fa = panel['cast_parity']['float32_artifact'].replace(chr(92), '/')
    assert fa.endswith(FP32.name), fa
    arms = {
        'supa': {r['id']: {'text': r['source']['raw'], 'terminated': r['source']['terminated'],
                           'new_tokens': r['source'].get('new_tokens')} for r in supa['rows']},
        'opallo_bf16': {i: {'text': v['text'], 'terminated': v['terminated']}
                        for i, v in panel['predictions'].items()},
        'opallo_fp32': {i: {'text': v['text'], 'terminated': v['terminated']}
                        for i, v in fp32['checkpoints'][-1]['outputs'].items()},
    }
    ids = [r['id'] for r in probe]
    assert len(set(ids)) == 150
    for a in arms.values():
        assert set(a) == set(ids)
    return R, hashes, probe, audit, targets, supa, panel, fp32, arms


def score_all(R, arms, targets):
    scored = {}
    for arm, preds in arms.items():
        for pol in R.BARE_POLICIES:
            scored[(arm, pol)] = {i: R.score_prediction(p['text'], p['terminated'],
                                                        targets[pol][i])
                                  for i, p in preds.items()}
    correct = {k: {i for i, r in v.items() if r['correct']} for k, v in scored.items()}
    return scored, correct


def sk(i):
    """Unique short key: suffixes 02944 and 00538 each occur in two classes."""
    parts = i.split('__')
    return parts[1][:4] + '_' + parts[-1]
