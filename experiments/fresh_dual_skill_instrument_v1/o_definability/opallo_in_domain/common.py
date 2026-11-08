"""Shared read-only loaders for the Opallo O-endpoint feasibility analysis (attempt 3).

CPU python over recorded JSON/JSONL only. No model, no GPU, no generation.
The scorer is imported under a private module name from its file, hash-asserted.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
AB = ROOT / 'experiments' / 'architecture_benchmark_001'
EXPL = ROOT / 'experiments' / 'rab001_exploration'
RETENTION_PY = AB / 'retention.py'
AUDIT = AB / 'evidence' / 'retention-design-v1' / 'PROBE_LABEL_AUDIT.json'
PROBE = AB / 'data' / 'monitor_probe.jsonl'
SUPA_REF = AB / 'outputs' / 'retention-a329eb7b' / 'B28' / 'retention.json'
PANEL = ROOT / 'Rineya_Published_Cores' / 'Rineya-Opallo-1.0.0-devfreeze' / 'PANEL.json'
TAG = 'B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95'
ARM_DOSE = AB / 'evidence' / 'retention-design-v1' / f'ARM_DOSE_{TAG}.json'
ARM_RUN = AB / 'outputs' / 'arm-runs' / f'{TAG}_20260914T025105210772Z'
ARM_CKPT = AB / 'outputs' / 'arm-checkpoints' / TAG
SPL = AB / 'outputs' / 'splits_large'
TRAIN = SPL / 'train.jsonl'
DEV = SPL / 'dev.jsonl'
TOKENIZER = ROOT / 'Supa' / 'tokenizer.json'
TITAN_DIR = Path('C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/rrp015-20260917')
TBANK = Path('C:/Users/clclc/BusinessProject/ChatBubble-titan-model-soup-mini-v1/experiments/'
             'fresh_dual_skill_instrument_v1/T_BANK_V2.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]


def load_scorer():
    h = sha(RETENTION_PY)
    assert h.startswith('c52b6a48'), h
    spec = importlib.util.spec_from_file_location('_oend_a3_retention_private', RETENTION_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, h


def typed_targets(R, audit):
    """Targets built per row with retention.target_from_audit, cross-checked against
    build_overlay (which is how validate_published_core.py / retention_probe.py / arm_dose.py
    build them)."""
    out = {}
    for pol in R.BARE_POLICIES:
        t = {row['id']: R.target_from_audit(row, pol) for row in audit['rows']}
        ov = R.build_overlay(audit, pol)['targets']
        assert t == ov, f'target_from_audit != build_overlay under {pol}'
        out[pol] = t
    return out


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False), encoding='utf-8')
    return sha(p)[:16]
