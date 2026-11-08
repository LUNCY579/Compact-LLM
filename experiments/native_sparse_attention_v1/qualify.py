"""Run isolated CPU mechanical tests and write a fresh, hash-bound receipt."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

from .budget import parameter_budget, access_budget
from .core import NSAConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    # Exclusive destination; never overwrite an earlier evidence packet.
    output.mkdir(parents=True, exist_ok=False)
    junit = output / 'junit.xml'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    command = [sys.executable, '-m', 'pytest', str(here / 'tests'), '-q', f'--junitxml={junit}']
    result = subprocess.run(command, cwd=here.parents[1], env=env, capture_output=True, text=True, timeout=120)
    (output / 'tests.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    cases = ET.parse(junit).getroot().findall('.//testcase') if junit.exists() else []
    skipped = sum(x.find('skipped') is not None for x in cases)
    failed = sum(x.find('failure') is not None or x.find('error') is not None for x in cases)
    config = json.loads((here / 'configs' / 'supa_scout.json').read_text(encoding='utf-8'))
    nsa = NSAConfig(**config['nsa'])
    budgets = {}
    for mode in ('independent', 'shared'):
        for layers in (4, 28):
            budgets[f'{mode}_{layers}_layers'] = parameter_budget(config['model_geometry'], replace(nsa, kv_mode=mode), layers)
    bound = [p for p in here.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and 'evidence' not in p.relative_to(here).parts and p.suffix in {'.py', '.json', '.md'}]
    receipt = {
        'experiment': 'NSA-QUAL-001', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': config['base_commit'], 'branch': 'research/native-sparse-attention-v1',
        'scope': 'Synthetic CPU mathematics and Qwen3-shaped fixtures, not trained Rineya evidence',
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
        'test_returncode': result.returncode, 'tests_collected': len(cases),
        'passed': len(cases)-skipped-failed, 'skipped': skipped, 'failed': failed,
        'skip_reasons': [x.find('skipped').attrib.get('message') for x in cases if x.find('skipped') is not None],
        'states': {'mechanical_tests': 'PASS' if result.returncode == 0 and cases else 'FAIL',
                   'native_checkpoint': 'NOT_RUN', 'training': 'NOT_RUN', 'quality': 'NOT_RUN',
                   'gpu_speed': 'NOT_RUN', 'fused_kernel': 'NOT_IMPLEMENTED', 'cached_nsa_decode': 'NOT_IMPLEMENTED'},
        'parameter_budgets': budgets, 'access_budget_4096': access_budget(4096, nsa),
        'sha256': {str(p.relative_to(here)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bound)},
        'test_log_sha256': hashlib.sha256((output/'tests.txt').read_bytes()).hexdigest(),
        'junit_sha256': hashlib.sha256(junit.read_bytes()).hexdigest() if junit.exists() else None,
        'limitations': ['Chunk scratch is not peak autograd memory', 'No model downloads or optimizer steps',
                        'Optional tiny random Transformers test is not published-checkpoint qualification']}
    (output / 'QUALIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('passed', 'skipped', 'failed', 'states')}, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
