"""Preparation audit: CPU tensor/metadata fixtures; prohibit updates/native loads."""
import contextlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

record_root = Path(__file__).resolve().parent
source_root = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-distillation-dynamics-v1')
sys.path.insert(0, str(source_root))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONPYCACHEPREFIX'] = str(record_root / 'bytecode')
import torch
import pytest
from experiments.rineya_distillation_dynamics_v1 import native

counters = {'optimizer_updates': 0, 'native_model_loads': 0, 'pickle_loads': 0, 'cuda_initializations': 0}

def forbid(name):
    def denied(*args, **kwargs):
        counters[name] += 1
        raise AssertionError('Preparation forbids ' + name)
    return denied

for cls in (torch.optim.Optimizer, torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
    cls.step = forbid('optimizer_updates')
native.load_backend = forbid('native_model_loads')
torch.load = forbid('pickle_loads')
torch.cuda._lazy_init = forbid('cuda_initializations')

class PreparationScope:
    def __init__(self):
        self.deselected = []
    def pytest_collection_modifyitems(self, config, items):
        accepted, skipped = [], []
        for item in items:
            if item.originalname in {'test_all_factorial_cells_and_sft', 'test_qualification_restores_state'}:
                skipped.append(item)
                self.deselected.append(item.nodeid)
            else:
                accepted.append(item)
        items[:] = accepted
        config.hook.pytest_deselected(items=skipped)

scope = PreparationScope()
version = sys.argv[1] if len(sys.argv) > 1 else 'v1'
log_path = record_root / ('offline-tests-' + version + '.log')
with log_path.open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    exit_code = pytest.main([str(source_root / 'experiments/rineya_distillation_dynamics_v1/tests'),
        '-q', '--basetemp', str(record_root / ('tmp-' + version)),
        '-o', 'cache_dir=' + str(record_root / ('pytest-cache-' + version)),
        '--junitxml', str(record_root / ('offline-junit-' + version + '.xml'))], plugins=[scope])
payload = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code,
    'evidence_kind': 'MECHANICAL_FIXTURES_NO_NATIVE_CAPABILITY_EVIDENCE', 'runtime': sys.executable,
    'torch': torch.__version__, 'cuda_initialized': torch.cuda.is_initialized(), 'tripwire_attempts': counters,
    'optimizer_tests_deselected': scope.deselected, 'log': str(log_path),
    'confirmation_or_sealed_bank_reads': 0,
    'note': 'Only local fixture JSON/adapter tensors were read. Eleven update tests are deliberately excluded.'}
with (record_root / ('offline-verification-' + version + '.json')).open('x', encoding='utf-8') as output:
    json.dump(payload, output, indent=2)
print(json.dumps(payload, indent=2))
sys.exit(exit_code or int(any(counters.values())))
