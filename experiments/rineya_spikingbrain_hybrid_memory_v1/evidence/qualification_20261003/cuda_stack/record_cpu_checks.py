"""Bind completed CPU-only checks to the isolated CUDA-capable stack."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
experiment = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(experiment))
from sbhm.protocol import code_hash

output = Path(__file__).resolve().parent
tests = (output/'pytest-full.txt').read_text(encoding='utf-8')
integration = (output/'pytest-integration.txt').read_text(encoding='utf-8')
match = re.search(r'(\d+) passed in ([\d.]+)s', tests)
assert match and 'skipped' not in tests and 'failed' not in tests
assert '1 passed, 56 deselected' in integration
assert 'exit=0' in (output/'compileall.txt').read_text(encoding='utf-8')
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_CPU_MECHANICS_IN_CUDA_STACK_NOT_NATIVE_FIT',
    'python_executable': sys.executable,
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'code_sha256_at_receipt': code_hash(),
    'tests_sha256': {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((experiment/'tests').glob('*.py'))},
    'full_suite_passed': int(match.group(1)), 'full_suite_skipped': 0,
    'full_suite_reported_seconds': float(match.group(2)),
    'official_library_random_cpu_integration_passed': True,
    'compileall_passed': True,
    'execution_environment': {'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1',
                              'TRANSFORMERS_OFFLINE': '1'},
    'model_allocation_device': 'cpu', 'pretrained_weights_loaded': False,
    'cuda_tensors_allocated': False, 'native_inference_or_training': False,
    'native_fit_established': False,
}
with (output/'cpu-qualification.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'passed': receipt['full_suite_passed'],
                  'code_sha256': receipt['code_sha256_at_receipt']}, indent=2))
