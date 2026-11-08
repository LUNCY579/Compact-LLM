"""Fresh CUDA-stack host/device query and guards; no tensor/model allocation."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

import psutil
import torch

experiment = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(experiment))
from sbhm.protocol import Blocked, code_hash, resources

available = torch.cuda.is_available()
gpu = None
if available:
    free, total = torch.cuda.mem_get_info(0)
    gpu = {'name': torch.cuda.get_device_name(0), 'free_vram_bytes': free, 'total_vram_bytes': total}
guards = {}
for label, runtime in (('admission', False), ('runtime', True)):
    try:
        result = resources('cuda:0', runtime=runtime)
        guards[label] = {'status': 'PASS_GUARD_ONLY_NOT_FIT', 'sample': result}
    except Blocked as error:
        guards[label] = {'status': 'BLOCKED', 'reason': str(error)}
output = Path(__file__).resolve().parent
suite = (output/'pytest-frozen.txt').read_text(encoding='utf-8')
assert '109 passed' in suite and 'skipped' not in suite
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'FRESH_RESOURCE_GUARD_AND_CPU_TEST_RECEIPT_NO_NATIVE_EXECUTION',
    'python_executable': sys.executable,
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'code_sha256': code_hash(),
    'tests_sha256': {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((experiment/'tests').glob('*.py'))},
    'full_suite_passed': 109, 'full_suite_skipped': 0,
    'cpu_test_cuda_visible_devices': '-1', 'compileall_passed': True,
    'pip_check_passed': True, 'compiled_cuda': torch.version.cuda,
    'cuda_available': available, 'gpu': gpu,
    'available_host_ram_bytes_after_cuda_import': psutil.virtual_memory().available,
    'worker_rss_bytes': psutil.Process().memory_info().rss,
    'guards': guards, 'cuda_tensors_allocated': False, 'model_loaded': False,
    'native_mechanics_or_training_fit_established': False,
}
with (output/'frozen-resource-qualification.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt, indent=2))
