"""Record final CPU mechanics coverage with an interior converted layer."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
experiment = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(experiment))
from sbhm.protocol import code_hash

output = Path(__file__).resolve().parent
targeted = (output/'pytest-middle-site-integration.txt').read_text(encoding='utf-8')
full = (output/'pytest-middle-site-final.txt').read_text(encoding='utf-8')
assert '1 passed, 56 deselected' in targeted
assert '109 passed' in full and 'skipped' not in full
assert 'head_dim=8,num_hidden_layers=5' in (experiment/'tests/test_experiment.py').read_text(encoding='utf-8')
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_FINAL_PINNED_RANDOM_CPU_MIDDLE_SITE_MECHANICS',
    'python_executable': sys.executable,
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'code_sha256': code_hash(),
    'tests_sha256': {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((experiment/'tests').glob('*.py'))},
    'targeted_integration_passed': 1, 'full_suite_passed': 109, 'full_suite_skipped': 0,
    'tiny_official_qwen3': {'random_initialization': True, 'device': 'cpu', 'dtype': 'float32',
                           'layers': 5, 'converted_site': 3, 'subsequent_standard_layer': 4,
                           'explicit_head_dim': 8, 'hidden_size_div_heads': 4,
                           'attention_implementation': 'sdpa', 'arms': ['A', 'B', 'C', 'D']},
    'checks': ['same-source A/B zero initialization and identity wrapper exact logits',
               'full/tokenwise/chunked prefix and greedy-token agreement',
               'fresh request independence', 'no full-history KV at converted site',
               'cache handling with a standard attention layer after converted site',
               'zero-mix hybrid equals local-only exactly'],
    'CUDA_VISIBLE_DEVICES': '-1', 'gpu_tensors_allocated': False,
    'pretrained_weights_loaded': False, 'native_fit_or_behavior_result': False,
    'prior_resource_receipt': 'frozen-resource-qualification.json',
}
with (output/'middle-site-final.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'passed': receipt['full_suite_passed'],
                  'code_sha256': receipt['code_sha256']}, indent=2))
