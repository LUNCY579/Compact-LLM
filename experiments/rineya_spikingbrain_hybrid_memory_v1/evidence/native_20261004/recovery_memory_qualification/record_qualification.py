"""Record completed pinned CPU recovery objective/gradient/update qualification."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

experiment = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(experiment))
from sbhm.protocol import code_hash

output = Path(__file__).resolve().parent
assert '13 passed' in (output/'pytest-targeted.txt').read_text(encoding='utf-8')
full = (output/'pytest-full.txt').read_text(encoding='utf-8')
assert '122 passed' in full and 'skipped' not in full
assert 'exit=0' in (output/'compileall.txt').read_text(encoding='utf-8')
assert 'No broken requirements found.' in (output/'pip-check.txt').read_text(encoding='utf-8')
package = Path(sys.prefix)/'Lib/site-packages/transformers'
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_PINNED_RANDOM_CPU_SUPERVISED_POSITION_LOSS_GRADIENT_UPDATE_PARITY',
    'python_executable': sys.executable,
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'code_sha256': code_hash(),
    'tests_sha256': {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((experiment/'tests').glob('*.py'))},
    'pinned_library_source_sha256': {
        name: sha256((package/name).read_bytes()).hexdigest()
        for name in ('models/qwen3/modeling_qwen3.py', 'loss/loss_utils.py')},
    'targeted_parity_tests_passed': 13, 'full_suite_passed': 122, 'full_suite_skipped': 0,
    'compileall_passed': True, 'pip_check_passed': True,
    'tiny_official_model': {'initialization': 'random', 'device': 'cpu', 'dtype': 'float32',
                            'layers': 5, 'selected_site': 3, 'hidden_size': 16,
                            'head_dim': 8, 'query_heads': 4, 'kv_heads': 2,
                            'vocab_size': 67, 'attention_implementation': 'sdpa'},
    'arms': ['B', 'C', 'D'],
    'label_masks': ['answer_and_eos', 'discontiguous', 'eos_only'],
    'checks': ['projection materializes exactly the supervised next-token positions',
               'selected logits equal corresponding full-head logits',
               'mean causal cross-entropy equals pinned full-logit labels path',
               'all trainable gradients match, with nonzero LoRA b factors',
               'D forget/mix/memory_scale receive finite nonzero matching gradients',
               'frozen parameters have no gradients',
               'clipped gradient norms and one AdamW update match the registered recipe',
               'multibatch, mismatched/rank-one labels and no-supervision block before forward'],
    'tolerances': {'logits_loss_norm': {'atol': 1e-6, 'rtol': 1e-6},
                   'gradients': {'atol': 2e-6, 'rtol': 2e-5},
                   'one_adamw_update': {'atol': 2e-7, 'rtol': 2e-6}},
    'equivalence_scope': 'Same mathematical objective and gradient; ordinary FP32 rounding permitted. This is not bitwise CUDA parity or a native memory-fit result.',
    'unchanged_contract': ['full teacher-forced input sequence/backbone', 'answer and EOS labels',
                           'dtype', 'resource guards', 'source', 'corpus', 'steps', 'optimizer'],
    'native_shape_one_vocab_buffer_estimate': {
        'full_259_tokens_bytes': 259*151936*4,
        'selected_max_14_labels_bytes': 14*151936*4,
        'bytes_saved': (259-14)*151936*4,
        'actual_peak_vram_savings_measured': False},
    'CUDA_VISIBLE_DEVICES': '-1', 'gpu_tensors_allocated': False,
    'pretrained_weights_loaded': False, 'native_training_or_fit_qualified': False,
    'requires_new_native_source_arm_delta_qualification_and_grant': True,
}
with (output/'qualification.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'passed': receipt['full_suite_passed'],
                  'code_sha256': receipt['code_sha256']}, indent=2))
