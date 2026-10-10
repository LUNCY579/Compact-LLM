"""Reproduce dependency and local config/tokenizer checks; never load weights."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import psutil
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sbhm.data import encode


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


output = Path(__file__).resolve().parent
source = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
              'Rineya-Titan-Transition-1.0.0-devfreeze')
created = datetime.now(timezone.utc).isoformat()
packages = ('torch', 'transformers', 'tokenizers', 'safetensors', 'psutil',
            'pytest', 'accelerate', 'numpy', 'huggingface-hub', 'pip')
environment = {
    'created_utc': created,
    'python_executable': sys.executable,
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'versions': {name: version(name) for name in packages},
    'compiled_cuda': torch.version.cuda,
    'available_host_ram_bytes': psutil.virtual_memory().available,
    'process_rss_bytes': psutil.Process().memory_info().rss,
    'scope': 'isolated experiment-local CPU-only environment; shared stacks unchanged',
    'native_cuda_qualified': False,
    'native_training_fit_established': False,
}
assert torch.version.cuda is None
assert version('transformers') == '4.51.3'
(output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n', encoding='utf-8')

# These calls read local JSON/tokenizer assets, not model tensors or source code.
raw = json.loads((source/'config.json').read_text(encoding='utf-8'))
config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
geometry_keys = ('model_type', 'hidden_size', 'head_dim', 'num_attention_heads',
                 'num_key_value_heads', 'num_hidden_layers', 'vocab_size',
                 'rope_theta', 'rms_norm_eps', 'max_position_embeddings',
                 'eos_token_id', 'tie_word_embeddings')
geometry = {key: getattr(config, key) for key in geometry_keys}
assert all(geometry[key] == raw[key] for key in geometry_keys)
assert geometry['head_dim'] == 128
assert geometry['head_dim'] != geometry['hidden_size']//geometry['num_attention_heads']
assert len(tokenizer) <= config.vocab_size
assert tokenizer.chat_template
assert tokenizer.eos_token_id == config.eos_token_id
row = {'id': 'local-tokenizer-smoke', 'family_id': 'unreviewed-engineering-fixture',
       'prompt': 'Register TEST_A is 17. Return only TEST_A.', 'target': '17'}
prompt_ids, _ = encode(tokenizer, row, training=False)
ids, labels = encode(tokenizer, row, training=True)
prompt_length = prompt_ids.shape[1]
assert torch.equal(ids[:, :prompt_length], prompt_ids)
assert (labels[:, :prompt_length] == -100).all()
assert labels[0, -1].item() == tokenizer.eos_token_id
assert labels[0, prompt_length:].tolist() == ids[0, prompt_length:].tolist()
receipt = {
    'created_utc': created,
    'status': 'PASS_LOCAL_CONFIG_TOKENIZER_ONLY',
    'source_path': str(source),
    'transformers_version': version('transformers'),
    'tokenizer_class': tokenizer.__class__.__name__,
    'tokenizer_length': len(tokenizer),
    'tokenizer_base_vocab_size': tokenizer.vocab_size,
    'geometry': geometry,
    'derived_hidden_size_per_head_for_contrast_only': config.hidden_size//config.num_attention_heads,
    'tokenizer_eos_token_id': tokenizer.eos_token_id,
    'tokenizer_eos_token': tokenizer.eos_token,
    'chat_template_sha256': sha256(tokenizer.chat_template.encode()).hexdigest(),
    'input_files_sha256': {name: sha(source/name) for name in
                          ('config.json', 'tokenizer_config.json', 'tokenizer.json')},
    'sample': {'row': row, 'prompt_tokens': prompt_length,
               'answer_plus_eos_tokens': ids.shape[1]-prompt_length,
               'training_input_ids': ids[0].tolist(),
               'training_labels': labels[0].tolist()},
    'weights_loaded': False,
    'gpu_execution': False,
    'model_download': False,
    'corpus_review_or_behavior_evidence': False,
    'complete_source_lock_or_native_qualification': False,
}
(output/'source-tokenizer-smoke.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'environment': environment, 'source_smoke_status': receipt['status'],
                  'source_path': receipt['source_path'], 'geometry': geometry,
                  'tokenizer_length': len(tokenizer), 'prompt_tokens': prompt_length,
                  'answer_plus_eos_tokens': ids.shape[1]-prompt_length}, indent=2))
