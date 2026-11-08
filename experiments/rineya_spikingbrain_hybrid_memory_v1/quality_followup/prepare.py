"""Offline source verification and one exact, unexecuted native proposal."""
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import math

from . import SCHEMA, RECIPE, REGISTRATION
from .utils import (Blocked, require, sha, digest, read_json, write_json,
                    external_path, code_inventory, PACKAGE_ROOT, now)

SOURCE_DIGEST = 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a'
WEIGHT_SHA = 'a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567'

def verify_source(source):
    root = Path(source['path']).resolve()
    expected = source['files']
    require(digest(expected) == SOURCE_DIGEST and len(expected) == 17,
            'only the audited exact Titan-Transition source is registered')
    files = {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()}
    require(set(files) == set(expected), 'exact source inventory changed')
    for name, path in files.items():
        require(path.resolve().is_relative_to(root), 'escaping source link')
        require(sha(path) == expected[name], 'source hash mismatch: ' + name)
    require(expected['model.safetensors'] == WEIGHT_SHA, 'source weight identity differs')
    config = read_json(root / 'config.json')
    geometry = {'hidden_size': 1024, 'head_dim': 128, 'num_attention_heads': 16,
                'num_key_value_heads': 8, 'num_hidden_layers': 28, 'vocab_size': 151936}
    require(all(config.get(k) == v for k, v in geometry.items()), 'exact source geometry differs')
    require(config.get('model_type') == 'qwen3' and not config.get('auto_map')
            and not config.get('quantization_config') and not config.get('rope_scaling')
            and not config.get('use_sliding_window') and config.get('tie_word_embeddings'),
            'unsupported source architecture')
    path = root / 'model.safetensors'
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        require(size < 10 * 2**20, 'unexpected source tensor header')
        tensors = json.loads(stream.read(size))
    tensors.pop('__metadata__', None)
    require(len(tensors) == 310 and all(t['dtype'] == 'F32' for t in tensors.values()),
            'source tensor inventory/dtype differs')
    required = {'model.embed_tokens.weight': [151936, 1024], 'model.norm.weight': [1024]}
    for layer in range(28):
        shapes = {'self_attn.q_proj.weight': [2048, 1024], 'self_attn.k_proj.weight': [1024, 1024],
                  'self_attn.v_proj.weight': [1024, 1024], 'self_attn.o_proj.weight': [1024, 2048],
                  'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128],
                  'mlp.gate_proj.weight': [3072, 1024], 'mlp.up_proj.weight': [3072, 1024],
                  'mlp.down_proj.weight': [1024, 3072], 'input_layernorm.weight': [1024],
                  'post_attention_layernorm.weight': [1024]}
        required.update({f'model.layers.{layer}.{k}': v for k, v in shapes.items()})
    require(set(tensors) == set(required) and all(tensors[k]['shape'] == v for k, v in required.items()),
            'source projections, QK norms or backbone tensors differ')
    spans = sorted(t['data_offsets'] for t in tensors.values())
    require(spans[0][0] == 0 and spans[-1][1] + size + 8 == path.stat().st_size
            and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'source tensor offsets invalid')
    require(all(math.prod(t['shape']) * 4 == t['data_offsets'][1] - t['data_offsets'][0]
                for t in tensors.values()), 'tensor byte extents differ')
    return {'status': 'PASS_EXACT_SOURCE_HASH_AND_HEADER_ONLY', 'created_utc': now(),
            'source_sha256': SOURCE_DIGEST, 'weight_sha256': WEIGHT_SHA, 'files': 17,
            'tensors': 310, 'parameters': sum(math.prod(t['shape']) for t in tensors.values()),
            'geometry': geometry, 'model_loaded': False,
            'published_confirmation_named_metadata': 'inventory bytes hashed only; no bank contents parsed'}

def native_environment(python):
    script = "import importlib.metadata as m,platform,json; print(json.dumps({'versions':{n:m.version(n) for n in ['torch','transformers','safetensors','psutil','tokenizers','accelerate']},'python':platform.python_version(),'platform':platform.platform(),'cuda':'12.4','device':'NVIDIA GeForce RTX 4070 Laptop GPU'}))"
    result = subprocess.run([str(python), '-B', '-c', script], check=True, capture_output=True, text=True)
    environment = json.loads(result.stdout)
    require(environment['versions']['torch'] == '2.6.0+cu124'
            and environment['versions']['transformers'] == '4.51.3'
            and environment['versions']['tokenizers'] == '0.21.4', 'native environment pins changed')
    return environment

def prepare_study(packet, source_config, review, output, owner_thread_id):
    from .corpus import load_packet
    from .runner import build_plan
    output = external_path(output)
    require(not output.exists(), 'refusing to overwrite a proposal directory')
    packet = external_path(packet)
    load_packet(packet)
    source = read_json(source_config)['source']  # Deliberately import no grant or old delta.
    source_receipt = verify_source(source)
    review = external_path(review)
    reviewed = read_json(review)
    require(reviewed.get('schema') == SCHEMA + '/corpus-review'
            and reviewed.get('reviewed') is True and reviewed.get('status') == 'PASS_OFFLINE_AUDIT',
            'independent corpus review is required')
    require(reviewed['corpus_sha256'] == sha(packet / 'manifest.json')
            and reviewed['ledger_sha256'] == sha(packet / 'exposure_ledger.json')
            and reviewed['code_sha256'] == digest(code_inventory()), 'corpus review binding changed')
    audit_path = external_path(reviewed.get('audit_path', ''))
    require(audit_path.is_file() and sha(audit_path) == reviewed.get('audit_sha256'),
            'independent audit bytes changed or are missing')
    audit = read_json(audit_path)
    require(audit.get('schema') == SCHEMA + '/corpus-audit'
            and audit.get('status') == 'PASS_OFFLINE_AUDIT'
            and audit.get('manifest_sha256') == reviewed['corpus_sha256']
            and audit.get('full_tokenizer_replay') is True
            and audit.get('counts') == {'control': 512, 'candidate': 512, 'dev': 672}
            and audit.get('ledger_steps') == 512, 'full independent tokenizer/oracle audit required')
    manifest = read_json(packet / 'manifest.json')
    expected_tokenizer = {name: source['files'][name] for name in
                          ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    require(manifest['tokenizer']['files'] == expected_tokenizer, 'corpus tokenizer differs from exact source')
    native_python = PACKAGE_ROOT / '.venv-cuda' / 'Scripts' / 'python.exe'
    require(native_python.is_file(), 'existing pinned isolated native Python missing')
    config = {'schema': SCHEMA, 'study_id': 'update-order-v1', 'source': source,
              'recipe': RECIPE, 'registration': REGISTRATION, 'corpus_packet': str(packet),
              'review': str(review), 'native_run_root': str(output / 'native_run'),
              'native_python': str(native_python), 'grant': str(output / 'native-grant.json'),
              'ownership': str(output / 'ownership-native.json'), 'owner_thread_id': owner_thread_id,
              'environment': native_environment(native_python)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source-lock-audit.json', source_receipt)
    write_json(output / 'study.json', config)
    proposal = build_plan(output / 'study.json')
    write_json(output / 'proposal.json', proposal)
    write_json(output / 'preparation.json', {'status': 'OFFLINE_PREPARED_NATIVE_NOT_AUTHORIZED',
              'created_utc': now(), 'code_sha256': digest(code_inventory()),
              'config_sha256': digest(config), 'corpus_sha256': sha(packet / 'manifest.json'),
              'proposal_file_sha256': sha(output / 'proposal.json'),
              'proposal_sha256': proposal['proposal_sha256'], 'model_loaded': False,
              'native_execution': False, 'grant_created': False,
              'gpu_identity': 'historical expected identity; fresh runtime check required after authorization'})
    return proposal
