"""Fail-closed provenance, operator permissions, resource guards and receipts."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import platform
import math
import psutil
import torch

SCHEMA = 'SBHM-001/v1'
ARMS = {'A': 'frozen', 'B': 'ordinary_recovery', 'C': 'local', 'D': 'hybrid'}
ROOT = Path(__file__).resolve().parent.parent
# Each native stage runs in its own process. Receipts can export these plain
# measurements without retaining CUDA tensors or changing allocation arithmetic.
CACHE_RELEASE_EVENTS = []


class Blocked(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def code_hash():
    return digest({str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'sbhm').glob('*.py'))})


def environment():
    versions = {}
    for name in ('torch', 'transformers', 'safetensors', 'psutil', 'tokenizers', 'accelerate'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'MISSING'
    return {'versions': versions, 'python': platform.python_version(), 'platform': platform.platform(),
            'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


def resources(device, runtime=False):
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    result = {'available_ram_bytes': vm.available, 'rss_bytes': rss}
    if vm.available < (6.0 if runtime else 7.4)*2**30:
        raise Blocked(f'free RAM below {6.0 if runtime else 7.4} GiB guard')
    if rss > 4*2**30:
        raise Blocked('worker RSS exceeds 4 GiB guard')
    if device != 'cuda:0' or not torch.cuda.is_available():
        raise Blocked('native pilot requires explicit cuda:0; no CPU fallback')
    free, total = torch.cuda.mem_get_info(0)
    if runtime and free < 1*2**30:
        reserved_before = torch.cuda.memory_reserved(0)
        allocated_before = torch.cuda.memory_allocated(0)
        if reserved_before > allocated_before:
            event = {'created_utc': now(), 'device': device,
                     'free_vram_before_bytes': free,
                     'reserved_before_bytes': reserved_before,
                     'allocated_before_bytes': allocated_before}
            # Only unused allocator segments can be released. Live parameters,
            # gradients, optimizer state and autograd activations remain intact.
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            event.update(free_vram_after_bytes=free,
                         reserved_after_bytes=torch.cuda.memory_reserved(0),
                         allocated_after_bytes=torch.cuda.memory_allocated(0),
                         recovered=free >= 1*2**30)
            CACHE_RELEASE_EVENTS.append(event)
    result.update(free_vram_bytes=free, total_vram_bytes=total)
    if free < 1*2**30:
        raise Blocked('less than 1 GiB free VRAM safety reserve (not a fit certificate)')
    return result


def safe_file(root, relative):
    root = Path(root).resolve()
    p = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise Blocked('absolute or escaping artifact path')
    # HF snapshot symlinks are allowed only if they resolve inside the locked root.
    if not p.is_file() or not p.resolve().is_relative_to(root):
        raise Blocked(f'missing or escaping artifact: {relative}')
    return p


def source_lock(config):
    source = config.get('source') or {}
    if not source.get('path') or not source.get('files'):
        raise Blocked('exact local source directory and file SHA256 map are required')
    root, expected = Path(source['path']), source['files']
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(expected) or not required <= actual or not any(x.endswith('.safetensors') for x in actual):
        raise Blocked('source inventory must exactly match locked files and include safe weights/tokenizer/config')
    if any(x.endswith(('.py', '.bin', '.pt', '.pth')) for x in actual):
        raise Blocked('custom code and pickle checkpoints are unsupported; no silent source conversion')
    for name, value in expected.items():
        if len(value) != 64 or sha(safe_file(root, name)) != value:
            raise Blocked(f'source hash mismatch: {name}')
    for name in actual:
        if name.endswith('.safetensors.index.json'):
            index = read_json(safe_file(root, name))
            for shard in index.get('weight_map', {}).values():
                if shard not in actual or not shard.endswith('.safetensors'):
                    raise Blocked('weight index references an unlocked shard')
                safe_file(root, shard)
    c = read_json(root/'config.json')
    if (c.get('model_type') != 'qwen3' or c.get('head_dim') is None or c.get('auto_map') or
            c.get('use_sliding_window', False) or c.get('attention_dropout', 0) != 0 or
            c.get('rope_scaling') is not None or c.get('quantization_config') is not None):
        raise Blocked('only explicit-head-dim, ungated standard Qwen3 with default RoPE is supported')
    if any('gate' in k.lower() for k in c):
        raise Blocked('gated/custom Rineya architectures require a separately qualified loader')
    if config['layer'] < 3 or config['layer'] >= c['num_hidden_layers']:
        raise Blocked('one valid layer beyond the protected first three is required')
    if config['limits']['max_prompt_tokens'] + config['limits']['max_new_tokens'] > c['max_position_embeddings']:
        raise Blocked('registered length exceeds source positional contract')
    return digest(expected)


def validate(config):
    if config.get('schema') != SCHEMA:
        raise Blocked('unknown schema')
    for name in ('seed', 'layer', 'window', 'rank'):
        if not isinstance(config.get(name), int) or isinstance(config[name], bool):
            raise Blocked(f'{name} must be an integer')
    if config['window'] < 1 or config['rank'] < 1 or config['dtype'] not in ('float32', 'bfloat16'):
        raise Blocked('invalid attention/precision registration')
    limits = config['limits']
    for k in ('max_prompt_tokens', 'max_new_tokens', 'max_train_tokens', 'steps', 'prefill_chunk'):
        if not isinstance(limits.get(k), int) or isinstance(limits[k], bool) or limits[k] < 1:
            raise Blocked(f'invalid {k}')
    if limits['max_train_tokens'] > limits['max_prompt_tokens'] or limits['max_prompt_tokens'] > 4096:
        raise Blocked('v1 is a bounded <=4096-token pilot, not a long-context fit certificate')
    if limits['steps'] > 1000 or not math.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise Blocked('invalid bounded training registration')


def bind(config, arm, delta=None):
    validate(config)
    if arm not in ARMS:
        raise Blocked('unknown arm')
    resources(config['device'])
    env = environment()
    if env['versions']['transformers'] != '4.51.3':
        raise Blocked('install the experiment-pinned transformers==4.51.3')
    binding = {'schema': SCHEMA, 'code_sha256': code_hash(), 'config_sha256': digest(config),
               'source_sha256': source_lock(config), 'arm': arm, 'environment': env,
               'delta_sha256': None if delta is None else sha(Path(delta)/'delta.safetensors')}
    binding['corpus_sha256'] = sha(config['corpus_manifest']) if config.get('corpus_manifest') else None
    binding['delta_manifest_sha256'] = sha(Path(delta)/'delta.json') if delta else None
    return binding


def authorize(config, action, binding, allow_model_load=False, allow_training=False):
    if not allow_model_load:
        raise Blocked('explicit --allow-model-load is required')
    if action in ('distill', 'recover') and not allow_training:
        raise Blocked('explicit --allow-training is required')
    path = config.get('grant')
    if not path:
        raise Blocked('a dated operator grant is required; implementation approval is not native-run approval')
    grant = read_json(path)
    now = datetime.now(timezone.utc)
    try:
        expiry = datetime.fromisoformat(grant['expires_utc'])
        issued = datetime.fromisoformat(grant['issued_utc'])
    except (KeyError, ValueError, TypeError) as e:
        raise Blocked('invalid dated grant') from e
    if expiry.tzinfo is None or issued.tzinfo is None or not issued <= now < expiry <= issued + timedelta(days=7):
        raise Blocked('grant expired, future dated, naive, or over seven days')
    if not grant.get('approved_by') or action not in grant.get('actions', []):
        raise Blocked('action not granted by a named operator')
    for key in ('code_sha256', 'source_sha256', 'config_sha256', 'corpus_sha256'):
        if grant.get(key) != binding[key]:
            raise Blocked(f'grant does not bind {key}')


def require_qualification(path, binding):
    if not path:
        raise Blocked('fresh exact-variant native qualification receipt required')
    receipt = read_json(path)
    if receipt.get('status') != 'PASS_NATIVE_MECHANICS_ONLY' or receipt.get('binding') != binding:
        raise Blocked('qualification belongs to another source/code/config/arm/delta/environment')
    age = datetime.now(timezone.utc) - datetime.fromisoformat(receipt['created_utc'])
    if not timedelta(0) <= age <= timedelta(hours=24):
        raise Blocked('qualification expired or future dated')


def now():
    return datetime.now(timezone.utc).isoformat()
