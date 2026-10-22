"""Offline provenance and a separate, fail-closed native permission boundary.

Importing this module neither imports torch nor queries a GPU. The existing
SBHM grant cannot authorize this study. Native environment/resource checks are
performed only after the separately reviewed, proposal-bound grant passes.
"""
from datetime import datetime, timedelta, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import sys

from . import CONDITIONS, RECIPE, REGISTRATION, SCHEMA
from .utils import (Blocked, PACKAGE_ROOT, REPO_ROOT, code_inventory, digest,
                    external_path, read_json, require, sha)

TITAN_SOURCE_SHA256 = 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a'
NATIVE_VERSIONS = {'torch': '2.6.0+cu124', 'transformers': '4.51.3',
                   'safetensors': '0.7.0', 'psutil': '7.2.2',
                   'tokenizers': '0.21.4', 'accelerate': '1.15.0'}
STABLE_KEYS = ('schema', 'study_id', 'code_sha256', 'config_sha256',
               'source_sha256', 'corpus_sha256', 'ledger_sha256',
               'tokenizer_sha256', 'environment')
GRANT_BINDINGS = tuple(key for key in STABLE_KEYS if key not in ('schema', 'study_id'))


def dated(value, label):
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise Blocked(f'invalid timezone-aware {label}') from exc
    require(result.tzinfo is not None, f'{label} requires a timezone')
    return result.astimezone(timezone.utc)


def safe_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'escaping relative source file')
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root),
            f'missing or escaping source file: {relative}')
    return path


def source_lock(config):
    """Verify only the registered source inventory; never discover data banks."""
    source = config.get('source', {})
    require(isinstance(source, dict) and source.get('path') and source.get('files'),
            'exact frozen Titan local source and complete hashes required')
    root, expected = Path(source['path']).resolve(), source['files']
    require(isinstance(expected, dict) and digest(expected) == TITAN_SOURCE_SHA256,
            'source is not the exact frozen Titan-Transition inventory')
    actual = {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file()}
    require(set(actual) == set(expected), 'source file inventory changed')
    for name, value in expected.items():
        require(isinstance(value, str) and len(value) == 64 and
                sha(safe_file(root, name)) == value, f'source hash changed: {name}')
    require(not any(name.endswith(('.py', '.bin', '.pt', '.pth')) for name in actual),
            'custom code and pickle sources are unsupported')
    shape = read_json(root / 'config.json')
    require(shape.get('model_type') == 'qwen3' and shape.get('head_dim') == 128 and
            shape.get('num_hidden_layers') == 28 and shape.get('hidden_size') == 1024 and
            shape.get('num_attention_heads') == 16 and shape.get('num_key_value_heads') == 8 and
            not shape.get('auto_map') and not shape.get('use_sliding_window', False) and
            shape.get('attention_dropout', 0) == 0 and shape.get('rope_scaling') is None and
            shape.get('quantization_config') is None and
            not any('gate' in key.lower() for key in shape),
            'frozen standard Qwen3 source contract changed')
    require(RECIPE['qualification_tokens'] <= shape['max_position_embeddings'],
            'qualification exceeds source position contract')
    tokenizer_files = {name: value for name, value in expected.items()
                       if Path(name).name in {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                                             'vocab.json', 'merges.txt'}}
    require(len(tokenizer_files) == 5, 'exact tokenizer/config inventory required')
    return digest(expected), digest(tokenizer_files)


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('study_id') == 'update-order-v1',
            'unknown quality study; SBHM permits/configs cannot be imported')
    require(config.get('recipe') == RECIPE, 'the registered recipe is immutable')
    require(config.get('registration') == REGISTRATION, 'measurement registration changed')
    require(isinstance(config.get('environment'), dict) and config['environment'],
            'exact native environment must be registered')
    require(config['environment'].get('versions') == NATIVE_VERSIONS and
            config['environment'].get('cuda') == '12.4' and
            config['environment'].get('device') == 'NVIDIA GeForce RTX 4070 Laptop GPU',
            'the registered native dependency/CUDA/device pins cannot be changed')
    require(isinstance(config.get('owner_thread_id'), str) and config['owner_thread_id'],
            'owner chat identity required')
    for key in ('corpus_packet', 'review', 'native_run_root', 'grant', 'ownership'):
        require(config.get(key), f'missing {key}')
        external_path(config[key])
    require(config.get('native_python') and Path(config['native_python']).is_file(),
            'the exact isolated native interpreter must exist')
    return config


def load_config(path):
    external_path(path)
    return validate_config(read_json(path))


def bind(config, condition, delta=None):
    validate_config(config)
    require(condition in CONDITIONS, 'unknown condition')
    require(condition != 'anchor' or delta is None, 'anchor must remain frozen')
    source_hash, tokenizer_hash = source_lock(config)
    packet = external_path(config['corpus_packet'])
    require((packet / 'manifest.json').is_file() and (packet / 'exposure_ledger.json').is_file(),
            'a frozen corpus packet and exposure ledger are required')
    binding = {'schema': SCHEMA, 'study_id': config['study_id'], 'condition': condition,
               'arm': 'A' if condition == 'anchor' else 'B',
               'code_sha256': digest(code_inventory()), 'config_sha256': digest(config),
               'source_sha256': source_hash, 'tokenizer_sha256': tokenizer_hash,
               'corpus_sha256': sha(packet / 'manifest.json'),
               'ledger_sha256': sha(packet / 'exposure_ledger.json'),
               'environment': config['environment'], 'delta_sha256': None,
               'delta_manifest_sha256': None}
    if delta is not None:
        path = external_path(delta)
        require(path == external_path(config['native_run_root']) / f'{condition}-train',
                'only this condition\'s exact proposed adaptation may be loaded')
        manifest = read_json(path / 'delta.json')
        prior = manifest.get('binding', {})
        require(prior == binding, 'delta is not bound to this exact condition/study/source')
        require(manifest.get('sha256') == sha(path / 'delta.safetensors'), 'delta bytes changed')
        record = manifest.get('record', {})
        require((path / 'receipt.json').is_file() and read_json(path / 'receipt.json') == record,
                'delta requires the exact completed training receipt artifact')
        require_training_record(config, record, binding)
        binding['delta_sha256'] = manifest['sha256']
        binding['delta_manifest_sha256'] = sha(path / 'delta.json')
    return binding


def require_training_record(config, record, binding):
    """Validate completed actual exposure before loading any adaptation tensor."""
    from .corpus import load_packet
    require(record.get('status') == 'TRAINING_COMPLETED_NOT_EFFICACY' and
            record.get('stage') == 'train' and record.get('binding') == binding and
            len(record.get('records', [])) == RECIPE['steps'],
            'delta requires the complete, bound 512-step quality training receipt')
    packet = load_packet(config['corpus_packet'])
    rows = {row['id']: row for row in packet[binding['condition']]}
    require(len(packet['ledger']) == 512, 'delta training ledger incomplete')
    for step, (runtime, slot) in enumerate(zip(record['records'], packet['ledger'])):
        require(isinstance(runtime, dict) and runtime.get('step') == step == slot['step'] and
                runtime.get('slot_id') == runtime.get('row_id') == slot['slot_id'] and
                runtime['row_id'] in rows and digest(rows[runtime['row_id']]) ==
                slot[f'{binding["condition"]}_row_sha256'], 'delta runtime row/exposure mismatch')
        for field in ('input_tokens', 'supervised_positions'):
            require(type(runtime.get(field)) is int and runtime[field] == slot[field] and runtime[field] > 0,
                    'delta actual token/label geometry differs from ledger')
        for actual, expected in (('input_token_sha256', 'input_sha256'),
                                 ('label_token_sha256', 'labels_sha256')):
            require(runtime.get(actual) == slot[expected][binding['condition']],
                    'delta actual runtime token/label hash differs from ledger')
        for field in ('loss', 'gradient_norm'):
            require(type(runtime.get(field)) in (int, float) and math.isfinite(runtime[field]) and
                    runtime[field] >= 0, 'delta training loss/gradient record is invalid')


def require_review(config, binding):
    path = external_path(config['review'])
    require(path.is_file(), 'independent frozen-corpus review is still pending')
    review = read_json(path)
    require(review.get('schema') == SCHEMA + '/corpus-review' and
            review.get('study_id') == config['study_id'] and review.get('reviewed') is True and
            review.get('status') == 'PASS_OFFLINE_AUDIT' and review.get('reviewed_by'),
            'independent passing corpus review required')
    for key in ('corpus_sha256', 'ledger_sha256', 'code_sha256'):
        require(review.get(key) == binding[key], f'corpus review does not bind {key}')
    require(isinstance(review.get('audit_sha256'), str) and len(review['audit_sha256']) == 64,
            'independent audit hash required')
    require(review.get('audit_path'), 'independent tokenizer-replayed audit artifact required')
    audit_path = external_path(review['audit_path'])
    require(audit_path.is_file() and sha(audit_path) == review['audit_sha256'],
            'independent audit bytes changed or are missing')
    audit = read_json(audit_path)
    require(audit.get('schema') == SCHEMA + '/corpus-audit' and
            audit.get('status') == 'PASS_OFFLINE_AUDIT' and
            audit.get('manifest_sha256') == binding['corpus_sha256'] and
            audit.get('full_tokenizer_replay') is True and
            audit.get('counts') == {'control': 512, 'candidate': 512, 'dev': 672} and
            audit.get('ledger_steps') == 512,
            'review needs a passing, exact-corpus, fully tokenizer-replayed offline audit')
    return review


def authorize(config, stage, binding, proposal, allow_model_load=False, allow_training=False):
    """Pure permission check: do this before GPU queries, imports, or allocation."""
    require(allow_model_load, 'separate native authorization and --allow-model-load required')
    require(stage in ('qualify', 'train', 'evaluate'), 'unknown native stage')
    if stage == 'train':
        require(allow_training, 'separate training authorization and --allow-training required')
        require(binding['condition'] != 'anchor', 'anchor is never trained')
    grant_path = external_path(config['grant'])
    require(grant_path.is_file(), 'separate dated quality native grant is missing')
    grant = read_json(grant_path)
    require(grant.get('schema') == SCHEMA + '/native-grant' and
            grant.get('study_id') == config['study_id'],
            'this study needs its own native grant; historical SBHM permission is invalid')
    now = datetime.now(timezone.utc)
    issued, expires = dated(grant.get('issued_utc'), 'issued_utc'), dated(grant.get('expires_utc'), 'expires_utc')
    require(issued <= now < expires <= issued + timedelta(days=7),
            'grant expired, future-dated, or exceeds seven days')
    require(isinstance(grant.get('approved_by'), str) and grant['approved_by'].strip(),
            'a separately authorizing named operator is required')
    require(grant.get('proposal_sha256') == proposal['proposal_sha256'], 'native proposal changed')
    require(grant.get('native_run_root') == str(external_path(config['native_run_root'])),
            'grant belongs to another run root')
    for key in GRANT_BINDINGS:
        require(grant.get(key) == binding[key], f'grant does not bind {key}')
    expected = [{'stage': stage_name, 'condition': condition}
                for condition in CONDITIONS for stage_name in ('qualify', 'train', 'evaluate')
                if not (condition == 'anchor' and stage_name == 'train')]
    actions = grant.get('actions')
    require(isinstance(actions, list) and len(actions) == len(expected) and
            sorted(actions, key=digest) == sorted(expected, key=digest),
            'grant must explicitly authorize exactly the reviewed stage/condition set')
    require({'stage': stage, 'condition': binding['condition']} in actions,
            'stage/condition is not separately granted')
    require_review(config, binding)
    return grant


def require_ownership(config, processes=None):
    """Fresh operator ownership evidence plus current competing-worker scan."""
    path = external_path(config['ownership'])
    require(path.is_file(), 'fresh ownership receipt is missing; reuse any active owner chat')
    record = read_json(path)
    require(record.get('schema') == SCHEMA + '/ownership' and
            record.get('owner_thread_id') == config['owner_thread_id'] and
            record.get('study_id') == config['study_id'], 'ownership belongs to another chat/study')
    import psutil
    current = psutil.Process()
    ancestors = {p.pid for p in current.parents()}
    excluded = {os.getpid()} | ancestors
    age = datetime.now(timezone.utc) - dated(record.get('checked_utc'), 'checked_utc')
    require(age >= timedelta(0), 'ownership acknowledgement is future-dated')
    lock_path = external_path(config['native_run_root']) / 'run-owner-lock.json'
    orchestrated = False
    if lock_path.is_file():
        lock = read_json(lock_path)
        require(lock.get('schema') == SCHEMA + '/run-owner-lock' and
                lock.get('owner_thread_id') == config['owner_thread_id'] and
                lock.get('config_sha256') == digest(config) and
                lock.get('acknowledgement_sha256') == sha(path), 'run ownership lock changed')
        pid = lock.get('orchestrator_pid')
        require(pid in ancestors | {os.getpid()}, 'existing run belongs to another orchestrator')
        owner = psutil.Process(pid)
        require(owner.is_running() and owner.create_time() == lock.get('orchestrator_create_time'),
                'original run orchestrator is absent or its process identity changed')
        command = ' '.join(owner.cmdline()).lower()
        require('-m quality_followup run' in command, 'ownership lock is not a native-run orchestrator')
        orchestrated = True
    require(orchestrated or age <= timedelta(minutes=5),
            'standalone ownership acknowledgement must be within five minutes')
    workers = psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']) if processes is None else processes
    found = []
    for item in workers:
        info = item.info if hasattr(item, 'info') else item
        if info.get('pid') in excluded or 'python' not in str(info.get('name', '')).lower():
            continue
        command = ' '.join(info.get('cmdline') or []).lower()
        is_native = ('-m sbhm' in command or '-m quality_followup stage' in command or
                     '-m quality_followup run' in command or 'run_final_qualification.py' in command or
                     'probe_recovery_memory.py' in command)
        if is_native:
            found.append(info['pid'])
    require(not found, f'competing native owner/worker active: {found}; reuse its owner chat')
    return {'schema': SCHEMA + '/ownership-check', 'study_id': config['study_id'],
            'owner_thread_id': config['owner_thread_id'],
            'checked_utc': datetime.now(timezone.utc).isoformat(),
            'acknowledgement': artifact(path), 'acknowledgement_record': record,
            'uninterrupted_orchestrator': orchestrated,
            'run_lock': artifact(lock_path) if orchestrated else None,
            'other_native_worker_pids': found}


def require_qualification(path, binding):
    require(path is not None, 'fresh condition-bound native qualification required')
    path = external_path(path)
    require(path.is_file(), 'native qualification receipt missing')
    receipt = read_json(path)
    require(receipt.get('status') == 'PASS_NATIVE_MECHANICS_ONLY' and
            receipt.get('binding') == binding, 'qualification source/code/corpus/condition/delta mismatch')
    age = datetime.now(timezone.utc) - dated(receipt.get('created_utc'), 'qualification created_utc')
    require(timedelta(0) <= age <= timedelta(hours=24), 'qualification expired or future-dated')
    require(receipt.get('max_prompt_tokens_checked') == RECIPE['qualification_tokens'],
            'qualification did not check the registered 2048-token prefix')
    return receipt


def native_admission(config):
    """Call only after authorize: reuse unchanged SBHM guards and pinned loader."""
    expected = config['environment']
    require(str(Path(sys.executable).resolve()).casefold() ==
            str(Path(config['native_python']).resolve()).casefold(), 'wrong isolated native interpreter')
    require(platform.python_version() == expected.get('python') and
            platform.platform() == expected.get('platform'), 'native Python/platform changed')
    for name, value in expected.get('versions', {}).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = 'MISSING'
        require(actual == value, f'native dependency {name} changed')
    from sbhm.protocol import environment, resources
    actual = environment()
    require(actual == expected, 'native CUDA/device/environment differs from registration')
    return resources(RECIPE['device'])


def native_config(config, qualification=False):
    r = config['recipe']
    return {'schema': 'SBHM-001/v1', 'source': config['source'],
            **{key: r[key] for key in ('device', 'dtype', 'seed', 'layer', 'rank', 'window', 'learning_rate')},
            'limits': {key: r[key] for key in ('max_train_tokens', 'steps', 'max_new_tokens', 'prefill_chunk')}
                      | {'max_prompt_tokens': r['qualification_tokens'] if qualification else r['max_prompt_tokens']},
            'reset_recurrent_at': None}


def artifact(path):
    path = external_path(path)
    require(path.is_file(), 'missing native artifact')
    return {'path': str(path), 'sha256': sha(path)}
