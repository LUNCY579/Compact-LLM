"""SBHM-001 offline artifact audit; standard library only, no model execution.

Run with Python -B. The only write is a new exclusive-create JSON receipt under
BusinessProject outside the repository. Protected-named source metadata is read
as bytes for inventory hashes only; no content is parsed or used as evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys


REPO = Path(r'C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1')
ROOT = REPO / 'experiments/rineya_spikingbrain_hybrid_memory_v1'
RUN = ROOT / 'runs/titan_transition_v2_20261004_memory_efficient'
EVIDENCE = ROOT / 'evidence/native_20261004'
QUAL = ROOT / 'evidence/qualification_20261003'
OUTROOT = Path(r'C:/Users/clclc/BusinessProject/sbhm-001-audits/20261005-closeout-01a10d69')
EXPECTED_HEAD = 'fe9183f5c8e884316f4a6446d3b5472e9d6ad18f'
EXPECTED_BRANCH = 'research/rineya-spikingbrain-hybrid-memory-v1'
EXPECTED_BINDING = {
    'code_sha256': '326d805bde96047ca0553f8c33e5124a6c346fc11caaf3c28b4221277ef82140',
    'config_sha256': '30b85a8bd3e24043bbbd2983d735eaf64df47e60626dc051792b45f3918d2400',
    'source_sha256': 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a',
    'corpus_sha256': '6f9c858b39c6c45d4fd5cd6bfa29d79b3c2f12773af7f392b44cfa171379105f',
}
failures = []
check_count = 0
inventory = {}


def check(condition, name):
    global check_count
    check_count += 1
    if not condition:
        failures.append(name)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 2**20), b''):
            h.update(chunk)
    value = h.hexdigest()
    inventory[str(Path(path).resolve())] = {'sha256': value, 'bytes': Path(path).stat().st_size}
    return value


def read(path):
    path = Path(path)
    if 'CONFIRM' in path.name.upper() or 'SEALED' in path.name.upper():
        raise ValueError('protected-named content must never be parsed: ' + str(path))
    sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def time(value):
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise ValueError('naive timestamp')
    return t


def tensor_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('short safetensors prefix')
        n = struct.unpack('<Q', prefix)[0]
        if not 2 <= n <= 10 * 2**20:
            raise ValueError('invalid safetensors header length')
        result = json.loads(stream.read(n))
    tensors = {k: v for k, v in result.items() if k != '__metadata__'}
    spans = sorted(v['data_offsets'] for v in tensors.values())
    if not spans or spans[0][0] != 0 or spans[-1][1] + n + 8 != Path(path).stat().st_size:
        raise ValueError('tensor spans do not exactly cover file')
    if any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('overlap or gap in tensor spans')
    for key, t in tensors.items():
        width = {'F32': 4, 'BF16': 2}.get(t['dtype'])
        if width is None or math.prod(t['shape']) * width != t['data_offsets'][1] - t['data_offsets'][0]:
            raise ValueError('tensor byte/shape/dtype mismatch: ' + key)
    return tensors


def cache(arm, n):
    per_layer_per_token = 2 * 8 * 128 * 4
    return (28 * per_layer_per_token * n if arm in 'AB' else
            27 * per_layer_per_token * n + per_layer_per_token * min(n, 128) + (8 * 128 * 128 * 4 if arm == 'D' else 0))


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def section(name, func, result):
    before = len(failures)
    try:
        result[name] = func()
    except Exception as exc:
        failures.append(name + ': ' + type(exc).__name__ + ': ' + str(exc))
        result[name] = {'exception': str(exc)}
    result.setdefault('section_status', {})[name] = 'PASS' if len(failures) == before else 'FAIL'


def main(output):
    global check_count
    config_path = ROOT / 'configs/titan_transition.v2.memory_efficient.json'
    config = read(config_path)
    manifest_path = Path(config['corpus_manifest'])
    manifest = read(manifest_path)
    design = read(manifest_path.parent / manifest['design']['path'])
    grant = read(config['grant'])
    report = {
        'schema': 'SBHM-001/offline-provenance-closeout-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'independent Codex provenance subagent',
        'script_path': str(Path(__file__).resolve()),
        'script_sha256': sha(__file__),
        'python': sys.version,
        'reproduction': ['python', '-B', str(Path(__file__).resolve()), '--output', 'NEW_provenance_receipt.json'],
        'scope': 'Read-only historical DEV artifact identity/completion audit, not native reexecution or present permission.',
        'native_or_gpu_execution': False, 'model_weights_loaded': False,
        'training_or_evaluation_executed': False, 'confirmation_bank_content_parsed': False,
        'new_native_authorization_claim': False,
        'run_root': str(RUN), 'configuration': str(config_path),
    }
    stages = ['A-qualification', 'teacher', 'A-evaluation']
    for arm in 'BCD':
        stages += [arm + '-' + x for x in ['initial-qualification', 'distill', 'distilled-qualification', 'recover', 'recovered-qualification', 'evaluation']]
    receipts = {name: read(RUN / name / 'receipt.json') for name in stages}
    requests = {name: read(RUN / name / 'request.json') for name in stages}
    environment = receipts['A-qualification']['binding']['environment']
    code_files = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'sbhm').glob('*.py'))}
    binding = {
        'code_sha256': digest(code_files), 'config_sha256': digest(config),
        'source_sha256': digest(config['source']['files']), 'corpus_sha256': sha(manifest_path),
    }
    report['binding'] = binding
    report['native_recorded_environment'] = environment
    report['code_files'] = code_files
    corpora = {}

    def identity():
        branch, head = git('branch', '--show-current'), git('rev-parse', 'HEAD')
        check(branch == EXPECTED_BRANCH, 'expected branch')
        check(head == EXPECTED_HEAD, 'expected HEAD')
        check(binding == EXPECTED_BINDING, 'current code/config/source/corpus digests equal native registration')
        check(config['device'] == 'cuda:0' and config['dtype'] == 'float32' and config['layer'] == 14 and config['window'] == 128 and config['rank'] == 8 and config['seed'] == 17 and config['learning_rate'] == .0001 and config['reset_recurrent_at'] is None, 'fixed registered recipe')
        check(config['limits'] == {'max_new_tokens': 32, 'max_prompt_tokens': 2048, 'max_train_tokens': 512, 'prefill_chunk': 64, 'steps': 512}, 'fixed registered limits')
        register = read(EVIDENCE / 'memory-efficient-rerun-registration.json')
        check(register['binding'] == binding and register['pool_prior_controls'] is False, 'fresh memory-equivalent matrix registration')
        check(set(environment) == {'cuda', 'device', 'platform', 'python', 'versions'}, 'native environment field contract')
        cuda_env = read(QUAL / 'cuda_stack/environment.json')
        check(environment['python'] == cuda_env['python_version'] and environment['cuda'] == cuda_env['compiled_cuda'] and environment['platform'] == cuda_env['platform'], 'native environment matches historical CUDA prerequisites')
        check(all(environment['versions'][key] == cuda_env['versions'][key] for key in environment['versions']), 'native versions match historical CUDA prerequisites')
        check(environment['versions']['transformers'] == '4.51.3' and environment['versions']['torch'] == '2.6.0+cu124' and environment['cuda'] == '12.4', 'native pinned transformers/CUDA stack')
        return {'branch': branch, 'head': head, 'local_changes_not_modified': True, 'native_environment_identity_verified_from_records_only': True}
    section('branch_registration_and_environment', identity, report)

    def source_check():
        snapshot = Path(config['source']['path'])
        lock = read(QUAL / 'titan_transition_source_lock.json')
        snapshot_record = read(QUAL / 'native_source_snapshot.json')
        check(lock['files'] == config['source']['files'] == snapshot_record['source']['files'], 'snapshot and published source locks agree')
        source_inventories = {}
        for label, root in [('native_snapshot', snapshot), ('published_source', Path(lock['path']))]:
            files = sorted(p for p in root.rglob('*') if p.is_file())
            observed = {str(p.relative_to(root)): sha(p) for p in files}
            check(observed == lock['files'], label + ': exact 17-file inventory and hashes')
            check(all(p.resolve().is_relative_to(root.resolve()) for p in files), label + ': no escaping file links')
            check(not any(name.endswith(('.py', '.bin', '.pt', '.pth')) for name in observed), label + ': no custom-code or pickle source')
            source_inventories[label] = {'path': str(root), 'files': observed, 'bytes': sum(p.stat().st_size for p in files)}
        check(digest(source_inventories['native_snapshot']['files']) == binding['source_sha256'], 'actual source digest')
        source_config = read(snapshot / 'config.json')
        expected_geometry = {'hidden_size': 1024, 'head_dim': 128, 'num_attention_heads': 16, 'num_key_value_heads': 8, 'num_hidden_layers': 28, 'vocab_size': 151936}
        check({k: source_config[k] for k in expected_geometry} == expected_geometry, 'source explicit Qwen3 geometry')
        check(source_config['architectures'] == ['Qwen3ForCausalLM'] and source_config['model_type'] == 'qwen3' and not source_config.get('auto_map') and not source_config.get('quantization_config') and not source_config.get('rope_scaling') and not source_config.get('use_sliding_window') and source_config['tie_word_embeddings'], 'supported ungated standard Qwen3 and tied embedding')
        tensors = tensor_header(snapshot / 'model.safetensors')
        expected = {'model.embed_tokens.weight': [151936, 1024], 'model.norm.weight': [1024]}
        layer_shapes = {'self_attn.q_proj.weight': [2048, 1024], 'self_attn.k_proj.weight': [1024, 1024], 'self_attn.v_proj.weight': [1024, 1024], 'self_attn.o_proj.weight': [1024, 2048], 'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128], 'mlp.gate_proj.weight': [3072, 1024], 'mlp.up_proj.weight': [3072, 1024], 'mlp.down_proj.weight': [1024, 3072], 'input_layernorm.weight': [1024], 'post_attention_layernorm.weight': [1024]}
        for layer in range(28):
            expected.update({f'model.layers.{layer}.{key}': value for key, value in layer_shapes.items()})
        check(set(tensors) == set(expected), 'all 310 source tensor names')
        check(all(tensors[k]['shape'] == v and tensors[k]['dtype'] == 'F32' for k, v in expected.items()), 'all source tensor shapes and actual F32 dtype')
        count = sum(math.prod(v['shape']) for v in tensors.values())
        check(count == 596049920, '596049920 source parameters')
        historical_geometry = read(QUAL / 'geometry_inventory.json')
        selected = next(c for c in historical_geometry['candidates'] if c['path'] == lock['path'])
        check(selected['full_tensor_inventory'] == {k: {'shape': v['shape'], 'dtype': v['dtype']} for k, v in tensors.items()}, 'selected October 3 tensor inventory agrees with current locked header')
        smoke = read(QUAL / 'source-tokenizer-smoke.json')
        check(all(lock['files'][k] == v for k, v in smoke['input_files_sha256'].items()), 'historical tokenizer/config smoke hashes agree')
        check(all(smoke['geometry'][k] == source_config[k] for k in smoke['geometry']), 'historical tokenizer/config smoke geometry agrees')
        tokenizer = read(snapshot / 'tokenizer_config.json')
        check(hashlib.sha256(tokenizer['chat_template'].encode()).hexdigest() == smoke['chat_template_sha256'], 'tokenizer chat-template hash')
        return {'inventories': source_inventories, 'tensor_count': len(tensors), 'parameter_count': count, 'parameter_bytes': count * 4, 'geometry': expected_geometry, 'weight_sha256': lock['files']['model.safetensors'], 'tokenizer': {'class_recorded': smoke['tokenizer_class'], 'eos_token_id': smoke['tokenizer_eos_token_id'], 'chat_template_sha256': smoke['chat_template_sha256']}, 'source_metadata_with_CONFIRM002_names': 'bytes hashed only; never parsed, no bank access or imported permit'}
    section('locked_source_and_tensor_geometry', source_check, report)

    def corpus_check():
        check(manifest['reviewed_for_sbhm'] is True and set(manifest['splits']) == {'train', 'dev'}, 'admitted TRAIN/DEV-only manifest')
        for split, entry in manifest['splits'].items():
            path = manifest_path.parent / entry['path']
            check(entry['role'] == split and entry['protected'] is False and sha(path) == entry['sha256'], split + ': role/unprotected/file lock')
            corpora[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            check(len(corpora[split]) == (256 if split == 'train' else 512), split + ': expected row count')
            check(all(set(row) == {'id', 'family_id', 'prompt', 'target'} for row in corpora[split]), split + ': raw row schema')
        for reference in ['design', 'review']:
            check(sha(manifest_path.parent / manifest[reference]['path']) == manifest[reference]['sha256'], reference + ': metadata hash')
        groups = manifest['evaluation_groups']
        families = [family for values in groups.values() for family in values]
        check(len(families) == len(set(families)) and set(families) == {r['family_id'] for r in corpora['dev']}, 'disjoint evaluation group coverage')
        for field in ['id', 'family_id', 'prompt']:
            normalize = (lambda x: ' '.join(x.split()).casefold()) if field == 'prompt' else (lambda x: x)
            a, b = [{normalize(r[field]) for r in corpora[split]} for split in ['train', 'dev']]
            check(a.isdisjoint(b), 'TRAIN/DEV separation: ' + field)
        return {'train_rows': len(corpora['train']), 'dev_rows': len(corpora['dev']), 'groups': {g: sum(r['family_id'] in f for r in corpora['dev']) for g, f in groups.items()}, 'only_train_and_dev_parsed': True, 'token_geometry_is_registered_metadata_not_retokenized_by_stdlib_auditor': True}
    section('corpus_locks_and_split_coverage', corpus_check, report)

    def expected_stage_binding(name):
        arm = 'A' if name == 'teacher' else name[0]
        previous = None
        if arm != 'A' and name.endswith(('recovered-qualification', 'evaluation')):
            previous = RUN / (arm + '-recover')
        elif name.endswith(('distilled-qualification', 'recover')):
            previous = RUN / (arm + '-distill')
        return {**binding, 'schema': 'SBHM-001/v1', 'environment': environment, 'arm': arm, 'delta_sha256': sha(previous / 'delta.safetensors') if previous else None, 'delta_manifest_sha256': sha(previous / 'delta.json') if previous else None}

    def stage_check():
        actual = {p.parent.name for p in RUN.glob('*/receipt.json')}
        check(actual == set(stages) and len(actual) == 21, 'exact 21 native receipt inventory')
        issued, expires = time(grant['issued_utc']), time(grant['expires_utc'])
        check(timedelta(0) < expires - issued <= timedelta(days=7) and bool(grant['approved_by']), 'dated nonempty maximum-seven-day grant contract')
        check(all(grant[k] == v for k, v in binding.items()), 'grant exact binding')
        check({'qualify', 'teacher', 'distill', 'recover', 'evaluate'} <= set(grant['actions']), 'grant covers historical stage actions')
        detail = {}
        for name in stages:
            r, q = receipts[name], requests[name]
            expected = expected_stage_binding(name)
            check(r['binding'] == q['binding'] == expected, name + ': receipt/request exact variant and environment binding')
            stage = 'qualify' if name.endswith('qualification') else 'teacher' if name == 'teacher' else 'evaluate' if name.endswith('evaluation') else name.split('-')[1]
            check(q['stage'] == stage, name + ': request stage')
            created, completed = time(q['created_utc']), time(r['created_utc'])
            check(issued <= created <= completed < expires, name + ': historical request and completion within recorded grant')
            check(isinstance(r['allocator_cache_release_events'], list) and len(r['allocator_cache_release_events']) == 0, name + ': zero reported allocator release events')
            detail[name] = {'request_utc': q['created_utc'], 'completed_utc': r['created_utc'], 'elapsed_request_to_receipt_seconds': (completed - created).total_seconds(), 'request_sha256': sha(RUN / name / 'request.json'), 'receipt_sha256': sha(RUN / name / 'receipt.json'), 'status': r['status']}
        check(read(RUN / 'teacher/teacher.json') == receipts['teacher'], 'teacher manifest identical to receipt')
        handoff = read(EVIDENCE / 'final_completed_handoff.json')
        check(handoff['matrix_exit_code'] == 0 and handoff['native_qualification_stages_completed'] == 10 and handoff['native_training_stages_completed'] == 6, 'historical handoff reports complete successful matrix')
        log = EVIDENCE / 'matrix-memory-efficient-execution.txt'
        log_digest = sha(log)
        log_text = log.read_text(encoding='utf-8')
        check('DESCRIPTIVE_PAIRED_DEV_ONLY' in log_text, 'matrix log reached completed comparison')
        return {'count': 21, 'matrix_commands': 22, 'stages': detail, 'grant': {'issued_utc': grant['issued_utc'], 'expires_utc': grant['expires_utc'], 'historical_scope_only': True, 'approval_is_self_recorded_not_cryptographic_or_independent_authorization_proof': True, 'not_permission_for_new_native_work': True}, 'completion_log_sha256': log_digest, 'exit_code_reported_historically': 0, 'historical_process_exit_not_reexecuted_or_independently_observed': True, 'failed_old_run_pooled': False}
    section('native_completion_requests_bindings_historical_grant', stage_check, report)

    def qualification_check():
        qualifications = {}
        for name in stages:
            if not name.endswith('qualification'):
                continue
            r = receipts[name]
            arm = name[0]
            expected_parameters = 0 if arm == 'A' else 1131664 if arm == 'D' else 81920
            check(r['status'] == 'PASS_NATIVE_MECHANICS_ONLY' and r['capability_claim'] is False and r['native_training_qualified'] is False, name + ': qualification status/mechanics only')
            check(r['identity_wrapper_max_abs_error'] == 0 and r['max_prompt_tokens_checked'] == 2048, name + ': identity and maximum tested prompt')
            check(r['trainable_parameters'] == expected_parameters and r['total_parameters'] == 596049920 + expected_parameters, name + ': parameter totals')
            check(set(r['parity']) == {'1:17', '64:2048'}, name + ': tokenwise short and chunked 2048 parity records')
            for label, p in r['parity'].items():
                check(math.isfinite(p['max_abs_logit_error']) and p['max_abs_logit_error'] <= 3e-4, name + ':' + label + ': finite bounded logit error')
                check(p['cache_storage_bytes'] == cache(arm, int(label.split(':')[1])), name + ':' + label + ': source-derived actual storage')
            qualifications[name] = {'trainable_parameters': r['trainable_parameters'], 'parity': r['parity']}
        next_stages = [('A-qualification', 'teacher'), ('A-qualification', 'A-evaluation')]
        for arm in 'BCD':
            next_stages += [(arm + '-initial-qualification', arm + '-distill'), (arm + '-distilled-qualification', arm + '-recover'), (arm + '-recovered-qualification', arm + '-evaluation')]
        freshness = {}
        for qname, sname in next_stages:
            q, request = receipts[qname], requests[sname]
            age = time(request['created_utc']) - time(q['created_utc'])
            check(q['binding'] == request['binding'], sname + ': exact matching prior qualification')
            check(timedelta(0) <= age <= timedelta(hours=24), sname + ': historical prior qualification age <=24h')
            freshness[sname] = {'qualification': qname, 'age_seconds_at_request': age.total_seconds()}
        check(len(qualifications) == 10, 'all 10 qualifications present')
        return {'count': 10, 'measurements': qualifications, 'freshness': freshness, 'at_2048_tokens_cache_bytes': {a: cache(a, 2048) for a in 'ABCD'}, 'limitations': ['Inference mechanics were not reexecuted.', 'Only saved numeric parity/cache records are independently arithmetically checked; greedy parity/request independence are implied by the bound PASS-producing code, not separately persisted raw trajectories.', 'Historical freshness does not qualify a newly proposed native run.']}
    section('qualifications_and_freshness', qualification_check, report)

    order = list(range(256))
    random.Random(config['seed']).shuffle(order)
    consumed = [corpora['train'][order[i % 256]] for i in range(512)]

    def teacher_check():
        r = receipts['teacher']
        check(r['status'] == 'TEACHER_TARGETS_CAPTURED' and r['split'] == 'train' and r['layer'] == 14 and r['corpus_sha256'] == binding['corpus_sha256'], 'teacher identity/role/status')
        entries = r['entries']
        check(len(entries) == 256 and len({e['id'] for e in entries}) == 256, '256 distinct teacher targets')
        check([e['id'] for e in entries] == [row['id'] for row in consumed[:256]], 'teacher inventory follows deterministic first exposure')
        checked, total = [], 0
        for entry, row in zip(entries, consumed):
            p = RUN / 'teacher' / entry['path']
            check(p.resolve().is_relative_to((RUN / 'teacher').resolve()) and p.suffix == '.safetensors', entry['id'] + ': safe target path')
            actual = sha(p)
            check(actual == entry['sha256'] and digest(row) == entry['row_sha256'], entry['id'] + ': target and raw row hashes')
            header = tensor_header(p)
            n = design['rows'][row['id']]['token_geometry']['train_tokens']
            expected = {'hidden': [1, n, 1024], 'target': [1, n, 1024], 'cos': [1, n, 128], 'sin': [1, n, 128]}
            check(set(header) == set(expected) and all(header[k]['shape'] == v and header[k]['dtype'] == 'F32' for k, v in expected.items()), entry['id'] + ': target source geometry and dtype')
            total += p.stat().st_size
            checked.append({'id': entry['id'], 'file': entry['path'], 'sha256': actual, 'bytes': p.stat().st_size, 'train_tokens': n})
        check({p.name for p in (RUN / 'teacher').glob('*.safetensors')} == {e['path'] for e in entries}, 'no missing or unlisted teacher targets')
        check(total == r['teacher_cache_file_bytes'] == 502904576 and total <= 512 * 2**20, 'teacher file bytes equal receipt and below 512 MiB cap')
        return {'count': len(checked), 'bytes': total, 'MiB': total / 2**20, 'cap_bytes': 512 * 2**20, 'margin_bytes': 512 * 2**20 - total, 'targets': checked, 'tensor_payload_not_deserialized': True}
    section('teacher_target_hashes_and_headers', teacher_check, report)

    def training_check():
        traces, deltas, training = {}, {}, {}
        lora_shapes = {'q_proj.a': [8, 1024], 'q_proj.b': [2048, 8], 'k_proj.a': [8, 1024], 'k_proj.b': [1024, 8], 'v_proj.a': [8, 1024], 'v_proj.b': [1024, 8], 'o_proj.a': [8, 2048], 'o_proj.b': [1024, 8]}
        for arm in 'BCD':
            for stage in ['distill', 'recover']:
                name = arm + '-' + stage
                r = receipts[name]
                delta = read(RUN / name / 'delta.json')
                p = RUN / name / 'delta.safetensors'
                actual = sha(p)
                header = tensor_header(p)
                # native.install keeps source attention in B and attaches its
                # LoRA projections directly to self_attn, exactly as C/D do.
                shapes = {'model.layers.14.self_attn.' + k: v for k, v in lora_shapes.items()}
                if arm == 'D':
                    shapes.update({'model.layers.14.self_attn.' + k: v for k, v in {'forget.bias': [1024], 'forget.weight': [1024, 1024], 'memory_scale': [128], 'mix': [16]}.items()})
                check(delta['record'] == r and delta['binding'] == r['binding'] and delta['sha256'] == actual, name + ': delta manifest embeds exact receipt/binding/hash')
                check(set(header) == set(r['parameter_names']) == set(delta['parameters']) == set(shapes), name + ': exact same-site trainable tensor inventory')
                check(all(header[k]['shape'] == delta['parameters'][k] == shapes[k] and header[k]['dtype'] == 'F32' for k in shapes), name + ': expected trainable shape/dtype')
                expected_count = 1131664 if arm == 'D' else 81920
                check(sum(math.prod(v['shape']) for v in header.values()) == r['trainable_parameters'] == expected_count, name + ': saved trainable parameter count')
                check(r['status'] == 'TRAINING_COMPLETED_NOT_EFFICACY' and r['stage'] == stage and r['resume_supported'] is False and r['corpus_sha256'] == binding['corpus_sha256'], name + ': completed training stage')
                check(r['parent_record'] == (None if stage == 'distill' else receipts[arm + '-distill']), name + ': parent distillation chain')
                check(r['optimizer'] == 'AdamW(.9,.95), wd=0, eps=1e-8, clip=1', name + ': matched registered optimizer')
                records = r['records']
                check(len(records) == 512, name + ': 512 updates')
                for i, (record, row) in enumerate(zip(records, consumed)):
                    g = design['rows'][row['id']]['token_geometry']
                    check(record['step'] == i and record['row_id'] == row['id'] and record['input_tokens'] == g['train_tokens'] and record['supervised_positions'] == (g['train_tokens'] if stage == 'distill' else g['target_tokens'] + 1), name + ': exact row/input/label exposure step ' + str(i))
                    check(all(type(record[k]) in [int, float] and math.isfinite(record[k]) and record[k] >= 0 for k in ['loss', 'gradient_norm']), name + ': finite nonnegative loss/gradient step ' + str(i))
                trace = [(v['row_id'], v['input_tokens'], v['supervised_positions']) for v in records]
                if stage in traces:
                    check(trace == traces[stage], name + ': matched control exposure')
                traces[stage] = trace
                check(Counter(v['row_id'] for v in records) == Counter({v['id']: 2 for v in corpora['train']}), name + ': each of 256 TRAIN rows twice')
                deltas[name] = {'sha256': actual, 'manifest_sha256': sha(RUN / name / 'delta.json'), 'bytes': p.stat().st_size, 'tensor_count': len(header), 'parameter_count': expected_count, 'tensor_shapes': shapes}
                training[name] = {'updates': len(records), 'unique_rows': len({v['row_id'] for v in records}), 'input_tokens_total': sum(v['input_tokens'] for v in records), 'supervised_positions_total': sum(v['supervised_positions'] for v in records), 'input_tokens_min': min(v['input_tokens'] for v in records), 'input_tokens_max': max(v['input_tokens'] for v in records), 'trace_sha256': digest(trace), 'loss_first': records[0]['loss'], 'loss_last': records[-1]['loss'], 'max_gradient_norm': max(v['gradient_norm'] for v in records)}
        return {'six_delta_files': deltas, 'six_training_stages': training, 'matched_exposure_is_not_matched_compute': True, 'actual_training_fit_only_42_to_259_tokens': True, '512_token_training_cap_not_qualified': True}
    section('binary_deltas_training_completion_and_exposure', training_check, report)

    def evaluation_check():
        details = {}
        for arm in 'ABCD':
            r = receipts[arm + '-evaluation']
            check(r['status'] == 'DEV_MEASUREMENT_ONLY' and r['evaluation_groups'] == manifest['evaluation_groups'] and r['corpus_sha256'] == binding['corpus_sha256'], arm + ': evaluation status/corpus/groups')
            check(r['adaptation_record'] == (None if arm == 'A' else receipts[arm + '-recover']), arm + ': correct recovered delta receipt embedded')
            check(r['energy_measured'] is False and r['speedup_claim'] is False and r['semantic_accuracy'] is None, arm + ': limits of recorded measurement')
            preds = r['predictions']
            check(len(preds) == 512, arm + ': complete 512 DEV rows')
            for p, row in zip(preds, corpora['dev']):
                check(all(p[k] == row[k] for k in ['id', 'family_id', 'target']), arm + ':' + row['id'] + ': exact DEV row pairing/order')
                check(type(p['strict_exact_match']) is bool and p['strict_exact_match'] == (p['prediction'] == row['target']) and type(p['normalized_exact_match']) is bool and p['normalized_exact_match'] == (p['prediction'].strip().casefold() == row['target'].strip().casefold()), arm + ':' + row['id'] + ': raw exact-match scores')
                check(p['reset_recurrent_at'] is None and p['prompt_tokens'] == design['rows'][row['id']]['token_geometry']['prompt_tokens'], arm + ':' + row['id'] + ': primary no reset/prompt count')
                ids = p['generated_ids']
                check(p['generated_tokens'] == len(ids) and 1 <= len(ids) <= 32 and all(type(i) is int and 0 <= i < 151936 for i in ids), arm + ':' + row['id'] + ': generated-id bounds')
                check(p['stop_reason'] == ('eos' if ids[-1] == 151645 else 'max_new_tokens') and (p['stop_reason'] == 'eos' or len(ids) == 32), arm + ':' + row['id'] + ': EOS/max-length stop')
                check(p['cache_storage_bytes'] == cache(arm, p['prompt_tokens'] + len(ids) - 1), arm + ':' + row['id'] + ': source-derived persistent cache storage')
                check(p['rss_sample_max_bytes'] <= 4 * 2**30 and p['peak_vram_allocated_bytes'] <= p['peak_vram_reserved_bytes'], arm + ':' + row['id'] + ': sampled RSS ceiling and allocation/reserve ordering')
            for metric in ['strict_exact_match', 'normalized_exact_match']:
                check(r[metric] == sum(p[metric] for p in preds) / 512, arm + ': exact aggregate ' + metric)
            details[arm] = {'rows': len(preds), 'strict_correct': sum(p['strict_exact_match'] for p in preds), 'normalized_correct': sum(p['normalized_exact_match'] for p in preds), 'sampled_rss_max_bytes': max(p['rss_sample_max_bytes'] for p in preds), 'allocated_peak_max_bytes': max(p['peak_vram_allocated_bytes'] for p in preds), 'reserved_peak_max_bytes': max(p['peak_vram_reserved_bytes'] for p in preds)}
        comparison = read(RUN / 'comparison.json')
        check(comparison['status'] == 'DESCRIPTIVE_PAIRED_DEV_ONLY' and comparison['promotion'] is False, 'comparison descriptive DEV only')
        check(receipts['A-qualification']['trainable_parameters'] == 0 and receipts['A-evaluation']['binding']['delta_sha256'] is None and receipts['A-evaluation']['adaptation_record'] is None, 'A is unadapted source control')
        for group, families in manifest['evaluation_groups'].items():
            expected_counts = {a: sum(p['strict_exact_match'] for p in receipts[a + '-evaluation']['predictions'] if p['family_id'] in families) for a in 'ABCD'}
            n = sum(p['family_id'] in families for p in receipts['A-evaluation']['predictions'])
            check(all(comparison['evaluation_groups'][group]['arms'][a]['strict_exact_match'] == expected_counts[a] / n for a in 'ABCD'), group + ': comparison raw strict group counts')
        return {'arms': details, 'all_2048_prediction_rows_paired_raw_scored_and_cache_checked': True, 'comparison_sha256': sha(RUN / 'comparison.json'), 'remaining_27_full_history_sites': True, 'metrics_are_exact_strings_not_semantic_JSON': True}
    section('evaluation_completion_pairing_and_cache', evaluation_check, report)

    def snapshots_check():
        snapshot = EVIDENCE / 'completed_receipts'
        m = read(snapshot / 'manifest.json')
        expected = {'comparison.json', 'teacher\\teacher.json'}
        for name in stages:
            expected.update([name + '\\receipt.json', name + '\\request.json'])
        for arm in 'BCD':
            for stage in ['distill', 'recover']:
                expected.add(arm + '-' + stage + '\\delta.json')
        check(set(m['files']) == expected and len(m['files']) == 50, 'exact expected 50 copied JSON receipt/manifests')
        actual_files = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()}
        check(actual_files == expected | {'manifest.json'}, 'snapshot directory contains exact copied files and manifest')
        verified, total = {}, 0
        for name, entry in m['files'].items():
            original, copy = RUN / name, snapshot / name
            o, c = sha(original), sha(copy)
            check(o == c == entry['sha256'] and original.stat().st_size == copy.stat().st_size == entry['bytes'], name + ': byte-identical authoritative/snapshot copy and manifest hash/length')
            verified[name] = {'sha256': o, 'bytes': entry['bytes']}
            total += entry['bytes']
        check(m['source_run'] == str(RUN) and m['binary_artifacts_remain_in_source_run'] is True, 'snapshot source/binary-artifact location')
        return {'count': len(verified), 'bytes': total, 'manifest_sha256': sha(snapshot / 'manifest.json'), 'copied_files': verified, 'no_copy_or_mutation_performed': True}
    section('all_50_completed_snapshot_copies', snapshots_check, report)

    def historical_support_check():
        equivalence = read(EVIDENCE / 'memory-equivalence-review.json')
        check(equivalence['current_binding'] == binding, 'memory-equivalence review current binding')
        for name, value in equivalence['input_sha256'].items():
            check(sha(ROOT / name) == value, 'memory-equivalence input hash ' + name)
        cpu_output = EVIDENCE / 'recovery_memory_qualification/full-with-cache.txt'
        cpu_text = cpu_output.read_text(encoding='utf-8')
        check(sha(cpu_output) == equivalence['completed_cpu_evidence']['full_with_cache_output_sha256'] and '136 passed' in cpu_text and 'skipped' not in cpu_text, 'historic 136-pass zero-skipped full CPU output lock')
        final = read(EVIDENCE / 'final_completed_handoff.json')
        for name, value in final['audit_sha256'].items():
            check(sha(EVIDENCE / name) == value, 'historic final handoff audit lock ' + name)
        older = read(QUAL / 'cuda_stack/middle-site-final.json')
        check(older['full_suite_passed'] == 109 and older['full_suite_skipped'] == 0 and older['pretrained_weights_loaded'] is False and older['native_fit_or_behavior_result'] is False, 'historic final pinned-library CPU-only 109-test qualification')
        return {'cpu_136_pass_output_sha256': sha(cpu_output), 'historical_109_passes_not_current_native_code_qualification': True, 'memory_equivalence_input_hashes_checked': equivalence['input_sha256'], 'historic_final_audit_hashes_checked': final['audit_sha256'], 'guards': {'admission_free_RAM_GiB': 7.4, 'runtime_free_RAM_GiB': 6, 'worker_RSS_GiB': 4, 'free_VRAM_reserve_GiB': 1, 'native_device': 'cuda:0'}, 'guard_implementation_bound_by_exact_protocol_hash': True, 'boundary_checks_not_continuous_resource_certificate': True}
    section('historical_supporting_receipt_chains', historical_support_check, report)

    report['audit_revision'] = {'version': 2, 'previous_receipt': str(OUTROOT / 'provenance_receipt.json'), 'previous_script': str(OUTROOT / 'provenance_audit.py'), 'correction': 'v1 auditor incorrectly expected B LoRA tensors below self_attn.base; native.install and saved parameter_names place them directly below self_attn. v1 failure is an auditor schema assumption, not failed native evidence. v1 script/receipt remain preserved; v2 rechecks all six training deltas/stages.'}
    report['limitations'] = [
        'Artifact consistency audit cannot prove actual execution independently of the historical self-recorded receipts and completion log.',
        'Historical dated grants are audit records, not cryptographic evidence of owner consent and not new permission under the current offline-only request.',
        'Native recorded environment identities were compared across records; no current CUDA environment or native model requalification was executed.',
        'Locked tensors were header/byte/hash checked only; no numerical model, optimizer, target payload, or capability computation was performed.',
        'Native PASS supports bound inference mechanics; saved receipts omit separate raw greedy-token and request-independence trajectories.',
        'Teacher tensor lengths and prompt/train exposures agree with locked registered token geometry; this stdlib audit does not rerun the tokenizer.',
        'Boundary resource guards and sampled evaluation RSS cannot certify continuous transient host/GPU headroom.',
        'Actual successful TRAIN shape range is 42-259 tokens; the nominal 512-token cap and additional sites/contexts remain unqualified.',
        'One source/site/seed, synthetic DEV, repeated inspection, dependent counterfactual/position worlds, and remaining 27 full-attention sites preclude broad retention, recurrent-route, or promotion claims.',
        'No failed older matrix control is used in the authoritative completed four-arm comparison.',
    ]
    report['checks_run'] = check_count
    report['failures'] = failures
    report['status'] = 'PASS_OFFLINE_HISTORICAL_NATIVE_ARTIFACT_AUDIT' if not failures else 'FAIL_OFFLINE_HISTORICAL_NATIVE_ARTIFACT_AUDIT'
    report['artifact_inventory'] = inventory
    output = Path(output).resolve()
    if not output.is_relative_to(OUTROOT.resolve()) or output.parent != OUTROOT.resolve():
        raise ValueError('receipt must be a new file directly under fixed audit output root')
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'checks_run': check_count, 'failures': failures, 'output': str(output), 'artifact_files_hashed': len(inventory), 'native_receipts': 21, 'qualifications': 10, 'deltas': 6, 'teacher_targets': 256, 'snapshot_copies': 50}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTROOT / 'provenance_receipt_v2.json')
    args = parser.parse_args()
    raise SystemExit(main(args.output))
