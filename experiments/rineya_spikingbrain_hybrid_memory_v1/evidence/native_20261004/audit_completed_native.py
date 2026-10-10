"""Independent saved-artifact audit; never import Torch or load a model.

Read only the registered source inventory, approved synthetic corpus, completed
run artifacts and qualification evidence. Write a new audit receipt, never alter
any native/configuration/data receipt. Tensor shapes come from safe headers.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/titan_transition_v2_20261004_memory_efficient'
CONFIG = ROOT / 'configs/titan_transition.v2.memory_efficient.json'
OUT = Path(__file__).resolve().parent / 'completed_native_audit.json'


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def header(path):
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        require(size < 10 * 2**20, 'unexpected safetensors header size')
        result = json.loads(stream.read(size))
    tensors = {key: value for key, value in result.items() if key != '__metadata__'}
    spans = sorted(value['data_offsets'] for value in tensors.values())
    require(spans[0][0] == 0 and spans[-1][1] + size + 8 == Path(path).stat().st_size,
            'tensor offsets do not cover the saved file exactly')
    require(all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'overlapping/gapped tensor offsets')
    for value in tensors.values():
        width = {'F32': 4, 'BF16': 2}.get(value['dtype'])
        require(width is not None and math.prod(value['shape']) * width == value['data_offsets'][1] - value['data_offsets'][0],
                'tensor shape/dtype offsets disagree')
    return tensors


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'naive artifact timestamp')
    return result


def summary(baseline, candidate):
    result = {'rows': len(candidate),
              'repairs': sum(not a['strict_exact_match'] and b['strict_exact_match'] for a, b in zip(baseline, candidate)),
              'damages': sum(a['strict_exact_match'] and not b['strict_exact_match'] for a, b in zip(baseline, candidate)),
              'original_model_retention': {}}
    for metric in ('strict_exact_match', 'normalized_exact_match'):
        correct = sum(row[metric] for row in baseline)
        retained = sum(a[metric] and b[metric] for a, b in zip(baseline, candidate))
        result[metric] = sum(row[metric] for row in candidate) / len(candidate)
        result['original_model_retention'][metric] = {
            'baseline_correct': correct, 'retained': retained, 'damages': correct - retained,
            'repairs': sum(not a[metric] and b[metric] for a, b in zip(baseline, candidate)),
            'retention_rate': retained / correct if correct else None}
    return result


def contrast(local, hybrid):
    result = {'paired_rows': len(local)}
    for metric in ('strict_exact_match', 'normalized_exact_match'):
        d_only = sum(not c[metric] and d[metric] for c, d in zip(local, hybrid))
        c_only = sum(c[metric] and not d[metric] for c, d in zip(local, hybrid))
        result[metric] = {'D_only_correct': d_only, 'C_only_correct': c_only,
                          'both_correct': sum(c[metric] and d[metric] for c, d in zip(local, hybrid)),
                          'neither_correct': sum(not c[metric] and not d[metric] for c, d in zip(local, hybrid)),
                          'D_accuracy': sum(row[metric] for row in hybrid) / len(hybrid),
                          'C_accuracy': sum(row[metric] for row in local) / len(local),
                          'delta_percentage_points': 100 * (d_only - c_only) / len(local)}
    return result


def main():
    config = read(CONFIG)
    code_files = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / 'sbhm').glob('*.py'))}
    code_hash = digest(code_files)
    require(code_hash == '326d805bde96047ca0553f8c33e5124a6c346fc11caaf3c28b4221277ef82140',
            'current native code differs from completed run')
    require(config['dtype'] == 'float32' and config['device'] == 'cuda:0' and config['layer'] == 14
            and config['window'] == 128 and config['rank'] == 8 and config['seed'] == 17
            and config['learning_rate'] == .0001 and config['reset_recurrent_at'] is None,
            'registered native recipe differs')
    require(config['limits'] == {'max_new_tokens': 32, 'max_prompt_tokens': 2048, 'max_train_tokens': 512,
                                 'prefill_chunk': 64, 'steps': 512}, 'registered shape/exposure limits differ')
    source = Path(config['source']['path'])
    source_actual = {str(path.relative_to(source)): sha(path) for path in sorted(source.rglob('*')) if path.is_file()}
    require(source_actual == config['source']['files'], 'current exact source inventory changed')
    require(all(path.resolve().is_relative_to(source.resolve()) for path in source.rglob('*') if path.is_file()),
            'escaping source link')
    require(not any(name.endswith(('.py', '.bin', '.pt', '.pth')) for name in source_actual), 'unsupported source format')
    source_config = read(source / 'config.json')
    require(source_config['architectures'] == ['Qwen3ForCausalLM'] and source_config['model_type'] == 'qwen3'
            and not source_config.get('auto_map') and not source_config.get('quantization_config')
            and not source_config.get('rope_scaling') and not source_config['use_sliding_window'], 'unsupported source architecture')
    require({key: source_config[key] for key in ('hidden_size', 'head_dim', 'num_attention_heads', 'num_key_value_heads',
                                                'num_hidden_layers', 'vocab_size')} ==
            {'hidden_size': 1024, 'head_dim': 128, 'num_attention_heads': 16,
             'num_key_value_heads': 8, 'num_hidden_layers': 28, 'vocab_size': 151936}, 'exact geometry differs')
    weights = header(source / 'model.safetensors')
    expected = {'model.embed_tokens.weight': [151936, 1024], 'model.norm.weight': [1024]}
    for layer in range(28):
        shapes = {'self_attn.q_proj.weight': [2048, 1024], 'self_attn.k_proj.weight': [1024, 1024],
                  'self_attn.v_proj.weight': [1024, 1024], 'self_attn.o_proj.weight': [1024, 2048],
                  'self_attn.q_norm.weight': [128], 'self_attn.k_norm.weight': [128],
                  'mlp.gate_proj.weight': [3072, 1024], 'mlp.up_proj.weight': [3072, 1024],
                  'mlp.down_proj.weight': [1024, 3072], 'input_layernorm.weight': [1024],
                  'post_attention_layernorm.weight': [1024]}
        expected.update({f'model.layers.{layer}.{key}': shape for key, shape in shapes.items()})
    require(set(weights) == set(expected) and all(weights[key]['shape'] == shape and weights[key]['dtype'] == 'F32'
                                                 for key, shape in expected.items()), 'source tensor names/shapes/dtypes differ')
    parameter_count = sum(math.prod(value['shape']) for value in weights.values())
    require(parameter_count == 596049920 and source_config['tie_word_embeddings'], 'source parameter/tie identity differs')
    manifest_path = Path(config['corpus_manifest'])
    manifest, design = read(manifest_path), read(manifest_path.parent / 'design.json')
    require(manifest['reviewed_for_sbhm'] is True and set(manifest['splits']) == {'train', 'dev'}, 'corpus not admitted')
    corpus = {}
    for split, entry in manifest['splits'].items():
        path = manifest_path.parent / entry['path']
        require(entry['role'] == split and entry['protected'] is False and sha(path) == entry['sha256'], 'split lock/role differs')
        corpus[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(corpus['train']) == 256 and len(corpus['dev']) == 512, 'corpus counts differ')
    require(sha(manifest_path.parent / manifest['design']['path']) == manifest['design']['sha256'], 'design lock differs')
    base_binding = read(RUN / 'A-qualification/receipt.json')['binding']
    global_binding = {'schema': 'SBHM-001/v1', 'code_sha256': code_hash, 'config_sha256': digest(config),
                      'source_sha256': digest(source_actual), 'corpus_sha256': sha(manifest_path),
                      'environment': base_binding['environment']}
    require(global_binding['environment']['versions']['transformers'] == '4.51.3'
            and global_binding['environment']['versions']['torch'] == '2.6.0+cu124'
            and global_binding['environment']['cuda'] == '12.4', 'native stack differs')
    grant = read(config['grant'])
    for key in ('source_sha256', 'code_sha256', 'config_sha256', 'corpus_sha256'):
        require(grant[key] == global_binding[key], 'grant binding differs')
    issued, expires = timestamp(grant['issued_utc']), timestamp(grant['expires_utc'])
    require(grant['approved_by'] and timedelta(0) < expires - issued <= timedelta(days=7), 'invalid operator grant')
    require({'qualify', 'teacher', 'distill', 'recover', 'evaluate'} <= set(grant['actions']), 'stage permission missing')

    def binding(arm, delta=None):
        return {**global_binding, 'arm': arm,
                'delta_sha256': sha(delta / 'delta.safetensors') if delta else None,
                'delta_manifest_sha256': sha(delta / 'delta.json') if delta else None}

    artifacts, receipts, requests = {}, {}, {}
    stages = ['A-qualification', 'teacher', 'A-evaluation']
    for arm in 'BCD':
        stages.extend(arm + '-' + stage for stage in ('initial-qualification', 'distill', 'distilled-qualification',
                                                     'recover', 'recovered-qualification', 'evaluation'))
    for name in stages:
        receipt_path, request_path = RUN / name / 'receipt.json', RUN / name / 'request.json'
        receipt, request = read(receipt_path), read(request_path)
        arm = name[0] if name != 'teacher' else 'A'
        delta = RUN / (arm + '-recover') if name.endswith(('recovered-qualification', 'evaluation')) and arm != 'A' else (
            RUN / (arm + '-distill') if name.endswith(('distilled-qualification', 'recover')) else None)
        expected_binding = binding(arm, delta)
        require(receipt['binding'] == request['binding'] == expected_binding, name + ': stage/variant binding differs')
        require(issued <= timestamp(request['created_utc']) < expires, name + ': operator grant stale at request')
        require(timestamp(receipt['created_utc']) >= timestamp(request['created_utc']), 'receipt precedes request')
        require(isinstance(receipt['allocator_cache_release_events'], list), 'allocator events missing')
        receipts[name], requests[name] = receipt, request
        artifacts[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        artifacts[str(request_path.relative_to(ROOT))] = sha(request_path)
    require(len(receipts) == 21, 'matrix stage missing')
    require(read(RUN / 'teacher/teacher.json') == receipts['teacher'], 'teacher receipt copies disagree')
    qualifiers = {}
    for name, receipt in receipts.items():
        if not name.endswith('qualification'):
            continue
        arm = receipt['binding']['arm']
        require(receipt['status'] == 'PASS_NATIVE_MECHANICS_ONLY' and receipt['identity_wrapper_max_abs_error'] == 0
                and receipt['capability_claim'] is False and receipt['native_training_qualified'] is False
                and receipt['max_prompt_tokens_checked'] == 2048, 'invalid qualification')
        trainable = 0 if arm == 'A' else 1131664 if arm == 'D' else 81920
        require(receipt['trainable_parameters'] == trainable and receipt['total_parameters'] == parameter_count + trainable,
                'qualification parameter count differs')
        require(set(receipt['parity']) == {'1:17', '64:2048'}, 'qualified paths differ')
        for shape, measurement in receipt['parity'].items():
            length = int(shape.split(':')[1])
            cache = 28 * 8192 * length if arm in 'AB' else 27 * 8192 * length + 8192 * min(length, 128) + (524288 if arm == 'D' else 0)
            require(measurement['cache_storage_bytes'] == cache, 'qualification allocated cache violates exact geometry')
            require(math.isfinite(measurement['max_abs_logit_error']) and measurement['max_abs_logit_error'] <= 3e-4,
                    'qualification parity error unexpectedly large')
        qualifiers[name] = {'receipt_sha256': sha(RUN / name / 'receipt.json'), 'parity': receipt['parity'],
                            'trainable_parameters': trainable}
    require(len(qualifiers) == 10, 'not all source/variant qualifications present')

    def fresh_for(qualification, stage):
        q, request = receipts[qualification], requests[stage]
        require(q['binding'] == request['binding'], stage + ': qualification belongs to a different exact variant')
        age = timestamp(request['created_utc']) - timestamp(q['created_utc'])
        require(timedelta(0) <= age <= timedelta(hours=24), stage + ': qualification stale at request')

    fresh_for('A-qualification', 'teacher')
    fresh_for('A-qualification', 'A-evaluation')
    order = list(range(256))
    random.Random(config['seed']).shuffle(order)
    consumed = [corpus['train'][order[step % 256]] for step in range(512)]
    teacher = receipts['teacher']
    require(teacher['status'] == 'TEACHER_TARGETS_CAPTURED' and teacher['split'] == 'train' and teacher['layer'] == 14,
            'teacher role or source differs')
    require(teacher['corpus_sha256'] == global_binding['corpus_sha256'], 'teacher corpus differs')
    require([entry['id'] for entry in teacher['entries']] == [row['id'] for row in consumed[:256]], 'teacher TRAIN inventory order differs')
    require(len({entry['id'] for entry in teacher['entries']}) == 256, 'duplicate teacher entry')
    files_total = 0
    for entry, row in zip(teacher['entries'], consumed[:256]):
        path = RUN / 'teacher' / entry['path']
        require(path.resolve().is_relative_to((RUN / 'teacher').resolve()) and path.suffix == '.safetensors', 'unsafe teacher target')
        require(sha(path) == entry['sha256'] and entry['row_sha256'] == digest(row), 'teacher target/row changed')
        tensors = header(path)
        length = design['rows'][row['id']]['token_geometry']['train_tokens']
        shapes = {'hidden': [1, length, 1024], 'target': [1, length, 1024],
                  'cos': [1, length, 128], 'sin': [1, length, 128]}
        require(set(tensors) == set(shapes) and all(tensors[key]['shape'] == shape and tensors[key]['dtype'] == 'F32'
                                                  for key, shape in shapes.items()), 'teacher tensor geometry differs')
        files_total += path.stat().st_size
    require(files_total == teacher['teacher_cache_file_bytes'] == 502904576 and files_total <= 512 * 2**20,
            'teacher cache byte cap/receipt differs')
    teacher_targets = {entry['path'] for entry in teacher['entries']}
    require({path.name for path in (RUN / 'teacher').glob('*.safetensors')} == teacher_targets, 'unlisted teacher target')
    trained, exposure = {}, {}
    for arm in 'BCD':
        fresh_for(arm + '-initial-qualification', arm + '-distill')
        fresh_for(arm + '-distilled-qualification', arm + '-recover')
        fresh_for(arm + '-recovered-qualification', arm + '-evaluation')
        for stage in ('distill', 'recover'):
            name, directory = arm + '-' + stage, RUN / (arm + '-' + stage)
            receipt, delta = receipts[name], read(directory / 'delta.json')
            require(receipt['status'] == 'TRAINING_COMPLETED_NOT_EFFICACY' and receipt['stage'] == stage
                    and receipt['resume_supported'] is False, 'training status/stage differs')
            require(receipt['corpus_sha256'] == global_binding['corpus_sha256'], 'training corpus differs')
            require(delta['record'] == receipt and delta['binding'] == receipt['binding'], 'delta training record differs')
            require(delta['sha256'] == sha(directory / 'delta.safetensors'), 'saved delta changed')
            tensors = header(directory / 'delta.safetensors')
            require(set(tensors) == set(receipt['parameter_names']) == set(delta['parameters']), 'delta parameter inventory differs')
            require(all(tensors[key]['shape'] == delta['parameters'][key] and tensors[key]['dtype'] == 'F32' for key in tensors),
                    'delta parameter shapes/dtypes differ')
            require(sum(math.prod(value['shape']) for value in tensors.values()) == receipt['trainable_parameters']
                    == (1131664 if arm == 'D' else 81920), 'delta trainable size differs')
            require(len(receipt['records']) == 512, 'training step count differs')
            for step, (record, row) in enumerate(zip(receipt['records'], consumed)):
                g = design['rows'][row['id']]['token_geometry']
                require(record['step'] == step and record['row_id'] == row['id'] and record['input_tokens'] == g['train_tokens'],
                        'training exposure/order/token count differs')
                require(record['supervised_positions'] == (g['train_tokens'] if stage == 'distill' else g['target_tokens'] + 1),
                        'training target/EOS exposure differs')
                require(all(type(record[key]) in (int, float) and math.isfinite(record[key]) and record[key] >= 0
                            for key in ('loss', 'gradient_norm')), 'nonfinite/invalid completed training measurement')
            trace = [(x['row_id'], x['input_tokens'], x['supervised_positions']) for x in receipt['records']]
            if stage in exposure:
                require(trace == exposure[stage], 'training exposure unmatched across controls')
            exposure[stage] = trace
            require(Counter(x['row_id'] for x in receipt['records']) == Counter({row['id']: 2 for row in corpus['train']}),
                    'TRAIN rows are not all consumed twice')
            require(receipt['parent_record'] == (None if stage == 'distill' else receipts[arm + '-distill']), 'distillation/recovery chain differs')
            require(receipt['optimizer'] == 'AdamW(.9,.95), wd=0, eps=1e-8, clip=1', 'optimizer recipe differs')
            trained[name] = {'steps': 512, 'unique_train_rows': 256, 'exposures_per_row': 2,
                             'input_tokens_total': sum(x['input_tokens'] for x in receipt['records']),
                             'supervised_positions_total': sum(x['supervised_positions'] for x in receipt['records']),
                             'input_tokens_max': max(x['input_tokens'] for x in receipt['records']),
                             'loss_first': receipt['records'][0]['loss'], 'loss_last': receipt['records'][-1]['loss'],
                             'loss_max': max(x['loss'] for x in receipt['records']),
                             'gradient_norm_max': max(x['gradient_norm'] for x in receipt['records']),
                             'delta_sha256': delta['sha256'], 'delta_manifest_sha256': sha(directory / 'delta.json'),
                             'delta_parameter_tensor_count': len(tensors)}
            artifacts[str((directory / 'delta.json').relative_to(ROOT))] = sha(directory / 'delta.json')
            artifacts[str((directory / 'delta.safetensors').relative_to(ROOT))] = delta['sha256']
    predictions = {}
    timing_and_storage = {}
    for arm in 'ABCD':
        receipt = receipts[arm + '-evaluation']
        require(receipt['status'] == 'DEV_MEASUREMENT_ONLY' and receipt['corpus_sha256'] == global_binding['corpus_sha256']
                and receipt['evaluation_groups'] == manifest['evaluation_groups'], 'evaluation corpus/groups/status differs')
        require(receipt['semantic_accuracy'] is None and receipt['speedup_claim'] is False and receipt['energy_measured'] is False,
                'unsupported evaluation claim')
        require(receipt['adaptation_record'] == (None if arm == 'A' else receipts[arm + '-recover']), 'evaluation uses wrong adaptation')
        require(len(receipt['predictions']) == 512, 'DEV row count differs')
        for prediction, row in zip(receipt['predictions'], corpus['dev']):
            require(all(prediction[key] == row[key] for key in ('id', 'family_id', 'target')), 'DEV row/target/order differs')
            require(type(prediction['strict_exact_match']) is bool and type(prediction['normalized_exact_match']) is bool
                    and prediction['strict_exact_match'] == (prediction['prediction'] == row['target'])
                    and prediction['normalized_exact_match'] == (prediction['prediction'].strip().casefold() == row['target'].strip().casefold()),
                    'raw prediction score differs')
            require(prediction['reset_recurrent_at'] is None, 'reset diagnostic pooled into main run')
            require(prediction['prompt_tokens'] == design['rows'][row['id']]['token_geometry']['prompt_tokens'], 'DEV prompt token count differs')
            ids = prediction['generated_ids']
            require(prediction['generated_tokens'] == len(ids) and 1 <= len(ids) <= 32
                    and all(type(token) is int and 0 <= token < 151936 for token in ids), 'generation token contract differs')
            require(prediction['stop_reason'] == ('eos' if ids[-1] == source_config['eos_token_id'] else 'max_new_tokens'), 'generation stopping differs')
            require(prediction['stop_reason'] != 'max_new_tokens' or len(ids) == 32, 'non-EOS generation stopped early')
            cached_length = prediction['prompt_tokens'] + len(ids) - 1
            cache = 28 * 8192 * cached_length if arm in 'AB' else 27 * 8192 * cached_length + 8192 * min(cached_length, 128) + (524288 if arm == 'D' else 0)
            require(prediction['cache_storage_bytes'] == cache, 'DEV actual cache bytes do not match bounded-site geometry')
            require(prediction['rss_sample_max_bytes'] <= 4 * 2**30, 'sampled worker RSS exceeds unchanged ceiling')
            require(prediction['backend'] == ('native_sdpa' if arm in 'AB' else 'mixed_sdpa_and_sequential_reference'), 'backend identity differs')
        for metric in ('strict_exact_match', 'normalized_exact_match'):
            require(receipt[metric] == sum(x[metric] for x in receipt['predictions']) / 512, 'aggregate DEV metric differs')
        predictions[arm] = receipt['predictions']
        timing_and_storage[arm] = {'rows': 512,
            'cache_storage_bytes_min': min(x['cache_storage_bytes'] for x in receipt['predictions']),
            'cache_storage_bytes_max': max(x['cache_storage_bytes'] for x in receipt['predictions']),
            'peak_vram_allocated_bytes_max': max(x['peak_vram_allocated_bytes'] for x in receipt['predictions']),
            'peak_vram_reserved_bytes_max': max(x['peak_vram_reserved_bytes'] for x in receipt['predictions']),
            'sampled_rss_bytes_max': max(x['rss_sample_max_bytes'] for x in receipt['predictions'])}
    compare = read(RUN / 'comparison.json')
    require(compare['status'] == 'DESCRIPTIVE_PAIRED_DEV_ONLY' and compare['promotion'] is False, 'unsupported comparison status')
    family_ids = sorted({row['family_id'] for row in corpus['dev']})
    for arm in 'ABCD':
        expected_summary = summary(predictions['A'], predictions[arm])
        expected_summary['per_family'] = {family: summary([r for r in predictions['A'] if r['family_id'] == family],
                                                         [r for r in predictions[arm] if r['family_id'] == family]) for family in family_ids}
        require(compare['arms'][arm] == expected_summary, 'independent overall/family retention calculation differs')
    overall_contrast = contrast(predictions['C'], predictions['D'])
    overall_contrast['per_family'] = {family: contrast([r for r in predictions['C'] if r['family_id'] == family],
                                                     [r for r in predictions['D'] if r['family_id'] == family]) for family in family_ids}
    require(compare['hybrid_vs_local'] == overall_contrast, 'independent hybrid/local calculation differs')
    grouped = {}
    for group, families in manifest['evaluation_groups'].items():
        selected = {arm: [r for r in rows if r['family_id'] in families] for arm, rows in predictions.items()}
        expected_group = {'family_ids': families, 'arms': {arm: summary(selected['A'], rows) for arm, rows in selected.items()},
                          'hybrid_vs_local': contrast(selected['C'], selected['D'])}
        require(compare['evaluation_groups'][group] == expected_group, 'independent grouped primary/retention calculation differs')
        grouped[group] = {'rows': len(selected['A']), 'correct': {arm: sum(r['strict_exact_match'] for r in rows) for arm, rows in selected.items()},
                          'hybrid_vs_local': expected_group['hybrid_vs_local'],
                          'original_model_retention': {arm: value['original_model_retention']['strict_exact_match']
                                                       for arm, value in expected_group['arms'].items()}}
    require(receipts['A-qualification']['trainable_parameters'] == 0
            and receipts['A-evaluation']['binding']['delta_sha256'] is None
            and receipts['A-evaluation']['adaptation_record'] is None, 'source A was adapted')
    register = read(Path(__file__).resolve().parent / 'memory-efficient-rerun-registration.json')
    equivalence = read(Path(__file__).resolve().parent / 'memory-equivalence-review.json')
    require(register['binding'] == {key: global_binding[key] for key in ('source_sha256', 'code_sha256', 'config_sha256', 'corpus_sha256')},
            'fresh rerun registration differs')
    full_tests = Path(__file__).resolve().parent / 'recovery_memory_qualification/full-with-cache.txt'
    require('136 passed' in full_tests.read_text(encoding='utf-8') and 'skipped' not in full_tests.read_text(encoding='utf-8'),
            'combined CPU check evidence incomplete')
    require(sha(full_tests) == equivalence['completed_cpu_evidence']['full_with_cache_output_sha256'], 'CPU evidence changed')
    require(sha(ROOT / 'sbhm/protocol.py') == equivalence['input_sha256']['sbhm\\protocol.py']
            and sha(ROOT / 'tests/test_allocator_pressure.py') == equivalence['input_sha256']['tests\\test_allocator_pressure.py'],
            'resource guard qualification changed')
    cache_events = {name: len(receipt['allocator_cache_release_events']) for name, receipt in receipts.items()}
    require(all(count == 0 for count in cache_events.values()), 'unexpected allocator intervention requires additional review')
    artifacts[str((RUN / 'comparison.json').relative_to(ROOT))] = sha(RUN / 'comparison.json')
    result = {'status': 'PASS_READ_ONLY_COMPLETED_NATIVE_MATRIX_AUDIT', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer': 'Codex independent geometry/completion-audit agent', 'audit_script_sha256': sha(__file__),
              'run_root': str(RUN), 'config_path': str(CONFIG), 'binding': global_binding,
              'model_or_gpu_workload_executed_by_auditor': False, 'native_data_config_or_source_modified_by_auditor': False,
              'matrix_exit_code_reported_by_parent': 0, 'exit_status_reexecuted_by_auditor': False,
              'requirements': {'all_21_stage_receipts_plus_native_comparison_present': True,
                  'exact_current_source_code_config_corpus_environment_binding': True,
                  'all_10_initial_distilled_recovered_native_mechanics_qualifications_pass': True,
                  'all_variant_qualifications_fresh_at_next_stage_request': True,
                  'all_6_training_stages_complete_512_finite_steps': True,
                  'all_256_train_rows_consumed_twice_each_stage_matched_order_tokens_labels': True,
                  'all_6_saved_deltas_hashed_header_checked_and_bound_to_qualification_evaluation': True,
                  'teacher_256_files_actual_hashes_shapes_bytes_valid_under_512_mib': True,
                  'all_4_evaluations_complete_512_rows_paired_with_raw_score_stop_cache_checks': True,
                  'source_A_unadapted_and_no_recurrent_reset_in_primary': True,
                  'native_comparison_independently_recomputed_all_counts_and_retention': True,
                  'unchanged_resource_guards_and_136_cpu_checks_documented': True},
              'source_geometry': {'source_path': str(source), 'source_file_count': len(source_actual),
                  'weight_sha256': source_actual['model.safetensors'], 'weight_file_bytes': (source / 'model.safetensors').stat().st_size,
                  'source_tensor_count': len(weights), 'source_tensor_dtype': 'F32', 'parameter_count': parameter_count,
                  'parameter_bytes_fp32': parameter_count * 4, 'layers': 28, 'hidden_size': 1024,
                  'query_heads': 16, 'kv_heads': 8, 'head_dim': 128, 'query_heads_per_kv_group': 2,
                  'q_projection': [2048, 1024], 'k_and_v_projections': [1024, 1024], 'o_projection': [1024, 2048],
                  'q_and_k_norm': [128], 'lm_head_tied_to_embedding': True},
              'qualifications': qualifiers, 'training_stages': trained,
              'teacher': {'unique_train_rows': 256, 'actual_target_file_bytes': files_total,
                          'cap_bytes': 512 * 2**20, 'margin_bytes': 512 * 2**20 - files_total},
              'observed_training_fit': {'actual_sequence_tokens_max': max(x['input_tokens_max'] for x in trained.values()),
                  'registered_training_cap_tokens': 512, 'dtype': 'float32', 'batch_size': 1,
                  'fit_established_for_consumed_shapes_only': True, 'all_registered_512_token_training_shapes_proved': False},
              'allocator_cache_release_event_counts': cache_events,
              'persistent_cache_at_2048_tokens': {'A_and_B_bytes': 469762048, 'C_bytes': 454033408,
                  'D_bytes': 454557696, 'D_converted_site_bytes': 1572864,
                  'D_recurrent_state_bytes': 524288, 'D_savings_vs_A_bytes': 15204352,
                  'D_total_model_cache_reduction_fraction': 15204352 / 469762048},
              'grouped_behavioral_counts': grouped, 'evaluation_storage_measurement_ranges': timing_and_storage,
              'artifact_sha256': artifacts,
              'caveats': ['A/B/C from the interrupted older code run are not pooled; this audit uses only the fresh completed matrix.',
                  'One source, one selected attention site, one seed and one synthetic DEV corpus; no confidence intervals or promotion claim.',
                  'Counterfactual and near/distant rows share worlds and are not independent observations.',
                  'Remaining 27 full-attention sites can transmit distant evidence; D has additional trainable parameters and no recurrent-reset diagnostic here.',
                  'Compact JSON exact matching includes formatting; normalized matching is strip/casefold, not semantic JSON accuracy.',
                  'Original-model retention concerns 64 narrow synthetic rows and 22 A-correct rows, not broad capability preservation.',
                  'Qualification PASS establishes inference mechanics; completed training establishes this actual-shape execution, not all future training fit.',
                  'Resource guards check stage/runtime boundaries; RSS is sampled, not an allocator-level transient peak or continuous free-VRAM certificate.',
                  'The Python token-scanning backend is a reference, and these receipts do not establish optimized-kernel speed, energy, or large total-memory savings.',
                  'Selected-logit CPU parity allows normal FP32 rounding and does not prove bitwise CUDA equality with the older full-logit trajectory.'],
              'objective_completion_claim': False, 'promotion': False}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'source_weight_sha256': result['source_geometry']['weight_sha256'],
                      'native_stages': len(receipts), 'qualifications': len(qualifiers), 'training_stages': len(trained),
                      'teacher_target_file_bytes': files_total, 'actual_training_max_tokens': result['observed_training_fit']['actual_sequence_tokens_max'],
                      'primary_correct': grouped['distant_heldout']['correct'], 'retention_correct': grouped['retention']['correct'],
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
