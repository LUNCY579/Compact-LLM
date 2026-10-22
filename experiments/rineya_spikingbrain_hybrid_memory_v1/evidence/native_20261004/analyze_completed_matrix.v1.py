"""Read-only supplement to the completed, locked four-arm DEV comparison.

Standard library only: no model/tokenizer imports, inference, training, or GPU.
No accuracy metric is replaced and no threshold/configuration is selected.
Counterfactual rows share worlds; outputs are descriptive without intervals.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


METRICS = ('strict_exact_match', 'normalized_exact_match')
RUNTIME = ('prompt_tokens', 'generated_tokens', 'cache_storage_bytes', 'model_ttft_seconds',
           'decode_seconds_excluding_first', 'decode_tokens_per_second',
           'peak_vram_allocated_bytes', 'peak_vram_reserved_bytes', 'rss_sample_max_bytes')
EXPECTED_ROLES = {'distant_heldout': 224, 'near_control': 224, 'retention': 64}
EVIDENCE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EVIDENCE_ROOT.parent.parent


class AnalysisBlocked(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise AnalysisBlocked(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bound_file(root, reference):
    name = reference['path']
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and '..' not in Path(name).parts and
            path.is_relative_to(root.resolve()) and path.is_file(), 'unsafe/missing locked metadata path')
    require(sha(path) == reference['sha256'], f'locked metadata hash changed: {name}')
    return path


def matches(value, expected):
    return type(value) in (int, float) and math.isfinite(value) and math.isclose(value, expected, abs_tol=1e-12, rel_tol=0)


def distribution(values):
    present = [value for value in values if value is not None]
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in present),
            'invalid/nonfinite runtime value')
    if not present:
        return {'count': 0, 'missing': len(values), 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'count': len(present), 'missing': len(values) - len(present),
            'mean': statistics.mean(present), 'median': statistics.median(present),
            'min': min(present), 'max': max(present)}


def exact_summary(rows, baseline, metric):
    n = len(rows)
    baseline_correct = sum(row[metric] for row in baseline)
    retained = sum(a[metric] and row[metric] for a, row in zip(baseline, rows))
    return {'rows': n, 'correct': sum(row[metric] for row in rows),
            'accuracy': sum(row[metric] for row in rows) / n if n else None,
            'A_correct': baseline_correct, 'retained_A_correct': retained,
            'damaged_A_correct': baseline_correct - retained,
            'repaired_A_incorrect': sum(not a[metric] and row[metric] for a, row in zip(baseline, rows)),
            'A_correct_retention_rate': retained / baseline_correct if baseline_correct else None}


def contrast(local, hybrid, metric):
    n = len(local)
    d_only = sum(not c[metric] and d[metric] for c, d in zip(local, hybrid))
    c_only = sum(c[metric] and not d[metric] for c, d in zip(local, hybrid))
    return {'paired_rows': n, 'D_only_correct': d_only, 'C_only_correct': c_only,
            'both_correct': sum(c[metric] and d[metric] for c, d in zip(local, hybrid)),
            'neither_correct': sum(not c[metric] and not d[metric] for c, d in zip(local, hybrid)),
            'D_accuracy': sum(row[metric] for row in hybrid) / n if n else None,
            'C_accuracy': sum(row[metric] for row in local) / n if n else None,
            'delta_percentage_points': 100 * (d_only - c_only) / n if n else None}


def cached_prefix(row):
    return row['prompt_tokens'] + row['generated_tokens'] - 1


def runtime_summary(rows):
    decoded = sum(max(0, row['generated_tokens'] - 1) for row in rows)
    decode_time = sum(row['decode_seconds_excluding_first'] for row in rows)
    return {'rows': len(rows), 'values': {key: distribution([row[key] for row in rows]) for key in RUNTIME},
            'aggregate_decode_tokens_excluding_first': decoded,
            'aggregate_decode_seconds_excluding_first': decode_time,
            'aggregate_decode_tokens_per_second': decoded / decode_time if decoded and decode_time else None,
            'stop_reasons': dict(Counter(row['stop_reason'] for row in rows)),
            'backends': dict(Counter(row['backend'] for row in rows)),
            'cached_prefix_tokens': distribution([cached_prefix(row) for row in rows])}


def summarize(ids, predictions):
    selected = {arm: [predictions[arm][row_id] for row_id in ids] for arm in 'ABCD'}
    return {'rows': len(ids),
            'arms': {arm: {metric: exact_summary(rows, selected['A'], metric) for metric in METRICS}
                     for arm, rows in selected.items()},
            'hybrid_vs_local': {metric: contrast(selected['C'], selected['D'], metric) for metric in METRICS},
            'runtime': {arm: runtime_summary(rows) for arm, rows in selected.items()}}


def validate_inputs(run_root, config_path):
    required = [run_root / 'comparison.json']
    required += [run_root / f'{arm}-evaluation' / 'receipt.json' for arm in 'ABCD']
    required += [run_root / f'{arm}-recovered-qualification' / 'receipt.json' for arm in 'BCD']
    required += [run_root / 'A-qualification' / 'receipt.json']
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, 'matrix is incomplete; missing: ' + ', '.join(missing))
    config = read(config_path)
    manifest_path = Path(config['corpus_manifest'])
    manifest = read(manifest_path)
    require(manifest.get('reviewed_for_sbhm') is True, 'corpus was not admitted')
    design_path = bound_file(manifest_path.parent, manifest['design'])
    review_path = bound_file(manifest_path.parent, manifest['review'])
    design, review = read(design_path), read(review_path)
    locked = {}
    for split in ('train', 'dev'):
        entry = manifest['splits'][split]
        require(entry['role'] == split and entry['protected'] is False, 'unsupported/protected split')
        path = bound_file(manifest_path.parent, entry)
        locked[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(locked['train']) == 256 and len(locked['dev']) == 512, 'registered v2 row counts changed')
    require(len({row['id'] for row in locked['dev']}) == 512, 'duplicate DEV IDs')
    require(len({row['id'] for row in locked['train']}) == 256, 'duplicate TRAIN IDs')
    config_hash, corpus_hash = digest(config), sha(manifest_path)
    source_hash = digest(config['source']['files'])
    code_hash = digest({str(path.relative_to(EXPERIMENT_ROOT)): sha(path)
                        for path in sorted((EXPERIMENT_ROOT / 'sbhm').glob('*.py'))})
    expected_binding = {'config_sha256': config_hash, 'corpus_sha256': corpus_hash,
                        'source_sha256': source_hash, 'code_sha256': code_hash}
    source_config_path = Path(config['source']['path']) / 'config.json'
    require(sha(source_config_path) == config['source']['files']['config.json'], 'source geometry metadata changed')
    source_config = read(source_config_path)
    require(config['dtype'] == 'float32' and config['reset_recurrent_at'] is None,
            'this descriptive analysis is frozen for the FP32 main v2 matrix')
    comparison = read(run_root / 'comparison.json')
    require(comparison['status'] == 'DESCRIPTIVE_PAIRED_DEV_ONLY' and comparison['promotion'] is False,
            'completed descriptive comparison required')
    reports, predictions, exposure, files = {}, {}, None, {}
    for arm in 'ABCD':
        receipt_path = run_root / f'{arm}-evaluation' / 'receipt.json'
        report = read(receipt_path)
        require(report['status'] == 'DEV_MEASUREMENT_ONLY' and report['binding']['arm'] == arm,
                'completed exact arm evaluation required')
        for key, value in expected_binding.items():
            require(report['binding'][key] == value, f'evaluation has changed {key}')
        require(report['corpus_sha256'] == corpus_hash and report['evaluation_groups'] == manifest['evaluation_groups'],
                'evaluation differs from locked corpus/groups')
        if reports:
            require(report['binding']['environment'] == reports['A']['binding']['environment'], 'environment mismatch')
        qualifier_path = run_root / ('A-qualification' if arm == 'A' else f'{arm}-recovered-qualification') / 'receipt.json'
        qualifier = read(qualifier_path)
        require(qualifier['status'] == 'PASS_NATIVE_MECHANICS_ONLY' and qualifier['binding'] == report['binding'],
                'evaluation does not match its completed exact-variant qualification')
        require(read(run_root / f'{arm}-evaluation' / 'request.json')['binding'] == report['binding'],
                'evaluation request binding changed')
        requested = datetime.fromisoformat(read(run_root / f'{arm}-evaluation' / 'request.json')['created_utc'])
        qualified = datetime.fromisoformat(qualifier['created_utc'])
        require(requested.tzinfo is not None and qualified.tzinfo is not None and
                0 <= (requested - qualified).total_seconds() <= 24 * 3600,
                'qualification was not fresh when evaluation was requested')
        if arm == 'A':
            require(report['binding']['delta_sha256'] is None and report['adaptation_record'] is None, 'A was adapted')
        else:
            adaptation = read(run_root / f'{arm}-recover' / 'delta.json')
            require(sha(run_root / f'{arm}-recover' / 'delta.safetensors') == report['binding']['delta_sha256'] == adaptation['sha256'],
                    'recovered delta hash mismatch')
            require(sha(run_root / f'{arm}-recover' / 'delta.json') == report['binding']['delta_manifest_sha256'],
                    'recovered delta manifest mismatch')
            require(report['adaptation_record'] == adaptation['record'] == read(run_root / f'{arm}-recover' / 'receipt.json'),
                    'recovery record mismatch')
            recovery = report['adaptation_record']
            distill = recovery['parent_record']
            require(distill == read(run_root / f'{arm}-distill' / 'receipt.json'), 'distillation record mismatch')
            indices = list(range(len(locked['train'])))
            random.Random(config['seed']).shuffle(indices)
            expected_rows = [locked['train'][indices[i % len(indices)]] for i in range(config['limits']['steps'])]
            trace = []
            for stage, label in ((distill, 'distill'), (recovery, 'recover')):
                require(stage['status'] == 'TRAINING_COMPLETED_NOT_EFFICACY' and stage['stage'] == label,
                        'unfinished adaptation stage')
                require(len(stage['records']) == len(expected_rows), 'training step count mismatch')
                for index, (record, row) in enumerate(zip(stage['records'], expected_rows)):
                    geometry = design['rows'][row['id']]['token_geometry']
                    expected_supervised = geometry['train_tokens'] if label == 'distill' else geometry['target_tokens'] + 1
                    require(record['step'] == index and record['row_id'] == row['id'] and
                            record['input_tokens'] == geometry['train_tokens'] and
                            record['supervised_positions'] == expected_supervised, 'registered exposure/token shape changed')
                    require(all(type(record[key]) in (int, float) and math.isfinite(record[key]) for key in ('loss', 'gradient_norm')),
                            'nonfinite training record')
                trace.append([(row['row_id'], row['input_tokens'], row['supervised_positions']) for row in stage['records']])
            require(exposure is None or trace == exposure, 'unequal B/C/D exposure')
            exposure = trace
        require(len(report['predictions']) == len(locked['dev']), 'incomplete DEV evaluation')
        by_id = {}
        for row, expected in zip(report['predictions'], locked['dev']):
            require((row['id'], row['family_id'], row['target']) ==
                    (expected['id'], expected['family_id'], expected['target']), 'DEV row/target/order mismatch')
            info = design['rows'][row['id']]
            require(info['row_sha256'] == digest(expected) and info['split'] == 'dev', 'locked row metadata mismatch')
            require(type(row['prediction']) is str and row['reset_recurrent_at'] is None, 'invalid prediction/reset diagnostic')
            calculated = (row['prediction'] == row['target'], row['prediction'].strip().casefold() == row['target'].strip().casefold())
            require(all(type(row[metric]) is bool and row[metric] == value for metric, value in zip(METRICS, calculated)),
                    'prediction scoring does not match raw text')
            require(row['prompt_tokens'] == info['token_geometry']['prompt_tokens'], 'prompt token shape differs from locked tokenizer audit')
            require(type(row['generated_tokens']) is int and 1 <= row['generated_tokens'] <= config['limits']['max_new_tokens'] and
                    len(row['generated_ids']) == row['generated_tokens'], 'invalid generated token accounting')
            require(all(type(token) is int and 0 <= token < source_config['vocab_size'] for token in row['generated_ids']),
                    'invalid generated token IDs')
            require(row['stop_reason'] in ('eos', 'max_new_tokens'), 'unexpected stop reason')
            require((row['stop_reason'] == 'eos') == (row['generated_ids'][-1] == design['tokenizer']['eos_token_id']),
                    'stop reason does not match source EOS token')
            for key in RUNTIME:
                value = row[key]
                require(value is None and key == 'decode_tokens_per_second' or
                        type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid runtime measurement')
            by_id[row['id']] = row
        for metric in METRICS:
            require(matches(report[metric], sum(row[metric] for row in by_id.values()) / len(by_id)), 'invalid exact-match aggregate')
        reports[arm], predictions[arm] = report, by_id
        files[str(receipt_path)] = sha(receipt_path)
        files[str(qualifier_path)] = sha(qualifier_path)
    return config, manifest, design, review, source_config, comparison, reports, predictions, expected_binding, files


def analyze(run_root, config_path):
    config, manifest, design, review, source, main, reports, predictions, binding, files = validate_inputs(run_root, config_path)
    metadata = {row_id: info for row_id, info in design['rows'].items() if info['split'] == 'dev'}
    ids = list(predictions['A'])
    require(Counter(metadata[row_id]['role'] for row_id in ids) == Counter(EXPECTED_ROLES), 'registered DEV group counts changed')
    grouped = {role: [row_id for row_id in ids if metadata[row_id]['role'] == role] for role in EXPECTED_ROLES}
    for role, group_ids in grouped.items():
        require(sorted({metadata[row_id]['family_id'] for row_id in group_ids}) == manifest['evaluation_groups'][role],
                'metadata group partition differs from manifest')
    subset_summaries = {role: summarize(group_ids, predictions) for role, group_ids in grouped.items()}
    for role, value in subset_summaries.items():
        for metric in METRICS:
            require(main['evaluation_groups'][role]['hybrid_vs_local'][metric] == value['hybrid_vs_local'][metric],
                    'native comparison contrast differs from independently recomputed rows')
            for arm in 'ABCD':
                native = main['evaluation_groups'][role]['arms'][arm]
                actual = value['arms'][arm][metric]
                require(native['rows'] == actual['rows'] and matches(native[metric], actual['accuracy']),
                        'native comparison arm accuracy differs from rows')
                retention = native['original_model_retention'][metric]
                require((retention['baseline_correct'], retention['retained'], retention['damages'], retention['repairs'], retention['retention_rate']) ==
                        (actual['A_correct'], actual['retained_A_correct'], actual['damaged_A_correct'],
                         actual['repaired_A_incorrect'], actual['A_correct_retention_rate']),
                        'native comparison source-correct retention differs from rows')
    by_task = {}
    for role, group_ids in grouped.items():
        by_task[role] = {category: summarize([row_id for row_id in group_ids if metadata[row_id]['category'] == category], predictions)
                         for category in sorted({metadata[row_id]['category'] for row_id in group_ids})}
    by_distance = {}
    by_task_distance = {}
    for role in ('distant_heldout', 'near_control'):
        band_ids = defaultdict(list)
        for row_id in grouped[role]:
            band = '-'.join(map(str, metadata[row_id]['token_geometry']['registered_gap_band']))
            band_ids[band].append(row_id)
        by_distance[role] = {band: summarize(selected, predictions) for band, selected in sorted(band_ids.items())}
        for band, selected in band_ids.items():
            by_distance[role][band]['actual_locked_evidence_to_query_gap_tokens'] = distribution([
                metadata[row_id]['token_geometry']['evidence_to_query_gap_tokens'] for row_id in selected])
        by_task_distance[role] = {
            category: {band: summarize([row_id for row_id in selected if metadata[row_id]['category'] == category], predictions)
                       for band, selected in sorted(band_ids.items())}
            for category in sorted({metadata[row_id]['category'] for row_id in grouped[role]})}
    worlds = defaultdict(list)
    for row_id in ids:
        worlds[metadata[row_id]['world_id']].append(row_id)
    world_rows = []
    cf_results = defaultdict(Counter)
    position_results = {arm: Counter() for arm in 'ABCD'}
    world_deltas = defaultdict(list)
    for world_id, world_ids in sorted(worlds.items()):
        first = metadata[world_ids[0]]
        if first['role'] == 'retention':
            continue
        require(len(world_ids) == 4, 'long-context world must have four counterfactual/position rows')
        roles = {role: sorted([row_id for row_id in world_ids if metadata[row_id]['role'] == role],
                              key=lambda row_id: metadata[row_id]['counterfactual_id'])
                 for role in ('distant_heldout', 'near_control')}
        require(all(len(selected) == 2 for selected in roles.values()), 'world lacks paired near/distant rows')
        entry = {'world_id': world_id, 'category': first['category'],
                 'registered_gap_band': first['token_geometry']['registered_gap_band'], 'row_ids': roles, 'arms': {}}
        for arm in 'ABCD':
            entry['arms'][arm] = {}
            for role, selected in roles.items():
                rows = [predictions[arm][row_id] for row_id in selected]
                require(rows[0]['target'] != rows[1]['target'], 'counterfactual truth did not change')
                entry['arms'][arm][role] = {metric: sum(row[metric] for row in rows) / 2 for metric in METRICS}
                for metric in METRICS:
                    correct = sum(row[metric] for row in rows)
                    counter = cf_results[(arm, role, metric)]
                    counter['worlds'] += 1
                    counter[('neither_correct', 'one_correct', 'both_correct')[correct]] += 1
                    counter['unchanged_raw_prediction_despite_changed_truth'] += rows[0]['prediction'] == rows[1]['prediction']
            for distant_id, near_id in zip(roles['distant_heldout'], roles['near_control']):
                distant, near = predictions[arm][distant_id], predictions[arm][near_id]
                require(metadata[distant_id]['position_pair_id'] == metadata[near_id]['position_pair_id'] and
                        distant['target'] == near['target'] and distant['prompt_tokens'] == near['prompt_tokens'],
                        'position control is not matched')
                for metric in METRICS:
                    counter = position_results[arm]
                    counter[f'{metric}:pairs'] += 1
                    counter[f'{metric}:near_only_correct'] += near[metric] and not distant[metric]
                    counter[f'{metric}:distant_only_correct'] += distant[metric] and not near[metric]
                    counter[f'{metric}:both_correct'] += distant[metric] and near[metric]
                    counter[f'{metric}:neither_correct'] += not distant[metric] and not near[metric]
        entry['hybrid_vs_local'] = {}
        for metric in METRICS:
            distant_delta = entry['arms']['D']['distant_heldout'][metric] - entry['arms']['C']['distant_heldout'][metric]
            near_delta = entry['arms']['D']['near_control'][metric] - entry['arms']['C']['near_control'][metric]
            increment = distant_delta - near_delta
            entry['hybrid_vs_local'][metric] = {'distant_delta_percentage_points': distant_delta * 100,
                                               'near_delta_percentage_points': near_delta * 100,
                                               'distance_specific_increment_percentage_points': increment * 100}
            world_deltas[(metric, 'distant')].append(distant_delta)
            world_deltas[(metric, 'near')].append(near_delta)
            world_deltas[(metric, 'distance_specific_increment')].append(increment)
        world_rows.append(entry)
    require(len(world_rows) == 112, 'registered long-context world count changed')
    cluster_summary = {}
    for metric in METRICS:
        cluster_summary[metric] = {}
        for contrast_name in ('distant', 'near', 'distance_specific_increment'):
            values = world_deltas[(metric, contrast_name)]
            cluster_summary[metric][contrast_name] = {'worlds': len(values),
                'mean_delta_percentage_points': statistics.mean(values) * 100,
                'worlds_positive': sum(value > 0 for value in values),
                'worlds_zero': sum(value == 0 for value in values),
                'worlds_negative': sum(value < 0 for value in values),
                'independent_row_assumption': False, 'confidence_interval': None}
    full_site_per_token = 2 * source['num_key_value_heads'] * source['head_dim'] * 4
    recurrent_state = source['num_key_value_heads'] * source['head_dim'] ** 2 * 4
    cache_validation, cache_pairs = {}, {}
    for arm in 'ABCD':
        mismatches = []
        for row_id in ids:
            row = predictions[arm][row_id]
            length = cached_prefix(row)
            expected = source['num_hidden_layers'] * full_site_per_token * length
            if arm in 'CD':
                expected = (source['num_hidden_layers'] - 1) * full_site_per_token * length + full_site_per_token * min(length, config['window'])
                if arm == 'D':
                    expected += recurrent_state
            if row['cache_storage_bytes'] != expected:
                mismatches.append({'id': row_id, 'measured_bytes': row['cache_storage_bytes'], 'expected_bytes': expected})
        cache_validation[arm] = {'rows': len(ids), 'formula_mismatches': mismatches,
                                 'formula_is_source_geometry_check_not_extra_per_site_measurement': True}
    for left, right in (('C', 'D'), ('A', 'D'), ('A', 'C')):
        equal_length = [row_id for row_id in ids if cached_prefix(predictions[left][row_id]) == cached_prefix(predictions[right][row_id])]
        cache_pairs[f'{right}_minus_{left}'] = {'equal_cached_length_rows': len(equal_length),
            'unequal_cached_length_rows': len(ids) - len(equal_length),
            'actual_byte_differences_on_equal_length_rows': distribution([
                abs(predictions[right][row_id]['cache_storage_bytes'] - predictions[left][row_id]['cache_storage_bytes'])
                for row_id in equal_length]),
            'signed_actual_byte_difference_mean': statistics.mean([
                predictions[right][row_id]['cache_storage_bytes'] - predictions[left][row_id]['cache_storage_bytes']
                for row_id in equal_length]) if equal_length else None}
    cf_serialized = {arm: {role: {metric: dict(cf_results[(arm, role, metric)]) for metric in METRICS}
                          for role in ('distant_heldout', 'near_control')} for arm in 'ABCD'}
    training = {arm: {'trainable_parameters': reports[arm]['adaptation_record']['trainable_parameters'],
                      'distillation': distribution([row['loss'] for row in reports[arm]['adaptation_record']['parent_record']['records']]),
                      'recovery': distribution([row['loss'] for row in reports[arm]['adaptation_record']['records']]),
                      'steps_each_stage': len(reports[arm]['adaptation_record']['records'])} for arm in 'BCD'}
    return {'status': 'SUPPLEMENTARY_DESCRIPTIVE_COMPLETED_DEV_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'analysis_script_sha256': sha(__file__), 'binding': binding,
            'input_sha256': {**files, str(config_path): sha(config_path),
                             str(Path(config['corpus_manifest'])): sha(config['corpus_manifest']),
                             str(run_root / 'comparison.json'): sha(run_root / 'comparison.json')},
            'primary_contrast': 'D minus C on distant_heldout; strict exact JSON string match',
            'secondary_metric': 'Whitespace-strip/casefold exact match; no semantic JSON equivalence',
            'evaluation_groups': subset_summaries, 'by_task': by_task, 'by_registered_distance': by_distance,
            'by_task_and_registered_distance': by_task_distance,
            'distance_grouping_contract': 'For near_control, the registered band identifies its matched distant counterpart; actual near gaps are reported separately.',
            'world_cluster_summary': cluster_summary, 'worlds': world_rows,
            'counterfactual_consistency': cf_serialized, 'matched_near_distant_controls': position_results,
            'cache_geometry_validation': cache_validation, 'cache_equal_length_paired_comparisons': cache_pairs,
            'training_descriptive_losses_not_efficacy': training,
            'source_geometry': {'layers': source['num_hidden_layers'], 'kv_heads': source['num_key_value_heads'],
                                'head_dim': source['head_dim'], 'bytes_per_full_site_per_token': full_site_per_token,
                                'computed_recurrent_state_bytes': recurrent_state, 'window': config['window']},
            'promotion': False, 'confidence_intervals': None, 'inference_or_training_executed_by_analysis': False,
            'limitations': [
                'One source, one registered site, one training seed, and a narrow synthetic DEV corpus.',
                'Counterfactual/position rows share worlds; no independent-row uncertainty estimate or promotion threshold is applied.',
                'Near controls diagnose task/format/source competence; the distance-specific increment is descriptive rather than exclusive memory-route proof.',
                'Remaining full-attention sites retain and can transmit distant evidence; D adds trainable parameters and unequal sequential compute.',
                'Strict compact-JSON scoring rejects alternate spacing/key order/spelling; normalized scoring does not measure semantic JSON correctness.',
                'Missing evidence is balanced inside its category, not across the full distant subset.',
                'Multi-hop relation atoms are adjacent inside one distant block; retention is limited synthetic source-correct retention.',
                'Measured cache is after decoding and completion lengths may differ; equal-length cache comparisons are reported separately.',
                'Instrumented TTFT/decode include guards; model loading/tokenization are excluded, CUDA peaks include the loaded model, and RSS is sampled.',
                'No optimized-kernel, speedup, energy, broad-quality, or universal recurrent-memory claim follows from this pilot.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(EVIDENCE_ROOT) and not output.exists(), 'analysis output must be a new file within its evidence directory')
    result = analyze(args.run_root.resolve(), args.config.resolve())
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'primary': result['evaluation_groups']['distant_heldout']['hybrid_vs_local'],
                      'world_clusters': result['world_cluster_summary']}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (AnalysisBlocked, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({'status': 'ANALYSIS_BLOCKED', 'reason': str(error)}, sort_keys=True))
        raise SystemExit(2)
