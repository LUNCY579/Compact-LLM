"""Dedicated order-intervention comparator; SBHM exposure rules stay unchanged."""
import math
from datetime import datetime, timedelta
from pathlib import Path

from . import CONDITIONS, SCHEMA
from .metrics import (METRICS, anchor_retention_gate, retention_changes,
                      scored_rows, summarize_rows)
from .utils import digest, external_path, read_json, require, sha
from .prepare import SOURCE_DIGEST

STABLE_BINDING = ('schema', 'study_id', 'source_sha256', 'code_sha256',
                  'config_sha256', 'corpus_sha256', 'ledger_sha256',
                  'tokenizer_sha256', 'environment')


def _timestamp(value, label):
    try:
        result = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        require(False, f'{label} requires a valid timezone-aware date')
    require(result.tzinfo is not None, f'{label} requires a timezone-aware date')
    return result


def _hash(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), f'invalid {label} SHA256')


def _same_binding(actual, expected, label, initial=False):
    require(isinstance(actual, dict), f'{label} binding missing')
    for key in STABLE_BINDING + ('condition', 'arm'):
        require(actual.get(key) == expected.get(key), f'{label} binding mismatch: {key}')
    for key in ('delta_sha256', 'delta_manifest_sha256'):
        require(actual.get(key) == (None if initial else expected.get(key)),
                f'{label} binding mismatch: {key}')


def _artifact(report, name, expected=None):
    refs = report.get('artifact_refs')
    require(isinstance(refs, dict) and name in refs, f'missing {name} artifact reference')
    reference = refs[name]
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'},
            f'invalid {name} artifact reference')
    _hash(reference['sha256'], name)
    path = Path(reference['path'])
    require(path.is_absolute() and path.is_file(), f'missing {name} artifact file')
    external_path(path)
    require(sha(path) == reference['sha256'], f'{name} artifact bytes changed')
    if expected is not None:
        require(read_json(path) == expected, f'{name} embedded receipt differs from artifact')
    return reference


def _qualification(report):
    qualification = report.get('qualification')
    require(isinstance(qualification, dict) and qualification.get('status') == 'PASS_NATIVE_MECHANICS_ONLY',
            'completed native qualification is required')
    _same_binding(qualification.get('binding'), report['binding'], 'qualification')
    require(qualification.get('max_prompt_tokens_checked') == 2048,
            'qualification must exercise the registered 2048-token prefix')
    require(qualification.get('capability_claim') is False and
            qualification.get('native_training_qualified') is False,
            'qualification is mechanics only')
    timestamp = _timestamp(qualification.get('created_utc'), 'qualification')
    measured = _timestamp(report.get('created_utc'), 'DEV report')
    require(timedelta(0) <= measured - timestamp <= timedelta(hours=24),
            'DEV measurement did not use fresh qualification')
    _artifact(report, 'qualification', qualification)


def _training(report, packet):
    condition = report['binding']['condition']
    record, manifest = report.get('training_record'), report.get('delta_record')
    binding = report['binding']
    if condition == 'anchor':
        require(record is None and manifest is None and binding.get('delta_sha256') is None
                and binding.get('delta_manifest_sha256') is None,
                'anchor must be unadapted with no delta or training record')
        return
    for field in ('delta_sha256', 'delta_manifest_sha256'):
        _hash(binding.get(field), field)
    require(isinstance(record, dict) and record.get('status') == 'TRAINING_COMPLETED_NOT_EFFICACY'
            and record.get('stage') == 'train', 'complete adaptation receipt is required')
    _same_binding(record.get('binding'), binding, 'training', initial=True)
    records = record.get('records')
    ledger = packet['ledger']
    require(isinstance(records, list) and len(records) == len(ledger) == 512,
            'training must contain exactly 512 completed updates')
    train_rows = packet[condition]
    by_id = {row['id']: row for row in train_rows}
    for step, (runtime, slot) in enumerate(zip(records, ledger)):
        require(isinstance(runtime, dict), 'runtime training record must be object')
        require(type(runtime.get('step')) is int and runtime['step'] == step == slot['step'],
                'runtime exposure step mismatch')
        require(runtime.get('slot_id') == slot['slot_id'], 'runtime exposure slot mismatch')
        row_id = runtime.get('row_id')
        require(row_id in by_id and digest(by_id[row_id]) == slot[f'{condition}_row_sha256'],
                'runtime training row differs from condition ledger')
        for runtime_name, ledger_name in (
                ('input_token_sha256', 'input_sha256'),
                ('label_token_sha256', 'labels_sha256')):
            require(runtime.get(runtime_name) == slot[ledger_name][condition],
                    f'actual runtime {runtime_name} differs from condition ledger')
            _hash(runtime.get(runtime_name), runtime_name)
        for name in ('input_tokens', 'supervised_positions'):
            require(type(runtime.get(name)) is int and runtime[name] == slot[name] and runtime[name] > 0,
                    f'runtime {name} differs from ledger')
        require(runtime['input_tokens'] <= 259, 'runtime exceeded registered TRAIN length')
        for field in ('loss', 'gradient_norm'):
            value = runtime.get(field)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    f'nonfinite or invalid runtime {field}')
    require(isinstance(manifest, dict) and manifest.get('sha256') == binding['delta_sha256']
            and manifest.get('record') == record,
            'delta manifest must bind exact completed training receipt')
    _same_binding(manifest.get('binding'), binding, 'delta manifest', initial=True)
    require(isinstance(manifest.get('parameters'), dict) and bool(manifest['parameters']),
            'delta requires nonempty trainable parameter inventory')
    _artifact(report, 'training', record)
    manifest_ref = _artifact(report, 'delta_manifest', manifest)
    delta_ref = _artifact(report, 'delta')
    require(manifest_ref['sha256'] == binding['delta_manifest_sha256']
            and delta_ref['sha256'] == binding['delta_sha256'], 'delta artifact binding mismatch')
    # Training qualification is a distinct initial B-variant receipt.
    initial = record.get('qualification')
    require(isinstance(initial, dict) and initial.get('status') == 'PASS_NATIVE_MECHANICS_ONLY',
            'training requires completed initial qualification')
    _same_binding(initial.get('binding'), binding, 'initial qualification', initial=True)
    require(initial.get('max_prompt_tokens_checked') == 2048,
            'initial qualification must exercise registered prefix')
    started = _timestamp(record.get('started_utc'), 'training start')
    finished = _timestamp(record.get('created_utc'), 'training completion')
    qualified = _timestamp(initial.get('created_utc'), 'initial qualification')
    require(timedelta(0) <= started - qualified <= timedelta(hours=24) and started <= finished,
            'training did not start under fresh initial qualification')
    require(finished <= _timestamp(report['qualification']['created_utc'], 'post-delta qualification'),
            'post-delta qualification must follow completed training')
    _artifact(report, 'initial_qualification', initial)


def _validate_panel(packet):
    rows, meta = packet['dev'], packet['design']['dev']
    require(len(rows) == 672 and len({row['id'] for row in rows}) == 672,
            'registered common DEV panel must have exactly 672 unique rows')
    require(set(meta) == {row['id'] for row in rows}, 'DEV metadata coverage mismatch')
    update = [info for info in meta.values() if info.get('category') == 'update']
    missing = [info for info in meta.values() if info.get('category') == 'missing']
    retained = [info for info in meta.values() if info.get('role') == 'retention']
    require(len(update) == 384 and len({info['world_id'] for info in update}) == 16,
            'registered update panel must contain 384 rows in 16 worlds')
    require(len(missing) == 64 and sum(info.get('answerable') is True for info in missing) == 32
            and sum(info.get('answerable') is False for info in missing) == 32,
            'registered answerability panel must contain 32 present and 32 absent rows')
    require(len(retained) == 64, 'registered retention panel must contain 64 rows')
    for position in range(3):
        require(sum(info.get('latest_position') == position for info in update) == 128,
                'each latest-record position must have 128 DEV rows')
    require(sum(info.get('latest_position') in (1, 2) for info in update) == 256,
            'primary greatest-not-first panel must contain 256 rows')


def _decision(scored, summaries, gate):
    anchor, control, candidate = (scored[name] for name in CONDITIONS)
    def count(rows, selector, metric='typed_value_correct'):
        return sum(row[metric] for row in rows if selector(row['metadata']))
    update = lambda info: info.get('category') == 'update'
    not_first = lambda info: update(info) and info['latest_position'] != 0
    net = count(candidate, not_first) - count(control, not_first)
    middle = count(candidate, lambda info: update(info) and info['latest_position'] == 1) - count(control, lambda info: update(info) and info['latest_position'] == 1)
    last = count(candidate, lambda info: update(info) and info['latest_position'] == 2) - count(control, lambda info: update(info) and info['latest_position'] == 2)
    first = count(candidate, lambda info: update(info) and info['latest_position'] == 0) - count(control, lambda info: update(info) and info['latest_position'] == 0)
    grouped_net = summaries['candidate']['typed_all_six_correct_groups'] - summaries['control']['typed_all_six_correct_groups']
    retained = {name: retention_changes(anchor, scored[name]) for name in ('control', 'candidate')}
    answerability = {}
    for present, name in ((True, 'present_fact'), (False, 'correct_abstention')):
        selector = lambda info, present=present: info.get('category') == 'missing' and info['answerable'] is present
        answerability[name] = {condition: {'correct': count(scored[condition], selector), 'total': 32,
                                          'accuracy': count(scored[condition], selector)/32}
                              for condition in CONDITIONS}
    checks = {
        'retention_instrument_adequate_each_metric': gate['adequate'],
        'primary_256_net_at_least_26': net >= 26,
        'middle_improves': middle > 0,
        'last_improves': last > 0,
        'fully_correct_six_order_groups_increase': grouped_net > 0,
        'greatest_first_no_decline': first >= 0,
        'json_contract_no_decline': sum(row['json_contract_compliance'] for row in candidate) >= sum(row['json_contract_compliance'] for row in control),
        'zero_new_strict_retention_damage': retained['candidate']['strict_exact_match']['damages'] == 0,
        'zero_new_typed_retention_damage': retained['candidate']['typed_value_correct']['damages'] == 0,
        'present_fact_no_decline': answerability['present_fact']['candidate']['correct'] >= answerability['present_fact']['control']['correct'],
        'correct_abstention_no_decline': answerability['correct_abstention']['candidate']['correct'] >= answerability['correct_abstention']['control']['correct'],
    }
    return {'decision': 'ADVANCE_TO_REVIEW_ONLY' if all(checks.values()) else 'HOLD',
            'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed],
            'primary': {'rows': 256, 'net_additional_correct': net,
                        'delta_percentage_points': 100*net/256,
                        'required_net_additional_correct': 26},
            'middle_net_additional_correct': middle, 'last_net_additional_correct': last,
            'greatest_first_net_additional_correct': first,
            'fully_correct_six_order_group_net': grouped_net,
            'source_correct_retention_and_source_incorrect_repairs': retained,
            'absolute_answerability': answerability, 'anchor_retention_gate': gate,
            'promotion': False, 'automatic_successor': False}


def compare_reports(reports, packet):
    """Audit raw native evidence before applying immutable prospective thresholds."""
    require(isinstance(reports, dict) and set(reports) == set(CONDITIONS),
            'exactly anchor/control/candidate native DEV reports are required')
    require(isinstance(packet, dict) and 'manifest_path' in packet,
            'comparison requires the exact frozen corpus manifest path')
    from .corpus import load_packet
    frozen = load_packet(external_path(packet['manifest_path']))
    for field in ('manifest', 'control', 'candidate', 'dev', 'design', 'ledger'):
        require(packet.get(field) == frozen[field],
                f'in-memory {field} differs from the frozen packet bytes')
    _validate_panel(packet)
    manifest = packet['manifest']
    source = reports['anchor'].get('binding')
    require(isinstance(source, dict) and source.get('schema') == SCHEMA,
            'native report schema mismatch')
    for key in STABLE_BINDING:
        require(key in source and source[key] is not None, f'anchor missing stable binding {key}')
    for key in STABLE_BINDING:
        if key.endswith('_sha256'):
            _hash(source[key], key)
    require(source['source_sha256'] == SOURCE_DIGEST,
            'only the audited exact frozen Titan source is registered')
    require(source['study_id'] == 'update-order-v1', 'unknown native quality study')
    require(source['ledger_sha256'] == manifest.get('files', {}).get('ledger', {}).get('sha256'),
            'receipt ledger binding differs from packet')
    require('manifest_path' in packet and source['corpus_sha256'] == sha(packet['manifest_path']),
            'receipt corpus binding differs from manifest bytes')
    tokenizer = {key: value for key, value in manifest.get('tokenizer', {}).get('files', {}).items()
                 if key in {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}}
    require(len(tokenizer) == 5 and source['tokenizer_sha256'] == digest(tokenizer),
            'receipt tokenizer binding differs from corpus')
    scored, summaries = {}, {}
    for name in CONDITIONS:
        report = reports[name]
        require(report.get('status') == 'DEV_MEASUREMENT_ONLY', 'completed native DEV report required')
        binding = report.get('binding')
        require(isinstance(binding, dict) and binding.get('condition') == name
                and binding.get('arm') == ('A' if name == 'anchor' else 'B'),
                'condition-bound original-attention variant mismatch')
        for key in STABLE_BINDING:
            require(binding.get(key) == source[key], f'unmatched native {key}')
        _qualification(report)
        _training(report, packet)
        scored[name] = scored_rows(report.get('predictions'), packet['dev'], packet['design'])
        for prediction in report['predictions']:
            meta = packet['design']['dev'][prediction['id']]
            for field in ('input_token_sha256', 'label_token_sha256'):
                _hash(meta.get(field), 'registered DEV ' + field)
                require(prediction.get(field) == meta[field],
                        f'actual runtime DEV {field} differs from frozen prompt')
            prompt_tokens = meta.get('token_geometry', {}).get('prompt_tokens')
            require(type(prompt_tokens) is int and 0 < prompt_tokens <= 1158 and
                    type(prediction.get('input_tokens')) is int and
                    prediction['input_tokens'] == prompt_tokens,
                    'actual runtime DEV length differs from frozen prompt')
            require(type(prediction.get('supervised_positions')) is int and
                    prediction['supervised_positions'] == 0,
                    'DEV measurement must not expose supervised labels to the model')
        summaries[name] = summarize_rows(report['predictions'], packet['dev'], packet['design'])
        require(report.get('summary') == summaries[name], 'reported summary contradicts raw predictions')
        for metric in METRICS:
            if metric in report:
                actual = report[metric]
                expected = summaries[name]['overall'][metric]['accuracy']
                require(type(actual) in (int, float) and math.isfinite(actual)
                        and actual == expected, f'reported {metric} contradicts raw predictions')
    gate = anchor_retention_gate(reports['anchor']['predictions'], packet['dev'], packet['design'])
    result = _decision(scored, summaries, gate)
    return {'status': 'AUDITED_SYNTHETIC_DEV_DECISION_ONLY', 'schema': SCHEMA,
            **result, 'conditions': summaries,
            'dependent_unit': 'world', 'update_worlds': 16, 'six_order_groups': 64,
            'confidence_intervals': None, 'training_exposure_intervention':
            'balanced six-order presentation versus fixed biased repetition; matched facts, labels, lengths and steps',
            'limitations': ['one frozen source, one site, one model seed', 'synthetic DEV only',
                'no broad retention or transfer claim', 'no separate attribution of diversity versus position frequency'],
            'report_bindings': {name: reports[name]['binding'] for name in CONDITIONS}}


def compare(paths, packet):
    """Read three reports without importing or loading a pretrained model."""
    require(isinstance(paths, (dict, list, tuple)), 'native report paths required')
    if isinstance(paths, dict):
        require(set(paths) == set(CONDITIONS), 'exact condition report paths required')
        reports = {name: read_json(paths[name]) for name in CONDITIONS}
    else:
        require(len(paths) == 3, 'exactly three native report paths required')
        reports = {name: read_json(path) for name, path in zip(CONDITIONS, paths)}
    return compare_reports(reports, packet)
