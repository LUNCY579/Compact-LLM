"""Offline adversarial scoring/receipt fixtures; no native capability evidence."""
import copy
import json

import pytest

from quality_followup import CONDITIONS, SCHEMA, RECIPE, DATA_SEED
from quality_followup.comparison import compare_reports
from quality_followup.metrics import (anchor_retention_gate, parse_json,
    retention_changes, score_prediction, scored_rows, summarize_rows)
from quality_followup.utils import Blocked, digest, sha
from quality_followup.prepare import SOURCE_DIGEST


@pytest.mark.parametrize('prediction', ['{"v":"x","v":"x"}', '{"v":NaN}',
    '{"v":Infinity}', '{"v":1e999}', '{', '```json\n{"v":"x"}\n```'])
def test_malformed_duplicate_and_nonfinite_answers_fail_closed(prediction):
    value = score_prediction(prediction, '{"v":"x"}')
    assert not value['typed_value_correct']
    assert not value['json_contract_compliance']
    assert not value['json_valid']


@pytest.mark.parametrize('prediction', ['{"v":1}', '{"v":true}',
    '{"v":"1","extra":2}', '{}', '1', '"1"'])
def test_types_and_contract_are_not_salvaged(prediction):
    value = score_prediction(prediction, '{"v":"1"}')
    assert not value['typed_value_correct']
    assert not value['json_contract_compliance']


def test_case_and_nested_types_are_literal_but_normalized_stays_historical():
    value = score_prediction('{"v":"YES"}', '{"v":"yes"}')
    assert value['normalized_exact_match']
    assert value['json_contract_compliance']
    assert not value['typed_value_correct']
    assert not score_prediction('{"v":true}', '{"v":1}')['typed_value_correct']
    assert not score_prediction('{"v":[1,2.0]}', '{"v":[1,2]}')['typed_value_correct']


def test_spacing_can_fix_strict_contract_without_value_repair():
    value = score_prediction(' { "v" : "x" } ', '{"v":"x"}')
    assert value['typed_value_correct'] and value['formatting_only_mismatch']
    assert not value['strict_exact_match']
    assert score_prediction('x', '{"v":"x"}')['bare_expected_value_diagnostic']
    assert not score_prediction('x', '{"v":"x"}')['typed_value_correct']


@pytest.mark.parametrize('target', ['[]', 'null', '{}', '{"v":NaN}', '{"v":0,"v":0}'])
def test_invalid_target_is_rejected(target):
    with pytest.raises(Blocked, match='target'):
        score_prediction('x', target)


def _panel():
    rows, metadata = [], {}
    def append(identifier, target, info):
        rows.append({'id': identifier, 'family_id': identifier,
                     'prompt': f'Fixture only {identifier}', 'target': target})
        metadata[identifier] = dict(info, input_token_sha256=digest([identifier, 'input']),
            label_token_sha256=digest([identifier, 'labels']), token_geometry={'prompt_tokens': 7})
    for world in range(16):
        for cf in range(2):
            for role in ('near_control', 'distant_heldout'):
                for permutation in range(6):
                    position = permutation // 2
                    identifier = f'update-{world}-{cf}-{role}-{permutation}'
                    append(identifier, '{"v":"latest"}', {
                        'category': 'update', 'role': role, 'world_id': f'u-{world}',
                        'counterfactual_id': cf, 'permutation': permutation,
                        'latest_position': position,
                        'first_record_value': 'latest' if position == 0 else 'first'})
    for index in range(64):
        present = index < 32
        append(f'missing-{index}', '{"v":"found"}' if present else '{"v":null}', {
            'category': 'missing', 'role': 'near_control', 'world_id': f'm-{index}',
            'answerable': present})
    for index in range(160):
        append(f'other-{index}', '{"v":"literal"}', {
            'category': 'lookup', 'role': 'near_control', 'world_id': f'o-{index}'})
    for index in range(64):
        append(f'retention-{index}', '{"v":"retained"}', {
            'category': f'skill-{index%8}', 'role': 'retention',
            'world_id': f'r-{index}', 'skill': f'skill-{index%8}'})
    return rows, metadata


def _predictions(rows, update='correct'):
    result = []
    for row in rows:
        prediction = row['target']
        if row['id'].startswith('update-') and update != 'correct':
            prediction = '{"v":"first"}'
        result.append({key: row[key] for key in ('id', 'family_id', 'target')}
                      | {'prediction': prediction, 'input_tokens': 7, 'supervised_positions': 0,
                         'input_token_sha256': digest([row['id'], 'input']),
                         'label_token_sha256': digest([row['id'], 'labels'])})
    return result


def test_consistently_wrong_six_order_groups_are_never_counted_correct():
    rows, meta = _panel()
    result = summarize_rows(_predictions(rows, 'wrong'), rows, meta)
    assert result['typed_all_six_correct_groups'] == 0
    assert len(result['six_order_groups']) == 64
    assert result['update_world_count'] == 16
    assert result['first_record_copy_errors'] == 256


def test_anchor_instrument_each_metric_requires_its_own_skills_and_denominator():
    rows, meta = _panel()
    predictions = _predictions(rows)
    for prediction in predictions:
        if prediction['id'].startswith('retention-'):
            prediction['prediction'] = '{ "v": "retained" }'
    result = anchor_retention_gate(predictions, rows, meta)
    assert not result['adequate']
    assert not result['checks']['strict_exact_match']
    assert result['checks']['typed_value_correct']
    assert result['per_metric']['strict_exact_match']['source_correct'] == 0
    assert result['per_metric']['typed_value_correct']['source_correct'] == 64


def test_source_damage_repairs_and_formatting_repairs_stay_separate():
    rows, meta = _panel()
    anchor = _predictions(rows)
    condition = copy.deepcopy(anchor)
    offset = next(index for index, row in enumerate(rows) if row['id'] == 'retention-0')
    condition[offset]['prediction'] = '{"v":"lost"}'
    anchor[offset+1]['prediction'] = '{ "v":"retained" }'
    anchor[offset+2]['prediction'] = '{"v":"wrong"}'
    anchor[offset+3]['prediction'] = 'retained'
    result = retention_changes(scored_rows(anchor, rows, meta), scored_rows(condition, rows, meta))
    assert result['strict_exact_match']['damages'] == 1
    assert result['strict_exact_match']['repairs'] == 3
    assert result['typed_value_correct']['damages'] == 1
    assert result['typed_value_correct']['repairs'] == 2
    assert result['strict_repairs_by_value']['formatting_only'] == ['retention-1']
    assert result['strict_repairs_by_value']['bare_value_wrapper_diagnostic'] == ['retention-3']
    assert result['strict_repairs_by_value']['literal_value_or_contract'] == ['retention-2']
    assert result['typed_repairs_by_value']['bare_value_wrapper_diagnostic'] == ['retention-3']
    assert result['typed_repairs_by_value']['literal_value_or_contract'] == ['retention-2']


def _reports(tmp_path):
    rows, meta = _panel()
    tokenizer_files = {name: digest(name) for name in
                       ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    packet = {'manifest': {'schema': SCHEMA, 'seed': DATA_SEED, 'recipe': RECIPE,
                          'tokenizer': {'files': tokenizer_files}},
              'dev': rows, 'design': {'dev': meta}, 'ledger': []}
    for condition in ('control', 'candidate'):
        packet[condition] = [{'id': f'{condition}-{step}', 'family_id': f'train-{step}',
            'prompt': f'Fixture order {condition} {step}', 'target': '{"v":"fixture"}'}
            for step in range(512)]
    for step in range(512):
        packet['ledger'].append({'step': step, 'slot_id': f'slot-{step}',
            'control_row_sha256': digest(packet['control'][step]),
            'candidate_row_sha256': digest(packet['candidate'][step]),
            'input_tokens': 12, 'supervised_positions': 6,
            'input_sha256': {condition: digest([condition, step, 'input']) for condition in ('control', 'candidate')},
            'labels_sha256': {condition: digest([step, 'labels']) for condition in ('control', 'candidate')}})
    stable = {field: digest(field) for field in ('source_sha256', 'code_sha256',
        'config_sha256', 'corpus_sha256', 'tokenizer_sha256')}
    files = {'control':'control_train.jsonl', 'candidate':'candidate_train.jsonl',
             'dev':'dev.jsonl', 'design':'design.json', 'ledger':'exposure_ledger.json'}
    packet['manifest']['files'] = {}
    for key, filename in files.items():
        path = tmp_path / filename
        contents = ('\n'.join(json.dumps(row) for row in packet[key])
                    if key in ('control', 'candidate', 'dev') else json.dumps(packet[key]))
        path.write_text(contents, encoding='utf-8')
        packet['manifest']['files'][key] = {'path': filename, 'sha256': sha(path),
                                           'bytes': path.stat().st_size}
    ledger_path = tmp_path / files['ledger']
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(packet['manifest']), encoding='utf-8')
    packet['manifest_path'] = str(manifest_path)
    stable.update(schema=SCHEMA, study_id='update-order-v1',
                  source_sha256=SOURCE_DIGEST, corpus_sha256=sha(manifest_path),
                  tokenizer_sha256=digest(tokenizer_files),
                  ledger_sha256=sha(ledger_path), environment={'fixture': True})
    reports = {}
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        return {'path': str(path.resolve()), 'sha256': sha(path)}
    def qualification(binding, created):
        return {'status': 'PASS_NATIVE_MECHANICS_ONLY', 'binding': copy.deepcopy(binding),
            'created_utc': created, 'max_prompt_tokens_checked': 2048,
            'capability_claim': False, 'native_training_qualified': False}
    for condition in CONDITIONS:
        binding = dict(stable, condition=condition, arm='A' if condition == 'anchor' else 'B',
                       delta_sha256=None, delta_manifest_sha256=None)
        predictions = _predictions(rows, 'wrong' if condition == 'control' else 'correct')
        report = {'status': 'DEV_MEASUREMENT_ONLY', 'binding': binding,
            'predictions': predictions, 'training_record': None, 'delta_record': None,
            'created_utc': '2026-10-05T03:00:00+00:00', 'artifact_refs': {},
            'summary': summarize_rows(predictions, rows, meta)}
        if condition != 'anchor':
            initial = qualification(binding, '2026-10-05T00:00:00+00:00')
            records = [{
                'step': step, 'slot_id': slot['slot_id'], 'row_id': packet[condition][step]['id'],
                'input_tokens': slot['input_tokens'], 'supervised_positions': slot['supervised_positions'],
                'input_token_sha256': slot['input_sha256'][condition],
                'label_token_sha256': slot['labels_sha256'][condition],
                'loss': 1.0, 'gradient_norm': .1} for step, slot in enumerate(packet['ledger'])]
            training = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': 'train',
                'binding': copy.deepcopy(binding), 'records': records, 'qualification': initial,
                'started_utc': '2026-10-05T00:30:00+00:00', 'created_utc': '2026-10-05T01:00:00+00:00'}
            delta_path = tmp_path / f'{condition}.safetensors'
            delta_path.write_bytes(b'offline receipt-validation fixture, not tensors')
            manifest = {'binding': copy.deepcopy(binding), 'record': training,
                        'parameters': {'fixture_only': [1]}, 'sha256': sha(delta_path)}
            manifest_ref = save(f'{condition}-delta.json', manifest)
            binding.update(delta_sha256=sha(delta_path), delta_manifest_sha256=manifest_ref['sha256'])
            report.update(training_record=training, delta_record=manifest)
            report['artifact_refs'].update(initial_qualification=save(f'{condition}-initial.json', initial),
                training=save(f'{condition}-training.json', training), delta_manifest=manifest_ref,
                delta={'path': str(delta_path.resolve()), 'sha256': sha(delta_path)})
        final = qualification(binding, '2026-10-05T02:00:00+00:00')
        report['qualification'] = final
        report['artifact_refs']['qualification'] = save(f'{condition}-qualification.json', final)
        reports[condition] = report
    return reports, packet


def test_valid_mechanical_receipts_apply_prospective_decision(tmp_path):
    reports, packet = _reports(tmp_path)
    result = compare_reports(reports, packet)
    assert result['decision'] == 'ADVANCE_TO_REVIEW_ONLY'
    assert result['promotion'] is False and result['automatic_successor'] is False
    assert result['primary']['net_additional_correct'] == 256
    assert result['absolute_answerability']['present_fact']['candidate']['correct'] == 32
    assert result['source_correct_retention_and_source_incorrect_repairs']['candidate']['strict_exact_match']['damages'] == 0


@pytest.mark.parametrize('mutation', ['incomplete', 'input_hash', 'label_hash', 'condition_swap',
    'nonfinite', 'summary', 'raw_flag', 'delta_bytes', 'qualification', 'source', 'row_order',
    'dev_input_hash', 'dev_labels_hash', 'dev_length', 'dev_supervision'])
def test_tampered_or_incomplete_native_chain_is_rejected(tmp_path, mutation):
    reports, packet = _reports(tmp_path)
    candidate = reports['candidate']
    records = candidate['training_record']['records']
    if mutation == 'incomplete':
        records.pop()
    elif mutation == 'input_hash':
        records[0]['input_token_sha256'] = digest('tampered')
    elif mutation == 'label_hash':
        records[0]['label_token_sha256'] = digest('tampered')
    elif mutation == 'condition_swap':
        records[0]['input_token_sha256'] = packet['ledger'][0]['input_sha256']['control']
    elif mutation == 'nonfinite':
        records[0]['loss'] = float('nan')
    elif mutation == 'summary':
        candidate['summary']['typed_all_six_correct_groups'] = 63
    elif mutation == 'raw_flag':
        candidate['predictions'][0]['typed_value_correct'] = False
    elif mutation == 'delta_bytes':
        from pathlib import Path
        Path(candidate['artifact_refs']['delta']['path']).write_bytes(b'tampered')
    elif mutation == 'qualification':
        candidate['qualification']['binding']['delta_sha256'] = digest('other')
    elif mutation == 'source':
        candidate['binding']['source_sha256'] = digest('other')
    elif mutation == 'row_order':
        candidate['predictions'][0], candidate['predictions'][1] = candidate['predictions'][1], candidate['predictions'][0]
    elif mutation == 'dev_input_hash':
        candidate['predictions'][0]['input_token_sha256'] = digest('tampered')
    elif mutation == 'dev_labels_hash':
        candidate['predictions'][0]['label_token_sha256'] = digest('tampered')
    elif mutation == 'dev_length':
        candidate['predictions'][0]['input_tokens'] = 8
    elif mutation == 'dev_supervision':
        candidate['predictions'][0]['supervised_positions'] = 1
    with pytest.raises(Blocked):
        compare_reports(reports, packet)


def test_damage_blocks_even_when_aggregate_retention_is_repaired(tmp_path):
    reports, packet = _reports(tmp_path)
    reports['candidate']['predictions'][-1]['prediction'] = '{"v":"damaged"}'
    reports['candidate']['summary'] = summarize_rows(reports['candidate']['predictions'], packet['dev'], packet['design'])
    result = compare_reports(reports, packet)
    assert result['decision'] == 'HOLD'
    assert 'zero_new_strict_retention_damage' in result['failed_checks']
    assert 'zero_new_typed_retention_damage' in result['failed_checks']


def test_consistency_without_correctness_cannot_advance(tmp_path):
    reports, packet = _reports(tmp_path)
    reports['candidate']['predictions'] = _predictions(packet['dev'], 'wrong')
    reports['candidate']['summary'] = summarize_rows(reports['candidate']['predictions'], packet['dev'], packet['design'])
    result = compare_reports(reports, packet)
    assert result['decision'] == 'HOLD'
    assert result['fully_correct_six_order_group_net'] == 0
    assert 'primary_256_net_at_least_26' in result['failed_checks']


def test_absolute_answerability_preserved_even_when_both_conditions_poor(tmp_path):
    reports, packet = _reports(tmp_path)
    for condition in ('control', 'candidate'):
        for prediction in reports[condition]['predictions']:
            if prediction['id'].startswith('missing-'):
                prediction['prediction'] = '{"v":null}'
        reports[condition]['summary'] = summarize_rows(reports[condition]['predictions'], packet['dev'], packet['design'])
    result = compare_reports(reports, packet)
    assert result['checks']['present_fact_no_decline']
    assert result['absolute_answerability']['present_fact']['candidate']['accuracy'] == 0
    assert result['absolute_answerability']['correct_abstention']['candidate']['accuracy'] == 1


@pytest.mark.parametrize('field', ['manifest', 'ledger', 'design', 'dev'])
def test_direct_comparator_rejects_in_memory_packet_tampering(tmp_path, field):
    reports, packet = _reports(tmp_path)
    if field == 'manifest':
        packet['manifest']['seed'] = 170106
    elif field == 'ledger':
        packet['ledger'][0]['input_tokens'] = 13
    elif field == 'design':
        next(iter(packet['design']['dev'].values()))['latest_position'] = 2
    else:
        packet['dev'][0]['prompt'] = 'changed'
    with pytest.raises(Blocked, match='in-memory'):
        compare_reports(reports, packet)
