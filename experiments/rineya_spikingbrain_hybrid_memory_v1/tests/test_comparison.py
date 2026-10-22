"""Paired DEV reporting contracts; these fixtures are not efficacy evidence."""
import copy
import json

import pytest

from sbhm import data, pipeline, protocol


def reports():
    targets = ('one', 'two', 'three', 'four')
    predictions = {'A': ('one', 'two', 'wrong', 'wrong'),
                   'B': ('one', 'two', 'wrong', 'wrong'),
                   'C': ('one', 'wrong', 'THREE', 'four'),
                   'D': ('one', 'two', 'three', 'wrong')}
    result = {}
    for arm, texts in predictions.items():
        rows = [{'id': f'dev-{i}', 'family_id': 'family-a' if i < 2 else 'family-b',
                 'target': target, 'prediction': prediction, 'reset_recurrent_at': None,
                 'strict_exact_match': prediction == target,
                 'normalized_exact_match': prediction.strip().casefold() == target.strip().casefold()}
                for i, (target, prediction) in enumerate(zip(targets, texts))]
        distill = {'stage': 'distill', 'records': [{'row_id': 'train-1', 'input_tokens': 6,
                                                'supervised_positions': 6}]}
        recovery = {'stage': 'recover', 'records': [{'row_id': 'train-1', 'input_tokens': 6,
                                                  'supervised_positions': 2}], 'parent_record': distill}
        binding = {'arm': arm, 'source_sha256': 'source', 'code_sha256': 'code',
                   'config_sha256': 'config', 'environment': {},
                   'delta_sha256': None if arm == 'A' else arm}
        for stage in (distill, recovery):
            stage.update(status='TRAINING_COMPLETED_NOT_EFFICACY', corpus_sha256='corpus', binding=copy.deepcopy(binding))
            stage['records'][0].update(step=0, loss=0.0, gradient_norm=0.0)
        result[arm] = {'status': 'DEV_MEASUREMENT_ONLY', 'corpus_sha256': 'corpus',
                       'binding': binding,
                       'adaptation_record': None if arm == 'A' else recovery,
                       'predictions': rows,
                       **{metric: sum(row[metric] for row in rows) / len(rows)
                          for metric in ('strict_exact_match', 'normalized_exact_match')}}
    return result


def run_compare(tmp_path, value):
    paths = []
    for arm in 'ABCD':
        path = tmp_path / f'{arm}.json'
        path.write_text(json.dumps(value[arm]), encoding='utf-8')
        paths.append(path)
    return pipeline.compare(paths)


def test_hybrid_local_contrast_and_original_retention_are_separate(tmp_path):
    result = run_compare(tmp_path, reports())
    contrast = result['hybrid_vs_local']
    assert contrast['paired_rows'] == 4
    assert contrast['strict_exact_match'] == {
        'D_only_correct': 2, 'C_only_correct': 1, 'both_correct': 1, 'neither_correct': 0,
        'D_accuracy': .75, 'C_accuracy': .5, 'delta_percentage_points': 25.0}
    assert contrast['normalized_exact_match']['D_only_correct'] == 1
    assert contrast['normalized_exact_match']['C_only_correct'] == 1
    assert contrast['normalized_exact_match']['delta_percentage_points'] == 0
    assert contrast['per_family']['family-a']['strict_exact_match']['delta_percentage_points'] == 50
    assert result['arms']['D']['repairs'] == 1
    assert result['arms']['D']['damages'] == 0
    assert result['arms']['C']['original_model_retention']['strict_exact_match'] == {
        'baseline_correct': 2, 'retained': 1, 'damages': 1, 'repairs': 1, 'retention_rate': .5}
    assert result['arms']['D']['original_model_retention']['strict_exact_match']['retention_rate'] == 1
    assert result['arms']['D']['per_family']['family-b']['original_model_retention']['strict_exact_match']['retention_rate'] is None
    assert result['status'] == 'DESCRIPTIVE_PAIRED_DEV_ONLY'
    assert result['promotion'] is False


@pytest.mark.parametrize('field', ['id', 'family_id', 'target'])
def test_same_report_ids_do_not_allow_mismatched_pairing(tmp_path, field):
    value = reports()
    # The raw text/metrics remain internally consistent; only pairing changes.
    row = value['C']['predictions'][1]
    row[field] = 'changed'
    with pytest.raises(protocol.Blocked, match='mismatch'):
        run_compare(tmp_path, value)


@pytest.mark.parametrize('metric', ['strict_exact_match', 'normalized_exact_match'])
@pytest.mark.parametrize('bad_value', [1, 'true', None])
def test_row_metrics_must_be_boolean(tmp_path, metric, bad_value):
    value = reports()
    value['C']['predictions'][0][metric] = bad_value
    with pytest.raises(protocol.Blocked, match='booleans'):
        run_compare(tmp_path, value)


@pytest.mark.parametrize('metric', ['strict_exact_match', 'normalized_exact_match'])
@pytest.mark.parametrize('bad_value', [.9, '0.5', True, None])
def test_aggregate_metrics_must_match_rows(tmp_path, metric, bad_value):
    value = reports()
    value['C'][metric] = bad_value
    with pytest.raises(protocol.Blocked, match='reported'):
        run_compare(tmp_path, value)


def test_row_metric_cannot_contradict_raw_prediction(tmp_path):
    value = reports()
    value['C']['predictions'][1]['normalized_exact_match'] = True
    with pytest.raises(protocol.Blocked, match='contradict'):
        run_compare(tmp_path, value)


def test_duplicate_prediction_ids_rejected(tmp_path):
    value = reports()
    value['C']['predictions'][1]['id'] = value['C']['predictions'][0]['id']
    with pytest.raises(protocol.Blocked, match='duplicate'):
        run_compare(tmp_path, value)


def test_empty_predictions_rejected(tmp_path):
    value = reports()
    value['A']['predictions'] = []
    with pytest.raises(protocol.Blocked, match='nonempty'):
        run_compare(tmp_path, value)


def test_state_reset_cannot_enter_primary_comparison(tmp_path):
    value = reports()
    value['D']['predictions'][0]['reset_recurrent_at'] = 1
    with pytest.raises(protocol.Blocked, match='state-reset'):
        run_compare(tmp_path, value)


def test_frozen_original_cannot_include_adaptation_record(tmp_path):
    value = reports()
    value['A']['adaptation_record'] = copy.deepcopy(value['B']['adaptation_record'])
    with pytest.raises(protocol.Blocked, match='unadapted'):
        run_compare(tmp_path, value)


@pytest.mark.parametrize('mutation', ['no_delta', 'empty_records', 'no_status', 'wrong_arm',
                                     'wrong_corpus', 'nonfinite_loss', 'missing_prediction'])
def test_incomplete_or_unbound_adaptation_cannot_enter_comparison(tmp_path, mutation):
    value = reports()
    report = value['D']
    stage = report['adaptation_record']
    if mutation == 'no_delta':
        report['binding']['delta_sha256'] = None
    elif mutation == 'empty_records':
        stage['records'] = []
    elif mutation == 'no_status':
        del stage['status']
    elif mutation == 'wrong_arm':
        stage['binding']['arm'] = 'C'
    elif mutation == 'wrong_corpus':
        stage['corpus_sha256'] = 'other'
    elif mutation == 'nonfinite_loss':
        stage['records'][0]['loss'] = float('nan')
    else:
        del report['predictions'][0]['prediction']
    with pytest.raises(protocol.Blocked):
        run_compare(tmp_path, value)


def test_registered_groups_separate_primary_contrast_and_retention(tmp_path):
    value = reports()
    # Labels here exercise reporting only; fixture text is not distant evidence.
    groups = {'distant_evidence': ['family-b'], 'short_retention_probe': ['family-a']}
    for report in value.values():
        report['evaluation_groups'] = copy.deepcopy(groups)
    result = run_compare(tmp_path, value)
    distant = result['evaluation_groups']['distant_evidence']
    short = result['evaluation_groups']['short_retention_probe']
    assert distant['family_ids'] == ['family-b']
    assert distant['hybrid_vs_local']['paired_rows'] == 2
    assert distant['hybrid_vs_local']['strict_exact_match']['delta_percentage_points'] == 0
    assert short['hybrid_vs_local']['strict_exact_match']['delta_percentage_points'] == 50
    assert short['arms']['C']['original_model_retention']['strict_exact_match']['retention_rate'] == .5
    assert short['arms']['D']['original_model_retention']['strict_exact_match']['retention_rate'] == 1
    assert distant['arms']['D']['original_model_retention']['strict_exact_match']['retention_rate'] is None


@pytest.mark.parametrize('drift', ['swapped', 'missing'])
def test_group_registration_must_match_all_four_arms(tmp_path, drift):
    value = reports()
    groups = {'distant_evidence': ['family-b'], 'short_retention_probe': ['family-a']}
    for report in value.values():
        report['evaluation_groups'] = copy.deepcopy(groups)
    value['D']['evaluation_groups'] = ({'distant_evidence': ['family-a'], 'short_retention_probe': ['family-b']}
                                     if drift == 'swapped' else None)
    with pytest.raises(protocol.Blocked, match='unpaired evaluation_groups'):
        run_compare(tmp_path, value)


@pytest.mark.parametrize('groups', [
    {}, {'': ['family-a', 'family-b']}, {'group': 'family-a'}, {'group': []},
    {'group': [None]}, {'group': ['  ']}, {'group': ['family-a', 'family-a', 'family-b']},
    {'first': ['family-a'], 'second': ['family-a', 'family-b']},
    {'group': ['family-a']}, {'group': ['family-a', 'family-b', 'unknown']},
])
def test_invalid_group_registration_rejected(groups):
    with pytest.raises(protocol.Blocked, match='evaluation_groups'):
        data.validate_evaluation_groups(groups, reports()['A']['predictions'])


def test_group_registration_canonicalizes_family_order_and_allows_absence():
    rows = reports()['A']['predictions']
    assert data.validate_evaluation_groups(None, rows) is None
    assert data.validate_evaluation_groups({'all': ['family-b', 'family-a']}, rows) == {
        'all': ['family-a', 'family-b']}


def test_tampered_group_coverage_cannot_enter_comparison(tmp_path):
    value = reports()
    for report in value.values():
        report['evaluation_groups'] = {'primary': ['family-a'], 'retention': ['family-a', 'family-b']}
    with pytest.raises(protocol.Blocked, match='overlapping'):
        run_compare(tmp_path, value)


def test_manifest_group_validation_preserves_four_field_rows(tmp_path):
    data.make_fixture_corpus(tmp_path / 'fixture')
    path = tmp_path / 'fixture' / 'manifest.json'
    manifest = protocol.read_json(path)
    manifest['reviewed_for_sbhm'] = True  # Temporary mechanics-test fixture only.
    manifest['evaluation_groups'] = {'engineering_fixture': ['updated_lookup']}
    path.write_text(json.dumps(manifest), encoding='utf-8')
    loaded, _ = data.load_corpus(path)
    assert set(loaded['dev'][0]) == {'id', 'family_id', 'prompt', 'target'}
    manifest['evaluation_groups'] = {'engineering_fixture': ['unknown']}
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(protocol.Blocked, match='every DEV family'):
        data.load_corpus(path)
    manifest['evaluation_groups'] = None
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(protocol.Blocked, match='present evaluation_groups'):
        data.load_corpus(path)


@pytest.mark.parametrize('reference', ['design', 'review'])
def test_manifest_analysis_metadata_is_hash_bound(tmp_path, reference):
    data.make_fixture_corpus(tmp_path / 'fixture')
    path = tmp_path / 'fixture/manifest.json'
    metadata = path.parent / f'{reference}.json'
    metadata.write_text('{}', encoding='utf-8')
    manifest = protocol.read_json(path)
    manifest['reviewed_for_sbhm'] = True  # Mechanics-only temporary fixture.
    manifest[reference] = {'path': metadata.name, 'sha256': protocol.sha(metadata)}
    path.write_text(json.dumps(manifest), encoding='utf-8')
    data.load_corpus(path)
    metadata.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(protocol.Blocked, match='metadata.*changed'):
        data.load_corpus(path)


def test_evaluation_receipt_carries_registered_groups(tmp_path, monkeypatch):
    import torch
    groups = {'distant_evidence': ['family-b'], 'short_retention_probe': ['family-a']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'evaluation_groups': groups}), encoding='utf-8')
    rows = [{'id': row['id'], 'family_id': row['family_id'], 'prompt': 'fixture', 'target': row['target']}
            for row in reports()['A']['predictions']]
    monkeypatch.setattr(pipeline, 'corpus', lambda _config: ({'dev': rows}, protocol.sha(path)))
    monkeypatch.setattr(pipeline, 'tokens', lambda *args: (torch.zeros((1, 1), dtype=torch.long), None))
    monkeypatch.setattr(pipeline, 'generate', lambda *args: {'prediction': 'wrong', 'reset_recurrent_at': None})
    monkeypatch.setattr(pipeline, 'resources', lambda *args, **kwargs: {})
    receipt = pipeline.evaluate(lambda **kwargs: None, None,
                                {'device': 'cpu', 'corpus_manifest': str(path)}, {}, tmp_path, 'A')
    assert receipt['evaluation_groups'] == groups
