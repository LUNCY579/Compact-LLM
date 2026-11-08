"""Tiny lifecycle fixtures; no checkpoint, donor, GPU or efficacy inference."""
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch

from cross_scale.generation_stage.v1.src import execution as ex
from shared.hashing import canonical_sha256, write_json, sha256_file
from armr.metrics import output_flags, logit_metrics


@pytest.fixture
def context(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.json'
    config = {'role': 'architecture_dev', 'cells': {'X0': {'arm': 'X0', 'gate': 0.},
                'XG_g0.03': {'arm': 'XG', 'gate': .03}}, 'shard_count': 8,
              'seed': 17, 'threads': 4, 'dtype': 'float32', 'device': 'cpu',
              'max_prompt_seconds': 900, 'hidden_rms_ratio_bounds': [.25, 4.], 'top_k': 3,
              'dataset_manifest': 'test.json', 'dataset_manifest_sha256': 'd' * 64,
              'upstream_export': {'manifest_sha256': 'e' * 64},
              'matched_control': {'artifact_sha256': 'f' * 64}, 'checkpoint_sha256': 'c' * 64}
    write_json(config_path, config)
    registration = {'registration_sha256': 'a' * 64, 'sources': {'source.py': 'b' * 64}, 'config': config}
    cases = [{'example_id': f'case_{i}', 'case_id': f'case_{i}', 'scenario_id': f'scenario_{i}',
              'track': 'cross_scale', 'split': 'architecture_dev', 'variant': 'original',
              'family': 'reasoning', 'template_id': 't', 'template_unseen_in_engineering': False,
              'scorer': {'type': 'exact', 'answers': ['yes']}} for i in range(128)]
    monkeypatch.setattr(ex, 'EVIDENCE', tmp_path / 'evidence')
    monkeypatch.setattr(ex, 'AUDITS', tmp_path / 'audits')
    monkeypatch.setattr(ex, '_context', lambda path: (config_path, config, registration, cases))
    from shared import integrity
    monkeypatch.setattr(integrity, 'validate_inference_receipt', lambda value, cfg: True)
    return config_path, config, registration, cases


def _row(case, cell, config, run_id):
    return {**{k: case[k] for k in ex.ROW_METADATA}, 'cell_id': cell,
            'arm': config['cells'][cell]['arm'], 'gate': config['cells'][cell]['gate'], 'run_id': run_id,
            'response': 'yes', 'token_ids': [1, 2], 'score': 1., 'correct': True,
            'score_ambiguous': False, 'safety_stopped': False, 'gold_nll': .5,
            'termination': 'eos', 'eos_emitted': True, 'truncated': False,
            'trajectory_min_rms_ratio': 1., 'trajectory_max_rms_ratio': 1.,
            'added_parameter_count': 0, 'added_active_parameters': 0, 'extra_matmul_flops_per_token': 0,
            **output_flags('yes', [1, 2]), **logit_metrics(torch.arange(4.).float(), torch.arange(4.).float(), 3)}


def _attempt(context, cell='X0', index=0, attempt=0, status='COMPLETE', failure_class='hard_failure'):
    config_path, config, registration, cases = context
    contract, selected = ex._expected_contract(ex._binding(config_path, config, registration), cell, config, cases, index, attempt)
    directory = ex._root(registration) / cell / f'shard_{index:03d}' / f'attempt_{attempt:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    started = {**contract, 'status': 'STARTED', 'start_utc': '2026-09-12T00:00:00+00:00',
               'pid': 999999999, 'run_id': f'{cell}:{index}:{attempt}'}
    write_json(directory / 'started.json', started)
    final = {**started, 'status': status, 'end_utc': '2026-09-12T00:01:00+00:00'}
    if status == 'COMPLETE':
        rows = []
        for case in selected:
            name = canonical_sha256(case['example_id'])
            row = _row(case, cell, config, started['run_id'])
            write_json(directory / 'cases' / (name + '.json'), row)
            (directory / 'logits').mkdir(exist_ok=True)
            with (directory / 'logits' / (name + '.pt')).open('xb') as file:
                torch.save(torch.arange(4.).float(), file)
            rows.append(row)
        write_json(directory / 'results.json', rows)
        final.update(completed_examples=len(selected), operator_integrity_passed=True,
                     operator={'added_parameter_count': 0, 'added_active_parameters': 0,
                               'extra_matmul_flops_per_token': 0})
    elif status == 'FAILED':
        failure = {'type': 'RuntimeError', 'message': 'retained test failure'}
        write_json(directory / 'failure.json', failure)
        final.update(failure_class=failure_class, failure_receipt=failure, completed_examples=0, model_loaded=False)
    if status != 'STARTED':
        final.update(started_sha256=sha256_file(directory / 'started.json'), artifacts_sha256=ex._artifacts(directory))
        write_json(directory / 'run_manifest.json', final)
    return directory


def test_registered_cells_and_shards_only(context):
    path, config, registration, cases = context
    binding = ex._binding(path, config, registration)
    for index, count in [(-1, 8), (8, 8), (True, 8)]:
        with pytest.raises((ValueError, ex.EvidenceError)):
            ex._expected_contract(binding, 'X0', config, cases, index, 0)
    with pytest.raises(ex.EvidenceError, match='unregistered cell'):
        ex._expected_contract(binding, 'XN', config, cases, 0, 0)
    with pytest.raises(ex.EvidenceError, match='attempt'):
        ex._expected_contract(binding, 'X0', config, cases, 0, -1)


def test_eight_shards_cover_each_scenario_once(context):
    _, config, _, cases = context
    shards = [ex.shard_cases(cases, i, config['shard_count']) for i in range(8)]
    assert all(shards)
    assert sorted(c['example_id'] for shard in shards for c in shard) == sorted(c['example_id'] for c in cases)


@pytest.mark.parametrize('key,value', [('split', 'confirmatory_dev'), ('track', 'native_armr'),
                                    ('variant', 'paraphrase')])
def test_generation_bank_rejects_foreign_roles_tracks_variants(context, key, value):
    cases = [dict(row) for row in context[3]]
    cases[0][key] = value
    with pytest.raises(ex.EvidenceError, match='foreign'):
        ex.validate_screen_cases(cases)


def test_generation_bank_rejects_partial_or_duplicate_scenarios(context):
    cases = [dict(row) for row in context[3]]
    with pytest.raises(ex.EvidenceError, match='128 distinct'):
        ex.validate_screen_cases(cases[:-1])
    cases[1]['scenario_id'] = cases[0]['scenario_id']
    with pytest.raises(ex.EvidenceError, match='128 distinct'):
        ex.validate_screen_cases(cases)


def test_baseline_requires_entire_bank(context):
    _attempt(context)
    path, config, registration, cases = context
    state = ex.inventory(path, config, registration, cases)
    with pytest.raises(ex.EvidenceError, match='complete matched X0'):
        ex._baseline(state, config, cases)
    for index in range(1, 8):
        _attempt(context, index=index)
    state = ex.inventory(path, config, registration, cases)
    assert len(ex._baseline(state, config, cases)) == 8


def test_duplicate_complete_shards_rejected(context):
    _attempt(context, attempt=0)
    _attempt(context, attempt=1)
    with pytest.raises(ex.EvidenceError, match='duplicate complete'):
        ex.inventory(*context)


def test_changed_raw_case_hash_rejected(context):
    directory = _attempt(context)
    next((directory / 'cases').glob('*.json')).write_text('{}', encoding='utf-8')
    with pytest.raises(ex.EvidenceError, match='artifact'):
        ex.inventory(*context)


def test_failed_cell_retains_failure_without_partial_denominator(context):
    _attempt(context, cell='XG_g0.03', status='FAILED')
    state = ex.inventory(*context)
    assert state['XG_g0.03']['failed']['status'] == 'FAILED'
    assert state['XG_g0.03']['complete'] == {}


def test_resource_admission_failure_does_not_reject_cell(context):
    _attempt(context, status='FAILED', failure_class='resource_admission')
    state = ex.inventory(*context)
    assert state['X0']['failed'] is None
    assert len(state['X0']['attempts'][0]) == 1


def test_unfinished_attempt_requires_explicit_abandonment(context, monkeypatch):
    directory = _attempt(context, status='STARTED')
    with pytest.raises(ex.EvidenceError, match='unfinished worker'):
        ex.inventory(*context)
    import psutil
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: True)
    with pytest.raises(ex.EvidenceError, match='still live'):
        ex.abandon_shard(context[0], directory, 'worker interruption')
    monkeypatch.setattr(psutil, 'pid_exists', lambda pid: False)
    result = ex.abandon_shard(context[0], directory, 'worker interruption')
    assert ex._read(result)['original_process_absent'] is True
    assert ex.inventory(*context)['X0']['failed'] is None
    with pytest.raises(ex.EvidenceError, match='already terminal'):
        ex.abandon_shard(context[0], directory, 'cannot overwrite')


@pytest.mark.parametrize('key,value', [('role', 'confirmatory_dev'), ('dataset_manifest_sha256', '0' * 64),
                                    ('source_sha256', '0' * 64), ('cell_id', 'XN')])
def test_started_binding_cannot_be_rewritten(context, key, value):
    directory = _attempt(context)
    final = ex._read(directory / 'run_manifest.json')
    final[key] = value
    # Deliberately mutate only a fixture to model tampering.
    (directory / 'run_manifest.json').write_text(json.dumps(final), encoding='utf-8')
    with pytest.raises(ex.EvidenceError, match='rewrote'):
        ex.inventory(*context)


def test_raw_rescoring_detects_internally_rehashed_false_score(context):
    directory = _attempt(context)
    path, config, registration, cases = context
    state = ex.inventory(*context)
    entry = state['X0']['complete'][0]
    raw = next((directory / 'cases').glob('*.json'))
    value = ex._read(raw)
    value['score'] = 0.
    raw.write_text(json.dumps(value), encoding='utf-8')
    final = ex._read(directory / 'run_manifest.json')
    final['artifacts_sha256'] = ex._artifacts(directory)
    (directory / 'run_manifest.json').write_text(json.dumps(final), encoding='utf-8')
    entry['manifest'] = final
    entry['manifest_sha256'] = sha256_file(directory / 'run_manifest.json')
    with pytest.raises(ex.EvidenceError, match='rescoring'):
        ex._read_complete(entry, config, cases, 'X0')


def test_unknown_cell_directory_rejected(context):
    (ex._root(context[2]) / 'unregistered').mkdir(parents=True)
    with pytest.raises(ex.EvidenceError, match='unknown'):
        ex.inventory(*context)


def test_exclusive_attempt_creation(context):
    _attempt(context)
    with pytest.raises(FileExistsError):
        _attempt(context)


@pytest.mark.parametrize('mode,gate,enabled', [(None, .03, True), ('zero_gate', 0., True), ('disabled', .03, False)])
def test_factory_diagnostic_constructor_flags_do_not_change_config(context, monkeypatch, mode, gate, enabled):
    from cross_scale.generation_stage.v1.src import compute_control, control_calibration
    config = json.loads(json.dumps(context[1]))
    config['cells']['XM_g0.03'] = {'arm': 'XM', 'gate': .03}
    config['selected_pair'] = {'supa_layer': 19, 'donor_layer': 24}
    config['matched_control'].update(alpha=.7, state_sha256='s' * 64, artifact_path='fixture.json')
    before = json.loads(json.dumps(config))
    monkeypatch.setattr(control_calibration, 'verify_calibration_artifact', lambda *a, **kw: {
        'alpha': .7, 'control_state_sha256': 's' * 64})
    monkeypatch.setattr(compute_control, 'CrossScaleComputeControl', lambda model, **kw: kw)
    wrapper = ex.operator_factory(object(), 'XM_g0.03', config, diagnostic_mode=mode)
    assert wrapper['gate'] == gate and wrapper['enabled'] is enabled
    assert wrapper['alpha'] == .7
    assert config == before


def test_factory_rejects_unregistered_diagnostic_mode_before_constructing(context):
    with pytest.raises(ex.EvidenceError, match='unknown qualification'):
        ex.operator_factory(object(), 'X0', context[1], diagnostic_mode='arbitrary')


def test_fragment_audit_rescores_and_recomputes_retained_logits(context):
    _attempt(context)
    state = ex.inventory(*context)
    entry = state['X0']['complete'][0]
    fragments = ex._audit_fragments(entry, context[1], context[3], entry)
    assert len(fragments) == len(ex.shard_cases(context[3], 0, 8))
    assert all(row['cell_id'] == 'X0' for row in fragments)


def test_saved_logit_metric_tampering_rejected(context):
    _attempt(context)
    entry = ex.inventory(*context)['X0']['complete'][0]
    row = ex._read_complete(entry, context[1], context[3], 'X0')[0]
    row['logit_cosine'] = .5
    with pytest.raises(ex.EvidenceError, match='metrics do not reproduce'):
        ex._audit_logits(entry, row, entry, context[1])


def test_real_shard_path_failure_preserves_partial_and_closes_cell(context, monkeypatch):
    """Exercise the actual worker lifecycle using tiny fake inference dependencies."""
    from cross_scale.generation_stage.v1.src import provenance, qualification
    from shared import model_loading, integrity, generation, timing
    from armr import hooks
    _, config, registration, _ = context
    monkeypatch.setattr(provenance, 'stage_sources', lambda: registration['sources'])
    monkeypatch.setattr(qualification, 'verify_cell_qualification', lambda *args: {
        'status': 'PASS', 'receipt_sha256': 'q' * 64, 'directory': '/fixture/qualification'})
    monkeypatch.setattr(ex, 'git_identity', lambda: {'branch': 'test', 'commit_sha': 'fixture'})
    monkeypatch.setattr(timing, 'machine_lease', lambda: nullcontext())
    monkeypatch.setattr(hooks, 'frozen_inference', lambda model: nullcontext())
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    identity = {'state_sha256_before': 's', 'model_path': '/fixture/model', 'checkpoint_files_before': {}}
    monkeypatch.setattr(model_loading, 'load_supa', lambda *args: (model, None, identity))
    monkeypatch.setattr(integrity, 'assert_unchanged', lambda *args: {'canonical_unchanged': True})
    monkeypatch.setattr(ex, 'operator_factory', lambda *args: nullcontext())
    seen = []
    def evaluate(model, tokenizer, case, config, wrapper, base):
        if seen:
            raise RuntimeError('malformed generation: repetition/script/Unicode guard')
        seen.append(case['example_id'])
        return _row(case, 'X0', config, 'replaced_by_worker'), torch.arange(4.).float()
    monkeypatch.setattr(generation, 'evaluate_case', evaluate)
    with pytest.raises(RuntimeError, match='malformed generation'):
        ex.run_shard(context[0], 'X0', 0)
    state = ex.inventory(*context)
    assert state['X0']['failed'] is not None
    directory = Path(state['X0']['failed']['directory'])
    assert len(list((directory / 'cases').glob('*.json'))) == 1
    assert len(list((directory / 'logits').glob('*.pt'))) == 1
    assert not (directory / 'results.json').exists()
    with pytest.raises(ex.CellRejected):
        ex.run_shard(context[0], 'X0', 1)
