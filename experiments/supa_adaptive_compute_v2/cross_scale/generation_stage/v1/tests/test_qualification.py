"""Synthetic qualification evidence and tiny tensors; no real model runs."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import pytest
import torch

from cross_scale.generation_stage.v1.src import qualification as q
from shared.hashing import write_json, sha256_file


def config():
    return {'device': 'cuda:0', 'seed': 2702, 'model_class': 'Qwen3AbstainForCausalLM',
        'checkpoint_sha256': 'a' * 64, 'dataset_manifest_sha256': 'b' * 64,
        'min_available_gib': .75, 'cells': {'X0': {'arm': 'X0', 'gate': 0.}},
        'qualification': {'capture_layers': [2, 19, 27], 'cpu_min_available_gib': 3.5,
            'max_abs_logit_error': .002, 'max_relative_logit_l2': .0002,
            'minimum_logit_cosine': .999999, 'max_abs_nll_error': .002,
            'max_relative_hidden_l2': .0002, 'minimum_hidden_cosine': .999999}}


def fixtures():
    return [{'example_id': f'case_{i}', 'prompt_sha256': str(i) * 64} for i in range(8)]


def report(phase='cpu'):
    cfg = config()
    result = {'cell_id': 'X0', 'fixture_ids': [r['example_id'] for r in fixtures()],
        'fixture_prompt_sha256': {r['example_id']: r['prompt_sha256'] for r in fixtures()},
        'model_identity': {'canonical_unchanged': True, 'model_class': cfg['model_class'],
            'state_sha256_before': 'a' * 64, 'state_sha256_after': 'a' * 64,
            'checkpoint_files_before': {'model.safetensors': cfg['checkpoint_sha256']},
            'checkpoint_files_after': {'model.safetensors': cfg['checkpoint_sha256']},
            'qwen_implementation_sha256': 'f' * 64,
            'trainable_parameters': 0, 'backbone_gradients': 0},
        'phase_runtime_identity': {'device': 'cpu' if phase == 'cpu' else 'cuda:0',
            'dtype': 'float32', 'threads': 4, 'seed': 2702, 'torch': '2.12.1+cu126',
            'transformers': '5.14.1', 'attention': 'eager', 'use_cache': False},
        'operator_before': {'parameters': 0, 'state_sha256': None, 'buffers': {}},
        'operator_after': {'parameters': 0, 'state_sha256': None, 'buffers': {}},
        'disabled_identity': True, 'zero_gate_identity': True, 'restored_identity': True,
        'cases': {}}
    for item in fixtures():
        result['cases'][item['example_id']] = {'row': {'token_ids': [2, 3], 'gold_nll': 1., 'safety_stopped': False},
            'repeat_verified': phase == 'gpu', 'hidden_shapes': {str(i): [1, 2, 4] for i in (2, 19, 27)}}
    return result


def test_complete_phase_fixture_validation():
    assert q.validate_phase_report(report(), config(), fixtures(), 'cpu')
    assert q.validate_phase_report(report('gpu'), config(), fixtures(), 'gpu')


@pytest.mark.parametrize('mutation', ['coverage', 'prompt', 'state', 'checkpoint', 'gradient', 'runtime',
                                    'disabled', 'zero', 'restore', 'operator', 'repeat', 'nll', 'hidden'])
def test_phase_rejects_incomplete_or_changed_proof(mutation):
    r = report('gpu')
    row = r['cases']['case_0']
    if mutation == 'coverage': del r['cases']['case_0']
    elif mutation == 'prompt': r['fixture_prompt_sha256']['case_0'] = 'changed'
    elif mutation == 'state': r['model_identity']['state_sha256_after'] = 'changed'
    elif mutation == 'checkpoint': r['model_identity']['checkpoint_files_after'] = {}
    elif mutation == 'gradient': r['model_identity']['backbone_gradients'] = 1
    elif mutation == 'runtime': r['phase_runtime_identity']['threads'] = 2
    elif mutation == 'disabled': r['disabled_identity'] = False
    elif mutation == 'zero': r['zero_gate_identity'] = False
    elif mutation == 'restore': r['restored_identity'] = False
    elif mutation == 'operator': r['operator_after']['parameters'] = 1
    elif mutation == 'repeat': row['repeat_verified'] = False
    elif mutation == 'nll': row['row']['gold_nll'] = float('nan')
    else: del row['hidden_shapes']['19']
    with pytest.raises(RuntimeError):
        q.validate_phase_report(r, config(), fixtures(), 'gpu')


def tensor_reports(tmp_path, perturb=False, change_tokens=False):
    dirs = []
    for phase in ('cpu', 'gpu'):
        directory = tmp_path / phase
        directory.mkdir()
        r = report(phase)
        for index, item in enumerate(r['cases'].values()):
            values = torch.arange(1, 9, dtype=torch.float32)
            if perturb and phase == 'gpu' and index == 0:
                values[0] += .01
            if change_tokens and phase == 'gpu' and index == 0:
                item['row']['token_ids'] = [2, 4]
            path = directory / f'{index}.pt'
            torch.save({'logits': values, 'hidden': {str(i): torch.ones(1, 2, 4) for i in (2, 19, 27)}}, path)
            item.update(tensor_path=path.name, tensor_sha256=sha256_file(path))
        write_json(directory / 'report.json', r)
        dirs.append(directory)
    return dirs


def test_numerical_gates_recomputed_from_tensors(tmp_path):
    cpu, gpu = tensor_reports(tmp_path)
    result = q.compare_phase_reports(cpu, gpu, config())
    assert len(result['comparisons']) == 8 and result['pass']


@pytest.mark.parametrize('kind', ['numeric', 'greedy'])
def test_numeric_or_greedy_mismatch_rejects_cell(tmp_path, kind):
    cpu, gpu = tensor_reports(tmp_path, perturb=kind == 'numeric', change_tokens=kind == 'greedy')
    result = q.compare_phase_reports(cpu, gpu, config())
    assert not result['pass']
    assert sum(not r['pass'] for r in result['comparisons']) == 1


def test_changed_tensor_bytes_rejected(tmp_path):
    cpu, gpu = tensor_reports(tmp_path)
    (gpu / '0.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='tensors changed'):
        q.compare_phase_reports(cpu, gpu, config())


def identity_report(tmp_path, phase='gpu'):
    r = report(phase)
    sample = {'logits': torch.arange(1, 9, dtype=torch.float32),
              'hidden': {str(i): torch.ones(1, 2, 4) for i in (2, 19, 27)}}
    row = {'token_ids': [2, 3], 'gold_nll': 1., 'safety_stopped': False}
    r['identity_probes'] = {mode: q._save_capture(tmp_path, mode, (copy.deepcopy(row), sample))
                            for mode in ('baseline', 'disabled', 'zero_gate', 'restored')}
    for identity, item in r['cases'].items():
        item.update(q._save_capture(tmp_path, identity, (copy.deepcopy(row), sample)))
        if phase == 'gpu':
            item['repeat_capture'] = q._save_capture(tmp_path, identity + '_repeat', (copy.deepcopy(row), sample))
    return r


def test_all_raw_repeat_and_identity_artifacts_recomputed(tmp_path):
    r = identity_report(tmp_path)
    assert q.validate_identity_artifacts(tmp_path, r, 'gpu')


@pytest.mark.parametrize('kind', ['repeat', 'disabled', 'zero_gate', 'restored', 'missing_repeat'])
def test_claimed_identity_cannot_hide_rehashed_raw_mismatch(tmp_path, kind):
    r = identity_report(tmp_path)
    if kind == 'missing_repeat':
        del r['cases']['case_0']['repeat_capture']
    else:
        entry = r['cases']['case_0']['repeat_capture'] if kind == 'repeat' else r['identity_probes'][kind]
        entry['row']['token_ids'] = [9]
    with pytest.raises(RuntimeError):
        q.validate_identity_artifacts(tmp_path, r, 'gpu')


def test_graft_phase_requires_unchanged_donor_identity():
    cfg = config()
    cfg['cells']['XG_g0.03'] = {'arm': 'XG', 'gate': .03}
    cfg['donor_revision'] = 'donor-revision'
    r = report()
    r['cell_id'] = 'XG_g0.03'
    snapshot = {'parameters': 100930816, 'trainable_parameters': 0, 'backbone_gradients': 0,
                'state_sha256': 'a' * 64, 'buffers': {}}
    r.update(operator_before=copy.deepcopy(snapshot), operator_after=copy.deepcopy(snapshot),
             donor_identity_before={'revision': 'donor-revision', 'files': {'weights': 'a' * 64}},
             donor_identity_after={'revision': 'donor-revision', 'files': {'weights': 'a' * 64}})
    assert q.validate_phase_report(r, cfg, fixtures(), 'cpu')
    r['donor_identity_after']['files']['weights'] = 'b' * 64
    with pytest.raises(RuntimeError, match='donor checkpoint'):
        q.validate_phase_report(r, cfg, fixtures(), 'cpu')


def test_gpu_candidate_requires_complete_baseline_comparison(tmp_path, monkeypatch):
    stage = tmp_path / 'stage'
    (stage / 'configs').mkdir(parents=True)
    path = stage / 'configs/config.json'; path.write_text('{}')
    cfg = config(); cfg['cells']['XG_g0.03'] = {'arm': 'XG', 'gate': .03}
    monkeypatch.setattr(q, 'STAGE', stage)
    monkeypatch.setattr(q, 'EVIDENCE', stage / 'evidence')
    monkeypatch.setattr(q, 'verify_registration', lambda _: {'config': cfg, 'sources': {}, 'registration_sha256': 'a' * 64})
    seen = []
    def baseline(*args):
        seen.append(args[2]); raise q.QualificationPendingError('baseline comparison absent')
    monkeypatch.setattr(q, 'verify_cell_qualification', baseline)
    with pytest.raises(q.QualificationPendingError, match='baseline comparison absent'):
        q.run_phase(path, 'XG_g0.03', 'gpu')
    assert seen == ['X0']
    assert not (stage / 'evidence').exists()


@pytest.mark.parametrize('error,loaded,expected', [
    (MemoryError('insufficient available physical RAM: low'), False, 'resource_admission'),
    (MemoryError('RSS budget'), True, 'runtime_resource'),
    (RuntimeError('CUDA context lost'), True, 'runtime_resource'),
    (RuntimeError('malformed generation: repetition/script/Unicode guard'), True, 'cell_rejection'),
    (RuntimeError('hidden norm emergency'), True, 'cell_rejection'),
    (RuntimeError('operator state changed'), True, 'fatal_integrity'),
])
def test_operational_failures_are_not_scientific_rejections(error, loaded, expected):
    assert q._failure_class(error, loaded) == expected


def test_lease_collision_has_terminal_resource_receipt(tmp_path, monkeypatch):
    import shared.timing
    stage = tmp_path / 'stage'
    (stage / 'configs').mkdir(parents=True)
    config_path = stage / 'configs/config.json'
    config_path.write_text('{}')
    cfg = config()
    monkeypatch.setattr(q, 'STAGE', stage)
    monkeypatch.setattr(q, 'EVIDENCE', stage / 'evidence')
    monkeypatch.setattr(q, 'verify_registration', lambda path: {'config': cfg, 'sources': {}, 'registration_sha256': 'a' * 64})
    monkeypatch.setattr(q, '_fixtures', lambda cfg: fixtures())
    monkeypatch.setattr(q, 'git_identity', lambda: {'branch': 'fixture', 'commit_sha': '0' * 40})
    @contextmanager
    def occupied():
        raise RuntimeError('fixture lease collision')
        yield
    monkeypatch.setattr(shared.timing, 'machine_lease', occupied)
    with pytest.raises(q.QualificationPendingError):
        q.run_phase(config_path, 'X0', 'cpu')
    path = stage / 'evidence' / ('a' * 64) / 'X0/cpu/attempt_000/run_manifest.json'
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'FAILED' and receipt['failure_class'] == 'resource_admission'
    assert receipt['completed_fixtures'] == 0


def test_incomplete_phase_never_counts_as_qualified(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'EVIDENCE', tmp_path)
    reg = {'registration_sha256': 'a' * 64, 'sources': {}}
    directory = tmp_path / ('a' * 64) / 'X0/cpu/attempt_000'
    directory.mkdir(parents=True)
    write_json(directory / 'started.json', {'status': 'STARTED'})
    with pytest.raises(q.QualificationPendingError):
        q._find_phase(config(), reg, 'X0', 'cpu', committed=False)
