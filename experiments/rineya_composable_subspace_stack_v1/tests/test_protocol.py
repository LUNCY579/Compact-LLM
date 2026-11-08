from copy import deepcopy
from pathlib import Path
import pytest
from experiments.rineya_composable_subspace_stack_v1.protocol import (
    default_plan, validate_plan, read_json, digest, write_new, audit_splits,
    source_digest, freeze, admit, execution_lease, registered_names)
from experiments.rineya_composable_subspace_stack_v1.artifacts import build, preflight


def grant_for(tmp_path, action='build'):
    plan = default_plan()
    grant = dict(experiment='CSS-001', action=action, plan_sha256=digest(plan), code_sha256=source_digest(),
                 authorized=True, operator_reviewed=True, arm='P1S1C1', expires_utc='2099-01-01T00:00:00Z',
                 grant_id='test-grant', max_attempts=1, lease_path=str(tmp_path/'shared.lock'),
                 attempt_directory=str(tmp_path), output_parent=str(tmp_path))
    return plan, grant


def ledgers():
    return {s: [{'id':s+'-id', 'parent_id':s+'-parent', 'prompt':s+' prompt'}] for s in ('TRAIN', 'CAL', 'CHECK')}


def test_default_plan_and_unique_ablation_aliases():
    p = default_plan()
    assert len(validate_plan(p)) == 11
    assert p['leave_one_out']['minus_ctm'] == 'P1S1C0'
    assert not p['native_authorized'] and not p['training_authorized']


@pytest.mark.parametrize('field,value', [('unknown', 1), ('coefficients', [.5]), ('coefficients', [1., 1.]),
    ('donors', ['opallo', 'opallo']), ('tensors', []), ('tensors', ['model.embed_tokens.weight']),
    ('max_tensor_elements', 8_000_000), ('native_authorized', 'true'), ('retention_margin', float('nan'))])
def test_bad_plans(field, value):
    p = default_plan(); p[field] = value
    with pytest.raises(ValueError):
        validate_plan(p)


@pytest.mark.parametrize('field,value', [('rank', -1), ('iterations', 1000), ('private_gain', 1.1),
                                        ('agreement', 0), ('beta', float('nan'))])
def test_bad_settings(field, value):
    p = default_plan(); p['settings'][field] = value
    with pytest.raises(ValueError):
        validate_plan(p)


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '[]'])
def test_json_refuses_ambiguous_or_nonfinite(tmp_path, text):
    p = tmp_path/'x.json'; p.write_text(text)
    with pytest.raises(ValueError):
        read_json(p)


def test_exclusive_evidence_output(tmp_path):
    p = tmp_path/'f.json'
    write_new(p, {'a': 1})
    with pytest.raises(FileExistsError):
        write_new(p, {'a': 2})
    assert read_json(p) == {'a': 1}


@pytest.mark.parametrize('field', ['id', 'parent_id', 'prompt'])
def test_cross_split_leakage(field):
    rows = ledgers(); rows['CHECK'][0][field] = rows['TRAIN'][0][field]
    with pytest.raises(ValueError, match='leakage'):
        audit_splits(rows)


def test_unicode_whitespace_prompt_leakage():
    rows = ledgers(); rows['TRAIN'][0]['prompt'] = 'Ａ   B'
    rows['CHECK'][0]['prompt'] = 'A B'
    with pytest.raises(ValueError, match='prompt leakage'):
        audit_splits(rows)


def test_disjointness_not_ancestry_completeness():
    report = audit_splits(ledgers())
    assert report['status'] == 'SUPPLIED_LEDGERS_DISJOINT'
    assert report['ancestry_completeness_verified'] is False


@pytest.mark.parametrize('field,value', [('authorized', False), ('operator_reviewed', False),
    ('plan_sha256', '0'*64), ('code_sha256', '0'*64), ('action', 'train'),
    ('expires_utc', '2000-01-01T00:00:00Z'), ('expires_utc', '2099-01-01T00:00:00'),
    ('grant_id', '../escape'), ('max_attempts', 2), ('lease_path', 'relative.lock'), ('arm', 'other')])
def test_admission_refusals(tmp_path, field, value):
    p, g = grant_for(tmp_path); g[field] = value
    with pytest.raises(ValueError):
        admit(p, g, 'build', arm='P1S1C1')


@pytest.mark.parametrize('action', ['collect', 'train'])
def test_native_remains_closed_even_with_grant(tmp_path, action):
    p, g = grant_for(tmp_path, action)
    with pytest.raises(ValueError, match='native'):
        admit(p, g, action, arm='P1S1C1')


def test_preflight_refuses_before_checkpoint_io(tmp_path, monkeypatch):
    from experiments.rineya_composable_subspace_stack_v1 import artifacts
    p, g = grant_for(tmp_path, 'preflight'); g['authorized'] = False
    def unexpected(*args):
        pytest.fail('checkpoint I/O before grant admission')
    monkeypatch.setattr(artifacts, 'file_hash', unexpected)
    with pytest.raises(ValueError, match='grant'):
        preflight(p, Path('/nonexistent/manifest'), g)


def test_lease_single_attempt_and_ownership(tmp_path):
    p, g = grant_for(tmp_path)
    admit(p, g, 'build', arm='P1S1C1')
    with execution_lease(g, memory_reader=lambda: 8*1024**3) as guard:
        assert Path(g['lease_path']).exists()
        guard()
    assert not Path(g['lease_path']).exists()
    assert (tmp_path/'test-grant.spent.json').exists()
    with pytest.raises(FileExistsError):
        with execution_lease(g, memory_reader=lambda: 8*1024**3):
            pass
    assert not Path(g['lease_path']).exists()


def test_lease_wont_steal_existing_lock(tmp_path):
    _, g = grant_for(tmp_path)
    lock = Path(g['lease_path']); lock.write_text('another experiment')
    with pytest.raises(FileExistsError):
        with execution_lease(g, memory_reader=lambda: 8*1024**3):
            pass
    assert lock.read_text() == 'another experiment'


def test_host_memory_race_consumes_attempt(tmp_path):
    _, g = grant_for(tmp_path)
    readings = iter([8*1024**3, 7*1024**3])
    with pytest.raises(ValueError, match='7.4'):
        with execution_lease(g, memory_reader=lambda: next(readings)):
            pytest.fail('admitted after memory drop')
    assert (tmp_path/'test-grant.spent.json').exists()
    assert not Path(g['lease_path']).exists()


def test_worker_memory_guard(tmp_path):
    _, g = grant_for(tmp_path)
    readings = iter([8*1024**3, 8*1024**3, 5*1024**3])
    with execution_lease(g, memory_reader=lambda: next(readings)) as guard:
        with pytest.raises(ValueError, match='6.0'):
            guard()


def test_freeze_completeness_and_hash_binding():
    p = default_plan(); artifacts = {name: digest(name) for name in registered_names(p)}
    f = freeze(p, artifacts, cal_sha256=digest('CAL'), check_sha256=digest('CHECK'), audit_sha256=digest('audit'))
    assert f['plan_sha256'] == digest(p) and not f['check_tuning_permitted']
    with pytest.raises(ValueError, match='base'):
        freeze(p, {}, cal_sha256=digest('CAL'), check_sha256=digest('CHECK'), audit_sha256=digest('audit'))
    with pytest.raises(ValueError, match='identical'):
        freeze(p, artifacts, cal_sha256=digest('CAL'), check_sha256=digest('CAL'), audit_sha256=digest('audit'))


def test_subset_freeze_does_not_require_nineteen_resident_checkpoints():
    p = default_plan()
    f = freeze(p, {'base':digest('base'), 'P1S1C1':digest('full')},
               cal_sha256=digest('CAL'),check_sha256=digest('CHECK'),audit_sha256=digest('audit'))
    assert len(f['artifacts']) == 2 and len(f['registered_but_not_screened']) == 17


def test_controls_include_full_endpoints_scoped_skills_and_matched_doses():
    from experiments.rineya_composable_subspace_stack_v1.protocol import control_recipes
    p = default_plan(); controls = control_recipes(p)
    assert len(controls) == 8
    assert controls['endpoint_titan_sparse']['source'] == 'titan_sparse'
    assert controls['scoped_titan_sparse']['coefficients'] == [1., 0.]
    assert controls['dose_opallo']['coefficients'] == [0., .5]
    assert controls['delta_sum']['coefficients'] == [1., 1.]


def test_native_collector_refuses_before_runtime_or_model_io(tmp_path, monkeypatch):
    from experiments.rineya_composable_subspace_stack_v1 import native
    p, g = grant_for(tmp_path, 'collect')
    def unexpected(*args):
        pytest.fail('native runtime inspected before disabled-plan refusal')
    monkeypatch.setattr(native, 'runtime_identity', unexpected)
    with pytest.raises(ValueError, match='native execution'):
        native.collect(p,g,Path('/missing'),Path('/missing'),tmp_path/'out',{})


def test_inherited_source_change_fails_closed(tmp_path, monkeypatch):
    from experiments.rineya_composable_subspace_stack_v1 import artifacts
    folder = tmp_path/'experiments'/'rineya_composable_subspace_stack_v1'
    folder.mkdir(parents=True)
    source = folder.parent/'titan_model_soup_mini_v1'/'soup.py'
    source.parent.mkdir(); source.write_text('raise RuntimeError("must not execute changed source")')
    monkeypatch.setattr(artifacts, '__file__', str(folder/'artifacts.py'))
    with pytest.raises(ValueError, match='differs'):
        artifacts.parent_bridge()
