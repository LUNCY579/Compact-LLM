import copy
import pytest
from experiments.rineya_self_distillation_axes_v1.data import messages, score, synthetic_rows, validate_rows
from experiments.rineya_self_distillation_axes_v1.analysis import behavioral_metrics, crossfit_rate, last_three_endpoint, paired_contrast, pareto_indices
from experiments.rineya_self_distillation_axes_v1.plan import Arm, arms, make_plan, write_json


@pytest.mark.parametrize('text, answer, scorer, correct', [
    (r'work \boxed{3}', '3', 'boxed_int', True),
    (r'\boxed{2} final \boxed{3}', '3', 'boxed_int', True),
    (r'3', '3', 'boxed_int', False),
    (r'\boxed{9}', '10', 'boxed_base9', False),
    (r'\boxed{10}', '10', 'boxed_base9', True),
    (r'\boxed{(2, -3)}', '(2,-3)', 'boxed_pair', True),
    (r'\boxed{(2.0,-3)}', '(2,-3)', 'boxed_pair', False),
    ('answer', 'answer', 'exact', True),
    ('{"b":2,"a":1}', '{"a":1,"b":2}', 'json_exact', True),
])
def test_strict_scorers(text, answer, scorer, correct):
    assert score(text, answer, scorer) is correct


def test_paired_contradiction_changes_only_reference_and_answer():
    rows = synthetic_rows(parents_per_split=3)
    ordinary = [r for r in rows if r['task'] == 'spatial_ordinary']
    rotated = [r for r in rows if r['task'] == 'spatial_rotated']
    for a, b in zip(ordinary, rotated):
        assert (a['prompt'], a['parent_id'], a['split']) == (b['prompt'], b['parent_id'], b['split'])
        assert a['answer'] != b['answer']
    selected = validate_rows(rows, 'spatial_rotated')
    assert all(r['task'] == 'spatial_rotated' for r in selected if r['split'] == 'train')
    assert any(r['purpose'] == 'retention' for r in selected)


@pytest.mark.parametrize('mutation', ['split', 'admission', 'duplicate', 'parent', 'prompt', 'reference', 'retention_train'])
def test_data_fail_closed(mutation):
    rows = synthetic_rows(parents_per_split=3)
    train = next(r for r in rows if r['split'] == 'train')
    dev = next(r for r in rows if r['split'] == 'dev' and r['purpose'] == 'acquisition')
    if mutation == 'split': train['split'] = 'sealed'
    if mutation == 'admission': train['admitted_for'] = 'dev'
    if mutation == 'duplicate': dev['id'] = train['id']
    if mutation == 'parent': dev['parent_id'] = train['parent_id']
    if mutation == 'prompt': dev['prompt'] = train['prompt']
    if mutation == 'reference': train['reference'] = 'wrong'
    if mutation == 'retention_train': train['purpose'] = 'retention'
    with pytest.raises(ValueError): validate_rows(rows, 'spatial_ordinary')


def test_privileged_context_never_enters_student_input():
    row = {'prompt': 'Original task?', 'reference': 'SECRET_REFERENCE_PAYLOAD'}
    assert messages(row) == [{'role': 'user', 'content': 'Original task?'}]
    assert 'SECRET_REFERENCE_PAYLOAD' in messages(row, True)[0]['content']
    assert len(messages(row, True)) == 1


def evaluation():
    return [{'id': str(i), 'parent_id': str(i), 'purpose': p, 'task': t, 'correct': c}
            for i, (p, t, c) in enumerate([('acquisition','new',False), ('retention','old_a',True),
                                           ('retention','old_b',False), ('retention','old_b',False)])]


def test_macro_retention_and_source_successes():
    before = evaluation()
    after = copy.deepcopy(before)
    after[0]['correct'], after[1]['correct'], after[2]['correct'] = True, False, True
    m = behavioral_metrics(before, after)
    assert m['acquisition_pp'] == 100 and m['retention_pp'] == -25
    old = next(r for r in m['tasks'] if r['task'] == 'old_b')
    assert old['retained_fraction_of_source_correct'] is None


def test_missing_coverage_is_not_silently_dropped():
    with pytest.raises(ValueError): behavioral_metrics(evaluation(), evaluation()[:-1])


def test_last_three_acquisition_final_retention():
    before = evaluation()
    after = copy.deepcopy(before)
    after[0]['correct'] = True
    result = last_three_endpoint(before, [{'step': 1, 'rows': before}, {'step': 2, 'rows': after}, {'step': 3, 'rows': after}])
    assert abs(result['acquisition_pp'] - 200/3) < 1e-12
    assert result['retention_pp'] == 0 and not result['promotion_allowed']
    with pytest.raises(ValueError): last_three_endpoint(before, [])


def test_pareto_and_paired_contrasts():
    records = [{'acquisition_pp': 1, 'retention_pp': -2}, {'acquisition_pp': 2, 'retention_pp': -1},
               {'acquisition_pp': 3, 'retention_pp': -3}]
    assert pareto_indices(records) == [1, 2]
    assert paired_contrast({17:records[0]}, {17:records[1]}, 'acquisition_pp')['normal_95_ci'] is None
    with pytest.raises(ValueError): paired_contrast({17:records[0]}, {29:records[1]}, 'acquisition_pp')


def test_crossfit_never_selects_on_held_out_seed():
    records = [{'seed': s, 'ema_rate': a, 'acquisition_pp': (10 if (s % 2 == 0) == (a == 0) else 0)}
               for s in range(4) for a in (0., .1)]
    selected = crossfit_rate(records)
    assert all(r['acquisition_pp'] == 0 for r in selected)
    assert all(r['selection_fold'] != r['heldout_fold'] for r in selected)
    with pytest.raises(ValueError): crossfit_rate(records[:-1])


def test_factorial_grid_counts_and_sft_axis_separation():
    assert len(arms()) == 21
    assert len({a.name for a in arms()}) == 21
    assert make_plan()['run_count'] == 126
    assert not make_plan()['launch_authorized']
    with pytest.raises(ValueError): Arm('sft', 'teacher', 'forward', .1)
    with pytest.raises(ValueError): arms([0., 0.])


def test_output_never_overwrites(tmp_path):
    path = tmp_path / 'record.json'
    write_json(path, {'a': 1})
    with pytest.raises(FileExistsError): write_json(path, {'a': 2})


@pytest.mark.parametrize("text,answer", [('true','1'), ('{"a":1,"a":2}','{"a":2}'), ('NaN','NaN'), ('1.0','1')])
def test_json_scorer_rejects_type_coercion_duplicates_and_nan(text, answer):
    assert not score(text, answer, "json_exact")


def test_crossfit_rejects_mixed_other_axes():
    records = [{"seed": s, "ema_rate": a, "acquisition_pp": 1., "kl_direction": "reverse"}
               for s in range(4) for a in (0., .1)]
    records[0]["kl_direction"] = "forward"
    with pytest.raises(ValueError): crossfit_rate(records)
