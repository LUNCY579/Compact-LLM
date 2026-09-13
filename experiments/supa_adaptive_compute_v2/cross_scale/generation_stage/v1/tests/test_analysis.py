"""Fixed-dose selection and failure handling on synthetic scored scenarios."""
import pytest

from cross_scale.generation_stage.v1.src import analysis
from cross_scale.generation_stage.v1.src.protocol import GATES, cell_id
from cross_scale.generation_stage.v1.tests.test_protocol import synthetic_config


@pytest.fixture(autouse=True)
def short_resampling(monkeypatch):
    # Exercise the real shared paired functions with fewer fixture-only draws;
    # production config retains its fixed 10,000 resamples.
    original = analysis.paired_analysis
    def short(*args, **kwargs):
        kwargs['bootstrap_samples'] = 10
        return original(*args, **kwargs)
    monkeypatch.setattr(analysis, 'paired_analysis', short)


def fixture():
    cases = [{'example_id': f'e{i}', 'scenario_id': f's{i}',
        'family': analysis.FAMILIES[i // 16], 'track': 'cross_scale',
        'split': 'architecture_dev', 'variant': 'original',
        'template_id': f'{analysis.FAMILIES[i // 16]}.toy',
        'template_unseen_in_engineering': False,
        'scorer': {'type': 'exact', 'answers': ['yes']}} for i in range(128)]
    correct = {i for i in range(128) if i % 4 in (0, 1)}
    def rows(indices):
        return [{**{key: value for key, value in case.items() if key != 'scorer'},
            'response': 'yes' if i in indices else 'no', 'token_ids': [1 if i in indices else 2],
            'score': float(i in indices), 'invalid': False, 'safety_stopped': False,
            'termination': 'eos', 'gold_nll': .2, 'logit_cosine': .99,
            'kl_base_to_arm': .001, 'wall_seconds': 1.,
            'generation_dense_matmul_flops_estimate': 100., 'peak_rss_bytes': 1000,
            'trajectory_min_rms_ratio': 1., 'trajectory_max_rms_ratio': 1.,
            'hidden': [{'rms': 1.}], 'added_parameter_count': 10} for i, case in enumerate(cases)]
    config = synthetic_config()
    by_cell = {name: rows(correct) for name in config['cells']}
    return config, cases, correct, rows, by_cell


def test_inert_cells_cannot_pass_without_arbitrary_identity_threshold():
    config, cases, _, _, by_cell = fixture()
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    assert result['status'] == 'SCREEN_NO_GO'
    assert result['selected'] is None
    assert len(result['doses']) == 4 and len(result['cells']) == 17
    assert result['cells']['XG_g0.03']['paired_statistics']['overall']['output_identity']['rate'] == 1.
    assert result['efficacy_claim_licensed'] is False


def test_exact_minimum_and_lower_gate_tiebreak():
    config, cases, correct, rows, by_cell = fixture()
    improved = correct | {2, 3}
    improved.remove(0)  # two paired wins, one loss; net one.
    for gate in GATES:
        by_cell[cell_id('XG', gate)] = rows(improved)
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    assert result['status'] == 'SCREEN_GO'
    assert result['ranked_eligible_gates'] == list(GATES)
    assert result['selected']['gate'] == .03
    assert result['confirmatory_generation_licensed'] is False


def test_ranking_prioritizes_margin_before_gain():
    config, cases, correct, rows, by_cell = fixture()
    by_cell['XG_g0.03'] = rows(correct | {2, 3, 6, 7})
    by_cell['XM_g0.03'] = rows(correct | {2, 3, 6})
    by_cell['XG_g0.10'] = rows(correct | {2, 3})
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    assert result['selected']['gate'] == .10


def test_control_ties_allowed_but_control_winner_rejects():
    config, cases, correct, rows, by_cell = fixture()
    for gate in (.03, .10):
        by_cell[cell_id('XG', gate)] = rows(correct | {2, 3})
    by_cell['XS_g0.03'] = rows(correct | {2, 3})
    by_cell['XR_g0.10'] = rows(correct | {2, 3, 6})
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    assert result['selected']['gate'] == .03
    assert 'aligned_graft_below_same_dose_control' in result['doses'][1]['rejection_reasons']


def test_additional_preservation_errors_reject_even_positive_overall_gain():
    config, cases, correct, rows, by_cell = fixture()
    changed = (correct | {2, 3, 6, 7}) - {32, 33}
    by_cell['XG_g0.03'] = rows(changed)
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    dose = result['doses'][0]
    assert dose['XG_net_correct_gain'] == 2
    assert dose['preservation']['additional_XG_errors'] == 2
    assert 'preservation_error_budget_exceeded' in dose['rejection_reasons']


def test_invalid_control_rejects_same_dose():
    config, cases, correct, rows, by_cell = fixture()
    by_cell['XG_g0.03'] = rows(correct | {2, 3})
    by_cell['XM_g0.03'][2].update(response='', invalid=True, score=0.)
    result = analysis.analyze_screen(cases, by_cell, {}, config)
    assert 'additional_invalid_output_in_same_dose_cell' in result['doses'][0]['rejection_reasons']


@pytest.mark.parametrize('phase', ['qualification', 'generation'])
def test_failed_cell_retains_partial_rows_without_accuracy_and_other_doses_continue(phase):
    config, cases, correct, rows, by_cell = fixture()
    for gate in GATES:
        by_cell[cell_id('XG', gate)] = rows(correct | {2, 3})
    by_cell['XM_g0.03'] = by_cell['XM_g0.03'][:3]
    failures = {'XM_g0.03': {'status': 'FAILED', 'phase': phase, 'reason': 'toy failure'}}
    result = analysis.analyze_screen(cases, by_cell, failures, config)
    assert result['selected']['gate'] == .10
    assert result['cells']['XM_g0.03']['accuracy'] is None
    assert result['cells']['XM_g0.03']['retained_partial_examples'] == 3
    assert result['cells']['XS_g0.03']['status'] == 'COMPLETE'


@pytest.mark.parametrize('mutator', [
    lambda c, rows: rows.pop('XM_g0.03'),
    lambda c, rows: rows['X0'].pop(),
    lambda c, rows: rows['XG_g0.03'].pop(),
    lambda c, rows: rows['XG_g0.03'][0].update(score=0.),
    lambda c, rows: rows['XG_g0.03'][0].update(safety_stopped=True),
    lambda c, rows: rows['XG_g0.03'][0].update(track='native_armr'),
    lambda c, rows: rows['XG_g0.03'][0].update(trajectory_max_rms_ratio=5.),
    lambda c, rows: rows['XG_g0.03'][0].update(invalid=True),
    lambda c, rows: c[0].update(split='confirmatory_dev'),
])
def test_incomplete_or_changed_evidence_refused(mutator):
    config, cases, _, _, by_cell = fixture()
    mutator(cases, by_cell)
    with pytest.raises(ValueError):
        analysis.analyze_screen(cases, by_cell, {}, config)


def test_synthetic_sibling_failure_and_failed_baseline_refused():
    config, cases, _, _, by_cell = fixture()
    for failed in ({'XM_g0.03': {'status': 'FAILED', 'phase': 'dose_rejected'}},
                   {'X0': {'status': 'FAILED', 'phase': 'generation'}}):
        with pytest.raises(ValueError):
            analysis.analyze_screen(cases, by_cell, failed, config)
