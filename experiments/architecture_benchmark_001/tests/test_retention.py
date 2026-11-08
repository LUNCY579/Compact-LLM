"""Tests for the typed retention scorer and the retention arithmetic.

The counterexample table in `TYPED_SCORING_CONTRACT.md` is transcribed here row for row. That
table was written and hashed BEFORE this code existed, so these are specification cases, not
cases reverse-engineered from an implementation that already passed.

The programme's rule is that a cheap control beats an argument that a bug cannot happen: four
harness defects here were found by controls and none by code review. So the fixtures below
deliberately include the exact shapes that defeated the legacy scorer -- AM/PM, wrong units,
negation, equivalent duration units, and trailing punctuation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.architecture_benchmark_001 import gates, retention

ROOT = Path(__file__).resolve().parents[1]


def _num(value, bare=False):
    return {'kind': 'number', 'value': str(value), 'bare_allowed': bare}


TAGS = {'kind': 'count', 'value': '54', 'bare_allowed': True,
        'unit_aliases': ['sensor tag', 'sensor tags', 'tag', 'tags']}
TAGS_STRICT = dict(TAGS, bare_allowed=False)
CHAIRS = {'kind': 'count', 'value': '15', 'bare_allowed': False, 'qualifier': 'blue',
          'unit_aliases': ['folding chair', 'folding chairs', 'chair', 'chairs']}
MONEY = {'kind': 'money', 'value': '40', 'bare_allowed': True}
PERCENT = {'kind': 'percent', 'value': '40', 'bare_allowed': True}
CUPS = {'kind': 'volume_cups', 'value': '3.2', 'bare_allowed': False}
WATER = {'kind': 'volume_cups', 'value': '12', 'bare_allowed': False, 'qualifier': 'water'}
KM = {'kind': 'distance_km', 'value': '5', 'bare_allowed': False}
DUR45 = {'kind': 'duration', 'value': '2700', 'bare_allowed': True, 'bare_unit': 'minute'}
DUR45_STRICT = {'kind': 'duration', 'value': '2700', 'bare_allowed': False}
DUR75 = {'kind': 'duration', 'value': '4500', 'bare_allowed': False}
DUR90 = {'kind': 'duration', 'value': '5400', 'bare_allowed': False}
CLOCK_1025AM = {'kind': 'clock', 'minute_of_day': 10 * 60 + 25}
CLOCK_420PM = {'kind': 'clock', 'minute_of_day': 16 * 60 + 20}
CLOCK_12AM = {'kind': 'clock', 'minute_of_day': 0}
CLOCK_12PM = {'kind': 'clock', 'minute_of_day': 12 * 60}
CLOCK_120PM = {'kind': 'clock', 'minute_of_day': 13 * 60 + 20}

#: Every row of the contract's counterexample table, in the contract's order.
CONTRACT_CASES = [
    (_num(3), '3.', True),
    (_num('62.5'), 'The answer is 62.500.', True),
    (_num('0.5'), '1/2', True),
    (_num(1000), '1,000', True),
    (_num(1000), '10,00', False),
    (_num(5), 'not 5', False),
    (_num(5), '5 or 6', False),
    (_num(5), '5, but actually 6', False),
    (_num(5), '5 hours', False),
    (_num(5), '$5', False),
    (_num(5), '5%', False),
    (_num(5), 'NaN', False),
    (_num(5), '1/0', False),
    (_num(5), '5e0', False),
    (_num(5), '5?', False),
    (_num(5), '5...', False),
    (TAGS, '54', True),
    (TAGS, '54 sensor tags.', True),
    (TAGS, '54 spare fuses.', False),
    (CHAIRS, '15 blue chairs', True),
    (CHAIRS, '15 red folding chairs', False),
    (MONEY, '40', True),
    (MONEY, '$40.00', True),
    (MONEY, '40 euros', False),
    (PERCENT, '40', True),
    (PERCENT, '40%', True),
    (PERCENT, '0.4', False),
    (CUPS, '16/5 cups', True),
    (CUPS, '3.7 cups', False),
    (CUPS, '0.8 liters', False),
    (WATER, '12 cups of water', True),
    (WATER, '12 cups of oil', False),
    (KM, '5000 meters', True),
    (KM, '5 miles', False),
    (DUR45, '45', True),
    (DUR45, '0.75 hours', True),
    (DUR45, '2700 seconds', True),
    (DUR75, '1 hour and 15 minutes', True),
    (DUR75, '15 minutes and 1 hour', True),
    (DUR90, '90', False),
    (DUR45, '45 hours', False),
    (DUR45, '45 minutes or 1 hour', False),
    (CLOCK_1025AM, '10:25 PM', False),
    (CLOCK_1025AM, '10:55 AM', False),
    (CLOCK_420PM, '16:20', True),
    (CLOCK_420PM, '4:20', False),
    (CLOCK_12AM, 'midnight', True),
    (CLOCK_12PM, '12 PM', True),
    (CLOCK_12PM, '12 AM', False),
    (CLOCK_120PM, '13:20 PM', False),
    (CLOCK_12AM, '24:00', False),
]


@pytest.mark.parametrize('target,prediction,expected', CONTRACT_CASES,
                         ids=[f'{t["kind"]}:{p}' for t, p, _ in CONTRACT_CASES])
def test_every_contract_counterexample(target, prediction, expected):
    record = retention.score_prediction(prediction, True, target)
    assert record['correct'] is expected, record
    # A rejected answer must say why: a silent False is indistinguishable from a scorer bug.
    assert expected or record['reason']


def test_contract_case_count_matches_the_hashed_document():
    """The table and this list must stay the same size, and the document must stay hashed."""
    text = (ROOT / 'TYPED_SCORING_CONTRACT.md').read_text(encoding='utf-8')
    rows = [line for line in text.splitlines()
            if line.startswith('|') and line.rstrip().endswith(('| yes |', '| no |'))]
    assert len(rows) == len(CONTRACT_CASES), (len(rows), len(CONTRACT_CASES))
    policy = json.loads((ROOT / 'retention_policy.proposed.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((ROOT / 'TYPED_SCORING_CONTRACT.md').read_bytes()).hexdigest()
    assert policy['correctness']['grammar_contract_sha256'] == digest


def test_the_legacy_scorer_fails_cases_this_one_must_not():
    """The four shapes that motivated a typed grammar, asserted as a regression guard.

    If `scoring.answer_matches` is ever repaired, this test should be updated deliberately --
    not deleted -- because the retention gate must not silently revert to number-sequence
    matching.
    """
    from experiments.rab001_exploration.scoring import answer_matches
    # Legacy accepts these; the typed grammar must not.
    assert answer_matches('10:25 PM', '10:25 AM.')
    assert not retention.score_prediction('10:25 PM', True, CLOCK_1025AM)['correct']
    assert answer_matches('5 hours', '5 minutes.')
    assert not retention.score_prediction('5 hours', True, DUR45_STRICT)['correct']
    assert answer_matches('not 5', '5.')
    assert not retention.score_prediction('not 5', True, _num(5))['correct']
    # Legacy rejects this; the typed grammar must accept it.
    assert not answer_matches('1 hour', '60 minutes.')
    assert retention.score_prediction('1 hour', True,
                                      {'kind': 'duration', 'value': '3600',
                                       'bare_allowed': False})['correct']


def test_termination_is_required_even_for_a_perfect_answer():
    record = retention.score_prediction('3', False, _num(3))
    assert record['correct'] is False and record['reason'] == 'did not terminate'
    assert record['parsed'] is None


def test_scoring_record_keeps_raw_text_and_cannot_be_overridden_by_a_caller():
    record = retention.score_prediction('  54  Sensor Tags. ', True, TAGS)
    assert record['correct'] is True
    assert record['raw'] == '  54  Sensor Tags. '          # raw text preserved verbatim
    assert record['parsed'] == '54'
    # score_prediction takes no correctness argument at all, so a supplied verdict has nowhere
    # to enter. Recomputation from the record reproduces the same answer.
    assert retention.score_prediction(record['raw'], True, TAGS)['correct'] is True


@pytest.mark.parametrize('bad', [
    {'kind': 'nonsense', 'value': '1', 'bare_allowed': True},
    {'kind': 'number', 'bare_allowed': True},
    {'kind': 'number', 'value': 'abc', 'bare_allowed': True},
    {'kind': 'clock', 'minute_of_day': 1440},
    {'kind': 'clock', 'minute_of_day': True},
    {'kind': 'number', 'value': '1'},
    {'kind': 'quantity', 'value': '1', 'bare_allowed': False},
    {'kind': 'count', 'value': '1', 'bare_allowed': False, 'unit_aliases': 'tags'},
    {'kind': 'duration', 'value': '1', 'bare_allowed': True, 'bare_unit': 'fortnight'},
])
def test_a_malformed_target_invalidates_rather_than_scoring_zero(bad):
    """A bad target spec is not a wrong answer: it must never be counted as a model failure."""
    with pytest.raises(retention.TargetError):
        retention.score_prediction('1', True, bad)


@pytest.mark.parametrize('prediction', ['1', '1 tags', '', 'not a number', '1:30', '1 hour'])
def test_target_validation_does_not_depend_on_the_prediction(prediction):
    """Regression: validation used to run lazily inside the parse, so a `quantity` target with
    no `unit` raised TargetError only when the prediction carried a suffix, and was scored as
    an ordinary wrong answer otherwise. A broken specification would then have been recorded
    as a capability loss. Every malformed target must fail on every path."""
    for bad in ({'kind': 'quantity', 'value': '1', 'bare_allowed': False},
                {'kind': 'count', 'value': '1', 'bare_allowed': True, 'unit_aliases': []},
                {'kind': 'number', 'value': '1'},
                {'kind': 'number', 'value': '1', 'bare_allow': True},
                {'kind': 'count', 'value': '1.5', 'bare_allowed': True, 'unit_aliases': ['x']}):
        with pytest.raises(retention.TargetError):
            retention.score_prediction(prediction, True, bad)


def test_a_typo_in_a_target_key_is_loud_rather_than_a_missing_permission():
    """`bare_allow` must not read as `bare_allowed: False` and silently fail a correct answer."""
    with pytest.raises(retention.TargetError, match='unknown target keys'):
        retention.score_prediction('54', True,
                                   {'kind': 'count', 'value': '54', 'bare_allowed': True,
                                    'unit_aliases': ['tags'], 'unit': 'tag'})
    # A clock target carrying a numeric `value` is a mixed-up specification, not a clock.
    with pytest.raises(retention.TargetError, match='not value/bare_allowed'):
        retention.score_prediction('10:25 AM', True,
                                   {'kind': 'clock', 'minute_of_day': 625, 'value': '625'})


def test_bare_permission_is_per_row_and_defaults_to_refusing():
    assert retention.score_prediction('54', True, TAGS)['correct'] is True
    assert retention.score_prediction('54', True, TAGS_STRICT)['correct'] is False
    # Only 3 of the 19 duration prompts fix the unit, so a bare duration is normally ambiguous.
    assert retention.score_prediction('45', True, DUR45)['correct'] is True
    assert retention.score_prediction('45', True, DUR45_STRICT)['correct'] is False


def test_clock_and_duration_are_distinct_types_with_the_same_digits():
    """`1:30` is a time of day and `1 hour 30 minutes` is a length; digits alone cannot decide."""
    assert retention.score_prediction('1:30', True, {'kind': 'clock', 'minute_of_day': 90})['correct']
    ninety = {'kind': 'duration', 'value': '5400', 'bare_allowed': False}
    assert not retention.score_prediction('1:30', True, ninety)['correct']
    assert retention.score_prediction('1 hour 30 minutes', True, ninety)['correct']


def test_compound_durations_reject_a_repeated_unit():
    assert not retention.score_prediction('1 hour 1 hour', True,
                                          {'kind': 'duration', 'value': '7200',
                                           'bare_allowed': False})['correct']


def test_exact_rationals_never_touch_a_float():
    assert retention.parse_rational('0.1') + retention.parse_rational('0.2') == Fraction(3, 10)
    assert retention.parse_rational('1,234,567') == 1234567
    for bad in ('5e0', '1/0', '1/-2', '10,00', '1..2', '', '1 2', '--1', '1/2/3', '0x10'):
        with pytest.raises(retention.ParseError):
            retention.parse_rational(bad)


# ---------------------------------------------------------------------------------------
# Target overlay
# ---------------------------------------------------------------------------------------

AUDIT = json.loads(
    (ROOT / 'evidence' / 'retention-design-v1' / 'PROBE_LABEL_AUDIT.json').read_text(
        encoding='utf-8'))


@pytest.mark.parametrize('policy', retention.BARE_POLICIES)
def test_every_audited_row_converts_to_a_valid_target(policy):
    """The audit and the scorer were specified independently and use different vocabularies.
    An argument that they agree is worth less than converting all 150 rows."""
    overlay = retention.build_overlay(AUDIT, policy)
    assert overlay['row_count'] == 150 == overlay['audit_row_count']
    counts = collections.Counter(t['kind'] for t in overlay['targets'].values())
    assert counts == {'clock': 31, 'count': 30, 'duration': 19, 'quantity': 16,
                      'money': 14, 'number': 14, 'volume_cups': 12, 'percent': 10,
                      'distance_km': 4}
    assert sum(counts.values()) == 150
    for target in overlay['targets'].values():
        retention.validate_target(target)


def test_overlay_carries_the_pending_review_state_rather_than_hiding_it():
    overlay = retention.build_overlay(AUDIT, 'permissive')
    assert overlay['label_human_review_status'] == 'pending'
    # Four references name a different counted object than their own question.
    assert overlay['unit_or_entity_label_mismatch_count'] == 4


def test_the_four_contradictory_references_use_the_prompt_entity_not_the_label():
    """`54 spare fuses` is the stored reference for a question that asks about sensor tags.
    Under number-sequence matching both scored alike; the typed grammar must follow the
    prompt."""
    overlay = retention.build_overlay(AUDIT, 'permissive')
    for row_id in AUDIT['mismatch_ids']:
        target = overlay['targets'][row_id]
        assert target['kind'] == 'count'
        stored = next(r['original_answer'] for r in AUDIT['rows'] if r['id'] == row_id)
        stored_noun = stored.split(None, 1)[1].rstrip('.').casefold()
        assert stored_noun not in target['unit_aliases'], (row_id, stored_noun)
    tags = overlay['targets'][AUDIT['mismatch_ids'][0]]
    assert retention.score_prediction('54 sensor tags.', True, tags)['correct'] is True
    assert retention.score_prediction('54 spare fuses.', True, tags)['correct'] is False


def test_bare_duration_permission_is_limited_to_the_three_contract_rows():
    overlay = retention.build_overlay(AUDIT, 'permissive')
    durations = {i: t for i, t in overlay['targets'].items() if t['kind'] == 'duration'}
    bare = [i for i, t in durations.items() if t['bare_allowed']]
    assert len(durations) == 19 and len(bare) == 3
    assert all(i.endswith(retention.DURATION_BARE_ROW_SUFFIXES) for i in bare)
    # The permissive policy must not reach the 16 ambiguous duration rows.
    strict = retention.build_overlay(AUDIT, 'strict')
    assert ({i for i, t in strict['targets'].items()
             if t['kind'] == 'duration' and t['bare_allowed']} == set(bare))


def test_bare_policy_is_the_only_axis_that_differs_between_the_two_overlays():
    """The pending review decision must be isolated: everything else has to be identical, or
    a sensitivity measurement across the two policies would not be measuring one thing."""
    permissive = retention.build_overlay(AUDIT, 'permissive')['targets']
    strict = retention.build_overlay(AUDIT, 'strict')['targets']
    assert permissive.keys() == strict.keys()
    for row_id in permissive:
        a, b = dict(permissive[row_id]), dict(strict[row_id])
        assert a.pop('bare_allowed', None) is not None or a['kind'] == 'clock'
        b.pop('bare_allowed', None)
        a.pop('bare_allowed', None)
        assert a == b, row_id


def test_an_unmapped_audit_type_or_unknown_entity_fails_closed():
    row = {'id': 'x', 'proposed_typed_target': {'type': 'temperature',
                                                'value': {'rational': '1'}, 'unit': 'c'}}
    with pytest.raises(retention.TargetError, match='unmapped audit target type'):
        retention.target_from_audit(row, 'permissive')
    row = {'id': 'x', 'proposed_typed_target': {'type': 'count', 'value': {'rational': '1'},
                                                'unit': 'widgets', 'qualifier': None}}
    with pytest.raises(retention.TargetError, match='no frozen alias list'):
        retention.target_from_audit(row, 'permissive')
    with pytest.raises(retention.TargetError, match='bare_policy'):
        retention.target_from_audit(AUDIT['rows'][0], 'whatever')


# ---------------------------------------------------------------------------------------
# Retention arithmetic
# ---------------------------------------------------------------------------------------

def test_binomial_bound_matches_the_closed_form_at_zero_losses():
    """With no observed losses the exact bound is 1 - alpha**(1/n). The solver is checked
    against that identity rather than trusted, and the registered design value is asserted."""
    for n in (10, 50, 137, 300):
        for alpha in (0.05, 0.05 / 9, 0.01):
            assert math.isclose(retention.exact_binomial_upper_bound(0, n, alpha),
                                1 - alpha ** (1 / n), rel_tol=1e-9)
    # The value quoted in RETENTION_GATE_DESIGN.md for m=50 at alpha=0.05/9.
    assert round(retention.exact_binomial_upper_bound(0, 50, 0.05 / 9), 6) == 0.098648


def test_binomial_bound_is_monotone_and_saturates():
    assert retention.exact_binomial_upper_bound(50, 50, 0.05) == 1.0
    bounds = [retention.exact_binomial_upper_bound(k, 100, 0.05) for k in range(0, 11)]
    assert bounds == sorted(bounds)
    # More evidence at the same loss rate must tighten the bound, never loosen it.
    assert (retention.exact_binomial_upper_bound(2, 200, 0.05)
            < retention.exact_binomial_upper_bound(1, 100, 0.05))
    for args in ((0, 0, 0.05), (-1, 10, 0.05), (11, 10, 0.05), (0, 10, 0.0), (0, 10, 1.0)):
        with pytest.raises(ValueError):
            retention.exact_binomial_upper_bound(*args)


def test_a_gain_never_offsets_a_lost_source_success():
    """The failure mode this gate exists to catch: aggregate accuracy rises while a capability
    the source had is destroyed."""
    counts = retention.monitor_counts(['a', 'b', 'c'], ['a', 'b', 'x', 'y', 'z'])
    assert counts == {'reference_n': 3, 'retained': 2, 'lost': 1, 'gained': 3, 'lost_ids': ['c']}


CLASSES = ('percentage_reasoning', 'ratio_proportion', 'time_duration')
POLICY = json.loads(
    (ROOT / 'retention_policy.proposed.json').read_text(encoding='utf-8'))


def _cls(rows=50, lost=0, terminated=50, source_ok=20):
    return {'rows': rows, 'lost': lost, 'terminated': terminated,
            'source_correct_terminated': source_ok, 'reference_n': source_ok}


def test_monitor_passes_only_with_zero_losses_in_every_class():
    monitor = POLICY['monitor']
    clean = {c: _cls() for c in POLICY['probe_classes']}
    assert retention.monitor_verdict(clean, monitor, CLASSES)['status'] == 'PASS'
    assert retention.monitor_verdict(clean, monitor, CLASSES)['permits_next_block'] is True
    one_loss = dict(clean, time_duration=_cls(lost=1))
    verdict = retention.monitor_verdict(one_loss, monitor, CLASSES)
    assert verdict['status'] == 'FAIL'
    assert verdict['failed_classes'] == ['time_duration']
    assert verdict['permits_next_block'] is False


def test_a_class_collapse_is_not_hidden_by_the_other_classes():
    """Two healthy classes must not average away a third that lost everything."""
    counts = {c: _cls() for c in POLICY['probe_classes']}
    counts['ratio_proportion'] = _cls(lost=20)
    verdict = retention.monitor_verdict(counts, POLICY['monitor'], CLASSES)
    assert verdict['status'] == 'FAIL' and verdict['failed_classes'] == ['ratio_proportion']


def test_monitor_reports_insufficient_when_the_source_cannot_do_the_class():
    """The arithmetic_chain lesson: a probe the teacher scores 0 on measures nothing, and must
    never read as a clean PASS."""
    counts = {c: _cls() for c in POLICY['probe_classes']}
    counts['percentage_reasoning'] = _cls(source_ok=0)
    verdict = retention.monitor_verdict(counts, POLICY['monitor'], CLASSES)
    assert verdict['status'] == 'INSUFFICIENT'
    assert verdict['uninformative_classes'] == ['percentage_reasoning']
    assert verdict['permits_next_block'] is False


def test_insufficient_outranks_fail_so_a_broken_probe_never_reads_as_a_model_verdict():
    counts = {c: _cls() for c in POLICY['probe_classes']}
    counts['percentage_reasoning'] = _cls(source_ok=0)
    counts['ratio_proportion'] = _cls(lost=5)
    assert retention.monitor_verdict(counts, POLICY['monitor'], CLASSES)['status'] == 'INSUFFICIENT'


def test_termination_floor_is_applied_per_class():
    counts = {c: _cls() for c in POLICY['probe_classes']}
    counts['time_duration'] = _cls(terminated=39)          # 39/50 = 0.78 < 0.8
    assert retention.monitor_verdict(counts, POLICY['monitor'], CLASSES)['status'] == 'FAIL'
    counts['time_duration'] = _cls(terminated=40)          # exactly 0.8 passes
    assert retention.monitor_verdict(counts, POLICY['monitor'], CLASSES)['status'] == 'PASS'


def _statements(losses=0, m=50):
    return {(seed, cls): {'source_successes': m, 'losses': losses}
            for seed in (17, 29, 43) for cls in POLICY['probe_classes']}


def test_confirmation_requires_every_seed_and_class_to_clear_its_own_bound():
    conf = POLICY['confirmation']
    assert retention.confirmation_verdict(_statements(), conf, _statements())['status'] == 'PASS'
    partial = _statements()
    partial[(29, 'time_duration')] = {'source_successes': 50, 'losses': 9}
    verdict = retention.confirmation_verdict(partial, conf, _statements())
    assert verdict['status'] == 'FAIL'
    assert verdict['failed_statements'] == ['seed29:time_duration']


def test_confirmation_refuses_a_statement_count_it_did_not_budget_alpha_for():
    """The Bonferroni allocation buys coverage for exactly nine statements. Silently scoring a
    different number is a different multiplicity problem, so it is INVALID, not a verdict."""
    conf = POLICY['confirmation']
    extra = _statements()
    extra[(17, 'extra_class')] = {'source_successes': 50, 'losses': 0}
    assert retention.confirmation_verdict(extra, conf, _statements())['status'] == 'INVALID'
    assert conf['simultaneous_statement_count'] == 9


def test_confirmation_is_insufficient_below_the_registered_source_success_floor():
    conf = POLICY['confirmation']
    thin = _statements(m=49)
    verdict = retention.confirmation_verdict(thin, conf, _statements(m=49))
    assert verdict['status'] == 'INSUFFICIENT'
    assert len(verdict['insufficient_statements']) == 9
    # Zero observed losses on too little evidence is not a pass: absence of evidence cannot
    # open a gate.
    assert all(s['losses'] == 0 for s in verdict['statements'].values())


def test_registered_policy_constants_are_the_design_values_and_stay_put():
    """These numbers were fixed in RETENTION_GATE_DESIGN.md at f7bc290 before any candidate was
    scored under this grammar. Changing one requires a new policy version, not an edit."""
    monitor, conf = POLICY['monitor'], POLICY['confirmation']
    assert POLICY['probe_classes'] == ['percentage_reasoning', 'ratio_proportion', 'time_duration']
    assert POLICY['excluded_probe_class'] == 'arithmetic_chain'
    assert monitor['max_correct_to_incorrect_per_class'] == 0
    assert monitor['min_termination_rate_per_class'] == {'numerator': 4, 'denominator': 5}
    assert monitor['gains_offset_losses'] is False
    assert monitor['check_every_applied_target_tokens'] == 5000
    assert conf['rows_per_class'] == 300 and conf['min_source_successes_per_class'] == 50
    assert conf['min_retention'] == {'numerator': 9, 'denominator': 10}
    assert conf['alpha_per_statement'] == {'numerator': 1, 'denominator': 180}
    assert conf['pool_families_across_seeds_as_independent'] is False
    assert POLICY['diagnostic_only'] == ['ce', 'reply_region_ce', 'kl_to_source', 'top1_agreement']
    assert POLICY['authorization']['licenses_registered_training'] is False
    assert POLICY['authorization']['G3_G5_execution_enabled'] is False


# ---------------------------------------------------------------------------------------
# Stage registration (HOTFIX-005)
# ---------------------------------------------------------------------------------------

ARMS = ('B28', 'D40', 'C28', 'C40')


def test_retention_stages_are_registered_for_every_arm_with_their_parents():
    for arm in ARMS:
        init = gates.STAGES[f'G1_RETENTION_{arm}']
        assert init['parents'] == [f'G1_QUALITY_{arm}']
        assert init['artifacts'] == ['retention.json']
        trained = gates.STAGES[f'G2_RETENTION_{arm}']
        # Both parents: the trained weights, and the step-zero reference they must have kept.
        assert trained['parents'] == [f'G2_CACHED_{arm}', f'G1_RETENTION_{arm}']
        for name in ('source_probe_informative', 'no_reference_capability_lost',
                     'generation_terminates_per_class', 'probe_disjoint_from_training',
                     'calibration_overlap_as_declared', 'typed_scorer_recomputed',
                     'source_unchanged', 'probe_unchanged'):
            assert name in init['checks'] and name in trained['checks'], (arm, name)
        # Only the trained stage can drift from a parent measurement, so only it carries these.
        for name in ('source_answers_reproduce_parent', 'reference_matches_parent'):
            assert name in trained['checks'] and name not in init['checks']


def test_quality_is_no_longer_a_leaf_in_the_receipt_dag():
    """G1_QUALITY used to be a leaf: nothing consumed it, so an arm could reach the engineering
    lane without its output ever being compared to the source."""
    consumers = {stage for stage, spec in gates.STAGES.items()
                 for parent in spec['parents'] if parent.startswith('G1_QUALITY')}
    assert consumers == {f'G1_RETENTION_{arm}' for arm in ARMS}


def test_retention_policy_is_registered_hashed_and_reuses_the_g0_termination_floor():
    p = gates.protocol()
    r = p['G1_RETENTION']
    assert r['scored_split'] == 'monitor_probe'
    assert r['probe_classes'] == ['percentage_reasoning', 'ratio_proportion', 'time_duration']
    assert r['excluded_probe_class'] == 'arithmetic_chain'
    # Reused rather than invented, so the floor cannot drift from G0's.
    assert r['min_termination_rate_per_class'] == p['G0']['min_termination_rate']
    assert r['max_correct_to_incorrect_per_class'] == 0
    assert r['min_source_correct_terminated_per_class'] == 1
    assert r['gains_offset_losses'] is False
    assert r['licenses_registered_training'] is False and r['licenses_efficacy_claim'] is False
    assert r['licenses_confirmation_certificate'] is False
    for forbidden in ('sealed_test_access', 'dev_access', 'threshold_tuning_on_student_outcomes',
                      'reference_reset_after_training'):
        assert forbidden in r['forbidden']
    # Thresholds live inside the hashed manifest, so they cannot move without invalidating
    # every receipt, and the amendment that justifies them travels with them.
    assert 'protocol.json' in gates.implementation_manifest()
    assert 'amendments/HOTFIX_005.md' in gates.implementation_manifest()
    assert 'amendments/HOTFIX_005.md' in p['amendments']


def test_registered_hashes_still_match_the_documents_they_bind():
    r = gates.protocol()['G1_RETENTION']
    for key, path in (('grammar_contract_sha256', ROOT / 'TYPED_SCORING_CONTRACT.md'),
                      ('label_audit_sha256',
                       ROOT / 'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json'),
                      ('policy_proposal_sha256', ROOT / 'retention_policy.proposed.json')):
        assert r[key] == hashlib.sha256(path.read_bytes()).hexdigest(), key


def test_probe_panel_is_the_registered_shape_and_is_disjoint_from_training():
    rows = [json.loads(line) for line in
            (ROOT / 'data/monitor_probe.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    assert len(rows) == 150
    assert collections.Counter(r['reply_class'] for r in rows) == {
        'percentage_reasoning': 50, 'ratio_proportion': 50, 'time_duration': 50}
    assert {r['split'] for r in rows} == {'task'}
    assert len({r['id'] for r in rows}) == 150 == len({r['group_id'] for r in rows})

    def load(name, split):
        return [json.loads(l) for l in (ROOT / 'data' / name).read_text(
            encoding='utf-8-sig').splitlines() if l.strip()]

    probe_fam = {r['group_id'] for r in rows}
    probe_ids = {r['id'] for r in rows}
    train = load('train-smoke.jsonl', 'train')
    assert not probe_fam & {r['group_id'] for r in train}
    assert not probe_ids & {r['id'] for r in train}
    pilot = load('pilot.jsonl', 'pilot')
    assert not probe_ids & {r['id'] for r in pilot}
    # The one disclosed exposure: declared in protocol.json and asserted here, so a silent data
    # change fails the suite rather than only the gate.
    calibration = load('calibration.jsonl', 'calibration')
    overlap = probe_fam & {r['group_id'] for r in calibration}
    assert len(overlap) == gates.protocol()['G1_RETENTION']['declared_calibration_family_overlap']
    assert not probe_ids & {r['id'] for r in calibration}


def test_retention_stages_bind_the_probe_file_not_calibration(tmp_path):
    pilot = tmp_path / 'p.jsonl'; pilot.write_text('pilot', encoding='utf-8')
    cal = tmp_path / 'c.jsonl'; cal.write_text('cal', encoding='utf-8')
    probe = tmp_path / 'm.jsonl'; probe.write_text('probe', encoding='utf-8')
    bindings = gates.data_bindings(pilot, cal, None, probe)
    for arm in ARMS:
        assert bindings[f'G1_RETENTION_{arm}'] == bindings[f'G2_RETENTION_{arm}']
        assert bindings[f'G1_RETENTION_{arm}'] != bindings[f'G1_QUALITY_{arm}']
    # Omitting the probe must not invent a binding for a stage that scores on it.
    assert 'G1_RETENTION_B28' not in gates.data_bindings(pilot, cal)


def test_a_reference_must_be_compared_under_the_grammar_that_produced_it():
    """Regression: the bare-policy sensitivity diagnostic scored the strict grammar against a
    reference the PERMISSIVE grammar produced. Those disagree by construction wherever a bare
    value is involved, so it reported 6 spurious losses and a FAIL on a checkpoint whose 150
    answers were byte-identical to the source. Each policy now carries its own reference."""
    permissive = {'a', 'b', 'c', 'd'}          # source got 4 right under the looser grammar
    strict = {'a', 'b'}                        # only 2 of those survive the stricter one
    correct_under_strict = {'a', 'b'}
    # Mixing them invents losses that are purely a grammar difference.
    assert retention.monitor_counts(permissive, correct_under_strict)['lost'] == 2
    # Comparing like with like reports the truth: nothing was forgotten.
    assert retention.monitor_counts(strict, correct_under_strict)['lost'] == 0


def test_probe_loader_rejects_a_row_that_would_silently_measure_nothing():
    from experiments.architecture_benchmark_001 import run
    good = {'split': 'task', 'id': 'a', 'group_id': 'g', 'reply_class': 'time_duration',
            'answer': '45 minutes.', 'messages': [{'role': 'user', 'content': 'q'}]}
    for damage in ({'split': 'dev'}, {'id': ''}, {'group_id': ''}, {'answer': '  '},
                   {'messages': []}, {'reply_class': 'arithmetic_chain'},
                   {'reply_class': 'counting_sets'}):
        path = Path(str(pytest.importorskip('tempfile').mktemp(suffix='.jsonl')))
        path.write_text(json.dumps({**good, **damage}) + '\n', encoding='utf-8')
        with pytest.raises(ValueError):
            run.load_probe_rows(path)
        path.unlink()


# ---------------------------------------------------------------------------------------
# Regressions from the adversarial review (all six confirmed by reproduction)
# ---------------------------------------------------------------------------------------

def test_an_empty_or_partial_class_set_is_never_a_pass():
    """`monitor_verdict({})` returned PASS with permits_next_block True -- a gate primitive
    whose 'no evidence' answer was 'proceed to the next optimizer block'."""
    with pytest.raises(ValueError, match='empty result is not a pass'):
        retention.monitor_verdict({}, POLICY['monitor'], CLASSES)
    with pytest.raises(ValueError, match='do not match the registered classes'):
        retention.monitor_verdict({'percentage_reasoning': _cls()}, POLICY['monitor'], CLASSES)
    with pytest.raises(ValueError, match='do not match the registered classes'):
        retention.monitor_verdict({**{c: _cls() for c in CLASSES}, 'invented': _cls()},
                                  POLICY['monitor'], CLASSES)


@pytest.mark.parametrize('prediction,ok', [
    ('45 minutes and', False), ('45 minutes,', False), ('45 and minutes', False),
    ('45 minutes and and 0 hours', False), ('45 minutes', True),
    ('and 45 minutes', False), ('45 minutes 0 hours', True),
])
def test_duration_separators_are_matched_not_deleted(prediction, ok):
    """Commas and `and` used to be stripped everywhere before tokenising, so dangling and
    interpolated separators scored correct at exactly the right value."""
    target = {'kind': 'duration', 'value': '2700', 'bare_allowed': False}
    assert retention.score_prediction(prediction, True, target)['correct'] is ok


def test_comma_grouped_numbers_survive_inside_a_duration():
    """The same blanket comma deletion made a legal grouped numeral unparseable, and 7200 s is
    a real target in the monitor panel."""
    for text, seconds in (('3,600 seconds', '3600'), ('7,200 seconds', '7200')):
        target = {'kind': 'duration', 'value': seconds, 'bare_allowed': False}
        assert retention.score_prediction(text, True, target)['correct'] is True


def test_confirmation_checks_statement_identity_not_just_count():
    """Nine fabricated seed/class labels returned PASS carrying the registered family-wise
    alpha. Counting nine statements is not checking they are the registered nine."""
    conf = POLICY['confirmation']
    fabricated = {(1, f'x{i}'): {'source_successes': 60, 'losses': 0} for i in range(9)}
    verdict = retention.confirmation_verdict(fabricated, conf, _statements())
    assert verdict['status'] == 'INVALID'
    assert 'identities do not match' in verdict['reason']


def test_the_registered_retention_floor_is_inert_and_that_is_recorded():
    """The 9/10 floor can never bind: for alpha=1/180 the exact upper limit always exceeds the
    point estimate, so bound <= 0.10 already implies retention > 0.9. And at the registered
    minimum m=50 only ZERO losses clear the bound, so the certificate demands 100% there.
    Recorded in protocol.json rather than silently repaired."""
    alpha = 1 / 180
    for m in (50, 70, 126, 300):
        passing = [k for k in range(m + 1)
                   if retention.exact_binomial_upper_bound(k, m, alpha) <= 0.10]
        assert all((m - k) / m > 0.9 for k in passing), m
    assert [k for k in range(51)
            if retention.exact_binomial_upper_bound(k, 50, alpha) <= 0.10] == [0]
    assert retention.exact_binomial_upper_bound(1, 70, alpha) <= 0.10
    assert retention.exact_binomial_upper_bound(1, 69, alpha) > 0.10
    note = gates.protocol()['G1_RETENTION']['confirmation_min_retention_is_inert']
    assert 'can never bind' in note and '100% retention' in note


def test_a_bare_policy_disagreement_can_fail_the_receipt():
    """The amendment calls a permissive/strict disagreement 'a blocking review item, not a
    rounding error'. It was recorded only in metrics, so it could block nothing."""
    for arm in ARMS:
        assert 'bare_policy_verdicts_agree' in gates.STAGES[f'G1_RETENTION_{arm}']['checks']
        assert 'bare_policy_verdicts_agree' in gates.STAGES[f'G2_RETENTION_{arm}']['checks']


def test_retention_module_is_pure_and_enters_the_hashed_manifest():
    """No torch: the scorer must be runnable in a test in milliseconds and identical for the
    source, the initialization and every checkpoint."""
    source = (ROOT / 'retention.py').read_text(encoding='utf-8')
    assert 'import torch' not in source and 'from torch' not in source
    assert 'retention.py' in gates.implementation_manifest()
