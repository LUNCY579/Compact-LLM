"""Fail-closed literal JSON scoring and grouped synthetic DEV descriptions.

The historical strict and normalized string measures are preserved. Typed
scoring adds a declared answer-contract measure; it is not a semantic judge.
"""
import json
import math
from collections import defaultdict

from .utils import Blocked, require

METRICS = ('strict_exact_match', 'normalized_exact_match',
           'typed_value_correct', 'json_contract_compliance', 'json_valid')
RETENTION_METRICS = ('strict_exact_match', 'typed_value_correct')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant')


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def parse_json(text):
    """Return (valid, value); no fenced-JSON, bare-text, or duplicate salvage."""
    if not isinstance(text, str):
        return False, None
    try:
        value = json.loads(text, object_pairs_hook=_object,
                           parse_constant=_reject_constant)
        return (_finite(value), value if _finite(value) else None)
    except (ValueError, TypeError, RecursionError):
        return False, None


def _same_contract(value, target):
    if type(value) is not type(target):
        return False
    if isinstance(target, dict):
        return set(value) == set(target) and all(
            _same_contract(value[key], target[key]) for key in target)
    if isinstance(target, list):
        return len(value) == len(target) and all(
            _same_contract(a, b) for a, b in zip(value, target))
    return True


def _same_value(value, target):
    return _same_contract(value, target) and value == target


def score_prediction(prediction, target):
    require(isinstance(prediction, str) and isinstance(target, str),
            'prediction and target must be raw strings')
    target_valid, expected = parse_json(target)
    require(target_valid and isinstance(expected, dict) and bool(expected),
            'target must be nonempty, finite, duplicate-free JSON object')
    valid, value = parse_json(prediction)
    contract = valid and _same_contract(value, expected)
    typed = contract and _same_value(value, expected)
    # Diagnostics never turn a bare value or a wrong type into typed correctness.
    bare = False
    if len(expected) == 1:
        only = next(iter(expected.values()))
        bare = ((valid and not isinstance(value, dict) and _same_value(value, only))
                or (isinstance(only, str) and prediction.strip() == only))
    return {
        'strict_exact_match': prediction == target,
        'normalized_exact_match': prediction.strip().casefold() == target.strip().casefold(),
        'typed_value_correct': bool(typed),
        'json_contract_compliance': bool(contract),
        'json_valid': bool(valid),
        'formatting_only_mismatch': bool(typed and prediction != target),
        'bare_expected_value_diagnostic': bool(bare),
    }


def _metadata(metadata):
    return metadata['dev'] if 'dev' in metadata else metadata


def scored_rows(predictions, devrows, metadata):
    """Validate pair identity and supplied flags before recomputing raw scores."""
    require(isinstance(predictions, list) and isinstance(devrows, list),
            'predictions and DEV rows must be lists')
    require(len(predictions) == len(devrows) and bool(devrows),
            'prediction count must match complete nonempty DEV panel')
    meta = _metadata(metadata)
    require(isinstance(meta, dict) and set(meta) == {row['id'] for row in devrows},
            'DEV metadata coverage must match rows')
    seen, result = set(), []
    for prediction, row in zip(predictions, devrows):
        require(isinstance(prediction, dict), 'prediction rows must be objects')
        require(row['id'] not in seen, 'duplicate DEV row IDs')
        seen.add(row['id'])
        for field in ('id', 'family_id', 'target'):
            require(prediction.get(field) == row[field], f'prediction {field} mismatch')
        require(prediction.get('reset_recurrent_at') is None,
                'state-reset diagnostic cannot enter primary comparison')
        flags = score_prediction(prediction.get('prediction'), row['target'])
        for metric, expected in flags.items():
            if metric in prediction:
                require(type(prediction[metric]) is bool,
                        f'{metric} flags must be booleans')
                require(prediction[metric] == expected,
                        f'{metric} flag contradicts raw prediction')
        info = meta[row['id']]
        scored = dict(prediction, **flags, metadata=info)
        if info.get('category') == 'update':
            valid, value = parse_json(prediction['prediction'])
            valid_target, target = parse_json(row['target'])
            require(valid_target and len(target) == 1,
                    'update answer contract must have one field')
            key = next(iter(target))
            first = info.get('first_record_value')
            scored['first_record_copy_error'] = bool(
                flags['json_contract_compliance'] and valid and key in value
                and _same_value(value[key], first) and not flags['typed_value_correct'])
        result.append(scored)
    return result


def _counts(rows):
    return {'rows': len(rows), **{
        metric: {'correct': sum(row[metric] for row in rows),
                 'total': len(rows),
                 'accuracy': sum(row[metric] for row in rows) / len(rows) if rows else None}
        for metric in METRICS}}


def _retention_base(rows):
    retained = [row for row in rows if row['metadata'].get('role') == 'retention']
    return {metric: {
        'source_correct': sum(row[metric] for row in retained),
        'source_correct_skill_count': len({row['metadata']['skill'] for row in retained if row[metric]}),
        'source_correct_by_skill': {skill: sum(row[metric] for row in retained
            if row['metadata']['skill'] == skill) for skill in sorted({row['metadata']['skill'] for row in retained})},
        'rows': len(retained),
    } for metric in RETENTION_METRICS}


def summarize_rows(predictions, devrows, metadata):
    rows = scored_rows(predictions, devrows, metadata)
    updates = [row for row in rows if row['metadata'].get('category') == 'update']
    missing = [row for row in rows if row['metadata'].get('category') == 'missing']
    strata = {}
    for position in range(3):
        for role in ('distant_heldout', 'near_control'):
            key = f'{position}:{role}'
            strata[key] = _counts([row for row in updates
                if row['metadata'].get('latest_position') == position
                and row['metadata'].get('role') == role])
    groups = defaultdict(list)
    worlds = defaultdict(list)
    for row in updates:
        info = row['metadata']
        key = (info['world_id'], info['counterfactual_id'], info['role'])
        groups[key].append(row)
        worlds[info['world_id']].append(row)
    group_results = []
    for key in sorted(groups):
        group = groups[key]
        require(len(group) == 6 and {row['metadata']['permutation'] for row in group} == set(range(6)),
                'update groups must contain all six distinct permutations')
        group_results.append({'world_id': key[0], 'counterfactual_id': key[1], 'role': key[2],
            'typed_all_six_correct': all(row['typed_value_correct'] for row in group),
            'strict_all_six_correct': all(row['strict_exact_match'] for row in group)})
    return {
        'overall': _counts(rows),
        'updates': _counts(updates),
        'update_position_and_distance': strata,
        'first_record_copy_errors': sum(row.get('first_record_copy_error', False) for row in updates),
        'six_order_groups': group_results,
        'typed_all_six_correct_groups': sum(group['typed_all_six_correct'] for group in group_results),
        'update_world_count': len(worlds),
        'per_update_world': {world: _counts(group) for world, group in sorted(worlds.items())},
        'missing_evidence': {
            'present_fact': _counts([row for row in missing if row['metadata'].get('answerable') is True]),
            'correct_abstention': _counts([row for row in missing if row['metadata'].get('answerable') is False]),
        },
        'retention_instrument': _retention_base(rows),
        'formatting_only_mismatches': sum(row['formatting_only_mismatch'] for row in rows),
        'bare_expected_value_diagnostics': sum(row['bare_expected_value_diagnostic'] for row in rows),
        'limitations': ['synthetic DEV', 'one model seed',
            'counterfactual, position, and permutation rows grouped by dependent world',
            'literal JSON contracts are not broad semantic capability'],
    }


def anchor_retention_gate(anchor_predictions, devrows, metadata):
    base = _retention_base(scored_rows(anchor_predictions, devrows, metadata))
    checks = {metric: value['source_correct'] >= 16 and value['source_correct_skill_count'] >= 4
              for metric, value in base.items()}
    return {'adequate': all(checks.values()), 'checks': checks,
            'per_metric': base, 'minimum_correct_per_metric': 16,
            'minimum_skills_per_metric': 4,
            'decision': 'ADEQUATE_SYNTHETIC_INSTRUMENT' if all(checks.values()) else 'HOLD_INADEQUATE_RETENTION_INSTRUMENT'}


def retention_changes(anchor_rows, condition_rows):
    require(len(anchor_rows) == len(condition_rows), 'retention pairing mismatch')
    paired = [(a, b) for a, b in zip(anchor_rows, condition_rows)
              if a['metadata'].get('role') == 'retention']
    result = {}
    for metric in RETENTION_METRICS:
        baseline = sum(a[metric] for a, _ in paired)
        damaged = [a['id'] for a, b in paired if a[metric] and not b[metric]]
        repaired = [a['id'] for a, b in paired if not a[metric] and b[metric]]
        result[metric] = {'baseline_correct': baseline,
            'retained': baseline - len(damaged), 'damages': len(damaged),
            'repairs': len(repaired), 'damaged_ids': damaged, 'repaired_ids': repaired,
            'retention_rate': (baseline-len(damaged))/baseline if baseline else None}
    # Partition strict repairs: valid existing values with wrapper/spacing fixes
    # remain distinct from newly correct literal values.
    result['strict_repairs_by_value'] = {
        'formatting_only': [a['id'] for a, b in paired
            if not a['strict_exact_match'] and b['strict_exact_match'] and a['typed_value_correct']],
        'literal_value_or_contract': [a['id'] for a, b in paired
            if not a['strict_exact_match'] and b['strict_exact_match']
            and not a['typed_value_correct'] and not a['bare_expected_value_diagnostic']],
        'bare_value_wrapper_diagnostic': [a['id'] for a, b in paired
            if not a['strict_exact_match'] and b['strict_exact_match'] and a['bare_expected_value_diagnostic']],
    }
    result['typed_repairs_by_value'] = {
        'bare_value_wrapper_diagnostic': [a['id'] for a, b in paired
            if not a['typed_value_correct'] and b['typed_value_correct'] and a['bare_expected_value_diagnostic']],
        'literal_value_or_contract': [a['id'] for a, b in paired
            if not a['typed_value_correct'] and b['typed_value_correct'] and not a['bare_expected_value_diagnostic']],
    }
    return result
