"""Complete-case fixed-dose screen selection, never a confirmatory claim."""
from __future__ import annotations

from collections import Counter
import math
from statistics import mean

from shared.scoring import score_response
from shared.statistics import paired_analysis
from shared.reporting import quality_cost_frontier
from armr.metrics import output_flags

from .protocol import ARMS, GATES, PRESERVATION, cell_id, validate_config

FAMILIES = ('compositional', 'counterfactual', 'instruction_evidence', 'mathematical',
            'ordinary_qa_preservation', 'relational', 'retrieval_multihop', 'symbolic_logic')
IDENTITY = ('scenario_id', 'family', 'track', 'split', 'variant', 'template_id',
            'template_unseen_in_engineering')


def _number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'finite numeric {name} required')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} outside permitted range')
    return value


def _cases(cases):
    if not isinstance(cases, list) or len(cases) != 128:
        raise ValueError('exactly 128 frozen architecture originals required')
    ids = [case.get('example_id') for case in cases]
    scenarios = [case.get('scenario_id') for case in cases]
    if (any(not isinstance(item, str) or not item for item in ids + scenarios)
            or len(set(ids)) != 128 or len(set(scenarios)) != 128):
        raise ValueError('128 unique nonempty example/scenario IDs required')
    if Counter(case.get('family') for case in cases) != Counter({family: 16 for family in FAMILIES}):
        raise ValueError('frozen eight-family architecture strata required')
    for case in cases:
        if (case.get('track') != 'cross_scale' or case.get('split') != 'architecture_dev'
                or case.get('variant') != 'original' or not case.get('template_id')
                or case.get('template_unseen_in_engineering') is not False):
            raise ValueError('only Track B engineering originals may enter screen selection')
    return {case['example_id']: case for case in cases}


def _complete_rows(rows, cases):
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError('partial output may not support cell accuracy')
    by_id = {}
    for row in rows:
        identity = row.get('example_id')
        if identity not in cases or identity in by_id:
            raise ValueError('unexpected or duplicate output example ID')
        case = cases[identity]
        if any(row.get(key) != case.get(key) for key in IDENTITY):
            raise ValueError('output does not match the frozen case identity')
        if (row.get('safety_stopped') is not False or row.get('termination') not in ('eos', 'token_cap')
                or not isinstance(row.get('response'), str)
                or not isinstance(row.get('token_ids'), list)
                or not row['token_ids']
                or any(type(token) is not int or token < 0 for token in row['token_ids'])):
            raise ValueError('complete uncorrupted raw generation evidence required')
        flags = output_flags(row['response'], row['token_ids'])
        if any(flags[key] for key in ('repetition_loop', 'script_corruption', 'control_count', 'replacement_count')):
            raise ValueError('hard-failed generation cannot be counted as a complete cell')
        if type(row.get('invalid')) is not bool or row['invalid'] != flags['invalid']:
            raise ValueError('raw output invalidity differs from saved flag')
        scored = score_response(row['response'], case['scorer'])
        if (type(row.get('score')) not in (float, int) or row['score'] not in (0., 1.)
                or row['score'] != scored['score']):
            raise ValueError('saved correctness differs from the unchanged strict scorer')
        _number(row.get('gold_nll'), 'gold_nll', minimum=0)
        _number(row.get('logit_cosine'), 'logit_cosine', minimum=-1.000001, maximum=1.000001)
        _number(row.get('kl_base_to_arm'), 'kl_base_to_arm', minimum=0)
        _number(row.get('wall_seconds'), 'wall_seconds', minimum=0)
        _number(row.get('generation_dense_matmul_flops_estimate'), 'generation FLOPs', minimum=0)
        _number(row.get('peak_rss_bytes'), 'peak RSS', minimum=0)
        lower = _number(row.get('trajectory_min_rms_ratio'), 'minimum hidden RMS ratio', minimum=.25, maximum=4.)
        upper = _number(row.get('trajectory_max_rms_ratio'), 'maximum hidden RMS ratio', minimum=.25, maximum=4.)
        if lower > upper or not isinstance(row.get('hidden'), list):
            raise ValueError('valid hidden trajectory records required')
        by_id[identity] = row
    if set(by_id) != set(cases):
        raise ValueError('complete identical cell coverage required')
    return [by_id[identity] for identity in sorted(cases)]


def _diagnostics(rows):
    rms = [record['rms'] for row in rows for record in row['hidden'] if 'rms' in record]
    for value in rms:
        _number(value, 'captured hidden RMS', minimum=0)
    cuda = [row.get('peak_cuda_allocated_bytes', 0) for row in rows]
    for value in cuda:
        _number(value, 'peak CUDA allocation', minimum=0)
    inventory = {}
    for key in ('added_parameter_count', 'added_active_parameters', 'extra_matmul_flops_per_token'):
        values = {row.get(key) for row in rows}
        if len(values) != 1:
            raise ValueError('operator inventory changed within a complete cell')
        value = next(iter(values))
        inventory[key] = None if value is None else _number(value, key, minimum=0)
    return {
        'examples': len(rows), 'mean_first_token_cosine': mean(row['logit_cosine'] for row in rows),
        'mean_kl_X0_to_cell': mean(row['kl_base_to_arm'] for row in rows),
        'mean_gold_nll': mean(row['gold_nll'] for row in rows),
        'trajectory_hidden_rms_ratio_min': min(row['trajectory_min_rms_ratio'] for row in rows),
        'trajectory_hidden_rms_ratio_max': max(row['trajectory_max_rms_ratio'] for row in rows),
        'captured_hidden_rms': {'records': len(rms), 'mean': mean(rms) if rms else None,
                               'minimum': min(rms) if rms else None, 'maximum': max(rms) if rms else None},
        'mean_wall_seconds': mean(row['wall_seconds'] for row in rows),
        'mean_generation_dense_matmul_flops_estimate': mean(row['generation_dense_matmul_flops_estimate'] for row in rows),
        'peak_rss_bytes': max(row['peak_rss_bytes'] for row in rows),
        'peak_cuda_allocated_bytes': max(cuda),
        'operator_inventory': inventory,
        'invalid_outputs': sum(row['invalid'] for row in rows), 'safety_stops': 0,
        'token_cap_terminations': sum(row['termination'] == 'token_cap' for row in rows),
    }


def analyze_screen(cases, by_cell, failed_cells, config):
    """Analyze all 17 terminal cells and select at most one preregistered dose.

    `by_cell` contains raw rows. A failed cell may retain partial rows there, but
    these are never scored or assigned accuracy. `failed_cells` must contain a
    terminal FAILED receipt with phase qualification or generation. Other fixed
    cells, including siblings of a failed dose, must still reach their own result.
    Receipt/source/registration/tensor integrity is verified by the lifecycle
    caller before this pure evidence analysis; raw responses are rescored here.
    """
    validate_config(config, verify_artifacts=False)
    expected = _cases(cases)
    if not isinstance(by_cell, dict) or not isinstance(failed_cells, dict):
        raise ValueError('cell outputs and terminal failures must be mappings')
    cells = config['cells']
    if set(by_cell) | set(failed_cells) != set(cells):
        raise ValueError('all fixed cells must be complete or terminal failed; unknown/unfinished cells remain')
    for name, receipt in failed_cells.items():
        if (not isinstance(receipt, dict) or receipt.get('status') != 'FAILED'
                or receipt.get('phase') not in ('qualification', 'generation')):
            raise ValueError('explicit terminal failure receipt required')
    if 'X0' in failed_cells or 'X0' not in by_cell:
        raise ValueError('complete 128-case X0 baseline is mandatory before screen analysis')
    complete = {name: _complete_rows(rows, expected) for name, rows in by_cell.items() if name not in failed_cells}
    base = complete['X0']
    policy = config['screen_policy']
    options = {'seed': policy['bootstrap_seed'], 'bootstrap_samples': policy['bootstrap_replicates'],
               'expected_example_ids': sorted(expected), 'analysis_role': 'candidate_selection'}
    summaries = {}
    costs = {}
    for name in cells:
        if name in failed_cells:
            summaries[name] = {'status': 'FAILED', 'failure': failed_cells[name],
                'retained_partial_examples': len(by_cell.get(name, [])),
                'accuracy': None, 'paired_statistics': None,
                'interpretation': 'Failure evidence only; no partial-cell accuracy or missing-control substitution.'}
            continue
        rows = complete[name]
        paired = paired_analysis(base, rows, **options)
        diagnostic = _diagnostics(rows)
        summaries[name] = {'status': 'COMPLETE', 'accuracy': paired['overall']['candidate_accuracy'],
            'paired_statistics': paired, 'diagnostics': diagnostic}
        # Real caller records wrapper inventory in every row; do not invent a
        # donor parameter count when analyzing fixtures or incomplete metadata.
        params = {row.get('added_parameter_count') for row in rows}
        if len(params) == 1 and next(iter(params)) is not None:
            count = _number(next(iter(params)), 'added parameter count', minimum=0)
            costs[name] = {'accuracy': summaries[name]['accuracy'],
                'mean_wall_seconds': diagnostic['mean_wall_seconds'],
                'flops': diagnostic['mean_generation_dense_matmul_flops_estimate'],
                'added_parameters': count, 'peak_memory_bytes': diagnostic['peak_rss_bytes'],
                'peak_cuda_allocated_bytes': diagnostic['peak_cuda_allocated_bytes']}
    preservation_ids = {identity for identity, case in expected.items() if case['family'] in PRESERVATION}
    base_preservation_errors = sum(1 - row['score'] for row in base if row['example_id'] in preservation_ids)
    baseline_invalid = sum(row['invalid'] for row in base)
    doses = []
    for gate in GATES:
        names = [cell_id(arm, gate) for arm in ARMS]
        failed = [name for name in names if name in failed_cells]
        row = {'gate': gate, 'cells': names, 'eligible': False, 'rejection_reasons': [],
               'failed_cells': failed, 'comparisons_with_controls': {}}
        if failed:
            row['rejection_reasons'].append('same_dose_cell_failed')
            if any(failed_cells[name]['phase'] == 'qualification' for name in failed):
                row['rejection_reasons'].append('GPU_qualification_rejected_entire_dose')
            doses.append(row)
            continue
        invalid = {name: sum(item['invalid'] for item in complete[name]) - baseline_invalid for name in names}
        if any(count > policy['maximum_additional_invalid_outputs'] for count in invalid.values()):
            row['rejection_reasons'].append('additional_invalid_output_in_same_dose_cell')
        graft_name = cell_id('XG', gate)
        graft = summaries[graft_name]['paired_statistics']['overall']
        preservation_errors = sum(1 - item['score'] for item in complete[graft_name] if item['example_id'] in preservation_ids)
        additional_errors = preservation_errors - base_preservation_errors
        if graft['paired_wins'] < policy['minimum_paired_wins_vs_X0']:
            row['rejection_reasons'].append('fewer_than_two_paired_wins')
        net = graft['paired_wins'] - graft['paired_losses']
        if net < policy['minimum_net_correct_vs_X0']:
            row['rejection_reasons'].append('less_than_one_net_correct_gain')
        if additional_errors > policy['maximum_additional_preservation_errors']:
            row['rejection_reasons'].append('preservation_error_budget_exceeded')
        for arm in ('XS', 'XR', 'XM'):
            name = cell_id(arm, gate)
            row['comparisons_with_controls'][arm] = paired_analysis(complete[name], complete[graft_name], **options)
        best_control = max(summaries[cell_id(arm, gate)]['accuracy'] for arm in ('XS', 'XR', 'XM'))
        margin = summaries[graft_name]['accuracy'] - best_control
        if margin < 0:
            row['rejection_reasons'].append('aligned_graft_below_same_dose_control')
        row.update({'additional_invalid_by_cell': invalid, 'XG_paired_wins': graft['paired_wins'],
            'XG_paired_losses': graft['paired_losses'], 'XG_net_correct_gain': net,
            'preservation': {'examples': len(preservation_ids), 'X0_errors': base_preservation_errors,
                'XG_errors': preservation_errors, 'additional_XG_errors': additional_errors},
            'XG_minus_best_control': margin, 'XG_minus_X0': graft['accuracy_difference'],
            'eligible': not row['rejection_reasons']})
        doses.append(row)
    eligible = sorted((row for row in doses if row['eligible']),
        key=lambda row: (-row['XG_minus_best_control'], -row['XG_minus_X0'], row['gate']))
    selected = None if not eligible else {'arm': 'XG', 'gate': eligible[0]['gate'],
        'cell_id': cell_id('XG', eligible[0]['gate']), 'selected_pair': config['selected_pair'],
        'objective': 'B3', 'rank_margin_to_best_control': eligible[0]['XG_minus_best_control'],
        'rank_gain_to_X0': eligible[0]['XG_minus_X0']}
    return {'schema_version': 'cross_scale_generation_screen_analysis_v1',
        'status': 'SCREEN_GO' if selected else 'SCREEN_NO_GO',
        'analysis_role': 'candidate_selection', 'examples': 128, 'independent_scenarios': 128,
        'selected': selected, 'ranked_eligible_gates': [row['gate'] for row in eligible],
        'cells': summaries, 'doses': doses,
        'quality_cost': quality_cost_frontier(costs) if len(costs) == len(complete) else {
            'status': 'UNAVAILABLE', 'reason': 'Every complete cell needs a consistent recorded wrapper parameter inventory.'},
        'native_comparator': config['native_comparator'],
        'confirmatory_generation_licensed': False, 'efficacy_claim_licensed': False,
        'next_stage': 'A SCREEN_GO requires a separate committed confirmation freeze; the 1024-case bank remains closed.',
        'limitations': ['128 architecture originals select a dose, not establish generalizable efficacy.',
            'Shared paired confidence intervals and exact McNemar tests are descriptive after this fixed multi-dose screen.',
            'No arbitrary output-identity or KL threshold was selected; required positive decision changes reject inert behavior.',
            'Failed cells have no accuracy estimate; every same-dose control is mandatory.',
            'No native efficacy row or native candidate-selection decision enters this analysis.']}
