"""Independent CPU audit of a completed CS-PILOT baseline and any completed arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .audit_lora596 import Inputs, fingerprint, unique
from .metrics import Outcome, compare, loss_upper_bound


def exact_prediction(text, target, family, terminated):
    if type(terminated) is not bool:
        raise ValueError('termination must be boolean')
    if not terminated:
        return False
    if family in {'ledger_replay', 'prerequisite_plan', 'register_machine'}:
        try:
            return json.dumps(json.loads(text), sort_keys=True, separators=(',', ':'), allow_nan=False) == target
        except (TypeError, ValueError):
            return False
    return text.strip() == target


def whole_value(text, family):
    """Post-hoc format sensitivity only: never extract a number from prose."""
    value = text.strip()
    if family == 'time':
        match = re.fullmatch(r'([01]?[0-9]|2[0-3]):([0-5][0-9])\.?', value)
        return 60 * int(match[1]) + int(match[2]) if match else None
    if family == 'arithmetic':
        return int(value.removesuffix('.')) if re.fullmatch(r'[+-]?[0-9]+\.?', value) else None
    raise ValueError('unsupported whole-answer type')


def audit_run(run: Path, coverage_path: Path | None = None):
    inputs = Inputs()
    plan = inputs.read(run / 'PLAN.json')
    banks = {'old_dev': unique(inputs.read(run / 'dev.json')),
             'acquisition': unique(inputs.read(Path(plan['acquisition']), jsonl=True))}
    for name, expected in plan['local_data'].items():
        path = run / name
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'local data changed: {name}')
    if inputs.receipts[str(Path(plan['acquisition']).resolve())] != plan['bindings'][plan['acquisition']]:
        raise ValueError('acquisition bank changed')

    def completed(stage):
        complete_path = run / stage / 'COMPLETE.json'
        if not complete_path.exists():
            return False
        complete = inputs.read(complete_path)
        supervisor = inputs.read(run / stage / 'supervisor.json')
        if complete['stage'] != stage or supervisor['status'] != 'COMPLETE' or supervisor['child_returncode'] != 0:
            raise ValueError(f'unsuccessful supervised stage: {stage}')
        for name, expected in complete['artifacts'].items():
            path = run / stage / name
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'completed artifact changed: {stage}/{name}')
        return True

    if not completed('baseline'):
        raise ValueError('baseline is incomplete')
    arm_results = {}
    outcomes = {}
    for arm, stage in [('base', 'baseline'), ('control', 'eval-control'), ('anchor', 'eval-anchor')]:
        if not completed(stage):
            continue
        if arm != 'base' and not completed(f'train-{arm}'):
            raise ValueError(f'training is incomplete: {arm}')
        outcomes[arm] = {}
        arm_results[arm] = {}
        for endpoint, rows in banks.items():
            predictions = unique(inputs.read(run / stage / f'{endpoint}.jsonl', jsonl=True))
            if predictions.keys() != rows.keys():
                raise ValueError(f'{arm}/{endpoint}: incomplete or mismatched rows')
            scored = []
            for key, row in rows.items():
                prediction = predictions[key]
                if prediction['reference'] != row['reference'] or prediction['family'] != row['family']:
                    raise ValueError(f'prediction reference/family drift: {key}')
                correct = exact_prediction(prediction['prediction'], row['reference'], row['family'], prediction['terminated'])
                if type(prediction['correct']) is not bool or correct != prediction['correct']:
                    raise ValueError(f'correctness flag does not reproduce: {key}')
                group = row.get('world_id', row.get('parent_id'))
                if prediction['world_id'] != group:
                    raise ValueError(f'prediction world drift: {key}')
                scored.append(Outcome(key, group, fingerprint([row['prompt'], row['reference'], row['family'], 'exact-terminated-v1']), correct))
            outcomes[arm][endpoint] = scored
            arm_results[arm][endpoint] = {'correct': sum(r.correct for r in scored), 'rows': len(scored)}
            if endpoint == 'acquisition':
                per_family = {}
                for family, floor in [('ledger_replay', 30/128), ('prerequisite_plan', 10/128), ('register_machine', 9/128)]:
                    subset = [r for r in scored if rows[r.id]['family'] == family]
                    n, hits = len(subset), sum(r.correct for r in subset)
                    if n != 128:
                        raise ValueError('wrong acquisition family count')
                    lower = 1 - loss_upper_bound(n - hits, n, alpha=.025)
                    per_family[family] = {'n': n, 'correct': hits, 'cp95_lower': lower, 'floor': floor, 'clears_floor': lower > floor}
                arm_results[arm][endpoint]['per_family'] = per_family
            else:
                binding = {k: r for k, r in rows.items() if r['family'] == 'binding'}
                grouped = {}
                for r in scored:
                    if r.id in binding:
                        grouped.setdefault(r.group, []).append(r.correct)
                if any(len(v) != 4 for v in grouped.values()):
                    raise ValueError('binding worlds require four paired variants')
                arm_results[arm]['binding_joint_worlds'] = {'correct': sum(all(v) for v in grouped.values()), 'n': len(grouped)}
                arm_results[arm]['abstention'] = {
                    'insufficient': sum(not r['answerable'] for r in binding.values()),
                    'true_unknown': sum(not r['answerable'] and predictions[k]['prediction'].strip() == 'UNKNOWN' and predictions[k]['terminated'] for k, r in binding.items()),
                    'answerable': sum(r['answerable'] for r in binding.values()),
                    'false_unknown': sum(r['answerable'] and predictions[k]['prediction'].strip() == 'UNKNOWN' for k, r in binding.items()),
                }
                arm_results[arm]['whole_answer_format_sensitivity_posthoc'] = {}
                for family in ['time', 'arithmetic']:
                    selected = {k: r for k, r in rows.items() if r['family'] == family}
                    rescued, typed_correct = [], 0
                    for key, row in selected.items():
                        target = whole_value(row['reference'], family)
                        if target is None:
                            raise ValueError('invalid typed diagnostic reference')
                        pred = predictions[key]
                        hit = pred['terminated'] and whole_value(pred['prediction'], family) == target
                        typed_correct += hit
                        if hit and not pred['correct']:
                            rescued.append(key)
                    arm_results[arm]['whole_answer_format_sensitivity_posthoc'][family] = {
                        'rows': len(selected), 'typed_correct': typed_correct, 'rescued_ids': rescued,
                        'changes_primary_endpoint_or_anchor_membership': False,
                        'grammar': 'whole 24h H:MM/HH:MM or signed integer; optional final period; no prose or numeric extraction'}

    contrasts = {}
    for left, right in [('base', 'control'), ('base', 'anchor'), ('control', 'anchor')]:
        if left not in outcomes or right not in outcomes:
            continue
        contrast = {}
        for endpoint, rows in banks.items():
            old, new = outcomes[left][endpoint], outcomes[right][endpoint]
            contrast[endpoint] = compare(old, new)
            contrast[endpoint]['per_family'] = {}
            for family in sorted({r['family'] for r in rows.values()}):
                subset = {k for k, r in rows.items() if r['family'] == family}
                contrast[endpoint]['per_family'][family] = compare([r for r in old if r.id in subset], [r for r in new if r.id in subset])
            if endpoint == 'old_dev':
                contrast[endpoint]['binding_by_answerability'] = {}
                for answerable in [True, False]:
                    subset = {k for k, r in rows.items() if r['family'] == 'binding' and r['answerable'] == answerable}
                    contrast[endpoint]['binding_by_answerability'][str(answerable)] = compare([r for r in old if r.id in subset], [r for r in new if r.id in subset])
        contrasts[f'{left}_to_{right}'] = contrast
    constraint_results = {}
    if coverage_path is not None:
        coverage = inputs.read(coverage_path)
        bank_sha = inputs.receipts[str(Path(plan['acquisition']).resolve())]
        if bank_sha not in coverage['input_sha256'].values():
            raise ValueError('constraint masks do not bind this acquisition bank')
        for family, detail in coverage['families'].items():
            for rule, ablation in detail['ablations'].items():
                ids = set(ablation['decisive_ids'])
                if len(ids) != len(ablation['decisive_ids']) or len(ids) != ablation['decisive']:
                    raise ValueError('invalid constraint mask count')
                if not ids <= {k for k, r in banks['acquisition'].items() if r['family'] == family}:
                    raise ValueError('constraint mask IDs do not belong to family')
                name = f'{family}/{rule}'
                diagnostic = {'rows': len(ids), 'informative': bool(ids), 'arms': {}, 'contrasts': {}}
                for arm, endpoints in outcomes.items():
                    diagnostic['arms'][arm] = sum(r.correct for r in endpoints['acquisition'] if r.id in ids)
                if ids:
                    for left, right in [('base', 'control'), ('base', 'anchor'), ('control', 'anchor')]:
                        if left in outcomes and right in outcomes:
                            diagnostic['contrasts'][f'{left}_to_{right}'] = compare(
                                [r for r in outcomes[left]['acquisition'] if r.id in ids],
                                [r for r in outcomes[right]['acquisition'] if r.id in ids])
                constraint_results[name] = diagnostic
    return {'schema': 1, 'run': str(run.resolve()), 'arms': arm_results, 'contrasts': contrasts,
            'rule_ablation_decisive_subsets_exploratory': constraint_results,
            'model_calls': 0, 'input_sha256': inputs.receipts, 'quality_improvement_established': False,
            'release_permitted': False, 'limitations': [
                'Visible synthetic development and reused selection acquisition; no confirmation.',
                'Matched HF inference only; no native deployment equivalence established.',
                'One training seed and a small number of worlds; paired probabilities are exploratory.',
                'A one-stage pilot does not establish sequential or jointly compositional learning.',
                'Grouped any-loss risk and row loss probability are different estimands.',
                'No source-weight rehash or model execution in this audit.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--constraint-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite audit')
    result = audit_run(args.run, args.constraint_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['arms'], indent=2))


if __name__ == '__main__':
    main()
