"""Recheck whether removing a named rule actually changes SELECT-VIEWS answers.

Uses the historical pure ablation predictors, never their main() or model calls.
These are rule ablations requiring arithmetic, not shortcut floors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def coverage(records: Path):
    root = records / 'select-views-001'
    path = root / 'structural_floor_probe.py'
    spec = importlib.util.spec_from_file_location('historical_constraint_predictors', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    views = root / 'SELECT_VIEWS.jsonl'
    rows = [json.loads(line) for line in views.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate row IDs')
    result = {}
    for family, function in [('ledger_replay', module.ledger_predictors),
                             ('prerequisite_plan', module.prereq_predictors),
                             ('register_machine', module.register_predictors)]:
        family_rows = [r for r in rows if r['family'] == family]
        masks = {}
        status_predicate_ids = []
        for row in family_rows:
            prompt = json.loads(next(m['content'] for m in row['prompt'] if m['role'] == 'user'))
            if family == 'ledger_replay' and prompt['event']['status'] != 'posted':
                status_predicate_ids.append(row['id'])
            for name, predicted in function(prompt['before'], prompt['event']).items():
                mask = masks.setdefault(name, {'applicable_ids': [], 'decisive_ids': []})
                mask['applicable_ids'].append(row['id'])
                if module.canon(predicted) != module.canon(row['reference']):
                    mask['decisive_ids'].append(row['id'])
        result[family] = {'rows': len(family_rows), 'ablations': {
            name: {**mask, 'applicable': len(mask['applicable_ids']), 'decisive': len(mask['decisive_ids'])}
            for name, mask in masks.items()}}
        if family == 'ledger_replay':
            decisive = set(masks['correct_kind_ignore_status']['decisive_ids'])
            result[family]['non_posted_predicate_rows'] = len(status_predicate_ids)
            result[family]['non_posted_but_answer_unchanged_ids'] = sorted(set(status_predicate_ids) - decisive)
    return {'schema': 1, 'families': result, 'model_calls': 0, 'quality_claim': False,
            'interpretation': 'Decisive means the ablated rule yields a different reference answer, not a new performance floor.',
            'input_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, views]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite coverage report')
    report = coverage(args.records_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({family: {name: value['decisive'] for name, value in rec['ablations'].items()}
                      for family, rec in report['families'].items()}, indent=2))


if __name__ == '__main__':
    main()
