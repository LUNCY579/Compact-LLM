"""Context-free nuisance-feature diagnostics for prospective joint rule worlds.

The best polarity is selected on this same visible bank: an optimistic diagnostic,
not trained-model evidence or a certified bound over all possible shortcuts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def features(row):
    payload = json.loads(row['prompt'][1]['content'])
    events = payload['events']
    result = {'first_kind_is_add': events[0]['kind'] == 'add',
              'events_have_identical_keys': events[0].keys() == events[1].keys()}
    if row['family'] == 'register_machine':
        add = next(e for e in events if e['kind'] == 'add')
        mix = next(e for e in events if e['kind'] == 'mix')
        result.update(same_value_parity=(add['value'] - mix['value']) % 2 == 0,
                      add_value_larger=add['value'] > mix['value'],
                      add_value_negative=add['value'] < 0,
                      add_value_above_modulus=add['value'] >= 97)
    elif row['family'] == 'ledger_replay':
        add = next(e for e in events if e['kind'] == 'add')
        scale = next(e for e in events if e['kind'] == 'scale')
        result.update(same_amount_parity=(add['amount'] - scale['amount']) % 2 == 0,
                      scale_amount_gt7=scale['amount'] > 7,
                      scale_amount_larger=scale['amount'] > add['amount'])
    else:
        union = events[0]['add'] | events[1]['add']
        result.update(add_touches_pre_on=bool(union & events[0]['pre_on']),
                      add_touches_pre_off=bool(union & events[0]['pre_off']),
                      first_add_has_more_bits=events[0]['add'].bit_count() > events[1]['add'].bit_count())
    return result


def audit(directory):
    files = [directory / 'train.jsonl', directory / 'dev.jsonl']
    rows = [json.loads(line) for file in files for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
    output = {}
    for split in ['train', 'dev']:
        for family in ['ledger_replay', 'prerequisite_plan', 'register_machine']:
            subset = [r for r in rows if r['split'] == split and r['family'] == family and r['task'] == 'joint']
            if not subset:
                raise ValueError('empty joint stratum')
            labels = [r['reference'] != 'UNKNOWN' for r in subset]
            values = [features(r) for r in subset]
            scores = {}
            for feature in values[0]:
                matches = sum(v[feature] == label for v, label in zip(values, labels))
                scores[feature] = {'best_polarity_correct': max(matches, len(subset) - matches),
                                   'n': len(subset), 'best_polarity_rate': max(matches, len(subset) - matches) / len(subset)}
            output[f'{split}/{family}'] = scores
    return {'schema': 1, 'features': output, 'model_calls': 0,
            'limitations': ['Same-bank optimistic polarity selection.', 'Incomplete feature class; no guarantee against other shortcuts.',
                            'Semantic arithmetic and bit operations are legitimate solutions, not automatically shortcut violations.'],
            'input_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite diagnostic')
    result = audit(args.bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({group: {feature: value['best_polarity_rate'] for feature, value in values.items()}
                      for group, values in result['features'].items()}, indent=2))


if __name__ == '__main__':
    main()
