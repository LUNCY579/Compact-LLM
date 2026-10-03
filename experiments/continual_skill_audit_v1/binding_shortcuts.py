"""Surface extraction diagnostics; not an exhaustive bound on binding shortcuts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def predictors(row):
    context, people = row['context'], row['people']
    first = {person: context.index(person) for person in people}
    last = {person: context.rindex(person) for person in people}
    counts = {person: context.count(person) for person in people}
    chronological = sorted(people, key=lambda p: first[p])
    question = row['prompt'][1]['content'].split('\nQUESTION\n', 1)[1]
    queried_names = [person for person in people if person in question]
    minimum = [person for person in people if counts[person] == min(counts.values())]
    maximum = [person for person in people if counts[person] == max(counts.values())]
    return {
        'always_unknown': 'UNKNOWN',
        'first_person': min(people, key=first.get),
        'last_person': max(people, key=last.get),
        'least_frequent_first_tie': min(chronological, key=counts.get),
        'most_frequent_first_tie': max(chronological, key=counts.get),
        'unique_least_frequent_else_unknown': minimum[0] if len(minimum) == 1 else 'UNKNOWN',
        'unique_most_frequent_else_unknown': maximum[0] if len(maximum) == 1 else 'UNKNOWN',
        'alphabetical_first': min(people),
        'shortest_name_first_tie': min(chronological, key=len),
        'question_name_else_unknown': queried_names[0] if len(queried_names) == 1 else 'UNKNOWN',
    }


def audit(directory):
    files = [directory / 'anchor.json', directory / 'dev.json']
    rows = [r for path in files for r in json.loads(path.read_text(encoding='utf-8')) if r['family'] == 'binding']
    results = {}
    for split in ['anchor', 'dev']:
        for task in ['direct', 'two_hop']:
            for salience in ['all', 'low', 'equal', 'high']:
                subset = [r for r in rows if r['split'] == split and r['task'] == task and (salience == 'all' or r['salience'] == salience)]
                names = predictors(subset[0]).keys()
                scores = {}
                for name in names:
                    available = [r for r in subset if r['answerable']]
                    scores[name] = {'correct': sum(predictors(r)[name] == r['reference'] for r in subset),
                                    'n': len(subset), 'answerable_correct': sum(predictors(r)[name] == r['reference'] for r in available),
                                    'answerable_n': len(available)}
                results[f'{split}/{task}/{salience}'] = scores
    return {'model_calls': 0, 'results': results,
            'limitations': ['Predefined surface predictors only; no exhaustive shortcut guarantee.',
                           'Derivative rows share worlds and cannot be treated as independent trials.',
                           'Always UNKNOWN succeeds on half this balanced bank without binding competence.',
                           'All-case and answerable-only counts are reported separately.'],
            'input_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite diagnostics')
    result = audit(args.bank)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for label, scores in result['results'].items():
        if label.endswith('/all'):
            print(label, {name: f"{value['answerable_correct']}/{value['answerable_n']}" for name, value in scores.items()})


if __name__ == '__main__':
    main()
