"""Engineering-only crossed relation fixtures; never a native corpus admission.

Shows how presence can be crossed with entity count, field count and record order.
No Rineya predictions are produced or altered. Not an intervention-gain result.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUERY = {'subject': 'Q', 'field': 'code'}

CELLS = {
    (1, 1, True): [('Q', 'code'), ('B', 'weight'), ('C', 'status'), ('D', 'note')],
    (1, 1, False): [('Q', 'weight'), ('B', 'code'), ('C', 'status'), ('D', 'note')],
    (1, 2, True): [('Q', 'code'), ('B', 'code'), ('C', 'weight'), ('D', 'status')],
    (1, 2, False): [('Q', 'weight'), ('B', 'code'), ('C', 'code'), ('D', 'status')],
    (2, 1, True): [('Q', 'code'), ('Q', 'weight'), ('B', 'status'), ('C', 'note')],
    (2, 1, False): [('Q', 'weight'), ('Q', 'status'), ('B', 'code'), ('C', 'note')],
    (2, 2, True): [('Q', 'code'), ('Q', 'weight'), ('B', 'code'), ('C', 'status')],
    (2, 2, False): [('Q', 'weight'), ('Q', 'status'), ('B', 'code'), ('C', 'code')],
}

def exists(events):
    return any(e['subject'] == QUERY['subject'] and e['field'] == QUERY['field'] for e in events)

def main():
    rows, cells, checks = [], Counter(), 0
    rules = {'entity_present': lambda e: any(x['subject'] == 'Q' for x in e),
             'requested_field_present_anywhere': lambda e: any(x['field'] == 'code' for x in e),
             'queried_entity_count_ge_2': lambda e: sum(x['subject'] == 'Q' for x in e) >= 2,
             'requested_field_count_ge_2': lambda e: sum(x['field'] == 'code' for x in e) >= 2,
             'first_record_subject_matches': lambda e: e[0]['subject'] == 'Q',
             'first_record_field_matches': lambda e: e[0]['field'] == 'code',
             'last_record_subject_matches': lambda e: e[-1]['subject'] == 'Q',
             'last_record_field_matches': lambda e: e[-1]['field'] == 'code',
             'subject_AND_field_pair_exists': exists}
    matches, by_margin = Counter(), {}
    for (entity_count, field_count, present), facts in CELLS.items():
        assert len(facts) == 4 and len(set(facts)) == 4
        for order_index, order in enumerate(permutations(range(4))):
            events = [{'subject': facts[i][0], 'field': facts[i][1], 'value': str(111 * (i + 1))}
                      for i in order]
            assert sum(e['subject'] == 'Q' for e in events) == entity_count
            assert sum(e['field'] == 'code' for e in events) == field_count
            assert exists(events) == present
            checks += 3
            id = f'fixture-e{entity_count}-f{field_count}-p{int(present)}-order{order_index:02d}'
            selected = [e['value'] for e in events if e['subject'] == 'Q' and e['field'] == 'code']
            assert len(selected) == int(present)
            prompt = '\n'.join(f'{e["subject"]}: {e["field"]}="{e["value"]}".' for e in events)
            prompt += '\nQuery Q code. Return {"v":"..."} or {"v":null} if absent.'
            row = {'id': id, 'prompt': prompt, 'query': QUERY, 'events': events,
                   'target': {'v': selected[0] if selected else None},
                   'entity_count': entity_count, 'field_count': field_count,
                   'present': present, 'engineering_fixture_only': True}
            rows.append(row)
            cells[f'e{entity_count}-f{field_count}-p{int(present)}'] += 1
            for name, rule in rules.items():
                matches[name] += rule(events) == present
            for margin in (('entity_count', entity_count), ('field_count', field_count),
                           ('entity_and_field_counts', (entity_count, field_count)),
                           ('record_order', order_index)):
                key = str(margin)
                by_margin.setdefault(key, Counter())[present] += 1
    assert len(rows) == 192 and all(v == 24 for v in cells.values())
    assert all(value[True] == value[False] for value in by_margin.values())
    assert matches['subject_AND_field_pair_exists'] == 192
    assert all(value == 96 for name, value in matches.items() if name != 'subject_AND_field_pair_exists')
    checks += len(by_margin) + 4
    # Byte-stable examples are fixtures only: no TRAIN/DEV role or reviewed flag.
    fixture_path = ROOT / 'relational-falsification-fixtures.jsonl'
    with fixture_path.open('x', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = {'schema': 'RINEYA-QUALITY/engineering-relation-falsification-v1',
              'status': 'PASS_CROSSED_RELATION_FIXTURES_ONLY',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              'rows': 192, 'checks': checks, 'cell_counts': dict(cells),
              'oracle_and_heuristic_label_matches': dict(matches),
              'interpretation': 'Every tested marginal-count/first/last shortcut falls to 50% on presence labels; the subject-and-field reference relation remains 100%.',
              'native_model_execution': False, 'native_corpus_admission': False,
              'model_quality_gain': False, 'existing_corpus_modified': False,
              'limits': ['Engineering fixture truth and shortcut behavior only.',
                         'Does not establish how Rineya will behave or learn.',
                         'Not a replacement for the frozen order-only pilot or its budget.',
                         'Native use requires separate realistic corpus/shape/admission review and permission.']}
    with (ROOT / 'relational-falsification-receipt.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
