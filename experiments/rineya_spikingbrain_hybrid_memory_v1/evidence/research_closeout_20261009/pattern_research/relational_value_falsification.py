"""Crossed relation and value engineering fixtures; no model or native admission.

Adds independent payload permutations to the earlier presence-only fixtures so
the requested value cannot be a constant. It does not replace any frozen corpus.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import permutations
import hashlib
import json
from pathlib import Path
from relational_fixture_falsification import CELLS, QUERY, exists

ROOT = Path(__file__).resolve().parent
VALUES = ('117', '349', '582', '906')

def main():
    labels, selected_values, selected_ranks, position_counts = Counter(), Counter(), Counter(), Counter()
    total = checks = present_rows = 0
    margin_tables = {}
    path = ROOT / 'relational-value-falsification-fixtures.jsonl'
    with path.open('x', encoding='utf8') as stream:
        for (entity_count, field_count, present), facts in CELLS.items():
            for value_index, values in enumerate(permutations(VALUES)):
                for order_index, order in enumerate(permutations(range(4))):
                    events = [{'subject': facts[i][0], 'field': facts[i][1], 'value': values[i]} for i in order]
                    selected = [e for e in events if e['subject'] == QUERY['subject'] and e['field'] == QUERY['field']]
                    assert exists(events) == present and len(selected) == int(present)
                    assert sum(e['subject'] == 'Q' for e in events) == entity_count
                    assert sum(e['field'] == 'code' for e in events) == field_count
                    checks += 4
                    row = {'id': f'fixture-e{entity_count}-f{field_count}-p{int(present)}-v{value_index}-o{order_index}',
                           'events': events, 'query': QUERY, 'target': {'v': selected[0]['value'] if selected else None},
                           'engineering_fixture_only': True}
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    total += 1
                    labels[present] += 1
                    for factor in [('entity_count', entity_count), ('field_count', field_count),
                                   ('joint_counts', (entity_count, field_count)), ('value_assignment', value_index),
                                   ('physical_order', order_index)]:
                        margin_tables.setdefault(str(factor), Counter())[present] += 1
                    if present:
                        present_rows += 1
                        value = selected[0]['value']
                        selected_values[value] += 1
                        selected_ranks[sorted(VALUES, key=int).index(value)] += 1
                        position_counts[next(i for i, e in enumerate(events) if e is selected[0])] += 1
    assert total == 4608 and labels[True] == labels[False] == 2304
    assert all(c[True] == c[False] for c in margin_tables.values())
    assert all(c == 576 for c in selected_values.values())
    assert all(c == 576 for c in selected_ranks.values()) and all(c == 576 for c in position_counts.values())
    checks += len(margin_tables) + 4
    result = {'schema': 'RINEYA-QUALITY/engineering-relational-value-falsification-v1',
              'status': 'PASS_BALANCED_RELATION_AND_VALUE_FIXTURES_ONLY',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'rows': total,
              'present': 2304, 'absent': 2304, 'checks': checks,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parent_script_sha256': hashlib.sha256((ROOT / 'relational_fixture_falsification.py').read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'present_target_value_counts': dict(selected_values), 'present_target_numeric_rank_counts': dict(selected_ranks),
              'present_target_record_position_counts': dict(position_counts),
              'static_opportunities': {'entity_or_field_marginals_presence_accuracy': 0.5,
                                       'constant_or_minmax_payload_accuracy_conditional_present': 0.25,
                                       'fixed_record_position_accuracy_conditional_present': 0.25,
                                       'exact_subject_field_reference_oracle_accuracy': 1.0},
              'interpretation': 'Independent presence, entity/field counts, payload values/ranks and record orders remove the earlier fixture constant-value opportunity.',
              'native_model_execution': False, 'model_gain_established': False, 'corpus_admitted': False,
              'frozen_inputs_changed': False,
              'limitations': ['Exhaustive structural engineering fixtures, not native Rineya behavior.',
                              'Only eight simple subject/field graph structures and four literal values.',
                              'Native use needs realistic wording, independent corpus review, tokenizer caps and separate permission.',
                              'No new training budget or replacement of the frozen order-only study.']}
    with (ROOT / 'relational-value-falsification-receipt.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
