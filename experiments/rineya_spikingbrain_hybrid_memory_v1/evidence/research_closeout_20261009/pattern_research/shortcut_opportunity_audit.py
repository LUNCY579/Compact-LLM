"""Offline corpus shortcut opportunities; not model evaluation or capability evidence.

Independent of producer oracles. All examples/labels are already unprotected
synthetic TRAIN/DEV. Never edits the frozen corpus or any native receipt.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

BASE = Path('C:/Users/clclc/BusinessProject/rineya-quality-records/update-order-v1/20261005-implementation-01a10d69')
OUT = Path(__file__).resolve().parent
PACKET = BASE / 'corpus-v2'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def reference_exists(events, query):
    return any(e['subject'] == query['subject'] and e['field'] == query['field'] for e in events)

def count_shortcut(events, query):
    # This deliberately never reads the requested field; useful as a falsifier.
    return sum(e['subject'] == query['subject'] for e in events) >= 2

def main():
    manifest, design = read(PACKET / 'manifest.json'), read(PACKET / 'design.json')
    for item in manifest['files'].values():
        path = PACKET / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    rows = {c: jsonl(PACKET / manifest['files'][c]['path']) for c in ('control', 'candidate', 'dev')}
    missing, updates, assertion_count = {}, {}, 0
    examples = []
    for condition, split in rows.items():
        count_table = Counter()
        correct, total = 0, 0
        update_hits, update_total = Counter(), 0
        for row in split:
            meta = design[condition][row['id']]
            if meta['category'] == 'missing':
                target = json.loads(row['target'])['v']
                truth = target is not None
                events, query = meta['events'], meta['query']
                assert truth == reference_exists(events, query)
                assertion_count += 1
                mentions = sum(e['subject'] == query['subject'] for e in events)
                predicted = count_shortcut(events, query)
                count_table[f'{mentions}_mentions_{"present" if truth else "absent"}'] += 1
                correct += predicted == truth
                total += 1
                if condition == 'dev' and len(examples) < 4:
                    examples.append({'id': row['id'], 'query': query, 'events': events,
                                     'answerable': truth, 'entity_mentions': mentions,
                                     'field_agnostic_count_prediction': predicted})
            elif meta['category'] == 'update':
                events = meta['events']
                assert len(events) == 3 and len({e['version'] for e in events}) == 3
                gold = json.loads(row['target'])['v']
                assert gold == max(events, key=lambda e: e['version'])['value']
                assert all(e['subject'] == meta['query']['subject'] and e['field'] == 'code' for e in events)
                assertion_count += 3
                rules = {'first_record': events[0]['value'], 'middle_record': events[1]['value'],
                         'last_record': events[2]['value'],
                         'greatest_payload_number': max(events, key=lambda e: int(e['value']))['value'],
                         'least_payload_number': min(events, key=lambda e: int(e['value']))['value'],
                         'least_revision': min(events, key=lambda e: e['version'])['value'],
                         'greatest_revision': max(events, key=lambda e: e['version'])['value']}
                for rule, value in rules.items():
                    update_hits[rule] += value == gold
                update_total += 1
        missing[condition] = {'rows': total, 'label_predictable_without_requested_field': correct,
                              'accuracy': correct / total, 'count_by_presence': dict(count_table)}
        updates[condition] = {'rows': update_total, 'oracle_agreements': dict(update_hits),
                              'source': 'ground-truth structure, not native model predictions'}
    # Falsify the field-agnostic rule on fresh engineering cases, not model rows.
    query = {'subject': 'E1', 'field': 'code'}
    fixtures = [
        {'name': 'present_with_one_entity_mention', 'events': [
            {'subject': 'E1', 'field': 'code', 'value': '111'},
            {'subject': 'E2', 'field': 'code', 'value': '222'},
            {'subject': 'E2', 'field': 'weight', 'value': '333'}]},
        {'name': 'absent_with_two_entity_mentions', 'events': [
            {'subject': 'E1', 'field': 'weight', 'value': '111'},
            {'subject': 'E1', 'field': 'status', 'value': '222'},
            {'subject': 'E2', 'field': 'code', 'value': '333'}]},
    ]
    for fixture in fixtures:
        fixture['truth'] = reference_exists(fixture['events'], query)
        fixture['count_shortcut_prediction'] = count_shortcut(fixture['events'], query)
        assert fixture['truth'] != fixture['count_shortcut_prediction']
        assertion_count += 1
    assert missing['dev']['label_predictable_without_requested_field'] == 64
    assert missing['control']['label_predictable_without_requested_field'] == 24
    assert updates['control']['oracle_agreements']['first_record'] == 288
    assert updates['candidate']['oracle_agreements']['first_record'] == 128
    assert updates['dev']['oracle_agreements']['first_record'] == 128
    assertion_count += 5
    result = {'schema': 'RINEYA-QUALITY/offline-shortcut-opportunity-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_OFFLINE_CORPUS_FALSIFICATION',
              'script_sha256': sha(Path(__file__)), 'assertions': assertion_count,
              'input_files': {str(path): sha(path) for path in (PACKET / 'manifest.json', PACKET / 'design.json',
                             PACKET / 'control_train.jsonl', PACKET / 'candidate_train.jsonl', PACKET / 'dev.jsonl')},
              'missing_label_shortcut': missing, 'update_rule_opportunities': updates,
              'missing_examples': examples, 'engineering_counterexamples': fixtures,
              'interpretation': 'The current missing category cannot distinguish requested-field binding from a queried-entity mention-count shortcut. This does not demonstrate that the native model uses that shortcut.',
              'unchanged_primary_question': 'The frozen order-only pilot remains untested; its corpus and registered primary metrics are not changed.',
              'next_measurement_requirement': 'Any later claim of robust answerability needs presence/absence crossed with queried-entity mention count and wrong-field/wrong-subject distractors.',
              'capability_claim': False, 'native_execution': False, 'serving_policy_changed': False,
              'limitations': ['Static label predictability is not observed model behavior.',
                              'Engineering counterexamples test the shortcut rule only.',
                              'Posthoc diagnostics do not replace frozen metrics or authorize a new model run.']}
    path = OUT / 'shortcut-opportunities.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'assertions': assertion_count,
                      'missing': missing, 'updates': updates}))

if __name__ == '__main__':
    main()
