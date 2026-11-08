"""Static answer-shape cue audit of old admitted DEV, never model evaluation.

The rules intentionally ignore the requested subject. Native response correctness
does not establish identity matching if answer shape uniquely exposes the target.
"""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

PACKAGE = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1/experiments/rineya_spikingbrain_hybrid_memory_v1')
CORPUS = PACKAGE / 'corpora/distant_evidence_v2'
ROOT = Path(__file__).resolve().parent
RULES = {'identifier': re.compile(r'^[A-Za-z]{3}-\d{4}-[xYz]$'),
         'number': re.compile(r'^-?\d{2}\.\d{2}$')}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    design = json.loads((CORPUS / 'design.json').read_text())['rows']
    rows = [json.loads(line) for line in (CORPUS / 'dev.jsonl').read_text().splitlines()]
    results = {}
    for category, rule in RULES.items():
        selected_rows = [row for row in rows if design[row['id']]['category'] == category]
        matches, unique, not_first, examples = 0, 0, 0, []
        for row in selected_rows:
            meta = design[row['id']]
            candidate_values = [e['value'] for e in meta['events'] if rule.fullmatch(e['value'])]
            value = candidate_values[0] if len(candidate_values) == 1 else None
            unique += len(candidate_values) == 1
            correct = value == json.loads(row['target'])['v']
            matches += correct
            if correct and meta['events'][0]['value'] != value:
                not_first += 1
            if len(examples) < 3:
                examples.append({'id': row['id'], 'events': meta['events'], 'query': meta['query'],
                                 'unique_shape_prediction': value, 'target': row['target']})
        assert len(selected_rows) == unique == matches == 64
        results[category] = {'rows': 64, 'unique_shape_candidates': unique,
                             'subject_agnostic_correct_values': matches,
                             'correct_when_target_not_first': not_first, 'examples': examples}
    fixtures = [
        {'category': 'identifier', 'events': [{'subject': 'Q', 'value': 'AbC-1729-x'},
                                             {'subject': 'D', 'value': 'XyZ-4938-Y'}]},
        {'category': 'number', 'events': [{'subject': 'Q', 'value': '-48.00'},
                                         {'subject': 'D', 'value': '99.90'}]},
    ]
    for fixture in fixtures:
        candidates = [e['value'] for e in fixture['events'] if RULES[fixture['category']].fullmatch(e['value'])]
        assert len(candidates) == 2
        fixture['shape_only_rule_has_unique_answer'] = False
        fixture['query_subject'] = 'Q'
        fixture['relation_reference_value'] = fixture['events'][0]['value']
        fixture['engineering_fixture_only'] = True
    result = {'schema': 'RINEYA-QUALITY/old-answer-shape-opportunity-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS_STATIC_ANSWER_SHAPE_FALSIFICATION',
              'script_sha256': sha(Path(__file__)),
              'input_sha256': {'design.json': sha(CORPUS / 'design.json'), 'dev.jsonl': sha(CORPUS / 'dev.jsonl')},
              'declared_target_blind_rules': {category: rule.pattern for category, rule in RULES.items()},
              'results': results, 'counterexamples': fixtures,
              'interpretation': 'All tested identifier/number targets are uniquely identifiable by value shape without the requested subject. Correct non-first selection refutes universal first copying but does not establish general entity binding.',
              'raw_native_predictions_or_registered_metrics_changed': False,
              'native_model_execution': False, 'model_gain_proved': False,
              'limits': ['This is a corpus opportunity, not proof that any native model uses the shape rule.',
                         'Matched native response repairs remain actual benchmark outcome changes, but their mechanism is not identified.',
                         'Counterexamples are fixtures only and were not presented to a native model.']}
    with (ROOT / 'answer-shape-shortcuts.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'results': {k: {n:v for n,v in r.items() if n!='examples'} for k,r in results.items()}}))

if __name__ == '__main__':
    main()
