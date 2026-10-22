"""Independent raw-prompt review of static shortcut opportunities; stdlib only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re

OUTROOT = Path(__file__).resolve().parent
PACKET = Path(r'C:/Users/clclc/BusinessProject/rineya-quality-records/update-order-v1/20261005-implementation-01a10d69/corpus-v2')
OUT = OUTROOT / 'historical_patterns_shortcut_review.json'
INVENTORY, FAILURES = {}, []
CHECKS = 0
SUBJECT = r'[A-Z][A-Z0-9]*'
MISSING_RECORD = [re.compile(r'(?P<subject>' + SUBJECT + r'): (?P<field>[a-z_]+)="(?P<value>[0-9]+)"\.'), re.compile(r'record \| (?P<subject>' + SUBJECT + r') \| (?P<field>[a-z_]+) \| "(?P<value>[0-9]+)"\.')]
UPDATE_RECORD = [re.compile(r'(?P<subject>' + SUBJECT + r'): code="(?P<value>[0-9]+)"; rev=(?P<version>[0-9]+)\.'), re.compile(r'revision (?P<version>[0-9]+) \| (?P<subject>' + SUBJECT + r') \| code \| "(?P<value>[0-9]+)"\.')]
MISSING_QUERY = [re.compile(r'Ask (?P<subject>' + SUBJECT + r') (?P<field>[a-z_]+), or null if missing\. Reply \{"v":"\.\.\."\}\.'), re.compile(r'Query: (?P<subject>' + SUBJECT + r') (?P<field>[a-z_]+), or null if missing\. Return \{"v":"\.\.\."\}\.')]
UPDATE_QUERY = [re.compile(r'Ask (?P<subject>' + SUBJECT + r') code greatest revision\. Reply \{"v":"\.\.\."\}\.'), re.compile(r'Query: (?P<subject>' + SUBJECT + r') code greatest revision\. Return \{"v":"\.\.\."\}\.')]


def check(condition, description):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(description)


def sha(path):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    INVENTORY[str(Path(path).resolve())] = {'sha256': value, 'bytes': Path(path).stat().st_size}
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parse(prompt, category):
    record_patterns, query_patterns = (MISSING_RECORD, MISSING_QUERY) if category == 'missing' else (UPDATE_RECORD, UPDATE_QUERY)
    events, queries = [], []
    for line in prompt.splitlines():
        for pattern in record_patterns:
            match = pattern.fullmatch(line)
            if match:
                values = match.groupdict()
                if category == 'update':
                    values['version'], values['field'] = int(values['version']), 'code'
                events.append(values)
        for pattern in query_patterns:
            match = pattern.fullmatch(line)
            if match:
                values = match.groupdict()
                if category == 'update':
                    values['field'] = 'code'
                queries.append(values)
    if len(events) != 3 or len(queries) != 1:
        raise ValueError('raw prompt does not contain exactly three complete records and one complete query')
    return events, queries[0]


def rules(events):
    return {'first_record': events[0]['value'], 'middle_record': events[1]['value'], 'last_record': events[2]['value'], 'greatest_payload_number': max(events, key=lambda e: int(e['value']))['value'], 'least_payload_number': min(events, key=lambda e: int(e['value']))['value'], 'least_revision': min(events, key=lambda e: e['version'])['value'], 'greatest_revision': max(events, key=lambda e: e['version'])['value']}


def main():
    root_tool = OUTROOT / 'shortcut_opportunity_audit.py'
    root_result = read(OUTROOT / 'shortcut-opportunities.json')
    reproduced = read(OUTROOT / 'historical_patterns_shortcut_reproduced.json')
    copy = OUTROOT / 'historical_patterns_shortcut_reproduction.py'
    original_code = root_tool.read_text(encoding='utf-8')
    copied_code = copy.read_text(encoding='utf-8')
    check(copied_code.replace("OUT / 'historical_patterns_shortcut_reproduced.json'", "OUT / 'shortcut-opportunities.json'") == original_code, 'reproduction script differs only in output filename')
    comparable = lambda result: {k: v for k, v in result.items() if k not in ['created_utc', 'script_sha256']}
    check(comparable(root_result) == comparable(reproduced), 'original and independent reproduction results identical except time/script digest')
    check(root_result['script_sha256'] == sha(root_tool), 'root receipt binds exact root audit script')
    check(reproduced['script_sha256'] == sha(copy), 'reproduction receipt binds exact copied audit script')
    manifest = read(PACKET / 'manifest.json')
    design = read(PACKET / 'design.json')
    for entry in manifest['files'].values():
        path = PACKET / entry['path']
        check(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'frozen corpus file lock ' + entry['path'])
    rows = {}
    for condition in ['control', 'candidate', 'dev']:
        path = PACKET / manifest['files'][condition]['path']
        rows[condition] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        check(manifest['splits'][condition]['protected'] is False, condition + ': explicitly unprotected split')
    missing, update_summary, parsed, details = {}, {}, {}, []
    for condition, rr in rows.items():
        table, raw_mentions, correct, total = Counter(), Counter(), 0, 0
        update_hits, ranks, update_total = Counter(), Counter(), 0
        parsed[condition] = {}
        for row in rr:
            meta = design[condition][row['id']]
            if meta['category'] not in ['missing', 'update']:
                continue
            events, query = parse(row['prompt'], meta['category'])
            check(events == meta['events'] and all(query[k] == meta['query'][k] for k in query), condition + ':' + row['id'] + ': raw events/query equal metadata')
            check(meta['row_sha256'] == digest(row), condition + ':' + row['id'] + ': raw row identity')
            blocks = row['prompt'].split('\n---\n')
            expected = [meta['evidence_block_text'], meta['filler_block_text']] if meta['role'] == 'distant_heldout' else [meta['filler_block_text'], meta['evidence_block_text']]
            check(len(blocks) == 4 and blocks[1:3] == expected, condition + ':' + row['id'] + ': raw block role')
            for name, text in [('evidence', meta['evidence_block_text']), ('filler', meta['filler_block_text'])]:
                start, end = meta['block_spans'][name]
                check(row['prompt'][start:end] == text, condition + ':' + row['id'] + ': raw character span ' + name)
            entry = {'row': row, 'metadata': meta, 'events': events, 'query': query}
            parsed[condition][row['id']] = entry
            gold = json.loads(row['target'])['v']
            if meta['category'] == 'missing':
                answers = [e['value'] for e in events if e['subject'] == query['subject'] and e['field'] == query['field']]
                check(len(answers) <= 1 and gold == (answers[0] if answers else None), condition + ':' + row['id'] + ': raw requested-field value or null truth')
                evidence_count = sum(e['subject'] == query['subject'] for e in events)
                prompt_count = len(re.findall(r'\b' + re.escape(query['subject']) + r'\b', row['prompt']))
                check(prompt_count == evidence_count + 1, condition + ':' + row['id'] + ': full prompt entity occurrences equal evidence record mentions plus query')
                truth = bool(answers)
                prediction = evidence_count >= 2
                correct += prediction == truth
                total += 1
                table[str(evidence_count) + '_mentions_' + ('present' if truth else 'absent')] += 1
                raw_mentions[str(prompt_count) + '_literal_mentions_' + ('present' if truth else 'absent')] += 1
                details.append({'condition': condition, 'id': row['id'], 'record_entity_count': evidence_count, 'whole_prompt_literal_entity_count': prompt_count, 'query': query, 'raw_parsed_events': events, 'target': row['target'], 'answerable': truth, 'field_agnostic_record_count_prediction': prediction})
            else:
                check(len({e['version'] for e in events}) == 3 and len({e['value'] for e in events}) == 3 and all(e['subject'] == query['subject'] and e['field'] == query['field'] for e in events), condition + ':' + row['id'] + ': distinct revisions/payloads and raw field/subject binding')
                choices = rules(events)
                check(gold == choices['greatest_revision'], condition + ':' + row['id'] + ': independently replayed raw greatest-revision target')
                for rule, value in choices.items():
                    update_hits[rule] += value == gold
                payload_rank = next(i + 1 for i, e in enumerate(sorted(events, key=lambda e: int(e['value']))) if e['value'] == gold)
                entry['payload_rank'] = payload_rank
                entry['gold'] = gold
                entry['choices'] = choices
                ranks[payload_rank] += 1
                update_total += 1
        missing[condition] = {'rows': total, 'label_predictable_without_requested_field': correct, 'accuracy': correct / total, 'count_by_presence': dict(table), 'whole_prompt_entity_occurrence_table': dict(raw_mentions), 'shortcut_record_count_threshold': 2, 'equivalent_whole_prompt_occurrence_threshold_in_this_corpus': 3}
        update_summary[condition] = {'rows': update_total, 'oracle_agreements': dict(update_hits), 'payload_rank_of_correct_revision': dict(sorted(ranks.items())), 'source': 'raw prompt structure and locked targets; no native predictions'}
        check({k: v for k, v in missing[condition].items() if k in ['rows', 'label_predictable_without_requested_field', 'accuracy', 'count_by_presence']} == root_result['missing_label_shortcut'][condition], condition + ': independent raw missing-label result equals root tool')
        check(update_summary[condition]['rows'] == root_result['update_rule_opportunities'][condition]['rows'] and update_summary[condition]['oracle_agreements'] == root_result['update_rule_opportunities'][condition]['oracle_agreements'], condition + ': independent raw numeric/order rule agreements equal root tool')
    check(set(r['id'] for r in rows['control']) == set(r['id'] for r in rows['candidate']), 'paired TRAIN control/candidate row IDs identical')
    control_by_id, candidate_by_id = [{r['id']: r for r in rows[c]} for c in ['control', 'candidate']]
    for ident, control in control_by_id.items():
        candidate = candidate_by_id[ident]
        meta = design['control'][ident]
        check(control['target'] == candidate['target'] and control['family_id'] == candidate['family_id'], ident + ': paired TRAIN target/family identical')
        if meta['category'] == 'update':
            a, b = parsed['control'][ident], parsed['candidate'][ident]
            signature = lambda e: sorted((x['subject'], x['field'], x['version'], x['value']) for x in e['events'])
            check(signature(a) == signature(b), ident + ': paired TRAIN record multiset identical')
            check(a['metadata']['query'] == b['metadata']['query'] and a['metadata']['filler_block_text'] == b['metadata']['filler_block_text'] and a['metadata']['token_geometry'] == b['metadata']['token_geometry'], ident + ': paired TRAIN query/filler and saved token geometry identical')
        else:
            check(control == candidate, ident + ': non-update TRAIN row unchanged')
    permutation_groups = {}
    for condition in ['control', 'candidate', 'dev']:
        groups = defaultdict(list)
        for entry in parsed[condition].values():
            m = entry['metadata']
            if m['category'] == 'update':
                groups[(m['world_id'], m['counterfactual_id'], m['role'])].append(entry)
        summary = Counter()
        for key, entries in groups.items():
            check(len(entries) == 6, condition + ':' + str(key) + ': six within-world rows')
            signatures = {tuple(sorted((e['version'], e['value']) for e in entry['events'])) for entry in entries}
            check(len(signatures) == 1 and len({entry['gold'] for entry in entries}) == 1 and len({entry['payload_rank'] for entry in entries}) == 1, condition + ':' + str(key) + ': record multiset/target/numeric rank unchanged across six rows')
            check(len({digest(entry['metadata']['token_geometry']) for entry in entries}) == 1, condition + ':' + str(key) + ': saved token geometry unchanged across six rows')
            event_orders = {tuple((e['version'], e['value']) for e in entry['events']) for entry in entries}
            if condition in ['candidate', 'dev']:
                expected_orders = set(itertools.permutations(next(iter(signatures))))
                check(event_orders == expected_orders, condition + ':' + str(key) + ': all six actual record permutations covered')
            else:
                check(len(event_orders) == 1, condition + ':' + str(key) + ': frozen control repeats one actual order')
            summary[(len(entries), len(event_orders))] += 1
        permutation_groups[condition] = {'worlds': len({key[0] for key in groups}), 'world_counterfactual_position_groups': len(groups), 'six_row_group_table': {str(k): v for k, v in summary.items()}, 'numeric_rank_invariant_within_groups': True, 'raw_record_permutation_verification': True, 'saved_token_geometry_only_not_retokenized': True}
    fixture_details = []
    for fixture in root_result['engineering_counterexamples']:
        text = '\n'.join(e['subject'] + ': ' + e['field'] + '="' + e['value'] + '".' for e in fixture['events'])
        prompt = 'Use these exhaustive records. Output compact JSON only.\n\n---\n' + text + '\n---\nNotes: quiet.\n---\nAsk E1 code, or null if missing. Reply {"v":"..."}.'
        events, query = parse(prompt, 'missing')
        truth = any(e['subject'] == query['subject'] and e['field'] == query['field'] for e in events)
        count = sum(e['subject'] == query['subject'] for e in events)
        prediction = count >= 2
        check(truth != prediction and truth == fixture['truth'] and prediction == fixture['count_shortcut_prediction'], fixture['name'] + ': independently rendered/raw-parsed engineering counterexample falsifies general rule')
        fixture_details.append({'name': fixture['name'], 'raw_fixture_prompt': prompt, 'truth': truth, 'record_entity_count': count, 'count_rule_prediction': prediction, 'model_executed': False})
    conflict_strata = {}
    for condition, entries in parsed.items():
        counts = Counter((entry['payload_rank'], next(i + 1 for i, e in enumerate(entry['events']) if e['value'] == entry['gold'])) for entry in entries.values() if entry['metadata']['category'] == 'update')
        conflict_strata[condition] = [{'oracle_payload_numeric_rank': rank, 'oracle_record_position': slot, 'rows': count} for (rank, slot), count in sorted(counts.items())]
    result = {'schema': 'Rineya-quality/offline-shortcut-independent-review-v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_RAW_PROMPT_SHORTCUT_REVIEW' if not FAILURES else 'FAIL_RAW_PROMPT_SHORTCUT_REVIEW', 'reviewer': 'independent Codex provenance/pattern subagent', 'script_sha256': sha(__file__), 'checks_run': CHECKS, 'issues': FAILURES, 'original_tool': str(root_tool), 'original_artifact': str(OUTROOT / 'shortcut-opportunities.json'), 'reproduction_artifact': str(OUTROOT / 'historical_patterns_shortcut_reproduced.json'), 'reproduction_matches_original_excluding_timestamp_and_script_hash': comparable(root_result) == comparable(reproduced), 'raw_missing_label_shortcut': missing, 'raw_update_rule_opportunities': update_summary, 'same_world_six_row_permutation_checks': permutation_groups, 'numeric_rank_by_true_record_position': conflict_strata, 'raw_missing_row_details': details, 'engineering_rule_counterexamples': fixture_details, 'interpretation': ['All existing missing labels are predictable by queried-entity evidence-record count alone:24/24 control TRAIN,24/24 candidate TRAIN,64/64 DEV. Direct raw parsing verifies record/entity counts and requested-field/null targets. This is static label predictability, not evidence a native model uses the shortcut.', 'Evidence-record counts are2 for present versus1 for absent. Complete prompts include one extra occurrence in the query, giving3 versus2 literal mentions; do not apply the >=2 record-count threshold to whole-prompt occurrences.', 'Payload-rank opportunity is shared by TRAIN control/candidate: least numeric payload180/384, greatest96/384. DEV reverses the dominant numeric shortcut: least48/384, greatest192/384. Payload numeric ranks are invariant across each same-world counterfactual/position group of six record orders.', 'Candidate and DEV cover all six permutations, while control repeats one order per six-row group.16 worlds per condition yield32 world-counterfactual combinations and64 world-counterfactual-position groups. These rows are dependent variants.', 'The shared numeric-rank nuisance shift does not confound the frozen control/candidate order-only contrast because paired records/targets/query/filler/exposure geometry are identical. Candidate positional balance alone leaves a competitive least-payload shortcut on TRAIN. Future behavioral interpretation must inspect order-versus-payload conflict strata without changing registered primary metrics.', 'Engineering counterexamples demonstrate that entity-count threshold is not a valid general answerability rule; they test the rule only, not any native model.', 'No source/corpus/policy/metric/model changes, no protected bank access or native retry. Frozen native pilot remains untested and RAM guard is not bypassed.'], 'limitations': ['Static corpus opportunities do not measure model behavior or capability.', 'Token geometry equivalence is checked against locked saved metadata; this stdlib review does not rerun tokenization.', 'No cross-corpus causal or generalization claim; no confirmation bank or new native inference.', 'Posthoc conflict strata are diagnostic and do not replace frozen primary metrics or authorize training.'], 'native_execution': False, 'model_or_GPU_execution': False, 'source_corpus_policy_or_metric_changes': False, 'input_and_artifact_sha256': INVENTORY}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'checks_run': CHECKS, 'issues': FAILURES, 'output': str(OUT), 'missing_correct': {c: m['label_predictable_without_requested_field'] for c, m in missing.items()}, 'numeric_min_max': {c: {k: u['oracle_agreements'][k] for k in ['least_payload_number', 'greatest_payload_number']} for c, u in update_summary.items()}}, indent=2))
    return 0 if not FAILURES else 1


if __name__ == '__main__':
    raise SystemExit(main())
