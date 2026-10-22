"""Read-only, standard-library analysis of the frozen 672-row native anchor.

Literal extraction is fixed before inspection: accept finite duplicate-free
JSON scalars, or an object with exactly one key named 'v' and a scalar value.
No substrings, fences, numeric tolerances, target-dependent recovery or judge.
Type-exact matches and canonical code-lexeme matches are separate diagnostics.
The latter maps strings to their content and other scalars to JSON text, so an
integer 682 can identify the visible code lexeme '682' without being called a
correct JSON string. Leading zeros, case, decimal spelling and types are not
silently repaired. Original strict and typed metrics remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import re

SOURCE_ROOT = Path('C:/Users/clclc/BusinessProject/rineya-quality-records/update-order-v1/20261005-implementation-01a10d69')
OUTPUT_ROOT = Path('C:/Users/clclc/BusinessProject/rineya-quality-records/update-order-v1/20261005-pattern-research-01a10d69')
FROZEN = {
    'proposal_sha256': '91fd07797c777c0c90284842544346cae089407abd763e3c4baefcc0b534fe97',
    'code_sha256': '26f1c07bbe354a422b1a6d4ea5988a21b063621f055c9a92155fa5e5e2a738ef',
    'corpus_sha256': 'ce2c1e1489ab5dc6806a4b1dc6a2de1e51c6623df9779313d81d28115ef36dc3',
    'anchor_receipt_sha256': '5e9a1843c2f5f7229d5a073c632078ebfa0174143accac0c72d1b23975e85d37',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON literal')


def finite(value):
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def parse(text):
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        require(finite(value), 'nonfinite parsed number')
        return True, value
    except (ValueError, TypeError, RecursionError):
        return False, None


def same_type(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(value, dict):
        return set(value) == set(expected) and all(same_type(value[k], expected[k]) for k in expected)
    if isinstance(value, list):
        return len(value) == len(expected) and all(same_type(a, b) for a, b in zip(value, expected))
    return True


def typed_equal(value, expected):
    return same_type(value, expected) and value == expected


def literal(text):
    valid, value = parse(text)
    if not valid:
        return {'extracted': False, 'form': 'invalid_json', 'value': None, 'type': None, 'code_lexeme': None}
    form = 'scalar'
    if isinstance(value, dict):
        if set(value) != {'v'}:
            return {'extracted': False, 'form': 'wrong_object_shape', 'value': None, 'type': None, 'code_lexeme': None}
        value, form = value['v'], 'exact_v_object'
    if isinstance(value, (dict, list)):
        return {'extracted': False, 'form': 'non_scalar', 'value': None, 'type': None, 'code_lexeme': None}
    # Never stringify against the target. This serialization is fixed for all
    # scalars and is separately labeled as a lexeme diagnostic.
    lexeme = value if type(value) is str else json.dumps(value, allow_nan=False, separators=(',', ':'))
    return {'extracted': True, 'form': form, 'value': value, 'type': type(value).__name__, 'code_lexeme': lexeme}


def payload_match(decoded, value):
    return decoded['extracted'] and type(value) is str and decoded['code_lexeme'] == value


def type_exact_match(decoded, value):
    return decoded['extracted'] and typed_equal(decoded['value'], value)


def verify_original_scores(prediction, target):
    valid, value = parse(prediction['prediction'])
    target_valid, expected = parse(target)
    require(target_valid and isinstance(expected, dict), 'invalid frozen target')
    contract = valid and same_type(value, expected)
    checks = {
        'strict_exact_match': prediction['prediction'] == target,
        'normalized_exact_match': prediction['prediction'].strip().casefold() == target.strip().casefold(),
        'typed_value_correct': contract and value == expected,
        'json_contract_compliance': contract,
        'json_valid': valid,
    }
    require(all(type(prediction[k]) is bool and prediction[k] == v for k, v in checks.items()),
            'original score disagrees with raw text')
    return checks


def actual_events(row, meta):
    blocks = row['prompt'].split('\n---\n')
    require(len(blocks) == 4, 'unexpected raw prompt blocks')
    index = 1 if meta['role'] == 'distant_heldout' else 2
    evidence = blocks[index]
    regex = re.compile(r'^(?:revision (\d+) |record )\| (N\d+) \| (\w+) \| ("(?:[^"\\]|\\.)*")\.$')
    events = []
    for line in evidence.splitlines():
        match = regex.fullmatch(line)
        require(match is not None, 'unexpected raw evidence record')
        version, subject, field, quoted = match.groups()
        event = {'subject': subject, 'field': field, 'value': json.loads(quoted)}
        if version is not None:
            event['version'] = int(version)
        events.append(event)
    require(events == meta['events'] and evidence == meta['evidence_block_text'],
            'sidecar events differ from actual prompt')
    query = blocks[3]
    match = re.fullmatch(r'Query: (N\d+) code (greatest revision|, or null if missing)\. Return \{"v":"\.\.\."\}\.', query)
    # The missing renderer puts its comma directly after 'code'.
    if match is None:
        match = re.fullmatch(r'Query: (N\d+) code(, or null if missing)\. Return \{"v":"\.\.\."\}\.', query)
    require(match is not None and match[1] == meta['query']['subject'], 'unexpected raw query')
    return events, query, evidence


def canonical(events):
    return sorted(events, key=lambda e: json.dumps(e, sort_keys=True, separators=(',', ':')))


def counts(rows):
    n = len(rows)
    result = {'rows': n, 'original_strict_correct': sum(r['strict'] for r in rows),
              'original_typed_correct': sum(r['typed'] for r in rows),
              'diagnostic_type_exact_payload_matches': sum(r['type_exact_payload_match'] for r in rows),
              'diagnostic_code_lexeme_payload_matches': sum(r['code_lexeme_payload_match'] for r in rows),
              'decoded_forms': dict(Counter(r['decoded']['form'] for r in rows)),
              'decoded_types': dict(Counter(r['decoded']['type'] for r in rows))}
    if rows and rows[0]['category'] == 'update':
        for key in ('matches_first_payload', 'matches_middle_payload', 'matches_last_payload',
                    'matches_greatest_code_payload', 'matches_any_revision_numeral', 'matches_greatest_revision_numeral'):
            result[key] = sum(r[key] for r in rows)
        result['exclusive_selection_classes'] = dict(Counter(r['selection_class'] for r in rows))
    return result


def update_row(row, prediction, meta):
    events, query, evidence = actual_events(row, meta)
    require(len(events) == 3 and len({e['version'] for e in events}) == 3, 'three distinct revisions required')
    require(all(e['subject'] == meta['query']['subject'] and e['field'] == 'code' for e in events), 'update records do not match query')
    require(all(re.fullmatch(r'[1-9]\d\d', e['value']) for e in events), 'registered three-digit code lexemes changed')
    latest = max(events, key=lambda e: e['version'])
    greatest_code = max(events, key=lambda e: int(e['value']))
    require(json.loads(row['target']) == {'v': latest['value']}, 'target differs from independent latest-revision oracle')
    decoded = literal(prediction['prediction'])
    latest_match = payload_match(decoded, latest['value'])
    max_code_match = payload_match(decoded, greatest_code['value'])
    any_payload = any(payload_match(decoded, e['value']) for e in events)
    any_revision = decoded['extracted'] and any(decoded['code_lexeme'] == str(e['version']) for e in events)
    category = ('latest_and_greatest_code_overlap' if latest_match and max_code_match else
                'latest_only' if latest_match else 'greatest_code_only' if max_code_match else
                'other_visible_payload' if any_payload else 'visible_revision_numeral' if any_revision else
                'unobserved_literal' if decoded['extracted'] else 'unextractable')
    code_rank = sorted(e['value'] for e in events).index(latest['value'])
    return {
        'id': row['id'], 'category': 'update', 'world_id': meta['world_id'],
        'counterfactual_id': meta['counterfactual_id'], 'role': meta['role'],
        'permutation': meta['permutation'], 'latest_position': meta['latest_position'],
        'distance_band': meta['token_geometry']['registered_gap_band'],
        'query': query, 'evidence': evidence, 'events': events,
        'latest_revision': latest['version'], 'latest_payload': latest['value'],
        'latest_payload_numeric_rank': code_rank,
        'greatest_code_payload': greatest_code['value'],
        'comparison_key_conflict': latest['value'] != greatest_code['value'],
        'prediction': prediction['prediction'], 'decoded': decoded,
        'strict': prediction['strict_exact_match'], 'typed': prediction['typed_value_correct'],
        'type_exact_payload_match': type_exact_match(decoded, latest['value']),
        'code_lexeme_payload_match': latest_match,
        'matches_first_payload': payload_match(decoded, events[0]['value']),
        'matches_middle_payload': payload_match(decoded, events[1]['value']),
        'matches_last_payload': payload_match(decoded, events[2]['value']),
        'matches_most_recent_in_text': payload_match(decoded, events[-1]['value']),
        'matches_greatest_code_payload': max_code_match,
        'matches_any_revision_numeral': bool(any_revision),
        'matches_greatest_revision_numeral': bool(decoded['extracted'] and decoded['code_lexeme'] == str(latest['version'])),
        'matched_visible_payload_text_positions': [i for i, e in enumerate(events) if payload_match(decoded, e['value'])],
        'selection_class': category,
    }


def update_bundles(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['world_id'], row['counterfactual_id'], row['role'])].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda r: r['permutation'])
        require(len(group) == 6 and {r['permutation'] for r in group} == set(range(6)), 'incomplete six-order bundle')
        require(all(canonical(r['events']) == canonical(group[0]['events']) for r in group), 'bundle facts changed')
        extracted = all(r['decoded']['extracted'] for r in group)
        typed_values = {(r['decoded']['type'], json.dumps(r['decoded']['value'], sort_keys=True, separators=(',', ':'))) for r in group}
        lexemes = {r['decoded']['code_lexeme'] for r in group}
        result.append({'world_id': key[0], 'counterfactual_id': key[1], 'role': key[2],
            'latest_payload': group[0]['latest_payload'], 'greatest_code_payload': group[0]['greatest_code_payload'],
            'comparison_key_conflict': group[0]['comparison_key_conflict'],
            'original_strict_all_six_correct': all(r['strict'] for r in group),
            'original_typed_all_six_correct': all(r['typed'] for r in group),
            'diagnostic_type_exact_all_six_payload_matches': all(r['type_exact_payload_match'] for r in group),
            'diagnostic_code_lexeme_all_six_payload_matches': all(r['code_lexeme_payload_match'] for r in group),
            'diagnostic_code_lexeme_latest_match_count': sum(r['code_lexeme_payload_match'] for r in group),
            'raw_prediction_consistent': len({r['prediction'] for r in group}) == 1,
            'decoded_type_sensitive_value_consistent': extracted and len(typed_values) == 1,
            'code_lexeme_consistent': extracted and len(lexemes) == 1,
            'code_lexeme_all_six_greatest_code_matches': all(r['matches_greatest_code_payload'] for r in group),
            'row_ids': [r['id'] for r in group],
            'permutation_predictions': [{'permutation': r['permutation'], 'latest_position': r['latest_position'],
                'prediction': r['prediction'], 'code_lexeme': r['decoded']['code_lexeme'],
                'latest_match': r['code_lexeme_payload_match'], 'selection_class': r['selection_class']} for r in group]})
    require(len(result) == 64 and len({r['world_id'] for r in result}) == 16, 'registered dependent update groups changed')
    return result


def update_pairs(rows):
    cf_groups, role_groups = defaultdict(dict), defaultdict(dict)
    for row in rows:
        cf_groups[(row['world_id'], row['role'], row['permutation'])][row['counterfactual_id']] = row
        role_groups[(row['world_id'], row['counterfactual_id'], row['permutation'])][row['role']] = row
    cf, role = [], []
    for key, pair in sorted(cf_groups.items()):
        require(set(pair) == {0, 1}, 'missing factual counterfactual')
        a, b = pair[0], pair[1]
        require(a['latest_payload'] != b['latest_payload'] and a['query'] == b['query'], 'counterfactual truth/query contract changed')
        require([e for e in a['events'] if e['version'] != a['latest_revision']] ==
                [e for e in b['events'] if e['version'] != b['latest_revision']], 'lower-revision counterfactual facts changed')
        both_extracted = a['decoded']['extracted'] and b['decoded']['extracted']
        responsive = both_extracted and a['decoded']['code_lexeme'] != b['decoded']['code_lexeme']
        cf.append({'world_id': key[0], 'role': key[1], 'permutation': key[2],
            'row_ids': [a['id'], b['id']], 'predictions': [a['prediction'], b['prediction']],
            'truths': [a['latest_payload'], b['latest_payload']],
            'code_lexeme_responsive': responsive, 'both_extracted': both_extracted,
            'raw_text_changed': a['prediction'] != b['prediction'],
            'both_latest_lexemes_match': a['code_lexeme_payload_match'] and b['code_lexeme_payload_match'],
            'only_cf0_latest_lexeme_matches': a['code_lexeme_payload_match'] and not b['code_lexeme_payload_match'],
            'only_cf1_latest_lexeme_matches': b['code_lexeme_payload_match'] and not a['code_lexeme_payload_match'],
            'both_latest_lexemes_wrong': not a['code_lexeme_payload_match'] and not b['code_lexeme_payload_match']})
    for key, pair in sorted(role_groups.items()):
        require(set(pair) == {'near_control', 'distant_heldout'}, 'missing near/distant pair')
        a, b = pair['near_control'], pair['distant_heldout']
        require(a['events'] == b['events'] and a['query'] == b['query'], 'near/distant actual facts/query changed')
        role.append({'world_id': key[0], 'counterfactual_id': key[1], 'permutation': key[2],
            'row_ids': [a['id'], b['id']], 'predictions': [a['prediction'], b['prediction']],
            'code_lexeme_changed': a['decoded']['code_lexeme'] != b['decoded']['code_lexeme'],
            'raw_text_changed': a['prediction'] != b['prediction'],
            'both_latest_lexemes_match': a['code_lexeme_payload_match'] and b['code_lexeme_payload_match'],
            'near_only_latest_lexeme_matches': a['code_lexeme_payload_match'] and not b['code_lexeme_payload_match'],
            'distant_only_latest_lexeme_matches': b['code_lexeme_payload_match'] and not a['code_lexeme_payload_match'],
            'neither_latest_lexeme_matches': not a['code_lexeme_payload_match'] and not b['code_lexeme_payload_match']})
    require(len(cf) == len(role) == 192, 'registered update pairing changed')
    return cf, role


def missing_rows(rows, predictions, metadata):
    result = []
    pairs = defaultdict(dict)
    for row in rows:
        meta = metadata[row['id']]
        if meta['category'] != 'missing':
            continue
        events, query, evidence = actual_events(row, meta)
        matching = [e for e in events if e['subject'] == meta['query']['subject'] and e['field'] == 'code']
        require(len(matching) <= 1, 'ambiguous missing-evidence query')
        expected = matching[0]['value'] if matching else None
        require(json.loads(row['target']) == {'v': expected} and meta['answerable'] == bool(matching), 'answerability sidecar/oracle differs')
        prediction = predictions[row['id']]
        decoded = literal(prediction['prediction'])
        item = {'id': row['id'], 'category': 'missing', 'world_id': meta['world_id'],
            'counterfactual_id': meta['counterfactual_id'], 'role': meta['role'], 'query_subject': meta['query']['subject'],
            'answerable': bool(matching), 'query': query, 'evidence': evidence, 'events': events,
            'prediction': prediction['prediction'], 'decoded': decoded,
            'strict': prediction['strict_exact_match'], 'typed': prediction['typed_value_correct'],
            'type_exact_payload_match': type_exact_match(decoded, expected),
            'code_lexeme_payload_match': payload_match(decoded, expected) if expected is not None else decoded['extracted'] and decoded['value'] is None,
            'selected_visible_records': [e for e in events if payload_match(decoded, e['value'])],
            'matching_queried_subject_code': any(payload_match(decoded, e['value']) for e in matching),
            'same_subject_wrong_field_weight': any(e['subject'] == meta['query']['subject'] and e['field'] == 'weight' and payload_match(decoded, e['value']) for e in events),
            'wrong_subject_code': any(e['subject'] != meta['query']['subject'] and e['field'] == 'code' and payload_match(decoded, e['value']) for e in events),
            'literal_null': decoded['extracted'] and decoded['value'] is None,
        }
        pairs[(meta['world_id'], meta['role'])][meta['counterfactual_id']] = item
        result.append(item)
    cf = []
    for key, pair in sorted(pairs.items()):
        require(set(pair) == {0, 1}, 'missing answerability counterfactual')
        present, absent = pair[0], pair[1]
        require(present['answerable'] and not absent['answerable'], 'answerability pair order changed')
        present_event = next(e for e in present['events'] if e['subject'] == present['query_subject'] and e['field'] == 'code')
        donor = [e for e in absent['events'] if e['field'] == 'code' and e['value'] == present_event['value']]
        require(len(donor) == 1 and donor[0]['subject'] != absent['query_subject'], 'subject-swap donor ambiguous')
        absent['donor_subject'] = donor[0]['subject']
        absent['donor_code_value'] = donor[0]['value']
        absent['donor_code_match'] = payload_match(absent['decoded'], donor[0]['value'])
        absent['other_decoy_code_match'] = any(e['field'] == 'code' and e['subject'] not in (absent['query_subject'], donor[0]['subject']) and payload_match(absent['decoded'], e['value']) for e in absent['events'])
        present['donor_code_match'] = False
        present['other_decoy_code_match'] = any(e['field'] == 'code' and e['subject'] != present['query_subject'] and payload_match(present['decoded'], e['value']) for e in present['events'])
        cf.append({'world_id': key[0], 'role': key[1], 'row_ids': [present['id'], absent['id']],
            'predictions': [present['prediction'], absent['prediction']],
            'code_lexeme_unchanged': present['decoded']['code_lexeme'] == absent['decoded']['code_lexeme'],
            'present_code_selected_and_absent_donor_selected': present['matching_queried_subject_code'] and absent['donor_code_match'],
            'correct_diagnostic_presence_and_abstention': present['code_lexeme_payload_match'] and absent['literal_null']})
    for item in result:
        item['classification_matches'] = [name for name in ('matching_queried_subject_code','same_subject_wrong_field_weight',
            'wrong_subject_code','donor_code_match','other_decoy_code_match','literal_null') if item[name]]
        item['unobserved_literal'] = item['decoded']['extracted'] and not item['selected_visible_records'] and not item['literal_null']
    require(len(result) == 64 and len(cf) == 32, 'registered answerability count changed')
    return result, cf


def summarize_booleans(rows, keys):
    return {'pairs': len(rows), **{key: sum(row[key] for row in rows) for key in keys}}


def analyze(source_root=SOURCE_ROOT):
    source_root = Path(source_root).resolve()
    manifest_path = source_root/'corpus-v2/manifest.json'
    receipt_path = source_root/'proposal/native_run/anchor-evaluation/receipt.json'
    proposal_path = source_root/'proposal/proposal.json'
    require(sha(manifest_path) == FROZEN['corpus_sha256'] and sha(receipt_path) == FROZEN['anchor_receipt_sha256'], 'frozen raw evidence changed')
    manifest, receipt, proposal = read(manifest_path), read(receipt_path), read(proposal_path)
    require(proposal['proposal_sha256'] == FROZEN['proposal_sha256'] and receipt['binding'] == proposal['bindings']['anchor'], 'proposal/anchor binding changed')
    require(receipt['binding']['code_sha256'] == FROZEN['code_sha256'], 'frozen code binding changed')
    inputs = {'manifest': manifest_path, 'receipt': receipt_path, 'proposal': proposal_path}
    for name in ('dev','design'):
        entry = manifest['files'][name]
        path = (manifest_path.parent/entry['path']).resolve()
        require(path.is_relative_to(manifest_path.parent) and sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'frozen packet file changed')
        inputs[name] = path
    rows = [json.loads(line) for line in inputs['dev'].read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata = read(inputs['design'])['dev']
    require(len(rows) == len(receipt['predictions']) == 672, 'native panel count changed')
    predictions = {}
    original_counts = Counter()
    for row, prediction in zip(rows, receipt['predictions']):
        require(all(row[k] == prediction[k] for k in ('id','family_id','target')), 'native row identity differs')
        scores = verify_original_scores(prediction,row['target'])
        original_counts.update({k:int(v) for k,v in scores.items()})
        require(prediction['id'] not in predictions, 'duplicate native row ID')
        predictions[prediction['id']] = prediction
    updates = [update_row(row,predictions[row['id']],metadata[row['id']]) for row in rows if metadata[row['id']]['category']=='update']
    require(len(updates) == 384, 'update panel count changed')
    bundles = update_bundles(updates)
    update_cf, update_role = update_pairs(updates)
    absent_rows, missing_cf = missing_rows(rows,predictions,metadata)
    def missing_summary(group):
        return {**counts(group), **{key:sum(row[key] for row in group) for key in (
            'matching_queried_subject_code','same_subject_wrong_field_weight','wrong_subject_code','donor_code_match',
            'other_decoy_code_match','literal_null','unobserved_literal')},
            'overlapping_classification_note':'A prediction can match more than one visible record if literal values coincide; matching lists are preserved.'}
    results = {
        'original_metrics_unchanged': {'rows':672, **dict(original_counts)},
        'update_summary': counts(updates),
        'update_by_latest_text_position': {str(position):counts([r for r in updates if r['latest_position']==position]) for position in range(3)},
        'update_by_latest_payload_numeric_rank': {str(rank):counts([r for r in updates if r['latest_payload_numeric_rank']==rank]) for rank in range(3)},
        'update_by_comparison_key_conflict': {str(conflict):counts([r for r in updates if r['comparison_key_conflict'] is conflict]) for conflict in (False,True)},
        'update_by_role': {role:counts([r for r in updates if r['role']==role]) for role in ('near_control','distant_heldout')},
        'update_by_world': {world:counts([r for r in updates if r['world_id']==world]) for world in sorted({r['world_id'] for r in updates})},
        'update_by_distant_gap_band': {','.join(map(str,band)):counts([r for r in updates if r['role']=='distant_heldout' and r['distance_band']==list(band)]) for band in sorted({tuple(r['distance_band']) for r in updates})},
        'six_order_bundle_summary': {'bundles':64,'dependent_worlds':16, **{key:sum(r[key] for r in bundles) for key in (
            'original_strict_all_six_correct','original_typed_all_six_correct','diagnostic_type_exact_all_six_payload_matches',
            'diagnostic_code_lexeme_all_six_payload_matches','raw_prediction_consistent','decoded_type_sensitive_value_consistent',
            'code_lexeme_consistent','code_lexeme_all_six_greatest_code_matches')}},
        'update_counterfactual_summary': summarize_booleans(update_cf,('code_lexeme_responsive','both_extracted','raw_text_changed',
            'both_latest_lexemes_match','only_cf0_latest_lexeme_matches','only_cf1_latest_lexeme_matches','both_latest_lexemes_wrong')),
        'update_near_distant_summary': summarize_booleans(update_role,('code_lexeme_changed','raw_text_changed','both_latest_lexemes_match',
            'near_only_latest_lexeme_matches','distant_only_latest_lexeme_matches','neither_latest_lexeme_matches')),
        'missing_present_summary':missing_summary([r for r in absent_rows if r['answerable']]),
        'missing_absent_summary':missing_summary([r for r in absent_rows if not r['answerable']]),
        'missing_counterfactual_summary':summarize_booleans(missing_cf,('code_lexeme_unchanged','present_code_selected_and_absent_donor_selected','correct_diagnostic_presence_and_abstention')),
        'update_rows':updates,'six_order_bundles':bundles,'update_counterfactual_pairs':update_cf,
        'update_near_distant_pairs':update_role,'missing_rows':absent_rows,'missing_counterfactual_pairs':missing_cf,
        'exact_examples': {
            'greatest_code_instead_of_revision': sorted([r for r in updates if r['selection_class']=='greatest_code_only'],key=lambda r:r['id'])[:8],
            'revision_numeral_instead_of_payload': sorted([r for r in updates if r['selection_class']=='visible_revision_numeral'],key=lambda r:r['id'])[:8],
            'order_sensitive_bundles': [r for r in bundles if not r['code_lexeme_consistent']][:8],
            'absent_donor_code_copy': sorted([r for r in absent_rows if not r['answerable'] and r['donor_code_match']],key=lambda r:r['id'])[:8],
            'absent_same_subject_wrong_field': sorted([r for r in absent_rows if not r['answerable'] and r['same_subject_wrong_field_weight']],key=lambda r:r['id'])[:8]},
    }
    return {
        'schema':'RINEYA-ANCHOR-OFFLINE-PATTERNS/v1','status':'DESCRIPTIVE_SAVED_NATIVE_DEV_DIAGNOSTICS_ONLY',
        'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_bindings':FROZEN,
        'input_files':{key:{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for key,path in inputs.items()},
        'analyzer_sha256':sha(__file__),
        'fixed_extractor':'finite duplicate-free JSON scalar or exact {v: scalar}; no salvage',
        'fixed_code_lexeme_rule':'str -> literal content; int/float/bool/null -> canonical compact JSON text; exact character equality against each visible quoted code',
        'correctness_boundary':'Lexeme matches are type-erasing diagnostics, not strict/typed correctness, semantic scoring, retained skill or quality gains.',
        'results_sha256':digest(results),'results':results,
        'limitations':['One frozen source and synthetic DEV; 16 dependent update worlds and 16 dependent answerability worlds.',
            'Counterfactual, ordering and position variants share worlds; no confidence intervals or independent-row statistical claim.',
            'The terse query renderer and payload/revision number distributions may affect the observed behaviors.',
            'No intervention, pretrained model call, GPU call, bank read, corpus change or proposal change occurred.',
            'Control/candidate adaptation failed to complete in the prior pilot; ordering gains remain untested.']}


def self_check():
    require(not literal('{"v":1,"v":1}')['extracted'],'duplicate keys must fail')
    require(not literal('{"v":NaN}')['extracted'],'nonfinite value must fail')
    require(not literal('```json\n{"v":"123"}\n```')['extracted'],'fenced salvage prohibited')
    require(not literal('{"v":"123","extra":0}')['extracted'],'extra-key salvage prohibited')
    require(payload_match(literal('123'),'123') and not type_exact_match(literal('123'),'123'),'numeric lexeme must remain wrong type')
    require(not payload_match(literal('123'),'0123'),'leading zero repair prohibited')
    require(not payload_match(literal('123.0'),'123'),'numeric tolerance prohibited')
    require(not payload_match(literal('"ABC"'),'abc'),'case repair prohibited')
    require(type_exact_match(literal('null'),None),'literal null parse changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',default=str(SOURCE_ROOT))
    parser.add_argument('--output',default=str(OUTPUT_ROOT/'anchor-patterns.json'))
    args=parser.parse_args()
    self_check()
    output=Path(args.output).resolve()
    require(output.is_relative_to(OUTPUT_ROOT.resolve()) and output!=OUTPUT_ROOT.resolve(),'output must remain in the new external pattern-research root')
    require(output.parent.is_dir() and not output.exists(),'exclusive new output required')
    result=analyze(args.source_root)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'output':str(output),'results_sha256':result['results_sha256'],
        'update':result['results']['update_summary'],'bundles':result['results']['six_order_bundle_summary'],
        'missing_present':result['results']['missing_present_summary'],'missing_absent':result['results']['missing_absent_summary']},sort_keys=True))


if __name__=='__main__':
    main()
