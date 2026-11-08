"""Read-only historical task-selector controls; no model/native execution.

Independently parse complete raw DEV prompts in both old template forms. Preserve
registered strict scores; use one fixed lexical whole-output decoder separately.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PACKAGE = Path(r'C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1/experiments/rineya_spikingbrain_hybrid_memory_v1')
CORPUS = PACKAGE / 'corpora/distant_evidence_v2'
RUN = PACKAGE / 'runs/titan_transition_v2_20261004_memory_efficient'
OUT = Path(__file__).resolve().parent / 'selector_controls.json'
FAMILIES = ['lookup', 'identifier', 'number', 'update', 'join']
BASIC = ['lookup', 'identifier', 'number']
BINDING = {'code_sha256': '326d805bde96047ca0553f8c33e5124a6c346fc11caaf3c28b4221277ef82140', 'config_sha256': '30b85a8bd3e24043bbbd2983d735eaf64df47e60626dc051792b45f3918d2400', 'source_sha256': 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a', 'corpus_sha256': '6f9c858b39c6c45d4fd5cd6bfa29d79b3c2f12773af7f392b44cfa171379105f'}
FILES, FAILURES = {}, []
CHECKS = 0


def sha(path):
    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    FILES[str(path.resolve())] = {'sha256': h, 'bytes': path.stat().st_size}
    return h


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(condition, name):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(name)


GENERIC_RECORD = [re.compile(r'The (?P<field>[a-z_]+) of (?P<subject>[A-Z]{5}) is "(?P<value>[^"\n]*)"\.'), re.compile(r'(?P<subject>[A-Z]{5}) \| (?P<field>[a-z_]+) \| "(?P<value>[^"\n]*)"\.')]
UPDATE_RECORD = [re.compile(r'Revision (?P<version>[0-9]+) \| (?P<subject>[A-Z]{5}) \| code \| "(?P<value>[^"\n]*)"\.'), re.compile(r'code of (?P<subject>[A-Z]{5}) was "(?P<value>[^"\n]*)" in release (?P<version>[0-9]+)\.')]
GENERIC_QUERY = [re.compile(r'Request: Copy the exact (?P<field>[a-z_]+) assigned to (?P<subject>[A-Z]{5}): \{"v":"\.\.\."\}\.'), re.compile(r'Request: Look up (?P<field>[a-z_]+) for (?P<subject>[A-Z]{5})\. Answer \{"v":"\.\.\."\}\.')]
UPDATE_QUERY = re.compile(r'Request: (?P<subject>[A-Z]{5}): copy code at the greatest numbered revision into \{"v":"\.\.\."\}\.')
JOIN_QUERY = [re.compile(r'Request: Find (?P<subject>[A-Z]{5})\x27s route then destination code\. Answer \{"v":"\.\.\."\}\.'), re.compile(r'Request: Use route then destination starting at (?P<subject>[A-Z]{5}); return its code as \{"v":"\.\.\."\}\.')]


def parse_raw(row, category):
    events, query_matches, forms = [], [], []
    patterns = UPDATE_RECORD if category == 'update' else GENERIC_RECORD
    for line in row['prompt'].splitlines():
        for i, pattern in enumerate(patterns):
            match = pattern.fullmatch(line)
            if match:
                event = match.groupdict()
                if category == 'update':
                    event['version'], event['field'] = int(event['version']), 'code'
                events.append(event)
                forms.append(i)
        queries = [UPDATE_QUERY] if category == 'update' else JOIN_QUERY if category == 'join' else GENERIC_QUERY
        for pattern in queries:
            match = pattern.fullmatch(line)
            if match:
                query = match.groupdict()
                if category == 'update':
                    query['field'] = 'code'
                if category == 'join':
                    query['join_chain'] = ['route', 'destination']
                query_matches.append(query)
    expected = 3 if category == 'update' else 4 if category == 'join' else 2
    if len(events) != expected or len(query_matches) != 1 or len(set(forms)) != 1:
        raise ValueError('complete raw parser contract failed: ' + row['id'])
    return events, query_matches[0], forms[0]


def resolve(events, query, category):
    def unique(subject, field):
        matches = [(i, e) for i, e in enumerate(events) if e['subject'] == subject and e['field'] == field]
        if len(matches) != 1:
            raise ValueError('expected unique entity/field record')
        return matches[0]
    if category == 'update':
        eligible = [(i, e) for i, e in enumerate(events) if e['subject'] == query['subject'] and e['field'] == query['field']]
        if len(eligible) != 3 or len({e['version'] for _, e in eligible}) != 3:
            raise ValueError('expected three distinct revisions for one entity/field')
        i, event = max(eligible, key=lambda p: p[1]['version'])
        return event['value'], i, [i], len(eligible)
    if category == 'join':
        subject, required = query['subject'], []
        for field in ['route', 'destination', 'code']:
            i, event = unique(subject, field)
            required.append(i)
            subject = event['value']
        return event['value'], i, required, 1
    i, event = unique(query['subject'], query['field'])
    return event['value'], i, [i], 1


class JSONNumber(str):
    """Keep a JSON numeric token's exact lexical spelling, such as48.00."""


ATOM = re.compile(r'[A-Za-z0-9_+-][A-Za-z0-9_.+-]*')


def decode(raw):
    """Whole JSON {'v': scalar}, JSON scalar, or one bare unquoted atom only.

    Strings preserve exact case/spelling. JSON numeric lexical spelling is kept
    exactly:48.00 !=48.0. No substring extraction, prose/fences, punctuation
    stripping, extra keys, numeric rounding, casefold or whitespace within atoms.
    """
    text = raw.strip()
    try:
        value = json.loads(text, parse_int=JSONNumber, parse_float=JSONNumber,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-JSON numeric constant')))
    except (ValueError, TypeError):
        if ATOM.fullmatch(text) and text not in ['NaN', 'Infinity', '-Infinity']:
            return {'decoded': True, 'value': text, 'form': 'bare_atom'}
        return {'decoded': False, 'value': None, 'form': 'rejected_nonliteral_output'}
    form = 'JSON_scalar'
    if isinstance(value, dict):
        if set(value) != {'v'}:
            return {'decoded': False, 'value': None, 'form': 'rejected_extra_or_missing_object_keys'}
        value, form = value['v'], 'JSON_v_object'
    if isinstance(value, JSONNumber):
        return {'decoded': True, 'value': str(value), 'form': form + '_numeric_lexeme'}
    if type(value) is str:
        return {'decoded': True, 'value': value, 'form': form + '_string'}
    if value is None:
        return {'decoded': True, 'value': None, 'form': form + '_null'}
    return {'decoded': False, 'value': None, 'form': 'rejected_nonstring_or_numeric_scalar'}


def summarize(rows, arm):
    nonfirst = [r for r in rows if r['true_value_record_position_one_based'] != 1]
    first = [r for r in rows if r['true_value_record_position_one_based'] == 1]
    def scores(rr):
        return {'rows': len(rr), 'strict_correct': sum(r['outputs'][arm]['strict_correct'] for r in rr), 'literal_correct': sum(r['outputs'][arm]['literal_correct'] for r in rr), 'literal_decodable': sum(r['outputs'][arm]['literal']['decoded'] for r in rr), 'first_physical_record_agreement': sum(r['outputs'][arm]['literal']['decoded'] and r['outputs'][arm]['literal']['value'] == r['first_physical_record_value'] for r in rr), 'first_target_field_record_agreement': sum(r['outputs'][arm]['literal']['decoded'] and r['outputs'][arm]['literal']['value'] == r['first_target_field_record_value'] for r in rr)}
    return {'all': scores(rows), 'true_value_record_first': scores(first), 'true_value_record_not_first': scores(nonfirst)}


def main():
    manifest = read(CORPUS / 'manifest.json')
    design = read(CORPUS / 'design.json')['rows']
    config = read(PACKAGE / 'configs/titan_transition.v2.memory_efficient.json')
    check(digest(config) == BINDING['config_sha256'] and sha(CORPUS / 'manifest.json') == BINDING['corpus_sha256'], 'historical config/corpus binding unchanged')
    check(sha(CORPUS / 'design.json') == manifest['design']['sha256'], 'historical design lock')
    code = {str(p.relative_to(PACKAGE)): sha(p) for p in sorted((PACKAGE / 'sbhm').glob('*.py'))}
    check(digest(code) == BINDING['code_sha256'], 'historical native code binding unchanged')
    dev_path = CORPUS / manifest['splits']['dev']['path']
    check(manifest['splits']['dev']['protected'] is False and sha(dev_path) == manifest['splits']['dev']['sha256'], 'explicitly unprotected locked DEV input')
    raw = [json.loads(line) for line in dev_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    evaluations = {arm: read(RUN / (arm + '-evaluation/receipt.json')) for arm in 'ABCD'}
    predictions = {arm: {p['id']: p for p in r['predictions']} for arm, r in evaluations.items()}
    for arm, r in evaluations.items():
        check(r['status'] == 'DEV_MEASUREMENT_ONLY' and all(r['binding'][k] == v for k, v in BINDING.items()) and r['binding']['arm'] == arm, arm + ': exact old evaluation binding')
        check([p['id'] for p in r['predictions']] == [row['id'] for row in raw], arm + ': full original DEV order')
        if arm == 'A':
            check(r['adaptation_record'] is None and r['binding']['delta_sha256'] is None, 'A is unchanged source')
        else:
            recovered = read(RUN / (arm + '-recover/receipt.json'))
            check(r['adaptation_record'] == recovered and all(recovered['binding'][k] == v for k, v in BINDING.items()), arm + ': recovered adaptation record bound')
    for raw_fixture, expected in [('48.00', '48.00'), ('{"v":48.00}', '48.00'), ('CtU-5469-Y', 'CtU-5469-Y'), ('"-30.98"', '-30.98')]:
        check(decode(raw_fixture)['value'] == expected, 'fixed lexical decoder fixture ' + raw_fixture)
    for rejected in ['{"v":"111","x":1}', 'Answer is111', '"...-93.49".']:
        check(not decode(rejected)['decoded'], 'fixed decoder rejects nonliteral/extras ' + rejected)
    rows = []
    for row in raw:
        meta = design[row['id']]
        category = meta['category']
        if category not in FAMILIES:
            continue
        events, query, form = parse_raw(row, category)
        gold, target_index, required, key_count = resolve(events, query, category)
        check(events == meta['events'] and all(query[k] == meta['query'][k] for k in query), row['id'] + ': raw parser equals saved events/query')
        expected_form = meta['template_index'] - 2 if category == 'update' else 3 - meta['template_index'] if category == 'join' else meta['template_index'] - 2
        check(form == expected_form, row['id'] + ': raw template form matches registered template')
        check(row['target'] == json.dumps({'v': gold}, separators=(',', ':')) and meta['row_sha256'] == digest(row), row['id'] + ': independently replayed target and row identity')
        field = 'code' if category == 'join' else query['field']
        first_field = next(e['value'] for e in events if e['field'] == field)
        outputs = {}
        for arm in 'ABCD':
            p = predictions[arm][row['id']]
            strict = p['prediction'] == row['target']
            check(p['target'] == row['target'] and p['family_id'] == row['family_id'] and strict == p['strict_exact_match'], arm + ':' + row['id'] + ': paired target and original strict flag')
            literal = decode(p['prediction'])
            outputs[arm] = {'raw': p['prediction'], 'strict_correct': strict, 'literal': literal, 'literal_correct': literal['decoded'] and literal['value'] == gold}
        rows.append({'id': row['id'], 'category': category, 'world_id': meta['world_id'], 'role': meta['role'], 'counterfactual_id': meta['counterfactual_id'], 'template_index': meta['template_index'], 'events': events, 'query': query, 'target': row['target'], 'gold_literal': gold, 'true_value_record_position_one_based': target_index + 1, 'required_record_positions_one_based': [i + 1 for i in required], 'queried_entity_field_candidate_records': key_count, 'first_physical_record_value': events[0]['value'], 'first_target_field_record_value': first_field, 'evidence_to_query_gap_tokens': meta['token_geometry']['evidence_to_query_gap_tokens'], 'outputs': outputs})
    check(len(rows) == 320 and Counter(r['category'] for r in rows) == Counter({c: 64 for c in FAMILIES}), 'all five task families64 DEV rows each')
    by_family = {}
    for family in FAMILIES:
        rr = [r for r in rows if r['category'] == family]
        by_family[family] = {'rows': len(rr), 'worlds': len({r['world_id'] for r in rr}), 'true_value_record_position_histogram': dict(sorted(Counter(r['true_value_record_position_one_based'] for r in rr).items())), 'queried_entity_field_candidate_record_histogram': dict(Counter(r['queried_entity_field_candidate_records'] for r in rr)), 'arms': {a: {'all_positions': summarize(rr, a), 'by_position': {role: summarize([r for r in rr if r['role'] == role], a) for role in ['distant_heldout', 'near_control']}, 'by_template': {str(t): summarize([r for r in rr if r['template_index'] == t], a) for t in [2, 3]}} for a in 'ABCD'}}
    paired = defaultdict(dict)
    for row in rows:
        paired[(row['category'], row['world_id'], row['role'])][row['counterfactual_id']] = row
    counterfactual = {c: {a: {'all_positions': [], 'distant_heldout': [], 'near_control': []} for a in 'ABCD'} for c in FAMILIES}
    for (family, world, role), variants in sorted(paired.items()):
        check(set(variants) == {0, 1}, world + ':' + role + ': complete factual counterfactual pair')
        a, b = variants[0], variants[1]
        changed = [i + 1 for i, (x, y) in enumerate(zip(a['events'], b['events'])) if x['value'] != y['value']]
        check(a['query'] == b['query'] and a['gold_literal'] != b['gold_literal'] and changed == [a['true_value_record_position_one_based']], world + ':' + role + ': only true requested-value event changes')
        for arm in 'ABCD':
            x, y = a['outputs'][arm], b['outputs'][arm]
            detail = {'world_id': world, 'role': role, 'true_record_position': a['true_value_record_position_one_based'], 'target_cf0': a['gold_literal'], 'target_cf1': b['gold_literal'], 'literal_cf0': x['literal']['value'], 'literal_cf1': y['literal']['value'], 'both_decodable': x['literal']['decoded'] and y['literal']['decoded'], 'literal_changed': x['literal']['decoded'] and y['literal']['decoded'] and x['literal']['value'] != y['literal']['value'], 'raw_changed': x['raw'] != y['raw'], 'both_strict_correct': x['strict_correct'] and y['strict_correct'], 'both_literal_correct': x['literal_correct'] and y['literal_correct']}
            counterfactual[family][arm]['all_positions'].append(detail)
            counterfactual[family][arm][role].append(detail)
    for family, arms in counterfactual.items():
        for arm, roles in arms.items():
            for role, pairs in list(roles.items()):
                roles[role] = {'pairs': len(pairs), 'all_targets_changed': True, 'both_decodable_pairs': sum(p['both_decodable'] for p in pairs), 'literal_prediction_changed_pairs': sum(p['literal_changed'] for p in pairs), 'raw_prediction_changed_pairs': sum(p['raw_changed'] for p in pairs), 'both_strict_correct_pairs': sum(p['both_strict_correct'] for p in pairs), 'both_literal_correct_pairs': sum(p['both_literal_correct'] for p in pairs), 'nonfirst_true_record_pairs': sum(p['true_record_position'] != 1 for p in pairs), 'both_literal_correct_nonfirst_pairs': sum(p['true_record_position'] != 1 and p['both_literal_correct'] for p in pairs), 'pair_details': pairs}
    examples = []
    for family in BASIC + ['join']:
        for template in [2, 3]:
            candidates = [r for r in rows if r['category'] == family and r['template_index'] == template and r['role'] == 'distant_heldout' and r['true_value_record_position_one_based'] != 1 and r['outputs']['D']['strict_correct']]
            if candidates:
                examples.append(max(candidates, key=lambda r: (r['evidence_to_query_gap_tokens'], r['id'])))
    update_failures = []
    for template in [2, 3]:
        candidates = [r for r in rows if r['category'] == 'update' and r['template_index'] == template and r['role'] == 'distant_heldout' and r['true_value_record_position_one_based'] != 1 and all(not r['outputs'][a]['literal_correct'] and r['outputs'][a]['literal']['value'] == r['first_physical_record_value'] for a in 'CD')]
        update_failures.append(max(candidates, key=lambda r: (r['evidence_to_query_gap_tokens'], r['id'])))
    groups = {'unique_entity_field_selection': [r for r in rows if r['category'] in BASIC], 'same_entity_revision_selection': [r for r in rows if r['category'] == 'update'], 'route_destination_code_chain': [r for r in rows if r['category'] == 'join']}
    result = {'schema': 'Rineya-quality/historical-selector-controls-v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_OFFLINE_TASK_SPECIFIC_SELECTOR_CONTROLS' if not FAILURES else 'FAIL_OFFLINE_TASK_SPECIFIC_SELECTOR_CONTROLS', 'checks_run': CHECKS, 'issues': FAILURES, 'script_sha256': sha(__file__), 'old_native_binding': BINDING, 'fixed_literal_decoder_contract': decode.__doc__, 'registered_strict_scores_unchanged': True, 'parsed_DEV_templates': [2, 3], 'task_families': by_family, 'task_structure_groups': {g: {a: summarize(rr, a) for a in 'ABCD'} for g, rr in groups.items()}, 'counterfactual_response': counterfactual, 'correct_nonfirst_distant_case_counterexamples_to_universal_first_record_rule': examples, 'distant_nonfirst_revision_selector_failure_cases': update_failures, 'all_row_details': rows, 'interpretation': ['The observed first-record collapse is task-specific. After adaptation, many unique queried-entity/field answers and joined endpoint codes are copied correctly from non-first records; update rows with several revisions for the same entity/field fail when the greatest revision is not first.', 'Identifier selection is an especially strong finite control: B/C/D answer all64 rows correctly, including all36 cases whose queried shipment record is second, across both template forms and both evidence positions. This falsifies a universal first-record output rule on these saved outputs.', 'C/D answer zero of26 distant update rows whose greatest-revision record is not first, while supplying first-record values on every distant update row. The generic entity-binding controls can work at distant positions, so the saved pattern does not justify a universal loss of attention or distant evidence access.', 'Join success on non-first final-code records shows some multi-hop selection outputs are correct. It is not proof of an exclusive latent algorithm or an exclusive recurrent route.', 'Counterfactual response requires changing to the new correct literal, not merely producing different text. Raw formatting and literal correctness remain separate; dependent position/counterfactual variants share worlds.', 'This is localization/falsification within the already completed old matched matrix. No new source quality gain, broad retention, confirmation or cross-corpus causal claim follows. Other27 sites retain full attention.', 'The actionable hypothesis concerns selecting the greatest revision among multiple same-entity/field candidates. A future separately permitted order-only control should preserve successful unique-entity selection and correct field/endpoint binding while moving the winning revision record among slots; include revision-versus-payload conflict cases. No native retry or corpus/run proposal is created here.'], 'limitations': ['Only saved old synthetic DEV outputs; one source/site/seed and16 dependent worlds per family.', 'Task families differ in field/value formats and semantic difficulty; this is a task-specific falsification, not a randomized causal isolation of selector complexity.', 'Literal decoder keeps lexical spelling and accepts one bare identifier atom; it is a descriptive output audit, not replacement of registered strict/normalized metrics.', 'No TRAIN model predictions or stage-specific behavioral trajectories are available.', 'No native model, Torch, GPU, sealed/confirmation bank, source/corpus/proposal/policy/metric change. Current offline evidence does not establish an actual new quality gain.'], 'native_execution': False, 'model_or_GPU_execution': False, 'new_quality_gain_claim': False, 'source_corpus_proposal_or_metrics_changed': False, 'input_and_artifact_sha256': FILES}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'checks_run': CHECKS, 'issues': FAILURES, 'output': str(OUT), 'all_positions_strict': {c: {a: by_family[c]['arms'][a]['all_positions']['all']['strict_correct'] for a in 'ABCD'} for c in FAMILIES}, 'all_positions_literal': {c: {a: by_family[c]['arms'][a]['all_positions']['all']['literal_correct'] for a in 'ABCD'} for c in FAMILIES}, 'D_nonfirst': {c: by_family[c]['arms']['D']['all_positions']['true_value_record_not_first'] for c in FAMILIES}}, indent=2))
    return 0 if not FAILURES else 1


if __name__ == '__main__':
    raise SystemExit(main())
