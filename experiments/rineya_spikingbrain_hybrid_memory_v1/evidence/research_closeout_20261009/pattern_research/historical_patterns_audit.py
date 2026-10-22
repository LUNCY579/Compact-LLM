"""Offline research of existing SBHM update-task prediction/order patterns.

Standard library only. Read the already-admitted historical TRAIN/DEV corpus,
metadata and completed October 4 matrix records. No model or new corpus is
executed or changed. Literal value decoding is a descriptive secondary audit;
registered strict exact-match scores remain untouched.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

PACKAGE = Path(r'C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1/experiments/rineya_spikingbrain_hybrid_memory_v1')
RUN = PACKAGE / 'runs/titan_transition_v2_20261004_memory_efficient'
CORPUS = PACKAGE / 'corpora/distant_evidence_v2'
OUTROOT = Path(r'C:/Users/clclc/BusinessProject/rineya-quality-records/update-order-v1/20261005-pattern-research-01a10d69')
OUT = OUTROOT / 'historical_patterns_receipt.json'
OLD_AUDIT = Path(r'C:/Users/clclc/BusinessProject/sbhm-001-audits/20261005-closeout-01a10d69/provenance_receipt_v2.json')
EXPECTED_BINDING = {
    'code_sha256': '326d805bde96047ca0553f8c33e5124a6c346fc11caaf3c28b4221277ef82140',
    'config_sha256': '30b85a8bd3e24043bbbd2983d735eaf64df47e60626dc051792b45f3918d2400',
    'source_sha256': 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a',
    'corpus_sha256': '6f9c858b39c6c45d4fd5cd6bfa29d79b3c2f12773af7f392b44cfa171379105f',
}
FILES, FAILURES = {}, []
CHECKS = 0


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    value = h.hexdigest()
    FILES[str(path.resolve())] = {'sha256': value, 'bytes': path.stat().st_size}
    return value


def read(path):
    path = Path(path)
    if any(word in path.name.upper() for word in ['CONFIRM', 'SEALED']):
        raise ValueError('protected-named contents must not be parsed')
    sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(condition, name):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(name)


# Independent parser: four complete anchored historical record grammars.
PATTERNS = [
    re.compile(r'(?P<subject>[A-Z]{5}): code="(?P<value>[0-9]+)"; version=(?P<version>[0-9]+)\.'),
    re.compile(r'At revision (?P<version>[0-9]+), (?P<subject>[A-Z]{5}) recorded code "(?P<value>[0-9]+)"\.'),
    re.compile(r'Revision (?P<version>[0-9]+) \| (?P<subject>[A-Z]{5}) \| code \| "(?P<value>[0-9]+)"\.'),
    re.compile(r'code of (?P<subject>[A-Z]{5}) was "(?P<value>[0-9]+)" in release (?P<version>[0-9]+)\.'),
]
QUERY = re.compile(r'Request: (?P<subject>[A-Z]{5}): copy code at the greatest numbered revision into \{"v":"\.\.\."\}\.')
DECIMAL = re.compile(r'[0-9]+')


def parse_raw(row):
    events, template_ids = [], []
    query = None
    for line in row['prompt'].splitlines():
        for index, grammar in enumerate(PATTERNS):
            match = grammar.fullmatch(line)
            if match:
                values = match.groupdict()
                events.append({'subject': values['subject'], 'field': 'code', 'value': values['value'], 'version': int(values['version'])})
                template_ids.append(index)
        q = QUERY.fullmatch(line)
        if q:
            if query is not None:
                raise ValueError('duplicate update query')
            query = q['subject']
    if len(events) != 3 or query is None or any(e['subject'] != query for e in events):
        raise ValueError('update raw parser contract failed: ' + row['id'])
    if len({e['version'] for e in events}) != 3 or len({e['value'] for e in events}) != 3:
        raise ValueError('ambiguous revisions/code values: ' + row['id'])
    if len(set(template_ids)) != 1:
        raise ValueError('mixed historical update templates')
    return events, query, template_ids[0]


def literal_decode(raw):
    """Parse one complete JSON value: {'v': decimal string/int}, decimal string/int.

    No prose extraction, substring matching, fences, booleans, floats, null,
    additional keys or numeric revision/code rewriting. Decimal strings preserve
    exact text; integers convert by ordinary decimal str().
    """
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return {'value': None, 'status': 'UNPARSED_COMPLETE_JSON'}
    kind = 'bare'
    if isinstance(value, dict):
        if set(value) != {'v'}:
            return {'value': None, 'status': 'REJECTED_OBJECT_KEYS'}
        value, kind = value['v'], 'v_object'
    if type(value) is str and DECIMAL.fullmatch(value):
        return {'value': value, 'status': kind + '_decimal_string'}
    if type(value) is int and value >= 0:
        return {'value': str(value), 'status': kind + '_nonnegative_integer'}
    return {'value': None, 'status': 'REJECTED_NONDECIMAL_LITERAL'}


def rule_values(events):
    greatest = max(range(3), key=lambda i: events[i]['version'])
    largest_code = max(range(3), key=lambda i: int(events[i]['value']))
    return {'first_record': events[0]['value'], 'last_record': events[-1]['value'], 'largest_code': events[largest_code]['value'], 'largest_revision': events[greatest]['value']}, greatest, largest_code


def agreement(rows, arm):
    rule_counts = {rule: sum(r['model_outputs'][arm]['decoded']['value'] == value for r in rows for name, value in r['rule_values'].items() if name == rule) for rule in ['first_record', 'last_record', 'largest_code', 'largest_revision']}
    ranking = sorted([{'rule': name, 'matching_predictions': count, 'rows': len(rows), 'fraction': count / len(rows) if rows else None} for name, count in rule_counts.items()], key=lambda r: (-r['matching_predictions'], r['rule']))
    wrong = [r for r in rows if not r['model_outputs'][arm]['literal_oracle_correct']]
    first_conflicts = [r for r in rows if r['rule_values']['first_record'] != r['rule_values']['largest_revision']]
    return {
        'rows': len(rows),
        'strict_correct': sum(r['model_outputs'][arm]['strict_correct'] for r in rows),
        'literal_oracle_correct': sum(r['model_outputs'][arm]['literal_oracle_correct'] for r in rows),
        'decodable': sum(r['model_outputs'][arm]['decoded']['value'] is not None for r in rows),
        'shortcut_ranking': ranking,
        'all_first_record_exceptions': [r['id'] for r in rows if r['model_outputs'][arm]['decoded']['value'] != r['rule_values']['first_record']],
        'literal_error_rows': len(wrong),
        'errors_equal_first_record': sum(r['model_outputs'][arm]['decoded']['value'] == r['rule_values']['first_record'] for r in wrong),
        'errors_equal_last_record': sum(r['model_outputs'][arm]['decoded']['value'] == r['rule_values']['last_record'] for r in wrong),
        'errors_equal_largest_code': sum(r['model_outputs'][arm]['decoded']['value'] == r['rule_values']['largest_code'] for r in wrong),
        'errors_equal_a_revision_number': sum(r['model_outputs'][arm]['decoded']['value'] in [str(e['version']) for e in r['events']] for r in wrong),
        'first_vs_true_rule_discriminating_rows': len(first_conflicts),
        'first_chosen_on_discriminating_rows': sum(r['model_outputs'][arm]['decoded']['value'] == r['rule_values']['first_record'] for r in first_conflicts),
        'greatest_revision_chosen_on_discriminating_rows': sum(r['model_outputs'][arm]['decoded']['value'] == r['rule_values']['largest_revision'] for r in first_conflicts),
        'ambiguous_rule_agreements_are_not_unique_internal_mechanism_identification': True,
    }


def paired_shift(rows, candidate):
    def score(row, arm, metric):
        output = row['model_outputs'][arm]
        if metric == 'first_record_agreement':
            return output['decoded']['value'] == row['rule_values']['first_record']
        return output[metric]
    result = {}
    for metric in ['first_record_agreement', 'literal_oracle_correct', 'strict_correct']:
        cells = Counter((score(r, 'A', metric), score(r, candidate, metric)) for r in rows)
        source, changed = sum(score(r, 'A', metric) for r in rows), sum(score(r, candidate, metric) for r in rows)
        result[metric] = {'rows': len(rows), 'A_count': source, 'candidate_count': changed, 'net_change': changed - source, 'both': cells[(True, True)], 'A_only': cells[(True, False)], 'candidate_only': cells[(False, True)], 'neither': cells[(False, False)]}
    return result


def main():
    old_audit = read(OLD_AUDIT)
    check(old_audit['status'] == 'PASS_OFFLINE_HISTORICAL_NATIVE_ARTIFACT_AUDIT' and old_audit['binding'] == EXPECTED_BINDING, 'prior independent successful artifact audit identity')
    config_path = PACKAGE / 'configs/titan_transition.v2.memory_efficient.json'
    config = read(config_path)
    manifest = read(CORPUS / 'manifest.json')
    design = read(CORPUS / 'design.json')['rows']
    code = {str(p.relative_to(PACKAGE)): sha(p) for p in sorted((PACKAGE / 'sbhm').glob('*.py'))}
    check(digest(code) == EXPECTED_BINDING['code_sha256'], 'historical native code binding unchanged')
    check(digest(config) == EXPECTED_BINDING['config_sha256'], 'historical native config binding unchanged')
    check(sha(CORPUS / 'manifest.json') == EXPECTED_BINDING['corpus_sha256'], 'historical native corpus binding unchanged')
    check(sha(CORPUS / 'design.json') == manifest['design']['sha256'], 'historical design lock')
    corpora = {}
    raw_ids = {}
    for split in ['train', 'dev']:
        path = CORPUS / manifest['splits'][split]['path']
        check(manifest['splits'][split]['role'] == split and manifest['splits'][split]['protected'] is False and sha(path) == manifest['splits'][split]['sha256'], split + ': admitted unprotected split lock')
        corpora[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        raw_ids.update({r['id']: r for r in corpora[split]})
    evaluations = {a: read(RUN / (a + '-evaluation/receipt.json')) for a in 'ABCD'}
    predictions = {a: {p['id']: p for p in evaluations[a]['predictions']} for a in 'ABCD'}
    for arm, receipt in evaluations.items():
        check(all(receipt['binding'][k] == v for k, v in EXPECTED_BINDING.items()) and receipt['binding']['arm'] == arm and receipt['status'] == 'DEV_MEASUREMENT_ONLY', arm + ': authoritative historical DEV binding')
        check([p['id'] for p in receipt['predictions']] == [r['id'] for r in corpora['dev']], arm + ': full DEV order identical')
        if arm == 'A':
            check(receipt['adaptation_record'] is None and receipt['binding']['delta_sha256'] is None, 'A remains unadapted source control')
    rows = []
    for split in ['train', 'dev']:
        for raw in corpora[split]:
            metadata = design[raw['id']]
            if metadata['category'] != 'update':
                continue
            events, query, template = parse_raw(raw)
            check(events == metadata['events'] and query == metadata['query']['subject'] and template == metadata['template_index'], raw['id'] + ': independent parser agrees with saved event/query/template metadata')
            check(metadata['row_sha256'] == digest(raw), raw['id'] + ': exact raw row identity')
            rules, oracle_slot, code_slot = rule_values(events)
            target = json.dumps({'v': rules['largest_revision']}, separators=(',', ':'))
            check(target == raw['target'], raw['id'] + ': independent max-revision target replay')
            blocks = raw['prompt'].split('\n---\n')
            check(len(blocks) == 4 and blocks[3] == metadata['query_text'], raw['id'] + ': complete prompt block/query shape')
            expected_blocks = [metadata['evidence_block_text'], metadata['filler_block_text']] if metadata['role'] == 'distant_heldout' else [metadata['filler_block_text'], metadata['evidence_block_text']]
            check(blocks[1:3] == expected_blocks, raw['id'] + ': evidence/filler positional role')
            row = {'id': raw['id'], 'split': split, 'world_id': metadata['world_id'], 'role': metadata['role'], 'counterfactual_id': metadata['counterfactual_id'], 'template_index': template, 'events': events, 'target': target, 'rule_values': rules, 'oracle_record_position_one_based': oracle_slot + 1, 'largest_code_record_position_one_based': code_slot + 1, 'registered_evidence_gap_tokens': metadata['token_geometry']['evidence_to_query_gap_tokens'], 'model_outputs': {}}
            if split == 'dev':
                for arm in 'ABCD':
                    prediction = predictions[arm][raw['id']]
                    check(prediction['target'] == raw['target'] and prediction['family_id'] == raw['family_id'], arm + ':' + raw['id'] + ': saved row/target match')
                    strict = prediction['prediction'] == raw['target']
                    check(strict == prediction['strict_exact_match'], arm + ':' + raw['id'] + ': strict raw flag verified unchanged')
                    decoded = literal_decode(prediction['prediction'])
                    row['model_outputs'][arm] = {'raw_prediction': prediction['prediction'], 'decoded': decoded, 'strict_correct': strict, 'literal_oracle_correct': decoded['value'] == rules['largest_revision'], 'matching_rules': [rule for rule, value in rules.items() if decoded['value'] == value], 'matching_record_positions_one_based': [i + 1 for i, e in enumerate(events) if decoded['value'] == e['value']]}
            rows.append(row)
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'dev']
    check(len(train) == 32 and len(dev) == 64, 'all 32 TRAIN and 64 DEV update rows covered')
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['split'], r['world_id'])][(r['role'], r['counterfactual_id'])] = r
    for (split, world), variants in groups.items():
        check(set(variants) == {(role, cf) for role in ['distant_heldout', 'near_control'] for cf in [0, 1]}, world + ': four paired counterfactual/position variants')
        for cf in [0, 1]:
            far, near = variants[('distant_heldout', cf)], variants[('near_control', cf)]
            check(far['events'] == near['events'] and far['target'] == near['target'], world + ': evidence/target invariant near versus distant cf ' + str(cf))
        for role in ['distant_heldout', 'near_control']:
            a, b = variants[(role, 0)], variants[(role, 1)]
            changed_slots = [i + 1 for i, (x, y) in enumerate(zip(a['events'], b['events'])) if x['value'] != y['value']]
            check(a['oracle_record_position_one_based'] == b['oracle_record_position_one_based'] and changed_slots == [a['oracle_record_position_one_based']] and a['target'] != b['target'], world + ':' + role + ': counterfactual replaces only correct event value')
            check(all(x['subject'] == y['subject'] and x['version'] == y['version'] for x, y in zip(a['events'], b['events'])), world + ':' + role + ': counterfactual preserves subject/revisions/order')
    structural = {}
    for split, rr in [('train', train), ('dev', dev)]:
        structural[split] = {'rows': len(rr), 'worlds': len({r['world_id'] for r in rr}), 'oracle_position_rows': dict(sorted(Counter(r['oracle_record_position_one_based'] for r in rr).items())), 'oracle_position_worlds': dict(sorted(Counter(variants[('near_control', 0)]['oracle_record_position_one_based'] for (s, _), variants in groups.items() if s == split).items())), 'shortcut_oracle_correct': {rule: sum(r['rule_values'][rule] == r['rule_values']['largest_revision'] for r in rr) for rule in ['first_record', 'last_record', 'largest_code', 'largest_revision']}, 'model_predictions_available': split == 'dev', 'all_actual_rows_listed_in_receipt': True}
    summaries = {arm: {'all_update_DEV': agreement(dev, arm), 'by_position': {role: agreement([r for r in dev if r['role'] == role], arm) for role in ['distant_heldout', 'near_control']}, 'by_counterfactual': {str(cf): agreement([r for r in dev if r['counterfactual_id'] == cf], arm) for cf in [0, 1]}, 'by_position_and_counterfactual': {role + '-cf-' + str(cf): agreement([r for r in dev if r['role'] == role and r['counterfactual_id'] == cf], arm) for role in ['distant_heldout', 'near_control'] for cf in [0, 1]}, 'by_true_revision_record_position': {str(slot): agreement([r for r in dev if r['oracle_record_position_one_based'] == slot], arm) for slot in [1, 2, 3]}, 'by_template': {str(t): agreement([r for r in dev if r['template_index'] == t], arm) for t in [2, 3]}} for arm in 'ABCD'}
    counterfactual, position = {}, {}
    for arm in 'ABCD':
        counterfactual[arm] = {}
        for role in ['distant_heldout', 'near_control']:
            pair_rows = []
            for (split, world), variants in sorted(groups.items()):
                if split != 'dev':
                    continue
                a, b = variants[(role, 0)], variants[(role, 1)]
                x, y = a['model_outputs'][arm]['decoded']['value'], b['model_outputs'][arm]['decoded']['value']
                pair_rows.append({'world_id': world, 'true_revision_record_position': a['oracle_record_position_one_based'], 'target_cf0': a['rule_values']['largest_revision'], 'target_cf1': b['rule_values']['largest_revision'], 'decoded_prediction_cf0': x, 'decoded_prediction_cf1': y, 'prediction_changed': x != y, 'both_literal_correct': a['model_outputs'][arm]['literal_oracle_correct'] and b['model_outputs'][arm]['literal_oracle_correct'], 'both_first_record': x == a['rule_values']['first_record'] and y == b['rule_values']['first_record']})
            counterfactual[arm][role] = {'pairs': len(pair_rows), 'all_targets_changed': True, 'prediction_changed_pairs': sum(p['prediction_changed'] for p in pair_rows), 'prediction_unchanged_pairs': sum(not p['prediction_changed'] for p in pair_rows), 'both_literal_correct_pairs': sum(p['both_literal_correct'] for p in pair_rows), 'both_first_record_pairs': sum(p['both_first_record'] for p in pair_rows), 'pair_details': pair_rows}
        pair_rows = []
        for (split, world), variants in sorted(groups.items()):
            if split != 'dev':
                continue
            for cf in [0, 1]:
                far, near = variants[('distant_heldout', cf)], variants[('near_control', cf)]
                x, y = far['model_outputs'][arm]['decoded']['value'], near['model_outputs'][arm]['decoded']['value']
                pair_rows.append({'world_id': world, 'cf': cf, 'decoded_distant': x, 'decoded_near': y, 'literal_prediction_same': x == y, 'raw_prediction_same': far['model_outputs'][arm]['raw_prediction'] == near['model_outputs'][arm]['raw_prediction']})
        position[arm] = {'pairs': len(pair_rows), 'same_literal_pairs': sum(p['literal_prediction_same'] for p in pair_rows), 'same_raw_pairs': sum(p['raw_prediction_same'] for p in pair_rows), 'discordances': [p for p in pair_rows if not p['literal_prediction_same']]}
    shifts = {arm: {'all_update_DEV': paired_shift(dev, arm), 'by_position': {role: paired_shift([r for r in dev if r['role'] == role], arm) for role in ['distant_heldout', 'near_control']}} for arm in 'BCD'}
    training = {}
    for arm in 'BCD':
        stages = {}
        for stage in ['distill', 'recover']:
            name = arm + '-' + stage
            receipt = read(RUN / name / 'receipt.json')
            delta = read(RUN / name / 'delta.json')
            delta_sha = sha(RUN / name / 'delta.safetensors')
            check(all(receipt['binding'][k] == v for k, v in EXPECTED_BINDING.items()) and receipt['status'] == 'TRAINING_COMPLETED_NOT_EFFICACY', name + ': matched completed adaptation binding')
            check(delta['record'] == receipt and delta['sha256'] == delta_sha, name + ': delta hash/full training record')
            update_steps = [r for r in receipt['records'] if r['row_id'] in {x['id'] for x in train}]
            check(len(update_steps) == 64 and Counter(r['row_id'] for r in update_steps) == Counter({r['id']: 2 for r in train}), name + ': each historical TRAIN update row seen twice')
            stages[stage] = {'update_task_steps': len(update_steps), 'updates_total': len(receipt['records']), 'delta_sha256': delta_sha, 'training_predictions_saved': False, 'first_position_correct_update_exposures': sum(design[r['row_id']]['events'][0]['version'] == max(e['version'] for e in design[r['row_id']]['events']) for r in update_steps), 'training_step_trace_sha256': digest([(r['row_id'], r['input_tokens'], r['supervised_positions']) for r in receipt['records']])}
            if stage == 'distill' and arm == 'B':
                check(all(r['loss'] == 0 and r['gradient_norm'] == 0 for r in receipt['records']), 'B self-teacher distillation all loss/gradient zero')
        training[arm] = stages
        check(evaluations[arm]['adaptation_record'] == read(RUN / (arm + '-recover/receipt.json')) and evaluations[arm]['binding']['delta_sha256'] == stages['recover']['delta_sha256'], arm + ': observed DEV outputs bound to recovered adaptation')
    for stage in ['distill', 'recover']:
        check(len({training[a][stage]['training_step_trace_sha256'] for a in 'BCD'}) == 1, stage + ': matched historical control exposures')
    first_exceptions = {a: [{'id': r['id'], 'role': r['role'], 'cf': r['counterfactual_id'], 'events': r['events'], 'target': r['target'], 'rule_values': r['rule_values'], 'model_output': r['model_outputs'][a]} for r in dev if r['model_outputs'][a]['decoded']['value'] != r['rule_values']['first_record']] for a in 'ABCD'}
    literal_only = {a: [{'id': r['id'], 'target': r['target'], 'prediction': r['model_outputs'][a]['raw_prediction']} for r in dev if r['model_outputs'][a]['literal_oracle_correct'] and not r['model_outputs'][a]['strict_correct']] for a in 'ABCD'}
    result = {
        'schema': 'Rineya-quality/historical-update-order-patterns-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_OFFLINE_HISTORICAL_UPDATE_PATTERN_AUDIT' if not FAILURES else 'FAIL_OFFLINE_HISTORICAL_UPDATE_PATTERN_AUDIT',
        'checks_run': CHECKS, 'failures': FAILURES,
        'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__), 'python': sys.version,
        'native_execution': False, 'model_or_GPU_execution': False, 'training': False,
        'source_corpus_or_historical_receipts_changed': False, 'new_anchor_contents_read': False,
        'sealed_confirmation_or_protected_bank_access': False,
        'binding': EXPECTED_BINDING, 'prior_independent_source_delta_audit': str(OLD_AUDIT),
        'source_snapshot_digest_from_previously_verified_receipts': EXPECTED_BINDING['source_sha256'],
        'literal_decoder_contract': literal_decode.__doc__,
        'registered_strict_metrics_unchanged': True,
        'structural_shortcut_opportunity': structural,
        'model_shortcut_agreement': summaries,
        'counterfactual_pairs': counterfactual, 'near_distant_position_pairs': position,
        'source_A_to_adapted_arm_paired_changes': shifts,
        'training_exposure_and_delta_records': training,
        'all_first_record_counterexamples': first_exceptions,
        'literal_correct_but_strict_wrong': literal_only,
        'row_details': rows,
        'deterministic_bounds': {
            'scope': 'These finite historical rows only, no generalization estimate or confidence interval.',
            'TRAIN_first_record_rule_correct': {'rows': 24, 'denominator': 32, 'worlds': 6, 'world_denominator': 8},
            'DEV_first_record_rule_correct': {'rows': 12, 'denominator': 64, 'worlds': 3, 'world_denominator': 16},
            'DEV_distant_first_record_rule_correct': {'rows': 6, 'denominator': 32, 'worlds': 3, 'world_denominator': 16},
            'observed_C_and_D_distant_predictions_equal_first_record': {'rows': 32, 'denominator': 32, 'correct': 6, 'wrong': 26, 'counterexamples': 0},
            'observed_C_near_first_record_exceptions': 3,
            'observed_D_near_first_record_exceptions': 2,
            'no_unconditional_all_position_deterministic_shortcut_claim': True,
        },
        'interpretation': [
            'Recovered C and D exactly implement the observable first-record output rule on all 32 saved distant update rows, across both counterfactuals and templates. Near rows have three C and two D counterexamples, so this is not an unconditional deterministic all-position rule.',
            'The old TRAIN update targets are first-record-biased (24/32 rows, 6/8 worlds), not always first. Old DEV correct revisions are first only 12/64 rows (3/16 worlds). Positional shortcut reward differs across these registered splits.',
            'Original-architecture B also shifts strongly toward first-record outputs (59/64 overall), so added recurrence is unnecessary for this historical adaptation-associated pattern.',
            'Fixed whole-output literal decoding isolates formatting: source A has 18 literal-correct update rows versus 14 strict-correct; adapted B/C have 13 under both and D12. Aggregate matrix gains elsewhere do not imply improved revision selection.',
            'Within the matched old A/B/C/D matrix, exposure and delta bindings support a descriptive adaptation-associated output shift. No stage-specific behavioral outputs, TRAIN model predictions or order-balanced control are saved, so learning mechanism and which training phase caused the shift remain unisolated.',
            'Counterfactual replacements alter only the correct event value while preserving order/revisions. Distant C/D predictions therefore change in only the three worlds whose correct revision is first, and remain unchanged in the other thirteen target-changing worlds.',
            'Rule agreement can overlap: first-record, largest-code and largest-revision may select the same literal. Rank fits describe outputs and cannot reveal an exclusive internal algorithm.',
            'No comparison to the new anchor is made. Old template/geometry differences and a different corpus cannot support a cross-corpus causal gain/loss claim.',
        ],
        'minimal_actionable_intervention_not_an_execution_proposal': {
            'mechanistic_hypothesis': 'The old recovery distribution rewards a first-record shortcut more often than genuine revision comparison, and adaptation increases use of that shortcut on unseen templates.',
            'smallest_behavioral_control': 'Hold subject, three distinct code values, three distinct revisions, query, filler and oracle target constant; change only evidence record order so the same greatest-revision event moves from first to a later slot. Choose values so greatest code and greatest revision are different events. Compare first-record and true-rule literal predictions, preserving strict output metrics separately.',
            'minimum_interpretation': 'A changed output following the new first event while the oracle target stays fixed supports a positional-output dependence at those rows; it does not by itself show an internal learned algorithm or broad capability change.',
            'balanced_recovery_intervention_if_separately_authorized_later': 'Counterbalance correct-revision position and numeric code ranking within equal world/exposure budgets, with a matched original-order training control and fixed held-out permutation worlds. Exact record-order change is the isolated intervention; keep source, tokenizer, optimizer, precision, site, steps and total exposure fixed.',
            'prospective_analysis': 'Separate greatest-revision literal correctness, first/last/largest-code agreement, target-changing counterfactual response, source-correct damage and formatting; group dependent variants by world.',
            'current_decision': 'Offline evidence only. No native retry, new corpus, new grant, training, evaluation, source edit or run proposal is created by this audit. RAM guard remains a stopping condition.',
        },
        'limitations': [
            'One source, site, seed and old synthetic DEV corpus; exploratory pattern inspection, no confirmation evidence or promotion/gain claim.',
            'The 64 DEV update rows are 16 dependent worlds with four variants each, not 64 independent samples.',
            'Only saved predictions are analyzed; no native model reexecution, numerical delta payload interpretation or GPU workload.',
            'No TRAIN predictions are saved: TRAIN analysis concerns stimulus/target ordering and known update exposure only.',
            'Literal decoding is a newly explicit descriptive audit of old output strings and does not replace registered strict metrics.',
            'No cross-corpus causal inference with any new anchor; no automatic longer context, recurrence campaign or recovery training.',
        ],
        'input_inventory': FILES,
    }
    # Fail closed if a manually summarized deterministic bound ever drifts.
    check(structural['train']['shortcut_oracle_correct']['first_record'] == 24 and structural['dev']['shortcut_oracle_correct']['first_record'] == 12, 'reported TRAIN/DEV shortcut bounds')
    check(all(summaries[a]['by_position']['distant_heldout']['shortcut_ranking'][0]['rule'] == 'first_record' and summaries[a]['by_position']['distant_heldout']['shortcut_ranking'][0]['matching_predictions'] == 32 for a in 'CD'), 'reported C/D distant deterministic bound')
    check(len(first_exceptions['C']) == 3 and len(first_exceptions['D']) == 2, 'reported C/D near exceptions')
    result['checks_run'], result['failures'] = CHECKS, FAILURES
    result['status'] = 'PASS_OFFLINE_HISTORICAL_UPDATE_PATTERN_AUDIT' if not FAILURES else 'FAIL_OFFLINE_HISTORICAL_UPDATE_PATTERN_AUDIT'
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'checks_run': CHECKS, 'failures': FAILURES, 'output': str(OUT), 'TRAIN_oracle_positions': structural['train']['oracle_position_rows'], 'DEV_oracle_positions': structural['dev']['oracle_position_rows'], 'first_record_agreement': {a: next(r['matching_predictions'] for r in summaries[a]['all_update_DEV']['shortcut_ranking'] if r['rule'] == 'first_record') for a in 'ABCD'}, 'literal_correct': {a: summaries[a]['all_update_DEV']['literal_oracle_correct'] for a in 'ABCD'}, 'strict_correct': {a: summaries[a]['all_update_DEV']['strict_correct'] for a in 'ABCD'}}, indent=2))
    return 0 if not FAILURES else 1


if __name__ == '__main__':
    raise SystemExit(main())
