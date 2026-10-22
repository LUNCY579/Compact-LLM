"""Read-only actual v3 data/source review; no package, teacher, or model import.

Usage: python -B this_script.py DATA_DIRECTORY [EXISTING_RECEIPT_FILE ...]
Reads only explicit open TRAIN/PILOT and metadata/test receipts. Does not generate
any corpus or run fixtures. Creates two exclusive external review artifacts.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('C:/Users/clclc/BusinessProject')
OUT = ROOT / 'Titan-Quality-Patterns-v1-runs'
BASE = ROOT / 'Titan-Joint-Quality-Pilot-v2-runs/data-001'
V2 = ROOT / 'Titan-Joint-Quality-Pilot-v2'
V3 = ROOT / 'Titan-Joint-Quality-Pilot-v3'
SOURCE_PINS = {
    'PROTOCOL.json': '594b6d22bdda4682a4802eba16cf87df35fe84528a45227d620163d62cc4a474',
    'README.md': '1b2026d231bf211f627f595b6c37d81f3dd7dcc611445653f1d59b072e3e34fc',
    'tests/test_data.py': '5df15ee9e27342bb5fb3c5ab94750dd3afb0d74200ba9c8d677ae5cf1c585597',
    'titan_joint_quality_v1/common.py': 'af33709b46ae99676ee615c5228bb5211769a398ef5c06a843507b69ed69a408',
    'titan_joint_quality_v1/data.py': '0f493036927066c4902c2939021dcae75576c615673fc56b3b76bda146fc7b2b',
    'tests/test_selected_authority.py': '25487f82a86e714cd0b0fd1bbaf841fdf8c132511e9615b99a1c77c388f95128',
}
BASE_PINS = {
    'train.jsonl': '6c82a98a4d66c64d15aa813dc1afa3f20898996c00f2d8590158a8eec8e1b11d',
    'pilot.prompts.jsonl': '1ba852d8b0bb78d42c543a876e24a38987284095ff17f3ec9887bd7882d46bf8',
    'pilot.references.jsonl': '51b3b96d8e557ddcf046efbb7a8afc81d1b49bbadd1289fb41023a7936aced2d',
}
BASE_SOURCE = 'dd4a39762f805b4d2a99d9c7416cbf454c2d99244ca2f42e4e73faa17d4dba1b'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def encoded(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
def digest(x): return hashlib.sha256(encoded(x)).hexdigest()
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def pin(p): return {'path': str(p), 'sha256': sha(p)}


def parse(row):
    messages = row['messages']
    assert [a['role'] for a in messages] == ['system', 'user']
    prefix, sep, payload = messages[1]['content'].partition('TASK_JSON\n')
    payload, ending, trailing = payload.partition('\nEND_TASK')
    assert sep and ending and not trailing
    return json.loads(payload), prefix


def truth(task):
    q = task['query']
    values = {a['value'] for a in task['facts'] if a['entity'] == q['entity'] and a['attribute'] == q['attribute'] and a['source'] == q['authoritative_source']}
    return (next(iter(values)) if len(values) == 1 else 'UNKNOWN'), len(values)


def projected(task, source_blind):
    t = deepcopy(task)
    if source_blind:
        t.pop('sources', None)
        t['query'].pop('authoritative_source')
        for fact in t['facts']: fact.pop('source')
    else: t.pop('query')
    return t


def format_answer(answer, prefix):
    p = prefix.strip()
    if p == 'Return the answer alone with no surrounding text.': return answer
    if p == 'Return exactly <answer>VALUE</answer> with no surrounding text.': return '<answer>' + answer + '</answer>'
    if p == 'Return exactly one JSON object with one string field named "answer".': return json.dumps({'answer': answer}, separators=(',', ':'))
    raise AssertionError(p)


def main():
    data = Path(sys.argv[1]).resolve()
    assert data.parent == (ROOT / 'Titan-Joint-Quality-Pilot-v3-runs').resolve(), 'Explicit existing v3 open epoch required'
    report_path = OUT / 'selected-authority-v3-independent-001.json'
    md_path = OUT / 'selected-authority-v3-independent-001.md'
    assert not report_path.exists() and not md_path.exists(), 'Exclusive reports already exist'
    for name, expected in SOURCE_PINS.items(): assert sha(V3 / name) == expected
    for name, expected in BASE_PINS.items(): assert sha(BASE / name) == expected
    assert sha(V2 / 'titan_joint_quality_v1/data.py') == BASE_SOURCE
    inventory = lambda base: {p.relative_to(base).as_posix(): sha(p) for p in base.rglob('*') if p.is_file() and p.suffix in {'.py', '.md', '.json'}}
    old_src, new_src = inventory(V2), inventory(V3)
    changed = sorted(k for k in old_src.keys() & new_src.keys() if old_src[k] != new_src[k])
    same = sorted(k for k in old_src.keys() & new_src.keys() if old_src[k] == new_src[k])
    new = sorted(new_src.keys() - old_src.keys())
    assert len(old_src) == 20 and len(new_src) == 21 and len(same) == 15
    assert set(changed) == set(SOURCE_PINS) - {'tests/test_selected_authority.py'} and new == ['tests/test_selected_authority.py']
    assert not old_src.keys() - new_src.keys()
    base_train, train = rows(BASE / 'train.jsonl'), rows(data / 'train.jsonl')
    assert len(train) == len(base_train) == 2000
    manifest = json.loads((data / 'DATA_MANIFEST.json').read_text(encoding='utf-8'))
    artifact_pins = {name: pin(data / name) for name in BASE_PINS}
    assert {n: p['sha256'] for n, p in artifact_pins.items()} == manifest['artifact_sha256']
    assert manifest['instrument_revision'] == 3 and manifest['seed'] == 17
    assert manifest['input_epoch']['input_sha256'] == BASE_PINS and manifest['input_epoch']['base_generator_source_sha256'] == BASE_SOURCE
    assert manifest['status'] == 'DRAFT_UNREVIEWED_NOT_FROZEN' and not manifest['frozen']
    assert manifest['native_model_calls'] == 0 and not manifest['native_authorized'] and not manifest['training_authorized'] and not manifest['confirmation_materialized'] and not manifest['confirmation_authorized']
    preservation = Counter()
    base_by_parent = defaultdict(list)
    for old in base_train: base_by_parent[old['parent_id']].append(old)
    groups = defaultdict(list)
    counts, known_sources, unknown_sources, unknown_kinds = Counter(), Counter(), Counter(), Counter()
    shapes, value_shapes = Counter(), Counter()
    examples = {}
    base_digest = digest(base_train)
    base_order = [r['case_id'] for r in base_train]
    assert [r['case_id'] for r in train] == base_order
    for line, (old, row) in enumerate(zip(base_train, train), 1):
        if old['domain'] != 'grounding':
            assert old == row
            preservation['non_grounding_rows_exact'] += 1
            continue
        preservation['grounding_replaced'] += old != row
        assert row['case_id'] == old['case_id'] and row['parent_id'] == old['parent_id']
        assert row['family_id'] == 'train_query_selected_authority_v3' and row['pair_kind'] == 'changed_answer_selected_authority_switch'
        assert row['requested_format'] == old['requested_format'] and row['variant_id'] == old['variant_id']
        task, prefix = parse(row)
        assert task['kind'] == 'authority_query_source_v3' and set(task['query']) == {'entity', 'attribute', 'authoritative_source'}
        assert 'query.authoritative_source' in row['messages'][0]['content'] and 'two or more distinct' in row['messages'][0]['content']
        answer, nvalues = truth(task)
        assert answer == row['canonical_answer'] and format_answer(answer, prefix) == row['target']
        counts.update(rows=1, independently_derived_truth_matches=1, target_wrapper_matches=1)
        selected = task['query']['authoritative_source']
        (unknown_sources if answer == 'UNKNOWN' else known_sources)[selected] += 1
        q = task['query']
        matched = [f for f in task['facts'] if (f['entity'], f['attribute']) == (q['entity'], q['attribute'])]
        values = {f['value'] for f in matched if f['value'].startswith('v')}
        blind = next(iter(values)) if len(values) == 1 else 'UNKNOWN'
        counts['old_sourceblind_prefix_control_correct'] += blind == answer
        for f in task['facts']:
            value_shapes[(f['source'], f['value'][:1], len(f['value']))] += 1
            assert f['value'].startswith('v') and len(f['value']) == 7
        trace = row['grounding_repair_trace']
        assert trace['base_input_digest'] == base_digest and trace['base_source_sha256'] == BASE_SOURCE and trace['repair_seed'] == 17 and trace['fresh_draw_creation_verified'] is False
        original_task, _ = parse(base_by_parent[row['parent_id']][0])
        assert trace['base_world_digest'] == digest({k: v for k, v in original_task.items() if k != 'query'})
        expected_unknown_kind = 'missing' if nvalues == 0 else 'conflict' if nvalues >= 2 else None
        if answer == 'UNKNOWN': assert trace['unknown_kind'] == expected_unknown_kind
        else: assert trace['known_source'] == selected
        assert trace['known_source'] != trace['unknown_source']
        groups[row['parent_id']].append((line, row, task, prefix))
        for key in ('case_id', 'parent_id', 'family_id', 'canonical_answer', 'target', 'solver_trace', 'grounding_repair_trace'):
            assert key not in task
        category = (trace['unknown_kind'], 'UNKNOWN' if answer == 'UNKNOWN' else 'known', selected)
        examples.setdefault('|'.join(category), {'case_id': row['case_id'], 'train_line': line, 'query': q, 'matching_public_facts': matched, 'independent_truth': answer, 'selected_source_distinct_values': nvalues, 'gold_matches': True})
    for parent, pair in groups.items():
        assert len(pair) == 2
        (_, a, ta, pa), (_, b, tb, pb) = pair
        assert a['canonical_answer'] != b['canonical_answer'] and 'UNKNOWN' in {a['canonical_answer'], b['canonical_answer']}
        assert pa == pb and a['messages'][0] == b['messages'][0]
        assert (ta['query']['entity'], ta['query']['attribute']) == (tb['query']['entity'], tb['query']['attribute'])
        assert {ta['query']['authoritative_source'], tb['query']['authoritative_source']} == {'charter', 'rumor'}
        assert projected(ta, False) == projected(tb, False)
        assert projected(ta, True) == projected(tb, True)
        counts.update(parents=1, queryblind_identical_distinct_truth_pairs=1, sourceblind_identical_distinct_truth_pairs=1)
        unknown_kinds[a['grounding_repair_trace']['unknown_kind']] += 1
        shapes[a['grounding_repair_trace']['unknown_kind'] + ':' + a['grounding_repair_trace']['known_source']] += 1
    assert dict(known_sources) == dict(unknown_sources) == {'charter': 75, 'rumor': 75}
    assert counts['parents'] == 150 and counts['rows'] == 300 and counts['old_sourceblind_prefix_control_correct'] == 150
    assert unknown_kinds == {'missing': 75, 'conflict': 75}
    for name in ('pilot.prompts.jsonl', 'pilot.references.jsonl'):
        assert (data / name).read_bytes() == (BASE / name).read_bytes()
        preservation[name + '_byte_identical'] = True
    # Verify existing final offline receipts without retokenization/fixture calls.
    run = data.parent
    final_path = run / 'FINAL_MECHANICAL_VERIFICATION_002.json'
    assert sha(final_path) == 'ec2caed6e59cbf05f0ad09d0b90ea55ae92affdced0a6229894a700b47d0377a'
    final = json.loads(final_path.read_text(encoding='utf-8'))
    assert final['final_source_files_sha256'] == new_src
    for n, expected in new_src.items(): assert sha(Path(final['final_source_archive']) / n) == expected
    for p, expected in final['receipt_sha256'].items(): assert sha(Path(p)) == expected
    source_path = run / 'source-002.json'
    plan_path = run / 'exposure-002/EXPOSURE_PLAN.json'
    token_path = run / 'exposure-002/TOKEN_COUNT_CONTRACT.json'
    handoff_path = run / 'handoff-002/OWNER_HANDOFF.json'
    test_path = run / 'pytest-affected-002.log'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    token = json.loads(token_path.read_text(encoding='utf-8'))
    handoff = json.loads(handoff_path.read_text(encoding='utf-8'))
    assert source['status'] == 'PASS_METADATA_ONLY' and source['errors'] == [] and not source['native_qualified'] and source['model_loads'] == source['optimizer_steps'] == 0
    assert plan['source_bindings']['data_sha256'] == sha(data / 'DATA_MANIFEST.json')
    assert plan['source_bindings']['source_receipt_sha256'] == sha(source_path)
    assert plan['common_source_sha256'] == new_src['titan_joint_quality_v1/common.py'] and plan['exposure_source_sha256'] == new_src['titan_joint_quality_v1/exposure.py']
    assert token['data_binding'] == plan['source_bindings'] and not token['native_qualified'] and token['native_model_calls'] == 0 and not token['trained_model_quality_evidence']
    assert handoff['data_manifest_sha256'] == sha(data / 'DATA_MANIFEST.json') and handoff['source_receipt_sha256'] == sha(source_path) and handoff['exposure_plan_sha256'] == sha(plan_path)
    assert {n.replace('\\', '/'): h for n, h in handoff['source_closure'].items()} == {n: h for n, h in new_src.items() if n.endswith('.py')}
    assert not handoff['quality_improvement_established'] and handoff['native_model_calls'] == 0 and handoff['executable_native_command'] is None and handoff['owner_messages_by_this_implementation'] == 0
    assert not any(handoff['contract'][key] for key in ('native_authorized', 'training_authorized', 'confirmation_authorized'))
    assert '67 passed in 12.37s' in test_path.read_text(encoding='utf-8')
    assert not plan['native_authorized'] and not plan['training_authorized'] and not plan['source_objective_qualified'] and not plan['capability_evidence'] and not plan['truncation'] and not plan['padding']
    encoded_rows = [row for group in plan['groups'] for row in group['rows']]
    actual_by_case = {row['case_id']: row for row in train}
    selected_parent_counts = Counter()
    total_tokens, total_labels, reasoning_labels, max_row = 0, 0, 0, 0
    for row in encoded_rows:
        assert row['source_record'] == actual_by_case[row['case_id']]
        assert row['row_sha256'] == digest(row['source_record'])
        assert len(row['ids']) == len(row['label_mask']) == row['processed_tokens'] <= 1024
        assert row['eos_position'] == row['response_end'] - 1 and row['ids'][row['eos_position']] == row['eos_token_id'] == 151645
        assert row['label_mask'] == [row['response_start'] <= i < row['response_end'] for i in range(len(row['ids']))]
        assert sum(row['label_mask']) == row['response_eos_tokens']
        selected_parent_counts[row['parent_id']] += 1
        total_tokens += row['processed_tokens']; total_labels += row['response_eos_tokens']
        reasoning_labels += row['response_eos_tokens'] if row['role'] == 'reasoning' else 0
        max_row = max(max_row, row['processed_tokens'])
    assert len(encoded_rows) == plan['selected_rows'] == 200 and len(selected_parent_counts) == plan['selected_parent_count'] == 100 and set(selected_parent_counts.values()) == {2}
    assert total_tokens == plan['processed_tokens_per_arm'] == 53913 <= 54172 and total_labels == plan['joint_label_tokens'] == 1653 and reasoning_labels == plan['reasoning_only_label_tokens'] == 785
    assert [row['case_id'] for row in encoded_rows] == plan['selected_case_ids']
    for group in plan['groups']:
        assert sum(row['processed_tokens'] for row in group['rows']) == group['processed_tokens'] <= 2048
        assert sum(row['response_eos_tokens'] for row in group['rows']) == group['joint_label_tokens'] == group['planned_denominator']
        assert sum(row['response_eos_tokens'] for row in group['rows'] if row['role'] == 'reasoning') == group['reasoning_label_tokens'] == group['reasoning_only_label_tokens'] > 0
    assert len(plan['groups']) == plan['updates'] == 28 and max_row == 626
    assert plan['planned_denominators'] == [group['planned_denominator'] for group in plan['groups']] and sum(plan['planned_denominators']) == 1653
    for role, target in {'reasoning': .5, 'preservation': .25, 'grounding': .15, 'format': .1}.items():
        positions = sum(row['processed_tokens'] for row in encoded_rows if row['role'] == role)
        assert positions == plan['processed_token_counts_by_role'][role] and positions / total_tokens == plan['processed_token_shares_by_role'][role]
        assert abs(positions / total_tokens - target) <= .05
    assert plan['same_sequences_order_groups'] and plan['same_reasoning_targets'] and plan['control_extra_reasoning_rows'] == 0
    assert token['count_contract']['qualification_forward_cap'] is None and token['count_contract']['qualification_backward_cap'] is None and final['qualification_counts_and_complete_native_forecast'] is None
    for name in BASE_PINS: assert (data / name).read_bytes() == (run / 'data-001' / name).read_bytes()
    qualification = {'source_status': source['status'], 'affected_tests': '67 passed in 12.37s (root log; not rerun)', 'final_source_archive_matches_all21files': True, 'root_receipt_hashes_match': True, 'handoff_source_closure_matches_all19Python_files': True, 'final_data_bytes_identical_to_initial_repaired_epoch': True, 'selected_rows': len(encoded_rows), 'selected_parents': len(selected_parent_counts), 'updates': len(plan['groups']), 'processed_tokens_per_arm': total_tokens, 'max_row_tokens': max_row, 'max_group_tokens': max(group['processed_tokens'] for group in plan['groups']), 'joint_common_labels': total_labels, 'reasoning_only_labels': reasoning_labels, 'processed_role_shares': plan['processed_token_shares_by_role'], 'every_update_reasoning_labels': True, 'whole_parent_selection': True, 'row_response_EOS_masks': True, 'native_qualification': False, 'qualification_counts': None, 'complete_native_forecast': None}
    # Receipts are separately pinned evidence, not native permission or data admission.
    receipt_paths = [final_path, source_path, plan_path, token_path, handoff_path, test_path] + [Path(p).resolve() for p in sys.argv[2:]]
    receipts = {str(p): pin(p) for p in receipt_paths}
    contexts = {str(p): pin(p) for p in [Path(__file__).resolve(), data / 'DATA_MANIFEST.json', OUT / 'semantic-admission-independent-001.json', OUT / 'semantic-admission-independent-001-bindings-addendum.json']}
    report = {
        'schema': 'TITAN_SELECTED_AUTHORITY_V3_INDEPENDENT_REVIEW_V1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'OFFLINE_SOURCE_AND_ACTUAL_G_MECHANICS_CONFIRMED_UNADMITTED',
        'review_identity': {'compiler_author': False, 'fresh_agent_review': True, 'prior_reports_and_builder_source_read': True, 'blind_review': False, 'independent_human_review': False, 'package_or_teacher_calls': 0},
        'scope': 'Static successor source review plus independent public-input derivation over actual300TRAIN G rows; exact non-G/PILOT preservation. Receipt verification remains offline metadata/token/exposure evidence.',
        'source_bindings': {'v2_root': str(V2), 'v2_relative_sha256': old_src, 'v3_root': str(V3), 'v3_relative_sha256': new_src, 'changed_original_files': changed, 'unchanged_original_files': same, 'new_files': new},
        'actual_input_bindings': artifact_pins, 'base_input_bindings': {n: pin(BASE / n) for n in BASE_PINS}, 'context_bindings': contexts, 'receipt_bindings': receipts,
        'preservation': dict(preservation), 'independent_public_checks': dict(counts), 'qualification_receipt_review': qualification, 'known_source_counts': dict(known_sources), 'UNKNOWN_source_counts': dict(unknown_sources), 'UNKNOWN_kind_parent_counts': dict(unknown_kinds), 'UNKNOWN_kind_known_source_parent_counts': dict(shapes), 'fact_value_shape_occurrences': {str(k): v for k, v in value_shapes.items()}, 'selected_public_examples': list(examples.values()),
        'stateless_pair_bounds': {'queryblind_accuracy_upper_bound': 0.5, 'sourceblind_accuracy_upper_bound': 0.5, 'reason': 'Each actualpair has identical respective projected public inputs and wrapper/system, but distinct verified truths. A deterministic stateless policy produces one answer per identical projection, hence matches at most one member.'},
        'static_review': [
            {'finding': 'Explicit public selected-source rule matches A distinct-value set and B conflict-detecting scan. Repeated equalvalues remain one distinctvalue; missing/conflicting selected-source evidence isUNKNOWN; other sources cannot supply answers.', 'location': 'data.py:61,174-178,270-279'},
            {'finding': 'Sourceblind projection first requires full exact canonical rendered messages, then removes selected-source and fact-source fields. It is an abstract erasure certificate, not a new legitimate sourceful evaluationprompt. Queryblind projection removes the entire query. Both actualpair equalities verified independently.', 'location': 'data.py:674-684,721-726'},
            {'finding': 'Review found P2: initial projection discarded extra public messages/text before equalinput certification. Corrected source rejects noncanonical full messages before erasure. Four existingepoch rowmutation regressions cover extraassistant, extrauser, trailing suffix, and customsystem hints. Superseded source/data qualification is kept separate from this finalsource epoch.', 'location': 'data.py:677-679; tests/test_selected_authority.py:test_source_blind_projection_rejects_different_full_public_hints_on_existing_row'},
            {'finding': 'Generation binds exact existing open v2 seed17 bytes and compiler pin; hashing/parsing uses the same read-once inputbuffer. Other public generationseeds refuse. Isolatedrepair RNG depends on suppliedbaseworld andrepairseed.', 'location': 'data.py:567-580,583-613,664-671'},
            {'finding': 'The puretransform records inputdigest and suppliedbasepin but does not authenticate futurefreshdrawcreation. fresh_draw_creation_verifiedfalse and independentdrawintegrationPENDING correctly keep that future qualification open.', 'location': 'data.py:616-624,657-658; PROTOCOL.json data.input_epoch'},
            {'finding': 'Native/confirmation CLI dispatch remains closed before outputcreation; confirmation split refuses before baseinputreads. common.py includes v1/v2/v3 source-root outputrefusal. Runtime/tokenizer/optimizer/analysis files are unchanged.', 'location': '__main__.py:144-145; data.py:509-511,664-668; common.py:10,51-59'},
        ],
        'limits': ['The G300 actualtruth/projection checks are independent standard-library derivations, not blind/human admission or trainedmodel evidence.', 'PILOT G remains Horn, including the49/64copy and56/64one-round floors. Preserving it creates no held-out selected-authority evaluation or source-sensitive generalization result.', 'Common value alphabet/prefix removes the discovered prefixproxy; the actualsourceblind paircertificate is stronger than relying on a distribution claim. It assumes deterministicstateless policies without metadata/history leakage.', 'Puretransform suppliedprovenance is a future integration seam. Freshdraw creation, rights/ancestry/overlap, usablebaseline denominators, exactnativebridge/qualification/cost and action-specificauthority remain open.', 'Existing token/exposure receipts must bind thissource/data epoch; priorv1/v2/native receipts cannotqualify it. No qualitywinner or instrumentfreeze is declared.'],
        'boundaries': {'native_model_calls': 0, 'training_calls': 0, 'protected_or_confirmation_reads': 0, 'new_bank_generated_by_reviewer': False, 'fixture_tests_run_by_reviewer': False, 'source_or_existing_data_modified': False, 'owner_messages': 0, 'permits_changed': False, 'serving_changes': False, 'native_admission_frozen': False, 'scientific_model_quality_proved': False},
    }
    md = '''# Independent selected-authority v3 review 001

The exact successor source and actual 300 TRAIN grounding rows pass the bounded source-sensitivity review. Public instructions explicitly select query.authoritative_source; both teachers implement one distinct selected-source value, with missing or conflicting evidence returning UNKNOWN. Independent public-input derivation matches all 300 actual gold answers and wrappers.

All 150 actual parents have the same world, entity/attribute, system instruction and requested wrapper, with selected source crossed and distinct known/UNKNOWN truths. Independent query-blind and source-blind projections are identical within every pair. Each deterministic stateless blind policy therefore has a paired accuracy bound of 50%. The old value-prefix control now scores 150/300. There are 75 missing and 75 conflict parents; known and UNKNOWN source counts are 75 charter and 75 rumor each. Both sources use seven-character v-prefix values.

The 1,700 non-grounding TRAIN rows are exact object matches in the same order; both PILOT files are byte-identical to the pinned v2 epoch. Source inventory has 20 old files, five changed originals, fifteen byte-identical originals, and one new test. Runtime, tokenizer, optimizer, analysis, native and confirmation dispatch implementations are unchanged. Public generation binds seed17 only; confirmation refuses before input reads and output guards include all three source epochs.

The review found that the initial source-blind projection could discard extra public text and certify unequal policy inputs. The corrected source requires full exact canonical messages before erasure; four existing-row mutation regressions cover extra assistant/user messages, trailing text, and altered system instructions. This review pins the corrected source and fresh qualification epoch; archived initial-source receipts remain separate.

Final receipts bind the corrected source/data epoch, and the root's affected log records 67 passes. Independent receipt arithmetic verifies 200 rows/100 complete parents, 28 updates, 53,913 processed tokens per arm, row maximum 626, group maximum 2,048, common labels 1,653 and reasoning labels 785. Every group has reasoning labels and all role shares meet the five-point tolerance. Response-plus-EOS masks and exact selected source records agree. These are offline stored-receipt checks; no retokenization, fixture rerun, or native execution was performed.

The pure transformation records the supplied base digest/source pin and explicitly leaves fresh-draw creation unverified. The current generator creates no independent replication draws. Preserved PILOT grounding remains Horn, with the prior 49/64 copied-fact and 56/64 one-round floors; it supplies no held-out selected-authority result.

This is compiler-independent agent review using only standard-library calculations over existing open data. It is neither blind nor human admission and establishes no model acquisition, quality, native qualification, permit, or serving result. The adjacent JSON pins all source files, exact inputs, script and provided receipts; prior v1/v2 receipts cannot qualify this successor. Native admission and remaining provenance, baseline, runtime/cost and action-specific authority gates remain open.
'''
    assert inventory(V3) == new_src and inventory(V2) == old_src
    assert all(sha(data / name) == p['sha256'] for name, p in artifact_pins.items())
    with report_path.open('x', encoding='utf-8', newline='\n') as f: json.dump(report, f, indent=2); f.write('\n')
    with md_path.open('x', encoding='utf-8', newline='\n') as f: f.write(md)
    print(json.dumps({'report': pin(report_path), 'markdown': pin(md_path), 'script': pin(Path(__file__)), 'checks': dict(counts), 'preservation': dict(preservation)}))


if __name__ == '__main__': main()
