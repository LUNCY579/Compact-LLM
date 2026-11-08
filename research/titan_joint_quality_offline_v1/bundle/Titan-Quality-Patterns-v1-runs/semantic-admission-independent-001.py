"""Independent, bounded public-input review; standard library only.

Does not import compiler/teachers, run a model, materialize a bank, or edit inputs.
All authority and Horn rows are read only to reproduce the two reported controls.
Truth/wording checks outside authority cover selected existing parents, not a full
corpus admission. Reports and this script are exclusive new external artifacts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/clclc/BusinessProject')
OUT = ROOT / 'Titan-Quality-Patterns-v1-runs'
DATA = ROOT / 'Titan-Joint-Quality-Pilot-v2-runs/data-001'
SOURCE = ROOT / 'Titan-Joint-Quality-Pilot-v2'
PIN = 'dd4a39762f805b4d2a99d9c7416cbf454c2d99244ca2f42e4e73faa17d4dba1b'
EXPECTED = {
    'train.jsonl': '6c82a98a4d66c64d15aa813dc1afa3f20898996c00f2d8590158a8eec8e1b11d',
    'pilot.prompts.jsonl': '1ba852d8b0bb78d42c543a876e24a38987284095ff17f3ec9887bd7882d46bf8',
    'pilot.references.jsonl': '51b3b96d8e557ddcf046efbb7a8afc81d1b49bbadd1289fb41023a7936aced2d',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lines(p):
    return [(i, json.loads(s)) for i, s in enumerate(p.read_text(encoding='utf-8').splitlines(), 1)]


def public(row):
    users = [m['content'] for m in row['messages'] if m['role'] == 'user']
    assert len(users) == 1
    before, sep, after = users[0].partition('TASK_JSON\n')
    payload, end, trailing = after.partition('\nEND_TASK')
    assert sep and end and not trailing and 'TASK_JSON\n' not in payload
    return json.loads(payload), before.strip()


def unique(values):
    values = set(values)
    return next(iter(values)) if len(values) == 1 else 'UNKNOWN'


def selected_truth(t):
    """Explicit working interpretations of the selected public finite tasks."""
    k, q = t['kind'], t['query']
    if k == 'authority':
        matched = [f for f in t['facts'] if f['entity'] == q['entity'] and f['attribute'] == q['attribute']]
        return unique(f['value'] for f in matched if f['source'] == t['authoritative_source']), {'selected_source': t['authoritative_source'], 'matching_facts': matched}
    if k == 'program':
        registers = dict(t['initial'])
        trace = [dict(registers)]
        for step in t['instructions']:
            dst, arg, op = step['dst'], step['arg'], step['op']
            if op == 'add': registers[dst] = registers[dst] + arg
            elif op == 'multiply': registers[dst] = registers[dst] * arg
            elif op == 'add_from': registers[dst] = registers[dst] + registers[arg]
            elif op == 'if_even_add': registers[dst] += arg if registers[dst] % 2 == 0 else 0
            else: raise AssertionError(op)
            trace.append(dict(registers))
        return str(registers[q]), {'current_register_trace': trace}
    if k == 'item_events':
        relevant = [e['bin'] for e in t['events'] if e['item'] == q]
        return (relevant[-1] if relevant else t['initial'].get(q, 'UNKNOWN')), {'initial_bin': t['initial'].get(q), 'moves_in_order': relevant}
    if k == 'baskets':
        products = [a['price'] * a['quantity'] for a in t['baskets'][q]]
        return str(sum(products)), {'quantity_price_products': products}
    if k == 'latest_record':
        relevant = [a for a in t['records'] if a['entity'] == q['entity']]
        highest = max(a['timestamp'] for a in relevant)
        winners = [a for a in relevant if a['timestamp'] == highest]
        assert len(winners) == 1
        return str(winners[0][q['field']]), {'matching_records': relevant, 'selected_timestamp': highest}
    if k == 'field_transform':
        values = [t['fields'][x] for x in q['keys']]
        answer = q['separator'].join(x.upper() if q['case'] == 'upper' else x.lower() for x in values)
        return answer, {'ordered_field_values': values, 'case': q['case'], 'separator': q['separator']}
    if k == 'alias_join':
        code = t['aliases'].get(q['alias'])
        matches = [a for a in t['records'] if a['code'] == code]
        assert len(matches) == 1
        return str(matches[0][q['field']]), {'resolved_code': code, 'unique_record': matches[0]}
    if k == 'filtered_roster':
        matches = [a for a in t['records'] if a['tier'] == q['tier'] and a['score'] >= q['minimum']]
        ordered = sorted((a['label'] for a in matches), reverse=q['order'] == 'desc')
        return ','.join(ordered) if ordered else 'NONE', {'selected_records': matches, 'ordered_labels': ordered}
    if k == 'fsm':
        trace = [t['initial']]
        for symbol in t['symbols'][:q['after_symbols']]:
            trace.append(t['transitions'][trace[-1]][symbol])
        return trace[-1], {'symbol_count': q['after_symbols'], 'state_trace': trace}
    if k == 'sharing':
        a = t['groups'][q['group']]
        quotient = a['count'] // a['people']
        remainder = a['count'] - a['people'] * quotient
        answer = quotient if q['component'] == 'quotient' else remainder
        return str(answer), {'count': a['count'], 'people': a['people'], 'quotient': quotient, 'remainder': remainder}
    if k == 'dependencies':
        evidence = []
        def calc(name):
            if name in t['leaves']: return t['leaves'][name]
            if name not in t['expressions']: raise KeyError(name)
            left, op, right = t['expressions'][name]
            a, b = calc(left), calc(right)
            value = {'+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b}[op]()
            evidence.append({'name': name, 'operands': [a, b], 'operator': op, 'value': value})
            return value
        try: answer = str(calc(q))
        except KeyError: answer = 'UNKNOWN'
        return answer, {'queried_derivation': evidence}
    if k == 'horn':
        # Simultaneous closure rounds expose depth; no forward compiler call.
        reached, rounds = set(t['facts']), [sorted(t['facts'])]
        while True:
            added = {a['conclusion'] for a in t['rules'] if all(x in reached for x in a['premises'])} - reached
            if not added: break
            reached |= added
            rounds.append(sorted(added))
        return ('SUPPORTED' if q in reached else 'UNKNOWN'), {'new_atoms_by_round': rounds}
    raise AssertionError(k)


def wrap(answer, prefix):
    if prefix == 'Return the answer alone with no surrounding text.': return answer
    if prefix == 'Return exactly <answer>VALUE</answer> with no surrounding text.': return '<answer>' + answer + '</answer>'
    if prefix == 'Return exactly one JSON object with one string field named "answer".': return json.dumps({'answer': answer}, separators=(',', ':'))
    raise AssertionError(prefix)


def main():
    destinations = [OUT / 'semantic-admission-independent-001.json', OUT / 'semantic-admission-independent-001.md']
    assert not any(p.exists() for p in destinations), 'Exclusive outputs already exist'
    inputs = {name: {'path': str(DATA / name), 'sha256': sha(DATA / name)} for name in EXPECTED}
    assert all(inputs[n]['sha256'] == h for n, h in EXPECTED.items())
    assert sha(SOURCE / 'titan_joint_quality_v1/data.py') == PIN
    train = lines(DATA / 'train.jsonl')
    pilot = lines(DATA / 'pilot.prompts.jsonl')
    references = {r['case_id']: r for _, r in lines(DATA / 'pilot.references.jsonl')}
    authority = Counter()
    authority_sources = Counter()
    source_prefixes = Counter()
    authority_examples = []
    for line, row in train:
        t, prefix = public(row)
        if t['kind'] != 'authority': continue
        q = t['query']
        matched = [f for f in t['facts'] if f['entity'] == q['entity'] and f['attribute'] == q['attribute']]
        source_aware = unique(f['value'] for f in matched if f['source'] == t['authoritative_source'])
        blind = unique(f['value'] for f in matched if f['value'].startswith('v'))
        switched_truth = unique(f['value'] for f in matched if f['source'] == 'rumor')
        authority.update(rows=1, source_aware_gold_matches=source_aware == row['canonical_answer'], source_blind_gold_matches=blind == row['canonical_answer'], source_blind_unknown=blind == 'UNKNOWN', source_blind_switched_truth_matches=blind == switched_truth)
        authority_sources[t['authoritative_source']] += 1
        source_prefixes.update((f['source'] + ':' + f['value'][:1]) for f in t['facts'])
        if not authority_examples or (authority_examples[0]['actual_gold'] == 'UNKNOWN' and source_aware != 'UNKNOWN' and len(authority_examples) == 1):
            authority_examples.append({'case_id': row['case_id'], 'train_line': line, 'query': q, 'actual_selected_source': t['authoritative_source'], 'actual_gold': row['canonical_answer'], 'blind_prediction': blind, 'in_memory_switched_source': 'rumor', 'proper_switched_truth': switched_truth, 'matching_public_facts': matched, 'switched_gold_persisted': False})
    horn = Counter()
    horn_examples = {}
    for line, row in pilot:
        t, prefix = public(row)
        if t['kind'] != 'horn': continue
        facts = set(t['facts'])
        first_round = facts | {a['conclusion'] for a in t['rules'] if all(x in facts for x in a['premises'])}
        copy = 'SUPPORTED' if t['query'] in facts else 'UNKNOWN'
        shallow = 'SUPPORTED' if t['query'] in first_round else 'UNKNOWN'
        truth, evidence = selected_truth(t)
        gold = references[row['case_id']]['canonical_answer']
        horn.update(rows=1, full_closure_gold_matches=truth == gold, copy_correct=copy == gold, copy_false_abstention=copy == 'UNKNOWN' and gold == 'SUPPORTED', copy_unsupported=copy == 'SUPPORTED' and gold == 'UNKNOWN', one_round_correct=shallow == gold, one_round_false_abstention=shallow == 'UNKNOWN' and gold == 'SUPPORTED', one_round_unsupported=shallow == 'SUPPORTED' and gold == 'UNKNOWN')
        category = 'unsupported' if truth == 'UNKNOWN' else 'direct_fact' if copy == 'SUPPORTED' else 'one_round' if shallow == 'SUPPORTED' else 'two_or_more_rounds'
        if category not in horn_examples:
            horn_examples[category] = {'case_id': row['case_id'], 'pilot_line': line, 'query': t['query'], 'facts': t['facts'], 'rules': t['rules'], 'gold': gold, 'derived_truth': truth, **evidence}
    first_parent = {}
    examples = []
    for filename, rows in [('train.jsonl', train), ('pilot.prompts.jsonl', pilot)]:
        for line, row in rows:
            t, prefix = public(row)
            ref = row if filename == 'train.jsonl' else references[row['case_id']]
            kind, parent = t['kind'], ref['parent_id']
            first_parent.setdefault(kind, parent)
            if first_parent[kind] != parent: continue
            derived, evidence = selected_truth(t)
            examples.append({'case_id': row['case_id'], 'input_file': filename, 'line': line, 'kind': kind, 'query': t['query'], 'public_rule': row['messages'][0]['content'], 'gold': ref['canonical_answer'], 'derived_truth': derived, 'truth_match': derived == ref['canonical_answer'], 'visible_wrapper': prefix, 'target_matches_visible_wrapper': wrap(derived, prefix) == ref['target'], 'derivation': evidence})
    assert len(examples) == 24 and len(first_parent) == 12
    selected_mismatches = [e['case_id'] for e in examples if not e['truth_match'] or not e['target_matches_visible_wrapper']]
    assert authority['rows'] == 300 and authority['source_aware_gold_matches'] == 300
    assert authority['source_blind_gold_matches'] == 300 and authority['source_blind_switched_truth_matches'] == 0
    assert horn['rows'] == 64 and horn['full_closure_gold_matches'] == 64
    assert horn['copy_correct'] == 49 and horn['one_round_correct'] == 56
    assert not selected_mismatches
    bindings = {}
    for p in [SOURCE / 'README.md', SOURCE / 'PROTOCOL.json', SOURCE / 'titan_joint_quality_v1/data.py', ROOT / 'Titan-Quality-Patterns-v1/README.md', OUT / 'AUTHOR_SEMANTIC_CROSSCHECK_001.json', OUT / 'SOURCE_PROXY_AUDIT_001.json', OUT / 'ROOT_SEMANTIC_REPLAY_001.json', OUT / 'GROUNDING_REPAIR_PROPOSAL_001.json', Path(__file__)]:
        bindings[p.name] = {'path': str(p), 'sha256': sha(p)}
    report = {
        'schema': 'TITAN_SEMANTIC_ADMISSION_INDEPENDENT_REVIEW_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'REPORTED_CONTROLS_CONFIRMED_SELECTED_PUBLIC_TRUTHS_MATCH_NATIVE_ADMISSION_OPEN',
        'review_identity': {'compiler_author': False, 'fresh_independent_agent': True, 'compiler_teachers_called': False, 'prior_reports_and_compiler_source_read': True, 'blind_review': False, 'independent_human_review': False},
        'scope': 'Existing open v2 only. Full300authority and64Horn rows for control replication; first existing two-row parent per kind for selected public truth/wording review. Not a full384/2000-row semantic admission.',
        'input_bindings': inputs, 'script_and_context_bindings': bindings,
        'authority_control': dict(authority), 'selected_source_counts': dict(authority_sources), 'fact_source_value_prefix_occurrences': dict(source_prefixes), 'authority_examples': authority_examples,
        'horn_control': dict(horn), 'horn_examples': horn_examples,
        'selected_examples_count': len(examples), 'selected_truth_or_wrapper_mismatches': selected_mismatches, 'selected_public_examples': examples,
        'findings': [
            {'classification': 'SOURCE_SENSITIVITY_IDENTIFICATION_GAP', 'finding': 'A query-aware policy ignoring source fields solves all300actual TRAIN authority rows by v-prefix matching. Same facts/entity/attribute with authority switched to rumor make it0/300. Correct actual gold does not identify source-sensitive acquisition.'},
            {'classification': 'DISCLOSED_QUERY_AWARE_FLOORS', 'finding': 'Horn copied-fact elseUNKNOWN is49/64; one simultaneous round using only original facts is56/64. Neither control is query-blind; the existing deterministic stateless query-blind50percent pair bound remains a different claim.'},
            {'classification': 'SELECTED_PUBLIC_TRUTHS_SUPPORTED', 'finding': 'No actualgold defect found in300authority,64Horn, or24selected public examples. Remaining train/pilot truth agreement is recorded author/root crosscheck evidence, not independently rederived here.'},
            {'classification': 'WORDING_HARDENING_NOT_ESTABLISHED_GOLD_ERROR', 'finding': 'Sequential register execution supports current-state add_from; explicitly defining this primitive in a future codebook reduces semantic assumptions. Plain wrapper lacks explicit canonical decimal spelling/terminal punctuation. String-only JSON is explicit. These do not establish wrong current gold.'},
            {'classification': 'PROSPECTIVE_SUCCESSOR_CONDITION', 'finding': 'A selected-authority successor should select source in query, preserve same-fact pairs with distinct answers, counterbalance missing/conflict evidence for both sources, and make value morphology source-independent. Actual blinded projections require successor review.'},
        ],
        'boundaries': {'native_model_calls': 0, 'training_calls': 0, 'protected_or_confirmation_reads': 0, 'new_bank_generated': False, 'existing_source_or_data_modified': False, 'owner_messages': 0, 'permits_changed': False, 'serving_changes': False, 'native_admission_frozen': False, 'scientific_model_quality_proved': False},
        'limits': ['Author independence is narrower than human/blind admission; this reviewer read previous findings before checking them.', 'Selected examples cannot certify every wording or corner case. Actual corpus lacks several prospective conflict/tie/duplicate/multi-premise conditions noted by the author review.', 'In-memory authority falsification creates no persisted scientific bank or changed gold.', 'No successor is admitted in this report; successor review is a later, separately pinned artifact.'],
    }
    md = '''# Independent semantic admission review 001

The source-control finding reproduces: a query-aware, source-blind v-prefix policy matches all 300 actual TRAIN authority gold answers (150 UNKNOWN). Switching only the named authority to rumor in memory yields zero correct answers for that policy. A source-aware public-input derivation matches all 300 actual gold answers.

Horn controls also reproduce: copying a queried original fact and otherwise returning UNKNOWN scores 49/64, with 15 false abstentions; one simultaneous rule round using only original facts scores 56/64, with 8 false abstentions. Both read the query. They do not contradict the separate deterministic, stateless query-blind paired 50% bound. Full public closure matches all 64 actual Horn gold answers.

The first existing paired parent in each of the twelve mechanisms was checked from public instructions and payloads: 24 selected truths and visible target wrappers match. This is selected evidence, not an independent full-corpus admission. The compiler writer's full 2,384-row crosscheck and root replay remain useful recorded checks, with their stated author-independence limits.

No demonstrated current gold error was found. Sequential register execution supports current-state add_from, though explicitly defining its primitive would harden future wording. Plain responses do not explicitly prescribe canonical numeric spelling or terminal punctuation; JSON explicitly requires a string. These are wording/format hardening issues rather than established wrong truth.

The current TRAIN authority fit does not identify source-sensitive acquisition. A prospective selected-authority successor needs same-fact distinct-truth pairs, query-selected source, crossed missing/conflict evidence for both named sources, and values whose morphology does not reveal source. Its actual source-blind projections require review.

This review was performed by a fresh agent who did not author the compiler. Previous reports and compiler source were read; the review is neither blind nor independent human admission. Only standard-library public-input calculations were run. No model, training, protected/confirmation bank, owner message, permit, or serving action occurred; existing source/data and actual gold were unchanged.

The adjacent JSON pins exact open inputs, compiler source, README/PROTOCOL, cited reports, and the independent script, and includes case IDs, line numbers, and derivations. Reports were created exclusively. Native admission and model-quality claims remain open. No successor is admitted here.
'''
    # Revalidate immutable inputs immediately before exclusive publication.
    assert all(sha(DATA / n) == h for n, h in EXPECTED.items())
    assert sha(SOURCE / 'titan_joint_quality_v1/data.py') == PIN
    with destinations[0].open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    with destinations[1].open('x', encoding='utf-8', newline='\n') as f:
        f.write(md)
    print(json.dumps({'report_path': str(destinations[0]), 'report_sha256': sha(destinations[0]), 'markdown_path': str(destinations[1]), 'markdown_sha256': sha(destinations[1]), 'script_sha256': sha(Path(__file__)), 'authority': dict(authority), 'horn': dict(horn), 'selected_examples': len(examples), 'selected_mismatches': selected_mismatches}, sort_keys=True))


if __name__ == '__main__':
    main()
