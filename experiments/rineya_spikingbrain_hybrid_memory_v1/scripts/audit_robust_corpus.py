"""Independent raw-prompt semantic replay for the fresh SBHM v2 proposal.

This audit does not import the generator or use its target function. It parses
the language actually presented to the model, solves its finite rules, checks
counterfactual/position pairs, and independently repeats tokenizer geometry.
Only the explicitly supplied synthetic corpus and local tokenizer are read.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_record(text):
    name, field, value = r'([A-Z]{5})', r'([a-z_]+)', r'("[^"\\]*")'
    specifications = [
        (name + r': ' + field + '=' + value + r'; version=(\d+)\.', ('subject', 'field', 'value', 'version')),
        (r'At revision (\d+), ' + name + r' recorded ' + field + ' ' + value + r'\.', ('version', 'subject', 'field', 'value')),
        (r'Revision (\d+) \| ' + name + r' \| ' + field + r' \| ' + value + r'\.', ('version', 'subject', 'field', 'value')),
        (field + r' of ' + name + r' was ' + value + r' in release (\d+)\.', ('field', 'subject', 'value', 'version')),
        (name + r': ' + field + '=' + value + r'\.', ('subject', 'field', 'value')),
        (name + r' recorded ' + field + ' ' + value + r'\.', ('subject', 'field', 'value')),
        (name + r' \| ' + field + r' \| ' + value + r'\.', ('subject', 'field', 'value')),
        (r'The ' + field + r' of ' + name + r' is ' + value + r'\.', ('field', 'subject', 'value')),
    ]
    matches = []
    for pattern, keys in specifications:
        matched = re.fullmatch(pattern, text)
        if matched:
            record = dict(zip(keys, matched.groups()))
            record['value'] = json.loads(record['value'])
            if 'version' in record:
                record['version'] = int(record['version'])
            matches.append(record)
    require(len(matches) == 1, 'record does not have exactly one independent language parse: ' + text)
    return matches[0]


def raw_long_target(row, info):
    prompt = row['prompt']
    pieces = prompt.split('\n---\n')
    require(len(pieces) == 4, row['id'] + ': unexpected block layout')
    evidence_at, filler_at = (1, 2) if info['role'] == 'distant_heldout' else (2, 1)
    require(pieces[evidence_at] == info['evidence_block_text'], 'actual evidence differs from metadata')
    require(pieces[filler_at] == info['filler_block_text'], 'actual filler differs from metadata')
    require(pieces[3] == info['query_text'], 'actual query differs from metadata')
    require('Records are exhaustive.' in pieces[0], 'missing exhaustive-world contract')
    records = [parse_record(text) for text in pieces[evidence_at].splitlines()]
    require(records == info['events'], row['id'] + ': raw records disagree with canonical events')
    subjects = re.findall(r'\b[A-Z]{5}\b', pieces[3])
    require(len(subjects) == 1, 'query must name exactly one requested subject')
    subject = subjects[0]
    require(subject == info['query']['subject'], 'raw requested subject differs from metadata')
    dependencies = set()

    def lookup(requested, field, nullable=False):
        found = [(i, r['value']) for i, r in enumerate(records) if r['subject'] == requested and r['field'] == field]
        require(len(found) <= 1, 'ambiguous unversioned record')
        if not found:
            require(nullable, 'required positive fact is absent')
            return None
        dependencies.add(found[0][0])
        return found[0][1]

    category, query = info['category'], pieces[3]
    if category == 'multiple_facts':
        require('north then south' in query and '{"a":"...","b":"..."}' in query, 'pair direction/format is ambiguous')
        truth = {'a': lookup(subject, 'north'), 'b': lookup(subject, 'south')}
    elif category == 'join':
        require('route then destination' in query and 'code' in query, 'join traversal contract differs')
        relay = lookup(subject, 'route')
        terminal = lookup(relay, 'destination')
        truth = {'v': lookup(terminal, 'code')}
    elif category == 'update':
        require('greatest numbered revision' in query, 'update rule must specify maximum revision')
        candidates = [r for r in records if r['subject'] == subject and r['field'] == 'code']
        require(len(candidates) == 3 and all('version' in r for r in candidates), 'invalid update history')
        require(len({r['version'] for r in candidates}) == 3, 'ambiguous tied revision')
        dependencies.update(i for i, r in enumerate(records) if r in candidates)
        truth = {'v': max(candidates, key=lambda r: r['version'])['value']}
    else:
        field = {'identifier': 'shipment', 'number': 'measurement'}.get(category, 'code')
        require(field in query, 'query requests the wrong field')
        if category == 'missing':
            require('{"v":null}' in query and 'when absent' in query, 'missing-answer contract differs')
        truth = {'v': lookup(subject, field, nullable=category == 'missing')}
        if category == 'missing':
            dependencies.update(range(len(records)))
    for block, span in info['block_spans'].items():
        require(prompt[slice(*span)] == info[block + '_block_text'] if block != 'query'
                else prompt[slice(*span)] == info['query_text'], 'block character span is incorrect')
    for index, span in enumerate(info['evidence_spans']):
        require(span['event_index'] == index and type(span['required']) is bool, 'invalid evidence span binding')
        require(prompt[span['start_char']:span['end_char']] == span['record_text'], 'event character span is incorrect')
        require(parse_record(span['record_text']) == records[index], 'event span does not bind actual raw record')
    require({span['event_index'] for span in info['evidence_spans'] if span['required']} == dependencies,
            'required evidence flags do not match independently replayed semantic dependencies')
    return compact(truth)


def raw_retention_target(prompt):
    suffix = ' Return only {"v":"..."}; put the result in a JSON string.'
    require(prompt.endswith(suffix), 'retention response contract differs')
    text = prompt[:-len(suffix)]
    matched = re.fullmatch(r'Compute (\d+) (plus|minus) (\d+)\.', text)
    if matched:
        a, operator, b = int(matched[1]), matched[2], int(matched[3])
        return compact({'v': str(a + b if operator == 'plus' else a - b)})
    matched = re.fullmatch(r'What is (\d+) multiplied by (\d+)\?', text)
    if matched:
        return compact({'v': str(int(matched[1]) * int(matched[2]))})
    matched = re.fullmatch(r'Add (\d+) and (\d+), then subtract (\d+)\.', text)
    if matched:
        return compact({'v': str(int(matched[1]) + int(matched[2]) - int(matched[3]))})
    matched = re.fullmatch(r'List (\[[^\]]+\])\. Return the (minimum|maximum|median|second smallest) number\.', text)
    if matched:
        ordered = sorted(json.loads(matched[1]))
        which = {'minimum': 0, 'maximum': -1, 'median': len(ordered) // 2, 'second smallest': 1}[matched[2]]
        return compact({'v': str(ordered[which])})
    matched = re.fullmatch(r'Rule: if reading is (at least|less than|greater than|at most) (\d+), output (\w+); otherwise output (\w+)\. Reading=(\d+)\.', text)
    if matched:
        operator, threshold, yes, no, reading = matched.groups()
        reading, threshold = int(reading), int(threshold)
        predicate = {'at least': reading >= threshold, 'less than': reading < threshold,
                     'greater than': reading > threshold, 'at most': reading <= threshold}[operator]
        return compact({'v': yes if predicate else no})
    matched = re.fullmatch(r'A=(true|false); B=(true|false)\. Evaluate (AND\(A,B\)|XOR\(A,B\)|OR\(A,B\)|NOT\(AND\(A,B\)\))\.', text)
    if matched:
        a, b = matched[1] == 'true', matched[2] == 'true'
        answer = {'AND(A,B)': a and b, 'XOR(A,B)': a != b, 'OR(A,B)': a or b,
                  'NOT(AND(A,B))': not (a and b)}[matched[3]]
        return compact({'v': str(answer).lower()})
    if text.startswith('revision '):
        records, instruction = text.split('. Return the status at the ')
        entries = [re.fullmatch(r'revision (\d+): (\w+)', x) for x in records.split('; ')]
        require(len(entries) == 3 and all(entries), 'invalid finite temporal history')
        ordered = sorted((int(x[1]), x[2]) for x in entries)
        position = {'latest.': -1, 'earliest.': 0, 'middle revision.': 1}[instruction]
        return compact({'v': ordered[position][1]})
    matched = re.fullmatch(r'Input ("[^"]+")\. (Convert every letter to uppercase|Convert every letter to lowercase|Reverse all characters; preserve their case|Keep only the final three characters; preserve their order and case)\.', text)
    if matched:
        source = json.loads(matched[1])
        answer = {'Convert every letter to uppercase': source.upper(),
                  'Convert every letter to lowercase': source.lower(),
                  'Reverse all characters; preserve their case': source[::-1],
                  'Keep only the final three characters; preserve their order and case': source[-3:]}[matched[2]]
        return compact({'v': answer})
    if text.startswith('Table: '):
        table, subject = text[len('Table: '):].split('. Copy the value assigned to ')
        require(re.fullmatch(r'[A-Z]{5}\.', subject), 'invalid short-table query')
        subject = subject[:-1]
        if '=' in table:
            entries = re.findall(r'([A-Z]{5})=(\d+)', table)
        elif ' records ' in table:
            entries = re.findall(r'([A-Z]{5}) records (\d+)', table)
        elif ' belongs to ' in table:
            entries = [(name, value) for value, name in re.findall(r'(\d+) belongs to ([A-Z]{5})', table)]
        else:
            entries = re.findall(r'\| ([A-Z]{5}) \| (\d+) \|', table)
        require(len(entries) == 3 and len(dict(entries)) == 3, 'ambiguous short table')
        require(subject in dict(entries), 'short-table requested subject is absent')
        return compact({'v': dict(entries)[subject]})
    matched = re.fullmatch(r'One unit A equals (\d+) units B\. Convert (\d+) units A to units B\.', text)
    if matched:
        return compact({'v': str(int(matched[1]) * int(matched[2]))})
    raise ValueError('retention prompt has no independent solver: ' + text)


def independent_geometry(tokenizer, row, info):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    require(rendered.count(row['prompt']) == 1, 'prompt text not uniquely preserved in source chat template')
    offset = rendered.index(row['prompt'])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    n = len(encoded['input_ids'])
    answer = len(tokenizer.encode(row['target'], add_special_tokens=False))
    actual = {'prompt_tokens': n, 'target_tokens': answer, 'train_tokens': n + answer + 1}
    if info['role'] != 'retention':
        def span_tokens(span):
            a, b = span[0] + offset, span[1] + offset
            selected = [i for i, (left, right) in enumerate(encoded['offset_mapping']) if right > a and left < b]
            require(selected, 'nonempty language span maps to no tokens')
            return [selected[0], selected[-1] + 1]
        blocks = {key: span_tokens(value) for key, value in info['block_spans'].items()}
        facts = [span_tokens([span['start_char'], span['end_char']]) for span in info['evidence_spans']]
        required = [fact for fact, span in zip(facts, info['evidence_spans']) if span['required']]
        require(required, 'no required evidence atoms')
        gap = blocks['query'][0] - blocks['evidence'][1]
        earliest, latest = n - min(x[0] for x in required), n - max(x[1] for x in required)
        actual.update(block_token_spans=blocks, evidence_token_spans=facts,
                      evidence_to_query_gap_tokens=gap, earliest_required_to_answer_tokens=earliest,
                      latest_required_to_answer_tokens=latest,
                      rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest())
        if info['role'] == 'near_control':
            require(earliest <= 128, 'near control has required facts outside local window')
        else:
            low, high = info['token_geometry']['registered_gap_band']
            require(low <= gap <= high and gap > 128, 'distant query gap is not in registered band')
    for key, value in actual.items():
        require(info['token_geometry'][key] == value, row['id'] + ': tokenizer geometry mismatch: ' + key)
    require(n <= 2048 and answer + 1 <= 32, 'inference length cap exceeded')
    require(info['split'] != 'train' or actual['train_tokens'] <= 512, 'training length cap exceeded')
    return actual


def audit(corpus, output):
    from transformers import AutoTokenizer, __version__ as transformers_version
    require(transformers_version == '4.51.3', 'audit must use pinned Transformers 4.51.3')
    manifest = json.loads((corpus / 'manifest.json').read_text(encoding='utf-8'))
    design = json.loads((corpus / 'design.json').read_text(encoding='utf-8'))
    require(set(manifest['splits']) == {'train', 'dev'}, 'unexpected split or protected bank')
    require(sha(corpus / manifest['design']['path']) == manifest['design']['sha256'], 'design hash mismatch')
    source = Path(design['tokenizer']['path'])
    for name, expected in design['tokenizer']['files'].items():
        require(sha(source / name) == expected, 'source tokenizer asset changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast, 'offset replay requires exact fast tokenizer')
    rows, roles, worlds, tokens, split_summary = {}, Counter(), defaultdict(list), {}, {}
    semantic_count, present, absent = 0, Counter(), Counter()
    for split, item in manifest['splits'].items():
        require(item['role'] == split and item['protected'] is False, 'unsafe corpus role')
        path = corpus / item['path']
        require(sha(path) == item['sha256'], 'split hash mismatch')
        loaded = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in loaded:
            require(set(row) == {'id', 'family_id', 'prompt', 'target'}, 'row schema differs')
            require(row['id'] not in rows, 'duplicate row ID')
            info = design['rows'][row['id']]
            require(info['split'] == split and info['family_id'] == row['family_id'], 'sidecar split/family mismatch')
            require(digest(row) == info['row_sha256'], 'row digest mismatch')
            oracle = raw_retention_target(row['prompt']) if info['role'] == 'retention' else raw_long_target(row, info)
            require(oracle == row['target'], row['id'] + ': target disagrees with actual-prompt independent oracle')
            semantic_count += 1
            rows[row['id']] = row
            tokens[row['id']] = independent_geometry(tokenizer, row, info)
            roles[(split, info['role'], info['category'])] += 1
            if info['role'] != 'retention':
                worlds[info['world_id']].append(row['id'])
            if info['category'] == 'missing':
                (absent if json.loads(row['target'])['v'] is None else present)[(split, info['role'])] += 1
        split_summary[split] = {'rows': len(loaded), 'prompt_tokens_min': min(tokens[r['id']]['prompt_tokens'] for r in loaded),
                                'prompt_tokens_max': max(tokens[r['id']]['prompt_tokens'] for r in loaded),
                                'train_tokens_max': max(tokens[r['id']]['train_tokens'] for r in loaded)}
    require(set(rows) == set(design['rows']), 'unlisted or unused design rows')
    require(present == absent, 'missing evidence is not balanced against presence')
    cf_pairs, position_pairs = 0, 0
    for world, row_ids in worlds.items():
        require(len(row_ids) == 4, 'world lacks two counterfactuals times two positions')
        entries = {(design['rows'][r]['role'], design['rows'][r]['counterfactual_id']): r for r in row_ids}
        require(set(entries) == {(role, cf) for role in ('distant_heldout', 'near_control') for cf in (0, 1)}, 'incomplete world matrix')
        for role in ('distant_heldout', 'near_control'):
            left, right = (entries[(role, cf)] for cf in (0, 1))
            a, b = design['rows'][left], design['rows'][right]
            require(rows[left]['target'] != rows[right]['target'], 'counterfactual failed to change truth')
            masked = []
            for row_id, info in ((left, a), (right, b)):
                start, end = info['block_spans']['evidence']
                masked.append(rows[row_id]['prompt'][:start] + '[REMOVED_EVIDENCE]' + rows[row_id]['prompt'][end:])
            require(masked[0] == masked[1], 'counterfactual changes non-evidence prompt content')
            require(a['query_text'] == b['query_text'] and a['filler_block_text'] == b['filler_block_text'], 'counterfactual leaks truth via query/filler')
            cf_pairs += 1
        for cf in (0, 1):
            far, near = entries[('distant_heldout', cf)], entries[('near_control', cf)]
            a, b = design['rows'][far], design['rows'][near]
            require(rows[far]['target'] == rows[near]['target'], 'position manipulation changes truth')
            require(tokens[far]['prompt_tokens'] == tokens[near]['prompt_tokens'], 'position pair differs in exact token length')
            require(all(a[key] == b[key] for key in ('events', 'evidence_block_text', 'filler_block_text', 'query_text')), 'position pair changes fact/filler/query content')
            require(sorted(rows[far]['prompt'].split('\n---\n')) == sorted(rows[near]['prompt'].split('\n---\n')), 'position pair changes block multiset')
            position_pairs += 1
    split_ids = {split: [row_id for row_id in rows if design['rows'][row_id]['split'] == split] for split in ('train', 'dev')}
    for field in ('id', 'family_id', 'prompt'):
        normalized = {split: {' '.join(rows[row_id][field].casefold().split()) for row_id in row_ids} for split, row_ids in split_ids.items()}
        require(not normalized['train'].intersection(normalized['dev']), 'train/dev overlap in ' + field)
    template_ids = {split: {design['rows'][row_id]['template_id'] for row_id in row_ids} for split, row_ids in split_ids.items()}
    require(not template_ids['train'].intersection(template_ids['dev']), 'template IDs overlap across splits')
    covered, groups = set(), manifest['evaluation_groups']
    for name, families in groups.items():
        require(families and len(set(families)) == len(families) and not covered.intersection(families), 'overlapping/duplicate evaluation families')
        covered.update(families)
        require(all(design['rows'][row_id]['role'] == name for row_id in split_ids['dev'] if rows[row_id]['family_id'] in families), 'group label does not match row role')
    require(covered == {rows[r]['family_id'] for r in split_ids['dev']}, 'evaluation groups do not exhaust DEV')
    payload = sum(tokens[row_id]['train_tokens'] for row_id in split_ids['train']) * 9216
    require(payload + 32 * 2**20 <= 512 * 2**20, 'full TRAIN teacher payload plus 32 MiB reserve exceeds cache cap')
    order_patterns = defaultdict(Counter)
    for row_id, row in rows.items():
        info = design['rows'][row_id]
        if info['role'] != 'distant_heldout' or info['counterfactual_id'] != 0:
            continue
        revisions = sorted(event['version'] for event in info['events'] if 'version' in event)
        pattern = [(event['field'], event['subject'] == info['query']['subject'], span['required'],
                    revisions.index(event['version']) if 'version' in event else None)
                   for event, span in zip(info['events'], info['evidence_spans'])]
        order_patterns[(info['split'], info['category'])][compact(pattern)] += 1
    require(all(len(patterns) > 1 for patterns in order_patterns.values()),
            'a category/split retains one fixed evidence record-order pattern')
    language_examples = sorted(row_id for row_id in rows
        if (design['rows'][row_id]['role'] == 'retention' and row_id.endswith(('world-000', 'world-001')))
        or (design['rows'][row_id]['role'] == 'distant_heldout'
            and ('world-000-' in row_id or 'world-001-' in row_id)
            and (design['rows'][row_id]['counterfactual_id'] == 0 or design['rows'][row_id]['category'] == 'missing')))
    result = {'status': 'PASS_INDEPENDENT_SYNTHETIC_CORPUS_AUDIT', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'reviewer': 'Codex independent geometry/semantic-review agent', 'human_review_performed': False,
              'audit_script_sha256': sha(__file__), 'corpus_path': str(corpus.resolve()),
              'corpus_file_sha256': {name: sha(corpus / name) for name in ('manifest.json', 'design.json', 'train.jsonl', 'dev.jsonl')},
              'manifest_reviewed_flag_at_audit': manifest['reviewed_for_sbhm'],
              'manifest_hash_scope': 'Snapshot used by the audit; a later review-flag/provenance amendment has separate bytes and must preserve reviewed design/split/group contents.',
              'source_tokenizer_path': str(source), 'source_tokenizer_files_sha256': design['tokenizer']['files'],
              'transformers_version': transformers_version, 'actual_prompt_targets_independently_replayed': semantic_count,
              'counterfactual_pairs': cf_pairs, 'near_distant_position_pairs': position_pairs,
              'worlds': len(worlds), 'split_summary': split_summary,
              'role_category_counts': {':'.join(key): value for key, value in sorted(roles.items())},
              'missing_present_balance': {':'.join(key): {'present': present[key], 'absent': absent[key]} for key in sorted(present)},
              'all_target_plus_eos_tokens_at_most_32': True, 'all_near_required_facts_within_128_tokens_of_first_answer': True,
              'all_distant_evidence_gaps_beyond_128_tokens': True, 'position_pair_token_counts_exactly_equal': True,
              'counterfactual_non_evidence_content_exactly_equal_with_changed_targets': True,
              'required_fact_flags_match_independently_solved_dependencies': True,
              'evidence_record_order_patterns': {':'.join(key): dict(patterns) for key, patterns in sorted(order_patterns.items())},
              'agent_stratified_language_review_example_ids': language_examples,
              'agent_language_review_scope': 'Every fact-rendering template and retention operation, with matched requested subjects, chain direction, greatest revision, literal copying and presence/absence semantics; no human review claimed.',
              'teacher_tensor_payload_all_unique_train_rows_bytes': payload, 'teacher_cache_reserved_overhead_bytes': 32 * 2**20,
              'teacher_cache_cap_bytes': 512 * 2**20, 'native_model_loaded': False, 'gpu_execution': False,
              'pretrained_behavior_or_training_fit_measured': False,
              'caveats': ['Synthetic English closed worlds and compact JSON scoring define a narrow learnability probe.',
                          'Semantic categories overlap across splits; independent worlds and rendering templates do not establish broad distributional independence.',
                          'Join atoms are adjacent inside a distant evidence block, rather than separated distant regions.',
                          'Short synthetic retention measures source-correct rows on this corpus and does not qualify broad original-model capability.',
                          'Remaining full-attention sites can transmit old evidence; near/distant results concern one selected attention-site intervention.',
                          'Exact matching does not accept all semantically equivalent JSON spellings.',
                          'Automated semantic replay and agent review are not an independent human review or protected confirmation.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(compact({'status': result['status'], 'targets_replayed': semantic_count,
                   'counterfactual_pairs': cf_pairs, 'position_pairs': position_pairs,
                   'teacher_payload_bytes': payload, 'receipt': str(output.resolve())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.corpus, args.output)


if __name__ == '__main__':
    main()
