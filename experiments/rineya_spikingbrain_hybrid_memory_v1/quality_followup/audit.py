"""Independent actual-prompt semantic and exposure audit. No pretrained model."""
from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
import random
import re

from . import DATA_SEED, MODEL_SEED, RECIPE, SCHEMA
from .corpus import load_packet
from .utils import PACKAGE_ROOT, digest, require, sha


def _events(text, split):
    result = []
    train = re.compile(r'^(N\d+): (\w+)=("(?:[^"\\]|\\.)*")(?:; rev=(\d+))?\.$')
    dev = re.compile(r'^(?:revision (\d+) |record )\| (N\d+) \| (\w+) \| ("(?:[^"\\]|\\.)*")\.$')
    for line in text.splitlines():
        match = (train if split == 'train' else dev).fullmatch(line)
        require(match is not None, 'unparseable evidence record')
        if split == 'train':
            subject, field, quoted, version = match.groups()
        else:
            version, subject, field, quoted = match.groups()
        item = {'subject': subject, 'field': field, 'value': json.loads(quoted)}
        if version is not None:
            item['version'] = int(version)
        result.append(item)
    require(result, 'empty evidence records')
    return result


def _query(text, split):
    prefix, suffix = ('Ask ', '. Reply ') if split == 'train' else ('Query: ', '. Return ')
    require(text.startswith(prefix) and text.endswith('.'), 'unparseable query')
    parts = text[len(prefix):-1].split(suffix)
    require(len(parts) == 2, 'invalid query contract')
    body, contract = parts
    subject, _, rule = body.partition(' ')
    require(re.fullmatch(r'N\d+', subject), 'invalid query subject')
    query = {'subject': subject}
    if rule == 'code greatest revision':
        category = 'update'
        query.update(field='code', revision_rule='greatest_number')
    elif rule == 'code, or null if missing':
        category = 'missing'
        query.update(field='code', missing_policy='closed_world_null')
    elif rule == 'fields north,south':
        category = 'multiple_facts'
        query['fields'] = ['north', 'south']
    elif rule == 'route then destination then code':
        category = 'join'
        query.update(field='code', join_chain=['route', 'destination'])
    else:
        require(rule.startswith('field '), 'unknown query rule')
        field = rule[len('field '):]
        require(field in ('code', 'shipment', 'measurement'), 'unknown requested field')
        category = {'code': 'lookup', 'shipment': 'identifier', 'measurement': 'number'}[field]
        query['field'] = field
    require(contract == ('{"a":"...","b":"..."}' if category == 'multiple_facts' else '{"v":"..."}'), 'invalid answer contract')
    return category, query


def _retention_raw(prompt):
    lines = prompt.splitlines()
    if lines[0].startswith('Rule:'):
        require(len(lines) == 3 and lines[1].startswith('Data:'), 'invalid TRAIN retention input')
        operation, inputs = lines[0][5:], json.loads(lines[1][5:])
    else:
        match = re.fullmatch(r'Operation=(\w+); values=(\{.*\})', lines[0])
        require(match is not None and len(lines) == 2, 'invalid DEV retention input')
        operation, inputs = match.group(1), json.loads(match.group(2))
    require(isinstance(inputs, dict), 'retention values must be an object')
    required = {'add': ('a', 'b', 'c'), 'subtract': ('a', 'b', 'c'), 'multiply': ('a', 'b', 'c'), 'two_step': ('a', 'b', 'c'),
                'minimum': ('values',), 'maximum': ('values',), 'median': ('values',), 'second_smallest': ('values',),
                'at_least': ('value', 'threshold', 'yes', 'no'), 'less_than': ('value', 'threshold', 'yes', 'no'),
                'greater_than': ('value', 'threshold', 'yes', 'no'), 'at_most': ('value', 'threshold', 'yes', 'no'),
                'and': ('left', 'right'), 'or': ('left', 'right'), 'xor': ('left', 'right'), 'not_and': ('left', 'right'),
                'latest': ('versions', 'statuses'), 'earliest': ('versions', 'statuses'), 'middle_revision': ('versions', 'statuses'),
                'uppercase': ('text',), 'lowercase': ('text',), 'reverse': ('text',), 'last_three': ('text',),
                'table_lookup': ('table', 'selected'), 'integer_unit_conversion': ('quantity', 'factor')}
    require(operation in required and set(inputs) == set(required[operation]), 'missing or unexpected retention fields')
    expected_instruction = {'add': 'Compute a+b.', 'subtract': 'Compute a-b.', 'multiply': 'Compute a*b.', 'two_step': 'Compute a+b-c.',
                            'minimum': 'Select minimum.', 'maximum': 'Select maximum.', 'median': 'Select median.', 'second_smallest': 'Select second smallest.',
                            'at_least': 'If value>=threshold use yes else no.', 'less_than': 'If value<threshold use yes else no.',
                            'greater_than': 'If value>threshold use yes else no.', 'at_most': 'If value<=threshold use yes else no.',
                            'and': 'Compute left AND right.', 'or': 'Compute left OR right.', 'xor': 'Compute left XOR right.', 'not_and': 'Compute NOT(left AND right).',
                            'latest': 'Select status at greatest revision.', 'earliest': 'Select status at smallest revision.', 'middle_revision': 'Select status at middle revision.',
                            'uppercase': 'Uppercase text.', 'lowercase': 'Lowercase text.', 'reverse': 'Reverse text preserving case.', 'last_three': 'Take final three text characters.',
                            'table_lookup': 'Look up selected key in table.', 'integer_unit_conversion': 'Multiply quantity by factor.'}[operation]
    require(lines[-1] == expected_instruction + ' Return only {"v":"..."} with a string result.', 'retention instruction contradicts its operation')
    if operation in ('add', 'subtract', 'multiply', 'two_step'):
        a, b, c = (inputs[k] for k in ('a', 'b', 'c'))
        require(all(type(v) is int for v in (a, b, c)), 'arithmetic requires integers')
        answer = a + b if operation == 'add' else (a - b if operation == 'subtract' else (a * b if operation == 'multiply' else a + b - c))
    elif operation in ('minimum', 'maximum', 'median', 'second_smallest'):
        values = inputs['values']
        require(isinstance(values, list) and len(values) == 5 and all(type(v) is int for v in values), 'ordering requires five integers')
        answer = sorted(values)[{'minimum': 0, 'maximum': 4, 'median': 2, 'second_smallest': 1}[operation]]
    elif operation in ('at_least', 'less_than', 'greater_than', 'at_most'):
        a, b = inputs['value'], inputs['threshold']
        require(type(a) is int and type(b) is int and isinstance(inputs['yes'], str) and isinstance(inputs['no'], str), 'invalid conditional inputs')
        flag = a >= b if operation == 'at_least' else (a < b if operation == 'less_than' else (a > b if operation == 'greater_than' else a <= b))
        answer = inputs['yes'] if flag else inputs['no']
    elif operation in ('and', 'or', 'xor', 'not_and'):
        a, b = inputs['left'], inputs['right']
        require(type(a) is bool and type(b) is bool, 'boolean inputs required')
        answer = str(a and b if operation == 'and' else (a or b if operation == 'or' else (a != b if operation == 'xor' else not (a and b)))).lower()
    elif operation in ('latest', 'earliest', 'middle_revision'):
        versions, statuses = inputs['versions'], inputs['statuses']
        require(isinstance(versions, list) and isinstance(statuses, list) and len(versions) == len(statuses) == 3 and len(set(versions)) == 3 and all(type(v) is int for v in versions), 'temporal revisions tied or invalid')
        require(all(isinstance(v, str) for v in statuses), 'temporal statuses require strings')
        order = sorted(range(3), key=lambda i: versions[i])
        answer = statuses[order[{'latest': 2, 'earliest': 0, 'middle_revision': 1}[operation]]]
    elif operation in ('uppercase', 'lowercase', 'reverse', 'last_three'):
        value = inputs['text']
        require(isinstance(value, str), 'string input required')
        answer = value.upper() if operation == 'uppercase' else (value.lower() if operation == 'lowercase' else (value[::-1] if operation == 'reverse' else value[-3:]))
    elif operation == 'table_lookup':
        require(isinstance(inputs['table'], dict) and inputs['selected'] in inputs['table'], 'table key missing')
        answer = inputs['table'][inputs['selected']]
    else:
        require(type(inputs['quantity']) is int and type(inputs['factor']) is int, 'integer conversion inputs required')
        answer = inputs['quantity'] * inputs['factor']
    require(prompt.endswith('Return only {"v":"..."} with a string result.'), 'retention output contract missing')
    return {'v': str(answer)}, operation, inputs


def parse_prompt(prompt, meta):
    """Parse literal prompt facts rather than trusting generator sidecars/targets."""
    if meta['role'] == 'retention':
        answer, operation, inputs = _retention_raw(prompt)
        require(meta.get('operation') == operation and meta.get('inputs') == inputs, 'retention sidecar differs from actual prompt')
        return answer, [], {}, {}
    blocks = prompt.split('\n---\n')
    require(len(blocks) == 4 and blocks[0] == 'Use these exhaustive records. Output compact JSON only.\n', 'incorrect evidence block structure')
    split = 'train' if blocks[-1].startswith('Ask ') else 'dev'
    evidence_index = 1 if meta['role'] == 'distant_heldout' else 2
    require(meta['role'] in ('distant_heldout', 'near_control'), 'invalid evidence role')
    evidence, filler = blocks[evidence_index], blocks[3 - evidence_index]
    require(filler.startswith('Notes: ') and filler.endswith('.') and re.fullmatch(r'Notes: (?:quiet|amber|parked|sealed|plain|idle|steady|silver)(?: (?:quiet|amber|parked|sealed|plain|idle|steady|silver))*\.', filler), 'unexpected filler facts')
    events = _events(evidence, split)
    category, query = _query(blocks[-1], split)
    require(category == meta['category'] and query == meta['query'] and events == meta['events'], 'actual prompt semantics differ from sidecar')
    require(evidence == meta['evidence_block_text'] and filler == meta['filler_block_text'], 'block text differs from sidecar')
    def lookup(subject, field):
        matches = [e['value'] for e in events if e['subject'] == subject and e['field'] == field]
        require(len(matches) <= 1, 'ambiguous duplicate fact')
        return matches[0] if matches else None
    subject = query['subject']
    if category == 'update':
        selected = [e for e in events if e['subject'] == subject and e['field'] == 'code']
        require(len(selected) == 3 and all('version' in e for e in selected) and len({e['version'] for e in selected}) == 3, 'update needs three distinct numeric revisions')
        answer = {'v': max(selected, key=lambda e: e['version'])['value']}
        latest = max(range(3), key=lambda i: events[i]['version'])
        require(meta.get('latest_position') == latest and meta.get('first_record_value') == events[0]['value'], 'update position sidecar incorrect')
    elif category == 'multiple_facts':
        answer = {'a': lookup(subject, 'north'), 'b': lookup(subject, 'south')}
        require(all(v is not None for v in answer.values()), 'required field missing')
    elif category == 'join':
        relay = lookup(subject, 'route')
        terminal = lookup(relay, 'destination') if relay is not None else None
        value = lookup(terminal, 'code') if terminal is not None else None
        require(value is not None, 'join required field missing')
        answer = {'v': value}
    else:
        value = lookup(subject, query['field'])
        require(category == 'missing' or value is not None, 'required field missing')
        answer = {'v': value}
        if category == 'missing':
            require(meta.get('answerable') == (value is not None), 'answerability sidecar incorrect')
    spans, cursor = {}, len(blocks[0])
    for i in range(1, 4):
        cursor += len('\n---\n')
        label = 'query' if i == 3 else ('evidence' if i == evidence_index else 'filler')
        spans[label] = [cursor, cursor + len(blocks[i])]
        cursor += len(blocks[i])
    require(spans == meta['block_spans'], 'sidecar block offsets differ from actual prompt')
    return answer, events, query, spans


def raw_oracle(prompt, meta):
    return parse_prompt(prompt, meta)[0]


def _canonical_invariant(row, meta, parsed):
    answer, events, query, _ = parsed
    return {'id': row['id'], 'family_id': row['family_id'], 'target': row['target'],
            'world_id': meta['world_id'], 'category': meta['category'], 'role': meta['role'],
            'counterfactual_id': meta['counterfactual_id'], 'exposure': meta['exposure'], 'query': query or None,
            'events': sorted(events, key=lambda e: json.dumps(e, separators=(',', ':'), ensure_ascii=False)),
            'filler': meta.get('filler_block_text'), 'operation': meta.get('operation'), 'inputs': meta.get('inputs')}


def _token_replay(tokenizer, row, spans):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    prompt = list(encoded['input_ids'])
    answer = list(tokenizer.encode(row['target'], add_special_tokens=False))
    require(tokenizer.eos_token_id is not None, 'EOS required for labels')
    target = answer + [tokenizer.eos_token_id]
    ids, labels = prompt + target, [-100] * len(prompt) + target
    offset = rendered.index(row['prompt'])
    token_spans = {}
    for label, (start, end) in spans.items():
        covered = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start + offset and a < end + offset]
        require(covered, 'nonempty prompt block lacks offsets')
        token_spans[label] = [covered[0], covered[-1] + 1]
    geometry = {'prompt_tokens': len(prompt), 'target_tokens': len(answer), 'train_tokens': len(ids), 'block_token_spans': token_spans}
    if 'evidence' in token_spans:
        geometry.update(evidence_to_query_gap_tokens=token_spans['query'][0] - token_spans['evidence'][1],
                        earliest_required_to_answer_tokens=len(prompt) - token_spans['evidence'][0])
    return ids, labels, geometry


def audit_packet(packet, tokenizer=None):
    data = load_packet(packet)
    manifest, design = data['manifest'], data['design']
    require(manifest.get('reviewed') is False, 'generated packet cannot self-approve')
    require(manifest.get('model_seed') == MODEL_SEED and set(design) == {'control', 'candidate', 'dev'}, 'wrong design/seed')
    require(set(manifest['splits']) == {'control', 'candidate', 'dev'}, 'only explicit TRAIN/DEV supported')
    require(manifest['generator_sha256'] == sha(PACKAGE_ROOT / 'quality_followup' / 'corpus.py'), 'generator revision changed')
    tokenizer_path = Path(manifest['tokenizer']['path'])
    locked_files = manifest['tokenizer']['files']
    require(set(locked_files) == {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}, 'tokenizer inventory incomplete')
    require(all(sha(tokenizer_path / name) == value for name, value in locked_files.items()), 'source tokenizer changed')
    if tokenizer is not None:
        require(tokenizer.eos_token_id == manifest['tokenizer']['eos_token_id'], 'wrong tokenizer EOS')
    rows_by_id, parsed, tokens = {}, {}, {}
    for condition, expected in (('control', 512), ('candidate', 512), ('dev', 672)):
        rows = data[condition]
        require(len(rows) == expected and manifest['splits'][condition] == {'protected': False, 'role': 'dev' if condition == 'dev' else 'train', 'rows': expected}, 'wrong exposure budget or protected split')
        require(len({r['id'] for r in rows}) == expected and set(design[condition]) == {r['id'] for r in rows}, 'duplicate IDs or missing sidecars')
        rows_by_id[condition] = {r['id']: r for r in rows}
        parsed[condition], tokens[condition] = {}, {}
        categories = Counter()
        for row in rows:
            require(set(row) == {'id', 'family_id', 'prompt', 'target'} and all(isinstance(v, str) and v for v in row.values()), 'invalid row schema')
            meta = design[condition][row['id']]
            require(meta['row_sha256'] == digest(row) and meta['family_id'] == row['family_id'], 'row hash/family mismatch')
            result = parse_prompt(row['prompt'], meta)
            require(json.loads(row['target']) == result[0] and row['target'] == json.dumps(result[0], separators=(',', ':'), ensure_ascii=False), 'oracle or compact target mismatch')
            parsed[condition][row['id']] = result
            categories[meta['category']] += 1
            g = meta['token_geometry']
            require(g['prompt_tokens'] <= RECIPE['max_prompt_tokens'] and g['target_tokens'] + 1 <= 32 and (condition == 'dev' or g['train_tokens'] <= 259), 'registered token cap violated')
            if meta['role'] != 'retention':
                require(meta['template_id'].startswith('uo1-' + ('dev' if condition == 'dev' else 'train') + '-'), 'template split mislabeled')
                if meta['role'] == 'distant_heldout':
                    low, high = g['registered_gap_band']
                    require((low, high) in ((145, 165), (385, 405), (1040, 1058)) if condition == 'dev' else (low, high) == (140, 157), 'distance band changed')
                    require(low <= g['evidence_to_query_gap_tokens'] <= high, 'distant gap out of band')
                else:
                    require(g['earliest_required_to_answer_tokens'] <= 128, 'near evidence outside window')
            if tokenizer is not None:
                ids, labels, replay = _token_replay(tokenizer, row, result[3])
                require(all(g.get(key) == value for key, value in replay.items()), 'token geometry replay differs')
                tokens[condition][row['id']] = (ids, labels)
                if condition == 'dev':
                    require(meta.get('input_token_sha256') == digest(ids[:replay['prompt_tokens']])
                            and meta.get('label_token_sha256') == digest([-100] * replay['prompt_tokens']), 'DEV inference runtime token/label hash replay differs')
            if condition == 'dev':
                require(all(isinstance(meta.get(key), str) and re.fullmatch('[0-9a-f]{64}', meta[key])
                            for key in ('input_token_sha256', 'label_token_sha256')), 'DEV runtime hashes missing')
        expected_categories = {'update': 384, 'missing': 64 if condition == 'dev' else 24}
        expected_categories.update({c: 32 if condition == 'dev' else 8 for c in ('lookup', 'multiple_facts', 'join', 'identifier', 'number')})
        expected_categories.update({c: 8 for c in ('arithmetic', 'ordering', 'conditional', 'boolean', 'temporal', 'string_rule', 'short_table', 'unit_conversion')})
        require(dict(categories) == expected_categories, 'category budgets changed')
    for field in ('id', 'family_id', 'prompt'):
        normalize = lambda v: ' '.join(v.casefold().split())
        train = {normalize(r[field]) for c in ('control', 'candidate') for r in data[c]}
        require(not train.intersection(normalize(r[field]) for r in data['dev']), f'TRAIN/DEV overlap: {field}')
    for field in ('world_id', 'template_id'):
        require(not {m[field] for m in design['control'].values()}.intersection(m[field] for m in design['dev'].values()), f'TRAIN/DEV overlap: {field}')
    # Only the explicit, historically unprotected SBHM TRAIN/DEV are inspected.
    history = PACKAGE_ROOT / 'corpora' / 'distant_evidence_v2'
    normalize = lambda value: ' '.join(value.casefold().split())
    historical_prompts = set()
    for name in ('train.jsonl', 'dev.jsonl'):
        path = history / name
        if path.is_file():
            historical_prompts.update(normalize(json.loads(line)['prompt']) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    require(not historical_prompts.intersection(normalize(r['prompt']) for c in ('control', 'candidate', 'dev') for r in data[c]), 'historical prompt repeated')
    history_design = history / 'design.json'
    if history_design.is_file():
        old = json.loads(history_design.read_text(encoding='utf-8'))['rows'].values()
        for key in ('world_id', 'template_id'):
            require(not {m[key] for m in old}.intersection(m[key] for c in design for m in design[c].values()), f'historical {key} repeated')
    require([r['id'] for r in data['control']] == [r['id'] for r in data['candidate']], 'slot row order differs')
    groups = {c: defaultdict(list) for c in design}
    for condition in design:
        for row_id, meta in design[condition].items():
            groups[condition][meta['world_id']].append((row_id, meta))
    expected_permutations = set(permutations(range(3)))
    for condition, worlds in groups.items():
        updates = {w: pairs for w, pairs in worlds.items() if pairs[0][1]['category'] == 'update'}
        require(len(updates) == 16, 'needs sixteen update worlds')
        positions = Counter()
        for world, pairs in updates.items():
            require(len(pairs) == 24, 'update world needs24 exposures')
            positions.update(m['latest_position'] for _, m in pairs)
            subgroups = defaultdict(list)
            for row_id, meta in pairs:
                events = parsed[condition][row_id][1]
                require(len(events) == 3 and len({e['value'] for e in events}) == 3, 'update values ambiguous')
                revisions = sorted(e['version'] for e in events)
                actual_permutation = tuple(revisions.index(e['version']) for e in events)
                require(meta['permutation'] == tuple(permutations(range(3))).index(actual_permutation), 'permutation sidecar incorrect')
                subgroups[(meta['counterfactual_id'], meta['role'])].append((meta['exposure'], actual_permutation))
            require(set(subgroups) == {(cf, role) for cf in (0, 1) for role in ('near_control', 'distant_heldout')}, 'missing update counterfactual/near group')
            for subgroup in subgroups.values():
                require({e for e, _ in subgroup} == set(range(6)), 'update exposure slots incomplete')
                if condition == 'control':
                    require(len({p for _, p in subgroup}) == 1, 'control order is not repeated')
                else:
                    require({p for _, p in subgroup} == expected_permutations, 'candidate/DEV permutations incomplete')
        require(dict(positions) == ({0: 288, 1: 48, 2: 48} if condition == 'control' else {0: 128, 1: 128, 2: 128}), 'latest position balance incorrect')
        for world, pairs in worlds.items():
            if pairs[0][1]['role'] == 'retention':
                continue
            pair_by_slot = defaultdict(dict)
            for row_id, meta in pairs:
                pair_by_slot[(meta['counterfactual_id'], meta['exposure'])][meta['role']] = row_id
            for roles in pair_by_slot.values():
                require(set(roles) == {'near_control', 'distant_heldout'}, 'position pair missing')
                a, b = (roles[r] for r in ('near_control', 'distant_heldout'))
                ma, mb = design[condition][a], design[condition][b]
                require(parsed[condition][a][:3] == parsed[condition][b][:3] and ma['filler_block_text'] == mb['filler_block_text'], 'near/distant truths differ')
                require(ma['token_geometry']['prompt_tokens'] == mb['token_geometry']['prompt_tokens'], 'near/distant token count differs')
            cf0 = [parsed[condition][row_id] for row_id, meta in pairs if meta['counterfactual_id'] == 0][0]
            cf1 = [parsed[condition][row_id] for row_id, meta in pairs if meta['counterfactual_id'] == 1][0]
            require(cf0[0] != cf1[0] and cf0[2] == cf1[2], 'counterfactual query changed or truth unchanged')
            if pairs[0][1]['category'] == 'update':
                lower = lambda result: sorted([e for e in result[1] if e['version'] != max(v['version'] for v in result[1])], key=lambda e: e['version'])
                require(lower(cf0) == lower(cf1), 'nonlatest counterfactual records changed')
    ledger = data['ledger']
    require(len(ledger) == 512 and [x['step'] for x in ledger] == list(range(512)), 'ledger steps incomplete')
    indices = list(range(512))
    random.Random(MODEL_SEED).shuffle(indices)
    require([x['slot_id'] for x in ledger] == [data['control'][i]['id'] for i in indices], 'shared shuffled schedule changed')
    for entry in ledger:
        row_id = entry['slot_id']
        invariant = None
        for condition in ('control', 'candidate'):
            row, meta = rows_by_id[condition][row_id], design[condition][row_id]
            current = _canonical_invariant(row, meta, parsed[condition][row_id])
            require(invariant is None or current == invariant, 'declared ordering intervention changes facts')
            invariant = current
            require(entry[condition + '_row_sha256'] == digest(row) and entry['invariant_sha256'] == digest(current), 'ledger row/invariant binding incorrect')
            require(all(entry[k] == meta[k] for k in ('world_id', 'category', 'role', 'counterfactual_id', 'exposure')), 'ledger exposure metadata incorrect')
            require(entry['input_tokens'] == meta['token_geometry']['train_tokens'] and entry['supervised_positions'] == meta['token_geometry']['target_tokens'] + 1, 'ledger lengths incorrect')
            if tokenizer is not None:
                ids, labels = tokens[condition][row_id]
                require(entry['input_sha256'][condition] == digest(ids) and entry['labels_sha256'][condition] == digest(labels), 'runtime token/label hash replay differs')
        require(entry['labels_sha256']['control'] == entry['labels_sha256']['candidate'], 'matched labels changed')
        if design['control'][row_id]['category'] != 'update':
            require(rows_by_id['control'][row_id] == rows_by_id['candidate'][row_id], 'shared background changed')
    path = Path(packet)
    path = path / 'manifest.json' if path.is_dir() else path
    return {'schema': SCHEMA + '/corpus-audit', 'status': 'PASS_OFFLINE_AUDIT', 'reviewed': False,
            'manifest_sha256': sha(path), 'full_tokenizer_replay': tokenizer is not None,
            'counts': {'control': 512, 'candidate': 512, 'dev': 672}, 'ledger_steps': 512,
            'checks': ['actual-prompt independent oracle', 'all six update orders', 'counterfactual and position pairs',
                       'joint intervention invariants', 'shared512slot schedule', 'split and historical disjointness',
                       'tokenizer inventory', 'runtime token/label replay' if tokenizer is not None else 'runtime replay pending'],
            'limitations': ['Offline engineering verification only; no model capability evidence.',
                            'Semantic review and separate native authorization remain required.']}
