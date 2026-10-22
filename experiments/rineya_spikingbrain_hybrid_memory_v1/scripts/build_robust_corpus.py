"""Build fresh controlled synthetic SBHM TRAIN/DEV with exact local tokenizer.

No pretrained weights, GPU, network, or existing evaluation bank is used.
The initial manifest stays unreviewed until an independent audit approves it.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import string


REVISION = 'distant_evidence_v2'
CATEGORIES = ('lookup', 'multiple_facts', 'join', 'update', 'identifier', 'number', 'missing')
RETENTION = ('arithmetic', 'ordering', 'conditional', 'boolean', 'temporal',
             'string_rule', 'short_table', 'unit_conversion')
SEPARATOR = '\n---\n'
INTRO = ('Records are exhaustive. Return compact JSON without spaces or explanation; '
         'preserve quoted values.\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rng(seed, split, category, world, stream):
    """Stream keys prevent names/filler/answers from sharing draws or formulas."""
    return random.Random(int(digest([REVISION, seed, split, category, world, stream]), 16))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def opaque_name(randomness):
    return ''.join(randomness.choice(string.ascii_uppercase) for _ in range(5))


def unique_names(randomness, count):
    result = []
    while len(result) < count:
        name = opaque_name(randomness)
        if name not in result:
            result.append(name)
    return result


def different_values(randomness, maker):
    first, second = maker(randomness), maker(randomness)
    while first == second:
        second = maker(randomness)
    return first, second


def literal(randomness):
    return str(randomness.randrange(100, 1000))


def identifier(randomness):
    letters = ''.join(randomness.choice(string.ascii_letters) for _ in range(3))
    return f'{letters}-{randomness.randrange(1000, 10000)}-{randomness.choice("xYz")}'


def measurement(randomness):
    sign = randomness.choice(('', '-'))
    return f'{sign}{randomness.randrange(10, 100)}.{randomness.randrange(100):02d}'


def event(subject, field, value, version=None):
    result = {'subject': subject, 'field': field, 'value': value}
    if version is not None:
        result['version'] = version
    return result


def render_event(item, template):
    subject, field, value = item['subject'], item['field'], compact(item['value'])
    if 'version' in item:
        version = item['version']
        variants = (f'{subject}: {field}={value}; version={version}.',
                    f'At revision {version}, {subject} recorded {field} {value}.',
                    f'Revision {version} | {subject} | {field} | {value}.',
                    f'{field} of {subject} was {value} in release {version}.')
    else:
        variants = (f'{subject}: {field}={value}.',
                    f'{subject} recorded {field} {value}.',
                    f'{subject} | {field} | {value}.',
                    f'The {field} of {subject} is {value}.')
    return variants[template]


def query_text(category, query, template):
    subject = query['subject']
    if category == 'multiple_facts':
        instruction = f'{subject}: give north then south as {{"a":"...","b":"..."}}.'
    elif category == 'join':
        chain = ' then '.join(query['join_chain'])
        variants = (f'From {subject}, follow {chain}; give the terminal code as {{"v":"..."}}.',
                    f'{subject}: traverse {chain}. Copy the resulting code into {{"v":"..."}}.',
                    f'Find {subject}\'s {chain} code. Answer {{"v":"..."}}.',
                    f'Use {chain} starting at {subject}; return its code as {{"v":"..."}}.')
        instruction = variants[template]
    elif category == 'update':
        instruction = f'{subject}: copy code at the greatest numbered revision into {{"v":"..."}}.'
    elif category == 'missing':
        instruction = f'{subject}: return its code as {{"v":"..."}}, or {{"v":null}} when absent.'
    else:
        field = query['field']
        variants = (f'{subject}: copy {field} into {{"v":"..."}}.',
                    f'Return {subject}\'s {field} as {{"v":"..."}}.',
                    f'Look up {field} for {subject}. Answer {{"v":"..."}}.',
                    f'Copy the exact {field} assigned to {subject}: {{"v":"..."}}.')
        instruction = variants[template]
    return 'Request: ' + instruction


def world(seed, split, category, index):
    names = unique_names(rng(seed, split, category, index, 'entity_names'), 6)
    values_rng = rng(seed, split, category, index, 'world_values')
    subject, relay, terminal, decoy, extra, spare = names
    template = index % 2 + (0 if split == 'train' else 2)
    pair = different_values(values_rng, identifier if category == 'identifier' else
                            measurement if category == 'number' else literal)
    ancillary = [literal(values_rng) for _ in range(4)]
    query = {'subject': subject, 'missing_policy': 'closed_world_null'}
    variants = []
    for cf, value in enumerate(pair):
        if category == 'multiple_facts':
            second = different_values(rng(seed, split, category, index, 'secondary_values'), literal)[cf]
            events = [event(subject, 'north', value), event(decoy, 'north', ancillary[0]),
                      event(subject, 'south', second)]
            required = [0, 2]
            query['fields'] = ['north', 'south']
            target = compact({'a': value, 'b': second})
        elif category == 'join':
            # Three explicit relations are adjacent; the whole relation block is distant.
            events = [event(subject, 'route', relay), event(relay, 'destination', terminal),
                      event(terminal, 'code', value), event(decoy, 'code', ancillary[0])]
            required = [0, 1, 2]
            query['join_chain'] = ['route', 'destination']
            target = compact({'v': value})
        elif category == 'update':
            versions = sorted(rng(seed, split, category, index, 'version_numbers').sample(range(10, 90), 3))
            # Deliberately scramble physical order; greatest revision determines current value.
            order = [2, 0, 1] if index % 2 else [1, 2, 0]
            truth_events = [event(subject, 'code', ancillary[0], versions[0]),
                            event(subject, 'code', ancillary[1], versions[1]),
                            event(subject, 'code', value, versions[2])]
            events = [truth_events[position] for position in order]
            required = list(range(3))
            query.update(field='code', revision_rule='greatest_number')
            target = compact({'v': value})
        elif category == 'missing':
            # Presence is not announced: same record count, requested subject becomes a decoy.
            factual_subject = subject if cf == 0 else spare
            events = [event(factual_subject, 'code', pair[0]),
                      event(subject, 'weight', ancillary[0]), event(decoy, 'code', ancillary[1])]
            required = [0, 1, 2]  # Exhaustive finite record set supports absence, no negative cue.
            query.update(field='code', missing_policy='closed_world_null')
            target = compact({'v': pair[0] if cf == 0 else None})
        else:
            field = {'identifier': 'shipment', 'number': 'measurement'}.get(category, 'code')
            events = [event(decoy, field, ancillary[0]), event(subject, field, value)]
            required = [1]
            query['field'] = field
            target = compact({'v': value})
        variants.append({'events': events, 'required_event_indices': required,
                         'target': target, 'counterfactual_id': cf,
                         'counterfactual_kind': 'subject_presence_swap' if category == 'missing' else 'evidence_value_replacement'})
    # Remove fixed requested-fact/terminal-code positions without changing truth.
    # The independent order is shared across every factual and position variant.
    order = list(range(len(variants[0]['events'])))
    rng(seed, split, category, index, 'record_order').shuffle(order)
    for variant in variants:
        original_required = set(variant['required_event_indices'])
        variant['events'] = [variant['events'][position] for position in order]
        variant['required_event_indices'] = [position for position, original in enumerate(order)
                                             if original in original_required]
    return {'names': names, 'query': dict(query), 'query_text': query_text(category, query, template),
            'template_id': f'{category}-template-{template}', 'template_index': template,
            'variants': variants}


def fillers(seed, split, category, index, excluded):
    randomness = rng(seed, split, category, index, 'fillers')
    result = []
    for _ in range(180):
        name = opaque_name(randomness)
        while name in excluded:
            name = opaque_name(randomness)
        number = literal(randomness)
        color = randomness.choice(('amber', 'blue', 'green', 'silver'))
        status = randomness.choice(('parked', 'sealed', 'idle', 'ready'))
        style = randomness.randrange(3)
        text = (f'{name}: meter={number}; color={color}; state={status}.',
                f'Unit {name} recorded {number} units, {color}, {status}.',
                f'{name} | meter {number} | {color} | {status}.')[style]
        result.append(text)
    return result


def assemble(evidence, filler, query, role):
    blocks = (('evidence', evidence), ('filler', filler)) if role == 'distant_heldout' else (
        ('filler', filler), ('evidence', evidence))
    prompt, spans = INTRO, {}
    for label, block in blocks:
        prompt += SEPARATOR
        start = len(prompt)
        prompt += block
        spans[label] = [start, len(prompt)]
    prompt += SEPARATOR
    spans['query'] = [len(prompt), len(prompt) + len(query)]
    prompt += query
    return prompt, spans


def token_geometry(tokenizer, prompt, target, spans, evidence_spans):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    offset = rendered.index(prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    intervals = encoded['offset_mapping']
    ids = encoded['input_ids']

    def token_span(char_span):
        start, end = (char_span[0] + offset, char_span[1] + offset)
        covered = [i for i, (left, right) in enumerate(intervals) if right > start and left < end]
        if not covered:
            raise ValueError('nonempty span has no tokenizer offsets')
        return [covered[0], covered[-1] + 1]

    token_spans = {name: token_span(span) for name, span in spans.items()}
    fact_spans = [token_span([item['start_char'], item['end_char']]) for item in evidence_spans]
    required = [span for span, item in zip(fact_spans, evidence_spans) if item['required']]
    answer_ids = tokenizer.encode(target, add_special_tokens=False)
    return {'prompt_tokens': len(ids), 'target_tokens': len(answer_ids),
            'train_tokens': len(ids) + len(answer_ids) + 1,
            'block_token_spans': token_spans, 'evidence_token_spans': fact_spans,
            'evidence_to_query_gap_tokens': token_spans['query'][0] - token_spans['evidence'][1],
            'earliest_required_to_answer_tokens': len(ids) - min(span[0] for span in required),
            'latest_required_to_answer_tokens': len(ids) - max(span[1] for span in required),
            'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest()}


def row_variant(tokenizer, split, category, index, description, variant, filler_text, role, gap_band):
    record_texts = [render_event(item, description['template_index']) for item in variant['events']]
    evidence_text = '\n'.join(record_texts)
    prompt, spans = assemble(evidence_text, filler_text, description['query_text'], role)
    facts, cursor = [], spans['evidence'][0]
    for event_index, record in enumerate(record_texts):
        facts.append({'start_char': cursor, 'end_char': cursor + len(record), 'record_text': record,
                      'event_index': event_index, 'required': event_index in variant['required_event_indices']})
        cursor += len(record) + 1
    geometry = token_geometry(tokenizer, prompt, variant['target'], spans, facts)
    if role == 'near_control' and geometry['earliest_required_to_answer_tokens'] > 128:
        raise ValueError(f'{split}/{category}/{index}: near evidence is outside the 128-token window: {geometry}')
    if role == 'distant_heldout' and not gap_band[0] <= geometry['evidence_to_query_gap_tokens'] <= gap_band[1]:
        raise ValueError(f'{split}/{category}/{index}: distant gap outside registered band')
    if geometry['prompt_tokens'] > 2048 or (split == 'train' and geometry['train_tokens'] > 512):
        raise ValueError(f'{split}/{category}/{index}: shape exceeds registered cap')
    if geometry['target_tokens'] + 1 > 32:
        raise ValueError('target plus EOS cannot fit 32-token completion budget')
    world_id = f'{split}-{category}-world-{index:03d}'
    family = f'{world_id}-{role}'
    row_id = f'{family}-cf-{variant["counterfactual_id"]}'
    row = {'id': row_id, 'family_id': family, 'prompt': prompt, 'target': variant['target']}
    sidecar = {'split': split, 'family_id': family, 'role': role, 'category': category,
               'world_id': world_id, 'group_id': world_id, 'counterfactual_id': variant['counterfactual_id'],
               'position_pair_id': f'{world_id}-cf-{variant["counterfactual_id"]}',
               'counterfactual_kind': variant['counterfactual_kind'],
               'template_id': description['template_id'], 'template_index': description['template_index'],
               'query': description['query'], 'query_text': description['query_text'], 'events': variant['events'],
               'evidence_block_text': evidence_text, 'filler_block_text': filler_text,
               'block_spans': spans, 'evidence_spans': facts,
               'token_geometry': {**geometry, 'registered_gap_band': list(gap_band)},
               'target_contract': 'compact JSON; values copied as strings; unknown is JSON null',
               'row_sha256': digest(row)}
    return row, sidecar


def distance_world(tokenizer, seed, split, category, index):
    description = world(seed, split, category, index)
    gap_band = (140, 160) if split == 'train' else (
        (145, 190), (385, 440), (1040, 1110))[index % 3]
    lines = fillers(seed, split, category, index, description['names'])
    chosen = None
    # Select length using tokenizer counts, never using answer correctness or model outputs.
    first = description['variants'][0]
    for count in range(1, len(lines) + 1):
        filler = '\n'.join(lines[:count])
        record_texts = [render_event(item, description['template_index']) for item in first['events']]
        evidence = '\n'.join(record_texts)
        prompt, spans = assemble(evidence, filler, description['query_text'], 'distant_heldout')
        one = [{'start_char': spans['evidence'][0], 'end_char': spans['evidence'][1], 'required': True}]
        gap = token_geometry(tokenizer, prompt, first['target'], spans, one)['evidence_to_query_gap_tokens']
        if gap_band[0] <= gap <= gap_band[1]:
            chosen = filler
            break
        if gap > gap_band[1]:
            raise ValueError('filler increment skipped registered gap band')
    if chosen is None:
        raise ValueError('could not build registered distant gap')
    rows, metadata = [], {}
    for variant in description['variants']:
        pair = []
        for role in ('distant_heldout', 'near_control'):
            row, info = row_variant(tokenizer, split, category, index, description, variant, chosen, role, gap_band)
            rows.append(row)
            metadata[row['id']] = info
            pair.append(info)
        if pair[0]['token_geometry']['prompt_tokens'] != pair[1]['token_geometry']['prompt_tokens']:
            raise ValueError(f'{split}/{category}/{index}: near/distant token lengths differ')
    return rows, metadata


def retention_case(seed, split, category, index):
    randomness = rng(seed, split, category, index, 'retention_inputs')
    names = unique_names(rng(seed, split, category, index, 'retention_names'), 5)
    template = index % 2 + (0 if split == 'train' else 2)
    a, b, c = [randomness.randrange(3, 30) for _ in range(3)]
    operation, inputs = None, {}
    if category == 'arithmetic':
        operation = ('add', 'subtract', 'multiply', 'two_step')[template]
        answer = (a + b, a - b, a * b, (a + b) - c)[template]
        text = (f'Compute {a} plus {b}.', f'Compute {a} minus {b}.',
                f'What is {a} multiplied by {b}?', f'Add {a} and {b}, then subtract {c}.')[template]
        inputs = {'a': a, 'b': b, 'c': c}
    elif category == 'ordering':
        values = randomness.sample(range(1, 100), 5)
        operation = ('minimum', 'maximum', 'median', 'second_smallest')[template]
        ordered = sorted(values)
        answer = (ordered[0], ordered[-1], ordered[2], ordered[1])[template]
        text = f'List {compact(values)}. Return the {operation.replace("_", " ")} number.'
        inputs = {'values': values}
    elif category == 'conditional':
        value, threshold = randomness.randrange(1, 30), randomness.randrange(1, 30)
        labels = randomness.sample(('amber', 'blue', 'green', 'silver'), 2)
        operator = ('at_least', 'less_than', 'greater_than', 'at_most')[template]
        predicate = (value >= threshold, value < threshold, value > threshold, value <= threshold)[template]
        answer = labels[0] if predicate else labels[1]
        text = (f'Rule: if reading is {operator.replace("_", " ")} {threshold}, output {labels[0]}; '
                f'otherwise output {labels[1]}. Reading={value}.')
        operation, inputs = operator, {'value': value, 'threshold': threshold, 'yes': labels[0], 'no': labels[1]}
    elif category == 'boolean':
        left, right = bool(randomness.randrange(2)), bool(randomness.randrange(2))
        operation = ('and', 'xor', 'or', 'not_and')[template]
        answer = (left and right, left != right, left or right, not (left and right))[template]
        expression = 'NOT(AND(A,B))' if operation == 'not_and' else f'{operation.upper()}(A,B)'
        text = f'A={str(left).lower()}; B={str(right).lower()}. Evaluate {expression}.'
        answer = str(answer).lower()
        inputs = {'left': left, 'right': right}
    elif category == 'temporal':
        versions = randomness.sample(range(10, 99), 3)
        statuses = randomness.sample(('amber', 'blue', 'green', 'silver'), 3)
        operation = ('latest', 'earliest', 'middle_revision', 'latest')[template]
        order = sorted(range(3), key=lambda position: versions[position])
        selected = order[(2, 0, 1, 2)[template]]
        answer = statuses[selected]
        text = '; '.join(f'revision {v}: {s}' for v, s in zip(versions, statuses))
        text += f'. Return the status at the {operation.replace("_", " ")}.'
        inputs = {'versions': versions, 'statuses': statuses}
    elif category == 'string_rule':
        text_value = ''.join(randomness.choice(string.ascii_letters) for _ in range(6))
        operation = ('uppercase', 'lowercase', 'reverse', 'last_three')[template]
        answer = (text_value.upper(), text_value.lower(), text_value[::-1], text_value[-3:])[template]
        instruction = ('Convert every letter to uppercase', 'Convert every letter to lowercase',
                       'Reverse all characters; preserve their case',
                       'Keep only the final three characters; preserve their order and case')[template]
        text = f'Input {compact(text_value)}. {instruction}.'
        inputs = {'text': text_value}
    elif category == 'short_table':
        values = randomness.sample(range(100, 1000), 3)
        selected = randomness.randrange(3)
        operation = 'table_lookup'
        formats = (', '.join(f'{n}={v}' for n, v in zip(names, values)),
                   '; '.join(f'{n} records {v}' for n, v in zip(names, values)),
                   '\n'.join(f'| {n} | {v} |' for n, v in zip(names, values)),
                   ', '.join(f'{v} belongs to {n}' for n, v in zip(names, values)))
        answer = str(values[selected])
        text = f'Table: {formats[template]}. Copy the value assigned to {names[selected]}.'
        inputs = {'names': names[:3], 'values': values, 'selected_subject': names[selected]}
    else:
        factor = (10, 100, 1000, 60)[template]
        operation = 'integer_unit_conversion'
        answer = str(a * factor)
        text = f'One unit A equals {factor} units B. Convert {a} units A to units B.'
        inputs = {'quantity': a, 'factor': factor}
    return text + ' Return only {"v":"..."}; put the result in a JSON string.', compact({'v': str(answer)}), {
        'operation': operation, 'inputs': inputs, 'template_id': f'retention-{category}-template-{template}'}


def build(output, tokenizer_path, seed, train_worlds=8, train_retention=4):
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError('exact fast source tokenizer and EOS are required')
    if output.exists():
        raise ValueError('refusing to overwrite corpus output')
    metadata, splits, summaries = {}, {}, {}
    split_rows = {}
    for split, worlds_per_category, retained_per_category in (('train', train_worlds, train_retention), ('dev', 16, 8)):
        rows = []
        for category in CATEGORIES:
            for index in range(worlds_per_category):
                generated, info = distance_world(tokenizer, seed, split, category, index)
                rows.extend(generated)
                metadata.update(info)
        for category in RETENTION:
            for index in range(retained_per_category):
                prompt, target, info = retention_case(seed, split, category, index)
                family = f'{split}-retention-{category}-world-{index:03d}'
                row_id = family
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
                prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
                target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
                if prompt_tokens + target_tokens + 1 > 512 or target_tokens + 1 > 32:
                    raise ValueError('retention shape exceeds registered caps')
                row = {'id': row_id, 'family_id': family, 'prompt': prompt, 'target': target}
                rows.append(row)
                metadata[row_id] = {'split': split, 'family_id': family, 'role': 'retention',
                                    'category': category, 'world_id': family, 'group_id': family,
                                    'counterfactual_id': None, 'position_pair_id': None, **info,
                                    'token_geometry': {'prompt_tokens': prompt_tokens, 'target_tokens': target_tokens,
                                                       'train_tokens': prompt_tokens + target_tokens + 1},
                                    'target_contract': 'compact JSON object; scalar answer represented as string',
                                    'row_sha256': digest(row)}
        # File order is deterministic, separately randomized from world/filler values.
        rng(seed, split, 'all', 0, 'row_order').shuffle(rows)
        split_rows[split] = rows
        geometry = [metadata[row['id']]['token_geometry'] for row in rows]
        summaries[split] = {'rows': len(rows), 'roles': dict(Counter(metadata[row['id']]['role'] for row in rows)),
                            'prompt_tokens_min': min(g['prompt_tokens'] for g in geometry),
                            'prompt_tokens_max': max(g['prompt_tokens'] for g in geometry),
                            'train_tokens_max': max(g['train_tokens'] for g in geometry),
                            'missing_targets': sum(row['target'] == '{"v":null}' for row in rows)}
        if split == 'train':
            total_tokens = sum(g['train_tokens'] for g in geometry)
            summaries[split].update(total_training_tokens=total_tokens,
                                    estimated_teacher_target_payload_bytes=total_tokens * 9216,
                                    teacher_cache_soft_budget_bytes=480 * 2**20)
            if total_tokens * 9216 > 480 * 2**20:
                raise ValueError(f'Full TRAIN teacher payload {total_tokens * 9216} bytes exceeds480MiB: '
                                 'choose a smaller prospective complete paired design, not a random subsample')
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in split_rows.items():
        path = output / f'{split}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        splits[split] = {'path': path.name, 'sha256': sha(path), 'role': split, 'protected': False}
    source_files = {p.name: sha(p) for p in tokenizer_path.iterdir()
                    if p.is_file() and ('token' in p.name or p.name in ('config.json', 'vocab.json', 'merges.txt'))}
    design = {'status': 'UNREVIEWED_PENDING_INDEPENDENT_PROGRAMMATIC_AND_SEMANTIC_AUDIT',
              'generator_revision': REVISION, 'generator_sha256': sha(__file__), 'seed': seed,
              'randomness': 'Independent SHA256-keyed RNG streams for entity_names/world_values/secondary_values/version_numbers/record_order/fillers/retention_inputs/retention_names/row_order',
              'tokenizer': {'path': str(tokenizer_path.resolve()), 'files': source_files,
                            'transformers_version': transformers.__version__, 'eos_token_id': tokenizer.eos_token_id,
                            'chat_contract': 'one user message; add_generation_prompt=True; enable_thinking=False; no extra special tokens'},
              'scope': 'Fresh unprotected synthetic closed-world DEV, generated without pretrained weights or model predictions',
              'summaries': summaries,
              'semantics_by_category': {
                  'lookup': 'Exact subject/field lookup; nonrequested subjects are distractors.',
                  'multiple_facts': 'Read both north and south; output a=north,b=south.',
                  'join': 'Follow subject.route then relay.destination; read terminal.code. Adjacent evidence records form one distant relation block.',
                  'update': 'Pick the subject code at greatest numeric revision, independent of record location/order.',
                  'identifier': 'Copy exact mixed-case/hyphen shipment string.',
                  'number': 'Copy exact signed two-decimal measurement string; do not numerically reformat.',
                  'missing': 'Closed-world requested code present vs absent, balanced by subject-presence counterfactual; output null if absent.'},
              'design_rationale': [
                  'Counterfactual values change truth with identical query/entities/filler; missing pairs replace factual subject rather than announce absence.',
                  'Evidence records are permuted with an independent record_order stream shared across a world; the requested fact has no fixed position.',
                  'Near and distant controls contain the same evidence/filler/query blocks and exactly the same token count; only evidence position changes.',
                  'Near controls place every required evidence record within the registered 128-token window of the first answer position.',
                  'DEV gap bands include just-beyond-window, medium, and more-than-1000-token distances. TRAIN remains capped at512tokens including target/EOS.',
                  'World/case families and templates are disjoint across splits; pairing group_id is shared only within a split/world.',
                  'Retention covers eight varied synthetic skills; it does not measure broad original pretrained quality.',
                  'Training exposure is separately registered; corpus size does not promise all rows fit the512MiB teacher target cache.'],
              'limitations': ['Synthetic English closed worlds and structured string matching are a narrow probe.',
                              'Seven task categories share underlying skills across splits; held-out templates do not imply general distributional independence.',
                              'Multi-hop evidence atoms are adjacent; this is not a benchmark of joins among separate distant regions.',
                              'No protected confirmation, semantic model judge, model selection, or pretrained capability result.'],
              'review_required': ['Independent actual-prompt semantic replay', 'Counterfactual and position-pair audit',
                                  'Tokenizer/shape/distance replay', 'Shortcut and split leakage audit'],
              'rows': metadata}
    write_json(output / 'design.json', design)
    write_json(output / 'manifest.json', {'reviewed_for_sbhm': False,
        'purpose': 'Robust controlled synthetic development corpus; independent review pending',
        'design': {'path': 'design.json', 'sha256': sha(output / 'design.json')},
        'evaluation_groups': {role: sorted({item['family_id'] for item in metadata.values()
                                         if item['split'] == 'dev' and item['role'] == role})
                              for role in ('distant_heldout', 'near_control', 'retention')}, 'splits': splits})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--train-worlds-per-category', type=int, default=8)
    parser.add_argument('--train-retention-per-category', type=int, default=4)
    args = parser.parse_args()
    print(compact({'status': 'UNREVIEWED_PENDING_INDEPENDENT_AUDIT',
                   'output': str(args.output.resolve()),
                   'summaries': build(args.output, args.tokenizer_path, args.seed,
                                      args.train_worlds_per_category, args.train_retention_per_category)}))


if __name__ == '__main__':
    main()
