"""Fresh TRAIN/DEV ordering intervention. Tokenizer-only preparation, no model load."""
from collections import Counter
from itertools import permutations
import importlib.metadata
import json
from pathlib import Path
import random

from . import DATA_SEED, MODEL_SEED, RECIPE, SCHEMA
from .utils import Blocked, PACKAGE_ROOT, digest, external_path, read_json, require, sha, write_json

OTHER = ('lookup', 'multiple_facts', 'join', 'identifier', 'number')
SKILLS = ('arithmetic', 'ordering', 'conditional', 'boolean', 'temporal', 'string_rule', 'short_table', 'unit_conversion')
PERMUTATIONS = tuple(permutations(range(3)))
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
FILES = {'control': 'control_train.jsonl', 'candidate': 'candidate_train.jsonl', 'dev': 'dev.jsonl',
         'design': 'design.json', 'ledger': 'exposure_ledger.json'}
INTRO = 'Use these exhaustive records. Output compact JSON only.\n'
SEP = '\n---\n'


def _rng(seed, split, category, world, stream, attempt=0):
    return random.Random(int(digest([SCHEMA, seed, split, category, world, stream, attempt]), 16))


def _compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def load_tokenizer(path):
    """Explicit local fast tokenizer; never a hub ID, download or weight read."""
    path = Path(path).resolve()
    require(path.is_dir() and all((path / name).is_file() for name in TOKENIZER_FILES), 'complete local tokenizer required')
    locked = read_json(PACKAGE_ROOT / 'configs' / 'titan_transition.v2.memory_efficient.json')['source']
    require(digest(locked['files']) == 'c169fba225bfa58804c966caba97c147d9239b7dbcd2f3b7be87cdd05e44125a'
            and path == Path(locked['path']).resolve(), 'only the locked Titan-Transition tokenizer is registered')
    require(all(not (path / name).is_symlink() and sha(path / name) == locked['files'][name] for name in TOKENIZER_FILES), 'locked tokenizer/config hash changed')
    require(importlib.metadata.version('transformers') == '4.51.3'
            and importlib.metadata.version('tokenizers') == '0.21.4', 'tokenizer environment pins changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast and tokenizer.eos_token_id is not None, 'fast tokenizer with EOS required')
    return tokenizer


def encode_lists(tokenizer, row, training=True):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    require(ids, 'empty prompt')
    labels = [-100] * len(ids)
    if training:
        target = list(tokenizer.encode(row['target'], add_special_tokens=False)) + [tokenizer.eos_token_id]
        require(tokenizer.eos_token_id is not None, 'tokenizer EOS missing')
        ids += target
        labels += target
    return ids, labels


def _geometry(tokenizer, prompt, target, spans):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    offset = rendered.index(prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    intervals = encoded['offset_mapping']
    token_spans = {}
    for name, (start, end) in spans.items():
        covered = [i for i, (a, b) in enumerate(intervals) if b > start + offset and a < end + offset]
        require(covered, f'no offsets for {name}')
        token_spans[name] = [covered[0], covered[-1] + 1]
    prompt_tokens = len(encoded['input_ids'])
    target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
    result = {'prompt_tokens': prompt_tokens, 'target_tokens': target_tokens,
              'train_tokens': prompt_tokens + target_tokens + 1, 'block_token_spans': token_spans}
    if 'evidence' in token_spans:
        result.update(evidence_to_query_gap_tokens=token_spans['query'][0] - token_spans['evidence'][1],
                      earliest_required_to_answer_tokens=prompt_tokens - token_spans['evidence'][0])
    return result


def _event_text(item, split):
    suffix = f'; rev={item["version"]}' if 'version' in item else ''
    if split == 'train':
        return f'{item["subject"]}: {item["field"]}={_compact(item["value"])}{suffix}.'
    prefix = f'revision {item["version"]} | ' if 'version' in item else 'record | '
    return f'{prefix}{item["subject"]} | {item["field"]} | {_compact(item["value"])}.'


def _query_text(category, query, split):
    subject = query['subject']
    if category == 'update':
        body = f'{subject} code greatest revision'
    elif category == 'missing':
        body = f'{subject} code, or null if missing'
    elif category == 'multiple_facts':
        body = f'{subject} fields north,south'
    elif category == 'join':
        body = f'{subject} route then destination then code'
    else:
        body = f'{subject} field {query["field"]}'
    contract = '{"a":"...","b":"..."}' if category == 'multiple_facts' else '{"v":"..."}'
    return (f'Ask {body}. Reply {contract}.' if split == 'train' else f'Query: {body}. Return {contract}.')


def _assemble(events, filler, query, role, split):
    evidence = '\n'.join(_event_text(e, split) for e in events)
    prompt, spans = INTRO, {}
    blocks = (('evidence', evidence), ('filler', filler)) if role == 'distant_heldout' else (('filler', filler), ('evidence', evidence))
    for label, text in blocks:
        prompt += SEP
        spans[label] = [len(prompt), len(prompt) + len(text)]
        prompt += text
    prompt += SEP
    spans['query'] = [len(prompt), len(prompt) + len(query)]
    return prompt + query, spans, evidence


def _world(seed, split, category, index, attempt):
    names = [f'N{x}' for x in _rng(seed, split, category, index, 'entities', attempt).sample(range(100, 1000), 6)]
    r = _rng(seed, split, category, index, 'answers', attempt)
    pair = [str(x) for x in r.sample(range(100, 1000), 2)]
    distractors = [str(x) for x in _rng(seed, split, category, index, 'distractors', attempt).sample(range(100, 1000), 4)]
    subject, relay, terminal, decoy, spare, _ = names
    query = {'subject': subject, 'field': 'code'}
    variants = []
    def event(s, f, v, version=None):
        item = {'subject': s, 'field': f, 'value': v}
        if version is not None:
            item['version'] = version
        return item
    revisions = sorted(_rng(seed, split, category, index, 'revisions', attempt).sample(range(10, 99), 3))
    if category == 'identifier':
        pair = [f'aB-{v}-xY' for v in pair]
        query['field'] = 'shipment'
    elif category == 'number':
        pair = [f'-{v[:2]}.{v[1:]}' for v in pair]
        query['field'] = 'measurement'
    if category == 'multiple_facts':
        query = {'subject': subject, 'fields': ['north', 'south']}
    elif category == 'join':
        query['join_chain'] = ['route', 'destination']
    elif category == 'update':
        query['revision_rule'] = 'greatest_number'
    elif category == 'missing':
        query['missing_policy'] = 'closed_world_null'
    secondary = [str(x) for x in _rng(seed, split, category, index, 'secondary_answers', attempt).sample(range(100, 1000), 2)]
    for cf, value in enumerate(pair):
        if category == 'update':
            events = [event(subject, 'code', distractors[0], revisions[0]), event(subject, 'code', distractors[1], revisions[1]), event(subject, 'code', value, revisions[2])]
            target = {'v': value}
        elif category == 'multiple_facts':
            events = [event(subject, 'north', value), event(decoy, 'north', distractors[0]), event(subject, 'south', secondary[cf])]
            target = {'a': value, 'b': secondary[cf]}
        elif category == 'join':
            events = [event(subject, 'route', relay), event(relay, 'destination', terminal), event(terminal, 'code', value), event(decoy, 'code', distractors[0])]
            target = {'v': value}
        elif category == 'missing':
            events = [event(subject if cf == 0 else spare, 'code', pair[0]), event(subject, 'weight', distractors[0]), event(decoy, 'code', distractors[1])]
            target = {'v': pair[0] if cf == 0 else None}
        else:
            events = [event(decoy, query['field'], distractors[0]), event(subject, query['field'], value)]
            target = {'v': value}
        variants.append((events, _compact(target)))
    order = list(range(len(variants[0][0])))
    _rng(seed, split, category, index, 'ordering', attempt).shuffle(order)
    return {'query': query, 'variants': variants, 'order': order, 'names': names}


def _filler(seed, split, category, index, attempt, count):
    r = _rng(seed, split, category, index, 'filler', attempt)
    words = ('quiet', 'amber', 'parked', 'sealed', 'plain', 'idle', 'steady', 'silver')
    return 'Notes: ' + ' '.join(r.choice(words) for _ in range(count)) + '.'


def _make_evidence_row(tokenizer, split, category, index, description, events, target, filler, role, exposure, permutation):
    world = f'uo1-{split}-{category}-world-{index:03d}'
    cf = description['cf']
    family = f'{world}-{role}'
    row_id = f'{family}-cf{cf}-slot{exposure}'
    prompt, spans, evidence = _assemble(events, filler, _query_text(category, description['query'], split), role, split)
    row = {'id': row_id, 'family_id': family, 'prompt': prompt, 'target': target}
    meta = {'category': category, 'role': role, 'world_id': world, 'group_id': world, 'family_id': family,
            'counterfactual_id': cf, 'exposure': exposure, 'permutation': permutation,
            'template_id': f'uo1-{split}-{category}-template-{index % 2}', 'query': description['query'],
            'events': events, 'evidence_block_text': evidence, 'filler_block_text': filler, 'block_spans': spans,
            'token_geometry': _geometry(tokenizer, prompt, target, spans), 'row_sha256': digest(row)}
    if category == 'update':
        latest = max(range(3), key=lambda i: events[i]['version'])
        meta.update(latest_position=latest, first_record_value=events[0]['value'])
    if category == 'missing':
        meta['answerable'] = cf == 0
    return row, meta


def _evidence_world(tokenizer, seed, split, category, index):
    # Admission/resampling is exclusively tokenizer geometry, never model output.
    for attempt in range(500):
        desc = _world(seed, split, category, index, attempt)
        if category == 'update' and any(len({event['value'] for event in events}) != 3 for events, _ in desc['variants']):
            continue
        first_events, first_target = desc['variants'][0]
        band = (140, 157) if split == 'train' else ((145, 165), (385, 405), (1040, 1058))[index % 3]
        filler = None
        for n in range(band[0] - 8, band[1] + 1):
            trial = _filler(seed, split, category, index, attempt, n)
            prompt, spans, _ = _assemble(first_events, trial, _query_text(category, desc['query'], split), 'distant_heldout', split)
            gap = _geometry(tokenizer, prompt, first_target, spans)['evidence_to_query_gap_tokens']
            if band[0] <= gap <= band[1]:
                filler = trial
                break
        if filler is None:
            continue
        records = {'control': [], 'candidate': [], 'dev': []}
        meta = {k: {} for k in records}
        for cf, (canonical, target) in enumerate(desc['variants']):
            for role in ('distant_heldout', 'near_control'):
                count = 6 if category == 'update' else 1
                for exposure in range(count):
                    conditions = ('control', 'candidate') if split == 'train' else ('dev',)
                    for condition in conditions:
                        if category == 'update':
                            latest_position = 0 if index < 12 else (1 if index < 14 else 2)
                            fixed = ([2, 0, 1], [0, 2, 1], [0, 1, 2])[latest_position]
                            order = fixed if condition == 'control' else PERMUTATIONS[exposure]
                            permutation = PERMUTATIONS.index(tuple(order))
                        else:
                            order, permutation = desc['order'], None
                        events = [canonical[i] for i in order]
                        row, info = _make_evidence_row(tokenizer, split, category, index, {**desc, 'cf': cf}, events, target, filler, role, exposure, permutation)
                        info['token_geometry']['registered_gap_band'] = list(band)
                        info['resample_attempt'] = attempt
                        records[condition].append(row)
                        meta[condition][row['id']] = info
        all_rows = [(row, meta[c][row['id']]) for c in records for row in records[c]]
        if any(m['token_geometry']['prompt_tokens'] > RECIPE['max_prompt_tokens'] or
               m['token_geometry']['target_tokens'] + 1 > RECIPE['max_new_tokens'] or
               (split == 'train' and m['token_geometry']['train_tokens'] > RECIPE['max_train_tokens']) or
               (m['role'] == 'near_control' and m['token_geometry']['earliest_required_to_answer_tokens'] > 128) or
               (m['role'] == 'distant_heldout' and not band[0] <= m['token_geometry']['evidence_to_query_gap_tokens'] <= band[1]) for row, m in all_rows):
            continue
        # All position/CF/permutation variants have equal input and label lengths for update worlds.
        if category == 'update' and len({(m['token_geometry']['train_tokens'], m['token_geometry']['target_tokens']) for _, m in all_rows}) != 1:
            continue
        # Position is the sole difference within each exact factual/permutation pair.
        lengths = {}
        matched = True
        for c in records:
            for row in records[c]:
                m = meta[c][row['id']]
                key = (c, m['counterfactual_id'], m['exposure'])
                length = m['token_geometry']['prompt_tokens']
                if key in lengths and lengths[key] != length:
                    matched = False
                lengths[key] = length
        if matched:
            return records, meta
    raise Blocked(f'failed geometry admission for {split}/{category}/{index} after 500 whole-world resamples')


def _retention(seed, split, skill, index, tokenizer):
    r = _rng(seed, split, skill, index, 'retention_inputs')
    a, b, c = r.sample(range(3, 30), 3)
    if skill == 'arithmetic':
        operation = ('add', 'subtract', 'multiply', 'two_step')[index % 4]
        inputs = {'a': a, 'b': b, 'c': c}
        answer = {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'two_step': a + b - c}[operation]
    elif skill == 'ordering':
        operation = ('minimum', 'maximum', 'median', 'second_smallest')[index % 4]
        values = r.sample(range(1, 100), 5)
        inputs = {'values': values}
        answer = sorted(values)[{'minimum': 0, 'maximum': 4, 'median': 2, 'second_smallest': 1}[operation]]
    elif skill == 'conditional':
        operation = ('at_least', 'less_than', 'greater_than', 'at_most')[index % 4]
        yes, no = r.sample(('amber', 'blue', 'green', 'silver'), 2)
        inputs = {'value': a, 'threshold': b, 'yes': yes, 'no': no}
        flag = {'at_least': a >= b, 'less_than': a < b, 'greater_than': a > b, 'at_most': a <= b}[operation]
        answer = yes if flag else no
    elif skill == 'boolean':
        operation = ('and', 'or', 'xor', 'not_and')[index % 4]
        x, y = bool(r.randrange(2)), bool(r.randrange(2))
        inputs = {'left': x, 'right': y}
        answer = str({'and': x and y, 'or': x or y, 'xor': x != y, 'not_and': not (x and y)}[operation]).lower()
    elif skill == 'temporal':
        operation = ('latest', 'earliest', 'middle_revision', 'latest')[index % 4]
        versions, statuses = r.sample(range(10, 99), 3), r.sample(('amber', 'blue', 'green', 'silver'), 3)
        inputs = {'versions': versions, 'statuses': statuses}
        pos = sorted(range(3), key=lambda p: versions[p])[{'latest': 2, 'earliest': 0, 'middle_revision': 1}[operation]]
        answer = statuses[pos]
    elif skill == 'string_rule':
        operation = ('uppercase', 'lowercase', 'reverse', 'last_three')[index % 4]
        value = ''.join(r.choice('abCdEfGhIjKl') for _ in range(6))
        inputs = {'text': value}
        answer = {'uppercase': value.upper(), 'lowercase': value.lower(), 'reverse': value[::-1], 'last_three': value[-3:]}[operation]
    elif skill == 'short_table':
        operation = 'table_lookup'
        values = [str(v) for v in r.sample(range(100, 1000), 3)]
        selected = r.randrange(3)
        inputs = {'table': dict(zip(('oak', 'elm', 'ash'), values)), 'selected': ('oak', 'elm', 'ash')[selected]}
        answer = values[selected]
    else:
        operation = 'integer_unit_conversion'
        inputs = {'quantity': a, 'factor': (10, 100, 1000, 60)[index % 4]}
        answer = inputs['quantity'] * inputs['factor']
    instruction = {'add': 'Compute a+b.', 'subtract': 'Compute a-b.', 'multiply': 'Compute a*b.', 'two_step': 'Compute a+b-c.',
                   'minimum': 'Select minimum.', 'maximum': 'Select maximum.', 'median': 'Select median.', 'second_smallest': 'Select second smallest.',
                   'at_least': 'If value>=threshold use yes else no.', 'less_than': 'If value<threshold use yes else no.',
                   'greater_than': 'If value>threshold use yes else no.', 'at_most': 'If value<=threshold use yes else no.',
                   'and': 'Compute left AND right.', 'or': 'Compute left OR right.', 'xor': 'Compute left XOR right.', 'not_and': 'Compute NOT(left AND right).',
                   'latest': 'Select status at greatest revision.', 'earliest': 'Select status at smallest revision.', 'middle_revision': 'Select status at middle revision.',
                   'uppercase': 'Uppercase text.', 'lowercase': 'Lowercase text.', 'reverse': 'Reverse text preserving case.', 'last_three': 'Take final three text characters.',
                   'table_lookup': 'Look up selected key in table.', 'integer_unit_conversion': 'Multiply quantity by factor.'}[operation]
    body = f'Rule:{operation}\nData:{_compact(inputs)}' if split == 'train' else f'Operation={operation}; values={_compact(inputs)}'
    prompt = f'{body}\n{instruction} Return only {{"v":"..."}} with a string result.'
    world = f'uo1-{split}-retention-{skill}-{index:03d}'
    row = {'id': world, 'family_id': world, 'prompt': prompt, 'target': _compact({'v': str(answer)})}
    meta = {'category': skill, 'skill': skill, 'role': 'retention', 'world_id': world, 'group_id': world, 'family_id': world,
            'counterfactual_id': None, 'exposure': 0, 'permutation': None,
            'operation': operation, 'inputs': inputs, 'template_id': f'uo1-{split}-retention-{skill}-{index % 4}',
            'token_geometry': _geometry(tokenizer, prompt, row['target'], {}), 'row_sha256': digest(row)}
    require(meta['token_geometry']['train_tokens'] <= 259, 'retention row exceeds training cap')
    return row, meta


def invariant(row, meta):
    """Everything held fixed by the declared intervention, except ordering."""
    return {'id': row['id'], 'family_id': row['family_id'], 'target': row['target'],
            'world_id': meta['world_id'], 'category': meta['category'], 'role': meta['role'],
            'counterfactual_id': meta['counterfactual_id'], 'exposure': meta['exposure'],
            'query': meta.get('query'), 'events': sorted(meta.get('events', []), key=_compact),
            'filler': meta.get('filler_block_text'), 'operation': meta.get('operation'), 'inputs': meta.get('inputs')}


def build_packet(output, tokenizer_path, seed=DATA_SEED):
    """Write one exclusive unreviewed packet after complete tokenizer admission."""
    output = external_path(output)
    require(not output.exists(), 'refusing to overwrite packet')
    require(seed == DATA_SEED, 'registered dataset seed must be170105')
    tokenizer = load_tokenizer(tokenizer_path)
    rows, design = {c: [] for c in ('control', 'candidate', 'dev')}, {c: {} for c in ('control', 'candidate', 'dev')}
    for split in ('train', 'dev'):
        categories = [('update', 16), ('missing', 6 if split == 'train' else 16)]
        categories += [(c, 2 if split == 'train' else 8) for c in OTHER]
        for category, count in categories:
            for index in range(count):
                generated, info = _evidence_world(tokenizer, seed, split, category, index)
                for condition in rows:
                    rows[condition].extend(generated[condition])
                    design[condition].update(info[condition])
        for skill in SKILLS:
            for index in range(8):
                row, meta = _retention(seed, split, skill, index, tokenizer)
                for condition in (('control', 'candidate') if split == 'train' else ('dev',)):
                    rows[condition].append(row)
                    design[condition][row['id']] = meta
    require([len(rows[c]) for c in rows] == [512, 512, 672], 'registered row counts changed')
    for row in rows['dev']:
        ids, labels = encode_lists(tokenizer, row, training=False)
        design['dev'][row['id']].update(input_token_sha256=digest(ids), label_token_sha256=digest(labels))
    # Same shuffled slot order, one update per unique exposure row.
    indices = list(range(512))
    random.Random(MODEL_SEED).shuffle(indices)
    ledger = []
    for step, index in enumerate(indices):
        a, b = rows['control'][index], rows['candidate'][index]
        ma, mb = design['control'][a['id']], design['candidate'][b['id']]
        require(invariant(a, ma) == invariant(b, mb), 'unmatched exposure invariants')
        ia, la = encode_lists(tokenizer, a)
        ib, lb = encode_lists(tokenizer, b)
        require(len(ia) == len(ib) and la == lb, 'unmatched runtime token/label counts')
        ledger.append({'step': step, 'slot_id': a['id'], 'world_id': ma['world_id'], 'category': ma['category'],
                       'role': ma['role'], 'counterfactual_id': ma['counterfactual_id'], 'exposure': ma['exposure'],
                       'control_row_sha256': digest(a), 'candidate_row_sha256': digest(b),
                       'invariant_sha256': digest(invariant(a, ma)), 'input_tokens': len(ia),
                       'supervised_positions': sum(v != -100 for v in la),
                       'input_sha256': {'control': digest(ia), 'candidate': digest(ib)},
                       'labels_sha256': {'control': digest(la), 'candidate': digest(lb)}})
    random.Random(DATA_SEED).shuffle(rows['dev'])
    output.mkdir(parents=True, exist_ok=False)
    for condition in rows:
        with (output / FILES[condition]).open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows[condition]:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    write_json(output / FILES['design'], design)
    write_json(output / FILES['ledger'], ledger)
    tokenizer_path = Path(tokenizer_path).resolve()
    files = {key: {'path': name, 'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for key, name in FILES.items()}
    manifest = {'schema': SCHEMA, 'reviewed': False, 'purpose': 'unprotected synthetic TRAIN/DEV; no native capability evidence',
                'seed': seed, 'model_seed': MODEL_SEED, 'recipe': RECIPE, 'files': files,
                'splits': {c: {'role': 'dev' if c == 'dev' else 'train', 'protected': False, 'rows': len(rows[c])} for c in rows},
                'tokenizer': {'path': str(tokenizer_path), 'files': {n: sha(tokenizer_path / n) for n in TOKENIZER_FILES},
                              'eos_token_id': tokenizer.eos_token_id, 'chat_contract': 'oneuser;add_generation_prompt=True;enable_thinking=False;noextraspecialtokens',
                              'transformers_version': importlib.metadata.version('transformers'),
                              'tokenizers_version': importlib.metadata.version('tokenizers')},
                'generator_sha256': sha(__file__),
                'limitations': ['One synthetic DEV panel;16 dependent update worlds;one model seed.',
                                'Ordering diversity and positional balance are a joint intervention.',
                                'No broad retention, promotion or recurrence claim.'],
                'summaries': {c: {'categories': dict(Counter(m['category'] for m in design[c].values())),
                                   'prompt_tokens_max': max(m['token_geometry']['prompt_tokens'] for m in design[c].values()),
                                   'train_tokens_max': max(m['token_geometry']['train_tokens'] for m in design[c].values())} for c in rows}}
    write_json(output / 'manifest.json', manifest)
    return {'manifest': str(output / 'manifest.json'), 'summaries': manifest['summaries'],
            'counts': {c: len(rows[c]) for c in rows}, 'reviewed': False}


def load_packet(packet):
    path = Path(packet).resolve()
    path = path / 'manifest.json' if path.is_dir() else path
    manifest, root = read_json(path), path.parent
    require(manifest.get('schema') == SCHEMA and manifest.get('seed') == DATA_SEED and manifest.get('recipe') == RECIPE, 'wrong packet schema/recipe')
    require(set(manifest.get('files', {})) == set(FILES), 'packet file inventory incomplete')
    result = {'manifest': manifest, 'manifest_path': str(path), 'root': str(root)}
    for key, item in manifest['files'].items():
        require(item.get('path') == FILES[key], 'unexpected corpus filename')
        file = (root / item['path']).resolve()
        require(file.is_relative_to(root) and file.is_file() and not (root / item['path']).is_symlink(), 'unsafe corpus path')
        require(file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], f'packet file changed: {key}')
        require(file.stat().st_size <= 32 * 1024 * 1024, 'packet file exceeds32MiB')
        result[key] = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()] if key in ('control', 'candidate', 'dev') else read_json(file)
    return result
