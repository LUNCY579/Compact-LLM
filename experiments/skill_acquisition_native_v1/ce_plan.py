"""Pure CE8 exposure/label bindings. No model, framework, execution grant or IO writes.

Proposal plans bind a caller-pinned preparation and its two target files. Fixture
plans are explicitly synthetic and may contain any positive multiple of eight
rows. Neither mode grants training or establishes native qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 'R-ORACLE-CE8-PLAN-BINDING-1'
SHARED_SCHEMA = 'R-ORACLE-CE8-EXPOSURE-PLAN-PROPOSAL-1'
FIXTURE_SHARED_SCHEMA = 'R-ORACLE-CE8-EXPOSURE-PLAN-SYNTHETIC-1'
ORDER = 'COMP-RS-R-ORACLE-SFT-ORDER-1|'
PERMUTE = 'COMP-RS-R-ORACLE-PERMUTE-1'
ARMS = ('canonical', 'permuted')
REFERENCE_FIELDS = ('recipe', 'anchors', 'runtime', 'source')
CHUNK = 1024 ** 2
MAX_JSON = 4 * CHUNK
MAX_LINE = 131072
HEX = frozenset('0123456789abcdef')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and set(value) <= HEX


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular bound JSON file required')
    parts, size = [], 0
    with path.open('rb') as stream:
        while block := stream.read(CHUNK):
            size += len(block)
            require(size <= MAX_JSON, 'bound JSON byte ceiling')
            parts.append(block)
    raw = b''.join(parts)
    return _json(raw), hashlib.sha256(raw).hexdigest(), size


def _order(ids):
    return sorted(ids, key=lambda row_id: (hashlib.sha256((ORDER + row_id).encode('utf8')).hexdigest(), row_id))


def _references(values):
    require(values is None or type(values) is dict, 'binding references mapping')
    values = values or {}
    require(set(values) <= set(REFERENCE_FIELDS), 'unknown future binding reference')
    for value in values.values():
        require(value is None or type(value) is dict, 'references are explicit metadata mappings or unresolved None')
    return _json(canonical({name: values.get(name) for name in REFERENCE_FIELDS}))


def _shared(ids, mode):
    return {'schema': SHARED_SCHEMA if mode == 'unapproved_proposal' else FIXTURE_SHARED_SCHEMA,
            'epochs': 1, 'rows': len(ids), 'accumulation_rows': 8, 'microbatch_rows': 1,
            'order_rule': ORDER,
            'boundaries': [{'boundary': index, 'row_ids': ids[8 * index:8 * index + 8],
                            'logical_exposure': 0} for index in range(len(ids) // 8)]}


def _finish(mode, arm, records, label_metadata, preparation, binding_references):
    ids = [record['id'] for record in records]
    pairs = [[record['id'], record['target_sha256']] for record in records]
    shared = _shared(ids, mode)
    labels = dict(label_metadata, arm=arm, ordered_target_digest=digest(pairs),
                  boundaries=[{'boundary': index, 'row_target_pairs': pairs[8 * index:8 * index + 8]}
                              for index in range(len(ids) // 8)])
    labels['boundary_labels_sha256'] = digest(labels['boundaries'])
    references = _references(binding_references)
    body = {'schema': SCHEMA, 'mode': mode, 'approved': False, 'training_authorized': False,
            'native_qualified': False, 'execution_ready': False,
            'classification': 'pure exposure/label binding; no execution permission',
            'builder_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'preparation': preparation, 'shared_exposure': shared,
            'shared_exposure_sha256': digest(shared), 'ordered_row_ids_sha256': digest(ids),
            'ordered_plan': shared['boundaries'], 'label_binding': labels,
            'label_binding_sha256': digest(labels), 'binding_references': references,
            'unbound_reference_fields': [name for name in REFERENCE_FIELDS if references[name] is None],
            'binding_reference_status': 'metadata only; presence is not qualification or authorization',
            'teacher_seed_is_training_rng_seed': False}
    result = _json(canonical(dict(body, plan_sha256=digest(body))))
    validate_plan(result)
    return result


def _target_records(path, reference, ids, arm):
    require(type(reference) is dict and reference.get('rows') == 3000
            and type(reference['rows']) is int and _hex(reference.get('sha256'))
            and type(reference.get('bytes')) is int and reference['bytes'] > 0
            and _hex(reference.get('ordered_target_digest')), 'target file reference')
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular target file required')
    rows, observed, size = [], hashlib.sha256(), 0
    with path.open('rb') as stream:
        while raw := stream.readline(MAX_LINE + 1):
            require(len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'target line byte/termination limit')
            observed.update(raw)
            size += len(raw)
            require(size <= reference['bytes'], 'target file exceeded bound bytes')
            row = _json(raw)
            require(type(row) is dict and set(row) == {'id', 'parent_id', 'line', 'logical_exposure',
                    'prompt', 'messages', 'target', 'target_sha256'}, 'target record schema')
            index = len(rows)
            require(index < 3000 and row.get('id') == ids[index]
                    and type(row.get('line')) is int and row['line'] == index + 1
                    and type(row.get('logical_exposure')) is int and row['logical_exposure'] == 0,
                    'target identity/ordinal/exposure/order drift')
            require(type(row.get('parent_id')) is str and row['parent_id'], 'target parent identity')
            prompt = row['prompt']
            require(type(prompt) is list and len(prompt) == 2
                    and [item.get('role') if type(item) is dict else None for item in prompt] == ['system', 'user']
                    and all(set(item) == {'role', 'content'} and type(item['content']) is str for item in prompt),
                    'prepared prompt messages')
            target = row['target']
            require(type(target) is str and target and _hex(row['target_sha256'])
                    and hashlib.sha256(target.encode('utf8')).hexdigest() == row['target_sha256']
                    and row['messages'] == prompt + [{'role': 'assistant', 'content': target}],
                    'target hash/messages drift')
            obj = _json(target)
            require(type(obj) is dict and set(obj) == {'answer'} and type(obj['answer']) is list
                    and obj['answer'] and all(type(item) is int and item >= 0 for item in obj['answer'])
                    and len(set(obj['answer'])) == len(obj['answer']) and canonical(obj).decode('ascii') == target,
                    'compact distinct-index R target')
            if arm == 'canonical':
                require(obj['answer'] == sorted(obj['answer']), 'canonical target is not ascending')
            rows.append(row)
    require(size == reference['bytes'] and observed.hexdigest() == reference['sha256']
            and len(rows) == 3000, 'target file bytes/hash/count drift')
    require(digest([[row['id'], row['target_sha256']] for row in rows]) == reference['ordered_target_digest'],
            'ordered target digest drift')
    return rows


def build_proposal(preparation_path, expected_sha256, arm, *, binding_references=None):
    """Bind the exact caller-pinned unapproved 3000-row preparation; writes nothing.

    Label legality remains evidence supplied by the pinned preparation producer;
    this planner verifies files, records, paired policy and exposure provenance.
    It does not turn those checks or a tokenizer census into qualification.
    Full canonical/policy source hashes are referenced from that preparation;
    only the atomic manifest is reread here. Native ingestion must revalidate
    every referenced source file before executing an approved future scope.
    """
    require(arm in ARMS and type(arm) is str, 'unknown proposal arm')
    require(_hex(expected_sha256), 'expected preparation raw SHA256 required')
    path = Path(preparation_path).resolve()
    preparation, actual, _ = _read_json(path)
    require(actual == expected_sha256, 'preparation raw hash drift')
    require(type(preparation) is dict and preparation.get('schema') == 'R-ORACLE-SUPERVISION-PREPARATION-1'
            and preparation.get('approved') is False and preparation.get('training_authorized') is False,
            'preparation must be unapproved; no grant flags')
    require(type(preparation.get('rows')) is int and preparation['rows'] == 3000
            and type(preparation.get('parent_ids')) is int and preparation['parent_ids'] == 3000
            and preparation.get('order_rule') == ORDER
            and preparation.get('permutation_rule') == [PERMUTE, 1301, 'row_id', 0, 'index']
            and _hex(preparation.get('source_sha256')), 'preparation count/order/policy/producer binding')
    data = preparation.get('data')
    require(type(data) is dict and type(data.get('dataset_root')) is str and _hex(data.get('manifest_sha256'))
            and type(data.get('rows')) is int and data['rows'] == 3000
            and _hex(data.get('selected_ids_sha256')) and type(data.get('files')) is dict,
            'preparation data binding')
    root = Path(data['dataset_root']).resolve()
    entry = data.get('files', {}).get('manifests/atomic_R.json')
    require(type(entry) is dict and _hex(entry.get('sha256')) and type(entry.get('bytes')) is int,
            'bound atomic R file required')
    atomic, atomic_sha, atomic_bytes = _read_json(root / 'manifests/atomic_R.json')
    require(type(atomic) is dict and atomic_sha == entry['sha256'] and atomic_bytes == entry['bytes']
            and digest(atomic) == preparation.get('atomic_R_sha256'), 'raw/parsed atomic manifest drift')
    ids = atomic.get('ids')
    require(atomic.get('name') == 'atomic_R' and type(ids) is list and len(ids) == 3000
            and atomic.get('rows') == 3000 and type(atomic['rows']) is int
            and all(type(row_id) is str and row_id for row_id in ids) and len(set(ids)) == 3000
            and digest(sorted(ids)) == data['selected_ids_sha256'], 'selected atomic identities/count/digest')
    ids = _order(ids)
    targets = preparation.get('targets')
    require(type(targets) is dict and set(targets) == set(ARMS), 'paired target arm references')
    paired = {}
    for name in ARMS:
        reference = targets[name]
        require(type(reference) is dict and type(reference.get('path')) is str
                and Path(reference['path']).is_absolute(), 'absolute target path binding')
        target_path = Path(reference['path'])
        require(target_path.resolve().parent == path.parent and target_path.name == name + '.messages.jsonl',
                'target file outside preparation/arm scope')
        paired[name] = _target_records(target_path, reference, ids, name)
    for original, permuted in zip(paired['canonical'], paired['permuted']):
        require(all(original[key] == permuted[key] for key in ('id', 'parent_id', 'prompt', 'logical_exposure', 'line')),
                'paired common exposure/prompt/parent drift')
        answer = _json(original['target'])['answer']
        expected = sorted(answer, key=lambda index: (digest([PERMUTE, 1301, original['id'], 0, index]), index))
        require(permuted['target'] == canonical({'answer': expected}).decode('ascii'), 'fixed permutation policy drift')
    require(len({row['parent_id'] for row in paired['canonical']}) == 3000, 'distinct prepared parent count')
    policy = ({'kind': 'oracle_first_compact_ascending'} if arm == 'canonical' else
              {'kind': 'fixed_hash_permutation', 'key': [PERMUTE, 1301, 'row_id', 0, 'index'],
               'digest_encoding': 'compact_ascii_json', 'tie_break': 'index'})
    label_metadata = {'target_policy': policy, 'file': targets[arm],
                      'census': preparation.get('census', {}).get(arm) if preparation.get('census') is not None else None}
    binding = {'path': str(path), 'sha256': actual, 'producer_source_sha256': preparation['source_sha256'],
               'data': data, 'parsed_atomic_R_sha256': preparation['atomic_R_sha256']}
    return _finish('unapproved_proposal', arm, paired[arm], label_metadata, binding, binding_references)


def build_synthetic(records, arm='fixture', *, binding_references=None):
    """Small explicit fixture mode, independent of production data or seed choice."""
    require(type(records) is list and records and len(records) % 8 == 0, 'fixture rows must be a positive multiple of eight')
    require(type(arm) is str and arm, 'fixture arm identity')
    require(all(type(row) is dict and type(row.get('id')) is str and row['id']
                and _hex(row.get('target_sha256')) for row in records), 'fixture id/target identities')
    require(len({row['id'] for row in records}) == len(records), 'duplicate fixture row identity')
    index = {row['id']: row for row in records}
    ordered = [index[row_id] for row_id in _order(index)]
    return _finish('synthetic_fixture', arm, ordered, {'target_policy': {'kind': 'declared_synthetic_fixture'},
                   'file': None, 'census': None}, None, binding_references)


def validate_plan(plan):
    require(type(plan) is dict and plan.get('schema') == SCHEMA
            and plan.get('mode') in ('unapproved_proposal', 'synthetic_fixture'), 'CE8 plan mode/schema')
    require(all(plan.get(field) is False for field in ('approved', 'training_authorized', 'native_qualified', 'execution_ready')),
            'CE8 plan never grants execution')
    require({'shared_exposure', 'shared_exposure_sha256', 'ordered_plan', 'ordered_row_ids_sha256',
             'label_binding', 'label_binding_sha256', 'binding_references', 'unbound_reference_fields',
             'teacher_seed_is_training_rng_seed', 'builder_source_sha256', 'preparation'} <= set(plan),
            'CE8 plan binding fields')
    require(_hex(plan.get('plan_sha256'))
            and digest({key: value for key, value in plan.items() if key != 'plan_sha256'}) == plan['plan_sha256'], 'CE8 plan digest drift')
    shared = plan['shared_exposure']
    require(type(shared) is dict and set(shared) == {'schema', 'epochs', 'rows', 'accumulation_rows',
            'microbatch_rows', 'order_rule', 'boundaries'} and type(shared['boundaries']) is list,
            'shared exposure schema')
    require(digest(shared) == plan['shared_exposure_sha256'] and plan['ordered_plan'] == shared['boundaries'], 'shared exposure binding drift')
    rows = []
    for index, boundary in enumerate(shared['boundaries']):
        require(type(boundary) is dict and type(boundary.get('boundary')) is int and boundary['boundary'] == index
                and type(boundary.get('logical_exposure')) is int and boundary['logical_exposure'] == 0
                and type(boundary.get('row_ids')) is list and len(boundary['row_ids']) == 8
                and all(type(row_id) is str and row_id for row_id in boundary['row_ids']), 'exact CE8 boundary identity/order/exposure')
        rows.append(boundary['row_ids'])
    ids = [row_id for boundary in rows for row_id in boundary]
    require(ids and len(set(ids)) == len(ids) and ids == _order(ids)
            and digest(ids) == plan['ordered_row_ids_sha256'] and type(shared['rows']) is int and shared['rows'] == len(ids)
            and shared['epochs'] == 1 and type(shared['epochs']) is int and shared['accumulation_rows'] == 8
            and type(shared['accumulation_rows']) is int and shared['microbatch_rows'] == 1
            and type(shared['microbatch_rows']) is int and shared['order_rule'] == ORDER, 'CE8 full ordered exposure scope')
    expected_schema = SHARED_SCHEMA if plan['mode'] == 'unapproved_proposal' else FIXTURE_SHARED_SCHEMA
    require(shared['schema'] == expected_schema and (plan['mode'] != 'unapproved_proposal' or len(ids) == 3000), 'proposal/fixture count scope')
    labels = plan['label_binding']
    require(type(labels) is dict and {'arm', 'target_policy', 'file', 'census', 'ordered_target_digest',
            'boundaries', 'boundary_labels_sha256'} <= set(labels)
            and type(labels['arm']) is str and labels['arm'] and type(labels['target_policy']) is dict
            and type(labels['boundaries']) is list, 'label binding schema')
    require(digest(labels) == plan['label_binding_sha256'] and digest(labels['boundaries']) == labels['boundary_labels_sha256']
            and len(labels['boundaries']) == len(rows), 'label binding digest/count drift')
    pairs = []
    for index, boundary in enumerate(labels['boundaries']):
        values = boundary['row_target_pairs']
        require(type(boundary.get('boundary')) is int and boundary['boundary'] == index and type(values) is list
                and len(values) == 8 and all(type(value) is list and len(value) == 2 and _hex(value[1]) for value in values)
                and [value[0] for value in values] == rows[index], 'labels differ from CE8 ordered exposure')
        pairs.extend(values)
    require(digest(pairs) == labels['ordered_target_digest'] and _references(plan['binding_references']) == plan['binding_references']
            and plan['teacher_seed_is_training_rng_seed'] is False
            and _hex(plan['builder_source_sha256'])
            and plan['unbound_reference_fields'] == [name for name in REFERENCE_FIELDS if plan['binding_references'][name] is None],
            'label/reference/seed binding drift')
    if plan['mode'] == 'synthetic_fixture':
        require(plan['preparation'] is None and labels['file'] is None and labels['census'] is None
                and labels['target_policy'] == {'kind': 'declared_synthetic_fixture'}, 'fixture source/label scope')
    else:
        require(labels['arm'] in ARMS and type(plan['preparation']) is dict
                and type(labels['file']) is dict, 'proposal source/label scope')
    return rows


def checkpoint_export(plan):
    """Export only fixture metadata for the separately restricted CPU adapter."""
    validate_plan(plan)
    require(plan['mode'] == 'synthetic_fixture', 'unapproved proposal is not a checkpoint execution scope')
    return {'checkpoint_mode': 'CE8_SYNTHETIC_CPU_1', 'synthetic_cpu': True,
            'ordered_plan': _json(canonical(plan['ordered_plan'])),
            'ordered_plan_sha256': digest(plan['ordered_plan']), 'ce_plan_sha256': plan['plan_sha256']}
