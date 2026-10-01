"""Bounded R teacher ingestion beneath a separately validated outer scope.

There is no model loader, CLI, training grant or qualification claim here.
The mandatory admission service runs before paths/IO and again before tokenizer
imports. Registered rendering uses only the pinned S0 tokenizer/renderer.
Observers must price inherited helper IO and opaque verification overhead;
reported file/hash lower bounds are not measurements of every helper operation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
import uuid

from . import ce_plan, ce_update

SCHEMA = 'CE8-INGESTION-1'
SOURCE_SCHEMA = 'CE8-INGESTION-SOURCES-1'
ADMISSION_SCHEMA = 'CE8-OUTER-ADMISSION-1'
ADMISSION_FIELDS = ('schema', 'mode', 'admitted', 'request_sha256', 'scope_sha256', 'phase', 'arm',
                    'static_recipe_sha256', 'source_sha256', 'runtime_sha256')
MAX_JSON = 4 * 1024 ** 2
MAX_LINE = 131072
CHUNK = 1024 ** 2

canonical, digest, require = ce_plan.canonical, ce_plan.digest, ce_plan.require


def _admit(callback, stage, refs):
    require(callable(callback), 'outer scope admission service required before paths/IO/framework')
    value = callback(stage, copy.deepcopy(refs))
    require(type(value) is dict and value.get('schema') == ADMISSION_SCHEMA
            and value.get('admitted') is True and value.get('mode') in ('synthetic_fixture', 'registered_scope')
            and value.get('phase') in ('qualification', 'training') and value.get('arm') == refs['arm']
            and all(ce_plan._hex(value.get(k)) for k in ('request_sha256', 'scope_sha256', 'static_recipe_sha256',
                                                       'source_sha256', 'runtime_sha256')),
            'unapproved outer ingestion scope; no paths/IO/framework permitted')
    if value['mode'] == 'registered_scope':
        from .ce_controller import OuterAdmission, NativeStagePermit
        require(isinstance(callback, OuterAdmission), 'registered ingestion requires an opaque outer stage capability')
        permit = callback.native_permit(stage, copy.deepcopy(refs))
        require(isinstance(permit, NativeStagePermit)
                and permit.matches(stage, refs, value['request_sha256'], value['scope_sha256']),
                'registered outer stage permit mismatch')
    return copy.deepcopy(value)


def _observe(operation, gate, stage, fields, work):
    require(callable(operation) and callable(gate), 'explicit observer and resource gate required')
    identifier, started = uuid.uuid4().hex, time.perf_counter()
    fields = dict(fields); kind = fields.pop('kind', 'input_io')
    operation(kind, 'started', operation_id=identifier, **fields)
    try:
        gate(stage)
        value = work()
    except BaseException as error:
        try:
            operation(kind, 'failed', operation_id=identifier, wall_s=time.perf_counter() - started,
                      error_type=type(error).__name__, error=str(error)[:1000])
        except BaseException as accounting_error:
            error.accounting_failure = {'type': type(accounting_error).__name__,
                                        'message': str(accounting_error)[:1000]}
            raise error from accounting_error
        raise
    operation(kind, 'returned', operation_id=identifier, wall_s=time.perf_counter() - started)
    return value


def _io(operation, gate, stage, fields, work):
    return _observe(operation, gate, stage, dict(fields, kind='input_io'), work)


def _reference(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['path']) is str
            and ce_plan._hex(ref['sha256']) and Path(ref['path']).is_absolute(), 'absolute raw-hash JSON reference required')
    return copy.deepcopy(ref)


def _read_ref(ref, operation, gate):
    ref = _reference(ref); path = Path(ref['path'])
    require(path.is_file() and not path.is_symlink(), 'regular bound JSON required')
    size = path.stat().st_size
    require(0 < size <= MAX_JSON, 'bound JSON byte ceiling')
    def read():
        result = hashlib.sha256(); parts, count = [], 0
        with path.open('rb') as stream:
            while block := stream.read(min(CHUNK, size + 1 - count)):
                count += len(block); require(count <= size, 'bound JSON grew')
                result.update(block); parts.append(block)
        require(count == size and result.hexdigest() == ref['sha256'], 'bound JSON raw hash drift')
        return ce_plan._json(b''.join(parts))
    return _io(operation, gate, 'ingestion_sources', {'path': str(path), 'read_bytes_upper': size,
        'write_bytes_upper': 0, 'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': size}, read)


class VerifiedSources:
    @property
    def receipt(self): return copy.deepcopy(self._receipt)
    @property
    def declared_census(self): return copy.deepcopy(self._expected_census)

    def _target(self, row_id, operation, gate):
        entry = self._index[row_id]; path = Path(self._target_ref['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == self._target_ref['bytes'],
                'teacher file path/size changed')
        def read():
            with path.open('rb') as stream:
                stream.seek(entry['offset']); raw = stream.read(entry['bytes'])
            require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['line_sha256'],
                    'teacher row changed after source verification')
            return ce_plan._json(raw)
        return _io(operation, gate, 'ingestion_boundary', {'path': str(path), 'row_id': row_id,
            'read_bytes_upper': entry['bytes'], 'write_bytes_upper': 0,
            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': entry['bytes']}, read)


def verify_sources(preparation_ref, plan_ref, arm, outer_admission, *, operation, gate):
    """Revalidate the full TRAIN streams and selected teacher; import no framework."""
    refs = {'preparation': preparation_ref, 'plan': plan_ref, 'arm': arm}
    admission = _admit(outer_admission, 'ingestion_sources', refs)
    require(arm in ce_plan.ARMS, 'one named oracle teacher arm required')
    preparation_ref, plan_ref = _reference(preparation_ref), _reference(plan_ref)
    preparation, plan = _read_ref(preparation_ref, operation, gate), _read_ref(plan_ref, operation, gate)
    ordered = ce_plan.validate_plan(plan)
    require(plan['mode'] == 'unapproved_proposal' and plan['label_binding']['arm'] == arm,
            'bound full R teacher proposal plan required; metadata is not execution authority')
    require(preparation.get('schema') == 'R-ORACLE-SUPERVISION-PREPARATION-1'
            and preparation.get('approved') is False and preparation.get('training_authorized') is False
            and preparation.get('rows') == 3000 and type(preparation['rows']) is int
            and preparation.get('parent_ids') == 3000 and type(preparation['parent_ids']) is int
            and preparation.get('order_rule') == ce_plan.ORDER
            and preparation.get('permutation_rule') == [ce_plan.PERMUTE, 1301, 'row_id', 0, 'index'],
            'exact unapproved teacher preparation/policy required')
    bound = plan['preparation']
    require(bound['sha256'] == preparation_ref['sha256']
            and Path(bound['path']).resolve() == Path(preparation_ref['path']).resolve()
            and bound['data'] == preparation['data'], 'plan/preparation source drift')
    from . import data
    from experiments.maxrl_composable_skills_v2 import oracle, corpus
    from experiments.maxrl_comp002_pretraining_v1 import canary
    source = preparation['data']; root = Path(source['dataset_root']).resolve()
    expected_manifest = source['manifest_sha256']
    if admission['mode'] == 'registered_scope':
        require(expected_manifest == data.MANIFEST, 'registered native dataset manifest drift')
    manifest_path = root / 'manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink()
            and manifest_path.stat().st_size <= MAX_JSON, 'bounded dataset manifest')
    file_bytes = manifest_path.stat().st_size + sum(source['files'][key]['bytes'] for key in
        ('manifests/atomic_R.json', data.TRAIN_FILE, data.POLICY_FILE))
    rows, manifests, actual = _io(operation, gate, 'ingestion_sources', {'path': str(root),
        'read_bytes_upper': file_bytes, 'write_bytes_upper': 0,
        'identity_hash_attempts_lower': 4, 'identity_hash_bytes_lower': file_bytes,
        'opaque_source_verification': True}, lambda: data.load_atomic_train(root, ('R',), expected_manifest=expected_manifest))
    require(actual == source and digest(manifests['R']) == preparation['atomic_R_sha256'], 'full TRAIN provenance drift')
    ids = [row_id for batch in ordered for row_id in batch]
    require(ids == ce_plan._order(rows) and len({row['parent_id'] for row in rows.values()}) == 3000,
            'exact ordered atomic R rows/parents required')
    reference = preparation['targets'][arm]
    require(reference == plan['label_binding']['file'] and type(reference['bytes']) is int
            and reference['bytes'] > 0 and reference['rows'] == 3000, 'chosen teacher reference drift')
    target_path = Path(reference['path'])
    require(target_path.is_file() and not target_path.is_symlink()
            and target_path.resolve().parent == Path(preparation_ref['path']).resolve().parent
            and target_path.name == arm + '.messages.jsonl' and target_path.stat().st_size == reference['bytes'],
            'teacher file outside preparation scope')
    index, pairs = {}, []
    def scan():
        hashed, count = hashlib.sha256(), 0
        with target_path.open('rb') as stream:
            for ordinal, row_id in enumerate(ids):
                offset = stream.tell(); raw = stream.readline(MAX_LINE + 1)
                require(raw and len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'teacher line bound/termination')
                count += len(raw); require(count <= reference['bytes'], 'teacher stream grew'); hashed.update(raw)
                target = ce_plan._json(raw); row = rows[row_id]
                require(type(target) is dict and set(target) == {'id', 'parent_id', 'line', 'logical_exposure',
                    'prompt', 'messages', 'target', 'target_sha256'} and target['id'] == row_id
                    and target['parent_id'] == row['parent_id'] and target['prompt'] == row['prompt']
                    and type(target['line']) is int and target['line'] == ordinal + 1
                    and type(target['logical_exposure']) is int and target['logical_exposure'] == 0,
                    'teacher identity/prompt/parent/ordinal drift')
                text = target['target']; require(type(text) is str and text, 'teacher content')
                require(hashlib.sha256(text.encode('utf8')).hexdigest() == target['target_sha256']
                        and target['messages'] == row['prompt'] + [{'role': 'assistant', 'content': text}], 'teacher hash/messages drift')
                expected = oracle.answers(row['verifier']['spec'], limit=1)[0]
                answer = ce_plan._json(expected)['answer']
                require(type(answer) is list and answer and all(type(i) is int for i in answer)
                        and len(set(answer)) == len(answer), 'nonempty distinct oracle R resolved set required')
                if arm == 'permuted':
                    answer = sorted(answer, key=lambda item: (digest([ce_plan.PERMUTE, 1301, row_id, 0, item]), item))
                    expected = canonical({'answer': answer}).decode('ascii')
                require(text == expected and canary.direct_strict(row, text) == corpus.reward(row, text) == 1,
                        'chosen teacher is not the independently verified declared oracle policy')
                index[row_id] = {'offset': offset, 'bytes': len(raw), 'line_sha256': hashlib.sha256(raw).hexdigest()}
                pairs.append([row_id, target['target_sha256']])
            require(not stream.read(1) and count == reference['bytes'] and hashed.hexdigest() == reference['sha256'],
                    'full teacher bytes/hash/count drift')
    _io(operation, gate, 'ingestion_sources', {'path': str(target_path), 'read_bytes_upper': reference['bytes'],
        'write_bytes_upper': 0, 'identity_hash_attempts_lower': 1, 'identity_hash_bytes_lower': reference['bytes'],
        'opaque_oracle_verification': True}, scan)
    require(digest(pairs) == reference['ordered_target_digest'] == plan['label_binding']['ordered_target_digest']
            and pairs == [pair for batch in plan['label_binding']['boundaries'] for pair in batch['row_target_pairs']],
            'ordered teacher target binding drift')
    expected_census = preparation['census'][arm]
    require(expected_census == plan['label_binding']['census'], 'declared tokenizer census drift')
    verified = VerifiedSources()
    verified._plan, verified._refs, verified._admission = copy.deepcopy(plan), copy.deepcopy(refs), admission
    verified._target_ref, verified._index = copy.deepcopy(reference), index
    verified._expected_census = copy.deepcopy(expected_census)
    # Bind this producer without an unpriced read/hash after the outer stage.
    # Reservation/guard failure must precede opening even this small source file.
    source_path = Path(__file__)
    require(source_path.is_file() and not source_path.is_symlink(), 'regular ingestion producer source required')
    source_bytes = source_path.stat().st_size
    require(0 < source_bytes <= MAX_JSON, 'ingestion producer source byte ceiling')
    def hash_source():
        observed, count = hashlib.sha256(), 0
        with source_path.open('rb') as stream:
            while block := stream.read(min(CHUNK, source_bytes + 1 - count)):
                count += len(block); require(count <= source_bytes, 'ingestion producer source grew')
                observed.update(block)
        require(count == source_bytes, 'ingestion producer source size drift')
        return observed.hexdigest()
    producer_sha = _io(operation, gate, 'ingestion_sources', {'path': str(source_path),
        'read_bytes_upper': source_bytes, 'write_bytes_upper': 0,
        'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': source_bytes}, hash_source)
    receipt = {'schema': SOURCE_SCHEMA, 'classification': 'full TRAIN and teacher revalidation only',
        'tokenizer_rendered': False, 'model_loads': 0, 'optimizer_steps': 0, 'training_authorized': False,
        'native_qualified': False, 'arm': arm, 'preparation_ref': preparation_ref, 'plan_ref': plan_ref,
        'plan_sha256': plan['plan_sha256'], 'static_recipe_sha256': admission['static_recipe_sha256'],
        'source': actual, 'teacher': copy.deepcopy(reference), 'declared_census': copy.deepcopy(expected_census),
        'source_sha256': producer_sha}
    verified._receipt = dict(receipt, receipt_sha256=digest(receipt))
    return verified


class PreparedRows:
    """Stream the full census once; retain at most one eight-row token batch."""
    def __init__(self, verified, tokenizer=None, *, outer_admission, operation, gate, renderer=None):
        require(type(verified) is VerifiedSources, 'verified sources required')
        admission = _admit(outer_admission, 'ingestion_tokenizer', verified._refs)
        require(all(admission[k] == verified._admission[k] for k in ADMISSION_FIELDS), 'source/tokenizer outer scope drift')
        gate('ingestion_tokenizer')
        expected = verified._expected_census
        if admission['mode'] == 'registered_scope':
            require(tokenizer is None and renderer is None, 'native tokenizer/renderer injection forbidden')
            from experiments.maxrl_comp002_pretraining_v1 import s0_runtime, s0_warmstart
            require(expected['tokenizer_receipt_sha256'] == s0_runtime.TOKENIZER_RECEIPT_SHA, 'pinned tokenizer receipt drift')
            _read_ref({'path': str(s0_runtime.TOKENIZER_RECEIPT), 'sha256': s0_runtime.TOKENIZER_RECEIPT_SHA}, operation, gate)
            _read_ref({'path': str(s0_runtime.ASSET_RECEIPT), 'sha256': s0_runtime.ASSET_RECEIPT_SHA}, operation, gate)
            tokenizer = _observe(operation, gate, 'ingestion_tokenizer', {'kind': 'tokenizer_load'},
                lambda: s0_runtime.load_tokenizer(s0_runtime.MODEL_DIR, s0_runtime.TOKENIZER_RECEIPT,
                                                  asset_receipt=s0_runtime.ASSET_RECEIPT))
            renderer = s0_warmstart.render_row
        else:
            require(tokenizer is not None and callable(renderer), 'explicit synthetic tokenizer/renderer required')
        self.verified, self.tokenizer, self.renderer = verified, tokenizer, renderer
        self.outer_admission, self.operation, self.gate = outer_admission, operation, gate
        self._cache, self._cache_boundary, self._bindings = [], None, {}
        ids = [rid for batch in verified._plan['ordered_plan'] for rid in batch['row_ids']]
        sequences, renders = hashlib.sha256(b'['), hashlib.sha256(b'[')
        Ps, Cs, witness = [], [], []
        for ordinal, row_id in enumerate(ids):
            item = self._render(row_id); binding = self._binding(item)
            self._bindings[row_id] = binding
            if ordinal: sequences.update(b','); renders.update(b',')
            sequences.update(canonical(item['prompt_ids'] + item['completion_ids'])); renders.update(canonical(item['render_sha256']))
            Ps.append(binding['prompt_tokens']); Cs.append(binding['labeled_tokens']); witness.append(binding)
        sequences.update(b']'); renders.update(b']')
        census = {'rows': len(ids), 'total_prompt_tokens': sum(Ps), 'total_supervised_tokens': sum(Cs),
            'total_sequence_positions': sum(Ps) + sum(Cs), 'max_prompt_tokens': max(Ps), 'max_labeled_tokens': max(Cs),
            'max_sequence_positions': max(r['full_sequence_tokens'] for r in witness),
            'prompt_render_sha256': renders.hexdigest(), 'sequence_ids_sha256': sequences.hexdigest()}
        require(expected['rows_rendered'] == census['rows'] and expected['eos_id'] == 151645
                and expected['max_target_tokens'] == 128 and expected['prompt_tokens'] == {'min': min(Ps), 'max': max(Ps), 'total': sum(Ps)}
                and expected['labelled_tokens'] == {'min': min(Cs), 'max': max(Cs), 'total': sum(Cs)}
                and all(expected[k] == census[k] for k in ('total_prompt_tokens', 'total_supervised_tokens',
                    'total_sequence_positions', 'max_sequence_positions', 'prompt_render_sha256', 'sequence_ids_sha256')),
                'actual streamed tokenizer census differs from pinned preparation')
        witnesses = {key: {'value': census[key], 'row_ids': [r['id'] for r in witness if r[field] == census[key]]}
            for key, field in (('max_prompt_tokens', 'prompt_tokens'), ('max_labeled_tokens', 'labeled_tokens'),
                               ('max_sequence_positions', 'full_sequence_tokens'))}
        receipt = {'schema': SCHEMA, 'classification': 'verified tokenizer-only ingestion; no model efficacy/grant',
            'tokenizer_rendered': True, 'model_loads': 0, 'optimizer_steps': 0, 'native_qualified': False,
            'training_authorized': False, 'arm': verified._admission['arm'], 'plan_sha256': verified._plan['plan_sha256'],
            'static_recipe_sha256': admission['static_recipe_sha256'], 'preparation_ref': verified._refs['preparation'],
            'plan_ref': verified._refs['plan'], 'source_receipt_sha256': verified._receipt['receipt_sha256'],
            'tokenizer_receipt_sha256': expected['tokenizer_receipt_sha256'], 'census': census,
            'prepared_row_bindings': witness, 'witnesses': witnesses}
        self._receipt = dict(receipt, receipt_sha256=digest(receipt))

    @property
    def receipt(self): return copy.deepcopy(self._receipt)
    @property
    def census(self): return copy.deepcopy(self._receipt['census'])
    @property
    def witnesses(self): return copy.deepcopy(self._receipt['witnesses'])
    @property
    def cached_rows(self): return len(self._cache)
    def _binding(self, item):
        # Hash input bytes depend on token magnitudes, id escaping and JSON
        # framing, so preserve their exact total while these arrays are live.
        prepared = {key: item[key] for key in ('id', 'prompt_ids', 'completion_ids')}
        hash_bytes = sum(len(canonical(value)) for value in
                         (item['prompt_ids'], item['completion_ids'], prepared))
        return dict(ce_update.prepared_row_binding(item), render_sha256=item['render_sha256'],
                    target_sha256=item['target_sha256'], prepared_hash_bytes_upper=hash_bytes)
    def _render(self, row_id):
        target = self.verified._target(row_id, self.operation, self.gate)
        item = _observe(self.operation, self.gate, 'ingestion_tokenizer', {'kind': 'tokenizer_render',
            'row_id': row_id, 'target_sha256': target['target_sha256']}, lambda: self.renderer(self.tokenizer, target))
        require(type(item) is dict and item['id'] == row_id and item['target_sha256'] == target['target_sha256']
                and ce_plan._hex(item.get('render_sha256')), 'rendered source identity/content drift')
        binding = ce_update.prepared_row_binding(item)
        require(item['completion_ids'][-1] == 151645 and [i for i, t in enumerate(item['completion_ids']) if t in (151645, 151643)]
                == [binding['labeled_tokens'] - 1] and binding['labeled_tokens'] <= 128
                and binding['labeled_tokens'] <= self.verified._expected_census['labelled_tokens']['max']
                and binding['prompt_tokens'] <= self.verified._expected_census['prompt_tokens']['max']
                and binding['full_sequence_tokens'] <= self.verified._expected_census['max_sequence_positions'], 'EOS/cap/sequence census drift')
        return item
    def iter_boundary(self, C):
        admission = _admit(self.outer_admission, 'ingestion_boundary', self.verified._refs)
        require(all(admission[k] == self.verified._admission[k] for k in ADMISSION_FIELDS), 'boundary outer scope changed')
        batches = self.verified._plan['ordered_plan']
        require(type(C) is int and 0 <= C < len(batches), 'exact boundary cursor')
        if self._cache_boundary != C:
            self._cache = []; self._cache_boundary = None
            items = []
            for row_id in batches[C]['row_ids']:
                item = self._render(row_id)
                require(self._binding(item) == self._bindings[row_id], 'row rendering changed after full census')
                items.append({'id': item['id'], 'prompt_ids': list(item['prompt_ids']), 'completion_ids': list(item['completion_ids'])})
            require(len(items) == 8, 'one full eight-row cache required')
            self._cache, self._cache_boundary = items, C
        return copy.deepcopy(self._cache)
