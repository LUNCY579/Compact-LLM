"""Bounded, framework-free readback of raw qualification numbers and masks.

These reducers prove arithmetic and exact metadata/artifact linkage only.
Expected bindings must come from a separately implemented execution verifier.
Neither paths/hashes, reader callbacks nor these results prove native execution;
there is no authority, dispatch, verifier registration or native PASS here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import uuid

SCHEMA = 'CE8-QUALIFICATION-RAW-EVIDENCE-1'
REDUCTION = 'CE8-RAW-NUMERIC-REDUCTION-1'
LOSS = 'mean_labeled_tokens_per_row_then_equal_rows'
CHUNK = 1024 ** 2
MAX_JSON = 4 * CHUNK
MAX_RESOURCE_JSON = 128 * CHUNK
# ensure_ascii encodes one astral character as two six-byte surrogate escapes.
# 1024 source keys * (512 * 12 +64 hash chars+JSON framing), plus the bounded
# stage/attempt and fixed hash fields, remains below eight MiB.
MAX_BINDING_BYTES = 8 * CHUNK
WITNESS_COUNT = 90  # exact union in the frozen V2 R-teacher registration
BINDING_FIELDS = ('stage', 'request_sha256', 'scope_sha256', 'static_recipe_sha256',
                  'code_sha256', 'worker_identity_sha256', 'attempt_id')
ROW_BINDING_FIELDS = ('id', 'prompt_tokens', 'labeled_tokens', 'full_sequence_tokens',
                     'prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')
BUDGET_FIELDS = ('operation_attempts_upper', 'io_read_bytes_upper',
                 'identity_hash_attempts_upper', 'identity_hash_bytes_upper',
                 'reducer_work_bytes_upper')


class EvidenceRejected(ValueError): pass
class AccountingFailure(RuntimeError): pass


def require(condition, message):
    if not condition: raise EvidenceRejected(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def _hex(value): return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')
def _int(value, minimum=0): return type(value) is int and value >= minimum
def _finite(value): return type(value) in (int, float) and math.isfinite(value)


def resource_json_bound(value):
    require(_int(value, 1) and value <= MAX_RESOURCE_JSON, 'finite bounded resource JSON ceiling')
    return value


def _accounting(error, failure):
    previous = getattr(error, 'accounting_failure', None)
    details = {'type': type(failure).__name__, 'message': str(failure)[:1000]}
    error.accounting_failure = details if previous is None else {'previous': previous, 'latest': details}


def binding(value):
    require(type(value) is dict and set(value) == set(BINDING_FIELDS), 'exact expected evidence binding')
    require(type(value['stage']) is str and 0 < len(value['stage']) <= 128
            and type(value['attempt_id']) is str and 0 < len(value['attempt_id']) <= 128,
            'explicit stage attempt identity')
    require(all(_hex(value[key]) for key in ('request_sha256', 'scope_sha256',
            'static_recipe_sha256', 'worker_identity_sha256')), 'request/scope/recipe/worker hashes')
    require(type(value['code_sha256']) is dict and 0 < len(value['code_sha256']) <= 1024
            and all(type(key) is str and 0 < len(key) <= 512 and _hex(h) for key, h in value['code_sha256'].items()),
            'exact source code map')
    return value


def _json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate raw JSON key'); out[key] = value
        return out
    try:
        return json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(EvidenceRejected('nonfinite raw JSON')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceRejected('invalid raw JSON') from error


def _rows(rows):
    require(type(rows) is list and rows, 'declared actual row bindings required')
    out = {}
    for row in rows:
        require(type(row) is dict and all(key in row for key in ROW_BINDING_FIELDS), 'actual row binding fields')
        value = {key: row[key] for key in ROW_BINDING_FIELDS}
        require(type(value['id']) is str and value['id'] and value['id'] not in out, 'unique actual row identity')
        require(all(_int(value[key], 1) for key in ('prompt_tokens', 'labeled_tokens', 'full_sequence_tokens'))
                and value['full_sequence_tokens'] == value['prompt_tokens'] + value['labeled_tokens']
                and all(_hex(value[key]) for key in ROW_BINDING_FIELDS[4:]), 'actual row lengths/content hashes')
        out[value['id']] = value
    return out


def _tokens(value):
    require(type(value) is list and value and all(_int(token) for token in value), 'raw nonempty token IDs')
    return value


def _completion(ids, eos_ids, target_eos_id):
    _tokens(ids)
    require(len(ids) <= 128 and ids[-1] == target_eos_id
            and [index for index, token in enumerate(ids) if token in eos_ids] == [len(ids) - 1],
            'exactly one terminal target EOS, no internal EOS/PAD')


def _result(kind, observations, references):
    return {'schema': REDUCTION, 'evidence_kind': kind, 'observations': observations,
            'artifact_refs': references, 'native_qualified': False, 'execution_verified': False,
            'classification': 'raw numeric/metadata consistency only; actual process/source execution proof still required'}


class RawEvidenceReader:
    """Permanent finite reservations precede path resolution, reads and hashes.

    ``operation(kind, phase, **fields)`` must perform external admission on
    ``started`` and retain failures. Local reservations are also never refunded.
    No value returned by that callback can establish native qualification.
    Reducer byte-work counts bounded parsing/serialization/arithmetic passes;
    it is an analytical admission contract, not measured CPU time or prices.
    """
    def __init__(self, operation, bounds, owned_root, *, chunk_bytes=CHUNK):
        require(callable(operation), 'pre-reservation observer required')
        require(type(bounds) is dict and set(bounds) == set(BUDGET_FIELDS)
                and all(_int(value) for value in bounds.values()), 'finite complete raw read/hash bounds')
        require(type(chunk_bytes) is int and 4 <= chunk_bytes <= CHUNK and chunk_bytes % 4 == 0,
                'aligned bounded chunk size')
        self.root = Path(owned_root)
        require(self.root.is_absolute(), 'absolute owned artifact root')
        self.operation, self.remaining, self.chunk_bytes = operation, dict(bounds), chunk_bytes

    @contextmanager
    def _overhead(self, reason, *, work_bytes, hash_attempts=0, hash_bytes=0):
        wanted = {'operation_attempts_upper': 1, 'io_read_bytes_upper': 0,
            'identity_hash_attempts_upper': hash_attempts, 'identity_hash_bytes_upper': hash_bytes,
            'reducer_work_bytes_upper': work_bytes}
        require(all(_int(value) and self.remaining[key] >= value for key, value in wanted.items()),
                'reducer verification budget exhausted')
        for key, value in wanted.items(): self.remaining[key] -= value
        identifier, start = uuid.uuid4().hex, time.perf_counter()
        self.operation('qualification_reducer_overhead', 'started', operation_id=identifier,
                       reason=reason, **wanted)
        try:
            yield
        except BaseException as error:
            try:
                self.operation('qualification_reducer_overhead', 'failed', operation_id=identifier,
                    reason=reason, wall_s=time.perf_counter() - start, error_type=type(error).__name__)
            except BaseException as accounting:
                _accounting(error, accounting)
                raise error from accounting
            raise
        self.operation('qualification_reducer_overhead', 'returned', operation_id=identifier,
                       reason=reason, wall_s=time.perf_counter() - start)

    def _prepare(self, reference, expected, role):
        # Binding shape bounds keep its canonical ASCII representation below
        # MAX_BINDING_BYTES even with escaped astral source paths; price first.
        with self._overhead('expected_binding', work_bytes=MAX_BINDING_BYTES,
                            hash_attempts=1, hash_bytes=MAX_BINDING_BYTES):
            binding(expected)
            expected_sha = digest(expected)
        require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes', 'role', 'binding_sha256'}
                and type(reference['path']) is str and Path(reference['path']).is_absolute()
                and _hex(reference['sha256']) and _int(reference['bytes'], 1)
                and reference['role'] == role and reference['binding_sha256'] == expected_sha,
                'exact stage/worker/source/attempt artifact reference')
        wanted = {'operation_attempts_upper': 1, 'io_read_bytes_upper': reference['bytes'] + 1,
                  'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': reference['bytes'] + 1,
                  'reducer_work_bytes_upper': 0}
        require(all(self.remaining[key] >= value for key, value in wanted.items()), 'raw evidence read/hash budget exhausted')
        for key, value in wanted.items(): self.remaining[key] -= value
        identifier = uuid.uuid4().hex
        self.operation('qualification_artifact_readback', 'started', operation_id=identifier,
                       path=reference['path'], role=role, binding_sha256=reference['binding_sha256'], **wanted)
        return identifier

    def _blocks(self, reference, expected, role):
        identifier = self._prepare(reference, expected, role)
        start, count, hasher = time.perf_counter(), 0, hashlib.sha256()
        try:
            path = Path(reference['path']); root = self.root.resolve()
            require(not self.root.is_symlink() and not path.is_symlink()
                    and path.is_file() and path.resolve().is_relative_to(root)
                    and path.stat().st_size == reference['bytes'], 'retained regular artifact path/size/ownership')
            with path.open('rb') as stream:
                while block := stream.read(min(self.chunk_bytes, reference['bytes'] + 1 - count)):
                    count += len(block)
                    require(count <= reference['bytes'], 'raw artifact grew')
                    hasher.update(block)
                    yield block
            require(count == reference['bytes'] and hasher.hexdigest() == reference['sha256'], 'raw artifact bytes/hash mismatch')
        except BaseException as error:
            try:
                self.operation('qualification_artifact_readback', 'failed', operation_id=identifier,
                    bytes_read=count, wall_s=time.perf_counter() - start, error_type=type(error).__name__)
            except BaseException as accounting:
                if isinstance(error, GeneratorExit):
                    # generator.close() suppresses GeneratorExit; surface the
                    # failed accounting separately for the consumer to attach.
                    failure = AccountingFailure('partial raw read accounting failed')
                    failure.accounting_failure = {'type': type(accounting).__name__, 'message': str(accounting)[:1000]}
                    raise failure from accounting
                _accounting(error, accounting)
                raise error from accounting
            raise
        self.operation('qualification_artifact_readback', 'returned', operation_id=identifier,
            bytes_read=count, identity_hash_attempts_actual=1, identity_hash_bytes_actual=count,
            wall_s=time.perf_counter() - start)

    def _envelope(self, reference, expected, kind, role, *, json_bytes_upper=None):
        maximum = MAX_JSON
        if json_bytes_upper is not None:
            require(kind == role == 'resource_io' and type(expected) is dict
                    and expected.get('stage') == 'resource_io_measurement',
                    'explicit JSON ceiling only for resource evidence')
            maximum = resource_json_bound(json_bytes_upper)
        require(_int(reference.get('bytes'), 1) and reference['bytes'] <= maximum, 'raw JSON byte ceiling')
        raw = b''.join(self._blocks(reference, expected, role))
        with self._overhead('JSON_parse_and_linkage', work_bytes=8 * reference['bytes']):
            value = _json(raw)
        require(type(value) is dict and set(value) == set(BINDING_FIELDS) | {'schema', 'evidence_kind', 'payload'}
                and value['schema'] == SCHEMA and value['evidence_kind'] == kind
                and {key: value[key] for key in BINDING_FIELDS} == expected
                and type(value['payload']) is dict, 'raw envelope source/worker/stage/attempt linkage')
        return value['payload']

    def _numeric_budget(self, reference, reason, *, hash_attempts, work_multiplier):
        require(type(reference) is dict and _int(reference.get('bytes'), 1)
                and reference['bytes'] <= MAX_JSON, 'bounded numeric JSON reference')
        require(self.remaining['io_read_bytes_upper'] >= reference['bytes'] + 1,
                'raw evidence read/hash budget exhausted')
        # Six ASCII bytes per input UTF8 byte covers escaped row IDs, with two
        # complete passes for the prompt/completion/prepared or LP preimages.
        return self._overhead(reason, work_bytes=work_multiplier * reference['bytes'],
                              hash_attempts=hash_attempts, hash_bytes=12 * reference['bytes'])

    def read_masks(self, reference, expected, witness_bindings, *, eos_ids=(151645, 151643),
                   target_eos_id=151645, pad_id=151643):
        with self._numeric_budget(reference, 'mask_reduction', hash_attempts=3 * WITNESS_COUNT,
                                  work_multiplier=32):
            return self._masks(reference, expected, witness_bindings, eos_ids, target_eos_id, pad_id)

    def _masks(self, reference, expected, witness_bindings, eos_ids, target_eos_id, pad_id):
        require(expected.get('stage') == 'longest_masks', 'mask evidence stage')
        wanted = _rows(witness_bindings)
        require(len(wanted) == WITNESS_COUNT, 'all 90 registered actual mask witnesses required')
        payload = self._envelope(reference, expected, 'masks', 'masks')
        rows = payload.get('rows')
        require(type(rows) is list and len(rows) == len(wanted)
                and all(type(row) is dict for row in rows)
                and {row.get('id') for row in rows} == set(wanted), 'all exact actual max-witness rows, no duplicates')
        observed = []
        for row in rows:
            prompt, completion = _tokens(row['prompt_ids']), _tokens(row['completion_ids'])
            _completion(completion, eos_ids, target_eos_id)
            P, C = len(prompt), len(completion); L = row.get('padding_tokens')
            require(_int(L) and row.get('padding_side') == ('none' if L == 0 else 'left'), 'explicit valid left/none padding')
            require(type(row.get('input_ids')) is list and len(row['input_ids']) == L + P + C
                    and all(type(token) is int for token in row['input_ids'])
                    and type(row.get('labels')) is list and all(type(token) is int for token in row['labels'])
                    and type(row.get('predictor_positions')) is list
                    and type(row.get('target_positions')) is list
                    and type(row.get('attention_mask')) is list
                    and all(type(value) is int and value in (0, 1) for value in row['attention_mask'])
                    and all(_int(position) for position in row['predictor_positions'] + row['target_positions']),
                    'bounded integer raw input/label/position vectors')
            prepared = {'id': row['id'], 'prompt_ids': prompt, 'completion_ids': completion}
            actual = {'id': row['id'], 'prompt_tokens': P, 'labeled_tokens': C, 'full_sequence_tokens': P + C,
                'prompt_ids_sha256': digest(prompt), 'completion_ids_sha256': digest(completion),
                'prepared_row_sha256': digest(prepared)}
            require(actual == wanted[row['id']], 'raw prompt/completion differs from actual witness binding')
            require(row.get('input_ids') == [pad_id] * L + prompt + completion
                    and row.get('attention_mask') == [0] * L + [1] * (P + C)
                    and row.get('labels') == [-100] * (L + P) + completion
                    and row.get('predictor_positions') == list(range(L + P - 1, L + P + C - 1))
                    and row.get('target_positions') == list(range(L + P, L + P + C)), 'raw one-shift attention/label/pad/EOS mask')
            observed.append({'id': row['id'], 'prepared_row_sha256': actual['prepared_row_sha256'],
                'predictor_positions': list(range(P - 1, P + C - 1)), 'target_positions': list(range(P, P + C)),
                'prompt_selected': 0, 'pad_selected': 0, 'eos_selected': 1, 'raw_padding_tokens': L})
        return _result('masks', {'rows': observed}, [reference])

    def read_likelihood(self, reference, expected, witness_bindings, *, eos_ids=(151645, 151643), target_eos_id=151645):
        with self._numeric_budget(reference, 'likelihood_reduction', hash_attempts=3 * WITNESS_COUNT,
                                  work_multiplier=32):
            return self._likelihood(reference, expected, witness_bindings, eos_ids, target_eos_id)

    def _likelihood(self, reference, expected, witness_bindings, eos_ids, target_eos_id):
        require(expected.get('stage') == 'likelihood_gradient_parity', 'likelihood evidence stage')
        wanted = _rows(witness_bindings)
        require(len(wanted) == WITNESS_COUNT, 'all 90 registered actual likelihood witnesses required')
        payload = self._envelope(reference, expected, 'likelihood', 'likelihood')
        rows = payload.get('rows')
        require(type(rows) is list and len(rows) == len(wanted)
                and all(type(row) is dict for row in rows)
                and {row.get('id') for row in rows} == set(wanted), 'all exact actual likelihood max-witness rows')
        observed = []
        for row in rows:
            actual = wanted[row['id']]; ids = _tokens(row['completion_ids'])
            _completion(ids, eos_ids, target_eos_id)
            require(len(ids) == actual['labeled_tokens'] and digest(ids) == actual['completion_ids_sha256']
                    and row.get('prepared_row_sha256') == actual['prepared_row_sha256'], 'likelihood actual target/EOS identity')
            cached, bulk = row.get('cached_logprobs'), row.get('bulk_logprobs')
            require(type(cached) is list and type(bulk) is list and len(cached) == len(bulk) == len(ids)
                    and all(_finite(value) and value <= 0 for value in cached + bulk), 'all finite selected-token likelihoods, EOS included')
            errors = [abs(left - right) for left, right in zip(cached, bulk)]
            require(all(math.isfinite(value) for value in errors), 'likelihood difference overflow')
            observed.append({'id': row['id'], 'positions': len(ids), 'max_abs_nats': max(errors),
                'cached_values_sha256': digest(cached), 'bulk_values_sha256': digest(bulk)})
        return _result('likelihood', {'per_token_comparisons': observed}, [reference])

    def read_gradient(self, reference, expected, declared_rows, parameter_inventory, *, start_anchor_sha256):
        require(expected.get('stage') == 'likelihood_gradient_parity' and _hex(start_anchor_sha256), 'gradient stage/start anchor')
        wanted = _rows(declared_rows)
        require(len(wanted) == 8, 'separate exact eight-row gradient reference batch')
        payload = self._envelope(reference, expected, 'gradient', 'gradient_manifest')
        require(payload.get('fixed_loss_denominator') == 8 and type(payload['fixed_loss_denominator']) is int
                and payload.get('loss_reduction') == LOSS and payload.get('dtype') == 'F32'
                and payload.get('start_anchor_sha256') == start_anchor_sha256
                and payload.get('rows') == list(wanted.values()), 'gradient objective/row order/token means/anchor identity')
        require(type(parameter_inventory) is list and parameter_inventory, 'actual selected parameter inventory')
        inventory, offset = [], 0
        for entry in parameter_inventory:
            require(type(entry) is dict and set(entry) == {'name', 'shape', 'dtype'}
                    and type(entry['name']) is str and entry['name'] and entry['dtype'] == 'F32'
                    and type(entry['shape']) is list and entry['shape']
                    and all(_int(size, 1) for size in entry['shape']), 'exact FP32 tensor inventory')
            count = math.prod(entry['shape'])
            inventory.append({'name': entry['name'], 'shape': entry['shape'], 'offset_elements': offset, 'numel': count})
            offset += count
        require(len({entry['name'] for entry in inventory}) == len(inventory)
                and payload.get('tensor_inventory') == inventory, 'exact contiguous gradient tensor layout/count')
        left, right = payload.get('reference_ref'), payload.get('candidate_ref')
        require(type(left) is dict and type(right) is dict and left.get('bytes') == right.get('bytes') == 4 * offset,
                'exact gradient raw byte count')
        require(type(left.get('path')) is str and type(right.get('path')) is str
                and Path(left['path']) != Path(right['path']), 'independent gradient artifacts require distinct retained paths')
        reference_norm = difference_norm = maximum = 0.0
        with self._overhead('streamed_gradient_arithmetic', work_bytes=16 * (left['bytes'] + right['bytes'])):
            reference_norm, difference_norm, maximum = self._gradient_values(left, right, expected)
        require(all(math.isfinite(value) for value in (reference_norm, difference_norm, maximum)), 'gradient norm overflow')
        return _result('gradient', {'gradient_reference': {'norm': 'l2', 'fixed_loss_denominator': 8,
            'reference_norm': reference_norm, 'difference_norm': difference_norm, 'max_abs_error': maximum},
            'numel': offset, 'tensor_count': len(inventory)}, [reference, left, right])

    def _gradient_values(self, left, right, expected):
        reference_norm = difference_norm = maximum = 0.0
        a, b = self._blocks(left, expected, 'gradient_reference'), self._blocks(right, expected, 'gradient_candidate')
        primary = None
        try:
            while True:
                x, y = next(a, None), next(b, None)
                require((x is None) == (y is None), 'gradient stream lengths differ')
                if x is None: break
                require(len(x) == len(y) and len(x) % 4 == 0, 'aligned paired gradient chunks')
                count = len(x) // 4
                xs, ys = struct.unpack('<%df' % count, x), struct.unpack('<%df' % count, y)
                require(all(math.isfinite(value) for value in xs + ys), 'nonfinite raw gradient')
                ds = [left_value - right_value for left_value, right_value in zip(xs, ys)]
                reference_norm = math.hypot(reference_norm, _l2(xs))
                difference_norm = math.hypot(difference_norm, _l2(ds))
                maximum = max(maximum, max(map(abs, ds), default=0.0))
            return reference_norm, difference_norm, maximum
        except BaseException as error:
            primary = error
            raise
        finally:
            failures = []
            for stream in (a, b):
                try: stream.close()
                except BaseException as error: failures.append(error)
            if failures:
                details = [{'type': type(error).__name__, 'message': str(error)[:1000],
                            'accounting_failure': getattr(error, 'accounting_failure', None)} for error in failures]
                if primary is not None:
                    previous = getattr(primary, 'accounting_failure', None)
                    primary.accounting_failure = {'stream_close_failures': details, 'previous': previous}
                    raise primary from failures[0]
                failures[0].accounting_failure = {'stream_close_failures': details}
                raise failures[0]


def _l2(values):
    scale = max(map(abs, values), default=0.0)
    if scale == 0: return 0.0
    return scale * math.sqrt(math.fsum((value / scale) ** 2 for value in values))
