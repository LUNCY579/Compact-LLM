"""Native FP32 C/U checkpoint mechanics; imports Torch only inside save/restore.

This module supplies storage mechanics, not a training grant or stage runner.
``gate(stage)`` must enforce the caller's resource/admission envelope. Bindings
pin the exact ordered plan, inventory, start/reference anchors and code/data/
runtime/recipe. Supply ``metadata_reserve`` before constructor writes, and call
``attach_ledger`` before a fresh-process restore. Metadata reservations are
permanent across failures, interrupted writes and rolling retirement.

The returned fence is a symbolic C/U/history snapshot, never a model policy ID.
The runner installs it and rejects every older-generation operation/trajectory.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import time
import uuid

from experiments.maxrl_comp002_pretraining_v1 import warmstart_checkpoint as streams

SCHEMA = 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1'
CHUNK = 1024 ** 2
MAX_JSON = 4 * CHUNK
MAX_JOURNAL = 32 * CHUNK
DISK_FLOOR = 14 * 1024 ** 3
METADATA_RESERVE = 64 * CHUNK
BINDING_FILE = 'CHECKPOINT_BINDINGS.json'
JOURNAL_FILE = 'CHECKPOINT_COSTS.jsonl'
PROGRESS_REQUIRED = {'C', 'U', 'history', 'generation', 'snapshot'}
HISTORY_REQUIRED = {'boundary', 'row_ids', 'updated'}
HEX = frozenset('0123456789abcdef')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def constructor_hash_lower_bound(bindings):
    """No IO/framework work: minimum for the enclosing constructor profile.

    Replay may hash many existing journal rows. Its measured opaque profile
    must cover those in addition to this unavoidable binding digest.
    """
    return {'identity_hash_attempts_lower': 1,
            'identity_hash_bytes_lower': len(canonical(bindings))}


def _payload_sizes(specifications):
    """Exact publication data/file bytes, matching the bounded stream format."""
    header, offset = {}, 0
    for name in sorted(specifications):
        dtype, shape = specifications[name]
        require(dtype in ('F32', 'U8') and type(shape) is list
                and all(type(size) is int and size >= 0 for size in shape), 'payload size schema')
        size = math.prod(shape) * (4 if dtype == 'F32' else 1)
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf8')
    return offset, 8 + len(raw) + (-len(raw)) % 8 + offset


def _hex(value):
    return type(value) is str and len(value) == 64 and set(value) <= HEX


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _plan_rows(bindings):
    plan = bindings.get('ordered_plan')
    if plan is None:
        plan = bindings.get('plan', {}).get('boundaries')
    require(type(plan) is list and plan, 'bindings need an exact nonempty ordered plan')
    rows = []
    for index, boundary in enumerate(plan):
        require(type(boundary) is dict, 'ordered boundary schema')
        ids = boundary.get('row_ids')
        if ids is None:
            require(type(boundary.get('prompts')) is list, 'ordered prompt schema')
            ids = [prompt['row_id'] for prompt in boundary['prompts']]
        require(type(ids) is list and len(ids) == 2 and len(set(ids)) == 2
                and all(type(row) is str and row for row in ids), 'exact B2 ordered row identities required')
        declared_index = boundary.get('logical_boundary', boundary.get('boundary', index))
        require(type(declared_index) is int and declared_index == index, 'boundary order drift')
        rows.append(ids)
    return rows


def validate_progress(progress, plan_rows):
    require(type(progress) is dict and PROGRESS_REQUIRED <= set(progress), 'C/U/progress schema')
    C, U, history = progress['C'], progress['U'], progress['history']
    require(type(C) is type(U) is int and 0 <= U <= C <= len(plan_rows), 'C/U ranges/types')
    require(type(progress['generation']) is int and progress['generation'] >= 0 and _hex(progress['snapshot']),
            'fenced generation/snapshot schema')
    require(type(history) is list and len(history) == C, 'history length differs from C')
    for index, entry in enumerate(history):
        require(type(entry) is dict and HISTORY_REQUIRED <= set(entry), 'history entry schema')
        require(type(entry['boundary']) is int and entry['boundary'] == index
                and entry['row_ids'] == plan_rows[index] and type(entry['row_ids']) is list,
                'history differs from exact ordered plan')
        require(type(entry['updated']) is bool, 'updated history must contain strict booleans')
    require(U == sum(entry['updated'] for entry in history), 'U differs from updated-history count')
    canonical(progress)  # rejects non-JSON/nonfinite optional bound operation/policy fields
    return progress


class NativeCheckpoints:
    def __init__(self, directory, bindings, numel, gate, *, metadata_reserve):
        require(callable(gate) and type(numel) is int and numel > 0, 'gate and positive inventory required')
        require(callable(metadata_reserve), 'metadata reservation callback required before constructor writes')
        self.metadata_reserve = metadata_reserve
        self.directory = Path(directory).resolve()
        self.bindings = _json(canonical(bindings))
        require(type(self.bindings) is dict, 'bindings mapping required')
        for field in ('code', 'data', 'runtime', 'recipe', 'inventory'):
            require(field in self.bindings and self.bindings[field] is not None, 'missing binding: ' + field)
        require(_hex(self.bindings.get('own_start_sha256', self.bindings.get('start_anchor_sha256')))
                and _hex(self.bindings.get('common_anchor_sha256', self.bindings.get('reference_anchor_sha256'))),
                'start/reference anchor identities required')
        self.plan_rows = _plan_rows(self.bindings)
        self.numel, self.gate, self.ledger = numel, gate, None
        self.binding_sha256 = digest(self.bindings)
        self.events, self._io, self._generation = [], {'bytes_read': 0, 'bytes_written': 0}, 0
        self._header_sizes = {}
        self._check_gate('constructor')
        self.directory.mkdir(parents=True, exist_ok=True)
        binding_path = self._path(BINDING_FILE)
        if binding_path.exists():
            require(self._read_json(binding_path) == self.bindings, 'checkpoint bindings changed')
        else:
            self._publish_json(binding_path, self.bindings)
        self._read_journal()

    def attach_ledger(self, ledger):
        """Attach before restore. Costs are never restored from a checkpoint."""
        self.ledger = ledger
        return self

    def _path(self, name):
        path = self.directory / name
        require(path.resolve().parent == self.directory and not path.is_symlink(), 'checkpoint path escapes owner')
        return path

    def _check_gate(self, stage):
        self.gate(stage)
        path = self.directory
        while not path.exists():
            path = path.parent
        free = shutil.disk_usage(path).free
        required = DISK_FLOOR + 28 * self.numel + METADATA_RESERVE
        require(free >= required, f'disk floor/reserve failed: {free} < {required}')
        return {'free_bytes': free, 'required_bytes': required}

    def _read_bytes(self, path, maximum=MAX_JSON):
        chunks, count = [], 0
        with path.open('rb') as stream:
            while True:
                block = stream.read(min(CHUNK, maximum + 1 - count))
                if not block:
                    break
                count += len(block)
                self._io['bytes_read'] += len(block)
                require(count <= maximum, 'metadata/journal byte budget exceeded')
                chunks.append(block)
        return b''.join(chunks)

    def _read_json(self, path):
        return _json(self._read_bytes(path))

    def _publish_json(self, path, value):
        raw = canonical(value) + b'\n'
        require(len(raw) <= MAX_JSON and not os.path.lexists(path), 'metadata budget/create-only publication')
        self._check_gate('metadata_write')
        temporary = self._path('.' + path.name + '.' + uuid.uuid4().hex + '.partial')
        self.metadata_reserve('checkpoint_json', len(raw), write_id=uuid.uuid4().hex, path=str(temporary))
        try:
            with temporary.open('xb') as stream:
                for start in range(0, len(raw), CHUNK):
                    block = raw[start:start + CHUNK]
                    count = stream.write(block)
                    require(count == len(block), 'short metadata write')
                    self._io['bytes_written'] += count
                stream.flush()
                os.fsync(stream.fileno())
            observed = self._read_bytes(temporary)
            # RNG tuples intentionally become JSON arrays. Compare their JSON
            # meaning while also proving the exact serialized bytes, rather
            # than comparing Python tuple/list container types.
            require(observed == raw and canonical(_json(observed)) == raw[:-1],
                    'metadata write/readback mismatch')
            os.link(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_journal(self):
        path = self._path(JOURNAL_FILE)
        if not path.exists():
            return
        raw = self._read_bytes(path, MAX_JOURNAL)
        require(not raw or raw.endswith(b'\n'), 'corrupt/truncated checkpoint cost journal')
        previous = None
        for line in raw.splitlines():
            entry = _json(line)
            body = {key: value for key, value in entry.items() if key != 'sha256'}
            require(body['seq'] == len(self.events) and body['previous'] == previous
                    and body['binding_sha256'] == self.binding_sha256 and digest(body) == entry['sha256'],
                    'checkpoint journal order/binding/hash drift')
            self.events.append(entry)
            self._generation = max(self._generation, entry.get('new_generation', 0))
            previous = entry['sha256']

    def _journal(self, kind, **fields):
        self._check_gate('journal_write')
        body = dict(fields, kind=kind, seq=len(self.events),
                    previous=self.events[-1]['sha256'] if self.events else None,
                    binding_sha256=self.binding_sha256,
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        entry = dict(body, sha256=digest(body))
        raw = canonical(entry) + b'\n'
        path = self._path(JOURNAL_FILE)
        require((path.stat().st_size if path.exists() else 0) + len(raw) <= MAX_JOURNAL, 'checkpoint journal byte ceiling')
        self.metadata_reserve('checkpoint_journal', len(raw), write_id=uuid.uuid4().hex, path=str(path))
        with path.open('ab') as stream:
            for start in range(0, len(raw), CHUNK):
                block = raw[start:start + CHUNK]
                count = stream.write(block)
                require(count == len(block), 'short cost-journal write')
                self._io['bytes_written'] += count
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(entry)
        return entry

    def _emit(self, kind, phase, operation_id, **fields):
        operation = None
        if self.ledger is not None:
            operation = getattr(self.ledger, 'operation', self.ledger if callable(self.ledger) else None)
            require(callable(operation), 'ledger needs operation(kind, phase, **fields)')
        # The cumulative ledger must admit every physical start before this
        # child's journal write. A denied start leaves no unpriced child IO.
        # If a child write then fails, the admitted parent start stays charged.
        if operation is not None and phase == 'started':
            operation(kind, phase, operation_id=operation_id, **fields)
        event = self._journal('operation', operation=kind, phase=phase, operation_id=operation_id, **fields)
        if operation is not None and phase != 'started':
            operation(kind, phase, operation_id=operation_id, checkpoint_event_sha256=event['sha256'], **fields)
        return event

    def cost_snapshot(self):
        counts = {}
        known_wall = 0.0
        pending = {}
        for event in self.events:
            if event['kind'] != 'operation':
                continue
            key = event['operation'] + '_' + event['phase']
            counts[key] = counts.get(key, 0) + 1
            if event['phase'] == 'started':
                pending[event['operation_id']] = event
            else:
                pending.pop(event['operation_id'], None)
                if event['operation'] in ('checkpoint_save', 'checkpoint_restore'):
                    known_wall += event.get('wall_seconds', 0.0)
        return {'counts': counts, 'known_storage_wall_seconds': known_wall,
                'wall_scope': 'top-level save/restore spans; nested identity spans excluded',
                'pending_operations': list(pending.values()),
                'interruption_rule': 'pending IO retains declared byte ceilings; missing wall end remains unknown',
                'journal_path': str(self._path(JOURNAL_FILE)),
                'journal_head': self.events[-1]['sha256'] if self.events else None,
                'physical_costs_rewound': False}

    @contextlib.contextmanager
    def _hash_span(self, state_bytes, reason, attempts=1):
        """Price an explicit streamed identity pass before touching its bytes.

        The ledger charges one builtin fingerprint attempt plus extra passes.
        A partial/failed pass keeps its whole admitted upper bound; no refund
        or invented precise count is claimed after an interruption.
        """
        require(type(state_bytes) is int and state_bytes >= 0
                and type(attempts) is int and attempts > 0, 'hash work bounds')
        identifier, start = uuid.uuid4().hex, time.perf_counter()
        self._emit('fingerprint', 'started', identifier, state_bytes=state_bytes,
                   identity_hash_attempts_upper=attempts - 1, identity_hash_bytes_upper=0,
                   reason=reason, checkpoint_hash_scope=True)
        try:
            yield
        except BaseException as error:
            self._emit('fingerprint', 'failed', identifier,
                       error={'type': type(error).__name__, 'message': str(error)[:1000]},
                       identity_hash_actual_known=False,
                       uncertainty_rule='entire admitted hash work remains charged',
                       wall_seconds=time.perf_counter() - start)
            raise
        self._emit('fingerprint', 'returned', identifier,
                   identity_hash_attempts_actual=attempts, identity_hash_bytes_actual=state_bytes,
                   wall_seconds=time.perf_counter() - start)

    def _metadata_hash_lower_bound(self, kind, progress=None):
        # Metadata hashing is priced by the enclosing opaque operation profile;
        # native state/payload passes have separate observable fingerprint spans.
        inventory = self.bindings['inventory']
        binding_bytes, inventory_bytes = len(canonical(self.bindings)), len(canonical(inventory))
        if kind == 'save':
            # New manifest + verification each hash their body and inventory.
            return {'identity_hash_attempts_lower': 4,
                    'identity_hash_bytes_lower': 2 * (binding_bytes + len(canonical(progress)) + inventory_bytes)}
        require(kind == 'restore', 'metadata hash operation')
        # Verified manifest body/inventory, history digest and fresh fence digest.
        return {'identity_hash_attempts_lower': 4,
                'identity_hash_bytes_lower': binding_bytes + inventory_bytes + len(canonical([])) + 64}

    @staticmethod
    def _groups(optimizer, parameters):
        names = {id(parameter): name for name, parameter in parameters.items()}
        seen, result = [], []
        for group in optimizer.param_groups:
            row = {key: value for key, value in group.items() if key != 'params'}
            require(row.get('amsgrad') is False, 'only Adam m/v state is qualified')
            row['parameter_names'] = [names.get(id(parameter)) for parameter in group['params']]
            require(None not in row['parameter_names'], 'optimizer contains frozen/foreign parameters')
            seen.extend(row['parameter_names'])
            result.append(_json(canonical(row)))
        require(len(seen) == len(set(seen)) and set(seen) == set(parameters), 'optimizer scope/order')
        return result

    def _scope(self, model, optimizer):
        import torch
        require(isinstance(optimizer, torch.optim.AdamW), 'actual AdamW required')
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        require(parameters and sum(value.numel() for value in parameters.values()) == self.numel, 'trainable parameter count')
        devices = {value.device for value in parameters.values()}
        require(len(devices) == 1, 'trainable parameters span devices')
        inventory = []
        for name in sorted(parameters):
            value = parameters[name]
            require(value.dtype == torch.float32 and value.is_contiguous() and value.numel() > 0
                    and value.grad is None, 'clean contiguous FP32 trainable boundary required')
            inventory.append({'name': name, 'shape': list(value.shape), 'dtype': 'F32'})
        declared = self.bindings['inventory']
        require(type(declared) is list, 'inventory binding must list names/shapes/dtypes')
        require(all(type(item) is dict and type(item.get('name')) is str and item['name']
                    and type(item.get('shape')) is list and all(type(size) is int and size > 0 for size in item['shape'])
                    and type(item.get('dtype')) is str for item in declared), 'inventory binding schema')
        normalized = sorted([{'name': item['name'], 'shape': item['shape'],
                              'dtype': 'F32' if item['dtype'] in ('F32', 'float32', 'torch.float32') else item['dtype']}
                             for item in declared], key=lambda item: item['name'])
        require(inventory == normalized, 'trainable inventory binding differs from actual model')
        require(set(optimizer.state) <= set(parameters.values()), 'optimizer has foreign parameter state')
        return parameters, inventory, self._groups(optimizer, parameters), next(iter(devices))

    @staticmethod
    def _tensor_hash(tensor):
        header = canonical({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)})
        result = hashlib.sha256(header + b'\n')
        for block in streams.tensor_chunks(tensor):
            result.update(block)
        return result.hexdigest()

    def _frozen_hash(self, model, parameters):
        active = {id(value) for value in parameters.values()}
        values = [('parameter:' + name, value) for name, value in model.named_parameters(remove_duplicate=False)
                  if id(value) not in active]
        values += [('buffer:' + name, value) for name, value in model.named_buffers(remove_duplicate=False)]
        state_bytes = 0
        for name, value in values:
            require(value.is_contiguous(), 'noncontiguous frozen state unsupported')
            state_bytes += (value.numel() * value.element_size()
                            + len(canonical({'dtype': str(value.dtype), 'shape': list(value.shape)})) + 1
                            + len(name.encode('utf8')) + 1 + 64 + 1)
        with self._hash_span(state_bytes, 'checkpoint_frozen_state'):
            result = hashlib.sha256()
            for name, value in sorted(values):
                result.update(name.encode('utf8') + b'\0' + self._tensor_hash(value).encode('ascii') + b'\n')
            return result.hexdigest()

    @staticmethod
    def _versions(tensors):
        return {name: value._version for name, value in tensors.items()}

    def _capture(self, parameters, optimizer, progress):
        import torch
        tensors, initialized = {}, {}
        for name, parameter in parameters.items():
            tensors['param::' + name] = parameter.detach()
            state = optimizer.state.get(parameter, {})
            initialized[name] = bool(state)
            if progress['U'] == 0:
                require(not state, 'U0 requires empty/lazy Adam state; no inherited moments')
                continue
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'dense Adam state incomplete/extra')
            require(state['step'].dtype == torch.float32 and state['step'].shape == torch.Size([])
                    and float(state['step']) == progress['U'], 'Adam scalar steps must equal U, not C')
            for key in ('exp_avg', 'exp_avg_sq'):
                require(state[key].dtype == torch.float32 and state[key].shape == parameter.shape
                        and state[key].is_contiguous(), 'Adam moment schema')
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                require(state[key].device == parameter.device, 'Adam state is not on parameter device')
                tensors[key + '::' + name] = state[key]
        rng, rng_tensors = streams.rng_state()
        tensors.update(rng_tensors)
        specifications = {name: ['U8' if value.dtype == torch.uint8 else 'F32', list(value.shape)]
                          for name, value in tensors.items()}
        return tensors, initialized, rng, specifications

    @staticmethod
    def _validate_live_payload(tensors):
        import torch
        for name, tensor in tensors.items():
            # The byte view shares the <=1 MiB CPU chunk owned by the generator.
            for block in streams.tensor_chunks(tensor):
                if name.startswith('exp_avg_sq::'):
                    part = torch.frombuffer(block, dtype=torch.float32)
                    require(bool((part >= 0).all()), 'negative live Adam second moment')

    def _latest(self):
        manifests = sorted(self.directory.glob('checkpoint-*.json'))
        for path in manifests:
            require(path.resolve().parent == self.directory and not path.is_symlink(), 'manifest ownership')
        return manifests

    def save(self, model, optimizer, progress, ledger):
        self.attach_ledger(ledger)
        progress = _json(canonical(progress))
        validate_progress(progress, self.plan_rows)
        require(progress['generation'] >= self._generation, 'runner has not installed the restored generation fence')
        operation_id, start = uuid.uuid4().hex, time.perf_counter()
        before_io = dict(self._io)
        maximum_write = 12 * self.numel + METADATA_RESERVE
        self._emit('checkpoint_save', 'started', operation_id, C=progress['C'], U=progress['U'],
                   generation=progress['generation'], symbolic_snapshot=progress['snapshot'],
                   write_bytes_upper=maximum_write, read_bytes_upper=8 * maximum_write,
                   **self._metadata_hash_lower_bound('save', progress))
        try:
            self._check_gate('save_validate')
            import torch
            parameters, inventory, groups, device = self._scope(model, optimizer)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            frozen = self._frozen_hash(model, parameters)
            tensors, initialized, rng, specifications = self._capture(parameters, optimizer, progress)
            self._validate_live_payload(tensors)
            versions = self._versions(tensors)
            manifests = self._latest()
            if len(manifests) >= 2:
                self._verify(manifests[-1], model, optimizer)
                for old in manifests[:-1]:
                    self._check_gate('rolling_retirement')
                    old.with_suffix('.safetensors').unlink()
                    old.unlink()
            sequence = max([event.get('sequence', 0) for event in self.events] + [0]) + 1
            payload = self._path('checkpoint-%09d.safetensors' % sequence)
            self._journal('save_intent', operation_id=operation_id, sequence=sequence,
                          C=progress['C'], U=progress['U'], generation=progress['generation'])
            self._check_gate('payload_write')
            data_bytes, file_bytes = _payload_sizes(specifications)
            with self._hash_span(2 * data_bytes + file_bytes, 'checkpoint_streamed_publication', attempts=3):
                saved = streams.stream_tensors(payload, specifications,
                                               lambda name: streams.tensor_chunks(tensors[name]))
            require(saved['bytes'] == file_bytes, 'publication size differs from analytical hash bound')
            # streams reads the complete file twice (data digest and file digest).
            self._io['bytes_written'] += saved['bytes']
            self._io['bytes_read'] += 2 * saved['bytes']
            require(versions == self._versions(tensors), 'live parameters/moments changed during save')
            require(self._frozen_hash(model, parameters) == frozen, 'frozen state changed during save')
            ledger_snapshot = ledger.snapshot() if hasattr(ledger, 'snapshot') else None
            body = {'schema': SCHEMA, 'sequence': sequence, 'save_operation_id': operation_id,
                    'bindings': self.bindings, 'bindings_sha256': self.binding_sha256,
                    'inventory': inventory, 'inventory_sha256': digest(inventory),
                    'progress': progress, 'Adam_initialized': initialized, 'Adam_scalar_steps': progress['U'],
                    'optimizer_groups': groups, 'frozen_state_sha256': frozen,
                    'module_training': {name: module.training for name, module in model.named_modules()},
                    'rng': rng, 'specifications': specifications, 'payload': saved,
                    'external_cumulative_ledger': ledger_snapshot,
                    'storage_costs_before_publication': self.cost_snapshot()}
            manifest = dict(body, manifest_sha256=digest(body))
            require(len(canonical(manifest)) <= MAX_JSON, 'checkpoint manifest metadata budget')
            self._journal('manifest_intent', sequence=sequence, operation_id=operation_id,
                          manifest_sha256=manifest['manifest_sha256'], payload_sha256=saved['file_sha256'])
            self._publish_json(payload.with_suffix('.json'), manifest)
            self._verify(payload.with_suffix('.json'), model, optimizer)
            self._emit('checkpoint_save', 'returned', operation_id, sequence=sequence,
                       C=progress['C'], U=progress['U'], generation=progress['generation'],
                       symbolic_snapshot=progress['snapshot'], manifest_sha256=manifest['manifest_sha256'],
                       bytes_read=self._io['bytes_read'] - before_io['bytes_read'],
                       bytes_written=self._io['bytes_written'] - before_io['bytes_written'],
                       wall_seconds=time.perf_counter() - start)
            return manifest
        except BaseException as error:
            self._emit('checkpoint_save', 'failed', operation_id, C=progress['C'], U=progress['U'],
                       generation=progress['generation'], symbolic_snapshot=progress['snapshot'],
                       error={'type': type(error).__name__, 'message': str(error)[:1000]},
                       known_bytes_read=self._io['bytes_read'] - before_io['bytes_read'],
                       known_bytes_written=self._io['bytes_written'] - before_io['bytes_written'],
                       write_bytes_upper=maximum_write, read_bytes_upper=8 * maximum_write,
                       wall_seconds=time.perf_counter() - start)
            raise

    def _file_hash(self, path):
        size = path.stat().st_size
        with self._hash_span(size, 'checkpoint_payload_file'):
            result, observed = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while block := stream.read(CHUNK):
                    self._io['bytes_read'] += len(block)
                    observed += len(block)
                    result.update(block)
            require(observed == size, 'payload size changed during hash')
            return result.hexdigest()

    def _header(self, path):
        with path.open('rb') as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, 'truncated payload header')
            length = struct.unpack('<Q', prefix)[0]
            require(0 < length <= CHUNK, 'payload header byte ceiling')
            raw = stream.read(length)
            self._io['bytes_read'] += 8 + len(raw)
            require(len(raw) == length, 'truncated payload header')
        header = _json(raw)
        end = 0
        for name, entry in sorted(header.items(), key=lambda pair: pair[1]['data_offsets'][0]):
            require(type(name) is str and type(entry) is dict and set(entry) == {'dtype', 'shape', 'data_offsets'}, 'payload header schema')
            require(entry['dtype'] in ('F32', 'U8') and type(entry['shape']) is list
                    and all(type(value) is int and value >= 0 for value in entry['shape']), 'payload dtype/shape')
            offsets = entry['data_offsets']
            require(type(offsets) is list and len(offsets) == 2 and all(type(value) is int for value in offsets)
                    and offsets[0] == end and offsets[1] - offsets[0] == math.prod(entry['shape']) * (4 if entry['dtype'] == 'F32' else 1),
                    'payload offsets/size')
            end = offsets[1]
        require(path.stat().st_size == 8 + length + end, 'payload truncation/trailing bytes')
        self._header_sizes[str(path)] = 8 + length
        return header

    def _tensor_chunks(self, path, name):
        # The reused primitive reads one <=1 MiB header and <=1 MiB data chunks.
        self._io['bytes_read'] += self._header_sizes[str(path)]
        for part in streams.file_tensor_chunks(path, name):
            self._io['bytes_read'] += part.numel() * part.element_size()
            yield part

    def _copy_tensor(self, path, name, target):
        flat, offset = target.reshape(-1), 0
        for part in self._tensor_chunks(path, name):
            flat[offset:offset + part.numel()].copy_(part)
            offset += part.numel()
        require(offset == target.numel(), 'restored tensor size differs')

    def _verify(self, manifest_path, model, optimizer):
        self._check_gate('verify')
        manifest = self._read_json(manifest_path)
        require(type(manifest) is dict and 'manifest_sha256' in manifest, 'manifest schema')
        body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
        require(digest(body) == manifest['manifest_sha256'] and manifest['schema'] == SCHEMA
                and manifest['bindings'] == self.bindings and manifest['bindings_sha256'] == self.binding_sha256,
                'manifest source/binding/hash drift')
        intents = [event for event in self.events if event['kind'] == 'manifest_intent'
                   and event['sequence'] == manifest['sequence']]
        require(len(intents) == 1 and intents[0]['manifest_sha256'] == manifest['manifest_sha256']
                and intents[0]['operation_id'] == manifest['save_operation_id'], 'manifest differs from durable publication intent')
        progress = validate_progress(manifest['progress'], self.plan_rows)
        require(type(manifest['Adam_scalar_steps']) is int and manifest['Adam_scalar_steps'] == progress['U'], 'Adam metadata differs from U')
        parameters, inventory, groups, device = self._scope(model, optimizer)
        require(inventory == manifest['inventory'] and digest(inventory) == manifest['inventory_sha256']
                and groups == manifest['optimizer_groups'], 'actual inventory/optimizer recipe drift')
        require(self._frozen_hash(model, parameters) == manifest['frozen_state_sha256'], 'frozen model/source state differs')
        modes = manifest['module_training']
        require(type(modes) is dict and set(modes) == {name for name, module in model.named_modules()}
                and all(type(mode) is bool for mode in modes.values()), 'module training-mode schema')
        initialized = manifest['Adam_initialized']
        require(set(initialized) == set(parameters) and all(type(value) is bool for value in initialized.values())
                and all(value == (progress['U'] > 0) for value in initialized.values()), 'Adam initialization differs from U')
        payload = manifest_path.with_suffix('.safetensors')
        require(payload.resolve().parent == self.directory and not payload.is_symlink(), 'payload ownership')
        require(payload.stat().st_size == manifest['payload']['bytes']
                and self._file_hash(payload) == manifest['payload']['file_sha256'], 'latest checkpoint payload bytes corrupt')
        header = self._header(payload)
        data_bytes = payload.stat().st_size - self._header_sizes[str(payload)]
        require(data_bytes >= 0, 'payload data byte range')
        with self._hash_span(data_bytes, 'checkpoint_payload_data'):
            require(streams.data_digest(payload) == manifest['payload']['data_section_sha256'], 'payload data digest corrupt')
        self._io['bytes_read'] += payload.stat().st_size
        specifications = manifest['specifications']
        require({name: [entry['dtype'], entry['shape']] for name, entry in header.items()} == specifications, 'header/name-bound metadata differs')
        expected = {'rng::cpu'} | {f'rng::cuda::{index}' for index in range(manifest['rng']['cuda_count'])}
        import torch
        import numpy as np
        for name, parameter in parameters.items():
            expected.add('param::' + name)
            require(specifications.get('param::' + name) == ['F32', list(parameter.shape)], 'parameter payload schema')
            if initialized[name]:
                for key in ('exp_avg', 'exp_avg_sq'):
                    expected.add(key + '::' + name)
                    require(specifications.get(key + '::' + name) == ['F32', list(parameter.shape)], 'moment payload schema')
                expected.add('step::' + name)
                require(specifications.get('step::' + name) == ['F32', []], 'scalar-step payload schema')
        require(set(specifications) == expected, 'payload has missing/foreign state')
        require(manifest['rng']['numpy_version'] == np.__version__, 'NumPy RNG runtime differs')
        current_cuda_count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
        require(type(manifest['rng']['cuda_count']) is int and manifest['rng']['cuda_count'] == current_cuda_count,
                'CUDA RNG device inventory differs')
        rng_tensors = {}
        for name, (dtype, shape) in specifications.items():
            if name.startswith('rng::'):
                require(dtype == 'U8' and len(shape) == 1 and 0 < math.prod(shape) <= 65536, 'RNG shape/byte budget')
                rng_tensors[name] = torch.empty(shape, dtype=torch.uint8)
            offset = 0
            for part in self._tensor_chunks(payload, name):
                if part.is_floating_point():
                    require(bool(torch.isfinite(part).all()), 'nonfinite native checkpoint state')
                if name.startswith('exp_avg_sq::'):
                    require(bool((part >= 0).all()), 'negative Adam second moment')
                if name.startswith('step::'):
                    require(part.numel() == 1 and float(part[0]) == progress['U'], 'payload Adam steps differ from U')
                if name in rng_tensors:
                    rng_tensors[name][offset:offset + part.numel()].copy_(part)
                offset += part.numel()
        # Validate RNG encodings on independent generators before touching global RNG/model state.
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value
        random.Random().setstate(tuples(manifest['rng']['python']))
        numpy_state = manifest['rng']['numpy']
        np.random.RandomState().set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        torch.Generator(device='cpu').set_state(rng_tensors['rng::cpu'])
        for index in range(current_cuda_count):
            torch.Generator(device='cuda:' + str(index)).set_state(rng_tensors[f'rng::cuda::{index}'])
        return manifest, payload, parameters, device, rng_tensors

    def restore(self, model, optimizer):
        require(self.ledger is not None, 'attach cumulative ledger before restore')
        manifests = self._latest()
        require(manifests, 'NO_VERIFIED_CHECKPOINT; caller must classify a zero-step attempt separately')
        operation_id, start = uuid.uuid4().hex, time.perf_counter()
        before_io = dict(self._io)
        maximum_read = 8 * (12 * self.numel + METADATA_RESERVE)
        maximum_write = METADATA_RESERVE
        self._emit('checkpoint_restore', 'started', operation_id, read_bytes_upper=maximum_read,
                   write_bytes_upper=maximum_write, **self._metadata_hash_lower_bound('restore'))
        try:
            # Latest corruption fails closed. No older manifest is read as a fallback.
            manifest, payload, parameters, device, rng_tensors = self._verify(manifests[-1], model, optimizer)
            self._check_gate('restore_copy')
            import torch
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            progress = manifest['progress']
            replacements = {}
            if progress['U'] > 0:
                for name, parameter in parameters.items():
                    replacements[parameter] = {'step': torch.empty((), dtype=torch.float32, device=device),
                                               'exp_avg': torch.empty_like(parameter), 'exp_avg_sq': torch.empty_like(parameter)}
                    for key, target in replacements[parameter].items():
                        self._copy_tensor(payload, key + '::' + name, target)
            require(self._file_hash(payload) == manifest['payload']['file_sha256'], 'payload changed before model mutation')
            with torch.no_grad():
                for name, parameter in parameters.items():
                    self._copy_tensor(payload, 'param::' + name, parameter)
                    parameter.grad = None
                optimizer.state.clear()
                optimizer.state.update(replacements)
                streams.restore_rng(manifest['rng'], rng_tensors)
                for name, module in model.named_modules():
                    module.training = manifest['module_training'][name]
            require(self._file_hash(payload) == manifest['payload']['file_sha256'], 'payload changed during restoration; discard live model')
            live_tensors, initialized, _, _ = self._capture(parameters, optimizer, progress)
            require(initialized == manifest['Adam_initialized'], 'restored Adam initialization differs')
            for name, tensor in live_tensors.items():
                tensor_header = canonical({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}) + b'\n'
                state_bytes = 2 * (len(tensor_header) + tensor.numel() * tensor.element_size())
                with self._hash_span(state_bytes, 'checkpoint_restored_saved_and_live_tensor', attempts=2):
                    expected_hash = hashlib.sha256(tensor_header)
                    for part in self._tensor_chunks(payload, name):
                        expected_hash.update(memoryview(part.view(torch.uint8).numpy()).cast('B'))
                    require(self._tensor_hash(tensor) == expected_hash.hexdigest(), 'restored native tensor differs: ' + name)
            external_generation = getattr(self.ledger, 'generation', 0)
            require(type(external_generation) is int and external_generation >= 0, 'external ledger generation schema')
            generation = max(progress['generation'], self._generation, external_generation) + 1
            snapshot = digest({'schema': SCHEMA + '-RESTORE-FENCE', 'manifest_sha256': manifest['manifest_sha256'],
                               'bindings_sha256': self.binding_sha256, 'C': progress['C'], 'U': progress['U'],
                               'history_sha256': digest(progress['history']), 'generation': generation,
                               'restored_snapshot': progress['snapshot']})
            self._journal('restore_fence', new_generation=generation, snapshot=snapshot,
                          C=progress['C'], U=progress['U'], manifest_sha256=manifest['manifest_sha256'])
            self._generation = generation
            self._emit('checkpoint_restore', 'returned', operation_id, C=progress['C'], U=progress['U'],
                       generation=generation, symbolic_snapshot=snapshot,
                       manifest_sha256=manifest['manifest_sha256'],
                       bytes_read=self._io['bytes_read'] - before_io['bytes_read'],
                       bytes_written=self._io['bytes_written'] - before_io['bytes_written'],
                       wall_seconds=time.perf_counter() - start)
            restored = dict(progress, generation=generation, snapshot=snapshot, symbolic_snapshot=snapshot)
            return {'C': restored['C'], 'U': restored['U'], 'history': restored['history'],
                    'progress': restored, 'generation': generation, 'snapshot': snapshot,
                    'symbolic_snapshot': snapshot, 'restored_generation': progress['generation'],
                    'restored_snapshot': progress['snapshot'], 'manifest_sha256': manifest['manifest_sha256'],
                    'cumulative_storage_costs': self.cost_snapshot(),
                    'install_fence_required': True, 'policy_id_rule': 'runner computes actual model fingerprint; symbolic snapshot is not policy_id'}
        except BaseException as error:
            self._emit('checkpoint_restore', 'failed', operation_id,
                       error={'type': type(error).__name__, 'message': str(error)[:1000]},
                       known_bytes_read=self._io['bytes_read'] - before_io['bytes_read'],
                       known_bytes_written=self._io['bytes_written'] - before_io['bytes_written'],
                       read_bytes_upper=maximum_read, write_bytes_upper=maximum_write,
                       wall_seconds=time.perf_counter() - start)
            raise
