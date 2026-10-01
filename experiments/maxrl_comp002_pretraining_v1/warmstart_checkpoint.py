"""Section 12 warmstart durability. No framework import until tensor operations.

Resume restores a complete optimizer boundary and discards later in-memory work.
Replayed work is charged again; the final applied dose follows the saved cursor.
"""
from __future__ import annotations

import hashlib
import datetime
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

from .preflight import digest, require, sha

GIB = 1024 ** 3
DISK_FLOOR = 14 * GIB
METADATA_RESERVE = 64 * 1024 ** 2
CHUNK_BYTES = 1024 ** 2
CLAIM = 'S0_WARMSTART_CLAIM.json'
JOURNAL = 'S0_WARMSTART_JOURNAL.jsonl'
ACTIVE = 'S0_WARMSTART_ACTIVE.json'
DIRECTORY = 'S0_WARMSTART_CHECKPOINTS'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def exclusive_json(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def disk_reserve(numel):
    require(type(numel) is int and numel > 0, 'positive disk inventory required')
    return 28 * numel + METADATA_RESERVE  # one FP32 delta and two parameter/Adam checkpoints


def disk_gate(directory, numel, *, usage=None):
    path = Path(directory).resolve()
    while not path.exists():
        path = path.parent
    free = (shutil.disk_usage if usage is None else usage)(path).free
    needed = DISK_FLOOR + disk_reserve(numel)
    require(free >= needed, f'section 12 disk floor/reserve: free {free} < required {needed}')
    return {'free_bytes': free, 'required_bytes': needed, 'floor_bytes': DISK_FLOOR,
            'ledger_bytes': disk_reserve(numel), 'inventory_numel': numel}


def stream_tensors(path, specifications, chunks):
    """Create-only safetensors publication, <=1 MiB CPU chunks, fsync then verify.

specifications: name -> (safetensors dtype, shape). chunks(name) yields byte
views, whose exact lengths and digest are checked before publication.
"""
    path = Path(path)
    require(not os.path.lexists(path), 'output exists (create-only): ' + str(path))
    sizes = {'F32': 4, 'U8': 1}
    offset, header = 0, {}
    for name in sorted(specifications):
        dtype, shape = specifications[name]
        require(dtype in sizes and name != '__metadata__', 'unsupported checkpoint tensor')
        size = math.prod(shape) * sizes[dtype]
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf8')
    raw += b' ' * ((-len(raw)) % 8)
    require(len(raw) <= 1024 ** 2, 'checkpoint header budget')
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.partial')
    expected = hashlib.sha256()
    try:
        with temporary.open('xb') as stream:
            stream.write(struct.pack('<Q', len(raw)))
            stream.write(raw)
            for name in sorted(header):
                written = 0
                for block in chunks(name):
                    require(len(block) <= CHUNK_BYTES, 'stream chunk budget')
                    written += len(block)
                    require(written <= header[name]['data_offsets'][1] - header[name]['data_offsets'][0], 'tensor grew')
                    stream.write(block)
                    expected.update(block)
                require(written == header[name]['data_offsets'][1] - header[name]['data_offsets'][0], 'tensor truncated')
            stream.flush()
            os.fsync(stream.fileno())
        observed = data_digest(temporary)
        require(observed == expected.hexdigest(), 'checkpoint data digest mismatch')
        os.link(temporary, path)  # exclusive publication; never replaces a destination
        return {'data_section_sha256': observed, 'file_sha256': sha(path), 'bytes': path.stat().st_size}
    finally:
        if temporary.exists():
            temporary.unlink()


def data_digest(path):
    with Path(path).open('rb') as stream:
        first = stream.read(8)
        require(len(first) == 8, 'truncated checkpoint header')
        length = struct.unpack('<Q', first)[0]
        require(0 < length <= 1024 ** 2, 'checkpoint header budget')
        raw = stream.read(length)
        require(len(raw) == length, 'truncated checkpoint header')
        header = json.loads(raw)
        entries = sorted((v['data_offsets'][0], v['data_offsets'][1]) for v in header.values())
        end = 0
        for start, stop in entries:
            require(start == end and stop >= start, 'invalid checkpoint offsets')
            end = stop
        h, remaining = hashlib.sha256(), end
        while remaining:
            block = stream.read(min(CHUNK_BYTES, remaining))
            require(block, 'truncated checkpoint data')
            h.update(block)
            remaining -= len(block)
        require(stream.read(1) == b'', 'checkpoint has trailing bytes')
        return h.hexdigest()


def tensor_chunks(tensor):
    import torch
    flat = tensor.detach().reshape(-1)
    count = CHUNK_BYTES // tensor.element_size()
    for start in range(0, flat.numel(), count):
        cpu = flat[start:start + count].to('cpu').contiguous()
        require(not cpu.is_floating_point() or bool(torch.isfinite(cpu).all()), 'nonfinite checkpoint tensor')
        yield memoryview(cpu.view(torch.uint8).numpy()).cast('B')


def file_tensor_chunks(path, name):
    """Read a tensor in bounded buffers, without touching a whole-file mmap."""
    import torch
    with Path(path).open('rb') as stream:
        raw = stream.read(8)
        require(len(raw) == 8, 'truncated checkpoint header')
        length = struct.unpack('<Q', raw)[0]
        require(0 < length <= 1024 ** 2, 'checkpoint header budget')
        raw = stream.read(length)
        require(len(raw) == length, 'truncated checkpoint header')
        entry = json.loads(raw)[name]
        dtype = {'F32': torch.float32, 'U8': torch.uint8}[entry['dtype']]
        start, end = entry['data_offsets']
        stream.seek(8 + length + start)
        remaining = end - start
        while remaining:
            block = stream.read(min(CHUNK_BYTES, remaining))
            require(block and len(block) % torch.empty((), dtype=dtype).element_size() == 0, 'truncated tensor chunk')
            remaining -= len(block)
            yield torch.frombuffer(bytearray(block), dtype=dtype)


def copy_file_tensor(path, name, target):
    flat, start = target.reshape(-1), 0
    for part in file_tensor_chunks(path, name):
        flat[start:start + part.numel()].copy_(part)
        start += part.numel()
    require(start == target.numel(), 'restored tensor size drift')


def rng_state():
    import torch
    import numpy as np
    numpy_state = np.random.get_state()
    metadata = {'python': random.getstate(), 'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                'numpy_version': np.__version__,
                'cuda_count': torch.cuda.device_count() if torch.cuda.is_initialized() else 0}
    tensors = {'rng::cpu': torch.get_rng_state()}
    for index in range(metadata['cuda_count']):
        tensors[f'rng::cuda::{index}'] = torch.cuda.get_rng_state(index)
    return metadata, tensors


def restore_rng(metadata, tensors):
    import torch
    import numpy as np
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    random.setstate(tuples(metadata['python']))
    value = metadata['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(tensors['rng::cpu'])
    for index in range(metadata['cuda_count']):
        torch.cuda.set_rng_state(tensors[f'rng::cuda::{index}'], index)


class Store:
    def __init__(self, directory, numel, bindings, *, resume=False, reason=None, usage=None, started=None):
        self.root, self.numel, self.bindings = Path(directory).resolve(), numel, json.loads(canonical(bindings))
        self.usage = usage
        self.token = uuid.uuid4().hex
        self.started = time.perf_counter() if started is None else started
        self.started_utc = (datetime.datetime.now(datetime.timezone.utc)
                            - datetime.timedelta(seconds=time.perf_counter() - self.started)).isoformat()
        self.gate()
        self.root.mkdir(parents=True, exist_ok=True)
        claim = self.root / CLAIM
        if resume:
            require(type(reason) is str and reason.strip(), 'resume needs a documented interruption reason')
            require(claim.is_file() and json.loads(claim.read_bytes())['bindings'] == self.bindings, 'attempt claim binding drift')
        else:
            exclusive_json(claim, {'schema': 1, 'bindings': self.bindings, 'token': self.token})
        import psutil
        try:
            boot = psutil.boot_time()
            self.boot_time = float(boot) if type(boot) in (int, float) and math.isfinite(boot) and boot > 0 else None
        except (OSError, NotImplementedError, AttributeError):
            self.boot_time = None
        active = self.root / ACTIVE
        stale = None
        if resume and active.exists():
            stale = json.loads(active.read_bytes())
            try:
                process = psutil.Process(stale['pid'])
                live = process.is_running() and process.create_time() == stale['created']
            except psutil.NoSuchProcess:
                live = False
            require(not live, 'another warmstart invocation is active')
            active.unlink()
        exclusive_json(active, {'token': self.token, 'pid': os.getpid(), 'created': psutil.Process().create_time()})
        self.events = []
        try:
            self.events = self.read_journal()
            if not resume:
                require(not self.events, 'new attempt has an existing journal')
            # A killed invocation has a known elapsed lower bound and an upper
            # bound including the delay until this restart (shared Windows clock).
            for invocation in self.invocations():
                if not invocation['ended'] and invocation['invocation_id'] != self.token:
                    upper = time.perf_counter() - invocation['perf_counter_started']
                    same_boot = self.boot_time is not None and invocation.get('boot_time') == self.boot_time
                    self.append('invocation_uncertain', target_invocation=invocation['invocation_id'],
                                wall_lower_s=invocation['wall_known_s'],
                                wall_upper_s=upper if same_boot and upper >= invocation['wall_known_s'] else None,
                                upper_rule='includes restart delay only with matching known OS boot identity; null across changed/unknown boot or counter epoch')
            self.append('resume' if resume else 'claim', reason=reason, stale_owner=stale,
                        invocation_started_utc=self.started_utc, perf_counter_started=self.started, boot_time=self.boot_time)
        except BaseException:
            self.close()
            raise

    def close(self):
        path = self.root / ACTIVE
        if path.is_file() and json.loads(path.read_bytes()).get('token') == self.token:
            path.unlink()

    def gate(self):
        return disk_gate(self.root, self.numel, usage=self.usage)

    def read_journal(self):
        path = self.root / JOURNAL
        if not path.exists():
            return []
        with path.open('rb') as stream:
            raw = stream.read(16 * 1024 ** 2 + 1)
        require(len(raw) <= 16 * 1024 ** 2 and (not raw or raw.endswith(b'\n')), 'journal truncated or oversized')
        events, previous = [], None
        for line in raw.splitlines():
            entry = json.loads(line)
            body = {k: v for k, v in entry.items() if k != 'sha256'}
            require(body['seq'] == len(events) and body['previous'] == previous and digest(body) == entry['sha256'], 'journal digest/order drift')
            events.append(entry)
            previous = entry['sha256']
        return events

    def append(self, kind, **fields):
        self.gate()
        body = dict(fields, kind=kind, seq=len(self.events), previous=self.events[-1]['sha256'] if self.events else None,
                    invocation_id=self.token,
                    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    invocation_elapsed_s=time.perf_counter() - self.started)
        entry = dict(body, sha256=digest(body))
        with (self.root / JOURNAL).open('ab') as stream:
            stream.write(canonical(entry))
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(entry)
        return entry

    def operation(self, operation, status, **counts):
        self.append('operation', operation=operation, status=status, **counts)

    def cost(self):
        result = {}
        for entry in self.events:
            if entry['kind'] == 'operation':
                key = entry['operation'] + '_' + entry['status']
                result[key] = result.get(key, 0) + 1
                for name in ('input_positions', 'supervised_positions', 'generated_tokens', 'emitted_tokens'):
                    key = name + '_' + entry['status']
                    result[key] = result.get(key, 0) + entry.get(name, 0)
        return result

    def invocations(self):
        groups = {}
        for event in self.events:
            key = event.get('invocation_id')
            if key is None:
                continue
            group = groups.setdefault(key, {'invocation_id': key, 'wall_known_s': 0.0, 'ended': False,
                                           'model_load_attempted': 0, 'model_load_returned': 0,
                                           'perf_counter_started': self.started})
            group['wall_known_s'] = max(group['wall_known_s'], event['invocation_elapsed_s'])
            if event['kind'] in ('claim', 'resume'):
                group['perf_counter_started'] = event['perf_counter_started']
                group['started_utc'] = event['invocation_started_utc']
                group['boot_time'] = event.get('boot_time')
            if event['kind'] == 'invocation_end':
                group.update(ended=True, outcome=event['outcome'])
            if event['kind'] == 'operation' and event['operation'] == 'model_load' and event['status'] in ('attempted', 'returned'):
                group['model_load_' + event['status']] += 1
        for event in self.events:
            if event['kind'] == 'invocation_uncertain' and event['target_invocation'] in groups:
                group = groups[event['target_invocation']]
                group['wall_upper_s'] = event['wall_upper_s']
                group['uncertainty_rule'] = event['upper_rule']
        for group in groups.values():
            group['model_load_pending'] = group['model_load_attempted'] - group['model_load_returned']
        return list(groups.values())

    def bind_envelope(self, binding):
        binding = json.loads(canonical(binding))
        existing = [event for event in self.events if event['kind'] == 'envelope_binding']
        if existing:
            require(len(existing) == 1 and existing[0]['binding'] == binding, 'envelope arm/decoding/prompt-scope drift')
        else:
            self.append('envelope_binding', binding=binding, binding_sha256=digest(binding))
        return digest(binding)

    def envelope_rows(self, binding):
        bound = self.bind_envelope(binding)
        rows = []
        for event in self.events:
            if event['kind'] == 'envelope_row':
                row = event['row']
                require(event['binding_sha256'] == bound and event['row_sha256'] == digest(row), 'envelope row hash/binding drift')
                require(len(rows) < len(binding['prompts']), 'too many envelope rows')
                prompt = binding['prompts'][len(rows)]
                require(row['id'] == prompt['id'] and row['prompt_ids_sha256'] == prompt['prompt_ids_sha256'], 'envelope rows differ from ordered prompt scope')
                rows.append(row)
        return rows

    def envelope_result(self, bound, row_id):
        records = [event for event in self.events if event['kind'] == 'envelope_returned'
                   and event['binding_sha256'] == bound and event['row_id'] == row_id]
        return records[-1] if records else None

    def envelope_cost(self):
        attempts = [event for event in self.events if event['kind'] == 'envelope_attempt']
        result = []
        for attempt in attempts:
            events = [event for event in self.events if event.get('attempt_id') == attempt['attempt_id']]
            returned = next((event for event in events if event['kind'] == 'envelope_returned'), None)
            emitted = [event['token_id'] for event in events if event['kind'] == 'operation'
                       and event['operation'] == 'generation_token' and event['status'] == 'returned']
            ended = any(event['kind'] == 'envelope_stream_end' for event in events)
            known = len(returned['result']['completion_ids']) if returned else len(emitted)
            upper = known if returned or ended else attempt['max_new_tokens']
            result.append({'attempt_id': attempt['attempt_id'], 'row_id': attempt['row_id'],
                           'returned': returned is not None, 'known_emitted_tokens': known,
                           'emitted_tokens_upper_bound': upper,
                           'decoder_forward_attempted': sum(event['kind'] == 'operation' and event['operation'] == 'decoder_forward' and event['status'] == 'attempted' for event in events),
                           'decoder_forward_returned': sum(event['kind'] == 'operation' and event['operation'] == 'decoder_forward' and event['status'] == 'returned' for event in events),
                           'decoder_forward_upper_bound': attempt['max_new_tokens'],
                           'input_positions_upper_bound': attempt['input_positions_upper_bound'],
                           'uncertainty': None if returned else 'partial/unknown completion remains charged; cap is a bound, not an observed count'})
        return {'attempts': result, 'known_emitted_tokens': sum(row['known_emitted_tokens'] for row in result),
                'emitted_tokens_upper_bound': sum(row['emitted_tokens_upper_bound'] for row in result)}

    def _manifests(self):
        folder = self.root / DIRECTORY
        return sorted(folder.glob('checkpoint-*.json')) if folder.exists() else []

    def verify(self, manifest_path):
        from safetensors import safe_open
        raw = manifest_path.read_bytes()
        require(len(raw) <= 1024 ** 2, 'checkpoint metadata budget')
        manifest = json.loads(raw)
        body = {k: v for k, v in manifest.items() if k != 'sha256'}
        require(digest(body) == manifest['sha256'] and manifest['bindings'] == self.bindings, 'checkpoint metadata/binding drift')
        path = manifest_path.with_suffix('.safetensors')
        require(sha(path) == manifest['tensor_file']['file_sha256'] and path.stat().st_size == manifest['tensor_file']['bytes'], 'checkpoint bytes drift')
        require(data_digest(path) == manifest['tensor_file']['data_section_sha256'], 'checkpoint data digest drift')
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            require(sorted(handle.keys()) == sorted(manifest['specifications']), 'checkpoint tensor names drift')
            for name, (dtype, shape) in manifest['specifications'].items():
                view = handle.get_slice(name)
                require(view.get_shape() == shape and view.get_dtype() == dtype, 'checkpoint tensor schema drift')
        return manifest, path

    def save(self, model, optimizer, cursor, progress):
        import torch
        self.gate()
        folder = self.root / DIRECTORY
        folder.mkdir(exist_ok=True)
        # Preserve the verified latest boundary while recycling only the older slot.
        manifests = self._manifests()
        if len(manifests) >= 2:
            self.verify(manifests[-1])
            for old in manifests[:-1]:
                require(old.parent.resolve() == folder.resolve() and not old.is_symlink(), 'checkpoint deletion scope')
                old.with_suffix('.safetensors').unlink()
                old.unlink()
        tensors, states = {}, {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            require(param.dtype == torch.float32 and param.grad is None, 'checkpoint requires clean FP32 optimizer boundary')
            tensors['param::' + name] = param
            state = optimizer.state.get(param, {})
            states[name] = bool(state)
            if state:
                require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'unexpected Adam state')
                for key in state:
                    tensors[key + '::' + name] = state[key]
        rng, rng_tensors = rng_state()
        tensors.update(rng_tensors)
        specs = {name: ['U8' if value.dtype == torch.uint8 else 'F32', list(value.shape)] for name, value in tensors.items()}
        path = folder / ('checkpoint-%06d.safetensors' % cursor)
        self.append('checkpoint_start', cursor=cursor)
        saved = stream_tensors(path, specs, lambda name: tensor_chunks(tensors[name]))
        value = {'schema': 1, 'cursor': cursor, 'bindings': self.bindings, 'specifications': specs,
                 'states': states, 'rng': rng, 'progress': progress, 'tensor_file': saved,
                 'optimizer_groups': [{k: list(v) if isinstance(v, tuple) else v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]}
        exclusive_json(path.with_suffix('.json'), dict(value, sha256=digest(value)))
        manifest, _ = self.verify(path.with_suffix('.json'))
        self.append('checkpoint_committed', cursor=cursor, manifest_sha256=manifest['sha256'])
        return manifest

    def restore(self, model, optimizer, *, initial_progress=None):
        import torch
        import numpy as np
        manifests = self._manifests()
        if not manifests:
            require(initial_progress is not None and not any(event['kind'] == 'step_begin' or
                    (event['kind'] == 'operation' and event['operation'] in ('optimizer_step', 'backward'))
                    for event in self.events), 'resume has no verified checkpoint after training work')
            self.append('base_restart', cursor=0, reason='same claim; fresh original base verified; no training step started; previous load/wall costs retained')
            folder = self.root / DIRECTORY
            for orphan in folder.glob('checkpoint-*.safetensors'):
                require(orphan.stem == 'checkpoint-000000' and not orphan.is_symlink() and orphan.resolve().parent == folder.resolve(), 'initial checkpoint cleanup scope')
                self.append('orphan_removed', cursor=0, file_sha256=sha(orphan), bytes=orphan.stat().st_size)
                orphan.unlink()
            return {'cursor': 0, 'progress': initial_progress, 'base_restart': True}
        value, path = self.verify(manifests[-1])  # never falls back after corruption
        commits = [e for e in self.events if e['kind'] == 'checkpoint_committed' and e['cursor'] == value['cursor']]
        if commits:
            require(commits[-1]['manifest_sha256'] == value['sha256'], 'checkpoint differs from durable journal commitment')
        else:
            require(any(e['kind'] == 'checkpoint_start' and e['cursor'] == value['cursor'] for e in self.events), 'unpublished checkpoint')
        begins = [e['cursor'] for e in self.events if e['kind'] == 'step_begin']
        if begins and max(begins) > value['cursor']:
            self.append('rollback', discarded_through_cursor=max(begins), restored_cursor=value['cursor'],
                        cost_before_replay=self.cost(), reason='reload verified boundary; later parameters/Adam/RNG discarded; repeated work remains charged')
        # An interrupted publication may have a complete tensor file but no
        # metadata. It is never used as a boundary and is removed before replay.
        folder = self.root / DIRECTORY
        for orphan in folder.glob('checkpoint-*.safetensors'):
            cursor = int(orphan.stem.split('-')[1])
            if cursor > value['cursor']:
                require(not orphan.is_symlink() and orphan.resolve().parent == folder.resolve(), 'checkpoint cleanup scope')
                self.append('orphan_removed', cursor=cursor, file_sha256=sha(orphan), bytes=orphan.stat().st_size)
                orphan.unlink()
        require(value['cursor'] == len(value['progress']['steps']), 'checkpoint cursor/count mismatch')
        require([step['index'] for step in value['progress']['steps']] == list(range(value['cursor'])), 'checkpoint dose/order drift')
        params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        require(set(params) == set(value['states']), 'resume trainable inventory drift')
        groups = [{k: list(v) if isinstance(v, tuple) else v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]
        require(groups == value['optimizer_groups'], 'resume optimizer recipe drift')
        require(value['rng']['cuda_count'] == (torch.cuda.device_count() if torch.cuda.is_initialized() else 0), 'resume RNG device drift')
        require(value['rng']['numpy_version'] == np.__version__, 'resume NumPy RNG runtime drift')
        specs = value['specifications']
        expected_names = {'rng::cpu'} | {f'rng::cuda::{i}' for i in range(value['rng']['cuda_count'])}
        for name, param in params.items():
            require(type(value['states'][name]) is bool, 'Adam initialization schema')
            expected_names.add('param::' + name)
            if value['states'][name]:
                expected_names.update(key + '::' + name for key in ('step', 'exp_avg', 'exp_avg_sq'))
            for key in ('param', 'exp_avg', 'exp_avg_sq'):
                if key + '::' + name in specs:
                    require(specs[key + '::' + name] == ['F32', list(param.shape)], 'resume state schema')
            if value['states'][name]:
                require(specs['step::' + name] == ['F32', []], 'Adam scalar step schema')
        require(set(specs) == expected_names, 'checkpoint state inventory drift')
        for name, (dtype, shape) in specs.items():
            if name.startswith('rng::'):
                require(dtype == 'U8' and len(shape) == 1 and 0 < math.prod(shape) <= 65536, 'RNG tensor schema/budget')
                observed = torch.get_rng_state() if name == 'rng::cpu' else torch.cuda.get_rng_state(int(name.rsplit('::', 1)[1]))
                require(list(observed.shape) == shape, 'RNG tensor runtime shape drift')
        # Validate all finite values and scalar steps before changing live state.
        for name in specs:
            for part in file_tensor_chunks(path, name):
                if part.is_floating_point():
                    require(bool(torch.isfinite(part).all()), 'nonfinite checkpoint state')
                if name.startswith('step::'):
                    require(part.numel() == 1 and float(part[0]) == value['cursor'], 'Adam step/cursor drift')
        with torch.no_grad():
            optimizer.state.clear()
            for name, param in params.items():
                copy_file_tensor(path, 'param::' + name, param)
                param.grad = None
                if value['states'][name]:
                    state = {'step': torch.empty((), dtype=torch.float32, device=param.device),
                             'exp_avg': torch.empty_like(param), 'exp_avg_sq': torch.empty_like(param)}
                    for key, target in state.items():
                        copy_file_tensor(path, key + '::' + name, target)
                    optimizer.state[param] = state
            rng = {}
            for name, (dtype, shape) in specs.items():
                if name.startswith('rng::'):
                    rng[name] = torch.empty(shape, dtype=torch.uint8)
                    copy_file_tensor(path, name, rng[name])
            restore_rng(value['rng'], rng)
        require(sha(path) == value['tensor_file']['file_sha256'], 'checkpoint changed during restoration')
        self.append('restored', cursor=value['cursor'], manifest_sha256=value['sha256'])
        return value

    def retire(self, delta):
        """Retire owned checkpoints only after a durable delta and final receipt."""
        require(sha(Path(delta['path'])) == delta['file_sha256'] and data_digest(delta['path']) == delta['data_section_sha256'], 'delta changed before checkpoint retirement')
        self.append('checkpoints_retired', delta_sha256=delta['data_section_sha256'])
        for path in self._manifests():
            self.verify(path)
            path.with_suffix('.safetensors').unlink()
            path.unlink()
