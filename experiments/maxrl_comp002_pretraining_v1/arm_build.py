"""FP32 arm construction and delta bookkeeping (COMP-RS-001 Amendment 1, sections 2, 3, 5, 7).

Never loads a model, never trains, never casts to BF16. Arithmetic is registered:
every base tensor is upcast exactly to FP32, then each component is applied strictly
in the registered order as

    arm = fl32(arm + fl32(float32(alpha) * delta))

i.e. one IEEE single multiply followed by a separate single add, never a fused
multiply-add, so an arm is defined bitwise and its digest is reproducible. FP32
addition is not associative ((-1e8 + 1e8) + 1 = 1 but (-1e8 + 1) + 1e8 = 0), so
the order is part of the arm's definition, not a convenience; the input list order
is irrelevant because components are sorted by the registered order first.

An arm's identity is the sha256 of its per-tensor data digests in sorted key order
(section 2). Norms, inner products and cross terms accumulate in float64 over fixed
chunks; they describe the arm and never enter its identity.

A rebuild passes the registered digest as expected_sha256; a different digest raises
ArmRebuildStop before the arm is returned (section 2: 'a rebuild must reproduce the
digest, or STOP'). Every build receipt records the intra-op thread count and whether
the CPU flushes subnormals, which is probed rather than assumed: flushing changes the
arm's bits whenever a sum lands in the subnormal range.

Donor deltas are stored as FP32 safetensors files: write_delta is create-only and
returns the data-section digest (sha256 over the raw tensor bytes after the header, so
it does not depend on header key order or padding); read_delta refuses a file whose
data section does not reproduce it. The data section alone does not bind tensor names;
pass expected_state_sha256 (the delta's state_digest) to bind those too.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .preflight import digest, require

ORDER = ('R', 'S', 'P')
TIED_PAIR = ('model.embed_tokens.weight', 'lm_head.weight')
UPCASTABLE = (torch.bfloat16, torch.float16, torch.float32)
CHUNK = 1 << 22
READ_CHUNK = 1 << 24
MAX_HEADER = 100_000_000
HEX = frozenset('0123456789abcdef')
ARITHMETIC = ('upcast(base) to FP32 exactly; per component in registered order: '
              'arm = fl32(arm + fl32(float32(alpha) * delta)), unfused; never BF16')
DENORMAL_PROBE = ('FP32 subnormal with bits 0x00000001 times 2.0, on 1 element and on 65536 '
                  'elements (above the intra-op grain): bits 0x00000002 = kept, 0 = flushed')


class ArmRebuildStop(RuntimeError):
    """STOP (section 2): a rebuild did not reproduce the registered arm digest.

    Not a ValueError, so a handler for input refusals cannot absorb it. The rebuilt
    state is discarded; .receipt is the failed rebuild's receipt, for the incident record.
    """

    def __init__(self, expected: str, receipt: dict):
        self.expected = expected
        self.actual = receipt['arm_sha256']
        self.receipt = receipt
        super().__init__(f'STOP (section 2): rebuild digest {self.actual} != registered '
                         f'{expected}; the arm was not returned')


def _hex_digest(value, what: str) -> str:
    require(type(value) is str and len(value) == 64 and set(value) <= HEX,
            what + ' must be a lowercase sha256 hex digest')
    return value


def _denormal_kept(count: int) -> bool:
    bits = torch.ones(count, dtype=torch.int32).view(torch.float32)
    out = (bits * 2.0).view(torch.int32)
    if bool((out == 2).all()):
        return True
    require(bool((out == 0).all()), 'unexpected subnormal probe result')
    return False


def flush_denormal_enabled() -> bool:
    """True iff FP32 subnormals are flushed to zero (torch.set_flush_denormal), probed on CPU."""
    kept = {_denormal_kept(1), _denormal_kept(1 << 16)}
    require(len(kept) == 1, 'flush-denormal state differs between small and pool-sized ops')
    return not kept.pop()


def _runtime() -> dict:
    return {'torch_num_threads': torch.get_num_threads(),
            'flush_denormal': flush_denormal_enabled(), 'flush_denormal_probe': DENORMAL_PROBE}


def tensor_digest(t) -> str:
    """sha256 over the (dtype, shape) header and the contiguous little-endian bytes."""
    require(isinstance(t, torch.Tensor), 'not a tensor')
    require(sys.byteorder == 'little', 'big-endian host unsupported')
    header = json.dumps({'dtype': str(t.dtype), 'shape': list(t.shape)},
                        sort_keys=True, separators=(',', ':'))
    flat = t.detach().to('cpu').contiguous().reshape(-1)
    result = hashlib.sha256(header.encode('ascii') + b'\n')
    result.update(memoryview(flat.view(torch.uint8).numpy()))
    return result.hexdigest()


def _require_keys(state, what: str) -> None:
    require(isinstance(state, dict) and state, f'empty/invalid {what}')
    require(all(type(k) is str and k and '\0' not in k and '\n' not in k for k in state),
            f'invalid tensor key in {what}')


def state_digest(state: dict) -> str:
    """Arm identity: sha256 over sorted keys of key + tensor_digest."""
    _require_keys(state, 'state')
    result = hashlib.sha256()
    for key in sorted(state):
        result.update(key.encode('utf-8') + b'\0' + tensor_digest(state[key]).encode('ascii')
                      + b'\n')
    return result.hexdigest()


def _flat_chunks(*tensors):
    flats = [t.detach().reshape(-1) for t in tensors]
    for start in range(0, flats[0].numel(), CHUNK):
        yield [f[start:start + CHUNK].to(torch.float64) for f in flats]


def _sumsq(t) -> float:
    total = 0.0
    for (x,) in _flat_chunks(t):
        total += float((x * x).sum())
    return total


def _dot(a, b) -> float:
    total = 0.0
    for x, y in _flat_chunks(a, b):
        total += float((x * y).sum())
    return total


def _diff_sumsq(a, b) -> float:
    total = 0.0
    for x, y in _flat_chunks(a, b):
        d = x - y
        total += float((d * d).sum())
    return total


def _require_upcastable(t, what: str) -> None:
    require(isinstance(t, torch.Tensor) and t.dtype in UPCASTABLE,
            'base dtype not exactly upcastable to FP32: ' + what)


def _require_float_state(state, what: str) -> None:
    _require_keys(state, what)
    for key, t in state.items():
        require(isinstance(t, torch.Tensor) and t.is_floating_point(),
                f'non-floating tensor in {what}: {key}')
        require(bool(torch.isfinite(t).all()), f'nonfinite tensor in {what}: {key}')


def global_norm(delta: dict) -> float:
    """Frobenius norm over all tensors, accumulated in float64 in sorted key order."""
    _require_float_state(delta, 'delta')
    total = 0.0
    for key in sorted(delta):
        total += _sumsq(delta[key])
    return math.sqrt(total)


def extract_delta(final_fp32: dict, base: dict, inventory: list[str]) -> dict:
    """FP32 donor delta over the trainable inventory: final - upcast(base).

    final_fp32 is the FP32 master and must hold exactly the inventory keys; base may
    hold more (frozen tensors). Refuses extra/missing keys, shape mismatch, a
    non-FP32 final and any non-finite value.
    """
    require(isinstance(inventory, (list, tuple)) and inventory
            and all(type(k) is str and k for k in inventory)
            and len(set(inventory)) == len(inventory), 'invalid inventory')
    require(isinstance(final_fp32, dict) and isinstance(base, dict), 'states must be dicts')
    extra = sorted(set(final_fp32) - set(inventory), key=str)
    if extra:
        raise ValueError('key outside inventory: ' + str(extra[0]))
    delta = {}
    for key in sorted(inventory):
        require(key in final_fp32, 'missing inventory key in final: ' + key)
        require(key in base, 'missing inventory key in base: ' + key)
        final, start = final_fp32[key], base[key]
        require(isinstance(final, torch.Tensor) and final.dtype == torch.float32,
                'final not FP32: ' + key)
        _require_upcastable(start, key)
        require(tuple(final.shape) == tuple(start.shape), 'shape mismatch: ' + key)
        require(final.device == start.device, 'device mismatch: ' + key)
        value = final.detach() - start.detach().to(torch.float32)
        require(bool(torch.isfinite(value).all()), 'nonfinite delta: ' + key)
        delta[key] = value
    return delta


def _alpha32(alpha) -> float:
    require(isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            and math.isfinite(alpha) and alpha > 0, 'alpha must be a finite positive number')
    value = float(torch.tensor(float(alpha), dtype=torch.float32))
    require(math.isfinite(value) and value > 0, 'alpha not representable in FP32')
    return value


def _plan(components, order) -> list:
    require(isinstance(order, (list, tuple)) and order
            and all(type(n) is str and n for n in order)
            and len(set(order)) == len(order), 'invalid registered order')
    require(isinstance(components, (list, tuple)), 'components must be a list')
    seen = set()
    plan = []
    for item in components:
        require(isinstance(item, (list, tuple)) and len(item) == 3,
                'component must be (name, source, alpha)')
        name, source, alpha = item
        require(type(name) is str and name in order, 'unknown component name: ' + repr(name))
        require(name not in seen, 'duplicate component: ' + name)
        seen.add(name)
        alpha32 = _alpha32(alpha)
        plan.append((name, source, float(alpha), alpha32))
    plan.sort(key=lambda entry: order.index(entry[0]))
    return plan


def _tied_alias(state, get_base, parts, base_keys) -> dict:
    embed, head = TIED_PAIR
    touched = [name for name, _, _, keys, _ in parts if embed in keys or head in keys]
    if embed not in base_keys or head not in base_keys:
        return {'pair_present': False, 'tied_in_base': False, 'touched_by': touched,
                'equal_in_arm': None}
    first, second = get_base(embed), get_base(head)
    tied = first is second or (first.dtype == second.dtype and first.shape == second.shape
                               and bool(torch.equal(first, second)))
    equal = bool(torch.equal(state[embed], state[head]))
    require(not (tied and touched and not equal),
            'tied embed_tokens/lm_head alias broken by delta: ' + ','.join(touched))
    return {'pair_present': True, 'tied_in_base': tied, 'touched_by': touched,
            'equal_in_arm': equal}


def _build(base_keys, get_base, parts, order, expected=None) -> tuple[dict, dict]:
    """Shared core for in-memory and streamed builds; parts are already in registered order.

    parts: (name, alpha, alpha_fp32, key_set, get_tensor). One base tensor and at most
    one tensor per component are held at a time besides the arm under construction.
    expected: the registered arm digest for a rebuild (already validated), or None.
    """
    runtime = _runtime()
    keys = sorted(base_keys)
    require(keys and all(type(k) is str and k and '\0' not in k and '\n' not in k
                         for k in keys), 'empty/invalid base state')
    known = set(keys)
    for name, _, _, part_keys, _ in parts:
        require(part_keys, 'empty delta: ' + name)
        stray = sorted(part_keys - known, key=str)
        if stray:
            raise ValueError(f'delta key not in base ({name}): {stray[0]}')
    count = len(parts)
    component_sq = [0.0] * count
    cross = {(i, j): 0.0 for i in range(count) for j in range(i + 1, count)}
    per_tensor = {}
    total_sq = 0.0
    base_dtypes = set()
    state = {}
    for key in keys:
        start = get_base(key)
        _require_upcastable(start, key)
        base_dtypes.add(str(start.dtype))
        arm = start.detach().to(dtype=torch.float32, copy=True)
        applied = []
        for index, (name, _, alpha32, part_keys, get) in enumerate(parts):
            if key not in part_keys:
                continue
            delta = get(key)
            require(isinstance(delta, torch.Tensor) and delta.dtype == torch.float32,
                    f'delta not FP32 ({name}): {key}')
            require(tuple(delta.shape) == tuple(arm.shape), f'shape mismatch ({name}): {key}')
            require(delta.device == arm.device, f'device mismatch ({name}): {key}')
            require(bool(torch.isfinite(delta).all()), f'nonfinite delta ({name}): {key}')
            arm.add_(delta.mul(alpha32))
            applied.append((index, delta))
        require(arm.dtype == torch.float32 and bool(torch.isfinite(arm).all()),
                'nonfinite arm tensor: ' + key)
        if applied:
            for index, delta in applied:
                component_sq[index] += _sumsq(delta)
            for x in range(len(applied)):
                for y in range(x + 1, len(applied)):
                    cross[(applied[x][0], applied[y][0])] += _dot(applied[x][1], applied[y][1])
            squared = _diff_sumsq(arm, start)
            per_tensor[key] = math.sqrt(squared)
            total_sq += squared
        state[key] = arm
        del applied, start
    require(all(t.dtype == torch.float32 for t in state.values()), 'arm dtype drift')
    tied = _tied_alias(state, get_base, parts, known)
    names = [part[0] for part in parts]
    components = []
    for index, (name, alpha, alpha32, part_keys, _) in enumerate(parts):
        norm = math.sqrt(component_sq[index])
        components.append({'name': name, 'alpha': alpha, 'alpha_fp32': alpha32,
                           'tensors': len(part_keys), 'keys_sha256': digest(sorted(part_keys)),
                           'delta_norm': norm, 'scaled_norm': alpha32 * norm})
    cross_terms = {f'{names[i]}|{names[j]}': parts[i][2] * parts[j][2] * value
                   for (i, j), value in cross.items()}
    from_components = (sum(parts[i][2] ** 2 * component_sq[i] for i in range(count))
                       + 2.0 * sum(cross_terms.values()))
    receipt = {
        'arithmetic': ARITHMETIC, 'accumulator': 'float64',
        'registered_order': list(order), 'applied_order': names,
        'components': components, 'cross_terms': cross_terms,
        'per_tensor_displacement_norm': per_tensor,
        'total_displacement_norm': math.sqrt(total_sq),
        'total_norm_from_components': math.sqrt(max(from_components, 0.0)),
        'tensors': len(state), 'displaced_tensors': len(per_tensor),
        'base_dtypes': sorted(base_dtypes), 'dtype': str(torch.float32), 'all_finite': True,
        'tied_alias': tied, 'runtime': runtime, 'arm_sha256': state_digest(state),
        'rebuild_verified_against': None}
    if expected is not None:
        if receipt['arm_sha256'] != expected:
            state.clear()  # the traceback keeps this frame alive; do not keep the arm with it
            raise ArmRebuildStop(expected, receipt)
        receipt['rebuild_verified_against'] = expected
    return state, receipt


def build_arm(base: dict, components: list[tuple[str, dict, float]],
              order=ORDER, *, expected_sha256: str | None = None) -> tuple[dict, dict]:
    """theta0 upcast to FP32 plus sum of alpha*delta, applied in the registered order.

    components: (name, fp32_delta_dict, alpha); names must be in `order`, no duplicates.
    An empty component list builds arm B (the FP32 upcast source). The base is never
    mutated. Returns (fp32_state, receipt). A rebuild passes the registered digest as
    expected_sha256: a mismatch raises ArmRebuildStop and returns nothing.
    """
    expected = None if expected_sha256 is None else _hex_digest(expected_sha256,
                                                                 'expected arm digest')
    _require_keys(base, 'base state')
    plan = _plan(components, order)
    parts = []
    for name, delta, alpha, alpha32 in plan:
        _require_keys(delta, 'delta ' + name)
        parts.append((name, alpha, alpha32, set(delta), delta.__getitem__))
    return _build(list(base), base.__getitem__, parts, tuple(order), expected)


def build_arm_streaming(base_path, components: list[tuple[str, object, float]],
                        device: str = 'cpu', order=ORDER, *,
                        expected_sha256: str | None = None) -> tuple[dict, dict]:
    """Same arm and receipt as build_arm, reading safetensors files tensor by tensor.

    components: (name, delta_safetensors_path, alpha). No delta file is materialised
    whole: each delta tensor is read once, added into the upcast base tensor on
    `device`, and released. Only device='cpu' is exercised by the tests. expected_sha256
    as in build_arm. The delta files' data-section digests are not checked here; verify
    them with delta_data_digest before a build that must bind them.
    """
    expected = None if expected_sha256 is None else _hex_digest(expected_sha256,
                                                                 'expected arm digest')
    require(isinstance(device, str) and device, 'invalid device')
    torch.device(device)
    plan = _plan(components, order)
    require(isinstance(base_path, (str, Path)) and Path(base_path).is_file(),
            'missing base file')
    with ExitStack() as stack:
        base_file = stack.enter_context(safe_open(str(base_path), framework='pt', device=device))
        parts = []
        for name, path, alpha, alpha32 in plan:
            require(isinstance(path, (str, Path)) and Path(path).is_file(),
                    f'missing delta file ({name})')
            handle = stack.enter_context(safe_open(str(path), framework='pt', device=device))
            parts.append((name, alpha, alpha32, set(handle.keys()), handle.get_tensor))
        return _build(list(base_file.keys()), base_file.get_tensor, parts, tuple(order),
                      expected)


def _raw_bytes(t) -> memoryview:
    return memoryview(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy())


def _read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    require(len(data) == count, 'truncated safetensors file')
    return data


def _unique_pairs(pairs) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key in safetensors header')
        result[key] = value
    return result


def _nonfinite_json(value):
    raise ValueError('nonfinite JSON in safetensors header: ' + value)


def _delta_header(stream, size: int) -> list:
    """Parse and check an FP32 delta file's header: (begin, end, key, shape) in data order.

    Every tensor is F32, its byte length matches its shape, and the entries tile the data
    section from offset 0 to the end of the file with no gap, overlap or trailing byte.
    """
    require(size >= 8, 'not a safetensors file')
    (length,) = struct.unpack('<Q', _read_exact(stream, 8))
    require(2 <= length <= MAX_HEADER and 8 + length <= size, 'invalid safetensors header length')
    try:
        header = json.loads(_read_exact(stream, length).decode('utf-8'),
                            object_pairs_hook=_unique_pairs, parse_constant=_nonfinite_json)
    except UnicodeDecodeError as error:
        raise ValueError('invalid safetensors header') from error
    require(isinstance(header, dict), 'invalid safetensors header')
    metadata = header.pop('__metadata__', None)
    require(metadata is None or (isinstance(metadata, dict) and all(
        type(k) is str and type(v) is str for k, v in metadata.items())),
        'invalid safetensors metadata')
    entries = []
    for key, spec in header.items():
        require(key and '\0' not in key and '\n' not in key, 'invalid tensor key in delta file')
        require(isinstance(spec, dict) and set(spec) == {'dtype', 'shape', 'data_offsets'},
                'invalid tensor entry: ' + key)
        require(spec['dtype'] == 'F32', 'delta file tensor not FP32: ' + key)
        shape, offsets = spec['shape'], spec['data_offsets']
        require(isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape),
                'invalid shape: ' + key)
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(o) is int for o in offsets) and 0 <= offsets[0] <= offsets[1],
                'invalid data offsets: ' + key)
        require(offsets[1] - offsets[0] == 4 * math.prod(shape),
                'data offsets disagree with shape: ' + key)
        entries.append((offsets[0], offsets[1], key, tuple(shape)))
    require(entries, 'empty delta file')
    entries.sort()
    position = 0
    for begin, end, key, _ in entries:
        require(begin == position, 'data section not tiled (gap or overlap): ' + key)
        position = end
    require(8 + length + position == size, 'data section length disagrees with the file size')
    return entries


def _scan_delta(path, keep: bool) -> tuple[str, list, dict]:
    """One pass over a delta file: (data-section sha256, entries, tensors if keep).

    The digest is taken over exactly the bytes the returned tensors are read into.
    """
    require(sys.byteorder == 'little', 'big-endian host unsupported')
    require(isinstance(path, (str, Path)) and Path(path).is_file(), 'missing delta file')
    result = hashlib.sha256()
    tensors = {}
    with open(path, 'rb') as stream:
        entries = _delta_header(stream, os.fstat(stream.fileno()).st_size)
        for begin, end, key, shape in entries:
            if keep:
                tensor = torch.empty(shape, dtype=torch.float32)
                buffer = memoryview(tensor.reshape(-1).numpy()).cast('B')
                filled = 0
                while filled < len(buffer):
                    count = stream.readinto(buffer[filled:])
                    require(count, 'truncated safetensors file')
                    filled += count
                result.update(buffer)
                tensors[key] = tensor
            else:
                remaining = end - begin
                while remaining:
                    chunk = _read_exact(stream, min(READ_CHUNK, remaining))
                    result.update(chunk)
                    remaining -= len(chunk)
        require(stream.read(1) == b'', 'trailing bytes after the data section')
    return result.hexdigest(), entries, tensors


def delta_data_digest(path) -> str:
    """Data-section sha256 of an FP32 delta file, streamed; the header is checked, not hashed."""
    return _scan_delta(path, keep=False)[0]


def write_delta(path, delta: dict) -> str:
    """Write an FP32 delta as a new safetensors file and return its data-section digest.

    Create-only: an existing path is refused, and the file appears at `path` only by a
    hard link from a verified temporary in the same directory, which fails rather than
    replace a file created meanwhile. The returned digest is read back from disk and
    checked against the source tensors' bytes in the file's data order.
    """
    require(isinstance(path, (str, Path)) and str(path), 'invalid delta path')
    target = Path(path)
    _require_keys(delta, 'delta')
    for key, t in delta.items():
        require(key != '__metadata__', 'reserved safetensors key in delta')
        require(isinstance(t, torch.Tensor) and t.dtype == torch.float32, 'delta not FP32: ' + key)
        require(t.device.type == 'cpu', 'delta not on CPU: ' + key)
        require(bool(torch.isfinite(t).all()), 'nonfinite delta: ' + key)
    require(target.parent.is_dir(), 'missing delta directory')
    require(not os.path.lexists(target), 'delta file exists (create-only): ' + str(target))
    handle, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.partial',
                                    dir=target.parent)
    os.close(handle)
    temp = Path(name)
    try:
        save_file({key: delta[key].detach().contiguous() for key in sorted(delta)}, name)
        with open(temp, 'rb+') as stream:
            os.fsync(stream.fileno())
        written, entries, _ = _scan_delta(temp, keep=False)
        require({key: shape for _, _, key, shape in entries}
                == {key: tuple(t.shape) for key, t in delta.items()},
                'written delta header disagrees with the source')
        source = hashlib.sha256()
        for _, _, key, _ in entries:
            source.update(_raw_bytes(delta[key]))
        require(source.hexdigest() == written, 'written delta bytes disagree with the source')
        try:
            os.link(temp, target)
        except FileExistsError:
            raise ValueError('delta file exists (create-only): ' + str(target)) from None
    finally:
        temp.unlink(missing_ok=True)
    return written


def read_delta(path, expected_digest: str, expected_state_sha256: str | None = None) -> dict:
    """Load an FP32 delta file, refusing it unless its data section reproduces expected_digest.

    The data-section digest binds the tensor bytes but not their names; a header that
    swaps the names of two same-shape tensors keeps it. expected_state_sha256 (the
    delta's state_digest) also binds names, dtypes and shapes.
    """
    expected = _hex_digest(expected_digest, 'expected delta digest')
    expected_state = (None if expected_state_sha256 is None
                      else _hex_digest(expected_state_sha256, 'expected delta state digest'))
    actual, _, tensors = _scan_delta(path, keep=True)
    if actual != expected:
        tensors.clear()
        raise ValueError(f'delta data-section digest mismatch: {actual} != {expected}')
    _require_float_state(tensors, 'delta file')
    if expected_state is not None and state_digest(tensors) != expected_state:
        tensors.clear()
        raise ValueError('delta state digest mismatch (names, shapes or bytes)')
    return {key: tensors[key] for key in sorted(tensors)}


def _target(target_norm) -> float:
    require(isinstance(target_norm, (int, float)) and not isinstance(target_norm, bool)
            and math.isfinite(target_norm) and target_norm > 0,
            'target norm must be finite and positive')
    return float(target_norm)


def twin_coefficient(delta: dict, target_norm: float) -> float | None:
    """c with ||c delta|| = target_norm; None when c > 1 (twin not built, never extrapolated)."""
    target = _target(target_norm)
    norm = global_norm(delta)
    require(norm > 0, 'zero-norm delta')
    c = target / norm
    return None if c > 1 else c


def placebo_coefficient(delta_p: dict, target_norm: float) -> tuple[float, float]:
    """(p, shortfall_fraction): p capped at 1; shortfall = (target - ||delta_p||)/target > 0 when capped."""
    target = _target(target_norm)
    norm = global_norm(delta_p)
    require(norm > 0, 'zero-norm delta')
    p = target / norm
    if p > 1:
        return 1.0, (target - norm) / target
    return p, 0.0


def placebo_label(shortfall_fraction: float) -> str | None:
    """Section 5 label: PLACEBO_UNDERSIZED(shortfall %) when the placebo was capped."""
    require(isinstance(shortfall_fraction, float) and 0.0 <= shortfall_fraction < 1.0,
            'invalid shortfall fraction')
    if shortfall_fraction == 0.0:
        return None
    return f'PLACEBO_UNDERSIZED({100.0 * shortfall_fraction:.2f}%)'


def retained_coefficient(displacement: dict, delta_first: dict) -> float:
    """<displacement, delta_first> / ||delta_first||^2 in float64 (section 7 checkpoint record)."""
    _require_float_state(displacement, 'displacement')
    _require_float_state(delta_first, 'first delta')
    require(set(displacement) == set(delta_first), 'displacement/delta key mismatch')
    numerator = denominator = 0.0
    for key in sorted(delta_first):
        a, d = displacement[key], delta_first[key]
        require(tuple(a.shape) == tuple(d.shape), 'shape mismatch: ' + key)
        numerator += _dot(a, d)
        denominator += _sumsq(d)
    require(denominator > 0, 'zero-norm first delta')
    return numerator / denominator
