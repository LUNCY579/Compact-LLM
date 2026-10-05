"""Actual barrier/resource and explicit IO readback; no native service registration.

This adapter records actual NativeResourceRecorder samples and retained ledger
operations. Reservation ceilings are never relabeled measurements. Missing raw
actual fields or declared barrier coverage refuse. Samples describe discrete
barriers; no continuous GPU/disk minimum or acquisition claim is made.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import ce_native_authority as authority, ce_resource_evidence as resource, update

SCHEMA = 'CE8-RESOURCE-MEASUREMENT-SPEC-1'
RAW = 'CE8-QUALIFICATION-RAW-EVIDENCE-1'
require = authority.require
_REGISTERED_RESOURCE_ADAPTERS = frozenset()


def _integer(value): return type(value) is int and value >= 0
def _finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_measurement_bounds(spec):
    from .ce_qualification_artifacts import resource_json_bound, MAX_RESOURCE_JSON
    require(type(spec) is dict, 'explicit resource measurement bounds required')
    maximum = resource_json_bound(spec.get('json_bytes_upper'))
    work = spec.get('reducer_work_bytes_upper')
    require(type(work) is int and 4 * maximum <= work <= 4 * MAX_RESOURCE_JSON,
            'positive bounded resource reducer work allowance')
    return maximum, work


def reduce_samples(samples, required_barriers, *, worker_pid, gpu_uuid, limits):
    """Pure arithmetic on exact retained OS/CUDA facts; not execution proof."""
    require(type(samples) is list and samples and type(required_barriers) is list and required_barriers
            and all(type(name) is str and name for name in required_barriers)
            and len(set(required_barriers)) == len(required_barriers)
            and all(type(sample) is dict and type(sample.get('stage')) is str and sample['stage'] for sample in samples),
            'explicit actual sample/barrier scope')
    require(set(required_barriers) <= {sample.get('stage') for sample in samples}, 'required native barriers were not sampled')
    require(type(worker_pid) is int and worker_pid > 0, 'actual worker OS PID required')
    previous, totals = None, set()
    for sample in samples:
        require(type(sample) is dict and sample.get('schema') == resource.SCHEMA
                and sample.get('worker_id') == worker_pid and type(sample['worker_id']) is int
                and sample.get('gpu_uuid') == gpu_uuid and _finite(sample.get('perf')),
                'same actual worker/GPU barrier evidence')
        require(previous is None or sample['perf'] >= previous, 'resource barrier clock regressed')
        previous = sample['perf']; resource.enforce(sample, limits, launch=sample.get('stage') == 'launch')
        require(type(sample.get('gpu_total_bytes')) is int and sample['gpu_total_bytes'] > 0,
                'integer displayed GPU total')
        totals.add(sample['gpu_total_bytes'])
    require(len(totals) == 1 and all(type(value) is int and value > 0 for value in totals), 'one actual GPU total inventory')
    cuda = [sample for sample in samples if 'cuda_peak_reserved_bytes' in sample]
    # The CLI rounds total memory to integer MiB. Retain that separate display
    # inventory; allocator fractions use the exact CUDA total, never that display.
    cuda_totals = {sample['cuda_total_bytes'] for sample in cuda}
    require(cuda and len(cuda_totals) == 1 and all(type(value) is int and value > 0 for value in cuda_totals),
            'actual initialized CUDA-phase resource samples required')
    gib = 1024 ** 3
    return {'peak_worker_rss_gib': max(sample['worker_peak_working_set_bytes'] for sample in samples) / gib,
        'gpu_total_gib': next(iter(cuda_totals)) / gib,
        'displayed_gpu_total_bytes': next(iter(totals)),
        'peak_reserved_gib': max(sample['cuda_peak_reserved_bytes'] for sample in cuda) / gib,
        'min_gpu_free_gib': min(resource.gpu_free_lower(sample) for sample in samples) / gib,
        'min_disk_free_bytes': min(sample['disk_free_bytes'] for sample in samples),
        'allocator_fraction': limits['allocator_fraction'], 'reserved_fraction_limit': limits['reserved_fraction_limit'],
        'sampled_barriers': [sample['stage'] for sample in samples],
        'gpu_observation_scope': 'declared discrete barrier samples and lifetime allocator peak; no continuous minimum assertion',
        'worker_peak_primitive': 'Windows PeakWorkingSetSize observed at barriers; metric retains RSS compatibility name'}


def _actual(result, keys, label, *, integer=True):
    values = [result[key] for key in keys if key in result]
    require(values and all((_integer(value) if integer else _finite(value)) for value in values)
            and all(value == values[0] for value in values), 'explicit consistent actual operation ' + label + ' missing')
    return values[0]


def _measurement_row(identifier, operation, worker_identity_sha256):
    require(type(identifier) is str and identifier and type(operation) is dict
            and operation.get('phase') == 'returned' and operation.get('worker_id') == worker_identity_sha256
            and type(operation.get('started_seq')) is int and type(operation.get('returned_seq')) is int
            and operation['returned_seq'] > operation['started_seq'], 'actual returned current-worker operation required')
    result = operation.get('result'); require(type(result) is dict, 'retained actual operation results required')
    return {'operation_id': identifier, 'kind': operation['kind'],
        'elapsed_seconds': _actual(result, ('wall_s', 'wall_seconds', 'elapsed_seconds'), 'elapsed time', integer=False),
        'io_read_bytes': _actual(result, ('bytes_read', 'io_read_bytes'), 'read bytes'),
        'io_write_bytes': _actual(result, ('bytes_written', 'io_write_bytes'), 'write bytes'),
        'identity_hash_attempts': _actual(result, ('identity_hash_attempts_actual', 'identity_hash_attempts'), 'hash attempts'),
        'identity_hash_bytes': _actual(result, ('identity_hash_bytes_actual', 'identity_hash_bytes'), 'hash bytes'),
        'io_observation_scope': result.get('io_observation_scope', 'tracked returned counters only'),
        'started_seq': operation['started_seq'], 'returned_seq': operation['returned_seq'],
        'qualification_stage': operation.get('stage'), 'attempt_id': operation.get('attempt_id')}


def reduce_operations(operations, measured_kinds, *, worker_identity_sha256):
    """Derive work from actual returned fields; reserved profiles are evidence only."""
    require(type(operations) is dict and type(measured_kinds) is list and measured_kinds
            and all(type(kind) is str and kind for kind in measured_kinds)
            and len(set(measured_kinds)) == len(measured_kinds)
            and authority._hex(worker_identity_sha256), 'explicit measured operation kinds and worker required')
    measurements = []; seen = set()
    for identifier, operation in operations.items():
        require(type(identifier) is str and identifier and type(operation) is dict, 'retained operation shape')
        if operation.get('kind') not in measured_kinds: continue
        measurements.append(_measurement_row(identifier, operation, worker_identity_sha256))
        seen.add(operation['kind'])
    require(seen == set(measured_kinds), 'declared physical operation measurement coverage missing')
    return {'measured': True, 'operation_measurements': measurements,
        'measurement_scope': 'exact selected returned operation kinds; opaque operations without actual counters remain unmeasured',
            'native_qualified': False, 'execution_verified': False}


def _snapshot_source_view(owner):
    """Reference-only equivalent of the frozen QualificationAccounting snapshot.

    No full snapshot/JSON clone occurs before the 512MiB gate. This exact field
    list must remain aligned with that source-owned snapshot; tests compare it.
    Small progress/prepayment records are bounded metadata, not operation graphs.
    """
    from .ce_qualification_accounting import SCHEMA as ACCOUNTING_SCHEMA
    require(type(owner.operations) is dict and len(owner.operations) <= 200000,
            'resource operation graph count before snapshot view')
    pending = [op for op in owner.operations.values() if op['phase'] == 'started']
    return {'schema': ACCOUNTING_SCHEMA, 'fixture': owner.fixture, 'native_qualified': False,
        'training_authorized': False, 'binding_sha256': owner.binding_sha256,
        'ledger_head': owner._head(), 'events': len(owner.events), 'worker_proof': owner.proof,
        'cumulative': owner.cumulative, 'stages': owner.by_stage,
        'reducer_reserved_bytes': owner.reducer_reserved_bytes, 'reducer_by_stage': owner.reducer_by_stage,
        'attempts': owner.attempts, 'pending_operations': pending,
        'fatal': owner.fatal, 'unknown_wall': owner.unknown_wall, 'progress': owner.progress_record(),
        **({'rejection_ticket': owner.rejection_ticket, 'rejection_used': owner.rejection_used,
            'rejection_record': owner.rejection_record, 'rejection_unknown': owner.rejection_unknown}
           if owner.rejection_policy is not None else {}),
        'reservation_rule': 'permanent full upper charges; failed/unknown/restarted work never refunded',
        'cost_scope': 'declared callback upper reservations and serialized journal costs; not complete native measured totals',
        'complete_wall_accounting': not owner.worker_open and not owner.unknown_wall,
        'actual_wall_seconds': owner.actual_wall_seconds, 'last_monotonic': owner.last_clock,
        'worker_open': owner.worker_open, 'bootstrap_prepayment': owner.bootstrap_handle.evidence,
        'native_accounting_status': 'FIXTURE_UNQUALIFIED' if owner.fixture else 'UNSUPPORTED_UNMEASURED',
        'native_activation_gaps': owner.native_activation_gaps}


def _report_preflight(recorder, accounting, spec, proof, binding, maximum):
    """Cheap bounded source scan plus one-row measurement census, no row list."""
    require(type(recorder.samples) is list and 0 < len(recorder.samples) <= 100000,
            'resource sample graph count bound before reporting')
    observed = reduce_samples(recorder.samples, [recorder.samples[0]['stage']],
        worker_pid=proof['os_identity']['pid'], gpu_uuid=recorder.spec['gpu_uuid'], limits=spec['limits'])
    observed.update(measured=True, operation_measurements=[],
        measurement_scope='exact selected returned operation kinds; opaque operations without actual counters remain unmeasured',
        native_qualified=False, execution_verified=False)
    payload = {'resource_samples': recorder.samples, 'raw_operations': accounting.operations,
        'accounting_snapshot': _snapshot_source_view(accounting), 'required_barriers': spec['required_barriers'],
        'measured_operation_kinds': spec['measured_operation_kinds'], 'json_bytes_upper': maximum,
        'reducer_work_bytes_upper': spec['reducer_work_bytes_upper'], 'limits': spec['limits'],
        'observations': observed, 'worker_proof': proof,
        'classification': 'actual barrier and explicit-operation evidence; independent OS/source/ledger verification still required'}
    envelope = {'schema': RAW, **binding, 'evidence_kind': 'resource_io', 'payload': payload}
    total = resource.bounded_json_graph_census(envelope, maximum_json_bytes=maximum)
    selected, seen = 0, set()
    for identifier, operation in accounting.operations.items():
        require(type(operation) is dict, 'resource source operation shape')
        if operation.get('kind') not in spec['measured_operation_kinds']: continue
        row = _measurement_row(identifier, operation, proof['worker_identity_sha256'])
        extent = resource.bounded_json_graph_census(row, maximum_json_bytes=maximum)
        total['json_bytes'] += extent['json_bytes'] + int(selected > 0)
        for key in ('nodes', 'containers', 'copied_container_bytes_upper'):
            total[key] += extent[key]
        total['max_scalar_json_bytes'] = max(total['max_scalar_json_bytes'], extent['max_scalar_json_bytes'])
        total['encoder_workspace_bytes_upper'] = max(total['encoder_workspace_bytes_upper'], extent['encoder_workspace_bytes_upper'])
        selected += 1; seen.add(operation['kind'])
        require(total['json_bytes'] + 1 <= maximum and total['nodes'] <= resource.REPORT_CENSUS_NODES,
                'complete resource graph/JSON bound before copying')
    require(seen == set(spec['measured_operation_kinds']), 'declared resource measurement coverage missing')
    # Measurement list pointers, deepcopy memo/container workspace, bounded
    # scalar encoder plus bytes chunk, buffered IO/readback chunk and metadata.
    total['copied_container_bytes_upper'] += 128 * selected
    total['transient_bytes_upper'] = (total['copied_container_bytes_upper']
        + total['encoder_workspace_bytes_upper'] + 8 * resource.CHUNK)
    total['reducer_work_bytes_upper'] = (4 * (total['json_bytes'] + 1)
        + total['copied_container_bytes_upper'] + total['encoder_workspace_bytes_upper'])
    require(total['json_bytes'] + 1 <= maximum and total['max_scalar_json_bytes'] <= resource.CHUNK,
            'resource scalar/JSON encoder extent exceeds preallocation bound')
    require(total['transient_bytes_upper'] <= resource.REPORT_TRANSIENT_BYTES,
            'resource graph/container/copy/encoder transient exceeds fixed512MiB')
    require(total['reducer_work_bytes_upper'] <= spec['reducer_work_bytes_upper'],
            'resource census/copy/reduction work exceeds fixed prepaid ceiling')
    return envelope, total


def _ascii_json_chunks(value):
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    for text in encoder.iterencode(value):
        require(len(text) <= resource.CHUNK, 'resource encoder scalar extent drift')
        yield text.encode('ascii')


def _stream_resource_envelope(path, envelope, *, maximum, expected_bytes):
    """Bounded chunks and independent streamed readback; no whole JSON buffer."""
    count = 0
    with path.open('xb') as stream:
        for block in _ascii_json_chunks(envelope):
            count += len(block); require(count + 1 <= maximum, 'resource streamed JSON ceiling')
            require(stream.write(block) == len(block), 'short resource artifact publication')
        require(stream.write(b'\n') == 1, 'short resource artifact publication')
        count += 1; stream.flush(); os.fsync(stream.fileno())
    require(count == expected_bytes, 'resource streaming extent differs from paid graph census')
    observed, read = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for expected in _ascii_json_chunks(envelope):
            block=stream.read(len(expected)); read += len(block)
            require(block == expected, 'resource artifact streamed raw readback mismatch')
            observed.update(block)
        newline=stream.read(1); read += len(newline)
        require(newline == b'\n' and stream.read(1) == b'', 'resource artifact newline/extent drift')
        observed.update(newline)
    require(read == count, 'resource artifact streamed raw readback extent mismatch')
    return count, observed.hexdigest()


def validate_binding(context, proof, accounting, binding):
    from .ce_qualification_artifacts import binding as checked_binding
    checked_binding(binding)
    require(binding['stage'] == 'resource_io_measurement'
            and all(binding[key] == context[key] for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256'))
            and binding['code_sha256'] == context['bindings']['source']['code_sha256']
            and binding['worker_identity_sha256'] == proof['worker_identity_sha256']
            and type(accounting.active) is dict and accounting.active.get('stage') == binding['stage']
            and accounting.active.get('attempt_id') == binding['attempt_id']
            and accounting.active.get('status') == 'STARTED',
            'resource artifact must bind the exact active source/request/scope/worker attempt')
    return binding


def resource_adapter(context, permit, prepared, backend, accounting, proof, binding):
    authority.require_execution_permit(permit, context, 'resource_io_measurement')
    from .ce_native_backend import NativeCEBackend
    from .ce_qualification_accounting import QualificationAccounting
    require(type(backend) is NativeCEBackend and backend.permit is permit and type(accounting) is QualificationAccounting,
            'exact source-bound native backend and qualification ledger required')
    authority.require_execution_permit(permit, context, 'resource_io_measurement', operation=backend.operation, gate=backend.gate)
    require(getattr(backend.operation, 'ledger', None) is accounting and accounting.scope_role == 'qualification_only'
            and accounting.request_sha256 == context['request_sha256'] and accounting.scope_sha256 == context['scope_sha256'],
            'actual separate resource accounting owner mismatch')
    validate_binding(context, proof, accounting, binding)
    recorder = context.get('native_services', {}).get('resource_recorder')
    require(type(recorder) is resource.NativeResourceRecorder and recorder.permit is permit
            and recorder.operation is backend.operation and recorder.context is context,
            'actual same-scope barrier recorder missing; sample flags are not measurements')
    spec = context['bindings']['scope'].get('resource_measurement')
    require(type(spec) is dict and spec.get('schema') == SCHEMA, 'explicit resource measurement/readback scope required')
    maximum, _ = validate_measurement_bounds(spec)
    require(type(spec.get('owned_directory')) is str and Path(spec['owned_directory']).is_absolute(),
            'resource artifact size/owner declared')
    transients = getattr(backend.gate, 'spec', {}).get('host_transient_bytes', {})
    require(type(transients) is dict
            and transients.get('qualification_reducer_overhead') == 64 * resource.CHUNK
            and type(transients.get('resource_io_measurement')) is int
            and transients['resource_io_measurement'] >= resource.REPORT_TRANSIENT_BYTES
            and type(transients.get('qualification_artifact')) is int
            and transients['qualification_artifact'] >= resource.REPORT_TRANSIENT_BYTES,
            'source reporting requires cheap64MiB and actual512MiB transient gates')
    recorder.framework = backend.torch
    with update._span(backend.operation, 'qualification_reducer_overhead', {
        'qualification_stage': binding['stage'], 'attempt_id': binding['attempt_id'],
        'read_bytes_upper': 0, 'write_bytes_upper': 0,
        'identity_hash_attempts_upper': 0, 'identity_hash_bytes_upper': 0,
        'reducer_work_bytes_upper': spec['reducer_work_bytes_upper']}) as reduced:
        # Fixed cheap64MiB guard precedes all census/view allocations. The
        # first scan may refuse before the512MiB gate or any full graph copy.
        backend.gate('qualification_reducer_overhead')
        recorder.sample('final')
        _report_preflight(recorder, accounting, spec, proof, binding, maximum)
        # This exact stage owns the registered512MiB transient floor. Its new
        # probe/clock facts are included by a second bounded census before copy.
        backend.gate('resource_io_measurement')
        source_envelope, plan = _report_preflight(recorder, accounting, spec, proof, binding, maximum)
        envelope = copy.deepcopy(source_envelope)
        payload = envelope['payload']
        samples, operations = payload['resource_samples'], payload['raw_operations']
        observed = reduce_samples(samples, spec['required_barriers'], worker_pid=proof['os_identity']['pid'],
                                  gpu_uuid=recorder.spec['gpu_uuid'], limits=spec['limits'])
        measured = reduce_operations(operations, spec['measured_operation_kinds'], worker_identity_sha256=proof['worker_identity_sha256'])
        observed.update(measured)
        from .ce_qualification import validate_stage_observations
        validate_stage_observations(context, 'resource_io_measurement', observed, prepared.receipt)
        # Raw reservations/failed/pending operations are retained separately
        # from selected actual measurements. Nothing is refunded as zero work.
        payload['observations'] = observed
        exact = resource.bounded_json_graph_census(envelope, maximum_json_bytes=maximum)
        require(exact['json_bytes'] + 1 == plan['json_bytes'] + 1,
                'resource captured graph differs from prospective exact encoder extent')
        reduced.update(bytes_read=0, bytes_written=0, identity_hash_attempts_actual=0,
                       identity_hash_bytes_actual=0, serialized_envelope_bytes=exact['json_bytes'] + 1,
                       resource_transient_bytes_upper=plan['transient_bytes_upper'],
                       resource_reducer_work_bytes_upper=plan['reducer_work_bytes_upper'])
    root = Path(spec['owned_directory'])
    path = root / ('resource.' + binding['attempt_id'] + '.' + uuid.uuid4().hex + '.json')
    with update._span(backend.operation, 'qualification_artifact', {'qualification_stage': binding['stage'],
        'attempt_id': binding['attempt_id'], 'path': str(path), 'write_bytes_upper': maximum,
        'read_bytes_upper': maximum + 1, 'identity_hash_attempts_upper': 2, 'identity_hash_bytes_upper': 2 * maximum,
        'reducer_work_bytes_upper': maximum}) as written:
        backend.gate('qualification_artifact')
        owner = Path(context['bindings']['scope']['native_backend']['output_directory']).resolve()
        require(not root.is_symlink() and not path.is_symlink() and root.resolve().is_relative_to(owner)
                and root.resolve() != owner and path.resolve().parent == root.resolve(), 'resource artifact escapes exact native owner')
        binding_census = resource.bounded_json_graph_census(binding, maximum_json_bytes=resource.CHUNK)
        binding_hasher = hashlib.sha256()
        for block in _ascii_json_chunks(binding): binding_hasher.update(block)
        binding_sha = binding_hasher.hexdigest()
        expected_bytes = exact['json_bytes'] + 1
        accounting.reserve_metadata('qualification_json', expected_bytes, write_id=uuid.uuid4().hex, path=str(path))
        root.mkdir(parents=True, exist_ok=True)
        size, hashed = _stream_resource_envelope(path, envelope, maximum=maximum, expected_bytes=expected_bytes)
        written.update(bytes_read=size, bytes_written=size, identity_hash_attempts_actual=2,
                       identity_hash_bytes_actual=binding_census['json_bytes'] + size)
    reference = {'path': str(path), 'sha256': hashed, 'bytes': size, 'role': 'resource_io', 'binding_sha256': binding_sha}
    return {'observations': observed, 'raw_artifacts': [reference], 'native_qualified': False, 'execution_verified': False,
            'classification': 'barrier-sampled resources and explicitly measured operation subset; not native PASS or full-unit pricing'}
