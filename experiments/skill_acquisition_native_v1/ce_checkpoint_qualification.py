"""Qualification-only checkpoint storage/branch contracts; activation closed.

Three fresh OS workers are required: uninterrupted U2, independent U1 prefix,
and fresh U1 restore followed by U2. Parent U2 is retained before GPU handoff,
then restored with a fresh fence for the subsequent delta stage. Numeric/state
readback never substitutes for actual source-issued process/lease evidence.
"""
from __future__ import annotations

import copy
import gc
import math
from pathlib import Path

from . import ce_native_authority as authority, ce_qualification_artifacts as raw

SPEC = 'CE8-CHECKPOINT-QUALIFICATION-SPEC-1'
MODE = 'CE8_NATIVE_QUALIFICATION_CHECKPOINT_1'
RESTORE_SCHEMA = 'CE8-QUALIFICATION-RESTORE-FENCE-1'
CHECKPOINT_SCHEMA = 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1'
BRANCHES = ('uninterrupted', 'prefix', 'resumed')
_REGISTERED_GPU_HANDOFF_SERVICES = frozenset()
_REGISTERED_CHECKPOINT_PROCESS_SERVICES = frozenset()
require = authority.require


class MissingCheckpointService(authority.NativeAdmissionError): pass


def qualification_batches(context):
    require(context.get('phase') == 'qualification', 'qualification checkpoint cannot use production scope')
    batches = context['bindings']['scope']['qualifier_execution'].get('adam_batches')
    require(type(batches) is list and len(batches) == 2
            and all(type(batch) is list and len(batch) == 8 and len(set(batch)) == 8
                    and all(type(rid) is str and rid for rid in batch) for batch in batches),
            'exact two declared eight-row qualification updates')
    return copy.deepcopy(batches)


def branch_plan(context):
    batches = qualification_batches(context)
    return [
        {'branch': 'uninterrupted', 'start_U': 0, 'updates': 2, 'final_U': 2,
         'row_batches': batches, 'restore_from': None},
        {'branch': 'prefix', 'start_U': 0, 'updates': 1, 'final_U': 1,
         'row_batches': batches[:1], 'restore_from': None},
        {'branch': 'resumed', 'start_U': 1, 'updates': 1, 'final_U': 2,
         'row_batches': batches[1:], 'restore_from': 'prefix'}]


def validate_restore_fence(record, before, restore_operation, *, current_binding):
    """Pure exact checkpoint/fence ancestry; no state mutation or installation."""
    raw.binding(current_binding)
    require(current_binding['stage'] == 'checkpoint_resume', 'current qualification restore stage binding')
    require(type(before) is dict and all(type(before.get(key)) is int for key in ('C', 'U', 'generation'))
            and 0 <= before['C'] == before['U'] <= 2 and before['generation'] >= 0,
            'strict before-restore C/U/generation')
    require(type(record) is dict and record.get('schema') == RESTORE_SCHEMA
            and raw._hex(record.get('manifest_sha256'))
            and record.get('source_worker_binding') == current_binding
            and type(record.get('progress')) is dict, 'actual source-bound restore progress proof')
    p = record['progress']
    require(type(p.get('C')) is type(p.get('U')) is int and p['C'] == p['U'] and 0 <= p['C'] <= before['C'] <= 2,
            'qualification restore C/U ancestor')
    require(p.get('history') == before['history'][:p['C']]
            and type(p.get('generation')) is int and p['generation'] > before['generation'], 'exact history prefix/new restore generation')
    require(all(type(record.get(key)) is int for key in ('C', 'U', 'generation'))
            and all(record.get(key) == p.get(key) for key in
            ('C', 'U', 'history', 'generation', 'snapshot', 'symbolic_snapshot'))
            and record.get('install_fence_required') is True, 'full restore fence fields')
    require(type(restore_operation) is dict and restore_operation.get('phase') == 'returned'
            and restore_operation.get('kind') == 'checkpoint_restore'
            and restore_operation.get('worker_id') == current_binding['worker_identity_sha256']
            and restore_operation.get('attempt_id') == current_binding['attempt_id']
            and restore_operation.get('stage') == current_binding['stage']
            and restore_operation.get('source_worker_binding') == current_binding
            and all(restore_operation.get('result', {}).get(key) == record.get(key) for key in
                ('manifest_sha256', 'C', 'U', 'generation', 'symbolic_snapshot')),
            'own invocation returned restore operation matches installed fence')
    return copy.deepcopy(p)


class QualificationCheckpoints:
    """Thin actual FP32/CUDA storage using only the two qualification batches."""
    def __init__(self, context, native_permit, directory, bindings, numel, *, operation, gate, metadata_reserve):
        authority.require_execution_permit(native_permit, context, 'checkpoint_resume', operation=operation, gate=gate)
        batches = qualification_batches(context); scope = context['bindings']['scope']
        spec = scope.get('checkpoint_qualification')
        require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit owned qualification checkpoint specification')
        require(type(bindings) is dict and bindings.get('checkpoint_mode') == MODE
                and bindings.get('synthetic_cpu') is False and bindings.get('role') == 'qualification_only'
                and bindings.get('qualification_batches') == batches, 'checkpoint exact qualification role/batches')
        admitted = context['bindings']
        require(bindings.get('code') == admitted['source']['code_sha256']
                and bindings.get('recipe') == admitted['recipe']['recipe']
                and bindings.get('runtime') == admitted['runtime']['decoding_preimage']
                and bindings.get('inventory') == scope['native_backend']['parameter_inventory']
                and all(bindings.get(key) == admitted['anchors'][key] for key in ('common_anchor_sha256', 'own_start_sha256')),
                'qualification storage source/runtime/inventory/anchor drift')
        require(type(numel) is int and numel == sum(math.prod(entry['shape']) for entry in bindings['inventory']),
                'qualification exact selected numel')
        owner = Path(spec['owned_directory']); path = Path(directory)
        require(owner.is_absolute() and path.is_absolute() and path.is_relative_to(owner) and path != owner,
                'separately owned qualification branch checkpoint path')
        ledger = getattr(operation, 'ledger', operation)
        require(callable(metadata_reserve) and metadata_reserve == getattr(ledger, 'reserve_metadata', None),
                'bound permanent qualification metadata reservation')
        self.context, self.permit, self.operation, self.gate = context, native_permit, operation, gate
        from . import checkpoint as base
        class Storage(base.NativeCheckpoints):
            def _extract_plan_rows(self, value): return copy.deepcopy(value['qualification_batches'])
            def _validate_progress(self, progress):
                base.validate_progress(progress, batches)
                require(progress['C'] == progress['U'] <= 2
                        and all(row['updated'] is True for row in progress['history']), 'exact qualification C=U<=2')
                return progress
            def _scope(self, model, optimizer):
                result = super()._scope(model, optimizer)
                require(result[-1].type == 'cuda' and result[-1].index == 0, 'actual qualification cuda:0 state')
                return result
        self.storage = Storage(directory, bindings, numel, gate, metadata_reserve=metadata_reserve)

    def _admit(self):
        authority.require_execution_permit(self.permit, self.context, 'checkpoint_resume', operation=self.operation, gate=self.gate)
    def attach_ledger(self, ledger):
        self._admit(); require(ledger is self.operation, 'qualification checkpoint accounting facade identity')
        self.storage.attach_ledger(ledger); return self
    def save(self, model, optimizer, progress, ledger):
        self._admit(); require(ledger is self.operation, 'qualification save owns exact accounting')
        return self.storage.save(model, optimizer, progress, ledger)
    def restore(self, model, optimizer):
        self._admit(); return self.storage.restore(model, optimizer)


def _read_manifest(reader, reference, expected):
    require(type(reference) is dict and type(reference.get('bytes')) is int
            and 0 < reference['bytes'] <= raw.MAX_JSON, 'bounded checkpoint JSON manifest before any read')
    require(expected['stage'] == 'checkpoint_resume', 'checkpoint state proof stage')
    blob = b''.join(reader._blocks(reference, expected, 'checkpoint_manifest'))
    with reader._overhead('checkpoint_manifest_semantics', work_bytes=32 * reference['bytes'],
                         hash_attempts=2, hash_bytes=16 * reference['bytes']):
        manifest = raw._json(blob)
        require(type(manifest) is dict and raw.canonical(manifest) + b'\n' == blob,
                'canonical original checkpoint JSON manifest')
        require(type(manifest) is dict and manifest.get('schema') == CHECKPOINT_SCHEMA
                and raw.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
                == manifest.get('manifest_sha256'), 'original checkpoint manifest body digest')
        require(manifest.get('bindings', {}).get('qualification_evidence_binding') == expected,
                'checkpoint manifest actual branch/source/worker binding')
        require(manifest.get('progress', {}).get('C') == manifest['progress'].get('U') == 2,
                'checkpoint comparison exact qualified U2 endpoint')
    return manifest


def _close_child(owned, timeout_seconds, primary=None, *, terminal_already_verified=False):
    """Use this retained owner's prepaid stop/terminal path before close.

    The caller receives failures instead of mistaking an unclosed/live worker
    for a terminal worker. No unrelated PID or status-file flag is consulted.
    """
    failures, terminal_verified = [], terminal_already_verified
    if primary is not None:
        try: owned._stop_owned('checkpoint_qualification_failure', primary=primary)
        except BaseException as error: failures.append(error)
    if not terminal_verified:
        try: owned.terminal(timeout_seconds); terminal_verified = True
        except BaseException as error: failures.append(error)
    try: owned.close()
    except BaseException as error: failures.append(error)
    if failures:
        details = [{'type': type(error).__name__, 'message': str(error)[:1000]} for error in failures]
        error = primary if primary is not None else failures[0]
        error.checkpoint_child_cleanup_failure = details
        error.checkpoint_child_terminal_verified = terminal_verified
        if error is failures[0]: raise error
        raise error from failures[0]
    return terminal_verified


def _payload_blocks(reader, reference, expected, specifications):
    """Validate actual safetensors header/layout and stream only data bytes."""
    stream = reader._blocks(reference, expected, 'checkpoint_payload')
    buffered, target, primary = b'', None, None
    try:
        for block in stream:
            buffered += block
            if target is None and len(buffered) >= 8:
                length = int.from_bytes(buffered[:8], 'little')
                require(0 < length <= raw.CHUNK, 'bounded actual checkpoint tensor header')
                target = 8 + length
            if target is not None and len(buffered) >= target:
                header = raw._json(buffered[8:target]); require(type(header) is dict, 'actual tensor header object')
                require(set(header) == set(specifications), 'actual payload keys equal manifest specifications')
                end = 0
                for name, entry in sorted(header.items(), key=lambda pair: pair[1]['data_offsets'][0]):
                    dtype, shape = specifications[name]
                    require(dtype in ('F32', 'U8') and entry == {'dtype': dtype, 'shape': shape,
                        'data_offsets': [end, end + math.prod(shape) * (4 if dtype == 'F32' else 1)]},
                        'actual contiguous tensor shape/dtype/byte layout')
                    end = entry['data_offsets'][1]
                require(reference['bytes'] == target + end, 'actual tensor payload exact byte count')
                require(list(header) == sorted(header), 'canonical tensor order for bitwise stream comparison')
                data = buffered[target:]
                if data: yield data
                for rest in stream: yield rest
                return
        require(False, 'truncated actual checkpoint tensor header')
    except BaseException as error:
        primary = error; raise
    finally:
        try: stream.close()
        except BaseException as error:
            if primary is not None:
                raw._accounting(primary, error); raise primary from error
            raise


def validate_checkpoint_relationship(left_binding, right_binding, *, parent_binding=None, child_lineages=None):
    """Exact declared study ancestry; this arithmetic check is not OS proof."""
    raw.binding(left_binding); raw.binding(right_binding)
    require(left_binding['stage'] == right_binding['stage'] == 'checkpoint_resume', 'checkpoint branch stage')
    require(left_binding['worker_identity_sha256'] != right_binding['worker_identity_sha256'], 'independent fresh checkpoint workers required')
    require(all(left_binding[key] == right_binding[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')),
            'same checkpoint study/source/recipe for separate workers')
    if parent_binding is None and child_lineages is None:
        require(left_binding['request_sha256'] == right_binding['request_sha256'],
                'different checkpoint requests require exact parent/child lineage')
        return
    raw.binding(parent_binding)
    require(parent_binding['stage'] == 'checkpoint_resume' and type(child_lineages) is dict,
            'explicit checkpoint parent binding and observed child lineage map')
    for value in (left_binding, right_binding):
        require(all(value[key] == parent_binding[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')),
                'checkpoint lineage parent study/source/recipe drift')
        identity = value['worker_identity_sha256']
        if identity == parent_binding['worker_identity_sha256']:
            require(value == parent_binding, 'exact parent attempt binding required')
            continue
        lineage = child_lineages.get(identity)
        require(type(lineage) is dict and lineage.get('schema') == 'CE8-CHECKPOINT-CHILD-PREDECESSOR-1'
                and lineage.get('worker_identity_sha256') == identity
                and lineage.get('study_request_sha256') == value['request_sha256'] == parent_binding['request_sha256']
                and lineage.get('parent_request_sha256') == parent_binding['request_sha256']
                and lineage.get('parent_worker_identity_sha256') == parent_binding['worker_identity_sha256']
                and all(lineage.get(key) == parent_binding[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')),
                'exact observed child request/worker/parent lineage required')
        proof = lineage.get('parent_proof')
        require(type(proof) is dict and proof.get('worker_identity_sha256') == parent_binding['worker_identity_sha256']
                and proof.get('request_sha256') == parent_binding['request_sha256']
                and proof.get('scope_sha256') == parent_binding['scope_sha256'],
                'retained exact parent worker proof required')
        child_proof, request_ref = lineage.get('child_proof'), lineage.get('child_request_ref')
        require(type(request_ref) is dict and type(request_ref.get('path')) is str and Path(request_ref['path']).is_absolute()
                and raw._hex(request_ref.get('sha256')) and raw._int(request_ref.get('bytes'), 1)
                and lineage.get('child_request_sha256') == request_ref['sha256']
                and type(child_proof) is dict and child_proof.get('worker_identity_sha256') == identity
                and child_proof.get('request_sha256') == value['request_sha256']
                and child_proof.get('scope_sha256') == value['scope_sha256']
                and child_proof.get('sources', {}).get('request_ref') == request_ref,
                'raw child request hash and observed child proof are separate from study identity')


def compare_checkpoint_states(reader, left_manifest_ref, left_binding, right_manifest_ref, right_binding, *, parameter_inventory,
                              parent_binding=None, child_lineages=None):
    """Actual byte comparison, with independent worker provenance left intact."""
    require(type(reader) is raw.RawEvidenceReader, 'actual bounded paid checkpoint reader required')
    with reader._overhead('checkpoint_study_ancestry', work_bytes=8 * raw.MAX_BINDING_BYTES):
        validate_checkpoint_relationship(left_binding, right_binding, parent_binding=parent_binding, child_lineages=child_lineages)
    left = _read_manifest(reader, left_manifest_ref, left_binding)
    right = _read_manifest(reader, right_manifest_ref, right_binding)
    require(left['specifications'] == right['specifications'], 'bitwise state requires exact parameter/Adam/RNG tensor keys/shapes')
    specifications = left['specifications']
    names = set(specifications)
    require(type(parameter_inventory) is list and parameter_inventory
            and all(type(entry) is dict and set(entry) == {'name', 'shape', 'dtype'}
                and entry['dtype'] == 'F32' and type(entry['name']) is str
                and type(entry['shape']) is list and all(type(size) is int and size > 0 for size in entry['shape'])
                for entry in parameter_inventory)
            and len({entry['name'] for entry in parameter_inventory}) == len(parameter_inventory), 'exact complete selected inventory required')
    expected_names = {'rng::cpu', 'rng::cuda::0'}
    for entry in parameter_inventory:
        for prefix in ('param::', 'exp_avg::', 'exp_avg_sq::'):
            name = prefix + entry['name']; expected_names.add(name)
            require(specifications.get(name) == ['F32', entry['shape']], 'complete selected parameter/Adam shapes')
        name = 'step::' + entry['name']; expected_names.add(name)
        require(specifications.get(name) == ['F32', []], 'all actual Adam scalar counters')
    require(names == expected_names and left['rng'].get('cuda_count') == right['rng'].get('cuda_count') == 1
            and all(specifications[name][0] == 'U8' for name in ('rng::cpu', 'rng::cuda::0')),
            'complete native parameter/Adam/CPU/CUDA RNG state proof')
    for key in ('module_training', 'optimizer_groups', 'Adam_initialized', 'Adam_scalar_steps', 'rng', 'frozen_state_sha256'):
        require(raw.canonical(left[key]) == raw.canonical(right[key]), 'bitwise checkpoint metadata mismatch: ' + key)
    refs = []
    for manifest, expected in ((left, left_binding), (right, right_binding)):
        saved = manifest['payload']
        ref = {'path': saved['path'], 'sha256': saved['file_sha256'], 'bytes': saved['bytes'],
               'role': 'checkpoint_payload', 'binding_sha256': raw.digest(expected)}
        refs.append(ref)
    total = sum(ref['bytes'] for ref in refs)
    with reader._overhead('checkpoint_bitwise_byte_comparison', work_bytes=16 * total,
                         hash_attempts=2, hash_bytes=2 * raw.MAX_BINDING_BYTES):
        a = _payload_blocks(reader, refs[0], left_binding, specifications)
        b = _payload_blocks(reader, refs[1], right_binding, specifications)
        xa = xb = b''; done_a = done_b = False; primary = None
        try:
            while True:
                if not xa and not done_a:
                    block = next(a, None); done_a = block is None; xa = b'' if block is None else block
                if not xb and not done_b:
                    block = next(b, None); done_b = block is None; xb = b'' if block is None else block
                if done_a and done_b:
                    require(not xa and not xb, 'checkpoint state stream length mismatch'); break
                require(not (done_a and not xa) and not (done_b and not xb), 'checkpoint state stream length mismatch')
                count = min(len(xa), len(xb)); require(xa[:count] == xb[:count], 'checkpoint parameter/Adam/RNG bytes differ')
                xa, xb = xa[count:], xb[count:]
        except BaseException as error:
            primary = error; raise
        finally:
            failures = []
            for stream in (a, b):
                try: stream.close()
                except BaseException as error: failures.append(error)
            if failures:
                if primary is not None:
                    for error in failures: raw._accounting(primary, error)
                    raise primary from failures[0]
                for error in failures[1:]: raw._accounting(failures[0], error)
                raise failures[0]
    return {'parameters_bitwise': True, 'adam_bitwise': True, 'module_modes_bitwise': True,
        'python_rng_bitwise': True, 'numpy_rng_bitwise': True, 'torch_cpu_rng_bitwise': True,
        'torch_cuda_rng_bitwise': True, 'execution_verified': False, 'native_qualified': False,
        'classification': 'actual retained state bytes equal; registered fresh-process/source proof still required'}


def release_parent_model(context, permit, backend):
    authority.require_execution_permit(permit, context, 'checkpoint_resume')
    from .ce_native_backend import NativeCEBackend
    require(type(backend) is NativeCEBackend, 'exact admitted native backend before callback access')
    authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=backend.operation, gate=backend.gate)
    require(backend.model is not None and backend.optimizer is not None and not backend.closed, 'actual already loaded parent model/Adam required')
    with backend._span('cleanup', reason='qualification_parent_GPU_handoff'):
        backend.gate('qualification_parent_release')
        primary = None
        try: backend.optimizer.zero_grad(set_to_none=True); backend.model.eval()
        except BaseException as error: primary = error
        finally:
            backend.model = backend.optimizer = None
            gc.collect()
            try: backend.torch.cuda.synchronize(0); backend.torch.cuda.empty_cache()
            except BaseException as error:
                if primary is not None: primary.release_failure = str(error); raise primary from error
                raise
        if primary is not None: raise primary


def reload_parent_model(context, permit, backend):
    authority.require_execution_permit(permit, context, 'checkpoint_resume')
    from .ce_native_backend import NativeCEBackend
    require(type(backend) is NativeCEBackend, 'exact admitted native backend before callback access')
    authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=backend.operation, gate=backend.gate)
    require(not backend.closed and backend.model is None and backend.optimizer is None, 'released parent remains live and owns no model/Adam')
    require(backend.runtime.decoding_preimage('cuda:0') == context['bindings']['runtime']['decoding_preimage'],
            'already configured pinned CUDA runtime unchanged; no second configure')
    with backend._span('model_load', reason='qualification_parent_reload'):
        backend.gate.launch()
        backend.model, identity = backend.runtime.load_anchor(backend.spec['model_dir'], 'cuda:0', backend.spec['asset_receipt_ref']['path'],
            delta_path=backend.spec['warm_delta_ref']['path'], delta_sha256=backend.spec['warm_delta_ref']['data_sha256'],
            arm_sha256=context['bindings']['anchors']['common_anchor_sha256'])
        require(identity['state_fingerprint'] == context['bindings']['anchors']['own_start_sha256'], 'actual parent reload warm anchor')
    with backend._span('inventory_configuration', reason='qualification_parent_reload'):
        selected = backend.pilot.set_inventory(backend.model, *backend.spec['inventory']['blocks'])
        require(selected['trainable_names'] == backend.trainable_names, 'reloaded parent exact inventory')
        backend.pilot.enable_checkpointing(backend.model)
        settings = backend.recipe['optimizer']; chosen = set(backend.trainable_names)
        backend.optimizer = backend.torch.optim.AdamW([p for name, p in backend.model.named_parameters() if name in chosen],
            lr=settings['lr'], betas=tuple(settings['betas']), eps=settings['eps'], weight_decay=settings['weight_decay'],
            foreach=settings['foreach'], capturable=settings['capturable'])
        require(not backend.optimizer.state, 'fresh reload Adam before exact checkpoint restore')
        backend.model.eval()
    return backend.fingerprint('qualification_parent_reloaded_warm', initial=True)


def checkpoint_adapter(context, permit, prepared, backend, accounting, proof, binding):
    """Closed source-owned coordination entry; no default callback/launcher."""
    authority.require_execution_permit(permit, context, 'checkpoint_resume')
    from .ce_native_backend import NativeCEBackend
    require(type(backend) is NativeCEBackend, 'exact admitted native backend before callback access')
    authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=backend.operation, gate=backend.gate)
    services = context.get('native_services', {}).get('checkpoint_qualification')
    require(type(services) is dict, 'explicit checkpoint process and lease handoff services')
    handoff, processes = services.get('handoff'), services.get('processes')
    if not any(type(handoff) is value for value in _REGISTERED_GPU_HANDOFF_SERVICES):
        raise MissingCheckpointService('source-registered GPU lease handoff is missing; parent model remains allocated')
    if not any(type(processes) is value for value in _REGISTERED_CHECKPOINT_PROCESS_SERVICES):
        raise MissingCheckpointService('source-registered child process/request/handle service is missing; no parent release or launch')
    plans = branch_plan(context)
    # The process owner must use ce_process_launch.OwnedProcess and its exact
    # handshake/terminal handle proofs; JSON response flags cannot satisfy this.
    from .ce_process_launch import OwnedProcess
    require(type(proof) is dict and raw._hex(proof.get('worker_identity_sha256')), 'parent worker metadata binding')
    require(accounting.progress_record()['C'] == accounting.progress_record()['U'] == 2, 'parent actual qualified U2 endpoint')
    parent = processes.save_parent(context, permit, backend, accounting, binding)
    require(type(parent) is dict and type(parent.get('manifest_ref')) is dict, 'retained verified parent U2 before release')
    results, lease, child_quiescent, main_error = [], None, True, None
    try:
        release_parent_model(context, permit, backend)
        lease = handoff.release_parent_lease(context, permit, proof)
        for plan in plans:
            owned = processes.launch(context, permit, plan, parent, results)
            require(type(owned) is OwnedProcess, 'actual source-owned retained child process required')
            primary, terminal_verified = None, False
            child_quiescent = False
            try:
                owned.handshake()
                response = owned.request('checkpoint_qualification', processes.bound_request(plan, parent, results))
                owned.request('finish', {})
                terminal = owned.terminal(processes.timeout_seconds(plan))
                terminal_verified = True
                # This operation consumes the retained OS handle, source-bound
                # request and own child/parent lineage; no response Boolean.
                result = processes.verify_child(owned, plan, response, terminal, proof)
                results.append(result)
            except BaseException as error:
                primary = error
                raise
            finally:
                child_quiescent = _close_child(owned, processes.timeout_seconds(plan), primary,
                                              terminal_already_verified=terminal_verified)
        handoff.reacquire_parent_lease(context, permit, lease)
        lease = None
        handoff.assert_parent_use()
        reload_parent_model(context, permit, backend)
        restored, operation = processes.restore_parent(context, permit, backend, accounting, parent)
        validate_restore_fence(restored, accounting.progress_record(), operation, current_binding=binding)
        accounting.install_restore(restored)
        backend.refresh_policy(accounting)
        return processes.retained_result(context, permit, results, parent, accounting, binding)
    except BaseException as error:
        main_error = error
        raise
    finally:
        if lease is not None and child_quiescent:
            # Only the source owner can reacquire its own lease. Failure remains
            # charged and terminal handling may need to discard the parent.
            try: handoff.reacquire_parent_lease(context, permit, lease)
            except BaseException as error:
                if main_error is not None:
                    main_error.checkpoint_lease_cleanup_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
                    raise main_error from error
                raise
