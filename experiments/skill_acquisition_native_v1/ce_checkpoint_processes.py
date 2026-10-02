"""Concrete checkpoint process coordination; all native issuers stay closed.

Parent/model mechanics remain in QualificationCheckpoints and the supervised
lease. Immutable child files contain static declarations; post-launch observed
delegation is a separate paid file. Raw artifacts and OS proofs are evidence,
never execution permissions or a native qualification result.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import time
import uuid

from . import ce_native_authority as authority, ce_qualification_artifacts as raw
from . import ce_bootstrap_accounting as boot, ce_qualification_accounting as accounting
from . import ce_checkpoint_qualification as cp, ce_checkpoint_worker as worker
from . import ce_checkpoint_initializer as initializer, ce_checkpoint_reduction as reduction
from . import ce_execution_verifier as execution, ce_process_launch as launcher
from . import ce_publication as publication

SPEC = 'CE8-CHECKPOINT-PROCESSES-SPEC-1'
BUNDLE = 'CE8-CHECKPOINT-BRANCHES-BUNDLE-1'
require = authority.require


def child_request(context, parent, branch, *, prefix=None):
    """Static request is serializable before its raw SHA or future child exists."""
    require(branch in cp.BRANCHES, 'one exact checkpoint process branch')
    batches = cp.qualification_batches(context); binding = parent['binding']
    declaration = {'schema': worker.DECLARATION, 'branch': branch, 'inherited_U': int(branch == 'resumed'),
        'inherited_history': [], 'study_request_sha256': context['request_sha256'],
        'parent_request_sha256': binding['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'static_recipe_sha256': context['static_recipe_sha256'], 'code_sha256': context['bindings']['source']['code_sha256'],
        'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
        'own_start_sha256': context['bindings']['anchors']['own_start_sha256'], 'qualification_batches': batches,
        'ingestion_receipt_sha256': context['bindings']['scope']['qualifier_accounting']['ingestion_receipt_sha256'],
        'parent_worker_identity_sha256': binding['worker_identity_sha256'], 'parent_proof': parent['proof'],
        'parent_stage_seal': parent['stage_seal'], 'parent_ledger_refs': parent['ledger_refs']}
    inputs = {'parent_binding': binding}
    if branch == 'resumed':
        require(type(prefix) is dict and prefix.get('branch') == 'prefix', 'verified independent U1 prefix branch required')
        manifest = prefix['manifest']
        require(type(manifest['progress']['C']) is type(manifest['progress']['U']) is int
                and manifest['progress']['C'] == manifest['progress']['U'] == 1
                and manifest['bindings']['qualification_batches'] == batches,
                'exact retained prefix U1/two-batch ancestry')
        declaration.update(inherited_history=manifest['progress']['history'], prefix_manifest=manifest,
            prefix_manifest_ref=prefix['manifest_ref'], prefix_checkpoint_bindings_sha256=manifest['bindings_sha256'],
            endpoint_raw_refs={key: prefix['store_refs']['payload'] for key in ('parameters', 'adam', 'rng')})
        inputs['prefix_store'] = prefix['store_refs']
    request = {'schema': worker.REQUEST, 'branch': branch,
        'plan_sha256': raw.digest({'qualification_batches': batches, 'branch': branch}),
        'study_request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'static_recipe_sha256': context['static_recipe_sha256'], 'code_sha256': context['bindings']['source']['code_sha256'],
        'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
        'ingestion_receipt_sha256': declaration['ingestion_receipt_sha256'], 'lineage_declaration': declaration,
        'common_context_ref': parent['common_context_ref'], 'initializer_inputs': inputs,
        'lease_delegation_path': str(initializer.child_directory(context, branch) / 'LEASE_DELEGATION.json')}
    # Validate without mutating the frozen common scope or inventing observed
    # child request/worker fields which cannot exist before this file is written.
    expected = dict(context, checkpoint_child_declaration=declaration)
    worker.validate_request(request, expected)
    return copy.deepcopy(request)


def child_launch_spec(context, branch, request_ref, parent_proof):
    templates = context['bindings']['scope']['checkpoint_processes']['launch_templates']
    require(type(templates) is dict and set(templates) == set(cp.BRANCHES), 'all three frozen native launch templates')
    spec = copy.deepcopy(templates[branch])
    require(not {'request_ref', 'request_sha256', 'scope_sha256', 'parent_identity', 'argv'} & set(spec),
            'launch template excludes derived raw request/parent fields')
    os_identity = parent_proof['os_identity']
    spec.update(schema=launcher.SPEC, mode='registered_native_worker', request_ref=copy.deepcopy(request_ref),
        request_sha256=context['request_sha256'], scope_sha256=context['scope_sha256'],
        parent_identity={key: os_identity[key] for key in ('pid', 'creation_filetime_100ns', 'boot_identity')})
    spec['argv'] = [spec['python_ref']['path'], '-m', spec['worker_module'], '--request', request_ref['path']]
    require(spec['approved_code_sha256'] == context['bindings']['source']['code_sha256']
            and spec['source_checkout'] == context['bindings']['scope']['native_backend']['source_checkout'],
            'child launch exact common source checkout/hash map')
    launcher.validate_spec(spec)
    return spec


def checkpoint_observations(owner, parent_snapshot, reduced, terminal_proofs, restored, binding):
    """Reduce retained mechanics; native entry separately consumes actual handles.

    Terminal dictionaries here are the already captured outputs of same-handle
    OS verification, not permissions. Calling this arithmetic helper cannot
    establish execution or populate any native issuer/verifier registry.
    """
    from . import ce_worker_evidence as evidence
    require(type(owner) is accounting.QualificationAccounting and owner.progress_record() == restored.get('progress')
            and owner.progress_record()['C'] == owner.progress_record()['U'] == 2
            and not owner.requires_restore and not owner.restore_pending
            and restored.get('schema') == cp.RESTORE_SCHEMA and restored.get('source_worker_binding') == binding,
            'actual installed parent U2 fresh restore fence')
    own = [op for op in owner.operations.values() if op['stage'] == 'checkpoint_resume'
            and op['worker_id'] == binding['worker_identity_sha256'] and op['attempt_id'] == binding['attempt_id']]
    returned = [op for op in own if op['kind'] == 'checkpoint_restore' and op['phase'] == 'returned'
        and all(op['result'].get(key) == restored.get(key) for key in ('manifest_sha256', 'C', 'U', 'generation', 'symbolic_snapshot'))]
    require(len(returned) == 1, 'current parent returned restore operation proves installed fence')
    cp.validate_restore_fence(restored, parent_snapshot['progress'],
        dict(returned[0], source_worker_binding=binding), current_binding=binding)
    branches = reduced.get('branches')
    require(type(branches) is dict and set(branches) == set(cp.BRANCHES)
            and reduced.get('parent_costs_imported') is False, 'all three raw-replayed independent child branches')
    require(type(terminal_proofs) is dict and set(terminal_proofs) == set(cp.BRANCHES), 'all actual retained child terminal results')
    identities = []
    for branch in cp.BRANCHES:
        proof, state = terminal_proofs[branch], branches[branch]
        require(type(proof) is dict and proof.get('schema') == evidence.TERMINAL and proof.get('status_file_used') is False
                and type(proof.get('exit_code')) is int and proof['exit_code'] == 0
                and raw._int(proof.get('exit_filetime_100ns'), 1)
                and type(proof.get('os_identity')) is dict
                and raw._int(proof['os_identity'].get('pid'), 1)
                and raw._int(proof['os_identity'].get('creation_filetime_100ns'), 1)
                and proof['exit_filetime_100ns'] >= proof['os_identity']['creation_filetime_100ns']
                and proof.get('worker_identity_sha256') == state['binding']['worker_identity_sha256']
                and all(proof.get(key) == binding[key] for key in ('request_sha256', 'scope_sha256')),
                'captured same-handle successful child terminal/branch identity')
        identities.append(proof['worker_identity_sha256'])
        require(state.get('negative_controls_independently_recomputed') is True
                and raw._int(state.get('failed_restore_operations'), 1)
                and raw._int(state.get('returned_restore_operations'), 1)
                and raw._int(state.get('checkpoint_save_operations'), 1), 'raw child save/corruption/stale/recovery evidence')
        accounting.q._costs(state['derived_costs'])
    require(len(set(identities)) == 3 and binding['worker_identity_sha256'] not in identities,
            'three distinct actual child identities and separate parent')
    require(not owner.fatal and not owner.unknown_wall
            and all(owner.cumulative[key] >= parent_snapshot['cumulative'][key] for key in accounting.COST_FIELDS),
            'current parent retains prior full reservations; unknown/fatal work cannot certify controls')
    comparison = reduced['U2_state_comparison']
    parent_comparison = reduced['parent_state_comparison']
    names = ('parameters_bitwise', 'adam_bitwise', 'module_modes_bitwise', 'python_rng_bitwise',
             'numpy_rng_bitwise', 'torch_cpu_rng_bitwise', 'torch_cuda_rng_bitwise')
    require(all(comparison.get(key) is True and parent_comparison.get(key) is True for key in names),
            'actual raw parent/uninterrupted/resumed U2 parameter/Adam/module/RNG byte comparison')
    return {**{key: comparison[key] and parent_comparison[key] for key in names},
        'checkpoint_saves': sum(op['kind'] == 'checkpoint_save' and op['phase'] == 'returned' for op in own)
            + sum(state['checkpoint_save_operations'] for state in branches.values()),
        'checkpoint_restores': sum(op['kind'] == 'checkpoint_restore' and op['phase'] == 'returned' for op in own)
            + sum(state['returned_restore_operations'] for state in branches.values()),
        'checkpoint_restore_attempts': sum(op['kind'] == 'checkpoint_restore' for op in own)
            + sum(state['failed_restore_operations'] + state['returned_restore_operations'] for state in branches.values()),
        'actual_processes': 1 + len(set(identities)), 'fresh_fence_installed': True,
        'stale_work_rejected': all(state['negative_controls_independently_recomputed'] for state in branches.values()),
        'corruption_fails_closed': all(state['failed_restore_operations'] >= 1 for state in branches.values()),
        'failed_unknown_costs_retained': True}


def external_result(reference, observations, reduced, counts):
    """Exact NativeQualificationExecutor._external contract, still unqualified."""
    return {'raw_artifacts': [reference], 'observations': observations, 'raw_reduction': reduced,
            'native_qualified': False, 'execution_verified': False, 'outer_charged_bounds': copy.deepcopy(counts)}


class CheckpointProcesses:
    """Actual source primitives, one separate durable outer ticket for process IO."""
    def __init__(self, context, permit, prepared_receipt, proof, *, backend, owner, handoff,
                 parent_handle, bootstrap_handle, bootstrap_verifier=None, launch_issuer=None,
                 launch_verifier=None):
        fixture = type(context) is dict and context.get('schema') == accounting.FIXTURE
        if not fixture:
            authority.require_execution_permit(permit, context, 'checkpoint_resume')
            boot.require_registered_verifier(bootstrap_verifier)
            require(any(launch_verifier is item for item in launcher._REGISTERED_LAUNCH_VERIFIERS),
                    'source-owned checkpoint launch issuer/verifier is not registered')
        else:
            require(context.get('mode') == 'synthetic_fixture' and context.get('approved') is False
                    and context.get('training_authorized') is False and permit is None,
                    'explicit checkpoint process metadata fixture only')
        ticket_owner = context['bindings']['scope']['checkpoint_processes_owner']
        boot.verify_prepayment(bootstrap_handle, context, proof, ticket_owner, verifier=bootstrap_verifier)
        self.handle = bootstrap_handle; self.handle.begin()
        self.context, self.permit, self.proof = context, permit, copy.deepcopy(proof)
        self.fixture, self.backend, self.accounting, self.handoff = fixture, backend, owner, handoff
        self.parent_handle, self.launch_issuer, self.launch_verifier = parent_handle, launch_issuer, launch_verifier
        self.spec = copy.deepcopy(context['bindings']['scope']['checkpoint_processes'])
        self.counts = dict.fromkeys(accounting.COST_FIELDS, 0); self.operations, self.events = {}, []
        # NativeLaunchPermit binds callback object identity, so retain this one
        # bound method across construction, launch and child observation.
        self.operation = self.operation
        self.started = self.last_clock = time.monotonic(); self.parent = self.restore_record = None
        self.units, self.results = {}, []
        self.terminal_handle_records, self.final_handle_lifetime_closed = {}, False
        self.prefix_terminal_handle = None
        try:
            self.operation('checkpoint_process_setup', 'started', 'setup', metadata_bytes_upper=0,
                identity_hash_attempts_upper=16, identity_hash_bytes_upper=64 * raw.MAX_JSON)
            require(self.spec.get('schema') == SPEC and self.spec.get('owned_directory') == str(
                    Path(context['bindings']['scope']['checkpoint_qualification']['owned_directory']) / 'processes'),
                    'exact separately owned checkpoint process directory')
            require(ticket_owner == {'accounting_spec_sha256': raw.digest(self.spec),
                'prepared_receipt_sha256': prepared_receipt['receipt_sha256']}, 'exact prepaid process policy/ingestion owner')
            accounting.q._costs(self.spec['bounds'])
            require(self.handle.evidence['bounds'] == self.spec['bounds']
                    and raw._int(self.spec.get('max_json_bytes'), 1) and self.spec['max_json_bytes'] <= raw.MAX_JSON,
                    'actual full outer reservation and bounded checkpoint process JSON')
            require(type(owner) is accounting.QualificationAccounting and owner.context is context,
                    'actual own parent qualification ledger required')
            if not fixture:
                from .ce_native_backend import NativeCEBackend
                from .ce_checkpoint_lease import SupervisorGpuLease
                from . import ce_worker_evidence as evidence
                require(type(backend) is NativeCEBackend and type(handoff) is SupervisorGpuLease
                        and backend.context is context and handoff.context is context and backend.permit is permit
                        and handoff.permit is permit and handoff.operation is backend.operation
                        and type(parent_handle) is evidence.WinWorkerHandle,
                        'same actual parent/backend/supervisor lease/observer required')
                authority.require_execution_permit(permit, context, 'checkpoint_resume', operation=backend.operation, gate=backend.gate)
                actual = evidence.verify_worker_handle(parent_handle, {key: proof[key] for key in
                    ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}, require_native=True, terminal=False)
                require(actual == proof, 'actual live parent proof differs from checkpoint process owner')
            else:
                require(backend is None and handoff is None and parent_handle is None,
                        'metadata fixture cannot supply model/process/lease objects')
            self.root = Path(context['bindings']['scope']['native_backend']['output_directory'])
            self.directory = Path(self.spec['owned_directory'])
            require(self.root.is_absolute() and self.directory.is_absolute()
                    and self.directory.is_relative_to(self.root), 'process artifacts beneath admitted output root')
            self.prepared = copy.deepcopy(prepared_receipt)
            self.reader = raw.RawEvidenceReader(self.operation, self.spec['reader_bounds'], self.root)
            self.operation('checkpoint_process_setup', 'returned', 'setup')
        except BaseException as error:
            self.handle.fail(error); raise

    def tick(self):
        now = time.monotonic(); require(now >= self.last_clock, 'checkpoint process monotonic clock regressed')
        self._charge({'wall_seconds_upper': now - self.last_clock}); self.last_clock = now

    def _charge(self, amounts):
        for key, amount in amounts.items():
            require(key in self.counts and accounting.q.finite(amount) and amount >= 0
                    and (key == 'wall_seconds_upper' or type(amount) is int), 'finite outer checkpoint process charge')
            require(self.counts[key] + amount <= self.spec['bounds'][key], 'checkpoint process outer ceiling: ' + key)
        for key, amount in amounts.items(): self.counts[key] += amount

    def operation(self, kind, phase, operation_id, **fields):
        """Full ticket is durably prepaid; local starts retain all upper charges."""
        self.tick()
        require(type(operation_id) is str and operation_id and phase in ('started', 'returned', 'failed'), 'exact outer operation identity/phase')
        if phase == 'started':
            require(operation_id not in self.operations, 'outer checkpoint operation identity reused')
            amount = {'operation_attempts_upper': fields.get('operation_attempts_upper', 1),
                'io_read_bytes_upper': fields.get('io_read_bytes_upper', fields.get('read_bytes_upper', 0)),
                'io_write_bytes_upper': fields.get('io_write_bytes_upper', fields.get('write_bytes_upper', 0)),
                'identity_hash_attempts_upper': fields.get('identity_hash_attempts_upper', 0),
                'identity_hash_bytes_upper': fields.get('identity_hash_bytes_upper', 0),
                'metadata_bytes_upper': fields.get('metadata_bytes_upper', 0)}
            self._charge(amount); self.operations[operation_id] = {'kind': kind, 'phase': phase, 'upper_charges': amount}
        else:
            require(operation_id in self.operations and self.operations[operation_id]['kind'] == kind
                    and self.operations[operation_id]['phase'] == 'started', 'matching pending outer checkpoint operation')
            self.operations[operation_id].update(phase=phase, result=copy.deepcopy(fields))
        self.events.append({'kind': kind, 'phase': phase, 'operation_id': operation_id, 'fields': copy.deepcopy(fields)})

    def _admit(self, context, permit, *, native=False):
        require(context is self.context and permit is self.permit, 'same checkpoint process context/permit')
        if not self.fixture:
            authority.require_execution_permit(permit, context, 'checkpoint_resume',
                operation=self.backend.operation, gate=self.backend.gate)
        require(not native or not self.fixture, 'metadata fixture cannot execute models/processes')

    def _file_ref(self, path, maximum):
        from . import update
        with update._span(self.operation, 'checkpoint_process_reference', {'read_bytes_upper': maximum + 1,
                'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': maximum + 1}) as event:
            path = Path(path)
            require(path.is_absolute() and path.resolve().is_relative_to(self.root.resolve())
                    and path.is_file() and not path.is_symlink(), 'regular owned checkpoint process artifact')
            size = path.stat().st_size; require(0 < size <= maximum, 'bounded checkpoint process artifact bytes')
            hashed, count = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while block := stream.read(min(raw.CHUNK, size + 1 - count)):
                    count += len(block); require(count <= size, 'checkpoint process artifact grew'); hashed.update(block)
            require(count == size, 'checkpoint process artifact truncated')
            result = {'path': str(path), 'sha256': hashed.hexdigest(), 'bytes': count}
            event.update(bytes_read=count, identity_hash_attempts_actual=1, identity_hash_bytes_actual=count)
            return result

    def write_json(self, path, value, *, role=None, binding=None):
        from . import update
        maximum = self.spec['max_json_bytes']
        with update._span(self.operation, 'checkpoint_process_publication', {'read_bytes_upper': 2 * maximum + 2,
                'write_bytes_upper': maximum, 'metadata_bytes_upper': maximum,
                'identity_hash_attempts_upper': 2, 'identity_hash_bytes_upper': maximum + raw.MAX_BINDING_BYTES}) as event:
            with self.reader._overhead('checkpoint_process_JSON_serialization', work_bytes=32 * maximum):
                blob = raw.canonical(value) + b'\n'; require(len(blob) <= maximum, 'checkpoint process JSON byte ceiling')
            path = Path(path); require(path.is_absolute() and path.resolve().is_relative_to(self.root.resolve())
                and not path.is_symlink() and not os.path.lexists(path), 'create-only checkpoint process artifact')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.parent / (path.name + '.' + uuid.uuid4().hex + '.partial')
            with temporary.open('xb') as stream:
                require(stream.write(blob) == len(blob), 'short checkpoint process JSON write'); stream.flush(); os.fsync(stream.fileno())
            with temporary.open('rb') as stream: observed = stream.read(len(blob) + 1)
            require(observed == blob, 'checkpoint process temporary readback mismatch')
            publication._atomic_no_replace(temporary, path, path.parent)
            with path.open('r+b') as stream: os.fsync(stream.fileno()); observed = stream.read(len(blob) + 1)
            require(observed == blob, 'checkpoint process final readback mismatch')
            reference = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
            if role is not None:
                raw.binding(binding); reference.update(role=role, binding_sha256=raw.digest(binding))
            event.update(bytes_read=2 * len(blob), bytes_written=len(blob), identity_hash_attempts_actual=1 + int(role is not None),
                identity_hash_bytes_actual=len(blob) + (len(raw.canonical(binding)) if role is not None else 0))
            return reference

    def copy_frozen_file(self, source, destination, maximum):
        """Retain a fixed ledger prefix; a reference to a growing file is invalid."""
        from . import update
        with update._span(self.operation, 'checkpoint_process_frozen_copy', {'read_bytes_upper': 2 * maximum + 2,
                'write_bytes_upper': maximum, 'metadata_bytes_upper': maximum,
                'identity_hash_attempts_upper': 2, 'identity_hash_bytes_upper': 2 * maximum + 2}) as event:
            source, destination = Path(source), Path(destination)
            require(source.is_absolute() and source.resolve().is_relative_to(self.root.resolve())
                    and source.is_file() and not source.is_symlink() and destination.is_absolute()
                    and destination.resolve().is_relative_to(self.root.resolve())
                    and not destination.is_symlink() and not os.path.lexists(destination), 'regular create-only fixed ledger copy')
            size = source.stat().st_size; require(0 < size <= maximum, 'bounded fixed ledger prefix before read')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / (destination.name + '.' + uuid.uuid4().hex + '.partial')
            hashed, count = hashlib.sha256(), 0
            with source.open('rb') as original, temporary.open('xb') as target:
                while block := original.read(min(raw.CHUNK, size + 1 - count)):
                    count += len(block); require(count <= size, 'parent ledger grew during retained prefix copy')
                    hashed.update(block); require(target.write(block) == len(block), 'short retained ledger prefix write')
                target.flush(); os.fsync(target.fileno())
            require(count == size == source.stat().st_size, 'parent ledger changed during retained prefix copy')
            publication._atomic_no_replace(temporary, destination, destination.parent)
            readback, observed = hashlib.sha256(), 0
            with destination.open('rb') as stream:
                while block := stream.read(min(raw.CHUNK, size + 1 - observed)):
                    observed += len(block); require(observed <= size, 'retained ledger copy grew'); readback.update(block)
            require(observed == size and readback.hexdigest() == hashed.hexdigest(), 'retained ledger prefix readback differs')
            event.update(bytes_read=count + observed, bytes_written=count,
                identity_hash_attempts_actual=2, identity_hash_bytes_actual=count + observed)
            return {'path': str(destination), 'sha256': hashed.hexdigest(), 'bytes': size}

    def capture_ledger(self, owner, name):
        require(owner is self.accounting and name in ('PARENT_PARTIAL', 'PARENT_POST_RESTORE'), 'own exact retained parent ledger phase')
        snapshot = owner.snapshot(); head, count = snapshot['ledger_head'], snapshot['events']
        refs = {'header': self.copy_frozen_file(owner.header_path, self.directory / (name + '_HEADER.json'), execution.MAX_HEADER),
            'journal': self.copy_frozen_file(owner.path, self.directory / (name + '_JOURNAL.jsonl'), execution.MAX_LEDGER),
            'seal': self.write_json(self.directory / (name + '_SNAPSHOT.json'), snapshot)}
        require(owner._head() == head and len(owner.events) == count and owner.snapshot() == snapshot,
                'parent state/head changed while retaining canonical ledger prefix')
        return refs, snapshot

    def save_parent(self, context, permit, backend, owner, binding):
        self._admit(context, permit, native=True)
        require(backend is self.backend and owner is self.accounting and self.parent is None
                and owner.active['stage'] == 'checkpoint_resume' and owner.progress_record()['C'] == owner.progress_record()['U'] == 2,
                'one actual parent U2 checkpoint before child handoff')
        self.handoff.assert_parent_use()
        bindings = {'checkpoint_mode': cp.MODE, 'synthetic_cpu': False, 'role': 'qualification_only',
            'qualification_batches': cp.qualification_batches(context), 'code': context['bindings']['source']['code_sha256'],
            'runtime': context['bindings']['runtime']['decoding_preimage'], 'recipe': context['bindings']['recipe']['recipe'],
            'inventory': context['bindings']['scope']['native_backend']['parameter_inventory'],
            'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
            'own_start_sha256': context['bindings']['anchors']['own_start_sha256'], 'qualification_evidence_binding': binding,
            'data': {'ingestion_receipt_sha256': self.prepared['receipt_sha256']}}
        from . import update
        with update._span(backend.operation, 'qualification_checkpoint_configuration', owner.profiles['qualification_checkpoint_configuration']):
            checkpoints = cp.QualificationCheckpoints(context, permit, self.directory / 'parent_checkpoints', bindings,
                sum(math.prod(item['shape']) for item in bindings['inventory']), operation=backend.operation,
                gate=backend.gate, metadata_reserve=owner.reserve_metadata)
        checkpoints.attach_ledger(backend.operation)
        manifest = checkpoints.save(backend.model, backend.optimizer, owner.progress_record(), backend.operation)
        manifest_ref = self._file_ref(Path(manifest['payload']['path']).with_suffix('.json'), self.spec['max_json_bytes'])
        common = dict(authority.context_preimage(context), prepared_receipt=self.prepared)
        common_ref = self.write_json(self.directory / 'COMMON_STUDY_CONTEXT.json', common)
        # All publication/reference work here is separately prepaid outer work;
        # it creates no Q events after the partial snapshot/head is captured.
        refs, snapshot = self.capture_ledger(owner, 'PARENT_PARTIAL')
        stage = {'schema': 'CE8-CHECKPOINT-PARENT-STAGE-SEAL-1', 'stage': 'checkpoint_resume', 'completed': False,
            'worker_open': True, 'C': 2, 'U': 2, 'worker_identity_sha256': binding['worker_identity_sha256'],
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'ledger_head': snapshot['ledger_head'], 'binding_sha256': snapshot['binding_sha256']}
        self.checkpoints = checkpoints
        self.parent_snapshot = copy.deepcopy(snapshot)
        self.parent = {'binding': copy.deepcopy(binding), 'proof': self.proof, 'manifest': manifest,
            'manifest_ref': manifest_ref, 'ledger_refs': refs, 'stage_seal': stage,
            'common_context_ref': common_ref, 'common_context': common}
        return copy.deepcopy(self.parent)

    def launch(self, context, permit, plan, parent, results):
        self._admit(context, permit, native=True)
        branch = plan['branch']
        require(parent == self.parent and branch == cp.BRANCHES[len(results)]
                and results == self.results and branch not in self.units, 'exact sequential owned checkpoint branch')
        prefix = next((result for result in results if result['branch'] == 'prefix'), None)
        with self.reader._overhead('checkpoint_child_request_derivation', work_bytes=32 * raw.MAX_JSON,
                hash_attempts=8, hash_bytes=32 * raw.MAX_JSON):
            request = child_request(context, parent, branch, prefix=prefix)
        reference = self.write_json(initializer.child_directory(context, branch) / 'IMMUTABLE_REQUEST.json', request)
        with self.reader._overhead('checkpoint_launch_scope_derivation', work_bytes=32 * raw.MAX_JSON,
                hash_attempts=8, hash_bytes=32 * raw.MAX_JSON):
            spec = child_launch_spec(context, branch, reference, self.proof)
            from .ce_checkpoint_configuration import prepare_child_configuration
            spec, configuration_ref = prepare_child_configuration(self, request, reference, spec, parent)
            require(self.launch_issuer is CheckpointProcesses, 'fixed source checkpoint launch issuer class marker')
            self._active_launch_spec = copy.deepcopy(spec)
            launch_permit = launcher.NativeLaunchPermit(spec, self, self.launch_verifier, self.operation)
        # Existing launch authority/observation-factory registration checks run
        # before any Popen call; neither raw JSON nor this class enables them.
        owned = launcher.launch_native(spec, launch_permit, self.operation)
        require(type(owned) is launcher.OwnedProcess, 'exact owned native checkpoint process')
        self.units[branch] = {'owned': owned, 'request': request, 'request_ref': reference, 'spec': spec,
            'configuration_ref': configuration_ref}
        return owned

    def bound_request(self, plan, parent, results):
        self._admit(self.context, self.permit, native=True)
        unit = self.units[plan['branch']]
        require(parent == self.parent and results == self.results and 'delegation' not in unit, 'one post-handshake child delegation')
        delegation = self.handoff.bind_child(unit['owned'], plan['branch'], unit['request_ref'])
        if plan['branch'] == 'resumed':
            from .ce_prefix_handle_transfer import transfer_prefix
            require(self.prefix_terminal_handle is not None, 'original terminal prefix handle retained before resumed launch')
            prefix = self.units['prefix']
            prefix_result = next(value for value in self.results if value['branch'] == 'prefix')
            delegation['prefix_handle_transfer'] = transfer_prefix(self.context, self.permit, self.prefix_terminal_handle,
                prefix['live_proof'], prefix_result['manifest_ref'], unit['owned'], delegation, self.operation, self.backend.gate)
        reference = self.write_json(unit['request']['lease_delegation_path'], delegation)
        unit.update(delegation=delegation, delegation_ref=reference, live_proof=copy.deepcopy(self.handoff.child[1]))
        request = unit['request']
        return {'schema': worker.COMMAND, 'branch': request['branch'], 'plan_sha256': request['plan_sha256']}

    def timeout_seconds(self, plan):
        self._admit(self.context, self.permit)
        return self.spec['launch_templates'][plan['branch']]['deadline_seconds']

    def verify_child(self, owned, plan, response, terminal, parent_proof):
        self._admit(self.context, self.permit, native=True)
        from . import ce_worker_evidence as evidence
        unit = self.units[plan['branch']]
        require(owned is unit['owned'] and type(owned) is launcher.OwnedProcess and type(owned.monitor) is evidence.WinWorkerHandle
                and parent_proof == self.proof, 'exact owned checkpoint child and original parent proof')
        primary, result = None, None
        try:
            terminal_proof = evidence.verify_worker_handle(owned.monitor, {
                'worker_identity_sha256': owned.worker_identity_sha256, 'request_sha256': self.context['request_sha256'],
                'scope_sha256': self.context['scope_sha256']}, require_native=True, terminal=True)
            require(terminal.get('exit_code') == 0 and terminal.get('status_file_used') is False
                    and all(terminal[key] == terminal_proof[key] for key in
                        ('worker_identity_sha256', 'request_sha256', 'scope_sha256', 'exit_code', 'exit_filetime_100ns')),
                    'actual same-handle successful checkpoint child terminal')
            require(type(response) is dict and type(response.get('raw_artifact')) is dict,
                    'raw checkpoint branch artifact response; flags do not prove execution')
            binding = {'stage': 'checkpoint_resume', 'request_sha256': self.context['request_sha256'],
                'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
                'code_sha256': self.context['bindings']['source']['code_sha256'],
                'worker_identity_sha256': owned.worker_identity_sha256,
                'attempt_id': self.context['bindings']['scope']['checkpoint_qualification']['child_initializer']['branch_attempt_ids'][plan['branch']]}
            payload = reduction._envelope(self.reader, response['raw_artifact'], binding)
            lineage = payload['observed_lineage']
            require(accounting.checkpoint_child_declaration(lineage) == unit['request']['lineage_declaration']
                    and lineage['child_request_ref'] == unit['request_ref']
                    and payload['child_context'].get('checkpoint_child_lease_delegation') == unit['delegation']
                    and payload['branch'] == plan['branch'], 'raw branch exact immutable request/OS/delegation lineage')
            # Current-child and supervisor observers retain different wording
            # about their view; compare the actual stable OS/source preimage.
            initializer.validate_parent_view(lineage['child_proof'], unit['live_proof'])
            reduced = reduction._branch(self.reader, self.parent['common_context'], response['raw_artifact'], binding, self.parent['binding'])
            manifest = reduction._json(self.reader, payload['final_manifest_ref'], binding, 'checkpoint_manifest')
            directory = Path(manifest['payload']['path']).parent
            from . import checkpoint as base
            refs = {'bindings': self._file_ref(directory / base.BINDING_FILE, raw.MAX_JSON),
                'journal': self._file_ref(directory / base.JOURNAL_FILE, base.MAX_JOURNAL),
                'manifest': payload['final_manifest_ref'],
                'payload': {'path': manifest['payload']['path'], 'sha256': manifest['payload']['file_sha256'], 'bytes': manifest['payload']['bytes']}}
            result = {'branch': plan['branch'], 'binding': binding, 'raw_ref': response['raw_artifact'],
                'manifest': manifest, 'manifest_ref': payload['final_manifest_ref'], 'store_refs': refs,
                'terminal_proof': terminal_proof, 'reduction': reduced, 'native_qualified': False}
            require(plan['branch'] not in self.terminal_handle_records, 'retain each actual terminal branch monitor once')
            final_clone = owned.monitor.retain_terminal()
            try:
                retained_terminal = evidence.verify_worker_handle(final_clone, {key: terminal_proof[key] for key in
                    ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}, require_native=True, terminal=True)
                require(retained_terminal == terminal_proof, 'retained original terminal clone changed child OS/source identity')
                terminal_ref = self.write_json(initializer.child_directory(self.context, plan['branch']) / 'RETAINED_TERMINAL.json',
                    {'schema': 'CE8-CHECKPOINT-RETAINED-TERMINAL-1', 'branch': plan['branch'], 'binding': binding,
                        'worker_proof': unit['live_proof'], 'terminal_proof': retained_terminal,
                        'native_qualified': False, 'execution_verified': False})
                self.terminal_handle_records[plan['branch']] = {'handle': final_clone, 'binding': copy.deepcopy(binding),
                    'proof': copy.deepcopy(unit['live_proof']), 'terminal': copy.deepcopy(retained_terminal), 'raw_ref': terminal_ref}
            except BaseException as error:
                try: final_clone.close()
                except BaseException as closing: raw._accounting(error, closing); raise error from closing
                raise
            if plan['branch'] == 'prefix':
                require(self.prefix_terminal_handle is None, 'retain original prefix terminal handle once')
                self.prefix_terminal_handle = owned.monitor.retain_terminal()
        except BaseException as error:
            primary = error
            try: self.handoff.mark_child_evidence_failed(owned, error)
            except BaseException as marking_error:
                primary.checkpoint_failure_marking = {'type': type(marking_error).__name__, 'message': str(marking_error)[:1000]}
        try: self.handoff.finish_child(owned, terminal)
        except BaseException as error:
            if primary is not None:
                primary.checkpoint_terminal_cleanup_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
                raise primary from error
            raise
        if primary is not None: raise primary
        if plan['branch'] == 'resumed': self.close_prefix_transfer()
        self.results.append(result); return copy.deepcopy(result)

    def retained_terminal_handles(self):
        """Return actual opaque objects to the source final verifier, never JSON."""
        self._admit(self.context, self.permit, native=True)
        from . import ce_worker_evidence as evidence
        require(not self.final_handle_lifetime_closed and set(self.terminal_handle_records) == set(cp.BRANCHES),
                'all three original child terminal handles retained through final replay')
        output = {}
        for record in self.terminal_handle_records.values():
            current = record['handle']
            require(type(current) is evidence.WinWorkerHandle and not current.closed, 'actual retained original terminal OS handle')
            actual = evidence.verify_worker_handle(current, {key: record['terminal'][key] for key in
                ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}, require_native=True, terminal=True)
            require(actual == record['terminal'], 'final replay same-handle terminal/source proof changed')
            output[record['binding']['worker_identity_sha256']] = current
        return output

    def close_terminal_handles(self, primary=None):
        """Release only these owned clones; final source verifier calls this last."""
        from . import ce_worker_evidence as evidence
        failures = []
        for branch, record in list(self.terminal_handle_records.items()):
            current = record['handle']
            try:
                require(type(current) is evidence.WinWorkerHandle, 'only actual retained terminal monitor may close')
                current.close()
            except BaseException as error: failures.append(error)
            finally:
                if type(current) is evidence.WinWorkerHandle and current.closed: self.terminal_handle_records.pop(branch, None)
        if not self.terminal_handle_records:
            self.final_handle_lifetime_closed = True
            if self.handle.evidence['status'] == 'STARTED':
                try:
                    if primary is None and not failures: self.handle.complete()
                    else: self.handle.fail(primary if primary is not None else failures[0])
                except BaseException as error: failures.append(error)
        if failures:
            target = primary if primary is not None else failures[0]
            target.checkpoint_terminal_handle_cleanup = [{'type': type(error).__name__, 'message': str(error)[:1000]} for error in failures]
            if primary is not None: raise primary from failures[0]
            raise target

    def close_prefix_transfer(self):
        """Read-only original clone; call only after resumed child quiescence."""
        current = self.prefix_terminal_handle
        if current is not None:
            try: current.close()
            finally: self.prefix_terminal_handle = None

    def restore_parent(self, context, permit, backend, owner, parent):
        self._admit(context, permit, native=True)
        require(parent == self.parent and backend is self.backend and owner is self.accounting, 'exact original parent checkpoint restore owner')
        self.handoff.assert_parent_use()
        restored = self.checkpoints.restore(backend.model, backend.optimizer)
        record, operation = initializer.bind_returned_restore(owner, restored, parent['binding'])
        require(record['manifest_sha256'] == parent['manifest']['manifest_sha256'], 'actual original retained parent U2 restored')
        self.restore_record = copy.deepcopy(record)
        return record, operation

    def retained_result(self, context, permit, results, parent, owner, binding):
        self._admit(context, permit, native=True)
        require(results == self.results and [value['branch'] for value in results] == list(cp.BRANCHES)
                and parent == self.parent and owner is self.accounting and binding == parent['binding']
                and self.restore_record is not None and owner.progress_record() == self.restore_record['progress']
                and not owner.requires_restore and not owner.restore_pending, 'all exact children and installed restored parent endpoint')
        branch_refs = {value['branch']: value['raw_ref'] for value in results}
        branch_bindings = {value['branch']: value['binding'] for value in results}
        reduced = reduction.reduce_checkpoint_branches(self.parent['common_context'], branch_refs, branch_bindings,
            binding, reader=self.reader, outer_preflight=self._reduction_preflight)
        child = reduced['branches']['uninterrupted']
        reduced['parent_state_comparison'] = cp.compare_checkpoint_states(self.reader, parent['manifest_ref'], binding,
            child['manifest_ref'], child['binding'],
            parameter_inventory=context['bindings']['scope']['native_backend']['parameter_inventory'],
            parent_binding=binding, child_lineages={child['binding']['worker_identity_sha256']: child['lineage']})
        terminal_proofs = {value['branch']: value['terminal_proof'] for value in results}
        with self.reader._overhead('checkpoint_stage_observation_reduction', work_bytes=32 * raw.MAX_JSON):
            observed = checkpoint_observations(owner, self.parent_snapshot, reduced, terminal_proofs, self.restore_record, binding)
        post_refs, snapshot = self.capture_ledger(owner, 'PARENT_POST_RESTORE')
        payload = {'schema': BUNDLE, 'branch_refs': branch_refs, 'branch_bindings': branch_bindings,
            'parent_binding': binding, 'parent_ledger_refs': parent['ledger_refs'], 'parent_manifest_ref': parent['manifest_ref'],
            'parent_restore_record': self.restore_record, 'child_terminal_proofs': terminal_proofs,
            'child_terminal_refs': {key: value['raw_ref'] for key, value in self.terminal_handle_records.items()},
            'parent_post_restore_snapshot_ref': post_refs['seal'], 'parent_post_restore_ledger_refs': post_refs,
            'outer_prepayment': self.handle.evidence,
            'native_qualified': False, 'execution_verified': False, 'classification': 'actual retained branch/replay/byte proofs; native qualification registry remains closed'}
        reference = self.write_json(self.directory / 'CHECKPOINT_BRANCHES.json', {'schema': raw.SCHEMA, **binding,
            'evidence_kind': 'checkpoint_branches', 'payload': payload}, role='checkpoint_branches', binding=binding)
        require(not any(op['phase'] == 'started' for op in self.operations.values()), 'all owned process/publication operations return before outer end')
        self.tick()
        # The outer ticket remains started until source final replay releases
        # all owned OS clones; later OS verification is still prepaid here.
        self.context['native_services']['checkpoint_terminal_owner'] = self
        return external_result(reference, observed, reduced, self.counts)

    def _reduction_preflight(self, name, contract):
        require(name == 'checkpoint_branch_reduction' and contract['reader_bounds'] == self.reader.remaining,
                'exact already-prepaid final reducer allowance')
        self.tick()
