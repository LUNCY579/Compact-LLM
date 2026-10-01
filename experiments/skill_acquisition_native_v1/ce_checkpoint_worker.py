"""Actual checkpoint child branch bodies behind closed local authority registries.

RPC carries commands and retained references only. A source-owned local issuer
must supply execution/observation permits, actual process evidence, bootstrap
prepayment and supervised GPU ownership before framework/model access. This
module does not activate any of those services or claim native qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys

from . import ce_native_authority as authority, ce_checkpoint_qualification as cp
from . import ce_qualification_artifacts as raw

REQUEST = 'CE8-CHECKPOINT-WORKER-REQUEST-1'
COMMAND = 'CE8-CHECKPOINT-WORKER-COMMAND-1'
RESULT = 'CE8-CHECKPOINT-WORKER-RESULT-1'
_REGISTERED_LOCAL_ISSUERS = frozenset()
_REGISTERED_CHILD_INITIALIZERS = frozenset()
_REGISTERED_CHILD_LEASE_SERVICES = frozenset()
_REGISTERED_CHILD_PUBLICATION_SERVICES = frozenset()
require = authority.require


class MissingCheckpointWorkerService(authority.NativeAdmissionError): pass


def validate_command(command, request):
    require(type(command) is dict and set(command) == {'schema', 'branch', 'plan_sha256'}
            and command['schema'] == COMMAND and command['branch'] == request['branch']
            and command['plan_sha256'] == request['plan_sha256'], 'exact source-bound checkpoint branch command')
    return command


def validate_request(request, context):
    require(type(request) is dict and request.get('schema') == REQUEST and request.get('branch') in cp.BRANCHES,
            'explicit checkpoint child request/branch')
    require(not {'approved', 'native_permit', 'observation_permit', 'training_authorized', 'qualification_authorized'} & set(request),
            'JSON request cannot supply execution capabilities')
    batches = cp.qualification_batches(context)
    plan = {'qualification_batches': batches, 'branch': request['branch']}
    require(request.get('plan_sha256') == raw.digest(plan)
            and request.get('request_sha256') == context['request_sha256']
            and request.get('scope_sha256') == context['scope_sha256']
            and request.get('static_recipe_sha256') == context['static_recipe_sha256']
            and request.get('code_sha256') == context['bindings']['source']['code_sha256']
            and request.get('common_anchor_sha256') == context['bindings']['anchors']['common_anchor_sha256']
            and request.get('ingestion_receipt_sha256') == context['bindings']['scope']['qualifier_accounting']['ingestion_receipt_sha256'],
            'checkpoint child plan/source/recipe/anchor/census binding')
    lineage = request.get('predecessor')
    require(type(lineage) is dict and lineage.get('schema') == 'CE8-CHECKPOINT-CHILD-PREDECESSOR-1'
            and lineage.get('branch') == request['branch'] and lineage.get('qualification_batches') == batches
            and lineage.get('inherited_U') == (1 if request['branch'] == 'resumed' else 0)
            and type(lineage['inherited_U']) is int, 'explicit independently verified child predecessor/prefix contract')
    require(request.get('worker_identity_sha256') == lineage.get('worker_identity_sha256')
            and raw._hex(request.get('worker_identity_sha256')), 'actual child worker identity bound to lineage')
    if request['branch'] == 'resumed':
        require(type(lineage.get('prefix_manifest_ref')) is dict
                and raw._hex(lineage.get('prefix_checkpoint_bindings_sha256'))
                and type(lineage.get('inherited_history')) is list and len(lineage['inherited_history']) == 1,
                'source-bound retained U1 checkpoint/history required')
    return copy.deepcopy(request)


def _service(value, registry, name):
    if not any(value is item for item in registry):
        raise MissingCheckpointWorkerService('source-registered checkpoint child ' + name + ' is missing before kernel')
    return value


class CheckpointChildHandler:
    def __init__(self, context, execution_permit, observation_permit, worker_handle, prepared, services):
        authority.require_execution_permit(execution_permit, context, 'checkpoint_resume')
        from . import ce_worker_evidence as worker
        require(type(observation_permit) is worker.WorkerObservationPermit
                and type(worker_handle) is worker.WinWorkerHandle, 'source-owned worker observation permit/actual handle required')
        require(type(services) is dict and callable(services.get('operation')) and callable(services.get('gate')),
                'stable child observer/resource gate required')
        self.operation, self.gate = services['operation'], services['gate']
        authority.require_execution_permit(execution_permit, context, 'checkpoint_resume',
                                           operation=self.operation, gate=self.gate)
        observation_permit.require(worker_handle.spec, self.operation)
        require(worker_handle.spec['expected_pid'] == os.getpid(), 'checkpoint child OS handle must describe this process')
        self.proof = worker.verify_worker_handle(worker_handle, {
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'worker_identity_sha256': worker_handle.identity_sha256}, require_native=True, terminal=False)
        self.initializer = _service(services.get('initializer'), _REGISTERED_CHILD_INITIALIZERS, 'predecessor/prefix initializer')
        self.lease = _service(services.get('lease'), _REGISTERED_CHILD_LEASE_SERVICES, 'supervised lease transfer')
        self.publisher = _service(services.get('publisher'), _REGISTERED_CHILD_PUBLICATION_SERVICES, 'retained terminal publication')
        require(all(callable(getattr(self.initializer, name, None)) for name in (
            'construct_accounting', 'predecessor_capability', 'begin_child_stage', 'checkpoint_directory',
            'admit_prefix_storage', 'bound_restore_result', 'open_owned_corruption_store', 'stale_fence_record',
            'finish_child_accounting')), 'complete recovery/prefix/negative-control initializer required before kernel')
        bounds = services.get('state_proof_bounds')
        inventory = context['bindings']['scope']['native_backend']['parameter_inventory']
        known_bytes = 12 * sum(math.prod(entry['shape']) for entry in inventory) + 4 * len(inventory)
        require(type(bounds) is dict and set(bounds) == {'read_bytes_upper', 'write_bytes_upper',
                'identity_hash_attempts_upper', 'identity_hash_bytes_upper'}
                and all(type(value) is int and value >= 0 for value in bounds.values())
                and bounds['identity_hash_attempts_upper'] >= 4 * len(inventory) + 3
                and bounds['identity_hash_bytes_upper'] >= known_bytes + raw.MAX_JSON,
                'complete state/recovery hash admission lower bound before kernel')
        self.context, self.permit, self.worker, self.prepared, self.services = context, execution_permit, worker_handle, prepared, services
        self.backend = self.accounting = self.checkpoints = None

    def _admit(self):
        authority.require_execution_permit(self.permit, self.context, 'checkpoint_resume',
                                           operation=self.operation, gate=self.gate)

    def _bindings(self, evidence_binding):
        admitted = self.context['bindings']; scope = admitted['scope']
        return {'checkpoint_mode': cp.MODE, 'synthetic_cpu': False, 'role': 'qualification_only',
            'qualification_batches': cp.qualification_batches(self.context),
            'code': admitted['source']['code_sha256'], 'runtime': admitted['runtime']['decoding_preimage'],
            'recipe': admitted['recipe']['recipe'], 'inventory': scope['native_backend']['parameter_inventory'],
            'common_anchor_sha256': admitted['anchors']['common_anchor_sha256'],
            'own_start_sha256': admitted['anchors']['own_start_sha256'],
            'qualification_evidence_binding': evidence_binding,
            'data': {'ingestion_receipt_sha256': self.prepared.receipt['receipt_sha256']}}

    def _state_digest(self, checkpoints):
        self._admit()
        backend, progress = self.backend, self.accounting.progress_record()
        from experiments.maxrl_comp002_pretraining_v1 import warmstart_checkpoint as streams
        bounds = self.services.get('state_proof_bounds')
        require(type(bounds) is dict and set(bounds) == {'read_bytes_upper', 'write_bytes_upper',
                'identity_hash_attempts_upper', 'identity_hash_bytes_upper'}
                and all(type(value) is int and value >= 0 for value in bounds.values()),
                'explicit prepriced complete state fingerprint bounds')
        from . import update
        with update._span(self.operation, 'qualification_state_fingerprint', dict(bounds, reason='negative_control_live_state')):
            self.gate('qualification_state_fingerprint')
            parameters, _, groups, _ = checkpoints.storage._scope(backend.model, backend.optimizer)
            tensors, initialized, rng, specifications = checkpoints.storage._capture(parameters, backend.optimizer, progress)
            expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            require(bounds['identity_hash_attempts_upper'] >= len(tensors) + 1
                    and bounds['identity_hash_bytes_upper'] >= expected_bytes + raw.MAX_JSON,
                    'complete parameter/Adam/RNG hash work lower bound before hashing')
            hashes = {}
            for name, tensor in tensors.items():
                hashed = hashlib.sha256()
                for chunk in streams.tensor_chunks(tensor): hashed.update(chunk)
                hashes[name] = hashed.hexdigest()
            metadata = {'tensor_hashes': hashes, 'specifications': specifications, 'Adam_initialized': initialized,
                'rng': rng, 'optimizer_groups': groups,
                'module_training': {name: module.training for name, module in backend.model.named_modules()}}
            return raw.digest(metadata)

    def _negative_controls(self, checkpoints, request, binding):
        before = self._state_digest(checkpoints)
        corrupt = self.initializer.open_owned_corruption_store(self.context, self.permit, self.accounting, request, binding)
        require(type(corrupt) is cp.QualificationCheckpoints, 'actual isolated corrupt checkpoint storage required')
        rejected = None
        try: corrupt.restore(self.backend.model, self.backend.optimizer)
        except Exception as error: rejected = {'type': type(error).__name__, 'message': str(error)[:1000]}
        require(rejected is not None, 'corrupted latest checkpoint must reject, never fallback')
        after_corruption = self._state_digest(checkpoints)
        require(before == after_corruption, 'corruption rejection changed live parameter/Adam/module/RNG state')
        stale = self.initializer.stale_fence_record(self.context, self.accounting, request, binding)
        stale_error = None
        try: self.accounting.install_restore(stale)
        except Exception as error: stale_error = {'type': type(error).__name__, 'message': str(error)[:1000]}
        require(stale_error is not None, 'stale generation/fence must reject')
        after_stale = self._state_digest(checkpoints)
        require(before == after_stale, 'stale fence rejection changed complete live state')
        # Failed native checkpoint work retains its discard/restore obligation
        # even when the no-mutation control passes. Perform a real saved-ancestor
        # restore and install the current child's fresh, returned-operation fence.
        before_progress = self.accounting.progress_record()
        restored = checkpoints.restore(self.backend.model, self.backend.optimizer)
        record, operation = self.initializer.bound_restore_result(self.accounting, restored, binding)
        cp.validate_restore_fence(record, before_progress, operation, current_binding=binding)
        self.accounting.install_restore(record); self.backend.refresh_policy(self.accounting)
        recovered = self._state_digest(checkpoints)
        require(recovered == before
                and self.accounting.progress_record()['history'] == before_progress['history']
                and self.accounting.progress_record()['C'] == self.accounting.progress_record()['U'] == before_progress['U']
                and not self.accounting.requires_restore and not self.accounting.restore_pending,
                'verified good ancestor restores exact endpoint/RNG/modes and clears fresh-fence obligation')
        return {'before_state_sha256': before, 'after_corruption_state_sha256': after_corruption,
            'after_stale_state_sha256': after_stale, 'corruption_error': rejected, 'stale_fence_error': stale_error,
            'recovered_state_sha256': recovered, 'recovery_restore_record': record,
            'classification': 'actual negative-control operations; failed reservations retained, not native PASS'}

    def run(self, request, command):
        self._admit()
        request = validate_request(request, self.context); validate_command(command, request)
        require(request['worker_identity_sha256'] == self.proof['worker_identity_sha256'], 'request actual child identity mismatch')
        require(self.backend is None, 'checkpoint child branch runs once')
        from .ce_qualification_accounting import QualificationAccounting, CheckpointChildAdmission
        from .ce_native_backend import NativeCEBackend
        # Source initializer proves parent/predecessor lineage under its own
        # admission handle. It supplies a durably prepaid independent ledger;
        # no direct C/U/history assignment is performed by this handler.
        accounting = self.initializer.construct_accounting(self.context, self.permit, self.proof, self.prepared.receipt, request)
        require(type(accounting) is QualificationAccounting and accounting.scope_role == 'qualification_only',
                'actual independent child qualification accounting required')
        self.accounting = accounting
        require(callable(getattr(self.operation, 'attach_qualification_accounting', None)), 'stable child accounting dispatcher')
        self.operation.attach_qualification_accounting(accounting)
        admission = self.initializer.predecessor_capability(self.context, self.permit, self.proof, request['predecessor'])
        require(type(admission) is CheckpointChildAdmission, 'source-issued exact checkpoint child predecessor capability')
        accounting.admit_checkpoint_child_prefix(request['predecessor'], admission_handle=admission,
            verifier=self.initializer.predecessor_verifier)
        initial_U = 1 if request['branch'] == 'resumed' else 0
        require(accounting.progress_record()['C'] == accounting.progress_record()['U'] == initial_U,
                'retained admitted predecessor installs exact logical child start')
        attempt_id = self.initializer.begin_child_stage(accounting, request)
        binding = {'stage': 'checkpoint_resume', 'request_sha256': self.context['request_sha256'],
            'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': attempt_id}
        backend = NativeCEBackend(self.context, self.permit); self.backend = backend
        primary, checkpoint, branch_result = None, None, None
        # The source-owned supervisor transfer does not compete for a second
        # lease while the parent holds it. Its registry is currently empty.
        with self.lease.enter_child(self.context, self.permit, self.proof, request):
            try:
                # Local bootstrap must leave CUDA uninitialized. This is the
                # one and only configure/load owner in this fresh OS process.
                backend.load(self.context, self.prepared, self.services['observer'])
                directory = self.initializer.checkpoint_directory(self.context, request)
                bindings = self._bindings(binding)
                checkpoints = cp.QualificationCheckpoints(self.context, self.permit, directory, bindings,
                    sum(math.prod(entry['shape']) for entry in bindings['inventory']),
                    operation=self.operation, gate=self.gate, metadata_reserve=accounting.reserve_metadata)
                checkpoints.attach_ledger(self.operation); self.checkpoints = checkpoints
                if request['branch'] == 'resumed':
                    # The original prefix binding/worker remains in an exact
                    # verified imported store. Resumed U2 is saved separately
                    # with this child's own source/evidence binding.
                    imported = self.initializer.admit_prefix_storage(self.context, self.permit, request, bindings)
                    require(type(imported) is dict and set(imported) == {'directory', 'bindings'}
                            and raw.digest(imported['bindings']) == request['predecessor']['prefix_checkpoint_bindings_sha256'],
                            'actual original imported prefix checkpoint binding')
                    prefix = cp.QualificationCheckpoints(self.context, self.permit, imported['directory'], imported['bindings'],
                        sum(math.prod(entry['shape']) for entry in bindings['inventory']),
                        operation=self.operation, gate=self.gate, metadata_reserve=accounting.reserve_metadata)
                    prefix.attach_ledger(self.operation)
                    restored = prefix.restore(backend.model, backend.optimizer)
                    record, operation = self.initializer.bound_restore_result(accounting, restored, binding)
                    cp.validate_restore_fence(record, accounting.progress_record(), operation, current_binding=binding)
                    accounting.install_restore(record); backend.refresh_policy(accounting)
                require(backend.U == initial_U, 'actual child Adam U equals retained prefix start')
                batches = cp.qualification_batches(self.context)
                stop_U = 1 if request['branch'] == 'prefix' else 2
                while accounting.progress_record()['U'] < stop_U:
                    U = accounting.progress_record()['U']
                    rows = list(self.prepared.iter_ids(batches[U]))
                    result = backend.update(rows, accounting); accounting.commit_boundary(result)
                    require(accounting.progress_record()['C'] == accounting.progress_record()['U'] == U + 1,
                            'one actual committed child CE update')
                checkpoint = checkpoints.save(backend.model, backend.optimizer, accounting.progress_record(), self.operation)
                controls = self._negative_controls(checkpoints, request, binding)
                # Raw state proof uses actual checkpoint payload/metadata. Its
                # source-owned publication service retains exact OS lineage.
                branch_result = self.publisher.retain_branch(self.context, self.permit, request, binding, checkpoint,
                    accounting, backend, negative_control_evidence=controls)
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    if backend.model is not None or backend.optimizer is not None: backend.cleanup()
                    self.initializer.finish_child_accounting(accounting, request, primary)
                    accounting.end()
                except BaseException as cleanup:
                    if primary is not None:
                        primary.checkpoint_worker_cleanup_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
                        raise primary from cleanup
                    raise
        # Terminal raw publication is deliberately after model cleanup and
        # supervisor lease return. The OS supervisor separately proves exit.
        return self.publisher.publish_after_release(self.context, self.permit, request, binding, branch_result, accounting,
            result_schema=RESULT, native_qualified=False)


def main(argv=None):
    # This denial precedes argument parsing, request paths, RPC/environment
    # inspection, source hashing, worker observation and framework imports.
    if not _REGISTERED_LOCAL_ISSUERS:
        raise MissingCheckpointWorkerService('source-owned checkpoint worker scope/bootstrap/permit issuer is not registered')
    require(len(_REGISTERED_LOCAL_ISSUERS) == 1, 'one reviewed source-owned child issuer required')
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    issuer = next(iter(_REGISTERED_LOCAL_ISSUERS))
    # Issuer owns bounded prepaid request/source/OS reads. It cannot be named
    # by JSON, imported from an arbitrary descriptor or serialized through RPC.
    return issuer.run_frozen_request(args.request, CheckpointChildHandler, validate_command)


if __name__ == '__main__': raise SystemExit(main())
