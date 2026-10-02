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
DECLARATION = 'CE8-CHECKPOINT-CHILD-LINEAGE-DECLARATION-1'
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
    require(not {'approved', 'native_permit', 'observation_permit', 'training_authorized', 'qualification_authorized',
                 'request_sha256', 'worker_identity_sha256', 'child_request_sha256', 'child_request_ref', 'child_proof'} & set(request),
            'JSON request cannot supply execution capabilities')
    batches = cp.qualification_batches(context)
    plan = {'qualification_batches': batches, 'branch': request['branch']}
    require(request.get('plan_sha256') == raw.digest(plan)
            and request.get('study_request_sha256') == context['request_sha256']
            and request.get('scope_sha256') == context['scope_sha256']
            and request.get('static_recipe_sha256') == context['static_recipe_sha256']
            and request.get('code_sha256') == context['bindings']['source']['code_sha256']
            and request.get('common_anchor_sha256') == context['bindings']['anchors']['common_anchor_sha256']
            and request.get('ingestion_receipt_sha256') == context['bindings']['scope']['qualifier_accounting']['ingestion_receipt_sha256'],
            'checkpoint child plan/source/recipe/anchor/census binding')
    lineage = request.get('lineage_declaration')
    require(type(lineage) is dict and lineage.get('schema') == DECLARATION
            and lineage.get('branch') == request['branch'] and lineage.get('qualification_batches') == batches
            and lineage.get('inherited_U') == (1 if request['branch'] == 'resumed' else 0)
            and type(lineage['inherited_U']) is int, 'explicit independently verified child predecessor/prefix contract')
    require(not {'worker_identity_sha256', 'child_request_sha256', 'child_request_ref', 'child_proof', 'declaration_sha256'} & set(lineage)
            and lineage == context.get('checkpoint_child_declaration'),
            'immutable lineage declaration excludes future OS identity/raw-file hashes')
    if request['branch'] == 'resumed':
        require(type(lineage.get('prefix_manifest_ref')) is dict
                and raw._hex(lineage.get('prefix_checkpoint_bindings_sha256'))
                and type(lineage.get('inherited_history')) is list and len(lineage['inherited_history']) == 1,
                'source-bound retained U1 checkpoint/history required')
    return copy.deepcopy(request)


def validate_observed_lineage(request, context, actual_proof):
    """Check the post-observation envelope; this function issues no capability.

    The native handler passes its refreshed actual WinWorkerHandle proof. The
    local issuer derives this envelope after bounded raw-file/OS observation and
    includes it in the execution permit preimage, never in the launch request.
    """
    from .ce_qualification_accounting import checkpoint_child_declaration
    observed = context.get('checkpoint_child_observed_lineage')
    require(type(observed) is dict and observed.get('schema') == 'CE8-CHECKPOINT-CHILD-PREDECESSOR-1'
            and checkpoint_child_declaration(observed) == request['lineage_declaration']
            and observed.get('declaration_sha256') == raw.digest(request['lineage_declaration']),
            'observed child envelope matches static frozen declaration')
    require(type(actual_proof) is dict and observed.get('child_proof') == actual_proof
            and observed.get('worker_identity_sha256') == actual_proof.get('worker_identity_sha256')
            and raw._hex(observed.get('worker_identity_sha256')), 'actual observed child proof/identity envelope')
    reference = observed.get('child_request_ref')
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
            and type(reference['path']) is str and raw._hex(reference['sha256']) and type(reference['bytes']) is int
            and reference['bytes'] > 0 and observed.get('child_request_sha256') == reference['sha256']
            and actual_proof.get('sources', {}).get('request_ref') == reference,
            'actual raw request reference binds OS-observed source without self-reference')
    require(observed.get('study_request_sha256') == context['request_sha256'],
            'stable logical study identity remains distinct from raw child file hash')
    return copy.deepcopy(observed)


def _service(value, registry, name):
    # These are exact source constructors. Subclasses and duck-typed objects
    # cannot replace constructor admission; every constructor separately binds
    # the full permit/context, real prepayment, OS handles and callbacks.
    if not any(type(value) is item for item in registry if type(item) is type):
        raise MissingCheckpointWorkerService('source-registered checkpoint child ' + name + ' is missing before kernel')
    return value


def expected_negative_error(error, message):
    """Accept a specific plain validation rejection, never infrastructure failure."""
    if type(error) is not ValueError or str(error) != message or vars(error) or error.__cause__ is not None:
        raise error
    return {'type': 'ValueError', 'message': message}


def validate_stale_attempt(record, current):
    require(type(record) is dict and record.get('schema') == cp.RESTORE_SCHEMA
            and type(record.get('progress')) is dict
            and type(current) is dict and all(type(current.get(key)) is int for key in ('C', 'U', 'generation'))
            and 0 <= current['C'] == current['U'] <= 2 and current['generation'] >= 0,
            'complete retained stale attempt')
    progress = record['progress']
    require(all(type(progress.get(key)) is int for key in ('C', 'U', 'generation'))
            and progress['C'] == progress['U'] == current['C'] == current['U']
            and progress.get('history') == current['history'] and 0 <= progress['generation'] <= current['generation']
            and all(type(record.get(key)) is int and record[key] == progress[key] for key in ('C', 'U', 'generation'))
            and record.get('history') == progress['history'] and record.get('install_fence_required') is True
            and raw._hex(record.get('manifest_sha256'))
            and raw._hex(progress.get('snapshot'))
            and progress['snapshot'] == progress.get('symbolic_snapshot') == record.get('snapshot') == record.get('symbolic_snapshot'),
            'exact stale endpoint/nested fields with nonfuture generation')
    return copy.deepcopy(record)


def failed_corruption_operation(operations, before_ids, binding):
    matches = [(identifier, op) for identifier, op in operations.items() if identifier not in before_ids and type(op) is dict
        and op.get('kind') == 'checkpoint_restore' and op.get('phase') == 'failed'
        and op.get('worker_id') == binding['worker_identity_sha256']
        and op.get('attempt_id') == binding['attempt_id'] and op.get('stage') == binding['stage']
        and type(op.get('result')) is dict
        and op['result'].get('error') == {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'}
        and type(op.get('started_seq')) is int and type(op.get('returned_seq')) is int
        and 0 <= op['started_seq'] < op['returned_seq']]
    require(len(matches) == 1, 'one newly retained current-worker corrupt checkpoint restore rejection')
    return {'operation_id': matches[0][0], 'operation': copy.deepcopy(matches[0][1])}


def run_child_lifetime(owner, enter, body, cleanup, finish):
    """Cleanup inside delegation; finish/end after its priced exit, also on failure.

    This ordering primitive supplies no execution authority. The native handler
    supplies its already admitted source-owned methods; pure fixtures can test
    the actual durable accounting guard without models or child processes.
    """
    from .ce_controller import _OwnedContext
    primary, result, last_failure = None, None, None
    try:
        with _OwnedContext(enter()):
            work_error = None
            try: result = body()
            except BaseException as error: work_error = error; raise
            finally:
                try: cleanup()
                except BaseException as error:
                    if work_error is not None:
                        work_error.checkpoint_worker_cleanup_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
                        raise work_error from error
                    raise
    except BaseException as error: primary = error
    try: finish(primary, result)
    except BaseException as error:
        if primary is None: primary = error
        else:
            primary.checkpoint_worker_finish_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
            last_failure = error
        # Retain a failed stage if readback/finish itself rejected before its
        # terminal row; never publish a completed stage or refund reservations.
        try:
            if owner.active is not None: owner.fail_stage(primary)
        except BaseException as record_error:
            primary.checkpoint_worker_failure_recording = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
            last_failure = record_error
    try: owner.end()
    except BaseException as error:
        if primary is None: primary = error
        else:
            primary.checkpoint_worker_end_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
            last_failure = error
    if primary is not None:
        if last_failure is not None: raise primary from last_failure
        raise primary
    require(result is not None, 'failed/incomplete child lifetime cannot publish terminal success')
    return result


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
        with update._span(self.operation, 'qualification_state_fingerprint', dict(bounds, reason='negative_control_live_state')) as record:
            self.gate('qualification_state_fingerprint')
            parameters, _, groups, _ = checkpoints.storage._scope(backend.model, backend.optimizer)
            tensors, initialized, rng, specifications = checkpoints.storage._capture(parameters, backend.optimizer, progress)
            expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            require(bounds['identity_hash_attempts_upper'] >= len(tensors) + 1
                    and bounds['identity_hash_bytes_upper'] >= expected_bytes + raw.MAX_JSON,
                    'complete parameter/Adam/RNG hash work lower bound before hashing')
            hashes, hashed_bytes = {}, 0
            for name, tensor in tensors.items():
                hashed = hashlib.sha256()
                for chunk in streams.tensor_chunks(tensor): hashed.update(chunk); hashed_bytes += len(chunk)
                hashes[name] = hashed.hexdigest()
            metadata = {'tensor_hashes': hashes, 'specifications': specifications, 'Adam_initialized': initialized,
                'rng': rng, 'optimizer_groups': groups,
                'module_training': {name: module.training for name, module in backend.model.named_modules()}}
            encoded = raw.canonical(metadata)
            require(len(encoded) <= raw.MAX_JSON and hashed_bytes == expected_bytes,
                    'bounded complete streamed state descriptor/hash byte count')
            state_hash = hashlib.sha256(encoded).hexdigest()
            record.update(state_sha256=state_hash, state_descriptor=metadata,
                identity_hash_attempts_actual=len(tensors) + 1,
                identity_hash_bytes_actual=hashed_bytes + len(encoded))
            return state_hash

    def _negative_controls(self, checkpoints, request, binding):
        before = self._state_digest(checkpoints)
        corrupt = self.initializer.open_owned_corruption_store(self.context, self.permit, self.accounting, request, binding)
        require(type(corrupt) is cp.QualificationCheckpoints, 'actual isolated corrupt checkpoint storage required')
        rejected = None; before_operations = set(self.accounting.operations)
        try: corrupt.restore(self.backend.model, self.backend.optimizer)
        except Exception as error: rejected = expected_negative_error(error, 'latest checkpoint payload bytes corrupt')
        require(rejected is not None, 'corrupted latest checkpoint must reject, never fallback')
        corruption_operation = failed_corruption_operation(self.accounting.operations, before_operations, binding)
        after_corruption = self._state_digest(checkpoints)
        require(before == after_corruption, 'corruption rejection changed live parameter/Adam/module/RNG state')
        stale = self.initializer.stale_fence_record(self.context, self.accounting, request, binding)
        stale = validate_stale_attempt(stale, self.accounting.progress_record())
        stale_error = None
        try: self.accounting.install_restore(stale)
        except Exception as error: stale_error = expected_negative_error(error, 'exact ancestor nested restore fence')
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
            'corruption_restore_operation': corruption_operation, 'stale_attempt_record': stale,
            'recovered_state_sha256': recovered, 'recovery_restore_record': record,
            'classification': 'actual negative-control operations; failed reservations retained, not native PASS'}

    def run(self, request, command):
        self._admit()
        request = validate_request(request, self.context); validate_command(command, request)
        predecessor = validate_observed_lineage(request, self.context, self.proof)
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
        admission = self.initializer.predecessor_capability(self.context, self.permit, self.proof, predecessor)
        require(type(admission) is CheckpointChildAdmission, 'source-issued exact checkpoint child predecessor capability')
        accounting.admit_checkpoint_child_prefix(predecessor, admission_handle=admission,
            verifier=self.initializer.predecessor_verifier)
        initial_U = 1 if request['branch'] == 'resumed' else 0
        require(accounting.progress_record()['C'] == accounting.progress_record()['U'] == initial_U,
                'retained admitted predecessor installs exact logical child start')
        attempt_id = self.initializer.begin_child_stage(accounting, request)
        binding = {'stage': 'checkpoint_resume', 'request_sha256': self.context['request_sha256'],
            'scope_sha256': self.context['scope_sha256'], 'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'attempt_id': attempt_id}
        def body():
            backend = NativeCEBackend(self.context, self.permit); self.backend = backend
            # Local bootstrap leaves CUDA uninitialized. This fresh process
            # owns its one configure/load while the supervised lease is held.
            backend.load(self.context, self.prepared, self.services['observer'])
            directory = self.initializer.checkpoint_directory(self.context, request)
            bindings = self._bindings(binding)
            checkpoints = cp.QualificationCheckpoints(self.context, self.permit, directory, bindings,
                sum(math.prod(entry['shape']) for entry in bindings['inventory']),
                operation=self.operation, gate=self.gate, metadata_reserve=accounting.reserve_metadata)
            checkpoints.attach_ledger(self.operation); self.checkpoints = checkpoints
            if request['branch'] == 'resumed':
                imported = self.initializer.admit_prefix_storage(self.context, self.permit, request, bindings)
                require(type(imported) is dict and set(imported) == {'directory', 'bindings'}
                        and raw.digest(imported['bindings']) == predecessor['prefix_checkpoint_bindings_sha256'],
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
            batches = cp.qualification_batches(self.context); stop_U = 1 if request['branch'] == 'prefix' else 2
            while accounting.progress_record()['U'] < stop_U:
                U = accounting.progress_record()['U']; rows = list(self.prepared.iter_ids(batches[U]))
                result = backend.update(rows, accounting); accounting.commit_boundary(result)
                require(accounting.progress_record()['C'] == accounting.progress_record()['U'] == U + 1,
                        'one actual committed child CE update')
            checkpoint = checkpoints.save(backend.model, backend.optimizer, accounting.progress_record(), self.operation)
            controls = self._negative_controls(checkpoints, request, binding)
            return self.publisher.retain_branch(self.context, self.permit, request, binding, checkpoint,
                accounting, backend, negative_control_evidence=controls)
        def cleanup():
            backend = self.backend
            if backend is not None and (backend.model is not None or backend.optimizer is not None): backend.cleanup()
        branch_result = run_child_lifetime(accounting,
            lambda: self.lease.enter_child(self.context, self.permit, self.proof, request), body, cleanup,
            lambda primary, retained: self.initializer.finish_child_accounting(accounting, request, primary, retained=retained))
        # Terminal raw publication is deliberately after model cleanup and
        # supervisor lease return. The OS supervisor separately proves exit.
        require(branch_result is not None, 'failed/incomplete child work cannot publish terminal success')
        return self.publisher.publish_after_release(self.context, self.permit, request, binding, branch_result, accounting,
            result_schema=RESULT, native_qualified=False)


def main(argv=None):
    # This denial precedes argument parsing, request paths, RPC/environment
    # inspection, source hashing, worker observation and framework imports.
    if not _REGISTERED_LOCAL_ISSUERS:
        from . import ce_checkpoint_configuration
        try: ce_checkpoint_configuration.install_child_configuration()
        except authority.NativeAdmissionError as error:
            if ce_checkpoint_configuration._SOURCE_ADOPTION is None:
                raise MissingCheckpointWorkerService('source-owned checkpoint worker scope/bootstrap/permit issuer is not registered') from error
            raise
    require(len(_REGISTERED_LOCAL_ISSUERS) == 1, 'one reviewed source-owned child issuer required')
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    issuer = next(iter(_REGISTERED_LOCAL_ISSUERS))
    # Issuer owns bounded prepaid request/source/OS reads. It cannot be named
    # by JSON, imported from an arbitrary descriptor or serialized through RPC.
    return issuer.run_frozen_request(args.request, CheckpointChildHandler, validate_command)


if __name__ == '__main__':
    from .ce_qualification_adoption import adopt_source_paths
    adopt_source_paths()
    raise SystemExit(main())
