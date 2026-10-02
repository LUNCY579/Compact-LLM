"""Controller bootstrap, stable dispatch and native resource/lifecycle services.

No source issuers/factories are registered. Construction requires a separately
issued metadata-service capability before IO. Concrete source bodies reuse the
qualified loader helpers; missing admission/publication/worker/lease-transfer
roles refuse explicitly. This module supplies no experiment decision, native
qualification, model default or execution grant.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from . import ce_controller as controller, ce_qualification as qualification, ce_worker_evidence as worker

SCHEMA = 'CE8-NATIVE-CONTROLLER-SERVICES-1'
BOOTSTRAP = 'CE8-SERVICES-BOOTSTRAP-1'
PROFILE = ('read_bytes_upper', 'write_bytes_upper', 'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_s_upper')
COUNTERS = ('read_bytes', 'write_bytes', 'hash_attempts', 'hash_bytes', 'metadata_bytes', 'operation_attempts', 'wall_seconds')
_REGISTERED_SERVICE_VERIFIERS = frozenset()
_REGISTERED_ROLE_FACTORIES = frozenset()
_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES = frozenset()
_SERVICE_SEAL = object()
require, digest, canonical = worker.require, worker.digest, worker.canonical


class MissingServiceRole(worker.WorkerEvidenceError): pass


class ControllerServicePermit:
    """Pre-model service metadata capability; no model or native launch authority."""
    __slots__ = ('_seal', '_request', '_spec', '_issuer', '_verifier')
    def __init__(self, request, spec, issuer, verifier):
        require(any(verifier is item for item in _REGISTERED_SERVICE_VERIFIERS), 'controller service issuer is not registered')
        require(issuer is not None and not isinstance(issuer, (dict, list, str, int, float, bool)), 'opaque service issuer required')
        request_sha, spec_sha = digest(controller.request_preimage(request)), digest(spec)
        require(verifier(issuer, request_sha, spec_sha) is True, 'controller service authority failed')
        for name, value in (('_seal', _SERVICE_SEAL), ('_request', request_sha), ('_spec', spec_sha), ('_issuer', issuer), ('_verifier', verifier)):
            object.__setattr__(self, name, value)
    def __setattr__(self, name, value): raise worker.WorkerEvidenceError('controller service permit immutable')
    def require(self, request, spec):
        require(getattr(self, '_seal', None) is _SERVICE_SEAL
                and any(getattr(self, '_verifier', None) is item for item in _REGISTERED_SERVICE_VERIFIERS), 'source-issued controller service authority required')
        require(digest(controller.request_preimage(request)) == self._request and digest(spec) == self._spec
                and self._verifier(self._issuer, self._request, self._spec) is True, 'controller request/spec authority drift')


def _admit(permit, request, spec):
    require(type(permit) is ControllerServicePermit, 'opaque controller service permit required before paths/IO/framework')
    permit.require(request, spec)


def _profile(value):
    require(type(value) is dict and set(value) == set(PROFILE)
            and all(type(value[key]) is int and value[key] >= 0 for key in PROFILE[:-1])
            and type(value['wall_s_upper']) in (int, float) and 0 < value['wall_s_upper'] < float('inf'), 'explicit finite service operation profile')
    return value


def resolve_role(descriptor, code):
    require(type(descriptor) is dict and set(descriptor) == {'module', 'symbol', 'source_sha256'}
            and type(descriptor['module']) is str and type(descriptor['symbol']) is str
            and worker._hex(descriptor['source_sha256']), 'JSON source-owned service role descriptor required')
    name = descriptor['module'].replace('.', '/') + '.py'
    require(code.get(name) == descriptor['source_sha256'], 'service role source bundle mismatch')
    matches = [item for item in _REGISTERED_ROLE_FACTORIES if getattr(item, '__module__', None) == descriptor['module']
               and getattr(item, '__name__', None) == descriptor['symbol']]
    if len(matches) != 1: raise MissingServiceRole('registered actual service role missing: ' + descriptor['symbol'])
    return matches[0]


class BootstrapLedger:
    """Create-only fsynced hash chain; failed/pending reservations never refund."""
    def __init__(self, spec, request_sha256, *, synthetic_fixture=False, prepayment=None):
        require(synthetic_fixture is True or any(type(prepayment) is item for item in _REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES)
                and callable(getattr(prepayment, 'require_constructor', None)), 'native bootstrap requires source-owned permanent outer constructor prepayment')
        if not synthetic_fixture: prepayment.require_constructor(digest(spec), request_sha256)
        self.spec, self.request_sha256 = copy.deepcopy(spec), request_sha256
        self.directory = Path(spec['owned_directory']); self.path = self.directory / 'SERVICE_BOOTSTRAP.jsonl'
        self.counters = dict.fromkeys(COUNTERS, 0); self.operations, self.events = {}, []
        self.started = time.perf_counter(); self.sealed = self.fatal = False
        require(self.directory.is_absolute() and not self.directory.is_symlink(), 'absolute owned bootstrap directory')
        require(type(spec.get('caps')) is dict and set(spec['caps']) == set(COUNTERS)
                and all(type(spec['caps'][key]) is int and spec['caps'][key] > 0 for key in COUNTERS if key != 'wall_seconds')
                and type(spec['caps']['wall_seconds']) in (int, float) and 0 < spec['caps']['wall_seconds'] < float('inf'), 'finite complete bootstrap caps')
        require(type(spec.get('operation_profiles')) is dict and spec['operation_profiles'], 'explicit bootstrap operation profiles')
        for profile in spec['operation_profiles'].values(): _profile(profile)
        # Actual outer accounting permanently pre-reserves this constructor;
        # the service permit alone is not prepayment.
        _profile(spec.get('setup_bounds'))
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not os.path.lexists(self.path), 'create-only service bootstrap ledger')
        self._event('setup', {'setup_bounds': spec['setup_bounds']}, self._price(spec['setup_bounds']))
    def _price(self, profile):
        return {'read_bytes': profile['read_bytes_upper'], 'write_bytes': profile['write_bytes_upper'],
            'hash_attempts': profile['identity_hash_attempts_upper'], 'hash_bytes': profile['identity_hash_bytes_upper'],
            'wall_seconds': profile['wall_s_upper']}
    def _event(self, kind, payload, charges):
        require(not self.sealed and not self.fatal, 'service bootstrap terminal/fatal; no further claims')
        body = {'schema': BOOTSTRAP, 'seq': len(self.events), 'previous': self.events[-1]['sha256'] if self.events else None,
            'request_sha256': self.request_sha256, 'kind': kind, 'payload': payload}
        preimage = canonical(body); hashed = hashlib.sha256(preimage).hexdigest()
        raw = canonical(dict(body, sha256=hashed)) + b'\n'
        require(len(raw) <= 1024 ** 2, 'bounded service metadata row')
        charges = dict(charges)
        for key, value in (('metadata_bytes', len(raw)), ('write_bytes', len(raw)), ('hash_attempts', 1), ('hash_bytes', len(preimage))):
            charges[key] = charges.get(key, 0) + value
        next_counters = dict(self.counters)
        for key, value in charges.items(): next_counters[key] += value
        require(all(next_counters[key] <= self.spec['caps'][key] for key in COUNTERS), 'service reservation cap exhausted before append')
        self.counters = next_counters  # remain charged if the fsynced append fails
        try:
            with self.path.open('ab') as stream:
                require(stream.write(raw) == len(raw), 'short service bootstrap append')
                stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            self.fatal = True; raise
        row = dict(body, sha256=hashed); self.events.append(row)
        return row
    def operation(self, kind, phase, operation_id, **fields):
        require(kind in self.spec['operation_profiles'] and phase in ('started', 'returned', 'failed'), 'priced service kind/phase required')
        if phase == 'started':
            require(operation_id not in self.operations, 'unique bootstrap physical attempt')
            profile = self.spec['operation_profiles'][kind]
            if 'wall_s_upper' in fields:
                declared_wall = fields['wall_s_upper']
                require(type(declared_wall) in (int, float) and 0 < declared_wall <= profile['wall_s_upper'],
                        'bootstrap profile below declared wall work')
            aliases = {'io_read_bytes_upper': 'read_bytes_upper', 'io_write_bytes_upper': 'write_bytes_upper'}
            for field in PROFILE[:-1]:
                value = fields.get(field, 0)
                for alias, target in aliases.items():
                    if target == field and alias in fields:
                        require(field not in fields or fields[field] == fields[alias], 'conflicting bootstrap IO aliases')
                        value = fields[alias]
                require(type(value) is int and value >= 0 and value <= profile[field], 'bootstrap profile below declared work')
            row = self._event('operation', {'operation': kind, 'phase': phase, 'operation_id': operation_id, 'fields': fields},
                              dict(self._price(profile), operation_attempts=1))
            self.operations[operation_id] = {'kind': kind, 'phase': phase, 'profile': profile, 'started_seq': row['seq']}
        else:
            previous = self.operations.get(operation_id)
            require(previous and previous['kind'] == kind and previous['phase'] == 'started', 'bootstrap matching return required')
            actuals = {'bytes_read': 'read_bytes_upper', 'bytes_written': 'write_bytes_upper',
                'identity_hash_attempts_actual': 'identity_hash_attempts_upper',
                'identity_hash_bytes_actual': 'identity_hash_bytes_upper', 'wall_s': 'wall_s_upper'}
            for field, upper in actuals.items():
                if field not in fields: continue
                value = fields[field]
                valid = (type(value) in (int, float) and 0 <= value < float('inf')) if field == 'wall_s' else (type(value) is int and value >= 0)
                if not valid or value > previous['profile'][upper]:
                    self.fatal = True
                    raise worker.WorkerEvidenceError('bootstrap actual work exceeds reserved profile: ' + field)
            row = self._event('operation', {'operation': kind, 'phase': phase, 'operation_id': operation_id, 'fields': fields}, {})
            previous.update(phase=phase, returned_seq=row['seq'])
        return row
    __call__ = operation
    def snapshot(self):
        return {'schema': BOOTSTRAP, 'request_sha256': self.request_sha256, 'head': self.events[-1]['sha256'],
            'cumulative': dict(self.counters), 'pending': [key for key, value in self.operations.items() if value['phase'] == 'started'],
            'path': str(self.path), 'fatal': self.fatal, 'native_qualified': False}


class StableDispatcher:
    """One callback identity across bootstrap and separately admitted ledgers."""
    def __init__(self, bootstrap):
        self.bootstrap, self.ledger, self.owners = bootstrap, None, {}
    def attach_qualification_accounting(self, ledger):
        require(self.ledger is None and getattr(ledger, 'scope_role', None) == 'qualification_only', 'one separate qualification accounting owner required')
        self.ledger = ledger
    def attach_journal(self, ledger):
        require(self.ledger is None and getattr(ledger, 'synthetic_cpu', True) is False, 'one separately admitted native CE ledger owner required')
        self.ledger = ledger
    def __call__(self, kind, phase, operation_id, **fields):
        require(type(operation_id) is str and operation_id and phase in ('started', 'returned', 'failed'),
                'explicit dispatch operation identity/phase')
        if phase == 'started':
            require(operation_id not in self.owners, 'dispatch operation identity reused')
            target = self.ledger if self.ledger is not None else self.bootstrap
            self.owners[operation_id] = (kind, target)
        else:
            require(operation_id in self.owners and self.owners[operation_id][0] == kind,
                    'dispatch matching original operation owner required')
            target = self.owners[operation_id][1]
        return target.operation(kind, phase, operation_id=operation_id, **fields)
    operation = __call__
    def snapshot(self): return (self.ledger if self.ledger is not None else self.bootstrap).snapshot()


class NativeResourceGate:
    """Actual RAM/disk/CUDA checks; numeric limits/transients all scope-supplied."""
    def __init__(self, spec, bootstrap):
        self.spec, self.bootstrap, self.framework = copy.deepcopy(spec), bootstrap, None
        require(type(spec.get('limits')) is dict and type(spec.get('resource_probe')) is dict
                and type(spec.get('host_transient_bytes')) is dict
                and all(type(value) is int and value >= 0 for value in spec['host_transient_bytes'].values()), 'explicit resource limits/probe/transients')
        limits = spec['limits']; gib = 1024 ** 3
        require(set(limits) == {'launch_host_gib', 'ongoing_host_gib', 'max_worker_gib', 'gpu_free_gib',
            'allocator_fraction', 'reserved_fraction_limit', 'disk_floor_bytes', 'launch_disk_bytes', 'active_storage_reserve_bytes'},
            'complete registered service resource barriers')
        require(all(type(limits[key]) in (int, float) and 0 < limits[key] < float('inf') for key in
            ('launch_host_gib', 'ongoing_host_gib', 'max_worker_gib', 'gpu_free_gib', 'allocator_fraction', 'reserved_fraction_limit'))
            and limits['launch_host_gib'] >= 5.5 and limits['ongoing_host_gib'] >= 1 and limits['max_worker_gib'] <= 4.5
            and limits['gpu_free_gib'] >= .5 and limits['allocator_fraction'] == .78 and limits['reserved_fraction_limit'] == .95
            and type(limits['disk_floor_bytes']) is int and limits['disk_floor_bytes'] >= 14 * gib
            and type(limits['launch_disk_bytes']) is int and limits['launch_disk_bytes'] >= 32 * gib
            and type(limits['active_storage_reserve_bytes']) is int and limits['active_storage_reserve_bytes'] >= 0,
            'registered service barriers cannot be lowered')
        from .ce_resource_evidence import validate_spec
        validate_spec(spec['resource_probe'])
    def _sample(self, *, launch=False, stage=None):
        from .ce_resource_evidence import readonly_sample, enforce
        sample = readonly_sample(self.spec['resource_probe'], self.bootstrap)
        limits = copy.deepcopy(self.spec['limits'])
        if stage is not None:
            require(stage in self.spec['host_transient_bytes'], 'unpriced service host transient: ' + stage)
            limits['ongoing_host_gib'] += self.spec['host_transient_bytes'][stage] / 1024 ** 3
        # is_initialized does not initialize CUDA. Loader.configure remains the
        # sole owner; OS/NVIDIA evidence supplies the earlier resource barrier.
        if self.framework is not None and self.framework.cuda.is_initialized():
            identifier, started = uuid.uuid4().hex, time.perf_counter()
            self.bootstrap.operation('resource_cuda_probe', 'started', operation_id=identifier)
            try:
                free, total = self.framework.cuda.mem_get_info(0)
                sample.update(cuda_free_bytes=int(free), cuda_total_bytes=int(total),
                              cuda_peak_reserved_bytes=int(self.framework.cuda.max_memory_reserved(0)))
            except BaseException as error:
                try: self.bootstrap.operation('resource_cuda_probe', 'failed', operation_id=identifier, error=str(error)[:1000])
                except BaseException as record_error:
                    error.resource_accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
                    raise error from record_error
                raise
            self.bootstrap.operation('resource_cuda_probe', 'returned', operation_id=identifier, wall_s=time.perf_counter() - started)
        return enforce(sample, limits, launch=launch)
    def launch(self):
        return self._sample(launch=True)
    def __call__(self, stage):
        return self._sample(stage=stage)


class NativeControllerServices:
    def __init__(self, request, service_permit, spec):
        _admit(service_permit, request, spec)
        controller.validate_request(request)
        require(type(spec) is dict and spec.get('schema') == SCHEMA and type(spec.get('bootstrap')) is dict
                and type(spec.get('resource')) is dict and type(spec.get('role_descriptors')) is dict, 'complete frozen service specification')
        self.request, self.spec, self.permit = copy.deepcopy(request), copy.deepcopy(spec), service_permit
        prepay = resolve_role(spec['role_descriptors'].get('bootstrap_constructor_prepayment'), spec.get('approved_code_sha256', {}))
        prepayment = prepay(request, spec)
        self.bootstrap = BootstrapLedger(spec['bootstrap'], digest(controller.request_preimage(request)), prepayment=prepayment)
        self.operation = StableDispatcher(self.bootstrap)
        self.gate = NativeResourceGate(spec['resource'], self.bootstrap)
        self.context, self.worker_handle, self.lease_token = None, None, None
    def _admit(self): _admit(self.permit, self.request, self.spec)
    def _role(self, name, context=None):
        self._admit(); descriptor = self.spec['role_descriptors'].get(name)
        code = (context or self.context or {}).get('bindings', {}).get('source', {}).get('code_sha256')
        if code is None: code = self.spec.get('approved_code_sha256', {})
        return resolve_role(descriptor, code)
    def _span(self, kind, work, **fields):
        identifier, started = uuid.uuid4().hex, time.perf_counter()
        self.bootstrap.operation(kind, 'started', operation_id=identifier, **fields)
        try: result = work()
        except BaseException as error:
            try: self.bootstrap.operation(kind, 'failed', operation_id=identifier, error=str(error)[:1000], wall_s=time.perf_counter() - started)
            except BaseException as record_error: error.service_accounting_failure = str(record_error)[:1000]
            raise
        self.bootstrap.operation(kind, 'returned', operation_id=identifier, wall_s=time.perf_counter() - started)
        return result
    def preflight_bootstrap(self, bounds, request_hash):
        self._admit()
        require(bounds == self.request['bootstrap_bounds'] and request_hash == digest(controller.request_preimage(self.request)), 'bootstrap request bounds drift')
        return self.bootstrap.snapshot()
    def admit_request(self, request):
        self._admit(); require(request == self.request, 'service request changed')
        if request['phase'] != 'qualification':
            raise MissingServiceRole('separate raw production-scope/qualification-result verifier remains unimplemented')
        bootstrap_outer = self._role('bootstrap_outer')(request, self.operation, self.spec)
        from .runner import code_bundle, verify_module_origins
        code = self._span('source_verification', lambda: code_bundle(self.operation))
        require(code == self.spec['approved_code_sha256'], 'complete running service/source bundle drift')
        verify_module_origins(code)
        context = self._span('scope_admission', lambda: qualification.admit_qualification(
            request, code_sha256=code, outer_admission=bootstrap_outer, io_observer=self.operation))
        require(type(context) is dict and context.get('admission_evidence_verified') is True
                and context.get('request_sha256') == digest(controller.request_preimage(request)), 'raw new-operator/scope/source admission required')
        # admit_qualification verifies §18 choice and every raw scope/recipe/
        # source/runtime/anchor/caps reference. This cannot be replaced by a
        # callback which merely returns admitted=True.
        context['native_services'] = {'operation': self.operation, 'gate': self.gate}
        context = self.finalize_worker_context(context)
        self.context = context
        return context
    def finalize_worker_context(self, context):
        self._admit()
        require('native_execution_permit' not in context.get('native_services', {}), 'worker JSON context must be complete before native permit issuance')
        spec, permit = self._role('worker_observation', context)(context, self.operation)
        handle = worker.open_worker(spec, permit, self.operation)
        try:
            proof = handle.proof()
            require(proof['fresh_process'] is True and proof['source_verified'] is True, 'registered actual worker provenance prerequisite')
            completed = dict(context, worker_proof_sha256=digest(proof))
        except BaseException as primary:
            try: handle.close()
            except BaseException as cleanup:
                primary.worker_close_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
                raise primary from cleanup
            raise
        self.worker_handle, self.worker_proof = handle, proof
        return completed
    def observer(self, context):
        self._admit(); require(context is self.context, 'service context identity')
        def observe(name, phase, **fields):
            # Controller stage summaries reserve only own metadata/verification;
            # native leaf charges stay in their linked bootstrap/Q/CE owner.
            return self.bootstrap.operation('controller_stage', phase, stage=name, **fields)
        return observe
    def make_outer_admission(self, request, context):
        self._admit()
        issuer = self._role('outer_issuer', context)(request, context, self.operation)
        verifier = self._role('outer_verifier', context)
        reserve = self._role('outer_reservation', context)
        return controller.OuterAdmission(request, context, issuer, reserve, verifier)
    def verify_sources(self, request, context, outer):
        self._admit()
        from . import ce_ingestion
        return ce_ingestion.verify_sources(request['preparation_ref'], request['plan_ref'], request['arm'], outer,
                                           operation=self.operation, gate=self.gate)
    def verify_worker(self, context):
        self._admit()
        require(self.worker_handle is not None and worker._hex(context.get('worker_proof_sha256')), 'finalize worker context before outer/native authority')
        proof = self.worker_handle.proof()
        require(proof['fresh_process'] is True and proof['source_verified'] is True, 'registered actual current worker provenance required')
        require(context['worker_proof_sha256'] == digest(proof), 'worker provenance changed after complete context freeze')
        return proof
    def configure_preparation(self, context):
        self._admit()
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime
        def configure():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = runtime.CUBLAS_WORKSPACE_CONFIG
            runtime.assert_runtime()
            import torch
            require(not torch.cuda.is_initialized(), 'native preparation must not initialize CUDA before loader configuration')
            return {'framework_versions_checked': True, 'cuda_initialized': False, 'native_qualified': False}
        return self._span('framework_preparation', configure)
    def prepare_rows(self, verified, context, outer):
        self._admit()
        from .ce_ingestion import PreparedRows
        return PreparedRows(verified, outer_admission=outer, operation=self.operation, gate=self.gate)
    @contextmanager
    def lease(self, context):
        self._admit()
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime
        require(self.lease_token is None, 'no nested wait on already owned GPU lease')
        with runtime.lease(self.spec['lease_label'], timeout_s=self.spec['lease_timeout_seconds']) as token:
            self.lease_token = token
            try: yield token
            finally: self.lease_token = None
    @contextmanager
    def watchdog(self, context):
        self._admit()
        seconds = self.spec['physical_deadline_seconds']
        require(type(seconds) in (int, float) and 0 < seconds < float('inf'), 'registered physical service deadline')
        timer = threading.Timer(seconds, lambda: os._exit(124))
        timer.daemon = True; timer.start()
        try: yield
        finally: timer.cancel()
    def guard(self, context, stage): self._admit(); return self.gate(stage)
    def open_journal(self, context, prepared, proof, resume, reason):
        self._admit()
        return self._role('native_journal', context)(context, prepared, proof, resume, reason, self.operation, self.gate)
    def checkpoints(self, context, backend, journal): self._admit(); return backend.checkpoints(context, journal)
    def cleanup(self, context, backend):
        # Backend's genuine prepaid deallocation handles later authority denial.
        try: self._admit()
        except BaseException as primary:
            from .ce_native_backend import NativeCEBackend, _PREPAID_CLEANUP_SEAL
            from .ce_native_qualification import NativeQualificationExecutor
            target = backend.backend if type(backend) is NativeQualificationExecutor else backend
            if type(target) is not NativeCEBackend or getattr(target, '_cleanup_seal', None) is not _PREPAID_CLEANUP_SEAL:
                raise
            try: backend.cleanup()
            except BaseException as cleanup_error:
                primary.service_cleanup_failure = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)[:1000]}
                raise primary from cleanup_error
            raise
        return backend.cleanup()
    def _publish(self, name, context, payload, journal):
        self._admit()
        role = self._role(name, context)
        return self._span(name, lambda: role(context, payload, journal, self.bootstrap.snapshot(), self.operation))
    def interruption(self, context, error, journal):
        return self._publish('interruption', context, {'type': type(error).__name__, 'message': str(error)[:1000]}, journal)
    def publish_completion(self, context, artifacts, journal): return self._publish('publication', context, artifacts, journal)
    def seal(self, context, publication, journal):
        self._admit()
        # Completion verifier must replay exact outer + native ledgers, include
        # failed/pending costs, publication and its own pre-reserved verification.
        return self._role('completion_seal', context)(context, publication, journal, self.bootstrap, self.operation)
    def build_child_launch_spec(self, context, branch, request_ref, parent_identity):
        self._admit()
        require(branch in ('uninterrupted', 'prefix', 'resumed'), 'exact sequential qualification branch')
        require(self.lease_token is not None, 'child needs existing supervisor-owned lease')
        # This source-owned transfer role establishes an inherited exclusive
        # child-use capability. It must not ask the child to reacquire our lock.
        return self._role('child_launch_and_lease_transfer', context)(context, branch, request_ref, parent_identity,
                                                                    self.lease_token, self.operation)
