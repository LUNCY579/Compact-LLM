"""Concrete pre-model checkpoint child metadata pipeline; native entry closed.

Static launch configuration and metadata-service authority precede all argv,
path, environment and OS work. Current identity is observed under the actual
durable service bootstrap, then existing WorkerObservationPermit admission
hashes pinned Python/request/source bytes. No CUDA configuration or model load
is performed here. The selected source issuer/factories remain absent.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys
import time
import uuid

from . import ce_checkpoint_local_issuer as local, ce_checkpoint_worker as child
from . import ce_native_services as services, ce_process_launch as launch
from . import ce_worker_evidence as worker, ce_native_authority as authority
from . import ce_controller as controller, ce_bootstrap_accounting as boot

SPEC = 'CE8-CHECKPOINT-ENTRY-CONFIGURATION-1'
FIXTURE = 'CE8-CHECKPOINT-ENTRY-FIXTURE-1'
# An operator/source freeze must install actual source-owned values. No API
# accepts these authorities from request JSON or a caller-supplied callback.
_SOURCE_CONFIGURATION = None
_SERVICE_ISSUER = None
_SERVICE_VERIFIER = None
_METADATA_VERIFIER = None
require, canonical, digest = worker.require, worker.canonical, worker.digest


def validate_configuration(configuration):
    require(type(configuration) is dict and configuration.get('schema') == SPEC,
            'frozen source-owned checkpoint entry configuration required')
    required = {'schema', 'launch_spec', 'controller_request', 'service_spec', 'common_context_ref',
                'observation_bounds', 'entry_factories'}
    require(set(configuration) == required, 'exact static checkpoint entry configuration fields')
    spec = launch.validate_spec(configuration['launch_spec'])
    request = controller.validate_request(configuration['controller_request'])
    require(digest(controller.request_preimage(request)) == spec['request_sha256']
            and configuration['service_spec'].get('approved_code_sha256') == spec['approved_code_sha256'],
            'static controller/launch/source request binding')
    reference = configuration['common_context_ref']
    require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
            and type(reference['path']) is str and Path(reference['path']).is_absolute()
            and worker._hex(reference['sha256']) and type(reference['bytes']) is int
            and 0 < reference['bytes'] <= local.raw.MAX_JSON, 'pinned bounded common context reference')
    bounds = configuration['observation_bounds']
    require(type(bounds) is dict and set(bounds) == {'operation_attempts_upper', 'io_read_bytes_upper',
            'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_seconds_upper'}
            and all(type(value) is int and value > 0 for value in bounds.values()),
            'finite prepaid initial OS/source observation bounds')
    factories = configuration['entry_factories']
    require(type(factories) is dict and set(factories) == {'prepared_rows', 'prefix_terminal_handle'},
            'existing reviewed source entry factory descriptors')
    for descriptor in factories.values():
        require(type(descriptor) is dict and set(descriptor) == {'module', 'symbol', 'source_sha256'}
                and type(descriptor['module']) is str and type(descriptor['symbol']) is str
                and descriptor.get('source_sha256') == spec['approved_code_sha256'].get(descriptor['module'].replace('.', '/') + '.py'),
                'entry factory must be an exact approved source symbol')
    return configuration


class MetadataEntryPayment:
    """One permanently paid metadata lifetime, never model execution authority."""
    def __init__(self, runtime, configuration):
        require(type(runtime) is services.NativeControllerServices, 'actual native metadata services owner required')
        services._admit(runtime.permit, configuration['controller_request'], configuration['service_spec'])
        self._initialize(runtime.bootstrap, configuration, native=True, runtime=runtime)

    def _initialize(self, bootstrap, configuration, *, native, runtime=None):
        require(type(bootstrap) is services.BootstrapLedger and not bootstrap.fatal and not bootstrap.sealed
                and bootstrap.spec == configuration['service_spec']['bootstrap']
                and bootstrap.request_sha256 == configuration['launch_spec']['request_sha256'],
                'actual durable entry bootstrap/request/spec owner')
        self.bootstrap, self.configuration, self.native, self.runtime = bootstrap, copy.deepcopy(configuration), native, runtime
        self.identifier, self.status, self.started = uuid.uuid4().hex, 'RESERVED', time.monotonic()
        self.proof = self.context = self.owner = None
        self.profile = bootstrap.spec['operation_profiles']['entry_lifetime']
        services._profile(self.profile)
        require(self.profile['wall_s_upper'] >= configuration['launch_spec']['deadline_seconds'],
                'whole child entry/RPC lifetime wall budget before work')
        require(self.profile['identity_hash_attempts_upper'] >= 64
                and self.profile['identity_hash_bytes_upper'] >= 64 * local.raw.MAX_JSON,
                'full bounded entry context/proof/service comparison envelope before work')
        row = bootstrap.operation('entry_lifetime', 'started', self.identifier,
            request_ref=configuration['launch_spec']['request_ref'], common_context_ref=configuration['common_context_ref'],
            scope_sha256=configuration['launch_spec']['scope_sha256'], service_spec_sha256=digest(configuration['service_spec']),
            metadata_only=True, **self.profile)
        self.reservation_sha256 = row['sha256']

    @classmethod
    def fixture(cls, bootstrap, configuration):
        value = object.__new__(cls); value._initialize(bootstrap, configuration, native=False); return value

    def _admit(self):
        require(type(self) is MetadataEntryPayment and type(self.bootstrap) is services.BootstrapLedger
                and not self.bootstrap.fatal and not self.bootstrap.sealed
                and self.bootstrap.spec == self.configuration['service_spec']['bootstrap']
                and self.bootstrap.request_sha256 == self.configuration['launch_spec']['request_sha256'],
                'live exact metadata payment/bootstrap owner')
        if self.native:
            require(type(self.runtime) is services.NativeControllerServices and self.runtime.bootstrap is self.bootstrap,
                    'exact original native metadata services owner')
            services._admit(self.runtime.permit, self.configuration['controller_request'], self.configuration['service_spec'])
        operation = self.bootstrap.operations.get(self.identifier)
        require(operation is not None and operation['kind'] == 'entry_lifetime' and operation['phase'] == 'started',
                'original retained entry reservation required')

    def _bind(self, context, proof):
        self._admit(); require(self.status == 'RESERVED' and self.proof is None, 'entry worker binding once before RPC')
        spec = self.configuration['launch_spec']
        require(context['request_sha256'] == spec['request_sha256'] and context['scope_sha256'] == spec['scope_sha256']
                and proof['request_sha256'] == spec['request_sha256'] and proof['scope_sha256'] == spec['scope_sha256']
                and proof['sources']['request_ref'] == spec['request_ref'], 'metadata payment current worker/raw request/scope binding')
        require(len(canonical(context)) <= local.raw.MAX_JSON and len(canonical(proof)) <= local.raw.MAX_JSON,
                'bounded complete entry context/proof serialization')
        owner = {'accounting_spec_sha256': digest(local.validate_spec(context['bindings']['scope']['checkpoint_qualification']['local_issuer'])),
                 'prepared_receipt_sha256': context['prepared_receipt']['receipt_sha256']}
        def retain(): return {'worker_proof_sha256': digest(proof), 'worker_identity_sha256': proof['worker_identity_sha256'],
                             'owner': owner, 'entry_reservation_sha256': self.reservation_sha256}
        identifier = uuid.uuid4().hex
        profile = self.bootstrap.spec['operation_profiles']['entry_worker_binding']
        self.bootstrap.operation('entry_worker_binding', 'started', identifier, **profile)
        try: binding = retain()
        except BaseException as error:
            self.bootstrap.operation('entry_worker_binding', 'failed', identifier, error=str(error)[:1000]); raise
        self.bootstrap.operation('entry_worker_binding', 'returned', identifier, **binding)
        self.context, self.proof, self.owner = copy.deepcopy(context), copy.deepcopy(proof), owner

    def bind_worker(self, context, handle, permit, operation):
        require(self.native is True and type(handle) is worker.WinWorkerHandle and type(permit) is worker.WorkerObservationPermit,
                'actual native metadata handle/observation permit required')
        permit.require(handle.spec, operation)
        proof = handle.proof()
        require(handle.native is True and proof['os_identity']['pid'] == os.getpid(), 'entry prepayment belongs to actual current child')
        self._bind(context, proof)
        return proof

    def bind_fixture(self, context, proof):
        require(self.native is False and proof.get('execution_verified') is False and proof.get('native_qualified') is False,
                'explicit non-native metadata fixture proof')
        self._bind(context, proof)

    def verify(self, context, proof, owner):
        # Native route refuses fixtures before traversing supplied evidence.
        require(self.native is True, 'fixture entry payment cannot admit native source/RPC work')
        self._admit()
        require(self.status in ('RESERVED', 'STARTED') and self.context == authority.context_preimage(context)
                and self.proof == proof and self.owner == owner, 'actual entry payment context/worker/owner drift')
        return self

    @property
    def evidence(self):
        return {'schema': services.BOOTSTRAP, 'native': self.native, 'ticket_id': self.identifier, 'status': self.status,
            'reservation_sha256': self.reservation_sha256, 'bootstrap_path': str(self.bootstrap.path),
            'request_sha256': self.configuration['launch_spec']['request_sha256'],
            'scope_sha256': self.configuration['launch_spec']['scope_sha256'], 'owner': copy.deepcopy(self.owner),
            'worker_proof_sha256': digest(self.proof) if self.proof is not None else None,
            'native_qualified': False, 'execution_verified': False}

    def begin(self):
        self._admit(); require(self.status == 'RESERVED' and self.proof is not None, 'entry begins once after actual worker binding')
        # The durable physical start already precedes OS/source work. No costs
        # are migrated or refunded when the RPC consumer begins its ABI phase.
        self.status = 'STARTED'

    def complete(self):
        self._admit(); require(self.status == 'STARTED', 'entry completion requires original started lifetime')
        self.bootstrap.operation('entry_lifetime', 'returned', self.identifier,
            wall_s=time.monotonic() - self.started, worker_proof_sha256=digest(self.proof), metadata_only=True)
        self.status = 'RETURNED'

    def fail(self, error):
        self._admit(); require(self.status in ('RESERVED', 'STARTED'), 'one metadata entry failure, never refund')
        self.bootstrap.operation('entry_lifetime', 'failed', self.identifier,
            wall_s=time.monotonic() - self.started, error={'type': type(error).__name__, 'message': str(error)[:1000]}, metadata_only=True)
        self.status = 'FAILED'


def validate_common(configuration, request, common):
    """Framework-free linkage; returned JSON alone carries no permission."""
    spec = configuration['launch_spec']
    require(type(common) is dict and common.get('request_sha256') == spec['request_sha256']
            and common.get('scope_sha256') == spec['scope_sha256']
            and common.get('bindings', {}).get('source', {}).get('code_sha256') == spec['approved_code_sha256']
            and request.get('common_context_ref') == configuration['common_context_ref'],
            'paid common scope/request/complete source binding drift')
    temporary = dict(common, checkpoint_child_declaration=request.get('lineage_declaration'))
    child.validate_request(request, temporary)
    rpc = local.validate_spec(common['bindings']['scope']['checkpoint_qualification']['local_issuer'])
    require(rpc['max_message_bytes'] <= spec['max_message_bytes'] and rpc['max_messages'] <= spec['max_messages']
            and rpc['deadline_seconds'] <= spec['deadline_seconds'], 'child RPC cannot exceed source-frozen launch caps/deadline')
    receipt = common.get('prepared_receipt')
    require(type(receipt) is dict and receipt.get('receipt_sha256') == common['bindings']['scope']['qualifier_accounting']['ingestion_receipt_sha256']
            and digest({key: value for key, value in receipt.items() if key != 'receipt_sha256'}) == receipt['receipt_sha256'],
            'full retained preparation receipt is bound before child preparation')
    root = Path(common['bindings']['scope']['checkpoint_qualification']['owned_directory'])
    branch = root / 'children' / request['branch']
    require(Path(spec['request_ref']['path']).parent == branch
            and request['lease_delegation_path'] == str(branch / 'LEASE_DELEGATION.json')
            and Path(configuration['service_spec']['bootstrap']['owned_directory']) == branch / 'service-bootstrap',
            'exact independently owned child request/delegation/bootstrap directories')
    return common


def read_static_inputs(configuration, bootstrap_operation):
    """Actual bounded request/common file reads, reserved before path metadata."""
    spec = configuration['launch_spec']; reference = configuration['common_context_ref']
    root = Path(spec['request_ref']['path']).parents[3]
    # The source-owned scope root is recovered from the retained common file;
    # source configuration must itself prebind a common owner for initial IO.
    require(Path(reference['path']).is_relative_to(root), 'common reference outside static launch request owner')
    profiles = configuration['service_spec']['bootstrap']['operation_profiles']
    profile = profiles['input_io']
    request = local.read_json(spec['request_ref'], root, bootstrap_operation, profile, local.raw.MAX_JSON)
    common = local.read_json(reference, root, bootstrap_operation, profile, local.raw.MAX_JSON)
    return request, validate_common(configuration, request, common)


def verify_environment_and_origins(configuration, request, common, bootstrap_operation):
    """Exact environment, pinned module origins and raw source bytes; no ML."""
    spec = configuration['launch_spec']; profile = configuration['service_spec']['bootstrap']['operation_profiles']['input_io']
    def verify_environment():
        expected = spec['environment']
        actual = dict(os.environ)
        rpc = actual.pop('CE8_RPC_BINDING', None)
        require(rpc is not None and actual == expected, 'whole current child environment differs from frozen launcher')
        binding = local.raw._json(rpc)
        require(type(binding) is dict and set(binding) == {'session_id', 'request_sha256', 'scope_sha256'}
                and type(binding['session_id']) is str and binding['session_id']
                and binding['request_sha256'] == spec['request_sha256'] and binding['scope_sha256'] == spec['scope_sha256'],
                'RPC environment cannot supply a foreign request/scope or authority')
        require(Path(sys.executable) == Path(spec['python_ref']['path']), 'current launcher Python origin drift')
        root = Path(spec['source_checkout']).resolve()
        for name, reference in spec['source_files'].items():
            path = Path(reference['path'])
            require(path.resolve() == (root / name).resolve() and path.resolve().is_relative_to(root),
                    'source map logical module origin drift')
            module_name = name[:-3].replace('/', '.').replace('\\', '.') if name.endswith('.py') else None
            module = sys.modules.get(module_name)
            if module is not None:
                require(type(getattr(module, '__file__', None)) is str and Path(module.__file__).resolve() == path.resolve(),
                        'running module origin differs from approved source checkout')
        require(child.__name__.replace('.', '/') + '.py' in spec['source_files']
                and __name__.replace('.', '/') + '.py' in spec['source_files'],
                'complete child/provider running source map required')
        return binding
    binding = local._paid(bootstrap_operation, profile, 'input_io', verify_environment)
    base = {'bounds': configuration['observation_bounds']}; budget = worker._Budget(base, bootstrap_operation)
    references = [spec['python_ref'], spec['process_image_ref'], spec['request_ref'], *spec['source_files'].values()]
    require(base['bounds']['io_read_bytes_upper'] >= sum(ref['bytes'] + 1 for ref in references)
            and base['bounds']['identity_hash_bytes_upper'] >= sum(ref['bytes'] for ref in references)
            and base['bounds']['identity_hash_attempts_upper'] >= len(references), 'full initial pinned source/Python raw hash envelope')
    for reference in references: worker._file(reference, budget)
    return binding


def current_observation_spec(configuration, bootstrap_operation):
    """Real current OS facts under metadata-service prepayment, no CUDA/model."""
    spec = configuration['launch_spec']; budget = worker._Budget({'bounds': configuration['observation_bounds']}, bootstrap_operation)
    api = budget.run('worker_os_binding', worker._WindowsOS)
    pid = budget.run('worker_os_binding', os.getpid)
    retained = None
    primary = None
    try:
        def open_owned():
            nonlocal retained
            retained = api.open(pid); return retained
        budget.run('worker_os_binding', open_owned)
        facts = budget.run('worker_os_binding', lambda: api.snapshot(retained))
        boot_id = budget.run('worker_os_binding', api.boot)
        parent_pid = budget.run('worker_os_binding', lambda: launch._parent_pid(api, retained))
        expected_argv = [spec['process_image_ref']['path'], *spec['argv'][1:]]
        require(facts['pid'] == pid and facts['signaled'] is False and facts['exit_code'] == 259
                and Path(facts['image_path']) == Path(spec['process_image_ref']['path']) and facts['argv'] == expected_argv
                and boot_id == spec['parent_identity']['boot_identity'], 'current OS image/argv/boot/live identity drift')
        if spec['worker_topology'] == 'direct':
            require(parent_pid == spec['parent_identity']['pid'], 'direct child OS parent drift')
        else:
            intermediate, redirector_primary = None, None
            try:
                def open_redirector():
                    nonlocal intermediate
                    intermediate = api.open(parent_pid); return intermediate
                budget.run('worker_os_binding', open_redirector)
                redirector = budget.run('worker_os_binding', lambda: api.snapshot(intermediate))
                ancestor = budget.run('worker_os_binding', lambda: launch._parent_pid(api, intermediate))
                require(ancestor == spec['parent_identity']['pid'] and redirector['signaled'] is False
                        and Path(redirector['image_path']) == Path(spec['python_ref']['path'])
                        and redirector['argv'] == spec['argv']
                        and spec['parent_identity']['creation_filetime_100ns'] <= redirector['creation_filetime_100ns'] <= facts['creation_filetime_100ns'],
                        'exact venv redirector/current-child/original-parent OS lineage')
            except BaseException as error: redirector_primary = error; raise
            finally:
                if intermediate is not None:
                    try: _close_observed(api, intermediate, budget)
                    except BaseException as cleanup:
                        if redirector_primary is not None:
                            redirector_primary.entry_redirector_close_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
                            raise redirector_primary from cleanup
                        raise
        require(facts['creation_filetime_100ns'] >= spec['parent_identity']['creation_filetime_100ns'], 'current child predates original parent')
    except BaseException as error: primary = error; raise
    finally:
        try:
            if retained is not None: _close_observed(api, retained, budget)
        except BaseException as error:
            if primary is not None:
                primary.entry_identity_close_failure = {'type': type(error).__name__, 'message': str(error)[:1000]}
                raise primary from error
            raise
    result = {'schema': worker.SCOPE, 'mode': 'registered_metadata', 'request_sha256': spec['request_sha256'],
        'scope_sha256': spec['scope_sha256'], 'expected_pid': pid,
        'expected_creation_filetime_100ns': facts['creation_filetime_100ns'],
        'launch_not_before_filetime_100ns': facts['creation_filetime_100ns'],
        'launch_not_after_filetime_100ns': facts['creation_filetime_100ns'], 'expected_boot_identity': boot_id,
        'source_checkout': spec['source_checkout'], 'argv': facts['argv'], 'source_files': spec['source_files'],
        'approved_code_sha256': spec['approved_code_sha256'], 'python_ref': spec['python_ref'],
        'process_image_ref': spec['process_image_ref'], 'request_ref': spec['request_ref'],
        'entry_module': spec['worker_module'], 'bounds': configuration['observation_bounds']}
    worker.validate_spec(result)
    return result


def _close_observed(api, retained, budget):
    """Retain denial/failure while attempting release of the owned OS handle."""
    entered = False
    def release():
        nonlocal entered
        entered = True; api.close(retained)
    try: budget.run('worker_handle_close', release)
    except BaseException as error:
        if not entered:
            try: api.close(retained)
            except BaseException as failure:
                error.physical_close_failure = {'type': type(failure).__name__, 'message': str(failure)[:1000]}
        raise


def parent_observation_spec(configuration, retained_proof):
    """Static retained parent facts still require a fresh native OS handle."""
    identity, sources = retained_proof['os_identity'], retained_proof['sources']
    spec = {'schema': worker.SCOPE, 'mode': 'registered_metadata',
        'request_sha256': retained_proof['request_sha256'], 'scope_sha256': retained_proof['scope_sha256'],
        'expected_pid': identity['pid'], 'expected_creation_filetime_100ns': identity['creation_filetime_100ns'],
        'launch_not_before_filetime_100ns': identity['creation_filetime_100ns'],
        'launch_not_after_filetime_100ns': identity['creation_filetime_100ns'], 'expected_boot_identity': identity['boot_identity'],
        'source_checkout': configuration['launch_spec']['source_checkout'], 'argv': identity['argv'],
        'source_files': sources['source_files'], 'approved_code_sha256': configuration['launch_spec']['approved_code_sha256'],
        'python_ref': sources['python_ref'], 'process_image_ref': sources['process_image_ref'], 'request_ref': sources['request_ref'],
        'entry_module': sources['entry'][:-3].replace('/', '.').replace('\\', '.'), 'bounds': configuration['observation_bounds']}
    worker.validate_spec(spec)
    return spec


class CheckpointEntryProvider:
    def open_prepaid_entry(self, request_path):
        # Source registration and metadata authority are checked before even
        # inspecting a supplied path or a frozen configuration object.
        require(local._SOURCE_ENTRY_PROVIDER is self and _SOURCE_CONFIGURATION is not None,
                'actual source-owned checkpoint entry provider is not selected')
        require(any(_SERVICE_VERIFIER is value for value in services._REGISTERED_SERVICE_VERIFIERS)
                and any(_METADATA_VERIFIER is value for value in worker._REGISTERED_METADATA_VERIFIERS),
                'actual checkpoint metadata service/OS issuer remains unregistered')
        configuration = copy.deepcopy(validate_configuration(_SOURCE_CONFIGURATION))
        service_permit = services.ControllerServicePermit(configuration['controller_request'], configuration['service_spec'],
                                                         _SERVICE_ISSUER, _SERVICE_VERIFIER)
        # Existing constructor role owns permanent prepayment before mkdir,
        # initial journal/hash, clocks or any argv/OS/environment inspection.
        runtime = services.NativeControllerServices(configuration['controller_request'], service_permit, configuration['service_spec'])
        payment = MetadataEntryPayment(runtime, configuration)
        return self._open_after_payment(configuration, runtime, payment, request_path)

    def _open_after_payment(self, configuration, runtime, payment, request_path):
        bootstrap_operation = runtime.bootstrap.operation
        profile = configuration['service_spec']['bootstrap']['operation_profiles']['input_io']
        handles = []
        try:
            local._paid(bootstrap_operation, profile, 'input_io', lambda:
                require(type(request_path) is str and Path(request_path) == Path(configuration['launch_spec']['request_ref']['path']),
                        'CLI path cannot select another raw child request'))
            request, common = read_static_inputs(configuration, bootstrap_operation)
            verify_environment_and_origins(configuration, request, common, bootstrap_operation)
            observation = current_observation_spec(configuration, runtime.operation)
            permit = worker.WorkerObservationPermit(observation, self, _METADATA_VERIFIER, runtime.operation)
            handle = worker.open_worker(observation, permit, runtime.operation); handles.append(handle)
            proof = handle.proof()
            payment.bind_worker(common, handle, permit, runtime.operation)
            parent_spec = parent_observation_spec(configuration, request['lineage_declaration']['parent_proof'])
            parent_permit = worker.WorkerObservationPermit(parent_spec, self, _METADATA_VERIFIER, runtime.operation)
            parent = worker.open_worker(parent_spec, parent_permit, runtime.operation); handles.append(parent)
            local.validate_parent_view(parent.proof(), request['lineage_declaration']['parent_proof'])
            runtime.context = common
            prepared_factory = services.resolve_role(configuration['entry_factories']['prepared_rows'], configuration['launch_spec']['approved_code_sha256'])
            prepared = prepared_factory(common, service_permit, runtime.operation, runtime.gate, proof)
            from .ce_ingestion import PreparedRows
            require(type(prepared) is PreparedRows and prepared.receipt == common['prepared_receipt'],
                    'reviewed real prepared rows/full census must exactly match retained receipt')
            controller.require_preparation_cuda_uninitialized()
            owner = {'accounting_spec_sha256': digest(local.validate_spec(common['bindings']['scope']['checkpoint_qualification']['local_issuer'])),
                     'prepared_receipt_sha256': prepared.receipt['receipt_sha256']}
            payment.verify(common, proof, owner)
            prefix = None
            if request['branch'] == 'resumed':
                factory = services.resolve_role(configuration['entry_factories']['prefix_terminal_handle'], configuration['launch_spec']['approved_code_sha256'])
                prefix = factory(common, request, service_permit, runtime.operation)
                require(type(prefix) is worker.WinWorkerHandle, 'actual retained terminal prefix OS handle factory is missing')
                handles.append(prefix)
            return {'context': common, 'request_ref': configuration['launch_spec']['request_ref'],
                'common_context_ref': configuration['common_context_ref'],
                'owned_root': common['bindings']['scope']['native_backend']['output_directory'],
                'worker_handle': handle, 'observation_permit': permit, 'parent_handle': parent, 'prefix_handle': prefix,
                'expected_worker': {key: proof[key] for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')},
                'entry_prepayment': payment, 'operation': runtime.operation, 'bootstrap_operation': bootstrap_operation,
                'gate': runtime.gate, 'observer': runtime.observer(common), 'prepared': prepared,
                'source_service_bootstrap': runtime.bootstrap.snapshot()}
        except BaseException as primary:
            for handle in reversed(handles):
                try: handle.close()
                except BaseException as failure: local.raw._accounting(primary, failure)
            try: payment.fail(primary)
            except BaseException as failure: local.raw._accounting(primary, failure)
            raise


def fixture_read_inputs(context, configuration, bootstrap):
    require(context == {'schema': FIXTURE, 'native_qualified': False, 'execution_verified': False}
            and type(bootstrap) is services.BootstrapLedger, 'explicit real-file non-native entry fixture')
    request, common = read_static_inputs(configuration, bootstrap.operation)
    return {'request': request, 'common': common, 'bootstrap_snapshot': bootstrap.snapshot(),
            'native_qualified': False, 'execution_verified': False, 'models_loaded': 0}
