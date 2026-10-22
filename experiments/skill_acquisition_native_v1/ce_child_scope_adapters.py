"""Concrete child issuer verifiers using existing retained source objects.

No registration occurs here. LocalCheckpointIssuer/CheckpointEntryProvider
retain their exact source entry/configuration before existing permit issuance.
Revalidation consumes cached sealed handles, not recursive priced OS refreshes.
"""
from __future__ import annotations

import copy

from . import ce_checkpoint_local_issuer as local, ce_checkpoint_entry_provider as provider
from . import ce_checkpoint_worker as child, ce_worker_evidence as worker
from . import ce_native_services as services, ce_native_authority as authority
from . import ce_bootstrap_accounting as boot, ce_qualification_accounting as accounting
from . import ce_checkpoint_initializer as initializer, ce_checkpoint_publication as publication
from . import ce_qualification_artifacts as raw
from . import ce_checkpoint_processes as processes, ce_process_launch as launch

require, digest = authority.require, raw.digest


def _provider(value):
    require(type(value) is provider.CheckpointEntryProvider and local._SOURCE_ENTRY_PROVIDER is value
            and provider._SOURCE_CONFIGURATION is not None, 'exact selected source child entry provider required')
    require(value._active_configuration == provider._SOURCE_CONFIGURATION,
            'retained source child provider configuration differs from selected freeze')
    return value


def child_provider_service_verifier(issuer, request_sha256, service_spec_sha256):
    current = _provider(issuer); configuration = current._active_configuration
    require(provider._SERVICE_ISSUER is issuer and provider._SERVICE_VERIFIER is child_provider_service_verifier
            and any(child_provider_service_verifier is item for item in services._REGISTERED_SERVICE_VERIFIERS),
            'existing source child service verifier remains unregistered')
    require(digest(services.controller.request_preimage(configuration['controller_request'])) == request_sha256
            == configuration['launch_spec']['request_sha256'] and digest(configuration['service_spec']) == service_spec_sha256,
            'child metadata service exact original controller request/spec')
    return True


def child_worker_metadata_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    if type(issuer) is processes.CheckpointProcesses:
        current = _parent_launch(issuer)
        bound = getattr(current, '_observed_child_spec', None)
        require(type(bound) is dict and digest(bound) == spec_sha256 and bound['request_sha256'] == request_sha256
                and bound['scope_sha256'] == scope_sha256, 'exact owned launch-observed native child metadata scope')
        return True
    current = _provider(issuer)
    require(provider._METADATA_VERIFIER is child_worker_metadata_verifier
            and any(child_worker_metadata_verifier is item for item in worker._REGISTERED_METADATA_VERIFIERS),
            'existing source child worker observation verifier remains unregistered')
    runtime, payment = current._active_runtime, current._active_payment
    require(type(runtime) is services.NativeControllerServices and type(payment) is provider.MetadataEntryPayment
            and payment.native is True and payment.runtime is runtime, 'original durable child metadata lifetime owner')
    runtime._admit(); payment._admit()
    observed = current._metadata_specs.get(spec_sha256)
    require(type(observed) is dict and digest(observed) == spec_sha256
            and observed['request_sha256'] == request_sha256 and observed['scope_sha256'] == scope_sha256
            and request_sha256 == current._active_configuration['launch_spec']['request_sha256']
            and scope_sha256 == current._active_configuration['launch_spec']['scope_sha256']
            and observed['approved_code_sha256'] == current._active_configuration['launch_spec']['approved_code_sha256'],
            'exact source-observed current/parent OS scope before metadata permit')
    return True


def _parent_launch(value):
    require(type(value) is processes.CheckpointProcesses and not value.fixture,
            'actual existing native checkpoint process launch owner required')
    value._admit(value.context, value.permit, native=True)
    require(value.launch_issuer is processes.CheckpointProcesses
            and value.launch_verifier is child_launch_verifier
            and any(child_launch_verifier is item for item in launch._REGISTERED_LAUNCH_VERIFIERS),
            'source parent launch verifier remains unregistered')
    bound = getattr(value, '_active_launch_spec', None)
    require(type(bound) is dict and bound['request_sha256'] == value.context['request_sha256']
            and bound['scope_sha256'] == value.context['scope_sha256']
            and bound['approved_code_sha256'] == value.context['bindings']['source']['code_sha256'],
            'exact derived parent launch spec retained before process creation')
    return value


def child_launch_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    owner = _parent_launch(issuer); spec = owner._active_launch_spec
    require(digest(spec) == spec_sha256 and spec['request_sha256'] == request_sha256
            and spec['scope_sha256'] == scope_sha256, 'exact prepaid parent-owned native launch specification')
    return True


def observed_child_spec(spec, identity, parent_identity):
    """Pure OS/source linkage; actual caller is a retained OwnedProcess body."""
    launch.validate_spec(spec)
    require(type(identity) is dict and type(identity.get('pid')) is int and identity['pid'] > 0
            and type(identity.get('creation_filetime_100ns')) is int and identity['creation_filetime_100ns'] > 0
            and identity['pid'] != parent_identity['pid'] and identity['creation_filetime_100ns'] >= parent_identity['creation_filetime_100ns']
            and identity.get('boot_identity') == parent_identity['boot_identity']
            and parent_identity == spec['parent_identity'], 'actual owned child creation/boot/parent identity')
    expected = [spec['process_image_ref']['path'], *spec['argv'][1:]]
    require(identity.get('image_path') == spec['process_image_ref']['path'] and identity.get('argv') == expected,
            'actual owned child dual-image/whole argv identity')
    bounds = spec.get('observation_bounds')
    require(type(bounds) is dict, 'explicit separately priced postlaunch OS/source observation envelope')
    result = {'schema': worker.SCOPE, 'mode': 'registered_metadata', 'request_sha256': spec['request_sha256'],
        'scope_sha256': spec['scope_sha256'], 'expected_pid': identity['pid'],
        'expected_creation_filetime_100ns': identity['creation_filetime_100ns'],
        'launch_not_before_filetime_100ns': identity['creation_filetime_100ns'],
        'launch_not_after_filetime_100ns': identity['creation_filetime_100ns'], 'expected_boot_identity': identity['boot_identity'],
        'source_checkout': spec['source_checkout'], 'argv': identity['argv'], 'source_files': spec['source_files'],
        'approved_code_sha256': spec['approved_code_sha256'], 'python_ref': spec['python_ref'],
        'process_image_ref': spec['process_image_ref'], 'request_ref': spec['request_ref'],
        'entry_module': spec['worker_module'], 'bounds': bounds}
    worker.validate_spec(result)
    return result


def child_launch_observation(spec, permit, identity, parent_identity, observer):
    require(type(permit) is launch.NativeLaunchPermit, 'existing opaque native launch permit before observation')
    permit.require(spec, observer); owner = _parent_launch(permit._issuer)
    require(observer is owner.operation, 'same original prepaid launch observer')
    # The actual OwnedProcess has already observed this child OS object and its
    # redirector/direct lineage. Parent handle proof is independently retained.
    require(type(owner.parent_handle) is worker.WinWorkerHandle, 'actual original parent retained OS handle')
    actual = worker.verify_worker_handle(owner.parent_handle, {key: owner.proof[key] for key in
        ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True, terminal=False)
    require({key: actual['os_identity'][key] for key in ('pid', 'creation_filetime_100ns', 'boot_identity')} == parent_identity,
            'actual parent OS handle differs from the admitted native launch')
    budget = worker._Budget({'bounds': spec['observation_bounds']}, observer)
    bound = budget.run('worker_source_verification', lambda: observed_child_spec(spec, identity, parent_identity))
    owner._observed_child_spec = copy.deepcopy(bound)
    metadata = worker.WorkerObservationPermit(bound, owner, child_worker_metadata_verifier, observer)
    return bound, metadata


def cached_worker(handle, proof, operation):
    """Actual source handle origins only; this helper performs no OS/observer IO."""
    require(type(handle) is worker.WinWorkerHandle and handle._seal is worker._HANDLE_SEAL
            and handle.native is True and not handle.closed and handle.observer is operation,
            'cached actual sealed native child worker handle/observer required')
    worker._admit(handle.spec, handle._permit, operation)
    require(type(proof) is dict and proof.get('schema') == worker.PROOF
            and proof.get('worker_identity_sha256') == handle.identity_sha256
            and proof.get('request_sha256') == handle.spec['request_sha256']
            and proof.get('scope_sha256') == handle.spec['scope_sha256']
            and digest(worker.worker_identity_preimage(handle.identity, handle.sources,
                handle.spec['request_sha256'], handle.spec['scope_sha256'])) == handle.identity_sha256
            and worker.worker_identity_preimage(proof['os_identity'], proof['sources'],
                proof['request_sha256'], proof['scope_sha256']) == worker.worker_identity_preimage(handle.identity,
                handle.sources, handle.spec['request_sha256'], handle.spec['scope_sha256']),
            'cached exact child process/source/request identity changed')
    return proof


def _local(value, context=None):
    require(type(value) is local.LocalCheckpointIssuer and any(value is item for item in child._REGISTERED_LOCAL_ISSUERS),
            'exact source singleton child execution issuer remains unregistered')
    entry, active = value._active_entry, value._active_context
    require(type(entry) is dict and type(active) is dict and (context is None or active is context),
            'existing local issuer has no exact active source entry/context')
    payment = entry.get('entry_prepayment')
    require(type(payment) is provider.MetadataEntryPayment and payment.native is True and payment.status == 'STARTED',
            'actual started source-owned child metadata entry lifetime')
    payment._admit()
    runtime = payment.runtime
    require(type(runtime) is services.NativeControllerServices and runtime.operation is entry['operation']
            and runtime.gate is entry['gate'] and entry['metadata_service_permit'] is runtime.permit,
            'active child operation/gate/original metadata permit bindings')
    cached_worker(entry['worker_handle'], value._active_proof, entry['operation'])
    require(active.get('worker_proof_sha256') == digest(value._active_proof)
            and active['checkpoint_child_observed_lineage']['child_proof'] == value._active_proof
            and active['checkpoint_child_observed_lineage']['child_request_ref'] == entry['request_ref']
            and active['checkpoint_child_declaration'] == value._active_request['lineage_declaration'],
            'retained source child request/proof/declaration binding')
    return value


def _context_issuer(context):
    require(type(context) is dict and type(context.get('native_services')) is dict,
            'complete active source child context required')
    return _local(context['native_services'].get('local_checkpoint_issuer'), context)


def child_native_authority_verifier(issuer, context_sha256, request_sha256, scope_sha256, phase, arm):
    current = _local(issuer); context = current._active_context
    require(local._AUTHORITY_VERIFIER is child_native_authority_verifier
            and any(child_native_authority_verifier is item for item in authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS),
            'existing child execution verifier remains unregistered')
    require(digest(authority.context_preimage(context)) == context_sha256
            and context['request_sha256'] == request_sha256 and context['scope_sha256'] == scope_sha256
            and context['phase'] == phase == 'qualification' and context['arm'] == arm,
            'exact retained child execution context identity')
    return True


def child_bootstrap_native_issuer(context, permit, proof):
    authority.require_execution_permit(permit, context, 'journal_construct')
    current = _context_issuer(context)
    require(current._active_proof == proof, 'child bootstrap exact active worker proof')
    return current


def child_bootstrap_prepayment_verifier(handle, context, proof, owner):
    current = _context_issuer(context)
    require(type(handle) is boot.PrepaymentHandle and handle._seal is boot._SEAL and handle._native is True
            and type(handle._ledger) is boot.BootstrapAccounting and handle._ledger.context is context
            and handle._ledger.proof == proof == current._active_proof and handle.evidence['owner'] == owner
            and handle.evidence['status'] == 'RESERVED' and not handle._ledger.closed
            and not handle._ledger.unknown_wall and not handle._ledger.fatal, 'actual child prepaid constructor/worker/owner ticket')
    return True


def child_predecessor_verifier(context, proof, payload):
    current = _context_issuer(context)
    require(local._PREDECESSOR_VERIFIER is child_predecessor_verifier
            and any(child_predecessor_verifier is item for item in accounting._REGISTERED_CHILD_PREFIX_VERIFIERS),
            'existing source child predecessor verifier remains unregistered')
    require(proof == current._active_proof and payload == context['checkpoint_child_observed_lineage'], 'exact active child predecessor evidence')
    accounting._validate_child_prefix(context, proof, payload)
    entry = current._active_entry; inputs = context['checkpoint_child_inputs']; parent_proof = payload['parent_proof']
    # Explicit lifecycle boundary refresh is paid outside Q and before the
    # child model/kernel. Rechecking an execution permit does not call here.
    actual = worker.verify_worker_handle(entry['parent_handle'], {key: parent_proof[key] for key in
        ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True, terminal=False)
    local.validate_parent_view(actual, parent_proof)
    policy = context['bindings']['scope']['checkpoint_qualification']['child_initializer']
    reader = raw.RawEvidenceReader(entry['bootstrap_operation'], policy['reader_bounds'],
        context['bindings']['scope']['native_backend']['output_directory'])
    parent = initializer.read_parent_evidence(reader, inputs['parent_context'], payload['parent_ledger_refs'],
        inputs['parent_binding'], payload)
    prefix = None
    if payload['branch'] == 'resumed':
        require(type(entry.get('prefix_handle')) is worker.WinWorkerHandle, 'actual transported retained prefix terminal handle')
        binding = payload['prefix_manifest']['bindings']['qualification_evidence_binding']
        worker.verify_worker_handle(entry['prefix_handle'], {key: binding[key] for key in
            ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}, require_native=True, terminal=True)
        prefix = initializer.read_checkpoint_store(reader, inputs['prefix_store'], binding,
            expected_manifest=payload['prefix_manifest'], maximum_bytes=policy['max_store_bytes'])
    current._predecessor_readback = {'payload_sha256': digest(payload), 'context_sha256': digest(authority.context_preimage(context)),
        'parent_ledger_head': parent['ledger_head'], 'prefix_manifest_sha256': None if prefix is None else prefix['manifest_sha256'],
        'native_qualified': False, 'execution_verified': False}
    return True


def child_publication_issuer(context, permit, proof):
    """Existing LocalCheckpointIssuer is the publisher identity, not a JSON flag."""
    authority.require_execution_permit(permit, context, 'completion_publication')
    current = _context_issuer(context)
    require(proof == current._active_proof and current._predecessor_readback is not None,
            'actual child predecessor readback precedes retained publication')
    return current
