"""Fixed discarded-qualification worker entry; source configuration is CLOSED.

Only a reviewed immutable source configuration can install these exact existing
implementations. RPC/argv/JSON cannot install authorities or choose callbacks.
The parent remains UNQUALIFIED after body completion; its external supervisor
must prove real OS termination and independently replay raw evidence/costs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import importlib
import os
from pathlib import Path
import sys
from types import MappingProxyType

from . import ce_native_services as services, ce_controller as controller
from . import ce_native_authority as authority, ce_checkpoint_local_issuer as local
from . import ce_qualification_artifacts as raw

SCHEMA = 'CE8-QUALIFICATION-SOURCE-CONFIGURATION-1'
ENTRY_MODULE = __package__ + '.ce_qualification_entry'
_SOURCE_CONFIGURATION = None
_INSTALLED_SHA256 = None
_INTAKE_EVIDENCE = None
require, digest = authority.require, authority.digest


def validate_configuration(configuration):
    """Pure static packet binding; validation alone installs/grants nothing."""
    require(type(configuration) is dict and set(configuration) == {'schema', 'request', 'service_spec',
            'request_ref', 'service_spec_ref', 'rpc_spec', 'configuration_sha256'} and configuration['schema'] == SCHEMA,
            'exact reviewed qualification source configuration')
    request, spec = configuration['request'], configuration['service_spec']
    controller.validate_request(request)
    require(request['phase'] == 'qualification' and request['arm'] == 'canonical'
            and request['training_authorized'] is False and spec.get('schema') == services.SCHEMA,
            'source entry is canonical discarded qualification only')
    require(configuration['configuration_sha256'] == digest({key: value for key, value in configuration.items()
            if key != 'configuration_sha256'}), 'immutable source configuration digest drift')
    for key in ('request_ref', 'service_spec_ref'):
        reference = configuration[key]
        require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
                and type(reference['path']) is str and Path(reference['path']).is_absolute()
                and authority._hex(reference['sha256']) and type(reference['bytes']) is int
                and 0 < reference['bytes'] <= raw.MAX_JSON, 'pinned bounded source ' + key)
    local.validate_spec(configuration['rpc_spec'])
    require(spec['scope_adapters']['worker']['request_ref'] == configuration['request_ref']
            and spec['scope_adapters']['worker']['entry_module'] == ENTRY_MODULE
            and spec['scope_adapters']['worker']['approved_code_sha256'] == spec['approved_code_sha256'],
            'fixed worker entry/raw request/complete source binding')
    raw.canonical(configuration)  # rejects callables, nonfinite data and capability objects
    return configuration


def _configuration():
    global _SOURCE_CONFIGURATION, _INTAKE_EVIDENCE
    if _SOURCE_CONFIGURATION is None:
        from . import ce_qualification_configuration as intake
        require(type(intake._SOURCE_ADOPTION) is MappingProxyType,
                'reviewed immutable qualification source configuration remains closed')
        configured, evidence = intake.load_source_configuration()
        _SOURCE_CONFIGURATION, _INTAKE_EVIDENCE = MappingProxyType(configured), evidence
    require(type(_SOURCE_CONFIGURATION) is MappingProxyType,
            'reviewed immutable qualification source configuration remains closed')
    return validate_configuration(copy.deepcopy(dict(_SOURCE_CONFIGURATION)))


def service_metadata_verifier(issuer, request_sha256, service_spec_sha256):
    require(issuer is source_issuer_identity(), 'fixed source entry metadata issuer identity')
    configuration = _configuration()
    require(request_sha256 == digest(controller.request_preimage(configuration['request']))
            and service_spec_sha256 == digest(configuration['service_spec']), 'source entry exact request/service metadata scope')
    return True


def source_issuer_identity():
    """The executing fixed module, including Python's real -m __main__ view."""
    current = sys.modules[__name__]
    require(current.__name__ in (ENTRY_MODULE, '__main__') and current.__package__ == __package__
            and current.__file__ == __file__, 'fixed canonical entry source module origin')
    return current


def registry_manifest():
    """Inspectable fixed symbol inventory; no import, registry change or grant."""
    return {
        'service': ['service_metadata_verifier'],
        'scope': ['bootstrap_outer', 'outer_issuer', 'outer_verifier', 'outer_reservation', 'worker_observation',
                  'worker_metadata_verifier', 'native_authority_verifier', 'bootstrap_native_issuer', 'bootstrap_prepayment_verifier'],
        'parent': ['create_parent_qualification', 'reserve_parent_accounting', 'parent_checkpoint_adapter', 'parent_resource_adapter'],
        'publication': ['publication_role', 'interruption_role', 'completion_seal_role'],
        'child': ['child_provider_service_verifier', 'child_worker_metadata_verifier', 'child_native_authority_verifier',
                  'child_bootstrap_native_issuer', 'child_bootstrap_prepayment_verifier', 'child_predecessor_verifier',
                  'child_launch_verifier', 'child_launch_observation'],
        'primitive': ['bootstrap_constructor_prepayment', 'ServiceConstructorPrepayment', 'prepare_checkpoint_entry_rows',
                      'prefix_terminal_handle', '_verify_prefix_adoption'],
    }


def _modules():
    names = ('ce_scope_adapters', 'ce_parent_qualification', 'ce_parent_publication', 'ce_child_scope_adapters',
        'ce_service_prepayment', 'ce_entry_factories', 'ce_prefix_handle_transfer', 'ce_checkpoint_entry_provider',
        'ce_checkpoint_worker', 'ce_bootstrap_accounting', 'ce_qualification_accounting', 'ce_publication',
        'ce_process_launch', 'ce_worker_evidence', 'ce_checkpoint_qualification', 'ce_native_qualification',
        'ce_checkpoint_lease', 'ce_checkpoint_processes', 'ce_checkpoint_initializer', 'ce_checkpoint_publication')
    return {name: importlib.import_module(__package__ + '.' + name) for name in names}


def install_source_registries():
    """No parameters: only the fixed source configuration can select this route."""
    global _INSTALLED_SHA256
    configuration = _configuration()  # before import/object/registry traversal
    require(_INSTALLED_SHA256 in (None, configuration['configuration_sha256']), 'source installation cannot switch packet')
    if _INSTALLED_SHA256 is not None: return configuration
    _install_registry_map(configuration)
    _INSTALLED_SHA256 = configuration['configuration_sha256']
    return configuration


def _install_registry_map(configuration):
    modules = _modules()
    scope, parent, publisher, child_scope = (modules[name] for name in
        ('ce_scope_adapters', 'ce_parent_qualification', 'ce_parent_publication', 'ce_child_scope_adapters'))
    # Check every concrete dependency before the first mutation. An unfinished
    # child launch interface cannot leave a partially installed parent runtime.
    for name in registry_manifest()['child']:
        require(callable(getattr(child_scope, name, None)), 'actual child source installer ABI missing: ' + name)
    prepay, rows, transfer = (modules[name] for name in ('ce_service_prepayment', 'ce_entry_factories', 'ce_prefix_handle_transfer'))
    role_functions = (scope.bootstrap_outer, scope.outer_issuer, scope.outer_verifier, scope.outer_reservation,
        scope.worker_observation, prepay.bootstrap_constructor_prepayment, rows.prepare_checkpoint_entry_rows,
        transfer.prefix_terminal_handle, publisher.publication_role, publisher.interruption_role, publisher.completion_seal_role)
    expected = {fn.__name__: {'module': fn.__module__, 'symbol': fn.__name__,
        'source_sha256': configuration['service_spec']['approved_code_sha256'].get(fn.__module__.replace('.', '/') + '.py')}
        for fn in role_functions}
    aliases = {'publication': 'publication_role', 'interruption': 'interruption_role', 'completion_seal': 'completion_seal_role'}
    descriptors = configuration['service_spec']['role_descriptors']
    for role in ('bootstrap_outer', 'outer_issuer', 'outer_verifier', 'outer_reservation', 'worker_observation',
                 'bootstrap_constructor_prepayment', 'publication', 'interruption', 'completion_seal'):
        descriptor = expected[aliases.get(role, role)]
        require(authority._hex(descriptor['source_sha256']) and descriptors.get(role) == descriptor,
                'source entry exact fixed role descriptor: ' + role)
    boot, accounting, publication, launch, worker, cp, executor = (modules[name] for name in
        ('ce_bootstrap_accounting', 'ce_qualification_accounting', 'ce_publication', 'ce_process_launch',
         'ce_worker_evidence', 'ce_checkpoint_qualification', 'ce_native_qualification'))
    provider, child_worker = modules['ce_checkpoint_entry_provider'], modules['ce_checkpoint_worker']
    issuer, entry_provider = local.LocalCheckpointIssuer(), provider.CheckpointEntryProvider()
    source_identity = source_issuer_identity()
    # Exact existing verifier and class registries, scoped to this source packet.
    services._REGISTERED_SERVICE_VERIFIERS = frozenset((service_metadata_verifier, child_scope.child_provider_service_verifier))
    services._REGISTERED_ROLE_FACTORIES = frozenset(role_functions)
    services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES = frozenset((prepay.ServiceConstructorPrepayment,))
    controller._REGISTERED_OUTER_AUTHORITY_VERIFIERS = frozenset((scope.outer_verifier, rows._verify_entry_outer_authority))
    controller._REGISTERED_NATIVE_FACTORIES = frozenset((parent.create_parent_qualification,))
    authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS = frozenset((scope.native_authority_verifier, child_scope.child_native_authority_verifier))
    worker._REGISTERED_METADATA_VERIFIERS = frozenset((scope.worker_metadata_verifier, child_scope.child_worker_metadata_verifier, transfer._verify_prefix_adoption))
    boot._REGISTERED_NATIVE_ISSUERS = frozenset((scope.bootstrap_native_issuer, child_scope.child_bootstrap_native_issuer))
    boot._REGISTERED_PREPAYMENT_VERIFIERS = frozenset((scope.bootstrap_prepayment_verifier, child_scope.child_bootstrap_prepayment_verifier))
    launch._REGISTERED_LAUNCH_VERIFIERS = frozenset((child_scope.child_launch_verifier,))
    launch._REGISTERED_OBSERVATION_FACTORIES = frozenset((child_scope.child_launch_observation,))
    accounting._REGISTERED_CHILD_PREFIX_VERIFIERS = frozenset((child_scope.child_predecessor_verifier,))
    publication._REGISTERED_PUBLICATION_ISSUERS = frozenset((source_identity, issuer))
    executor._REGISTERED_ACCOUNTING_FACTORIES = frozenset((parent.reserve_parent_accounting,))
    executor._REGISTERED_CHECKPOINT_ADAPTERS = frozenset((parent.parent_checkpoint_adapter,))
    executor._REGISTERED_RESOURCE_ADAPTERS = frozenset((parent.parent_resource_adapter,))
    cp._REGISTERED_GPU_HANDOFF_SERVICES = frozenset((modules['ce_checkpoint_lease'].SupervisorGpuLease,))
    cp._REGISTERED_CHECKPOINT_PROCESS_SERVICES = frozenset((modules['ce_checkpoint_processes'].CheckpointProcesses,))
    child_worker._REGISTERED_LOCAL_ISSUERS = frozenset((issuer,))
    child_worker._REGISTERED_CHILD_INITIALIZERS = frozenset((modules['ce_checkpoint_initializer'].CheckpointChildInitializer,))
    child_worker._REGISTERED_CHILD_LEASE_SERVICES = frozenset((modules['ce_checkpoint_lease'].DelegatedGpuLease,))
    child_worker._REGISTERED_CHILD_PUBLICATION_SERVICES = frozenset((modules['ce_checkpoint_publication'].CheckpointBranchPublisher,))
    parent._BOOTSTRAP_ISSUER, parent._BOOTSTRAP_VERIFIER = scope.bootstrap_native_issuer, scope.bootstrap_prepayment_verifier
    parent._LAUNCH_ISSUER, parent._LAUNCH_VERIFIER = modules['ce_checkpoint_processes'].CheckpointProcesses, child_scope.child_launch_verifier
    publisher._PUBLICATION_ISSUER = source_identity
    local._SOURCE_ENTRY_PROVIDER = entry_provider
    local._AUTHORITY_VERIFIER = child_scope.child_native_authority_verifier
    local._BOOTSTRAP_ISSUER, local._BOOTSTRAP_VERIFIER = child_scope.child_bootstrap_native_issuer, child_scope.child_bootstrap_prepayment_verifier
    local._PREDECESSOR_VERIFIER, local._PUBLICATION_ISSUER = child_scope.child_predecessor_verifier, issuer
    provider._SERVICE_ISSUER, provider._SERVICE_VERIFIER, provider._METADATA_VERIFIER = entry_provider, child_scope.child_provider_service_verifier, child_scope.child_worker_metadata_verifier
    return configuration


def install_child_source_registries(configuration):
    """Source intake identity is required; arbitrary JSON cannot call this ABI."""
    from . import ce_checkpoint_configuration as transport
    require(type(transport._SOURCE_ADOPTION) is MappingProxyType and configuration is transport._LOADED
            and type(transport._EVIDENCE) is dict and transport._EVIDENCE.get('source_files_verified') is True,
            'exact source-adopted paid child configuration required before registry installation')
    from . import ce_checkpoint_entry_provider as provider
    provider.validate_configuration(configuration)
    _install_registry_map({'service_spec': configuration['service_spec'],
        'configuration_sha256': transport._EVIDENCE['configuration_sha256']})


def serve_parent(pipe, proof, qualify):
    """Bounded actual RPC body; pure streams can test ordering without children."""
    pipe.emit('ready', 0, {'worker_proof': proof})
    require(pipe.receive(1) == {'command': 'qualify', 'data': {}}, 'exact qualify command only; early finish denied')
    with redirect_stdout(sys.stderr):
        result = qualify()
    seal = result.get('seal') if type(result) is dict else None
    require(type(seal) is dict and seal.get('status') == 'UNQUALIFIED_BODY_SEAL' and type(seal.get('body_ref')) is dict,
            'only actual retained parent body seal can respond')
    reference = seal['body_ref']
    require(all(key in reference for key in ('path', 'sha256', 'bytes')) and type(reference['path']) is str
            and Path(reference['path']).is_absolute() and authority._hex(reference['sha256'])
            and type(reference['bytes']) is int and 0 < reference['bytes'] <= raw.MAX_JSON,
            'bounded actual parent body artifact reference')
    response = {'raw_artifact': {key: reference[key] for key in ('path', 'sha256', 'bytes')}, 'body_completed': True,
        'os_terminal_verified': False, 'native_qualified': False}
    if result.get('final_seal_ref') is not None:
        response['final_seal_ref'] = {key: result['final_seal_ref'][key] for key in ('path', 'sha256', 'bytes')}
    pipe.emit('response', 1, response)
    require(pipe.receive(2) == {'command': 'finish', 'data': {}}, 'finish only after parent body completion')
    pipe.emit('response', 2, {'finished': True, 'native_qualified': False})
    return 0


def retain_final_body_seal(runtime, context, result):
    """Carry exact late scope/child refs across exit in one paid raw artifact."""
    runtime._admit()
    require(type(runtime) is services.NativeControllerServices and runtime.context is context
            and type(result) is dict and result.get('native_qualified') is False
            and result.get('seal', {}).get('status') == 'UNQUALIFIED_BODY_SEAL', 'actual unqualified parent body result')
    from . import update, ce_publication
    profile = services._profile(runtime.spec['bootstrap']['operation_profiles']['publication'])
    with update._span(runtime.operation, 'publication', dict(profile)) as returned:
        value = {'schema': 'CE8-PARENT-BODY-SEAL-1', 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'body_seal': result['seal'],
            'parent_service_specification': runtime.spec['bootstrap'],
            'metadata_intake_evidence': runtime.metadata_intake_evidence,
            'native_qualified': False, 'execution_verified': False, 'os_terminal_verified': False}
        blob = raw.canonical(value) + b'\n'; require(len(blob) <= raw.MAX_JSON, 'final parent seal metadata ceiling')
        directory = runtime.bootstrap.directory; final = directory / 'PARENT_BODY_SEAL.json'
        require(not os.path.lexists(final), 'create-only final parent seal')
        temporary = directory / ('PARENT_BODY_SEAL.' + __import__('uuid').uuid4().hex + '.partial')
        with temporary.open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short final parent seal'); stream.flush(); os.fsync(stream.fileno())
        with temporary.open('rb') as stream: observed = stream.read(len(blob) + 1)
        require(observed == blob, 'final parent seal temporary readback')
        ce_publication._atomic_no_replace(temporary, final, directory)
        with final.open('r+b') as stream: os.fsync(stream.fileno()); observed = stream.read(len(blob) + 1)
        require(observed == blob, 'final parent seal readback')
        returned.update(bytes_read=2 * len(blob), bytes_written=len(blob), identity_hash_attempts_actual=1,
                        identity_hash_bytes_actual=len(blob))
        result['final_seal_ref'] = {'path': str(final), 'sha256': __import__('hashlib').sha256(blob).hexdigest(), 'bytes': len(blob)}
    return result


def main(argv=None):
    configuration = install_source_registries()  # closed before argv/env/files
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    require(Path(args.request) == Path(configuration['request_ref']['path']), 'argv cannot select another qualification packet')
    request, spec = configuration['request'], configuration['service_spec']
    permit = services.ControllerServicePermit(request, spec, source_issuer_identity(), service_metadata_verifier)
    runtime = services.NativeControllerServices(request, permit, spec)
    runtime.metadata_intake_evidence = copy.deepcopy(_INTAKE_EVIDENCE)
    primary = None
    try:
        profile = services._profile(spec['bootstrap']['operation_profiles']['scope_metadata'])
        for reference, expected in ((configuration['request_ref'], request), (configuration['service_spec_ref'], spec)):
            observed = local.read_json(reference, Path(reference['path']).parent, runtime.bootstrap.operation, profile, raw.MAX_JSON)
            require(observed == expected, 'actual frozen source request/service file differs from reviewed configuration')
        # The service scope adapter independently reads exact raw request/source
        # refs; source config itself cannot replace that verified metadata chain.
        context = runtime.admit_request(request)
        proof = runtime.verify_worker(context)
        binding = raw._json(os.environ['CE8_RPC_BINDING'])
        require(binding['request_sha256'] == context['request_sha256'] and binding['scope_sha256'] == context['scope_sha256'],
                'parent RPC binding cannot alter source scope')
        pipe = local.BoundedRPC(configuration['rpc_spec'], binding, os.getpid(), sys.stdin.buffer,
            sys.stdout.buffer, runtime.bootstrap.operation)
        parent = importlib.import_module(__package__ + '.ce_parent_qualification')
        def qualify():
            return retain_final_body_seal(runtime, context,
                controller.run_controller(request, runtime, parent.create_parent_qualification))
        return serve_parent(pipe, proof, qualify)
    except BaseException as error: primary = error; raise
    finally:
        if runtime.worker_handle is not None:
            try: runtime.worker_handle.close()
            except BaseException as cleanup:
                if primary is not None:
                    primary.parent_entry_close_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
                    raise primary from cleanup
                raise
        if primary is None:
            runtime.bootstrap.end()
            runtime.bootstrap.inspect_evidence()


if __name__ == '__main__':
    from .ce_qualification_adoption import adopt_source_paths
    adopt_source_paths()
    raise SystemExit(main())
