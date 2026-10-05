"""Actual owned-process qualification supervision; no source activation.

The worker publishes an unqualified body artifact while alive. This supervisor
retains its real terminal handle after exit for subsequent independent replay.
RPC flags are checked as protocol fields and never establish native PASS.
"""
from __future__ import annotations

from . import ce_operation_contracts as contracts

import copy
import hashlib
import os
from pathlib import Path
import uuid
import sys

from . import ce_native_services as services, ce_process_launch as launch
from . import ce_worker_evidence as worker, ce_checkpoint_local_issuer as local
from . import ce_publication as publication, update

SPEC = 'CE8-QUALIFICATION-SUPERVISION-SPEC-1'
SCHEMA = 'CE8-QUALIFICATION-SUPERVISION-1'
ENTRY_MODULE = 'experiments.skill_acquisition_native_v1.ce_qualification_entry'
require = worker.require


def supervisor_specification(configuration):
    """Derive a separate metadata owner; the supervised worker owns models."""
    from . import ce_qualification_entry as entry
    entry.validate_configuration(configuration)
    spec = copy.deepcopy(configuration['service_spec'])
    root = Path(spec['resource']['resource_probe']['output_root'])
    owner = str(root / 'supervisor-service-bootstrap')
    spec['bootstrap']['owned_directory'] = owner
    spec['constructor_prepayment']['owned_directory'] = owner
    spec['scope_adapters']['owned_directory'] = str(Path(owner) / 'capsules')
    return spec


def supervisor_service_verifier(issuer, request_sha256, spec_sha256):
    require(issuer is sys.modules[__name__], 'fixed source supervisor metadata issuer')
    from . import ce_qualification_entry as entry
    config = entry._configuration()
    require(request_sha256 == worker.digest(services.controller.request_preimage(config['request']))
            and spec_sha256 == worker.digest(supervisor_specification(config)), 'source-adopted supervisor request/spec binding')
    return True


def supervisor_launch_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    require(type(issuer) is services.NativeControllerServices and issuer.context is None,
            'actual metadata-only supervisor service issuer')
    issuer._admit(); spec = getattr(issuer, '_qualification_launch_spec', None)
    require(type(spec) is dict and spec['worker_module'] == ENTRY_MODULE
            and worker.digest(spec) == spec_sha256 and spec['request_sha256'] == request_sha256
            and request_sha256 == worker.digest(services.controller.request_preimage(issuer.request))
            and spec['scope_sha256'] == scope_sha256 == issuer.request['scope_ref']['sha256']
            and spec['approved_code_sha256'] == issuer.spec['approved_code_sha256'],
            'fixed actual source-owned supervisor launch binding')
    return True


def supervisor_worker_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    require(type(issuer) is services.NativeControllerServices and issuer.context is None,
            'existing metadata supervisor observation owner')
    issuer._admit(); spec = getattr(issuer, '_qualification_observation_spec', None)
    require(type(spec) is dict and worker.digest(spec) == spec_sha256
            and spec['request_sha256'] == request_sha256 and spec['scope_sha256'] == scope_sha256,
            'actual launch-observed exact qualification worker metadata scope')
    return True


def supervisor_launch_observation(spec, permit, identity, parent_identity, observer):
    require(type(permit) is launch.NativeLaunchPermit, 'source launch permit before worker observation')
    permit.require(spec, observer); service = permit._issuer
    require(type(service) is services.NativeControllerServices and observer is service.operation,
            'same original source supervisor dispatcher')
    from .ce_child_scope_adapters import observed_child_spec
    budget = worker._Budget({'bounds': spec['observation_bounds']}, observer)
    bound = budget.run('worker_source_verification', lambda: observed_child_spec(spec, identity, parent_identity))
    service._qualification_observation_spec = copy.deepcopy(bound)
    return bound, worker.WorkerObservationPermit(bound, service, supervisor_worker_verifier, observer)


def build_supervisor_launch_spec(configuration, parent_identity):
    from . import ce_qualification_entry as entry
    entry.validate_configuration(configuration)
    policy = configuration['service_spec']['scope_adapters']['worker']
    module = __package__ + '.ce_qualification_supervisor'
    code = configuration['service_spec']['approved_code_sha256']
    spec = {'schema': launch.SPEC, 'mode': 'registered_native_worker',
        'request_sha256': worker.digest(services.controller.request_preimage(configuration['request'])),
        'scope_sha256': configuration['request']['scope_ref']['sha256'], 'cwd': policy['source_checkout'],
        'source_checkout': policy['source_checkout'], 'worker_module': ENTRY_MODULE,
        'argv': [policy['python_ref']['path'], '-m', ENTRY_MODULE, '--request', configuration['request_ref']['path']],
        'python_ref': policy['python_ref'], 'process_image_ref': policy['process_image_ref'],
        'request_ref': configuration['request_ref'], 'source_files': policy['source_files'], 'approved_code_sha256': code,
        'worker_topology': 'venv_redirector', 'parent_identity': copy.deepcopy(parent_identity),
        'environment': __import__(__package__ + '.ce_registration', fromlist=['child_environment']).child_environment(),
        'deadline_seconds': 19200, 'handshake_seconds': 180, 'rpc_seconds': 19200, 'terminate_grace_seconds': 30,
        'max_message_bytes': 1024 ** 2, 'max_messages': 8, 'max_stderr_bytes': 1024 ** 2,
        'bounds': {'operation_attempts_upper': 100000, 'io_read_bytes_upper': 4 * 1024 ** 3,
            'identity_hash_attempts_upper': 100000, 'identity_hash_bytes_upper': 4 * 1024 ** 3, 'wall_seconds_upper': 19200},
        'observation_bounds': copy.deepcopy(policy['bounds']),
        'observation_factory_descriptor': {'module': module, 'symbol': 'supervisor_launch_observation',
            'source_sha256': code[module.replace('.', '/') + '.py']}}
    launch.validate_spec(spec); return spec


def main():
    """Fixed source-adopted supervisor entry; no argv/env selects a scope."""
    from .ce_qualification_adoption import adopt_source_paths
    adopt_source_paths()
    from . import ce_qualification_entry as entry
    configuration = entry.install_source_registries()
    spec = supervisor_specification(configuration)
    # Register exact source bodies only after the same immutable adoption and
    # the concrete parent/child installer dependencies have been checked.
    services._REGISTERED_SERVICE_VERIFIERS |= frozenset((supervisor_service_verifier,))
    launch._REGISTERED_LAUNCH_VERIFIERS |= frozenset((supervisor_launch_verifier,))
    launch._REGISTERED_OBSERVATION_FACTORIES |= frozenset((supervisor_launch_observation,))
    worker._REGISTERED_METADATA_VERIFIERS |= frozenset((supervisor_worker_verifier,))
    service_permit = services.ControllerServicePermit(configuration['request'], spec, sys.modules[__name__], supervisor_service_verifier)
    service = services.NativeControllerServices(configuration['request'], service_permit, spec)
    budget = worker._Budget({'bounds': spec['scope_adapters']['worker']['bounds']}, service.operation)
    api = budget.run('worker_os_binding', worker._WindowsOS); current = None; result = None; primary = None; final_owner = None
    try:
        def open_current():
            nonlocal current
            current = api.open(os.getpid()); return current
        budget.run('worker_os_binding', open_current)
        facts = budget.run('worker_os_binding', lambda: api.snapshot(current))
        boot = budget.run('worker_os_binding', api.boot)
        parent = {'pid': facts['pid'], 'creation_filetime_100ns': facts['creation_filetime_100ns'], 'boot_identity': boot}
        budget.run('worker_handle_close', lambda: api.close(current)); current = None
        prepared = build_supervisor_launch_spec(configuration, parent)
        service._qualification_launch_spec = copy.deepcopy(prepared)
        permit = launch.NativeLaunchPermit(prepared, service, supervisor_launch_verifier, service.operation)
        result = supervise_qualification(service, prepared, permit)
        from .ce_finalizer_accounting import FinalizerCostOwner
        from . import ce_qualification_finalizer as finalizer
        final_owner = FinalizerCostOwner(service)
        verified = finalizer.verify_supervision(service, result, final_owner)
        result['terminal_handle'].close(); result['terminal_handle'] = None
        ended = final_owner.observe_supervisor_end(service)
        result['supervisor_final_evidence'] = ended.evidence
        candidate = finalizer.consume_verified(verified, ended)
        costs = final_owner.close_and_seal()
        result['qualification_finalization'] = finalizer.finish_cost_seal(candidate, final_owner, costs)
        result['finalizer_cost_evidence'] = costs
        print(worker.canonical({k: v for k, v in result.items() if k != 'terminal_handle'}).decode('ascii'))
        return 0
    except BaseException as error:
        primary = error
        if final_owner is not None:
            try: final_owner.fail(primary)
            except BaseException as failure: local.raw._accounting(primary, failure); raise primary from failure
        raise
    finally:
        if current is not None:
            try: api.close(current)
            except BaseException as failure:
                if primary is not None: local.raw._accounting(primary, failure); raise primary from failure
                raise
        if result is not None and result.get('terminal_handle') is not None:
            try: result['terminal_handle'].close()
            except BaseException as failure:
                if primary is not None: local.raw._accounting(primary, failure); raise primary from failure
                raise


def _reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'}
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and worker._hex(value['sha256']) and type(value['bytes']) is int
            and 0 < value['bytes'] <= 4 * 1024 ** 2, 'bounded exact parent body reference')
    return value


def _read_body(service, reference, actual_live):
    profile = services._profile(service.spec['bootstrap']['operation_profiles']['input_io'])
    require(profile['read_bytes_upper'] >= reference['bytes'] + 1
            and profile['identity_hash_attempts_upper'] >= 16
            and profile['identity_hash_bytes_upper'] >= 16 * 4 * 1024 ** 2,
            'body byte and source/worker comparison envelope reserved before read')
    with update._span(service.operation, 'input_io', dict(profile, wall_contract=contracts.OPAQUE_INPUT_FRAME)) as returned:
        root = Path(service.spec['resource']['resource_probe']['output_root']).resolve()
        path = Path(reference['path'])
        require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
                and path.stat().st_size == reference['bytes'], 'exact regular body artifact in admitted owner')
        with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
        require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == reference['sha256'],
                'parent body bytes/hash differ after actual worker exit')
        body = local.raw._json(blob)
        require(type(body) is dict and local.raw.canonical(body) + b'\n' == blob
                and body.get('schema') == 'CE8-PARENT-QUALIFICATION-BODY-1'
                and body.get('status') == 'UNQUALIFIED_BODY_COMPLETE' and body.get('body_completed') is True
                and body.get('os_terminal_verified') is False and body.get('native_qualified') is False
                and body.get('execution_verified') is False
                and body.get('request_sha256') == actual_live['request_sha256']
                and body.get('scope_sha256') == actual_live['scope_sha256'], 'same actual source/worker unqualified body artifact')
        local.validate_parent_view(actual_live, body['worker_proof'])
        identity_bytes = len(worker.canonical(worker.worker_identity_preimage(actual_live['os_identity'],
            actual_live['sources'], actual_live['request_sha256'], actual_live['scope_sha256'])))
        returned.update(bytes_read=len(blob), identity_hash_attempts_actual=2,
                        identity_hash_bytes_actual=len(blob) + identity_bytes,
                        io_observation_scope='tracked file read and raw/worker-preimage digests; OS query IO remains separate')
        return body


def _compare_ready(service, actual_live, ready_proof):
    profile = services._profile(service.spec['bootstrap']['operation_profiles']['input_io'])
    require(profile['identity_hash_attempts_upper'] >= 1
            and profile['identity_hash_bytes_upper'] >= 4 * 1024 ** 2, 'ready identity comparison priced before hashing')
    with update._span(service.operation, 'input_io', dict(profile, wall_contract=contracts.OPAQUE_INPUT_FRAME)) as returned:
        local.validate_parent_view(actual_live, ready_proof)
        preimage = worker.worker_identity_preimage(actual_live['os_identity'], actual_live['sources'],
            actual_live['request_sha256'], actual_live['scope_sha256'])
        size = len(worker.canonical(preimage))
        require(size <= 4 * 1024 ** 2, 'ready source/worker identity comparison ceiling')
        returned.update(bytes_read=0, bytes_written=0, identity_hash_attempts_actual=1,
                        identity_hash_bytes_actual=size)


def _read_final_seal(service, reference, body_reference, actual_live):
    profile = services._profile(service.spec['bootstrap']['operation_profiles']['input_io'])
    with update._span(service.operation, 'input_io', dict(profile, wall_contract=contracts.OPAQUE_INPUT_FRAME)) as returned:
        root = Path(service.spec['resource']['resource_probe']['output_root']).resolve(); path = Path(reference['path'])
        require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
                and path.stat().st_size == reference['bytes'], 'exact owned final body seal')
        with path.open('rb') as stream: blob = stream.read(reference['bytes'] + 1)
        require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == reference['sha256'], 'final body seal bytes/hash drift')
        seal = local.raw._json(blob)
        require(seal.get('schema') == 'CE8-PARENT-BODY-SEAL-1' and seal.get('request_sha256') == actual_live['request_sha256']
                and seal.get('scope_sha256') == actual_live['scope_sha256'] and seal.get('native_qualified') is False
                and seal.get('os_terminal_verified') is False, 'same actual worker final body seal')
        body = seal['body_seal']; require(body['status'] == 'UNQUALIFIED_BODY_SEAL'
            and all(body['body_ref'][key] == body_reference[key] for key in ('path', 'sha256', 'bytes'))
            and type(body.get('scope_bootstrap_evidence_ref')) is dict
            and type(body.get('checkpoint_child_terminal_replay_ref')) is dict, 'exact final scope and child replay refs retained')
        returned.update(bytes_read=len(blob), identity_hash_attempts_actual=1, identity_hash_bytes_actual=len(blob))
        return seal


def _retain_parent_service_refs(service, final_seal):
    """After actual exit, pin the configured worker's three ended ledger files."""
    from . import ce_qualification_entry as entry
    require(final_seal['parent_service_specification'] == entry._configuration()['service_spec']['bootstrap'],
            'final worker service ledger must equal the exact adopted parent specification')
    profile = services._profile(service.spec['bootstrap']['operation_profiles']['input_io'])
    with update._span(service.operation, 'input_io', dict(profile, wall_contract=contracts.OPAQUE_INPUT_FRAME)) as returned:
        root = Path(service.spec['resource']['resource_probe']['output_root']).resolve()
        owner = Path(final_seal['parent_service_specification']['owned_directory'])
        require(owner.resolve().is_relative_to(root) and not owner.is_symlink(), 'actual configured parent service evidence owner')
        names = {'header_ref': 'SERVICE_BOOTSTRAP_HEADER.json', 'journal_ref': 'SERVICE_BOOTSTRAP.jsonl',
                 'seal_ref': 'SERVICE_BOOTSTRAP_SEAL.json'}
        references = {}; count = 0
        for key, name in names.items():
            path = owner / name
            require(path.is_file() and not path.is_symlink(), 'parent service did not end before actual successful exit')
            size = path.stat().st_size
            require(0 < size and count + size <= profile['read_bytes_upper'], 'final parent service read ceiling before open')
            hashed = hashlib.sha256(); actual = 0
            with path.open('rb') as stream:
                while block := stream.read(min(1024 ** 2, size + 1 - actual)):
                    actual += len(block); require(actual <= size, 'ended parent evidence grew'); hashed.update(block)
            require(actual == size, 'ended parent service evidence truncated')
            count += actual; references[key] = {'path': str(path), 'bytes': actual, 'sha256': hashed.hexdigest()}
        returned.update(bytes_read=count, identity_hash_attempts_actual=3, identity_hash_bytes_actual=count)
        return {'specification': final_seal['parent_service_specification'], **references,
                'native_qualified': False, 'execution_verified': False}


def _retain(service, policy, envelope):
    profile = services._profile(service.spec['bootstrap']['operation_profiles']['terminal_publication'])
    maximum = policy['max_json_bytes']
    require(profile['read_bytes_upper'] >= 2 * maximum + 2 and profile['write_bytes_upper'] >= maximum
            and profile['identity_hash_attempts_upper'] >= 1 and profile['identity_hash_bytes_upper'] >= maximum,
            'whole terminal envelope publication reserved before paths')
    with update._span(service.operation, 'terminal_publication', dict(profile)) as returned:
        blob = worker.canonical(envelope) + b'\n'
        require(len(blob) <= maximum, 'bounded terminal envelope')
        directory = Path(policy['owned_directory']); root = Path(service.spec['resource']['resource_probe']['output_root'])
        require(directory.is_absolute() and directory.resolve().is_relative_to(root.resolve())
                and not directory.is_symlink(), 'supervision envelope stays in admitted output owner')
        path = directory / 'QUALIFICATION_SUPERVISION.json'
        require(not os.path.lexists(path), 'create-only supervision envelope')
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / (path.name + '.' + uuid.uuid4().hex + '.partial')
        with temporary.open('xb') as stream:
            require(stream.write(blob) == len(blob), 'short terminal envelope write')
            stream.flush(); os.fsync(stream.fileno())
        with temporary.open('rb') as stream: observed = stream.read(len(blob) + 1)
        require(observed == blob, 'terminal temporary readback')
        publication._atomic_no_replace(temporary, path, directory)
        with path.open('r+b') as stream: os.fsync(stream.fileno()); observed = stream.read(len(blob) + 1)
        require(observed == blob, 'terminal final readback')
        returned.update(bytes_read=2 * len(blob), bytes_written=len(blob), identity_hash_attempts_actual=1,
                        identity_hash_bytes_actual=len(blob))
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def supervise_qualification(service, launch_spec, launch_permit):
    """Launch one exactly admitted worker, then retain actual exit provenance."""
    require(type(service) is services.NativeControllerServices, 'actual original metadata supervisor service required')
    service._admit()  # existing closed registry denies before paths/launch
    launch._admit(launch_spec, launch_permit, service.operation)
    require(launch_spec['worker_module'] == ENTRY_MODULE
            and launch_spec['request_sha256'] == services.digest(services.controller.request_preimage(service.request))
            and launch_spec['scope_sha256'] == service.request['scope_ref']['sha256']
            and launch_spec['approved_code_sha256'] == service.spec['approved_code_sha256'],
            'supervisor can launch only the exact admitted qualification entry/request/source')
    policy = service.spec.get('supervision')
    require(type(policy) is dict and set(policy) == {'schema', 'owned_directory', 'max_json_bytes', 'finalizer'}
            and policy['schema'] == SPEC and type(policy['max_json_bytes']) is int
            and 0 < policy['max_json_bytes'] <= 4 * 1024 ** 2, 'explicit bounded supervisor publication policy')
    owned = retained = None; primary = None; completed = False
    try:
        service.gate.launch()
        owned = launch.launch_native(launch_spec, launch_permit, service.operation)
        require(type(owned) is launch.OwnedProcess and type(owned.monitor) is worker.WinWorkerHandle,
                'actual owned process and retained native worker monitor required')
        ready = owned.handshake()
        require(type(ready) is dict and set(ready) == {'worker_proof'}, 'exact qualification ready protocol')
        actual_live = owned.monitor.proof()
        _compare_ready(service, actual_live, ready['worker_proof'])
        response = owned.request('qualify', {})
        require(type(response) is dict and set(response) == {'body_completed', 'os_terminal_verified', 'native_qualified', 'raw_artifact', 'final_seal_ref'}
                and response['body_completed'] is True and response['os_terminal_verified'] is False
                and response['native_qualified'] is False, 'unqualified body response protocol')
        body_reference = _reference(response['raw_artifact'])
        seal_reference = _reference(response['final_seal_ref'])
        require(owned.request('finish', {}) == {'finished': True, 'native_qualified': False}, 'explicit qualification finish handshake')
        process_terminal = owned.terminal(launch_spec['deadline_seconds'])
        require(type(process_terminal['exit_code']) is int and process_terminal['exit_code'] == 0,
                'actual qualification worker failed before completion')
        terminal = worker.verify_worker_handle(owned.monitor, {'request_sha256': launch_spec['request_sha256'],
            'scope_sha256': launch_spec['scope_sha256'], 'worker_identity_sha256': actual_live['worker_identity_sha256']},
            require_native=True, terminal=True)
        identity_fields = ('pid', 'creation_filetime_100ns', 'boot_identity', 'image_path', 'argv')
        require(terminal['exit_code'] == 0
                and all(terminal['os_identity'][key] == process_terminal['os_identity'][key] for key in identity_fields)
                and terminal['exit_filetime_100ns'] == process_terminal['exit_filetime_100ns'],
                'owned terminal and independently retained native handle describe the same exit')
        _read_body(service, body_reference, actual_live)
        final_seal = _read_final_seal(service, seal_reference, body_reference, actual_live)
        parent_service_evidence = _retain_parent_service_refs(service, final_seal)
        retained = owned.monitor.retain_terminal()
        envelope = {'schema': SCHEMA, 'status': 'UNQUALIFIED_NATIVE_BODY_EXITED',
            'request_sha256': launch_spec['request_sha256'], 'scope_sha256': launch_spec['scope_sha256'],
            'code_sha256': copy.deepcopy(launch_spec['approved_code_sha256']), 'body_ref': copy.deepcopy(body_reference),
            'final_body_seal_ref': copy.deepcopy(seal_reference),
            'final_scope_bootstrap_evidence_ref': final_seal['body_seal']['scope_bootstrap_evidence_ref'],
            'checkpoint_child_terminal_replay_ref': final_seal['body_seal']['checkpoint_child_terminal_replay_ref'],
            'parent_service_evidence': parent_service_evidence,
            'live_worker_proof': actual_live, 'terminal_worker_proof': terminal, 'owned_process_terminal': process_terminal,
            'supervisor_bootstrap_prefix': service.bootstrap.snapshot(),
            'supervisor_constructor_reservation_ref': copy.deepcopy(service.constructor_prepayment.reference),
            'supervisor_cost_scope': 'original supervisor prefix; final publication and owned close follow it and require separate raw reduction',
            'complete_supervisor_clock_verified': False,
            'OS_terminal_verified': True, 'native_qualified': False, 'execution_verified': False,
                'production_training_authorized': False, 'status_file_used': False}
        reference = _retain(service, policy, envelope)
        completed = True
        result = {'raw_artifact': reference, 'terminal_handle': retained, 'envelope': envelope,
                  'native_qualified': False, 'execution_verified': False}
        require(getattr(service, '_qualification_supervision_result', None) is None, 'one source-owned qualification supervision result')
        service._qualification_supervision_result = result
        return result
    except BaseException as error: primary = error; raise
    finally:
        failure = None
        if owned is not None:
            if not completed:
                # Protocol and artifact validation can fail before terminal().
                # Close() requires quiescence; use only this launch's original
                # prepaid stop ticket before releasing its handles and pipes.
                try: owned._stop_owned('qualification_supervisor_failure', primary)
                except BaseException as error: failure = error
            try: owned.close()
            except BaseException as error:
                if failure is None: failure = error
                else: local.raw._accounting(failure, error)
        if retained is not None and (not completed or failure is not None):
            try: retained.close()
            except BaseException as error:
                if failure is None: failure = error
                else: local.raw._accounting(failure, error)
        if failure is not None:
            if primary is not None: local.raw._accounting(primary, failure); raise primary from failure
            raise failure


if __name__ == '__main__':
    # Registry descriptors use canonical source names, including a real -m
    # invocation. Execute the one canonical module body rather than register
    # __main__ callbacks under descriptors for another module name.
    import importlib
    raise SystemExit(importlib.import_module(__package__ + '.ce_qualification_supervisor').main())
