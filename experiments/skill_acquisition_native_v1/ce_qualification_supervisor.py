"""Actual owned-process qualification supervision; no source activation.

The worker publishes an unqualified body artifact while alive. This supervisor
retains its real terminal handle after exit for subsequent independent replay.
RPC flags are checked as protocol fields and never establish native PASS.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import uuid

from . import ce_native_services as services, ce_process_launch as launch
from . import ce_worker_evidence as worker, ce_checkpoint_local_issuer as local
from . import ce_publication as publication, update

SPEC = 'CE8-QUALIFICATION-SUPERVISION-SPEC-1'
SCHEMA = 'CE8-QUALIFICATION-SUPERVISION-1'
ENTRY_MODULE = 'experiments.skill_acquisition_native_v1.ce_qualification_entry'
require = worker.require


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
    with update._span(service.operation, 'input_io', dict(profile)) as returned:
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
    with update._span(service.operation, 'input_io', dict(profile)) as returned:
        local.validate_parent_view(actual_live, ready_proof)
        preimage = worker.worker_identity_preimage(actual_live['os_identity'], actual_live['sources'],
            actual_live['request_sha256'], actual_live['scope_sha256'])
        size = len(worker.canonical(preimage))
        require(size <= 4 * 1024 ** 2, 'ready source/worker identity comparison ceiling')
        returned.update(bytes_read=0, bytes_written=0, identity_hash_attempts_actual=1,
                        identity_hash_bytes_actual=size)


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
    require(type(policy) is dict and set(policy) == {'schema', 'owned_directory', 'max_json_bytes'}
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
        require(type(response) is dict and set(response) == {'body_completed', 'os_terminal_verified', 'native_qualified', 'raw_artifact'}
                and response['body_completed'] is True and response['os_terminal_verified'] is False
                and response['native_qualified'] is False, 'unqualified body response protocol')
        body_reference = _reference(response['raw_artifact'])
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
        retained = owned.monitor.retain_terminal()
        envelope = {'schema': SCHEMA, 'status': 'UNQUALIFIED_NATIVE_BODY_EXITED',
            'request_sha256': launch_spec['request_sha256'], 'scope_sha256': launch_spec['scope_sha256'],
            'code_sha256': copy.deepcopy(launch_spec['approved_code_sha256']), 'body_ref': copy.deepcopy(body_reference),
            'live_worker_proof': actual_live, 'terminal_worker_proof': terminal, 'owned_process_terminal': process_terminal,
            'supervisor_bootstrap_prefix': service.bootstrap.snapshot(),
            'supervisor_constructor_reservation_ref': copy.deepcopy(service.constructor_prepayment.reference),
            'supervisor_cost_scope': 'original supervisor prefix; final publication and owned close follow it and require separate raw reduction',
            'complete_supervisor_clock_verified': False,
            'OS_terminal_verified': True, 'native_qualified': False, 'execution_verified': False,
                'production_training_authorized': False, 'status_file_used': False}
        reference = _retain(service, policy, envelope)
        completed = True
        return {'raw_artifact': reference, 'terminal_handle': retained, 'envelope': envelope,
                'native_qualified': False, 'execution_verified': False}
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
