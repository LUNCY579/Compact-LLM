"""Read-only prefix OS handle transfer into one actual resumed child.

Default native authorities/factories remain closed. The supervisor retains the
original process object; the receiver never opens a terminated PID. Numeric
handle evidence is accepted only with typed metadata permits, exact target and
prefix identities, real access/flags and same-handle terminal observations.

Windows contracts: Microsoft Learn DuplicateHandle, NtQueryObject and
GetHandleInformation. All production OS calls are behind existing permits.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from . import ce_native_authority as authority, ce_worker_evidence as worker
from . import ce_native_services as services, ce_process_launch as launch
from . import ce_checkpoint_local_issuer as local

SCHEMA = 'CE8-PREFIX-HANDLE-TRANSFER-1'
READ_ONLY = 0x1000 | 0x100000  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
_ADOPTION_SEAL = object()
require, digest = worker.require, worker.digest


def duplicate_readonly(api, source_handle, target_process):
    c = api.ctypes
    current = api.kernel.GetCurrentProcess
    current.argtypes, current.restype = [], c.c_void_p
    duplicate = api.kernel.DuplicateHandle
    duplicate.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.POINTER(c.c_void_p), c.c_uint32, c.c_int, c.c_uint32]
    duplicate.restype = c.c_int
    result = c.c_void_p()
    require(bool(duplicate(current(), source_handle, target_process, c.byref(result), READ_ONLY, False, 0))
            and result.value, 'read-only prefix transfer into owned child failed')
    return result.value


def handle_access(api, handle):
    c = api.ctypes
    flags = c.c_uint32()
    get_flags = api.kernel.GetHandleInformation
    get_flags.argtypes, get_flags.restype = [c.c_void_p, c.POINTER(c.c_uint32)], c.c_int
    require(bool(get_flags(handle, c.byref(flags))), 'actual prefix handle flags unavailable')
    class Basic(c.Structure):
        _fields_ = [('attributes', c.c_uint32), ('access', c.c_uint32), ('handles', c.c_uint32),
                    ('pointers', c.c_uint32), ('reserved', c.c_uint32 * 10)]
    value, length = Basic(), c.c_uint32()
    query = api.ntdll.NtQueryObject
    query.argtypes, query.restype = [c.c_void_p, c.c_uint32, c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32)], c.c_long
    require(query(handle, 0, c.byref(value), c.sizeof(value), c.byref(length)) == 0
            and length.value == c.sizeof(value), 'actual prefix handle access unavailable')
    require(flags.value == 0 and value.access == READ_ONLY, 'transferred prefix handle must be read-only/noninheritable/closable')
    return {'granted_access': int(value.access), 'handle_flags': int(flags.value)}


def validate_record(record, context, request, current, delegation):
    require(type(record) is dict and record.get('schema') == SCHEMA and record.get('branch') == 'resumed'
            and request['branch'] == 'resumed' and type(record.get('remote_handle')) is int and 0 < record['remote_handle'] < 2 ** 64
            and record.get('desired_access') == READ_ONLY and record.get('inheritable') is False,
            'exact observed resumed prefix read-only handle transfer')
    local.validate_parent_view(current, record['target_proof'])
    require(record.get('target_request_ref') == current['sources']['request_ref']
            and record.get('parent_os_identity') == delegation['parent_os_identity']
            and record.get('parent_worker_identity_sha256') == delegation['parent_worker_identity_sha256']
            and all(record.get(key) == context[key] == current[key] for key in ('request_sha256', 'scope_sha256')),
            'prefix transfer current target/parent/source/request/scope binding')
    terminal, live = record.get('prefix_terminal_proof'), record.get('prefix_live_proof')
    expected = request['lineage_declaration']['prefix_manifest']['bindings']['qualification_evidence_binding']
    require(type(terminal) is dict and terminal.get('schema') == worker.TERMINAL
            and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0
            and type(terminal.get('exit_filetime_100ns')) is int
            and terminal['exit_filetime_100ns'] >= terminal['os_identity']['creation_filetime_100ns']
            and terminal.get('status_file_used') is False and type(live) is dict
            and terminal.get('os_identity') == live.get('os_identity')
            and terminal.get('worker_identity_sha256') == live.get('worker_identity_sha256') == expected['worker_identity_sha256']
            and all(terminal.get(key) == live.get(key) == context[key] for key in ('request_sha256', 'scope_sha256'))
            and record.get('prefix_manifest_ref') == request['lineage_declaration']['prefix_manifest_ref']
            and live.get('sources', {}).get('source_files') == record.get('prefix_observation_spec', {}).get('source_files')
            and {name: ref['sha256'] for name, ref in live['sources']['source_files'].items()} == context['bindings']['source']['code_sha256']
            and digest(worker.worker_identity_preimage(live['os_identity'], live['sources'], context['request_sha256'], context['scope_sha256'])) == live['worker_identity_sha256'],
            'original retained successful prefix OS/source/manifest identity')
    spec = worker.validate_spec(record['prefix_observation_spec'])
    require(spec['expected_pid'] == terminal['os_identity']['pid']
            and spec['expected_creation_filetime_100ns'] == terminal['os_identity']['creation_filetime_100ns']
            and spec['expected_boot_identity'] == terminal['os_identity']['boot_identity'] == current['os_identity']['boot_identity']
            and spec['request_ref'] == live['sources']['request_ref'], 'transferred prefix exact original observation scope')
    return record


def transfer_prefix(context, permit, prefix_handle, prefix_live_proof, prefix_manifest_ref, owned, delegation, operation, gate):
    authority.require_execution_permit(permit, context, 'checkpoint_resume')
    require(type(prefix_handle) is worker.WinWorkerHandle and type(owned) is launch.OwnedProcess
            and owned._seal is launch._OWNED_SEAL and owned.native is True and not owned.closed
            and type(owned.permit) is launch.NativeLaunchPermit and type(owned.monitor) is worker.WinWorkerHandle,
            'actual retained prefix and one owned native resumed launch required')
    owned.permit.require(owned.spec, owned.observer)
    profile = services._profile(context['bindings']['scope']['checkpoint_processes']['prefix_transfer_bounds'])
    remote = None
    def work():
        nonlocal remote
        require(type(gate) is services.NativeResourceGate,
                'source-owned resource gate for delegated metadata transfer')
        gate.metadata_for_delegated_child('checkpoint_resume', owned, delegation)
        terminal = worker.verify_worker_handle(prefix_handle, {key: prefix_live_proof[key] for key in
            ('worker_identity_sha256', 'request_sha256', 'scope_sha256')}, require_native=True, terminal=True)
        require(terminal['exit_code'] == 0 and terminal['os_identity'] == prefix_live_proof['os_identity'], 'original prefix terminal object')
        target = worker.verify_worker_handle(owned.monitor, {'worker_identity_sha256': owned.worker_identity_sha256,
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256']}, require_native=True)
        require(delegation['child_worker_identity_sha256'] == target['worker_identity_sha256']
                and delegation['child_os_identity'] == target['os_identity']
                and all(owned.identity[key] == target['os_identity'][key] for key in
                    ('pid', 'creation_filetime_100ns', 'boot_identity', 'image_path', 'argv'))
                and owned.spec['request_ref'] == target['sources']['request_ref'],
                'exact current owned transfer recipient process/OS/request')
        remote = duplicate_readonly(owned.os, prefix_handle.handle, owned.worker_handle)
        return {'schema': SCHEMA, 'branch': 'resumed', 'remote_handle': remote, 'desired_access': READ_ONLY, 'inheritable': False,
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'parent_os_identity': delegation['parent_os_identity'],
            'parent_worker_identity_sha256': delegation['parent_worker_identity_sha256'],
            'target_proof': target, 'target_request_ref': target['sources']['request_ref'],
            'prefix_terminal_proof': terminal, 'prefix_live_proof': copy.deepcopy(prefix_live_proof),
            'prefix_observation_spec': copy.deepcopy(prefix_handle.spec), 'prefix_manifest_ref': copy.deepcopy(prefix_manifest_ref),
            'native_qualified': False}
    try: return local._paid(operation, profile, 'prefix_handle_transfer', work)
    except BaseException as primary:
        if remote is not None:
            # The numeric value belongs to the target process and may be reused
            # after adoption. Never close it by value from the sender; existing
            # owned child termination closes all target handles on this failure.
            primary.prefix_remote_cleanup = {'handle': remote, 'owner': 'owned target process',
                'requires_actual_target_terminal': True, 'remote_close_attempted': False}
        raise


class _AdoptionAuthority:
    __slots__ = ('permit', 'request', 'service_spec', 'observation_sha256', '_seal')
    def __init__(self, permit, request, service_spec, observation):
        services._admit(permit, request, service_spec)
        for key, value in (('permit', permit), ('request', copy.deepcopy(request)), ('service_spec', copy.deepcopy(service_spec)),
                ('observation_sha256', digest(observation)), ('_seal', _ADOPTION_SEAL)):
            object.__setattr__(self, key, value)
    def __setattr__(self, key, value): raise worker.WorkerEvidenceError('prefix adoption authority is immutable')


def _verify_prefix_adoption(value, observation_sha, request_sha, scope_sha):
    require(type(value) is _AdoptionAuthority and getattr(value, '_seal', None) is _ADOPTION_SEAL,
            'exact source-owned prefix adoption metadata authority')
    services._admit(value.permit, value.request, value.service_spec)
    require(observation_sha == value.observation_sha256 and request_sha == digest(services.controller.request_preimage(value.request))
            and scope_sha == value.request['scope_ref']['sha256'], 'prefix adoption scope changed')
    return True


def prefix_terminal_handle(common, request, service_permit, operation, *, controller_request, service_spec,
                           current_handle, observation_permit, delegation):
    require(any(_verify_prefix_adoption is value for value in worker._REGISTERED_METADATA_VERIFIERS),
            'prefix adoption metadata verifier remains unregistered')
    services._admit(service_permit, controller_request, service_spec)
    require(controller_request['phase'] == common['phase'] == 'qualification', 'prefix transport is qualification metadata only')
    require(type(current_handle) is worker.WinWorkerHandle and type(observation_permit) is worker.WorkerObservationPermit,
            'actual current recipient observation required')
    observation_permit.require(current_handle.spec, operation)
    current = current_handle.proof()
    require(current['os_identity']['pid'] == os.getpid(), 'prefix handle is adopted only inside actual target child')
    record = delegation['prefix_handle_transfer']; validate_record(record, common, request, current, delegation)
    profile = services._profile(service_spec['bootstrap']['operation_profiles']['prefix_handle_transfer'])
    owned = None
    def adopt():
        nonlocal owned
        spec = copy.deepcopy(record['prefix_observation_spec'])
        issuer = _AdoptionAuthority(service_permit, controller_request, service_spec, spec)
        permit = worker.WorkerObservationPermit(spec, issuer, _verify_prefix_adoption, operation)
        budget = worker._Budget(spec, operation); api = budget.run('worker_os_binding', worker._WindowsOS)
        def identify():
            nonlocal owned
            require(record['remote_handle'] < 1 << (8 * api.ctypes.sizeof(api.ctypes.c_void_p)), 'remote prefix value fits current pointer width')
            handle_access(api, record['remote_handle'])
            facts = api.snapshot(record['remote_handle'], command=False, terminal=True)
            terminal = record['prefix_terminal_proof']; expected = terminal['os_identity']
            require(facts['pid'] == expected['pid'] and facts['creation_filetime_100ns'] == expected['creation_filetime_100ns']
                    and facts['signaled'] is True and facts['exit_code'] == 0 and api.boot() == expected['boot_identity'],
                    'actually transferred terminal prefix handle identity, never reopened PID')
            owned = record['remote_handle']
            return facts
        facts = budget.run('worker_os_binding', identify)
        clone = object.__new__(worker.WinWorkerHandle)
        clone.spec, clone.observer, clone.native, clone._permit = spec, operation, True, permit
        clone.budget, clone.os, clone.handle, clone.closed, clone._seal = budget, api, owned, False, worker._HANDLE_SEAL
        clone.identity, clone.sources = copy.deepcopy(record['prefix_live_proof']['os_identity']), copy.deepcopy(record['prefix_live_proof']['sources'])
        clone.identity_sha256 = record['prefix_live_proof']['worker_identity_sha256']
        for ref in (spec['python_ref'], spec['process_image_ref'], spec['request_ref'], *spec['source_files'].values()): worker._file(ref, budget)
        require(clone.terminal_evidence() == record['prefix_terminal_proof'], 'adopted prefix same-object terminal/source proof changed')
        return clone
    try: return local._paid(operation, profile, 'prefix_handle_transfer', adopt)
    except BaseException as primary:
        if owned is not None:
            try: current_handle.os.close(owned)
            except BaseException as cleanup:
                primary.prefix_adoption_close_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
        raise
