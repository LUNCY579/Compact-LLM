"""Inert typed envelope around actual-produced bytes; no native qualification.

This supplies declared CPU resource/containment metadata for source composition.
It never rewrites a producer document and never creates PRESERVATION.json. The
actual selected preserver must make all raw/lease/source copies and its receipt.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import math

PRODUCER_NAMES = frozenset(('PHASE_OBSERVATIONS.jsonl', 'ownership_verified.json',
    'input_check_summary.json', 'worker_result.json', 'PHASE_ACCOUNTING_FINAL.json',
    'OWNED_WORKER_RETURN.json'))
LEASE_NAMES = ('reservation.json', 'worker.json', 'resume.json', 'terminal.json')
GIB = 1024 ** 3
CPU_BIRTH = 134000000000000001
GPU_UUID = 'GPU-00000000-0000-0000-0000-000000000001'


def _descriptor(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _new_bytes(C, path, raw):
    import os
    C.require(type(raw) is bytes and len(raw) <= C.MAX_JSON, 'bounded CPU envelope bytes')
    with Path(path).open('xb') as stream:
        C.require(stream.write(raw) == len(raw), 'complete exclusive CPU envelope write')
        stream.flush()
        os.fsync(stream.fileno())
    C.require(C.read_bytes(path) == raw, 'exact CPU envelope readback')


def _raw(C, value):
    return (C.canonical(value) + '\n').encode('ascii')


def _identity(C, value):
    return C.identity({key: value.get(key) for key in ('pid', 'creation_filetime_100ns')})


def _topology(C, membership, supplied):
    """Validate the supplied normalized inert topology, without process APIs."""
    injected = supplied is None
    if supplied is None:
        root = membership['launcher_root_pid']
        C.require(membership['actual_worker_pid'] == root and
            membership['actual_parent_pid'] == membership['supervisor_pid'],
            'distinct interpreter requires explicit complete CPU FILETIME topology')
        supplied = {'job_name': membership['job_name'],
            'owned_birth_roster': [{'pid': root, 'parent_pid': membership['supervisor_pid'],
                'creation_filetime_100ns': CPU_BIRTH}],
            'supervisor_identity': {'pid': membership['supervisor_pid'],
                'creation_filetime_100ns': CPU_BIRTH - 1}, 'membership': membership}
    C.require(type(supplied) is dict and set(supplied) == {
        'job_name', 'owned_birth_roster', 'supervisor_identity', 'membership'},
        'exact normalized CPU owned topology')
    C.require(supplied['membership'] == membership and supplied['job_name'] == membership['job_name'],
        'unchanged actual-produced membership topology join')
    rows = supplied['owned_birth_roster']
    C.require(type(rows) is list and 0 < len(rows) <= 256, 'finite CPU birth roster')
    mapping = {}
    for row in rows:
        C.require(type(row) is dict and set(row) == {'pid', 'parent_pid', 'creation_filetime_100ns'} and
            type(row['parent_pid']) is int and row['parent_pid'] > 0, 'typed CPU parent/FILETIME row')
        birth = _identity(C, row)
        C.require(birth['pid'] not in mapping, 'unique CPU birth PID')
        mapping[birth['pid']] = {**birth, 'parent_pid': row['parent_pid']}
    supervisor = _identity(C, supplied['supervisor_identity'])
    root = membership['launcher_root_pid']
    actual = membership['actual_worker_pid']
    C.require(supervisor['pid'] == membership['supervisor_pid'] and supervisor['pid'] not in mapping and
        root in mapping and actual in mapping and mapping[root]['parent_pid'] == supervisor['pid'] and
        mapping[actual]['parent_pid'] == membership['actual_parent_pid'], 'complete CPU launcher/interpreter/supervisor topology')
    for pid in mapping:
        seen = set()
        current = pid
        while current != root:
            C.require(current in mapping and current not in seen, 'finite rooted CPU parent topology')
            seen.add(current)
            current = mapping[current]['parent_pid']
    result = {'job_name': supplied['job_name'], 'owned_birth_roster': [mapping[p] for p in sorted(mapping)],
        'supervisor_identity': supervisor, 'membership': deepcopy(membership)}
    return result, injected


def materialize_full_native_CPU_fixture(fixture, actual_raw_directory, *, context=None,
        topology=None, ownership=None):
    """Return raw/lease paths, topology and explicit CPU proof for preservation.

    ``fixture`` is an existing OuterContextFixture. ``actual_raw_directory`` must
    hold the actual six producer files. ``topology`` is the normalized binding
    topology observation; ``ownership`` is its original six-field owner. Neither
    argument is a current host observation. Stage provenance mismatches remain
    in proof.compatibility_diagnostics and do not get repaired or relabelled.
    """
    C = fixture.C
    context = fixture.context() if context is None else context
    value = C.context_value(context)
    C.require(value.get('fixture_only') is True and value.get('native_PASS') is False and
        value.get('native_authority_issued') is False, 'sealed CPU context required')
    C.require(value['config_path'] == str(fixture.config_path) and value['anchors'] == fixture.anchors,
        'same original selected CPU fixture context anchors')
    source = Path(actual_raw_directory).resolve()
    C.require(source.is_dir() and source.is_relative_to(fixture.root.parent),
        'CPU producer directory confined to fixture workspace')
    names = sorted(p.name for p in source.iterdir())
    C.require(PRODUCER_NAMES <= set(names) and len(names) <= 64 and
        'PRESERVATION.json' not in names and 'SUPERVISOR_RECEIPT.json' not in names and
        all((source / n).is_file() and not (source / n).is_symlink() for n in names),
        'finite actual-produced flat files without prebuilt preservation/terminal')
    original = {n: C.read_bytes(source / n) for n in names}
    C.require(sum(map(len, original.values())) <= 32 * 1024 ** 2, 'bounded actual producer bytes')
    container = C.parse(original['OWNED_WORKER_RETURN.json'])
    C.require(type(container) is dict and set(container) == {'report', 'runtime_execution_modules',
        'owned_membership_verified', 'IO', 'native_PASS'} and container['native_PASS'] is False,
        'actual returned producer container schema, no approval')
    membership = container['owned_membership_verified']
    C.require(type(membership) is dict and set(membership) == {'actual_worker_pid', 'actual_parent_pid',
        'launcher_root_pid', 'supervisor_pid', 'job_name', 'owned_job_membership_verified',
        'query_job_handle_closed_before_framework_import'} and all(type(membership[k]) is int and
        membership[k] > 0 for k in ('actual_worker_pid', 'actual_parent_pid', 'launcher_root_pid', 'supervisor_pid')) and
        type(membership['job_name']) is str and membership['job_name'] and
        membership['owned_job_membership_verified'] is membership['query_job_handle_closed_before_framework_import'] is True,
        'actual produced typed CPU membership')
    execution = container['runtime_execution_modules']
    C.require(type(execution) is dict and execution, 'actual produced module metadata')
    found = [entry['topology'] for entry in execution.values() if type(entry) is dict and 'topology' in entry]
    if topology is None and found:
        C.require(all(row == found[0] for row in found), 'consistent actual binding topology observations')
        topology = found[0]
    topology, injected_topology = _topology(C, membership, topology)
    births = [_identity(C, row) for row in topology['owned_birth_roster']]
    root_birth = next(row for row in births if row['pid'] == membership['launcher_root_pid'])
    if ownership is None:
        ownership = {'worker_pid': root_birth['pid'], 'root_worker_pid': root_birth['pid'],
            'supervisor_pid': membership['supervisor_pid'], 'supervisor_create_time': 1000.25,
            'job_name': membership['job_name'], 'config_sha256': value['anchors']['expected_config_file']}
    C.require(type(ownership) is dict and set(ownership) == {'worker_pid', 'root_worker_pid',
        'supervisor_pid', 'supervisor_create_time', 'job_name', 'config_sha256'} and
        ownership['worker_pid'] == ownership['root_worker_pid'] == root_birth['pid'] and
        ownership['supervisor_pid'] == membership['supervisor_pid'] and
        ownership['job_name'] == membership['job_name'] and
        ownership['config_sha256'] == value['anchors']['expected_config_file'] and
        type(ownership['supervisor_create_time']) in (int, float) and
        math.isfinite(ownership['supervisor_create_time']) and ownership['supervisor_create_time'] > 0,
        'exact CPU original ownership/config join')
    ledger = [C.parse(line) for line in original['PHASE_OBSERVATIONS.jsonl'].splitlines()]
    C.require(0 < len(ledger) <= 128 and all(type(row) is dict and type(row.get('monotonic')) in (int, float)
        and math.isfinite(row['monotonic']) and row['monotonic'] >= 0 for row in ledger),
        'bounded actual producer phase clocks')
    clocks = [row['monotonic'] for row in ledger]
    C.require(clocks == sorted(clocks), 'unchanged actual producer chronological ledger')
    first, last = clocks[0], clocks[-1]
    if 'LIBRARY_BOUNDARY_EVIDENCE.json' in original:
        # Explicit CPU envelope assumptions span actual-produced library clocks
        # as well as core clocks. These cadence/resources are never OS facts.
        packet=C.parse(original['LIBRARY_BOUNDARY_EVIDENCE.json']).get('raw_packet')
        points=[] if packet is None else [row.get('time_ns') for row in packet.get('observations',[])]
        known=[point/10**9 for point in points if type(point) is int and point>=0]
        if known:first=min(first,min(known));last=max(last,max(known))
    resume = first - .001
    C.require(resume >= 2.04 and last - first < 590, 'finite inert envelope admission/work times')
    anchors = {'config_path': value['config_path'], 'anchors': deepcopy(value['anchors'])}
    if 'supervisor_ownership.json' in original:
        C.require(C.parse(original['supervisor_ownership.json']) == ownership, 'unaltered actual producer ownership file')
    if 'OWNED_EXTERNAL_ANCHORS.json' in original:
        C.require(C.parse(original['OWNED_EXTERNAL_ANCHORS.json']) == anchors, 'unaltered actual producer selected anchors')
    output = fixture.package / 'artifacts/prefix_input_owned_run_CPU'
    C.require(source == output or not output.exists(), 'fresh owned CPU output leaf')
    lease = fixture.package / 'artifacts/_native_gpu_leases/CPU_FIXTURE_LEASE'
    C.require(not lease.exists(), 'fresh inert owned CPU lease leaf')
    if source != output:
        output.mkdir()
        for name, raw in original.items():
            _new_bytes(C, output / name, raw)
    lease.mkdir(parents=True)
    def publish(name, item):
        if name in original:
            C.require(C.parse(original[name]) == item, 'same original CPU envelope metadata')
        else:
            _new_bytes(C, output / name, _raw(C, item))
    publish('supervisor_ownership.json', deepcopy(ownership))
    publish('OWNED_EXTERNAL_ANCHORS.json', anchors)
    for name in ('worker.stdout.log', 'worker.stderr.log'):
        if name not in original:
            _new_bytes(C, output / name, b'')
    if 'supervisor_heartbeat.jsonl' not in original:
        _new_bytes(C, output / 'supervisor_heartbeat.jsonl', _raw(C,
            {'unix_seconds': 1000.5, 'supervisor_pid': membership['supervisor_pid']}))
    def admission(clock, index):
        return {'sample_started_monotonic': clock, 'sample_completed_monotonic': clock,
            'host_available_bytes': 8 * GIB, 'worker_tree_rss_bytes': 512 * 1024 ** 2,
            'disk_free_bytes': 64 * GIB, 'gpu_sample_started_monotonic': clock,
            'gpu_sample_completed_monotonic': clock, 'gpu_free_bytes': 16 * GIB,
            'gpu_total_bytes': 24 * GIB, 'GPU_index': 0, 'gpu_uuid': GPU_UUID,
            'gpu_observation_id': 'CPU_INERT_METADATA_' + str(index)}
    samples = []
    # Explicit CPU metadata cadence spans even slow source preflight gaps. None
    # of these points is represented as a measured host or GPU observation.
    cadence_clocks = [resume + .1 * index for index in range(int((last - resume) / .1) + 1)]
    for clock in sorted(set(cadence_clocks + clocks + [last])):
        samples.extend([{'kind': 'CPU', 'sample_started_monotonic': clock,
            'host_available_bytes': 8 * GIB, 'worker_tree_rss_bytes': 512 * 1024 ** 2,
            'owned_process_identities': deepcopy(births)},
            {'kind': 'GPU', 'gpu_sample_started_monotonic': clock,
                'gpu_free_bytes': 16 * GIB, 'gpu_total_bytes': 24 * GIB, 'gpu_uuid': GPU_UUID}])
    publish('RESOURCE_OBSERVATIONS.json', samples)
    full_owner = {**root_birth, 'create_time': 1000.5, 'parent_pid': membership['supervisor_pid'],
        'assigned_before_resume': True, 'job_kill_on_close': True, 'breakaway_allowed': False,
        'job_name': membership['job_name'], 'process_handle': 101, 'job_handle': 102}
    successful = container['report']['status'] == 'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT'
    partial = [{'path': p.name, **_descriptor(C.read_bytes(p))} for p in sorted(output.iterdir())]
    now = datetime.now(timezone.utc).isoformat()
    supervisor = {'schema_version': 1, 'started_utc': now, 'finished_utc': now,
        'output_dir': str(output), 'config_sha256': ownership['config_sha256'],
        'preflight': {'config_sha256': ownership['config_sha256'], 'config_path': value['config_path'],
            'worker_path': str(fixture.package / 'prefix_input_owned_worker.py'),
            'authorization': deepcopy(value['config']['authorization']),
            'resources': deepcopy(value['config']['resources']), 'source_commit': value['expected_head'],
            'profile_sha256': value['registration']['profile_sha256'],
            'source_inventory_sha256': C.digest(value['config']['source_files']),
            'runtime_inventory_sha256': C.digest(value['runtime']['files'])},
        'worker_attempted': True, 'worker_launched': True, 'worker_resumed': True,
        'preflight_revalidated_before_current_observations': True, 'job_closed_kill_on_close': True,
        'sampler_threads_terminal_verified': True, 'gpu_telemetry_cleanup_verified': True,
        'partial_outputs_frozen': True, 'partial_outputs': partial,
        'owned_worker_identity_currently_alive': False, 'unjoined_sampler_threads': [],
        'elapsed_seconds': 2.05 + last - first, 'worker_identity': full_owner,
        'owned_process_identities': deepcopy(births), 'owned_topology': deepcopy(topology),
        'worker_resume_monotonic': resume, 'deadline_monotonic': resume + 600,
        'admission_observations': [admission(first - 2.03, 0), admission(first - .025, 1)],
        'dispatch_observation': admission(first - .005, 2), 'lease_path': str(lease),
        'status': 'SUPERVISED_WORKER_EXITED_ZERO' if successful else 'SUPERVISED_WORKER_FAILED',
        'returncode': 0 if successful else 2, 'fixture_only': True, 'native_PASS': False,
        'native_authority_issued': False, 'mechanical_qualification_established': False,
        'production_training_authorized': False, 'protected_bank_access': False,
        'gpu_telemetry_backend': 'CPU_INJECTED_METADATA_NO_GPU_CALL'}
    _new_bytes(C, output / 'SUPERVISOR_RECEIPT.json', _raw(C, supervisor))
    lease_values = {'reservation.json': {'config_sha256': ownership['config_sha256'],
        'output_dir': str(output), 'fixture_only': True}, 'worker.json': full_owner,
        'resume.json': {'worker_identity': full_owner, 'external_monitor_ready_before_resume': True,
            'resumed_monotonic': resume, 'deadline_monotonic': resume + 600}, 'terminal.json': supervisor}
    for name in LEASE_NAMES:
        _new_bytes(C, lease / name, _raw(C, lease_values[name]))
    diagnostics = []
    known = {**value['config']['source_files'], **value['runtime']['files']}
    for name, observed in execution.items():
        path = observed.get('path') if type(observed) is dict else None
        if path not in known or observed.get('sha256') != known[path]['sha256']:
            diagnostics.append({'module': name, 'observed_path': path,
                'reason': 'actual-produced module is absent from selected context map or hash differs'})
    snapshots = {}
    with fixture.activate():
        for path, descriptor in value['config']['source_files'].items():
            C.file_descriptor(path, descriptor)
            snapshots[path] = deepcopy(descriptor)
    for name, raw in original.items():
        C.require(C.read_bytes(source / name) == C.read_bytes(output / name) == raw,
            'every actual producer byte conserved without relabel')
    C.require(source == output or set(p.name for p in source.iterdir()) == set(names),
        'original producer roster conserved')
    C.require(C.context_value(context) == value and not (output / 'PRESERVATION.json').exists() and
        set(p.name for p in lease.iterdir()) == set(LEASE_NAMES), 'CPU original context/no prebuilt preservation/exact lease roster')
    proof = {'kind': 'CPU_ACTUAL_PRODUCER_TYPED_ENVELOPE_ONLY', 'fixture_only': True,
        'native_PASS': False, 'native_authority_issued': False, 'native_fit_established': False,
        'current_runtime_qualified': False, 'model_execution_performed': False,
        'process_execution_performed': False, 'GPU_operations_performed': False,
        'producer_bytes_unchanged': True, 'producer_files': {n: _descriptor(raw) for n, raw in original.items()},
        'source_map_snapshot_metadata': snapshots, 'context_projection_sha256': C.digest(value),
        'metadata_fields_injected': ['containment/terminal flags', 'CPU/GPU resource values',
            'admission times', 'heartbeat/logs', 'handle values', 'lease reservation'],
        'topology_metadata_injected_by_envelope': injected_topology,
        'workflow_source_map_compatible': not diagnostics, 'compatibility_diagnostics': diagnostics,
        'preservation_created': False}
    proof['sha256'] = C.digest(proof)
    return {'raw': output, 'lease_root': lease, 'context': context, 'supervisor': supervisor,
        'owner': root_birth, 'ownership': deepcopy(ownership), 'owned_identities': births,
        'topology': topology, 'anchors': anchors, 'producer_raw': source, 'proof': proof}


def verify_full_native_CPU_fixture_conservation(fixture, envelope):
    """Check producer, SUP inventory, lease and selected anchor conservation."""
    C = fixture.C
    output = envelope['raw']
    for name, descriptor in envelope['proof']['producer_files'].items():
        original = C.read_bytes(envelope['producer_raw'] / name)
        C.require(_descriptor(original) == descriptor and C.read_bytes(output / name) == original,
            'unchanged actual producer original/envelope bytes')
    supervisor = C.parse(C.read_bytes(output / 'SUPERVISOR_RECEIPT.json'))
    C.require(supervisor == envelope['supervisor'], 'exact original CPU SUP fields')
    actual = {p.name for p in output.iterdir()}
    expected = {row['path'] for row in supervisor['partial_outputs']} | {'SUPERVISOR_RECEIPT.json'}
    C.require(actual == expected, 'complete SUP partial-output denominator')
    for row in supervisor['partial_outputs']:
        C.require(_descriptor(C.read_bytes(output / row['path'])) == {k: row[k] for k in ('bytes', 'sha256')},
            'same frozen SUP raw descriptor')
    C.require(C.read_bytes(envelope['lease_root'] / 'terminal.json') ==
        C.read_bytes(output / 'SUPERVISOR_RECEIPT.json'), 'same terminal lease/SUP bytes')
    C.require(C.parse(C.read_bytes(output / 'OWNED_EXTERNAL_ANCHORS.json')) == envelope['anchors'] and
        C.parse(C.read_bytes(output / 'supervisor_ownership.json')) == envelope['ownership'],
        'same original selected anchor/ownership fields')
    C.require(not (output / 'PRESERVATION.json').exists(), 'actual preserver remains responsible for receipt')
    return True
