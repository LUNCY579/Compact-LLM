"""Fixed production worker supervision and paid raw completion verification.

Default source configuration remains closed. Actual same-handle exit, retained
CE reservations and saved donor bytes establish body completion only; this
module makes no skill acquisition, retention or measured-price claim.
"""
from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import sys

from . import ce_training_entry as entry, ce_training_runtime as runtime
from . import ce_native_services as services, ce_process_launch as launch
from . import ce_worker_evidence as worker, ce_checkpoint_local_issuer as local
from . import ce_qualification_supervisor as shared, ce_qualification_artifacts as raw
from . import ce_execution_verifier as files, ce_journal as journal
from . import ce_service_bootstrap_verifier as service_replay, ce_delta_artifacts as delta
from . import ce_finalizer_accounting as costs, ce_plan

SPEC = 'CE8-TRAINING-SUPERVISION-SPEC-1'
SCHEMA = 'CE8-TRAINING-COMPLETION-1'
MAX_EVENTS, MAX_JOURNAL = 200000, 256 * raw.CHUNK
require, canonical, digest = raw.require, raw.canonical, raw.digest


def supervisor_specification(configuration):
    entry.validate_configuration(configuration)
    spec = copy.deepcopy(configuration['service_spec'])
    policy = spec.get('supervision')
    require(type(policy) is dict and set(policy) == {'schema', 'owned_directory', 'max_json_bytes', 'finalizer'}
            and policy['schema'] == SPEC and raw._int(policy['max_json_bytes'], 1)
            and policy['max_json_bytes'] <= raw.MAX_JSON, 'exact finite training supervision/finalizer policy')
    root = Path(spec['resource']['resource_probe']['output_root'])
    directory = root / 'supervisor-service-bootstrap'
    spec['bootstrap']['owned_directory'] = str(directory)
    spec['constructor_prepayment']['owned_directory'] = str(directory)
    spec['scope_adapters']['owned_directory'] = str(directory / 'capsules')
    return spec


def service_metadata_verifier(issuer, request_sha256, spec_sha256):
    require(issuer is sys.modules[__name__], 'fixed source training supervisor issuer')
    config = entry._configuration()
    require(request_sha256 == digest(services.controller.request_preimage(config['request']))
            and spec_sha256 == digest(supervisor_specification(config)), 'exact source training supervisor metadata')
    return True


def launch_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    require(type(issuer) is services.NativeControllerServices and issuer.context is None, 'original source metadata supervisor')
    issuer._admit(); spec = getattr(issuer, '_training_launch_spec', None)
    require(type(spec) is dict and digest(spec) == spec_sha256 and spec['worker_module'] == entry.ENTRY_MODULE
            and request_sha256 == spec['request_sha256'] == digest(services.controller.request_preimage(issuer.request))
            and scope_sha256 == spec['scope_sha256'] == issuer.request['scope_ref']['sha256']
            and spec['approved_code_sha256'] == issuer.spec['approved_code_sha256'], 'exact fixed source training launch')
    return True


def worker_metadata_verifier(issuer, spec_sha256, request_sha256, scope_sha256):
    require(type(issuer) is services.NativeControllerServices and issuer.context is None, 'original supervisor observation issuer')
    issuer._admit(); spec = getattr(issuer, '_training_observation_spec', None)
    require(type(spec) is dict and digest(spec) == spec_sha256 and spec['request_sha256'] == request_sha256
            and spec['scope_sha256'] == scope_sha256, 'exact launch-observed training worker metadata')
    return True


def launch_observation(spec, permit, identity, parent_identity, observer):
    require(type(permit) is launch.NativeLaunchPermit, 'source native launch permit before OS observation')
    permit.require(spec, observer); service = permit._issuer
    require(type(service) is services.NativeControllerServices and observer is service.operation, 'one original paid launch observer')
    from .ce_child_scope_adapters import observed_child_spec
    bound = worker._Budget({'bounds': spec['observation_bounds']}, observer).run('worker_source_verification',
        lambda: observed_child_spec(spec, identity, parent_identity))
    service._training_observation_spec = copy.deepcopy(bound)
    return bound, worker.WorkerObservationPermit(bound, service, worker_metadata_verifier, observer)


def build_launch_spec(configuration, parent_identity):
    entry.validate_configuration(configuration)
    service = configuration['service_spec']; policy = service['scope_adapters']['worker']; code = service['approved_code_sha256']
    deadline = service['physical_deadline_seconds']; module = __package__ + '.ce_training_supervisor'
    from .ce_registration import child_environment
    spec = {'schema': launch.SPEC, 'mode': 'registered_native_worker', 'request_sha256': digest(services.controller.request_preimage(configuration['request'])),
        'scope_sha256': configuration['request']['scope_ref']['sha256'], 'cwd': policy['source_checkout'], 'source_checkout': policy['source_checkout'],
        'worker_module': entry.ENTRY_MODULE, 'argv': [policy['python_ref']['path'], '-m', entry.ENTRY_MODULE, '--request', configuration['request_ref']['path']],
        'python_ref': policy['python_ref'], 'process_image_ref': policy['process_image_ref'], 'request_ref': configuration['request_ref'],
        'source_files': policy['source_files'], 'approved_code_sha256': code, 'worker_topology': 'venv_redirector',
        'parent_identity': copy.deepcopy(parent_identity), 'environment': child_environment(), 'deadline_seconds': deadline,
        'handshake_seconds': 180, 'rpc_seconds': deadline, 'terminate_grace_seconds': 30,
        'max_message_bytes': raw.CHUNK, 'max_messages': 8, 'max_stderr_bytes': raw.CHUNK,
        'bounds': dict(copy.deepcopy(policy['bounds']), wall_seconds_upper=deadline), 'observation_bounds': copy.deepcopy(policy['bounds']),
        'observation_factory_descriptor': {'module': module, 'symbol': 'launch_observation', 'source_sha256': code[module.replace('.', '/') + '.py']}}
    launch.validate_spec(spec)
    return spec


def _shadow(header, directory, *, fixture):
    """Initialize only existing semantic replay state; no constructor or IO."""
    state = object.__new__(journal.CeJournal)
    state.synthetic_cpu = fixture
    state.header = copy.deepcopy(header); state.binding_sha256 = digest(header)
    state.plan, state.bindings, state.caps = (copy.deepcopy(header[key]) for key in ('plan', 'bindings', 'caps'))
    state.plan_rows = [list(rows) for rows in ce_plan.validate_plan(state.plan)]
    state.allowed_kinds = journal.KINDS if fixture else journal.NATIVE_KINDS
    state.profiles, state.census = state.bindings['operation_bounds'], copy.deepcopy(header['token_census'])
    require(set(state.profiles) == state.allowed_kinds, 'all original operation profiles retained')
    for profile in state.profiles.values(): journal._profile(profile)
    fields = ('id', 'prompt_tokens', 'labeled_tokens', 'full_sequence_tokens',
              'prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')
    if fixture:
        from .ce_update import prepared_row_binding
        expected_census = {row['id']: prepared_row_binding(row) for row in state.bindings['prepared_rows']}
    else:
        expected_census = {row['id']: {key: row[key] for key in fields} for row in state.bindings['prepared_rows']}
    require(state.census == expected_census, 'original token census differs from admitted prepared rows')
    state.static_recipe_sha256 = digest(state.bindings['recipe'])
    state.directory, state.path = Path(directory), Path(directory) / journal.JOURNAL
    state.metadata_roots = [state.directory] if fixture else [Path(root) for root in state.bindings['metadata_roots']]
    state.counters, state.starts = dict.fromkeys(journal.COUNTERS, 0), dict.fromkeys(state.allowed_kinds, 0)
    state.operations, state.events, state.invocations = {}, [], {}
    state.used_restores, state.metadata_ids = set(), set()
    state.active_attempt = state.current_invocation = state._last_wall = None
    state.requires_restore = state.restore_pending = state.unknown_wall = state.fatal = False
    state._known_size = 0
    state.progress = {'C': 0, 'U': 0, 'history': [], 'generation': 0}
    state.progress['snapshot'] = digest(dict(state.progress, plan_sha256=state.plan['plan_sha256']))
    return state


def replay_journal(artifact, header_ref, journal_ref, snapshot, binding, *, reader, preflight, fixture=False):
    require(type(fixture) is bool and type(reader) is raw.RawEvidenceReader and callable(preflight), 'bounded independently paid raw journal replay')
    require(journal_ref['bytes'] <= MAX_JOURNAL, 'finite original CE journal bytes')
    preflight('qualification_execution_verifier', {'reader_bounds': dict(reader.remaining), 'references': [header_ref, journal_ref],
        'mode': 'production_CE_journal', 'max_events': MAX_EVENTS, 'max_line_bytes': journal.MAX_LINE})
    with reader._overhead('training_journal_original_header', work_bytes=128 * raw.MAX_JSON, hash_attempts=128, hash_bytes=128 * raw.MAX_JSON):
        header = files._read_json(reader, header_ref, binding, 'CE_header')
        require(header.get('schema') == journal.SCHEMA and header['plan'] == artifact['plan']
                and header['bindings'] == artifact['bindings'] and header['caps'] == artifact['caps'], 'CE original header differs from source ledger artifact')
        require(header_ref['path'] == str(Path(artifact['owned_directory']) / journal.HEADER)
                and journal_ref['path'] == str(Path(artifact['owned_directory']) / journal.JOURNAL), 'exact admitted CE journal paths')
        state = _shadow(header, artifact['owned_directory'], fixture=fixture)
        state._charge({'io_write_bytes': header_ref['bytes'], 'metadata_reserved_bytes': header_ref['bytes'],
                       'identity_hash_attempts': 1, 'identity_hash_bytes': len(canonical(header))})
    wrapped = files._decorated(reader, journal_ref, binding, 'CE_journal')
    with reader._overhead('training_original_chain_and_semantic_replay', work_bytes=64 * journal_ref['bytes'],
                         hash_attempts=2 * MAX_EVENTS, hash_bytes=16 * journal_ref['bytes']):
        stream = reader._blocks(wrapped, binding, 'CE_journal'); buffer, primary = b'', None
        try:
            for block in stream:
                buffer += block
                while b'\n' in buffer:
                    split = buffer.index(b'\n') + 1; line, buffer = buffer[:split], buffer[split:]
                    require(len(line) <= journal.MAX_LINE and len(state.events) < MAX_EVENTS, 'bounded original CE event')
                    row = raw._json(line); body = {key: value for key, value in row.items() if key != 'sha256'}
                    require(set(row) == {'seq', 'previous', 'binding_sha256', 'kind', 'payload', 'sha256'}
                            and type(row['seq']) is int and row['seq'] == len(state.events) and row['previous'] == state._head()
                            and row['binding_sha256'] == state.binding_sha256 and digest(body) == row['sha256']
                            and canonical(row) + b'\n' == line, 'original CE hash/order/body chain')
                    require(state.events or row['kind'] == 'setup' and row['payload'] == {'resume': False}, 'original CE setup first')
                    if row['kind'] == 'setup' and row['payload'].get('resume') is True:
                        state._charge({'io_write_bytes': header_ref['bytes'], 'metadata_reserved_bytes': header_ref['bytes'],
                            'identity_hash_attempts': 1, 'identity_hash_bytes': len(canonical(header))})
                    charges, action = state._prepare(row['kind'], row['payload'])
                    state._charge(state._combined(charges, len(line), len(canonical(body))))
                    action(); state.events.append(row); state._known_size += len(line)
                require(len(buffer) <= journal.MAX_LINE, 'bounded CE partial tail')
            require(not buffer and state.events, 'partial CE tail or missing original work; no repair')
        except BaseException as error: primary = error; raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise
    require(state.snapshot() == snapshot, 'CE snapshot refund/progress/pending differs from original raw replay')
    return state


def _read_worker_service(owner, configuration, binding):
    spec = configuration['service_spec']['bootstrap']; directory = Path(spec['owned_directory'])
    refs = {}
    for name, filename in (('header_ref', 'SERVICE_BOOTSTRAP_HEADER.json'), ('journal_ref', 'SERVICE_BOOTSTRAP.jsonl'), ('seal_ref', 'SERVICE_BOOTSTRAP_SEAL.json')):
        upper = service_replay.MAX_JOURNAL if name == 'journal_ref' else service_replay.MAX_HEADER
        identifier = __import__('uuid').uuid4().hex
        owner.operation('finalizer_metadata', 'started', identifier, io_read_bytes_upper=upper + 1, identity_hash_attempts_upper=1, identity_hash_bytes_upper=upper + 1)
        try: refs[name] = runtime._stream_reference(directory / filename, upper)
        except BaseException as error:
            owner.operation('finalizer_metadata', 'failed', identifier, error_type=type(error).__name__); raise
        owner.operation('finalizer_metadata', 'returned', identifier, reference=refs[name])
    context = {'mode': 'registered_scope', 'request_sha256': binding['request_sha256'], 'verification_binding': binding}
    replay = service_replay.verify_service_bootstrap(context, spec, refs['header_ref'], refs['journal_ref'], refs['seal_ref'],
                                                    reader=owner.reader, outer_preflight=owner.preflight)
    return refs, replay


def verify_body(owner, body_ref, terminal_observation, configuration):
    proof = terminal_observation.proof
    binding = {'stage': 'training_supervisor', 'request_sha256': proof['request_sha256'], 'scope_sha256': proof['scope_sha256'],
        'static_recipe_sha256': configuration['service_spec']['scope_adapters']['static_recipe_sha256'],
        'code_sha256': configuration['service_spec']['approved_code_sha256'], 'worker_identity_sha256': proof['worker_identity_sha256'],
        'attempt_id': 'independent-production-completion'}
    body = files._read_json(owner.reader, body_ref, binding, 'training_body')
    require(body.get('schema') == runtime.PUBLICATION and body.get('status') == 'TRAINING_BODY_COMPLETE'
            and body.get('request_sha256') == proof['request_sha256'] and body.get('scope_sha256') == proof['scope_sha256'],
            'actual training body differs from original worker/request/scope')
    request = configuration['request']; scope_doc = files._read_json(owner.reader, request['scope_ref'], binding, 'production_scope')
    ledger = files._read_json(owner.reader, scope_doc['native_ledger_ref'], binding, 'production_ledger_spec')
    anchors = files._read_json(owner.reader, request['anchors_ref'], binding, 'production_anchors')
    require(scope_doc.get('role') == 'production_training' and scope_doc.get('production_training_authorized') is True
            and ledger['bindings']['scope'] == journal.NATIVE_SCOPE and ledger['bindings']['recipe']['synthetic_cpu'] is False
            and ledger['bindings']['source'] == configuration['service_spec']['approved_code_sha256']
            and digest(ledger['bindings']['recipe']) == body['static_recipe_sha256'] == scope_doc['static_recipe_sha256']
            and body.get('qualifiedreceipt_ref') == request['qualifiedreceipt_ref'],
            'body and raw ledger differ from source-approved qualified production recipe/source')
    state = replay_journal(ledger, body['raw_refs']['header'], body['raw_refs']['journal'], body['journal_snapshot'], binding,
                           reader=owner.reader, preflight=owner.preflight)
    require(state.current_invocation is None and body['invocation_id'] in state.invocations
            and state.invocations[body['invocation_id']]['outcome'] == 'COMPLETE', 'actual completed production invocation required')
    runtime.validate_completion(state.snapshot(), body['donor'], {'bindings': {'anchors': anchors}})
    require(body['donor']['final_master_sha256'] == state.progress['history'][-1]['final_policy_id'], 'donor is not the actually committed final master')
    ref = body['raw_refs']['donor']; require(ref['sha256'] == body['donor']['file_sha256'] and ref['path'] == body['donor']['path'], 'same exact source donor bytes')
    wrapped = files._decorated(owner.reader, ref, binding, 'qualification_delta')
    saved = delta.read_delta(owner.reader, wrapped, binding, scope_doc['native_backend']['parameter_inventory'])
    require(all(saved[key] == body['donor'][key] for key in ('data_section_sha256', 'state_sha256', 'numel', 'tensors')),
            'saved donor bytes/inventory differ from source result')
    require(raw._finite(body['donor'].get('delta_norm')) and abs(saved['delta_norm'] - body['donor']['delta_norm'])
            <= 1e-12 + 1e-10 * saved['delta_norm'], 'saved donor norm differs from actual streamed values')
    refs, service = _read_worker_service(owner, configuration, binding)
    suffix = owner.validate_service_suffix(service, terminal_observation)
    snapshot = service['derived_snapshot']
    require(terminal_observation.monotonic - snapshot['original_origin'] <= configuration['service_spec']['bootstrap']['physical_wall_seconds_upper'],
            'worker physical deadline includes actual terminal suffix')
    return {'binding': binding, 'body_ref': body_ref, 'original_journal_refs': body['raw_refs'], 'derived_journal': state.snapshot(),
        'saved_donor': saved, 'worker_service_refs': refs, 'worker_service_replay': service, 'worker_terminal_suffix': suffix,
        'actual_worker_terminal': proof, 'learning_claim': False, 'native_qualified': False}


def supervise_training(service, launch_spec, launch_permit):
    require(type(service) is services.NativeControllerServices, 'original source training supervisor service')
    service._admit(); launch._admit(launch_spec, launch_permit, service.operation)
    require(launch_spec['worker_module'] == entry.ENTRY_MODULE and service.context is None, 'fixed production worker only')
    owned = retained = owner = None; primary = None; terminal_reached = False
    try:
        service.gate.launch(); owned = launch.launch_native(launch_spec, launch_permit, service.operation)
        require(type(owned) is launch.OwnedProcess and type(owned.monitor) is worker.WinWorkerHandle, 'actual owned training process/handle')
        ready = owned.handshake(); require(type(ready) is dict and set(ready) == {'worker_proof'}, 'exact ready protocol')
        live = owned.monitor.proof(); shared._compare_ready(service, live, ready['worker_proof'])
        response = owned.request('train', {})
        require(type(response) is dict and set(response) == {'raw_artifact', 'body_completed', 'os_terminal_verified', 'native_qualified'}
                and response['body_completed'] is True and response['os_terminal_verified'] is False and response['native_qualified'] is False,
                'source body protocol is not a terminal or learning claim')
        shared._reference(response['raw_artifact'])
        require(owned.request('finish', {}) == {'finished': True, 'native_qualified': False}, 'finish exactly after body')
        physical = owned.terminal(launch_spec['deadline_seconds'])
        require(type(physical.get('exit_code')) is int and physical['exit_code'] == 0, 'actual successful original training exit')
        retained = owned.monitor.retain_terminal(); owner = costs.FinalizerCostOwner(service)
        observation = owner.observe_parent_terminal(retained, {**{key: launch_spec[key] for key in ('request_sha256', 'scope_sha256')},
            'worker_identity_sha256': live['worker_identity_sha256']})
        require(observation.proof['os_identity'] == physical['os_identity'] and observation.proof['exit_filetime_100ns'] == physical['exit_filetime_100ns'],
                'same original OS object and actual exit time')
        terminal_reached = True
        verified = verify_body(owner, response['raw_artifact'], observation, entry._configuration())
        replay_ref = owner.write_json('TRAINING_PRE_CLOSE_REPLAY.json', verified)
        owned.close(); owned = None; retained.close(); retained = None
        ended = owner.observe_supervisor_end(service)
        ctx = {'mode': 'registered_scope', 'request_sha256': launch_spec['request_sha256'], 'verification_binding': verified['binding']}
        sr = service_replay.verify_service_bootstrap(ctx, service.spec['bootstrap'], ended.evidence['header_ref'],
            ended.evidence['journal_ref'], ended.evidence['seal_ref'], reader=owner.reader, outer_preflight=owner.preflight)
        supervisor_suffix = owner.validate_service_suffix(sr, ended)
        receipt = {'schema': SCHEMA, 'status': 'BODY_EXIT_VERIFIED_REQUIRES_COST_SEAL', 'updates': 375,
            'request_sha256': launch_spec['request_sha256'], 'scope_sha256': launch_spec['scope_sha256'],
            'body_ref': response['raw_artifact'], 'independent_replay_ref': replay_ref,
            'supervisor_service_replay': sr, 'supervisor_terminal_suffix': supervisor_suffix,
            'finalizer_header_ref': copy.deepcopy(owner.header_ref), 'expected_finalizer_seal_path': str(owner.seal_path),
            'cost_owner_binding_sha256': digest(owner.header), 'native_qualified': False, 'learning_claim': False}
        reference = owner.write_json('TRAINING_COMPLETION.json', receipt)
        evidence = owner.close_and_seal(); owner.require_closed_evidence(evidence)
        require(evidence['receipt_refs']['TRAINING_COMPLETION.json'] == reference, 'completion paired with actual successful owned cost seal')
        return {'schema': SCHEMA, 'status': 'TRAINING_BODY_EXIT_COSTS_VERIFIED', 'completed': True, 'updates': 375,
                'receipt_ref': reference, 'cost_evidence': evidence, 'native_qualified': False, 'learning_claim': False}
    except BaseException as error:
        primary = error
        if owner is not None:
            try: owner.fail(primary)
            except BaseException as failure: raw._accounting(primary, failure)
        raise
    finally:
        failures = []
        if owned is not None:
            if not terminal_reached:
                try: owned._stop_owned('training_supervisor_failure', primary)
                except BaseException as failure: failures.append(failure)
            try: owned.close()
            except BaseException as failure: failures.append(failure)
        if retained is not None:
            try: retained.close()
            except BaseException as failure: failures.append(failure)
        if failures:
            failure = failures[0]
            for extra in failures[1:]: raw._accounting(failure, extra)
            if primary is not None: raw._accounting(primary, failure); raise primary from failure
            raise failure


def main():
    configuration = entry.install_source_registries()  # default source CLOSED before IO/OS/launch
    spec = supervisor_specification(configuration)
    services._REGISTERED_SERVICE_VERIFIERS |= frozenset((service_metadata_verifier,))
    launch._REGISTERED_LAUNCH_VERIFIERS |= frozenset((launch_verifier,))
    launch._REGISTERED_OBSERVATION_FACTORIES |= frozenset((launch_observation,))
    worker._REGISTERED_METADATA_VERIFIERS |= frozenset((worker_metadata_verifier,))
    permit = services.ControllerServicePermit(configuration['request'], spec, sys.modules[__name__], service_metadata_verifier)
    service = services.NativeControllerServices(configuration['request'], permit, spec)
    budget = worker._Budget({'bounds': spec['scope_adapters']['worker']['bounds']}, service.operation)
    api = budget.run('worker_os_binding', worker._WindowsOS); handle = None
    try:
        handle = budget.run('worker_os_binding', lambda: api.open(os.getpid()))
        facts = budget.run('worker_os_binding', lambda: api.snapshot(handle)); boot = budget.run('worker_os_binding', api.boot)
        parent = {'pid': facts['pid'], 'creation_filetime_100ns': facts['creation_filetime_100ns'], 'boot_identity': boot}
    finally:
        if handle is not None: budget.run('worker_handle_close', lambda: api.close(handle))
    spec = build_launch_spec(configuration, parent)
    service._training_launch_spec = spec
    service._qualification_launch_spec = spec  # original finalizer's generic metadata-owner binding, no Q dispatch
    permit = launch.NativeLaunchPermit(spec, service, launch_verifier, service.operation)
    result = supervise_training(service, spec, permit)
    print(canonical(result).decode('ascii'))
    return 0


if __name__ == '__main__':
    raise SystemExit(importlib.import_module(__package__ + '.ce_training_supervisor').main())
