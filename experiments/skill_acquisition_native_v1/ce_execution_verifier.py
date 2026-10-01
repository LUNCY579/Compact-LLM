"""Bounded retained-ledger replay and raw numeric verification, never native PASS.

Accounting headers, worker dictionaries, stage status strings and supplied
totals are evidence to check, not execution authorities. Actual OS execution
requires source-issued ce_worker_evidence handles and a registered service.
This module supplies neither registration nor qualification activation.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import ce_qualification_artifacts as raw, ce_qualification as q
from . import ce_qualification_accounting as accounting

SCHEMA = 'CE8-EXECUTION-EVIDENCE-REPLAY-1'
MAX_HEADER = 4 * 1024 ** 2
MAX_LINE = 4 * 1024 ** 2
MAX_LEDGER = 256 * 1024 ** 2
MAX_EVENTS = 200000
require, canonical, digest = raw.require, raw.canonical, raw.digest


def _reference(value):
    require(type(value) is dict and {'path', 'sha256', 'bytes'} <= set(value)
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and raw._hex(value['sha256']) and raw._int(value['bytes'], 1), 'retained bounded file reference')
    return value


def _decorated(reader, reference, verifier_binding, role):
    _reference(reference)
    with reader._overhead('verifier_reference_binding', work_bytes=raw.MAX_BINDING_BYTES,
                          hash_attempts=1, hash_bytes=raw.MAX_BINDING_BYTES):
        raw.binding(verifier_binding)
        hashed = digest(verifier_binding)
    return {key: reference[key] for key in ('path', 'sha256', 'bytes')} | {'role': role, 'binding_sha256': hashed}


def _read_json(reader, reference, verifier_binding, role):
    require(reference['bytes'] <= MAX_HEADER, 'bounded replay JSON header/seal')
    wrapped = _decorated(reader, reference, verifier_binding, role)
    blob = b''.join(reader._blocks(wrapped, verifier_binding, role))
    with reader._overhead('replay_JSON_parse', work_bytes=16 * len(blob)):
        value = raw._json(blob)
        require(type(value) is dict and canonical(value) + b'\n' == blob, 'canonical retained header/seal bytes')
    return value


def _header(context, observed):
    bindings = context['bindings']; scope = bindings['scope']; spec = scope['qualifier_accounting']
    expected = {'schema': accounting.SCHEMA, 'fixture': context.get('mode') == 'synthetic_fixture',
        'native_qualified': False, 'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'arm': context['arm'], 'static_recipe_sha256': context['static_recipe_sha256'], 'specification': spec,
        'prepared_receipt_sha256': context['prepared_receipt']['receipt_sha256'],
        'recipe': bindings['recipe']['recipe'], 'anchors': bindings['anchors'],
        'source': bindings['source'], 'runtime': bindings['runtime'],
        'bootstrap_owner': {'accounting_spec_sha256': digest(spec),
                            'prepared_receipt_sha256': context['prepared_receipt']['receipt_sha256']},
        'wall_rule': 'source-owned time.monotonic intervals with same-boot worker/end evidence'}
    require(observed == expected and digest(expected['recipe']) == context['static_recipe_sha256'],
            'original header scope/source/recipe/runtime/anchor/schema binding')
    require(digest({k: v for k, v in context['prepared_receipt'].items() if k != 'receipt_sha256'})
            == observed['prepared_receipt_sha256'] == spec['ingestion_receipt_sha256'], 'exact prepared census self binding')
    require(set(spec['stage_caps']) == set(q.STAGES), 'all replay stage caps')
    q._costs(spec['global_caps'])
    for caps in spec['stage_caps'].values(): q._costs(caps, stage=True)
    for profile in spec['operation_profiles'].values(): accounting._profile(profile)
    require(set(spec['operation_profiles']) == accounting.KINDS, 'exact immutable operation profile set')
    return expected


def _shadow(context, header):
    """Pure semantic state; no constructor, permit, framework or publication."""
    return accounting.replay_state(context, header, context['prepared_receipt'])


def _worker(proof, context, allowed_workers):
    require(type(proof) is dict and proof.get('schema') == accounting.PROOF
            and raw._hex(proof.get('worker_identity_sha256')) and raw._hex(proof.get('boot_identity_sha256')),
            'retained worker proof shape')
    require(proof['worker_identity_sha256'] in allowed_workers, 'foreign worker outside declared handle/binding set')
    if 'code_sha256' in proof:
        require(proof['code_sha256'] == context['bindings']['source']['code_sha256'], 'worker source code mismatch')
    sources = proof.get('sources', {}).get('source_files') if type(proof.get('sources')) is dict else None
    if sources is not None:
        require(type(sources) is dict and all(type(value) is dict and raw._hex(value.get('sha256'))
                and raw._int(value.get('bytes'), 1) and type(value.get('path')) is str for value in sources.values()),
                'worker retained source-file references')
        require({key: value['sha256'] for key, value in sources.items()} == context['bindings']['source']['code_sha256'],
                'worker source code mismatch')
    else:
        require(context.get('mode') == 'synthetic_fixture', 'native worker requires complete actual source-file observation map')


def _replay(reader, reference, verifier_binding, context, header, allowed_workers):
    require(reference['bytes'] <= MAX_LEDGER, 'ledger replay byte ceiling')
    wrapped = _decorated(reader, reference, verifier_binding, 'qualification_journal')
    state = _shadow(context, header); workers, attempt_workers = {}, {}
    header_blob = canonical(header) + b'\n'
    initial_header_cost = {'metadata_bytes_upper': len(header_blob), 'io_write_bytes_upper': len(header_blob),
        'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(header))}
    with reader._overhead('ledger_chain_and_semantic_replay', work_bytes=32 * reference['bytes'],
                         hash_attempts=MAX_EVENTS + 1, hash_bytes=4 * reference['bytes'] + MAX_HEADER):
        stream = reader._blocks(wrapped, verifier_binding, 'qualification_journal')
        buffer, setup_count, primary = b'', 0, None
        try:
            for chunk in stream:
                buffer += chunk
                while b'\n' in buffer:
                    end = buffer.index(b'\n') + 1; line, buffer = buffer[:end], buffer[end:]
                    require(len(line) <= MAX_LINE and len(state.events) < MAX_EVENTS, 'bounded complete ledger event')
                    entry = raw._json(line)
                    require(type(entry) is dict and set(entry) == {'seq', 'previous', 'binding_sha256', 'kind', 'payload', 'sha256'}, 'original ledger row schema')
                    body = {key: value for key, value in entry.items() if key != 'sha256'}
                    require(type(body['seq']) is int and body['seq'] == len(state.events)
                            and body['previous'] == state._head() and body['binding_sha256'] == state.binding_sha256
                            and digest(body) == entry['sha256'] and canonical(entry) + b'\n' == line,
                            'ledger original hash/body/order/binding chain')
                    kind, payload = body['kind'], body['payload']
                    require(type(payload) is dict, 'retained event payload object')
                    if kind == 'setup':
                        require(payload.get('profile') == state.spec['setup_bounds'], 'immutable setup profile; no refund/repricing')
                        require(payload.get('header_costs') == (initial_header_cost if setup_count == 0 else {}), 'exact original header charges retained')
                        setup_count += 1
                    elif kind == 'worker':
                        _worker(payload.get('proof'), context, allowed_workers)
                        workers[payload['proof']['worker_identity_sha256']] = payload['proof']
                        if not state.proof: state.proof = copy.deepcopy(payload['proof'])
                    elif kind == 'begin':
                        require(state._current_proof and state.worker_open, 'worker evidence precedes stage')
                        attempt_workers[payload['attempt_id']] = state._current_proof['worker_identity_sha256']
                    state._apply(kind, payload, replay=True)
                    state.events.append(entry)
                require(len(buffer) <= MAX_LINE, 'ledger partial/oversized tail')
            require(not buffer and setup_count >= 1, 'partial ledger tail or missing original setup; no repair')
        except BaseException as error:
            primary = error
            raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None:
                    raw._accounting(primary, error); raise primary from error
                raise
    return state, workers, attempt_workers


def _seal(observed, state):
    require(observed.get('schema') == accounting.SCHEMA
            and observed.get('fixture') is state.header['fixture']
            and observed.get('native_qualified') is False and observed.get('training_authorized') is False
            and observed.get('binding_sha256') == state.binding_sha256
            and observed.get('ledger_head') == state._head() and observed.get('events') == len(state.events),
            'seal header/chain/sequence binding')
    require(observed.get('cumulative') == state.cumulative and observed.get('stages') == state.by_stage
            and observed.get('reducer_reserved_bytes') == state.reducer_reserved_bytes
            and observed.get('reducer_by_stage') == state.reducer_by_stage
            and observed.get('attempts') == state.attempts, 'seal totals/refunds differ from derived retained reservations')
    require(observed.get('progress') == state.progress_record()
            and observed.get('fatal') is state.fatal and observed.get('unknown_wall') is state.unknown_wall
            and observed.get('worker_proof') == state.proof, 'seal CE progress/worker/failure state differs from replay')
    require(observed.get('actual_wall_seconds') == state.actual_wall_seconds
            and observed.get('last_monotonic') == state.last_clock and observed.get('worker_open') is state.worker_open
            and observed.get('complete_wall_accounting') is (not state.worker_open and not state.unknown_wall),
            'seal actual wall/clock/worker interval differs from replay')
    pending = [copy.deepcopy(op) for op in state.operations.values() if op['phase'] == 'started']
    require(observed.get('pending_operations') == pending, 'seal pending/cleanup state differs from replay')
    complete = (state.active is None and not state.worker_open and not state.unknown_wall and not state.fatal
        and not state._ce.active_attempt and not state.requires_restore and not state.restore_pending and not pending
        and state.external_predecessors | {a['stage'] for a in state.attempts if a['status'] == 'RETURNED'} == set(q.STAGES))
    # A seal must not promote incomplete/failed/unknown/lifetime-cleanup work.
    require(not (observed.get('completed', False) or observed.get('status') == 'FIXTURE_SEALED_UNQUALIFIED') or complete,
            'seal claims completion with incomplete/pending work')
    return complete, pending


def _binding_for(context, attempt, worker):
    return {'stage': attempt['stage'], 'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'static_recipe_sha256': context['static_recipe_sha256'], 'code_sha256': context['bindings']['source']['code_sha256'],
        'worker_identity_sha256': worker, 'attempt_id': attempt['attempt_id']}


def _reductions(context, state, attempt_workers, reader, child_bindings):
    reductions, gaps = {}, []
    receipt = context['prepared_receipt']; rows = {row['id']: row for row in receipt['prepared_row_bindings']}
    execution = context['bindings']['scope']['qualifier_execution']
    witnesses = [rows[rid] for rid in execution['all_witness_ids']]
    gradient = [rows[rid] for rid in execution['gradient_batch_ids']]
    for attempt in state.attempts:
        if attempt['status'] != 'RETURNED': continue
        expected = _binding_for(context, attempt, attempt_workers[attempt['attempt_id']])
        artifacts = attempt['evidence']['raw_refs']
        by_role = {}
        for ref in artifacts:
            require(type(ref) is dict and type(ref.get('role')) is str, 'typed stage raw evidence references required')
            require(ref['role'] not in by_role, 'duplicate stage artifact role')
            if ref['path'] in child_bindings:
                child = child_bindings[ref['path']]; raw.binding(child)
                require(child['stage'] == 'checkpoint_resume' and attempt['stage'] == 'checkpoint_resume'
                        and child['worker_identity_sha256'] != expected['worker_identity_sha256']
                        and all(child[key] == expected[key] for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256', 'code_sha256'))
                        and ref.get('binding_sha256') == digest(child), 'exact separate checkpoint child evidence binding')
                gaps.append('checkpoint child requires its own registered OS handle and verified parent relationship')
            else:
                require(ref.get('binding_sha256') == digest(expected), 'unrelated artifact stage/source/worker/attempt binding')
            by_role[ref['role']] = ref
        stage = attempt['stage']
        if stage == 'longest_masks':
            require('masks' in by_role, 'raw masks artifact required')
            reductions[stage] = reader.read_masks(by_role['masks'], expected, witnesses)['observations']
        elif stage == 'likelihood_gradient_parity':
            require('likelihood' in by_role and 'gradient_manifest' in by_role, 'both raw likelihood and gradient artifacts required')
            values = reader.read_likelihood(by_role['likelihood'], expected, witnesses)['observations']
            values.update(reader.read_gradient(by_role['gradient_manifest'], expected, gradient,
                context['bindings']['scope']['native_backend']['parameter_inventory'],
                start_anchor_sha256=context['bindings']['anchors']['own_start_sha256'])['observations'])
            reductions[stage] = values
        else:
            gaps.append(stage + ': dedicated raw runtime/state/CP/resource verification is still required; stage Boolean observations ignored')
    return reductions, gaps


def _workers(handles, workers, context):
    if handles is None: return ['source-issued actual worker execution handles/service are missing']
    require(type(handles) is dict and all(value is not None and not isinstance(value, (dict, list, str, int, float, bool)) for value in handles.values()),
            'worker execution requires source-issued handles, not input dictionaries/flags')
    try:
        from .ce_worker_evidence import verify_worker_handle
    except ImportError:
        return ['source-owned worker evidence implementation is unavailable']
    gaps = []
    for identity in workers:
        require(identity in handles, 'every retained parent/child worker needs its own OS handle')
        expected = {'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'static_recipe_sha256': context['static_recipe_sha256'], 'code_sha256': context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': identity}
        try: verify_worker_handle(handles[identity], expected, require_native=True, terminal=True)
        except Exception as error: gaps.append('registered actual worker verification refused: ' + type(error).__name__)
    return gaps


def verify_qualification(context, header_ref, journal_ref, seal_ref, *, reader, outer_preflight,
                         worker_handles=None, child_bindings=None):
    """Replay retained evidence beneath an explicit independently priced outer scope.

    The preflight callback reserves verifier IO/hash/reducer/metadata/time before
    any read or setup hashing. Its return value is never native authority.
    Registration/complete wall and worker execution verification remain separate.
    """
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight), 'bounded reader and outer preflight required')
    for ref in (header_ref, journal_ref, seal_ref): _reference(ref)
    require(type(context) is dict and type(context.get('verification_binding')) is dict, 'explicit verifier identity/binding')
    require(header_ref['bytes'] <= MAX_HEADER and seal_ref['bytes'] <= MAX_HEADER and journal_ref['bytes'] <= MAX_LEDGER,
            'finite verifier input byte ceilings')
    outer_preflight('qualification_execution_verifier', {
        'references': [header_ref, journal_ref, seal_ref], 'reader_bounds': dict(reader.remaining),
        'max_events': MAX_EVENTS, 'max_line_bytes': MAX_LINE,
        'required_coverage': ['IO', 'file and arithmetic hashes', 'JSON/reducer work', 'metadata', 'elapsed and tick wall']})
    verifier = context['verification_binding']; child_bindings = {} if child_bindings is None else child_bindings
    require(type(child_bindings) is dict, 'explicit child artifact binding map')
    # Upper setup covers header equality/digests and shadow initialization; it
    # does not impersonate an operation-price measurement or admission service.
    with reader._overhead('verifier_header_scope_setup', work_bytes=32 * MAX_HEADER,
                          hash_attempts=16, hash_bytes=32 * MAX_HEADER):
        header = _header(context, _read_json(reader, header_ref, verifier, 'qualification_header'))
    allowed = set(context.get('allowed_worker_identities', [])) | {b['worker_identity_sha256'] for b in child_bindings.values()}
    require(allowed and all(raw._hex(value) for value in allowed), 'explicit retained worker identity allowlist')
    state, workers, attempt_workers = _replay(reader, journal_ref, verifier, context, header, allowed)
    seal = _read_json(reader, seal_ref, verifier, 'qualification_seal')
    with reader._overhead('seal_and_stage_artifact_linkage', work_bytes=32 * seal_ref['bytes'] + 32 * journal_ref['bytes'],
                          hash_attempts=MAX_EVENTS, hash_bytes=8 * journal_ref['bytes']):
        complete, pending = _seal(seal, state)
        reductions, gaps = _reductions(context, state, attempt_workers, reader, child_bindings)
    gaps += _workers(worker_handles, workers, context)
    gaps += ['outer bootstrap/verification prepayment and full clock chain require independent registered-service verification',
             'this source does not activate a native execution/qualification verifier registry']
    return {'schema': SCHEMA, 'status': 'UNQUALIFIED', 'native_qualified': False, 'execution_verified': False,
        'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'ledger_head': state._head(), 'events': len(state.events), 'derived_totals': state.cumulative,
        'derived_by_stage': state.by_stage, 'derived_attempts': state.attempts, 'pending_operations': pending,
        'metadata_cost_seal_consistent': True, 'all_stages_complete_by_replay': complete,
        'raw_numeric_reductions': reductions, 'activation_gaps': gaps,
        'classification': 'bounded retained arithmetic and linkage replay; no native PASS or learning claim'}
