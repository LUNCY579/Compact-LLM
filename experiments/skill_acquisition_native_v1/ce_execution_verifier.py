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
    bindings = context['bindings']; scope = bindings['scope']; spec = accounting.accounting_specification(context)
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


def _runtime_readback(context, state, payload, binding):
    scope, anchors = context['bindings']['scope'], context['bindings']['anchors']
    observed, identity = payload.get('observations'), payload.get('identity')
    require(type(observed) is dict and type(identity) is dict
            and identity.get('state_fingerprint') == anchors['own_start_sha256']
            and identity.get('weights_sha256') == anchors['base_weights_sha256']
            and type(identity.get('anchor')) is dict
            and identity['anchor'].get('base_weights_sha256') == anchors['base_weights_sha256']
            and identity['anchor'].get('delta_sha256') == scope['native_backend']['warm_delta_ref']['data_sha256']
            and identity['anchor'].get('arm_sha256') == anchors['common_anchor_sha256'],
            'raw runtime loader/anchor identity differs from admitted common warm bytes')
    require(payload.get('inventory') == scope['native_backend']['parameter_inventory']
            and type(payload.get('worker_proof')) is dict
            and payload['worker_proof'].get('worker_identity_sha256') == binding['worker_identity_sha256'],
            'raw runtime exact selected inventory/current worker')
    configurations = [operation for operation in state.operations.values() if operation['stage'] == binding['stage']
        and operation['attempt_id'] == binding['attempt_id'] and operation['phase'] == 'returned']
    require(all(any(operation['kind'] == kind for operation in configurations)
                for kind in ('configured_runtime', 'process_rng_seed', 'model_load', 'inventory_configuration', 'fingerprint')),
            'raw runtime observations lack original returned initialization/load operations')
    q.validate_stage_observations(context, 'runtime_anchor', observed, context['prepared_receipt'])
    return {'recorded_observations': observed, 'loader_identity_and_operation_linkage_consistent': True,
            'execution_verified': False, 'native_qualified': False}


def _delta_readback(context, payload, reference, binding, reader):
    from . import ce_delta_artifacts as delta
    donor = payload.get('donor'); anchors = context['bindings']['anchors']
    inventory = context['bindings']['scope']['native_backend']['parameter_inventory']
    require(type(donor) is dict and donor.get('schema') == 'SKILL-ACQUISITION-DONOR-DELTA-1'
            and donor.get('path') == reference['path'] and donor.get('file_sha256') == reference['sha256']
            and donor.get('bytes') == reference['bytes']
            and donor.get('common_anchor_sha256') == anchors['common_anchor_sha256']
            and donor.get('own_start_sha256') == anchors['own_start_sha256']
            and donor.get('reference') == 'EXACT_REBUILT_COMMON_WARM_ANCHOR',
            'raw saved donor file/reference/common-anchor provenance')
    saved = delta.read_delta(reader, reference, binding, inventory)
    require(all(donor.get(key) == saved[key] for key in ('data_section_sha256', 'state_sha256', 'numel', 'tensors'))
            and donor.get('trainable_names_sha256') == digest(sorted(entry['name'] for entry in inventory))
            and q.finite(donor.get('delta_norm')) and abs(donor['delta_norm'] - saved['delta_norm']) <= 1e-12 + 1e-10 * saved['delta_norm'],
            'raw saved FP32 delta digest/inventory/norm differs from independent stream')
    require(type(payload.get('observations')) is dict, 'retained roundtrip report required')
    q.validate_stage_observations(context, 'delta_roundtrip', payload['observations'], context['prepared_receipt'])
    require(payload['observations']['delta_file_sha256'] == reference['sha256']
            and payload['observations']['delta_data_sha256'] == saved['data_section_sha256']
            and payload['observations']['final_master_sha256'] == donor['final_master_sha256']
            and payload['observations']['rebuilt_endpoint_sha256'] == donor['deployed_arm_sha256'],
            'reported roundtrip file/endpoint identities differ from retained donor')
    return {'saved_delta': saved, 'recorded_roundtrip_observations': payload['observations'],
            'roundtrip_independently_rebuilt': False, 'execution_verified': False, 'native_qualified': False}


def _retained_prefix(context, original, snapshot):
    """Replay a retained head of the already verified main journal."""
    require(type(snapshot) is dict and raw._hex(snapshot.get('ledger_head'))
            and raw._int(snapshot.get('events'), 1), 'exact retained main journal prefix head/count')
    count = snapshot['events']
    require(count <= len(original.events) and original.events[count - 1]['sha256'] == snapshot['ledger_head'],
            'snapshot is not an actual ancestor of the verified main journal')
    state = _shadow(context, original.header)
    for entry in original.events[:count]:
        if entry['kind'] == 'worker' and not state.proof:
            state.proof = copy.deepcopy(entry['payload']['proof'])
        state._apply(entry['kind'], entry['payload'], replay=True)
        state.events.append(entry)
    _seal(snapshot, state)
    return state


def _adam_readback(context, state, payload, binding):
    batches = context['bindings']['scope']['qualifier_execution']['adam_batches']
    commits = [entry['payload']['result'] for entry in state.events if entry['kind'] == 'commit'
               and entry['payload']['result'].get('optimizer_operation', {}).get('operation_id') in state.operations
               and state.operations[entry['payload']['result']['optimizer_operation']['operation_id']]['attempt_id'] == binding['attempt_id']]
    require(type(payload) is dict and payload.get('boundaries') == commits and len(commits) == len(batches) == 2,
            'exact two retained qualification Adam commits, not supplied optimizer flags')
    orders = []
    for index, result in enumerate(commits):
        require([row['id'] for row in result['rows']] == batches[index]
                and result['U_before'] == index and result['U_after'] == index + 1,
                'raw Adam batches/update counters differ from declared qualification')
        clip = state.operation_evidence(result['clipping_operation']['operation_id'])
        optimizer = state.operation_evidence(result['optimizer_operation']['operation_id'])
        require(clip['kind'] == 'gradient_clipping' and optimizer['kind'] == 'optimizer'
                and clip['returned_seq'] < optimizer['started_seq'], 'actual returned clipping precedes retained Adam start')
        orders.append({'clipping': clip, 'optimizer': optimizer})
    require(payload.get('operation_order') == orders and type(payload.get('progress')) is dict
            and payload['progress']['C'] == payload['progress']['U'] == 2
            and payload['progress']['history'] == state.progress_record()['history'],
            'raw Adam operation order/history differs from original replay')
    observed = {'actual_optimizer_steps': 2, 'applied_U': [1, 2],
        'finite_confined_gradients': all(result['gradient_validation_operation']['finite_confined'] is True for result in commits),
        'frozen_state_unchanged': len({result['frozen_state_sha256'] for result in commits}) == 1,
        'clip_before_adam': True, 'clip_max_norm': context['bindings']['recipe']['recipe']['gradient_clipping']['max_norm']}
    q.validate_stage_observations(context, 'adam_clipping', observed, context['prepared_receipt'])
    return observed


def _resource_readback(context, state, payload, binding, *, reader, reference_bytes):
    from . import ce_resource_qualification as resource
    spec = context['bindings']['scope']['resource_measurement']
    require(type(payload) is dict and payload.get('required_barriers') == spec['required_barriers']
            and payload.get('measured_operation_kinds') == spec['measured_operation_kinds']
            and payload.get('limits') == spec['limits'], 'raw resource scope differs from frozen measurement policy')
    with reader._overhead('resource_original_journal_prefix', work_bytes=32 * reference_bytes + 32 * MAX_HEADER,
                          hash_attempts=MAX_EVENTS, hash_bytes=8 * reference_bytes + MAX_HEADER):
        prefix = _retained_prefix(context, state, payload.get('accounting_snapshot'))
        require(payload.get('raw_operations') == prefix.operations and payload.get('worker_proof') == prefix._current_proof,
                'resource operations/worker differ from independently replayed original head')
        proof = prefix._current_proof
        require(proof['worker_identity_sha256'] == binding['worker_identity_sha256'], 'resource prefix current worker mismatch')
        samples = payload.get('resource_samples')
        require(type(samples) is list and samples, 'actual retained barrier samples required')
        recorded = [operation['result'].get('resource_observation') for operation in prefix.operations.values()
                    if operation['kind'] in ('resource_probe', 'resource_cuda_probe') and operation['phase'] == 'returned']
        require(all(sample in recorded or {key: value for key, value in sample.items() if key != 'stage'} in recorded
                    for sample in samples), 'resource sample lacks its original returned probe operation')
        observed = resource.reduce_samples(samples, spec['required_barriers'], worker_pid=proof['os_identity']['pid'],
            gpu_uuid=context['bindings']['scope']['qualifier_resources']['gpu_uuid'], limits=spec['limits'])
        observed.update(resource.reduce_operations(prefix.operations, spec['measured_operation_kinds'],
            worker_identity_sha256=proof['worker_identity_sha256']))
    require(payload.get('observations') == observed, 'supplied resource observations differ from raw recomputation')
    q.validate_stage_observations(context, 'resource_io_measurement', observed, context['prepared_receipt'])
    return observed


def _checkpoint_prefix(context, original, refs, binding, reader):
    require(type(refs) is dict and set(refs) == {'header', 'journal', 'seal'}, 'all retained parent checkpoint prefix refs')
    header = _header(context, _read_json(reader, refs['header'], binding, 'qualification_header'))
    prefix, workers, _ = _replay(reader, refs['journal'], binding, context, header, {binding['worker_identity_sha256']})
    snapshot = _read_json(reader, refs['seal'], binding, 'qualification_seal')
    _seal(snapshot, prefix)
    require(prefix.header == original.header and prefix.events == original.events[:len(prefix.events)],
            'retained checkpoint parent prefix is not the original verified journal')
    require(not prefix.fatal and not prefix.unknown_wall and prefix.worker_open and prefix.active is not None
            and prefix.active['stage'] == 'checkpoint_resume' and prefix.active['attempt_id'] == binding['attempt_id']
            and prefix.progress_record()['C'] == prefix.progress_record()['U'] == 2
            and not prefix.requires_restore and not prefix.restore_pending,
            'exact live resolved parent U2 checkpoint stage in retained prefix')
    require(all(operation.get('lifetime_cleanup') for operation in prefix.operations.values() if operation['phase'] == 'started'),
            'unexpected pending parent operation in checkpoint prefix')
    return prefix, snapshot


def _checkpoint_readback(context, state, payload, binding, *, reader, outer_preflight):
    from . import ce_checkpoint_reduction as reduction, ce_checkpoint_qualification as checkpoint
    require(type(payload) is dict and payload.get('schema') == 'CE8-CHECKPOINT-BRANCHES-BUNDLE-1'
            and payload.get('parent_binding') == binding
            and payload.get('native_qualified') is False and payload.get('execution_verified') is False,
            'exact unqualified parent-bound raw checkpoint bundle')
    initial, _ = _checkpoint_prefix(context, state, payload.get('parent_ledger_refs'), binding, reader)
    post, post_snapshot = _checkpoint_prefix(context, state, payload.get('parent_post_restore_ledger_refs'), binding, reader)
    require(initial.events == post.events[:len(initial.events)]
            and payload.get('parent_post_restore_snapshot_ref') == payload['parent_post_restore_ledger_refs']['seal'],
            'post-restore parent prefix must retain exact original ancestry')
    record = payload.get('parent_restore_record')
    matches = [operation for operation in post.operations.values() if operation['kind'] == 'checkpoint_restore'
        and operation['phase'] == 'returned' and operation.get('result', {}).get('manifest_sha256') == (record or {}).get('manifest_sha256')
        and operation.get('worker_id') == binding['worker_identity_sha256'] and operation.get('attempt_id') == binding['attempt_id']]
    require(len(matches) == 1, 'one actual returned current-parent restore operation')
    bound_operation = dict(copy.deepcopy(matches[0]), source_worker_binding=copy.deepcopy(binding))
    checkpoint.validate_restore_fence(record, initial.progress_record(), bound_operation, current_binding=binding)
    require(post.progress_record() == record['progress'] and post_snapshot['progress'] == record['progress'],
            'returned parent restore fence was not installed in original ledger')
    parent_manifest = checkpoint._read_manifest(reader, payload['parent_manifest_ref'], binding)
    require(record['manifest_sha256'] == parent_manifest['manifest_sha256'], 'parent restored a different saved endpoint')
    reduced = reduction.reduce_checkpoint_branches(context, payload['branch_refs'], payload['branch_bindings'], binding,
        reader=reader, outer_preflight=outer_preflight)
    child_lineages = {child['binding']['worker_identity_sha256']: child['lineage'] for child in reduced['branches'].values()}
    parent_comparison = checkpoint.compare_checkpoint_states(reader, payload['parent_manifest_ref'], binding,
        reduced['branches']['uninterrupted']['manifest_ref'], reduced['branches']['uninterrupted']['binding'],
        parameter_inventory=context['bindings']['scope']['native_backend']['parameter_inventory'],
        parent_binding=binding, child_lineages=child_lineages)
    terminals = payload.get('child_terminal_proofs')
    require(type(terminals) is dict and set(terminals) == set(reduction.BRANCHES), 'all original child terminal records required')
    for branch, terminal in terminals.items():
        child = reduced['branches'][branch]
        require(type(terminal) is dict and terminal.get('worker_identity_sha256') == child['binding']['worker_identity_sha256']
                and terminal.get('request_sha256') == binding['request_sha256'] and terminal.get('scope_sha256') == binding['scope_sha256']
                and terminal.get('status_file_used') is False and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0,
                'child terminal metadata must match its exact raw branch; actual OS handles remain required')
    saved = [operation for operation in initial.operations.values() if operation['kind'] == 'checkpoint_save' and operation['phase'] == 'returned']
    observed = {'checkpoint_saves': len(saved) + sum(child['checkpoint_save_operations'] for child in reduced['branches'].values()),
        'checkpoint_restores': len(matches) + sum(child['returned_restore_operations'] for child in reduced['branches'].values()),
        **{key: value and parent_comparison[key] for key, value in reduced['U2_state_comparison'].items() if key.endswith('_bitwise')},
        'fresh_fence_installed': True, 'stale_work_rejected': all(child['negative_controls_independently_recomputed'] for child in reduced['branches'].values()),
        'corruption_fails_closed': all(child['failed_restore_operations'] >= 1 for child in reduced['branches'].values()),
        'failed_unknown_costs_retained': True}
    return {'observations': observed, 'raw_branches': reduced, 'parent_state_comparison': parent_comparison,
            'parent_derived_reservations': post.cumulative,
            'child_terminal_metadata_consistent': True, 'execution_verified': False, 'native_qualified': False}


def _reductions(context, state, attempt_workers, reader, child_bindings, child_lineages, outer_preflight=None):
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
                        and ref.get('binding_sha256') == digest(child), 'exact separate checkpoint child evidence binding')
                from .ce_checkpoint_qualification import validate_checkpoint_relationship
                validate_checkpoint_relationship(expected, child, parent_binding=expected, child_lineages=child_lineages)
                gaps.append('checkpoint child requires its own registered OS handle and verified parent relationship')
            else:
                require(ref.get('binding_sha256') == digest(expected), 'unrelated artifact stage/source/worker/attempt binding')
            by_role[ref['role']] = ref
        stage = attempt['stage']
        if stage == 'runtime_anchor':
            require('runtime_anchor' in by_role, 'raw runtime/anchor artifact required')
            payload = reader._envelope(by_role['runtime_anchor'], expected, 'runtime_anchor', 'runtime_anchor')
            with reader._overhead('runtime_identity_and_operation_readback', work_bytes=32 * by_role['runtime_anchor']['bytes'],
                                  hash_attempts=16, hash_bytes=32 * by_role['runtime_anchor']['bytes']):
                reductions[stage] = _runtime_readback(context, state, payload, expected)
            gaps.append('runtime loader identity and returned operations require actual source-issued model execution provenance')
        elif stage == 'longest_masks':
            require('masks' in by_role, 'raw masks artifact required')
            reductions[stage] = reader.read_masks(by_role['masks'], expected, witnesses)['observations']
        elif stage == 'likelihood_gradient_parity':
            require('likelihood' in by_role and 'gradient_manifest' in by_role, 'both raw likelihood and gradient artifacts required')
            values = reader.read_likelihood(by_role['likelihood'], expected, witnesses)['observations']
            values.update(reader.read_gradient(by_role['gradient_manifest'], expected, gradient,
                context['bindings']['scope']['native_backend']['parameter_inventory'],
                start_anchor_sha256=context['bindings']['anchors']['own_start_sha256'])['observations'])
            reductions[stage] = values
        elif stage == 'adam_clipping':
            require('adam_clipping' in by_role, 'raw Adam operation artifact required')
            payload = reader._envelope(by_role['adam_clipping'], expected, 'adam_clipping', 'adam_clipping')
            with reader._overhead('adam_original_operation_readback', work_bytes=32 * by_role['adam_clipping']['bytes']):
                reductions[stage] = _adam_readback(context, state, payload, expected)
        elif stage == 'delta_roundtrip':
            require('delta_roundtrip' in by_role and 'qualification_delta' in by_role, 'raw report and saved delta both required')
            payload = reader._envelope(by_role['delta_roundtrip'], expected, 'delta_roundtrip', 'delta_roundtrip')
            with reader._overhead('delta_report_linkage', work_bytes=32 * by_role['delta_roundtrip']['bytes'],
                                  hash_attempts=8, hash_bytes=32 * by_role['delta_roundtrip']['bytes']):
                reductions[stage] = _delta_readback(context, payload, by_role['qualification_delta'], expected, reader)
            gaps.append('saved delta bytes/digests/norm are checked; full common-anchor rebuild/master residual and native execution still require verification')
        elif stage == 'resource_io_measurement':
            require('resource_io' in by_role, 'raw barrier/operation resource artifact required')
            payload = reader._envelope(by_role['resource_io'], expected, 'resource_io', 'resource_io')
            reductions[stage] = _resource_readback(context, state, payload, expected, reader=reader,
                reference_bytes=by_role['resource_io']['bytes'])
            gaps.append('resource measurements retain their explicit IO subset; complete native unit prices and actual probe execution require verification')
        elif stage == 'checkpoint_resume' and 'checkpoint_branches' in by_role:
            require(callable(outer_preflight), 'independently priced checkpoint bundle reduction scope')
            payload = reader._envelope(by_role['checkpoint_branches'], expected, 'checkpoint_branches', 'checkpoint_branches')
            reductions[stage] = _checkpoint_readback(context, state, payload, expected, reader=reader, outer_preflight=outer_preflight)
            gaps.append('checkpoint raw equality and terminal metadata need actual retained parent/child OS handles and outer prepayment verification')
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
                         worker_handles=None, child_bindings=None, child_lineages=None, bootstrap_evidence=None):
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
    child_lineages = {} if child_lineages is None else child_lineages
    require(type(child_bindings) is dict and type(child_lineages) is dict, 'explicit child artifact binding/observed lineage maps')
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
        reductions, gaps = _reductions(context, state, attempt_workers, reader, child_bindings, child_lineages, outer_preflight)
    gaps += _workers(worker_handles, workers, context)
    bootstrap_reductions = []
    if bootstrap_evidence is not None:
        from . import ce_bootstrap_verifier as bootstrap
        require(type(bootstrap_evidence) is list and len(bootstrap_evidence) <= 16,
                'bounded exact bootstrap evidence scopes')
        for evidence in bootstrap_evidence:
            require(type(evidence) is dict and set(evidence) == {'context', 'refs', 'expected_workers', 'expected_owners', 'outer_evidence'},
                    'explicit original bootstrap context/ref/contract evidence')
            original = evidence['context']
            require(type(original) is dict and all(original.get(key) == context.get(key)
                        for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256', 'arm'))
                    and original['bindings']['source'] == context['bindings']['source'],
                    'bootstrap scope/source differs from qualification study')
            refs = evidence['refs']
            require(type(refs) is dict and set(refs) == {'header', 'journal', 'seal'}, 'complete original bootstrap references')
            result = bootstrap.verify_bootstrap(original, refs['header'], refs['journal'], refs['seal'],
                reader=reader, outer_preflight=outer_preflight, expected_workers=evidence['expected_workers'],
                expected_owners=evidence['expected_owners'], outer_evidence=evidence['outer_evidence'])
            bootstrap_reductions.append(result); gaps += result['activation_gaps']
    gaps += ['outer bootstrap/verification prepayment and full clock chain require independent registered-service verification',
             'this source does not activate a native execution/qualification verifier registry']
    return {'schema': SCHEMA, 'status': 'UNQUALIFIED', 'native_qualified': False, 'execution_verified': False,
        'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'ledger_head': state._head(), 'events': len(state.events), 'derived_totals': state.cumulative,
        'derived_by_stage': state.by_stage, 'derived_attempts': state.attempts, 'pending_operations': pending,
        'metadata_cost_seal_consistent': True, 'all_stages_complete_by_replay': complete,
        'raw_numeric_reductions': reductions, 'bootstrap_raw_reductions': bootstrap_reductions, 'activation_gaps': gaps,
        'classification': 'bounded retained arithmetic and linkage replay; no native PASS or learning claim'}
