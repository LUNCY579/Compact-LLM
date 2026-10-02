"""Paid framework-free reduction of retained checkpoint branch evidence.

Logical inherited U1 is distinct from a child's physical CE updates. Parent and
child reservations are reported separately and never refunded or copied into
another history. Arithmetic/state comparisons do not establish native execution.
"""
from __future__ import annotations

import copy

from . import ce_qualification_artifacts as raw, ce_execution_verifier as execution
from . import ce_checkpoint_qualification as checkpoint, ce_qualification_accounting as accounting
from . import ce_checkpoint_worker as worker

SCHEMA = 'CE8-CHECKPOINT-BRANCH-REDUCTION-1'
RESULT = 'CE8-CHECKPOINT-WORKER-RESULT-1'
BRANCHES = ('uninterrupted', 'prefix', 'resumed')
require, digest = raw.require, raw.digest


def _json(reader, reference, expected, role):
    require(reference['bytes'] <= raw.MAX_JSON, 'checkpoint descriptor JSON ceiling')
    decorated = execution._decorated(reader, reference, expected, role)
    blob = b''.join(reader._blocks(decorated, expected, role))
    with reader._overhead('checkpoint_descriptor_parse', work_bytes=32 * len(blob), hash_attempts=2, hash_bytes=4 * len(blob)):
        value = raw._json(blob)
        require(type(value) is dict and raw.canonical(value) + b'\n' == blob, 'canonical checkpoint descriptor')
    return value


def _envelope(reader, reference, expected):
    value = _json(reader, reference, expected, 'checkpoint_branch')
    require(value.get('schema') == raw.SCHEMA and value.get('evidence_kind') == 'checkpoint_branch'
            and all(value.get(key) == item for key, item in expected.items()), 'exact raw checkpoint branch/source/worker envelope')
    payload = value.get('payload')
    require(type(payload) is dict and payload.get('schema') == RESULT, 'actual checkpoint worker result payload')
    return payload


def reduce_negative_controls(state, controls, parameter_inventory=None, binding=None):
    hashes_equal = len({controls[key] for key in ('before_state_sha256', 'after_corruption_state_sha256',
        'after_stale_state_sha256', 'recovered_state_sha256')}) == 1
    failed = [op for op in state.operations.values() if op['kind'] == 'checkpoint_restore' and op['phase'] == 'failed']
    restored = [op for op in state.operations.values() if op['kind'] == 'checkpoint_restore' and op['phase'] == 'returned']
    require(hashes_equal and failed and restored and type(controls.get('corruption_error')) is dict
            and type(controls.get('stale_fence_error')) is dict,
            'negative-control equality needs actual failed and recovery restore records')
    require(controls['corruption_error'] == {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'}
            and controls['stale_fence_error'] == {'type': 'ValueError', 'message': 'exact ancestor nested restore fence'},
            'specific source corruption/stale rejection causes required')
    selected = controls.get('corruption_restore_operation')
    require(type(selected) is dict and set(selected) == {'operation_id', 'operation'}
            and selected['operation_id'] in state.operations
            and selected['operation'] == state.operations[selected['operation_id']],
            'exact retained corruption restore operation required')
    rejected = selected['operation']
    require(rejected['kind'] == 'checkpoint_restore' and rejected['phase'] == 'failed'
            and rejected.get('result', {}).get('error') ==
                {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'},
            'corruption must be actual payload rejection, not unrelated error')
    require(type(rejected.get('started_seq')) is type(rejected.get('returned_seq')) is int
            and 0 <= rejected['started_seq'] < rejected['returned_seq'], 'strict failed restore operation order')
    if binding is not None:
        require(all(rejected.get(key) == binding[value] for key, value in
                    (('worker_id', 'worker_identity_sha256'), ('attempt_id', 'attempt_id'), ('stage', 'stage'))),
                'corruption rejection belongs to exact current child stage/worker')
    stale = controls.get('stale_attempt_record')
    final_progress = state.progress_record()
    require(type(stale) is dict and type(stale.get('progress')) is dict,
            'retained exact stale attempted fence required')
    progress = stale['progress']
    require(type(progress.get('generation')) is int and progress['generation'] < final_progress['generation']
            and type(progress.get('C')) is type(progress.get('U')) is int
            and progress['C'] == progress['U'] == final_progress['U']
            and progress.get('history') == final_progress['history'], 'actual nonfuture stale generation and same history')
    worker.validate_stale_attempt(stale, final_progress)
    require(max(op['returned_seq'] for op in failed) < max(op['returned_seq'] for op in restored),
            'good recovery must return after corruption failure')
    fingerprints = sorted((op for op in state.operations.values()
        if op['kind'] == 'qualification_state_fingerprint' and op['phase'] == 'returned'), key=lambda op: op['returned_seq'])
    require(len(fingerprints) == 4, 'exact four returned negative-control state fingerprints')
    descriptor_hashes = []
    for op in fingerprints:
        descriptor = op['result'].get('state_descriptor')
        require(type(descriptor) is dict and set(descriptor) == {'tensor_hashes', 'specifications', 'Adam_initialized',
                'rng', 'optimizer_groups', 'module_training'}, 'complete retained live state descriptor')
        require(type(descriptor['tensor_hashes']) is dict and descriptor['tensor_hashes']
                and all(raw._hex(value) for value in descriptor['tensor_hashes'].values())
                and set(descriptor['tensor_hashes']) == set(descriptor['specifications']), 'all actual state tensor hashes/specifications')
        require(type(descriptor['rng']) is dict and descriptor['rng'].get('cuda_count') == 1
                and type(descriptor['module_training']) is dict and descriptor['module_training']
                and all(type(mode) is bool for mode in descriptor['module_training'].values()), 'complete RNG/module mode descriptor')
        if parameter_inventory is not None:
            names = {'rng::cpu', 'rng::cuda::0'}
            for entry in parameter_inventory:
                for prefix in ('param::', 'exp_avg::', 'exp_avg_sq::'):
                    name = prefix + entry['name']; names.add(name)
                    require(descriptor['specifications'].get(name) == ['F32', entry['shape']], 'selected complete parameter/Adam shape')
                name = 'step::' + entry['name']; names.add(name)
                require(descriptor['specifications'].get(name) == ['F32', []], 'selected scalar Adam counter')
            require(set(descriptor['specifications']) == names
                    and set(descriptor['Adam_initialized']) == {entry['name'] for entry in parameter_inventory},
                    'full native selected inventory in live state descriptors')
        require(digest(descriptor) == op['result'].get('state_sha256'), 'state descriptor does not derive returned fingerprint')
        descriptor_hashes.append(digest(descriptor))
    require(descriptor_hashes == [controls[key] for key in ('before_state_sha256', 'after_corruption_state_sha256',
        'after_stale_state_sha256', 'recovered_state_sha256')], 'raw control hashes differ from actual returned descriptors')
    require(fingerprints[0]['returned_seq'] < min(op['started_seq'] for op in failed)
            and max(op['returned_seq'] for op in failed) < fingerprints[1]['started_seq']
            and fingerprints[1]['returned_seq'] < fingerprints[2]['started_seq']
            and max(op['returned_seq'] for op in restored) < fingerprints[3]['started_seq'],
            'corruption/stale/recovery descriptor causal order')
    return {'negative_fingerprint_claims_equal': hashes_equal, 'failed_restore_operations': len(failed),
        'returned_restore_operations': len(restored), 'negative_controls_independently_recomputed': True}


def _branch(reader, context, reference, binding, parent_binding):
    payload = _envelope(reader, reference, binding)
    name = payload.get('branch'); require(name in BRANCHES, 'one exact checkpoint branch')
    lineage = payload.get('observed_lineage')
    require(type(lineage) is dict and lineage.get('branch') == name, 'retained observed child lineage required')
    checkpoint.validate_checkpoint_relationship(parent_binding, binding,
        parent_binding=parent_binding, child_lineages={binding['worker_identity_sha256']: lineage})
    child = payload.get('child_context')
    require(type(child) is dict and child.get('request_sha256') == binding['request_sha256']
            and child.get('scope_sha256') == binding['scope_sha256']
            and child.get('static_recipe_sha256') == binding['static_recipe_sha256']
            and child.get('prepared_receipt') == context['prepared_receipt']
            and child['bindings']['source']['code_sha256'] == binding['code_sha256']
            and child['bindings']['anchors'] == context['bindings']['anchors']
            and child['bindings']['scope']['qualifier_execution']['adam_batches']
                == context['bindings']['scope']['qualifier_execution']['adam_batches'], 'child immutable source/recipe/anchor/census/batches')
    refs = payload.get('ledger_refs')
    require(type(refs) is dict and set(refs) == {'header', 'journal', 'seal'}, 'original child ledger/header/stage seal references')
    header = execution._header(child, _json(reader, refs['header'], binding, 'qualification_header'))
    state, workers, _ = execution._replay(reader, refs['journal'], binding, child, header, {binding['worker_identity_sha256']})
    seal = _json(reader, refs['seal'], binding, 'qualification_seal')
    _, pending = execution._seal(seal, state)
    require(not pending and not state.fatal and not state.unknown_wall and not state.worker_open
            and not state.active and not state._ce.active_attempt and not state.requires_restore and not state.restore_pending,
            'closed resolved child ledger/clock/cleanup required')
    prefixes = [entry['payload'] for entry in state.events if entry['kind'] == 'checkpoint_child_prefix']
    require(prefixes == [lineage], 'one exact replayed observed child-prefix event')
    commits = [entry['payload']['result'] for entry in state.events if entry['kind'] == 'commit']
    inherited = 1 if name == 'resumed' else 0
    expected_updates = 1 if name in ('prefix', 'resumed') else 2
    expected_U = 1 if name == 'prefix' else 2
    require(lineage['inherited_U'] == inherited and len(commits) == expected_updates
            and state.progress_record()['C'] == state.progress_record()['U'] == expected_U
            and [row['U_after'] for row in commits] == list(range(inherited + 1, expected_U + 1)),
            'own physical child updates differ from explicit branch plan')
    if inherited:
        restores = [entry for entry in state.events if entry['kind'] == 'restore']
        require(restores and restores[0]['seq'] < next(entry['seq'] for entry in state.events if entry['kind'] == 'commit'),
                'resumed actual restore/fresh fence must precede its own update')
    manifest_ref = payload.get('final_manifest_ref')
    manifest = _json(reader, manifest_ref, binding, 'checkpoint_manifest')
    require(manifest.get('schema') == checkpoint.CHECKPOINT_SCHEMA
            and digest({key: value for key, value in manifest.items() if key != 'manifest_sha256'}) == manifest.get('manifest_sha256')
            and manifest.get('bindings_sha256') == digest(manifest['bindings'])
            and manifest['bindings'].get('qualification_evidence_binding') == binding
            and manifest['bindings'].get('qualification_batches') == context['bindings']['scope']['qualifier_execution']['adam_batches']
            and manifest['progress']['history'] == state.progress_record()['history']
            and manifest['progress']['C'] == manifest['progress']['U'] == expected_U,
            'original final checkpoint differs from replayed child endpoint/history')
    controls_ref = payload.get('negative_controls_ref')
    controls = _json(reader, controls_ref, binding, 'checkpoint_negative_controls')
    require(controls.get('schema') == raw.SCHEMA and controls.get('evidence_kind') == 'checkpoint_negative_controls'
            and all(controls.get(key) == value for key, value in binding.items()), 'negative-control raw source/worker binding')
    controls = controls.get('payload')
    require(type(controls) is dict and all(raw._hex(controls.get(key)) for key in
        ('before_state_sha256', 'after_corruption_state_sha256', 'after_stale_state_sha256', 'recovered_state_sha256')),
        'all retained negative-control state fingerprints')
    negative = reduce_negative_controls(state, controls, context['bindings']['scope']['native_backend']['parameter_inventory'], binding)
    return {'branch': name, 'binding': binding, 'lineage': lineage, 'manifest_ref': manifest_ref,
        'logical_inherited_U': inherited, 'physical_optimizer_updates': len(commits),
        'final_U': expected_U, 'derived_costs': copy.deepcopy(state.cumulative),
        'ledger_head': state._head(), **negative}


def reduce_checkpoint_branches(context, branch_refs, branch_bindings, parent_binding, *, reader, outer_preflight):
    """Recompute original child ledgers and actual U2 checkpoint bytes, never PASS."""
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight), 'explicit paid bounded reader/preflight required')
    require(type(branch_refs) is dict and set(branch_refs) == set(BRANCHES)
            and type(branch_bindings) is dict and set(branch_bindings) == set(BRANCHES), 'all three exact branch outputs required')
    outer_preflight('checkpoint_branch_reduction', {'references': list(branch_refs.values()),
        'reader_bounds': dict(reader.remaining), 'coverage': ['raw child JSON/ledger/state IO', 'hashes', 'reducer/metadata/wall']})
    with reader._overhead('checkpoint_parent_binding', work_bytes=raw.MAX_BINDING_BYTES,
                          hash_attempts=4, hash_bytes=4 * raw.MAX_BINDING_BYTES):
        raw.binding(parent_binding)
        require(parent_binding['stage'] == 'checkpoint_resume', 'exact parent checkpoint stage')
        require(len({value['worker_identity_sha256'] for value in branch_bindings.values()}) == 3
                and parent_binding['worker_identity_sha256'] not in {value['worker_identity_sha256'] for value in branch_bindings.values()},
                'three fresh child workers and separate parent required')
    branches = {name: _branch(reader, context, branch_refs[name], branch_bindings[name], parent_binding) for name in BRANCHES}
    require(all(value['branch'] == name for name, value in branches.items()), 'branch role substitution')
    lineage_map = {value['binding']['worker_identity_sha256']: value['lineage'] for value in branches.values()}
    comparison = checkpoint.compare_checkpoint_states(reader, branches['uninterrupted']['manifest_ref'], branch_bindings['uninterrupted'],
        branches['resumed']['manifest_ref'], branch_bindings['resumed'],
        parameter_inventory=context['bindings']['scope']['native_backend']['parameter_inventory'],
        parent_binding=parent_binding, child_lineages=lineage_map)
    totals = {key: sum(value['derived_costs'][key] for value in branches.values()) for key in accounting.COST_FIELDS}
    return {'schema': SCHEMA, 'status': 'UNQUALIFIED', 'native_qualified': False, 'execution_verified': False,
        'branches': branches, 'child_derived_reservations': totals, 'U2_state_comparison': comparison,
        'parent_costs_imported': False, 'parent_external_ledger_refs': branches['prefix']['lineage']['parent_ledger_refs'],
        'activation_gaps': ['source-owned OS terminal handles and exact launch/parent provenance must be verified',
            'bootstrap child prepayment chains and parent external costs need independent raw replay',
            'source-owned child accounting derivation from shared frozen caps/owner policy must be verified',
            'descriptor tensor hashes require actual source-issued worker execution provenance'],
        'classification': 'raw child ledger and byte-state reduction; no native qualification or learning claim'}
