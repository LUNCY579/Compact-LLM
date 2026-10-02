"""Independent post-exit qualification replay beneath a permanent cost owner.

The preclose result is a one-use source object, never a JSON permission. Raw
stage booleans, saved digests and earlier UNQUALIFIED status labels cannot
establish qualification. Every numerical stage and retained cost chain is
reduced again before the final receipt can be composed.
"""
from __future__ import annotations

import copy

from . import ce_qualification_artifacts as raw, ce_execution_verifier as execution
from . import ce_qualification as q, ce_qualification_accounting as accounting
from . import ce_bootstrap_accounting as boot, ce_bootstrap_verifier as bootstrap
from . import ce_service_bootstrap_verifier as service_replay
from . import ce_checkpoint_reduction as checkpoint, ce_worker_evidence as worker
from . import ce_native_services as services, ce_checkpoint_local_issuer as local

SCHEMA = 'CE8-QUALIFICATION-INDEPENDENT-FINALIZATION-1'
_SEAL = object()
_ISSUED = {}
_CANDIDATES = {}
require, digest = raw.require, raw.digest
CLAIM_SCOPE = {'native_numerical_mechanics': True, 'native_checkpoint_and_delta_bytes': True,
    'native_resource_fit': True, 'cost_basis': 'permanent finite hard upper reservations',
    'measured_operation_kinds': ['resource_probe', 'resource_cuda_probe'],
    'measured_io_scope': 'tracked Python file IO and registered resource probes', 'unit_prices_measured': False,
    'full_model_checkpoint_os_unit_prices': 'NOT_MEASURED', 'production_training_authorized': False}
EXECUTION_GAPS = {
    'runtime loader identity and returned operations require actual source-issued model execution provenance': 'parent_source',
    'checkpoint child requires its own registered OS handle and verified parent relationship': 'child_source',
    'checkpoint raw equality and terminal metadata need actual retained parent/child OS handles and outer prepayment verification': 'child_source',
    'full saved-byte common-anchor/master/deployed rebuild and residuals require actual source execution provenance': 'delta_source',
    'resource measurements retain their explicit IO subset; complete native unit prices and actual probe execution require verification': 'resource_source'}
BOOTSTRAP_GAPS = (
    'actual source-owned OS/entry provenance has not been verified',
    'source/OS terminal evidence must verify actual final record/readback suffix within prepaid wall/deadline',
    'publisher/process inner operation actuals are not BootstrapAccounting events; only their matching full ticket reservations are retained')


def discharge_known_gaps(gaps, supported, evidence):
    """Every exact known limitation needs evidence; unknown messages refuse."""
    require(type(gaps) is list and all(type(gap) is str for gap in gaps)
            and type(supported) is dict and type(evidence) is dict, 'explicit verifier gaps and source evidence')
    require(all(gap in supported for gap in gaps), 'unknown raw verification gap cannot certify qualification')
    result = {}
    for gap in gaps:
        key = supported[gap]
        require(key in evidence and type(evidence[key]) is dict and evidence[key], 'actual evidence missing for recognized verification gap')
        result[gap] = copy.deepcopy(evidence[key])
    return result


class VerifiedQualificationEvidence:
    """Exact source result retained across owned-handle closure, consumed once."""
    __slots__ = ('_seal', '_owner', '_service', '_supervision', '_handle', '_observation', '_payload', '_used')
    def __init__(self, *args, **kwargs):
        raise raw.EvidenceRejected('qualification evidence is issued only by the fixed finalizer')
    def __setattr__(self, name, value):
        raise AttributeError('immutable source qualification evidence')


def _issue(owner, service, supervision, observation, payload):
    value = object.__new__(VerifiedQualificationEvidence)
    require(id(owner) not in _ISSUED, 'one source preclose result per permanent owner')
    for name, item in {'_seal': _SEAL, '_owner': owner, '_service': service, '_supervision': supervision,
                       '_handle': supervision['terminal_handle'],
                       '_observation': observation, '_payload': copy.deepcopy(payload), '_used': False}.items():
        object.__setattr__(value, name, item)
    _ISSUED[id(owner)] = value
    return value


def _binding(envelope, recipe):
    return {'stage': 'qualification_finalizer', 'request_sha256': envelope['request_sha256'],
        'scope_sha256': envelope['scope_sha256'], 'static_recipe_sha256': recipe,
        'code_sha256': envelope['code_sha256'], 'worker_identity_sha256': envelope['terminal_worker_proof']['worker_identity_sha256'],
        'attempt_id': 'independent-post-exit-finalization'}


def clock_suffix(last, origin, observed, reservation_upper, physical_upper):
    """Arithmetic only: caller must supply a source-owned actual observation."""
    require(all(raw._finite(value) for value in (last, origin, observed, reservation_upper, physical_upper))
            and origin <= last <= observed and observed - last <= reservation_upper
            and observed - origin <= physical_upper, 'actual terminal suffix exceeds original clock reservation/deadline')
    return {'last_recorded_monotonic': last, 'observed_terminal_monotonic': observed,
        'suffix_seconds_upper_observed': observed - last, 'reserved_suffix_seconds_upper': reservation_upper,
        'physical_elapsed_seconds_upper_observed': observed - origin}


def _expected_bootstrap_owners(context, *, child=False):
    scope = context['bindings']['scope']; receipt_sha = context['prepared_receipt']['receipt_sha256']
    spec = accounting.accounting_specification(context)
    setup = accounting._price(spec['setup_bounds']); setup['metadata_bytes_upper'] = 2 * accounting.MAX_LINE
    result = [{'owner': {'accounting_spec_sha256': digest(spec), 'prepared_receipt_sha256': receipt_sha}, 'bounds': setup}]
    if not child:
        result.append({'owner': scope['checkpoint_processes_owner'], 'bounds': scope['checkpoint_processes']['bounds']})
        publication_spec = scope['publication']
    else:
        from .ce_checkpoint_publication import publication_specification
        publication_spec = publication_specification(context)
    result.append({'owner': {'accounting_spec_sha256': digest(publication_spec), 'prepared_receipt_sha256': receipt_sha},
                   'bounds': publication_spec['bounds']})
    return result


def replay_scope_bootstrap(context, capsule, proof, *, reader, preflight, child=False, outer_evidence=()):
    require(type(capsule) is dict and capsule.get('schema') == boot.SCHEMA
            and type(capsule.get('snapshot')) is dict, 'exact retained scope bootstrap capsule')
    result = bootstrap.verify_bootstrap(context, capsule['header_ref'], capsule['journal_ref'],
        reader=reader, outer_preflight=preflight, expected_workers={proof['worker_identity_sha256']: proof},
        expected_owners=_expected_bootstrap_owners(context, child=child), outer_evidence=outer_evidence)
    snapshot = capsule['snapshot']
    require(snapshot['binding_sha256'] == result['binding_sha256'] and snapshot['ledger_head'] == result['ledger_head']
            and snapshot['cumulative'] == result['derived_totals'] and snapshot['tickets'] == result['derived_tickets']
            and snapshot['actual_wall_seconds'] == result['recorded_wall_seconds']
            and snapshot['original_origin'] == result['first_monotonic'] and snapshot['last_monotonic'] == result['last_monotonic']
            and snapshot['end_reservation'] == result['end_reservation'] and snapshot['proof'] == proof
            and snapshot.get('complete_recorded_clock') is result['clock_coverage_complete'],
            'scope capsule refund/clock/owner differs from original raw replay')
    require(result['clock_coverage_complete'] and not result['ticket_statuses']['RESERVED']
            and not result['ticket_statuses']['STARTED'], 'scope bootstrap has pending/unknown clock or work')
    require(all(gap in BOOTSTRAP_GAPS for gap in result['activation_gaps']), 'unknown bootstrap provenance/completion gap')
    return result


def compare_child_replay(retained, independent, bundle, *, parent_binding):
    """Check every raw child terminal record against independent state replay."""
    require(type(retained) is dict and retained.get('schema') == 'CE8-PARENT-CHILD-TERMINAL-REPLAY-1'
            and retained.get('raw_numeric_reductions') == independent
            and retained.get('checkpoint_bundle_ref') == bundle['reference'],
            'retained child/numeric replay differs from independent raw reduction')
    results = retained.get('child_terminal_results')
    require(type(results) is dict and set(results) == set(checkpoint.BRANCHES), 'all source-linked child terminal observations')
    cp = independent['checkpoint_resume']; payload = bundle['payload']
    for branch in checkpoint.BRANCHES:
        result, child = results[branch], cp['raw_branches']['branches'][branch]
        require(result.get('binding') == child['binding'] and result.get('terminal_ref') == payload['child_terminal_refs'][branch]
                and result.get('child_derived_reservations') == child['derived_costs'], 'child terminal/cost/source linkage differs')
        terminal = result['terminal_proof']; proof = child['lineage']['child_proof']
        require(terminal.get('schema') == worker.TERMINAL and terminal.get('status_file_used') is False
                and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0
                and all(terminal.get(key) == child['binding'][key] for key in
                        ('worker_identity_sha256', 'request_sha256', 'scope_sha256'))
                and terminal.get('os_identity') == proof['os_identity']
                and proof['request_sha256'] == parent_binding['request_sha256']
                and proof['scope_sha256'] == parent_binding['scope_sha256']
                and raw._finite(result.get('terminal_observed_monotonic')),
                'source-linked actual child terminal/OS/time proof required')
    return results


def _child_costs(context, bundle, terminal_results, binding, owner):
    """Late source-ended child capsules are separate from live branch bodies."""
    refs = bundle.get('child_final_cost_clock_refs')
    if type(refs) is not dict or set(refs) != set(checkpoint.BRANCHES):
        return {}, ['all three source-ended child bootstrap/service cost-clock capsules are missing']
    reduced = {}
    for branch in checkpoint.BRANCHES:
        child_binding = bundle['branch_bindings'][branch]
        branch_payload = checkpoint._envelope(owner.reader, bundle['branch_refs'][branch], child_binding)
        child = branch_payload['child_context']
        child_original = copy.deepcopy(child)
        child['verification_binding'] = binding
        late = execution._read_json(owner.reader, refs[branch], binding, 'child_final_cost_clock')
        require(late.get('schema') == 'CE8-CHECKPOINT-CHILD-FINAL-COST-CLOCK-1' and late.get('branch') == branch
                and late.get('request_sha256') == context['request_sha256'] and late.get('scope_sha256') == context['scope_sha256'],
                'late child capsule exact branch/request/scope')
        proof = branch_payload['observed_lineage']['child_proof']
        require(late.get('worker_proof') == proof and late.get('branch_ref') == bundle['branch_refs'][branch],
                'late child capsule worker/immutable branch provenance')
        outer = replay_scope_bootstrap(child, late['scope_bootstrap_evidence'], proof, reader=owner.reader,
            preflight=owner.preflight, child=True, outer_evidence=[branch_payload['outer_publication_prepayment']])
        service = late['service_bootstrap_evidence']; specification = late['service_specification']
        configuration = execution._read_json(owner.reader, late['configuration_ref'], binding, 'child_configuration')
        require(configuration['service_spec'] == specification, 'child final service differs from original frozen configuration')
        if child.get('mode') != 'synthetic_fixture':
            require(late.get('static_recipe_sha256') == child['static_recipe_sha256']
                    and late.get('child_context_sha256') == digest(child_original), 'late child frozen context/recipe identity')
            payment = late['constructor_prepayment']
            constructor = execution._read_json(owner.reader, payment['reservation_ref'], binding, 'child_constructor_prepayment')
            require(payment.get('schema') == constructor.get('schema') == 'CE8-SERVICE-CONSTRUCTOR-PREPAYMENT-1'
                    and payment.get('consumed') is True and payment['request_sha256'] == child['request_sha256']
                    and payment['service_spec_sha256'] == digest(specification)
                    and payment['bootstrap_spec_sha256'] == digest(specification['bootstrap'])
                    and constructor['bounds'] == payment['bounds'] == specification['constructor_prepayment']['bounds']
                    and constructor['service_spec_sha256'] == payment['service_spec_sha256']
                    and constructor['request_sha256'] == child['request_sha256'], 'actual child constructor reservation/source linkage')
        sr = service_replay.verify_service_bootstrap(child, specification['bootstrap'], service['header_ref'], service['journal_ref'],
            service['seal_ref'], reader=owner.reader, outer_preflight=owner.preflight)
        stamp = terminal_results[branch]['terminal_observed_monotonic']
        physical = boot.bootstrap_specification(child)['physical_wall_seconds_upper']
        outer_suffix = clock_suffix(outer['last_monotonic'], outer['first_monotonic'], stamp,
            outer['end_reservation']['bounds']['wall_seconds_upper'], physical)
        snap = sr['derived_snapshot']; service_suffix = clock_suffix(snap['last_monotonic'], snap['original_origin'], stamp,
            snap['end_reservation']['bounds']['wall_s_upper'], specification['bootstrap']['physical_wall_seconds_upper'])
        require(sr['recorded_clock_ended'] and not snap['pending'], 'child service did not end after physical handle cleanup')
        discharge = discharge_known_gaps(outer['activation_gaps'], {gap: 'actual_child' for gap in BOOTSTRAP_GAPS},
            {'actual_child': {'source_parent_retained_terminal_observation': terminal_results[branch],
                'raw_child_binding': child_binding, 'raw_child_scope_tickets': outer['derived_tickets'],
                'raw_child_service_costs': sr['derived_totals'], 'actual_scope_suffix': outer_suffix,
                'actual_service_suffix': service_suffix, 'source_enforced_cost_basis': CLAIM_SCOPE['cost_basis']}})
        reduced[branch] = {'scope_bootstrap': outer, 'service_bootstrap': sr, 'scope_terminal_suffix': outer_suffix,
                           'service_terminal_suffix': service_suffix, 'capsule_ref': refs[branch],
                           'discharged_bootstrap_gaps': discharge}
    return reduced, []


def _reduce_body(configuration, envelope, body, context, prepared, child_replay, scope_capsule, observation, owner):
    binding = context['verification_binding']; context['prepared_receipt'] = prepared
    with owner.reader._overhead('finalizer_source_scope_census', work_bytes=128 * raw.MAX_JSON,
                              hash_attempts=256, hash_bytes=128 * raw.MAX_JSON):
        require(context['request_sha256'] == envelope['request_sha256'] and context['scope_sha256'] == envelope['scope_sha256']
                and context['static_recipe_sha256'] == configuration['service_spec']['scope_adapters']['static_recipe_sha256']
                and context['bindings']['source']['code_sha256'] == envelope['code_sha256']
                == configuration['service_spec']['approved_code_sha256'], 'body source/request/scope differs from actual launch')
        require(context['bindings']['scope'].get('qualification_claim_scope') == CLAIM_SCOPE,
                'exact registered qualification measurement scope; full unit prices are not measured')
        q.validate_actual_ingestion(context, prepared)
        local.validate_parent_view(envelope['live_worker_proof'], body['worker_proof'])
    refs = body['ledger_refs']
    with owner.reader._overhead('finalizer_Q_header_scope', work_bytes=128 * raw.MAX_JSON,
                              hash_attempts=128, hash_bytes=128 * raw.MAX_JSON):
        header = execution._header(context, execution._read_json(owner.reader, refs['header'], binding, 'qualification_header'))
    state, _, attempts = execution._replay(owner.reader, refs['journal'], binding, context, header,
                                           {body['worker_proof']['worker_identity_sha256']})
    seal = execution._read_json(owner.reader, refs['seal'], binding, 'qualification_seal')
    with owner.reader._overhead('finalizer_Q_seal_and_stage_linkage', work_bytes=64 * (refs['seal']['bytes'] + refs['journal']['bytes']),
                              hash_attempts=execution.MAX_EVENTS, hash_bytes=8 * refs['journal']['bytes']):
        complete, pending = execution._seal(seal, state)
        require(complete and not pending, 'parent retained Q does not prove complete known qualification work')
    cp_attempts = [value for value in state.attempts if value['stage'] == 'checkpoint_resume' and value['status'] == 'RETURNED']
    require(len(cp_attempts) == 1 and len(cp_attempts[0]['evidence']['raw_refs']) == 1, 'one original parent checkpoint bundle')
    cp_binding = execution._binding_for(context, cp_attempts[0], body['worker_proof']['worker_identity_sha256'])
    cp_ref = cp_attempts[0]['evidence']['raw_refs'][0]
    bundle = owner.reader._envelope(cp_ref, cp_binding, 'checkpoint_branches', 'checkpoint_branches')
    child_bindings, child_lineages = {}, {}
    for branch in checkpoint.BRANCHES:
        ref, cb = bundle['branch_refs'][branch], bundle['branch_bindings'][branch]
        payload = checkpoint._envelope(owner.reader, ref, cb)
        child_bindings[ref['path']] = cb; child_lineages[cb['worker_identity_sha256']] = payload['observed_lineage']
    with owner.reader._overhead('finalizer_all_stage_source_linkage', work_bytes=128 * raw.MAX_JSON,
                              hash_attempts=256, hash_bytes=128 * raw.MAX_JSON):
        reductions, reduction_gaps = execution._reductions(context, state, attempts, owner.reader, child_bindings, child_lineages,
                                              outer_preflight=owner.preflight)
    require(set(reductions) == set(q.STAGES[1:]), 'all seven independently recomputed native numeric stages')
    terminals = compare_child_replay(child_replay, reductions, {'reference': cp_ref, 'payload': bundle}, parent_binding=cp_binding)
    # Read the original retained child observations; a replay result dictionary
    # cannot substitute for the source's separate raw terminal records.
    for branch in checkpoint.BRANCHES:
        record = execution._read_json(owner.reader, terminals[branch]['terminal_ref'], cp_binding, 'checkpoint_retained_terminal')
        require(record['binding'] == terminals[branch]['binding'] and record['terminal_proof'] == terminals[branch]['terminal_proof']
                and record['terminal_observed_monotonic'] == terminals[branch]['terminal_observed_monotonic'],
                'child terminal observation is not its exact original raw record')
    outer = replay_scope_bootstrap(context, scope_capsule, body['worker_proof'], reader=owner.reader, preflight=owner.preflight,
        outer_evidence=[body['publication_prepayment']])
    parent_service = envelope['parent_service_evidence']
    require(parent_service['specification'] == configuration['service_spec']['bootstrap'], 'exact original parent service scope')
    sr = service_replay.verify_service_bootstrap(context, parent_service['specification'], parent_service['header_ref'],
        parent_service['journal_ref'], parent_service['seal_ref'], reader=owner.reader, outer_preflight=owner.preflight)
    service_suffix = owner.validate_service_suffix(sr, observation)
    stamp = observation.monotonic
    snap = sr['derived_snapshot']
    service_suffix['physical_deadline'] = clock_suffix(snap['last_monotonic'], snap['original_origin'], stamp,
        snap['end_reservation']['bounds']['wall_s_upper'], parent_service['specification']['physical_wall_seconds_upper'])
    scope_suffix = clock_suffix(outer['last_monotonic'], outer['first_monotonic'], stamp,
        outer['end_reservation']['bounds']['wall_seconds_upper'], boot.bootstrap_specification(context)['physical_wall_seconds_upper'])
    children, gaps = _child_costs(context, bundle, terminals, binding, owner)
    evidence = {'parent_source': {'terminal_proof': observation.proof, 'body_worker_proof': body['worker_proof'],
        'source_ref': configuration['request']['source_ref'], 'original_service_refs': parent_service},
        'child_source': {'source_parent_terminal_proof': observation.proof, 'child_terminal_records': terminals,
            'independent_checkpoint_reduction': reductions['checkpoint_resume'], 'child_cost_clock_reductions': children},
        'delta_source': {'source_parent_terminal_proof': observation.proof,
            'byte_derived_rebuild': reductions['delta_roundtrip']['byte_derived_rebuild'],
            'parent_master_source': reductions['delta_roundtrip']['parent_master_source']},
        'resource_source': {'source_parent_terminal_proof': observation.proof,
            'original_probe_and_barrier_reduction': reductions['resource_io_measurement'],
            'qualification_claim_scope': copy.deepcopy(CLAIM_SCOPE),
            'outside_qualification_measurement_contract': 'full model/checkpoint/OS unit prices NOT_MEASURED'}}
    discharged = discharge_known_gaps(reduction_gaps, EXECUTION_GAPS, evidence)
    boot_discharge = discharge_known_gaps(outer['activation_gaps'], {gap: 'actual_scope' for gap in BOOTSTRAP_GAPS},
        {'actual_scope': {'actual_parent_terminal_proof': observation.proof, 'recorded_scope_suffix': scope_suffix,
            'matching_full_permanent_ticket_reservations': outer['derived_tickets'],
            'independent_parent_Q_costs': state.cumulative, 'source_enforced_cost_basis': CLAIM_SCOPE['cost_basis']}})
    return {'schema': SCHEMA, 'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'static_recipe_sha256': context['static_recipe_sha256'], 'arm': context['arm'],
        'raw_numeric_reductions': reductions, 'parent_derived_costs': state.cumulative,
        'parent_scope_bootstrap': outer, 'parent_service_bootstrap': sr, 'parent_scope_suffix': scope_suffix,
        'parent_service_suffix': service_suffix, 'child_cost_clock_reductions': children,
        'child_terminal_observations': terminals, 'parent_terminal_proof': observation.proof,
        'original_execution_gaps': reduction_gaps, 'discharged_execution_gaps': discharged,
        'discharged_parent_bootstrap_gaps': boot_discharge, 'qualification_claim_scope': copy.deepcopy(CLAIM_SCOPE),
        'evidence_refs': {'body': envelope['body_ref'], 'context': body['context_ref'], 'prepared': body['prepared_receipt_ref'],
            'Q': refs, 'scope_bootstrap': envelope['final_scope_bootstrap_evidence_ref'],
            'child_replay': envelope['checkpoint_child_terminal_replay_ref'], 'parent_service': parent_service},
        'qualification_gaps': gaps, 'native_qualified': False, 'execution_verified': False}


def verify_supervision(service, result, owner):
    """Consume actual retained parent provenance before owned-handle closure."""
    require(type(service) is services.NativeControllerServices, 'actual source metadata service required')
    service._admit()
    from .ce_finalizer_accounting import FinalizerCostOwner
    require(type(owner) is FinalizerCostOwner and owner.native is True, 'actual native permanent finalizer cost owner required')
    require(type(result) is dict and getattr(service, '_qualification_supervision_result', None) is result
            and owner.service is service and type(result.get('terminal_handle')) is worker.WinWorkerHandle,
            'exact source-owned supervision result and retained parent handle')
    envelope = result['envelope']; expected = {key: envelope['terminal_worker_proof'][key] for key in
                                            ('request_sha256', 'scope_sha256', 'worker_identity_sha256')}
    observation = owner.observe_parent_terminal(result['terminal_handle'], expected)
    require(observation.proof == envelope['terminal_worker_proof'], 'retained parent handle differs from actual supervised terminal')
    from . import ce_qualification_entry as entry
    configuration = entry._configuration()
    binding = _binding(envelope, configuration['service_spec']['scope_adapters']['static_recipe_sha256'])
    retained = execution._read_json(owner.reader, result['raw_artifact'], binding, 'qualification_supervision')
    require(retained == envelope, 'actual supervision artifact differs from original source result')
    body = execution._read_json(owner.reader, envelope['body_ref'], binding, 'parent_body_complete')
    context = execution._read_json(owner.reader, body['context_ref'], binding, 'parent_context')
    context['verification_binding'] = binding
    prepared = execution._read_json(owner.reader, body['prepared_receipt_ref'], binding, 'prepared_receipt')
    children = execution._read_json(owner.reader, envelope['checkpoint_child_terminal_replay_ref'], binding, 'checkpoint_terminal_replay')
    scope = execution._read_json(owner.reader, envelope['final_scope_bootstrap_evidence_ref'], binding, 'scope_bootstrap_evidence')
    payload = _reduce_body(configuration, envelope, body, context, prepared, children, scope, observation, owner)
    payload['preclose_replay_ref'] = owner.write_json('PRE_CLOSE_QUALIFICATION_REPLAY.json', payload)
    return _issue(owner, service, result, observation, payload)


def consume_verified(value, supervisor_end_observation):
    """Finish raw supervisor replay after source-owned handle cleanup/end.

    The finalizer owner remains live through receipt write/readback and its own
    prepaid final seal. It cannot certify itself through a caller Boolean.
    """
    require(type(value) is VerifiedQualificationEvidence and getattr(value, '_seal', None) is _SEAL
            and not value._used and _ISSUED.get(id(value._owner)) is value, 'one exact source-issued preclose qualification result')
    from .ce_finalizer_accounting import FinalizerCostOwner
    require(type(value._owner) is FinalizerCostOwner and value._owner.native is True
            and value._observation.native is True, 'exact native permanent finalizer owner')
    object.__setattr__(value, '_used', True)
    _ISSUED.pop(id(value._owner))
    owner, service = value._owner, value._service
    require(value._supervision.get('terminal_handle') is None and type(value._handle) is worker.WinWorkerHandle
            and value._handle._seal is worker._HANDLE_SEAL and value._handle.closed is True,
            'actual original parent clone closure precedes final supervisor seal')
    binding = _binding(value._supervision['envelope'], value._payload['static_recipe_sha256'])
    retained = execution._read_json(owner.reader, value._payload['preclose_replay_ref'], binding, 'preclose_qualification_replay')
    require(retained == {key: item for key, item in value._payload.items() if key != 'preclose_replay_ref'},
            'source preclose result differs from its exact retained paid replay')
    evidence = supervisor_end_observation.evidence
    context = {'mode': 'registered_scope', 'request_sha256': value._payload['request_sha256'],
               'verification_binding': binding}
    result = service_replay.verify_service_bootstrap(context, service.spec['bootstrap'], evidence['header_ref'], evidence['journal_ref'],
        evidence['seal_ref'], reader=owner.reader, outer_preflight=owner.preflight)
    suffix = owner.validate_service_suffix(result, supervisor_end_observation)
    require(result['recorded_clock_ended'] and not result['derived_snapshot']['pending'], 'final supervisor cost/clock remains unfinished')
    snap = result['derived_snapshot']
    suffix['physical_deadline'] = clock_suffix(snap['last_monotonic'], snap['original_origin'], supervisor_end_observation.monotonic,
        snap['end_reservation']['bounds']['wall_s_upper'], service.spec['bootstrap']['physical_wall_seconds_upper'])
    gaps = list(value._payload['qualification_gaps'])
    passed = not gaps
    body = {'schema': q.RECEIPT, 'phase': 'qualification', 'status': 'PROOF_COMPLETE_REQUIRES_FINALIZER_SEAL' if passed else 'UNQUALIFIED',
        'passed': passed, 'native_actual': passed, 'native_qualified': passed, 'execution_verified': passed,
        'production_training_authorized': False, 'request_sha256': value._payload['request_sha256'],
        'scope_sha256': value._payload['scope_sha256'], 'static_recipe_sha256': value._payload['static_recipe_sha256'],
        'arm': value._payload['arm'], 'preclose_replay_ref': value._payload['preclose_replay_ref'],
        'finalizer_header_ref': copy.deepcopy(owner.header_ref), 'expected_finalizer_seal_path': str(owner.seal_path),
        'cost_owner_binding_sha256': digest(owner.header),
        'qualification_claim_scope': copy.deepcopy(CLAIM_SCOPE),
        'supervisor_service_replay': result, 'supervisor_terminal_suffix': suffix,
        'qualification_gaps': gaps, 'classification': 'discarded native mechanics qualification; no acquisition or training grant'}
    receipt = dict(body, receipt_sha256=digest(body))
    reference = owner.write_json('QUALIFICATION_RECEIPT.json', receipt)
    candidate = {'receipt_ref': reference, 'proof_complete': passed, 'native_qualified': False,
                 'completion_pending_cost_seal': True, 'qualification_gaps': gaps}
    _CANDIDATES[id(owner)] = (candidate, copy.deepcopy(candidate))
    return candidate


def finish_cost_seal(candidate, owner, cost_evidence):
    """Source completion only after this owner's actual create-only cost seal."""
    from .ce_finalizer_accounting import FinalizerCostOwner
    cached = _CANDIDATES.get(id(owner))
    require(cached is not None and cached[0] is candidate and candidate == cached[1],
            'exact unchanged source candidate, never caller qualification flags')
    owner.require_closed_evidence(cost_evidence)
    require(type(owner) is FinalizerCostOwner and owner.native is True and owner.closed is True and not owner.fatal
            and type(cost_evidence) is dict and cost_evidence.get('receipt_refs') == owner.receipts
            and cost_evidence['seal_ref']['path'] == str(owner.seal_path)
            and type(candidate) is dict and owner.receipts.get('QUALIFICATION_RECEIPT.json') == candidate.get('receipt_ref'),
            'actual successful source cost seal and exact receipt pairing required')
    _CANDIDATES.pop(id(owner))
    passed = cached[1]['proof_complete']
    return dict(candidate, status='NATIVE_QUALIFIED' if passed else 'UNQUALIFIED',
        passed=passed, native_qualified=passed,
        completion_pending_cost_seal=False, finalizer_cost_evidence=copy.deepcopy(cost_evidence))
