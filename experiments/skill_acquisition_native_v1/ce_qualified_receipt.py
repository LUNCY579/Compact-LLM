"""Read-only consumption of a source-selected completed qualification pair.

This verifies the retained finalizer accounting and its exact preclose linkage.
It does not repeat model arithmetic or reopen historical OS handles, and grants
no model, training or execution authority. The default source selection is closed.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MappingProxyType

from . import ce_finalizer_accounting as costs, ce_qualification_finalizer as finalizer
from . import ce_qualification_artifacts as raw, ce_execution_verifier as files
from . import ce_qualification as qualification

SCHEMA = 'CE8-QUALIFIED-RECEIPT-SELECTION-1'
RESULT = 'CE8-COMPLETED-QUALIFICATION-PAIR-1'
MAX_JOURNAL, MAX_EVENTS = 256 * raw.CHUNK, 200000
_SOURCE_SELECTION = None
_SOURCE_SELECTION_SHA256 = None
REF_NAMES = frozenset(('completion', 'receipt', 'preclose', 'header', 'journal', 'seal',
                      'context', 'prepared', 'scope'))
require, canonical, digest = raw.require, raw.canonical, raw.digest


def selected_qualification():
    require(type(_SOURCE_SELECTION) is MappingProxyType and raw._hex(_SOURCE_SELECTION_SHA256),
            'completed qualification source pin absent; consumer closed')
    return _SOURCE_SELECTION


def _shape(selection):
    require(type(selection) is dict and set(selection) == {'schema', 'owned_root', 'arm', 'binding', 'references',
            'finalizer_specification', 'claim_scope', 'runtime_sha256', 'anchors_sha256', 'prepared_receipt_sha256'}
            and selection['schema'] == SCHEMA and selection['arm'] == 'canonical', 'exact source qualification selection')
    raw.binding(selection['binding'])
    require(selection['binding']['stage'] == 'qualification_finalizer'
            and selection['binding']['attempt_id'] == 'independent-post-exit-finalization', 'exact finalizer binding')
    require(type(selection['owned_root']) is str and Path(selection['owned_root']).is_absolute()
            and all(raw._hex(selection[key]) for key in ('runtime_sha256', 'anchors_sha256', 'prepared_receipt_sha256')),
            'source-selected root/runtime/anchor/census hashes')
    refs = selection['references']
    require(type(refs) is dict and set(refs) == REF_NAMES, 'complete source-selected qualification references')
    for name, reference in refs.items():
        files._reference(reference)
        require(set(reference) == {'path', 'sha256', 'bytes'} and reference['bytes'] <=
                (MAX_JOURNAL if name == 'journal' else costs.MAX_JSON), 'bounded exact qualification reference: ' + name)
    require(refs['scope']['sha256'] == selection['binding']['scope_sha256'], 'source-selected raw scope hash')
    spec = costs.validate_specification(selection['finalizer_specification'], selection['owned_root'])
    root = Path(spec['owned_directory'])
    for name, filename in (('receipt', 'QUALIFICATION_RECEIPT.json'), ('preclose', 'PRE_CLOSE_QUALIFICATION_REPLAY.json'),
                           ('header', 'FINALIZER_HEADER.json'), ('journal', 'FINALIZER.jsonl'), ('seal', 'FINALIZER_SEAL.json')):
        require(Path(refs[name]['path']) == root / filename, 'exact finalizer owned path: ' + name)
    require(selection['claim_scope'] == finalizer.CLAIM_SCOPE, 'unchanged qualification measurement limits')
    return selection


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _counts(value):
    require(type(value) is dict and set(value) == set(costs.COST_FIELDS)
            and all(_finite(value[key]) and (key == 'wall_seconds_upper' or type(value[key]) is int)
                    for key in costs.COST_FIELDS), 'complete strict finalizer counters')
    return value


def _replay(header, header_ref, journal_ref, selection, reader, *, fixture):
    """Derive the equations implemented by FinalizerCostOwner, without IO writes."""
    spec, binding = selection['finalizer_specification'], selection['binding']
    origin = header.get('original_origin')
    require(_finite(origin) and header == {'schema': costs.SCHEMA, 'native': not fixture,
            'request_sha256': binding['request_sha256'], 'scope_sha256': binding['scope_sha256'],
            'specification': spec, 'original_origin': origin, 'permanent_reserved_bounds': spec['self_bounds'],
            'authority': 'original source supervisor metadata permit; no model authority'},
            'original native finalizer header/source scope differs')
    # Reuse the source owner's pure cap/suffix equation. No constructor, clock,
    # authority, native callback or file method is invoked on this shadow.
    shadow = object.__new__(costs.FinalizerCostOwner)
    shadow.spec, shadow.counts = spec, dict.fromkeys(costs.COST_FIELDS, 0)
    shadow._suffix_before = None
    charge = shadow._charge
    charge({'operation_attempts_upper': 1, 'io_write_bytes_upper': header_ref['bytes'],
            'metadata_bytes_upper': header_ref['bytes'], 'identity_hash_attempts_upper': 1,
            'identity_hash_bytes_upper': header_ref['bytes']})
    operations, receipts = {}, {}
    last, suffix_origin, head, count, ended = origin, None, None, 0, False
    wrapped = files._decorated(reader, journal_ref, binding, 'completed_finalizer_journal')
    with reader._overhead('completed_finalizer_chain_replay', work_bytes=64 * journal_ref['bytes'],
                          hash_attempts=2 * MAX_EVENTS, hash_bytes=16 * journal_ref['bytes']):
        stream = reader._blocks(wrapped, binding, 'completed_finalizer_journal')
        buffer, primary = b'', None
        try:
            for block in stream:
                buffer += block
                while b'\n' in buffer:
                    split = buffer.index(b'\n') + 1
                    line, buffer = buffer[:split], buffer[split:]
                    require(len(line) <= costs.MAX_JSON and count < MAX_EVENTS and not ended,
                            'bounded finalizer chain; no row after end')
                    row = raw._json(line)
                    require(type(row) is dict and set(row) == {'schema', 'seq', 'previous', 'request_sha256',
                            'scope_sha256', 'kind', 'payload', 'sha256'}, 'exact finalizer event fields')
                    body = {key: value for key, value in row.items() if key != 'sha256'}
                    require(row['schema'] == costs.SCHEMA and type(row['seq']) is int and row['seq'] == count
                            and row['previous'] == head and row['sha256'] == digest(body)
                            and row['request_sha256'] == binding['request_sha256']
                            and row['scope_sha256'] == binding['scope_sha256'] and canonical(row) + b'\n' == line,
                            'finalizer canonical hash/order/request/scope chain')
                    kind, payload = row['kind'], row['payload']
                    require(type(payload) is dict and (count != 0 or kind == 'permanent_self_reservation'),
                            'finalizer first permanent reservation')
                    if kind == 'permanent_self_reservation':
                        require(count == 0 and payload == {'bounds': spec['self_bounds'], 'original_origin': origin},
                                'one unchanged original full reservation')
                    elif kind == 'wall':
                        now, elapsed = payload.get('monotonic'), payload.get('elapsed_delta')
                        require(set(payload) == {'original_origin', 'monotonic', 'elapsed_delta'}
                                and payload['original_origin'] == origin and _finite(now) and now > last
                                and _finite(elapsed) and elapsed == now - last
                                and now - origin <= spec['physical_wall_seconds_upper']
                                and (suffix_origin is None or now - suffix_origin <= spec['finalization_bounds']['wall_seconds_upper']),
                                'original finalizer wall interval/deadline')
                        charge({'wall_seconds_upper': elapsed}); last = now
                    elif kind == 'permanent_final_suffix':
                        require(suffix_origin is None and set(payload) == {'bounds', 'monotonic'}
                                and payload['bounds'] == spec['finalization_bounds'] and _finite(payload['monotonic'])
                                and last <= payload['monotonic'] <= origin + spec['physical_wall_seconds_upper'],
                                'one original finalizer suffix reservation')
                        suffix_origin, shadow._suffix_before = payload['monotonic'], dict(shadow.counts)
                    elif kind == 'operation':
                        require(set(payload) == {'kind', 'phase', 'operation_id', 'fields'}, 'exact finalizer physical operation')
                        op, phase, ident, fields = (payload[key] for key in ('kind', 'phase', 'operation_id', 'fields'))
                        require(op in costs.KINDS and type(ident) is str and ident and type(fields) is dict,
                                'priced finalizer operation identity')
                        if phase == 'started':
                            require(ident not in operations, 'unique finalizer physical attempt')
                            wanted = {'operation_attempts_upper': fields.get('operation_attempts_upper', 1)}
                            for key in costs.COST_FIELDS:
                                if key in ('operation_attempts_upper', 'wall_seconds_upper'): continue
                                alias = {'io_read_bytes_upper': 'read_bytes_upper', 'io_write_bytes_upper': 'write_bytes_upper'}.get(key)
                                for present in (key, alias):
                                    if present in fields:
                                        require(type(fields[present]) is int and fields[present] >= 0,
                                                'strict original finalizer cost declaration')
                                require(alias not in fields or key not in fields or fields[alias] == fields[key],
                                        'conflicting finalizer IO declarations')
                                wanted[key] = fields.get(key, fields.get(alias, 0))
                            charge(wanted)
                            operations[ident] = {'kind': op, 'phase': phase, 'reservation': wanted, 'started_seq': count}
                        else:
                            previous = operations.get(ident)
                            require(phase == 'returned' and previous and previous['kind'] == op and previous['phase'] == 'started',
                                    'successful matching finalizer return; failed costs cannot qualify')
                            for field, upper in {'bytes_read': 'io_read_bytes_upper', 'bytes_written': 'io_write_bytes_upper',
                                    'identity_hash_attempts_actual': 'identity_hash_attempts_upper',
                                    'identity_hash_bytes_actual': 'identity_hash_bytes_upper'}.items():
                                if field in fields:
                                    require(type(fields[field]) is int and 0 <= fields[field] <= previous['reservation'][upper],
                                            'actual finalizer work exceeds original reservation')
                            previous.update(phase=phase, returned_seq=count, result=copy.deepcopy(fields))
                            if op == 'finalizer_json_write':
                                reference = fields.get('reference'); files._reference(reference)
                                require(set(reference) == {'path', 'sha256', 'bytes'}
                                        and Path(reference['path']).parent == Path(spec['owned_directory'])
                                        and reference['bytes'] <= spec['max_receipt_bytes'], 'owned returned finalizer JSON')
                                name = Path(reference['path']).name
                                require(name not in receipts, 'create-only finalizer JSON references')
                                receipts[name] = reference
                    elif kind == 'preflight':
                        require(set(payload) == {'name', 'contract'} and payload['name'] in costs.PREFLIGHTS
                                and type(payload['contract']) is dict, 'finite source finalizer preflight')
                        if payload['name'] == 'saved_delta_source_rebuild':
                            require(payload['contract'].get('wall_seconds_upper') == 3600,
                                    'unchanged saved delta rebuild wall limit')
                    elif kind == 'end':
                        require(suffix_origin is not None and not any(op['phase'] == 'started' for op in operations.values())
                                and payload == {'original_origin': origin, 'monotonic': last, 'receipts': receipts},
                                'ended original finalizer clock with no pending work')
                        ended = True
                    else: require(False, 'failed or unknown finalizer event')
                    charge({'io_write_bytes_upper': len(line), 'metadata_bytes_upper': len(line),
                            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))})
                    head, count = row['sha256'], count + 1
                require(len(buffer) <= costs.MAX_JSON, 'bounded partial finalizer row')
            require(not buffer and ended, 'complete finalizer end; partial tail or missing seal history')
        except BaseException as error: primary = error; raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise
    return shadow, {'head': head, 'receipts': receipts, 'operations': operations, 'origin': origin,
                    'last': last, 'suffix_origin': suffix_origin, 'event_count': count}


def _cost_pair(selection, header, seal, evidence, reader, *, fixture):
    refs, binding, spec = selection['references'], selection['binding'], selection['finalizer_specification']
    shadow, state = _replay(header, refs['header'], refs['journal'], selection, reader, fixture=fixture)
    expected = {'schema': costs.SCHEMA, 'request_sha256': binding['request_sha256'], 'scope_sha256': binding['scope_sha256'],
        'header_ref': refs['header'], 'head': state['head'], 'permanent_reserved_bounds': spec['self_bounds'],
        'consumed_declared_bounds': shadow.counts, 'pending': [], 'failed': [], 'original_origin': state['origin'],
        'last_monotonic': state['last'], 'fatal': False, 'closed': False, 'receipt_refs': state['receipts'], 'native_qualified': False}
    require(seal == expected, 'finalizer seal differs from raw replay; no refunds, pending work or clock changes')
    require(type(evidence) is dict and set(evidence) == {'schema', 'header_ref', 'journal_ref', 'seal_ref', 'receipt_refs',
            'owner_closed', 'permanent_reserved_bounds', 'consumed_declared_bounds', 'native_qualified', 'claim'}
            and evidence['schema'] == costs.SCHEMA and evidence['owner_closed'] is True
            and evidence['native_qualified'] is False and evidence['receipt_refs'] == state['receipts']
            and evidence['permanent_reserved_bounds'] == spec['self_bounds']
            and all(evidence[key + '_ref'] == refs[key] for key in ('header', 'journal', 'seal')),
            'joint finalizer close evidence/reference pairing')
    # close_and_seal writes the frozen snapshot, then hashes exactly these
    # three retained files. These post-snapshot costs must not disappear.
    shadow._charge({'operation_attempts_upper': 1, 'io_write_bytes_upper': refs['seal']['bytes'],
                    'metadata_bytes_upper': refs['seal']['bytes']})
    for key in ('header', 'journal', 'seal'):
        size = refs[key]['bytes'] + 1
        shadow._charge({'operation_attempts_upper': 1, 'io_read_bytes_upper': size,
                        'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': size})
    consumed = _counts(evidence['consumed_declared_bounds'])
    require(all(consumed[key] == shadow.counts[key] for key in costs.COST_FIELDS if key != 'wall_seconds_upper'),
            'finalizer close/readback costs changed or refunded')
    suffix_wall = consumed['wall_seconds_upper'] - shadow.counts['wall_seconds_upper']
    require(_finite(suffix_wall), 'finalizer final inspection wall refunded')
    shadow._charge({'wall_seconds_upper': suffix_wall})
    final_clock = state['last'] + suffix_wall
    require(final_clock - state['origin'] <= spec['physical_wall_seconds_upper']
            and final_clock - state['suffix_origin'] <= spec['finalization_bounds']['wall_seconds_upper'],
            'finalizer inspection outside original clock/deadline')
    require(state['receipts'].get('QUALIFICATION_RECEIPT.json') == refs['receipt']
            and state['receipts'].get('PRE_CLOSE_QUALIFICATION_REPLAY.json') == refs['preclose'],
            'receipt/preclose absent from original returned writes')
    return {'derived_consumed_bounds': dict(shadow.counts), 'permanent_reserved_bounds': copy.deepcopy(spec['self_bounds']),
            'ledger_head': state['head'], 'event_count': state['event_count'], 'recorded_end': state['last'],
            'inspection_end_derived': final_clock}, state['operations']


def _verify(selection, *, reader, outer_preflight, fixture, source_pin=None):
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight), 'paid bounded qualification consumer required')
    # The independent owner pays before path metadata, raw reads, or hashes.
    outer_preflight('completed_qualification_pair', {'reader_bounds': dict(reader.remaining),
        'max_events': MAX_EVENTS, 'max_line_bytes': costs.MAX_JSON,
        'required_coverage': ['raw read/hash', 'finalizer replay', 'preclose/source/census linkage', 'verification wall']})
    with reader._overhead('completed_qualification_source_selection', work_bytes=128 * raw.MAX_JSON,
                          hash_attempts=128, hash_bytes=128 * raw.MAX_JSON):
        selection = _shape(copy.deepcopy(selection))
        if not fixture: require(digest(selection) == source_pin, 'immutable source selection changed')
        require(Path(reader.root) == Path(selection['owned_root']), 'reader uses exact selected owned root')
    refs, binding = selection['references'], selection['binding']
    observed = {name: files._read_json(reader, reference, binding, 'completed_qualification_' + name)
                for name, reference in refs.items() if name != 'journal'}
    with reader._overhead('completed_qualification_pair_linkage', work_bytes=128 * raw.MAX_JSON,
                          hash_attempts=128, hash_bytes=128 * raw.MAX_JSON):
        completion = observed['completion']
        joint = completion.get('qualification_finalization', completion)
        require(type(joint) is dict and set(joint) == {'receipt_ref', 'proof_complete', 'native_qualified',
                'completion_pending_cost_seal', 'qualification_gaps', 'status', 'passed', 'finalizer_cost_evidence'}
                and joint['status'] == 'NATIVE_QUALIFIED' and joint['passed'] is True and joint['proof_complete'] is True
                and joint['native_qualified'] is True and joint['completion_pending_cost_seal'] is False
                and joint['qualification_gaps'] == [] and joint['receipt_ref'] == refs['receipt'],
                'completed joint qualification receipt and finalizer seal required')
        if joint is not completion:
            require(completion.get('finalizer_cost_evidence') == joint['finalizer_cost_evidence'], 'same supervisor finalizer close evidence')
        receipt = observed['receipt']; body = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
        require(receipt.get('schema') == qualification.RECEIPT and receipt.get('phase') == 'qualification'
                and receipt.get('status') == 'PROOF_COMPLETE_REQUIRES_FINALIZER_SEAL'
                and all(receipt.get(key) is True for key in ('passed', 'native_actual', 'native_qualified', 'execution_verified'))
                and receipt.get('production_training_authorized') is False and receipt.get('qualification_gaps') == []
                and receipt.get('receipt_sha256') == digest(body), 'original qualified receipt digest/claims')
        require(receipt.get('request_sha256') == binding['request_sha256'] and receipt.get('scope_sha256') == binding['scope_sha256']
                and receipt.get('static_recipe_sha256') == binding['static_recipe_sha256'] and receipt.get('arm') == selection['arm']
                and receipt.get('preclose_replay_ref') == refs['preclose'] and receipt.get('finalizer_header_ref') == refs['header']
                and receipt.get('expected_finalizer_seal_path') == refs['seal']['path']
                and receipt.get('cost_owner_binding_sha256') == digest(observed['header'])
                and receipt.get('qualification_claim_scope') == selection['claim_scope'], 'exact receipt source/owner/preclose pairing')
    derived, operations = _cost_pair(selection, observed['header'], observed['seal'], joint['finalizer_cost_evidence'], reader, fixture=fixture)
    with reader._overhead('completed_qualification_preclose_census', work_bytes=128 * raw.MAX_JSON,
                          hash_attempts=128, hash_bytes=128 * raw.MAX_JSON):
        preclose, context, prepared = observed['preclose'], observed['context'], observed['prepared']
        require(preclose.get('schema') == finalizer.SCHEMA and preclose.get('qualification_gaps') == []
                and preclose.get('arm') == selection['arm']
                and all(preclose.get(key) == binding[key] for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256'))
                and preclose.get('qualification_claim_scope') == selection['claim_scope']
                and set(preclose.get('raw_numeric_reductions', {})) == set(qualification.STAGES[1:]),
                'complete exact preclose numerical replay linkage')
        require(preclose['evidence_refs']['context'] == refs['context'] and preclose['evidence_refs']['prepared'] == refs['prepared']
                and context['bindings']['scope'] == observed['scope']
                and context['bindings']['source']['code_sha256'] == binding['code_sha256']
                and context['request_sha256'] == binding['request_sha256'] and context['scope_sha256'] == binding['scope_sha256']
                and context['static_recipe_sha256'] == binding['static_recipe_sha256'] and context['arm'] == selection['arm']
                and digest(context['bindings']['recipe']['recipe']) == binding['static_recipe_sha256']
                and digest(context['bindings']['runtime']) == selection['runtime_sha256']
                and digest(context['bindings']['anchors']) == selection['anchors_sha256']
                and prepared.get('receipt_sha256') == selection['prepared_receipt_sha256'],
                'preclose source/runtime/anchor/scope/census changed')
        qualification.validate_actual_ingestion(context, prepared)
        proof = preclose['parent_terminal_proof']
        require(proof.get('request_sha256') == binding['request_sha256'] and proof.get('scope_sha256') == binding['scope_sha256']
                and proof.get('worker_identity_sha256') == binding['worker_identity_sha256']
                and type(proof.get('exit_code')) is int and proof['exit_code'] == 0 and proof.get('status_file_used') is False,
                'source-selected successful terminal proof linkage; no fresh OS claim')
        terminal = [op for op in operations.values() if op['kind'] == 'finalizer_terminal_observation']
        service = [op for op in operations.values() if op['kind'] == 'finalizer_service_end']
        writes = {Path(op['result']['reference']['path']).name: op for op in operations.values()
                  if op['kind'] == 'finalizer_json_write'}
        require(len(terminal) == len(service) == 1 and terminal[0]['result'].get('terminal_proof') == proof
                and terminal[0]['returned_seq'] < writes['PRE_CLOSE_QUALIFICATION_REPLAY.json']['started_seq']
                < service[0]['started_seq'] < service[0]['returned_seq']
                < writes['QUALIFICATION_RECEIPT.json']['started_seq'],
                'original observed terminal/preclose/service-end/receipt causal linkage')
        for op in (terminal[0], service[0]):
            before, after = op['result'].get('observed_before'), op['result'].get('observed_after')
            require(_finite(before) and _finite(after) and observed['header']['original_origin'] <= before <= after <= derived['recorded_end'],
                    'observed terminal/service clock outside original finalizer lifecycle')
        replay = receipt['supervisor_service_replay']; suffix = receipt['supervisor_terminal_suffix']
        snapshot = replay['derived_snapshot']
        require(replay.get('recorded_clock_ended') is True and snapshot.get('clock_ended') is True
                and snapshot.get('fatal') is False and snapshot.get('pending') == []
                and service[0]['result']['evidence']['snapshot'] == snapshot
                and snapshot['request_sha256'] == binding['request_sha256']
                and service[0]['result']['observed_before'] <= snapshot['last_monotonic'] <= service[0]['result']['observed_after']
                and suffix['recorded_end'] == snapshot['last_monotonic']
                and suffix['observed_after'] == service[0]['result']['observed_after']
                and suffix['suffix_upper_seconds'] == snapshot['end_reservation']['bounds']['wall_s_upper']
                and suffix['observed_suffix_seconds_upper'] == suffix['observed_after'] - suffix['recorded_end']
                and 0 <= suffix['observed_suffix_seconds_upper'] <= suffix['suffix_upper_seconds']
                and suffix.get('native_observation') is True,
                'receipt supervisor replay/clock differs from original observed service end')
    return {'schema': RESULT, 'qualification_request_sha256': binding['request_sha256'],
        'qualification_scope_sha256': binding['scope_sha256'], 'arm': selection['arm'],
        'static_recipe_sha256': binding['static_recipe_sha256'], 'source_code_sha256': copy.deepcopy(binding['code_sha256']),
        'runtime': copy.deepcopy(context['bindings']['runtime']), 'anchors': copy.deepcopy(context['bindings']['anchors']),
        'census': copy.deepcopy(prepared['census']), 'prepared_receipt_sha256': prepared['receipt_sha256'],
        'context_ref': copy.deepcopy(refs['context']), 'scope_ref': copy.deepcopy(refs['scope']),
        'receipt_ref': copy.deepcopy(refs['receipt']),
        'finalizer_header_ref': copy.deepcopy(refs['header']), 'finalizer_journal_ref': copy.deepcopy(refs['journal']),
        'finalizer_seal_ref': copy.deepcopy(refs['seal']), 'preclose_ref': copy.deepcopy(refs['preclose']),
        'claim_scope': copy.deepcopy(selection['claim_scope']), 'unit_prices_measured': selection['claim_scope']['unit_prices_measured'],
        'completed_pair_verified': not fixture, 'synthetic_pair_consistent': fixture, 'derived_finalizer_costs': derived,
        'execution_reverified': False, 'production_training_authorized': False,
        'classification': 'source-selected completed mechanics receipt linkage; separate training authority and prices required'}


def verify_completed_qualification(selection, *, reader, outer_preflight):
    require(type(_SOURCE_SELECTION) is MappingProxyType and selection is _SOURCE_SELECTION
            and raw._hex(_SOURCE_SELECTION_SHA256),
            'exact immutable completed qualification source selection required; consumer closed')
    return _verify(dict(selection), reader=reader, outer_preflight=outer_preflight, fixture=False,
                   source_pin=_SOURCE_SELECTION_SHA256)


def verify_fixture_pair(selection, *, reader, outer_preflight):
    """Explicit arithmetic/file fixture; never produces an admitted completion."""
    return _verify(selection, reader=reader, outer_preflight=outer_preflight, fixture=True)
