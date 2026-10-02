"""Independent bounded replay of actual BootstrapAccounting raw evidence.

No constructor, ledger writes, authority issuance, framework or OS observation
is used. Matching reservations proves retained upper costs and ticket linkage;
it does not prove that a publisher/process body executed under that ticket.
"""
from __future__ import annotations

import copy

from . import ce_bootstrap_accounting as bootstrap, ce_qualification_artifacts as raw
from . import ce_execution_verifier as files, ce_qualification as q

SCHEMA = 'CE8-BOOTSTRAP-RAW-REPLAY-1'
MAX_HEADER, MAX_LINE = raw.MAX_JSON, raw.MAX_JSON
MAX_JOURNAL, MAX_EVENTS = 256 * raw.CHUNK, 200000
require, digest, canonical = raw.require, raw.digest, raw.canonical


def _owner(value):
    require(type(value) is dict and set(value) == {'accounting_spec_sha256', 'prepared_receipt_sha256'}
            and all(raw._hex(item) for item in value.values()), 'exact expected dependent owner contract')
    return value


def _worker(proof, context, expected_workers):
    require(type(proof) is dict and proof.get('schema') == 'CE8-QUALIFICATION-WORKER-PROOF-1'
            and raw._hex(proof.get('worker_identity_sha256')) and raw._hex(proof.get('boot_identity_sha256'))
            and proof.get('fresh_process') is True and proof.get('boot_verified') is True,
            'explicit original bootstrap worker proof')
    identity = proof['worker_identity_sha256']
    require(identity in expected_workers and proof == expected_workers[identity], 'foreign bootstrap worker/proof outside frozen expectations')
    if context.get('mode') != 'synthetic_fixture':
        from . import ce_worker_evidence as evidence
        require(proof.get('request_sha256') == context['request_sha256'] and proof.get('scope_sha256') == context['scope_sha256'],
                'bootstrap worker exact native logical request/scope')
        sources = proof.get('sources', {}).get('source_files')
        require(type(sources) is dict and all(type(ref) is dict and raw._hex(ref.get('sha256')) for ref in sources.values())
                and {name: ref['sha256'] for name, ref in sources.items()} == context['bindings']['source']['code_sha256'],
                'bootstrap native worker exact observed source map')
        require(digest(evidence.worker_identity_preimage(proof['os_identity'], proof['sources'],
            context['request_sha256'], context['scope_sha256'])) == identity
            and digest(proof['os_identity']['boot_identity']) == proof['boot_identity_sha256'],
            'bootstrap native OS/source/boot identity preimage')
    return proof


def _shadow(header):
    state = object.__new__(bootstrap.BootstrapAccounting)
    state.spec = copy.deepcopy(header['specification'])
    state.cumulative, state.events, state.tickets = bootstrap._zero(), [], {}
    state.fatal = state.unknown_wall = False
    state.proof, state.last_clock, state.started_clock = None, None, None
    state.actual_wall_seconds, state._self_prepaid = 0.0, False
    state.binding_sha256 = digest(header)
    state.header = copy.deepcopy(header)
    state.closed, state._worker_seen, state._end_reserved, state._end_readback_used = False, False, None, False
    # Original first self_reservation charges the one full self bound. Header
    # bytes are included there; replay adds original serialized row overheads.
    return state


def _transition(state, kind, payload, context, expected_workers, owner_contracts, clocks):
    require(type(payload) is dict, 'original bootstrap event payload object')
    if kind in ('self_reservation', 'inspection'):
        require(set(payload) == {'bounds'}, 'exact original bootstrap setup/inspection schema')
        if kind == 'self_reservation':
            q._costs(payload['bounds'])
            require(payload['bounds'] == state.spec['self_bounds'], 'immutable original self bounds; no repricing')
            require(not clocks['expect_worker'] and not any(ticket['status'] in ('RESERVED', 'STARTED') for ticket in state.tickets.values()),
                    'new original setup cannot conceal pending ticket work')
            clocks['expect_worker'] = 'resume' if state._worker_seen else 'worker'
        else:
            require(state.proof is not None, 'worker evidence precedes inspection')
            require(type(payload['bounds']) is dict and set(payload['bounds']) == set(bootstrap.COST_FIELDS)
                    and all(q.finite(value) and (key == 'wall_seconds_upper' or type(value) is int)
                            for key, value in payload['bounds'].items()), 'finite source inspection cost amounts')
            size = clocks['journal_offset'] + clocks['header_bytes'] + MAX_LINE
            expected = dict(bootstrap._zero(), operation_attempts_upper=1,
                io_read_bytes_upper=size, identity_hash_attempts_upper=2, identity_hash_bytes_upper=size)
            require(payload['bounds'] == expected, 'original source inspection reservation formula')
    elif kind == 'worker':
        require(set(payload) == {'proof', 'monotonic'} and q.finite(payload['monotonic']), 'exact bootstrap worker clock schema')
        require(clocks['expect_worker'] == 'worker' and not state._worker_seen,
                'one original worker starts after its setup reservation')
        clocks['expect_worker'] = False
        proof = _worker(payload['proof'], context, expected_workers)
        clocks['first_origin'] = payload['monotonic']
        state.proof = copy.deepcopy(proof)
    elif kind == 'resume':
        require(set(payload) == {'worker_identity_sha256', 'boot_identity_sha256', 'monotonic', 'original_origin'}
                and clocks['expect_worker'] == 'resume', 'exact source resume after immutable setup bounds')
        clocks['expect_worker'] = False
    elif kind == 'wall':
        require(set(payload) == {'boot_identity_sha256', 'worker_identity_sha256', 'monotonic', 'elapsed_delta'}
                and state.proof is not None and payload['worker_identity_sha256'] == state.proof['worker_identity_sha256']
                and payload['boot_identity_sha256'] == state.proof['boot_identity_sha256']
                and q.finite(payload['monotonic']) and q.finite(payload['elapsed_delta'])
                and payload['monotonic'] >= state.last_clock
                and payload['elapsed_delta'] == payload['monotonic'] - state.last_clock,
                'exact same-worker/boot monotonic bootstrap wall interval')
    elif kind == 'reserve':
        require(set(payload) == {'ticket_id', 'owner', 'bounds'} and state.proof is not None
                and type(payload['ticket_id']) is str and 0 < len(payload['ticket_id']) <= 128,
                'exact unique original bootstrap ticket reservation')
        _owner(payload['owner']); q._costs(payload['bounds'])
        matching = [contract for contract in owner_contracts if contract['owner'] == payload['owner']]
        require(len(matching) == 1 and matching[0]['bounds'] == payload['bounds'], 'foreign dependent owner or repriced ticket bounds')
    elif kind in ('setup_started', 'setup_returned', 'setup_failed'):
        required = {'ticket_id', 'monotonic', 'worker_identity_sha256'}
        require(set(payload) == required | ({'error'} if kind == 'setup_failed' else set())
                and state.proof is not None and payload.get('ticket_id') in state.tickets,
                'exact retained original bootstrap ticket transition')
        if kind == 'setup_failed':
            error = payload['error']
            require(type(error) is dict and set(error) == {'type', 'message'}
                    and type(error['type']) is str and 0 < len(error['type']) <= 128
                    and type(error['message']) is str and len(error['message']) <= 1000,
                    'failed dependent body retains its specific error')
    elif kind == 'interrupted':
        require(set(payload) == {'worker_identity_sha256', 'boot_identity_sha256', 'last_recorded_monotonic',
            'observed_monotonic', 'pending_ticket_ids', 'pending_end_reservation_sha256'}, 'exact retained source interruption schema')
    elif kind == 'end_reservation':
        require(set(payload) == {'bounds', 'prefix_bytes', 'worker_identity_sha256', 'boot_identity_sha256', 'original_origin', 'monotonic'}
                and type(payload['prefix_bytes']) is int and payload['prefix_bytes'] == clocks['journal_offset'],
                'exact prepaid final record/source prefix reservation')
        q._costs(payload['bounds'])
    elif kind == 'end':
        require(set(payload) == {'worker_identity_sha256', 'boot_identity_sha256', 'original_origin', 'monotonic', 'reservation_sha256'},
                'exact final source clock end schema')
    else: raise raw.EvidenceRejected('unsupported bootstrap event kind/schema')
    if 'monotonic' in payload:
        require(payload['monotonic'] - clocks['first_origin'] <= state.spec['physical_wall_seconds_upper'],
                'bootstrap scope physical deadline from original first worker')
    # Same source transition implementation, never its constructor or _event.
    state._apply(kind, payload)


def _evidence(value, header, header_sha, state):
    require(type(value) is dict and set(value) == {'schema', 'native', 'ticket_id', 'bootstrap_binding_sha256',
        'reservation_sha256', 'owner', 'bounds', 'status'} and value['schema'] == bootstrap.SCHEMA,
        'unsupported outer evidence; exact actual PrepaymentHandle schema required')
    require(value['native'] is header['native'] and value['bootstrap_binding_sha256'] == header_sha
            and value['ticket_id'] in state.tickets, 'outer evidence belongs to exact original bootstrap ledger')
    ticket = state.tickets[value['ticket_id']]
    require(all(value[key] == ticket[key] for key in ('reservation_sha256', 'owner', 'bounds')),
            'outer prepayment owner/reservation/status drift')
    status_for = {'reserve': 'RESERVED', 'setup_started': 'STARTED', 'setup_returned': 'RETURNED', 'setup_failed': 'FAILED'}
    observed = {status_for[row['kind']] for row in state.events if row['kind'] in status_for
                and row['payload'].get('ticket_id') == value['ticket_id']}
    require(value['status'] in observed, 'outer prepayment status never occurred in original ticket lifecycle')
    return {'ticket_id': value['ticket_id'], 'observed_status': value['status'], 'terminal_status': ticket['status'],
            'owner': copy.deepcopy(ticket['owner']), 'artifact_birth_order_verified': False,
            'full_upper_reservation_retained': True, 'body_operation_actuals_verified': False}


def verify_bootstrap(context, header_ref, journal_ref, seal_ref=None, *, reader, outer_preflight,
                     expected_workers, expected_owners, outer_evidence=()):
    """Read raw bytes only after external preflight and permanent local bounds.

    expected_owners is a finite list of {owner:<two hashes>,bounds:<seven costs>}
    from frozen caller scope. expected_workers binds original full raw proofs;
    source-owned OS verification is deliberately a separate unimplemented gate.
    """
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight), 'bounded reader and independent outer preflight')
    refs = [header_ref, journal_ref] + ([] if seal_ref is None else [seal_ref])
    for ref in refs: files._reference(ref)
    require(header_ref['bytes'] <= MAX_HEADER and journal_ref['bytes'] <= MAX_JOURNAL
            and (seal_ref is None or seal_ref['bytes'] <= MAX_HEADER), 'finite bootstrap raw file byte ceilings')
    outer_preflight('bootstrap_raw_verifier', {'references': refs, 'reader_bounds': dict(reader.remaining),
        'max_events': MAX_EVENTS, 'max_line_bytes': MAX_LINE,
        'required_coverage': ['original IO/hash', 'JSON/chain/reducer work', 'metadata', 'elapsed verification wall']})
    binding = context['verification_binding']
    with reader._overhead('bootstrap_header_and_expected_scope', work_bytes=128 * MAX_HEADER,
                          hash_attempts=128, hash_bytes=128 * MAX_HEADER):
        raw.binding(binding)
        require(type(expected_workers) is dict and 0 < len(expected_workers) <= 16
                and all(raw._hex(key) for key in expected_workers), 'finite original expected bootstrap worker map')
        require(type(expected_owners) is list and 0 < len(expected_owners) <= 128, 'finite expected dependent owner contracts')
        for contract in expected_owners:
            require(type(contract) is dict and set(contract) == {'owner', 'bounds'}, 'exact frozen owner/bounds contract')
            _owner(contract['owner']); q._costs(contract['bounds'])
        require(len({digest(contract['owner']) for contract in expected_owners}) == len(expected_owners), 'distinct frozen owner contracts')
        spec = bootstrap.bootstrap_specification(context)
        q._costs(spec['caps']); q._costs(spec['self_bounds'])
        require(spec.get('schema') == bootstrap.SPEC and q.finite(spec.get('physical_wall_seconds_upper'), True),
                'actual bounded bootstrap scope specification')
        header = files._read_json(reader, header_ref, binding, 'bootstrap_header')
        expected = {'schema': bootstrap.SCHEMA, 'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'arm': context['arm'], 'specification': spec, 'source_code_sha256': context['bindings']['source'],
            'native': context.get('schema') != bootstrap.FIXTURE}
        require(header == expected, 'original bootstrap header/request/scope/source/specification drift')
        require(type(outer_evidence) in (tuple, list) and len(outer_evidence) <= 128, 'finite exact outer ticket evidence list')
        state, clocks = _shadow(header), {'first_origin': None, 'gaps': [], 'expect_worker': False,
            'journal_offset': 0, 'header_bytes': header_ref['bytes']}
        header_sha = digest(header)
    offsets, offset, self_count = [], 0, 0
    wrapped = files._decorated(reader, journal_ref, binding, 'bootstrap_journal')
    with reader._overhead('bootstrap_original_chain_and_transitions', work_bytes=64 * journal_ref['bytes'],
                          hash_attempts=2 * MAX_EVENTS, hash_bytes=16 * journal_ref['bytes']):
        stream = reader._blocks(wrapped, binding, 'bootstrap_journal'); buffer, primary = b'', None
        try:
            for block in stream:
                buffer += block
                while b'\n' in buffer:
                    end = buffer.index(b'\n') + 1; line, buffer = buffer[:end], buffer[end:]
                    require(len(line) <= MAX_LINE and len(state.events) < MAX_EVENTS, 'bounded original bootstrap journal row')
                    row = raw._json(line)
                    require(type(row) is dict and set(row) == {'seq', 'previous', 'binding_sha256', 'kind', 'payload', 'sha256'}, 'exact original bootstrap row schema')
                    body = {key: value for key, value in row.items() if key != 'sha256'}
                    require(type(row['seq']) is int and row['seq'] == len(state.events) and row['previous'] == state._head()
                            and row['binding_sha256'] == header_sha and digest(body) == row['sha256']
                            and canonical(row) + b'\n' == line, 'original bootstrap canonical hash/order/offset chain')
                    if not state.events: require(row['kind'] == 'self_reservation', 'original full self reservation must be first')
                    overhead = {'io_write_bytes_upper': len(line), 'metadata_bytes_upper': len(line),
                        'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))}
                    state._charge(overhead)
                    require(not clocks['expect_worker'] or row['kind'] == clocks['expect_worker']
                            or clocks['expect_worker'] == 'resume' and row['kind'] == 'wall',
                            'setup reservation must be followed by its original worker or bridged resume')
                    _transition(state, row['kind'], row['payload'], context, expected_workers, expected_owners, clocks)
                    state.events.append(row)
                    offsets.append({'seq': row['seq'], 'offset': offset, 'bytes': len(line), 'sha256': row['sha256']})
                    offset += len(line); self_count += row['kind'] == 'self_reservation'
                    clocks['journal_offset'] = offset
                require(len(buffer) <= MAX_LINE, 'oversized partial bootstrap journal row')
            require(not buffer and not clocks['expect_worker'] and offset == journal_ref['bytes'] and self_count >= 1 and state.proof is not None,
                    'partial bootstrap tail or missing original reservation/worker; no repair')
        except BaseException as error: primary = error; raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise
    with reader._overhead('bootstrap_seal_and_outer_linkage', work_bytes=64 * MAX_HEADER,
                          hash_attempts=256, hash_bytes=64 * MAX_HEADER):
        if seal_ref is not None:
            seal = files._read_json(reader, seal_ref, binding, 'bootstrap_seal')
            require(seal == state.snapshot(), 'bootstrap seal refund/ticket/proof/clock totals differ from original replay')
        matched = [_evidence(value, header, header_sha, state) for value in outer_evidence]
    statuses = {name: [key for key, ticket in state.tickets.items() if ticket['status'] == name]
                for name in ('RESERVED', 'STARTED', 'RETURNED', 'FAILED')}
    gaps = ['actual source-owned OS/entry provenance has not been verified',
        'source/OS terminal evidence must verify actual final record/readback suffix within prepaid wall/deadline',
        'publisher/process inner operation actuals are not BootstrapAccounting events; only their matching full ticket reservations are retained']
    if not state.closed: gaps.append('source clock end has not been retained')
    if state.unknown_wall: gaps.append('explicit source interruption leaves unknown clock/completion; retained reservations cannot be reset')
    if statuses['RESERVED'] or statuses['STARTED']: gaps.append('unfinished dependent work retains full prepaid upper costs; completion remains unknown')
    return {'schema': SCHEMA, 'status': 'UNQUALIFIED', 'native_qualified': False, 'execution_verified': False,
        'binding_sha256': header_sha, 'ledger_head': state._head(), 'events': len(state.events),
        'original_offsets': offsets, 'derived_totals': copy.deepcopy(state.cumulative),
        'derived_tickets': copy.deepcopy(state.tickets), 'ticket_statuses': statuses,
        'recorded_wall_seconds': state.actual_wall_seconds, 'first_monotonic': clocks['first_origin'],
        'last_monotonic': state.last_clock, 'worker_origin_gaps': clocks['gaps'],
        'clock_coverage_complete': state.closed and not state.unknown_wall and not state.fatal,
        'physical_final_verified': False, 'end_reservation': copy.deepcopy(state._end_reserved), 'seal_consistent': seal_ref is not None,
        'matched_outer_evidence': matched, 'activation_gaps': gaps}
