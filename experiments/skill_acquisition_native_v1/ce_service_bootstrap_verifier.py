"""Paid raw replay of original service costs and clock; never execution proof."""
from __future__ import annotations

import copy
from pathlib import Path

from . import ce_native_services as services, ce_qualification_artifacts as raw
from . import ce_execution_verifier as files
from . import ce_rejected_return as rejected
from . import ce_journal_limits as journal_limits

SCHEMA = 'CE8-SERVICE-BOOTSTRAP-RAW-REPLAY-1'
MAX_HEADER = MAX_LINE = 1024 ** 2
MAX_JOURNAL, MAX_EVENTS = 256 * MAX_LINE, 200000
require, canonical, digest = raw.require, raw.canonical, raw.digest


def _finite(value):
    return type(value) in (int, float) and 0 <= value < float('inf')


def _price(profile):
    services._profile(profile)
    return {'read_bytes': profile['read_bytes_upper'], 'write_bytes': profile['write_bytes_upper'],
            'hash_attempts': profile['identity_hash_attempts_upper'], 'hash_bytes': profile['identity_hash_bytes_upper'],
            'wall_seconds': profile['wall_s_upper']}


def verify_service_bootstrap(context, specification, header_ref, journal_ref, seal_ref=None, *, reader, outer_preflight):
    """Check immutable files against external frozen scope, with no authority grant.

    All reads, hashes and semantic reductions use the independent reader budget.
    A final reservation covers the suffix after the recorded clock end; the
    replay does not infer actual suffix duration or OS process cleanup from it.
    """
    require(type(reader) is raw.RawEvidenceReader and callable(outer_preflight), 'paid independent service reader/preflight')
    require(type(specification) is dict, 'original service specification object')
    event_limit = 400008 if specification.get('journal_limits') is not None else MAX_EVENTS
    refs = [header_ref, journal_ref] + ([] if seal_ref is None else [seal_ref])
    for reference in refs: files._reference(reference)
    require(header_ref['bytes'] <= MAX_HEADER and journal_ref['bytes'] <= MAX_JOURNAL
            and (seal_ref is None or seal_ref['bytes'] <= MAX_HEADER), 'bounded service replay files')
    outer_preflight('service_bootstrap_raw_verifier', {'references': refs, 'reader_bounds': dict(reader.remaining),
        'max_events': event_limit, 'max_line_bytes': MAX_LINE,
        'required_coverage': ['raw read/hash', 'JSON/hash-chain/reducer work', 'verification wall']})
    binding = context['verification_binding']
    with reader._overhead('service_header_and_frozen_scope', work_bytes=128 * MAX_HEADER,
                          hash_attempts=128, hash_bytes=128 * MAX_HEADER):
        raw.binding(binding)
        require(raw._hex(context.get('request_sha256')) and binding['request_sha256'] == context['request_sha256'], 'external logical request binding')
        spec = copy.deepcopy(specification)
        caps = spec['caps']
        require(type(caps) is dict and set(caps) == set(services.COUNTERS)
                and all(type(caps[key]) is int and caps[key] > 0 for key in services.COUNTERS if key != 'wall_seconds')
                and _finite(caps['wall_seconds']) and caps['wall_seconds'] > 0, 'strict original service caps')
        require(type(spec['operation_profiles']) is dict and spec['operation_profiles'], 'original finite operation profiles')
        for profile in spec['operation_profiles'].values(): services._profile(profile)
        services.validate_preparation_policy(spec)
        if spec.get('operation_contracts') is not None:
            services.contracts.validate_manifest(spec['operation_contracts'], spec['operation_profiles'], owner='bootstrap')
        elif spec.get('input_io_row_wall_s_upper') is not None:
            services.reservation_profile(spec,'input_io',{})
        services._profile(spec['setup_bounds'])
        rejection_policy = rejected.policy_from_spec(spec, 'bootstrap')
        limits = journal_limits.limits_from_spec(spec, 'bootstrap')
        if context.get('mode') != 'synthetic_fixture':
            require(spec.get('operation_profile_rule')==services.INPUT_RESERVATION_RULE,'exact native input reservation rule')
            services._profile(spec.get('finalization_bounds'))
            require(_finite(spec.get('physical_wall_seconds_upper')) and spec['physical_wall_seconds_upper'] > 0, 'original native physical deadline')
        header = files._read_json(reader, header_ref, binding, 'service_bootstrap_header')
        origin = header.get('original_origin')
        require(_finite(origin) and header == {'schema': services.BOOTSTRAP, 'request_sha256': context['request_sha256'],
            'specification': spec, 'native': context.get('mode') != 'synthetic_fixture', 'original_origin': origin},
            'original service header/scope/request mismatch')
        require(Path(header_ref['path']) == Path(spec['owned_directory']) / 'SERVICE_BOOTSTRAP_HEADER.json'
                and Path(journal_ref['path']) == Path(spec['owned_directory']) / 'SERVICE_BOOTSTRAP.jsonl'
                and (seal_ref is None or Path(seal_ref['path']) == Path(spec['owned_directory']) / 'SERVICE_BOOTSTRAP_SEAL.json'),
                'exact owned service raw paths')
    totals = dict.fromkeys(services.COUNTERS, 0)
    def charge(values):
        for key, value in values.items(): totals[key] += value
        require(all(totals[key] <= caps[key] for key in services.COUNTERS), 'raw original service cap exhausted')
    charge({'write_bytes': header_ref['bytes'], 'metadata_bytes': header_ref['bytes'],
            'hash_attempts': 1, 'hash_bytes': header_ref['bytes']})
    operations, offsets = {}, []
    head, offset, monotonic, actual_wall, reservation, ended = None, 0, origin, 0.0, None, False
    rejection_ticket, rejection_record, rejection_used, fatal = None, None, False, False
    wrapped = files._decorated(reader, journal_ref, binding, 'service_bootstrap_journal')
    with reader._overhead('service_chain_and_semantic_replay', work_bytes=64 * journal_ref['bytes'],
                          hash_attempts=2 * event_limit, hash_bytes=16 * journal_ref['bytes']):
        stream = reader._blocks(wrapped, binding, 'service_bootstrap_journal'); buffer, primary = b'', None
        try:
            for block in stream:
                buffer += block
                while b'\n' in buffer:
                    split = buffer.index(b'\n') + 1; line, buffer = buffer[:split], buffer[split:]
                    require(len(line) <= MAX_LINE and len(offsets) < event_limit and not ended and not fatal,
                            'bounded original service row; no work after end or rejected return')
                    row = raw._json(line)
                    require(type(row) is dict and set(row) == {'schema', 'seq', 'previous', 'request_sha256', 'kind', 'payload', 'sha256'}, 'exact service event schema')
                    body = {key: value for key, value in row.items() if key != 'sha256'}
                    require(row['schema'] == services.BOOTSTRAP and row['request_sha256'] == context['request_sha256']
                            and type(row['seq']) is int and row['seq'] == len(offsets) and row['previous'] == head
                            and row['sha256'] == digest(body) and canonical(row) + b'\n' == line, 'original service hash/order/request chain')
                    kind, payload = row['kind'], row['payload']; require(type(payload) is dict, 'service payload object')
                    journal_limits.check_append(limits, rows=len(offsets), size=offset,
                        line_bytes=len(line), terminal=kind=='terminal_rejected')
                    charges = {}
                    if not offsets: require(kind == 'setup', 'original service setup first')
                    elif len(offsets) == 1: require(kind == 'origin', 'original service clock origin second')
                    if kind == 'setup':
                        require(not offsets and payload == {'setup_bounds': spec['setup_bounds']}, 'one immutable service setup reservation')
                        charges = _price(spec['setup_bounds'])
                    elif kind == 'origin':
                        require(len(offsets) == 1 and payload == {'monotonic': origin, 'original_origin': origin}, 'one original monotonic origin')
                    elif kind == 'rejection_reservation':
                        require(rejection_policy is not None and len(offsets) == 2 and rejection_ticket is None
                                and payload == rejected.reservation_payload(rejection_policy, services.BOOTSTRAP, header),
                                'one exact prepaid rejection ticket before dependent work')
                        rejection_ticket = copy.deepcopy(payload)
                        charges = services.rejection_service_costs(rejection_policy)
                    elif kind == 'wall':
                        now = payload.get('monotonic'); delta = payload.get('elapsed_delta')
                        require(set(payload) == {'original_origin', 'monotonic', 'elapsed_delta'} and payload['original_origin'] == origin
                                and _finite(now) and now > monotonic and _finite(delta) and delta == now - monotonic
                                and now - origin <= spec.get('physical_wall_seconds_upper', caps['wall_seconds']), 'original service wall interval/deadline')
                        monotonic = now; actual_wall += delta; charges = {'wall_seconds': delta}
                    elif kind == 'operation':
                        require(rejection_policy is None or rejection_ticket is not None,
                                'dependent service work needs its prepaid rejection ticket')
                        require(reservation is None and set(payload) == {'operation', 'phase', 'operation_id', 'fields'} |
                                ({'reservation_profile'} if payload.get('phase')=='started' else set()), 'exact operation reservation; no service work after final reservation')
                        op, phase, ident, fields = (payload[key] for key in ('operation', 'phase', 'operation_id', 'fields'))
                        require(op in spec['operation_profiles'] and type(ident) is str and ident and type(fields) is dict, 'priced exact service operation identity')
                        profile = spec['operation_profiles'][op]
                        if phase == 'started':
                            require(ident not in operations, 'unique service physical operation')
                            profile=services.reservation_profile(spec,op,fields,contract_replay=True)
                            require(payload['reservation_profile']==profile,'started permanent input reservation differs from original declarations/maxima')
                            charges = dict(_price(profile), operation_attempts=1)
                            operations[ident] = {'kind': op, 'phase': phase,'profile':profile,
                                'wall_contract':fields.get('wall_contract'),'start_row':row}
                        else:
                            require(phase in ('returned', 'failed') and ident in operations
                                    and operations[ident]['kind']==op and operations[ident]['phase']=='started', 'original matching service return')
                            profile=operations[ident]['profile']
                            if spec.get('operation_contracts') is not None:
                                require(fields.get('wall_contract') == operations[ident]['wall_contract'],
                                        'terminal service source contract differs from permanent start reservation')
                            services.contracts.check_wall_return(fields, profile)
                            for field, upper in {'bytes_read': 'read_bytes_upper', 'bytes_written': 'write_bytes_upper',
                                'identity_hash_attempts_actual': 'identity_hash_attempts_upper', 'identity_hash_bytes_actual': 'identity_hash_bytes_upper', 'wall_s': 'wall_s_upper'}.items():
                                if field in fields:
                                    value = fields[field]
                                    require((_finite(value) if field == 'wall_s' else type(value) is int and value >= 0)
                                            and value <= profile[upper], 'actual service work exceeds original reservation')
                            operations[ident]['phase'] = phase
                    elif kind == 'terminal_rejected':
                        ident = payload.get('operation_id')
                        require(rejection_policy is not None and rejection_ticket is not None and not rejection_used
                                and reservation is None and ident in operations
                                and operations[ident]['phase'] == 'started',
                                'final rejected return needs the original pending operation and unused prepaid ticket')
                        previous = operations[ident]
                        rejected.validate_terminal_payload(payload, rejection_policy, rejection_ticket,
                            previous['start_row'], previous['profile'],
                            {'request_sha256':context['request_sha256'],
                             'owner_header_sha256':rejected.digest(header)})
                        require(len(line) <= rejection_policy['record_bytes_upper']
                                and len(canonical(body)) <= rejection_policy['hash_bytes_upper'],
                                'rejected service record exceeds its permanent byte/hash ticket')
                        rejection_record = copy.deepcopy(payload)
                        rejection_used = fatal = True
                    elif kind == 'end_reservation':
                        profile = spec.get('finalization_bounds', spec['setup_bounds']); services._profile(profile)
                        lower = offset + header_ref['bytes'] + 3 * MAX_HEADER
                        require(reservation is None and not any(op['phase'] == 'started' for op in operations.values())
                                and payload == {'original_origin': origin, 'monotonic': monotonic, 'prefix_bytes': offset, 'bounds': profile}
                                and profile['read_bytes_upper'] >= lower and profile['write_bytes_upper'] >= MAX_HEADER
                                and profile['identity_hash_attempts_upper'] >= 4 and profile['identity_hash_bytes_upper'] >= lower + 2 * MAX_HEADER,
                                'permanent final service reservation identity/lower bounds')
                        reservation = copy.deepcopy(payload); charges = dict(_price(profile), operation_attempts=2, metadata_bytes=MAX_HEADER)
                    elif kind == 'end':
                        require(reservation is not None and payload == {'original_origin': origin, 'monotonic': monotonic,
                                'reservation_sha256': digest(reservation)}, 'ended service original reservation/clock binding')
                        ended = True
                    else: require(False, 'unknown service raw event')
                    if kind != 'terminal_rejected':
                        for key, value in (('metadata_bytes', len(line)), ('write_bytes', len(line)), ('hash_attempts', 1), ('hash_bytes', len(canonical(body)))):
                            charges[key] = charges.get(key, 0) + value
                    charge(charges); offsets.append({'seq': row['seq'], 'offset': offset, 'bytes': len(line), 'sha256': row['sha256']})
                    offset += len(line); head = row['sha256']
                require(len(buffer) <= MAX_LINE, 'bounded service partial row')
            require(not buffer and len(offsets) >= 2 and offset == journal_ref['bytes'], 'partial service tail or missing original clock; no repair')
            require(rejection_policy is None or rejection_ticket is not None,
                    'opted-in service ledger missing its prepaid rejection ticket')
        except BaseException as error: primary = error; raise
        finally:
            try: stream.close()
            except BaseException as error:
                if primary is not None: raw._accounting(primary, error); raise primary from error
                raise
    snapshot = {'schema': services.BOOTSTRAP, 'request_sha256': context['request_sha256'], 'head': head,
        'cumulative': totals, 'pending': [ident for ident, op in operations.items() if op['phase'] == 'started'],
        'path': str(Path(spec['owned_directory']) / 'SERVICE_BOOTSTRAP.jsonl'), 'fatal': fatal, 'native_qualified': False,
        'original_origin': origin, 'last_monotonic': monotonic, 'actual_wall_seconds': actual_wall, 'clock_ended': ended,
        'end_reservation': reservation, 'failed_operations': [ident for ident, op in operations.items() if op['phase'] == 'failed'],
        **({'rejection_ticket':rejection_ticket,'rejection_used':rejection_used,
            'rejection_record':rejection_record,'rejection_unknown':None}
           if rejection_policy is not None else {}),
        'claim': 'retained service cost/clock evidence; independent raw replay and OS execution verification required'}
    if seal_ref is not None:
        seal = files._read_json(reader, seal_ref, binding, 'service_bootstrap_seal')
        require(ended and not fatal and seal == snapshot, 'service seal refund/status/clock/fatal mismatch')
        require(sum(ref['bytes'] for ref in refs) <= reservation['bounds']['read_bytes_upper'], 'final actual readback exceeds original service reservation')
    return {'schema': SCHEMA, 'status': 'UNQUALIFIED', 'derived_totals': totals, 'derived_snapshot': snapshot,
        'original_offsets': offsets,
        'original_journal_ref': {key:journal_ref[key] for key in ('path','sha256','bytes')},
        'recorded_wall_seconds': actual_wall, 'original_origin': origin,
        'recorded_clock_ended': ended, 'final_suffix_reserved': reservation is not None,
        'suffix_actual_wall_verified': False, 'execution_verified': False, 'native_qualified': False,
        'rejected_return_retained': rejection_used,
        'rejection_publication_wall_verified': False,
        'claim': 'raw cost/clock consistency only; original OS/process execution and final suffix require independent evidence'}
