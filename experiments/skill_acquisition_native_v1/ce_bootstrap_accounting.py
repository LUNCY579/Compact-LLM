"""Durable outer prepayment primitives. Native issuer/verifier registries are empty.

This source creates no execution authority. Fixture handles are inspectable and
explicitly non-native. A reviewed native issuer must own a permanent reservation
before any dependent qualification constructor hashes or reads its files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from . import ce_qualification as q, ce_native_authority as authority

SCHEMA = 'CE8-BOOTSTRAP-ACCOUNTING-1'
SPEC = 'CE8-BOOTSTRAP-SPEC-1'
HEADER, JOURNAL = 'BOOTSTRAP_BINDINGS.json', 'BOOTSTRAP_RESERVATIONS.jsonl'
MAX_LINE, CHUNK = 4 * 1024 ** 2, 1024 ** 2
COST_FIELDS = q.COST_FIELDS
FIXTURE = 'CE8-QUALIFICATION-ACCOUNTING-FIXTURE-1'
_REGISTERED_NATIVE_ISSUERS = frozenset()
_REGISTERED_PREPAYMENT_VERIFIERS = frozenset()
_SEAL = object()
canonical, digest, require = q.canonical, q.digest, q.require


def _clone(value): return json.loads(canonical(value))
def _add(a, b): return {key: a.get(key, 0) + b.get(key, 0) for key in COST_FIELDS}
def _zero(): return dict.fromkeys(COST_FIELDS, 0)


def bootstrap_specification(context):
    """Derive an independent child directory under unchanged common prices."""
    scope = context['bindings']['scope']; common = scope['qualifier_bootstrap']
    child = context.get('checkpoint_child_bootstrap')
    if child is None: return _clone(common)
    declaration = context.get('checkpoint_child_declaration')
    require(type(declaration) is dict and declaration.get('branch') in ('uninterrupted', 'prefix', 'resumed'),
            'child bootstrap needs its source-bound branch declaration')
    root = Path(scope['checkpoint_qualification']['owned_directory'])
    require(root.is_absolute(), 'absolute common checkpoint bootstrap owner')
    expected = dict(common, owned_directory=str(root / 'children' / declaration['branch'] / 'bootstrap'))
    require(type(child) is dict and child == expected,
            'child bootstrap may change only exact directory, never common caps/prices/deadline')
    return _clone(expected)


class PrepaymentHandle:
    __slots__ = ('_seal', '_ledger', '_ticket', '_native')
    def __init__(self, ledger, ticket, *, seal):
        require(seal is _SEAL, 'prepayment handles are source issued only')
        object.__setattr__(self, '_seal', seal)
        object.__setattr__(self, '_ledger', ledger)
        object.__setattr__(self, '_ticket', ticket)
        object.__setattr__(self, '_native', not ledger.fixture)
    def __setattr__(self, key, value): raise ValueError('prepayment handle is immutable')
    @property
    def evidence(self):
        require(self._seal is _SEAL and self._ticket in self._ledger.tickets, 'unknown prepayment ticket')
        ticket = self._ledger.tickets[self._ticket]
        return {'schema': SCHEMA, 'native': self._native, 'ticket_id': self._ticket,
            'bootstrap_binding_sha256': self._ledger.binding_sha256,
            'reservation_sha256': ticket['reservation_sha256'], 'owner': _clone(ticket['owner']),
            'bounds': _clone(ticket['bounds']), 'status': ticket['status']}
    def begin(self): self._ledger.ticket_event(self, 'setup_started')
    def complete(self): self._ledger.ticket_event(self, 'setup_returned')
    def fail(self, error): self._ledger.ticket_event(self, 'setup_failed', error)


def require_registered_verifier(verifier):
    require(any(verifier is value for value in _REGISTERED_PREPAYMENT_VERIFIERS),
            'native prepayment verifier is unregistered; dependent constructor denied')


def verify_prepayment(handle, context, proof, owner, *, verifier=None):
    fixture = type(context) is dict and context.get('schema') == FIXTURE
    if not fixture:
        # Refuse before touching callback objects, context hashes or paths.
        require_registered_verifier(verifier)
    require(type(handle) is PrepaymentHandle and getattr(handle, '_seal', None) is _SEAL, 'source-owned opaque prepayment required')
    require(handle._native is (not fixture), 'fixture prepayment cannot admit native work')
    ledger, evidence = handle._ledger, handle.evidence
    require(ledger.request_sha256 == context.get('request_sha256')
            and ledger.scope_sha256 == context.get('scope_sha256') and ledger.arm == context.get('arm')
            and ledger.proof == proof and evidence['owner'] == owner
            and evidence['status'] == 'RESERVED' and not ledger.fatal and not ledger.unknown_wall and not ledger.closed,
            'prepayment request/scope/arm/worker/owner/status drift')
    if not fixture:
        require(verifier(handle, context, proof, owner) is True, 'registered prepayment evidence verification failed')
    return handle


class BootstrapAccounting:
    def __init__(self, context, permit, proof, *, clock=None, native_issuer=None):
        self.fixture = type(context) is dict and context.get('schema') == FIXTURE
        if self.fixture:
            require(context.get('mode') == 'synthetic_fixture' and context.get('approved') is False
                    and context.get('training_authorized') is False and permit is None, 'explicit bootstrap fixture only')
        else:
            require(any(native_issuer is value for value in _REGISTERED_NATIVE_ISSUERS), 'native bootstrap issuer unregistered')
            require(clock is None, 'native monotonic clock cannot be injected')
            authority.require_execution_permit(permit, context, 'journal_construct')
        self.clock = time.monotonic if clock is None else clock
        require(callable(self.clock), 'monotonic clock required')
        started = self.clock()
        require(q.finite(started), 'finite bootstrap monotonic origin')
        require(type(proof) is dict and proof.get('fresh_process') is True and proof.get('boot_verified') is True
                and q.hex64(proof.get('boot_identity_sha256')) and q.hex64(proof.get('worker_identity_sha256')), 'source-bound worker/boot proof required')
        if not self.fixture: require(context.get('worker_proof_sha256') == digest(proof), 'bootstrap worker evidence drift')
        spec = bootstrap_specification(context)
        require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit bootstrap specification required')
        q._costs(spec.get('caps')); q._costs(spec.get('self_bounds'))
        require(type(spec.get('owned_directory')) is str and Path(spec['owned_directory']).is_absolute(), 'absolute bootstrap owner')
        require(q.finite(spec.get('physical_wall_seconds_upper'), True), 'explicit physical bootstrap deadline')
        self.context, self.permit, self.proof, self.spec = context, permit, _clone(proof), _clone(spec)
        self.request_sha256, self.scope_sha256, self.arm = context['request_sha256'], context['scope_sha256'], context['arm']
        require(q.hex64(self.request_sha256) and q.hex64(self.scope_sha256) and self.arm in ('canonical', 'permuted'), 'one exact bootstrap identity')
        self.cumulative, self.events, self.tickets = _zero(), [], {}
        self.fatal = self.unknown_wall = False
        self.closed, self._worker_seen, self._end_reserved, self._end_readback_used = False, False, None, False
        self.actual_wall_seconds, self.last_clock, self.started_clock = 0.0, started, started
        self.directory = Path(spec['owned_directory']).resolve()
        require(not Path(spec['owned_directory']).is_symlink(), 'regular bootstrap directory')
        self.path, self.header_path = self.directory / JOURNAL, self.directory / HEADER
        self.header = {'schema': SCHEMA, 'request_sha256': self.request_sha256, 'scope_sha256': self.scope_sha256,
            'arm': self.arm, 'specification': self.spec, 'source_code_sha256': _clone(context['bindings']['source']),
            'native': not self.fixture}
        self.binding_sha256 = digest(self.header)
        raw_header = canonical(self.header) + b'\n'; require(len(raw_header) <= MAX_LINE, 'bootstrap header bound')
        self._known_size = 0
        self._self_prepaid = False
        if self.path.exists() or self.header_path.exists():
            require(self.path.is_file() and self.header_path.is_file() and not self.path.is_symlink()
                    and not self.header_path.is_symlink(), 'regular paired bootstrap history required')
            size = self.path.stat().st_size + self.header_path.stat().st_size
            require(size <= spec['self_bounds']['io_read_bytes_upper'] and size <= spec['self_bounds']['identity_hash_bytes_upper']
                    and spec['self_bounds']['identity_hash_attempts_upper'] >= size // 200 + 1, 'prepaid bounded bootstrap replay')
            with self.header_path.open('rb') as stream: saved = stream.read(MAX_LINE + 1)
            require(saved == raw_header, 'bootstrap binding drift')
            count = len(saved)
            with self.path.open('rb') as stream:
                while raw := stream.readline(MAX_LINE + 1):
                    count += len(raw); require(count <= size and len(raw) <= MAX_LINE and raw.endswith(b'\n'), 'bootstrap corrupt/growing tail')
                    entry = json.loads(raw); body = {key: value for key, value in entry.items() if key != 'sha256'}
                    require(canonical(entry) + b'\n' == raw and body['seq'] == len(self.events)
                            and body['previous'] == self._head() and body['binding_sha256'] == self.binding_sha256
                            and digest(body) == entry['sha256'], 'bootstrap hash/order drift')
                    overhead = {'io_write_bytes_upper': len(raw), 'metadata_bytes_upper': len(raw),
                                'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))}
                    self._charge(overhead)
                    self._apply(body['kind'], body['payload']); self.events.append(entry); self._known_size += len(raw)
            require(count == size, 'bootstrap replay byte drift')
            if not self.closed and not self.unknown_wall and (self._end_reserved is not None or any(ticket['status'] in ('STARTED', 'RESERVED') for ticket in self.tickets.values())):
                self.unknown_wall = True
                self._event('interrupted', {'worker_identity_sha256': self.proof['worker_identity_sha256'],
                    'boot_identity_sha256': self.proof['boot_identity_sha256'], 'last_recorded_monotonic': self.last_clock,
                    'observed_monotonic': started, 'pending_ticket_ids': sorted(key for key, ticket in self.tickets.items()
                        if ticket['status'] in ('STARTED', 'RESERVED')),
                    'pending_end_reservation_sha256': digest(self._end_reserved) if self._end_reserved is not None else None}, recovery=True)
            if self.closed: self._end_readback_used = True
        else:
            # Self-construction is admitted from the separately source-bound
            # outer specification, before its owned files can exist.
            require(spec['self_bounds']['io_write_bytes_upper'] >= len(raw_header)
                    and spec['self_bounds']['metadata_bytes_upper'] >= len(raw_header), 'bootstrap self-write bounds')
            self._charge(spec['self_bounds']); self._self_prepaid = True
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.header_path.open('xb') as stream:
                require(stream.write(raw_header) == len(raw_header), 'short bootstrap header'); stream.flush(); os.fsync(stream.fileno())
            with self.path.open('xb') as stream: stream.flush(); os.fsync(stream.fileno())
        if not self.unknown_wall and not self.closed:
            self._event('self_reservation', {'bounds': spec['self_bounds']})
            if not self._worker_seen:
                self._event('worker', {'proof': self.proof, 'monotonic': started})
                self.tick_wall()
            else:
                # Replay never changes the original origin. Bridge ALL elapsed
                # same-worker time, including constructor/replay work.
                self.tick_wall()
                self._event('resume', {'worker_identity_sha256': self.proof['worker_identity_sha256'],
                    'boot_identity_sha256': self.proof['boot_identity_sha256'], 'monotonic': self.last_clock,
                    'original_origin': self.started_clock})

    def _head(self): return self.events[-1]['sha256'] if self.events else None
    def _check(self, amount, *, recovery=False):
        require(not self.fatal and not self.closed and (not self.unknown_wall or recovery), 'bootstrap ended, fatal or interrupted unknown wall')
        for key, value in amount.items():
            require(key in COST_FIELDS and q.finite(value) and (key == 'wall_seconds_upper' or type(value) is int), 'finite bootstrap reservation')
            require(self.cumulative[key] + value <= self.spec['caps'][key], 'bootstrap cumulative ceiling: ' + key)
    def _charge(self, amount, *, recovery=False): self._check(amount, recovery=recovery); self.cumulative = _add(self.cumulative, amount)
    def _end_bounds(self, prefix_bytes):
        # Permanent suffix wall upper and one readback are declared before the
        # final row. Its actual physical duration needs external source/OS proof.
        read = len(canonical(self.header)) + 1 + prefix_bytes + 2 * MAX_LINE
        return dict(_zero(), wall_seconds_upper=self.spec['self_bounds']['wall_seconds_upper'],
            operation_attempts_upper=2, io_read_bytes_upper=read, io_write_bytes_upper=MAX_LINE,
            identity_hash_attempts_upper=4, identity_hash_bytes_upper=read + 2 * MAX_LINE,
            metadata_bytes_upper=MAX_LINE)
    def _apply(self, kind, payload):
        if kind == 'self_reservation':
            require(self._end_reserved is None and payload['bounds'] == self.spec['self_bounds'], 'immutable setup before final reservation')
            if not self._self_prepaid: self._charge(payload['bounds'])
            self._self_prepaid = False
        elif kind == 'inspection':
            require(self._end_reserved is None, 'inspection must precede final reservation')
            self._charge(payload['bounds'])
        elif kind == 'worker':
            require(not self._worker_seen and payload['proof'] == self.proof and q.finite(payload['monotonic']),
                    'one exact original bootstrap worker/origin')
            self.last_clock = payload['monotonic']; self.started_clock = payload['monotonic']
            self._worker_seen = True
        elif kind == 'resume':
            require(self._worker_seen and self._end_reserved is None and payload['original_origin'] == self.started_clock
                    and payload['worker_identity_sha256'] == self.proof['worker_identity_sha256']
                    and payload['boot_identity_sha256'] == self.proof['boot_identity_sha256']
                    and payload['monotonic'] == self.last_clock, 'same-worker resume preserves original clock/deadline')
        elif kind == 'wall':
            require(self._worker_seen and payload['boot_identity_sha256'] == self.proof['boot_identity_sha256']
                    and payload['worker_identity_sha256'] == self.proof['worker_identity_sha256']
                    and q.finite(payload['monotonic']) and payload['monotonic'] >= self.last_clock
                    and payload['elapsed_delta'] == payload['monotonic'] - self.last_clock
                    and payload['monotonic'] - self.started_clock <= self.spec['physical_wall_seconds_upper'], 'same-boot monotonic wall evidence')
            self._charge({'wall_seconds_upper': payload['elapsed_delta']})
            self.last_clock = payload['monotonic']; self.actual_wall_seconds += payload['elapsed_delta']
        elif kind == 'reserve':
            require(self._worker_seen and self._end_reserved is None, 'reserve only inside open worker lifecycle')
            self._charge(payload['bounds'])
            require(payload['ticket_id'] not in self.tickets, 'unique bootstrap ticket')
            self.tickets[payload['ticket_id']] = dict(payload, status='RESERVED', reservation_sha256=digest(payload))
        elif kind in ('setup_started', 'setup_returned', 'setup_failed'):
            require(self._end_reserved is None, 'dependent work cannot run after final reservation')
            ticket = self.tickets[payload['ticket_id']]
            require(payload.get('worker_identity_sha256') == self.proof['worker_identity_sha256']
                    and q.finite(payload.get('monotonic')) and payload['monotonic'] == self.last_clock,
                    'exact prepaid lifecycle worker/monotonic point')
            require(ticket['status'] == ('RESERVED' if kind == 'setup_started' else 'STARTED'), 'one dependent setup lifecycle')
            ticket['status'] = {'setup_started': 'STARTED', 'setup_returned': 'RETURNED', 'setup_failed': 'FAILED'}[kind]
            ticket['result'] = payload
        elif kind == 'interrupted':
            require(self._worker_seen and payload['worker_identity_sha256'] == self.proof['worker_identity_sha256']
                    and payload['boot_identity_sha256'] == self.proof['boot_identity_sha256']
                    and payload['last_recorded_monotonic'] == self.last_clock
                    and q.finite(payload['observed_monotonic']) and payload['observed_monotonic'] >= self.last_clock
                    and payload['pending_ticket_ids'] == sorted(key for key, ticket in self.tickets.items()
                        if ticket['status'] in ('RESERVED', 'STARTED'))
                    and payload['pending_end_reservation_sha256'] == (digest(self._end_reserved) if self._end_reserved is not None else None)
                    and (payload['pending_ticket_ids'] or self._end_reserved is not None),
                    'retained explicit interruption with unchanged pending reservations')
            self.unknown_wall = True
        elif kind == 'end_reservation':
            require(self._worker_seen and self._end_reserved is None and not any(ticket['status'] in ('RESERVED', 'STARTED') for ticket in self.tickets.values())
                    and payload['worker_identity_sha256'] == self.proof['worker_identity_sha256']
                    and payload['boot_identity_sha256'] == self.proof['boot_identity_sha256']
                    and payload['original_origin'] == self.started_clock and payload['monotonic'] == self.last_clock
                    and payload['bounds'] == self._end_bounds(payload['prefix_bytes']), 'prepaid exact final clock record reservation')
            self._charge(payload['bounds']); self._end_reserved = _clone(payload)
        elif kind == 'end':
            require(self._end_reserved is not None and self._worker_seen
                    and payload['worker_identity_sha256'] == self.proof['worker_identity_sha256']
                    and payload['boot_identity_sha256'] == self.proof['boot_identity_sha256']
                    and payload['original_origin'] == self.started_clock and payload['monotonic'] == self.last_clock
                    and payload['reservation_sha256'] == digest(self._end_reserved)
                    and not any(ticket['status'] in ('RESERVED', 'STARTED') for ticket in self.tickets.values()),
                    'exact source end after all terminal dependent tickets')
            self.closed = True
        else: raise ValueError('unknown bootstrap event')
    def _event(self, kind, payload, *, recovery=False):
        body = {'seq': len(self.events), 'previous': self._head(), 'binding_sha256': self.binding_sha256,
                'kind': kind, 'payload': _clone(payload)}
        raw = canonical(dict(body, sha256=digest(body))) + b'\n'; require(len(raw) <= MAX_LINE, 'bootstrap event ceiling')
        overhead = {'io_write_bytes_upper': len(raw), 'metadata_bytes_upper': len(raw),
                    'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': len(canonical(body))}
        amount = payload['bounds'] if kind in ('inspection', 'reserve', 'end_reservation') or kind == 'self_reservation' and not self._self_prepaid else {'wall_seconds_upper': payload['elapsed_delta']} if kind == 'wall' else {}
        self._check(_add(overhead, amount), recovery=recovery)
        # Validate the complete transition on a pure independent state before
        # persisting anything. Invalid lifecycle rows must never poison replay.
        shadow = object.__new__(BootstrapAccounting)
        for key in ('cumulative', 'spec', 'proof', 'tickets'):
            setattr(shadow, key, _clone(getattr(self, key)))
        shadow.header = _clone(self.header)
        shadow._end_reserved = _clone(self._end_reserved)
        for key in ('fatal', 'unknown_wall', 'closed', '_worker_seen', 'last_clock', 'started_clock', 'actual_wall_seconds', '_self_prepaid'):
            setattr(shadow, key, getattr(self, key))
        shadow._charge(overhead, recovery=recovery); shadow._apply(kind, payload)
        self._charge(overhead, recovery=recovery)
        try:
            require(self.path.stat().st_size == self._known_size, 'bootstrap single writer history')
            with self.path.open('ab') as stream:
                require(stream.write(raw) == len(raw), 'short bootstrap event'); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            self.fatal = True; raise
        self._apply(kind, payload); self.events.append(json.loads(raw)); self._known_size += len(raw)
    def tick_wall(self):
        require(self._worker_seen and not self.closed and not self.unknown_wall, 'open bootstrap worker clock required; ended or unknown wall cannot reset')
        now = self.clock()
        require(q.finite(now) and now >= self.last_clock, 'monotonic bootstrap clock')
        require(now - self.started_clock <= self.spec['physical_wall_seconds_upper'], 'bootstrap physical deadline')
        if now > self.last_clock:
            self._event('wall', {'boot_identity_sha256': self.proof['boot_identity_sha256'],
                'worker_identity_sha256': self.proof['worker_identity_sha256'], 'monotonic': now, 'elapsed_delta': now - self.last_clock})
    def reserve_setup(self, owner, bounds):
        self.tick_wall(); q._costs(bounds)
        require(type(owner) is dict and set(owner) == {'accounting_spec_sha256', 'prepared_receipt_sha256'}
                and all(q.hex64(value) for value in owner.values()), 'exact dependent owner contract')
        ticket_id = uuid.uuid4().hex
        self._event('reserve', {'ticket_id': ticket_id, 'owner': _clone(owner), 'bounds': _clone(bounds)})
        return PrepaymentHandle(self, ticket_id, seal=_SEAL)
    def ticket_event(self, handle, kind, error=None):
        require(type(handle) is PrepaymentHandle and handle._ledger is self and handle._seal is _SEAL, 'owned bootstrap handle')
        require(handle._ticket in self.tickets and kind in ('setup_started', 'setup_returned', 'setup_failed')
                and self.tickets[handle._ticket]['status'] == ('RESERVED' if kind == 'setup_started' else 'STARTED'),
                'one dependent setup lifecycle')
        self.tick_wall()
        payload = {'ticket_id': handle._ticket, 'monotonic': self.last_clock, 'worker_identity_sha256': self.proof['worker_identity_sha256']}
        if error is not None: payload['error'] = {'type': type(error).__name__, 'message': str(error)[:1000]}
        self._event(kind, payload)
    def snapshot(self):
        failed = sorted(key for key, ticket in self.tickets.items() if ticket['status'] == 'FAILED')
        return {'schema': SCHEMA, 'native_qualified': False, 'binding_sha256': self.binding_sha256,
            'ledger_head': self._head(), 'cumulative': _clone(self.cumulative), 'actual_wall_seconds': self.actual_wall_seconds,
            'proof': _clone(self.proof), 'tickets': _clone(self.tickets), 'fatal': self.fatal, 'unknown_wall': self.unknown_wall,
            'original_origin': self.started_clock, 'last_monotonic': self.last_clock, 'clock_ended': self.closed,
            'clock_recorded_end': self.closed, 'failed_ticket_ids': failed,
            'body_completed': self.closed and not self.fatal and not self.unknown_wall and bool(self.tickets) and not failed,
            'outcome': 'FAILED' if failed else 'RECORDED_END' if self.closed else 'INCOMPLETE',
            'end_reservation': _clone(self._end_reserved), 'complete_recorded_clock': self.closed and not self.fatal and not self.unknown_wall}
    def end(self):
        require(not self.closed and not self.unknown_wall and not self.fatal
                and not any(ticket['status'] in ('RESERVED', 'STARTED') for ticket in self.tickets.values()),
                'end requires every dependent ticket terminal and known source clock')
        self.tick_wall()
        bounds = self._end_bounds(self._known_size)
        self._event('end_reservation', {'bounds': bounds, 'prefix_bytes': self._known_size,
            'worker_identity_sha256': self.proof['worker_identity_sha256'], 'boot_identity_sha256': self.proof['boot_identity_sha256'],
            'original_origin': self.started_clock, 'monotonic': self.last_clock})
        self.tick_wall()
        self._event('end', {'worker_identity_sha256': self.proof['worker_identity_sha256'],
            'boot_identity_sha256': self.proof['boot_identity_sha256'], 'original_origin': self.started_clock,
            'monotonic': self.last_clock, 'reservation_sha256': digest(self._end_reserved)})
        self._check_final_suffix()
        return self.snapshot()
    def _check_final_suffix(self):
        now = self.clock()
        try:
            require(q.finite(now) and now >= self.last_clock
                    and now - self.last_clock <= self._end_reserved['bounds']['wall_seconds_upper']
                    and now - self.started_clock <= self.spec['physical_wall_seconds_upper'],
                    'final bootstrap write/readback suffix exceeds prepaid wall/deadline')
        except BaseException:
            self.fatal = self.unknown_wall = True; raise
    def inspect_evidence(self):
        # Inspection itself remains permanently priced; it grants no permission.
        if self.closed:
            require(self._end_reserved is not None and not self._end_readback_used and not self.fatal and not self.unknown_wall,
                    'one prepaid final bootstrap readback only')
            self._check_final_suffix(); self._end_readback_used = True
            size = self._end_reserved['bounds']['io_read_bytes_upper']
        else:
            size = self.path.stat().st_size + self.header_path.stat().st_size + MAX_LINE
            self._event('inspection', {'bounds': dict(_zero(), operation_attempts_upper=1,
                io_read_bytes_upper=size, identity_hash_attempts_upper=2, identity_hash_bytes_upper=size)})
        refs = []
        total = 0
        for path in (self.header_path, self.path):
            observed, count = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while block := stream.read(min(CHUNK, size + 1 - total)):
                    count += len(block); total += len(block); require(total <= size, 'bootstrap inspection grew beyond reservation'); observed.update(block)
            refs.append({'path': str(path), 'bytes': count, 'sha256': observed.hexdigest()})
        if self.closed: self._check_final_suffix()
        return {'schema': SCHEMA, 'header_ref': refs[0], 'journal_ref': refs[1], 'snapshot': self.snapshot(),
                'native_qualified': False, 'classification': 'retained prepayment evidence only; no native execution proof'}
