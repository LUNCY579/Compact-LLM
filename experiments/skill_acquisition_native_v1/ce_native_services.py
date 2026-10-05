"""Controller bootstrap, stable dispatch and native resource/lifecycle services.

No source issuers/factories are registered. Construction requires a separately
issued metadata-service capability before IO. Concrete source bodies reuse the
qualified loader helpers; missing admission/publication/worker/lease-transfer
roles refuse explicitly. This module supplies no experiment decision, native
qualification, model default or execution grant.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from . import ce_controller as controller, ce_qualification as qualification, ce_worker_evidence as worker
from . import ce_native_authority as authority

SCHEMA = 'CE8-NATIVE-CONTROLLER-SERVICES-1'
BOOTSTRAP = 'CE8-SERVICES-BOOTSTRAP-1'
PROFILE = ('read_bytes_upper', 'write_bytes_upper', 'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_s_upper')
COUNTERS = ('read_bytes', 'write_bytes', 'hash_attempts', 'hash_bytes', 'metadata_bytes', 'operation_attempts', 'wall_seconds')
INPUT_RESERVATION_RULE = 'CE8-INPUT-IO-EXPLICIT-RESERVATIONS-1'
INPUT_ROW_BYTES = 64 * 1024
_REGISTERED_SERVICE_VERIFIERS = frozenset()
_REGISTERED_ROLE_FACTORIES = frozenset()
_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES = frozenset()
_SERVICE_SEAL = object()
require, digest, canonical = worker.require, worker.digest, worker.canonical


class MissingServiceRole(worker.WorkerEvidenceError): pass


class ControllerServicePermit:
    """Pre-model service metadata capability; no model or native launch authority."""
    __slots__ = ('_seal', '_request', '_spec', '_issuer', '_verifier')
    def __init__(self, request, spec, issuer, verifier):
        require(any(verifier is item for item in _REGISTERED_SERVICE_VERIFIERS), 'controller service issuer is not registered')
        require(issuer is not None and not isinstance(issuer, (dict, list, str, int, float, bool)), 'opaque service issuer required')
        request_sha, spec_sha = digest(controller.request_preimage(request)), digest(spec)
        require(verifier(issuer, request_sha, spec_sha) is True, 'controller service authority failed')
        for name, value in (('_seal', _SERVICE_SEAL), ('_request', request_sha), ('_spec', spec_sha), ('_issuer', issuer), ('_verifier', verifier)):
            object.__setattr__(self, name, value)
    def __setattr__(self, name, value): raise worker.WorkerEvidenceError('controller service permit immutable')
    def require(self, request, spec):
        require(getattr(self, '_seal', None) is _SERVICE_SEAL
                and any(getattr(self, '_verifier', None) is item for item in _REGISTERED_SERVICE_VERIFIERS), 'source-issued controller service authority required')
        require(digest(controller.request_preimage(request)) == self._request and digest(spec) == self._spec
                and self._verifier(self._issuer, self._request, self._spec) is True, 'controller request/spec authority drift')


def _admit(permit, request, spec):
    require(type(permit) is ControllerServicePermit, 'opaque controller service permit required before paths/IO/framework')
    permit.require(request, spec)


def _profile(value):
    require(type(value) is dict and set(value) == set(PROFILE)
            and all(type(value[key]) is int and value[key] >= 0 for key in PROFILE[:-1])
            and type(value['wall_s_upper']) in (int, float) and 0 < value['wall_s_upper'] < float('inf'), 'explicit finite service operation profile')
    return value


def reservation_profile(spec,kind,fields):
    """Permanent declared input reservation, bounded by frozen maxima.

    Opaque verification keeps its complete hash ceiling. This never consumes
    returned measurements, discounts completed work or changes other kinds.
    """
    maximum=_profile(spec['operation_profiles'][kind]);reserved=copy.deepcopy(maximum)
    rule=spec.get('operation_profile_rule')
    require(rule in (None,INPUT_RESERVATION_RULE),'unknown source service reservation rule')
    explicit_kinds=spec.get('explicit_reservation_kinds',['input_io'])
    require(type(explicit_kinds) is list and explicit_kinds in (['input_io'],['input_io','worker_source_verification']),
            'exact finite source explicit reservation kinds')
    row_wall=spec.get('input_io_row_wall_s_upper')
    if row_wall is not None:
        require(rule==INPUT_RESERVATION_RULE and type(row_wall) in (int,float) and row_wall==5
                and spec['operation_profiles']['input_io']['wall_s_upper']>=row_wall,
                'explicit source fixed five-second small-row input ceiling')
    if 'wall_s_upper' in fields:
        wall=fields['wall_s_upper']
        require(type(wall) in (int,float) and 0<wall<=maximum['wall_s_upper'],'bootstrap profile below declared wall work')
    aliases={'read_bytes_upper':'io_read_bytes_upper','write_bytes_upper':'io_write_bytes_upper'}
    declared={}
    for field in PROFILE[:-1]:
        alias=aliases.get(field)
        for present in (field,alias):
            if present in fields:
                require(type(fields[present]) is int and 0<=fields[present]<=maximum[field],
                        'bootstrap profile below declared work')
        if alias in fields:
            require(field not in fields or fields[field]==fields[alias],'conflicting bootstrap IO aliases')
        if field in fields or alias in fields:
            value=fields[alias] if alias in fields else fields[field]
            require(type(value) is int and 0<=value<=maximum[field],'bootstrap profile below declared work')
            declared[field]=value
    for field in ('identity_hash_attempts_lower','identity_hash_bytes_lower'):
        if field in fields:
            require(type(fields[field]) is int and 0<=fields[field]<=maximum[field.replace('_lower','_upper')],
                    'bootstrap profile below declared opaque lower work')
    if kind in explicit_kinds and rule==INPUT_RESERVATION_RULE:
        for field in aliases:
            if field in declared:reserved[field]=declared[field]
        hash_fields=('identity_hash_attempts_upper','identity_hash_bytes_upper')
        opaque=any(key.endswith('_lower') and key.startswith('identity_hash_') or key.startswith('opaque_') for key in fields)
        file_read=(kind!='worker_source_verification' or declared.get('read_bytes_upper',0)>0
            and declared.get('identity_hash_attempts_upper',0)>0
            and declared.get('identity_hash_bytes_upper',0)>=declared.get('read_bytes_upper',0))
        if not opaque and file_read and all(field in declared for field in hash_fields):
            require((declared[hash_fields[0]]==0)==(declared[hash_fields[1]]==0),'coherent explicit input hash reservation')
            for field in hash_fields:reserved[field]=declared[field]
        small_row=(kind=='input_io' and row_wall is not None and type(fields.get('row_id')) is str and fields['row_id']
            and not opaque and all(field in declared for field in PROFILE[:-1])
            and declared['write_bytes_upper']==0 and declared['read_bytes_upper']<=INPUT_ROW_BYTES
            and declared['identity_hash_attempts_upper']<=4 and declared['identity_hash_bytes_upper']<=INPUT_ROW_BYTES
            and (declared['identity_hash_attempts_upper']==0)==(declared['identity_hash_bytes_upper']==0))
        if small_row:
            require('wall_s_upper' not in fields or fields['wall_s_upper']<=row_wall,'declared row work exceeds source fixed row wall ceiling')
            reserved['wall_s_upper']=row_wall
    return reserved


def resolve_role(descriptor, code):
    require(type(descriptor) is dict and set(descriptor) == {'module', 'symbol', 'source_sha256'}
            and type(descriptor['module']) is str and type(descriptor['symbol']) is str
            and worker._hex(descriptor['source_sha256']), 'JSON source-owned service role descriptor required')
    name = descriptor['module'].replace('.', '/') + '.py'
    require(code.get(name) == descriptor['source_sha256'], 'service role source bundle mismatch')
    matches = [item for item in _REGISTERED_ROLE_FACTORIES if getattr(item, '__module__', None) == descriptor['module']
               and getattr(item, '__name__', None) == descriptor['symbol']]
    if len(matches) != 1: raise MissingServiceRole('registered actual service role missing: ' + descriptor['symbol'])
    return matches[0]


class BootstrapLedger:
    """Create-only fsynced hash chain; failed/pending reservations never refund."""
    def __init__(self, spec, request_sha256, *, synthetic_fixture=False, prepayment=None, clock=None):
        require(synthetic_fixture is True or any(type(prepayment) is item for item in _REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES)
                and callable(getattr(prepayment, 'require_constructor', None)), 'native bootstrap requires source-owned permanent outer constructor prepayment')
        if not synthetic_fixture: prepayment.require_constructor(digest(spec), request_sha256)
        self.spec, self.request_sha256 = copy.deepcopy(spec), request_sha256
        self._native = synthetic_fixture is not True
        self.directory = Path(spec['owned_directory']); self.path = self.directory / 'SERVICE_BOOTSTRAP.jsonl'
        self.counters = dict.fromkeys(COUNTERS, 0); self.operations, self.events = {}, []
        require(synthetic_fixture is True or clock is None, 'native service clock cannot be injected')
        self.clock=time.perf_counter if clock is None else clock
        self.started=self.last_clock=self.clock(); self.actual_wall_seconds=0.0
        require(type(self.started) in (int,float) and 0<=self.started<float('inf'), 'finite service clock origin')
        self.sealed = self.fatal = False
        self._known_size=0;self._end_reserved=None;self._inspection_used=False
        self.header_path=self.directory/'SERVICE_BOOTSTRAP_HEADER.json'
        self.seal_path=self.directory/'SERVICE_BOOTSTRAP_SEAL.json'
        require(self.directory.is_absolute() and not self.directory.is_symlink(), 'absolute owned bootstrap directory')
        require(type(spec.get('caps')) is dict and set(spec['caps']) == set(COUNTERS)
                and all(type(spec['caps'][key]) is int and spec['caps'][key] > 0 for key in COUNTERS if key != 'wall_seconds')
                and type(spec['caps']['wall_seconds']) in (int, float) and 0 < spec['caps']['wall_seconds'] < float('inf'), 'finite complete bootstrap caps')
        require(type(spec.get('operation_profiles')) is dict and spec['operation_profiles'], 'explicit bootstrap operation profiles')
        for profile in spec['operation_profiles'].values(): _profile(profile)
        # Actual outer accounting permanently pre-reserves this constructor;
        # the service permit alone is not prepayment.
        _profile(spec.get('setup_bounds'))
        require(spec.get('operation_profile_rule') in (None,INPUT_RESERVATION_RULE),'unknown source service reservation rule')
        require(type(spec.get('explicit_reservation_kinds',['input_io'])) is list
                and spec.get('explicit_reservation_kinds',['input_io']) in (['input_io'],['input_io','worker_source_verification']),
                'exact finite source explicit reservation kinds')
        if spec.get('input_io_row_wall_s_upper') is not None:
            reservation_profile(spec,'input_io',{})
        if not synthetic_fixture:
            require(spec.get('operation_profile_rule')==INPUT_RESERVATION_RULE,'explicit native service input reservation rule required')
            _profile(spec.get('finalization_bounds'))
            require(type(spec.get('physical_wall_seconds_upper')) in (int,float)
                    and 0<spec['physical_wall_seconds_upper']<float('inf'), 'native service physical deadline required')
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not any(os.path.lexists(path) for path in (self.path,self.header_path,self.seal_path)), 'create-only service bootstrap ledger/header/seal')
        self.header={'schema':BOOTSTRAP,'request_sha256':request_sha256,'specification':copy.deepcopy(spec),
                     'native':not synthetic_fixture,'original_origin':self.started}
        self._specification_bytes=canonical(spec)
        raw_header=canonical(self.header)+b'\n'
        require(len(raw_header)<=1024**2, 'bounded service bootstrap header')
        for key,value in (('write_bytes',len(raw_header)),('metadata_bytes',len(raw_header)),('hash_attempts',1),('hash_bytes',len(raw_header))):
            require(self.counters[key]+value<=spec['caps'][key], 'service header exceeds prepaid cap');self.counters[key]+=value
        try:
            with self.header_path.open('xb') as stream:
                require(stream.write(raw_header)==len(raw_header),'short service bootstrap header');stream.flush();os.fsync(stream.fileno())
        except BaseException:self.fatal=True;raise
        self._event('setup', {'setup_bounds': spec['setup_bounds']}, self._price(spec['setup_bounds']))
        self._event('origin',{'monotonic':self.started,'original_origin':self.started},{})
    def _price(self, profile):
        return {'read_bytes': profile['read_bytes_upper'], 'write_bytes': profile['write_bytes_upper'],
            'hash_attempts': profile['identity_hash_attempts_upper'], 'hash_bytes': profile['identity_hash_bytes_upper'],
            'wall_seconds': profile['wall_s_upper']}
    def _event(self, kind, payload, charges):
        require(not self.sealed and not self.fatal, 'service bootstrap terminal/fatal; no further claims')
        body = {'schema': BOOTSTRAP, 'seq': len(self.events), 'previous': self.events[-1]['sha256'] if self.events else None,
            'request_sha256': self.request_sha256, 'kind': kind, 'payload': payload}
        preimage = canonical(body); hashed = hashlib.sha256(preimage).hexdigest()
        raw = canonical(dict(body, sha256=hashed)) + b'\n'
        require(len(raw) <= 1024 ** 2, 'bounded service metadata row')
        charges = dict(charges)
        for key, value in (('metadata_bytes', len(raw)), ('write_bytes', len(raw)), ('hash_attempts', 1), ('hash_bytes', len(preimage))):
            charges[key] = charges.get(key, 0) + value
        next_counters = dict(self.counters)
        for key, value in charges.items(): next_counters[key] += value
        require(all(next_counters[key] <= self.spec['caps'][key] for key in COUNTERS), 'service reservation cap exhausted before append')
        self.counters = next_counters  # remain charged if the fsynced append fails
        try:
            require(not self.path.exists() and self._known_size==0 or self.path.stat().st_size==self._known_size,
                    'single-writer service bootstrap history changed')
            with self.path.open('ab') as stream:
                require(stream.write(raw) == len(raw), 'short service bootstrap append')
                stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            self.fatal = True; raise
        row = dict(body, sha256=hashed); self.events.append(row);self._known_size+=len(raw)
        return row
    def operation(self, kind, phase, operation_id, **fields):
        self._require_specification()
        require(self._end_reserved is None, 'no service work after final reservation')
        require(not self.sealed and not self.fatal, 'service bootstrap terminal/fatal; no further claims')
        require(kind in self.spec['operation_profiles'] and phase in ('started', 'returned', 'failed'), 'priced service kind/phase required')
        if phase == 'started':
            require(type(operation_id) is str and operation_id and operation_id not in self.operations, 'unique bootstrap physical attempt')
            profile = reservation_profile(self.spec,kind,fields)
            self.tick_wall()
            row = self._event('operation', {'operation': kind, 'phase': phase, 'operation_id': operation_id, 'fields': fields,
                              'reservation_profile':profile},
                              dict(self._price(profile), operation_attempts=1))
            self.operations[operation_id] = {'kind': kind, 'phase': phase, 'profile': profile, 'started_seq': row['seq']}
        else:
            previous = self.operations.get(operation_id)
            require(previous and previous['kind'] == kind and previous['phase'] == 'started', 'bootstrap matching return required')
            actuals = {'bytes_read': 'read_bytes_upper', 'bytes_written': 'write_bytes_upper',
                'identity_hash_attempts_actual': 'identity_hash_attempts_upper',
                'identity_hash_bytes_actual': 'identity_hash_bytes_upper', 'wall_s': 'wall_s_upper'}
            for field, upper in actuals.items():
                if field not in fields: continue
                value = fields[field]
                valid = (type(value) in (int, float) and 0 <= value < float('inf')) if field == 'wall_s' else (type(value) is int and value >= 0)
                if not valid or value > previous['profile'][upper]:
                    self.fatal = True
                    raise worker.WorkerEvidenceError('bootstrap actual work exceeds reserved profile: ' + field)
            self.tick_wall()
            row = self._event('operation', {'operation': kind, 'phase': phase, 'operation_id': operation_id, 'fields': fields}, {})
            previous.update(phase=phase, returned_seq=row['seq'])
        return row
    __call__ = operation
    def _require_specification(self):
        require(not self._native or canonical(self.spec) == self._specification_bytes,
                'native service specification changed after constructor prepayment')
    def tick_wall(self):
        self._require_specification()
        require(not self.sealed and not self.fatal, 'ended/fatal service clock cannot restart')
        try:
            now=self.clock();require(type(now) in (int,float) and self.last_clock<=now<float('inf'), 'monotonic service clock regressed')
            deadline=self.spec.get('physical_wall_seconds_upper',self.spec['caps']['wall_seconds'])
            require(now-self.started<=deadline,'service physical deadline exceeded')
        except BaseException:self.fatal=True;raise
        if now>self.last_clock:
            self._event('wall',{'original_origin':self.started,'monotonic':now,'elapsed_delta':now-self.last_clock},
                        {'wall_seconds':now-self.last_clock})
            self.actual_wall_seconds+=now-self.last_clock;self.last_clock=now

    def end(self):
        require(not self.sealed and not self.fatal and self._end_reserved is None
                and not any(value['phase']=='started' for value in self.operations.values()),
                'service end requires all owned operations/process cleanup terminal')
        self.tick_wall()
        profile=self.spec.get('finalization_bounds',self.spec['setup_bounds'])
        _profile(profile);read=self._known_size+len(canonical(self.header))+1+3*1024**2
        require(profile['read_bytes_upper']>=read and profile['write_bytes_upper']>=1024**2
                and profile['identity_hash_attempts_upper']>=4 and profile['identity_hash_bytes_upper']>=read+2*1024**2,
                'permanent final service write/hash/readback lower bounds')
        payload={'original_origin':self.started,'monotonic':self.last_clock,'prefix_bytes':self._known_size,
                 'bounds':copy.deepcopy(profile)}
        self._event('end_reservation',payload,dict(self._price(profile),operation_attempts=2,metadata_bytes=1024**2))
        self._end_reserved=copy.deepcopy(payload)
        self.tick_wall()
        self._event('end',{'original_origin':self.started,'monotonic':self.last_clock,
                          'reservation_sha256':digest(self._end_reserved)}, {})
        self.sealed=True;self._check_suffix()
        raw_seal=canonical(self.snapshot())+b'\n'
        require(len(raw_seal)<=1024**2,'bounded prepaid service seal')
        try:
            with self.seal_path.open('xb') as stream:
                require(stream.write(raw_seal)==len(raw_seal),'short final service seal');stream.flush();os.fsync(stream.fileno())
            self._check_suffix()
        except BaseException:self.fatal=True;raise
        return self.snapshot()

    def _check_suffix(self):
        self._require_specification()
        now=self.clock()
        try:
            require(self._end_reserved is not None and type(now) in (int,float) and self.last_clock<=now<float('inf')
                    and now-self.last_clock<=self._end_reserved['bounds']['wall_s_upper']
                    and now-self.started<=self.spec.get('physical_wall_seconds_upper',self.spec['caps']['wall_seconds']),
                    'final service suffix exceeds permanent wall/deadline')
        except BaseException:self.fatal=True;raise

    def inspect_evidence(self):
        require(self.sealed and not self.fatal and not self._inspection_used and self._end_reserved is not None,
                'one permanently prepaid ended service inspection')
        self._check_suffix();self._inspection_used=True
        upper=self._end_reserved['bounds']['read_bytes_upper'];refs=[];total=0
        try:
            for path in (self.header_path,self.path,self.seal_path):
                hashed=hashlib.sha256();count=0
                with path.open('rb') as stream:
                    while block:=stream.read(min(1024**2,upper+1-total)):
                        count+=len(block);total+=len(block);require(total<=upper,'service final inspection grew');hashed.update(block)
                refs.append({'path':str(path),'sha256':hashed.hexdigest(),'bytes':count})
            self._check_suffix()
        except BaseException:self.fatal=True;raise
        return {'schema':BOOTSTRAP,'header_ref':refs[0],'journal_ref':refs[1],'seal_ref':refs[2],'snapshot':self.snapshot(),
                'native_qualified':False,'execution_verified':False}
    def snapshot(self):
        return {'schema': BOOTSTRAP, 'request_sha256': self.request_sha256, 'head': self.events[-1]['sha256'],
            'cumulative': dict(self.counters), 'pending': [key for key, value in self.operations.items() if value['phase'] == 'started'],
            'path': str(self.path), 'fatal': self.fatal, 'native_qualified': False,
            'original_origin':self.started,'last_monotonic':self.last_clock,'actual_wall_seconds':self.actual_wall_seconds,
            'clock_ended':self.sealed,'end_reservation':copy.deepcopy(self._end_reserved),
            'failed_operations':[key for key,value in self.operations.items() if value['phase']=='failed'],
            'claim':'retained service cost/clock evidence; independent raw replay and OS execution verification required'}


class StableDispatcher:
    """One callback identity across bootstrap and separately admitted ledgers."""
    def __init__(self, bootstrap):
        self.bootstrap, self.ledger, self.owners = bootstrap, None, {}
    def attach_qualification_accounting(self, ledger):
        require(self.ledger is None and getattr(ledger, 'scope_role', None) == 'qualification_only', 'one separate qualification accounting owner required')
        self.ledger = ledger
    def attach_journal(self, ledger):
        require(self.ledger is None and getattr(ledger, 'synthetic_cpu', True) is False, 'one separately admitted native CE ledger owner required')
        self.ledger = ledger
    def __call__(self, kind, phase, operation_id, **fields):
        require(type(operation_id) is str and operation_id and phase in ('started', 'returned', 'failed'),
                'explicit dispatch operation identity/phase')
        if phase == 'started':
            require(operation_id not in self.owners, 'dispatch operation identity reused')
            # Finite source metadata kinds keep their original outer owner
            # between and after Q stages; no caller-selected owner flags.
            target = self.bootstrap if kind in ('outer_reservation','supervisor_lease','supervisor_lease_acquire','supervisor_lease_release','worker_os_binding','worker_handle_close','worker_handle_duplicate') else (
                self.ledger if self.ledger is not None else self.bootstrap)
            if kind in ('resource_probe', 'resource_cuda_probe') and self.ledger is not None:
                if getattr(self.ledger, 'scope_role', None) == 'qualification_only' and (
                        getattr(self.ledger, 'active', None) is None or getattr(self.ledger, 'worker_open', True) is False):
                    target = self.bootstrap
            self.owners[operation_id] = (kind, target)
        else:
            require(operation_id in self.owners and self.owners[operation_id][0] == kind,
                    'dispatch matching original operation owner required')
            target = self.owners[operation_id][1]
        return target.operation(kind, phase, operation_id=operation_id, **fields)
    operation = __call__
    def resource_evidence(self, identifier):
        require(identifier in self.owners, 'known physically reserved resource operation')
        kind, owner = self.owners[identifier]
        require(kind in ('resource_probe', 'resource_cuda_probe'), 'resource linkage cannot borrow another operation')
        op = owner.operations[identifier]
        require(op['phase'] == 'returned' and type(op.get('returned_seq')) is int, 'returned resource operation required')
        row = owner.events[op['returned_seq']]
        return {'operation_id': identifier, 'kind': kind, 'phase': op['phase'],
            'owner_schema': owner.snapshot()['schema'], 'owner_binding_sha256': getattr(owner, 'binding_sha256', None),
            'journal_path': str(owner.path), 'returned_seq': op['returned_seq'], 'returned_event_sha256': row['sha256']}
    def snapshot(self): return (self.ledger if self.ledger is not None else self.bootstrap).snapshot()


class NativeResourceGate:
    """Actual RAM/disk/CUDA checks; numeric limits/transients all scope-supplied."""
    def __init__(self, spec, bootstrap):
        self.spec, self.bootstrap, self.framework = copy.deepcopy(spec), bootstrap, None
        self.supervisor = None
        self.samples, self.recorder = [], None
        self.sample_attempts = 0
        self.sample_attempts_upper = spec.get('sample_attempts_upper')
        require(self.sample_attempts_upper is None or type(self.sample_attempts_upper) is int
                and 0 < self.sample_attempts_upper <= 100000, 'finite source resource sample attempt ceiling')
        require(type(spec.get('limits')) is dict and type(spec.get('resource_probe')) is dict
                and type(spec.get('host_transient_bytes')) is dict
                and all(type(value) is int and value >= 0 for value in spec['host_transient_bytes'].values()), 'explicit resource limits/probe/transients')
        limits = spec['limits']; gib = 1024 ** 3
        require(set(limits) == {'launch_host_gib', 'ongoing_host_gib', 'max_worker_gib', 'gpu_free_gib',
            'allocator_fraction', 'reserved_fraction_limit', 'disk_floor_bytes', 'launch_disk_bytes', 'active_storage_reserve_bytes'},
            'complete registered service resource barriers')
        require(all(type(limits[key]) in (int, float) and 0 < limits[key] < float('inf') for key in
            ('launch_host_gib', 'ongoing_host_gib', 'max_worker_gib', 'gpu_free_gib', 'allocator_fraction', 'reserved_fraction_limit'))
            and limits['launch_host_gib'] >= 5.5 and limits['ongoing_host_gib'] >= 1 and limits['max_worker_gib'] <= 4.5
            and limits['gpu_free_gib'] >= .5 and limits['allocator_fraction'] == .78 and limits['reserved_fraction_limit'] == .95
            and type(limits['disk_floor_bytes']) is int and limits['disk_floor_bytes'] >= 14 * gib
            and type(limits['launch_disk_bytes']) is int and limits['launch_disk_bytes'] >= 32 * gib
            and type(limits['active_storage_reserve_bytes']) is int and limits['active_storage_reserve_bytes'] >= 0,
            'registered service barriers cannot be lowered')
        from .ce_resource_evidence import validate_spec
        validate_spec(spec['resource_probe'])
    def _sample(self, *, launch=False, stage=None):
        require(self.sample_attempts_upper is None or self.sample_attempts < self.sample_attempts_upper,
                'resource sample attempt ceiling exhausted before observation')
        self.sample_attempts += 1
        if self.supervisor is not None: self.supervisor.assert_parent_use()
        from .ce_resource_evidence import readonly_sample, enforce
        label = 'launch' if launch else stage
        require(type(label) is str and label, 'explicit controller resource barrier label')
        limits = copy.deepcopy(self.spec['limits'])
        if stage is not None:
            require(stage in self.spec['host_transient_bytes'], 'unpriced service host transient: ' + stage)
            limits['ongoing_host_gib'] += self.spec['host_transient_bytes'][stage] / 1024 ** 3
        if self.recorder is not None:
            self.recorder.framework = self.framework
            sample = self.recorder.sample(label)
        else:
            sample = readonly_sample(self.spec['resource_probe'], self.bootstrap, barrier=label)
            sample['stage'] = label
            evidence = getattr(self.bootstrap, 'resource_evidence', None)
            if callable(evidence): sample['operation_links'] = [evidence(sample['resource_probe_operation'])]
            self.samples.append(copy.deepcopy(sample))
        return enforce(sample, limits, launch=launch)
    def launch(self):
        return self._sample(launch=True)
    def __call__(self, stage):
        return self._sample(stage=stage)
    def bind_recorder(self, recorder):
        from .ce_resource_evidence import NativeResourceRecorder
        require(type(recorder) is NativeResourceRecorder and self.recorder is None
                and recorder.operation is self.bootstrap and recorder.spec == self.spec['resource_probe']
                and recorder.samples is self.samples, 'exact one shared resource recorder/spec/dispatcher')
        self.recorder = recorder
        return recorder


class NativeControllerServices:
    def __init__(self, request, service_permit, spec):
        _admit(service_permit, request, spec)
        controller.validate_request(request)
        require(type(spec) is dict and spec.get('schema') == SCHEMA and type(spec.get('bootstrap')) is dict
                and type(spec.get('resource')) is dict and type(spec.get('role_descriptors')) is dict, 'complete frozen service specification')
        self.request, self.spec, self.permit = copy.deepcopy(request), copy.deepcopy(spec), service_permit
        prepay = resolve_role(spec['role_descriptors'].get('bootstrap_constructor_prepayment'), spec.get('approved_code_sha256', {}))
        prepayment = prepay(request, spec, service_permit)
        self.constructor_prepayment = prepayment
        self.bootstrap = BootstrapLedger(spec['bootstrap'], digest(controller.request_preimage(request)), prepayment=prepayment)
        self.operation = StableDispatcher(self.bootstrap)
        self.operation.service_owner = self
        self.gate = NativeResourceGate(spec['resource'], self.operation)
        self.context, self.worker_handle, self.lease_token = None, None, None
        self.supervisor_lease = self.bound_backend = None
        self.resource_recorder = None
    def _admit(self): _admit(self.permit, self.request, self.spec)
    def _role(self, name, context=None):
        self._admit(); descriptor = self.spec['role_descriptors'].get(name)
        code = (context or self.context or {}).get('bindings', {}).get('source', {}).get('code_sha256')
        if code is None: code = self.spec.get('approved_code_sha256', {})
        return resolve_role(descriptor, code)
    def _span(self, kind, work, **fields):
        identifier, started = uuid.uuid4().hex, time.perf_counter()
        self.bootstrap.operation(kind, 'started', operation_id=identifier, **fields)
        try: result = work()
        except BaseException as error:
            try: self.bootstrap.operation(kind, 'failed', operation_id=identifier, error=str(error)[:1000], wall_s=time.perf_counter() - started)
            except BaseException as record_error: error.service_accounting_failure = str(record_error)[:1000]
            raise
        self.bootstrap.operation(kind, 'returned', operation_id=identifier, wall_s=time.perf_counter() - started)
        return result
    def preflight_bootstrap(self, bounds, request_hash):
        self._admit()
        require(bounds == self.request['bootstrap_bounds'] and request_hash == digest(controller.request_preimage(self.request)), 'bootstrap request bounds drift')
        return self.bootstrap.snapshot()
    def admit_request(self, request):
        self._admit(); require(request == self.request, 'service request changed')
        if self.context is not None:
            # The owned entry observes this same worker before its ready RPC.
            # Controller entry reuses that frozen admission after refreshing
            # the retained actual handle; it cannot replace request or context.
            self.verify_worker(self.context)
            return self.context
        if request['phase'] != 'qualification':
            raise MissingServiceRole('separate raw production-scope/qualification-result verifier remains unimplemented')
        bootstrap_outer = self._role('bootstrap_outer')(request, self.operation, self.spec)
        from .runner import code_bundle, verify_module_origins
        code = self._span('source_verification', lambda: code_bundle(self.operation))
        require(code == self.spec['approved_code_sha256'], 'complete running service/source bundle drift')
        verify_module_origins(code)
        context = self._span('scope_admission', lambda: qualification.admit_qualification(
            request, code_sha256=code, outer_admission=bootstrap_outer, io_observer=self.operation))
        require(type(context) is dict and context.get('admission_evidence_verified') is True
                and context.get('request_sha256') == digest(controller.request_preimage(request)), 'raw new-operator/scope/source admission required')
        # admit_qualification verifies §18 choice and every raw scope/recipe/
        # source/runtime/anchor/caps reference. This cannot be replaced by a
        # callback which merely returns admitted=True.
        context['native_services'] = {'operation': self.operation, 'gate': self.gate}
        context = self.finalize_worker_context(context)
        self.context = context
        return context
    def finalize_worker_context(self, context):
        self._admit()
        require('native_execution_permit' not in context.get('native_services', {}), 'worker JSON context must be complete before native permit issuance')
        spec, permit = self._role('worker_observation', context)(context, self.operation)
        handle = worker.open_worker(spec, permit, self.operation)
        try:
            proof = handle.proof()
            require(proof['fresh_process'] is True and proof['source_verified'] is True, 'registered actual worker provenance prerequisite')
            completed = dict(context, worker_proof_sha256=digest(proof))
        except BaseException as primary:
            try: handle.close()
            except BaseException as cleanup:
                primary.worker_close_failure = {'type': type(cleanup).__name__, 'message': str(cleanup)[:1000]}
                raise primary from cleanup
            raise
        self.worker_handle, self.worker_proof = handle, proof
        return completed
    def observer(self, context):
        self._admit(); require(context is self.context, 'service context identity')
        def observe(name, phase, **fields):
            # Controller stage summaries reserve only own metadata/verification;
            # native leaf charges stay in their linked bootstrap/Q/CE owner.
            return self.bootstrap.operation('controller_stage', phase, stage=name, **fields)
        return observe
    def make_outer_admission(self, request, context):
        self._admit()
        issuer = self._role('outer_issuer', context)(request, context, self.operation)
        verifier = self._role('outer_verifier', context)
        reserve = self._role('outer_reservation', context)
        return controller.OuterAdmission(request, context, issuer, reserve, verifier)
    def verify_sources(self, request, context, outer):
        self._admit()
        from . import ce_ingestion
        return ce_ingestion.verify_sources(request['preparation_ref'], request['plan_ref'], request['arm'], outer,
                                           operation=self.operation, gate=self.gate)
    def verify_worker(self, context):
        self._admit()
        require(self.worker_handle is not None and worker._hex(context.get('worker_proof_sha256')), 'finalize worker context before outer/native authority')
        proof = self.worker_handle.proof()
        require(proof['fresh_process'] is True and proof['source_verified'] is True, 'registered actual current worker provenance required')
        require(context['worker_proof_sha256'] == digest(proof), 'worker provenance changed after complete context freeze')
        return proof
    def configure_preparation(self, context):
        self._admit()
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime
        def configure():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = runtime.CUBLAS_WORKSPACE_CONFIG
            runtime.assert_runtime()
            import torch
            require(not torch.cuda.is_initialized(), 'native preparation must not initialize CUDA before loader configuration')
            return {'framework_versions_checked': True, 'cuda_initialized': False, 'native_qualified': False}
        return self._span('framework_preparation', configure)
    def prepare_rows(self, verified, context, outer):
        self._admit()
        from .ce_ingestion import PreparedRows
        return PreparedRows(verified, outer_admission=outer, operation=self.operation, gate=self.gate)
    @contextmanager
    def lease(self, context):
        self._admit()
        require(context is self.context and context.get('phase') == 'qualification', 'exact qualification supervisor context')
        permit=context.get('native_execution_permit')
        authority.require_execution_permit(permit,context,'checkpoint_resume',operation=self.operation,gate=self.gate)
        require(self.supervisor_lease is None and self.lease_token is None and self.operation.ledger is None,
                'one supervisor lease before Q attachment; no nested self-wait')
        from .ce_checkpoint_lease import SupervisorGpuLease
        owner=SupervisorGpuLease(context,permit,self.worker_handle,None,self.operation,self.gate)
        primary=None
        try:
            owner.acquire()
            self.supervisor_lease,self.lease_token=owner,owner.token
            self.gate.supervisor=owner
            services=context['native_services']
            checkpoint=services.setdefault('checkpoint_qualification',{})
            require(type(checkpoint) is dict and checkpoint.get('handoff') in (None,owner), 'foreign checkpoint lease holder')
            checkpoint['handoff']=owner
            yield owner
        except BaseException as error:
            primary=error
            raise
        finally:
            if owner.release_seal is not None and not owner.closed:
                try:
                    try: owner.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            primary.supervisor_lease_cleanup_failure={'type':type(cleanup).__name__,'message':str(cleanup)[:1000]}
                            raise primary from cleanup
                        raise
                finally:
                    if owner.closed:
                        self.lease_token=None; self.gate.supervisor=None

    def bind_backend(self,context,backend):
        self._admit(); require(context is self.context and self.supervisor_lease is not None
                               and self.bound_backend is None, 'existing supervisor before one backend bind')
        from .ce_native_backend import NativeCEBackend
        from .ce_native_qualification import NativeQualificationExecutor
        require(type(backend) in (NativeCEBackend,NativeQualificationExecutor), 'exact native backend/executor required')
        target=backend.backend if type(backend) is NativeQualificationExecutor else backend
        self.supervisor_lease.bind_backend(target)
        self.bound_backend=target
    @contextmanager
    def watchdog(self, context):
        self._admit()
        seconds = self.spec['physical_deadline_seconds']
        require(type(seconds) in (int, float) and 0 < seconds < float('inf'), 'registered physical service deadline')
        timer = threading.Timer(seconds, lambda: os._exit(124))
        timer.daemon = True; timer.start()
        try: yield
        finally: timer.cancel()
    def guard(self, context, stage): self._admit(); return self.gate(stage)
    def bind_resource_recorder(self, context, permit):
        self._admit(); require(context is self.context and self.resource_recorder is None, 'one complete admitted resource context')
        authority.require_execution_permit(permit, context, 'resource_io_measurement', operation=self.operation, gate=self.gate)
        from .ce_resource_evidence import NativeResourceRecorder
        recorder = NativeResourceRecorder(context, permit, self.operation, samples=self.gate.samples)
        self.resource_recorder = self.gate.bind_recorder(recorder)
        return self.resource_recorder
    def resource_barrier(self, context, barrier):
        self._admit(); require(context is self.context and self.resource_recorder is not None, 'bound native recorder before explicit barrier')
        return self.gate(barrier)
    def open_journal(self, context, prepared, proof, resume, reason):
        self._admit()
        return self._role('native_journal', context)(context, prepared, proof, resume, reason, self.operation, self.gate)
    def checkpoints(self, context, backend, journal): self._admit(); return backend.checkpoints(context, journal)
    def cleanup(self, context, backend):
        # Backend's genuine prepaid deallocation handles later authority denial.
        try: self._admit()
        except BaseException as primary:
            from .ce_native_backend import NativeCEBackend, _PREPAID_CLEANUP_SEAL
            from .ce_native_qualification import NativeQualificationExecutor
            target = backend.backend if type(backend) is NativeQualificationExecutor else backend
            if (type(target) is NativeCEBackend and target.context is context and target.operation is self.operation
                    and target.gate is self.gate and target.model is target.optimizer is target.tokenizer is target.torch is None):
                # Close the already-created metadata object even if its service
                # permission expires before first allocation; retain the denial.
                target.closed=True
                raise
            if type(target) is not NativeCEBackend or getattr(target, '_cleanup_seal', None) is not _PREPAID_CLEANUP_SEAL:
                raise
            try: backend.cleanup()
            except BaseException as cleanup_error:
                primary.service_cleanup_failure = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)[:1000]}
                raise primary from cleanup_error
            raise
        return backend.cleanup()
    def _publish(self, name, context, payload, journal):
        self._admit()
        role = self._role(name, context)
        return self._span(name, lambda: role(context, payload, journal, self.bootstrap.snapshot(), self.operation))
    def interruption(self, context, error, journal):
        return self._publish('interruption', context, {'type': type(error).__name__, 'message': str(error)[:1000]}, journal)
    def publish_completion(self, context, artifacts, journal): return self._publish('publication', context, artifacts, journal)
    def seal(self, context, publication, journal):
        self._admit()
        # Completion verifier must replay exact outer + native ledgers, include
        # failed/pending costs, publication and its own pre-reserved verification.
        return self._role('completion_seal', context)(context, publication, journal, self.bootstrap, self.operation)
    def build_child_launch_spec(self, context, branch, request_ref, parent_identity):
        self._admit()
        require(branch in ('uninterrupted', 'prefix', 'resumed'), 'exact sequential qualification branch')
        require(self.lease_token is not None, 'child needs existing supervisor-owned lease')
        # This source-owned transfer role establishes an inherited exclusive
        # child-use capability. It must not ask the child to reacquire our lock.
        return self._role('child_launch_and_lease_transfer', context)(context, branch, request_ref, parent_identity,
                                                                    self.lease_token, self.operation)
