"""Source-paid final receipt costs and opaque OS/clock observations.

No model authority or registry is introduced. The original supervisor metadata
permit admits one separate permanent owner before files are inspected. Previous
ended owners are independently replayed; this owner checks its own reserved
bounds in source rather than recursively certifying its own journal.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import re
import time
import uuid

from . import ce_native_services as services, ce_qualification as q
from . import ce_qualification_artifacts as raw, ce_worker_evidence as worker
from . import update

SCHEMA='CE8-QUALIFICATION-FINALIZER-COSTS-1'
SPEC='CE8-QUALIFICATION-FINALIZER-SPEC-1'
COST_FIELDS=q.COST_FIELDS+('reducer_work_bytes_upper',)
MAX_JSON=4*1024**2
_OWNER_SEAL,_OBSERVATION_SEAL,_FIXTURE_SEAL=object(),object(),object()
require,canonical,digest=raw.require,raw.canonical,raw.digest
KINDS=frozenset(('qualification_reducer_overhead','qualification_artifact_readback','qualification_artifact',
    'finalizer_json_write','finalizer_terminal_observation','finalizer_service_end','finalizer_metadata'))
PREFLIGHTS=frozenset(('qualification_execution_verifier','bootstrap_raw_verifier','service_bootstrap_raw_verifier',
    'checkpoint_branch_reduction','saved_delta_source_rebuild'))


def _bounds(value):
    require(type(value) is dict and set(value)==set(COST_FIELDS)
            and all(type(value[key]) is int and value[key]>=0 for key in COST_FIELDS if key!='wall_seconds_upper')
            and type(value['wall_seconds_upper']) in (int,float) and 0<value['wall_seconds_upper']<float('inf'),
            'explicit complete finite finalizer bounds')
    require(value['operation_attempts_upper']>0 and value['metadata_bytes_upper']>0,'positive finalizer attempt/metadata bounds')
    return value


def validate_specification(spec,output_root):
    require(type(spec) is dict and set(spec)=={'schema','owned_directory','max_receipt_bytes',
        'physical_wall_seconds_upper','self_bounds','finalization_bounds','reader_bounds'} and spec['schema']==SPEC,
        'exact immutable finalizer policy')
    require(type(spec['owned_directory']) is str and Path(spec['owned_directory']).is_absolute()
            and Path(spec['owned_directory']).is_relative_to(Path(output_root)), 'finalizer owned output directory')
    require(type(spec['max_receipt_bytes']) is int and 0<spec['max_receipt_bytes']<=MAX_JSON
            and type(spec['physical_wall_seconds_upper']) in (int,float)
            and 0<spec['physical_wall_seconds_upper']<float('inf'),'finite finalizer receipt/deadline')
    full,suffix=_bounds(spec['self_bounds']),_bounds(spec['finalization_bounds'])
    require(all(suffix[key]<=full[key] for key in COST_FIELDS)
            and suffix['io_write_bytes_upper']>=MAX_JSON and suffix['metadata_bytes_upper']>=MAX_JSON
            and suffix['io_read_bytes_upper']>=2*MAX_JSON+2 and suffix['identity_hash_attempts_upper']>=8
            and suffix['identity_hash_bytes_upper']>=8*MAX_JSON
            and suffix['operation_attempts_upper']>=8 and suffix['wall_seconds_upper']<=120,
            'permanent finalizer suffix below known write/hash/readback costs')
    require(type(spec['reader_bounds']) is dict and set(spec['reader_bounds'])==set(raw.BUDGET_FIELDS)
            and all(type(value) is int and value>=0 for value in spec['reader_bounds'].values())
            and all(value<=full[key] for key,value in spec['reader_bounds'].items()),'finalizer reader within full permanent envelope')
    return copy.deepcopy(spec)


class TerminalObservation:
    __slots__=('_seal','_owner','_service','_proof','_before','_after','_native')
    def __init__(self,owner,service,proof,before,after,*,seal):
        require(seal is _OBSERVATION_SEAL,'source-owned terminal observation required')
        for key,value in (('_seal',seal),('_owner',owner),('_service',service),('_proof',copy.deepcopy(proof)),
                          ('_before',before),('_after',after),('_native',owner.native)):
            object.__setattr__(self,key,value)
    def __setattr__(self,name,value):raise ValueError('terminal observation immutable')
    @property
    def proof(self):return copy.deepcopy(self._proof)
    @property
    def monotonic(self):return self._after
    @property
    def native(self):return self._native


class ServiceEndObservation:
    __slots__=('_seal','_owner','_service','_evidence','_before','_after','_native')
    def __init__(self,owner,service,evidence,before,after,*,seal):
        require(seal is _OBSERVATION_SEAL,'source-owned service end observation required')
        for key,value in (('_seal',seal),('_owner',owner),('_service',service),('_evidence',copy.deepcopy(evidence)),
                          ('_before',before),('_after',after),('_native',owner.native)):
            object.__setattr__(self,key,value)
    def __setattr__(self,name,value):raise ValueError('service end observation immutable')
    @property
    def evidence(self):return copy.deepcopy(self._evidence)
    @property
    def monotonic(self):return self._after
    @property
    def native(self):return self._native


class FinalizerCostOwner:
    def __init__(self,service):
        require(type(service) is services.NativeControllerServices,'actual source supervisor service before finalizer work')
        services._admit(service.permit,service.request,service.spec)
        require(service.context is None and type(getattr(service,'_qualification_launch_spec',None)) is dict,
                'original metadata-only qualification supervisor')
        spec=service.spec['supervision']['finalizer'];root=service.spec['resource']['resource_probe']['output_root']
        profile=services._profile(service.spec['bootstrap']['operation_profiles']['qualification_finalizer_constructor'])
        require(profile['write_bytes_upper']>=MAX_JSON and profile['identity_hash_attempts_upper']>=8
                and profile['identity_hash_bytes_upper']>=8*MAX_JSON,'bounded constructor self-write/hash envelope')
        with update._span(service.operation,'qualification_finalizer_constructor',dict(profile)):
            self._initialize(spec,root,service._qualification_launch_spec['request_sha256'],
                service._qualification_launch_spec['scope_sha256'],service,native=True,clock=time.monotonic)

    def _initialize(self,spec,root,request_sha256,scope_sha256,service,*,native,clock):
        self.spec=validate_specification(spec,root);self._spec_bytes=canonical(self.spec)
        require(raw._hex(request_sha256) and raw._hex(scope_sha256),'finalizer exact logical request/scope')
        self._seal,self.native,self.service=_OWNER_SEAL,native,service
        self.request_sha256,self.scope_sha256=request_sha256,scope_sha256
        self.clock=clock;self.origin=self.last_clock=clock()
        require(type(self.origin) in (int,float) and 0<=self.origin<float('inf'),'finalizer source monotonic origin')
        self.counts=dict.fromkeys(COST_FIELDS,0);self.operations={};self.events=[];self.receipts={}
        self.fatal=self.closed=False;self._suffix_origin=self._suffix_before=None;self._known_size=0
        self._terminal_observation=self._service_end_observation=None
        self._final_evidence=self._final_evidence_bytes=None
        self._delta_rebuild_started=self._delta_rebuild_operation=self._delta_rebuild_wall=None
        self.operation=self.operation # one callback identity for all independent readers
        self.root=Path(root);self.directory=Path(self.spec['owned_directory'])
        self.path=self.directory/'FINALIZER.jsonl';self.header_path=self.directory/'FINALIZER_HEADER.json'
        self.seal_path=self.directory/'FINALIZER_SEAL.json'
        self.header={'schema':SCHEMA,'native':native,'request_sha256':request_sha256,'scope_sha256':scope_sha256,
            'specification':copy.deepcopy(self.spec),'original_origin':self.origin,
            'permanent_reserved_bounds':copy.deepcopy(self.spec['self_bounds']),
            'authority':'original source supervisor metadata permit; no model authority'}
        require(not self.directory.is_symlink(),'regular finalizer owner')
        self.directory.mkdir(parents=True,exist_ok=True)
        require(not any(os.path.lexists(path) for path in (self.path,self.header_path,self.seal_path)), 'create-only finalizer owner files')
        blob=canonical(self.header)+b'\n';require(len(blob)<=MAX_JSON,'bounded finalizer header')
        self._charge({'operation_attempts_upper':1,'io_write_bytes_upper':len(blob),'metadata_bytes_upper':len(blob),
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':len(blob)})
        try:
            with self.header_path.open('xb') as stream:
                require(stream.write(blob)==len(blob),'short finalizer header');stream.flush();os.fsync(stream.fileno())
        except BaseException:self.fatal=True;raise
        self.header_ref={'path':str(self.header_path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
        self._event('permanent_self_reservation',{'bounds':self.spec['self_bounds'],'original_origin':self.origin})
        self.reader=raw.RawEvidenceReader(self.operation,self.spec['reader_bounds'],self.root)

    def _require(self):
        require(type(self) is FinalizerCostOwner and self._seal is _OWNER_SEAL
                and not self.closed and not self.fatal and canonical(self.spec)==self._spec_bytes,
                'actual live immutable finalizer cost owner')

    def _charge(self,values):
        _bounds(self.spec['self_bounds'])
        for key,value in values.items():
            require(key in COST_FIELDS and type(value) in (int,float) and not isinstance(value,bool)
                    and 0<=value<float('inf') and (key=='wall_seconds_upper' or type(value) is int)
                    and self.counts[key]+value<=self.spec['self_bounds'][key],'permanent finalizer allowance exhausted: '+key)
            if self._suffix_before is not None:
                require(self.counts[key]-self._suffix_before[key]+value<=self.spec['finalization_bounds'][key],
                        'permanent finalizer suffix allowance exhausted: '+key)
        for key,value in values.items():self.counts[key]+=value

    def _event(self,kind,payload):
        self._require()
        body={'schema':SCHEMA,'seq':len(self.events),'previous':self.events[-1]['sha256'] if self.events else None,
            'request_sha256':self.request_sha256,'scope_sha256':self.scope_sha256,'kind':kind,'payload':copy.deepcopy(payload)}
        preimage=canonical(body);row=dict(body,sha256=digest(body));blob=canonical(row)+b'\n'
        require(len(blob)<=MAX_JSON,'bounded finalizer event')
        self._charge({'io_write_bytes_upper':len(blob),'metadata_bytes_upper':len(blob),
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':len(preimage)})
        try:
            require(not self.path.exists() and self._known_size==0 or self.path.stat().st_size==self._known_size,
                    'original single-writer finalizer journal changed')
            with self.path.open('ab') as stream:
                require(stream.write(blob)==len(blob),'short finalizer append');stream.flush();os.fsync(stream.fileno())
        except BaseException:self.fatal=True;raise
        self.events.append(row);self._known_size+=len(blob)

    def tick(self):
        self._require();now=self.clock()
        if self._delta_rebuild_started is not None:
            require(0<=now-self._delta_rebuild_started<=self._delta_rebuild_wall,
                    'registered finalizer delta rebuild wall ceiling exceeded')
        require(type(now) in (int,float) and self.last_clock<=now<float('inf')
                and now-self.origin<=self.spec['physical_wall_seconds_upper'],'original finalizer monotonic deadline')
        if self._suffix_origin is not None:require(now-self._suffix_origin<=self.spec['finalization_bounds']['wall_seconds_upper'],'finalizer suffix deadline')
        elapsed=now-self.last_clock
        if elapsed:
            self._charge({'wall_seconds_upper':elapsed})
            self._event('wall',{'original_origin':self.origin,'monotonic':now,'elapsed_delta':elapsed})
            self.last_clock=now

    def operation(self,kind,phase,operation_id,**fields):
        delta_scope=(kind=='qualification_reducer_overhead'
                     and fields.get('reason')=='delta_readback_source_report_linkage')
        if delta_scope and phase=='started':
            require(self._delta_rebuild_started is not None and self._delta_rebuild_operation is None,
                    'registered source delta window before reader operation')
            self._delta_rebuild_operation=operation_id
        closing=delta_scope and phase in ('returned','failed')
        if closing:require(self._delta_rebuild_operation==operation_id,'same source delta reader closes finalizer window')
        try:return self._operation(kind,phase,operation_id,**fields)
        finally:
            if closing:self._delta_rebuild_started=self._delta_rebuild_operation=self._delta_rebuild_wall=None

    def _operation(self,kind,phase,operation_id,**fields):
        self._require();require(kind in KINDS and phase in ('started','returned','failed')
            and type(operation_id) is str and operation_id,'finite source finalizer operation')
        self.tick()
        if phase=='started':
            require(operation_id not in self.operations,'unique finalizer attempt')
            wanted={'operation_attempts_upper':fields.get('operation_attempts_upper',1)}
            for key in COST_FIELDS:
                if key in ('operation_attempts_upper','wall_seconds_upper'):continue
                alias={'io_read_bytes_upper':'read_bytes_upper','io_write_bytes_upper':'write_bytes_upper'}.get(key)
                for present in (key,alias):
                    if present in fields:require(type(fields[present]) is int and fields[present]>=0,'strict finalizer declared count')
                require(alias not in fields or key not in fields or fields[alias]==fields[key],'conflicting finalizer IO aliases')
                wanted[key]=fields.get(key,fields.get(alias,0))
            self._charge(wanted)
            self.operations[operation_id]={'kind':kind,'phase':'started','reservation':copy.deepcopy(wanted)}
        else:
            previous=self.operations.get(operation_id)
            require(previous and previous['kind']==kind and previous['phase']=='started','matching finalizer terminal operation')
            for field,upper in {'bytes_read':'io_read_bytes_upper','bytes_written':'io_write_bytes_upper',
                'identity_hash_attempts_actual':'identity_hash_attempts_upper','identity_hash_bytes_actual':'identity_hash_bytes_upper'}.items():
                if field in fields:
                    if not (type(fields[field]) is int and 0<=fields[field]<=previous['reservation'][upper]):
                        self.fatal=True;raise ValueError('actual finalizer work exceeded permanent reservation: '+field)
        self._event('operation',{'kind':kind,'phase':phase,'operation_id':operation_id,'fields':fields})
        if phase!='started':self.operations[operation_id]['phase']=phase

    def preflight(self,name,contract):
        self._require();require(name in PREFLIGHTS and type(contract) is dict,'finite original finalizer preflight')
        self.tick()
        if name!='saved_delta_source_rebuild':require(contract.get('reader_bounds')==self.reader.remaining,'same prepaid finalizer reader allowance')
        else:
            wall=contract.get('wall_seconds_upper')
            require(type(wall) in (int,float) and wall==3600 and wall<=self.spec['physical_wall_seconds_upper']
                    and self._delta_rebuild_started is None,'explicit registered 3600-second delta rebuild ceiling')
            self._delta_rebuild_started=self.clock();self._delta_rebuild_operation=None;self._delta_rebuild_wall=wall
        self._event('preflight',{'name':name,'contract':contract})

    def observe_parent_terminal(self,handle,expected):
        self._require();require(self.native and type(handle) is worker.WinWorkerHandle
            and handle.observer is self.service.operation and self._terminal_observation is None,
            'original source parent terminal handle before any heavy final verification')
        require(type(expected) is dict and expected.get('request_sha256')==self.request_sha256
            and expected.get('scope_sha256')==self.scope_sha256,'same finalizer terminal scope')
        identifier=uuid.uuid4().hex;self.operation('finalizer_terminal_observation','started',identifier)
        before=self.clock();proof=worker.verify_worker_handle(handle,expected,require_native=True,terminal=True);after=self.clock()
        require(before<=after and type(proof.get('exit_code')) is int and proof['exit_code']==0
                and proof.get('status_file_used') is False,'actual successful parent OS terminal observation')
        observation=TerminalObservation(self,self.service,proof,before,after,seal=_OBSERVATION_SEAL)
        self._terminal_observation=observation
        self.operation('finalizer_terminal_observation','returned',identifier,terminal_proof=proof,observed_before=before,observed_after=after)
        return observation

    def observe_supervisor_end(self,service):
        self._require();require(self.native and service is self.service and self._terminal_observation is not None
            and self._service_end_observation is None,'same original supervisor and previously observed parent terminal')
        identifier=uuid.uuid4().hex;self.operation('finalizer_service_end','started',identifier)
        before=self.clock();service.bootstrap.end();evidence=service.bootstrap.inspect_evidence();after=self.clock()
        observation=ServiceEndObservation(self,service,evidence,before,after,seal=_OBSERVATION_SEAL)
        self._service_end_observation=observation
        self.operation('finalizer_service_end','returned',identifier,observed_before=before,observed_after=after,evidence=evidence)
        return observation

    def validate_service_suffix(self,replay,observation):
        self._require();require(type(observation) in (TerminalObservation,ServiceEndObservation)
            and observation._seal is _OBSERVATION_SEAL and observation._owner is self
            and observation._native is self.native and type(replay) is dict,'same source-owned opaque suffix observation')
        state=replay.get('derived_snapshot');require(type(state) is dict and state.get('request_sha256')==self.request_sha256
            and state.get('clock_ended') is True and not state.get('fatal') and not state.get('pending'), 'independently replayed ended service required')
        end=state['last_monotonic'];bounds=state['end_reservation']['bounds'];stamp=observation._after
        require(type(end) in (int,float) and end<=stamp<=end+bounds['wall_s_upper'],
                'actual observed service suffix outside original permanent wall reservation')
        if type(observation) is TerminalObservation:
            require(observation is self._terminal_observation and observation._proof['request_sha256']==self.request_sha256
                and observation._proof['scope_sha256']==self.scope_sha256,'original successful parent terminal binding')
        else:
            require(observation is self._service_end_observation and observation._service is self.service
                and observation._before<=end<=observation._after
                and observation._evidence['snapshot']==state,'actual original supervisor end/readback evidence')
        return {'observed_after':stamp,'recorded_end':end,'suffix_upper_seconds':bounds['wall_s_upper'],
                'observed_suffix_seconds_upper':stamp-end,'native_observation':self.native}

    def write_json(self,name,value):
        self._require();require(type(name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.json',name),'owned finite finalizer JSON name')
        if name=='QUALIFICATION_RECEIPT.json':self.begin_finalization()
        encode=uuid.uuid4().hex
        self.operation('finalizer_metadata','started',encode,reducer_work_bytes_upper=8*self.spec['max_receipt_bytes'])
        try:
            blob=canonical(value)+b'\n';require(len(blob)<=self.spec['max_receipt_bytes'],'finalizer receipt ceiling')
            self.operation('finalizer_metadata','returned',encode)
        except BaseException as error:
            self.operation('finalizer_metadata','failed',encode,error={'type':type(error).__name__,'message':str(error)[:1000]});raise
        identifier=uuid.uuid4().hex;maximum=self.spec['max_receipt_bytes']
        self.operation('finalizer_json_write','started',identifier,io_write_bytes_upper=len(blob),
            io_read_bytes_upper=len(blob)+1,identity_hash_attempts_upper=1,identity_hash_bytes_upper=len(blob),
            metadata_bytes_upper=len(blob))
        try:
            path=self.directory/name;require(not os.path.lexists(path),'create-only finalizer JSON')
            with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short finalizer JSON');stream.flush();os.fsync(stream.fileno())
            with path.open('rb') as stream:observed=stream.read(len(blob)+1)
            require(observed==blob,'finalizer raw JSON readback')
            reference={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
            self.receipts[name]=reference
            self.operation('finalizer_json_write','returned',identifier,reference=reference)
            return reference
        except BaseException as primary:
            try:self.operation('finalizer_json_write','failed',identifier,error={'type':type(primary).__name__,'message':str(primary)[:1000]})
            except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
            raise

    def begin_finalization(self):
        self._require()
        if self._suffix_origin is not None:return
        self.tick();self._suffix_origin=self.clock();self._suffix_before=dict(self.counts)
        self._event('permanent_final_suffix',{'bounds':self.spec['finalization_bounds'],'monotonic':self._suffix_origin})

    def close_and_seal(self):
        self._require();require(not any(op['phase']=='started' for op in self.operations.values()),'all finalizer operations terminal before seal')
        self.begin_finalization()
        self.tick();self._event('end',{'original_origin':self.origin,'monotonic':self.last_clock,'receipts':self.receipts})
        snapshot=self.snapshot();blob=canonical(snapshot)+b'\n';require(len(blob)<=MAX_JSON,'bounded finalizer cost seal')
        self._charge({'operation_attempts_upper':1,'io_write_bytes_upper':len(blob),'metadata_bytes_upper':len(blob)})
        references={}
        try:
            with self.seal_path.open('xb') as stream:require(stream.write(blob)==len(blob),'short finalizer cost seal');stream.flush();os.fsync(stream.fileno())
            for key,path in (('header_ref',self.header_path),('journal_ref',self.path),('seal_ref',self.seal_path)):
                size=path.stat().st_size;self._charge({'operation_attempts_upper':1,'io_read_bytes_upper':size+1,'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':size+1})
                hashed=hashlib.sha256();count=0
                with path.open('rb') as stream:
                    while block:=stream.read(min(raw.CHUNK,size+1-count)):
                        count+=len(block);require(count<=size,'finalizer cost evidence grew');hashed.update(block)
                require(count==size,'finalizer cost evidence truncated')
                references[key]={'path':str(path),'sha256':hashed.hexdigest(),'bytes':count}
        except BaseException:self.fatal=True;raise
        # The frozen seal refers to the exact ended journal prefix. Final
        # inspection consumes its permanent suffix without appending/reopening.
        now=self.clock();require(self.last_clock<=now and now-self.origin<=self.spec['physical_wall_seconds_upper']
            and now-self._suffix_origin<=self.spec['finalization_bounds']['wall_seconds_upper'],'finalizer inspection deadline')
        self._charge({'wall_seconds_upper':now-self.last_clock});self.last_clock=now;self.closed=True
        result={'schema':SCHEMA,**references,'receipt_refs':copy.deepcopy(self.receipts),'owner_closed':True,
            'permanent_reserved_bounds':copy.deepcopy(self.spec['self_bounds']),
            'consumed_declared_bounds':copy.deepcopy(self.counts),'native_qualified':False,
            'claim':'source-enforced permanent self costs; previous ended owners independently replayed, no recursive self qualification'}
        self._final_evidence=result;self._final_evidence_bytes=canonical(result)
        return result

    def require_closed_evidence(self,evidence):
        require(type(self) is FinalizerCostOwner and self._seal is _OWNER_SEAL and self.closed and not self.fatal
            and not any(op['phase']=='started' for op in self.operations.values())
            and type(evidence) is dict and evidence is self._final_evidence and self._final_evidence_bytes is not None
            and canonical(evidence)==self._final_evidence_bytes and canonical(self.spec)==self._spec_bytes,
            'exact successful source-owned finalizer close evidence required')
        return evidence

    def fail(self,primary):
        require(type(primary) is not dict and isinstance(primary,BaseException),'original finalizer failure required')
        if self.closed:return self.snapshot()
        try:
            self._require();self._event('failed',{'type':type(primary).__name__,'message':str(primary)[:1000],
                'pending':[key for key,value in self.operations.items() if value['phase']=='started'],
                'permanent_reserved_bounds':self.spec['self_bounds'],'charges_refunded':False})
        finally:self.fatal=True
        return self.snapshot()

    def snapshot(self):
        return {'schema':SCHEMA,'request_sha256':self.request_sha256,'scope_sha256':self.scope_sha256,
            'header_ref':copy.deepcopy(self.header_ref),'head':self.events[-1]['sha256'],
            'permanent_reserved_bounds':copy.deepcopy(self.spec['self_bounds']),'consumed_declared_bounds':copy.deepcopy(self.counts),
            'pending':[key for key,value in self.operations.items() if value['phase']=='started'],
            'failed':[key for key,value in self.operations.items() if value['phase']=='failed'],
            'original_origin':self.origin,'last_monotonic':self.last_clock,'fatal':self.fatal,'closed':self.closed,
            'receipt_refs':copy.deepcopy(self.receipts),'native_qualified':False}


def fixture_owner(spec,output_root,*,clock=None):
    """Explicit pure cost/IO fixture, incapable of observing native OS proof."""
    owner=object.__new__(FinalizerCostOwner)
    owner._initialize(spec,output_root,'a'*64,'b'*64,None,native=False,clock=time.monotonic if clock is None else clock)
    return owner
