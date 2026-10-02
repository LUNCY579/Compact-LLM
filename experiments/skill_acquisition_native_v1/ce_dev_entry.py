"""Fixed source DEV read entry; no training permits or default activation.

Only a compiled metadata declaration and existing completed-training source pin
select five fixed read roles. Native OS proof and the exact owned runtime lease
precede DevReadPermit. JSON/RPC may not select implementations or metadata paths.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import re
from pathlib import Path
import sys
import time
import uuid
from contextlib import contextmanager,redirect_stdout
from types import MappingProxyType

from . import ce_dev_runtime as runtime,ce_native_services as services,ce_worker_evidence as worker
from . import ce_qualification_artifacts as raw,ce_checkpoint_local_issuer as rpc,ce_checkpoint_lease as locks
from . import ce_qualification_configuration as intake,update

SCHEMA='CE8-DEV-SOURCE-CONFIGURATION-1'
SPEC='CE8-DEV-ENTRY-SPEC-1'
ENTRY_MODULE=__package__+'.ce_dev_entry'
RUN_IDS=tuple(row[0] for row in runtime.RUNS)
_SOURCE_DECLARATION=_SOURCE_DECLARATION_SHA256=_CONFIGURATION=None
_CTOR_SEAL=object()
require,canonical,digest=raw.require,raw.canonical,raw.digest


def _source_pin():
    require(type(_SOURCE_DECLARATION) is MappingProxyType and raw._hex(_SOURCE_DECLARATION_SHA256),
            'compiled DEV source declaration absent; entry closed')
    value=copy.deepcopy(dict(_SOURCE_DECLARATION))
    require(set(value)=={'configuration_path','output_root','source_checkout','training_selection_sha256','intake_bounds'}
        and all(type(value[key]) is str and Path(value[key]).is_absolute() for key in ('configuration_path','output_root','source_checkout'))
        and raw._hex(value['training_selection_sha256']) and digest(value)==_SOURCE_DECLARATION_SHA256,
        'immutable literal DEV metadata declaration')
    bounds=value['intake_bounds']
    require(type(bounds) is dict and set(bounds)==set(services.COUNTERS)
        and all(type(bounds[key]) is int and bounds[key]>0 for key in services.COUNTERS if key!='wall_seconds')
        and type(bounds['wall_seconds']) in (int,float) and 0<bounds['wall_seconds']<=120,
        'finite DEV intake permanent envelope before files')
    require(bounds['read_bytes']>=intake.MAX_CONFIGURATION+1 and bounds['hash_attempts']>=16
        and bounds['hash_bytes']>=4*intake.MAX_CONFIGURATION and bounds['write_bytes']>=65536
        and bounds['metadata_bytes']>=65536 and bounds['operation_attempts']>=8,
        'DEV whole configuration read/self publication envelope')
    from . import ce_completed_training as trained
    selected=trained.selected_training() # no actual source pin refuses before FS
    require(digest(dict(selected))==value['training_selection_sha256'],'source completed training pin differs')
    return value,selected


def _configuration():
    global _CONFIGURATION
    declaration,selected=_source_pin()
    if _CONFIGURATION is not None:
        require(_CONFIGURATION['training_selection_sha256']==declaration['training_selection_sha256'],'DEV source configuration cannot switch')
        return copy.deepcopy(_CONFIGURATION)
    policy={'owned_directory':str(Path(declaration['output_root'])/'entry-intake'),'bounds':declaration['intake_bounds']}
    # Each of five independently launched readers has its own immutable intake.
    role=_fixed_read_id(sys.argv[1:])
    directory=Path(policy['owned_directory']);require(not directory.is_symlink(),'regular source DEV intake owner')
    directory.mkdir(parents=True,exist_ok=True)
    journal=directory/(role+'.jsonl');require(not os.path.lexists(journal),'create-only fixed DEV role intake')
    origin=time.monotonic();counts=dict.fromkeys(policy['bounds'],0)
    with journal.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
    head=intake._append(journal,_SOURCE_DECLARATION_SHA256,None,'PERMANENT_RESERVATION',
        {'bounds':policy['bounds'],'read_id':role,'metadata_only':True},policy['bounds'],counts,origin)
    try:
        ref={'path':declaration['configuration_path']};path=Path(ref['path'])
        require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=intake.MAX_CONFIGURATION,'bounded fixed DEV config')
        with path.open('rb') as stream:blob=stream.read(intake.MAX_CONFIGURATION+1)
        counts['read_bytes']+=len(blob);counts['hash_attempts']+=3;counts['hash_bytes']+=3*intake.MAX_CONFIGURATION
        require(all(counts[key]<=policy['bounds'][key] for key in counts),'DEV intake bounds exhausted')
        config=raw._json(blob);validate_configuration(config)
        require(config['training_selection']==dict(selected) and config['training_selection_sha256']==declaration['training_selection_sha256'],
                'fixed source DEV completed-training dependency')
        intake._append(journal,_SOURCE_DECLARATION_SHA256,head,'RETURNED',{'configuration_sha256':config['configuration_sha256'],
            'file_sha256':hashlib.sha256(blob).hexdigest(),'read_id':role},policy['bounds'],counts,origin)
        _CONFIGURATION=copy.deepcopy(config);return config
    except BaseException as primary:
        try:intake._append(journal,_SOURCE_DECLARATION_SHA256,head,'FAILED',{'type':type(primary).__name__,'message':str(primary)[:1000],
            'charges_refunded':False},policy['bounds'],counts,origin)
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise


def _fixed_read_id(argv):
    parser=argparse.ArgumentParser();parser.add_argument('--read-id',required=True,choices=RUN_IDS);parser.add_argument('--request')
    return parser.parse_args(argv).read_id


def _fixed_request(argv,config,role):
    parser=argparse.ArgumentParser();parser.add_argument('--read-id',required=True,choices=RUN_IDS);parser.add_argument('--request')
    args=parser.parse_args(argv)
    require(args.read_id==role and (args.request is None or Path(args.request)==Path(config['context_refs'][role]['path'])),
            'request argument must equal source-selected DEV context')


def validate_configuration(value):
    require(type(value) is dict and set(value)=={'schema','context_refs','entry_specs','training_selection',
        'training_selection_sha256','configuration_sha256'} and value['schema']==SCHEMA,'exact fixed DEV source configuration')
    require(set(value['context_refs'])==set(value['entry_specs'])==set(RUN_IDS)
        and digest(value['training_selection'])==value['training_selection_sha256']
        and digest({key:item for key,item in value.items() if key!='configuration_sha256'})==value['configuration_sha256'],
        'five source-pinned contexts/specs and immutable DEV config')
    for role in RUN_IDS:
        runtime._reference(value['context_refs'][role]);spec=value['entry_specs'][role]
        require(type(spec) is dict and set(spec)=={'schema','bootstrap','constructor_bounds','resource','worker','lease','rpc_spec','training_reader_bounds'}
            and spec['schema']==SPEC,'fixed DEV entry composition')
        bounds=spec['constructor_bounds']
        require(type(bounds) is dict and set(bounds)==set(services.COUNTERS) and all(type(bounds[key]) is int and bounds[key]>0
            for key in services.COUNTERS if key!='wall_seconds') and type(bounds['wall_seconds']) in (int,float)
            and 0<bounds['wall_seconds']<float('inf'),'finite DEV constructor envelope')
        require(spec['worker']['entry_module']==ENTRY_MODULE and spec['worker']['request_ref']==value['context_refs'][role]
            and spec['worker']['argv'][1:]==['-m',ENTRY_MODULE,'--request',value['context_refs'][role]['path'],'--read-id',role],
            'fixed DEV raw context/entry/whole argv')
        rpc.validate_spec(spec['rpc_spec'])
        reader=spec['training_reader_bounds']
        require(type(reader) is dict and set(reader)==set(raw.BUDGET_FIELDS)
            and all(type(amount) is int and amount>0 for amount in reader.values()),'explicit completed-training reader allowance')
        services._profile(spec['bootstrap']['operation_profiles']['dev_entry_verify'])
        for kind in ('qualification_artifact_readback','qualification_reducer_overhead','worker_runtime_binding'):
            services._profile(spec['bootstrap']['operation_profiles'][kind])
        require(re.fullmatch(r'comp-rs-001-[a-z0-9][a-z0-9.-]{2,80}',spec['lease']['label']) is not None,
                'source label matches pinned runtime lease protocol')
    return value


def _read(ref,operation,profile):
    runtime._reference(ref);require(profile['read_bytes_upper']>=ref['bytes']+1
        and profile['identity_hash_attempts_upper']>=1 and profile['identity_hash_bytes_upper']>=ref['bytes']+1,'prepaid raw context/source bytes')
    with update._span(operation,'dev_entry_source',dict(profile)):
        path=Path(ref['path']);require(path.is_file() and not path.is_symlink() and path.stat().st_size==ref['bytes'],'fixed regular DEV source file')
        with path.open('rb') as stream:blob=stream.read(ref['bytes']+1)
        require(len(blob)==ref['bytes'] and hashlib.sha256(blob).hexdigest()==ref['sha256'],'DEV source raw bytes/hash drift')
        return blob


class DevConstructorPrepayment:
    """One exact native BootstrapLedger constructor, independently prepaid."""
    def __init__(self,owner):
        require(any(DevConstructorPrepayment is item for item in services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES),
                'source DEV constructor type remains closed')
        require(type(owner) is DevEntryOwner,'exact original DEV entry constructor owner')
        owner._admit();spec=owner.spec['bootstrap'];bounds=owner.spec['constructor_bounds'];started=time.monotonic()
        self._seal,self.owner,self.spec_sha,self.request_sha=_CTOR_SEAL,owner,digest(spec),None
        self.consumed=False
        ref=owner.config['context_refs'][owner.role]
        record={'schema':'CE8-DEV-CONSTRUCTOR-PREPAYMENT-1','context_ref':copy.deepcopy(ref),'bootstrap_spec_sha256':self.spec_sha,
            'bounds':copy.deepcopy(bounds),'status':'PERMANENTLY_RESERVED','evaluation_only':True,'model_authority':False}
        blob=canonical(record)+b'\n';setup=services._profile(spec['setup_bounds'])
        require(bounds['write_bytes']>=len(blob)+setup['write_bytes_upper']+2*raw.MAX_JSON
            and bounds['metadata_bytes']>=len(blob)+2*raw.MAX_JSON and bounds['hash_attempts']>=setup['identity_hash_attempts_upper']+16
            and bounds['hash_bytes']>=setup['identity_hash_bytes_upper']+16*raw.MAX_JSON
            and bounds['read_bytes']>=setup['read_bytes_upper']+ref['bytes']+1 and bounds['wall_seconds']>=setup['wall_s_upper'],
            'DEV constructor full dependent/self metadata costs before files')
        declaration,_=_source_pin()
        directory=Path(spec['owned_directory']);require(directory.is_absolute() and not directory.is_symlink()
            and directory.resolve().is_relative_to(Path(declaration['output_root']).resolve()),'source DEV constructor owner')
        directory.mkdir(parents=True,exist_ok=True);path=directory/'DEV_CONSTRUCTOR_PREPAYMENT.json'
        with path.open('xb') as stream:require(stream.write(blob)==len(blob),'short DEV prepayment');stream.flush();os.fsync(stream.fileno())
        self.reference={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
        self.started,self.bounds=started,copy.deepcopy(bounds)
        require(time.monotonic()-started<=bounds['wall_seconds'],'DEV constructor physical bound')
    def require_constructor(self,spec_sha256,request_sha256):
        require(type(self) is DevConstructorPrepayment and self._seal is _CTOR_SEAL and not self.consumed
            and spec_sha256==self.spec_sha and request_sha256==self.request_sha,'one source DEV constructor prepayment')
        self.owner._admit();self.consumed=True
    def read_context(self):
        require(self.request_sha is None and not self.consumed,'one prepaid source context read')
        ref=self.owner.config['context_refs'][self.owner.role];path=Path(ref['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==ref['bytes'],'fixed source DEV raw context')
        with path.open('rb') as stream:blob=stream.read(ref['bytes']+1)
        require(len(blob)==ref['bytes'] and hashlib.sha256(blob).hexdigest()==ref['sha256']
            and time.monotonic()-self.started<=self.bounds['wall_seconds'],'prepaid source DEV context bytes/deadline')
        context=raw._json(blob);runtime.validate_context(context);self.request_sha=context['request_sha256']
        return context


def metadata_verifier(issuer,spec_sha,request_sha,scope_sha):
    require(type(issuer) is DevEntryOwner,'source-owned DEV metadata issuer')
    issuer._admit();require(type(issuer._worker_spec) is dict and digest(issuer._worker_spec)==spec_sha
        and issuer.context['request_sha256']==request_sha and issuer.context['scope_sha256']==scope_sha,'exact source DEV observation spec')
    return True


def dev_verifier(issuer,context_sha,scope_sha,request_sha):
    require(type(issuer) is DevEntryOwner,'source DEV worker and lease issuer')
    issuer._admit();require(context_sha==digest(runtime.context_preimage(issuer.context))
        and scope_sha==issuer.context['scope_sha256'] and request_sha==issuer.context['request_sha256'],'fixed source DEV context')
    handle=issuer.dev_worker_handle;require(type(handle) is worker.WinWorkerHandle and handle.native and not handle.closed
        and handle.observer is issuer.operation and handle.identity_sha256==issuer.context['worker_proof_sha256'],
        'original retained DEV worker identity')
    require(handle.identity_sha256==digest(worker.worker_identity_preimage(handle.identity,handle.sources,request_sha,scope_sha))
        and {name:ref['sha256'] for name,ref in handle.sources['source_files'].items()}==issuer.context['source']['code_sha256']
        and handle.spec['source_files']==issuer.context['source']['source_files'],
        'cached source/OS identity preimage remains exact')
    issuer.dev_lease_owner.assert_owned()
    return True


class _LeaseOwner:
    def __init__(self,owner,module,manager,token,record):
        self.owner,self.module,self.manager,self.token,self.record=owner,module,manager,token,copy.deepcopy(record)
        self.closed=False
    def assert_owned(self):
        require(self.owner.dev_lease_owner is self and not self.closed,'exact current source-owned DEV lease pointer')
        profile=self.owner.spec['lease']['operation_bounds']['assert']
        with update._span(self.owner.operation,'dev_entry_lease_assert',dict(profile)):
            record,_=locks._read_lock(Path(self.owner.spec['lease']['lock_path']),self.owner.spec['lease']['lock_bytes_upper'])
            require(locks._owner(record)==locks._owner(self.record) and record['token']==self.token
                and record['pid']==self.owner.dev_worker_handle.identity['pid'] and record['worker_pids']==[],
                'original DEV lock token/PID/create marker changed')


class DevEntryOwner:
    def __init__(self,config,role):
        require(config==_configuration() and role in RUN_IDS,'source-fixed DEV configuration/read role before objects')
        self.config,self.role,self.spec=config,role,copy.deepcopy(config['entry_specs'][role]);self._spec_sha=digest(self.spec)
        self.context={'request_sha256':'0'*64};self.dev_worker_handle=self.dev_lease_owner=None;self._worker_spec=None
        # Context logical identities are bound by the source raw ref. Read it
        # under the independent compiled intake before the dependent ledger.
        prepayment=DevConstructorPrepayment(self)
        self.context=prepayment.read_context()
        require(self.context['read_id']==role,'source context cannot select another read')
        self.bootstrap=services.BootstrapLedger(self.spec['bootstrap'],self.context['request_sha256'],prepayment=prepayment)
        self.operation=self.bootstrap.operation
        self.dev_verify_profile=copy.deepcopy(self.spec['bootstrap']['operation_profiles']['dev_entry_verify'])
        self.resource_gate=services.NativeResourceGate(self.spec['resource'],self.operation);self.gate=self._gate
        self.context['native_services']={'operation':self.operation,'gate':self.gate}
        from . import ce_completed_training as trained
        _,selected=_source_pin()
        reader=raw.RawEvidenceReader(self.operation,self.spec['training_reader_bounds'],selected['owned_root'])
        def preflight(name,contract):
            require(name in ('completed_training_pair','completed_training_CE_journal','completed_training_service') and contract['reader_bounds']==reader.remaining,
                    'exact source completed training dependency allowance')
            self.bootstrap.tick_wall()
        completed=trained.verify_completed_training(selected,reader=reader,outer_preflight=preflight)
        model=self.context['model'];donor=completed['donor_ref']
        require(completed['completed_pair_verified'] is True and completed['updates']==375
            and completed['runtime']==self.context['runtime'] and completed['warm_common_sha256']==model['warm_anchor_sha256']
            and donor['path']==model['donor_ref']['path'] and donor['sha256']==model['donor_ref']['sha256']
            and donor['data_sha256']==model['donor_ref']['data_sha256'] and donor['deployed_arm_sha256']==model['CE375_donor_sha256'],
            'actual source-completed375 donor/runtime/common anchor before DEV permit')
    def _admit(self):
        require(self.config==_configuration() and digest(self.spec)==self._spec_sha,'immutable source DEV configuration/spec')
    def _gate(self,stage):
        if self.dev_lease_owner is not None:
            with update._span(self.operation,'dev_entry_verify',dict(self.dev_verify_profile)):
                dev_verifier(self,digest(runtime.context_preimage(self.context)),self.context['scope_sha256'],self.context['request_sha256'])
        else:self._admit()
        framework=sys.modules.get('torch')
        if framework is not None:self.resource_gate.framework=framework
        sample=self.resource_gate(stage)
        if framework is not None and framework.cuda.is_initialized():
            from .ce_resource_evidence import enforce
            bounds=self.spec['bootstrap']['operation_profiles']['resource_cuda_probe']
            with update._span(self.operation,'resource_cuda_probe',dict(bounds)) as returned:
                free,total=framework.cuda.mem_get_info(0)
                require(framework.cuda.device_count()==1,'one registered visible DEV GPU')
                sample.update(cuda_free_bytes=free,cuda_total_bytes=total,cuda_reserved_bytes=framework.cuda.memory_reserved(0),
                    cuda_peak_reserved_bytes=framework.cuda.max_memory_reserved(0))
                limits=copy.deepcopy(self.spec['resource']['limits'])
                limits['ongoing_host_gib']+=self.spec['resource']['host_transient_bytes'][stage]/1024**3
                enforce(sample,limits)
                returned.update(observation=copy.deepcopy(sample),bytes_read=0,bytes_written=0,
                    identity_hash_attempts_actual=0,identity_hash_bytes_actual=0)
            self.resource_gate.samples[-1]=copy.deepcopy(sample)
        return sample
    def observe_worker(self):
        with update._span(self.operation,'dev_entry_metadata',dict(self.spec['bootstrap']['operation_profiles']['dev_entry_metadata'])):
            return self._observe_worker()
    def close_worker(self):
        handle=self.dev_worker_handle
        require(type(handle) is worker.WinWorkerHandle,'original owned DEV OS handle before release')
        entered=False
        try:
            with update._span(self.operation,'dev_entry_metadata',dict(self.spec['bootstrap']['operation_profiles']['dev_entry_metadata'])):
                entered=True;handle.close()
        except BaseException as primary:
            if not entered:
                # Already-owned read-only handle deallocation survives denied
                # bookkeeping; it cannot introduce new observation/model work.
                try:handle.close()
                except BaseException as closing:raw._accounting(primary,closing);raise primary from closing
            raise
    def _observe_worker(self):
        self._admit();policy=self.spec['worker'];budget=worker._Budget(policy,self.operation)
        api=budget.run('worker_os_binding',worker._WindowsOS);handle=None;primary=None
        try:
            def open_current():
                nonlocal handle
                handle=api.open(os.getpid());return handle
            budget.run('worker_os_binding',open_current)
            facts=budget.run('worker_os_binding',lambda:api.snapshot(handle));boot=budget.run('worker_os_binding',api.boot)
            static=copy.deepcopy(policy)
            require(facts['pid']==os.getpid() and facts['argv']==static['argv'] and facts['image_path']==static['process_image_ref']['path'],
                    'actual current DEV worker image/whole argv')
            self._worker_spec=dict(static,schema=worker.SCOPE,mode='registered_metadata',expected_pid=facts['pid'],
                expected_creation_filetime_100ns=facts['creation_filetime_100ns'],expected_boot_identity=boot,
                request_sha256=self.context['request_sha256'],scope_sha256=self.context['scope_sha256'])
            permit=worker.WorkerObservationPermit(self._worker_spec,self,metadata_verifier,self.operation)
            self.dev_worker_handle=worker.open_worker(self._worker_spec,permit,self.operation)
            proof=self.dev_worker_handle.proof();self.context['worker_proof_sha256']=proof['worker_identity_sha256']
            return proof
        except BaseException as error:primary=error;raise
        finally:
            if handle is not None:
                try:budget.run('worker_handle_close',lambda:api.close(handle))
                except BaseException as failure:
                    try:api.close(handle)
                    except BaseException as closing:
                        if primary is not None:raw._accounting(primary,closing)
                    if primary is not None:raw._accounting(primary,failure);raise primary from failure
                    raise
    @contextmanager
    def lease(self):
        with update._span(self.operation,'dev_entry_verify',dict(self.dev_verify_profile)):
            self._admit();require(self.dev_worker_handle is not None and self.dev_lease_owner is None,'one owned DEV lease after actual worker')
        policy=self.spec['lease'];release_id=uuid.uuid4().hex
        self.operation('dev_entry_lease_release','started',operation_id=release_id,**policy['operation_bounds']['release'])
        manager=None;owner=None;primary=None;entered=False;released_start=time.perf_counter()
        try:
            with update._span(self.operation,'dev_entry_lease',dict(policy['operation_bounds']['acquire'])):
                _read(policy['helper_ref'],self.operation,self.spec['bootstrap']['operation_profiles']['dev_entry_source'])
                from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as rt
                require(policy['helper_ref']['sha256']==rt.GPU_LEASE_SHA and Path(policy['helper_ref']['path'])==rt.GPU_LEASE_PATH,'literal pinned runtime lease helper')
                module=rt.gpu_lease();require(Path(module.LOCK)==Path(policy['lock_path']),'exact shared DEV GPU lock')
                manager=rt.lease(policy['label'],policy['timeout_seconds'],module=module);token=manager.__enter__();entered=True
                record,_=locks._read_lock(Path(policy['lock_path']),policy['lock_bytes_upper'])
                require(record['token']==token and record['pid']==self.dev_worker_handle.identity['pid'],'actual own acquired DEV lease')
                owner=_LeaseOwner(self,module,manager,token,record);self.dev_lease_owner=owner
            yield owner
        except BaseException as error:primary=error;raise
        finally:
            cleanup=None
            if entered:
                try:manager.__exit__(type(primary) if primary else None,primary,primary.__traceback__ if primary else None)
                except BaseException as failure:
                    cleanup=failure
                    if primary is not None:raw._accounting(primary,failure)
                finally:
                    if owner is not None:owner.closed=True
                    self.dev_lease_owner=None
            failure=primary or cleanup
            try:self.operation('dev_entry_lease_release','failed' if failure else 'returned',operation_id=release_id,
                wall_s=time.perf_counter()-released_start,error_type=type(failure).__name__ if failure else None)
            except BaseException as accounting:
                if failure is not None:raw._accounting(failure,accounting);raise failure from accounting
                raise
            if cleanup is not None:
                if primary is not None:raise primary from cleanup
                raise cleanup


def serve_read(pipe,proof,read):
    pipe.emit('ready',0,{'worker_proof':proof})
    require(pipe.receive(1)=={'command':'read','data':{}},'one fixed DEV read command only')
    with redirect_stdout(sys.stderr):result=read()
    require(type(result) is dict and result.get('rows')==1950 and result.get('native_qualified') is False
        and type(result.get('body_ref')) is dict,'actual complete readonly DEV body')
    runtime._reference(result['body_ref'])
    pipe.emit('response',1,{'raw_artifact':result['body_ref'],'body_completed':True,'native_qualified':False,'os_terminal_verified':False})
    require(pipe.receive(2)=={'command':'finish','data':{}},'finish only after DEV body')
    pipe.emit('response',2,{'finished':True,'native_qualified':False});return 0


def main(argv=None):
    config=_configuration();role=_fixed_read_id(sys.argv[1:] if argv is None else argv)
    _fixed_request(sys.argv[1:] if argv is None else argv,config,role)
    worker._REGISTERED_METADATA_VERIFIERS|=frozenset((metadata_verifier,))
    runtime._REGISTERED_DEV_VERIFIERS|=frozenset((dev_verifier,))
    services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES|=frozenset((DevConstructorPrepayment,))
    owner=DevEntryOwner(config,role);primary=None
    try:
        owner.resource_gate.launch();proof=owner.observe_worker()
        binding=raw._json(os.environ['CE8_RPC_BINDING'])
        require(binding['request_sha256']==owner.context['request_sha256'] and binding['scope_sha256']==owner.context['scope_sha256'],'DEV RPC cannot choose scope')
        pipe=rpc.BoundedRPC(owner.spec['rpc_spec'],binding,os.getpid(),sys.stdin.buffer,sys.stdout.buffer,owner.operation)
        def read():
            with owner.lease() as lease:
                with update._span(owner.operation,'dev_entry_verify',dict(owner.dev_verify_profile)):
                    owner.context['dev_read_permit']=runtime.DevReadPermit(owner.context,owner,dev_verifier,owner.operation,owner.gate)
                return runtime.run_dev_read(owner.context,role,operation=owner.operation,gate=owner.gate,worker_handle=owner.dev_worker_handle,lease_owner=lease)
        return serve_read(pipe,proof,read)
    except BaseException as error:primary=error;raise
    finally:
        if owner.dev_worker_handle is not None:
            try:owner.close_worker()
            except BaseException as closing:
                if primary is not None:raw._accounting(primary,closing);raise primary from closing
                raise
        if primary is None:owner.bootstrap.end();owner.bootstrap.inspect_evidence()


if __name__=='__main__':
    raise SystemExit(importlib.import_module(__package__+'.ce_dev_entry').main())
