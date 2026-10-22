"""One supervisor GPU lock with serial, OS-verified child use; no native grants.

The parent suspends its own GPU use without releasing the shared lock. Children
verify a permit-bound observed delegation and never acquire or remove that lock.
Only retained terminal process handles let the supervisor resume or delegate to
another child. Native authority and all existing service registries stay closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from . import ce_native_authority as authority, ce_worker_evidence as worker, update

SPEC = 'CE8-CHECKPOINT-LEASE-SPEC-1'
DELEGATION = 'CE8-CHECKPOINT-GPU-DELEGATION-1'
FIXTURE = 'CE8-CHECKPOINT-LEASE-FIXTURE-1'
BRANCHES = ('uninterrupted', 'prefix', 'resumed')
PROFILE = ('read_bytes_upper', 'write_bytes_upper', 'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_s_upper')
_SEAL, _FIXTURE_SEAL, _RELEASE_SEAL, _TICKET_SEAL = object(), object(), object(), object()
require, canonical, digest = authority.require, authority.canonical, authority.digest


def validate_spec(spec):
    require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit supervisor lease scope required')
    require(type(spec.get('lock_path')) is str and Path(spec['lock_path']).is_absolute()
            and Path(spec['lock_path']).name == 'rineya-gpu-0.lock'
            and type(spec.get('label')) is str and spec['label'].startswith('comp-rs-001-'), 'exact shared GPU lock and label')
    require(type(spec.get('timeout_seconds')) in (int, float) and math.isfinite(spec['timeout_seconds'])
            and spec['timeout_seconds'] > 0 and type(spec.get('lock_bytes_upper')) is int and spec['lock_bytes_upper'] > 0,
            'finite lease wait and raw lock ceiling')
    require(type(spec.get('operation_bounds')) is dict and set(spec['operation_bounds']) == {'acquire', 'transfer', 'release'},
            'explicit acquisition/transfer/lifetime-release prices')
    for profile in spec['operation_bounds'].values():
        require(type(profile) is dict and set(profile) == set(PROFILE)
                and all(type(profile[key]) is int and profile[key] >= 0 for key in PROFILE[:-1])
                and type(profile['wall_s_upper']) in (int,float) and math.isfinite(profile['wall_s_upper']) and profile['wall_s_upper'] > 0,
                'strict prepaid lease profile')
        require(profile['read_bytes_upper'] >= 2*(spec['lock_bytes_upper']+1)
                and profile['identity_hash_attempts_upper'] >= 2
                and profile['identity_hash_bytes_upper'] >= 2*spec['lock_bytes_upper'], 'raw lease ownership verification lower bound')
    require(type(spec.get('terminal_verify_timeout_seconds')) in (int,float)
            and math.isfinite(spec['terminal_verify_timeout_seconds']) and spec['terminal_verify_timeout_seconds']>0,
            'bounded retained child terminal verification')
    require(spec['operation_bounds']['acquire']['wall_s_upper'] >= spec['timeout_seconds']
            and spec['operation_bounds']['transfer']['write_bytes_upper'] >= spec['lock_bytes_upper']
            and spec['operation_bounds']['transfer']['read_bytes_upper'] >= 4*(spec['lock_bytes_upper']+1)
            and spec['operation_bounds']['transfer']['identity_hash_attempts_upper'] >= 3
            and spec['operation_bounds']['transfer']['identity_hash_bytes_upper'] >= 3*spec['lock_bytes_upper'],
            'lease wait/update and original-owner verification must be prepaid')
    for key in ('parent_released_cuda_bytes_upper','child_released_cuda_bytes_upper'):
        require(type(spec.get(key)) is int and spec[key] >= 0, 'explicit released CUDA allocation bound')
    reference=spec.get('helper_ref')
    require(type(reference) is dict and set(reference)=={'path','sha256','bytes'} and type(reference['path']) is str
            and Path(reference['path']).is_absolute() and authority._hex(reference['sha256'])
            and type(reference['bytes']) is int and reference['bytes']>0, 'pinned lease helper raw reference')
    require(spec['operation_bounds']['acquire']['read_bytes_upper'] >= reference['bytes']+1+2*(spec['lock_bytes_upper']+1)
            and spec['operation_bounds']['acquire']['identity_hash_attempts_upper'] >= 3
            and spec['operation_bounds']['acquire']['identity_hash_bytes_upper'] >= reference['bytes']+2*spec['lock_bytes_upper'],
            'pinned helper load/ownership IO and hash lower bound')
    return copy.deepcopy(spec)


def _read_lock(path, maximum):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum, 'bounded regular current lease lock required')
    with path.open('rb') as stream: raw=stream.read(maximum+1)
    require(0<len(raw)<=maximum, 'lease raw lock ceiling')
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result, 'duplicate lease owner key'); result[key]=value
        return result
    record=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite lease owner')))
    require(type(record) is dict and type(record.get('pid')) is int and record['pid']>0
            and type(record.get('token')) is str and record['token']
            and type(record.get('created')) in (int,float) and math.isfinite(record['created'])
            and type(record.get('label')) is str and type(record.get('protocol')) is str
            and type(record.get('worker_pids')) is list
            and all(type(pid) is int and pid>0 for pid in record['worker_pids']), 'strict actual lease owner metadata')
    return record, {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def _owner(record): return {key:record[key] for key in ('pid','token','created','label','protocol')}
def _os_keys(identity): return {key:identity[key] for key in ('pid','creation_filetime_100ns','boot_identity','image_path','argv')}


class _Ticket:
    __slots__=('seal','owner','identity')
    def __init__(self,owner): self.seal,self.owner,self.identity=_TICKET_SEAL,owner,uuid.uuid4().hex


class _OwnedHelper:
    """Guard the pinned helper's own-token release with original owner identity."""
    def __init__(self,module,spec,parent_pid):
        self.module,self.spec,self.parent_pid,self.marker=module,spec,parent_pid,None
        self.LOCK=module.LOCK
    def acquire(self,*,timeout_s,label):
        token=self.module.acquire(timeout_s=timeout_s,label=label)
        record,_=_read_lock(Path(self.LOCK),self.spec['lock_bytes_upper'])
        require(record['token']==token and record['pid']==self.parent_pid and record['label']==label
                and record['worker_pids']==[], 'new pinned helper lock belongs to exact supervisor')
        self.marker=_owner(record)
        return token
    def _check(self,token):
        record,_=_read_lock(Path(self.LOCK),self.spec['lock_bytes_upper'])
        require(record['token']==token and self.marker is not None and _owner(record)==self.marker,
                'never modify or remove an unknown lease owner')
    def set_workers(self,token,pids): self._check(token); self.module.set_workers(token,pids)
    def release(self,token): self._check(token); self.module.release(token)


class SupervisorGpuLease:
    """Own the runtime lease context for the entire serial qualification unit."""
    def __init__(self,context,permit,parent_handle,backend,operation,gate):
        authority.require_execution_permit(permit,context,'checkpoint_resume',operation=operation,gate=gate)
        from .ce_native_backend import NativeCEBackend
        require(type(parent_handle) is worker.WinWorkerHandle and (backend is None or type(backend) is NativeCEBackend
                and backend.permit is permit and backend.operation is operation and backend.gate is gate),
                'exact source-owned parent worker/backend before lease construction')
        self._initialize(context,permit,parent_handle,backend,operation,gate,native=True)

    def _initialize(self,context,permit,parent_handle,backend,operation,gate,*,native,module=None,proof=None):
        self.context,self.permit,self.parent_handle,self.backend=context,permit,parent_handle,backend
        self.operation,self.gate,self.native=operation,gate,native
        self.spec=validate_spec(context['bindings']['scope']['checkpoint_lease'])
        self.context_preimage=copy.deepcopy(authority.context_preimage(context))
        self.module,self.fixture_proof=module,copy.deepcopy(proof)
        self.state,self.fatal,self.closed='NEW',False,False
        self.token=self.cm=self.ticket=self.child=None
        self.completed=[]; self.evidence=[]; self.owner_marker=self.initial_lock_ref=None
        self.last_finished_child=None
        self.release_id=self.release_started=self.release_seal=None
        self.seal=_SEAL if native else _FIXTURE_SEAL

    def _admit(self,*,recovery=False):
        require(self.seal is (_SEAL if self.native else _FIXTURE_SEAL) and not self.closed, 'source-owned live supervisor lease required')
        if self.native: authority.require_execution_permit(self.permit,self.context,'checkpoint_resume',operation=self.operation,gate=self.gate)
        else: require(self.context.get('schema')==FIXTURE and self.context.get('mode')=='synthetic_fixture'
                      and self.context.get('approved') is False and self.context.get('training_authorized') is False,
                      'fixture lease cannot promote native authority')
        require(authority.context_preimage(self.context)==self.context_preimage, 'frozen lease evidence changed')
        require(recovery or not self.fatal, 'failed supervisor lease cannot grant new GPU work')

    def _parent(self):
        if not self.native: return copy.deepcopy(self.fixture_proof)
        proof=worker.verify_worker_handle(self.parent_handle,{'request_sha256':self.context['request_sha256'],
            'scope_sha256':self.context['scope_sha256'],'worker_identity_sha256':self.parent_handle.identity_sha256})
        require(digest(proof)==self.context['worker_proof_sha256'] and proof['os_identity']['pid']==os.getpid(),
                'exact current original supervisor OS/source proof')
        return proof

    def _lock(self,workers=None):
        record,reference=_read_lock(Path(self.spec['lock_path']),self.spec['lock_bytes_upper'])
        require(record['token']==self.token and _owner(record)==self.owner_marker, 'current lease is not the original runtime owner')
        if workers is not None: require(record['worker_pids']==workers, 'current lease worker delegation changed')
        return record,reference

    @contextmanager
    def _span(self,action):
        from .ce_operation_contracts import LEASE_PARENT_CHECK, OPAQUE_LEASE_FRAME
        contract = {'wall_contract': LEASE_PARENT_CHECK if action == 'check_parent_use' else OPAQUE_LEASE_FRAME}
        try:
            with update._span(self.operation,'supervisor_lease',dict(self.spec['operation_bounds']['transfer'],lease_action=action,**contract)) as record:
                yield record
        except BaseException:
            self.fatal=True
            raise

    def acquire(self):
        self._admit(); require(self.state=='NEW', 'supervisor must not wait on its already-held lease')
        self.release_id,self.release_started=uuid.uuid4().hex,time.perf_counter()
        self.operation('supervisor_lease_release','started',operation_id=self.release_id,deferred_owned_lease_release=True,
                       **self.spec['operation_bounds']['release'])
        self.release_seal=_RELEASE_SEAL
        try:
            with update._span(self.operation,'supervisor_lease_acquire',dict(self.spec['operation_bounds']['acquire'],lease_action='acquire')):
                parent=self._parent()
                from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime
                if self.native:
                    reference=self.spec['helper_ref']
                    require(Path(reference['path'])==runtime.GPU_LEASE_PATH and reference['sha256']==runtime.GPU_LEASE_SHA
                            and runtime.GPU_LEASE_PATH.stat().st_size==reference['bytes'], 'original pinned runtime lease helper required')
                    self.module=runtime.gpu_lease()
                require(Path(self.module.LOCK)==Path(self.spec['lock_path']), 'pinned helper shared lock path mismatch')
                self.module=_OwnedHelper(self.module,self.spec,parent['os_identity']['pid'])
                self.cm=runtime.lease(self.spec['label'],self.spec['timeout_seconds'],module=self.module)
                self.token=self.cm.__enter__()
                record,reference=_read_lock(Path(self.spec['lock_path']),self.spec['lock_bytes_upper'])
                require(record['token']==self.token and record['pid']==parent['os_identity']['pid']
                        and record['label']==self.spec['label'] and record['worker_pids']==[], 'new runtime lock belongs to current supervisor')
                self.owner_marker,self.initial_lock_ref=_owner(record),reference
                self.state='PARENT'; self.evidence.append({'action':'acquired','lock_ref':reference,'parent_proof':parent,'native_qualified':False})
        except BaseException as primary:
            self.fatal=True
            if self.token is not None:
                try: self.close()
                except BaseException as cleanup:
                    primary.lease_cleanup_failure={'type':type(cleanup).__name__,'message':str(cleanup)[:1000]}
                    raise primary from cleanup
            raise
        return self

    def bind_backend(self,backend):
        self._admit(); require(self.state=='PARENT' and self.backend is None, 'bind one metadata backend under the existing supervisor lease')
        if self.native:
            from .ce_native_backend import NativeCEBackend
            require(type(backend) is NativeCEBackend and backend.permit is self.permit and backend.context is self.context
                    and backend.operation is self.operation and backend.gate is self.gate,
                    'exact same-scope native backend binding required')
        require(backend.model is None and backend.optimizer is None and getattr(backend,'tokenizer',None) is None
                and getattr(backend,'torch',None) is None and not getattr(backend,'closed',False),
                'backend factory must remain metadata-only before supervisor binding')
        self.backend=backend

    def release_parent_lease(self,context,permit,proof):
        self._admit(); require(context is self.context and permit is self.permit and self.state=='PARENT', 'one original parent lease suspension')
        with self._span('suspend_parent'):
            require(self.backend is not None, 'bind actual parent backend before delegation')
            require(proof==self._parent(), 'caller parent flags cannot replace actual retained handle proof')
            require(self.backend.model is None and self.backend.optimizer is None, 'actual parent model/Adam must be released before transfer')
            self._released_cuda(self.spec['parent_released_cuda_bytes_upper'])
            _,ref=self._lock([])
            self.ticket=_Ticket(self); self.state='SUSPENDED'
            self.evidence.append({'action':'parent_suspended','lock_ref':ref,'ticket_id':self.ticket.identity,'native_qualified':False})
        return self.ticket

    def _released_cuda(self,maximum):
        if not self.native: return
        framework=self.backend.torch
        require(framework is sys.modules.get('torch') and framework.cuda.is_initialized(), 'actual configured parent CUDA runtime required')
        framework.cuda.synchronize(0)
        require(framework.cuda.memory_allocated(0)<=maximum, 'parent CUDA allocations remain after model release')

    def bind_child(self,owned,branch,request_ref):
        self._admit(); require(self.state=='SUSPENDED' and self.child is None and len(self.completed)<len(BRANCHES)
                and branch==BRANCHES[len(self.completed)], 'exact serial checkpoint branch; no concurrent child use')
        with self._span('delegate_child'):
            parent=self._parent(); _,before=self._lock([])
            proof=self._child_proof(owned)
            require(owned.spec['request_ref']==request_ref and proof['sources']['request_ref']==request_ref
                    and owned.spec['request_sha256']==self.context['request_sha256'] and owned.spec['scope_sha256']==self.context['scope_sha256']
                    and owned.spec['parent_identity']=={key:parent['os_identity'][key] for key in ('pid','creation_filetime_100ns','boot_identity')},
                    'exact owned child raw request and original parent OS lineage')
            pid=proof['os_identity']['pid']; require(pid!=parent['os_identity']['pid'], 'child must be a distinct actual OS process')
            self.module.set_workers(self.token,[pid]); _,after=self._lock([pid])
            record={'schema':DELEGATION,'branch':branch,'ticket_id':self.ticket.identity,
                'request_sha256':self.context['request_sha256'],'scope_sha256':self.context['scope_sha256'],
                'parent_worker_identity_sha256':parent['worker_identity_sha256'],'parent_os_identity':parent['os_identity'],
                'child_worker_identity_sha256':proof['worker_identity_sha256'],'child_os_identity':proof['os_identity'],
                'child_request_ref':copy.deepcopy(request_ref),'lock_path':self.spec['lock_path'],
                'original_lock_ref':self.initial_lock_ref,'current_lock_ref':after,'owner_marker':copy.deepcopy(self.owner_marker),
                'use':'one owned serial child; shared lock remains held by supervisor','native_qualified':False}
            self.child=(owned,proof,copy.deepcopy(record)); self.state='CHILD'
            self.evidence.append({'action':'child_delegated','before_lock_ref':before,'delegation':copy.deepcopy(record)})
        return record

    def _child_proof(self,owned,*,terminal=False):
        if not self.native:
            if terminal:
                self.last_owned_terminal=owned.terminal(self.spec['terminal_verify_timeout_seconds'])
                return owned.terminal_proof()
            return owned.proof()
        from .ce_process_launch import OwnedProcess,_OWNED_SEAL
        require(type(owned) is OwnedProcess and owned._seal is _OWNED_SEAL and owned.native is True and not owned.closed
                and type(owned.monitor) is worker.WinWorkerHandle, 'actual source-owned retained native child process/monitor required')
        if terminal: self.last_owned_terminal=owned.terminal(self.spec['terminal_verify_timeout_seconds'])
        actual=worker.verify_worker_handle(owned.monitor,{'request_sha256':self.context['request_sha256'],
            'scope_sha256':self.context['scope_sha256'],'worker_identity_sha256':owned.worker_identity_sha256},terminal=terminal)
        require(owned.spec['approved_code_sha256']==self.context['bindings']['source']['code_sha256'],
                'owned child source bundle must equal the complete admitted source map')
        if not terminal:
            require({name:ref['sha256'] for name,ref in actual['sources']['source_files'].items()}==owned.spec['approved_code_sha256'],
                    'actual child observed source files differ from admitted source bundle')
        return actual

    def finish_child(self,owned,terminal):
        self._admit(recovery=True); require(self.state=='CHILD' and self.child is not None and owned is self.child[0], 'only current owned delegated child may return')
        with self._span('verify_child_terminal'):
            self._parent(); proof=self.child[1]; actual=self._child_proof(owned,terminal=True)
            require(type(terminal) is dict and terminal.get('status_file_used') is False
                    and terminal==self.last_owned_terminal
                    and actual.get('status_file_used') is False and type(actual.get('exit_code')) is int
                    and all(terminal.get(key)==actual.get(key) for key in ('worker_identity_sha256','exit_code','exit_filetime_100ns','request_sha256','scope_sha256'))
                    and _os_keys(terminal['os_identity'])==_os_keys(actual['os_identity'])==_os_keys(proof['os_identity']),
                    'same retained OS-terminal child required; status/response flags cannot establish quiescence')
            _,before=self._lock([proof['os_identity']['pid']])
            self.module.set_workers(self.token,[]); _,after=self._lock([])
            branch=self.child[2]['branch']; self.last_finished_child=owned; self.child=None; self.state='SUSPENDED'
            self.evidence.append({'action':'child_terminal','branch':branch,'terminal':copy.deepcopy(actual),
                                  'before_lock_ref':before,'current_lock_ref':after,'native_qualified':False})
            if actual['exit_code']!=0:
                self.fatal=True; raise authority.NativeAdmissionError('owned child exited unsuccessfully; no next child or parent model reload')
            self.completed.append(branch)
        return copy.deepcopy(actual)

    def mark_child_evidence_failed(self,owned,error):
        self._admit(recovery=True)
        require(isinstance(error,BaseException) and (self.child is not None and owned is self.child[0]
                or owned is self.last_finished_child and owned is not None), 'only exact owned child evidence failure may invalidate this unit')
        self.fatal=True
        with self._span('child_evidence_failed'):
            self.evidence.append({'action':'child_evidence_failed','error':{'type':type(error).__name__,'message':str(error)[:1000]},
                                  'native_qualified':False})

    def reacquire_parent_lease(self,context,permit,ticket):
        self._admit(recovery=True)
        require(context is self.context and permit is self.permit and type(ticket) is _Ticket and ticket.seal is _TICKET_SEAL
                and ticket is self.ticket and ticket.owner is self and self.state=='SUSPENDED' and self.child is None,
                'parent cannot resume before exact owned child OS quiescence')
        with self._span('resume_parent'):
            self._parent(); _,ref=self._lock([])
            if self.completed!=list(BRANCHES): self.fatal=True
            self.ticket=None; self.state='PARENT'
            self.evidence.append({'action':'parent_resumed','lock_ref':ref,'failed_unit':self.fatal,'native_qualified':False})

    def assert_parent_use(self):
        self._admit(); require(self.state=='PARENT' and self.child is None, 'parent GPU use is suspended while a child may use it')
        with self._span('check_parent_use'): self._parent(); self._lock([])

    def assert_delegated_metadata(self, owned, delegation):
        """Observe one exact delegated child without granting parent GPU use."""
        self._admit()
        require(self.state == 'CHILD' and self.child is not None and self.child[0] is owned
                and delegation == self.child[2] and self.backend is not None
                and self.backend.model is None and self.backend.optimizer is None,
                'delegated metadata requires the exact child and released parent model/Adam')
        with self._span('check_delegated_metadata'):
            self._parent()
            proof = self._child_proof(owned)
            require(proof == self.child[1] and proof['os_identity'] == delegation['child_os_identity'],
                    'delegated metadata child OS/source identity changed')
            self._lock([proof['os_identity']['pid']])

    def close(self):
        require(getattr(self,'seal',None) in (_SEAL,_FIXTURE_SEAL), 'source-owned lease release only')
        if self.closed: return
        require(self.child is None and self.state in ('NEW','PARENT','SUSPENDED'), 'live or unverified child keeps supervisor lock held')
        require(self.backend is None or self.backend.model is None and self.backend.optimizer is None,
                'parent model/Adam must be released before shared lock release')
        require(self.release_seal is _RELEASE_SEAL and self.release_id is not None, 'physical lease release must be prepaid before acquisition')
        primary=None
        try: self._admit(recovery=True)
        except BaseException as error: primary=error
        try:
            if self.token is not None and self.owner_marker is not None: self._lock([])
        except BaseException as error:
            if primary is None: primary=error
            else: primary.lease_owner_failure=str(error)[:1000]
        if self.native and getattr(self.backend,'torch',None) is not None:
            try:
                require(self.backend.torch is sys.modules.get('torch'), 'actual retained parent framework identity')
                if self.backend.torch.cuda.is_initialized(): self._released_cuda(self.spec['parent_released_cuda_bytes_upper'])
            except BaseException as error:
                # A live allocation forbids releasing exclusivity. The caller
                # must finish source-owned backend cleanup; keep the lock held.
                if primary is not None:
                    primary.lease_cuda_quiescence_failure=str(error)[:1000]
                    raise primary from error
                raise
        try:
            if self.cm is not None and self.token is not None: self.cm.__exit__(None,None,None)
        except BaseException as error:
            if primary is None: primary=error
            else: primary.lease_physical_release_failure=str(error)[:1000]
        self.closed=True; self.state='CLOSED'
        try:
            self.operation('supervisor_lease_release','failed' if primary else 'returned',operation_id=self.release_id,
                wall_s=time.perf_counter()-self.release_started,lease_action='release_original_supervisor',
                error_type=type(primary).__name__ if primary else None,unknown_external_helper_IO=True)
        except BaseException as error:
            if primary is None: primary=error
            else: primary.lease_release_accounting_failure=str(error)[:1000]
        if primary is not None: raise primary


class DelegatedGpuLease:
    """Child's source-permitted use of the parent's existing lock; never owns it."""
    def __init__(self,context,permit,handle,operation,gate):
        authority.require_execution_permit(permit,context,'checkpoint_resume',operation=operation,gate=gate)
        require(type(handle) is worker.WinWorkerHandle, 'actual retained current child OS handle required')
        self.context,self.permit,self.handle,self.operation,self.gate=context,permit,handle,operation,gate
        self.spec=validate_spec(context['bindings']['scope']['checkpoint_lease']); self.entered=False

    @contextmanager
    def enter_child(self,context,permit,proof,request):
        authority.require_execution_permit(permit,context,'checkpoint_resume',operation=self.operation,gate=self.gate)
        require(context is self.context and permit is self.permit and not self.entered, 'one exact child delegation context')
        with update._span(self.operation,'qualification_lease',dict(self.spec['operation_bounds']['transfer'],lease_action='child_enter')):
            actual=worker.verify_worker_handle(self.handle,{key:proof[key] for key in ('worker_identity_sha256','request_sha256','scope_sha256')})
            record=context.get('checkpoint_child_lease_delegation')
            lineage=context.get('checkpoint_child_observed_lineage')
            require(type(record) is dict and record.get('schema')==DELEGATION and proof==actual
                    and record.get('child_worker_identity_sha256')==actual['worker_identity_sha256']
                    and record.get('child_os_identity')==actual['os_identity'] and actual['os_identity']['pid']==os.getpid()
                    and record.get('branch')==request.get('branch')
                    and all(record.get(key)==context[key]==actual[key] for key in ('request_sha256','scope_sha256'))
                    and record.get('child_request_ref')==actual['sources']['request_ref'], 'actual source-permit-bound postlaunch delegation required')
            require(type(lineage) is dict and lineage.get('child_proof')==actual
                    and lineage.get('parent_worker_identity_sha256')==record['parent_worker_identity_sha256']
                    and lineage.get('parent_proof',{}).get('os_identity')==record['parent_os_identity'],
                    'delegation must equal the independently observed parent/child lineage')
            lock,current_ref=_read_lock(Path(self.spec['lock_path']),self.spec['lock_bytes_upper'])
            require(record.get('lock_path')==self.spec['lock_path'] and _owner(lock)==record.get('owner_marker')
                    and current_ref==record.get('current_lock_ref')
                    and lock['pid']==record['parent_os_identity']['pid'] and lock['pid']!=actual['os_identity']['pid']
                    and lock['worker_pids']==[actual['os_identity']['pid']], 'current original supervisor lock delegates exactly this child')
            self.entered=True
        primary=None
        try: yield copy.deepcopy(record)
        except BaseException as error: primary=error; raise
        finally:
            try:
                with update._span(self.operation,'qualification_lease',dict(self.spec['operation_bounds']['transfer'],lease_action='child_exit')):
                    authority.require_execution_permit(permit,context,'checkpoint_resume',operation=self.operation,gate=self.gate)
                    framework=sys.modules.get('torch')
                    require(framework is not None and framework.cuda.is_initialized(), 'actual child configured runtime required for release evidence')
                    framework.cuda.synchronize(0)
                    require(framework.cuda.memory_allocated(0)<=self.spec['child_released_cuda_bytes_upper'], 'child CUDA allocations remain before delegated return')
                    current,_=_read_lock(Path(self.spec['lock_path']),self.spec['lock_bytes_upper'])
                    require(_owner(current)==record['owner_marker'] and current['worker_pids']==[actual['os_identity']['pid']], 'child does not own or edit supervisor lock')
                    self.entered=False
            except BaseException as cleanup:
                if primary is not None:
                    primary.delegated_release_failure={'type':type(cleanup).__name__,'message':str(cleanup)[:1000]}
                    raise primary from cleanup
                raise


def fixture_supervisor(context,backend,parent_proof,module,operation):
    """Explicit metadata-only FSM fixture; cannot issue native child permission."""
    require(type(context) is dict and context.get('schema')==FIXTURE and context.get('mode')=='synthetic_fixture'
            and context.get('approved') is False and context.get('training_authorized') is False
            and parent_proof.get('fresh_process') is False and parent_proof.get('source_verified') is False,
            'explicit non-native lease fixture only')
    value=object.__new__(SupervisorGpuLease)
    value._initialize(context,None,None,backend,operation,lambda stage:None,native=False,module=module,proof=parent_proof)
    return value
