"""One cumulative round4 component candidate. Actual dispatch is NOT_READY.

Concrete typed Job-list birth, original-handle ownership, framed request parsing,
canonical store sequencing and completed caller-leaf checks are implemented.
There is no C kernel-RPC bootstrap, qualified O-loss keeper or external OE
producer. Passing these components cannot qualify the adopted complete contract.
No accepted library, operations API or historical counter is changed here.
"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import time

GIB=2**30
LIB_SHA='82f211ebd5a49dfa9c2b8cbb098566f459181fab29f9ee19a2c6a917989d9cb0'
OPS_SHA='ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30'
CONTRACT_SHA='61a20d059308ea84c83503876eda513ec52586204b0aae9df8051cb9b0b8b25d'
UV='C:/Users/clclc/.local/bin/uv.exe'
UV_SHA='5a7ec85884c2ccb1be560cb8fac3eb890df1adf49bfcc070a270ba70401bdd68'
JOB_LIST=0x2000D
HANDLE_LIST=0x20002
NOT_READY_GAPS=(
    'Actual C kernel-RPC/bootstrap and controlled caller process are not implemented',
    'External O-loss holder/takeover and atomic O/C birth graph are not implemented',
    'Externally witnessed completed OE disposition producer is not implemented',
    'Combined actual O/C/U enforcement and physical Windows Job-list/runtime binding are unqualified',
    'Actual independent caller10/observer2 stop and current immutable role/admission/lease path are unqualified',
)

class Refused(RuntimeError):pass

def need(value,why):
    if not value:raise Refused(why)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(65536),b''):h.update(b)
    return h.hexdigest()

def verify_file_pins(rows):
    """Physical file bytes, never an accepted=True assertion or launch permit."""
    need(isinstance(rows,list) and rows,'empty immutable binding set')
    for row in rows:
        need(set(row)=={'path','bytes','sha256'},'binding schema')
        need(type(row['bytes']) is int and row['bytes']>=0,'binding size')
        need(Path(row['path']).stat().st_size==row['bytes'],'binding size drift')
        need(sha_file(row['path'])==row['sha256'],'binding hash drift')
    return {'verified_files':len(rows),'actual_dispatch_qualified':False}

class AtomicUVBirth:
    """Exact Win32 signatures and STARTUPINFOEX Job/handle-list calls.

    The outer and inner Job handles are retained by this object. The supplied
    kernel is the real typed API or a controlled kernel in the sole fixture.
    OS support/loader identity and whole O/C lifetime ownership remain gates.
    There is no fallback to CreateProcess without a birth Job list.
    """
    def __init__(self,kernel,library,outer_job,inner_job,last_error,clock=time.monotonic):
        self.k=kernel;self.lib=library;self.outer=outer_job;self.inner=inner_job
        self.last_error=last_error;self.clock=clock
        self.attempted=False;self.info=None;self.partial_handles={};self.identity_value=None
        self.acknowledged_copies={};self.errors=[];self.created_at=None
        signatures=[
          ('InitializeProcThreadAttributeList',[c.c_void_p,w.DWORD,w.DWORD,c.POINTER(c.c_size_t)],w.BOOL),
          ('UpdateProcThreadAttribute',[c.c_void_p,w.DWORD,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p,c.c_void_p],w.BOOL),
          ('DeleteProcThreadAttributeList',[c.c_void_p],None),
          ('CreateProcessW',[w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,w.DWORD,c.c_void_p,w.LPCWSTR,c.POINTER(library.StartupEx),c.POINTER(library.ProcessInfo)],w.BOOL),
          ('GetProcessTimes',[w.HANDLE,c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME)],w.BOOL),
          ('QueryFullProcessImageNameW',[w.HANDLE,w.DWORD,w.LPWSTR,c.POINTER(w.DWORD)],w.BOOL),
          ('IsProcessInJob',[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)],w.BOOL),
          ('WaitForSingleObject',[w.HANDLE,w.DWORD],w.DWORD),
          ('TerminateProcess',[w.HANDLE,w.UINT],w.BOOL),
          ('ResumeThread',[w.HANDLE],w.DWORD),('CloseHandle',[w.HANDLE],w.BOOL),
          ('GetCurrentProcess',[],w.HANDLE),
          ('DuplicateHandle',[w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD],w.BOOL),
          ('GetProcessMemoryInfo',[w.HANDLE,c.c_void_p,w.DWORD],w.BOOL),
          ('QueryInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p],w.BOOL),
        ]
        for name,args,result in signatures:
            fn=getattr(kernel,name);fn.argtypes=args;fn.restype=result
    def create(self,cwd,temp,inherited):
        need(not self.attempted and self.info is None,'one U creation attempt only')
        need(self.outer and self.inner and self.outer!=self.inner,'distinct held outer/inner Jobs')
        need(len(inherited)==3 and all(type(h) is int and h>0 for h in inherited),'exact inherited three handles')
        size=c.c_size_t();attribute=None
        sized=self.k.InitializeProcThreadAttributeList(None,2,0,c.byref(size))
        need(not sized and self.last_error()==122 and 0<size.value<=65536,'two-attribute sizing')
        storage=c.create_string_buffer(size.value)
        try:
            need(self.k.InitializeProcThreadAttributeList(storage,2,0,c.byref(size)),'two-attribute initialization')
            attribute=c.cast(storage,c.c_void_p)
            jobs=(w.HANDLE*2)(self.outer,self.inner)
            handles=(w.HANDLE*3)(*inherited)
            need(self.k.UpdateProcThreadAttribute(attribute,0,JOB_LIST,jobs,c.sizeof(jobs),None,None),'atomic birth Job list refused')
            need(self.k.UpdateProcThreadAttribute(attribute,0,HANDLE_LIST,handles,c.sizeof(handles),None,None),'inherited handle list refused')
            startup=self.lib.StartupEx();startup.StartupInfo.cb=c.sizeof(startup)
            startup.StartupInfo.dwFlags=0x100
            startup.StartupInfo.hStdInput=inherited[0];startup.StartupInfo.hStdOutput=inherited[1];startup.StartupInfo.hStdError=inherited[2]
            startup.lpAttributeList=attribute
            command=c.create_unicode_buffer('"'+UV+'" --version')
            env=c.create_unicode_buffer('\0'.join(sorted(('SystemRoot=C:/Windows','WINDIR=C:/Windows','PATH=C:/Windows/System32','TEMP='+temp,'TMP='+temp)))+'\0\0')
            info=self.lib.ProcessInfo()
            self.attempted=True  # BEFORE call: no-return/false is not no-attempt.
            ok=self.k.CreateProcessW(UV,command,None,None,True,4|0x400|0x80000|0x08000000,env,cwd,c.byref(startup),c.byref(info))
            # Own every returned handle before any later fallible check.
            if info.hProcess:self.info=info;self.partial_handles['process']=info.hProcess
            if info.hThread:self.partial_handles['thread']=info.hThread
            self.created_at=self.clock()
            need(ok and info.hProcess and info.hThread and info.dwProcessId>0,'creation failed/partial/ambiguous; full exposure retained')
            need(self.in_job(self.outer) and self.in_job(self.inner),'birth membership witness unavailable')
            return info
        finally:
            if attribute:
                try:self.k.DeleteProcThreadAttributeList(attribute)
                except BaseException as exc:self.errors.append('attribute deletion: '+str(exc))
    def in_job(self,job):
        need(self.info is not None,'no exact created process handle')
        value=w.BOOL();need(self.k.IsProcessInJob(self.info.hProcess,job,c.byref(value)),'held membership query unavailable')
        return bool(value.value)
    def identity(self):
        need(self.info is not None,'created identity unavailable')
        times=[w.FILETIME() for _ in range(4)]
        need(self.k.GetProcessTimes(self.info.hProcess,*(c.byref(t) for t in times)),'held creation-time unavailable')
        image=c.create_unicode_buffer(32768);size=w.DWORD(len(image))
        need(self.k.QueryFullProcessImageNameW(self.info.hProcess,0,image,c.byref(size)),'held image unavailable')
        need(image.value.replace('\\','/').casefold()==UV.casefold(),'held image drift')
        self.identity_value={'pid':int(self.info.dwProcessId),
          'creation_time':str((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime),
          'image':image.value.replace('\\','/'),
          'command_hash':hashlib.sha256(canonical([UV,'--version'])).hexdigest(),
          'command_hash_kind':'REQUESTED_ARGV_NOT_OBSERVED_OS_COMMAND_LINE'}
        return dict(self.identity_value)
    def duplicate_for_c(self,caller_handle):
        need(self.identity_value is not None,'no identity before copy')
        copies={}
        # Record each returned remote copy before the next duplicate can fail.
        # This is bookkeeping, not the missing C transport/lifetime bootstrap.
        self.acknowledged_copies={'caller_handle':caller_handle,'copies':copies,'ack_digest':None}
        for name in ('process','thread'):
            out=w.HANDLE()
            need(self.k.DuplicateHandle(self.k.GetCurrentProcess(),self.partial_handles[name],caller_handle,c.byref(out),0,False,2),'duplicate exact '+name)
            copies[name]=out.value
        # Originals remain owned even if C never receives/acknowledges copies.
        return copies
    def acknowledge(self,frame_digest,copies):
        need(self.acknowledged_copies.get('copies')==copies,'copy acknowledgement drift')
        need(isinstance(frame_digest,str) and re.fullmatch(r'[0-9a-f]{64}',frame_digest),'ack frame identity')
        self.acknowledged_copies['ack_digest']=frame_digest
    def resume(self):
        need(self.acknowledged_copies.get('ack_digest'),'C copy acknowledgement absent')
        need(self.in_job(self.outer) and self.in_job(self.inner),'membership drift before resume')
        need(self.k.ResumeThread(self.partial_handles['thread'])==1,'exact primary suspend count')
    def retained_cleanup(self):
        errors=[];h=self.partial_handles.get('process');signal=False
        if h:
            try:
                state=self.k.WaitForSingleObject(h,0)
                if state==258:
                    need(self.k.TerminateProcess(h,2),'owned termination failed')
                    state=self.k.WaitForSingleObject(h,0)
                need(state==0,'owned exit signal unproved');signal=True
            except BaseException as exc:errors.append(str(exc))
        # No Job/process closure when live or ambiguous. O-loss keeper is absent.
        if signal:
            try:
                pids=self.lib.Pids()
                need(self.k.QueryInformationJobObject(self.inner,3,c.byref(pids),c.sizeof(pids),None),'inner Job empty witness unavailable')
                need(pids.assigned==pids.returned and pids.returned==0,'inner Job not proved empty')
            except BaseException as exc:
                signal=False;errors.append(str(exc))
        if signal:
            for name in ('thread','process'):
                value=self.partial_handles.get(name)
                if value:
                    try:
                        need(self.k.CloseHandle(value),'original handle close failed')
                        self.partial_handles[name]=None
                    except BaseException as exc:errors.append(str(exc))
        return {'signal_proved':signal,'retained':dict(self.partial_handles),'errors':errors,
                'owner':self,'O_loss_recovery_implemented':False}

class TruthfulStoreDelegate:
    """Exact existing operations calls; refusals are exceptions, not fake writes."""
    def __init__(self,store,birth,gate):
        self.store=store;self.birth=birth;self.gate=gate;self.journal=[]
    def __getattr__(self,name):return getattr(self.store,name)
    def mark_unknown(self,key,reason):
        row=self.store.get_job(key)
        if not self.birth.attempted and self.birth.info is None and row['state']=='RESERVED' and row['worker'] is None:
            self.journal.append({'event':'UNKNOWN_REFUSED_PRISTINE_NO_ATTEMPT','stored_state':'RESERVED'})
            raise Refused('no creation attempt: keep genuine RESERVED for truthful no-dispatch reconciliation')
        return self.store.mark_unknown(key,reason)
    def check_reservation(self):
        g=self.gate;self.store._require_lease(g['owner'],g['fence'])
        row=self.store.get_job(g['key'])
        need(row['state']=='RESERVED' and row['request_hash']==g['request_hash'] and row['epoch']==g['epoch'] and
             row['owner']==g['owner'] and row['fence']==g['fence'] and row['reserved_seconds']==10,'exact canonical reservation required')
        return row
    def create_and_record(self,cwd,temp,inherited):
        self.check_reservation();info=self.birth.create(cwd,temp,inherited)
        identity=self.birth.identity()
        self.record_same_created_identity(identity)
        return info,identity
    def record_same_created_identity(self,identity=None):
        g=self.gate;identity=identity or self.birth.identity()
        need(identity==self.birth.identity_value,'no fabricated future worker')
        row=self.store.get_job(g['key'])
        if row['state']=='RUNNING':
            need(row['worker']==identity,'committed worker drift');return row
        need(row['state']=='RESERVED','created worker cannot be reset from UNKNOWN')
        try:self.store.mark_running(g['key'],g['owner'],g['fence'],identity)
        except BaseException as exc:
            self.journal.append({'event':'REGISTRATION_RESPONSE_UNKNOWN','error':type(exc).__name__})
            row=self.store.get_job(g['key'])
            if row['state']=='RUNNING' and row['worker']==identity:return row
            raise
        row=self.store.get_job(g['key'])
        need(row['state']=='RUNNING' and row['worker']==identity,'durable actual worker binding not proved')
        return row
    def resume(self):
        self.store._require_lease(self.gate['owner'],self.gate['fence'])
        row=self.store.get_job(self.gate['key'])
        need(row['state']=='RUNNING' and row['worker']==self.birth.identity_value,'no return/resume before recorded identity')
        return self.birth.resume()
    def reconcile_pristine_failure(self):
        need(not self.birth.attempted and self.birth.info is None,'attempted/created is not pristine no-dispatch')
        row=self.store.get_job(self.gate['key']);need(row['state']=='RESERVED' and row['worker'] is None,'not pristine RESERVED')
        g=self.gate
        proof={'observer':g['owner'],'request_hash':g['request_hash'],'owner':g['owner'],'fence':g['fence'],
               'never_dispatched':True,'launch_attempted':False,'elapsed_native_seconds':0}
        ref=self.store.write_receipt('native_outcome_observation',g['key'],proof)
        return self.store.reconcile(g['key'],'FAILED',0,{'native_outcome_proof_ref':ref,'CPU_cost_not_zeroed':'separate adopted phase evidence required'},owner=g['owner'],fence=g['fence'])

class FixedFrame:
    """Concrete bounded frame bytes. No OS kernel-RPC bootstrap is provided."""
    ALLOWED={'IDENTITY','ACK_COPIES','RESUME','CALLER_LEAF'}
    def __init__(self,nonce):
        need(isinstance(nonce,str) and len(nonce)==64,'fixed channel nonce');self.nonce=nonce;self.sequence=0
    def encode(self,operation,payload):
        need(operation in self.ALLOWED,'out-of-scope RPC operation')
        raw=canonical({'nonce':self.nonce,'sequence':self.sequence,'operation':operation,'payload':payload})
        need(len(raw)<=8192,'RPC frame cap');return struct.pack('<I',len(raw))+raw
    def decode(self,frame):
        need(len(frame)>=4,'truncated frame');size=struct.unpack('<I',frame[:4])[0]
        need(size<=8192 and len(frame)==size+4,'frame count/trailing bytes')
        value=json.loads(frame[4:]);need(canonical(value)==frame[4:],'noncanonical frame')
        need(set(value)=={'nonce','sequence','operation','payload'} and value['nonce']==self.nonce and
             value['sequence']==self.sequence and value['operation'] in self.ALLOWED,'frame owner/sequence/op drift')
        self.sequence+=1
        return value

def combined_peak_witness(kernel,library,held_graph):
    need(set(held_graph)=={'O','C','U'} and all(held_graph.values()),'complete original O/C/U graph required')
    total=0;samples={}
    for role,handle in held_graph.items():
        memory=library.Memory();memory.cb=c.sizeof(memory)
        need(kernel.GetProcessMemoryInfo(handle,c.byref(memory),c.sizeof(memory)),'graph peak witness unavailable')
        current=int(memory.WorkingSetSize);peak=int(memory.PeakWorkingSetSize)
        need(0<=current<=peak,'invalid graph RSS/peak')
        samples[role]={'current':current,'peak':peak};total+=peak
    need(total<=2*GIB,'combined graph peak RSS cap')
    return {'conservative_sum_of_exact_process_peaks':total,'samples':samples,'actual_graph_birth_enforcement_qualified':False}

def publish_completed_leaf(store,kind,key,payload,clock=time.monotonic):
    """Actual durable write + verify, with completion clock AFTER both return."""
    started=clock();ref=store.write_receipt(kind,key,payload)
    receipt=store.verify_receipt(ref);completed=clock()
    return {'ref':ref,'receipt':receipt,'started':started,'completed_after_publication':completed}

def witness_caller_completion(birth,leaf,stream_EOF,graph_witness,start,clock=time.monotonic):
    need(leaf['receipt']['kind']=='tool_caller_leaf','wrong completed caller leaf')
    need(birth.info and birth.k.WaitForSingleObject(birth.info.hProcess,0)==0,'owned exit missing')
    need(stream_EOF=={'stdout':True,'stderr':True},'both EOF witnesses required')
    need(set(graph_witness['samples'])=={'O','C','U'},'graph witness missing')
    completed=clock();need(completed>=leaf['completed_after_publication'],'completion precedes publication')
    need(0<=completed-start<=10,'caller complete10s overrun')
    # This is a component witness of held U/leaf. Exact C exit/EOF and outer
    # producer are NOT implemented, so it is not actual TC or whole ACCEPTED.
    return {'component_elapsed':completed-start,'actual_TC_qualified':False}

def assess_observer_interval(start,cpu_started,clock=time.monotonic,cpu_clock=time.process_time):
    wall=clock()-start;cpu=cpu_clock()-cpu_started
    need(math.isfinite(wall) and math.isfinite(cpu) and wall>=0 and cpu>=0,'unknown observer costs')
    return {'wall_seconds':wall,'CPU_seconds':cpu,'phase_within_2s':wall<=2 and cpu<=2,
            'whole_status':'NOT_READY','external_OE_producer_implemented':False}

def actual_dispatch(*args,**kwargs):
    raise Refused('NOT_READY: '+ '; '.join(NOT_READY_GAPS))

if __name__=='__main__':
    raise SystemExit('NOT_READY round4: component candidate has no authorized actual dispatch; missing C-RPC/O-loss/OE bootstrap')
