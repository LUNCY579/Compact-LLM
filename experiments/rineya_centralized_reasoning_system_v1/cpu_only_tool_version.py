"""Candidate3 library mechanics for one UV metadata observation; NOT_READY.

No quarantined loader is imported. WindowsJob structure lineage: opallo
supervisor d31370de; suspended assignment/exit lineage: CRS native84531289;
accounting is unchanged operations ba27ec8a. Runtime/source/role/admission gates
are supplied only by a separately reviewed caller; this candidate is NOT_READY.
"""
from __future__ import annotations
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

PACKAGE=Path(__file__).resolve().parent
GIB=2**30
UV_PATH='C:/Users/clclc/.local/bin/uv.exe'
UV_SHA='5a7ec85884c2ccb1be560cb8fac3eb890df1adf49bfcc070a270ba70401bdd68'
OPS_SHA='ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30'
FLAGS=0x2000|0x8|0x200  # kill-on-close, active-process limit, job committed memory
VERSION_LINE=re.compile(rb'uv ([0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+_-]*))(?: \(([A-Za-z0-9 .+_-]{1,160})\))?\r?\n?')

class Refused(RuntimeError):pass

def require(ok,why):
    if not ok:raise Refused(why)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

class Basic(c.Structure):
    _fields_=[('PerProcessUserTimeLimit',c.c_longlong),('PerJobUserTimeLimit',c.c_longlong),
              ('LimitFlags',w.DWORD),('MinimumWorkingSetSize',c.c_size_t),('MaximumWorkingSetSize',c.c_size_t),
              ('ActiveProcessLimit',w.DWORD),('Affinity',c.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
class IO(c.Structure):
    _fields_=[(n,c.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                        'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class Extended(c.Structure):
    _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',c.c_size_t),
              ('JobMemoryLimit',c.c_size_t),('PeakProcessMemoryUsed',c.c_size_t),('PeakJobMemoryUsed',c.c_size_t)]
class Pids(c.Structure):
    _fields_=[('assigned',w.DWORD),('returned',w.DWORD),('pids',c.c_size_t*2)]

class OwnedJob:
    """Actual typed Win32 call interface, with injected kernel for fixtures."""
    def __init__(self,kernel,guard=None):
        self.kernel=kernel;self.handle=None
        for name,args,result in [
            ('CreateJobObjectW',[c.c_void_p,w.LPCWSTR],w.HANDLE),
            ('SetInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD],w.BOOL),
            ('QueryInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p],w.BOOL),
            ('AssignProcessToJobObject',[w.HANDLE,w.HANDLE],w.BOOL),
            ('TerminateJobObject',[w.HANDLE,w.UINT],w.BOOL),('CloseHandle',[w.HANDLE],w.BOOL)]:
            fn=getattr(kernel,name);fn.argtypes=args;fn.restype=result
        self.handle=kernel.CreateJobObjectW(None,None);require(self.handle,'create owned Job')
        if guard:guard.own_job(self)
        limits=Extended();limits.BasicLimitInformation.LimitFlags=FLAGS
        limits.BasicLimitInformation.ActiveProcessLimit=1;limits.JobMemoryLimit=2*GIB
        try:
            require(kernel.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits)),'set Job limits')
            self.verify_limits()
        except BaseException:
            if guard:require(guard.quiesce(),'partial Job timer not quiescent; handle retained')
            self.close();raise
    def verify_limits(self):
        limits=Extended()
        require(self.kernel.QueryInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits),None),'query Job limits')
        b=limits.BasicLimitInformation
        require(b.LimitFlags==FLAGS and b.ActiveProcessLimit==1 and limits.JobMemoryLimit==2*GIB,'Job flag/process/commit drift')
        return {'limit_flags':int(b.LimitFlags),'ActiveProcessLimit':1,'JobMemoryLimit':int(limits.JobMemoryLimit),'breakaway':False}
    def assign(self,process):
        require(self.kernel.AssignProcessToJobObject(self.handle,int(process.hProcess)),'suspended Job assignment')
        require(self.pids()==[int(process.dwProcessId)],'exact one suspended PID in Job')
    def pids(self):
        p=Pids();require(self.kernel.QueryInformationJobObject(self.handle,3,c.byref(p),c.sizeof(p),None),'complete Job PID query')
        require(p.assigned==p.returned and p.returned<=1,'Job child/incomplete list')
        return [int(p.pids[i]) for i in range(p.returned)]
    def kill(self):
        require(self.kernel.TerminateJobObject(self.handle,2),'owned Job termination')
    def close(self):
        if self.handle:
            handle=self.handle
            require(self.kernel.CloseHandle(handle),'owned Job handle close')
            self.handle=None


class Deadline:
    """Independent owned-worker kill timer. Cannot abort blocked SQLite/OS calls.

    A separately qualified external caller must bound/observe the whole
    controller including durable finalization. No terminal success is published
    here; actual late/unknown costs survive for that external observer.
    """
    def __init__(self, clock=time.monotonic, timer_factory=threading.Timer):
        self.clock=clock;self.started=clock();self.until=self.started+10
        self.expired=threading.Event();self.port=None;self.job=None;self.errors=[]
        self.timer=timer_factory(10,self.trip);self.timer.daemon=True;self.timer.start()
    def own_job(self,job):self.job=job
    def own_port(self,port):self.port=port
    def trip(self):
        self.expired.set()
        if self.job and self.job.handle:
            try:self.job.kill()
            except BaseException as e:self.errors.append('timer Job kill: '+str(e))
        if self.port:
            try:self.port.terminate_created()
            except BaseException as e:self.errors.append('timer created-handle kill: '+str(e))
    def check(self):
        require(not self.expired.is_set() and self.clock()<self.until,'combined10s deadline')
    def call(self,fn,*args):
        self.check();value=fn(*args);self.check();return value
    def cancel(self):
        self.timer.cancel()
        # No assertion that a blocked callback or mainline returned within10s.
    def quiesce(self):
        self.timer.cancel()
        if self.timer.is_alive():self.timer.join(timeout=max(0,self.until-self.clock()))
        return not self.timer.is_alive()

class Security(c.Structure):
    _fields_=[('nLength',w.DWORD),('lpSecurityDescriptor',c.c_void_p),('bInheritHandle',w.BOOL)]
class Startup(c.Structure):
    _fields_=[('cb',w.DWORD),('lpReserved',w.LPWSTR),('lpDesktop',w.LPWSTR),('lpTitle',w.LPWSTR),
      ('dwX',w.DWORD),('dwY',w.DWORD),('dwXSize',w.DWORD),('dwYSize',w.DWORD),
      ('dwXCountChars',w.DWORD),('dwYCountChars',w.DWORD),('dwFillAttribute',w.DWORD),
      ('dwFlags',w.DWORD),('wShowWindow',w.WORD),('cbReserved2',w.WORD),
      ('lpReserved2',c.POINTER(c.c_ubyte)),('hStdInput',w.HANDLE),('hStdOutput',w.HANDLE),('hStdError',w.HANDLE)]
class StartupEx(c.Structure):
    _fields_=[('StartupInfo',Startup),('lpAttributeList',c.c_void_p)]
class ProcessInfo(c.Structure):
    _fields_=[('hProcess',w.HANDLE),('hThread',w.HANDLE),('dwProcessId',w.DWORD),('dwThreadId',w.DWORD)]
class Memory(c.Structure):
    _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(n,c.c_size_t) for n in
      ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
       'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

class WindowsPort:
    """Typed adapter. Injected kernel/psapi exercise these exact calls as data."""
    def __init__(self,guard,kernel=None,psapi=None,last_error=None):
        self.guard=guard;self.job=None;self.process=None;self.handles={}
        self.cleanup_exit_proved=False;self.cleanup_exit_at=None
        self.cleanup_job_empty=False;self.cleanup_timer_quiescent=False
        self.created_termination_attempted=False
        self.attribute=None;self.attribute_buffer=None;self.inherit_array=None
        self.kernel=kernel;self.psapi=psapi;self.last_error=last_error or c.get_last_error
        guard.own_port(self)
        try:
            if kernel is None:
                require(os.name=='nt','Windows only')
                self.kernel=c.WinDLL('C:/Windows/System32/kernel32.dll',use_last_error=True)
            if psapi is None:self.psapi=c.WinDLL('C:/Windows/System32/psapi.dll',use_last_error=True)
            specs=[
              ('GetProcessTimes',[w.HANDLE,c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME)],w.BOOL),
              ('QueryFullProcessImageNameW',[w.HANDLE,w.DWORD,w.LPWSTR,c.POINTER(w.DWORD)],w.BOOL),
              ('PeekNamedPipe',[w.HANDLE,c.c_void_p,w.DWORD,c.c_void_p,c.POINTER(w.DWORD),c.c_void_p],w.BOOL),
              ('ReadFile',[w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p],w.BOOL),
              ('CreatePipe',[c.POINTER(w.HANDLE),c.POINTER(w.HANDLE),c.POINTER(Security),w.DWORD],w.BOOL),
              ('SetHandleInformation',[w.HANDLE,w.DWORD,w.DWORD],w.BOOL),
              ('InitializeProcThreadAttributeList',[c.c_void_p,w.DWORD,w.DWORD,c.POINTER(c.c_size_t)],w.BOOL),
              ('UpdateProcThreadAttribute',[c.c_void_p,w.DWORD,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p,c.c_void_p],w.BOOL),
              ('DeleteProcThreadAttributeList',[c.c_void_p],None),
              ('CreateProcessW',[w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,w.DWORD,c.c_void_p,w.LPCWSTR,c.POINTER(StartupEx),c.POINTER(ProcessInfo)],w.BOOL),
              ('ResumeThread',[w.HANDLE],w.DWORD),('WaitForSingleObject',[w.HANDLE,w.DWORD],w.DWORD),
              ('GetExitCodeProcess',[w.HANDLE,c.POINTER(w.DWORD)],w.BOOL),
              ('TerminateProcess',[w.HANDLE,w.UINT],w.BOOL),('CloseHandle',[w.HANDLE],w.BOOL)]
            for name,args,restype in specs:
                fn=getattr(self.kernel,name);fn.argtypes=args;fn.restype=restype
            fn=self.psapi.GetProcessMemoryInfo;fn.argtypes=[w.HANDLE,c.c_void_p,w.DWORD];fn.restype=w.BOOL
            self.job=OwnedJob(self.kernel,guard)
        except BaseException:
            # Guard retains ownership even when constructor never returns.
            self.constructor_cleanup_errors=self.cleanup()
            guard.errors.extend(self.constructor_cleanup_errors)
            raise
    def close_handle(self,key):
        value=self.handles.get(key)
        if value:
            require(self.kernel.CloseHandle(value),'close '+key)
            self.handles[key]=None
    def spawn_suspended(self,cwd,env):
        sa=Security(c.sizeof(Security),None,True)
        for name in ('stdin','stdout','stderr'):
            r=w.HANDLE();v=w.HANDLE()
            self.guard.check()
            ok=self.kernel.CreatePipe(c.byref(r),c.byref(v),c.byref(sa),0)
            # Preserve even partial returned handles before deadline/refusal.
            if r.value:self.handles[name+'_read']=r.value
            if v.value:self.handles[name+'_write']=v.value
            require(ok and r.value and v.value,'create '+name+' pipe')
            self.guard.check()
        for name in ('stdin_write','stdout_read','stderr_read'):
            require(self.guard.call(self.kernel.SetHandleInformation,self.handles[name],1,0),'noninheritable '+name)
        self.close_handle('stdin_write')  # No input; exact child receives EOF.
        size=c.c_size_t()
        sized=self.guard.call(self.kernel.InitializeProcThreadAttributeList,None,1,0,c.byref(size))
        require(not sized and self.last_error()==122 and 0<size.value<=65536,'attribute sizing')
        self.attribute_buffer=c.create_string_buffer(size.value)
        self.guard.check()
        initialized=self.kernel.InitializeProcThreadAttributeList(self.attribute_buffer,1,0,c.byref(size))
        if initialized:self.attribute=c.cast(self.attribute_buffer,c.c_void_p)
        require(initialized,'attribute initialization');self.guard.check()
        self.inherit_array=(w.HANDLE*3)(self.handles['stdin_read'],self.handles['stdout_write'],self.handles['stderr_write'])
        require(self.guard.call(self.kernel.UpdateProcThreadAttribute,self.attribute,0,0x20002,self.inherit_array,c.sizeof(self.inherit_array),None,None),'exact inherited handle list')
        startup=StartupEx();startup.StartupInfo.cb=c.sizeof(startup)
        startup.StartupInfo.dwFlags=0x100
        startup.StartupInfo.hStdInput=self.handles['stdin_read'];startup.StartupInfo.hStdOutput=self.handles['stdout_write'];startup.StartupInfo.hStdError=self.handles['stderr_write']
        startup.lpAttributeList=self.attribute
        info=ProcessInfo();command=c.create_unicode_buffer('"'+UV_PATH+'" --version')
        envbuf=c.create_unicode_buffer('\0'.join(k+'='+v for k,v in sorted(env.items()))+'\0\0')
        # Register ownership immediately after CreateProcess returns, BEFORE a
        # post-call deadline/identity/pipe-close check can fail.
        self.guard.check()
        ok=self.kernel.CreateProcessW(UV_PATH,command,None,None,True,4|0x400|0x80000|0x08000000,
                                      envbuf,cwd,c.byref(startup),c.byref(info))
        if info.hProcess:
            self.process=info;self.handles['process']=info.hProcess
        if info.hThread:self.handles['primary_thread']=info.hThread
        require(ok and self.process and info.hThread and info.dwProcessId>0,'suspended CreateProcess')
        self.guard.check()
        for key in ('stdin_read','stdout_write','stderr_write'):self.close_handle(key)
        return info
    def identity(self,p):
        ft=[w.FILETIME() for _ in range(4)]
        require(self.guard.call(self.kernel.GetProcessTimes,p.hProcess,*(c.byref(x) for x in ft)),'creation identity')
        buf=c.create_unicode_buffer(32768);size=w.DWORD(len(buf))
        require(self.guard.call(self.kernel.QueryFullProcessImageNameW,p.hProcess,0,buf,c.byref(size)),'image identity')
        return {'pid':int(p.dwProcessId),'creation_time':str((ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime),
                'image':buf.value.replace('\\','/'),
                'command_hash':hashlib.sha256(canonical([UV_PATH,'--version'])).hexdigest(),
                'command_hash_kind':'REQUESTED_ARGV_NOT_OBSERVED_OS_COMMAND_LINE'}
    def resume(self,p):
        require(self.guard.call(self.kernel.ResumeThread,p.hThread)==1,'exact created primary thread suspend count')
    def exit_code(self,p):
        status=self.guard.call(self.kernel.WaitForSingleObject,p.hProcess,0)
        if status==258:return None
        require(status==0,'owned process wait unknown')
        code=w.DWORD();require(self.guard.call(self.kernel.GetExitCodeProcess,p.hProcess,c.byref(code)),'exit code unknown')
        require(code.value!=259,'signaled process still active')
        return int(code.value)
    def output(self,remaining):
        chunks=[];eof={}
        for name in ('stdout','stderr'):
            h=self.handles[name+'_read'];available=w.DWORD()
            ok=self.guard.call(self.kernel.PeekNamedPipe,h,None,0,None,c.byref(available),None)
            if not ok:
                require(self.last_error()==109,'pipe read error/unknown EOF')
                eof[name]=True;continue
            eof[name]=False
            if available.value:
                # One combined overflow sentinel (maximum captured8193), never
                # one sentinel independently per stream.
                count=min(int(available.value),remaining+1)
                require(count>0,'combined capture overflow')
                b=c.create_string_buffer(count);read=w.DWORD()
                require(self.guard.call(self.kernel.ReadFile,h,b,count,c.byref(read),None),'bounded pipe read')
                require(0<read.value<=count,'pipe byte count')
                data=b.raw[:read.value];chunks.append((name,data));remaining-=len(data)
                if remaining<0:break
        return {'chunks':chunks,'EOF':eof}
    def memory(self,p):
        value=Memory();value.cb=c.sizeof(value)
        require(self.guard.call(self.psapi.GetProcessMemoryInfo,p.hProcess,c.byref(value),c.sizeof(value)),'RSS/peak witness unavailable')
        return {'rss_bytes':int(value.WorkingSetSize),'peak_rss_bytes':int(value.PeakWorkingSetSize),
                'source':'GetProcessMemoryInfo exact created process handle','measured_at':self.guard.clock()}
    def terminate_created(self):
        # Exact owned handle, independent of first identity or a PID lookup.
        h=self.handles.get('process')
        if h:
            status=self.kernel.WaitForSingleObject(h,0)
            if status==258:
                require(not self.created_termination_attempted,'prior exact-handle termination unresolved; ownership retained without retry')
                self.created_termination_attempted=True
                require(self.kernel.TerminateProcess(h,2),'terminate exact created handle')
                # A successful asynchronous termination request is not exit.
                # One nonblocking signal witness; no cleanup retry/poll loop.
                status=self.kernel.WaitForSingleObject(h,0)
            require(status==0,'created process exit signal unproved during cleanup')
        return True
    def recovery_state(self):
        # Snapshot only. The reachable guard/port object, not these numbers,
        # retains the original exact handles and their cleanup implementation.
        job=self.job or self.guard.job
        return {'process_handle':self.handles.get('process'),
                'Job_handle':job.handle if job else None,
                'other_handles':{k:v for k,v in self.handles.items() if v and k!='process'},
                'attribute_retained':bool(self.attribute),
                'exact_process_signal_proved':self.cleanup_exit_proved,
                'owned_Job_empty_proved':self.cleanup_job_empty,
                'timer_quiescent_at_cleanup':self.cleanup_timer_quiescent,
                'created_termination_attempted':self.created_termination_attempted,
                'in_memory_only_not_durable_or_transferred':True}
    def cleanup(self):
        errors=[]
        def step(name,fn):
            try:return fn()
            except BaseException as e:errors.append(name+': '+type(e).__name__+': '+str(e))
        # Quiesce before touching callback targets. If it still runs, preserve
        # those targets while independently closing unrelated safe resources.
        self.cleanup_timer_quiescent=self.guard.quiesce()
        job=self.job or self.guard.job
        self.cleanup_exit_proved=False;self.cleanup_job_empty=False
        if self.cleanup_timer_quiescent:
            if job and job.handle:step('Job terminate',job.kill)
            self.cleanup_exit_proved=step('created handle terminate/signal',self.terminate_created) is True
            if self.cleanup_exit_proved:self.cleanup_exit_at=self.guard.clock()
            if job and job.handle:
                ids=step('owned Job empty witness',job.pids)
                self.cleanup_job_empty=ids==[]
            else:self.cleanup_job_empty=True
        else:errors.append('independent timer callback not quiescent; exact Job/process targets retained')
        if self.attribute:
            def delete():
                self.kernel.DeleteProcThreadAttributeList(self.attribute);self.attribute=None
            step('attribute delete',delete)
        for name in ('stdin_read','stdin_write','stdout_read','stdout_write','stderr_read','stderr_write','primary_thread'):
            step(name,lambda name=name:self.close_handle(name))
        # Never discard the last stable process/Job recovery witness on failed
        # termination, unknown wait, missing signal, or uncertain membership.
        if self.cleanup_timer_quiescent and self.cleanup_exit_proved and self.cleanup_job_empty:
            if job and job.handle:step('Job close',job.close)
            step('process close',lambda:self.close_handle('process'))
        elif self.handles.get('process') or (job and job.handle):
            errors.append('exact process/Job ownership retained pending verified exit and empty Job; no handoff implemented')
        return errors


def validate_admission(gate,clock):
    require(gate.get('separately_registered') is True and gate.get('source_roles_commit_accepted') is True,'launch registration/reviews absent')
    require(gate.get('loader_documentary_accepted') is True and gate.get('trusted_metadata_boundary_accepted') is True,'trusted loader/tool boundary unknown')
    for k in ('key','owner','epoch','request_hash','source_commit','contract_sha256','runtime_pin','admission_pin'):
        require(isinstance(gate.get(k),str) and gate[k],'missing '+k)
    require(type(gate.get('fence')) is int and gate['fence']>0,'fence identity')
    require(gate.get('telemetry_age_seconds',math.inf)<=5 and gate.get('preworker_available_bytes',0)>=3.3125*GIB,'fresh host admission')
    require(gate.get('disk_free_bytes',0)>=4*GIB and gate.get('host_reserve_bytes',0)>=1.25*GIB,'disk/reserve admission')
    require(gate.get('gpu_status')=='UNMEASURED_NOT_QUALIFIED','no GPU-zero inference')
    require(gate.get('argv')==['--version'] and gate.get('tool_path')==UV_PATH and gate.get('tool_sha256')==UV_SHA,'tool/argv pins')
    require(gate.get('canonical_operations_sha256')==OPS_SHA,'accounting source drift')


def observe_version(gate,store,port_factory,telemetry,*,clock=time.monotonic,
                    sleep=time.sleep,guard_factory=None):
    """Unqualified library endpoint, not a dispatcher or a terminal success.

    The canonical reservation remains10 while an external immutable observer
    binds actual caller completion, exit, source/runtime/admission and outcome.
    Synthetic inputs here exercise mechanics; they are not those attestations.
    """
    guard=guard_factory() if guard_factory else Deadline(clock)
    p=identity=None;port=None;reserved=running=False;assigned=False
    native_started=native_end=None;exit_proved=False;code=None
    out=bytearray();err=bytearray();eof={};resources=[];reason=None
    result={'status':'NOT_READY_LIBRARY_RESULT','source_candidate':3,
            'source_or_launch_qualified':False,'external_complete_observer_required':True}
    try:
        guard.call(validate_admission,gate,clock)
        require(file_sha(PACKAGE/'operations.py')==OPS_SHA,'canonical ops hash changed');guard.check()
        require(gate.get('physical_paths_verified') is True and gate.get('empty_scoped_paths_verified') is True,
                'external fresh canonical/reparse/path proof absent')
        cwd=gate['cwd'];temp=gate['temp']
        require(Path(cwd).is_relative_to(PACKAGE/'.venv/tool-version-007') and
                Path(temp).is_relative_to(PACKAGE/'.venv/tool-version-007'),'scoped paths')
        env={'SystemRoot':'C:/Windows','WINDIR':'C:/Windows','PATH':'C:/Windows/System32','TEMP':temp,'TMP':temp}
        guard.call(store._require_lease,gate['owner'],gate['fence'])
        row=guard.call(store.get_job,gate['key'])
        require(row['request_hash']==gate['request_hash'] and row['epoch']==gate['epoch'] and row['state']=='READY',
                'pre-registered exact request only')
        # Canonical reserve may commit then block. Record admission intent first,
        # and conservatively leave any unknown reservation for reconciliation.
        guard.check()
        admitted=store.reserve(gate['key'],gate['owner'],gate['fence'],10)
        reserved=True;guard.check()
        require(admitted['state']=='RESERVED','reservation witness')
        port=guard.call(port_factory,guard)
        native_started=clock()
        p=port.spawn_suspended(cwd,env)
        identity=port.identity(p)
        require(identity['image'].casefold()==UV_PATH.casefold(),'launched image drift')
        guard.call(port.job.assign,p);assigned=True
        witness=guard.call(port.job.verify_limits)
        require(guard.call(port.job.pids)==[int(p.dwProcessId)],'zero children before resume')
        marked=guard.call(store.mark_running,gate['key'],gate['owner'],gate['fence'],identity)
        running=True
        require(marked['state']=='RUNNING' and marked['worker']==identity,'canonical RUNNING binding')
        guard.call(store._require_lease,gate['owner'],gate['fence'])
        port.resume(p)
        while True:
            guard.call(store._require_lease,gate['owner'],gate['fence'])
            require(port.identity(p)==identity,'PID/ctime/image drift')
            ids=guard.call(port.job.pids);require(ids in ([int(p.dwProcessId)],[]),'unexpected child/Job drift')
            limits=guard.call(port.job.verify_limits)
            sample=guard.call(telemetry)
            require(type(sample['age_seconds']) in (int,float) and math.isfinite(sample['age_seconds']) and
                    0<=sample['age_seconds']<=5 and sample['host_available_bytes']>=1.25*GIB and
                    sample['disk_free_bytes']>=4*GIB,'stale/host/disk telemetry')
            # Peak is requested even when the one process has already exited.
            memory=port.memory(p)
            require(memory['rss_bytes']<=2*GIB and memory['peak_rss_bytes']<=2*GIB,
                    'current/peak whole one-process RSS cap')
            resources.append({'at':clock(),'memory':memory,'telemetry':sample,'Job':limits})
            require(len(resources)<=128,'resource journal cap')
            batch=port.output(8192-len(out)-len(err))
            for name,b in batch['chunks']:(out if name=='stdout' else err).extend(b)
            require(len(out)+len(err)<=8192,'combined output overflow (one sentinel captured)')
            eof=batch['EOF'];code=port.exit_code(p)
            if code is not None and not ids and eof=={'stdout':True,'stderr':True}:
                native_end=clock();exit_proved=True
                require(native_end>native_started,'zero/unknown native interval')
                guard.check();break
            sleep(min(.1,max(0,guard.until-clock())))
        require(code==0 and not err,'nonzero exit/unexpected stderr')
        match=VERSION_LINE.fullmatch(bytes(out));require(match is not None,'empty/malformed/ambiguous version')
        result.update(status='METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION',
          claimed_line=bytes(out).decode('ascii'),release_lookup_key=match.group(1).decode(),
          publisher_source_or_setup_qualified=False,worker_identity=identity)
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
        result.update(status='FAILED_OR_UNKNOWN_PENDING_EXTERNAL_OBSERVER',reason=reason)
    finally:
        # A factory exception may occur before assignment to local "port".
        # The guard already owns that partial object/resource.
        port=port or guard.port
        cleanup_errors=[]
        if port:
            cleanup_errors.extend(port.cleanup())
            if port.process and port.cleanup_exit_proved and port.cleanup_job_empty:
                exit_proved=True
                if native_started is not None and native_end is None:native_end=port.cleanup_exit_at
        cleanup_errors.extend(guard.errors)
        if cleanup_errors:
            result['cleanup_errors']=cleanup_errors
            result['status']='FAILED_OR_UNKNOWN_PENDING_EXTERNAL_OBSERVER'
        # Never publish canonical terminal SUCCEEDED/FAILED with a pre-write
        # sample. RUNNING/OUTCOME_UNKNOWN keeps the full reservation; a qualified
        # external finalization observer supplies the immutable terminal proof.
        outcome_ref=None;ledger_errors=[]
        try:
            actual_row=store.get_job(gate.get('key'))
            reserved=reserved or actual_row.get('state') in {'RESERVED','RUNNING','OUTCOME_UNKNOWN'}
            if reserved:
                store.mark_unknown(gate['key'],'await independent complete exit/finalization observer')
                native_elapsed=(native_end-native_started) if native_end is not None and native_started is not None else None
                if running and identity and exit_proved and native_elapsed is not None and native_elapsed>0:
                    proof={'observer':gate['owner'],'request_hash':gate['request_hash'],
                      'owner':gate['owner'],'fence':gate['fence'],'worker_identity':identity,
                      'process_exited':True,'exit_observed_at':store.clock(),
                      'elapsed_native_seconds':native_elapsed,'never_dispatched':False}
                    outcome_ref=store.write_receipt('native_outcome_observation',gate['key'],proof)
        except BaseException as exc:ledger_errors.append(type(exc).__name__+': '+str(exc))
        # Every return records quiescence. If a callback or exact handle remains,
        # keep its actual owner reachable in the returned library result. This
        # is in-memory retention only; no cross-process recovery is qualified.
        timer_quiescent=guard.quiesce()
        recovery=port.recovery_state() if port else {
            'Job_handle':guard.job.handle if guard.job else None,
            'in_memory_only_not_durable_or_transferred':True}
        retained=not timer_quiescent or bool(recovery.get('process_handle') or recovery.get('Job_handle') or
                                             recovery.get('other_handles') or recovery.get('attribute_retained'))
        completed=clock()
        late=guard.expired.is_set() or completed>=guard.until
        if late or ledger_errors:
            result['status']='LATE_OR_UNKNOWN_PENDING_EXTERNAL_OBSERVER'
        result.update(raw_stdout_hex=out.hex(),raw_stderr_hex=err.hex(),stream_EOF=eof,
          resource_samples=resources,worker_exit_proved=exit_proved,worker_identity=identity,
          native_interval_started=native_started,native_interval_ended=native_end,
          native_elapsed_seconds=(native_end-native_started) if native_end is not None and native_started is not None else None,
          caller_wall_through_internal_finalization_seconds=completed-guard.started,
          caller_terminal_serialization_and_external_observer_complete=False,
          combined_deadline_exceeded=late,canonical_terminal_published=False,
          outcome_observation_ref=outcome_ref,ledger_errors=ledger_errors,
          reservation_retained_seconds=10 if reserved else None,
          independent_worker_kill_timer=True,independent_caller_wall_limit_qualified=False,
          timer_quiescent_at_return=timer_quiescent,retained_ownership=recovery,
          recovery_owner=guard if retained else None,
          recovery_owner_scope='SAME_CALLER_IN_MEMORY_ONLY_NO_DURABLE_RECOVERY_OR_ACKNOWLEDGED_TRANSFER')
    return result


if __name__=='__main__':
    raise SystemExit('NOT_READY: no actual launch CLI; separate exact source/role/commit/admission/accounting registration required')
