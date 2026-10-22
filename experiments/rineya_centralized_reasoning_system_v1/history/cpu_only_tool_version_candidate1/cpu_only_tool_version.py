"""Candidate1 for one UV metadata observation. No CLI dispatch is provided.

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
    def __init__(self,kernel):
        self.kernel=kernel;self.handle=None
        for name,args,result in [
            ('CreateJobObjectW',[c.c_void_p,w.LPCWSTR],w.HANDLE),
            ('SetInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD],w.BOOL),
            ('QueryInformationJobObject',[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p],w.BOOL),
            ('AssignProcessToJobObject',[w.HANDLE,w.HANDLE],w.BOOL),
            ('TerminateJobObject',[w.HANDLE,w.UINT],w.BOOL),('CloseHandle',[w.HANDLE],w.BOOL)]:
            fn=getattr(kernel,name);fn.argtypes=args;fn.restype=result
        self.handle=kernel.CreateJobObjectW(None,None);require(self.handle,'create owned Job')
        limits=Extended();limits.BasicLimitInformation.LimitFlags=FLAGS
        limits.BasicLimitInformation.ActiveProcessLimit=1;limits.JobMemoryLimit=2*GIB
        try:
            require(kernel.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits)),'set Job limits')
            self.verify_limits()
        except BaseException:
            self.close();raise
    def verify_limits(self):
        limits=Extended()
        require(self.kernel.QueryInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits),None),'query Job limits')
        b=limits.BasicLimitInformation
        require(b.LimitFlags==FLAGS and b.ActiveProcessLimit==1 and limits.JobMemoryLimit==2*GIB,'Job flag/process/commit drift')
        return {'limit_flags':int(b.LimitFlags),'ActiveProcessLimit':1,'JobMemoryLimit':int(limits.JobMemoryLimit),'breakaway':False}
    def assign(self,process):
        require(self.kernel.AssignProcessToJobObject(self.handle,int(process._handle)),'suspended Job assignment')
        require(self.pids()==[process.pid],'exact one suspended PID in Job')
    def pids(self):
        p=Pids();require(self.kernel.QueryInformationJobObject(self.handle,3,c.byref(p),c.sizeof(p),None),'complete Job PID query')
        require(p.assigned==p.returned and p.returned<=1,'Job child/incomplete list')
        return [int(p.pids[i]) for i in range(p.returned)]
    def kill(self):
        require(self.kernel.TerminateJobObject(self.handle,2),'owned Job termination')
    def close(self):
        if self.handle:
            handle=self.handle;self.handle=None
            require(self.kernel.CloseHandle(handle),'owned Job handle close')

class WindowsPort:
    """No OS call occurs until constructed by a separately admitted caller."""
    def __init__(self):
        require(os.name=='nt','Windows only')
        self.kernel=c.WinDLL('C:/Windows/System32/kernel32.dll',use_last_error=True)
        self.psapi=c.WinDLL('C:/Windows/System32/psapi.dll',use_last_error=True)
        self.job=OwnedJob(self.kernel)
        self.kernel.GetProcessTimes.argtypes=[w.HANDLE,c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME)]
        self.kernel.GetProcessTimes.restype=w.BOOL
        self.kernel.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,c.POINTER(w.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype=w.BOOL
        self.kernel.PeekNamedPipe.argtypes=[w.HANDLE,c.c_void_p,w.DWORD,c.c_void_p,c.POINTER(w.DWORD),c.c_void_p]
        self.kernel.PeekNamedPipe.restype=w.BOOL
    def spawn_suspended(self,cwd,env):
        return subprocess.Popen([UV_PATH,'--version'],executable=UV_PATH,cwd=cwd,env=env,shell=False,
          stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,close_fds=True,
          creationflags=4|0x400|0x08000000)  # CREATE_SUSPENDED/UNICODE_ENVIRONMENT/NO_WINDOW
    def identity(self,p):
        ft=[w.FILETIME() for _ in range(4)]
        require(self.kernel.GetProcessTimes(int(p._handle),*(c.byref(x) for x in ft)),'creation identity')
        buf=c.create_unicode_buffer(32768);size=w.DWORD(len(buf))
        require(self.kernel.QueryFullProcessImageNameW(int(p._handle),0,buf,c.byref(size)),'image identity')
        return {'pid':p.pid,'creation_time':str((ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime),
                'image':buf.value.replace('\\','/'),'command_hash':hashlib.sha256(canonical([UV_PATH,'--version'])).hexdigest()}
    def resume(self,p):
        class Thread(c.Structure):
            _fields_=[('size',w.DWORD),('usage',w.DWORD),('tid',w.DWORD),('pid',w.DWORD),('priority',w.LONG),('delta',w.LONG),('flags',w.DWORD)]
        k=self.kernel
        k.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];k.CreateToolhelp32Snapshot.restype=w.HANDLE
        k.Thread32First.argtypes=[w.HANDLE,c.POINTER(Thread)];k.Thread32First.restype=w.BOOL
        k.Thread32Next.argtypes=[w.HANDLE,c.POINTER(Thread)];k.Thread32Next.restype=w.BOOL
        k.OpenThread.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenThread.restype=w.HANDLE
        k.ResumeThread.argtypes=[w.HANDLE];k.ResumeThread.restype=w.DWORD
        snap=k.CreateToolhelp32Snapshot(4,0);require(snap and snap!=c.c_void_p(-1).value,'thread snapshot')
        tids=[];count=0
        try:
            entry=Thread();entry.size=c.sizeof(entry);more=k.Thread32First(snap,c.byref(entry))
            while more:
                count+=1;require(count<=65536,'thread inventory cap')
                if entry.pid==p.pid:tids.append(entry.tid)
                more=k.Thread32Next(snap,c.byref(entry))
        finally:k.CloseHandle(snap)
        require(len(tids)==1 and p.poll() is None,'exact one suspended primary thread')
        h=k.OpenThread(2,False,tids[0]);require(h,'primary thread handle')
        try:require(k.ResumeThread(h)==1,'prior suspend count exactly one')
        finally:k.CloseHandle(h)
    def output(self,p,remaining):
        import msvcrt
        chunks=[]
        for name,f in [('stdout',p.stdout),('stderr',p.stderr)]:
            available=w.DWORD();handle=msvcrt.get_osfhandle(f.fileno())
            ok=self.kernel.PeekNamedPipe(handle,None,0,None,c.byref(available),None)
            if not ok:
                require(c.get_last_error()==109 and p.poll() is not None,'pipe observation unknown')
            if available.value:chunks.append((name,os.read(f.fileno(),min(available.value,remaining+1))))
        return chunks
    def rss(self,p):
        class Memory(c.Structure):
            _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(n,c.c_size_t) for n in
              ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        fn=self.psapi.GetProcessMemoryInfo;fn.argtypes=[w.HANDLE,c.c_void_p,w.DWORD];fn.restype=w.BOOL
        m=Memory();m.cb=c.sizeof(m);require(fn(int(p._handle),c.byref(m),c.sizeof(m)),'RSS unknown')
        return int(m.WorkingSetSize)
    def kill_prejob(self,p,identity):
        require(self.identity(p)==identity,'pre-Job known PID/image/creation drift')
        p.kill()
    def close_pipes(self,p):
        for f in (p.stdout,p.stderr):
            if f:f.close()

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

def observe_version(gate,store,port_factory,telemetry,*,clock=time.monotonic,sleep=time.sleep):
    """Store is canonical OperationsStore; fixtures inject a stdlib fake only.

    This library does not instantiate/open a host database or supply a launch
    key. Accepted independent receipts/current clean-commit validation are a
    separate caller prerequisite; booleans below are not role-attestation proof.
    """
    started=clock();validate_admission(gate,clock);deadline=started+10
    require(file_sha(PACKAGE/'operations.py')==OPS_SHA,'canonical ops hash changed')
    require(gate.get('physical_paths_verified') is True and gate.get('empty_scoped_paths_verified') is True,'fresh canonical/reparse/path proof absent')
    cwd=gate['cwd'];temp=gate['temp']
    require(Path(cwd).is_relative_to(PACKAGE/'.venv/tool-version-007') and Path(temp).is_relative_to(PACKAGE/'.venv/tool-version-007'),'scoped paths')
    env={'SystemRoot':'C:/Windows','WINDIR':'C:/Windows','PATH':'C:/Windows/System32','TEMP':temp,'TMP':temp}
    # No inherited mapping, proxies, tokens, user profiles, Python/UV/PIP paths.
    store._require_lease(gate['owner'],gate['fence'])
    jobrow=store.get_job(gate['key'])
    require(jobrow['request_hash']==gate['request_hash'] and jobrow['epoch']==gate['epoch'] and jobrow['state']=='READY','pre-registered exact request only')
    store.reserve(gate['key'],gate['owner'],gate['fence'],10)
    p=port=identity=None;assigned=False;running=False;out=bytearray();err=bytearray();reason=None;exit_proved=False
    try:
        require(clock()<deadline,'deadline before port creation')
        port=port_factory();p=port.spawn_suspended(cwd,env);identity=port.identity(p)
        require(identity['image'].casefold()==UV_PATH.casefold(),'launched image drift')
        port.job.assign(p);assigned=True
        witness=port.job.verify_limits();require(port.job.pids()==[p.pid],'zero children before resume')
        store.mark_running(gate['key'],gate['owner'],gate['fence'],identity);running=True
        require(clock()<deadline,'late suspended completion');store._require_lease(gate['owner'],gate['fence'])
        port.resume(p)
        while True:
            store._require_lease(gate['owner'],gate['fence'])
            require(clock()<deadline,'10 second deadline')
            require(port.identity(p)==identity,'PID/ctime/image drift')
            ids=port.job.pids();require(ids in ([p.pid],[]),'unexpected child/Job drift')
            port.job.verify_limits()
            sample=telemetry()
            require(sample['age_seconds']<=5 and sample['host_available_bytes']>=1.25*GIB and sample['disk_free_bytes']>=4*GIB,'stale/host/disk telemetry')
            if p.poll() is None:require(port.rss(p)<=2*GIB,'whole one-process RSS cap')
            for name,b in port.output(p,8192-len(out)-len(err)):
                (out if name=='stdout' else err).extend(b)
                require(len(out)+len(err)<=8192,'combined output overflow')
            if p.poll() is not None and not ids:
                # Drain once more; success needs EOF/exit within same deadline.
                for name,b in port.output(p,8192-len(out)-len(err)):(out if name=='stdout' else err).extend(b)
                require(len(out)+len(err)<=8192 and clock()<deadline,'late/overflow finalization')
                exit_proved=True;break
            sleep(min(.1,max(0,deadline-clock())))
        require(p.returncode==0 and not err,'nonzero exit/unexpected stderr')
        match=VERSION_LINE.fullmatch(bytes(out));require(match is not None,'empty/malformed/ambiguous version')
        result={'status':'SELF_REPORTED_VERSION_ONLY','raw_stdout_hex':out.hex(),'raw_stderr_hex':err.hex(),
                'claimed_line':bytes(out).decode('ascii'),'release_lookup_key':match.group(1).decode(),
                'publisher_source_or_setup_qualified':False,'worker_identity':identity,'Job':witness}
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
        result={'status':'FAILED_OR_UNKNOWN','reason':reason,'raw_stdout_hex':out.hex(),'raw_stderr_hex':err.hex()}
    finally:
        if p and port:
            try:
                if assigned:port.job.kill()
                elif p.poll() is None:port.kill_prejob(p,identity)
                # Bounded finalization; unknown retains reservation, no late success.
                while clock()<deadline:
                    if p.poll() is not None and (not assigned or not port.job.pids()):exit_proved=True;break
                    sleep(.01)
                if assigned and port.job.pids():exit_proved=False
                port.close_pipes(p);port.job.close()
            except BaseException as exc:
                exit_proved=False;result['cleanup_error']=type(exc).__name__+': '+str(exc)
        elapsed=clock()-started
        result.update(elapsed_seconds=elapsed,process_exited=exit_proved,source_stage_qualified=False)
        if not exit_proved or not running:
            store.mark_unknown(gate['key'],'unknown dispatch/exit: '+str(reason));result['reservation_retained_seconds']=10
        else:
            outcome='SUCCEEDED' if reason is None and elapsed<10 else 'FAILED'
            proof={'observer':gate['owner'],'request_hash':gate['request_hash'],'owner':gate['owner'],'fence':gate['fence'],
                   'worker_identity':identity,'process_exited':True,'exit_observed_at':store.clock(),
                   'elapsed_native_seconds':elapsed,'never_dispatched':False}
            ref=store.write_receipt('native_outcome_observation',gate['key'],proof)
            store.reconcile(gate['key'],outcome,elapsed,{'native_outcome_proof_ref':ref,'metadata_result':result},owner=gate['owner'],fence=gate['fence'])
    return result

if __name__=='__main__':
    raise SystemExit('NOT_READY: no actual launch CLI; separate exact source/role/commit/admission/accounting registration required')
