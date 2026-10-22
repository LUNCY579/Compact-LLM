"""Concrete original CE8 owner caller. One CPU OS fixture; no GPU/model scope."""
import ctypes as c
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import threading
import time
import types
import uuid

import admission as a
from accounting import Budget
from collections.abc import Mapping


class MemoryCounters(c.Structure):
    _fields_ = [('cb', c.c_uint32), ('faults', c.c_uint32)] + [
        (name, c.c_size_t) for name in ('peak_working', 'working', 'peak_paged', 'paged',
                                      'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]

class AvailableMemory(c.Structure):
    _fields_ = [('length', c.c_uint32), ('load', c.c_uint32)] + [
        (name, c.c_uint64) for name in ('total', 'available', 'total_page', 'available_page', 'virtual', 'available_virtual', 'extended')]

_POLL_KEYS=('supervisor_signaled','actual_supervisor_returncode','job_active_processes','job_total_processes','job_cumulative_io')
_IO_KEYS=('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')
class CurrentIOView(Mapping):
    __slots__=('_backend',)
    def __init__(self,backend):self._backend=backend
    def __getitem__(self,key):
        if key not in _IO_KEYS:raise KeyError(key)
        return int(getattr(self._backend._poll_job.io,key))
    def __iter__(self):return iter(_IO_KEYS)
    def __len__(self):return len(_IO_KEYS)
class CurrentPollView(Mapping):
    __slots__=('_backend',)
    def __init__(self,backend):self._backend=backend
    def __getitem__(self,key):
        b=self._backend
        if key=='supervisor_signaled':return b._poll_waited==b.provider.WAIT_OBJECT_0
        if key=='actual_supervisor_returncode':return int(b._poll_exit.value) if b._poll_waited==b.provider.WAIT_OBJECT_0 else None
        if key=='job_active_processes':return int(b._poll_job.basic.active)
        if key=='job_total_processes':return int(b._poll_job.basic.total)
        if key=='job_cumulative_io':return b._poll_io_view
        raise KeyError(key)
    def __iter__(self):return iter(_POLL_KEYS)
    def __len__(self):return len(_POLL_KEYS)


class NativeObservationFrame(c.Structure):
    """Fixed retained scalar payload; object references live in four slots.

    The original reusable 8192-byte selection is a logical named workspace,
    not a Python allocator byte count. Current process RSS/commit gates remain.
    """
    _pack_=1
    _fields_=[('flags',c.c_uint32),('windows_error',c.c_uint32),
      ('stop_windows_error',c.c_uint32),('descriptor_attempts',c.c_uint8),
      ('descriptor_selected_count',c.c_uint8),('descriptor_return_count',c.c_uint8),
      ('descriptor_attempt_mask',c.c_uint8),('descriptor_return_mask',c.c_uint8),
      ('descriptor_afterguard_mask',c.c_uint8),('descriptor_error_mask',c.c_uint8),
      ('descriptor_tail_selected',c.c_uint8),
      ('stop_started_at',c.c_double),('stop_returned_at',c.c_double)]

_OBSERVATION_FLAGS=('_native_selected','_ffi_attempt_entered','_ffi_returned',
  '_stop_started','_stop_native_selected','_stop_ffi_attempt_entered',
  '_stop_ffi_returned','_stop_wrapper_finished','_descriptor_selected',
  '_descriptor_entered','_descriptor_returned','_native_afterguard_finished',
  '_descriptor_afterguard_finished','_descriptor_batch_selected',
  '_descriptor_batch_started')
_OBSERVATION_REFS=('_native_name','_native_error','_stop_error','_descriptor_error')

def observation_flag(index):
    mask=1<<index
    def get(self):return bool(self._observation.flags&mask)
    def set(self,value):
        a.require(type(value) is bool,'fixed observation boolean')
        self._observation.flags=(self._observation.flags|mask) if value else (self._observation.flags&~mask)
    return property(get,set)

def observation_reference(index):
    def get(self):return self._observation_refs[index]
    def set(self,value):self._observation_refs[index]=value
    return property(get,set)

def observation_optional_scalar(field,valid_flag):
    def get(self):
        if field=='windows_error' and self._native_name=='NtQuerySystemInformation':return None
        return getattr(self._observation,field) if getattr(self,valid_flag) else None
    def set(self,value):
        if value is None:return
        setattr(self._observation,field,value)
    return property(get,set)

class FixedDescriptorSlots:
    """Exactly two retained CRT ownership slots; no shrinking temporary list."""
    __slots__=('_values',)
    def __init__(self):self._values=(c.c_int64*2)(-1,-1)
    def __len__(self):return int(self._values[0]>=0)+int(self._values[1]>=0)
    def __bool__(self):return bool(len(self))
    def __iter__(self):
        for index in range(2):
            value=int(self._values[index])
            if value>=0:yield value
    def __getitem__(self,index):
        a.require(type(index) is int and 0<=index<len(self),'fixed descriptor index')
        for value in self:
            if index==0:return value
            index-=1
        raise IndexError(index)
    def append(self,value):
        a.require(type(value) is int and value>=0,'owned CRT descriptor')
        a.require(value not in self,'one original descriptor ownership slot')
        for index in range(2):
            if self._values[index]<0:self._values[index]=value;return
        a.require(False,'two descriptor ownership slots exhausted')
    def remove(self,value):
        for index in range(2):
            if self._values[index]==value:self._values[index]=-1;return
        raise ValueError(value)
    def empty(self):return len(self)==0

def control(metadata):
    """Logical source-control envelope; physical heap remains an RSS gate."""
    def decorate(work):
        def controlled(self,*args,**kwargs):
            return self.owner.run('control',{'operation_attempts':1,'metadata_bytes':metadata},lambda:work(self,*args,**kwargs))
        return controlled
    return decorate


# No native API is called by this preparation. The immutable bytes below are a
# native INPUT ONLY: CreateProcessW must not write the environment block.
# Python object mutation is outside the trusted source contract; exported data
# cannot recreate this registered same-process/source/owner capability.
class ImageFileTime(c.Structure):
    _fields_=[('low',c.c_uint32),('high',c.c_uint32)]
class ImageFileInformation(c.Structure):
    _fields_=[('attributes',c.c_uint32),('creation',ImageFileTime),('access',ImageFileTime),
      ('write',ImageFileTime),('volume',c.c_uint32),('size_hi',c.c_uint32),('size_lo',c.c_uint32),
      ('links',c.c_uint32),('id_hi',c.c_uint32),('id_lo',c.c_uint32)]

class BootEnvironment(c.Structure):
    # Once-defined paid source ABI; no DLL binding or query occurs at import.
    _fields_=[('identifier',c.c_ubyte*16),('firmware',c.c_uint32),('flags',c.c_uint64)]

_CREATION_SEAL=object()
_CREATION_PLANS={}
class PreparedCreationPlan:
    __slots__=('_seal','_pid','_prep_owner','_root_owner','_static','_environment','_environment_buffer',
      '_environment_pointer','_commands','_runtime','_cwd','_environment_keys',
      '_expected_path','_runtime_units','_boot_library','_boot_query','_log_inputs',
      '_custody','_runtime_pin','_runtime_pin_identity','_runtime_pin_reference','_final_units','_image_library','_image_functions')
    def __setattr__(self,*args):raise RuntimeError('immutable prepared creation plan')
    def __delattr__(self,*args):raise RuntimeError('immutable prepared creation plan')
    @classmethod
    def prepare(cls,owner,provider,target,*,custody):
        a.require(type(owner) is a.cost.PreparationOwner and owner is a.preparation_owner(),
          'same original paid PREP owner for explicit creation inputs')
        # 8 retained/environment render copies <=65536 each are bounded below
        # by the stricter 32768-byte full source-input domain. Physical RSS and
        # commitment gates remain independent. No interpreter heap claim.
        ticket=owner.before('control',operation_attempts=256,metadata_bytes=524288)
        names=('SystemRoot','WINDIR','TEMP','TMP','PATH','LOCALAPPDATA','APPDATA','USERPROFILE')
        a.require(all(key not in os.environ or (type(os.environ[key]) is str
          and len(os.environ[key])<=16384 and '\x00' not in os.environ[key]) for key in names),
          'bounded exact whitelist input before environment dictionary copy')
        environment=provider.explicit_environment()
        a.require(type(environment) is dict and len(environment)<=17 and all(type(k) is str
          and type(v) is str and '\x00' not in k and '\x00' not in v and '=' not in k
          for k,v in environment.items()),'finite exact explicit environment')
        # UTF16 sizing is exact; all temporaries are owned by the PREP selection.
        a.require(sum(len(k.encode('utf-16-le'))+len(v.encode('utf-16-le'))+4
          for k,v in environment.items())+4<=32768,'complete prepared environment source domain')
        text='\x00'.join(k+'='+v for k,v in sorted(environment.items(),key=lambda pair:pair[0].upper()))+'\x00\x00'
        a.require(len(text)<=32767,'original bounded Unicode environment guard')
        raw=text.encode('utf-16-le')+b'\x00\x00'
        a.require(c.sizeof(c.c_wchar)==2,'Windows UTF16 ABI mandatory for prepared environment')
        # Actual native Unicode buffer allocated once here, under PREP. Root
        # borrows this exact privately retained object. No refresh/copy occurs.
        buffer=c.create_unicode_buffer(text)
        a.require(c.sizeof(buffer)==len(raw),'exact prepared native environment byte extent')
        runtime=target['runtime_ref']['path'];artifact=target['artifact_ref']['path']
        a.require(type(runtime) is str and type(artifact) is str and '\x00' not in runtime
          and '\x00' not in artifact and len(runtime.encode('utf-16-le'))<1024 and Path(runtime).is_absolute(),
          'registered exact canonical runtime image fits512wchar input domain')
        expected_path=Path(runtime).resolve()
        a.require(str(expected_path)==runtime,'prepared runtime path must already be canonical')
        runtime_units=len(runtime.encode('utf-16-le'))//2
        a.require(runtime.isascii() and len(runtime)>3 and runtime[0].isalpha() and runtime[1:3]==':\\'
          and '\\' in runtime and '/' not in runtime,'strict registered local DOS path; aliases/UNC refused')
        extended='\\\\?\\'+runtime
        final_units=tuple(int.from_bytes(extended.encode('utf-16-le')[i:i+2],'little')
          for i in range(0,len(extended.encode('utf-16-le')),2))
        a.require(len(final_units)+1<=516 and c.sizeof(ImageFileInformation)==52,'bounded normalized DOS image ABI')
        runtime_pin=custody.borrow_runtime_pin(target['runtime_ref'],owner)
        image_library=c.WinDLL('kernel32',use_last_error=True)
        image_library.CreateFileW.argtypes=[c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_void_p,c.c_uint32,c.c_uint32,c.c_void_p]
        image_library.CreateFileW.restype=c.c_void_p
        image_library.GetFileInformationByHandle.argtypes=[c.c_void_p,c.POINTER(ImageFileInformation)]
        image_library.GetFileInformationByHandle.restype=c.c_int
        image_library.GetFinalPathNameByHandleW.argtypes=[c.c_void_p,c.c_void_p,c.c_uint32,c.c_uint32]
        image_library.GetFinalPathNameByHandleW.restype=c.c_uint32
        image_functions=(image_library.CreateFileW,image_library.GetFileInformationByHandle,image_library.GetFinalPathNameByHandleW)
        # Binding/signature preparation is in the already positive whole PREP
        # plan selection. No native query is performed and no GUID is cached.
        boot_library=c.WinDLL('ntdll',use_last_error=True)
        boot_query=boot_library.NtQuerySystemInformation
        boot_query.argtypes=[c.c_uint32,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32)]
        boot_query.restype=c.c_long
        commands=[]
        log_inputs=[]
        for case in a.CASE_NAMES:
            argv=(runtime,'exec',artifact,'--qualification-probe',str(a.ROOT/'FIXTURE_GRANT.json'),
              '0'*64,str(a.ACTOR_OUTPUT/case))
            rendered=subprocess.list2cmdline(argv)
            a.require(rendered.count('0'*64)==1,'unique fixed fresh grant insertion slot')
            prefix,suffix=rendered.split('0'*64)
            a.require((len(prefix.encode('utf-16-le'))+128+len(suffix.encode('utf-16-le'))+2)<=2048,
              'complete future mutable grant-bound command fits2048bytes')
            commands.append((case,argv,prefix.encode('utf-16-le'),suffix.encode('utf-16-le')))
            # Complete exact log paths/names are prepared once under this
            # existing positive PREP plan. Root borrows by object identity.
            directory=a.OUTPUT/('parent_'+case)
            paths=(directory/'supervisor.stdout.log',directory/'supervisor.stderr.log')
            names=('ce8-log-'+case+'-stdout','ce8-log-'+case+'-stderr')
            a.require(all(len(name)<=32 for name in names),'registered fixed log thread names')
            path_strings=tuple(str(path) for path in paths)
            sidecars=tuple(str(directory/('late-block-'+str(index)+'.bin')) for index in range(2))
            a.require(all(len(value)<=320 and value.isascii() for value in path_strings+sidecars),
              'registered exact rendered log/sidecar path extent')
            log_inputs.append((case,directory,paths,names,path_strings,sidecars))
        obj=object.__new__(cls)
        fields={'_seal':_CREATION_SEAL,'_pid':os.getpid(),'_prep_owner':owner,'_root_owner':None,
          '_static':None,'_environment':raw,'_environment_buffer':buffer,'_environment_pointer':c.cast(buffer,c.c_void_p),
          '_commands':tuple(commands),'_runtime':runtime,'_cwd':str(a.ROOT),'_environment_keys':len(environment),
          '_expected_path':expected_path,'_runtime_units':runtime_units,'_boot_library':boot_library,'_boot_query':boot_query,
          '_log_inputs':tuple(log_inputs),'_custody':custody,'_runtime_pin':runtime_pin,
          '_runtime_pin_identity':runtime_pin['identity'],'_runtime_pin_reference':runtime_pin['reference'],
          '_final_units':final_units,'_image_library':image_library,'_image_functions':image_functions}
        for key,value in fields.items():object.__setattr__(obj,key,value)
        _CREATION_PLANS[id(obj)]=obj;ticket.finish();return obj
    def bind_original_root(self,owner,static):
        a.require(self._seal is _CREATION_SEAL and _CREATION_PLANS.get(id(self)) is self
          and self._pid==os.getpid() and self._prep_owner is a._PREP_OWNER
          and self._root_owner is None and type(owner) is Budget,'one exact original creation-plan handoff')
        object.__setattr__(self,'_root_owner',owner);object.__setattr__(self,'_static',static)
    def command(self,owner,permit,case,argv,cwd):
        a.require(self._seal is _CREATION_SEAL and _CREATION_PLANS.get(id(self)) is self
          and self._pid==os.getpid() and self._root_owner is owner and self._static is permit._static
          and permit._static.creation_plan is self,'same source-owned prepared creation object')
        permit._static.require(owner,permit.custody)
        a.require(str(cwd)==self._cwd and case in a.CASE_NAMES,'exact registered root/case command')
        grant=permit.grant_ref['sha256']
        a.require(type(grant) is str and len(grant)==64 and all(ch in '0123456789abcdef' for ch in grant),
          'actual fresh grant exact64hex domain')
        for named,fixed,prefix,suffix in self._commands:
            if named==case:
                a.require(type(argv) is tuple and len(argv)==7 and argv[:5]==fixed[:5]
                  and argv[5]==grant and argv[6]==fixed[6],'exact target argv including current fresh grant')
                # The complete mutable command allocation is inside the
                # positively selected root creation workspace, never PREP.
                a.require(len(prefix)+128+len(suffix)+2<=2048,
                  'positive2048 command limit before allocation')
                command=c.create_unicode_buffer(1024)
                # Three bounded source-copy loops, <=1023 UTF16 units total.
                # No post-origin concatenation, encoding, sort or join copy.
                cursor=0
                for offset in range(0,len(prefix),2):
                    command[cursor]=chr(prefix[offset]|prefix[offset+1]<<8);cursor+=1
                for char in grant:command[cursor]=char;cursor+=1
                for offset in range(0,len(suffix),2):
                    command[cursor]=chr(suffix[offset]|suffix[offset+1]<<8);cursor+=1
                a.require(cursor<1024 and command[cursor]=='\x00','complete grant-bound command terminator')
                return command
        a.require(False,'unregistered exact command case')

def backend_type(provider,max_members=6,*,owner=None,census=None,creation_plan=None):
    a.require(type(owner) is Budget,'same original CPU owner required before backend class construction')
    construction=owner.before('control',operation_attempts=1,metadata_bytes=256)
    a.require(max_members in (4, 6, 8), 'fixed registered fixture topology census')
    a.require(type(census) is types.ModuleType and census.__file__==a.JOB_CENSUS['path']
      and census.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==a.JOB_CENSUS['sha256'],
      'prepared exact census object; no backend metadata reload')
    a.require(type(provider) is types.ModuleType and provider.__file__==a.PROVIDER['path']
      and provider.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==a.PROVIDER['sha256'],
      'prepared exact existing provider object')
    a.require(type(creation_plan) is PreparedCreationPlan and creation_plan._seal is _CREATION_SEAL
      and _CREATION_PLANS.get(id(creation_plan)) is creation_plan and creation_plan._root_owner is owner
      and creation_plan._pid==os.getpid(),'exact source-owned prepared creation/boot function before class construction')
    class FixtureBackend(provider.Win32Backend):
        def __init__(self):
            self.owner=owner
            setup=owner.before('control',operation_attempts=1,metadata_bytes=2048)
            # Select the COMPLETE shared frame before any state/buffer allocation,
            # including provider construction and every partial-failure path.
            reusable=owner.before('control',operation_attempts=128,metadata_bytes=8192)
            self._observation=NativeObservationFrame()
            self._observation_refs=[None]*9
            self.closed=False;self.job=self.process=self.thread=None
            self.attributes_initialized=False;self.attribute_pointer=None
            self.read_fds=FixedDescriptorSlots();self.write_fds=FixedDescriptorSlots();self.stdin_fd=None
            self._owned_rss_handles=[None]*max_members;self._rss_open_slot=None
            self._job_created=False;self._job_empty_observed=False
            self._process_created=False;self._process_terminal_observed=False
            self._stop_lock=threading.Lock()
            self._owned_logs=None;self._logs_joined=False
            self._cleanup_local=threading.local()
            # Exact provider constructor fields, with descriptor containers
            # already replaced by this workspace's retained fixed slots.
            # Keep this concrete constructor unqualified until Native checks.
            a.require(os.name=='nt','Windows native backend only')
            self.clock=time.monotonic;self.calls=0;self.counter_lock=threading.Lock()
            self.pi=None;self.identity=None;self.abort_evidence=None
            self.attribute_buffer=None;self.attribute_arrays=None
            self.kernel=c.WinDLL('kernel32',use_last_error=True)
            self._declare()
            self.kernel.K32GetProcessMemoryInfo.argtypes = [c.c_void_p, c.c_void_p, c.c_uint32]
            self.kernel.K32GetProcessMemoryInfo.restype = c.c_int
            self.kernel.OpenProcess.argtypes = [c.c_uint32, c.c_int, c.c_uint32]
            self.kernel.OpenProcess.restype = c.c_void_p
            self.rss_samples = 0
            self.max_observed_rss = 0
            self.last_rss_sample = None
            self.max_rss_sample_gap = 0
            self.pointer_bytes=c.sizeof(c.c_void_p)
            self.last_job_census_evidence=[]
            self.census=census;self.provider=provider
            self._poll_job=provider.JobAccounting();self._poll_job_pointer=c.byref(self._poll_job);self._poll_exit=c.c_uint32();self._poll_waited=provider.WAIT_TIMEOUT
            self._ram=AvailableMemory();self._ram.length=c.sizeof(self._ram)
            self._own_mem=MemoryCounters();self._own_mem.cb=c.sizeof(self._own_mem)
            self._member_mem=MemoryCounters();self._member_mem.cb=c.sizeof(self._member_mem)
            self._inside=c.c_int();self._initial_members=(c.c_uint64*max_members)();self._initial_count=0
            self._active_raw_capture=None;self._terminal_result=None;self._polling=False
            # Two scalar/reference slots belong to the original positive
            # reusable backend workspace; no extra result/metadata ticket.
            self._identity_callback_active=False;self._creation_plan=creation_plan
            self._image_file=None;self._image_check_active=False;self._image_check_complete=False
            self._image_check_started=False;self._image_close_attempted=False
            self._poll_io_view=CurrentIOView(self);self._poll_view=CurrentPollView(self)
            self.census.prepare_storage(self,max_members)
            self.kernel.GlobalMemoryStatusEx.argtypes=[c.POINTER(AvailableMemory)];self.kernel.GlobalMemoryStatusEx.restype=c.c_int
            self.owner.register_owned_backend(self,reusable,job_bytes=c.sizeof(self._poll_job))
            self._terminal_verification_context=None
            reusable.finish()
            # Fixed slots retain every original RSS handle until an observed
            # successful close, including a failure of a later return guard.
            setup.finish()
        def native_selection_started(self,name):
            self._native_name=name;self._native_selected=False
            self._ffi_attempt_entered=False;self._ffi_returned=False
            self._native_error=None;self._observation.windows_error=0;self._native_afterguard_finished=False
        def native_selected(self,name):
            a.require(name==self._native_name,'same source-owned selected native API')
            self._native_selected=True
            if name=='TerminateJobObject':self._stop_native_selected=True
        def _invoke_primitive(self,name,args,allow_false):
            # Exact existing provider counter/ceiling and false-result guard,
            # now source-owned to observe the actual FFI boundary. The old
            # qualified provider is unchanged; this override is unqualified.
            a.require(self._native_selected and name==self._native_name,
              'actual original reservation required before native primitive')
            with self.counter_lock:
                self.calls+=1
                provider.require(self.calls<=1000000,'finite native API callback budget')
            if name=='NtQuerySystemInformation':
                a.require(self._identity_callback_active and not self._stop_started and self._creation_plan is creation_plan
                  and creation_plan._root_owner is self.owner and creation_plan._pid==os.getpid(),
                  'fresh boot query only within exact original nonreentrant identity callback')
                function=creation_plan._boot_query
            elif name in ('CreateFileW','GetFileInformationByHandle','GetFinalPathNameByHandleW'):
                a.require(self._image_check_active and self._creation_plan is creation_plan
                  and creation_plan._root_owner is self.owner,'only exact original bounded image check')
                function=creation_plan._image_functions[('CreateFileW','GetFileInformationByHandle','GetFinalPathNameByHandleW').index(name)]
                if name=='CreateFileW':
                    a.require(self._image_file is None and args==(creation_plan._runtime,0x80000000,1,None,3,0x80,None),
                      'only one registered read-only OPEN_EXISTING noninherited image handle')
            else:function=getattr(self.kernel,name)
            if name=='CloseHandle' and args and args[0]==self._image_file:
                a.require(not self._image_close_attempted,'original image file close is never retried')
                self._image_close_attempted=True
            # This is an FFI CALL ATTEMPT marker. ctypes argument conversion
            # can still raise before OS entry, which this source cannot see.
            self._ffi_attempt_entered=True
            if name=='TerminateJobObject':self._stop_ffi_attempt_entered=True
            try:value=function(*args)
            except BaseException as error:
                self._native_error=error
                raise
            self._ffi_returned=True
            # NTSTATUS uses zero-success semantics and supplies no Win32
            # GetLastError evidence. The existing Windows-error view returns
            # None for this API. Local status/ABI guards validate its result.
            invalid_image_handle=name=='CreateFileW' and value in (None,0,c.c_void_p(-1).value)
            actual_error=c.get_last_error() if invalid_image_handle else (0 if name=='NtQuerySystemInformation' or value else c.get_last_error())
            if name=='CreateFileW' and not invalid_image_handle:
                # Capture inside the primitive, before ANY outer return guard.
                self._image_file=value
            self._native_windows_error=actual_error
            if name=='TerminateJobObject':
                self._stop_ffi_returned=True;self._stop_windows_error=actual_error
            if not allow_false:
                provider.require(bool(value) and not invalid_image_handle,'owned native API failed: '+name+' / '+str(actual_error))
            return value
        def call(self,name,*args,**kwargs):
            # The concrete harness subclass routes native calls exactly once.
            # Ordinary RSS handle closes use that regular native partition.
            try:
                a.require(not kwargs or (len(kwargs)==1 and 'allow_false' in kwargs),
                  'closed existing primitive keyword recipe')
                result=self._invoke_primitive(name,args,kwargs.get('allow_false',False))
                if name=='QueryInformationJobObject' and len(args)>1 and args[1]==3 and self._active_raw_capture is not None:
                    actual_error=0 if result else c.get_last_error()
                    self._active_raw_capture.record_native_return(bool(result),actual_error,int(self._active_raw_capture.returned.value))
                if name=='CreateJobObjectW' and result:
                    self.job=result;self._job_created=True
                if name=='CreateProcessW' and result:
                    pi=c.cast(args[-1],c.POINTER(provider.ProcessInformation)).contents
                    self.pi=pi;self.process=pi.process;self.thread=pi.thread;self._process_created=True
                if name=='InitializeProcThreadAttributeList' and args and args[0] is not None and result:
                    self.attributes_initialized=True
                if name=='WaitForSingleObject' and args and args[0]==self.process and result==provider.WAIT_OBJECT_0:
                    self._process_terminal_observed=True
                if name=='QueryInformationJobObject' and len(args)>1 and args[1]==8 and result:
                    observed=c.cast(args[2],c.POINTER(provider.JobAccounting)).contents
                    self._job_empty_observed=int(observed.basic.active)==0
                if name=='OpenProcess' and result:
                    slot=self._rss_open_slot
                    a.require(type(slot) is int and self._owned_rss_handles[slot] is None,
                      'original RSS slot selected before owned open')
                    self._owned_rss_handles[slot]=result
                if name=='CloseHandle' and result:
                    for slot,handle in enumerate(self._owned_rss_handles):
                        if handle==args[0]:self._owned_rss_handles[slot]=None
                    # Qualified base close() clears these only after call()
                    # returns. Record the observed native release here, before
                    # an outer afterguard can fail and make that value stale.
                    for field in ('thread','process','job','_image_file'):
                        if getattr(self,field,None)==args[0]:setattr(self,field,None)
                if name=='DeleteProcThreadAttributeList' and args and args[0]==self.attribute_pointer:
                    self.attributes_initialized=False;self.attribute_pointer=None
                return result
            except BaseException as primary:
                self._native_error=primary
                # Store only actually observed same-thread native failure;
                # publication happens after unexpected-case classification.
                if type(primary) is RuntimeError and primary.args and type(primary.args[0]) is str and primary.args[0].startswith('owned native API failed:'):
                    primary.__dict__['_ce8_cached_windows_error']=c.get_last_error()
                    primary.__dict__['_ce8_actual_api']=name
                raise
        def release_deferred_handles(self):
            a.require(self.cleanup_mode() is True,
              'only explicit final owned release may use reserved cleanup')
            primary=None;closed=0
            for slot in range(max_members):
                handle=self._owned_rss_handles[slot]
                if handle is None:continue
                try:
                    self.call('CloseHandle',handle)
                    closed+=1
                except BaseException as error:
                    if primary is None:primary=error
                    else:primary.add_note('additional deferred original RSS close: '+type(error).__name__)
            if primary is not None:raise primary
            a.require(all(handle is None for handle in self._owned_rss_handles),'original RSS release incomplete')
            return {'closed_deferred_original_handles':closed,'remaining':0}
        @control(256)
        def pre_run_current_handle_checks(self):
            handle=self.call('GetCurrentProcess')
            pid=self.call('GetProcessId',handle)
            live=self.call('WaitForSingleObject',handle,0,allow_false=True)
            a.require(pid==os.getpid() and live==258,'paid current-handle PID/live boundary')
            actual_io=self.own_io();memory=self.own_memory();ram=self.available_ram()
            a.require(ram>=8*1024**3,'paid current-handle8GiB RAM boundary')
            return {'classification':'ACTUAL_READ_ONLY_CURRENT_HANDLE_BOUNDARIES','pid':pid,'wait_result':live,
                'actual_io':actual_io,'actual_memory':memory,'actual_ram_bytes':ram,'native_calls_upper':8,
                'native_learning_qualified':False}
        @control(0)
        def available_ram(self):
            self.call('GlobalMemoryStatusEx',c.byref(self._ram))
            return int(self._ram.available)
        @control(0)
        def _sample_own_memory(self):
            self.call('K32GetProcessMemoryInfo',self.call('GetCurrentProcess'),c.byref(self._own_mem),c.sizeof(self._own_mem))
        @control(64)
        def create_job(self):
            super().create_job()
            limits = provider.ExtendedLimits()
            limits.basic.flags = provider.KILL_ON_JOB_CLOSE | 0x200 | 0x8
            limits.basic.active_limit = max_members
            limits.job_memory = 320 * 1024**2
            self.call('SetInformationJobObject', self.job, 9, c.byref(limits), c.sizeof(limits))
        @control(128)
        def own_memory(self):
            mem = MemoryCounters(); mem.cb = c.sizeof(mem)
            self.call('K32GetProcessMemoryInfo', self.call('GetCurrentProcess'), c.byref(mem), c.sizeof(mem))
            return {'working_set_bytes': int(mem.working), 'peak_working_set_bytes': int(mem.peak_working),
                    'peak_commit_bytes': int(mem.peak_pagefile)}
        @control(128)
        def own_identity(self):
            # Whole callback borrows original positive1024 scratch inside the
            # backend8192 constructor selection.128 remains RESULT envelope.
            a.require(not self._identity_callback_active and not self._polling and not self._stop_started,
              'original identity/poll scratch is strictly nonreentrant')
            self._identity_callback_active=True
            try:
                handle=self.call('GetCurrentProcess')
                birth,ended,kernel,user=provider.FileTime(),provider.FileTime(),provider.FileTime(),provider.FileTime()
                self.call('GetProcessTimes',handle,c.byref(birth),c.byref(ended),c.byref(kernel),c.byref(user))
                return {'pid':os.getpid(),'creation_filetime_100ns':provider.filetime(birth),
                  'boot_identity':self.boot_identity()}
            finally:self._identity_callback_active=False
        def boot_identity(self):
            # Every identity request performs one fresh actual admitted query.
            # Module-level ABI and original PREP function are borrowed by
            # identity. Cached GUIDs or cold DLL/class preparation are forbidden.
            a.require(self._identity_callback_active and not self._stop_started and self._creation_plan is creation_plan
              and creation_plan._root_owner is self.owner,'exact active original identity callback')
            information=BootEnvironment();returned=c.c_uint32()
            status=self.call('NtQuerySystemInformation',90,c.byref(information),c.sizeof(information),c.byref(returned),allow_false=True)
            a.require(status==0 and returned.value==c.sizeof(information),'actual boot GUID query unavailable')
            identifier=uuid.UUID(bytes_le=bytes(information.identifier))
            a.require(identifier.int!=0 and information.firmware in (1,2) and not information.flags&2,
              'ambiguous boot/hibernation epoch refused')
            return 'windows-boot-guid-'+str(identifier)
        @control(256)
        def own_io(self):
            # Source-owned six-uint64 IO snapshot used at final accounting.
            # The inherited native helper creates this dictionary; price it
            # explicitly rather than treating a read-only result as free.
            return super().own_io()
        def _retain_refusal(self,primary):
            # Original fixed storage is already retained; defer its copying to
            # the single prepaid durable failure exporter.
            primary.__dict__['_ce8_census_storage']=getattr(self,'_reusable_census',None)
            primary.__dict__['_ce8_job_active_total']={
              'active':int(self._poll_job.basic.active),'total':int(self._poll_job.basic.total),
              'scope':'retained class8 before PIDquery; not simultaneous guarantee'}
        @control(64)
        def job_pids(self):
            try:return self.census.query_complete(self,max_members)
            except BaseException as primary:self._retain_refusal(primary);raise
        @control(0)
        def job_contains(self,pid):
            try:return self.census.query_contains(self,max_members,pid)
            except BaseException as primary:self._retain_refusal(primary);raise
        def check_image_file(self):
            plan=self._creation_plan
            a.require(not self._image_check_active and self._image_file is None and not self._image_check_started,
              'one original bounded image check before suspended target release')
            plan._custody.require_source_owner()
            pin=plan._runtime_pin
            a.require(any(row is pin for row in plan._custody._rows) and not pin['closed']
              and pin['handle'] is not None and pin['identity'] is plan._runtime_pin_identity
              and pin['reference'] is plan._runtime_pin_reference and pin['reference']['path']==plan._runtime,
              'same live original runtime pin capability and source identity')
            self._image_check_started=True;self._image_check_active=True
            try:
                self.call('CreateFileW',plan._runtime,0x80000000,1,None,3,0x80,None)
                info=ImageFileInformation()
                self.call('GetFileInformationByHandle',self._image_file,c.byref(info))
                a.require((info.volume,info.id_hi,info.id_lo,(info.size_hi<<32)|info.size_lo,info.attributes)==plan._runtime_pin_identity,
                  'opened image path target equals original volume/file/extent/attribute pin')
                capacity=len(plan._final_units)+1
                final=(c.c_uint16*capacity)()
                returned=self.call('GetFinalPathNameByHandleW',self._image_file,c.byref(final),capacity,0)
                a.require(returned==len(plan._final_units) and returned<capacity,
                  'exact bounded normalized DOS length before comparison; no resize/fallback')
                a.require(all(final[index]==unit for index,unit in enumerate(plan._final_units)) and final[returned]==0,
                  'exact borrowed normalized DOS UTF16 units; no materialized string')
                self.call('CloseHandle',self._image_file)
                a.require(self._image_file is None,'observed exact original image handle close')
                self._image_check_complete=True
            finally:self._image_check_active=False
        def create_suspended(self, argv, cwd, creation_plan, *, permit, case):
            # Retarget the removed existing8192 environment selection to the
            # COMPLETE bounded64-operation creation group BEFORE any pipes,
            # command/attribute/image/ABI allocations or CRT setup calls.
            creation=self.owner.before('control',operation_attempts=64,metadata_bytes=8192)
            a.require(c.sizeof(c.c_wchar)==2,'Windows UTF16 ABI mandatory')
            command=creation_plan.command(self.owner,permit,case,argv,cwd)
            a.require(self.available_ram()>=8589934592,'fresh8GiB parent actor birth floor')
            a.require(self.job is not None and self.process is None, 'fresh owned Job precedes creation')
            # Qualified provider._pipes has exactly11 CRT setup attempts:
            # open1, inheritable1, pipe2, inheritable4, get_osfhandle3.
            # The positively selected complete source group counts all11;
            # native Win32 reservations remain independently selected.
            handles = self._pipes()
            size = c.c_size_t()
            self.call('InitializeProcThreadAttributeList', None, 2, 0, c.byref(size), allow_false=True)
            a.require(0 < size.value <= 1024**2, 'bounded Job/handle attribute allocation')
            a.require(size.value<=4096,'registered4096 creation attribute allocation ceiling before allocation')
            self.attribute_buffer = c.create_string_buffer(size.value)
            self.attribute_pointer = c.cast(self.attribute_buffer, c.c_void_p)
            self.call('InitializeProcThreadAttributeList', self.attribute_pointer, 2, 0, c.byref(size))
            self.attributes_initialized = True
            job_array = (c.c_void_p * 1)(self.job)
            handle_array = (c.c_void_p * 3)(*handles)
            self.attribute_arrays = (job_array, handle_array)
            self.call('UpdateProcThreadAttribute', self.attribute_pointer, 0, provider.JOB_LIST_ATTRIBUTE,
                      c.cast(job_array, c.c_void_p), c.sizeof(job_array), None, None)
            self.call('UpdateProcThreadAttribute', self.attribute_pointer, 0, provider.HANDLE_LIST_ATTRIBUTE,
                      c.cast(handle_array, c.c_void_p), c.sizeof(handle_array), None, None)
            startup = provider.StartupInfoEx()
            startup.startup.cb = c.sizeof(startup)
            startup.startup.flags = 0x100
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
            startup.attributes = self.attribute_pointer
            pi = provider.ProcessInformation()
            flags = provider.CREATE_SUSPENDED | provider.CREATE_NO_WINDOW | provider.EXTENDED_STARTUPINFO_PRESENT | provider.CREATE_UNICODE_ENVIRONMENT
            try:
                self.call('CreateProcessW', argv[0], command, None, None, True, flags,
                          c.cast(creation_plan._environment_pointer, c.c_void_p), creation_plan._cwd, c.byref(startup), c.byref(pi))
                # This line is reached only after the actual primitive returned
                # and its outer admission return guard completed. The mutable
                # command is no longer an input; release before image allocation.
                command=None
                # Birth Job membership is atomic with CreateProcessW. If this
                # wrapper dies before these assignments, KILL_ON_JOB_CLOSE stops
                # the suspended member through the already owned Job handle.
                self.pi, self.process, self.thread = pi, pi.process, pi.thread
                inside = c.c_int()
                self.call('IsProcessInJob', self.process, self.job, c.byref(inside))
                a.require(inside.value == 1, 'exact new suspended process belongs to original Job')
                birth, exit_time, kernel, user = provider.FileTime(), provider.FileTime(), provider.FileTime(), provider.FileTime()
                self.call('GetProcessTimes', self.process, c.byref(birth), c.byref(exit_time), c.byref(kernel), c.byref(user))
                capacity=creation_plan._runtime_units+1
                a.require(1<capacity<=512,'registered exact expected UTF16 image capacity and original512upper')
                image=c.create_unicode_buffer(capacity)
                image_size=c.c_uint32(capacity)
                self.call('QueryFullProcessImageNameW',self.process,0,image,c.byref(image_size))
                # A mismatched return length refuses BEFORE materialization,
                # including unexpected long/short/truncated replies.
                a.require(image_size.value==creation_plan._runtime_units,'actual exact retained image UTF16 length')
                observed_image=image.value
                a.require(observed_image==creation_plan._runtime
                  and self.call('GetProcessId',self.process)==pi.pid
                  and self._creation_plan is creation_plan,
                  'retained exact supervisor image/PID')
                self.check_image_file()
                self.identity={'pid':int(pi.pid),'creation_filetime_100ns':provider.filetime(birth),
                  'image_path':observed_image,'argv':list(argv)}
                a.require(self.identity['creation_filetime_100ns'] > 0, 'actual retained supervisor birth')
                return dict(self.identity)
            finally:
                first=None
                if self.attributes_initialized:
                    try:self.call('DeleteProcThreadAttributeList',self.attribute_pointer,allow_false=True)
                    except BaseException as error:first=error
                try:self.close_descriptors(include_reads=False)
                except BaseException as error:
                    if first is None:first=error
                    else:first.add_note('owned creation descriptor cleanup: '+type(error).__name__)
                if first is not None:raise first
                creation.finish()
        def stop(self):
            with self._stop_lock:
                if self._stop_started:return
                self._stop_started=True;self._stop_started_at=self.owner._clock()
            # Never hold the arbitration lock during native work. A second
            # caller can observe selection and return while the first call
            # remains unresolved; it cannot infer termination completion.
            try:
                old=self.cleanup_mode()
                # A log refusal may race with the observer marking failure.
                # Termination always selects the existing original bounded
                # cleanup path before dispatch; selection is never retried.
                self._cleanup_local.active=True
                try:super().stop()
                finally:self._cleanup_local.active=old
            except BaseException as error:
                self._stop_error=error;raise
            finally:
                self._stop_wrapper_finished=True
                self._stop_returned_at=self.owner._clock()
        def cleanup_mode(self):return getattr(self._cleanup_local,'active',False)
        def _close_descriptor_admitted(self,fd,slot_index):
            # This primitive can run only inside the exact selected batch.
            a.require(self._descriptor_batch_selected,'original complete descriptor group required')
            self.owner.cleanup_after() if self.cleanup_mode() else self.owner._tick()
            a.require(not self._observation.descriptor_attempt_mask&(1<<slot_index),'original descriptor attempt never retried')
            a.require(self._observation.descriptor_attempts<5,'fixed whole-batch descriptor attempt ceiling')
            self._descriptor_selected=True;self._descriptor_entered=False
            self._descriptor_returned=False;self._descriptor_error=None;self._descriptor_afterguard_finished=False
            self._observation.descriptor_attempts+=1;self._descriptor_entered=True
            self._observation.descriptor_attempt_mask|=1<<slot_index
            try:os.close(fd)
            except BaseException as error:
                self._descriptor_error=error;self._observation_refs[4+slot_index]=error
                self._observation.descriptor_error_mask|=1<<slot_index
                raise
            self._descriptor_returned=True;self._observation.descriptor_return_count+=1
            self._observation.descriptor_return_mask|=1<<slot_index
            # An observed return clears exact ownership before a late guard.
            if fd in self.read_fds:self.read_fds.remove(fd)
            elif fd in self.write_fds:self.write_fds.remove(fd)
            elif fd==self.stdin_fd:self.stdin_fd=None
            self.owner.cleanup_after() if self.cleanup_mode() else self.owner._tick()
            self._descriptor_afterguard_finished=True
            self._observation.descriptor_afterguard_mask|=1<<slot_index
        def close_descriptors(self,include_reads=True):
            count=0
            for index in range(5):
                if index<2:
                    if not include_reads:continue
                    fd=int(self.read_fds._values[index])
                elif index<4:fd=int(self.write_fds._values[index-2])
                else:fd=self.stdin_fd
                if fd is not None and fd>=0 and not self._observation.descriptor_attempt_mask&(1<<index):count+=1
            if not count:return
            def group():
                self._descriptor_batch_started=True;self._descriptor_batch_selected=True
                self._observation.descriptor_selected_count+=count
                first=None
                for index in range(5):
                    if index<2:
                        if not include_reads:continue
                        fd=int(self.read_fds._values[index])
                    elif index<4:fd=int(self.write_fds._values[index-2])
                    else:fd=self.stdin_fd
                    if fd is None or fd<0 or self._observation.descriptor_attempt_mask&(1<<index):continue
                    try:self._close_descriptor_admitted(fd,index)
                    except BaseException as error:
                        self._observation_refs[4+index]=error
                        self._observation.descriptor_error_mask|=1<<index
                        if first is None:first=error
                if first is not None:raise first
            # ONE source group reserves the COMPLETE <=5 fixed CRT close recipe,
            # before its first dispatch. Actual FFI attempt/return counts remain
            # separate; no Native-host debit, retry, deadline or cap is added.
            if self.cleanup_mode():
                a.require(not self._observation.descriptor_tail_selected,'one original final-tail descriptor group')
                def selected_group():
                    self._observation.descriptor_tail_selected=1
                    return group()
                return self.owner.tail_run(1,selected_group)
            # All mutable descriptor state is in the positively selected fixed
            # frame/slots; this bounded consumer creates no per-FD evidence
            # container. Normal work still selects every actual CRT call.
            return self.owner.run('control',{'operation_attempts':count,'metadata_bytes':0},group)
        def close(self):
            # Changed source-owned close consumer: each failed admission stops
            # dispatch for that resource. Original live handles/descriptors
            # remain strongly referenced; a later lease release requires full
            # observed closure. Failure iteration never grants fresh counters.
            first=None
            def attempt(action):
                nonlocal first
                try:action()
                except BaseException as error:
                    if first is None:first=error
            if self.attributes_initialized:
                attempt(lambda:self.call('DeleteProcThreadAttributeList',self.attribute_pointer,allow_false=True))
            for name in ('thread','process','job','_image_file'):
                handle=getattr(self,name,None)
                if handle is not None:attempt(lambda:self.call('CloseHandle',handle))
            attempt(lambda:self.close_descriptors(include_reads=True))
            self.closed=first is None and self.closure_observation()['all_owned_handles_closed']
            if first is not None:raise first
            a.require(self.closed,'source-owned handle/descriptor closure incomplete')
        def closure_observation(self):
            if not hasattr(self,'_owned_rss_handles'):
                return {'job_created':False,'job_empty_observed':False,'process_created':False,'process_terminal_observed':False,
                  'all_owned_handles_closed':False,'closed_flag':False,'image_file_owned':getattr(self,'_image_file',None) is not None,
                  'image_check_complete':getattr(self,'_image_check_complete',False)}
            return {'job_created':self._job_created,'job_empty_observed':self._job_empty_observed,
              'process_created':self._process_created,'process_terminal_observed':self._process_terminal_observed,
              'all_owned_handles_closed':all(getattr(self,k,None) is None for k in ('job','process','thread','attribute_pointer','_image_file'))
                and not self.attributes_initialized and not self.read_fds and not self.write_fds and self.stdin_fd is None
                and all(h is None for h in self._owned_rss_handles),
              'closed_flag':self.closed,'image_file_owned':self._image_file is not None,'image_check_complete':self._image_check_complete}
        def verify_job_empty(self):
            # One source-owned context validates only exact original handles,
            # the registered class8 buffer/size, and zero-time retained wait.
            context=self.owner.terminal_verification(self)
            self._terminal_verification_context=context
            try:
                if self.job is not None:
                    self.call('QueryInformationJobObject',self.job,8,self._poll_job_pointer,c.sizeof(self._poll_job),None)
                    a.require(self._job_empty_observed,'owned Job not empty after stop; preserve lease')
                if self.process is not None:
                    waited=self.call('WaitForSingleObject',self.process,0,allow_false=True)
                    a.require(waited==provider.WAIT_OBJECT_0,'owned process not terminal; preserve lease')
            finally:self._terminal_verification_context=None
        def raw_job_pid_query_into(self,size,buffer,returned,capture):
            a.require(8<=size<=8+128*self.pointer_bytes,'bounded raw Job PID bytes')
            record=self.owner.before('control',operation_attempts=1)
            a.require(self._active_raw_capture is None,'one original raw query in flight')
            self._active_raw_capture=capture
            try:
                self.call('QueryInformationJobObject',self.job,3,buffer,size,c.byref(returned),allow_false=True)
                a.require(capture.captured,'native return captured before afterguard')
                record.finish()
            finally:self._active_raw_capture=None
        @control(0)
        def poll(self):
            a.require(not self._polling and not self._identity_callback_active,'source-fixed nonreentrant current observation')
            self._polling=True
            try:
                self._poll_waited=self.call('WaitForSingleObject',self.process,0,allow_false=True)
                a.require(self._poll_waited in (provider.WAIT_OBJECT_0,provider.WAIT_TIMEOUT),'owned process wait failed')
                self.call('QueryInformationJobObject',self.job,8,self._poll_job_pointer,c.sizeof(self._poll_job),None)
                if self._poll_waited==provider.WAIT_OBJECT_0:self.call('GetExitCodeProcess',self.process,c.byref(self._poll_exit))
                a.require(self.available_ram()>=2*1024**3,'original-owner ongoing2GiB CPU RAM floor')
                self._sample_own_memory();total=int(self._own_mem.working)
                a.require(int(self._own_mem.peak_working)<=128*1024**2,'controller RSS peak exceeds128MiB')
                a.require(int(self._own_mem.peak_pagefile)<=192*1024**2,'controller lifetime peak commitment exceeds192MiB')
                view=self.job_pids();self._initial_count=len(view)
                # Fixed independent slots protect outer iteration when a later
                # vanished-member requery invalidates the generation-bound view.
                for index in range(self._initial_count):self._initial_members[index]=view[index]
                for index in range(self._initial_count):
                    pid=int(self._initial_members[index])
                    slot=0
                    while slot<max_members and self._owned_rss_handles[slot] is not None:slot+=1
                    if slot==max_members:slot=None
                    a.require(slot is not None,'bounded original RSS handle slots exhausted before OpenProcess')
                    self._rss_open_slot=slot
                    try:handle=self.call('OpenProcess',0x1000|0x100000,False,pid,allow_false=True)
                    finally:self._rss_open_slot=None
                    if not handle:
                        a.require(not self.job_contains(pid),'unknown current owned member RSS; refuse');continue
                    primary=None
                    try:
                        self._inside.value=0
                        self.call('IsProcessInJob',handle,self.job,c.byref(self._inside))
                        a.require(self._inside.value==1 and self.call('GetProcessId',handle)==pid,'read-only RSS handle is same actual Job member')
                        available=self.call('K32GetProcessMemoryInfo',handle,c.byref(self._member_mem),c.sizeof(self._member_mem),allow_false=True)
                        if not available:
                            a.require(self.call('WaitForSingleObject',handle,0,allow_false=True)==0,'unknown live-member memory measurement');continue
                        total+=int(self._member_mem.working)
                    except BaseException as error:primary=error;raise
                    finally:
                        try:self.call('CloseHandle',handle)
                        except BaseException as close_error:
                            if primary is None:raise
                            primary.add_note('deferred original RSS close retained: '+type(close_error).__name__)
                self.rss_samples+=1;sample=time.monotonic()
                if self.last_rss_sample is not None:self.max_rss_sample_gap=max(self.max_rss_sample_gap,sample-self.last_rss_sample)
                self.last_rss_sample=sample;self.max_observed_rss=max(total,self.max_observed_rss)
                a.require(total<=512*1024**2,'observed whole fixture RSS exceeds512MiB')
                a.require(self._poll_job.io.ReadTransferCount<=a.BOUNDS['read_bytes'] and self._poll_job.io.WriteTransferCount<=a.BOUNDS['write_bytes'],'native Job IO cap')
                if self._poll_waited==provider.WAIT_OBJECT_0 and self._poll_job.basic.active==0:
                    if self._terminal_result is None:
                        terminal=self.owner.before('control',operation_attempts=1,metadata_bytes=8192)
                        self._terminal_result={key:self._poll_view[key] for key in _POLL_KEYS}
                        self._terminal_result['job_cumulative_io']={key:self._poll_io_view[key] for key in _IO_KEYS}
                        self._terminal_result['job_census_evidence']=self.census.snapshot_evidence(self)
                        terminal.finish()
                    return self._terminal_result
                return self._poll_view
            finally:self._polling=False
    # Descriptor-backed properties use fixed storage; publication keeps every
    # pre-existing field name and adds no unbounded per-call journal.
    for index,name in enumerate(_OBSERVATION_FLAGS):setattr(FixtureBackend,name,observation_flag(index))
    for index,name in enumerate(_OBSERVATION_REFS):setattr(FixtureBackend,name,observation_reference(index))
    FixtureBackend._native_windows_error=observation_optional_scalar('windows_error','_ffi_returned')
    FixtureBackend._stop_windows_error=observation_optional_scalar('stop_windows_error','_stop_ffi_returned')
    FixtureBackend._stop_started_at=observation_optional_scalar('stop_started_at','_stop_started')
    FixtureBackend._stop_returned_at=observation_optional_scalar('stop_returned_at','_stop_wrapper_finished')
    construction.finish()
    return FixtureBackend
