"""Selected bounded native collector adapter; no OS work at module import.

Native creation is stage/context/registration gated before ctypes or DLL work.
The CPU factory exercises identical prefix validation with explicit inert APIs.
Actual Windows observer completeness/lifetime still requires a reviewed run.
"""
from pathlib import Path
import hashlib

ABI={'version':1,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,
     'row_bytes':2600,'snapshot_bytes':72,'architecture':'x64'}
STATUS_UNSET=-2147483648
# Native callback code/BSS and API handles remain resident for process lifetime.
# No adapter path unloads the collector after unregister/drain.
_PROCESS_LIFETIME_TRANSPORTS=[]
SNAPSHOT_FIELDS=('abi_version','row_bytes','snapshot_bytes','row_capacity','full_capacity','base_capacity',
 'phase','faults','attempted','reserved','published_prefix','inflight','cookie_live','register_status','unregister_status','drained','complete','dropped')
ROW_FIELDS={'sequence','base_address','reason','image_size','full_units','base_units','flags','ready','full_name','base_name'}

def require(value,message):
    if not value:raise ValueError(message)

def _ordinary_snapshot(row,*,closed=False):
    require(type(row) is dict and set(row)==set(SNAPSHOT_FIELDS),'exact bounded collector snapshot ABI')
    require(all(type(value) is int and (-2**31<=value<2**31 if key in ('register_status','unregister_status') else 0<=value<2**32) for key,value in row.items()),'typed fixed width collector snapshot')
    require({k:row[k] for k in ('abi_version','row_bytes','snapshot_bytes','row_capacity','full_capacity','base_capacity')}=={'abi_version':1,'row_bytes':2600,'snapshot_bytes':72,'row_capacity':4096,'full_capacity':1024,'base_capacity':256},'selected collector ABI version and capacities')
    require(row['faults']==row['dropped']==row['inflight']==0 and row['attempted']==row['reserved']==row['published_prefix']<=4096,'lossless published collector prefix; no overflow truncation inflight or dropped rows')
    require(row['register_status']==0,'successful original native notification registration')
    if closed:require(row['phase']==5 and row['cookie_live']==0 and row['unregister_status']==0 and row['drained']==row['complete']==1,'final stopped drained collector observation')
    else:require(row['phase']==2 and row['cookie_live']==1 and row['unregister_status']==STATUS_UNSET and row['drained']==row['complete']==0,'active selected collector observation')
    return dict(row)

def _u16(value,units,capacity):
    require(type(value) is list and type(units) is int and 0<units<capacity and len(value)==units and all(type(v) is int and 0<=v<65536 for v in value),'exact nontruncated counted UTF16 observer path')
    raw=b''.join(v.to_bytes(2,'little') for v in value)
    text=raw.decode('utf-16-le','strict');require('\0' not in text,'embedded NUL observer path refused')
    return text

def _observer_capability_factory():
    token=object()
    def get():return token
    def check(candidate):require(candidate is token,'private CPU collector adapter capability')
    return get,check
_CPU_capability,_check_CPU_capability=_observer_capability_factory()

class NativeLibraryObserver:
    def __init__(self,context,contract,transport,*,fixture_only,check,owner=None,_capability=None):
        if fixture_only:_check_CPU_capability(_capability)
        else:contract.stage_native_gate()
        self.context=context;self.contract=contract;self.value=contract.context_value(context)
        require(type(fixture_only) is bool and self.value.get('fixture_only',False) is fixture_only,'original observer context mode')
        self.fixture_only=fixture_only;self.native_PASS=False;self.native_authority_issued=False
        self.transport=transport;self.check=check;self.started=False;self.closed=False;self.failed=False
        self.prefix=[];self.raw_rows=[];self.final_snapshot=None;self.birth=None;self.owner=owner
        self.registration=self.value['runtime']['library_event_observer']
        require(self.registration['abi']==ABI,'exact selected observer ABI registration')
        self._binding=transport.abi_binding()
        expected=self.registration['binary'];require(type(self._binding) is dict and set(self._binding)=={'path','descriptor','abi','dll_handle'} and self._binding['path']==expected['path'] and self._binding['descriptor']==expected['descriptor'] and self._binding['abi']==ABI and type(self._binding['dll_handle']) is int and self._binding['dll_handle']>0,'actual selected collector binary and ABI binding')
    def abi_binding(self):
        return self.contract.parse(self.contract.canonical(self._binding))
    def _check(self):
        self.check();require(self.contract.context_value(self.context)==self.value,'original observer context cannot be rebased')
        require(self.transport.abi_binding()==self._binding,'original collector handle/path/ABI cannot change')
    def _path(self,raw):
        # Only exact qualified identities and the one reviewed interpreter alias
        # may normalize. Original counted UTF16 rows remain in raw_rows.
        files=self.value['runtime']['files'];matches=[p for p in files if p.casefold()==raw.casefold()]
        if not matches:
            layout=self.value['runtime']['stdlib_layout'];logical=Path(layout['logical_base_prefix'])
            path=Path(raw)
            if path.is_relative_to(logical):
                physical=str(Path(layout['physical_base_prefix'])/path.relative_to(logical))
                matches=[p for p in files if p.casefold()==physical.casefold()]
        require(len(matches)==1,'one original qualified observed library identity')
        return matches[0]
    def normalize_library_path(self,raw):
        self._check();require(type(raw) is str and raw,'ordinary observed library path');return self._path(raw)
    def bootstrap(self,stage):
        self._check();require(type(stage) is str and 0<len(stage)<=128,'bounded bootstrap observation stage')
        observed=self.transport.self_identity()
        require(type(observed) is dict and set(observed)=={'pid','creation_filetime_100ns'} and all(type(v) is int and 0<v<2**64 for v in observed.values()),'actual fresh current process birth observation')
        if self.birth is None:self.birth=dict(observed)
        require(observed==self.birth,'same current observer process birth')
        # The selected fresh bootstrap document is external evidence, never a
        # new claim about executing namespace or universal foreign cookies.
        self.transport.verify_preloaded_bootstrap(self.value)
        self.transport.verify_owned_birth(self.value,self.owner,self.birth)
        return {'kind':'VERIFIED_CURRENT_LIBRARY_BOOTSTRAP_PROVENANCE','stage':stage,
            'fresh_process_identity_verified':True,'provider_source_verified':True,
            'preloaded_sources_verified':True,'module_event_scope_provenance_verified':True}
    def begin(self):
        self._check();require(not self.started and not self.closed,'collector scope starts exactly once')
        try:
            status=self.transport.start();require(type(status) is int and status==0,'native notification scope start succeeded')
            self.started=True;_ordinary_snapshot(self.transport.snapshot())
            return {'kind':'OBSERVED_PROCESS_LIBRARY_EVENT_SCOPE_START','complete':True,'ordered_scope_log_started':True}
        except BaseException:self.failed=True;raise
    def events(self):
        self._check();require(self.started,'collector events require begun scope')
        try:
            before=_ordinary_snapshot(self.transport.snapshot(),closed=self.closed)
            rows=self.transport.copy_rows(before['published_prefix'])
            after=_ordinary_snapshot(self.transport.snapshot(),closed=self.closed)
            require(before==after,'stable collector snapshot across counted row copy')
            require(type(rows) is list and len(rows)==before['published_prefix'],'complete exact copied event population')
            parsed=[];raw_rows=[]
            for i,row in enumerate(rows):
                require(type(row) is dict and set(row)==ROW_FIELDS and all(type(row[k]) is int for k in ('sequence','base_address','reason','image_size','full_units','base_units','flags','ready')),'ordinary exact native event row')
                require(row['sequence']==i+1 and 0<row['base_address']<2**64 and row['reason'] in (1,2) and 0<row['image_size']<2**32 and row['flags']==0 and row['ready']==1,'ordered valid native load/unload row')
                full=_u16(row['full_name'],row['full_units'],1024);base=_u16(row['base_name'],row['base_units'],256)
                require(Path(full).name.casefold()==base.casefold(),'native full/base library name join')
                parsed.append({'sequence':i+1,'event':'load' if row['reason']==1 else 'unload','path':self._path(full),'module_handle':row['base_address']})
                raw_rows.append(self.contract.parse(self.contract.canonical(row)))
            require(parsed[:len(self.prefix)]==self.prefix and raw_rows[:len(self.raw_rows)]==self.raw_rows,'append-only immutable collector event prefix')
            self.prefix=parsed;self.raw_rows=raw_rows
            if self.closed:self.final_snapshot=after
            return {'kind':'OBSERVED_PROCESS_LIBRARY_EVENTS','complete':True,'events':self.contract.parse(self.contract.canonical(parsed))}
        except BaseException:self.failed=True;raise
    def close(self):
        require(not self.closed,'collector cleanup exactly once')
        primary=None;cleanup=None;stopped=None
        try:self._check()
        except BaseException as failure:primary=failure;self.failed=True
        # Unregister/drain the original captured transport even if freshness or
        # ABI-binding validation failed. Never rescue the completion claim.
        try:
            status,stopped=self.transport.stop();require(type(status) is int and status==0,'successful native notification unregister and drain')
            _ordinary_snapshot(stopped,closed=True);self.closed=True
        except BaseException as failure:cleanup=failure;self.failed=True
        if primary is not None:
            if cleanup is not None:raise primary from cleanup
            raise primary
        if cleanup is not None:raise cleanup
        try:
            self.events();require(not self.failed,'poisoned collector cannot report complete cleanup')
            return {'kind':'OBSERVED_PROCESS_LIBRARY_EVENT_CLEANUP','complete':True,'closed':True}
        except BaseException:self.failed=True;raise

class _Win32Transport:
    def __init__(self,context,contract):
        contract.stage_native_gate();value=contract.context_value(context)
        require(value.get('fixture_only',False) is False,'CPU context cannot load native observer')
        self.C=contract;self.value=value;self.registration=value['runtime']['library_event_observer']
        binary=self.registration['binary'];contract.file_descriptor(binary['path'],binary['descriptor'])
        import sys
        require(all(name in sys.modules for name in ('ctypes','ctypes.wintypes','_ctypes')),'selected ctypes namespaces must be qualified preloaded bootstrap before observer loading')
        import ctypes
        self.ctypes=ctypes
        class Row(ctypes.Structure):
            _fields_=[('sequence',ctypes.c_uint64),('base_address',ctypes.c_uint64)]+[(n,ctypes.c_uint32) for n in ('reason','image_size','full_units','base_units','flags','ready')]+[('full_name',ctypes.c_uint16*1024),('base_name',ctypes.c_uint16*256)]
        class Snapshot(ctypes.Structure):
            _fields_=[(n,ctypes.c_int32 if n in ('register_status','unregister_status') else ctypes.c_uint32) for n in SNAPSHOT_FIELDS]
        require(ctypes.sizeof(Row)==2600 and ctypes.sizeof(Snapshot)==72 and ctypes.sizeof(ctypes.c_void_p)==8,'actual x64 native ctypes ABI sizes')
        self.Row=Row;self.Snapshot=Snapshot;self.dll=ctypes.WinDLL(binary['path'],use_last_error=True,winmode=0x800)
        self.dll.plobs_abi_version.argtypes=[];self.dll.plobs_abi_version.restype=ctypes.c_uint32
        require(self.dll.plobs_abi_version()==1,'actual loaded collector ABI version')
        self.dll.plobs_start.argtypes=[ctypes.c_void_p,ctypes.c_void_p];self.dll.plobs_start.restype=ctypes.c_int32
        self.dll.plobs_snapshot.argtypes=[ctypes.POINTER(Snapshot),ctypes.c_uint32];self.dll.plobs_snapshot.restype=ctypes.c_int32
        self.dll.plobs_copy_rows.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(Row),ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)];self.dll.plobs_copy_rows.restype=ctypes.c_int32
        self.dll.plobs_stop.argtypes=[ctypes.POINTER(Snapshot),ctypes.c_uint32];self.dll.plobs_stop.restype=ctypes.c_int32
        # System images must already be selected/byte qualified. Loading by an
        # exact System32 path avoids an unregistered name-based DLL resolution.
        for path in (r'C:\Windows\System32\ntdll.dll',r'C:\Windows\System32\kernel32.dll'):
            require(path in value['runtime']['files'],'system observer API binary explicitly qualified')
            contract.file_descriptor(path,value['runtime']['files'][path])
        self.ntdll=ctypes.WinDLL(r'C:\Windows\System32\ntdll.dll',use_last_error=True,winmode=0x800)
        self.kernel=ctypes.WinDLL(r'C:\Windows\System32\kernel32.dll',use_last_error=True,winmode=0x800)
    def abi_binding(self):
        return {'path':self.registration['binary']['path'],'descriptor':self.registration['binary']['descriptor'],'abi':dict(ABI),'dll_handle':int(self.dll._handle)}
    def start(self):
        c=self.ctypes
        return int(self.dll.plobs_start(c.cast(self.ntdll.LdrRegisterDllNotification,c.c_void_p),c.cast(self.ntdll.LdrUnregisterDllNotification,c.c_void_p)))
    def _snapshot_dict(self,s):return {n:int(getattr(s,n)) for n in SNAPSHOT_FIELDS}
    def snapshot(self):
        s=self.Snapshot();require(self.dll.plobs_snapshot(self.ctypes.byref(s),72)==0,'native collector snapshot return');return self._snapshot_dict(s)
    def copy_rows(self,count):
        c=self.ctypes;array=(self.Row*max(1,count))();copied=c.c_uint32()
        require(self.dll.plobs_copy_rows(0,count,array,c.sizeof(array),c.byref(copied))==0 and copied.value==count,'native collector exact row copy')
        rows=[]
        for x in array[:count]:
            require(x.full_units<1024 and x.base_units<256,'native string lengths fit copied row')
            row={n:int(getattr(x,n)) for n in ('sequence','base_address','reason','image_size','full_units','base_units','flags','ready')}
            row.update(full_name=[int(v) for v in x.full_name[:x.full_units]],base_name=[int(v) for v in x.base_name[:x.base_units]]);rows.append(row)
        return rows
    def stop(self):
        s=self.Snapshot();status=int(self.dll.plobs_stop(self.ctypes.byref(s),72));return status,self._snapshot_dict(s)
    def self_identity(self):
        c=self.ctypes
        class FileTime(c.Structure):_fields_=[('low',c.c_uint32),('high',c.c_uint32)]
        k=self.kernel;k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=c.c_void_p
        k.GetCurrentProcessId.argtypes=[];k.GetCurrentProcessId.restype=c.c_uint32
        k.GetProcessTimes.argtypes=[c.c_void_p]+[c.POINTER(FileTime)]*4;k.GetProcessTimes.restype=c.c_int
        created=FileTime();exited=FileTime();kernel=FileTime();user=FileTime()
        require(k.GetProcessTimes(k.GetCurrentProcess(),c.byref(created),c.byref(exited),c.byref(kernel),c.byref(user)),'actual current observer process FILETIME query')
        return {'pid':int(k.GetCurrentProcessId()),'creation_filetime_100ns':(int(created.high)<<32)|int(created.low)}
    def verify_preloaded_bootstrap(self,value):
        import sys
        policy=value['runtime']['cache_policy'];selected=policy['preloaded_bootstrap_modules']
        for name in ('ctypes','ctypes.wintypes','_ctypes'):
            module=sys.modules.get(name);row=selected.get(name)
            require(module is not None and type(row) is dict,'explicit qualified preloaded observer bootstrap module')
            actual=self.C.qualified_path(module.__file__,layout=value['runtime']['stdlib_layout'])
            require(str(actual)==row['path'] and row['path'] in value['runtime']['files'],'actual selected observer bootstrap module origin')
            self.C.file_descriptor(row['path'],value['runtime']['files'][row['path']],layout=value['runtime']['stdlib_layout'])


    def verify_owned_birth(self,value,owner,birth):
        require(type(owner) is dict and owner.get('config_sha256')==value['anchors']['expected_config_file'],'original observer ownership/config receipt')
        # Source-extract the unchanged read-only Job membership function. The
        # entire mechanical worker/model dispatcher is not imported here.
        import ast,os
        path=Path(value['config_path']).parent.parent/'native_worker.py'
        selected=value['config']['source_files'][str(path)];self.C.file_descriptor(path,selected)
        raw=self.C.read_bytes(path);require(hashlib.sha256(raw).hexdigest()==selected['sha256'],'original membership function source bytes')
        tree=ast.parse(raw,str(path));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='in_owned_job')
        namespace={'os':os,'ctypes':self.ctypes,'require':require}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        require(namespace['in_owned_job'](owner['job_name']) is True,'current observer process belongs to original owned Job')
        require(self.self_identity()==birth,'current observer FILETIME remains exact')
        c=self.ctypes;k=self.kernel
        k.OpenJobObjectW.argtypes=[c.c_uint32,c.c_int,c.c_wchar_p];k.OpenJobObjectW.restype=c.c_void_p
        k.QueryInformationJobObject.argtypes=[c.c_void_p,c.c_int,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32)];k.QueryInformationJobObject.restype=c.c_int
        k.CloseHandle.argtypes=[c.c_void_p];k.CloseHandle.restype=c.c_int
        handle=k.OpenJobObjectW(4,0,owner['job_name']);require(handle,'original named Job query handle')
        try:
            class Pids(c.Structure):_fields_=[('assigned',c.c_uint32),('count',c.c_uint32),('pids',c.c_size_t*256)]
            pids=Pids();returned=c.c_uint32()
            require(k.QueryInformationJobObject(handle,3,c.byref(pids),c.sizeof(pids),c.byref(returned)),'complete bounded original Job PID query')
            require(0<pids.count<=256 and pids.assigned==pids.count,'complete assigned Job PID roster')
            actual=[int(pid) for pid in pids.pids[:pids.count]]
            require(len(set(actual))==len(actual) and birth['pid'] in actual and owner['worker_pid'] in actual and owner['supervisor_pid'] not in actual,'current observer and original launcher in same bounded Job')
        finally:require(k.CloseHandle(handle),'original observer Job query handle closed')
        # The launcher/root and supervisor receipt remain bound here; complete
        # executing parent/FILETIME topology is checked before framework work
        # by the original binding and again by the independent terminal audit.
        require(type(owner.get('worker_pid')) is int and owner['worker_pid']>0 and type(owner.get('supervisor_pid')) is int and owner['supervisor_pid']>0 and owner['worker_pid']!=owner['supervisor_pid'],'typed original observer launcher/supervisor receipt')

def make_native_observer(context,contract,owner):
    contract.stage_native_gate();value=contract.context_value(context)
    require(value.get('fixture_only',False) is False,'CPU context has no native collector authority')
    expected=value['runtime']['library_event_observer']['adapter_source']
    actual=globals().get('__compiled_source_descriptor__')
    require(str(Path(__file__).absolute())==expected['path'] and type(actual) is dict and actual.get('bytes')==expected['descriptor']['bytes'] and actual.get('sha256')==expected['descriptor']['sha256'],'actual selected adapter source compile')
    contract.file_descriptor(expected['path'],expected['descriptor'])
    transport=_Win32Transport(context,contract)
    _PROCESS_LIFETIME_TRANSPORTS.append(transport)
    observer=NativeLibraryObserver(context,contract,transport,fixture_only=False,check=lambda:contract.verify_native_library_event_observer(observer),owner=owner)
    # Actual selected module object is supplied by trusted main; factory globals
    # alone cannot rediscover a separately compiled equivalent module instance.
    return observer

def _CPU_observer_fixture(context,contract,transport,*,_capability):
    _check_CPU_capability(_capability)
    require(contract.context_value(context).get('fixture_only') is True,'explicit CPU observer fixture context')
    return NativeLibraryObserver(context,contract,transport,fixture_only=True,check=lambda:_check_CPU_capability(_capability),_capability=_capability)
