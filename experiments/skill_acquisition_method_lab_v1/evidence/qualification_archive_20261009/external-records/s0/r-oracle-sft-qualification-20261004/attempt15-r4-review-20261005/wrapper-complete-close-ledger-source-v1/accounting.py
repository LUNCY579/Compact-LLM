"""Explicit preparation and original CPU-root accounting; no authority flags.

Preparation bounds are a prospective registration for independent review,
never an amendment to the CPU60 caps. No native API/timer runs at import.
"""
import hashlib,json,math,os,threading,time,ctypes as c,traceback,sys,types
from collections.abc import Mapping
from pathlib import Path
FIELDS=('operation_attempts','read_bytes','write_bytes','identity_hash_attempts','identity_hash_bytes','metadata_bytes','wall_seconds')
ROOT_BOUNDS=(150000,268435456,1048576,2048,33554432,1048576,60)
PREPARATION_BOUNDS=(50000,134217728,4194304,2048,134217728,8388608,1800)
def require(v,m):
    if v is not True:raise RuntimeError(m)
def vector(v):
    require(type(v) is dict and set(v)==set(FIELDS),'closed all7 vector')
    require(all((type(v[k]) is int if k!='wall_seconds' else type(v[k]) in (int,float)) and math.isfinite(v[k]) and v[k]>=0 for k in FIELDS),'finite nonnegative vector')
    return tuple(v[k] for k in FIELDS)
def timer(delay,callback):
    result=threading.Timer(delay,callback);result.daemon=True;result.start();return result
class FrozenMap(Mapping):
    __slots__=('_rows',)
    def __init__(self,rows):object.__setattr__(self,'_rows',tuple(rows))
    def __setattr__(self,*args):raise RuntimeError('immutable source-owned view')
    def __getitem__(self,key):
        for name,value in self._rows:
            if name==key:return value
        raise KeyError(key)
    def __iter__(self):return (key for key,_ in self._rows)
    def __len__(self):return len(self._rows)
    def __eq__(self,other):
        return isinstance(other,Mapping) and len(other)==len(self) and all(key in other and value==other[key] for key,value in self._rows)
def frozen(value):
    if type(value) is dict:return FrozenMap((key,frozen(item)) for key,item in value.items())
    if type(value) in (list,tuple):return tuple(frozen(item) for item in value)
    require(value is None or type(value) in (str,bool,int,float),'registered immutable scalar')
    require(type(value) is not float or math.isfinite(value),'finite immutable scalar');return value
def encoded_extent(value):
    """Exact canonical ASCII byte extent without allocating encoded payload."""
    def string(text):
        require(type(text) is str,'canonical JSON string key')
        return 2+sum(2 if ch in ('"','\\','\b','\f','\n','\r','\t') else 6 if ord(ch)<32 or 127<=ord(ch)<=65535 else 12 if ord(ch)>65535 else 1 for ch in text)
    if isinstance(value,Mapping):return 2+max(0,len(value)-1)+sum(string(key)+1+encoded_extent(item) for key,item in value.items())
    if type(value) in (list,tuple):return 2+max(0,len(value)-1)+sum(encoded_extent(item) for item in value)
    if type(value) is str:return string(value)
    if value is None:return 4
    if type(value) is bool:return 4 if value else 5
    require(type(value) in (int,float) and math.isfinite(value),'finite canonical scalar')
    return len(repr(value))
class Ticket:
    __slots__=('_owner','_role','_finished')
    def __init__(self,owner,role):self._owner=owner;self._role=role;self._finished=False
    def finish(self):
        require(not self._finished,'one source return');self._finished=True
        self._owner.after(self._role)
class Owner:
    __slots__=('_selected','_partitions','_used','_roles','_origin','_deadline','_clock','_failed','_closed','_lock','_resource','_resource_active')
    def __setattr__(self,k,v):
        if hasattr(self,k):raise RuntimeError('immutable source owner fields')
        object.__setattr__(self,k,v)
    def __init__(self,selected,partitions,origin,clock,resource=None):
        self._selected=tuple(selected);self._partitions={key:tuple(value) for key,value in partitions.items()}
        self._used=[0]*7;self._roles={key:[0]*7 for key in partitions};self._origin=origin;self._deadline=origin+selected[6]
        self._clock=clock;self._failed=False;self._closed=False;self._lock=threading.RLock();self._resource=resource;self._resource_active=False
    @property
    def origin(self):return self._origin
    @property
    def deadline(self):return self._deadline
    @property
    def fatal(self):return self._failed
    @property
    def used(self):return dict(zip(FIELDS,self._used))
    @property
    def role_used(self):return {key:dict(zip(FIELDS,value)) for key,value in self._roles.items()}
    def _tick(self,allow_failed=False):
        require(not self._closed and (allow_failed or not self._failed),'same live original owner')
        if not self._origin<=self._clock()<self._deadline:
            object.__setattr__(self,'_failed',True);raise RuntimeError('original physical phase deadline expired')
    def _resources(self):
        if self._resource is None:return
        require(not self._resource_active,'no recursive resource callback')
        object.__setattr__(self,'_resource_active',True)
        try:
            # Source sampler performs three native read-only API attempts.
            require(self._used[0]+3<=self._selected[0] and self._roles['control'][0]+3<=self._partitions['control'][0],'resource sampling reserve')
            self._used[0]+=3;self._roles['control'][0]+=3
            sample=self._resource()
            require(type(sample) is dict and set(sample)=={'available_ram_bytes','process_rss_bytes','process_peak_commit_bytes'}
              and all(type(v) is int and v>=0 for v in sample.values()) and sample['available_ram_bytes']>=2147483648
              and sample['process_rss_bytes']<=134217728 and sample['process_peak_commit_bytes']<=201326592,'original metadata-owner RAM/RSS/commit gates')
        except BaseException:
            object.__setattr__(self,'_failed',True);raise
        finally:object.__setattr__(self,'_resource_active',False)
    def before(self,role='control',**counts):
        with self._lock:
            self._tick();self._resources();self._tick();require(role in self._roles and set(counts)<=set(FIELDS),'closed owner/partition unit')
            delta=[counts.get(key,0) for key in FIELDS]
            require(all(type(v) is int and v>=0 for v in delta),'positive integral operation counters')
            total=[max(self._used[i],delta[i]) if i==6 else self._used[i]+delta[i] for i in range(7)]
            local=[max(self._roles[role][i],delta[i]) if i==6 else self._roles[role][i]+delta[i] for i in range(7)]
            if any(total[i]>self._selected[i] or local[i]>self._partitions[role][i] for i in range(7)):
                object.__setattr__(self,'_failed',True);raise RuntimeError('original partition exhausted before work')
            self._used[:]=total;self._roles[role][:]=local
            return Ticket(self,role)
    def after(self,role):self._tick();self._resources();self._tick()
    def mark_failed(self):object.__setattr__(self,'_failed',True)
    def run(self,role,counts,work):
        ticket=self.before(role,**counts)
        try:result=work();ticket.finish();return result
        except BaseException:
            object.__setattr__(self,'_failed',True);raise
    def canonical(self,value,maximum=16384):
        def encode():
            raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False,
              default=lambda item:dict(item.items()) if type(item) is FrozenMap else (_ for _ in ()).throw(TypeError('unsupported JSON object'))).encode()+b'\n'
            require(len(raw)<=maximum,'canonical publication bound');return raw
        return self.run('control',{'operation_attempts':1,'metadata_bytes':maximum},encode)
    def strict(self,blob,maximum):
        def decode():
            def unique(pairs):
                value={}
                for key,item in pairs:require(key not in value,'duplicate JSON key');value[key]=item
                return value
            def finite(token):
                value=float(token);require(math.isfinite(value),'overflowed JSON scalar');return value
            return json.loads(blob,object_pairs_hook=unique,parse_float=finite,parse_constant=lambda _:require(False,'nonfinite JSON scalar'))
        require(type(blob) is bytes and len(blob)<=maximum,'bounded JSON input')
        return self.run('control',{'operation_attempts':1,'metadata_bytes':maximum},decode)
    def sha(self,blob):
        require(type(blob) is bytes,'immutable SHA input')
        return self.run('control',{'operation_attempts':1,'identity_hash_attempts':1,'identity_hash_bytes':len(blob)},lambda:hashlib.sha256(blob).hexdigest())
    def freeze(self,value,maximum=65536):
        def owned_freeze():
            require(encoded_extent(value)<=maximum,'prospective immutable source graph extent exceeds selected envelope')
            return frozen(value)
        return self.run('control',{'operation_attempts':128,'metadata_bytes':maximum},owned_freeze)
    def snapshot(self):
        return self.run('control',{'operation_attempts':1,'metadata_bytes':2048},lambda:{'selected':dict(zip(FIELDS,self._selected)),
          'used':dict(zip(FIELDS,self._used)),'partition_used':{key:dict(zip(FIELDS,value)) for key,value in self._roles.items()},
          'pending':[],'failed':[] if not self._failed else ['SOURCE_OWNER_FAILED'],'refunds':0})
class PreparationOwner(Owner):
    """Prospective bounded PRE-GRANT phase, not new CPU-root cap authority."""
    __slots__=('_failure_custody','_failure_closes','_failure_export_attempted','_failure_stream','_failure_identity','_failure_close_attempted','_failure_window_codes','_root_handoff','_prepared_failure_path')
    FAILURE=(256,65537,65536,2,65536,131072,1800)
    def __init__(self,clock=time.monotonic,resource=None):
        control=tuple(PREPARATION_BOUNDS[i]-self.FAILURE[i] if i!=6 else 1800 for i in range(7))
        super().__init__(PREPARATION_BOUNDS,{'control':control,'prep_failure':self.FAILURE},clock(),clock,resource)
        self._failure_custody=None;self._failure_closes=0;self._failure_export_attempted=False;self._failure_stream=None;self._failure_identity=None;self._failure_close_attempted=False;self._failure_window_codes=();self._root_handoff=False
        # Whole original-PREP failure subset selected BEFORE any normal work.
        # There is no new owner, root, duration or failure-phase allocation.
        self.before('prep_failure',**dict(zip(FIELDS,self.FAILURE)))
        object.__setattr__(self,'_failure_identity',OutputDescriptorIdentity())
        object.__setattr__(self,'_prepared_failure_path',str(Path(__file__).resolve().parent/'FAILED_PREPARATION.json'))
    def bind_failure_custody(self,custody):
        admission=sys.modules.get('admission');module=admission._MODULES.get('ce8_prepared_runtime_custody')
        frame=sys._getframe(1)
        require(type(module) is types.ModuleType and frame.f_code is module.RuntimeCustody.prepare.__func__.__code__
          and frame.f_globals is module.__dict__ and type(custody) is module.RuntimeCustody
          and module._LIVE.get(id(custody)) is custody and custody._pid==os.getpid()
          and self._failure_custody is None,'only exact original source custody binds PREP cleanup')
        self._tick();object.__setattr__(self,'_failure_custody',custody)
    def failure_pin_before(self,custody,row):
        require(not self._root_handoff and self.fatal and custody is self._failure_custody and (custody._failed or self.fatal),
          'only authenticated actual failed original PREP can use emergency cleanup')
        self._tick(allow_failed=True)
        require(custody is self._failure_custody and any(item is row for item in custody._rows)
          and not row['closed'] and not row.get('close_attempted',False) and self._failure_closes<240,
          'only exact original owned pin and fixed selected PREP close subset')
        object.__setattr__(self,'_failure_closes',self._failure_closes+1)
        return PreparationFailureTicket(self)
    def notify_custody_failure(self,custody,error):
        admission=sys.modules.get('admission');module=admission._MODULES.get('ce8_prepared_runtime_custody')
        frame=sys._getframe(1)
        require(type(module) is types.ModuleType and frame.f_code is module.RuntimeCustody.prepare.__func__.__code__
          and frame.f_globals is module.__dict__ and custody is self._failure_custody and custody._failed
          and isinstance(error,BaseException),'exact original failed-custody event activates cleanup-only PREP')
        self.mark_failed()
    def register_metadata_window(self):
        # Called after original owner creation; only the fixed trusted layer0
        # metadata code is registered, after its bytes/code are authenticated.
        frame=sys._getframe(1);namespace=frame.f_globals
        path=Path(__file__).resolve().parent/'metadata_entry.py'
        require(not self._root_handoff and _ENTRY_PREPARATION is not None and _ENTRY_PREPARATION[1] is self
          and namespace.get('__file__')==str(path) and frame.f_code is namespace['main'].__code__
          and type(sys.modules.get(namespace.get('__name__'))) is types.ModuleType
          and sys.modules[namespace['__name__']].__dict__ is namespace,
          'same original fixed metadata entry window')
        ticket=self.before('control',operation_attempts=32,read_bytes=1048577,identity_hash_attempts=1,
          identity_hash_bytes=1048576,metadata_bytes=1048576)
        with path.open('rb') as stream:raw=stream.read(1048577)
        require(len(raw)<=1048576,'fixed metadata entry code bound')
        code=compile(raw,str(path),'exec')
        matches=tuple(item for item in code.co_consts if type(item) is types.CodeType and item.co_name=='main')
        require(len(matches)==1 and matches[0]==frame.f_code,'actual running metadata code equals exact fixed source')
        digest=hashlib.sha256(raw).hexdigest()
        object.__setattr__(self,'_failure_window_codes',self._failure_window_codes+((frame.f_code,namespace),))
        ticket.finish()
        return {'path':str(path),'bytes':len(raw),'sha256':digest}
    def abort_owned_entry(self,error):
        frame=sys._getframe(1)
        require(_ENTRY_PREPARATION is not None and _ENTRY_PREPARATION[1] is self
          and any(frame.f_code is code and frame.f_globals is namespace for code,namespace in self._failure_window_codes)
          and isinstance(error,BaseException) and not self._closed and not self._root_handoff,
          'only exact original owned entry exception window activates failed PREP')
        self.mark_failed()
        if self._failure_custody is not None and not self._failure_custody._closed:
            try:self._failure_custody.close_all(failure_owner=self)
            except BaseException as secondary:error.add_note('original PREP pins: '+type(secondary).__name__)
        if not self._failure_export_attempted:
            try:self.failure_receipt(error)
            except BaseException as secondary:error.add_note('original PREP receipt: '+type(secondary).__name__)
    def failure_receipt(self,error):
        admission=sys.modules.get('admission');frame=sys._getframe(1)
        issuer=None if admission is None else admission._MODULES.get('ce8_prepared_issuer_entry')
        internal=frame.f_code is PreparationOwner.abort_owned_entry.__code__ and frame.f_globals is globals()
        original_issuer=type(issuer) is types.ModuleType and frame.f_code is issuer.prepare_and_issue.__code__ and frame.f_globals is issuer.__dict__
        require(not self._root_handoff and self.fatal and (internal or original_issuer) and not self._failure_export_attempted,
          'one exact actual failed original PREP receipt; no caller blob authority')
        self._tick(allow_failed=True);object.__setattr__(self,'_failure_export_attempted',True)
        primary=None;raw=readback=None
        try:
            # Direct bounded scalar snapshot avoids normal failed-owner.run.
            name=type(error).__name__
            require(name.isascii() and len(name)<=128,'bounded original PREP error name')
            state={'schema':'CE8-PRESELECTED-PREP-FAILURE-1','error_type':name,
              'selected':dict(zip(FIELDS,self._selected)),'used':dict(zip(FIELDS,self._used)),
              'partition_used':{key:dict(zip(FIELDS,value)) for key,value in self._roles.items()},
              'origin':self.origin,'deadline':self.deadline,'failed':self.fatal,'refunds':0,
              'preselected_failure_subset':dict(zip(FIELDS,self.FAILURE)),
              'actual_original_pin_close_attempts':self._failure_closes,
              'all_original_pins_closed':self._failure_custody is None or self._failure_custody._closed,
              'model_loads':0,'optimizer_updates':0}
            require(encoded_extent(state)+1<=2048,'closed two-role/seven-field PREP failure state bound')
            raw=json.dumps(state,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()+b'\n'
            self._tick(allow_failed=True)
            path=self._prepared_failure_path
            actual=open(path,'xb+',buffering=0)
            object.__setattr__(self,'_failure_stream',actual)
            self._failure_identity.capture(actual)
            self._tick(allow_failed=True)
            require(self._failure_stream.write(raw)==len(raw),'full original PREP failure write');self._tick(allow_failed=True)
            self._failure_stream.flush();self._tick(allow_failed=True)
            os.fsync(self._failure_stream.fileno());self._tick(allow_failed=True)
            self._failure_stream.seek(0);self._tick(allow_failed=True)
            readback=self._failure_stream.read(len(raw)+1);self._tick(allow_failed=True)
            require(readback==raw,'exact original PREP failure readback')
            digest=hashlib.sha256(readback).hexdigest();self._tick(allow_failed=True)
            self._tick(allow_failed=True);object.__setattr__(self,'_failure_close_attempted',True);self._failure_identity.close_attempted=True
            try:self._failure_stream.close()
            except BaseException:self._failure_identity.close_failed=True;raise
            self._failure_identity.close_returned=True;object.__setattr__(self,'_failure_stream',None);self._tick(allow_failed=True)
            return {'path':path,'bytes':len(raw),'sha256':digest}
        except BaseException as actual:primary=actual;raise
        finally:
            raw=readback=None
            if self._failure_stream is not None and not self._failure_close_attempted:
                try:
                    self._tick(allow_failed=True)
                    object.__setattr__(self,'_failure_close_attempted',True)
                    self._failure_identity.close_attempted=True
                    try:self._failure_stream.close()
                    except BaseException:self._failure_identity.close_failed=True;raise
                    self._failure_identity.close_returned=True
                    object.__setattr__(self,'_failure_stream',None);self._tick(allow_failed=True)
                except BaseException as secondary:
                    if primary is None:raise
                    primary.add_note('original PREP output unknown close: '+type(secondary).__name__)
    def attach_resources(self,sampler):
        require(type(sampler) is NativeResourceSampler and self._resource is None,'one genuine source resource sampler')
        object.__setattr__(self,'_resource',sampler);self._resources()
    def receipt(self):
        result=self.snapshot();result.update(schema='CE8-BOUNDED-PRE-GRANT-ACCOUNTING-5',classification='PROSPECTIVE_BOUND_REQUIRES_INDEPENDENT_REVIEW',
          origin=self.origin,deadline=self.deadline,work_claimed_free=False,experiment_root_debit=False)
        return result
class PreparationFailureTicket:
    __slots__=('_owner','_finished')
    def __init__(self,owner):self._owner=owner;self._finished=False
    def finish(self):
        require(not self._finished,'one original PREP cleanup return')
        self._finished=True;self._owner._tick(allow_failed=True)


_ENTRY_PREPARATION=None
def begin_entry_preparation(entry_path,entry_sha256,*,clock=time.monotonic,resource=None):
    """One same-process owner after the explicit trusted localcaller boundary.

    This registry is not a permit or proof of interpreter startup cost. The
    source-owned entry must price every subsequent admission/module load.
    """
    global _ENTRY_PREPARATION
    require(_ENTRY_PREPARATION is None,'one retained entry preparation; no second phase allocation')
    path=Path(entry_path)
    require(path==Path(__file__).resolve().parent/'bootstrap_entry.py' and path.is_absolute(),
      'exact local source entry context')
    require(type(entry_sha256) is str and len(entry_sha256)==64 and not set(entry_sha256)-set('0123456789abcdef'),
      'exact source entry SHA context')
    owner=PreparationOwner(clock=clock,resource=resource)
    frame=sys._getframe(1)
    require(frame.f_code.co_name=='prepare_entry' and frame.f_globals.get('__file__')==str(path),
      'trusted fixed layer0 entry frame before owned window')
    object.__setattr__(owner,'_failure_window_codes',tuple((frame.f_globals[name].__code__,frame.f_globals)
      for name in ('prepare_entry','main')))
    _ENTRY_PREPARATION=(os.getpid(),owner,str(path),entry_sha256)
    return owner
def entry_preparation_owner():
    require(_ENTRY_PREPARATION is not None and _ENTRY_PREPARATION[0]==os.getpid(),
      'existing same-process entry owner required; no cold/cross-process adoption')
    return _ENTRY_PREPARATION[1]
def entry_preparation_context():
    owner=entry_preparation_owner()
    return {'pid':_ENTRY_PREPARATION[0],'entry_path':_ENTRY_PREPARATION[2],
      'entry_sha256':_ENTRY_PREPARATION[3],'origin':owner.origin,'deadline':owner.deadline,
      'trusted_localcaller_startup_qualified':False,'startup_claimed_free':False}
class NativeResourceSampler:
    """Read-only source sampler; construction/calls are owned preparation work."""
    def __init__(self):
        import ctypes as c
        self.c=c;self.kernel=c.WinDLL('kernel32',use_last_error=True)
        class RAM(c.Structure):
            _fields_=[('length',c.c_uint32),('load',c.c_uint32)]+[(key,c.c_uint64) for key in ('total','available','total_page','available_page','virtual','available_virtual','extended')]
        class MEM(c.Structure):
            _fields_=[('cb',c.c_uint32),('faults',c.c_uint32)]+[(key,c.c_size_t) for key in ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        self.RAM=RAM;self.MEM=MEM
        self.kernel.GetCurrentProcess.argtypes=[];self.kernel.GetCurrentProcess.restype=c.c_void_p
        self.kernel.GlobalMemoryStatusEx.argtypes=[c.POINTER(RAM)];self.kernel.GlobalMemoryStatusEx.restype=c.c_int
        self.kernel.K32GetProcessMemoryInfo.argtypes=[c.c_void_p,c.POINTER(MEM),c.c_uint32];self.kernel.K32GetProcessMemoryInfo.restype=c.c_int
    def __call__(self):
        c=self.c;ram=self.RAM();ram.length=c.sizeof(ram);mem=self.MEM();mem.cb=c.sizeof(mem)
        require(bool(self.kernel.GlobalMemoryStatusEx(c.byref(ram))),'actual preparation RAM sample')
        require(bool(self.kernel.K32GetProcessMemoryInfo(self.kernel.GetCurrentProcess(),c.byref(mem),mem.cb)),'actual preparation RSS/commit sample')
        return {'available_ram_bytes':int(ram.available),'process_rss_bytes':int(mem.working),'process_peak_commit_bytes':int(mem.peak_pagefile)}
class OutputDescriptorIdentity:
    __slots__=('fd','handle','close_attempted','close_returned','close_failed','capture_error','capture_status')
    def __init__(self,stream=None):
        self.fd=self.handle=None;self.capture_error=None;self.capture_status='NOT_CAPTURED'
        self.close_attempted=self.close_returned=self.close_failed=False
        if stream is not None:self.capture(stream)
    def capture(self,stream):
        try:
            import msvcrt
            fd=stream.fileno()
            require(type(fd) is int and fd>=0,'valid original FileIO numeric diagnostic')
            self.fd=fd;self.capture_status='FD_CAPTURED'
            handle=msvcrt.get_osfhandle(fd)
            require(type(handle) is int and handle!=-1,'valid original underlying HANDLE diagnostic')
            self.handle=handle;self.capture_status='COMPLETE'
        except BaseException as error:
            name=type(error).__name__
            self.capture_error=name if name.isascii() and len(name)<=128 else 'UNBOUNDED_CAPTURE_ERROR_TYPE'
            self.capture_status='PARTIAL_FAILED'
            # This is a new diagnostic-only failure, separate from operational
            # stream/read/write/close errors. Preserve its bounded type/status,
            # clear dormant getter traceback locals and retain original stream.
            traceback.clear_frames(error.__traceback__)
    def value(self):
        return {'fd':self.fd,'handle':self.handle,'capture_status':self.capture_status,
          'capture_error':self.capture_error,'close_attempted':self.close_attempted,
          'close_returned':self.close_returned,'close_failed':self.close_failed,
          'terminal_status':'OBSERVED_CLOSED' if self.close_returned else 'UNKNOWN' if self.close_attempted else 'NOT_ATTEMPTED',
          'diagnostic_only':True}

_TAIL_WORK_SEAL=object()
_LIVE_BUDGETS={}
_TAIL_AUTHORITIES={}
_TAIL_AUTHORITY_SEAL=object()

class PreparedTailAuthority:
    __slots__=('_seal','_prep','_pid','_root','_accounting','_harness','_census','_custody','_record','_sidecar','_lifecycle','_snapshot','_diagnostic','_failure','_dimensions','_phase_plan','_catalogue','_sha')
    def __init__(self,*args):raise RuntimeError('only exact paid static producer registration')
    def __setattr__(self,*args):raise RuntimeError('immutable prepared producer authority')
    @classmethod
    def prepare(cls,prep,packet,custody):
        admission=sys.modules.get('admission');frame=sys._getframe(3)
        runner=sys._getframe(2);work=sys._getframe(1)
        require(type(admission) is types.ModuleType and frame.f_code is admission.PreparedStaticViews.create.__func__.__code__
          and frame.f_globals is admission.__dict__ and prep is admission._PREP_OWNER
          and type(prep) is PreparationOwner and runner.f_code is Owner.run.__code__
          and runner.f_globals is sys.modules['accounting'].__dict__
          and work.f_globals is admission.__dict__ and work.f_code in frame.f_code.co_consts,
          'only exact positive paid static PREP construction frame can register producers')
        accounting=sys.modules.get('accounting');harness=admission._MODULES.get('ce8_prepared_harness_entry')
        census=admission._MODULES.get('ce8_prepared_parent_census')
        failure=admission._MODULES.get('ce8_prepared_failure_envelope')
        require(type(accounting) is types.ModuleType and accounting.Budget is Budget
          and type(harness) is types.ModuleType and type(census) is types.ModuleType
          and packet['source_refs']['accounting.py']==admission._ENTRY_IDENTITIES['accounting_ref'],
          'actual exact already authenticated accounting/harness/census modules')
        for module,name in ((accounting,'accounting.py'),(harness,'harness.py'),(census,'reusable_census.py'),(failure,'failure_envelope.py')):
            ref=packet['source_refs'][name]
            require(module.__file__==ref['path'] and module.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==ref['sha256'],
              'registered actual prepared module/path/source SHA identity')
        custody.require_source_owner()
        records=tuple(item for item in Budget.failure_record.__code__.co_consts
          if type(item) is types.CodeType and item.co_name=='record')
        require(len(records)==1,'one exact authenticated nested failure producer')
        obj=object.__new__(cls)
        values={'_seal':_TAIL_AUTHORITY_SEAL,'_prep':prep,'_pid':os.getpid(),'_root':None,
          '_accounting':accounting,'_harness':harness,'_census':census,'_custody':custody,'_record':records[0],
          '_sidecar':harness.OriginalOwnerLogs._publish_sidecar.__code__,
          '_lifecycle':harness.OwnedLifecycle.failure_state.__code__,
          '_snapshot':census.CensusStorage.snapshot.__code__,'_diagnostic':failure.describe.__code__,'_failure':failure,
          '_dimensions':(4,1,2,128,4,8,6,7,2,2066,24,1024),
          '_phase_plan':(('graph_gate',22528),('graph_candidate',20488),('probe',8192),('piece',2066),
            ('wire',32768),('working',32768),('readback',1024),('sidecar_readback',2049),('frame_locals_dict',528)),
          '_catalogue':None,
          '_sha':tuple((module,packet['source_refs'][name]['sha256']) for module,name in
            ((accounting,'accounting.py'),(harness,'harness.py'),(census,'reusable_census.py'),(failure,'failure_envelope.py')))}
        for key,value in values.items():object.__setattr__(obj,key,value)
        _TAIL_AUTHORITIES[id(obj)]=obj;return obj
    def prepare_catalogue(self,context):
        frame=sys._getframe(3);work=sys._getframe(1);runner=sys._getframe(2)
        admission=sys.modules.get('admission')
        require(type(admission) is types.ModuleType and frame.f_code is admission.PreparedStaticViews.create.__func__.__code__
          and frame.f_globals is admission.__dict__ and runner.f_code is Owner.run.__code__
          and work.f_globals is admission.__dict__ and any(work.f_code is code for code in frame.f_code.co_consts)
          and self._root is None and self._prep is admission._PREP_OWNER and self._catalogue is None,
          'only exact positively paid static PREP code builds borrowed identity catalogue')
        require(type(context) is admission.PreparedStaticViews and context._seal is admission._STATIC_SEAL
          and admission._STATIC_CONTEXTS.get(id(context)) is context and context._custody is self._custody
          and context._root_owner is None,'same original static/custody context catalogue')
        values=[];codes=[]
        def add(value):
            if value is None or type(value) in (str,bytes,bool,int,float):
                if not any(value is old for old in values):
                    require(len(values)<2048,'bounded paid literal-object identity catalogue')
                    values.append(value)
            elif type(value) is tuple:
                require(len(value)<=240,'bounded immutable source tuple domain')
                for item in value:add(item)
        def code_constants(code):
            if any(code is old for old in codes):return
            require(len(codes)<1024,'bounded paid code-object census')
            codes.append(code)
            for value in code.co_consts:
                if type(value) is types.CodeType:code_constants(value)
                else:add(value)
        source_modules=list(self._sha)
        for module,role in ((context._modules[0],'provider'),(context._modules[1],'backend.py')):
            ref=admission.PROVIDER if role=='provider' else context.packet['source_refs'][role]
            require(type(module) is types.ModuleType and module.__file__==ref['path']
              and module.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==ref['sha256'],
              'actual already authenticated provider/backend before literal borrowing')
            if not any(module is old for old,sha in source_modules):source_modules.append((module,ref['sha256']))
        for module,sha in source_modules:
            require(module.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==sha,'same paid exact module before catalogue')
            for value in module.__dict__.values():
                if type(value) is types.FunctionType and value.__globals__ is module.__dict__:code_constants(value.__code__)
                elif type(value) is type and value.__module__==module.__name__:
                    for member in value.__dict__.values():
                        function=member.__func__ if type(member) in (classmethod,staticmethod) else member
                        if type(function) is types.FunctionType and function.__globals__ is module.__dict__:code_constants(function.__code__)
                elif value is None or type(value) in (str,bytes,bool,int,float,tuple):add(value)
        plan=context.creation_plan
        for value in (plan._runtime,plan._cwd,plan._commands,plan._log_inputs):add(value)
        # Runtime Path objects themselves are never excluded as owned output
        # containers. Only the actual paid rendered immutable names are lent.
        for case,directory,paths,names,path_strings,sidecars in plan._log_inputs:
            for value in (*names,*path_strings,*sidecars):add(value)
        object.__setattr__(self,'_catalogue',tuple(values))
    def borrowed_scalar(self,owner,value):
        require(self._root is owner and self._catalogue is not None and _TAIL_AUTHORITIES.get(id(self)) is self
          and self._pid==os.getpid(),'original source-owned retained literal catalogue')
        return any(value is original for original in self._catalogue)
    def bind(self,owner):
        require(self._catalogue is not None,'complete positively prepared actual literal catalogue before root')
        frame=sys._getframe(1)
        require(self._seal is _TAIL_AUTHORITY_SEAL and _TAIL_AUTHORITIES.get(id(self)) is self
          and frame.f_code is Budget.__init__.__code__ and frame.f_globals is self._accounting.__dict__
          and type(owner) is Budget and self._root is None and self._pid==os.getpid(),
          'one exact original constructor binds already prepared producer authority')
        object.__setattr__(self,'_root',owner)
    def require_producer(self,owner,recipe,frame):
        expected=self._record if recipe=='failure' else self._sidecar
        module=self._accounting if recipe=='failure' else self._harness
        require(type(self) is PreparedTailAuthority and _TAIL_AUTHORITIES.get(id(self)) is self
          and self._seal is _TAIL_AUTHORITY_SEAL and self._root is owner and self._pid==os.getpid()
          and frame.f_code is expected and frame.f_globals is module.__dict__
          and all(item.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==sha for item,sha in self._sha),
          'exact prepared code object/module/SHA producer identity before any allocation')

    def authorize_data(self,kind,obj=None):
        frame=sys._getframe(1);owner=self._root
        code,module={'lifecycle':(self._lifecycle,self._harness),'snapshot':(self._snapshot,self._census),
          'diagnostic':(self._diagnostic,self._failure),'accounting':(self._record,self._accounting)}[kind]
        require(type(self) is PreparedTailAuthority and _TAIL_AUTHORITIES.get(id(self)) is self
          and self._seal is _TAIL_AUTHORITY_SEAL and type(owner) is Budget and owner._tail_authority is self
          and self._pid==os.getpid() and frame.f_code is code and frame.f_globals is module.__dict__
          and self._dimensions==(4,1,2,128,4,8,6,7,2,2066,24,1024)
          and all(item.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==sha for item,sha in self._sha),
          'exact prepared original producer/field plan before graph allocation')
        scope=owner._tail_work_active
        if kind!='snapshot' or scope is not None:
            require(type(scope) is TailWorkingScope and scope._recipe=='failure' and scope._phase==0,
              'original selected build workspace before failure producer allocation')
            scope.require_live()
        else:
            # Normal terminal evidence uses the already selected parent8192
            # construction record. It keeps its original live owner/census gate.
            owner._tick()
            require(any(row is not None and row[0] is obj.backend for row in owner._owned_backends)
              and obj.backend._polling,'only original terminal snapshot source context')
        if kind=='lifecycle':
            require(type(obj) is self._harness.OwnedLifecycle and obj.budget is owner and obj.custody is self._custody
              and type(obj.count) is int and 0<=obj.count<=4 and len(obj.backends)==len(obj.roles)==len(obj.errors)==4,
              'original lifecycle four fixed slots before rows allocation')
            for index in range(obj.count):
                require(obj.roles[index]==('initial_checker','normal_exit','timeout','final_checker')[index]
                  and (obj.errors[index] is None or (type(obj.errors[index]) is str and len(obj.errors[index])<=128
                    and obj.errors[index].isascii())),'original ordered roles/one cleanup-error source domain')
                backend=obj.backends[index]
                for key in ('_native_name',):
                    value=getattr(backend,key,None)
                    require(value is None or (type(value) is str and value.isascii() and len(value)<=128),
                      'bounded original native name before lifecycle output')
                for key in ('_native_error','_descriptor_error'):
                    value=getattr(backend,key,None)
                    require(value is None or (type(value).__name__.isascii() and len(type(value).__name__)<=128),
                      'bounded original error-type name before lifecycle output')
                logs=getattr(backend,'_owned_logs',None)
                if logs is not None and getattr(logs,'case',None) is not None:
                    require(type(logs) is self._harness.OriginalOwnerLogs and logs.backend is backend
                      and logs.case==obj.roles[index] and len(logs.blocks)==len(logs.sidecars)==2,
                      'original current-case two sidecar slots before output allocation')
                    for path in logs.sidecars:require(type(path) is str and path.isascii() and len(path)<=320,
                      'registered bounded sidecar path before producer graph')
            require(type(obj.lease_status) is str and obj.lease_status.isascii() and len(obj.lease_status)<=64
              and len(str(obj.lock))<=320,'bounded original lease data before lifecycle output')
        elif kind=='snapshot':
            require(type(obj) is self._census.CensusStorage and obj.backend.owner is owner and not obj.in_query
              and len(obj.slots)==2 and type(obj.attempts) is int and 0<=obj.attempts<=2,'original fixed raw producer before allocation')
            for index in range(obj.attempts):
                slot=obj.slots[index]
                require(type(slot.size) is int and 8<=slot.size<=1032 and slot.pointer_bytes in (4,8)
                  and type(slot.capacity) is int and 0<=slot.capacity<=128 and type(slot.guard) is int and 1<=slot.guard<=128,
                  'registered raw/decoded dimensions before byte/hex/PID allocation')
                if index==0:
                    require(1<=obj.guard<=8 and slot.capacity==obj.guard and slot.size==8+obj.guard*slot.pointer_bytes,
                      'first original raw request exact registered topology; second descriptive bound128')
        elif kind=='accounting':
            require(tuple(owner._roles)==('normal_exit','timeout','native_host','logs','final_tail','control')
              and len(owner._selected)==len(owner._used)==7 and all(len(v)==7 for v in owner._roles.values()),
              'six original roles/seven fields before fresh accounting maps')
            for vector in (owner._selected,owner._used,*owner._roles.values()):
                require(all(type(value) in (int,float) and math.isfinite(value) and value>=0
                  and (value<=60 if index==6 else type(value) is int and value.bit_length()<=64)
                  for index,value in enumerate(vector)),'finite original scalar plan before map construction')
        elif kind=='diagnostic':
            require(isinstance(obj,BaseException) and type(obj.args) is tuple
              and type(getattr(obj,'__notes__',())) in (list,tuple),
              'original four-argument/eight-note projection before diagnostics allocation')
        return scope

_TAIL_FAILURE_PATHS=tuple(str(Path(__file__).resolve().parent/name)
  for name in ('FAILED_QUALIFICATION.json','FAILED_ROOT_START.json'))
class TailWorkingScope:
    """Same-source nonreentrant working partition, separate from wire storage.

    This capability enforces ownership/phase limits. It is not proof that the
    producer's Python/C allocation graph obeys those selected logical maxima.
    """
    __slots__=('_seal','_owner','_pid','_recipe','_released','_phase','_working','_wire','_dirty','_producer_code')
    def __setattr__(self,*args):raise RuntimeError('immutable original tail working scope')
    def require_live(self):
        owner=self._owner
        require(owner._tail_authority._phase_plan==(('graph_gate',22528),('graph_candidate',20488),('probe',8192),('piece',2066),('wire',32768),('working',32768),('readback',1024),('sidecar_readback',2049),('frame_locals_dict',528)),'exact original phase plan')
        require(type(self) is TailWorkingScope and self._seal is _TAIL_WORK_SEAL
          and type(owner) is Budget and _LIVE_BUDGETS.get(id(owner)) is owner
          and owner._workspace_pid==self._pid==os.getpid() and owner._tail_work_active is self
          and not self._released and owner._tail_arm_state=='ARMED'
          and owner._tail_start is not None and owner._tail_deadline==min(owner.deadline,owner._tail_start+5),
          'exact live original source/owner/authenticated tail working capability')
        owner._tick(allow_failed=True);return owner
    def phase(self,name,working,wire=0):
        self.require_live()
        allowed={'sidecar0':{'sidecar':(8192,0)},'sidecar1':{'sidecar':(8192,0)},
          'failure':{'build':(32768,0),'encode':(32768,32768),'publish':(8192,32768)}}
        require(type(name) is str and name in allowed[self._recipe] and (working,wire)==allowed[self._recipe][name]
          and working<=32768 and wire<=32768,'fixed named original working/wire phase maxima')
        frame=sys._getframe(1)
        self._owner._tail_authority.require_producer(self._owner,self._recipe,frame)
        if name=='encode':
            require(frame.f_locals.get('domain') is None and frame.f_locals.get('state') is None,
              'obsolete preflight/accounting references released before encoder phase')
        if name=='publish':
            require(all(frame.f_locals.get(key) is None for key in ('domain','state','value','encoder','iterator','piece','encoded')),
              'producer graph/encoder/generator locals released before publication')
        order={'sidecar0':('sidecar',),'sidecar1':('sidecar',),'failure':('build','encode','publish')}[self._recipe]
        position=order.index(name)
        require(position==self._phase+1,'working phase cannot overlap/reopen/reset')
        object.__setattr__(self,'_phase',position);object.__setattr__(self,'_working',working)
        object.__setattr__(self,'_wire',wire);object.__setattr__(self,'_dirty',True)
    def release(self):
        frame=sys._getframe(1)
        self._owner._tail_authority.require_producer(self._owner,self._recipe,frame)
        require(frame.f_code is self._producer_code and frame.f_locals.get('scope') is self,
          'same actual acquired producer frame releases its own working scope')
        names=('readback',) if self._recipe!='failure' else ('state','value','domain','wire_view','wire','piece','encoded','chunk','hasher','encoder','iterator','copy_index','copy_end')
        require(all(frame.f_locals.get(name) is None for name in names),
          'actual producer temporary locals are cleared before working release')
        # Even an expired clock may release a cleared source allocation scope.
        # This operation confers no I/O permission or extra tail duration.
        owner=self._owner
        require(type(self) is TailWorkingScope and self._seal is _TAIL_WORK_SEAL
          and _LIVE_BUDGETS.get(id(owner)) is owner and owner._tail_work_active is self
          and owner._workspace_pid==self._pid==os.getpid() and not self._released,
          'same original active tail scope release once')
        object.__setattr__(self,'_dirty',False);object.__setattr__(self,'_released',True)
        object.__setattr__(owner,'_tail_work_active',None)

def tail_graph_payload(value,*,authority=None,owner=None):
    """Unique live graph identity payload; exact reference slots still count.

    Fixed8192 probe table avoids an unbounded set/ID integer container. The
    source graph remains live during traversal, so identities cannot be reused.
    This is conservative (borrowed constants count once), not a heap measure.
    """
    seen=(c.c_uint64*1024)();count=0
    def visit(item,depth):
        nonlocal count
        require(depth<=32,'fixed failure graph depth')
        identity=id(item)
        for index in range(count):
            if seen[index]==identity:return 0
        require(count<1024,'fixed unique failure graph identity census')
        seen[count]=identity;count+=1
        if type(item) is dict:
            require(len(item)<=128,'fixed failure object member domain')
            return 16+sum(16+visit(key,depth+1)+visit(member,depth+1) for key,member in item.items())
        if type(item) in (list,tuple):
            require(len(item)<=240,'fixed failure sequence domain')
            return 16+sum(8+visit(member,depth+1) for member in item)
        if item is None or type(item) in (str,bool,int,float):
            if authority is not None and authority.borrowed_scalar(owner,item):return 0
        if type(item) is str:
            require(item.isascii() and len(item)<=4096,'bounded source failure string payload')
            return len(item)
        if item is None:return 0
        require(type(item) in (bool,int,float),'closed source failure scalar payload')
        return 1 if type(item) is bool else 8
    try:return visit(value,0)
    finally:seen=None

class Budget(Owner):
    __slots__=('_watchdog','_tail_watchdog','_tail_start','_tail_deadline','_timer_factory','_native_cleanup','_pin_cleanup','_pin_limit','_tail_control','_tail_arm_state','_tail_arm_attempts','_root_started','_owned_backends','_workspace_pid','_tail_work_active','_tail_work_used','_tail_failure_stream','_tail_failure_close_attempted','_tail_failure_identity','_tail_authority')
    def __init__(self,partitions,clock=time.monotonic,watchdog_factory=timer,resource=None,pin_limit=0,preparation=None,tail_authority=None):
        # The immutable tables are constructed in bounded preparation; origin is
        # sampled once here before any grant construction/encoding/hash/fsync.
        object.__setattr__(self,'_root_started',False)
        prep_ticket=None if preparation is None else preparation.before('control',operation_attempts=128,metadata_bytes=4096)
        table={key:vector(value) for key,value in partitions.items()}
        require(all(sum(value[i] for value in table.values())==ROOT_BOUNDS[i] for i in range(6)),'original all6 reservation conserved')
        require(all(value[6]<=60 for value in table.values()),'one nested original wall clock')
        require(type(pin_limit) is int and 0<=pin_limit<=240 and pin_limit+16+16<=256,'actual closed pin census plus16native+16failure fits original256 tail')
        super().__init__(ROOT_BOUNDS,table,clock(),clock,resource)
        self._timer_factory=watchdog_factory;self._tail_start=None;self._tail_deadline=None;self._tail_watchdog=None;self._native_cleanup=0;self._pin_cleanup=0;self._pin_limit=pin_limit;self._tail_control=0;self._tail_arm_state='NOT_STARTED';self._tail_arm_attempts=0
        self._watchdog=None
        self._owned_backends=[None]*4
        # Five fields and one registry row are selected by the existing
        # positive PREP128/4096 owner constructor envelope.
        self._workspace_pid=os.getpid();self._tail_work_active=None;self._tail_work_used=0
        self._tail_failure_stream=None;self._tail_failure_close_attempted=False;self._tail_failure_identity=OutputDescriptorIdentity()
        _LIVE_BUDGETS[id(self)]=self
        require(type(tail_authority) is PreparedTailAuthority,'registered exact prepared tail producers required')
        self._tail_authority=tail_authority;tail_authority.bind(self)
        if prep_ticket is not None:prep_ticket.finish()
        origin=clock();object.__setattr__(self,'_origin',origin);object.__setattr__(self,'_deadline',origin+60)
        if preparation is not None:
            require(type(preparation) is PreparationOwner and _ENTRY_PREPARATION is not None
              and _ENTRY_PREPARATION[1] is preparation and not preparation._root_handoff,
              'exact source-owned PREP handoff before CPU origin authority')
            object.__setattr__(preparation,'_root_handoff',True)
        object.__setattr__(self,'_root_started',True)
        self.before('final_tail',operation_attempts=256,read_bytes=65536,write_bytes=65536,identity_hash_attempts=16,identity_hash_bytes=65536,metadata_bytes=65536)
        def arm_original_watchdog():
            watchdog=watchdog_factory(max(0,self.deadline-clock()),lambda:os._exit(124))
            # Retain the actual returned timer BEFORE the paid work's return
            # guard can fail; the partial owner keeps this same root handle.
            object.__setattr__(self,'_watchdog',watchdog)
            return watchdog
        self.run('control',{'operation_attempts':1,'metadata_bytes':256},arm_original_watchdog)
    def begin_final_tail(self):
        with self._lock:
            if self._tail_start is None:
                now=self._clock();object.__setattr__(self,'_tail_start',now);object.__setattr__(self,'_tail_deadline',min(self.deadline,now+5))
            self._tail_tick()
            if self._tail_arm_state!='ARMED':
                require(self._tail_arm_attempts<2,'same-tail watchdog arm attempts exhausted; owner quarantined')
                require(self._tail_control+1<=16,'prepaid finalizer/arm attempt subset exhausted')
                object.__setattr__(self,'_tail_control',self._tail_control+1)
                object.__setattr__(self,'_tail_arm_attempts',self._tail_arm_attempts+1)
                object.__setattr__(self,'_tail_arm_state','ARMING')
                try:
                    watchdog=self._timer_factory(max(0,self._tail_deadline-self._clock()),lambda:os._exit(124))
                    object.__setattr__(self,'_tail_watchdog',watchdog);self._tail_tick()
                    object.__setattr__(self,'_tail_arm_state','ARMED')
                except BaseException:
                    object.__setattr__(self,'_tail_arm_state','FAILED');object.__setattr__(self,'_failed',True);raise
    def _tail_tick(self):
        if self._tail_deadline is not None and self._clock()>=self._tail_deadline:
            object.__setattr__(self,'_failed',True);raise RuntimeError('same original single5s final tail expired')
    def _tick(self,allow_failed=False):super()._tick(allow_failed);self._tail_tick()
    def native(self,name,cleanup=False):
        if not cleanup:return self.before('native_host',operation_attempts=1)
        require(name in ('CloseHandle','TerminateJobObject','DeleteProcThreadAttributeList'),'owned cleanup API only')
        self.begin_final_tail()
        with self._lock:
            require(self._native_cleanup<16,'independent original16 native cleanup attempts')
            object.__setattr__(self,'_native_cleanup',self._native_cleanup+1)
        return CleanupTicket(self)
    def register_owned_backend(self,backend,ticket,*,job_bytes):
        # Existing positive reusable-constructor ticket owns these four slots.
        # Registration confers no process rights or native launch authority.
        require(type(ticket) is Ticket and ticket._owner is self and ticket._role=='control'
          and not ticket._finished,'same prepaid unfinished backend construction')
        require(getattr(backend,'owner',None) is self and type(job_bytes) is int and 1<=job_bytes<=4096
          and getattr(backend,'_poll_job',None) is not None
          and getattr(getattr(backend,'_poll_job_pointer',None),'_obj',None) is backend._poll_job,
          'exact original backend/buffer/retained source pointer construction')
        require(all(row is None or row[0] is not backend for row in self._owned_backends),'one backend registration')
        index=next((i for i,row in enumerate(self._owned_backends) if row is None),None)
        require(index is not None,'four fixed original backend registrations')
        self._owned_backends[index]=[backend,backend._poll_job,job_bytes,None,backend._poll_job_pointer]
    def terminal_verification(self,backend):
        def prepare():
            row=next((item for item in self._owned_backends if item is not None and item[0] is backend),None)
            require(row is not None and row[3] is None,'one final verification context for exact registered backend')
            require(getattr(backend,'owner',None) is self and backend._poll_job is row[1]
              and backend._poll_job_pointer is row[4],
              'original same-owner verification buffer')
            context=OwnedTerminalVerification(self,row,getattr(backend,'process',None),getattr(backend,'job',None))
            row[3]=context
            return context
        return self.tail_run(1,prepare)
    def verify_native(self,context,name,args,kwargs):
        # Unlike emergency handle release, a rejected read-only verification
        # MUST NOT dispatch. Validation precedes selecting the native attempt.
        require(type(context) is OwnedTerminalVerification and context._owner is self
          and any(item is context._row for item in self._owned_backends) and context._row[3] is context,
          'exact retained owned terminal-verification context')
        row=context._row;backend=row[0]
        require(type(args) is tuple and type(kwargs) is dict and getattr(backend,'owner',None) is self
          and backend._poll_job is row[1] and backend._poll_job_pointer is row[4],
          'exact source verification arguments')
        if name=='WaitForSingleObject':
            require(not context._wait_selected and context._process is not None and backend.process==context._process
              and len(args)==2 and args[0]==context._process and type(args[1]) is int and args[1]==0
              and kwargs=={'allow_false':True},'only retained originalprocess zero-time terminal wait')
            context._wait_selected=True
        elif name=='QueryInformationJobObject':
            require(not context._job_selected and context._job is not None and backend.job==context._job
              and len(args)==5 and args[0]==context._job and type(args[1]) is int and args[1]==8
              and args[2] is row[4] and type(args[3]) is int and args[3]==row[2]
              and args[4] is None and not kwargs,'only retained originalJob class8 originalbuffer terminal query')
            context._job_selected=True
        else:require(False,'terminal verification API outside closed source recipe')
        self.begin_final_tail()
        with self._lock:
            require(self._native_cleanup<16,'independent original16 native cleanup attempts')
            object.__setattr__(self,'_native_cleanup',self._native_cleanup+1)
        return CleanupTicket(self)
    def pin_close(self):
        self.begin_final_tail()
        with self._lock:
            require(self._pin_cleanup<self._pin_limit,'independent actual original pin-close subset exhausted')
            object.__setattr__(self,'_pin_cleanup',self._pin_cleanup+1)
        return CleanupTicket(self)
    def cleanup_after(self):self._tick(allow_failed=True)
    def tail_run(self,attempts,work):
        self.begin_final_tail()
        require(type(attempts) is int and attempts>0 and self._tail_control+attempts<=16,'fixed prepaid failure/publication attempts')
        object.__setattr__(self,'_tail_control',self._tail_control+attempts)
        try:result=work();self.cleanup_after();return result
        except BaseException:
            object.__setattr__(self,'_failed',True);raise
    def tail_working_scope(self,recipe):
        frame=sys._getframe(1)
        self._tail_authority.require_producer(self,recipe,frame)
        require(type(self) is Budget and _LIVE_BUDGETS.get(id(self)) is self and self._workspace_pid==os.getpid(),
          'same exact original source Budget before working allocation')
        self._tick(allow_failed=True)
        require(self._tail_arm_state=='ARMED' and self._tail_start is not None
          and self._tail_deadline==min(self.deadline,self._tail_start+5)
          and self._roles['final_tail'][5]==65536,'already selected original32768work+32768wire tail')
        require(recipe in ('sidecar0','sidecar1','failure'),'fixed original tail recipe')
        with self._lock:
            mask=1<<('sidecar0','sidecar1','failure').index(recipe)
            require(self._tail_work_active is None and not self._tail_work_used&mask,
              'tail working owner is nonreentrant and each recipe selected once')
            object.__setattr__(self,'_tail_work_used',self._tail_work_used|mask)
            # Sentinel retains quarantine if allocation/initialization fails.
            object.__setattr__(self,'_tail_work_active',self)
            try:
                scope=object.__new__(TailWorkingScope)
                fields={'_seal':_TAIL_WORK_SEAL,'_owner':self,'_pid':self._workspace_pid,'_recipe':recipe,
                  '_released':False,'_phase':-1,'_working':0,'_wire':0,'_dirty':False,'_producer_code':frame.f_code}
                for name,value in fields.items():object.__setattr__(scope,name,value)
                object.__setattr__(self,'_tail_work_active',scope)
                return scope
            except BaseException:
                # No producer allocation/IO has happened; partial capability
                # fields own no external resource. Keep recipe consumed, clear
                # an empty borrowed scope rather than false quarantine.
                object.__setattr__(self,'_tail_work_active',None);raise
    def failure_record(self,error,path,*,lifecycle=None,envelope=None):
        # The final-tail vector was selected before any root work. Eight of its
        # sixteen control attempts select this COMPLETE exporter before either
        # lifecycle extraction or accounting snapshot. No extra native query.
        require(envelope is not None and getattr(envelope,'MAXIMUM',None)==32768
          and getattr(envelope,'WORKSPACE',None)==32768 and getattr(envelope,'ATTEMPTS',None)==8,
          'exact prepared finite failure-envelope module required')
        def record():
            scope=self.tail_working_scope('failure')
            state=value=domain=wire=wire_view=piece=encoded=chunk=hasher=encoder=iterator=None;stream=None;path_text=None;primary=None
            def checked(action,returned=None):
                self._tick(allow_failed=True);result=action()
                if returned is not None:returned(result)
                try:self._tick(allow_failed=True)
                except BaseException:result=None;raise
                return result
            def opened(actual):
                object.__setattr__(self,'_tail_failure_stream',actual)
                self._tail_failure_identity.capture(actual)
            def closed(actual):
                self._tail_failure_identity.close_returned=True
                object.__setattr__(self,'_tail_failure_stream',None)
            try:
                scope.phase('build',32768)
                require((type(path) is str or isinstance(path,Path)) and str(path) in _TAIL_FAILURE_PATHS
                  and len(str(path))<=320 and str(path).isascii(),
                  'fixed bounded failure path/native wide conversion input')
                path_text=str(path)
                self._tail_authority.authorize_data('accounting')
                state={'selected':dict(zip(FIELDS,self._selected)),'used':dict(zip(FIELDS,self._used)),
                  'partition_used':{key:dict(zip(FIELDS,value)) for key,value in self._roles.items()},
                  'failed':True,'refunds':0,'native_cleanup_attempts':self._native_cleanup,
                  'pin_cleanup_attempts':self._pin_cleanup,'pin_cleanup_limit':self._pin_limit,
                  'tail_control_attempts_selected':self._tail_control,'tail_start':self._tail_start,
                  'tail_deadline':self._tail_deadline,'original_origin':self.origin,'original_deadline':self.deadline,
                  'complete_failure_envelope_selected_before_snapshot':True,'publication_operations_selected':8,
                  'publication_bytes_selected':32768,'publication_metadata_including_workspace_selected':65536,
                  'external_original_process_exit_required':True}
                domain=envelope.preflight(error,lifecycle,state,scope=scope)
                value=envelope.build_preflight(domain);self.cleanup_after()
                envelope.authenticate_pieces(value)
                # Actual input graph bound before encoder/wire allocation. The
                # builder's preallocation shape proof remains independently required.
                require(tail_graph_payload(value,authority=self._tail_authority,owner=self)<=22528 and encoded_extent(value)+1<=32768,
                  'bounded logical source failure graph and separate wire extent before streaming encoding')
                # Output owns the only retained graph. Discard preflight tuple
                # and redundant accounting references before encoder allocation.
                domain=state=None
                scope.phase('encode',32768,32768)
                wire=bytearray(32768);wire_view=memoryview(wire);wire_used=0
                encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
                iterator=encoder.iterencode(value)
                for piece in iterator:
                    require(type(piece) is str and len(piece)<=2066 and piece.isascii(),
                      'authenticated exact escaped ASCII piece domain before any copy')
                    require(wire_used+len(piece)+1<=32768,
                      'original wire capacity includes final newline before first write')
                    # Direct character writes remove the second full-sized
                    # encoded bytes allocation. At most2066 writes per piece.
                    # Each <=64-character source chunk checks the same original
                    # clock before and after; expiry permits no following chunk.
                    copy_index=0
                    while copy_index<len(piece):
                        self.cleanup_after();copy_end=min(copy_index+64,len(piece))
                        while copy_index<copy_end:
                            wire[wire_used]=ord(piece[copy_index]);wire_used+=1;copy_index+=1
                        self.cleanup_after()
                    copy_index=copy_end=None;piece=encoded=None
                    self.cleanup_after()
                wire[wire_used]=10;wire_used+=1;iterator=encoder=value=state=domain=None
                scope.phase('publish',8192,32768)
                checked(lambda:open(path_text,'xb+',buffering=0),opened)
                written=checked(lambda:self._tail_failure_stream.write(wire_view[:wire_used]))
                require(written==wire_used,'full original failure write')
                checked(lambda:self._tail_failure_stream.flush())
                checked(lambda:os.fsync(self._tail_failure_stream.fileno()))
                checked(lambda:self._tail_failure_stream.seek(0))
                hasher=checked(hashlib.sha256);offset=0
                while offset<wire_used:
                    chunk=checked(lambda:self._tail_failure_stream.read(min(1024,wire_used-offset)))
                    require(len(chunk)>0 and chunk==wire_view[offset:offset+len(chunk)],'exact original failure readback')
                    checked(lambda:hasher.update(chunk));offset+=len(chunk);chunk=None
                require(checked(lambda:self._tail_failure_stream.read(1))==b'','original failure grew; partial preserved')
                digest=checked(hasher.hexdigest)
                def close():
                    require(not self._tail_failure_close_attempted,'original failure stream close once')
                    object.__setattr__(self,'_tail_failure_close_attempted',True)
                    self._tail_failure_identity.close_attempted=True
                    try:return self._tail_failure_stream.close()
                    except BaseException:self._tail_failure_identity.close_failed=True;raise
                checked(close,closed)
                result={'path':path_text,'bytes':wire_used,'sha256':digest,
                  'output_identity':self._tail_failure_identity.value()}
                wire_view=wire=chunk=hasher=None
                return result
            except BaseException as actual:primary=actual;raise
            finally:
                if iterator is not None:iterator.close()
                state=value=wire_view=wire=piece=encoded=chunk=hasher=encoder=iterator=copy_index=copy_end=None
                domain=None
                # Keep the exact producer exception/traceback locations while
                # clearing dormant helper-frame payload references. The source
                # owner and original retained exceptions/blocks stay intact.
                if primary is not None and primary is not error:
                    # Clear only the new producer failure; preserve the original
                    # retained exception and its attached raw source evidence.
                    traceback.clear_frames(primary.__traceback__)
                if self._tail_failure_stream is not None and not self._tail_failure_close_attempted:
                    try:
                        def close():
                            object.__setattr__(self,'_tail_failure_close_attempted',True)
                            self._tail_failure_identity.close_attempted=True
                            try:return self._tail_failure_stream.close()
                            except BaseException:self._tail_failure_identity.close_failed=True;raise
                        checked(close,closed)
                    except BaseException as actual:
                        if primary is None:raise
                        primary.add_note('original failure stream unresolved close: '+type(actual).__name__)
                scope.release()
        return self.tail_run(8,record)
    def final_envelope(self,path,build,maximum=32768,validator=None):
        # Select ALL final construction/snapshot/JSON/write/readback-reference
        # work before the record can observe its accounting state.
        ticket=self.before('control',operation_attempts=8,write_bytes=maximum,identity_hash_attempts=1,
          identity_hash_bytes=maximum,metadata_bytes=maximum+4096)
        try:
            state={'selected':dict(zip(FIELDS,self._selected)),'used':dict(zip(FIELDS,self._used)),
              'partition_used':{key:dict(zip(FIELDS,value)) for key,value in self._roles.items()},'pending':[],
              'failed':[],'refunds':0,'complete_final_envelope_selected_before_snapshot':True}
            value=build(state)
            if validator is not None:validator(value)
            self.after('control')
            raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False,
              default=lambda item:dict(item.items()) if type(item) is FrozenMap else (_ for _ in ()).throw(TypeError('unsupported JSON object'))).encode()+b'\n'
            require(len(raw)<=maximum,'selected final envelope bound');self.after('control')
            digest=hashlib.sha256(raw).hexdigest();self.after('control')
            with Path(path).open('xb') as stream:
                require(stream.write(raw)==len(raw),'full selected final envelope write');stream.flush();os.fsync(stream.fileno())
            ticket.finish();return {'path':str(path),'bytes':len(raw),'sha256':digest}
        except BaseException:object.__setattr__(self,'_failed',True);raise
    @property
    def pin_closes(self):return self._pin_cleanup
    @property
    def cleanup(self):return self._native_cleanup
    def finish(self):
        self._tick();self._resources();self._tick();self._watchdog.cancel()
        if self._tail_watchdog is not None:self._tail_watchdog.cancel()
        object.__setattr__(self,'_closed',True)
class CleanupTicket:
    __slots__=('_owner','_finished')
    def __init__(self,owner):self._owner=owner;self._finished=False
    def finish(self):
        require(not self._finished,'one original cleanup return');self._finished=True;self._owner.cleanup_after()
class OwnedTerminalVerification:
    __slots__=('_owner','_row','_process','_job','_wait_selected','_job_selected')
    def __init__(self,owner,row,process,job):
        self._owner=owner;self._row=row;self._process=process;self._job=job
        self._wait_selected=False;self._job_selected=False
