"""Same original metadata owner holds exact files across the CPU60 root.

Streaming SHA work is recorded PRE-GRANT preparation. Original Windows handles
deny concurrent write/delete; later integrity checks use those handles, not a
second raw hash or a cached caller status. No native API runs at import.
"""
import copy,ctypes as c,hashlib,os,time
from pathlib import Path
import admission as a
MAX_PINS=240
_SEAL=object();_LIVE={}
class _FT(c.Structure):_fields_=[('low',c.c_uint32),('high',c.c_uint32)]
class _INFO(c.Structure):
    _fields_=[('attributes',c.c_uint32),('creation',_FT),('access',_FT),('write',_FT),('volume',c.c_uint32),
      ('size_hi',c.c_uint32),('size_lo',c.c_uint32),('links',c.c_uint32),('id_hi',c.c_uint32),('id_lo',c.c_uint32)]
class WinPinsIO:
    def __init__(self):
        a.require(os.name=='nt','actual Windows read custody only')
        self.kernel=c.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateFileW.argtypes=[c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_void_p,c.c_uint32,c.c_uint32,c.c_void_p];self.kernel.CreateFileW.restype=c.c_void_p
        self.kernel.GetFileInformationByHandle.argtypes=[c.c_void_p,c.POINTER(_INFO)];self.kernel.GetFileInformationByHandle.restype=c.c_int
        self.kernel.ReadFile.argtypes=[c.c_void_p,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32),c.c_void_p];self.kernel.ReadFile.restype=c.c_int
        self.kernel.CloseHandle.argtypes=[c.c_void_p];self.kernel.CloseHandle.restype=c.c_int
    def open(self,path):
        # GENERIC_READ / FILE_SHARE_READ ONLY / OPEN_EXISTING. Sharing denies
        # writes and deletion throughout original-handle custody.
        handle=self.kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
        a.require(handle not in (None,c.c_void_p(-1).value),'original deny-write/delete read handle unavailable')
        return handle
    def info(self,handle):
        value=_INFO();a.require(bool(self.kernel.GetFileInformationByHandle(handle,c.byref(value))),'original file identity unavailable')
        return (value.volume,value.id_hi,value.id_lo,(value.size_hi<<32)|value.size_lo,value.attributes)
    def read(self,handle,maximum):
        buffer=c.create_string_buffer(maximum);size=c.c_uint32()
        a.require(bool(self.kernel.ReadFile(handle,buffer,maximum,c.byref(size),None)),'original pinned source read failed')
        return buffer.raw[:size.value]
    def close(self,handle):a.require(bool(self.kernel.CloseHandle(handle)),'original pinned source close failed')
class RuntimeCustody:
    __slots__=('_seal','_pid','_io','_native','_rows','_closed','_failed','_records','_preparation','_clock','_namespaces')
    def __init__(self,*args):raise RuntimeError('source preparation creates original custody')
    def __setattr__(self,k,v):raise RuntimeError('immutable source custody identity')
    @classmethod
    def prepare(cls,refs,io=None,clock=time.monotonic,owner=None,namespaces=()):
        a.require(type(refs) is list and 0<len(refs)<=MAX_PINS,'finite exact original read-pin census')
        if owner is None:owner=a.preparation_owner()
        def paid(counts,work):
            ticket=owner.before('control',**counts)
            result=work();ticket.finish();return result
        for namespace in namespaces:
            paid({'operation_attempts':1024,'metadata_bytes':65536},lambda namespace=namespace:require_namespace(namespace))
        unique={}
        for item in refs:
            a.require(type(item) is dict and set(item)=={'path','bytes','sha256'} and type(item['bytes']) is int and 0<item['bytes']<=33554432,'bounded raw pin reference')
            path=Path(item['path']);a.require(path.is_absolute() and '..' not in path.parts,'absolute pin role')
            if item['path'] in unique:a.require(unique[item['path']]==item,'same path cannot carry two source identities')
            unique[item['path']]=paid({'operation_attempts':1,'metadata_bytes':1024},lambda item=item:dict(item))
        a.require(len(unique)<=MAX_PINS,'pin cleanup fits original reserved subset')
        value=object.__new__(cls);backend=paid({'operation_attempts':32,'metadata_bytes':4096},lambda:WinPinsIO() if io is None else io)
        for k,v in {'_seal':_SEAL,'_pid':os.getpid(),'_io':backend,'_native':type(backend) is WinPinsIO,'_rows':[],
          '_closed':False,'_failed':False,'_records':[],'_preparation':None,'_clock':clock,
          '_namespaces':paid({'operation_attempts':1,'metadata_bytes':131072},lambda:tuple(copy.deepcopy(x) for x in namespaces))}.items():object.__setattr__(value,k,v)
        _LIVE[id(value)]=value;owner.bind_failure_custody(value);started=clock();counts={'open_attempts':0,'info_attempts':0,'read_attempts':0,'requested_read_bytes':0,'actual_read_bytes':0,'sha_attempts':0,'sha_bytes':0,'close_attempts':0}
        try:
            for item in unique.values():
                path=Path(item['path'])
                if value._native:
                    a.require(path.resolve()==path and all(not p.is_symlink() and not(getattr(p.stat(),'st_file_attributes',0)&1024) for p in (path,*path.parents)),'regular current execution pin path')
                counts['open_attempts']+=1;opening=owner.before('control',operation_attempts=1,metadata_bytes=1024)
                handle=backend.open(path)
                # Retain the original handle BEFORE its return/deadline/resource
                # check can fail, so the exception path can still close it.
                row={'reference':item,'handle':handle,'identity':None,'closed':False,'close_attempted':False};value._rows.append(row);opening.finish()
                counts['info_attempts']+=1;identity=paid({'operation_attempts':1},lambda:backend.info(handle))
                a.require(type(identity) is tuple and len(identity)==5 and identity[3]==item['bytes'] and not identity[4]&(0x10|0x400),'regular exact original file extent')
                row['identity']=identity;digest=paid({'operation_attempts':1,'identity_hash_attempts':1},hashlib.sha256);counts['sha_attempts']+=1;total=0
                while True:
                    maximum=min(1048576,item['bytes']-total+1);counts['read_attempts']+=1;counts['requested_read_bytes']+=maximum
                    block=paid({'operation_attempts':1,'read_bytes':maximum},lambda:backend.read(handle,maximum));a.require(type(block) is bytes and len(block)<=maximum,'bounded original handle read')
                    total+=len(block);counts['actual_read_bytes']+=len(block);a.require(total<=item['bytes'],'pinned extent grew')
                    if not block:break
                    paid({'operation_attempts':1,'identity_hash_bytes':len(block)},lambda:digest.update(block));counts['sha_bytes']+=len(block)
                actual_sha=paid({'operation_attempts':1},digest.hexdigest)
                a.require(total==item['bytes'] and actual_sha==item['sha256'],'same original pinned extent/SHA')
            prep={'schema':'CE8-PRE-GRANT-READ-CUSTODY-PREPARATION-1','owner_thread_id':a.OWNER,'owner_process_id':value._pid,
              'phase':'PRE_GRANT_METADATA_PREPARATION_OUTSIDE_TARGET_OBSERVATION_ROOT','started_monotonic':started,'ended_monotonic':clock(),
              'counts':counts,'pins':[{k:v for k,v in row.items() if k not in ('handle','closed')} for row in value._rows],
              'native_read_pins':value._native,'experiment_root_SHA_debit':False,'preparation_work_claimed_free':False,'models_loaded':0,'optimizer_updates':0}
            object.__setattr__(value,'_preparation',prep);return value
        except BaseException as error:
            object.__setattr__(value,'_failed',True)
            owner.notify_custody_failure(value,error)
            value._records.append({'phase':'PREPARATION_FAILED','error_type':type(error).__name__,'counts':counts})
            try:error._ce8_original_pin_failure=value
            except BaseException:pass
            try:value.close_all(failure_owner=owner)
            except BaseException as close_error:error.add_note('original pin cleanup: '+type(close_error).__name__)
            raise
    def require_source_owner(self,native_required=True):
        a.require(type(self) is RuntimeCustody and self._seal is _SEAL and _LIVE.get(id(self)) is self and self._pid==os.getpid()
          and not self._closed and not self._failed and (self._native or not native_required),'same private live original source owner')
        return self
    def borrow_runtime_pin(self,reference,owner):
        # Caller has already positively selected the complete PREP plan.
        # Borrow an original row, not decoded fields or new file authority.
        a.require(owner is a._PREP_OWNER,'same original preparation owner for runtime pin borrow')
        self.require_source_owner()
        for row in self._rows:
            if row['reference']==reference:
                a.require(not row['closed'] and row['handle'] is not None and row['identity'] is not None,
                  'exact live original runtime pin before plan preparation')
                return row
        a.require(False,'registered runtime is absent from original custody')
    def require_live(self,before_info=None,after_info=None,native_required=True,before_namespace=None,after_namespace=None):
        self.require_source_owner(native_required)
        for row in self._rows:
            primary=None
            ticket=None
            try:
                if before_info is not None:ticket=before_info()
                a.require(not row['closed'] and self._io.info(row['handle'])==row['identity'],'original held file identity/extent changed')
            except BaseException as error:primary=error
            try:
                if after_info is not None:after_info()
                elif ticket is not None:ticket.finish()
            except BaseException as error:
                if primary is None:primary=error
                else:primary.add_note('original file observation return guard: '+type(error).__name__)
            try:
                if primary is not None:raise primary
            except BaseException:
                object.__setattr__(self,'_failed',True);raise
        for namespace in getattr(self,'_namespaces',()):
            primary=None;namespace_ticket=None
            try:
                if before_namespace is not None:namespace_ticket=before_namespace(namespace)
                require_namespace(namespace)
            except BaseException as error:primary=error
            try:
                if after_namespace is not None:after_namespace(namespace)
                elif namespace_ticket is not None:namespace_ticket.finish()
            except BaseException as error:
                if primary is None:primary=error
                else:primary.add_note('original namespace observation return guard: '+type(error).__name__)
            if primary is not None:object.__setattr__(self,'_failed',True);raise primary
        return self
    def covers(self,item):return any(row['reference']==item and not row['closed'] for row in self._rows)
    def close_all(self,before_close=None,after_close=None,failure_owner=None):
        failures=[]
        for row in self._rows:
            if row['closed']:continue
            if row.get('close_attempted',False):failures.append('UNKNOWN_PRIOR_CLOSE');continue
            # Failed original selection preserves this exact owned pin.
            ticket=None
            try:
                if failure_owner is not None:ticket=failure_owner.failure_pin_before(self,row)
                elif before_close is not None:ticket=before_close()
            except BaseException as error:
                failures.append(type(error).__name__)
                continue
            try:
                row['close_attempted']=True
                self._io.close(row['handle']);row['closed']=True
            except BaseException as error:failures.append(type(error).__name__)
            try:
                if after_close is not None:after_close()
                elif ticket is not None:ticket.finish()
            except BaseException as error:failures.append('RETURN_'+type(error).__name__)
        object.__setattr__(self,'_closed',all(row['closed'] for row in self._rows))
        if failures:object.__setattr__(self,'_failed',True)
        self._records.append({'phase':'PIN_RELEASE','all_original_handles_closed':self._closed,'failure_types':failures,'refunds':0})
        a.require(not failures and self._closed,'original pin closure incomplete; evidence/strong references retained')
        _LIVE.pop(id(self),None);return {'closed_pins':len(self._rows),'all_original_handles_closed':True,'refunds':0}
def preparation_refs(packet,target):
    refs=[*packet['source_refs'].values(),a.PROVIDER,a.JOB_CENSUS,a.AUTH,packet['target_manifest_ref'],target['actor_source_ref'],
      target['actor_source_freeze_ref'],target['artifact_ref'],target['runtimeconfig_ref'],target['runtime_ref'],
      target['runtime_package_manifest_ref'],*target['runtime_dependency_refs'].values()]
    unique={}
    for item in refs:
        if item['path'] in unique:a.require(unique[item['path']]==item,'runtime path identity conflict')
        unique[item['path']]=item
    a.require(len(unique)<=224,'complete private runtime census fits224pin subset plus16native and16records inside original256tail')
    return list(unique.values())

def require_namespace(namespace):
    a.require(type(namespace) is dict and set(namespace)=={'root','directories','files'},'exact private runtime namespace recipe')
    root=Path(namespace['root']);a.require(root.is_absolute() and root.is_dir() and root.resolve()==root
      and all(not p.is_symlink() and not(getattr(p.stat(),'st_file_attributes',0)&1024) for p in (root,*root.parents)),
      'exact private runtime root without ancestor reparse')
    directories=[];files=[]
    for item in root.rglob('*'):
        a.require(len(directories)+len(files)<=len(namespace['directories'])+len(namespace['files']),'bounded private namespace additions')
        a.require(not item.is_symlink() and not(getattr(item.stat(),'st_file_attributes',0)&1024),'runtime namespace reparse entry refused')
        relative=item.relative_to(root).as_posix()
        if item.is_dir():directories.append(relative)
        elif item.is_file():files.append(relative)
        else:a.require(False,'runtime namespace nonregular entry')
    a.require(sorted(directories)==namespace['directories'] and sorted(files)==sorted(namespace['files']),
      'private runtime loader namespace gained or lost a candidate')
    return True
