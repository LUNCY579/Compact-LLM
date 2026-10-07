"""Separate zero-allocation observer load/unload diagnostic worker.

No model, Torch, CUDA, tensor context or tensor registry is consumed. Public
entry refuses at the distinct location/registration gate before native APIs.
CPU fixtures exercise the same bounded operation body with inert providers.
"""
from pathlib import Path
from types import ModuleType
import hashlib, math, os, time

HERE=Path(__file__).resolve().parent
def _module(path,expected=None):
    path=Path(path);raw=path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('exact observer-only bootstrap contract source')
    module=ModuleType('observer_only_'+path.stem);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module
C=_module(HERE/'contract.py','b6d722cefb32c4f5735b2937c7edd6d68c551e2a1555bbe41f4e7518fc20ad1c')
OPS=('source_preflight','owned_identity','observer_binding','observer_start','initial_module_baseline',
    'fixture_load','loaded_module_snapshot','fixture_unload','unloaded_module_snapshot',
    'observer_stop','final_event_drain','final_source_preflight')
SNAPSHOT_FIELDS=('abi_version','row_bytes','snapshot_bytes','row_capacity','full_capacity','base_capacity',
    'phase','faults','attempted','reserved','published_prefix','inflight','cookie_live',
    'register_status','unregister_status','drained','complete','dropped')
ROW_FIELDS={'sequence','base_address','reason','image_size','full_units','base_units','flags','ready','full_name','base_name'}

def _error(error):
    cls=type(error);name=type.__getattribute__(cls,'__name__');message=None;exact=False
    try:
        args=BaseException.args.__get__(error,cls)
        if type.__getattribute__(cls,'__str__') is BaseException.__str__:
            if type(args) is tuple and len(args)==1 and type(args[0]) is str:message=args[0][:2048];exact=len(args[0])<=2048
            elif type(args) is tuple and not args:message='';exact=True
    except BaseException:pass
    return {'type':name[:128] if type(name) is str else 'BaseException','message':message,'message_exact':exact}

def _snapshot(c,value,*,closed):
    c.require(type(value) is dict and set(value)==set(SNAPSHOT_FIELDS),'exact observer-only collector ABI snapshot')
    c.require(all(type(v) is int and (-2**31<=v<2**31 if k in ('register_status','unregister_status') else 0<=v<2**32) for k,v in value.items()),'typed bounded observer snapshot')
    expected={'abi_version':1,'row_bytes':2600,'snapshot_bytes':72,'row_capacity':4096,'full_capacity':1024,'base_capacity':256}
    c.require(all(value[k]==v for k,v in expected.items()),'original observer-only ABI capacities')
    c.require(value['faults']==value['dropped']==value['inflight']==0 and value['attempted']==value['reserved']==value['published_prefix']<=4096,'lossless observer prefix; no overflow late callbacks or inflight rows')
    c.require(value['register_status']==0,'original observer registration succeeded')
    if closed:c.require(value['phase']==5 and value['cookie_live']==0 and value['unregister_status']==0 and value['drained']==value['complete']==1,'unregistered drained sealed collector state')
    else:c.require(value['phase']==2 and value['cookie_live']==1 and value['unregister_status']==-2147483648 and value['drained']==value['complete']==0,'active registered collector state')
    return c.parse(c.canonical(value))

def _modules(c,value):
    c.require(type(value) is dict and set(value)=={'kind','complete','libraries'} and value['kind']=='OBSERVED_PROCESS_LIBRARY_SNAPSHOT' and value['complete'] is True and type(value['libraries']) is list and 0<len(value['libraries'])<=4096,'complete bounded module baseline/snapshot')
    result={};handles=set()
    for row in value['libraries']:
        c.require(type(row) is dict and set(row)=={'path','module_handle'} and type(row['path']) is str and row['path'] and type(row['module_handle']) is int and 0<row['module_handle']<2**64 and row['path'] not in result and row['module_handle'] not in handles,'unique complete module path/handle identities')
        result[row['path']]=row['module_handle'];handles.add(row['module_handle'])
    return result

def _u16(c,value,units,limit):
    c.require(type(units) is int and 0<units<limit and type(value) is list and len(value)==units and all(type(v) is int and 0<=v<65536 for v in value),'bounded counted UTF16 observer names')
    text=b''.join(v.to_bytes(2,'little') for v in value).decode('utf-16-le','strict')
    c.require('\0' not in text,'observer name has no embedded NUL');return text

def _operation_body(context,contract,make_transport,make_provider,output,clock,revalidate,bootstrap_actor):
    c=contract;value=c.context_value(context);config=value['config'];output=Path(output).absolute()
    c.require(config['scope']==c.SCOPE and c.canonical(config['permission'])==c.canonical(c.PERMISSION) and c.canonical(config['resources'])==c.canonical(c.RESOURCES),'distinct no-model observer-only scope/permission/resource profile')
    c.require(output.resolve()==output and output.is_dir(),'existing exact observer-only output leaf')
    started=clock();c.require(type(started) in (int,float) and math.isfinite(started),'finite observer-only main clock')
    counts={op:{'attempted':0,'returned':0,'failed':0,'pending':0} for op in OPS};errors=[];ordinal=0;last_clock=started
    state={'provider':None,'transport':None,'start_attempted':False,'handle_live':False,'unload_attempted':False,
        'owned_identity':None,'abi_binding':None,'fixture_handle':None,'initial_modules':None,
        'loaded_modules':None,'unloaded_modules':None,'events':None,'final_collector_snapshot':None,'prefix':[],'raw_rows':[]}
    phase_stream=(output/'OBSERVER_PHASES.jsonl').open('xb')
    startup={'schema_version':1,'kind':'OBSERVER_STARTUP_INVENTORY_V1','scope':c.SCOPE,
        'source_commit':value['expected_head'],'implementation_commit':config['implementation_commit'],
        'config_sha256':value['anchors']['expected_config_file'],'context_sha256':c.digest(value),
        'bootstrap_policy_sha256':c.digest(value['runtime'].get('cache_policy',{})),
        'bootstrap':None,'initial_module_snapshots':[],'unknown_libraries':[],
        'fixture_only':value['fixture_only'],'namespace_execution_proven':False,
        'native_PASS':False,'native_authority_issued':False}
    def tick():
        nonlocal last_clock
        now=clock();c.require(type(now) in (int,float) and math.isfinite(now) and now>=last_clock and now-started<=c.RESOURCES['whole_scope_seconds'],'bounded nondecreasing observer-only clock')
        last_clock=now;return float(now)
    def row(operation,event,data=None,exception=None):
        nonlocal ordinal
        ordinal+=1
        record={'schema_version':1,'ordinal':ordinal,'operation':operation,'event':event,'monotonic':tick(),'data':data,'exception':exception}
        raw=(c.canonical(record)+'\n').encode('ascii');c.require(phase_stream.tell()+len(raw)<=c.MAX_JSON,'bounded complete observer phase ledger')
        phase_stream.write(raw);phase_stream.flush();os.fsync(phase_stream.fileno())
    def invoke(op,action):
        c.require(counts[op]['attempted']==0,'observer-only operation attempted at most once')
        counts[op].update(attempted=1,pending=1)
        try:
            row(op,'started')
            result=action();row(op,'returned',result);counts[op].update(returned=1,pending=0);return result
        except BaseException as error:
            counts[op].update(failed=1,pending=0);record=_error(error);errors.append({'operation':op,'error':record})
            try:row(op,'failed',exception=record)
            except BaseException as ledger_error:errors.append({'operation':op,'error':_error(ledger_error)})
            raise
    def preflight(*,first=False):
        current=revalidate();c.require(current==value,'original observer-only context cannot rebase')
        total=0
        for mapping in (config['source_files'],value['runtime']['files']):
            for path,descriptor in mapping.items():c.file_descriptor(path,descriptor);total+=descriptor['bytes']
        if first:
            observed=bootstrap_actor()
            startup['bootstrap']=c.parse(c.canonical(observed))
            c.require(type(observed) is dict and set(observed)=={'kind','imports','modules','unknown_modules','fixture_only','namespace_execution_proven'} and observed['kind']=='MEASURED_BOOTSTRAP_IMPORTS_V1' and observed['imports']==['ctypes','ctypes.wintypes','_ctypes'] and type(observed['modules']) is list and type(observed['unknown_modules']) is list and observed['fixture_only'] is value['fixture_only'] and observed['namespace_execution_proven'] is False,'original measured bootstrap import projection')
            c.require(not observed['unknown_modules'],'unselected original bootstrap modules; fixture work withheld')
        if state['events'] is not None:
            before=_snapshot(c,state['transport'].snapshot(),closed=True)
            rows=state['transport'].copy_rows(before['published_prefix'])
            after=_snapshot(c,state['transport'].snapshot(),closed=True)
            c.require(before==after==state['final_collector_snapshot'] and rows==state['raw_rows'],'final sealed collector/event bytes immutable after source checks')
        return {'context_sha256':c.digest(value),'config_sha256':value['anchors']['expected_config_file'],
            'source_count':len(config['source_files']),'runtime_count':len(value['runtime']['files']),'verified_bytes':total}
    def identity():
        state['provider']=make_provider();birth=state['provider'].owned_identity()
        c.require(type(birth) is dict and set(birth)=={'pid','creation_filetime_100ns'} and all(type(v) is int and 0<v<2**64 for v in birth.values()),'exact current owned observer-only process birth')
        state['owned_identity']=dict(birth);return dict(birth)
    def binding():
        state['transport']=make_transport();actual=state['transport'].abi_binding();selected=config['observer_registration']
        c.require(type(actual) is dict and set(actual)=={'path','descriptor','abi','dll_handle'} and actual['path']==selected['binary']['path'] and actual['descriptor']==selected['binary']['descriptor'] and actual['abi']==c.ABI and type(actual['dll_handle']) is int and actual['dll_handle']>0,'actual original observer-only DLL/ABI binding')
        state['abi_binding']=c.parse(c.canonical(actual));return state['abi_binding']
    def start():
        state['start_attempted']=True;status=state['transport'].start();c.require(type(status) is int and status==0,'observer-only collector start succeeded')
        snapshot=_snapshot(c,state['transport'].snapshot(),closed=False);c.require(snapshot['published_prefix']==0,'empty original event prefix before fixture work')
        return {'status':status,'snapshot':snapshot}
    def modules(which):
        observed=state['provider'].modules();state[which]=c.parse(c.canonical(observed))
        if which=='initial_modules':startup['initial_module_snapshots'].append(state[which])
        mapping=_modules(c,observed)
        repeated=state['provider'].modules()
        if which=='initial_modules':startup['initial_module_snapshots'].append(c.parse(c.canonical(repeated)))
        c.require(_modules(c,repeated)==mapping,'stable complete module snapshot across two bounded observations')
        fixture=config['binary_fixture']['path']
        if which=='initial_modules':
            admitted=set(value['runtime']['allowed_initial_loaded_libraries'])
            startup['unknown_libraries']=[row for row in observed['libraries'] if row['path'] not in admitted]
            c.require(not startup['unknown_libraries'] and fixture not in mapping,'observed initial modules inside prospective byte admission; fixture work withheld on unknown modules')
            c.require(mapping.get(state['abi_binding']['path'])==state['abi_binding']['dll_handle'],'collector binding joins actual initial modules')
        elif which=='loaded_modules':
            before=_modules(c,state['initial_modules']);c.require(mapping=={**before,fixture:state['fixture_handle']},'exact one fixture module added; every initial handle conserved')
        else:c.require(mapping==_modules(c,state['initial_modules']),'exact initial module population after one fixture unload')
        state[which]=c.parse(c.canonical(observed));return state[which]
    def load():
        fixture=config['binary_fixture'];c.file_descriptor(fixture['path'],fixture['descriptor'])
        handle=state['provider'].load_fixture(fixture['path']);c.require(type(handle) is int and 0<handle<2**64,'actual nonzero fixture DLL handle')
        state['fixture_handle']=handle;state['handle_live']=True;return {'path':fixture['path'],'module_handle':handle}
    def unload():
        state['unload_attempted']=True;returned=state['provider'].unload_fixture(state['fixture_handle'])
        c.require(returned is True,'fixture unload returned actual success');state['handle_live']=False
        return {'path':config['binary_fixture']['path'],'module_handle':state['fixture_handle'],'returned':True}
    def stop():
        state['stop_attempted']=True
        status,snapshot=state['transport'].stop();state['final_collector_snapshot']=c.parse(c.canonical(snapshot))
        c.require(type(status) is int and status==0,'original observer-only unregister/drain returned success')
        stopped=_snapshot(c,snapshot,closed=True);state['final_collector_snapshot']=stopped;return {'status':status,'snapshot':stopped}
    def drain():
        transport=state['transport'];before=_snapshot(c,transport.snapshot(),closed=True)
        rows=transport.copy_rows(before['published_prefix']);after=_snapshot(c,transport.snapshot(),closed=True)
        c.require(before==after==state['final_collector_snapshot'],'stable immutable sealed event prefix; late callback poison refused')
        c.require(type(rows) is list and len(rows)==2 and before['published_prefix']==2,'exact fixture load/unload event population')
        parsed=[];fixture=config['binary_fixture']['path'];image=None
        for i,item in enumerate(rows):
            c.require(type(item) is dict and set(item)==ROW_FIELDS and all(type(item[k]) is int for k in ('sequence','base_address','reason','image_size','full_units','base_units','flags','ready')),'exact typed observer-only raw row')
            c.require(item['sequence']==i+1 and item['reason']==i+1 and item['base_address']==state['fixture_handle'] and 0<item['image_size']<2**32 and item['flags']==0 and item['ready']==1,'ordered exact fixture load then unload at same handle')
            full=_u16(c,item['full_name'],item['full_units'],1024);base=_u16(c,item['base_name'],item['base_units'],256)
            c.require(full.casefold()==fixture.casefold() and Path(full).name.casefold()==base.casefold(),'exact selected fixture full/base UTF16 paths')
            c.require(image is None or image==item['image_size'],'same fixture image size load/unload');image=item['image_size']
            parsed.append({'sequence':i+1,'event':'load' if item['reason']==1 else 'unload','path':fixture,'module_handle':item['base_address']})
        state['events']={'kind':'OBSERVED_PROCESS_LIBRARY_EVENTS','complete':True,'events':parsed}
        state['raw_rows']=c.parse(c.canonical(rows));state['final_collector_snapshot']=after
        return {'events':state['events'],'snapshot':after,'raw_rows':state['raw_rows']}
    primary=None
    try:
        invoke('source_preflight',lambda:preflight(first=True));invoke('owned_identity',identity);invoke('observer_binding',binding)
        invoke('observer_start',start);invoke('initial_module_baseline',lambda:modules('initial_modules'))
        invoke('fixture_load',load);invoke('loaded_module_snapshot',lambda:modules('loaded_modules'))
        invoke('fixture_unload',unload);invoke('unloaded_module_snapshot',lambda:modules('unloaded_modules'))
        invoke('observer_stop',stop);invoke('final_event_drain',drain);invoke('final_source_preflight',preflight)
    except BaseException as error:
        primary=error
        if not errors:errors.append({'operation':next((op for op in OPS if counts[op]['pending']),OPS[0]),'error':_error(error)})
    finally:
        # Cleanup attempts only unreached operations; never retry a failed API.
        if state['handle_live'] and not state['unload_attempted']:
            try:invoke('fixture_unload',unload)
            except BaseException:
                if not state['unload_attempted']:
                    try:unload()
                    except BaseException as error:errors.append({'operation':'fixture_unload','error':_error(error)})
        if state['start_attempted'] and counts['observer_stop']['attempted']==0:
            try:invoke('observer_stop',stop)
            except BaseException:
                if not state.get('stop_attempted',False):
                    try:stop()
                    except BaseException as error:errors.append({'operation':'observer_stop','error':_error(error)})
        if counts['observer_stop']['returned']==1 and counts['final_event_drain']['attempted']==0:
            try:invoke('final_event_drain',drain)
            except BaseException:pass
        phase_stream.close()
    startup=c.seal(startup);startup_raw=(c.canonical(startup)+'\n').encode('ascii')
    c.require(len(startup_raw)<=c.MAX_JSON,'bounded complete original startup inventory')
    with (output/'OBSERVER_STARTUP_INVENTORY.json').open('xb') as stream:stream.write(startup_raw);stream.flush();os.fsync(stream.fileno())
    c.require(c.read_bytes(output/'OBSERVER_STARTUP_INVENTORY.json')==startup_raw,'exact durable original startup inventory')
    complete=primary is None and not errors and all(row['returned']==1 for row in counts.values())
    result={'schema_version':1,'kind':'OBSERVER_ONLY_WORKER_RESULT','scope':c.SCOPE,
        'status':'OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT' if complete else 'OBSERVER_OPERATIONS_FAILED',
        'source_commit':value['expected_head'],'config_sha256':value['anchors']['expected_config_file'],
        **{key:state[key] for key in ('owned_identity','abi_binding','fixture_handle','initial_modules','loaded_modules','unloaded_modules','events','final_collector_snapshot')},
        'phase_counts':counts,'errors':errors,'fixture_only':value['fixture_only'],'native_PASS':False,'native_authority_issued':False}
    result=c.seal(result);raw=(c.canonical(result)+'\n').encode('ascii');c.require(len(raw)<=c.MAX_JSON,'bounded full observer-only result')
    with (output/'OBSERVER_WORKER_RESULT.json').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    c.require(c.read_bytes(output/'OBSERVER_WORKER_RESULT.json')==raw,'exact durable observer-only result readback')
    return result

def _CPU_run_fixture(context,contract,transport,provider,output,*,_capability,clock=time.monotonic,bootstrap_actor=None):
    contract._check_CPU_capability(_capability);value=contract.context_value(context)
    contract.require(value['fixture_only'] is True and value['native_PASS'] is value['native_authority_issued'] is False,'private separate observer-only CPU context only')
    if bootstrap_actor is None:
        def bootstrap_actor():return {'kind':'MEASURED_BOOTSTRAP_IMPORTS_V1','imports':['ctypes','ctypes.wintypes','_ctypes'],
            'modules':[],'unknown_modules':[],'fixture_only':True,'namespace_execution_proven':False}
    return _operation_body(context,contract,lambda:transport,lambda:provider,output,clock,
        lambda:contract._CPU_revalidate_context(context,_capability=_capability),bootstrap_actor)

class _NativeProvider:
    def __init__(self,context,contract,owner):
        contract.native_gate();value=contract.context_value(context);contract.require(value['fixture_only'] is False,'CPU observer-only context has no native provider authority')
        contract.require(type(owner) is dict and set(owner)=={'worker_pid','root_worker_pid','supervisor_pid','supervisor_create_time','job_name','config_sha256'} and type(owner['worker_pid']) is type(owner['root_worker_pid']) is type(owner['supervisor_pid']) is int and owner['worker_pid']==owner['root_worker_pid']>0 and owner['supervisor_pid']>0 and owner['worker_pid']!=owner['supervisor_pid'] and type(owner['supervisor_create_time']) in (int,float) and math.isfinite(owner['supervisor_create_time']) and type(owner['job_name']) is str and owner['job_name'] and owner['config_sha256']==value['anchors']['expected_config_file'],'exact original observer-only ownership receipt')
        self.C=contract;self.owner=owner;self.value=value
        for path in (r'C:\Windows\System32\kernel32.dll',r'C:\Windows\System32\psapi.dll'):
            contract.require(path in value['runtime']['files'],'native provider system DLL selected in original map');contract.file_descriptor(path,value['runtime']['files'][path])
        import sys
        contract.require(all(name in sys.modules for name in ('ctypes','ctypes.wintypes','_ctypes')),'qualified preloaded ctypes before native provider')
        import ctypes
        self.c=ctypes;self.k=ctypes.WinDLL(r'C:\Windows\System32\kernel32.dll',use_last_error=True,winmode=0x800)
        self.p=ctypes.WinDLL(r'C:\Windows\System32\psapi.dll',use_last_error=True,winmode=0x800)
    def owned_identity(self):
        c=self.c;k=self.k
        class FT(c.Structure):_fields_=[('low',c.c_uint32),('high',c.c_uint32)]
        k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=c.c_void_p
        k.GetCurrentProcessId.argtypes=[];k.GetCurrentProcessId.restype=c.c_uint32
        k.GetProcessTimes.argtypes=[c.c_void_p]+[c.POINTER(FT)]*4;k.GetProcessTimes.restype=c.c_int
        created=FT();others=[FT(),FT(),FT()];self.C.require(k.GetProcessTimes(k.GetCurrentProcess(),c.byref(created),*[c.byref(x) for x in others]),'actual current observer-only FILETIME')
        birth={'pid':int(k.GetCurrentProcessId()),'creation_filetime_100ns':(int(created.high)<<32)|int(created.low)}
        k.OpenJobObjectW.argtypes=[c.c_uint32,c.c_int,c.c_wchar_p];k.OpenJobObjectW.restype=c.c_void_p
        k.QueryInformationJobObject.argtypes=[c.c_void_p,c.c_int,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32)];k.QueryInformationJobObject.restype=c.c_int
        k.CloseHandle.argtypes=[c.c_void_p];k.CloseHandle.restype=c.c_int
        handle=k.OpenJobObjectW(4,0,self.owner['job_name']);self.C.require(handle,'original observer-only Job handle')
        try:
            class PIDS(c.Structure):_fields_=[('assigned',c.c_uint32),('count',c.c_uint32),('pids',c.c_size_t*256)]
            rows=PIDS();returned=c.c_uint32();self.C.require(k.QueryInformationJobObject(handle,3,c.byref(rows),c.sizeof(rows),c.byref(returned)),'bounded actual original Job membership')
            pids=[int(v) for v in rows.pids[:rows.count]]
            self.C.require(0<rows.count<=256 and rows.assigned==rows.count and len(set(pids))==len(pids) and birth['pid'] in pids and self.owner['worker_pid'] in pids and self.owner['supervisor_pid'] not in pids,'current observer-only interpreter belongs to original owned Job')
        finally:self.C.require(k.CloseHandle(handle),'original Job query handle closed')
        return birth
    def modules(self):
        c=self.c;k=self.k;p=self.p
        k.GetCurrentProcess.argtypes=[];k.GetCurrentProcess.restype=c.c_void_p
        p.EnumProcessModules.argtypes=[c.c_void_p,c.POINTER(c.c_void_p),c.c_uint32,c.POINTER(c.c_uint32)];p.EnumProcessModules.restype=c.c_int
        p.GetModuleFileNameExW.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p,c.c_uint32];p.GetModuleFileNameExW.restype=c.c_uint32
        array=(c.c_void_p*4096)();needed=c.c_uint32();process=k.GetCurrentProcess()
        self.C.require(p.EnumProcessModules(process,array,c.sizeof(array),c.byref(needed)) and 0<needed.value<=c.sizeof(array) and needed.value%c.sizeof(c.c_void_p)==0,'complete bounded current module enumeration')
        out=[]
        for handle in array[:needed.value//c.sizeof(c.c_void_p)]:
            text=c.create_unicode_buffer(4096);length=p.GetModuleFileNameExW(process,handle,text,len(text))
            self.C.require(0<length<len(text)-1,'complete nontruncated module path')
            raw=text.value;layout=self.value['runtime']['stdlib_layout'];logical=Path(layout['logical_base_prefix'])
            path=Path(raw)
            if path.is_relative_to(logical):raw=str(Path(layout['physical_base_prefix'])/path.relative_to(logical))
            matches=[path for path in self.value['runtime']['files'] if path.casefold()==raw.casefold()]
            self.C.require(len(matches)<=1,'unambiguous original module byte identity')
            # Unknown originals must survive in the raw baseline before the
            # shared body refuses fixture work; they are never hashed here.
            out.append({'path':matches[0] if matches else raw,'module_handle':int(handle)})
        return {'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':sorted(out,key=lambda row:row['path'])}
    def load_fixture(self,path):
        self.k.LoadLibraryExW.argtypes=[self.c.c_wchar_p,self.c.c_void_p,self.c.c_uint32];self.k.LoadLibraryExW.restype=self.c.c_void_p
        return int(self.k.LoadLibraryExW(path,None,0x800) or 0)
    def unload_fixture(self,handle):
        self.k.FreeLibrary.argtypes=[self.c.c_void_p];self.k.FreeLibrary.restype=self.c.c_int
        return bool(self.k.FreeLibrary(handle))

def _native_transport(context,contract):
    contract.native_gate();value=contract.context_value(context);selected=value['config']['observer_registration']['adapter_source']
    contract.file_descriptor(selected['path'],selected['descriptor']);raw=contract.read_bytes(selected['path'])
    contract.require(hashlib.sha256(raw).hexdigest()==selected['descriptor']['sha256'],'original selected collector adapter source')
    module=ModuleType('observer_only_selected_adapter');module.__file__=selected['path']
    exec(compile(raw,selected['path'],'exec'),module.__dict__);module.__compiled_source_descriptor__=dict(selected['descriptor'])
    return module._Win32Transport(context,contract)

def _native_bootstrap(context,contract):
    """Measure a first explicit import after approval and complete byte checks.

    Origin/selected-byte joins do not prove the execution history of preloaded
    Python namespaces or defend against malicious arbitrary same-process code.
    """
    contract.native_gate();value=contract.context_value(context)
    contract.require(value['fixture_only'] is False,'CPU cannot import native bootstrap')
    import sys
    from importlib.machinery import PathFinder
    policy=value['runtime']['cache_policy'];selected=policy['allowed_bootstrap_modules'];layout=value['runtime']['stdlib_layout']
    def normalized(path):return str(contract.qualified_path(path,layout))
    # Find exact admitted source/extension origins before executing any of the
    # three explicitly registered bootstrap imports.
    for name in policy['imports']:
        parent=[str(Path(selected['ctypes']['path']).parent)] if name=='ctypes.wintypes' else None
        spec=PathFinder.find_spec(name,parent)
        contract.require(spec is not None and type(spec.origin) is str and normalized(spec.origin)==normalized(selected[name]['path']),'byte-admitted bootstrap resolver origin before explicit import')
        contract.file_descriptor(selected[name]['path'],value['runtime']['files'][selected[name]['path']],layout=layout)
    __import__('ctypes');__import__('ctypes.wintypes');__import__('_ctypes')
    contract.require(len(sys.modules)<=4096,'bounded original startup namespace population')
    rows=[];unknown=[]
    for name,module in sorted(tuple(sys.modules.items())):
        if module is None:continue
        namespace=vars(module);spec=namespace.get('__spec__');origin=getattr(spec,'origin',None) or namespace.get('__file__')
        classification='builtin_or_frozen' if origin in ('built-in','frozen') else 'source_or_extension'
        row={'name':name,'origin':origin if type(origin) is str else None,'classification':classification,'source_sha256':None,'error':None}
        if classification!='builtin_or_frozen':
            entry=selected.get(name)
            if entry is None or type(origin) is not str:unknown.append(row)
            else:
                try:
                    actual=normalized(origin);expected=normalized(entry['path'])
                    contract.require(actual==expected,'original measured bootstrap module origin')
                    contract.file_descriptor(entry['path'],{**value['config']['source_files'],**value['runtime']['files']}[entry['path']],layout=layout)
                    row['origin']=actual;row['source_sha256']=entry['source_sha256']
                except (OSError,ValueError) as error:row['error']=_error(error);unknown.append(row)
        rows.append(row)
    return {'kind':'MEASURED_BOOTSTRAP_IMPORTS_V1','imports':list(policy['imports']),'modules':rows,
        'unknown_modules':unknown,'fixture_only':False,'namespace_execution_proven':False}

def main():
    C.native_gate()
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    external=C.parse(C.read_bytes(args.output_dir/'OWNED_EXTERNAL_ANCHORS.json'))
    C.require(type(external) is dict and set(external)=={'config_path','anchors'} and external['config_path']==str(args.config.absolute()),'original bounded observer-only external anchors')
    context=C.validate_context(args.config,**external['anchors']);owner=C.parse(C.read_bytes(args.output_dir/'supervisor_ownership.json'))
    C.require(owner['config_sha256']==external['anchors']['expected_config_file'],'original observer-only owner/config receipt')
    value=C.context_value(context)
    for key,setting in value['config']['offline_environment'].items():os.environ[key]=setting
    result=_operation_body(context,C,lambda:_native_transport(context,C),lambda:_NativeProvider(context,C,owner),args.output_dir,time.monotonic,lambda:C.revalidate_context(context),lambda:_native_bootstrap(context,C))
    return 0 if result['status']=='OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT' else 2

if __name__=='__main__':raise SystemExit(main())
