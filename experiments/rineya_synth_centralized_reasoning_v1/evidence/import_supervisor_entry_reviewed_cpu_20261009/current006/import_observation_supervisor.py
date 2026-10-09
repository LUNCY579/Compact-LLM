"""Separate import-only adapter to the unchanged owned supervisor lifecycle."""
from pathlib import Path
from types import ModuleType
from datetime import datetime,timezone
import ast,builtins,hashlib,importlib,importlib.machinery,importlib.util,json,os,sys,time
HERE=Path(__file__).resolve().parent
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
ORIGINAL_SUPERVISOR_PIN='edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691'
def require(v,m):
    if not v:raise ValueError(m)
def _adapt_supervise(raw,source_path,worker_path):
    require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SUPERVISOR_PIN,'unchanged original supervisor bytes before source adaptation')
    tree=ast.parse(raw,str(source_path));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise');changes=[]
    class Scope(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='native_worker.py':changes.append('worker_path');return ast.copy_location(ast.Constant(Path(worker_path).name),node)
            if node.value=='mechanical_native_qualification_only':changes.append('scope');return ast.copy_location(ast.Constant(SCOPE),node)
            return node
    function=Scope().visit(function);require(changes==['worker_path','scope','worker_path'],'exact original two worker/one purpose adaptations')
    counts=[]
    class NoSite(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='argv':
                require(isinstance(node.value,ast.List) and len(node.value.elts)==8 and node.value.elts[1].value=='-I' and node.value.elts[2].value=='-B','exact original isolated argv');node.value.elts.insert(2,ast.Constant('-S'));counts.append('flag')
            return node
        def visit_Subscript(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='argv' and isinstance(node.ctx,ast.Store):
                require(isinstance(node.slice,ast.Constant) and node.slice.value in (3,5),'exact original worker/config stores');counts.append(node.slice.value);node.slice=ast.Constant(node.slice.value+1)
            return self.generic_visit(node)
    function=NoSite().visit(function);require(counts==['flag',3,5],'exact no-site flag and 4/6 stores');ast.fix_missing_locations(function)
    return ast.Module(body=[function],type_ignores=[]),{'purpose_worker_changes':changes,'argv_changes':counts,'total_changes':6}
def _preflight(context,contract,config_path):
    value=contract.revalidate_context(context);config=value['config'];require(Path(config_path).absolute()==Path(value['config_path']).absolute(),'exact selected import config path')
    worker=Path(config['main_source']);require(worker.name=='import_observation_worker.py' and str(worker) in config['source_files'],'own registered import worker source')
    return {'config_sha256':value['anchors']['expected_config_file'],'config_path':value['config_path'],'worker_path':str(worker),
        'authorization':{'scope':SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False},
        'resources':config['resources'],'source_commit':value['expected_head'],'profile_sha256':value['registration']['profile_sha256'],
        'source_inventory_sha256':contract.digest(config['source_files']),'runtime_inventory_sha256':contract.digest(value['runtime']['files'])}

class _SupervisorFinder:
    """Finite selected parent dependencies; never the worker import observer."""
    def __init__(self,value,contract):
        self.value=value;self.contract=contract
        policy=value['runtime']['cache_policy'];files=value['runtime']['files'];self.rows={}
        for name,row in policy['preloaded_bootstrap_modules'].items():
            descriptor=files[row['path']];cache=None if row['cache_path'] is None else files[row['cache_path']]
            require(descriptor['role'] in ('python_source','extension_binary') and descriptor['sha256']==row['source_sha256'] and (cache is None and row['cache_sha256'] is None or cache is not None and cache['role']=='qualified_bytecode_cache' and cache['sha256']==row['cache_sha256']),'historical parent source/cache pins join selected inventory')
            self.rows[name]={**row,'descriptor':descriptor,'cache_descriptor':cache}
        self.rows.update(policy['additional_preload_sources'])
        for name in ('native_process','native_nvml'):
            path=str(contract.PACKAGE/(name+'.py'));self.rows[name]={'path':path,'descriptor':value['config']['source_files'][path],'cache_path':None,'cache_descriptor':None}
        base=Path(value['runtime']['stdlib_layout']['physical_base_prefix'])
        for name,path,role in (('uuid',base/'Lib/uuid.py','python_source'),('_uuid',base/'DLLs/_uuid.pyd','extension_binary')):
            text=str(path);d=value['runtime']['files'][text];require(d['role']==role,'selected parent UUID dependency role');self.rows[name]={'path':text,'descriptor':d,'cache_path':None,'cache_descriptor':None}
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('pwd','org'):
            require(importlib.machinery.BuiltinImporter.find_spec(fullname) is None and importlib.machinery.FrozenImporter.find_spec(fullname) is None and importlib.machinery.PathFinder.find_spec(fullname,path,target) is None,'available unselected parent optional module refused');raise ModuleNotFoundError("No module named "+repr(fullname),name=fullname)
        if importlib.machinery.BuiltinImporter.find_spec(fullname) or importlib.machinery.FrozenImporter.find_spec(fullname):return None
        require(fullname in self.rows,'unknown parent dependency import refused')
        row=self.rows[fullname];C=self.contract;layout=self.value['runtime']['stdlib_layout'];spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        require(spec is not None and spec.origin is not None and str(C.qualified_path(spec.origin,layout=layout))==row['path'],'parent dependency resolution matches selected source')
        C.file_descriptor(row['path'],row['descriptor'],layout=layout)
        if row['cache_path'] is not None:C.file_descriptor(row['cache_path'],row['cache_descriptor'],layout=layout)
        if fullname in ('native_process','native_nvml','uuid'):
            # Package helpers compile the checked source, never an unselected pyc.
            class SelectedSource(importlib.machinery.SourceFileLoader):
                def get_code(self,name):
                    require(name==fullname,'exact parent helper loader name');C.file_descriptor(row['path'],row['descriptor']);return compile(C.read_bytes(row['path']),row['path'],'exec')
            spec.loader=SelectedSource(fullname,row['path'])
        return spec

def _supervisor_dependency_scope(value,contract):
    contract.stage_native_gate()
    roots=[str(contract.qualified_path(contract.PACKAGE)),str(contract.qualified_path(contract.SITE))]
    require(all(Path(p).is_dir() for p in roots),'physical selected parent PACKAGE/SITE roots')
    path_container=sys.path;before=tuple(path_container);meta_container=sys.meta_path;meta_before=tuple(meta_container)
    require(list(before)==value['runtime']['cache_policy']['interpreter_profile']['path'],'selected no-site parent initial roots')
    require(all(name not in sys.modules for name in ('psutil','native_process','native_nvml','pwd','org')),'fresh parent dependency identities')
    finder=_SupervisorFinder(value,contract);sys.path.extend(roots);after=tuple(sys.path);sys.meta_path.insert(0,finder)
    def finish():
        stable=sys.path is path_container and tuple(sys.path)==after and sys.meta_path is meta_container and tuple(sys.meta_path)==(finder,*meta_before)
        sys.path=path_container;path_container[:]=before;sys.meta_path=meta_container;meta_container[:]=meta_before
        require(stable,'owned parent search roots/finder unchanged before restore')
    return finish

def _selected_monitor(value,contract,original):
    path=contract.PACKAGE/'short_ram_monitor.py';d=value['config']['source_files'][str(path)];raw=contract.read_bytes(path);require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'registered original full-RSS monitor source')
    tree=ast.parse(raw,str(path));removed=[];body=[]
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='sys.path.insert':
            require(ast.unparse(node.value)=='sys.path.insert(0, str(PACKAGE))','exact original monitor path insertion');removed.append('path_insert')
        elif isinstance(node,ast.Import) and any(a.name=='native_supervisor' for a in node.names):
            require(len(node.names)==1 and node.names[0].asname=='shared','exact original monitor shared import');removed.append('shared_import');body.append(node)
        else:body.append(node)
    require(removed==['path_insert','shared_import'],'only two parent monitor source adaptations');tree.body=body
    def selected_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='native_supervisor':
            require(level==0 and not fromlist,'exact original shared module import');return original
        return builtins.__import__(name,globals,locals,fromlist,level)
    # Retain the import symbol in compilation to preserve original method code.
    module=ModuleType('owned_import_selected_monitor');module.__file__=str(path);module.__dict__['__builtins__']={**vars(builtins),'__import__':selected_import};exec(compile(tree,str(path),'exec'),module.__dict__)
    require(module.shared is original and module.ShortRamMonitor.__bases__==(original._Monitor,),'exact original monitor superclass/namespace')
    return module

def _verify_parent_preloads(value,contract):
    path=contract.PACKAGE/'import_bootstrap.py';d=value['config']['source_files'][str(path)];raw=contract.read_bytes(path);require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'selected existing preload verifier source')
    module=ModuleType('owned_import_parent_preload_verifier');module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__)
    importlib.import_module('psutil')
    files={**value['runtime']['files'],**value['config']['source_files']};rows=value['runtime']['cache_policy']['additional_preload_sources']
    for name in ('psutil','psutil._common','psutil._ntuples','psutil._pswindows','psutil._psutil_windows'):
        require(name in sys.modules and name in rows,'complete selected parent psutil family')
    for name,row in rows.items():
        if name in sys.modules:module.verify_current_module(name,sys.modules[name],row,contract,files,value['runtime']['stdlib_layout'])

def supervise_registered(config_path,output_dir,*,contract,grant_reference=None,owner_receipt_reference=None,**anchors):
    require(contract.SCOPE==SCOPE,'exact separate import contract')
    binding={'config_path':str(Path(config_path).absolute()),'anchors':anchors};contract.enroll_native_gate(grant_reference,owner_receipt_reference,binding);contract.stage_native_gate();context=contract.validate_native_manifest(config_path,**anchors);value=contract.context_value(context)
    finish=_supervisor_dependency_scope(value,contract)
    try:
        source=contract.PACKAGE/'native_supervisor.py';descriptor=value['config']['source_files'][str(source)];raw=contract.read_bytes(source);require(len(raw)==descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'unchanged registered original lifecycle source')
        original=ModuleType('owned_import_original_supervisor');original.__file__=str(source);exec(compile(raw,str(source),'exec'),original.__dict__)
        worker=Path(value['config']['main_source']);require(worker.parent==contract.PACKAGE,'registered public import worker location');tree,proof=_adapt_supervise(raw,source,worker)
        namespace=dict(original.__dict__);original_new=original._new_output
        def new_output(request):
            output=original_new(request);original._json_new(output/'OWNED_EXTERNAL_ANCHORS.json',{'schema_version':2,**binding,'grant_reference':grant_reference,'owner_receipt_reference':owner_receipt_reference});return output
        namespace['_new_output']=new_output;namespace['_preflight']=lambda request:_preflight(context,contract,request)
        monitor=_selected_monitor(value,contract,original);namespace['_Monitor']=monitor.ShortRamMonitor
        _verify_parent_preloads(value,contract)
        exec(compile(tree,str(source),'exec'),namespace);return namespace['supervise'](config_path,output_dir)
    finally:finish()

def preserve_terminal(output,target,*,contract,context,death_verifier=None,_CPU_capability=None):
    """Copy exact finite raw denominator, including failed/no worker result."""
    value=contract.revalidate_context(context)
    if _CPU_capability is not None:contract._check_CPU(_CPU_capability);require(value.get('fixture_only') is True and callable(death_verifier),'explicit CPU-only death fixture')
    else:
        require(death_verifier is None and value.get('fixture_only') is False,'native preservation never accepts caller death proof')
        source=contract.PACKAGE/'native_supervisor.py';d=value['config']['source_files'][str(source)];raw=contract.read_bytes(source);require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SUPERVISOR_PIN==d['sha256'],'actual selected original death verifier source');module=ModuleType('owned_import_current_death_verifier');module.__file__=str(source);exec(compile(raw,str(source),'exec'),module.__dict__);death_verifier=module._pid_alive
    output=Path(output).absolute();target=Path(target).absolute();require(output.parent==contract.PACKAGE/'artifacts' and output.is_dir() and output.resolve()==output and target.parent==contract.PACKAGE/'evidence' and target.parent.resolve()==target.parent and not target.exists(),'owned output/new evidence locations')
    rawsup=contract.read_bytes(output/'SUPERVISOR_RECEIPT.json');supervisor=contract.parse(rawsup);require(supervisor['config_sha256']==value['anchors']['expected_config_file'] and supervisor['output_dir']==str(output) and supervisor['job_closed_kill_on_close'] is supervisor['sampler_threads_terminal_verified'] is supervisor['gpu_telemetry_cleanup_verified'] is supervisor['partial_outputs_frozen'] is True and supervisor['owned_worker_identity_currently_alive'] is False and supervisor['unjoined_sampler_threads']==[],'original actual terminal cleanup/frozen denominator')
    resources=contract.parse(contract.read_bytes(output/'RESOURCE_OBSERVATIONS.json'));births={}
    def visit(item):
        if type(item) is dict:
            for born in item.get('owned_process_identities',[]):
                if type(born) is dict and 'pid' in born and 'creation_filetime_100ns' in born:
                    selected={k:born[k] for k in ('pid','creation_filetime_100ns')};require(all(type(v) is int and v>0 for v in selected.values()),'typed original owned birth');births[tuple(selected.values())]=selected
            for child in item.values():visit(child)
        elif type(item) is list:
            for child in item:visit(child)
    visit(resources);owner=supervisor['worker_identity'];born={k:owner[k] for k in ('pid','creation_filetime_100ns')};births[tuple(born.values())]=born
    deaths=[]
    for born in births.values():require(death_verifier(dict(born)) is False,'all current exact owned births dead');deaths.append({'identity':born,'currently_alive':False})
    names=sorted(p.name for p in output.iterdir());require(len(names)<=4096 and all((output/n).is_file() for n in names) and sum((output/n).stat().st_size for n in names)<=128*1024**2,'bounded complete partial raw roster');raw={n:contract.read_bytes(output/n) for n in names};partial={Path(v['path']).name:v for v in supervisor['partial_outputs']};require(set(partial)==set(names)-{'SUPERVISOR_RECEIPT.json'} and all(partial[n]['bytes']==len(raw[n]) and partial[n]['sha256']==hashlib.sha256(raw[n]).hexdigest() for n in partial),'original SUP partial byte denominator')
    lease=Path(supervisor['lease_path']);lease_names=('reservation.json','worker.json','resume.json','terminal.json');require(set(p.name for p in lease.iterdir())==set(lease_names) and lease.parent==contract.PACKAGE/'artifacts/_native_gpu_leases' and lease.resolve()==lease,'exact original four-file owned lease');leases={n:contract.read_bytes(lease/n) for n in lease_names};require(leases['terminal.json']==rawsup,'actual lease terminal bytes')
    target.mkdir();(target/'source_snapshots').mkdir();rawmap={};leasemap={};snapshots={}
    def copy(path,data):
        with path.open('xb') as f:require(f.write(data)==len(data),'complete exclusive preservation write');f.flush();os.fsync(f.fileno())
        require(contract.read_bytes(path)==data,'preserved byte readback');return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    for n,data in raw.items():rawmap[n]=copy(target/n,data)
    for n,data in leases.items():leasemap[n]={'source_path':str(lease/n),'path':'lease_'+n,**copy(target/('lease_'+n),data)}
    for text,d in value['config']['source_files'].items():
        require(Path(text).suffix.lower() in ('.py','.json','.md','.c','.h') and d['role'] in ('python_source','source_json','source_markdown','native_observer_source'),'explicit source-only preservation role')
        contract.file_descriptor(text,d);data=contract.read_bytes(text);name=hashlib.sha256(text.encode()).hexdigest()+Path(text).suffix;snapshots[text]={'path':'source_snapshots/'+name,'role':d['role'],**copy(target/'source_snapshots'/name,data)}
    require(sorted(p.name for p in output.iterdir())==names and all(contract.read_bytes(output/n)==raw[n] for n in names) and all(contract.read_bytes(lease/n)==leases[n] for n in lease_names),'original raw/lease bytes unchanged through copy')
    for text,d in value['config']['source_files'].items():contract.file_descriptor(text,d);require(contract.read_bytes(text)==contract.read_bytes(target/snapshots[text]['path']),'original/source-copy stable final bytes')
    require(set(p.name for p in target.iterdir())==set(names)|{'source_snapshots'}|{'lease_'+n for n in lease_names} and set(p.name for p in (target/'source_snapshots').iterdir())=={Path(x['path']).name for x in snapshots.values()},'complete exact copied raw/lease/source roster');contract.revalidate_context(context)
    record={'kind':'PRESERVED_OWNED_TORCH_IMPORT_OBSERVATION','scope':SCOPE,'config_body_sha256':value['config']['sha256'],'raw_files':rawmap,'lease_files':leasemap,'source_files':value['config']['source_files'],'source_snapshot_files':snapshots,'current_death_observation':{'observed_monotonic':time.monotonic(),'observed_utc':datetime.now(timezone.utc).isoformat(),'observations':deaths,'supervisor_file_sha256':hashlib.sha256(rawsup).hexdigest()},'worker_result_present':any('WORKER_RESULT' in n or n=='worker_result.json' for n in names),'native_authority_issued':False,'native_PASS':False};record['sha256']=contract.digest(record);copy(target/'PRESERVATION.json',(contract.canonical(record)+'\n').encode());return record
