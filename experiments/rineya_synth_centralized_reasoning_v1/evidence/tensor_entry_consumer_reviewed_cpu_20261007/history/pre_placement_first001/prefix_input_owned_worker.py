"""Separate owned caller. CPU simulation cannot be converted to native work."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent
COMPILED_SOURCE_OBSERVATIONS={}
OWNED_TOPOLOGY_OBSERVATIONS={}
def exact_module(path,pin=None,*,_meter=None):
    path=Path(path).absolute()
    if path.resolve()!=path:raise ValueError('exact module physical path')
    for part in (path,*path.parents):
        info=part.lstat()
        if part.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('module reparse alias')
    before=path.lstat()
    if not path.is_file() or path.is_symlink() or not 0<=before.st_size<=16*1024**2:raise ValueError('bounded regular source module')
    def selected_read():
        with path.open('rb') as stream:return stream.read(before.st_size+1)
    raw=selected_read() if _meter is None else _meter.invoke('worker.exact_module_read',selected_read,category='source',byte_count=len)
    after=path.lstat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or (pin is not None and hashlib.sha256(raw).hexdigest()!=pin):raise ValueError('exact stable source module')
    spec=importlib.util.spec_from_file_location('prefix_owned_exact_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    if path.name=='prefix_native_pipeline.py':module.__dict__['__owned_contract__']=C
    exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__)
    observation={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'loader_kind':'explicit_selected_source_compile','current_module_execution_observed':True}
    module.__compiled_source_descriptor__=observation;COMPILED_SOURCE_OBSERVATIONS[str(path)]=observation;return module
ENTRY_PIPELINE_PIN='9efb6f15444a6449734fe85d5eddd72d3cefb2a76c0b7af46264b44ed68cbb2b'
CONTRACT_PIN='0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263'
C=None if __name__=='__main__' else exact_module(HERE/'prefix_tensor_contract.py',CONTRACT_PIN)

def source_core():
    contract=exact_module(C.CORE/'contract.py',C.CORE_PINS['contract.py'])
    path=C.CORE/'pipeline.py';raw=C.read_bytes(path);C.require(hashlib.sha256(raw).hexdigest()==C.CORE_PINS['pipeline.py'],'unchanged accepted CPU pipeline')
    tree=ast.parse(raw.decode('utf-8'),str(path));removed=[];kept=[]
    # Four module-initialization statements are replaced with the exact source
    # contract object. Function/class bodies and CPU status literals are intact.
    for node in tree.body:
        name=node.targets[0].id if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) else None
        if name in ('_spec','C') or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and
           ((isinstance(node.value.func.value,ast.Attribute) and node.value.func.value.attr=='loader') or (isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='C' and node.value.func.attr=='require'))):removed.append(node)
        else:kept.append(node)
    C.require(len(removed)==4,'exact four source-loader adaptations')
    namespace={'__file__':str(path),'__name__':'accepted_prefix_core_source_only','C':contract};exec(compile(ast.Module(body=kept,type_ignores=[]),str(path),'exec'),namespace)
    frame=C.PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py'
    contract.frame_helper=lambda:exact_module(frame,'253caa71942c300c225abef9f33d09f2ac28c4e6f169570ecbea09b926817d0b')
    return namespace

def original_guard(host,owner,started):
    path=C.PACKAGE/'inference_qual_worker.py';raw=C.read_bytes(path)
    C.require(hashlib.sha256(raw).hexdigest()=='1172838c31c1b9440337707826573012e06f9a6a26b606a62a43976d19e99f1e','unchanged original resource guard source')
    tree=ast.parse(raw.decode('utf-8'),str(path));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_guard')
    ns={'Path':Path,'time':time,'require':C.require,'GIB':1024**3,'owned_job_resources':host.owned_resources,'read_heartbeat':host.heartbeat}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    view=[None]
    guard=ns['make_guard'](host.output,{'job_name':host.job_name,'supervisor_pid':host.supervisor_pid},started,lambda:view[0],
       owned_resources=host.owned_resources,heartbeat_reader=host.heartbeat,available=host.available,monotonic=host.clock,wall=host.wall)
    def invoke(stage,torch_view=None):
        view[0]=torch_view;host.guard_sample(stage);return guard(stage)
    return invoke

def run_injected(manifest,admission,owner,api,host,output):
    C.verify_cpu_admission(admission,manifest,owner);C.require(Path(output).absolute()==Path(host.output).absolute(),'exact injected owned output')
    core=source_core();bound=dict(api);original_preflight=bound['config_preflight'];original_ownership=bound['ownership_verify']
    initial_manifest=C.canonical(manifest)
    def preflight():
        C.require(C.canonical(manifest)==initial_manifest,'immutable injected original manifest');C.validate_fixture(manifest);actual=original_preflight()
        C.require(type(actual) is dict and actual,'actual nonempty preflight metadata');actual=C.parse(C.canonical(actual))
        for path,descriptor in manifest['source_pins'].items():C.file_descriptor(path,descriptor)
        return {'scope':C.CPU_SCOPE,'manifest_sha256':manifest['sha256'],'worker_identity':C.identity(owner),'permission_sha256':C.digest(manifest['permission']),'actual_preflight_sha256':C.digest(actual),'actual_preflight':actual}
    def ownership():
        C.verify_cpu_admission(admission,manifest,owner);result=original_ownership();C.require(result is True and host.membership(owner) is True,'assigned owned birth before framework import');return {'identity':owner,'assigned_before_resume':True,'fixture_only':True}
    bound['config_preflight']=preflight;bound['ownership_verify']=ownership
    return core['run_cpu'](bound,original_guard(host,owner,host.started),output,source_pins={str(C.CORE/name):pin for name,pin in C.CORE_PINS.items()},runtime_pins={})

def real_torch_bindings(context,owner,output,**untrusted):
    C.stage_native_gate();value=C.context_value(context);C.require(value.get('fixture_only',False) is False,'CPU context cannot create real native bindings')
    C.require(False,'native bindings require actual owned main entry; caller callbacks refused')

def _installed_framework_boundary(value,requested=None):
    C.require(False,'native library selector requires actual owned main entry; caller callbacks refused')

def _CPU_installed_framework_boundary_fixture(value,requested=None,*,_capability):
    C._check_CPU_native_context_fixture(_capability)
    C.require(value.get('fixture_only') is True and value.get('native_authority_issued') is False,'CPU installed-selector fixture only')
    return _installed_framework_boundary_body(value,requested,_CPU_capability=_capability)

def _installed_framework_boundary_body(value,requested=None,*,_CPU_capability=None):
    import sys,types
    selected=str(C.PACKAGE/'prefix_registered_loader.py');expected=value['config']['source_files'][selected]
    matches=[]
    for finder in sys.meta_path:
        method=getattr(finder,'framework_import',None)
        if type(method) is not types.MethodType:continue
        function=method.__func__
        if function.__code__.co_filename!=selected:continue
        descriptor=getattr(finder,'selected_loader_source',None)
        if type(descriptor) is not dict or descriptor.get('sha256')!=expected['sha256'] or descriptor.get('bytes')!=expected['bytes']:continue
        if getattr(finder,'selected_context_projection',None)!=C.canonical(value):continue
        guard=getattr(finder,'library_guard',None)
        if guard is None or (_CPU_capability is None and getattr(finder,'fixture_only',False)):continue
        C.require(guard is getattr(finder,'selected_library_guard',None),'actual installed selected library guard identity')
        method=getattr(guard,'framework_import',None);helper=str(C.PACKAGE/'prefix_library_policy.py')
        C.require(type(method) is types.MethodType and method.__func__ is type(guard).framework_import and method.__func__.__code__.co_filename==helper,'selected library guard method/source identity')
        actual=method.__func__.__globals__.get('__compiled_source_descriptor__');wanted=value['config']['source_files'].get(helper)
        C.require(type(actual) is dict and wanted is not None and actual==wanted,'selected library guard compiled descriptor')
        C.require(getattr(guard,'active',False) is True and getattr(guard,'failed',True) is False and getattr(guard,'closed',True) is False,'active unfailed unclosed library guard before native import')
        C.require(function is type(finder).framework_import,'actual installed selected finder method')
        # Authenticate the closure validator's executable code against the
        # selected bytes before consuming its source-owned installation records.
        raw=C.read_bytes(selected)
        C.require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],'exact selected installer bytes before validator authentication')
        tree=ast.parse(raw,selected);code=compile(tree,selected,'exec')
        factory=next(item for item in code.co_consts if type(item) is types.CodeType and item.co_name=='_installation_registry_factory')
        expected_check=next(item for item in factory.co_consts if type(item) is types.CodeType and item.co_name=='check')
        validator=function.__globals__.get('_check_installed_record')
        finder_class=next(item for item in code.co_consts if type(item) is types.CodeType and item.co_name=='RegisteredFinder')
        expected_method=next(item for item in finder_class.co_consts if type(item) is types.CodeType and item.co_name=='framework_import')
        C.require(function.__code__==expected_method and function.__globals__.get('RegisteredFinder') is type(finder),'exact selected finder executable code and class globals')
        C.require(type(validator) is types.FunctionType and validator.__code__==expected_check and validator.__globals__ is function.__globals__,'exact selected installer validator executable code and globals')
        helper_raw=C.read_bytes(helper)
        C.require(len(helper_raw)==wanted['bytes'] and hashlib.sha256(helper_raw).hexdigest()==wanted['sha256'],'exact selected guard bytes before method authentication')
        helper_code=compile(ast.parse(helper_raw,helper),helper,'exec')
        guard_class=next(item for item in helper_code.co_consts if type(item) is types.CodeType and item.co_name=='LibraryGuard')
        expected_guard=next(item for item in guard_class.co_consts if type(item) is types.CodeType and item.co_name=='framework_import')
        C.require(method.__func__.__code__==expected_guard and method.__func__.__globals__.get('LibraryGuard') is type(guard),'exact selected guard executable code and class globals')
        matches.append(validator(finder,value,_CPU_capability=_CPU_capability))
    C.require(len(matches)==1,'one actual installed selected context-bound library finder before native framework binding')
    actual=matches[0]
    C.require(requested is None or (type(requested) is types.MethodType and requested.__self__ is actual.__self__ and requested.__func__ is actual.__func__),'caller callback cannot replace installed library boundary')
    return actual
def _native_binding_suppliers():
    # Every native import remains behind the public stage/context refusal and
    # the operation's verified ownership/disk order in the shared body.
    def worker():
        import native_worker
        return native_worker
    def disk(output):
        import shutil
        return shutil.disk_usage(output).free
    def framework():
        import torch
        return torch
    def growth():
        import psutil
        return {'available_bytes':psutil.virtual_memory().available,'worker_RSS_bytes':psutil.Process().memory_info().rss}
    def seed(value):
        import random
        return random.seed(value)
    def topology(owner,membership):
        import native_worker,native_process,psutil
        sampler=native_process.Sampler()
        def snapshot(pid,supervisor=False):
            value=sampler.process(pid);value=dict(value)
            if supervisor:value['observed_psutil_create_time']=psutil.Process(pid).create_time()
            else:value['parent_pid']=psutil.Process(pid).ppid()
            return value
        before=native_worker.owned_job_resources(owner['job_name']);first=[snapshot(pid) for pid in before['queried_job_pids']]
        supervisor_first=snapshot(owner['supervisor_pid'],True)
        second=[snapshot(pid) for pid in before['queried_job_pids']];supervisor_second=snapshot(owner['supervisor_pid'],True)
        after=native_worker.owned_job_resources(owner['job_name'])
        return {'job_name':owner['job_name'],'resources_before':before,'resources_after':after,'process_snapshots_before':first,'process_snapshots_after':second,'supervisor_snapshot_before':supervisor_first,'supervisor_snapshot_after':supervisor_second}
    return {'native_worker':worker,'disk_free':disk,'framework':framework,'growth':growth,'python_seed':seed,'owned_topology':topology}
def _CPU_real_torch_bindings(context,owner,output,*,_capability,suppliers):
    C._check_CPU_native_context_fixture(_capability);value=C.context_value(context)
    C.require(value.get('fixture_only') is True and value.get('native_PASS') is value.get('native_authority_issued') is False,'explicit CPU binding source fixture only')
    original={k:v for k,v in value.items() if k not in ('fixture_only','native_PASS','native_authority_issued','status')}
    def revalidate():
        C._check_CPU_native_context_fixture(_capability);current=C._native_manifest_value(original['config_path'],**original['anchors'])
        C.require(current==original,'immutable original CPU context source/runtime/config/HEAD identity');return current
    result=_tensor_binding_body(context,owner,output,value,suppliers,revalidate)
    result.update(kind='CPU_ACTUAL_TENSOR_BINDING_FACTORY_SOURCE_FIXTURE',fixture_only=True,native_PASS=False,native_authority_issued=False)
    return result
def _binding_owner(owner,value):
    C.require(type(owner) is dict and set(owner)=={'worker_pid','root_worker_pid','job_name','supervisor_pid','supervisor_create_time','config_sha256'},'exact typed native ownership dictionary')
    C.require(all(type(owner[k]) is int and owner[k]>0 for k in ('worker_pid','root_worker_pid','supervisor_pid')) and owner['worker_pid']==owner['root_worker_pid'] and owner['worker_pid']!=owner['supervisor_pid'],'exact typed owned worker/supervisor PIDs')
    C.require(type(owner['supervisor_create_time']) in (int,float) and C.math.isfinite(owner['supervisor_create_time']) and owner['supervisor_create_time']>0,'exact finite supervisor creation time')
    name=owner['job_name'];prefix='Local\\SYNTH_NQ_JOB_';C.require(type(name) is str and name.startswith(prefix) and len(name)==len(prefix)+32 and all(x in '0123456789abcdef' for x in name[len(prefix):]),'exact owned native Job name')
    C.require(owner['config_sha256']==value['anchors']['expected_config_file'],'exact owned selected config file anchor');return C.parse(C.canonical(owner))
def _binding_compiled_source(value):
    selected=str(C.PACKAGE/'prefix_input_owned_worker.py');expected=value['config']['source_files'][selected]
    compiled=globals().get('__compiled_source_descriptor__');C.require(type(compiled) is dict and compiled.get('bytes')==expected['bytes'] and compiled.get('sha256')==expected['sha256'],'actual selected worker compile descriptor retained')
    path=str(HERE/'prefix_input_owned_worker.py');observation={'path':path,'bytes':compiled['bytes'],'sha256':compiled['sha256'],'loader_kind':'explicit_selected_source_compile','current_module_execution_observed':True}
    COMPILED_SOURCE_OBSERVATIONS[path]=observation
    return {'selected_source_path':selected,'compiled_source':observation,'fixture_rebased_source_path':path!=selected}
def _binding_topology(owner,membership,evidence):
    fields={'job_name','resources_before','resources_after','process_snapshots_before','process_snapshots_after','supervisor_snapshot_before','supervisor_snapshot_after'}
    C.require(type(evidence) is dict and set(evidence)==fields and evidence['job_name']==owner['job_name'],'exact complete owned topology evidence')
    rosters=[];resource_fields={'tree_rss_bytes','queried_job_pids','observed_pids','vanished_pids','query_handle_closed','scope'}
    for name in ('resources_before','resources_after'):
        row=evidence[name];C.require(type(row) is dict and set(row)==resource_fields and type(row['tree_rss_bytes']) is int and row['tree_rss_bytes']>=0 and row['query_handle_closed'] is True and row['scope']=='Windows_JobObjectBasicProcessIdList_owned_PIDs_only','original exact closed-query owned resource evidence')
        pids=row['queried_job_pids'];observed=row['observed_pids'];C.require(type(pids) is list and 0<len(pids)<=256 and all(type(pid) is int and 0<pid<=0xffffffff for pid in pids) and len(set(pids))==len(pids) and type(observed) is list and len(observed)==len(pids) and set(observed)==set(pids) and all(type(pid) is int for pid in observed) and row['vanished_pids']==[],'complete live typed owned PID roster before framework import');rosters.append(set(pids))
    C.require(rosters[0]==rosters[1] and owner['worker_pid'] in rosters[0] and membership['actual_worker_pid'] in rosters[0] and owner['supervisor_pid'] not in rosters[0],'same complete current Job roster with launcher/interpreter and external supervisor')
    snapshots=[]
    for name in ('process_snapshots_before','process_snapshots_after'):
        rows=evidence[name];C.require(type(rows) is list and len(rows)==len(rosters[0]),'complete owned process snapshot count');mapping={}
        for row in rows:
            C.require(type(row) is dict and row.get('terminal_confirmed') is False and type(row.get('parent_pid')) is int and row['parent_pid']>0,'live typed owned process parent snapshot')
            birth=C.identity({k:row.get(k) for k in ('pid','creation_filetime_100ns')});C.require(birth['creation_filetime_100ns']<=0xffffffffffffffff and birth['pid'] not in mapping,'unique exact uint64 raw FILETIME process roster');mapping[birth['pid']]={**birth,'parent_pid':row['parent_pid']}
        C.require(set(mapping)==rosters[0],'every and only current owned PID has a raw FILETIME snapshot');snapshots.append(mapping)
    C.require(snapshots[0]==snapshots[1],'stable exact owned FILETIME identities and parent snapshots')
    supervisors=[]
    for name in ('supervisor_snapshot_before','supervisor_snapshot_after'):
        row=evidence[name];C.require(type(row) is dict and row.get('terminal_confirmed') is False and type(row.get('observed_psutil_create_time')) in (int,float) and row['observed_psutil_create_time']==owner['supervisor_create_time'],'current supervisor legacy identity joins original ownership')
        birth=C.identity({k:row.get(k) for k in ('pid','creation_filetime_100ns')});C.require(birth['creation_filetime_100ns']<=0xffffffffffffffff and birth['pid']==owner['supervisor_pid'],'exact external supervisor uint64 raw FILETIME identity');supervisors.append(birth)
    C.require(supervisors[0]==supervisors[1],'stable current supervisor raw FILETIME identity')
    mapping=snapshots[0];root=owner['worker_pid'];C.require(mapping[root]['parent_pid']==owner['supervisor_pid'] and mapping[membership['actual_worker_pid']]['parent_pid']==membership['actual_parent_pid'],'launcher/supervisor and actual interpreter/parent topology joins')
    for pid in mapping:
        seen=set();current=pid
        while current!=root:
            C.require(current in mapping and current not in seen,'every owned member has a finite launcher parent chain');seen.add(current);parent=mapping[current]['parent_pid'];C.require(parent in mapping,'owned parent has an exact raw FILETIME roster identity');current=parent
    return {'job_name':owner['job_name'],'owned_birth_roster':[mapping[pid] for pid in sorted(mapping)],'supervisor_identity':supervisors[0],'membership':dict(membership)}
def _tensor_binding_body(context,owner,output,value,suppliers,revalidate):
    C.require(value['config']['scope']==C.NATIVE_SCOPE and value['config']['permission']==C.PERMISSION,'separate registered tensor purpose')
    owned=_binding_owner(owner,value);original_owner=C.canonical(owned);output=Path(output).absolute()
    C.require(output.parent==C.PACKAGE/'artifacts' and output.resolve()==output and output.is_dir(),'exact existing owned binding output leaf')
    C.require(type(suppliers) is dict and set(suppliers)=={'native_worker','disk_free','framework','growth','python_seed','owned_topology'} and all(callable(x) for x in suppliers.values()) and callable(revalidate),'exact private binding suppliers')
    compiled=_binding_compiled_source(value);state={'preflight':False,'ownership':None,'disk':False}
    topology_observations=[]
    topology_source=str(Path(value['config_path']).parent.parent/'native_worker.py')
    torch_holder=[None]
    def unchanged():C.require(C.canonical(owner)==original_owner,'immutable exact original ownership dictionary')
    def imported():
        unchanged();C.require(state['preflight'] is True and torch_holder[0] is not None,'framework operation requires actual ordered import and current successful preflight');return torch_holder[0]
    def lazy_torch():
        unchanged();C.require(state['preflight'] is True and state['ownership'] is not None and state['disk'] is True and torch_holder[0] is None,'verified current preflight/ownership/disk before exactly one framework import')
        torch=suppliers['framework']();C.require(torch is not None,'actual nonempty framework supplier');torch_holder[0]=torch;return torch
    def configure():
        torch=imported();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    def growth():
        imported();result=suppliers['growth']();C.require(type(result) is dict and set(result)=={'available_bytes','worker_RSS_bytes'} and all(type(x) is int and x>=0 for x in result.values()),'exact typed post-import growth observation');return dict(result)
    def ownership():
        unchanged();C.require(state['preflight'] is True and state['ownership'] is None and torch_holder[0] is None,'source preflight before one ownership verification and framework import')
        module=suppliers['native_worker']();method=getattr(module,'verify_worker_ownership',None);C.require(callable(method),'actual owned native-worker supplier verification method')
        result=method(dict(owned),value['config_path']);fields={'actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid','job_name','owned_job_membership_verified','query_job_handle_closed_before_framework_import'}
        C.require(type(result) is dict and set(result)==fields and all(type(result[k]) is int and result[k]>0 for k in ('actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid')),'exact typed actual Job membership dictionary')
        C.require(result['launcher_root_pid']==owned['worker_pid'] and result['supervisor_pid']==owned['supervisor_pid'] and result['job_name']==owned['job_name'] and result['owned_job_membership_verified'] is result['query_job_handle_closed_before_framework_import'] is True,'same launcher root/supervisor/Job and closed query handle')
        topology=_binding_topology(owned,result,suppliers['owned_topology'](dict(owned),dict(result)));topology_observations.append(topology)
        OWNED_TOPOLOGY_OBSERVATIONS[owned['job_name']]={'path':topology_source,'sha256':value['config']['source_files'][topology_source]['sha256'],'loader_kind':'original_owned_job_and_raw_FILETIME_topology_evidence','current_module_execution_observed':False,'fixture_only':value.get('fixture_only',False),'topology':topology}
        state['ownership']=C.parse(C.canonical(result));return dict(state['ownership'])
    def disk():
        unchanged();C.require(state['preflight'] is True and state['ownership'] is not None and state['disk'] is False and torch_holder[0] is None,'current source preflight/owned membership before disk and framework import')
        free=suppliers['disk_free'](output);C.require(type(free) is int and free>=4*1024**3,'unchanged4GiB typed disk admission');state['disk']=True;return free
    def preflight():
        state['preflight']=False;unchanged();C.require(C.context_value(context)==value,'same original opaque factory context');current=revalidate();state['preflight']=True
        return {'source_commit':current['expected_head'],'config_sha256':current['config']['sha256'],'scope':C.NATIVE_SCOPE,'permission_sha256':C.digest(current['config']['permission']),'source_inventory_sha256':C.digest(current['config']['source_files']),'runtime_inventory_sha256':C.digest(current['runtime']['files'])}
    def seed(number):
        imported();C.require(type(number) is int and number==101,'exact original Python seed101');return suppliers['python_seed'](number)
    api={'config_preflight':preflight,'ownership_verify':ownership,'disk_free':disk,'torch_import':lazy_torch,'torch_version_read':lambda:imported().__version__,
       'runtime_version':value['config']['torch_version'],'growth_observation':growth,'configure':configure,'python_seed':seed}
    return {'api':api,'factory_source':compiled,'owner_sha256':C.digest(owned),'context_projection_sha256':C.digest(value),'supplied_inert_boundaries':sorted(suppliers),'topology_observations':topology_observations}

def run_native(*args,**kwargs):
    C.stage_native_gate()
    return _integrated_native_worker(*args,**kwargs)

def _integrated_native_worker(*args,**kwargs):
    C.require(False,'native worker requires actual owned main execution closure')

def _owned_execution_factory(loader,*,_CPU_capability=None,suppliers=None,host=None,execution=None,owned_ledger=None):
    """Internal main factory captures its exact installer and original worker.

    No public entry accepts this checker/worker namespace or boundary callbacks.
    This preserves approved control flow, not hostile Python-memory isolation.
    """
    contract=C;check_authority=loader._check_installation_authority
    worker=SimpleNamespace(C=contract,source_core=source_core,exact_module=exact_module,
        original_guard=original_guard,_CPU_real_torch_bindings=_CPU_real_torch_bindings,
        COMPILED_SOURCE_OBSERVATIONS=COMPILED_SOURCE_OBSERVATIONS,
        OWNED_TOPOLOGY_OBSERVATIONS=OWNED_TOPOLOGY_OBSERVATIONS)
    prepared={}
    def build(context,owner,output,finder):
        value=contract.context_value(context);token=finder.installed_authority
        boundary=check_authority(token,value,finder=finder,_CPU_capability=_CPU_capability)
        if _CPU_capability is None:
            contract.require(contract.library_boundary_plan(value['runtime']) is not None,'externally selected library boundary plan before native suppliers')
            current=contract.revalidate_context(context)
            contract.require(current==value,'same original owned main context before suppliers')
        else:
            contract._check_CPU_native_context_fixture(_CPU_capability)
            original={k:v for k,v in value.items() if k not in ('fixture_only','native_PASS','native_authority_issued','status')}
            current=contract._native_manifest_value(value['config_path'],**value['anchors'])
            contract.require(current==original,'same original CPU main context before suppliers')
        _binding_compiled_source(value)
        name=str(contract.PACKAGE/'prefix_native_pipeline.py');descriptor=value['config']['source_files'][name]
        pipeline=exact_module(name,descriptor['sha256']);contract.require(Path(name).stat().st_size==descriptor['bytes'],'actual selected owned pipeline source size')
        contract.require(pipeline.C is contract,'owned pipeline retains original main contract')
        supplied=dict(_native_binding_suppliers() if _CPU_capability is None else suppliers)
        framework=supplied['framework']
        if boundary is not None:
            def guarded_framework():
                actual=check_authority(token,contract.context_value(context),finder=finder,_CPU_capability=_CPU_capability)
                with actual():return framework()
            supplied['framework']=guarded_framework
        else:contract.require(_CPU_capability is not None,'native framework needs actual installed library authority')
        if _CPU_capability is not None:
            def run_fixture():
                proof=pipeline._CPU_run_registered_fixture(context,owner,output,contract=contract,
                    _capability=_CPU_capability,suppliers=supplied,host=host,worker=worker)
                proof['owned_entry']={'original_contract_retained':pipeline.C is contract,
                    'original_context_retained':True,'opaque_installer_authority_consumed':True,
                    'library_boundary_consumed':boundary is not None,'fixture_only':True,
                    'native_PASS':False,'native_authority_issued':False}
                if execution is not None:execution.append(proof)
                return proof['report']
            return run_fixture
        api=_tensor_binding_body(context,owner,output,value,supplied,lambda:contract.revalidate_context(context))['api']
        return pipeline._prepare_owned_main_entry(worker,context,owner,output,api,owned_ledger=owned_ledger)
    def prepare(context,owner,output,finder):
        contract.require(not prepared,'one source-owned owned execution preparation')
        run=build(context,owner,output,finder)
        prepared.update(context=context,owner=contract.canonical(owner),output=Path(output).absolute(),finder=finder,token=finder.installed_authority,run=run,consumed=False)
        return {'prepared_without_framework_import':True,'native_authority_issued':False}
    def execute(context,owner,output,finder):
        if not prepared:return build(context,owner,output,finder)()
        contract.require(prepared['context'] is context and prepared['owner']==contract.canonical(owner) and prepared['output']==Path(output).absolute() and prepared['finder'] is finder and prepared['token'] is finder.installed_authority and not prepared['consumed'],'original context/owner/finder/token one-use prepared core handoff')
        check_authority(prepared['token'],contract.context_value(context),finder=finder,_CPU_capability=_CPU_capability)
        prepared['consumed']=True;return prepared['run']()
    execute.prepare=prepare
    return execute

def write_owned_return(output,report,module_observations,ownership,*,publisher_descriptor=None):
    # Shared actual publication body; CPU source fixtures call this writer but
    # retain their separate fixture proof/status and never obtain NativeContext.
    from_path=HERE/'prefix_input_owned_supervisor.py'
    if HERE==C.PACKAGE:C.require(type(publisher_descriptor) is dict and publisher_descriptor.get('role')=='python_source','native publisher exact selected source descriptor')
    publisher=exact_module(from_path,None if publisher_descriptor is None else publisher_descriptor['sha256'])
    if publisher_descriptor is not None:C.require(from_path.stat().st_size==publisher_descriptor['bytes'],'native publisher exact source size')
    module_observations=dict(module_observations)
    for path,observation in COMPILED_SOURCE_OBSERVATIONS.items():module_observations['explicit:'+path]=dict(observation)
    for name,observation in OWNED_TOPOLOGY_OBSERVATIONS.items():module_observations['owned_topology:'+name]=dict(observation)
    actual={'attempted':1,'returned':0,'failed':0,'pending':1};projected={'attempted':1,'returned':1,'failed':0,'pending':0}
    receipt={'report':report,'runtime_execution_modules':module_observations,'owned_membership_verified':ownership,'IO':{'owned_return_write':projected},'native_PASS':False}
    try:publisher.write_new(Path(output)/'OWNED_WORKER_RETURN.json',receipt)
    except BaseException:actual.update(failed=1,pending=0);raise
    actual.update(returned=1,pending=0);C.require(actual==projected,'actual owned return durable/write/readback counters reconcile');return receipt
def _registered_main_inputs(config,output,validate):
    """Shared bounded file/anchor prelude; validators are internal factories."""
    config=Path(config).absolute();output=Path(output).absolute()
    C.require(output.parent==C.PACKAGE/'artifacts' and output.resolve()==output and output.is_dir(),'owned output leaf')
    external=C.parse(C.read_bytes(output/'OWNED_EXTERNAL_ANCHORS.json'))
    C.require(type(external) is dict and set(external)=={'config_path','anchors'} and type(external['config_path']) is str and external['config_path']==str(config),'owned external anchor config join')
    fields={'expected_config_file','expected_config_body','registration_path','expected_registration_file','expected_registration_body','runtime_path','expected_runtime_file','expected_runtime_body','expected_head'}
    C.require(type(external['anchors']) is dict and set(external['anchors'])==fields,'exact owned external anchor fieldset')
    context=validate(config,**external['anchors']);value=C.context_value(context)
    owner=C.parse(C.read_bytes(output/'supervisor_ownership.json'));_binding_owner(owner,value)
    return {'context':context,'owner':owner,'output':output,'external':external}

def _registered_main_body(context,owner,output,*,loader_install,loader_remove,ownership_verify,execute,environment,_defer_cleanup=None,_entry_handoff=None):
    """Shared real main consumption order; suppliers are internal native bindings
    or a separately checked private CPU fixture. No public API accepts them.
    """
    value=C.context_value(context);selected_owner=C.parse(C.canonical(owner));finder=None;primary=None;cleanup=None;publication=None;result=None;stage='install';identity_error=None
    OWNED_TOPOLOGY_OBSERVATIONS.pop(selected_owner['job_name'],None)
    try:
        entry=None if _entry_handoff is None else _consume_entry_handoff(_entry_handoff,context,owner,output)
        finder=loader_install(context,C) if entry is None else entry['finder']
        if value.get('fixture_only',False) is False:
            C.require('library_boundary_plan' in value['runtime'],'externally selected library boundary plan before native owned work')
        stage='ownership'
        owned=_binding_owner(owner,value);original_owner=C.canonical(owned)
        verified=ownership_verify(owner,value['config_path']) if entry is None else entry['verified']
        fields={'actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid','job_name','owned_job_membership_verified','query_job_handle_closed_before_framework_import'}
        C.require(type(verified) is dict and set(verified)==fields and all(type(verified[k]) is int and verified[k]>0 for k in ('actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid')),'exact typed main owned membership dictionary')
        C.require(C.canonical(owner)==original_owner and verified['launcher_root_pid']==owned['worker_pid'] and verified['supervisor_pid']==owned['supervisor_pid'] and verified['job_name']==owned['job_name'] and verified['owned_job_membership_verified'] is verified['query_job_handle_closed_before_framework_import'] is True,'same immutable main launcher/supervisor/Job and verified closed query')
        for key,setting in value['config']['offline_environment'].items():environment[key]=setting
        environment['CUDA_VISIBLE_DEVICES']='0'
        stage='execute';report=execute(context,owner,output,finder)
        stage='owned_return'
        receipt=None if _defer_cleanup is not None else write_owned_return(output,report,finder.observed,verified,publisher_descriptor=value['config']['source_files'][str(C.PACKAGE/'prefix_input_owned_supervisor.py')])
        result={'returncode':0 if report['status']=='NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT' else 2,'report':report,'container':receipt,'finder':finder}
    except BaseException as failure:
        primary=failure
    def finish_owned_lifecycle():
        nonlocal result,cleanup,publication,identity_error
        if _defer_cleanup is not None and result is not None:
            try:result['container']=write_owned_return(output,result['report'],finder.observed,verified,publisher_descriptor=value['config']['source_files'][str(C.PACKAGE/'prefix_input_owned_supervisor.py')])
            except BaseException as failure:publication=failure
        try:identity=_library_evidence_identity(value,selected_owner)
        except BaseException as failure:
            identity_error=failure;identity=_library_evidence_identity(value,selected_owner,use_topology=False)
        if finder is not None:
            finder.library_evidence_identity=identity
            try:loader_remove(finder)
            except BaseException as failure:cleanup=failure
            raw_packet=finder.library_raw_evidence
        else:raw_packet=getattr(primary,'__library_raw_packet__',None)
        if raw_packet is not None and raw_packet.get('identity') is None:
            # A constructor's actual partial journal has no owned identity yet.
            # Bind it to this original context without changing any observations,
            # failures, terminal state or producer completeness claims.
            raw_packet=C.parse(C.canonical(raw_packet));raw_packet['identity']=identity
            raw_packet['sha256']=C.digest({k:v for k,v in raw_packet.items() if k!='sha256'})
        errors=[]
        if primary is not None:errors.append({'stage':stage,'error':_library_error_record(primary)})
        if identity_error is not None:errors.append({'stage':'export','error':_library_error_record(identity_error)})
        if finder is not None:
            for name,error in (('cleanup',finder.library_cleanup_error),('export',finder.library_export_error)):
                if error is not None:errors.append({'stage':name,'error':_library_error_record(error)})
        elif cleanup is not None:errors.append({'stage':'cleanup','error':_library_error_record(cleanup)})
        try:
            evidence=_write_library_boundary_evidence(output,value,selected_owner,raw_packet,errors)
            if result is not None:result['library_evidence']=evidence
        except BaseException as failure:publication=failure
        if publication is not None and (primary is not None or cleanup is not None or identity_error is not None):
            retained=primary if primary is not None else cleanup if cleanup is not None else identity_error
            # Builtin exception storage avoids subclass setters/stringifiers
            # replacing the already-retained primary/cleanup exception.
            try:
                attributes=BaseException.__dict__['__dict__'].__get__(retained,BaseException)
                attributes['__library_publication_error__']=publication
                attributes['__library_publication_IO__']={'attempted':1,'returned':0,'failed':1,'pending':0}
                record=_library_error_record(publication)
                if hasattr(BaseException,'add_note'):BaseException.add_note(retained,'Secondary library evidence publication failed: '+record['type']+': '+record['message'])
            except BaseException:pass
        if primary is not None:
            if cleanup is not None:raise primary from cleanup
            raise primary
        if cleanup is not None:raise cleanup
        if identity_error is not None:raise identity_error
        if publication is not None:raise publication
        return result
    if _defer_cleanup is not None:
        C.require(type(_defer_cleanup) is list and not _defer_cleanup,'one private original lifecycle cleanup handoff')
        _defer_cleanup.append(finish_owned_lifecycle)
        if result is not None:return result
        if primary is not None:raise primary
        C.require(False,'actual main retained report or primary failure')
    return finish_owned_lifecycle()

def _library_error_record(error):
    cls=type(error);name=type.__getattribute__(cls,'__name__');message='<message unavailable>';exact=False
    try:
        args=BaseException.args.__get__(error,cls)
        ordinary=type.__getattribute__(cls,'__str__') is BaseException.__str__
        if ordinary and type(args) is tuple and len(args)==1 and type(args[0]) is str:
            message=args[0][:2048];exact=len(args[0])<=2048
        elif ordinary and type(args) is tuple and not args:message='';exact=True
    except BaseException:pass
    return {'type':name[:128] if type(name) is str else 'BaseException','message':message,'message_exact':exact}

def _library_evidence_identity(value,owner,*,use_topology=True):
    birth=None;record=OWNED_TOPOLOGY_OBSERVATIONS.get(owner['job_name']) if use_topology else None
    if record is not None:
        topology=record['topology'];membership=topology['membership']
        C.require(topology['job_name']==owner['job_name'] and membership['launcher_root_pid']==owner['worker_pid'] and membership['supervisor_pid']==owner['supervisor_pid'],'library journal original owned membership join')
        rows=[row for row in topology['owned_birth_roster'] if row['pid']==membership['actual_worker_pid']]
        C.require(len(rows)==1,'one actual executing interpreter birth for library journal')
        birth=C.identity({k:rows[0][k] for k in ('pid','creation_filetime_100ns')})
    path=str(C.PACKAGE/'prefix_input_owned_worker.py')
    return {'source_commit':value['expected_head'],'original_scope':C.NATIVE_SCOPE,
        'owned_birth':birth,'owned_birth_status':'VALIDATED' if birth is not None else 'UNAVAILABLE',
        'context_sha256':C.digest(value),'producer_source':{'path':path,'descriptor':value['config']['source_files'][path]},
        'owner_receipt_sha256':C.digest(owner),'fixture_only':value.get('fixture_only',False),
        'native_PASS':False,'native_authority_issued':False}

def _write_library_boundary_evidence(output,value,owner,raw_packet,errors):
    policy=value['runtime']['library_search_policy'].get('enforced_boundary')
    provider=None if policy is None else policy['provider_source']
    envelope={'schema_version':1,'kind':'OWNED_LIBRARY_BOUNDARY_EVIDENCE_V1',
        'source_commit':value['expected_head'],'original_scope':C.NATIVE_SCOPE,'context_sha256':C.digest(value),
        'external_anchors':value['anchors'],'policy_sha256':None if policy is None else C.digest(policy),
        'provider_source':provider,'owner_receipt_sha256':C.digest(owner),'raw_packet':raw_packet,
        'raw_packet_sha256':None if raw_packet is None else C.digest(raw_packet),'main_errors':errors,
        'fixture_only':value.get('fixture_only',False),'native_PASS':False,'native_authority_issued':False}
    envelope['sha256']=C.digest(envelope);raw=(C.canonical(envelope)+'\n').encode('ascii')
    C.require(len(raw)<=C.MAX_JSON,'bounded complete library evidence envelope; no truncation')
    path=HERE/'prefix_input_owned_supervisor.py';descriptor=value['config']['source_files'][str(C.PACKAGE/'prefix_input_owned_supervisor.py')]
    publisher=exact_module(path,descriptor['sha256']);C.require(path.stat().st_size==descriptor['bytes'],'actual selected library evidence publisher bytes')
    publisher.write_new(Path(output)/'LIBRARY_BOUNDARY_EVIDENCE.json',envelope)
    return envelope

def _CPU_registered_main_fixture(context,owner,output,*,_capability,loader,suppliers,host,environment,library_supplier=None):
    C._check_CPU_native_context_fixture(_capability);value=C.context_value(context)
    C.require(value.get('fixture_only') is True and value.get('native_PASS') is False and value.get('native_authority_issued') is False,'CPU main context cannot authorize native entry');_binding_owner(owner,value);_binding_compiled_source(value)
    def validate(config,**anchors):
        return C._CPU_native_context_fixture(config,_capability=_capability,**anchors)
    inputs=_registered_main_inputs(value['config_path'],output,validate)
    C.require(C.context_value(inputs['context'])==value,'same original CPU main selected context from actual file/anchor factory')
    C.require(C.canonical(inputs['owner'])==C.canonical(owner),'same original CPU main owned receipt')
    context=inputs['context'];owner=inputs['owner'];output=inputs['output']
    C.require(type(environment) is dict,'isolated CPU environment only')
    selected=str(C.PACKAGE/'prefix_registered_loader.py');expected=value['config']['source_files'][selected]
    compiled=getattr(loader,'__compiled_source_descriptor__',None);C.require(type(compiled) is dict and compiled.get('bytes')==expected['bytes'] and compiled.get('sha256')==expected['sha256'] and str(Path(loader.__file__).absolute())==selected,'actual selected CPU loader compiled bytes/origin')
    C.require(type(suppliers) is dict and set(suppliers)=={'native_worker','disk_free','framework','growth','python_seed','owned_topology'} and all(callable(x) for x in suppliers.values()),'exact private main inert suppliers')
    execution=[];execute=_owned_execution_factory(loader,_CPU_capability=_capability,suppliers=suppliers,host=host,execution=execution)
    result=_registered_main_body(context,owner,output,loader_install=lambda ctx,contract:loader._CPU_install_fixture(ctx,contract,_capability=_capability,library_supplier=library_supplier),loader_remove=loader.remove,ownership_verify=lambda owned,path:suppliers['native_worker']().verify_worker_ownership(owned,path),execute=execute,environment=environment)
    return {'kind':'CPU_REGISTERED_WORKER_MAIN_SOURCE_FIXTURE','status':'CPU_REGISTERED_MAIN_RETURNED' if result['returncode']==0 else 'CPU_REGISTERED_MAIN_FAILURE_PRESERVED','fixture_only':True,'native_PASS':False,'native_authority_issued':False,'original_context':context,'library_terminal_evidence_join_complete':False,'input_prelude':{'config_path':value['config_path'],'external':inputs['external'],'owner_sha256':C.digest(owner),'context_projection_sha256':C.digest(value),'actual_selected_files_reconsumed':True,'fixture_only':True,'native_authority_issued':False},'pipeline_proof':execution[0],**result}

def _observer_install_factory(context,owner,loader,*,_CPU_capability=None,transport=None,_source_meter=None):
    """Trusted original main observer/loader binding; no public suppliers."""
    if _CPU_capability is None:C.stage_native_gate()
    else:C._check_CPU_native_context_fixture(_CPU_capability)
    value=C.context_value(context);adapter_path=str(C.PACKAGE/'prefix_library_observer.py')
    C.require('library_event_observer' in value['runtime'],'externally selected native observer registration required')
    adapter=exact_module(adapter_path,value['config']['source_files'][adapter_path]['sha256'],_meter=_source_meter)
    if _CPU_capability is None:
        C.stage_native_gate();observer=adapter.make_native_observer(context,C,owner)
        C._register_library_event_observer(context,adapter,observer,observer.abi_binding() if _source_meter is None else _source_meter.api(adapter,observer,'abi_binding'))
    else:
        C._check_CPU_native_context_fixture(_CPU_capability)
        C.require(value.get('fixture_only') is True and transport is not None,'explicit CPU startup collector transport only')
        observer=adapter._CPU_observer_fixture(context,C,transport,_capability=adapter._CPU_capability())
        C._CPU_register_library_event_observer_fixture(context,adapter,observer,observer.abi_binding(),_capability=_CPU_capability)
        observer.check=lambda:C._CPU_verify_library_event_observer_fixture(observer,_capability=_CPU_capability)
    def install(ctx,contract):
        C.require(ctx is context and contract is C,'same original main observer context and contract')
        return loader.install(ctx,contract,library_bootstrap_verifier=observer.bootstrap if _source_meter is None else lambda stage:_source_meter.api(adapter,observer,'bootstrap',stage),library_event_observer=observer)
    return install,observer,adapter

def _prepare_acquired_entry(state,*,install,verify,prepare,_CPU_capability=None):
    """Common source-owned acquisition sequence; CPU capability is not permit."""
    if _CPU_capability is None:C.stage_native_gate()
    else:
        C._check_CPU_native_context_fixture(_CPU_capability)
        C.require(C.context_value(state['inputs']['context']).get('fixture_only') is True,'private inert acquired-entry context only')
    inputs=state['inputs'];context=inputs['context'];owner=inputs['owner']
    finder=install(context,C);state['acquired_finder']=finder
    verified=verify(owner,C.context_value(context).get('config_path'))
    fields={'actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid','job_name','owned_job_membership_verified','query_job_handle_closed_before_framework_import'}
    C.require(type(verified) is dict and set(verified)==fields and all(type(verified[k]) is int and verified[k]>0 for k in ('actual_worker_pid','actual_parent_pid','launcher_root_pid','supervisor_pid')) and verified['launcher_root_pid']==owner['worker_pid'] and verified['supervisor_pid']==owner['supervisor_pid'] and verified['job_name']==owner['job_name'] and verified['owned_job_membership_verified'] is verified['query_job_handle_closed_before_framework_import'] is True,'original exact membership before native preparation')
    prepare(context,owner,inputs['output'],finder)
    state['entry_handoff']=_record_entry_handoff(context,owner,inputs['output'],finder,verified)
    return {'original_owned_execution_prepared':True,'fixture_only':_CPU_capability is not None,'native_authority_issued':False}


def _entry_handoff_factory():
    records={}
    class Handoff:__slots__=()
    def record(context,owner,output,finder,verified):
        token=Handoff();records[token]=(context,C.canonical(owner),Path(output).absolute(),finder,C.canonical(verified),finder.installed_authority);return token
    def consume(token,context,owner,output):
        C.require(type(token) is Handoff and token in records,'source-issued private entry handoff')
        selected,owner_pin,selected_output,finder,verified,authority=records[token]
        C.require(selected is context and owner_pin==C.canonical(owner) and selected_output==Path(output).absolute() and finder.installed_authority is authority,'original source-owned installed finder/context/owner handoff')
        del records[token];return {'finder':finder,'verified':C.parse(verified)}
    return record,consume
_record_entry_handoff,_consume_entry_handoff=_entry_handoff_factory()


class _SelectedSourceMeter:
    """Bounded source/API boundary counters; no permission or permit factory.

    Nested boundaries retain their own call counts. Byte totals count read
    bytes and streamed descriptor hashes separately; repeat reads are explicit.
    """
    def __init__(self):self.phase=None;self.records={};self.bindings=[];self.source_pins={}
    def enter(self,phase,costs):
        if self.phase is not None:raise ValueError('one current source-meter phase')
        self.phase=(phase,costs);self.records.setdefault(phase,{})
    def leave(self):self.phase=None
    def invoke(self,name,action,*,category='boundary',byte_count=None):
        if self.phase is None:return action()
        phase,costs=self.phase;rows=self.records[phase]
        if type(name) is not str or not 0<len(name)<=160 or name not in rows and len(rows)>=64:raise ValueError('finite selected source/API counters')
        key=name+'.'+category if category in ('source','cache') else name
        item=rows.setdefault(key,{'attempted':0,'returned':0,'failed':0,'pending':0,'bytes':0,'category':category})
        if item['category']!=category:raise ValueError('stable selected boundary category')
        item['attempted']+=1;item['pending']+=1
        if category in ('source','cache','IO'):costs['IO_calls']=costs.get('IO_calls',0)+1
        if category=='API':costs['API_calls']=costs.get('API_calls',0)+1
        try:
            value=action();size=0 if byte_count is None else byte_count(value)
            if type(size) is not int or not 0<=size<=16*1024**3:raise ValueError('bounded actual selected consumption bytes')
            item['bytes']+=size
            if category in ('source','cache'):
                costs[category+'_reads']=costs.get(category+'_reads',0)+1;costs[category+'_bytes']=costs.get(category+'_bytes',0)+size
        except BaseException:item['pending']-=1;item['failed']+=1;raise
        item['pending']-=1;item['returned']+=1;return value
    def pin_module(self,module):
        namespace=module if type(module) is dict else module.__dict__;path=namespace.get('__file__');descriptor=namespace.get('__compiled_source_descriptor__')
        if type(path) is not str or type(descriptor) is not dict or type(descriptor.get('bytes')) is not int or not 0<=descriptor['bytes']<=16*1024**2 or type(descriptor.get('sha256')) is not str or len(descriptor['sha256'])!=64:raise ValueError('source meter requires actual selected compiled descriptor')
        pin={'bytes':descriptor['bytes'],'sha256':descriptor['sha256'],'role':'python_source'}
        if len(self.source_pins)>=32 and path not in self.source_pins or path in self.source_pins and self.source_pins[path]!=pin:raise ValueError('finite immutable metered source pins')
        self.source_pins[path]=pin
    def bind_contract(self,module):
        self.pin_module(module)
        from types import FunctionType
        for name in ('read_bytes','sealed_json','file_descriptor','physical_roster'):
            original=module.__dict__[name]
            if type(original) is not FunctionType or original.__globals__ is not module.__dict__:raise ValueError('actual selected contract function identity')
            def wrapper(*args,__name=name,__original=original,**kwargs):
                category='boundary';bytes_fn=None
                if __name=='read_bytes':category='cache' if Path(args[0]).suffix=='.pyc' else 'source';bytes_fn=len
                elif __name=='physical_roster':category='IO'
                elif __name=='file_descriptor':category='cache' if Path(args[0]).suffix=='.pyc' else 'source';bytes_fn=lambda value:args[1]['bytes']
                return self.invoke('contract.'+__name,lambda:__original(*args,**kwargs),category=category,byte_count=bytes_fn)
            module.__dict__[name]=wrapper;self.bindings.append((module,name,original,wrapper))
        module.__dict__['__source_meter__']=self
    def bind_git(self,module):
        self.pin_module(module)
        from types import FunctionType,SimpleNamespace
        original=module.__dict__['read'];process=module.__dict__['subprocess']
        if type(original) is not FunctionType or original.__globals__ is not module.__dict__:raise ValueError('selected Git disk reader identity')
        def read(*args,**kwargs):return self.invoke('git.disk_read',lambda:original(*args,**kwargs),category='source',byte_count=len)
        def run(argv,*args,**kwargs):
            blob=type(argv) is list and len(argv)>=2 and argv[-3:-1]==['cat-file','blob']
            return self.invoke('git.immutable_blob' if blob else 'git.metadata_command',lambda:process.run(argv,*args,**kwargs),category='source' if blob else 'IO',byte_count=(lambda result:len(result.stdout)) if blob else None)
        module.__dict__['read']=read;module.__dict__['subprocess']=SimpleNamespace(run=run)
        self.bindings.extend(((module,'read',original,read),(module,'subprocess',process,module.__dict__['subprocess'])))
    def api(self,module,instance,name,*args,**kwargs):
        self.pin_module(module)
        from types import MethodType
        cls=type(instance);method=getattr(instance,name)
        if type(method) is not MethodType or method.__self__ is not instance or method.__func__.__globals__ is not (module if type(module) is dict else module.__dict__) or method.__func__ is not next((base.__dict__.get(name) for base in cls.__mro__ if name in base.__dict__),None):raise ValueError('original selected API class/method/module identity')
        return self.invoke('API.'+cls.__name__+'.'+name,lambda:method(*args,**kwargs),category='API')
    def snapshot(self):
        import copy
        return copy.deepcopy(self.records)
    def restore(self):
        for module,name,original,owned in reversed(self.bindings):
            if module.__dict__.get(name) is not owned:raise ValueError('source-meter binding mutated; no clobber restoration')
        for module,name,original,owned in reversed(self.bindings):module.__dict__[name]=original
        self.bindings=[]


def _entry_journal_bootstrap(output):
    """Pinned original writer AST only, before qualified C/source execution.

    These finite source reads establish the writer itself and are recorded as
    bootstrap provenance; interpreter/parser initialization is external scope.
    """
    from types import SimpleNamespace
    from copy import deepcopy
    from datetime import datetime,timezone
    import json,os
    package=HERE.parents[1] if HERE.parent.name=='artifacts' else HERE
    core_path=package/'evidence/prefix_input_core_reviewed_cpu_20261005/pipeline.py'
    contract_path=core_path.with_name('contract.py')
    pins={str(core_path):'700d6934e0fb36b158eacfe5f53330e5e498d9fd9a2bf264b40a077c95bbb9c4',str(contract_path):'b83228e3ae797d2a410b459f73662daba75ecb6f203f3edd8edaf206eb757836',str(HERE/'prefix_native_pipeline.py'):ENTRY_PIPELINE_PIN}
    raws={};provenance=[]
    for text,pin in pins.items():
        path=Path(text);before=path.lstat()
        if path.resolve()!=path or path.is_symlink() or not 0<=before.st_size<=16*1024**2:raise ValueError('bounded physical writer bootstrap source')
        with path.open('rb') as stream:raw=stream.read(before.st_size+1)
        after=path.lstat()
        if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('exact writer bootstrap source pin')
        raws[text]=raw;provenance.append({'path':text,'bytes':len(raw),'sha256':pin})
    constants={}
    for node in ast.parse(raws[str(contract_path)],str(contract_path)).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('LIMITS','OPS','CPU_SCOPE'):
                    expression=ast.fix_missing_locations(ast.Expression(node.value))
                    if any(not isinstance(n,(ast.Expression,ast.Constant,ast.Dict,ast.Tuple,ast.Load,ast.BinOp,ast.Mult,ast.Pow)) for n in ast.walk(expression)):raise ValueError('finite pinned writer constant expression')
                    constants[target.id]=eval(compile(expression,str(contract_path),'eval'),{'__builtins__':{}},{})
    if set(constants)!={'LIMITS','OPS','CPU_SCOPE'} or constants['LIMITS']['maximum_markers']!=128:raise ValueError('exact source-derived writer constants')
    def require(value,message):
        if not value:raise ValueError(message)
    facade=SimpleNamespace(**constants,require=require,canonical=lambda v:json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False))
    nodes=[n for n in ast.parse(raws[str(core_path)],str(core_path)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('counts','projected','error_value','Ledger')]
    require(len(nodes)==4,'exact original writer definition set')
    ns={'C':facade,'Path':Path,'deepcopy':deepcopy,'datetime':datetime,'timezone':timezone,'os':os,'time':time}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(core_path),'exec'),ns)
    names=('STARTUP_PHASES','_shared_scope_ledger')
    selected=[n for n in ast.parse(raws[str(HERE/'prefix_native_pipeline.py')],str(HERE/'prefix_native_pipeline.py')).body if isinstance(n,ast.FunctionDef) and n.name==names[1] or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==names[0] for t in n.targets)]
    require(len(selected)==2,'exact pinned shared journal definitions')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(HERE/'prefix_native_pipeline.py'),'exec'),ns)
    ledger=ns['_shared_scope_ledger'](ns,output);ledger.entry_bootstrap_sources=provenance
    return ledger,ns['STARTUP_PHASES']


def _entry_phase_body(ledger,phases,actions,execute,*,_native_mode=False):
    """Internal exact-main sequence; supplied actions never grant authority."""
    if type(actions) is not dict or set(actions)!=set(phases):raise ValueError('exact six private entry actions')
    ledger.entry_native_mode=_native_mode
    state={'source_meter':_SelectedSourceMeter()};primary=None;cleanup=None;report=None
    try:
        for name in phases[:5]:
            costs={}
            def action(name=name):
                state['source_meter'].enter(name,costs)
                try:return actions[name](state,costs)
                finally:state['source_meter'].leave()
            state[name]=ledger.startup(name,action,costs)
        report=execute(state,ledger)
    except BaseException as error:primary=error;state['entry_primary']=error
    try:
        costs={}
        def cleanup_action():
            state['source_meter'].enter(phases[5],costs)
            try:return actions[phases[5]](state,costs)
            finally:state['source_meter'].leave()
        ledger.startup(phases[5],cleanup_action,costs)
    except BaseException as error:cleanup=error
    if report is None:report={'status':'NATIVE_INPUT_CHECKS_FAILED' if _native_mode else 'CPU_PREFIX_INPUT_CHECKS_FAILED'}
    if primary is not None or cleanup is not None:
        report['status']='NATIVE_INPUT_CHECKS_FAILED' if _native_mode else 'CPU_PREFIX_INPUT_CHECKS_FAILED';report['entry_errors']=[_library_error_record(e) for e in (primary,cleanup) if e is not None]
    report['entry_bootstrap_sources']=ledger.entry_bootstrap_sources;report['selected_boundary_counters']=state['source_meter'].snapshot();report['selected_boundary_source_pins']=dict(state['source_meter'].source_pins)
    try:
        report['final_phase_summary']=ledger.finish_owned(report)
        import copy
        report['operation_accounting']=copy.deepcopy(ledger.ops);report['IO_accounting']=copy.deepcopy(ledger.io);report['publication_errors']=copy.deepcopy(ledger.errors)
    except BaseException as publication:
        if primary is not None:raise primary from publication
        if cleanup is not None:raise cleanup from publication
        raise
    if primary is not None:
        if cleanup is not None:raise primary from cleanup
        raise primary
    if cleanup is not None:raise cleanup
    return report


def _native_entry_actions(args,ledger,phases):
    """Closed production closures retain original context/installer lifecycle."""
    def source(state,costs):
        global C
        C=exact_module(HERE/'prefix_tensor_contract.py',CONTRACT_PIN,_meter=state['source_meter']);state['source_meter'].bind_contract(C);ledger.entry_scope=C.NATIVE_SCOPE;C.stage_native_gate()
        inputs=_registered_main_inputs(args.config,args.output_dir,C.validate_native_manifest);state['inputs']=inputs
        return {'context_projection_sha256':C.digest(C.context_value(inputs['context']))}
    def metadata(state,costs):
        import sys
        inputs=state['inputs'];return _measure_no_site_profile(C,C.context_value(inputs['context']),sys)
    def preload(state,costs):
        return _site_disabled_bootstrap_setup(state['inputs']['context'])
    def roots(state,costs):
        inputs=state['inputs'];context=inputs['context'];value=C.context_value(context)
        state['transition']=_registered_search_roots(context)
        name=str(C.PACKAGE/'prefix_registered_loader.py');descriptor=value['config']['source_files'][name]
        state['loader']=exact_module(name,descriptor['sha256'],_meter=state['source_meter'])
        install,observer,adapter=_observer_install_factory(context,inputs['owner'],state['loader'],_source_meter=state['source_meter']);state['install']=install;state['observer']=observer
        return {'registered_roots':state['transition']['registered_roots']}
    def provider(state,costs):
        inputs=state['inputs'];loader=state['loader']
        state['execute']=_owned_execution_factory(loader,owned_ledger=ledger)
        def verify(owner,path):
            import native_worker
            return native_worker.verify_worker_ownership(owner,path)
        return _prepare_acquired_entry(state,install=state['install'],verify=verify,prepare=state['execute'].prepare)
    def cleanup(state,costs):
        primary=None
        try:
            if state.get('lifecycle_cleanup'):
                for finish in state['lifecycle_cleanup']:finish()
            elif 'entry_primary' in state and 'inputs' in state:
                finder=state.get('acquired_finder');inputs=state['inputs'];context=inputs['context'];owner=inputs['owner'];retained=state['entry_primary'];prior_cause=retained.__cause__
                def retained_failure(*args):raise retained
                try:_registered_main_body(context,owner,inputs['output'],loader_install=(lambda *a:finder) if finder is not None else retained_failure,loader_remove=state['loader'].remove,ownership_verify=retained_failure,execute=retained_failure,environment={})
                except BaseException as failure:
                    publication=getattr(retained,'__library_publication_error__',None)
                    if failure is not retained:raise
                    if publication is not None:raise publication
                    if retained.__cause__ is not prior_cause:raise retained.__cause__
        except BaseException as error:primary=error
        try:
            if 'transition' in state:_restore_registered_roots(state['transition'])
        except BaseException as secondary:
            if primary is not None:raise primary from secondary
            raise
        state['source_meter'].restore()
        if primary is not None:raise primary
        return {'owned_root_cleanup':True}
    def execute(state,owned):
        import os
        inputs=state['inputs'];context=inputs['context'];owner=inputs['owner'];loader=state['loader']
        def verify(selected,config):
            import native_worker
            return native_worker.verify_worker_ownership(selected,config)
        state['lifecycle_cleanup']=[]
        result=_registered_main_body(context,owner,inputs['output'],loader_install=state['install'],loader_remove=loader.remove,ownership_verify=verify,execute=state['execute'],environment=os.environ,_defer_cleanup=state['lifecycle_cleanup'],_entry_handoff=state['entry_handoff'])
        return result['report']
    return dict(zip(phases,(source,metadata,preload,roots,provider,cleanup))),execute


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    output=args.output_dir.absolute()
    package=HERE.parents[1] if HERE.parent.name=='artifacts' else HERE
    if output.parent!=package/'artifacts' or not output.is_dir() or output.resolve()!=output:raise ValueError('owned existing physical entry output')
    ledger,phases=_entry_journal_bootstrap(output)
    actions,execute=_native_entry_actions(args,ledger,phases)
    report=_entry_phase_body(ledger,phases,actions,execute,_native_mode=True)
    return 0 if report['status']=='NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT' else 2


def _measure_no_site_profile(contract, value, interpreter):
    """Metadata only; never call a path hook or infer execution history."""
    from importlib.machinery import BuiltinImporter, FrozenImporter, PathFinder, FileFinder, ExtensionFileLoader, SourceFileLoader, SourcelessFileLoader, EXTENSION_SUFFIXES, SOURCE_SUFFIXES, BYTECODE_SUFFIXES
    from types import FunctionType
    import zipimport
    expected = value['runtime']['cache_policy']['interpreter_profile']
    contract.require(len(interpreter.meta_path) == 3 and all((a is b for a, b in zip(interpreter.meta_path, (BuiltinImporter, FrozenImporter, PathFinder)))), 'exact actual standard meta-path identities')
    hooks = interpreter.path_hooks
    contract.require(len(hooks) == 2 and hooks[0] is zipimport.zipimporter and (type(hooks[1]) is FunctionType), 'exact actual standard path-hook kinds')
    outer = FileFinder.path_hook.__func__
    expected_code = next((c for c in outer.__code__.co_consts if type(c) is type(outer.__code__) and c.co_name == 'path_hook_for_FileFinder'))
    hook = hooks[1]
    contract.require(hook.__code__ is expected_code and hook.__globals__ is outer.__globals__ and (type(hook.__closure__) is tuple) and (len(hook.__closure__) == 2) and (hook.__closure__[0].cell_contents is FileFinder), 'original standard FileFinder path-hook code/globals/class')
    details = hook.__closure__[1].cell_contents
    wanted = ((ExtensionFileLoader, EXTENSION_SUFFIXES), (SourceFileLoader, SOURCE_SUFFIXES), (SourcelessFileLoader, BYTECODE_SUFFIXES))
    contract.require(type(details) is tuple and len(details) == 3 and all((type(row) is tuple and len(row) == 2 and (row[0] is target[0]) and (type(row[1]) is list) and (row[1] == target[1]) for row, target in zip(details, wanted))), 'exact standard FileFinder suffix/loader closure')
    contract.require(type(interpreter.path) is list and len(interpreter.path) == 4 and all((type(p) is str for p in interpreter.path)), 'exact typed finite initial interpreter path')
    flags = {name: getattr(interpreter.flags, name) for name in expected['flags']}
    contract.require(all((type(v) is int for v in flags.values())), 'actual interpreter flag integers')
    observed = {**{name: getattr(interpreter, name) for name in ('executable', 'prefix', 'exec_prefix', 'base_prefix', 'base_exec_prefix')}, 'flags': flags, 'path': list(interpreter.path), 'meta_path': list(expected['meta_path']), 'path_hooks': list(expected['path_hooks']), 'source_execution_history_proven': False}
    contract.require(contract.canonical(observed) == contract.canonical(expected), 'actual no-site interpreter flags/prefix/path profile')
    return observed

def _site_disabled_bootstrap_setup(context):
    """Preserve actual009 source/cache assumptions; no invented cached None."""
    C.stage_native_gate();value=C.context_value(context);C.revalidate_context(context)
    import sys,importlib.machinery
    profile=_measure_no_site_profile(C,value,sys);rows=value['runtime']['cache_policy']['preloaded_bootstrap_modules'];files=value['runtime']['files']
    names=('ctypes','ctypes.wintypes','_ctypes','ctypes._endian')
    C.require(all(name not in sys.modules for name in names),'fresh selected ctypes family before standard preload')
    for name in names:
        row=rows[name];C.file_descriptor(row['path'],files[row['path']],layout=value['runtime']['stdlib_layout'])
        if row['cache_path'] is not None:C.file_descriptor(row['cache_path'],files[row['cache_path']],layout=value['runtime']['stdlib_layout'])
        parent=[str(Path(rows['ctypes']['path']).parent)] if name.startswith('ctypes.') else None;spec=importlib.machinery.PathFinder.find_spec(name,parent)
        C.require(spec is not None and str(C.qualified_path(spec.origin,layout=value['runtime']['stdlib_layout']))==row['path'],'exact observed standard ctypes source/extension resolution')
    for name in ('ctypes','ctypes.wintypes','_ctypes'):__import__(name)
    for name in names:
        module=sys.modules[name];row=rows[name];C.require(str(C.qualified_path(module.__file__,layout=value['runtime']['stdlib_layout']))==row['path'],'actual observed ctypes module origin')
        cache=vars(module).get('__cached__');C.require((cache is None and row['cache_path'] is None) or (type(cache) is str and str(C.qualified_path(cache,layout=value['runtime']['stdlib_layout']))==row['cache_path']),'preserved observed ctypes cache field')
    return {'initial_profile':profile,'policy':'preserved_observed_native_source_cache','native_authority_issued':False}

def _registered_search_roots(context):
    C.stage_native_gate();value=C.context_value(context);import sys
    require=C.require;roots=[str(C.qualified_path(C.PACKAGE)),str(C.qualified_path(C.SITE))]
    require(all(Path(root).is_dir() for root in roots),'exact physical registered PACKAGE and SITE roots')
    C.revalidate_context(context)
    initial=sys.path;before=tuple(initial);require(list(before)==value['runtime']['cache_policy']['interpreter_profile']['path'],'immutable initial no-site path before registered root transition')
    for root in roots:
        if root not in sys.path:sys.path.append(root)
    return {'registered_roots':roots,'path_after':list(sys.path),'namespace_execution_proven':False,'original_container':initial,'before':before}

def _restore_registered_roots(transition):
    import sys
    C.require(sys.path is transition['original_container'] and sys.path==transition['path_after'],'owned registered root transition unchanged before restore')
    sys.path[:]=transition['before']

if __name__=='__main__':raise SystemExit(main())
