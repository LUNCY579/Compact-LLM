"""Separate owned caller. CPU simulation cannot be converted to native work."""
from pathlib import Path
import ast,hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent
COMPILED_SOURCE_OBSERVATIONS={}
OWNED_TOPOLOGY_OBSERVATIONS={}
def exact_module(path,pin=None):
    path=Path(path).absolute()
    if path.resolve()!=path:raise ValueError('exact module physical path')
    for part in (path,*path.parents):
        info=part.lstat()
        if part.is_symlink() or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('module reparse alias')
    before=path.lstat()
    if not path.is_file() or path.is_symlink() or not 0<=before.st_size<=16*1024**2:raise ValueError('bounded regular source module')
    with path.open('rb') as stream:raw=stream.read(before.st_size+1)
    after=path.lstat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or (pin is not None and hashlib.sha256(raw).hexdigest()!=pin):raise ValueError('exact stable source module')
    spec=importlib.util.spec_from_file_location('prefix_owned_exact_'+path.stem,path);module=importlib.util.module_from_spec(spec);exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__)
    observation={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'loader_kind':'explicit_selected_source_compile','current_module_execution_observed':True}
    module.__compiled_source_descriptor__=observation;COMPILED_SOURCE_OBSERVATIONS[str(path)]=observation;return module
C=exact_module(HERE/'prefix_tensor_contract.py','88cc80d8a730eab39af3c240e1becacc6fc3b35cd5ef0ec21885472a28d9663c')

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

def real_torch_bindings(context,owner,output):
    # Only a validated integrated supervisor may reach this factory. No caller
    # can pass a fake Torch object or select a model API.
    C.stage_native_gate();value=C.context_value(context);C.require(value.get('fixture_only',False) is False,'CPU context cannot create real native bindings')
    return _tensor_binding_body(context,owner,output,value,_native_binding_suppliers(),lambda:C.revalidate_context(context))['api']
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

def _integrated_native_worker(context,owner,output):
    value=C.revalidate_context(context)
    pipeline=exact_module(C.PACKAGE/'prefix_native_pipeline.py',value['config']['source_files'][str(C.PACKAGE/'prefix_native_pipeline.py')]['sha256'])
    return pipeline.run_registered(value['config_path'],owner,output,**value['anchors'])
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
def main():
    C.stage_native_gate()
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    output=args.output_dir.absolute();C.require(output.parent==C.PACKAGE/'artifacts' and output.is_dir(),'owned output leaf')
    external=C.parse(C.read_bytes(output/'OWNED_EXTERNAL_ANCHORS.json'));C.require(external['config_path']==str(args.config.absolute()),'owned external anchor config join')
    context=C.validate_native_manifest(args.config,**external['anchors']);value=C.context_value(context)
    owner=C.parse(C.read_bytes(output/'supervisor_ownership.json'))
    loader=exact_module(C.PACKAGE/'prefix_registered_loader.py',value['config']['source_files'][str(C.PACKAGE/'prefix_registered_loader.py')]['sha256']);finder=loader.install(context,C)
    try:
        import native_worker,os
        verified=native_worker.verify_worker_ownership(owner,args.config)
        for key,setting in value['config']['offline_environment'].items():os.environ[key]=setting
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        report=_integrated_native_worker(context,owner,output)
        write_owned_return(output,report,finder.observed,verified,publisher_descriptor=value['config']['source_files'][str(C.PACKAGE/'prefix_input_owned_supervisor.py')])
        return 0 if report['status']=='NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT' else 2
    finally:loader.remove(finder)
if __name__=='__main__':raise SystemExit(main())
