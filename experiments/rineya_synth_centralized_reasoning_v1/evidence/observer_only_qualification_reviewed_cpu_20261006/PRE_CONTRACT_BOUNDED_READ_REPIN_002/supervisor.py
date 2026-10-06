"""Observer-only original supervised lifecycle; no native work at import.

The exact pinned original supervise body supplies containment, lease, admission,
watchdogs and cleanup. Only two worker constants and one purpose constant change.
CPU fixtures run that same adapted body with a separate capability/inert host.
"""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,json,math,stat

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
SCOPE='NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD'
ORIGINAL_SHA='edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691'
CONTRACT_BOOTSTRAP_SHA='822416b36e6eb6ad86d2d66bf99b5d6966bc195b9f5b5b501bc6a4d12a530f35'
WORKER_RELATIVE='observer_only_qualification_v1/worker.py'
BOUNDS={'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'worker_process_tree_RSS_GiB':2.0,
 'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'disk_free_GiB':4.0,
 'external_watchdog_milliseconds':100,'whole_scope_seconds':600}
FIXED={'GIB':1024**3,'BOUNDS':BOUNDS,'CPU_INTERVAL':.1,'CPU_FRESHNESS':.2,'GPU_INTERVAL':.5,
       'GPU_FRESHNESS':2.,'GPU_TIMEOUT':2.,'ADMISSION_INTERVAL':2.,'ADMISSION_DEADLINE':30.}

def require(value,message):
    if not value:raise ValueError(message)
def cleanup_verified(receipt):
    """No success authority from exit code alone; original receipt stays intact."""
    return (receipt.get('status')=='SUPERVISED_WORKER_EXITED_ZERO' and
            type(receipt.get('returncode')) is int and receipt['returncode']==0 and
            all(receipt.get(key) is True for key in ('worker_attempted','worker_launched','worker_resumed',
                'sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified','partial_outputs_frozen','job_closed_kill_on_close')) and
            receipt.get('owned_worker_identity_currently_alive') is False and receipt.get('unjoined_sampler_threads')==[])
def read_source(path):
    path=Path(path).absolute();require(path.resolve()==path and path.suffix=='.py','exact physical source path')
    for part in (path,*path.parents):
        info=part.lstat();require(not part.is_symlink() and not(getattr(info,'st_file_attributes',0)&0x400),'source reparse refused')
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=16*1024**2,'bounded regular source')
    with path.open('rb') as stream:raw=stream.read(before.st_size+1)
    after=path.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'stable source read')
    return raw
def contract_module():
    path=HERE/'contract.py';raw=read_source(path)
    require(hashlib.sha256(raw).hexdigest()==CONTRACT_BOOTSTRAP_SHA,'exact trusted observer-only contract bootstrap bytes')
    module=ModuleType('selected_observer_only_contract');module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module

def adapted_body(raw,namespace):
    require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,'exact original supervisor source SHA')
    tree=ast.parse(raw,str(PACKAGE/'native_supervisor.py'))
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='supervise')
    changes=[]
    class Purpose(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='native_worker.py':changes.append('worker');return ast.copy_location(ast.Constant(WORKER_RELATIVE),node)
            if node.value=='mechanical_native_qualification_only':changes.append('purpose');return ast.copy_location(ast.Constant(SCOPE),node)
            return node
    body=Purpose().visit(body);ast.fix_missing_locations(body)
    require(changes==['worker','purpose','worker'],'exact three source-derived observer lifecycle adaptations')
    exec(compile(ast.Module(body=[body],type_ignores=[]),str(PACKAGE/'native_supervisor.py'),'exec'),namespace)
    return namespace['supervise'],{'source_sha256':ORIGINAL_SHA,'changes':changes,'mutation_count':3,'original_body':'supervise'}

def preflight_projection(contract,context,request,*,cpu_capability=None):
    value=contract.context_value(context)
    require(Path(request).absolute()==Path(value['config_path']).absolute(),'original observer-only config path')
    if cpu_capability is None:current=contract.revalidate_context(context)
    else:current=contract._CPU_revalidate_context(context,_capability=cpu_capability)
    require(current==value,'original observer-only context cannot rebase')
    require(value['config']['kind']=='REGISTERED_OBSERVER_ONLY_QUALIFICATION_V1' and value['config']['resources']==contract.RESOURCES,'registered observer-only scope/resources')
    expected=contract.PACKAGE/WORKER_RELATIVE
    require(value['config']['worker_path']==str(expected),'exact observer-only worker entry')
    projected=contract.preflight(context)
    require(projected['worker_path']==str(expected) and projected['authorization']['scope']==SCOPE and
        all(projected['authorization'].get(name) is False for name in ('production_training','protected_bank_access','paid_services')),
        'new observer-only exact purpose projection')
    return projected

def supervise(config_path,output_dir,**anchors):
    contract=contract_module();contract.native_gate()
    context=contract.validate_context(config_path,**anchors);value=contract.context_value(context)
    require(value.get('fixture_only',False) is False,'CPU context cannot dispatch observer-only native worker')
    path=contract.PACKAGE/'native_supervisor.py';raw=read_source(path)
    require(value['config']['source_files'][str(path)]['sha256']==ORIGINAL_SHA and hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,
            'original supervisor selected source pin and compile bytes')
    original=ModuleType('observer_selected_original_supervisor');original.__file__=str(path)
    exec(compile(raw,str(path),'exec'),original.__dict__)
    namespace=dict(original.__dict__)
    namespace['_preflight']=lambda request:preflight_projection(contract,context,request)
    original_new=namespace['_new_output']
    def output(request):
        leaf=original_new(request)
        original._json_new(leaf/'OWNED_EXTERNAL_ANCHORS.json',{'config_path':value['config_path'],'anchors':value['anchors']})
        return leaf
    namespace['_new_output']=output
    body,source_proof=adapted_body(raw,namespace)
    receipt=body(config_path,output_dir)
    # The original terminal/lease bytes stay identical to the returned receipt.
    # Source adaptation/observer scope are a separate return envelope.
    return {'kind':'OBSERVER_ONLY_SUPERVISOR_RETURN','scope':SCOPE,'source_adaptation':source_proof,
            'receipt':receipt,'cleanup_verified':cleanup_verified(receipt),'native_PASS':False,'native_authority_issued':False}

def _capability_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private CPU supervisor capability')
    return get,check
_CPU_capability,_check_CPU_capability=_capability_factory()

def _CPU_supervise_fixture(context,host,output_dir,*,contract,_capability,admission_only=False):
    _check_CPU_capability(_capability);value=contract.context_value(context)
    require(value.get('fixture_only') is True and value.get('native_PASS') is value.get('native_authority_issued') is False,'explicit CPU observer-only original context')
    raw=read_source(PACKAGE/'native_supervisor.py')
    require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,'original lifecycle bytes before helper/function compile')
    tree=ast.parse(raw,str(PACKAGE/'native_supervisor.py'))
    def output(request):
        leaf=host.new_output(request)
        host.write_new(leaf/'OWNED_EXTERNAL_ANCHORS.json',{'config_path':value['config_path'],'anchors':value['anchors']})
        return leaf
    namespace={**FIXED,'Path':Path,'math':math,'PACKAGE':contract.PACKAGE,'sys':SimpleNamespace(executable='CPU_INERT_PYTHON'),
        'time':SimpleNamespace(monotonic=host.clock,sleep=host.sleep),'_utc':host.utc,'_json_safe':lambda v:v,
        '_new_output':output,'_json_new':host.write_new,'_acquire_lease':host.acquire_lease,
        '_release_lease':host.release_lease,'resource_snapshot':host.observation,'_cpu_snapshot':host.cpu_observation,
        '_WindowsOwnedWorker':host.worker,'_Monitor':host.monitor,'_shutdown_gpu_probe':host.close_NVML,
        '_gpu_pending_threads':host.pending_threads,'_pid_alive':host.current_alive,'_inventory':host.inventory,
        '_LaunchFailure':host.LaunchFailure}
    # Actual original admission/bounds predicates execute; they are not mirrored.
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_bounds_valid','_fresh_cpu','_fresh_gpu','_admission_failure')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(PACKAGE/'native_supervisor.py'),'exec'),namespace)
    cap=contract._CPU_capability()
    namespace['_preflight']=lambda request:preflight_projection(contract,context,request,cpu_capability=cap)
    body,proof=adapted_body(raw,namespace)
    result=body(value['config_path'],output_dir,admission_only=admission_only)
    return {'kind':'CPU_ORIGINAL_OBSERVER_ONLY_SUPERVISOR_BODY_FIXTURE','fixture_only':True,'native_PASS':False,
            'native_authority_issued':False,'scope':SCOPE,'source_adaptation':proof,'receipt':result,'cleanup_verified':cleanup_verified(result)}

if __name__=='__main__':
    contract=contract_module();contract.native_gate()
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True);args=parser.parse_args()
    anchors=contract.parse(contract.read_bytes(args.anchors))
    result=supervise(args.config,args.output_dir,**anchors)
    raise SystemExit(0 if result['cleanup_verified'] else 2)
