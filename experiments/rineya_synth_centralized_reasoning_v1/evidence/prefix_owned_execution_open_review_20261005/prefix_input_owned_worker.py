"""Separate owned caller. CPU simulation cannot be converted to native work."""
from pathlib import Path
import ast,hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent
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
    spec=importlib.util.spec_from_file_location('prefix_owned_exact_'+path.stem,path);module=importlib.util.module_from_spec(spec);exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__);return module
C=exact_module(HERE/'prefix_tensor_contract.py','b3d88de97266b7d5466fc9bcaa35c985c8d80ead1602059f23ce91406cd8522f')

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
    C.stage_native_gate();value=C.context_value(context);C.require(value['config']['scope']==C.NATIVE_SCOPE,'separate registered tensor purpose')
    import random,shutil
    torch_holder=[None]
    def lazy_torch():
        import torch
        torch_holder[0]=torch;return torch
    def configure():
        torch=torch_holder[0];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    def growth():
        import psutil
        return {'available_bytes':psutil.virtual_memory().available,'worker_RSS_bytes':psutil.Process().memory_info().rss}
    import native_worker
    def preflight():
        current=C.revalidate_context(context);return {'source_commit':current['expected_head'],'config_sha256':current['config']['sha256'],'scope':C.NATIVE_SCOPE,'permission_sha256':C.digest(current['config']['permission']),'source_inventory_sha256':C.digest(current['config']['source_files']),'runtime_inventory_sha256':C.digest(current['runtime']['files'])}
    return {'config_preflight':preflight,'ownership_verify':lambda:native_worker.verify_worker_ownership(owner,value['config_path']),
       'disk_free':lambda:shutil.disk_usage(output).free,'torch_import':lazy_torch,'torch_version_read':lambda:torch_holder[0].__version__,
       'runtime_version':value['config']['torch_version'],'growth_observation':growth,'configure':configure,'python_seed':random.seed}

def run_native(*args,**kwargs):
    C.stage_native_gate()
    return _integrated_native_worker(*args,**kwargs)

def _integrated_native_worker(context,owner,output):
    value=C.revalidate_context(context)
    pipeline=exact_module(C.PACKAGE/'prefix_native_pipeline.py',value['config']['source_files'][str(C.PACKAGE/'prefix_native_pipeline.py')]['sha256'])
    return pipeline.run_registered(value['config_path'],owner,output,**value['anchors'])
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
        from prefix_input_owned_supervisor import write_new
        actual={'attempted':1,'returned':0,'failed':0,'pending':1};projected={'attempted':1,'returned':1,'failed':0,'pending':0}
        receipt={'report':report,'runtime_execution_modules':finder.observed,'owned_membership_verified':verified,'IO':{'owned_return_write':projected},'native_PASS':False}
        try:write_new(output/'OWNED_WORKER_RETURN.json',receipt)
        except BaseException:actual.update(failed=1,pending=0);raise
        actual.update(returned=1,pending=0);C.require(actual==projected,'actual owned return durable/write/readback counters reconcile')
        return 0 if report['status']=='NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT' else 2
    finally:loader.remove(finder)
if __name__=='__main__':raise SystemExit(main())
