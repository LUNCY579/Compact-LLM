"""Actual registered tensor caller and original Windows lifecycle adaptation.

All public paths stage-refuse. These bodies are ready for separately reviewed
root integration, not a permission or current binary/runtime qualification.
"""
from pathlib import Path
import ast,hashlib,importlib.util,stat,time
HERE=Path(__file__).resolve().parent
CONTRACT_BOOTSTRAP={'bytes':36813,'sha256':'88cc80d8a730eab39af3c240e1becacc6fc3b35cd5ef0ec21885472a28d9663c','role':'python_source'}
def source_module(path,descriptor=None):
    path=Path(path).absolute()
    if descriptor is None and path==HERE/'prefix_tensor_contract.py':descriptor=CONTRACT_BOOTSTRAP
    def admitted():
        if path.suffix!='.py' or path.resolve()!=path:raise ValueError('exact physical Python source module')
        for part in (path,*path.parents):
            metadata=part.lstat()
            if part.is_symlink() or getattr(metadata,'st_file_attributes',0)&0x400:raise ValueError('compiled source has reparse alias')
        metadata=path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not 0<=metadata.st_size<=16*1024**2:raise ValueError('bounded regular source module')
        return metadata
    before=admitted()
    if descriptor is not None:
        if type(descriptor) is not dict or set(descriptor)!={'bytes','sha256','role'} or type(descriptor['bytes']) is not int or descriptor['bytes']!=before.st_size or descriptor['role']!='python_source' or type(descriptor['sha256']) is not str or len(descriptor['sha256'])!=64 or any(c not in '0123456789abcdef' for c in descriptor['sha256']):raise ValueError('exact selected source descriptor before read')
    if descriptor is None and HERE.name=='rineya_synth_centralized_reasoning_v1':raise ValueError('native bootstrap requires externally selected compiled source pin')
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=admitted()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('stable selected source bytes before compile')
    actual={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
    if descriptor is not None and actual!=descriptor:raise ValueError('exact selected compiled source descriptor')
    spec=importlib.util.spec_from_file_location('prefix_native_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__);module.__compiled_source_descriptor__=actual;return module
C=source_module(HERE/'prefix_tensor_contract.py')
def _registered_core_executor(worker):
    contract=worker.C;core=worker.source_core();core_contract=core['C']
    # Source core function/class bodies are unchanged except two literal status
    # names and the private native caller name. Binding is factory-qualified and
    # constructed by real_torch_bindings; CPU entry/results remain untouched.
    path=contract.CORE/'pipeline.py';raw=contract.read_bytes(path);contract.require(hashlib.sha256(raw).hexdigest()==contract.CORE_PINS['pipeline.py'],'pinned original pipeline source')
    function=next(n for n in ast.parse(raw.decode('utf-8'),str(path)).body if isinstance(n,ast.FunctionDef) and n.name=='run_cpu')
    replaced=[]
    class NativeNames(ast.NodeTransformer):
        def visit_Constant(self,node):
            table={'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED':'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT','CPU_PREFIX_INPUT_CHECKS_FAILED':'NATIVE_INPUT_CHECKS_FAILED'}
            if type(node.value) is str and node.value in table:replaced.append(node.value);return ast.copy_location(ast.Constant(table[node.value]),node)
            return node
    function=NativeNames().visit(function);function.name='_registered_tensor_execute';ast.fix_missing_locations(function)
    contract.require(replaced.count('CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED')==2 and replaced.count('CPU_PREFIX_INPUT_CHECKS_FAILED')==3,'exact five private native status adaptations')
    core_contract.CPU_SCOPE=contract.NATIVE_SCOPE
    core_contract.frame_helper=lambda:worker.exact_module(contract.PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py','253caa71942c300c225abef9f33d09f2ac28c4e6f169570ecbea09b926817d0b')
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),core)
    return core['_registered_tensor_execute']

def run_registered(config_path,owner,output,**anchors):
    C.stage_native_gate();context=C.validate_native_manifest(config_path,**anchors);selected=C.context_value(context)
    worker=source_module(HERE/'prefix_input_owned_worker.py',selected['config']['source_files'][str(HERE/'prefix_input_owned_worker.py')]);contract=worker.C;context=contract.validate_native_manifest(config_path,**anchors);value=contract.context_value(context)
    contract.require(contract.parse(contract.read_bytes(Path(output)/'supervisor_ownership.json'))==owner,'actual owned receipt identity before framework import')
    import native_worker
    native_worker.verify_worker_ownership(owner,config_path)
    api=worker.real_torch_bindings(context,owner,output);execute=_registered_core_executor(worker)
    class Host:
        def __init__(self):self.output=Path(output);self.job_name=owner['job_name'];self.supervisor_pid=owner['supervisor_pid'];self.started=time.monotonic()
        clock=staticmethod(time.monotonic);wall=staticmethod(time.time)
        def owned_resources(self,name):
            import native_worker
            return native_worker.owned_job_resources(name)
        def heartbeat(self,path):
            import native_worker
            return native_worker.read_heartbeat(path)
        def available(self):
            import worker_ram_probe
            return worker_ram_probe.available_bytes()
        def guard_sample(self,stage):pass  # actual100ms/.5s original external monitor records this scope
    host=Host();guard=worker.original_guard(host,owner,host.started)
    report=execute(api,guard,output,fixture_identity=contract.NATIVE_SCOPE,source_pins={p:d['sha256'] for p,d in value['config']['source_files'].items()},runtime_pins={p:d['sha256'] for p,d in value['runtime']['files'].items()})
    # The status is a returned primitive report; only independent terminal audit
    # can accept actual owned cleanup/source/runtime/API/IO/resource closure.
    return report

def run_source_fixture(manifest,owner,api,host,output):
    # Separate CPU capability. This calls the actual private native source body;
    # neither a NativeContext nor a production real-API binding can be minted.
    C.validate_fixture(manifest);C.require(host.membership(owner) is True,'explicit CPU assigned/resumed fixture birth')
    token=C.mint_cpu_admission(manifest,owner);C.verify_cpu_admission(token,manifest,owner)
    worker=source_module(HERE/'prefix_input_owned_worker.py');execute=_registered_core_executor(worker)
    report=execute(api,worker.original_guard(host,owner,host.started),output,fixture_identity=C.NATIVE_SCOPE,source_pins={p:d['sha256'] for p,d in manifest['source_pins'].items()},runtime_pins={})
    return {'kind':'CPU_NATIVE_CALLER_SOURCE_SIMULATION','status':'CPU_NATIVE_SOURCE_RETURNED_UNQUALIFIED','fixture_only':True,'native_PASS':False,'native_authority_issued':False,'report':report}
def supervise_registered(config_path,output_dir,**anchors):
    C.stage_native_gate();context=C.validate_native_manifest(config_path,**anchors);value=C.context_value(context)
    import native_supervisor
    path=C.PACKAGE/'native_supervisor.py';raw=C.read_bytes(path);C.require(hashlib.sha256(raw).hexdigest()==value['config']['helper_pins']['native_supervisor.py'],'unchanged original owned lifecycle source')
    function=next(n for n in ast.parse(raw.decode('utf-8'),str(path)).body if isinstance(n,ast.FunctionDef) and n.name=='supervise');changes=[]
    class NewScope(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='native_worker.py':changes.append('worker_path');return ast.copy_location(ast.Constant('prefix_input_owned_worker.py'),node)
            if node.value=='mechanical_native_qualification_only':changes.append('scope');return ast.copy_location(ast.Constant(C.NATIVE_SCOPE),node)
            return node
    function=NewScope().visit(function);ast.fix_missing_locations(function);C.require(changes.count('worker_path')==2 and changes.count('scope')==1,'exact three original lifecycle adaptations')
    namespace=dict(vars(native_supervisor));original_new=native_supervisor._new_output
    from short_ram_monitor import ShortRamMonitor
    namespace['_Monitor']=ShortRamMonitor
    def new_output(request):
        output=original_new(request);native_supervisor._json_new(output/'OWNED_EXTERNAL_ANCHORS.json',{'config_path':str(Path(config_path).absolute()),'anchors':anchors});return output
    def preflight(request):
        C.revalidate_context(context);config=value['config'];C.require(Path(request).absolute()==Path(config_path).absolute(),'exact owned config path')
        return {'config_sha256':anchors['expected_config_file'],'config_path':str(Path(config_path).absolute()),'worker_path':str(C.PACKAGE/'prefix_input_owned_worker.py'),'authorization':config['authorization'],'resources':config['resources'],'source_commit':config['source_commit'],'profile_sha256':value['registration']['profile_sha256'],'source_inventory_sha256':C.digest(config['source_files']),'runtime_inventory_sha256':C.digest(value['runtime']['files'])}
    namespace['_preflight']=preflight;namespace['_new_output']=new_output
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['supervise'](config_path,output_dir)
