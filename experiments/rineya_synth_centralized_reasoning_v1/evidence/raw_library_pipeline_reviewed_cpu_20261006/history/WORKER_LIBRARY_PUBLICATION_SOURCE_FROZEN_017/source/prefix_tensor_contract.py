"""New two-tensor purpose, immutable injected admission and stage-first gate."""
from pathlib import Path
import hashlib,json,math,stat,time
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in (HERE,*HERE.parents) if p.name=='rineya_synth_centralized_reasoning_v1')
CPU_SCOPE='CPU_OWNED_TENSOR_PIPELINE_SIMULATION'
NATIVE_SCOPE='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
INPUT_FILE='195f1b3a658959ccbe517ce5aefaef1d0d2c298db0095ace08f744bd6488feb5'
INPUT_BODY='a22c5b27225fc75ae697a5f06a96da18c1cb995a7bf7c6d03eb05f0cb6f06ec9'
PREFIX_SHA='e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1'
CORE=PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005'
CORE_PINS={'contract.py':'b83228e3ae797d2a410b459f73662daba75ecb6f203f3edd8edaf206eb757836','pipeline.py':'700d6934e0fb36b158eacfe5f53330e5e498d9fd9a2bf264b40a077c95bbb9c4','fixture_support.py':'a8d42ca5bb1f481c54db5a76e9eaf9605d486e429a7f422211529d3194426eff','test_prefix_input.py':'9fc7797cde34a1c4c48334b38857b7f682f0ec9e8c1584bca1d2b10563e04924'}
RESOURCES={'CPU_sample_max_age_seconds':.2,'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'GPU_probe_interval_seconds':.5,'GPU_probe_timeout_seconds':2.0,'GPU_sample_max_age_seconds':2.0,'admission_poll_seconds':2.0,'admission_seconds':30,'disk_free_GiB':4.0,'external_watchdog_milliseconds':100,'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'whole_scope_seconds':600,'worker_process_tree_RSS_GiB':2.0}
PERMISSION={'allocations':2,'dtype':'int64','shape':[1,422],'logical_bytes':6752,'model':False,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'M0_membership':False}
RAW_CORE=('PHASE_OBSERVATIONS.jsonl','ownership_verified.json','input_check_summary.json','worker_result.json','PHASE_ACCOUNTING_FINAL.json')
TIMELINE=('admission_first','admission_second','worker_created_suspended','Job_assigned','before_resume_admission','worker_resumed','worker_returned','worker_dead','samplers_joined','NVML_closed','Job_closed','reservation_released','lease_terminal')
MAX_JSON=16*1024**2
QUALIFIED_LAYOUT={'logical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none','physical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none','file_attributes':1040,'reparse_tag':2684354563,'is_symlink':False,'python_version':'3.11.15'}
SITE=Path(r'C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages')
WINDOWS_SYSTEM_DIRECTORY=Path(r'C:\Windows\System32')
EXTERNAL_PACKAGES=frozenset(('psutil','safetensors','transformers','filelock','typing_extensions','sympy','networkx','fsspec','jinja2','markupsafe','numpy','packaging','mpmath','torchgen','functorch'))
STDLIB_CANDIDATE=PACKAGE/'evidence/stdlib_inventory_reviewed_cpu_20261005/STDLIB_CANDIDATE_001.json'

class QualificationBudget:
    """One read/hash/metadata deadline; independent of the600s worker scope."""
    def __init__(self,clock=None):self.clock=time.monotonic if clock is None else clock;self.started=self.clock();self.total=0
    def check(self,amount=0):
        require(type(amount) is int and amount>=0,'nonnegative qualification byte accounting');self.total+=amount
        require(self.total<=16*1024**3 and self.clock()-self.started<=600,'bounded complete qualification deadline/total')
    def remaining(self):self.check();return max(.001,600-(self.clock()-self.started))

def qualified_path(path,layout=None):
    path=Path(path).absolute()
    if layout is not None:
        require(layout==QUALIFIED_LAYOUT,'exact separately qualified interpreter alias/layout')
        logical=Path(layout['logical_base_prefix']);base=Path(layout['physical_base_prefix'])
        if path.is_relative_to(logical):
            info=logical.lstat();require(not logical.is_symlink() and getattr(info,'st_file_attributes',0)==layout['file_attributes'] and getattr(info,'st_reparse_tag',None)==layout['reparse_tag'] and logical.resolve()==base,'qualified known interpreter alias changed')
            resolved=base/path.relative_to(logical);require(path.resolve()==resolved,'known alias contained mapping changed');path=resolved
    require(path.resolve()==path,'exact physical path')
    for part in (path,*path.parents):
        meta=part.lstat();require(not part.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'no reparse alias')
    return path

def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def pairs(items):
    value={}
    for k,v in items:require(k not in value,'duplicate JSON key');value[k]=v
    return value
def floating(v):
    n=float(v);require(math.isfinite(n),'nonfinite/overflow JSON float');return n
def constant(v):raise ValueError('nonfinite JSON constant')
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_float=floating,parse_constant=constant)
def read_bytes(path,maximum=MAX_JSON,*,layout=None,budget=None):
    if budget is not None:budget.check()
    path=qualified_path(path,layout)
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=maximum,'bounded regular file')
    if budget is not None:budget.check(before.st_size)
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'stable bounded bytes')
    if budget is not None:budget.check()
    return raw
def load_input(*,budget=None):
    raw=read_bytes(INPUT,16384,budget=budget);require(hashlib.sha256(raw).hexdigest()==INPUT_FILE,'fixed exposed prefix file');v=parse(raw)
    require(v['sha256']==INPUT_BODY==digest({k:x for k,x in v.items() if k!='sha256'}),'fixed exposed input body')
    require(len(v['prefix_ids'])==422 and all(type(x) is int and 0<=x<151936 for x in v['prefix_ids']) and v['attention_mask']==[1]*422 and digest(v['prefix_ids'])==PREFIX_SHA,'fixed422 input/mask')
    require(v['native_launch_authorized'] is v['M0_population_membership_allowed'] is v['training_or_grading_allowed'] is False,'exposed input confers no study authority')
    return v
def identity(value):
    require(type(value) is dict and set(value)=={'pid','creation_filetime_100ns'} and all(type(x) is int and x>0 for x in value.values()),'exact PID/FILETIME birth')
    return dict(value)
def fixture_manifest(source_pins=None):
    value={'schema_version':1,'kind':CPU_SCOPE,'scope':CPU_SCOPE,'permission':PERMISSION,'resources':RESOURCES,'input_file_sha256':INPUT_FILE,'input_body_sha256':INPUT_BODY,'prefix_ids_sha256':PREFIX_SHA,
      'source_pins':{} if source_pins is None else dict(source_pins),'runtime_pins':{},'native_runtime_closure_complete':False,'cache_execution_policy_qualified':False,'external_dependencies_qualified':False,'native_launch_authorized':False}
    value=parse(canonical(value));value['sha256']=digest(value);return value
def validate_fixture(value):
    require(type(value) is dict and value.get('kind')==CPU_SCOPE and value.get('scope')==CPU_SCOPE,'explicit CPU owned simulation')
    require(set(value)=={'schema_version','kind','scope','permission','resources','input_file_sha256','input_body_sha256','prefix_ids_sha256','source_pins','runtime_pins','native_runtime_closure_complete','cache_execution_policy_qualified','external_dependencies_qualified','native_launch_authorized','sha256'} and type(value['schema_version']) is int and value['schema_version']==1,'exact CPU manifest fieldset')
    require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'immutable manifest checksum')
    require(value['permission']==PERMISSION and value['resources']==RESOURCES,'exact tensor permission/resources')
    require(value['input_file_sha256']==INPUT_FILE and value['input_body_sha256']==INPUT_BODY and value['prefix_ids_sha256']==PREFIX_SHA,'exact input anchors')
    require(all(value[k] is False for k in ('native_runtime_closure_complete','cache_execution_policy_qualified','external_dependencies_qualified','native_launch_authorized')),'CPU evidence cannot authorize native')
    require(value['runtime_pins']=={} and type(value['source_pins']) is dict and 0<len(value['source_pins'])<=4096,'CPU source closure only; no fabricated runtime qualification')
    for path,descriptor in value['source_pins'].items():
        require(type(path) is str and Path(path).suffix=='.py' and type(descriptor) is dict and set(descriptor)=={'bytes','sha256','role'} and descriptor['role']=='python_source' and type(descriptor['bytes']) is int and 0<=descriptor['bytes']<=16*1024**2 and sha(descriptor['sha256']),'exact finite CPU source descriptors')
    return value

def _admission_factory():
    records={}
    class Admission:
        __slots__=()
    def mint(manifest,owner):
        validate_fixture(manifest);identity(owner);token=Admission();records[token]=(canonical(manifest),canonical(owner));return token
    def check(token,manifest,owner):
        require(type(token) is Admission and records.get(token)==(canonical(manifest),canonical(identity(owner))),'unmodified factory admission and exact owned birth')
        validate_fixture(manifest)
    return mint,check
mint_cpu_admission,verify_cpu_admission=_admission_factory()

def stage_native_gate():
    # Integration must install a separately reviewed direct package module.
    require(HERE==PACKAGE,'ignored/preserved stage has no native authority')
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def sealed_json(path,file_pin,body_pin,*,budget=None):
    require(sha(file_pin) and sha(body_pin),'external SHA256 anchors');raw=read_bytes(path,budget=budget);require(hashlib.sha256(raw).hexdigest()==file_pin,'external file anchor');value=parse(raw)
    require(type(value) is dict and value.get('sha256')==body_pin==digest({k:v for k,v in value.items() if k!='sha256'}),'external body anchor');return value
def file_descriptor(path,expected,*,layout=None,budget=None):
    require(type(expected) is dict and set(expected)=={'bytes','sha256','role'} and type(expected['bytes']) is int and 0<=expected['bytes']<=8*1024**3 and sha(expected['sha256']),'typed registered source/runtime descriptor')
    if budget is not None:budget.check()
    path=qualified_path(path,layout)
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'],'registered exact runtime size');hasher=hashlib.sha256();remaining=before.st_size
    if budget is not None:budget.check(before.st_size)
    with path.open('rb') as stream:
        while remaining:
            chunk=stream.read(min(1024**2,remaining));require(bool(chunk),'complete bounded runtime hash');hasher.update(chunk);remaining-=len(chunk)
            if budget is not None:budget.check()
        require(not stream.read(1),'registered runtime did not grow')
    after=path.lstat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and hasher.hexdigest()==expected['sha256'],'registered runtime exact stable bytes')
def native_metadata(config,registration,runtime,expected_head,*,budget=None):
    # Pure schema/join validation is independently CPU-testable; never a permit.
    require(type(expected_head) is str and len(expected_head)==40 and all(c in '0123456789abcdef' for c in expected_head),'exact full source commit')
    require(set(config)=={'schema_version','kind','scope','permission','resources','input_file_sha256','input_body_sha256','prefix_ids_sha256','source_commit','native_launch_authorized','source_files','runtime_files','helper_pins','torch_version','authorization','offline_environment','tolerances','runtime_versions','GPU_physical_ordinal','sha256'},'exact native config fieldset')
    require(set(registration)=={'kind','source_commit','native_launch_authorized','config_body_sha256','runtime_body_sha256','source_files','runtime_files','config_path','qualification_review_path','qualification_review_file_sha256','qualification_review_body_sha256','profile_sha256','git_output_exceptions','sha256'},'exact external registration fieldset')
    runtime_fields={'kind','source_commit','current_binary_bytes_qualified','external_dependencies_complete','interpreter_layout_qualified','cache_execution_policy_qualified','files','inherited65_map','stdlib779_map','cache_policy','external_dependency_files','rosters','library_search_policy','stdlib_layout','sha256'}
    require(runtime_fields<=set(runtime)<=runtime_fields|{'windows_library_identities','library_boundary_plan'},'exact runtime qualifier fieldset')
    require(config['kind']=='REGISTERED_ONE_PREFIX_TENSOR_OWNED_DIAGNOSTIC' and config['scope']==NATIVE_SCOPE and config['permission']==PERMISSION and config['resources']==RESOURCES,'new exact native tensor purpose/resource contract')
    require(config['input_file_sha256']==INPUT_FILE and config['input_body_sha256']==INPUT_BODY and config['prefix_ids_sha256']==PREFIX_SHA,'fixed exposed prefix anchors')
    require(config['source_commit']==registration['source_commit']==runtime['source_commit']==expected_head,'exact registered source HEAD')
    require(config['native_launch_authorized'] is True and registration['native_launch_authorized'] is True,'separately registered owned permission')
    require(config['authorization']=={'scope':NATIVE_SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False,'model':False,'forward':False,'decoder':False,'scoring':False},'exact new tensor-only authorization')
    require(registration['profile_sha256']==digest({'permission':PERMISSION,'resources':RESOURCES,'torch_version':config['torch_version']}),'external exact purpose/profile digest')
    require(registration['kind']=='ROOT_REVIEWED_ONE_PREFIX_OWNED_REGISTRATION' and registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'],'external exact registration joins')
    require(runtime['kind']=='ROOT_QUALIFIED_TENSOR_RUNTIME_SOURCE_BYTE_CLOSURE' and all(runtime[k] is True for k in ('current_binary_bytes_qualified','external_dependencies_complete','interpreter_layout_qualified','cache_execution_policy_qualified')),'full current binary/external/interpreter/cache qualification')
    files=runtime['files'];sources=config['source_files'];require(type(files) is dict and type(sources) is dict and 0<len(files)<=32768 and 0<len(sources)<=4096,'finite complete byte closure')
    require(config['runtime_files']==files and registration['source_files']==sources and registration['runtime_files']==files,'same exact worker/preflight/auditor byte maps')
    original=sealed_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62',budget=budget)
    require(runtime['inherited65_map']==original['runtime_pins'] and all(p in files and files[p]['sha256']==h for p,h in original['runtime_pins'].items()),'original exact65 preserved inside full closure')
    stdlib=sealed_json(STDLIB_CANDIDATE,'e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5','f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149',budget=budget)
    require(runtime['stdlib779_map']==stdlib['inventory']['runtime_pins'] and len(runtime['stdlib779_map'])==779 and all(p in files and files[p]['sha256']==h for p,h in runtime['stdlib779_map'].items()),'preserved exact779 stdlib byte superset inside full closure')
    require(config['helper_pins']==original['helper_pins'],'unchanged owned/resource helper identity')
    require(config['offline_environment']==original['offline_environment'] and config['tolerances']==original['tolerances'] and config['runtime_versions']==original['runtime_versions'] and config['torch_version']==original['runtime_versions']['torch'] and type(config['GPU_physical_ordinal']) is int and config['GPU_physical_ordinal']==0,'unchanged offline/numeric/version semantics and CUDA0/NVML0 binding')
    required={'prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_input_owned_supervisor.py','prefix_input_owned_auditor.py','prefix_registered_loader.py','prefix_native_pipeline.py'}
    require(required<={Path(p).name for p in sources} and all(str(PACKAGE/name) in sources for name in required),'all new exact direct-package source boundaries')
    require(all(str(PACKAGE/name) in sources and sources[str(PACKAGE/name)]['sha256']==pin for name,pin in original['helper_pins'].items()),'helper source map byte pins')
    cache=runtime['cache_policy'];require(type(cache) is dict and set(cache)=={'mode','allow_unregistered','source_loader_sha256','Torch_root','preloaded_module_cache_pins','namespace_roots','preloaded_bootstrap_modules','fresh_bootstrap_evidence'},'exact qualified cache/bootstrap policy schema');require(cache['mode']=='all_registered_python_source_compiled_without_cache' and cache['allow_unregistered'] is False and cache['source_loader_sha256']==sources[str(PACKAGE/'prefix_registered_loader.py')]['sha256'] and cache['Torch_root']==str(SITE/'torch'),'qualified fail-closed source/cache loader')
    require(set(runtime['external_dependency_files'])<=set(files) and runtime['external_dependency_files'],'explicit nonempty external bootstrap dependency closure')
    windows=_windows_library_identities(runtime,budget=budget)
    for mapping in (files,sources):
        for path,entry in mapping.items():
            require(type(path) is str and 0<len(path)<=4096 and type(entry) is dict and set(entry)=={'bytes','sha256','role'} and type(entry['bytes']) is int and 0<=entry['bytes']<=8*1024**3 and sha(entry['sha256']),'typed exact closure map')
            allowed={'python_source':{'.py'},'python_stub':{'.pyi'},'source_json':{'.json'},'source_markdown':{'.md'},'qualified_bytecode_cache':{'.pyc'},'extension_binary':{'.pyd','.dll'},'torch_binary':{'.pyd','.dll'},'interpreter_binary':{'.pyd','.dll','.exe','.zip'},'windows_system_binary':{'.dll'}}
            require(entry['role'] in allowed and Path(path).suffix.lower() in allowed[entry['role']],'exact source/runtime role suffix; no arbitrary payload')
            require(entry['bytes']>0 or entry['role'] in ('python_source','python_stub','source_markdown'),'nonempty binary/cache/JSON closure bytes; empty Python sources permitted')
            if mapping is files:require(entry['role']==('windows_system_binary' if path in windows else runtime_role(path,runtime['inherited65_map'])),'exact runtime root/role/suffix assignment')
            if budget is not None:budget.check()
    require(not(set(files)&set(sources)) or all(files[p]==sources[p] for p in set(files)&set(sources)),'conflict-free runtime/source overlap')
    library_boundary_plan(runtime)
    _git_exception_policy(registration['git_output_exceptions'],sources,files)
    return {'config':config,'registration':registration,'runtime':runtime,'expected_head':expected_head}
def runtime_role(text,inherited):
    p=Path(text);suffix=p.suffix.lower();base=Path(QUALIFIED_LAYOUT['physical_base_prefix'])
    require(p.is_absolute() and '..' not in p.parts and str(p)==text,'exact canonical absolute runtime path')
    if suffix=='.pyc':
        require(p.parent.name=='__pycache__' and (p.is_relative_to(base/'Lib') or p.is_relative_to(SITE)),'cache only in qualified Python runtime roots');return 'qualified_bytecode_cache'
    if p.is_relative_to(base/'Lib'):require(suffix=='.py','stdlib source suffix');return 'python_source'
    if p.is_relative_to(base/'DLLs'):require(suffix in ('.pyd','.dll'),'stdlib extension suffix');return 'extension_binary'
    if p.parent==base or p==SITE.parents[1]/'Scripts/python.exe':require(suffix in ('.exe','.dll','.zip'),'interpreter root suffix');return 'interpreter_binary'
    if p.is_relative_to(SITE):
        relative=p.relative_to(SITE);package=relative.parts[0]
        require(package=='torch' or package in EXTERNAL_PACKAGES or p==SITE/'typing_extensions.py','finite declared external runtime root')
        require(suffix in ('.py','.pyi','.pyd','.dll'),'external source/extension suffix')
        return 'python_source' if suffix=='.py' else 'python_stub' if suffix=='.pyi' else 'torch_binary' if package=='torch' else 'extension_binary'
    require(text in inherited and p.parent==WINDOWS_SYSTEM_DIRECTORY and suffix=='.dll','only exact inherited Windows binary leaves');return 'windows_system_binary'

def _windows_library_identities(runtime,*,budget=None):
    """Optional additive identity map; no directory wildcard or native permit.

    Current bytes are still verified by the complete qualifier's descriptor loop.
    The inherited maps remain independently joined to their external anchors.
    """
    windows=runtime.get('windows_library_identities',{})
    require(type(windows) is dict and len(windows)<=4096,'finite selected Windows DLL identity roster')
    require(not(set(windows)&(set(runtime['inherited65_map'])|set(runtime['stdlib779_map']))),'Windows identity additions never replace inherited65/stdlib779')
    if not windows:return windows
    boundary=runtime['library_search_policy'].get('enforced_boundary')
    require(type(boundary) is dict and type(boundary.get('library_files')) is dict,'Windows identities require qualified enforced library map')
    libraries=boundary['library_files']
    for text,entry in windows.items():
        require(type(text) is str and 0<len(text)<=4096 and not any(c in text for c in '*?'),'exact Windows DLL path without wildcard')
        path=Path(text)
        require(path.is_absolute() and str(path)==text and '..' not in path.parts and path.parent==WINDOWS_SYSTEM_DIRECTORY and path.suffix.lower()=='.dll','only exact absolute System32 DLL leaves')
        require(type(entry) is dict and set(entry)=={'bytes','sha256','role'} and type(entry['bytes']) is int and 0<entry['bytes']<=8*1024**3 and sha(entry['sha256']) and entry['role']=='windows_system_binary','typed Windows DLL descriptor')
        require(text in runtime['files'] and runtime['files'][text]==entry and text in libraries and libraries[text]==entry,'selected Windows DLL joins runtime and qualified library maps')
        physical=qualified_path(path);meta=physical.lstat()
        require(stat.S_ISREG(meta.st_mode) and meta.st_size==entry['bytes'],'selected Windows DLL exact regular physical size')
        if budget is not None:budget.check()
    return windows
def library_boundary_plan(runtime):
    """Externally selected observation topology; never reconstructed from output."""
    if 'library_boundary_plan' not in runtime:return None
    plan=runtime['library_boundary_plan'];boundary=runtime['library_search_policy'].get('enforced_boundary')
    require(type(boundary) is dict and type(boundary.get('extension_transitions')) is dict,'selected boundary plan requires enforced library policy')
    require(type(plan) is list and 0<len(plan)<=2048,'finite nonempty externally selected library boundary plan')
    ids=set();parents={};names={};framework_count=0;open_chain=[]
    for row in plan:
        require(type(row) is dict and set(row)=={'id','parent_id','name'} and type(row['id']) is int and 0<row['id']<2**64 and row['id'] not in ids,'unique typed selected boundary identity')
        parent=row['parent_id'];require(parent is None or (type(parent) is int and parent in ids),'earlier typed selected boundary parent')
        name=row['name'];require(type(name) is str and 0<len(name)<=8192,'bounded selected boundary name')
        if name=='framework_import':
            framework_count+=1;require(parent is None,'framework boundary has no parent');open_chain=[]
        else:
            require(name.startswith('extension:'),'selected framework or extension boundary')
            selected,separator,operation=name[len('extension:'):].rpartition(':')
            require(separator and selected in boundary['extension_transitions'] and operation in {'create','exec'},'selected extension operation in policy')
            transition=boundary['extension_transitions'][selected]
            require(type(transition) is dict and set(transition)=={'create','exec'},'typed selected extension transitions')
            require(parent is not None and parent in open_chain,'selected extension has an open earlier parent')
            open_chain=open_chain[:open_chain.index(parent)+1]
        require(len(open_chain)<64,'bounded selected boundary nesting')
        open_chain.append(row['id'])
        ids.add(row['id']);parents[row['id']]=parent;names[row['id']]=name
    require(framework_count==1 and plan[0]['name']=='framework_import','one first framework boundary in selected plan')
    return parse(canonical(plan))

def physical_roster(root,suffixes,excluded,*,recursive=True,budget=None):
    import os
    if budget is not None:budget.check()
    root=qualified_path(root);require(root.is_dir(),'exact physical finite runtime root');require(type(suffixes) is list and suffixes and len(set(suffixes))==len(suffixes) and set(suffixes)<={'.py','.pyi','.dll','.pyd','.exe','.zip'} and excluded in (['__pycache__'],['site-packages','__pycache__']) and type(recursive) is bool,'finite explicit executable/cache roster policy');result={}
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in excluded)
        for name in dirs:
            p=Path(current)/name;meta=p.lstat();require(not p.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'runtime roster no aliases')
        for name in files:
            p=Path(current)/name
            if p.suffix.lower() not in suffixes:continue
            meta=p.lstat();require(stat.S_ISREG(meta.st_mode) and not p.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400) and len(result)<32768,'finite regular runtime roster');result[str(p)]={'bytes':meta.st_size}
            if budget is not None:budget.check()
        if not recursive:dirs[:]=[]
    return result
def package_source_roster():
    import os
    result=set()
    for current,dirs,files in os.walk(PACKAGE,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in ('artifacts','evidence','__pycache__'))
        for name in files:
            if name.endswith('.py'):result.add(str(Path(current)/name))
    result|={str(CORE/name) for name in CORE_PINS}
    result|={str(PACKAGE/name) for name in ('PILOT_DECISIONS.json','pilot_trainable_inventory.json','NATIVE_QUALIFICATION_PROPOSAL.json','AGENTS.md')}
    result|={str(PACKAGE.parents[1]/name) for name in ('AGENTS.md','RINEYA_SYNTH_CENTRALIZED_REASONING.md')}
    result.add(str(PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py'))
    return result
def roster_policy(name):
    base=Path(QUALIFIED_LAYOUT['physical_base_prefix'])
    policies={'stdlib_python_source':(base/'Lib',['.py'],['site-packages','__pycache__'],True),'stdlib_extension_runtime':(base/'DLLs',['.pyd','.dll'],['__pycache__'],True),'interpreter_root_executable_runtime':(base,['.exe','.dll','.zip'],['__pycache__'],False),'torch':(SITE/'torch',['.py','.pyi','.pyd','.dll'],['__pycache__'],True)}
    if name.startswith('external_package:'):
        package=name.split(':',1)[1];require(package in EXTERNAL_PACKAGES and package!='typing_extensions','exact finite declared external package root');return (SITE/package,['.py','.pyi','.pyd','.dll'],['__pycache__'],True)
    require(name in policies,'exact finite qualified runtime roster name');return policies[name]

def _git_exception_policy(exceptions,sources,files):
    require(type(exceptions) is dict and len(exceptions)<=4096,'finite registration-owned exact Git output leaves')
    for text,entry in exceptions.items():
        require(type(text) is str and type(entry) is dict and entry.get('role') in ('run_output','evidence_output','evidence_source_snapshot') and set(entry)==({'role','source_path'} if entry['role']=='evidence_source_snapshot' else {'role'}),'typed registration-owned exact Git exception')
        path=Path(text);require(path.is_absolute() and '..' not in path.parts and str(path)==text and text not in sources and text not in files,'Git exceptions never original source/config leaves')
        relative=path.relative_to(PACKAGE);parts=relative.parts
        if entry['role']=='evidence_source_snapshot':
            source=entry['source_path'];require(source in sources and len(parts)==4 and parts[0]=='evidence' and parts[1].startswith('prefix_input_owned_') and parts[2]=='source_snapshots' and parts[3]==hashlib.sha256(source.encode('utf-8')).hexdigest()+Path(source).suffix and Path(source).suffix in ('.py','.json','.md'),'exact source-key-derived preservation snapshot leaf')
        else:
            require(path.suffix.lower() in ('.json','.jsonl','.txt','.log','.snapshot'),'Git output exceptions never executable leaves')
            if entry['role']=='run_output':require(len(parts)>=3 and parts[0]=='artifacts' and (parts[1]=='_native_gpu_leases' or parts[1].startswith('prefix_input_owned_run_')),'exact run/lease leaf under owned output root')
            else:require(len(parts)>=3 and parts[0]=='evidence' and parts[1].startswith('prefix_input_owned_'),'exact owned preservation evidence leaf')

def _bootstrap_policy(value,budget=None):
    runtime=value['runtime'];files={**runtime['files'],**value['config']['source_files']};cache=runtime['cache_policy'];layout=runtime['stdlib_layout'];require(layout==QUALIFIED_LAYOUT,'exact separately qualified interpreter alias/layout')
    pins=cache['preloaded_module_cache_pins'];require(type(pins) is dict and len(pins)<=4096 and all(p in files and files[p]==d and Path(p).suffix=='.pyc' and Path(p).parent.name=='__pycache__' for p,d in pins.items()),'exact explicit preloaded bytecode cache leaves')
    modules=cache['preloaded_bootstrap_modules'];require(type(modules) is dict and 0<len(modules)<=4096,'explicit finite external-qualified fresh bootstrap modules')
    used=set()
    for name,entry in modules.items():
        require(type(name) is str and name and name.split('.',1)[0]!='torch' and type(entry) is dict and set(entry)=={'path','source_sha256','cache_path','cache_sha256'},'exact fresh non-Torch bootstrap module assumption')
        p=entry['path'];require(p in files and Path(p).suffix in ('.py','.pyd') and entry['source_sha256']==files[p]['sha256'] and not Path(p).is_relative_to(SITE/'torch'),'bootstrap module exact qualified source/extension bytes')
        cp=entry['cache_path'];require((cp is None and entry['cache_sha256'] is None) or (Path(p).suffix=='.py' and cp in pins and entry['cache_sha256']==pins[cp]['sha256'] and Path(cp)==Path(p).parent/'__pycache__'/(Path(p).stem+'.cpython-311.pyc')),'bootstrap exact3.11 cache/source assumptions');used.add(cp) if cp is not None else None
        if budget is not None:budget.check()
    require(used==set(pins),'no arbitrary unrostered cache payloads')
    namespaces=cache['namespace_roots'];require(type(namespaces) is dict and len(namespaces)<=4096,'finite exact qualified namespace roots')
    for name,roots in namespaces.items():
        require(type(name) is str and name and type(roots) is list and roots and len(roots)==len(set(roots)) and all(type(p) is str and str(Path(p))==p and any(Path(q).parent==Path(p) for q in files) for p in roots),'exact namespace roots with registered immediate source children')
    anchor=cache['fresh_bootstrap_evidence'];require(type(anchor) is dict and set(anchor)=={'path','file_sha256','body_sha256'},'external selected fresh-bootstrap assumptions anchor')
    review=sealed_json(anchor['path'],anchor['file_sha256'],anchor['body_sha256'],budget=budget)
    require(set(review)=={'kind','source_commit','preloaded_bootstrap_modules_sha256','fresh_process_bootstrap_qualified','environment_and_loader_assumptions_qualified','current_module_execution_proven','sha256'} and review['kind']=='ROOT_QUALIFIED_FRESH_BOOTSTRAP_BYTE_SOURCE_ASSUMPTIONS' and review['source_commit']==value['expected_head'] and review['preloaded_bootstrap_modules_sha256']==digest(modules) and review['fresh_process_bootstrap_qualified'] is True and review['environment_and_loader_assumptions_qualified'] is True and review['current_module_execution_proven'] is False,'external bootstrap assumptions; equivalence is not current execution proof')

def qualification_rosters(value,*,budget=None):
    config=value['config'];runtime=value['runtime'];registration=value['registration']
    require(set(config['source_files'])==package_source_roster(),'exact complete current package/tests/transitive source roster')
    rosters=runtime['rosters'];require(type(rosters) is dict and 0<len(rosters)<=128,'finite qualified physical rosters');covered=set()
    for name,roster in rosters.items():
        require(type(name) is str and type(roster) is dict and set(roster)=={'root','suffixes','excluded_directory_names','recursive','files'},'exact runtime roster schema')
        root,suffixes,excluded,recursive=roster_policy(name);require((roster['root'],roster['suffixes'],roster['excluded_directory_names'],roster['recursive'])==(str(root),suffixes,excluded,recursive),'exact finite qualified runtime roots/roles/suffixes')
        observed=physical_roster(roster['root'],roster['suffixes'],roster['excluded_directory_names'],recursive=roster['recursive'],budget=budget);require(observed==roster['files'],'exact complete physical runtime roster')
        require(set(observed)<=set(runtime['files']) and all(runtime['files'][p]['bytes']==v['bytes'] for p,v in observed.items()),'every physical source/binary roster file in qualifier map');covered.update(observed)
    single=str(SITE/'typing_extensions.py');external=set(runtime['external_dependency_files']);require(type(runtime['external_dependency_files']) is dict and all(runtime['external_dependency_files'][p]==runtime['files'][p] for p in external),'exact declared external dependency descriptors');require(external<=covered|{single} and (single not in external or runtime['files'][single]['role']=='python_source'),'all declared external dependencies in exact finite physical rosters/singleton')
    known=set(runtime['inherited65_map'])|set(runtime['stdlib779_map']);windows=_windows_library_identities(runtime,budget=budget)
    require(set(runtime['files'])==covered|known|external|set(windows)|set(runtime['cache_policy']['preloaded_module_cache_pins']),'no unrostered extra runtime payloads or omitted qualified files')
    _bootstrap_policy(value,budget)
    review=sealed_json(registration['qualification_review_path'],registration['qualification_review_file_sha256'],registration['qualification_review_body_sha256'],budget=budget)
    review_fields={'kind','source_commit','source_inventory_sha256','runtime_closure_sha256','rosters_sha256','cache_policy_sha256','library_search_policy_sha256','complete_current_byte_review','sha256'}
    require(set(review)==(review_fields|{'library_boundary_plan_sha256'} if 'library_boundary_plan' in runtime else review_fields),'exact independent qualified review schema')
    if 'library_boundary_plan' in runtime:require(review['library_boundary_plan_sha256']==digest(library_boundary_plan(runtime)),'independent selected library boundary plan review join')
    require(review['kind']=='ROOT_REVIEWED_TENSOR_RUNTIME_ROSTER_LIBRARY_SOURCE_CACHE_CLOSURE' and review['source_commit']==value['expected_head'] and review['complete_current_byte_review'] is True,'externally selected current native qualification review')
    require(review['source_inventory_sha256']==digest(config['source_files']) and review['runtime_closure_sha256']==digest(runtime['files']) and review['rosters_sha256']==digest(rosters) and review['cache_policy_sha256']==digest(runtime['cache_policy']) and review['library_search_policy_sha256']==digest(runtime['library_search_policy']),'qualification review exact maps/rules joins')
    search=runtime['library_search_policy'];require(type(search) is dict and set(search) in ({'mode','roster_names','unregistered_search_allowed'},{'mode','roster_names','unregistered_search_allowed','enforced_boundary'}) and search['mode']=='qualified_complete_registered_DLL_search_rosters' and search['unregistered_search_allowed'] is False and search['roster_names'] and set(search['roster_names'])<=set(rosters),'qualified exact loader-search-roster policy')
    if 'enforced_boundary' in search:
        boundary=search['enforced_boundary'];require(type(boundary) is dict and set(boundary)=={'kind','search_directories','system_directory','library_files','initial_loaded_libraries','extension_transitions','framework_transition','provider_source'} and boundary['kind'] in {'FINITE_WINDOWS_DLL_BOUNDARY_V1','FINITE_WINDOWS_DLL_BOUNDARY_V2'},'exact selected enforced library boundary')
        provider=boundary['provider_source'];path=str(PACKAGE/'prefix_library_policy.py')
        require(type(provider) is dict and set(provider)=={'path','descriptor'} and provider['path']==path and path in config['source_files'] and provider['descriptor']==config['source_files'][path],'enforced library provider exact source join')
        libraries=boundary['library_files'];require(type(libraries) is dict and libraries and len(libraries)<=4096 and all(p in runtime['files'] and d==runtime['files'][p] and Path(p).suffix.lower() in ('.pyd','.dll','.exe') for p,d in libraries.items()),'selected library identities exact qualified runtime subset')
def current_head(expected,exceptions=None,*,budget=None):
    import subprocess
    repository=PACKAGE.parents[1]
    timeout=600 if budget is None else budget.remaining()
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repository,capture_output=True,text=True,check=True,timeout=timeout).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=repository,capture_output=True,check=True,timeout=600 if budget is None else budget.remaining()).stdout
    require(head==expected,'current exact committed source HEAD');_git_status(dirty,{} if exceptions is None else exceptions,repository)
    if budget is not None:budget.check(len(dirty))

def _git_status(raw,exceptions,repository):
    require(type(raw) is bytes and len(raw)<=MAX_JSON,'bounded exact Git status bytes')
    for row in raw.split(b'\0'):
        if not row:continue
        require(row.startswith(b'?? '),'tracked source/config changes never exempted')
        path=Path(repository)/row[3:].decode('utf-8','strict');require(str(path) in exceptions,'unregistered Git leaf; no directory mask')
def _native_manifest_value(config_path,*,expected_config_file,expected_config_body,registration_path,expected_registration_file,expected_registration_body,runtime_path,expected_runtime_file,expected_runtime_body,expected_head):
    budget=QualificationBudget()
    config=sealed_json(config_path,expected_config_file,expected_config_body,budget=budget);registration=sealed_json(registration_path,expected_registration_file,expected_registration_body,budget=budget);runtime=sealed_json(runtime_path,expected_runtime_file,expected_runtime_body,budget=budget)
    value=native_metadata(config,registration,runtime,expected_head,budget=budget);current_head(expected_head,registration['git_output_exceptions'],budget=budget);load_input(budget=budget);qualification_rosters(value,budget=budget)
    config_path=Path(config_path).absolute();require(config_path.parent==PACKAGE/'configs' and config_path.resolve()==config_path,'owned direct configs file')
    import subprocess
    committed=subprocess.run(['git','show',expected_head+':'+config_path.relative_to(PACKAGE.parents[1]).as_posix()],cwd=PACKAGE.parents[1],capture_output=True,check=True,timeout=budget.remaining()).stdout
    budget.check(len(committed));require(committed==read_bytes(config_path,budget=budget),'exact selected config bytes committed at source HEAD')
    for mapping in (config['source_files'],runtime['files']):
        for path,entry in mapping.items():
            budget.check();file_descriptor(path,entry,layout=runtime['stdlib_layout'],budget=budget)
    require(str(Path(config_path).absolute())==registration['config_path'],'exact external config path')
    value['config_path']=str(config_path);value['anchors']={'expected_config_file':expected_config_file,'expected_config_body':expected_config_body,'registration_path':str(registration_path),'expected_registration_file':expected_registration_file,'expected_registration_body':expected_registration_body,'runtime_path':str(runtime_path),'expected_runtime_file':expected_runtime_file,'expected_runtime_body':expected_runtime_body,'expected_head':expected_head}
    budget.check();return value
def validate_native_manifest(config_path,**anchors):
    stage_native_gate();return _mint_native_context(_native_manifest_value(config_path,**anchors))
def _CPU_context_capability_factory():
    class CPUContextFixture:
        __slots__=()
    records=set()
    def mint():
        token=CPUContextFixture();records.add(token);return token
    def check(token):require(type(token) is CPUContextFixture and token in records,'private CPU native-context source fixture capability')
    return mint,check
_mint_CPU_native_context_fixture,_check_CPU_native_context_fixture=_CPU_context_capability_factory()
def _CPU_native_context_fixture(config_path,*,_capability,**anchors):
    _check_CPU_native_context_fixture(_capability)
    value=_native_manifest_value(config_path,**anchors)
    value.update(fixture_only=True,native_PASS=False,native_authority_issued=False,status='CPU_NATIVE_CONTEXT_SOURCE_FIXTURE_COMPLETE')
    return _mint_native_context(value)
def _native_context_factory():
    records={}
    class NativeContext:
        __slots__=()
    def mint(value):
        token=NativeContext();records[token]=canonical(value);return token
    def get(token):
        require(type(token) is NativeContext and token in records,'actual integrated full-qualification factory context');return parse(records[token])
    return mint,get
_mint_native_context,context_value=_native_context_factory()
def revalidate_context(context):
    stage_native_gate();original=context_value(context);require(original.get('fixture_only',False) is False,'CPU native-context fixture has no native consumer authority');current=validate_native_manifest(original['config_path'],**original['anchors']);require(context_value(current)==original,'immutable original native source/runtime/config/HEAD identity');return original
def native_entry(*args,**kwargs):
    stage_native_gate();return validate_native_manifest(*args,**kwargs)
if __name__=='__main__':native_entry()
