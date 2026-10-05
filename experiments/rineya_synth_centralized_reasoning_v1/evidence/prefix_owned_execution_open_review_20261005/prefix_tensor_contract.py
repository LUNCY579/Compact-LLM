"""New two-tensor purpose, immutable injected admission and stage-first gate."""
from pathlib import Path
import hashlib,json,math,stat
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
def read_bytes(path,maximum=MAX_JSON):
    path=Path(path).absolute();require(path.resolve()==path,'exact physical path')
    for part in (path,*path.parents):
        meta=part.lstat();require(not part.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'no reparse alias')
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=maximum,'bounded regular file')
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'stable bounded bytes')
    return raw
def load_input():
    raw=read_bytes(INPUT,16384);require(hashlib.sha256(raw).hexdigest()==INPUT_FILE,'fixed exposed prefix file');v=parse(raw)
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
def sealed_json(path,file_pin,body_pin):
    require(sha(file_pin) and sha(body_pin),'external SHA256 anchors');raw=read_bytes(path);require(hashlib.sha256(raw).hexdigest()==file_pin,'external file anchor');value=parse(raw)
    require(type(value) is dict and value.get('sha256')==body_pin==digest({k:v for k,v in value.items() if k!='sha256'}),'external body anchor');return value
def file_descriptor(path,expected):
    require(type(expected) is dict and set(expected)=={'bytes','sha256','role'} and type(expected['bytes']) is int and 0<=expected['bytes']<=8*1024**3 and sha(expected['sha256']),'typed registered source/runtime descriptor')
    path=Path(path).absolute();require(path.resolve()==path,'registered exact physical runtime path')
    for part in (path,*path.parents):
        metadata=part.lstat();require(not part.is_symlink() and not(getattr(metadata,'st_file_attributes',0)&0x400),'registered runtime has no alias')
    before=path.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'],'registered exact runtime size');hasher=hashlib.sha256();remaining=before.st_size
    with path.open('rb') as stream:
        while remaining:
            chunk=stream.read(min(1024**2,remaining));require(bool(chunk),'complete bounded runtime hash');hasher.update(chunk);remaining-=len(chunk)
        require(not stream.read(1),'registered runtime did not grow')
    after=path.lstat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and hasher.hexdigest()==expected['sha256'],'registered runtime exact stable bytes')
def native_metadata(config,registration,runtime,expected_head):
    # Pure schema/join validation is independently CPU-testable; never a permit.
    require(type(expected_head) is str and len(expected_head)==40 and all(c in '0123456789abcdef' for c in expected_head),'exact full source commit')
    require(set(config)=={'schema_version','kind','scope','permission','resources','input_file_sha256','input_body_sha256','prefix_ids_sha256','source_commit','native_launch_authorized','source_files','runtime_files','helper_pins','torch_version','authorization','offline_environment','tolerances','runtime_versions','GPU_physical_ordinal','sha256'},'exact native config fieldset')
    require(set(registration)=={'kind','source_commit','native_launch_authorized','config_body_sha256','runtime_body_sha256','source_files','runtime_files','config_path','qualification_review_path','qualification_review_file_sha256','qualification_review_body_sha256','profile_sha256','sha256'},'exact external registration fieldset')
    require(set(runtime)=={'kind','source_commit','current_binary_bytes_qualified','external_dependencies_complete','interpreter_layout_qualified','cache_execution_policy_qualified','files','inherited65_map','cache_policy','external_dependency_files','rosters','library_search_policy','stdlib_layout','sha256'},'exact runtime qualifier fieldset')
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
    original=sealed_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62')
    require(runtime['inherited65_map']==original['runtime_pins'] and all(p in files and files[p]['sha256']==h for p,h in original['runtime_pins'].items()),'original exact65 preserved inside full closure')
    require(config['helper_pins']==original['helper_pins'],'unchanged owned/resource helper identity')
    require(config['offline_environment']==original['offline_environment'] and config['tolerances']==original['tolerances'] and config['runtime_versions']==original['runtime_versions'] and config['torch_version']==original['runtime_versions']['torch'] and type(config['GPU_physical_ordinal']) is int and config['GPU_physical_ordinal']==0,'unchanged offline/numeric/version semantics and CUDA0/NVML0 binding')
    required={'prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_input_owned_supervisor.py','prefix_input_owned_auditor.py','prefix_registered_loader.py','prefix_native_pipeline.py'}
    require(required<={Path(p).name for p in sources} and all(str(PACKAGE/name) in sources for name in required),'all new exact direct-package source boundaries')
    require(all(str(PACKAGE/name) in sources and sources[str(PACKAGE/name)]['sha256']==pin for name,pin in original['helper_pins'].items()),'helper source map byte pins')
    cache=runtime['cache_policy'];require(cache['mode']=='all_registered_python_source_compiled_without_cache' and cache['allow_unregistered'] is False and cache['source_loader_sha256']==sources[str(PACKAGE/'prefix_registered_loader.py')]['sha256'],'qualified fail-closed source/cache loader')
    require(set(runtime['external_dependency_files'])<=set(files) and runtime['external_dependency_files'],'explicit nonempty external bootstrap dependency closure')
    for mapping in (files,sources):
        for path,entry in mapping.items():
            require(type(path) is str and 0<len(path)<=4096 and type(entry) is dict and set(entry)=={'bytes','sha256','role'} and type(entry['bytes']) is int and 0<=entry['bytes']<=8*1024**3 and sha(entry['sha256']),'typed exact closure map')
            require(entry['role'] in ('python_source','python_stub','source_json','source_markdown','qualified_bytecode_cache','extension_binary','torch_binary','interpreter_binary','windows_system_binary','installation_metadata') and Path(path).suffix.lower() not in ('.safetensors','.pt','.pth','.bin','.ckpt'),'only registered executable/source roles; no model payload')
    require(not(set(files)&set(sources)) or all(files[p]==sources[p] for p in set(files)&set(sources)),'conflict-free runtime/source overlap')
    return {'config':config,'registration':registration,'runtime':runtime,'expected_head':expected_head}
def physical_roster(root,suffixes,excluded):
    import os
    root=Path(root).absolute();require(root.is_dir() and root.resolve()==root,'exact physical finite runtime root');require(type(suffixes) is list and suffixes and set(suffixes)<={'.py','.pyi','.dll','.pyd','.exe','.zip'} and excluded==['__pycache__'],'finite explicit executable/cache roster policy');result={}
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in excluded)
        for name in dirs:
            p=Path(current)/name;meta=p.lstat();require(not p.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'runtime roster no aliases')
        for name in files:
            p=Path(current)/name
            if p.suffix.lower() not in suffixes:continue
            meta=p.lstat();require(stat.S_ISREG(meta.st_mode) and not p.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400) and len(result)<32768,'finite regular runtime roster');result[str(p)]={'bytes':meta.st_size}
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
def qualification_rosters(value):
    config=value['config'];runtime=value['runtime'];registration=value['registration']
    require(set(config['source_files'])==package_source_roster(),'exact complete current package/tests/transitive source roster')
    rosters=runtime['rosters'];require(type(rosters) is dict and 0<len(rosters)<=128,'finite qualified physical rosters');covered=set()
    for name,roster in rosters.items():
        require(type(name) is str and type(roster) is dict and set(roster)=={'root','suffixes','excluded_directory_names','files'},'exact runtime roster schema')
        observed=physical_roster(roster['root'],roster['suffixes'],roster['excluded_directory_names']);require(observed==roster['files'],'exact complete physical runtime roster')
        require(set(observed)<=set(runtime['files']) and all(runtime['files'][p]['bytes']==v['bytes'] for p,v in observed.items()),'every physical source/binary roster file in qualifier map');covered.update(observed)
    require(set(runtime['external_dependency_files'])<=covered,'all declared external dependencies in complete physical rosters')
    known=set(runtime['inherited65_map']);require(set(runtime['files'])<=covered|known|set(runtime['cache_policy'].get('preloaded_module_cache_pins',{})),'no unrostered extra runtime payloads')
    layout=runtime['stdlib_layout'];require(layout=={'logical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none','physical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none','file_attributes':1040,'reparse_tag':2684354563,'is_symlink':False,'python_version':'3.11.15'},'exact separately qualified interpreter alias/layout')
    review=sealed_json(registration['qualification_review_path'],registration['qualification_review_file_sha256'],registration['qualification_review_body_sha256'])
    require(set(review)=={'kind','source_commit','source_inventory_sha256','runtime_closure_sha256','rosters_sha256','cache_policy_sha256','library_search_policy_sha256','complete_current_byte_review','sha256'},'exact independent qualified review schema')
    require(review['kind']=='ROOT_REVIEWED_TENSOR_RUNTIME_ROSTER_LIBRARY_SOURCE_CACHE_CLOSURE' and review['source_commit']==value['expected_head'] and review['complete_current_byte_review'] is True,'externally selected current native qualification review')
    require(review['source_inventory_sha256']==digest(config['source_files']) and review['runtime_closure_sha256']==digest(runtime['files']) and review['rosters_sha256']==digest(rosters) and review['cache_policy_sha256']==digest(runtime['cache_policy']) and review['library_search_policy_sha256']==digest(runtime['library_search_policy']),'qualification review exact maps/rules joins')
    search=runtime['library_search_policy'];require(type(search) is dict and set(search)=={'mode','roster_names','unregistered_search_allowed'} and search['mode']=='qualified_complete_registered_DLL_search_rosters' and search['unregistered_search_allowed'] is False and search['roster_names'] and set(search['roster_names'])<=set(rosters),'qualified exact loader-search-roster policy')
def current_head(expected):
    import subprocess
    repository=PACKAGE.parents[1]
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repository,capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=repository,capture_output=True,text=True,check=True).stdout
    require(head==expected and not dirty,'current clean exact committed source HEAD')
def validate_native_manifest(config_path,*,expected_config_file,expected_config_body,registration_path,expected_registration_file,expected_registration_body,runtime_path,expected_runtime_file,expected_runtime_body,expected_head):
    stage_native_gate()
    config=sealed_json(config_path,expected_config_file,expected_config_body);registration=sealed_json(registration_path,expected_registration_file,expected_registration_body);runtime=sealed_json(runtime_path,expected_runtime_file,expected_runtime_body)
    value=native_metadata(config,registration,runtime,expected_head);current_head(expected_head);load_input();qualification_rosters(value)
    config_path=Path(config_path).absolute();require(config_path.parent==PACKAGE/'configs' and config_path.resolve()==config_path,'owned direct configs file')
    import subprocess,time
    committed=subprocess.run(['git','show',expected_head+':'+config_path.relative_to(PACKAGE.parents[1]).as_posix()],cwd=PACKAGE.parents[1],capture_output=True,check=True).stdout
    require(committed==read_bytes(config_path),'exact selected config bytes committed at source HEAD')
    started=time.monotonic();total=0
    for mapping in (config['source_files'],runtime['files']):
        for path,entry in mapping.items():
            total+=entry['bytes'];require(total<=16*1024**3 and time.monotonic()-started<=600,'bounded full runtime/source byte closure deadline/total');file_descriptor(path,entry)
    require(str(Path(config_path).absolute())==registration['config_path'],'exact external config path')
    value['config_path']=str(config_path);value['anchors']={'expected_config_file':expected_config_file,'expected_config_body':expected_config_body,'registration_path':str(registration_path),'expected_registration_file':expected_registration_file,'expected_registration_body':expected_registration_body,'runtime_path':str(runtime_path),'expected_runtime_file':expected_runtime_file,'expected_runtime_body':expected_runtime_body,'expected_head':expected_head}
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
    stage_native_gate();original=context_value(context);current=validate_native_manifest(original['config_path'],**original['anchors']);require(context_value(current)==original,'immutable original native source/runtime/config/HEAD identity');return original
def native_entry(*args,**kwargs):
    stage_native_gate();return validate_native_manifest(*args,**kwargs)
if __name__=='__main__':native_entry()
