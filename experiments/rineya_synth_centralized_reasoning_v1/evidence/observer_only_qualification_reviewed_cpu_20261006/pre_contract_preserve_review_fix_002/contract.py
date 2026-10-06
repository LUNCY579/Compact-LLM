"""Separate observer-only scope, exact external anchors and opaque context.

No tensor/model scope is reused. Public execution is location- and approval-gated.
CPU source fixtures never mint a native-authorized context.
"""
from pathlib import Path
from types import ModuleType
import hashlib,json,math,stat,subprocess,time
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'observer_only_qualification_v1'
SCOPE='NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD'
PERMISSION={'observer_dll':True,'fixture_dll':True,'register_notifications':True,'load_fixture':True,'unload_fixture':True,'model':False,'torch_import':False,'CUDA_context':False,'tensor_allocation':False,'allocations':0,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'protected_bank_access':False,'M0_membership':False,'paid_services':False}
RESOURCES={'CPU_sample_max_age_seconds':.2,'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'GPU_probe_interval_seconds':.5,'GPU_probe_timeout_seconds':2.0,'GPU_sample_max_age_seconds':2.0,'admission_poll_seconds':2.0,'admission_seconds':30,'disk_free_GiB':4.0,'external_watchdog_milliseconds':100,'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'whole_scope_seconds':600,'worker_process_tree_RSS_GiB':2.0}
OPS=('source_preflight','owned_identity','observer_binding','observer_start','initial_module_baseline','fixture_load','loaded_module_snapshot','fixture_unload','unloaded_module_snapshot','observer_stop','final_event_drain','final_source_preflight')
MAX_JSON=16*1024**2
ABI={'version':1,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,'row_bytes':2600,'snapshot_bytes':72,'architecture':'x64'}

def require(value,message):
    if not value:raise ValueError(message)

def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def pairs(items):
    out={}
    for k,v in items:require(k not in out,'duplicate JSON field');out[k]=v
    return out
def constant(value):raise ValueError('nonfinite JSON constant')
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'};return {**body,'sha256':digest(body)}
def native_gate():require(HERE==EXPECTED,'staged/preserved observer-only source has no native authority')
stage_native_gate=native_gate
def physical(path):
    p=Path(path).absolute();require(p.resolve()==p,'exact physical observer-only path')
    for part in (p,*p.parents):
        meta=part.lstat();require(not part.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'observer-only reparse/alias refused')
    return p

def read_bytes(path,maximum=MAX_JSON):
    p=physical(path);before=p.lstat();require(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=maximum,'bounded regular observer-only bytes')
    with p.open('rb') as f:raw=f.read(before.st_size+1)
    after=p.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'stable original selected bytes');return raw

def qualified_path(path,layout=None):
    p=Path(path).absolute()
    if layout is not None:
        logical=Path(layout['logical_base_prefix']);base=Path(layout['physical_base_prefix'])
        if p.is_relative_to(logical):
            require(logical.resolve()==base,'one exact qualified interpreter alias')
            resolved=base/p.relative_to(logical);require(p.resolve()==resolved,'exact known alias child');return physical(resolved)
    return physical(p)

def file_descriptor(path,d,*,layout=None):
    require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=8*1024**3 and sha(d['sha256']),'typed selected observer-only descriptor')
    p=qualified_path(path,layout);before=p.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==d['bytes'],'exact observer-only file size');h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(2*1024**2),b''):h.update(data)
    after=p.lstat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and h.hexdigest()==d['sha256'],'exact current observer-only file bytes')
def _selected_json(path,file_sha,body_sha):
    require(sha(file_sha) and sha(body_sha),'external file/body anchors');raw=read_bytes(path);require(hashlib.sha256(raw).hexdigest()==file_sha,'original external file bytes');value=parse(raw)
    require(type(value) is dict and value.get('sha256')==body_sha==digest({k:v for k,v in value.items() if k!='sha256'}),'original external JSON body');return value

def _schema(config,runtime,registration,head,*,fixture_only):
    fields={'kind','scope','permission','resources','source_commit','source_files','runtime_files','observer_registration','binary_fixture','worker_path','offline_environment','native_launch_authorized','sha256'}
    require(type(config) is dict and set(config)==fields and config['kind']=='REGISTERED_OBSERVER_ONLY_QUALIFICATION_V1' and config['scope']==SCOPE,'exact observer-only config scope')
    require(canonical(config['permission'])==canonical(PERMISSION) and canonical(config['resources'])==canonical(RESOURCES),'unchanged observer-only permission/resource envelope')
    require(type(head) is str and len(head)==40 and all(c in '0123456789abcdef' for c in head) and config['source_commit']==runtime['source_commit']==registration['source_commit']==head,'same observer-only registered source HEAD')
    require(type(runtime) is dict and set(runtime)=={'kind','source_commit','files','inherited65_map','stdlib779_map','windows_library_identities','library_event_observer','stdlib_layout','cache_policy','initial_loaded_libraries','qualification_review','sha256'} and runtime['kind']=='OBSERVER_ONLY_COMPLETE_CURRENT_RUNTIME_V1','exact observer-only runtime schema')
    require(config['runtime_files']==runtime['files'] and config['observer_registration']==runtime['library_event_observer'],'same selected observer/runtime maps')
    require(type(registration) is dict and set(registration)=={'kind','source_commit','config_body_sha256','runtime_body_sha256','source_files','runtime_files','profile_sha256','worker_path','owner_approval','native_launch_authorized','sha256'} and registration['kind']=='ROOT_REGISTERED_OBSERVER_ONLY_LAUNCH_V1','exact observer-only registration')
    require(registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'] and registration['source_files']==config['source_files'] and registration['runtime_files']==runtime['files'],'external observer-only registration joins')
    require(registration['profile_sha256']==digest({'scope':SCOPE,'permission':PERMISSION,'resources':RESOURCES,'ABI':ABI}) and registration['worker_path']==config['worker_path'],'fixed observer-only profile/worker joins')
    require(config['worker_path']==str(EXPECTED/'worker.py'),'exact distinct observer-only worker entry')
    required={str(EXPECTED/name) for name in ('contract.py','worker.py','supervisor.py','audit.py')}
    require(required<=set(config['source_files']),'complete concrete observer-only worker/supervisor/audit source roster')
    require(type(config['offline_environment']) is dict and config['offline_environment'],'explicit original offline environment')
    approval=registration['owner_approval'];require(type(approval) is dict and set(approval)=={'approved','scope','evidence_reference'} and type(approval['approved']) is bool and approval['scope']==SCOPE and type(approval['evidence_reference']) is str,'explicit separate owner approval record')
    require(type(config['native_launch_authorized']) is type(registration['native_launch_authorized']) is bool,'typed launch authority fields')
    if not fixture_only:require(approval['approved'] is config['native_launch_authorized'] is registration['native_launch_authorized'] is True and approval['evidence_reference'],'explicit observer-only launch approval required')
    else:require(approval['approved'] is config['native_launch_authorized'] is registration['native_launch_authorized'] is False,'CPU fixture cannot carry launch authority')
    sources=config['source_files'];files=runtime['files'];require(type(sources) is dict and 0<len(sources)<=4096 and type(files) is dict and 0<len(files)<=32768,'bounded observer-only source/runtime population')
    fixture=config['binary_fixture'];require(type(fixture) is dict and set(fixture)=={'path','descriptor'} and fixture['path']==str(PACKAGE/'observer_only_fixture.dll') and fixture['descriptor']['role']=='observer_fixture_binary' and files.get(fixture['path'])==fixture['descriptor'],'exact separate inert fixture binary identity')
    observer=runtime['library_event_observer'];require(type(observer) is dict and set(observer)=={'kind','adapter_source','native_source_files','binary','abi','build_receipt'} and observer['kind']=='ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1' and canonical(observer['abi'])==canonical(ABI),'exact selected observer ABI/build registration')
    require(observer['binary']['path']==str(PACKAGE/'native_library_observer.dll') and files.get(observer['binary']['path'])==observer['binary']['descriptor'] and observer['binary']['descriptor']['role']=='observer_binary','exact selected observer DLL')
    require(observer['adapter_source']['path']==str(PACKAGE/'prefix_library_observer.py') and sources.get(observer['adapter_source']['path'])==observer['adapter_source']['descriptor'],'exact source-bound observer adapter')
    for path,d in observer['native_source_files'].items():require(path in {str(PACKAGE/'prefix_library_observer.c'),str(PACKAGE/'prefix_library_observer.h')} and sources.get(path)==d and d['role']=='native_observer_source','exact native observer source input')
    require(set(observer['native_source_files'])=={str(PACKAGE/'prefix_library_observer.c'),str(PACKAGE/'prefix_library_observer.h')},'two complete native observer source leaves')
    for mapping in (sources,files):
        for text,d in mapping.items():
            require(type(text) is str and Path(text).is_absolute() and '..' not in Path(text).parts and type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=8*1024**3 and sha(d['sha256']),'typed bounded original byte map')
            suffix=Path(text).suffix.lower();allowed={'python_source':{'.py'},'source_json':{'.json'},'source_markdown':{'.md'},'native_observer_source':{'.c','.h'},'fixture_source':{'.c'},'observer_binary':{'.dll'},'observer_fixture_binary':{'.dll'},'interpreter_binary':{'.exe','.dll','.zip','.pyd'},'windows_system_binary':{'.dll'},'extension_binary':{'.dll','.pyd'},'python_stub':{'.pyi'},'qualified_bytecode_cache':{'.pyc'},'torch_binary':{'.dll','.pyd'}}
            require(d['role'] in allowed and suffix in allowed[d['role']],'observer-only source/runtime role suffix; no model assets')
            if d['role']=='observer_binary':require(text==observer['binary']['path'] and mapping is files,'observer binary only exact singleton')
            if d['role']=='observer_fixture_binary':require(text==fixture['path'] and mapping is files,'fixture binary only exact singleton')
    require(type(runtime['initial_loaded_libraries']) is list and runtime['initial_loaded_libraries'] and len(set(runtime['initial_loaded_libraries']))==len(runtime['initial_loaded_libraries']) and all(p in files and Path(p).suffix.lower() in {'.exe','.dll','.pyd'} for p in runtime['initial_loaded_libraries']),'complete selected initial module baseline')
    return {'config':config,'runtime':runtime,'registration':registration,'expected_head':head,'fixture_only':fixture_only,'native_PASS':False,'native_authority_issued':False}

def _shared_inventory_checks(value):
    """Reuse exact inventory/role readers, never the tensor-purpose context."""
    path=PACKAGE/'prefix_tensor_contract.py';entry=value['config']['source_files'].get(str(path))
    require(type(entry) is dict and entry.get('role')=='python_source','selected shared inventory validator source')
    file_descriptor(path,entry);raw=read_bytes(path);module=ModuleType('observer_only_selected_inventory');module.__file__=str(path)
    exec(compile(raw.decode('utf-8'),str(path),'exec'),module.__dict__)
    config=value['config'];runtime=value['runtime'];files=runtime['files']
    original=module.sealed_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62')
    require(runtime['inherited65_map']==original['runtime_pins'] and all(p in files and files[p]['sha256']==h for p,h in original['runtime_pins'].items()),'unchanged inherited65 observer-only byte map')
    stdlib=module.sealed_json(module.STDLIB_CANDIDATE,'e5e185f980e2f0feae7bf49384089eedd4d47f5b41c10a828c6ef2a30ae777a5','f6284b28e689b2c5e42a97c12256b3e35a41395a6b9612c0a01ee786eb28b149')
    require(runtime['stdlib779_map']==stdlib['inventory']['runtime_pins'] and all(p in files and files[p]['sha256']==h for p,h in runtime['stdlib779_map'].items()),'unchanged stdlib779 observer-only byte map')
    require(config['offline_environment']==original['offline_environment'],'unchanged original offline environment')
    observer=runtime['library_event_observer']
    windows=runtime['windows_library_identities'];require(type(windows) is dict and len(windows)<=4096 and not(set(windows)&(set(runtime['inherited65_map'])|set(runtime['stdlib779_map']))),'finite additive Windows identities preserve original maps')
    for path,d in windows.items():require(Path(path).parent==Path(r'C:\Windows\System32') and Path(path).suffix.lower()=='.dll' and files.get(path)==d and d['role']=='windows_system_binary','exact selected Windows dependency singleton')
    for path,d in files.items():
        if path in (config['binary_fixture']['path'],observer['binary']['path']):continue
        role='windows_system_binary' if path in windows else module.runtime_role(path,runtime['inherited65_map'])
        require(d['role']==role,'exact finite observer-only runtime role/root assignment')
    # The old observer registration validator needs the selected library map;
    # this is a pure adapter of metadata fields, not a tensor launch/context.
    observer=runtime['library_event_observer'];selected_libraries={p:d for p,d in files.items() if Path(p).suffix.lower() in {'.dll','.pyd','.exe'}}
    proxy={**runtime,'library_search_policy':{'enforced_boundary':{'library_files':selected_libraries,'search_directories':sorted({str(Path(p).parent) for p in selected_libraries})}}}
    module.library_event_observer_registration(config,proxy)
    cache=runtime['cache_policy'];require(type(cache) is dict and set(cache)=={'preloaded_bootstrap_modules','fresh_bootstrap_evidence'},'observer-only fresh bootstrap source policy')
    modules=cache['preloaded_bootstrap_modules'];require(type(modules) is dict and modules and all(name.split('.',1)[0]!='torch' for name in modules),'fresh non-Torch observer bootstrap modules')
    for name,row in modules.items():
        require(type(name) is str and type(row) is dict and set(row)=={'path','source_sha256'} and row['path'] in files and files[row['path']]['sha256']==row['source_sha256'] and Path(row['path']).suffix.lower() in {'.py','.pyd'},'selected observer-only bootstrap source/extension identity')
    anchor=cache['fresh_bootstrap_evidence'];require(type(anchor) is dict and set(anchor)=={'path','file_sha256','body_sha256'},'external observer-only bootstrap evidence anchor')
    bootstrap=_selected_json(anchor['path'],anchor['file_sha256'],anchor['body_sha256'])
    require(set(bootstrap)=={'kind','source_commit','preloaded_modules_sha256','fresh_process_bootstrap_qualified','environment_and_loader_assumptions_qualified','current_module_execution_proven','sha256'} and bootstrap['kind']=='ROOT_QUALIFIED_OBSERVER_ONLY_FRESH_BOOTSTRAP_V1' and bootstrap['source_commit']==value['expected_head'] and bootstrap['preloaded_modules_sha256']==digest(modules) and bootstrap['fresh_process_bootstrap_qualified'] is bootstrap['environment_and_loader_assumptions_qualified'] is True and bootstrap['current_module_execution_proven'] is False,'externally qualified observer bootstrap assumptions; namespace execution not inferred')
    review_anchor=runtime['qualification_review'];require(type(review_anchor) is dict and set(review_anchor)=={'path','file_sha256','body_sha256'},'external current byte review anchor')
    review=_selected_json(review_anchor['path'],review_anchor['file_sha256'],review_anchor['body_sha256'])
    require(set(review)=={'kind','source_commit','source_inventory_sha256','runtime_inventory_sha256','observer_registration_sha256','initial_loaded_libraries_sha256','bootstrap_sha256','complete_current_byte_review','sha256'} and review['kind']=='ROOT_REVIEWED_OBSERVER_ONLY_CURRENT_BYTE_CLOSURE_V1' and review['source_commit']==value['expected_head'] and review['source_inventory_sha256']==digest(config['source_files']) and review['runtime_inventory_sha256']==digest(files) and review['observer_registration_sha256']==digest(observer) and review['initial_loaded_libraries_sha256']==digest(runtime['initial_loaded_libraries']) and review['bootstrap_sha256']==bootstrap['sha256'] and review['complete_current_byte_review'] is True,'independent exact observer-only byte/bootstrap/module review')

def _factory():
    records={}
    class Context:__slots__=()
    def mint(value):obj=Context();records[obj]=canonical(value);return obj
    def get(context):require(type(context) is Context and context in records,'actual observer-only original factory context');return parse(records[context])
    return mint,get
_mint,context_value=_factory()
def _capability_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private observer-only CPU capability')
    return get,check
_CPU_capability,_check_CPU_capability=_capability_factory()

def _read_context(config_path,*,expected_config_file,expected_config_body,runtime_path,expected_runtime_file,expected_runtime_body,registration_path,expected_registration_file,expected_registration_body,expected_head,fixture_only=False):
    config=_selected_json(config_path,expected_config_file,expected_config_body);runtime=_selected_json(runtime_path,expected_runtime_file,expected_runtime_body);registration=_selected_json(registration_path,expected_registration_file,expected_registration_body)
    value=_schema(config,runtime,registration,expected_head,fixture_only=fixture_only)
    require(Path(config_path).parent==PACKAGE/'configs' and Path(config_path).absolute()==physical(config_path),'exact observer-only configs location')
    _shared_inventory_checks(value)
    started=time.monotonic();total=sum(d['bytes'] for mapping in (config['source_files'],runtime['files']) for d in mapping.values())
    require(total<=16*1024**3,'bounded full observer-only byte qualification population')
    for mapping in (config['source_files'],runtime['files']):
        for path,d in mapping.items():
            file_descriptor(path,d,layout=runtime['stdlib_layout']);require(time.monotonic()-started<=600,'bounded original observer-only byte qualification time')
    value['config_path']=str(Path(config_path).absolute());value['anchors']={'expected_config_file':expected_config_file,'expected_config_body':expected_config_body,'runtime_path':str(runtime_path),'expected_runtime_file':expected_runtime_file,'expected_runtime_body':expected_runtime_body,'registration_path':str(registration_path),'expected_registration_file':expected_registration_file,'expected_registration_body':expected_registration_body,'expected_head':expected_head}
    return value

def validate_context(config_path,**anchors):
    native_gate();value=_read_context(config_path,**anchors)
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout
    require(head==value['expected_head'] and not dirty,'current clean committed observer-only source HEAD')
    committed=subprocess.run(['git','show',head+':'+Path(config_path).relative_to(PACKAGE.parents[1]).as_posix()],cwd=PACKAGE.parents[1],capture_output=True,check=True).stdout
    require(committed==read_bytes(config_path),'observer-only config committed at registered HEAD')
    return _mint(value)
def revalidate_context(context):
    native_gate();value=context_value(context);require(value['fixture_only'] is False,'CPU observer-only context has no native authority')
    fresh=context_value(validate_context(value['config_path'],**value['anchors']));require(fresh==value,'original observer-only context cannot rebase');return fresh

def _CPU_context_fixture(config_path,*,_capability,**anchors):
    _check_CPU_capability(_capability);return _mint(_read_context(config_path,fixture_only=True,**anchors))
def _CPU_context_value_fixture(value,*,_capability):
    _check_CPU_capability(_capability);require(value.get('fixture_only') is True and value.get('native_PASS') is value.get('native_authority_issued') is False,'explicit CPU observer-only value');return _mint(value)
def _CPU_revalidate_context(context,*,_capability):
    _check_CPU_capability(_capability);value=context_value(context);require(value['fixture_only'] is True,'CPU original observer-only context');return value

def preflight(context):
    value=context_value(context);config=value['config']
    return {'config_sha256':value['anchors']['expected_config_file'],'config_path':value['config_path'],'worker_path':config['worker_path'],
      'authorization':{'scope':SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False},
      'resources':dict(RESOURCES),'source_commit':value['expected_head'],'profile_sha256':value['registration']['profile_sha256'],
      'source_inventory_sha256':digest(config['source_files']),'runtime_inventory_sha256':digest(value['runtime']['files'])}
