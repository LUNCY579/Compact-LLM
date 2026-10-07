"""Separate observer-only scope, exact external anchors and opaque context.

No tensor/model scope is reused. Public execution is location- and approval-gated.
CPU source fixtures never mint a native-authorized context.
"""
from pathlib import Path
from types import ModuleType
import hashlib,json,math,stat,subprocess,time
HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'observer_startup_measurement_v1'
SCOPE='NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD'
PERMISSION={'observer_dll':True,'fixture_dll':True,'register_notifications':True,'load_fixture':True,'unload_fixture':True,'model':False,'torch_import':False,'CUDA_context':False,'tensor_allocation':False,'allocations':0,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'protected_bank_access':False,'M0_membership':False,'paid_services':False}
RESOURCES={'CPU_sample_max_age_seconds':.2,'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'GPU_probe_interval_seconds':.5,'GPU_probe_timeout_seconds':2.0,'GPU_sample_max_age_seconds':2.0,'admission_poll_seconds':2.0,'admission_seconds':30,'disk_free_GiB':4.0,'external_watchdog_milliseconds':100,'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'whole_scope_seconds':600,'worker_process_tree_RSS_GiB':2.0}
OPS=('source_preflight','owned_identity','observer_binding','observer_start','initial_module_baseline','fixture_load','loaded_module_snapshot','fixture_unload','unloaded_module_snapshot','observer_stop','final_event_drain','final_source_preflight')
MAX_JSON=16*1024**2
ABI={'version':1,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,'row_bytes':2600,'snapshot_bytes':72,'architecture':'x64'}
STARTUP_SITE=Path(r'C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages')
NO_SITE_MODE='MEASURE_BYTE_VERIFIED_STARTUP_NO_SITE_V3'
PRELUDE_MODE='MEASURE_FROZEN_TENSOR_PRELUDE_NO_SITE_V4'
def expected_no_site_profile(layout=None):
    logical=Path(r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none') if layout is None else Path(layout['logical_base_prefix'])
    return {'flags':{'isolated':1,'no_site':1,'ignore_environment':1,'dont_write_bytecode':1,'no_user_site':1},
      'executable':str(STARTUP_SITE.parents[1]/'Scripts/python.exe'),'prefix':str(logical),'exec_prefix':str(logical),'base_prefix':str(logical),'base_exec_prefix':str(logical),
      'path':[str(logical/'python311.zip'),str(logical/'DLLs'),str(logical/'Lib'),str(logical)],
      'meta_path':[{'module':'_frozen_importlib','name':'BuiltinImporter'},{'module':'_frozen_importlib','name':'FrozenImporter'},{'module':'_frozen_importlib_external','name':'PathFinder'}],
      'path_hooks':[{'module':'zipimport','name':'zipimporter'},{'module':'_frozen_importlib_external','name':'path_hook_for_FileFinder'}],
      'source_execution_history_proven':False}

STARTUP_MODULE_LEAVES={'__editable___hermes_agent_0_17_0_finder':'__editable___hermes_agent_0_17_0_finder.py','__editable___recursive_state_forgetting_0_1_0_finder':'__editable___recursive_state_forgetting_0_1_0_finder.py','__editable___rineya_tgsa_research_0_2_0_finder':'__editable___rineya_tgsa_research_0_2_0_finder.py','_distutils_hack':'_distutils_hack/__init__.py','_virtualenv':'_virtualenv.py','pywin32_bootstrap':'win32/lib/pywin32_bootstrap.py'}
STARTUP_HOOK_LEAVES=('__editable__.hermes_agent-0.17.0.pth','__editable__.recursive_state_forgetting-0.1.0.pth','__editable__.rineya_tgsa_research-0.2.0.pth','_virtualenv.pth','distutils-precedence.pth','pywin32.pth')
STARTUP_NAMESPACE_LEAVES=('pythoncom311.dll','pywintypes311.dll')

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
    started=time.monotonic();remaining=d['bytes']
    with p.open('rb') as f:
        while remaining:
            require(time.monotonic()-started<=600,'bounded in-file observer byte verification deadline')
            data=f.read(min(2*1024**2,remaining));require(data and len(data)<=remaining,'exact bounded observer read progress')
            h.update(data);remaining-=len(data)
        require(f.read(1)==b'','growing selected observer file refused')
    after=p.lstat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and h.hexdigest()==d['sha256'],'exact current observer-only file bytes')
def _selected_json(path,file_sha,body_sha):
    require(sha(file_sha) and sha(body_sha),'external file/body anchors');raw=read_bytes(path);require(hashlib.sha256(raw).hexdigest()==file_sha,'original external file bytes');value=parse(raw)
    require(type(value) is dict and value.get('sha256')==body_sha==digest({k:v for k,v in value.items() if k!='sha256'}),'original external JSON body');return value

def _schema(config,runtime,registration,head,*,fixture_only):
    fields={'kind','scope','permission','resources','implementation_commit','source_files','runtime_files','observer_registration','binary_fixture','worker_path','offline_environment','native_launch_authorized','sha256'}
    require(type(config) is dict and set(config)==fields and config['kind']=='REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1' and config['scope']==SCOPE,'exact observer-only config scope')
    require(canonical(config['permission'])==canonical(PERMISSION) and canonical(config['resources'])==canonical(RESOURCES),'unchanged observer-only permission/resource envelope')
    implementation=config['implementation_commit']
    require(type(head) is str and len(head)==40 and all(c in '0123456789abcdef' for c in head) and type(implementation) is str and len(implementation)==40 and all(c in '0123456789abcdef' for c in implementation) and implementation!=head and implementation==runtime['implementation_commit']==registration['implementation_commit'],'same implementation ancestor; distinct externally selected dispatch HEAD')
    require(type(runtime) is dict and set(runtime)=={'kind','implementation_commit','files','inherited65_map','stdlib779_map','windows_library_identities','library_event_observer','stdlib_layout','cache_policy','allowed_initial_loaded_libraries','qualification_review','sha256'} and runtime['kind']=='PROSPECTIVE_OBSERVER_STARTUP_BYTE_ADMISSION_V1','exact observer-only runtime schema')
    require(config['runtime_files']==runtime['files'] and config['observer_registration']==runtime['library_event_observer'],'same selected observer/runtime maps')
    require(type(registration) is dict and set(registration)=={'kind','implementation_commit','config_body_sha256','runtime_body_sha256','source_files','runtime_files','profile_sha256','worker_path','owner_approval','native_launch_authorized','sha256'} and registration['kind']=='ROOT_REGISTERED_OBSERVER_STARTUP_LAUNCH_V1','exact observer-only registration')
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
            suffix=Path(text).suffix.lower();allowed={'python_source':{'.py'},'startup_hook_source':{'.pth'},'source_json':{'.json'},'source_markdown':{'.md'},'native_observer_source':{'.c','.h'},'fixture_source':{'.c'},'observer_binary':{'.dll'},'observer_fixture_binary':{'.dll'},'interpreter_binary':{'.exe','.dll','.zip','.pyd'},'windows_system_binary':{'.dll'},'extension_binary':{'.dll','.pyd'},'python_stub':{'.pyi'},'qualified_bytecode_cache':{'.pyc'},'torch_binary':{'.dll','.pyd'}}
            require(d['role'] in allowed and suffix in allowed[d['role']],'observer-only source/runtime role suffix; no model assets')
            if d['role']=='observer_binary':require(text==observer['binary']['path'] and mapping is files,'observer binary only exact singleton')
            if d['role']=='observer_fixture_binary':require(text==fixture['path'] and mapping is files,'fixture binary only exact singleton')
    require(type(runtime['allowed_initial_loaded_libraries']) is list and runtime['allowed_initial_loaded_libraries'] and len(set(runtime['allowed_initial_loaded_libraries']))==len(runtime['allowed_initial_loaded_libraries']) and all(p in files and Path(p).suffix.lower() in {'.exe','.dll','.pyd'} for p in runtime['allowed_initial_loaded_libraries']),'finite allowed initial module population; no prior loaded-baseline claim')
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
    startup_roles=_startup_admitted_roles(config,runtime)
    for path,d in files.items():
        if path in (config['binary_fixture']['path'],observer['binary']['path']):continue
        role=startup_roles[path] if path in startup_roles else 'windows_system_binary' if path in windows else module.runtime_role(path,runtime['inherited65_map'])
        require(d['role']==role,'exact finite observer-only runtime role/root assignment')
    # The old observer registration validator needs the selected library map;
    # this is a pure adapter of metadata fields, not a tensor launch/context.
    observer=runtime['library_event_observer'];selected_libraries={p:d for p,d in files.items() if Path(p).suffix.lower() in {'.dll','.pyd','.exe'}}
    proxy={**runtime,'library_search_policy':{'enforced_boundary':{'library_files':selected_libraries,'search_directories':sorted({str(Path(p).parent) for p in selected_libraries})}}}
    module.library_event_observer_registration({**config,'source_commit':config['implementation_commit']},proxy)
    _bootstrap_admission_checks(config,runtime)


def _directory_roster(root,selected,*,suffix=None,layout=None):
    """Read exact finite directory identities without importing their contents."""
    root=qualified_path(root,layout);require(root.is_dir() and type(selected) is dict and 0<len(selected)<=64,'finite physical startup directory roster')
    expected={str(qualified_path(path,layout)):d for path,d in selected.items()}
    require(len(expected)==len(selected) and all(Path(path).parent==root for path in expected),'immediate exact startup file leaves')
    def capture():
        children=tuple(root.iterdir());require(len(children)<=32768,'bounded immediate startup directory')
        chosen={}
        for child in children:
            if suffix is not None and child.suffix.lower()!=suffix:continue
            meta=child.lstat();require(stat.S_ISREG(meta.st_mode) and not child.is_symlink() and not(getattr(meta,'st_file_attributes',0)&0x400),'startup roster regular physical files only')
            chosen[str(child)]=meta
        require(set(chosen)==set(expected),'exact complete startup directory roster')
        return {path:(s.st_size,s.st_mtime_ns,s.st_ino) for path,s in chosen.items()}
    before=capture()
    for path,d in expected.items():file_descriptor(path,d,layout=layout)
    require(capture()==before,'stable startup roster during byte verification')
    return expected

def _startup_admitted_roles(config,runtime):
    """Observer-only exact exceptions; shared tensor runtime roots stay closed."""
    cache=runtime['cache_policy']
    if cache.get('mode') in {'MEASURE_BYTE_VERIFIED_STARTUP_V1',NO_SITE_MODE,PRELUDE_MODE}:return {}
    require(set(cache)=={'mode','allowed_bootstrap_modules','imports','startup_hook_sources','allowed_namespace_packages'} and cache['mode']=='MEASURE_BYTE_VERIFIED_STARTUP_V2','exact prospective startup V2 fields')
    files=runtime['files'];modules=cache['allowed_bootstrap_modules'];roles={}
    for name,leaf in STARTUP_MODULE_LEAVES.items():
        path=str(STARTUP_SITE/leaf);entry=modules.get(name);d=files.get(path)
        require(type(d) is dict and d.get('role')=='python_source' and entry=={'path':path,'source_sha256':d.get('sha256')},'exact observer-only startup module singleton')
        roles[path]='python_source'
    hooks=cache['startup_hook_sources'];require(type(hooks) is dict and set(hooks)=={'directory','files'} and hooks['directory']==str(STARTUP_SITE) and type(hooks['files']) is dict and set(hooks['files'])=={str(STARTUP_SITE/leaf) for leaf in STARTUP_HOOK_LEAVES},'exact six startup hook source leaves')
    for path,d in hooks['files'].items():require(files.get(path)==d and d.get('role')=='startup_hook_source','startup hook selected bytes/role');roles[path]='startup_hook_source'
    namespaces=cache['allowed_namespace_packages'];require(type(namespaces) is dict and set(namespaces)=={'pywin32_system32'},'one exact prospective originless namespace')
    entry=namespaces['pywin32_system32'];root=STARTUP_SITE/'pywin32_system32'
    require(type(entry) is dict and set(entry)=={'search_locations','files','files_sha256'} and entry['search_locations']==[str(root)] and type(entry['files']) is dict and set(entry['files'])=={str(root/leaf) for leaf in STARTUP_NAMESPACE_LEAVES} and entry['files_sha256']==digest(entry['files']),'exact namespace path and two-file roster')
    require('pywin32_system32' not in modules,'namespace distinct from source module admission')
    for path,d in entry['files'].items():require(files.get(path)==d and d.get('role')=='extension_binary','namespace selected DLL byte/role joins');roles[path]='extension_binary'
    require(len(roles)==14 and not(set(roles)&(set(runtime['inherited65_map'])|set(runtime['stdlib779_map'])|set(config['source_files']))),'finite additive observer startup leaves only')
    return roles

def _bootstrap_admission_checks(config,runtime):
    files=runtime['files'];observer=runtime['library_event_observer']
    cache=runtime['cache_policy'];require(type(cache) is dict and cache.get('mode') in {'MEASURE_BYTE_VERIFIED_STARTUP_V1','MEASURE_BYTE_VERIFIED_STARTUP_V2',NO_SITE_MODE,PRELUDE_MODE} and cache['imports']==['ctypes','ctypes.wintypes','_ctypes'],'prospective byte-bound explicit startup import policy')
    if cache['mode']=='MEASURE_BYTE_VERIFIED_STARTUP_V1':require(set(cache)=={'mode','allowed_bootstrap_modules','imports'},'original startup V1 fields')
    elif cache['mode'] in (NO_SITE_MODE,PRELUDE_MODE):
        require(set(cache)==({'mode','allowed_bootstrap_modules','imports','interpreter_profile','tensor_prelude','allowed_cache_files'} if cache['mode']==PRELUDE_MODE else {'mode','allowed_bootstrap_modules','imports','interpreter_profile'}) and canonical(cache['interpreter_profile'])==canonical(expected_no_site_profile(runtime['stdlib_layout'])),'exact registered no-site interpreter profile')
        if cache['mode']==PRELUDE_MODE:
            entry=cache['tensor_prelude'];require(type(entry) is dict and set(entry)=={'path','file_sha256','body_sha256'},'selected frozen source prelude plan anchor')
            plan=_selected_json(entry['path'],entry['file_sha256'],entry['body_sha256'])
            caches=cache['allowed_cache_files'];require(type(caches) is dict and 0<len(caches)<=4096 and all(files.get(p)==d and d['role']=='qualified_bytecode_cache' and Path(p).suffix=='.pyc' and Path(p).parent.name=='__pycache__' for p,d in caches.items()),'finite exact selected cache bytes, no inferred execution')
            require(plan['kind']=='FROZEN_CLOSED_TENSOR_SOURCE_PRELUDE_PLAN' and plan['compile_order']==['prefix_input_owned_worker.py','registration_git_join.py','prefix_registered_loader.py'] and plan['native_authority_issued'] is plan['source_execution_history_proven'] is False,'exact closed tensor definition-only prelude')
            for row in plan['sources'].values():require(config['source_files'].get(row['path'])==row['descriptor'],'all frozen tensor source leaves selected in observer source inventory')
            for path,d in plan['helper_sources'].items():require(config['source_files'].get(path)==d,'selected frozen Git helper definition')
        require(not(set(STARTUP_MODULE_LEAVES)&set(cache['allowed_bootstrap_modules'])),'site startup modules excluded from no-site bootstrap')
    else:
        _startup_admitted_roles(config,runtime)
        hooks=cache['startup_hook_sources'];_directory_roster(hooks['directory'],hooks['files'],suffix='.pth',layout=runtime['stdlib_layout'])
        for entry in cache['allowed_namespace_packages'].values():_directory_roster(entry['search_locations'][0],entry['files'],layout=runtime['stdlib_layout'])
    modules=cache['allowed_bootstrap_modules'];require(type(modules) is dict and 0<len(modules)<=4096 and set(cache['imports'])<=set(modules) and all(name.split('.',1)[0]!='torch' for name in modules),'finite prospective non-Torch bootstrap module admission')
    for name,row in modules.items():
        require(type(name) is str and type(row) is dict and set(row)=={'path','source_sha256'} and row['path'] in {**config['source_files'],**files} and {**config['source_files'],**files}[row['path']]['sha256']==row['source_sha256'] and {**config['source_files'],**files}[row['path']]['role'] in {'python_source','extension_binary'} and Path(row['path']).suffix.lower() in {'.py','.pyd'},'selected observer-only bootstrap source/extension identity')
    review_anchor=runtime['qualification_review'];require(type(review_anchor) is dict and set(review_anchor)=={'path','file_sha256','body_sha256'},'external current byte review anchor')
    review=_selected_json(review_anchor['path'],review_anchor['file_sha256'],review_anchor['body_sha256'])
    require(set(review)=={'kind','implementation_commit','source_inventory_sha256','runtime_inventory_sha256','observer_registration_sha256','allowed_initial_loaded_libraries_sha256','bootstrap_policy_sha256','complete_current_byte_review','sha256'} and review['kind']=='ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1' and review['implementation_commit']==config['implementation_commit'] and review['source_inventory_sha256']==digest(config['source_files']) and review['runtime_inventory_sha256']==digest(files) and review['observer_registration_sha256']==digest(observer) and review['allowed_initial_loaded_libraries_sha256']==digest(runtime['allowed_initial_loaded_libraries']) and review['bootstrap_policy_sha256']==digest(cache) and review['complete_current_byte_review'] is True,'independent prospective byte/import/library admission review; no prior loaded/bootstrap qualification')

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

def _committed_dispatch_checks(value,config_path):
    """Separate committed implementation bytes from later registered JSON bytes."""
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain'],cwd=PACKAGE.parents[1],capture_output=True,text=True,check=True).stdout
    require(head==value['expected_head'] and not dirty,'current clean committed observer-only source HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',value['config']['implementation_commit'],head],cwd=PACKAGE.parents[1],check=True,capture_output=True)
    repository=PACKAGE.parents[1]
    for selected in (config_path,value['anchors']['runtime_path'],value['anchors']['registration_path']):
        committed=subprocess.run(['git','show',head+':'+Path(selected).relative_to(repository).as_posix()],cwd=repository,capture_output=True,check=True).stdout
        require(committed==read_bytes(selected),'exact external config/runtime/registration bytes committed at dispatch HEAD')
    for selected,d in value['config']['source_files'].items():
        committed=subprocess.run(['git','show',value['config']['implementation_commit']+':'+Path(selected).relative_to(repository).as_posix()],cwd=repository,capture_output=True,check=True).stdout
        require(len(committed)==d['bytes'] and hashlib.sha256(committed).hexdigest()==d['sha256'],'exact selected implementation-ancestor source bytes')

def validate_context(config_path,**anchors):
    native_gate();value=_read_context(config_path,**anchors)
    _committed_dispatch_checks(value,config_path)
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
