"""Authorized parent psutil metadata observation; no worker/model qualification."""
from pathlib import Path
from types import ModuleType
import hashlib,importlib,json,os,shutil,sys,time
P=Path(__file__).resolve().parents[2];D=P/'artifacts/import_approved_dispatch008';START=time.monotonic();GIB=1024**3
EXPECTED={'C:\\Users\\clclc\\BusinessProject\\ChatBubble-rineya-synth-centralized-reasoning-v1\\experiments\\rineya_synth_centralized_reasoning_v1\\artifacts\\import_approved_dispatch008\\IMPORT_OBSERVATION_CONFIG_008.json': 'faf7d393a18e443dd7d55a252c8a0d45a13671608e563bbf851ecf73e6a164f1', 'C:\\Users\\clclc\\BusinessProject\\ChatBubble-rineya-synth-centralized-reasoning-v1\\experiments\\rineya_synth_centralized_reasoning_v1\\artifacts\\import_approved_dispatch008\\IMPORT_OBSERVATION_RUNTIME_008.json': '217678ea66fe5af7b4073ad66253216ffe89c8c55c6889262eafe0c3ee96c6b4'}
def require(v,m):
    if not v:raise ValueError(m)
def read_packet(name):
    p=D/name;raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==EXPECTED[str(p)],'exact approved008 packet bytes');v=json.loads(raw);body={k:x for k,x in v.items() if k!='sha256'};require(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==v['sha256'],'sealed approved008 packet body');return v
config=read_packet('IMPORT_OBSERVATION_CONFIG_008.json');runtime=read_packet('IMPORT_OBSERVATION_RUNTIME_008.json');value={'config':config,'runtime':runtime}
def source(name):
    p=P/name;raw=p.read_bytes();d=config['source_files'][str(p)];require(d=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'},'exact selected observation source '+name);m=ModuleType('selected_parent_observation_'+p.stem);m.__file__=str(p);exec(compile(raw,str(p),'exec'),m.__dict__);return m
C=source('import_observation_contract.py');S=source('import_observation_supervisor.py');B=source('import_bootstrap.py')
path_container=sys.path;before=tuple(path_container);meta_container=sys.meta_path;meta_before=tuple(meta_container);finder=None
record={'kind':'ACTUAL_SELECTED_PARENT_PSUTIL_METADATA_OBSERVATION','scope':'AUTHORIZED_PARENT_DEPENDENCY_METADATA_ONLY','fixture_only':False,'native_PASS':False,'native_authority_issued':False,'Torch_CUDA_context_tensor_model_worker_or_optimizer_called':False,'host_GPU_admission_claimed':False,'selected_packets':EXPECTED,'selected_source_pins':{n:config['source_files'][str(P/n)] for n in ('import_observation_contract.py','import_observation_supervisor.py','import_bootstrap.py','native_process.py')},'module_rows':{},'dependency_resolutions':[],'resources':[],'error':None}
rows=runtime['cache_policy']['additional_preload_sources'];descriptors={row['path']:row['descriptor'] for row in rows.values()};descriptors.update({row['cache_path']:row['cache_descriptor'] for row in rows.values() if row['cache_path'] is not None});descriptor_context={'layout':runtime['stdlib_layout']}
def pins():
    for path,d in descriptors.items():C.file_descriptor(path,d,**descriptor_context)
def resource(native,pre):
    available=native.system_memory_status()['available_bytes'];process=native.Sampler().process(os.getpid());disk=shutil.disk_usage(P).free;require(available>=(3.3125 if pre else 1.25)*GIB and process['rss_bytes']<=2*GIB and disk>=4*GIB,'unchanged RAM/RSS/disk resource floors');row={'available_bytes':available,'process':process,'disk_free_bytes':disk,'pre_import':pre,'monotonic':time.monotonic()};record['resources'].append(row);require(time.monotonic()-START<=60,'bounded60s parent observation');return row
try:
    require(list(before)==runtime['cache_policy']['interpreter_profile']['path'],'selected no-site original parent roots');require(all(n not in sys.modules for n in ('psutil','native_process','native_nvml','pwd','org')),'fresh exact parent observation dependencies');pins()
    roots=[str(C.qualified_path(P)),str(C.qualified_path(C.SITE))];require(all(Path(p).is_dir() for p in roots),'physical selected PACKAGE/SITE roots');sys.path.extend(roots);after=tuple(sys.path);finder=S._SupervisorFinder(value,C)
    selected_find=finder.find_spec
    def trace_resolution(fullname,path=None,target=None):
        require(len(record['dependency_resolutions'])<256,'finite parent import trace');item={'name':fullname,'parent_paths':None if path is None else list(path)};record['dependency_resolutions'].append(item)
        try:
            spec=selected_find(fullname,path,target);item['origin']=None if spec is None else spec.origin;return spec
        except BaseException as error:item['error']={'type':type(error).__name__,'message':str(error)};raise
    finder.find_spec=trace_resolution;sys.meta_path.insert(0,finder)
    native=importlib.import_module('native_process');resource(native,True)
    # Replay the same selected parent setup that returned before actual008 failed.
    original=source('native_supervisor.py');monitor=S._selected_monitor(value,C,original)
    record['original_monitor_shared_module_same']=monitor.shared is original
    importlib.import_module('psutil')
    for name,row in rows.items():
        module=sys.modules.get(name)
        if module is None:record['module_rows'][name]={'present':False};continue
        data=vars(module);spec=data.get('__spec__');loader=data.get('__loader__');item={'present':True,'module_type':type(module).__name__,'module_name':data.get('__name__'),'module_file':data.get('__file__'),'spec_name':None if spec is None else spec.name,'spec_origin':None if spec is None else spec.origin,'loader_type':type(loader).__name__,'loader_name':getattr(loader,'name',None),'loader_path':getattr(loader,'path',None),'cached_present':'__cached__' in data,'cache':data.get('__cached__'),'selected_row':row}
        try:item['verifier']=B.verify_current_module(name,module,row,C,{**runtime['files'],**config['source_files']},runtime['stdlib_layout']);item['verification_passed']=True
        except Exception as error:item['verification_passed']=False;item['verification_error']={'type':type(error).__name__,'message':str(error)}
        record['module_rows'][name]=item
    resource(native,False);pins()
except BaseException as error:record['error']={'type':type(error).__name__,'message':str(error),'last_dependency_resolution':None if not record['dependency_resolutions'] else record['dependency_resolutions'][-1]}
finally:
    stable=sys.path is path_container and (finder is None or tuple(sys.path)==after) and sys.meta_path is meta_container and (finder is None or tuple(sys.meta_path)==(finder,*meta_before));sys.path=path_container;path_container[:]=before;sys.meta_path=meta_container;meta_container[:]=meta_before;record['original_root_and_meta_containers_restored']=sys.path is path_container and tuple(sys.path)==before and sys.meta_path is meta_container and tuple(sys.meta_path)==meta_before;record['owned_search_state_unchanged_before_restore']=stable;record['elapsed_seconds']=time.monotonic()-START;record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest();target=Path(sys.argv[1]).absolute();require(target.parent==Path(__file__).resolve().parent and not target.exists(),'owned new diagnostic receipt');require(stable and record['original_root_and_meta_containers_restored'] and record['elapsed_seconds']<=60,'bounded restored parent observation');target.write_bytes((json.dumps(record,sort_keys=True,indent=2)+'\n').encode())
raise SystemExit(0 if record['error'] is None else 2)
