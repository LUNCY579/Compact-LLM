"""Closed Torch-import observation registration construction; no context mint."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,math
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
PERMISSION={'torch_import':True,'CUDA_driver_interfaces_may_be_reached':True,'user_CUDA_context':False,
    'tensor_allocations':0,'model':False,'forward':False,'backward':False,'optimizer':False,
    'decoder':False,'scoring':False,'M0':False,'protected_bank_access':False,'paid_services':False}
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def seal(v):return {**v,'sha256':digest(v)}
def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def commit(v):return type(v) is str and len(v)==40 and all(c in '0123456789abcdef' for c in v)
def sealed(v):require(type(v) is dict and v.get('sha256')==digest({k:x for k,x in v.items() if k!='sha256'}),'exact selected document body')
def descriptor(d):require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=8*1024**3 and sha(d['sha256']),'exact bounded source/runtime descriptor')
def build(*,runtime_candidate,roster_candidate,observation,initial35_expectation,source_files,main_source,resources,implementation_commit,dispatch_head,packet_paths,metadata_references):
    """Join selected immutable candidates; this never grants run permission."""
    for value in (runtime_candidate,roster_candidate,observation):sealed(value)
    require((implementation_commit is dispatch_head is None) or
        (commit(implementation_commit) and commit(dispatch_head) and implementation_commit!=dispatch_head), 'pending or distinct actual implementation/dispatch selections')
    files=runtime_candidate['files'];require(len(files)==8716 and len(roster_candidate['rosters'])==18,'exact finite current passive byte/roster candidates')
    require(type(source_files) is dict and 0<len(source_files)<=4096 and main_source in source_files,'selected program source graph and separate main')
    for text,d in {**files,**source_files}.items():
        descriptor(d);p=Path(text);require(p.is_absolute() and '..' not in p.parts and str(p)==text,'literal selected path identity')
    require(all(p in files and files[p]==d for p,d in roster_candidate['files'].items()),'retained original expanded tensor runtime subset')
    rosters=roster_candidate['rosters'];covered=set().union(*(set(r['files']) for r in rosters.values()));require(covered<=set(files) and all(files[p]['bytes']==r['files'][p]['bytes'] for r in rosters.values() for p in r['files']),'exact current physical roster byte joins')
    maps=observation['maps'];pins=maps['preloaded_module_cache_pins'];require(len(pins)==59 and all(p in files and files[p]==d for p,d in pins.items()),'exact observed59 cache pins')
    require(set(maps['preloaded_class_aliases'])=={'typing.io','typing.re'} and '__main__' not in maps['preloaded_bootstrap_modules'],'finite observed aliases and separate future main')
    require(type(resources) is dict and resources['whole_scope_seconds']==600 and resources['GPU_free_at_admission_GiB']==4.0 and resources['GPU_free_during_work_GiB']==.5 and resources['worker_process_tree_RSS_GiB']==2.0,'unchanged registered resource limits')
    require(type(packet_paths) is dict and set(packet_paths)=={'config','runtime','registration'} and all(type(v) is str and Path(v).is_absolute() and Path(v).suffix=='.json' for v in packet_paths.values()) and len(set(packet_paths.values()))==3,'three distinct exact packet paths')
    require(type(metadata_references) is dict and metadata_references,'externally selected source/runtime references')
    require(type(initial35_expectation) is list and len(initial35_expectation)==len(set(initial35_expectation))==35 and all(p in files for p in initial35_expectation),'exact selected observer009 initial35 expectation')
    runtime=seal({'kind':'CLOSED_TORCH_IMPORT_OBSERVATION_RUNTIME_CANDIDATE','source_commit':implementation_commit,'files':deepcopy(files),'rosters':deepcopy(rosters),'cache_policy':{'mode':'SOURCE_COMPILE_WITH_OBSERVED009_FINITE_CACHE_ASSUMPTIONS','preloaded_module_cache_pins':deepcopy(pins),'preloaded_bootstrap_modules':deepcopy(maps['preloaded_bootstrap_modules']),'preloaded_class_aliases':deepcopy(maps['preloaded_class_aliases']),'interpreter_profile':deepcopy(observation['interpreter_profile']),'separate_main_source':{'path':main_source,'descriptor':deepcopy(source_files[main_source])},'source_execution_history_proven':False},'initial35_expectation':deepcopy(initial35_expectation),'initial35_provenance':'observer009 expectation; future program must measure and reject mismatch','observed_library_allowed_files':{p:d for p,d in files.items() if Path(p).suffix.lower() in ('.dll','.pyd','.exe')},'exact_effects_or_order_prequalified':False,'native_runtime_qualified':False,'metadata_references':deepcopy(metadata_references)})
    config=seal({'schema_version':1,'kind':'CLOSED_ONE_OWNED_TORCH_IMPORT_OBSERVATION','scope':SCOPE,'source_commit':implementation_commit,'permission':deepcopy(PERMISSION),'resources':deepcopy(resources),'source_files':deepcopy(source_files),'runtime_files':deepcopy(files),'main_source':main_source,'native_launch_authorized':False,'native_authority_issued':False,'owner_approval':None,'model':False,'whole_program_entry_observed':False})
    registration=seal({'kind':'CLOSED_TORCH_IMPORT_OBSERVATION_REGISTRATION','scope':SCOPE,'source_commit':implementation_commit,'config_path':packet_paths['config'],'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256'],'source_files':deepcopy(source_files),'runtime_files':deepcopy(files),'profile_sha256':digest({'scope':SCOPE,'permission':PERMISSION,'resources':resources}),'git_output_exceptions':{},'native_launch_authorized':False,'owner_approval':None})
    return {'config':config,'runtime':runtime,'registration':registration,'external_dispatch_head':dispatch_head,'packet_paths':deepcopy(packet_paths),'native_launch_authorized':False,
        'selected_inputs':{'runtime_map_sha256':digest(files),'source_map_sha256':digest(source_files),'rosters_sha256':digest(rosters),
            'cache_policy_sha256':digest(runtime['cache_policy']),'initial35_sha256':digest(initial35_expectation)}}
def anchors(bundle,packet_bytes):
    require(commit(bundle['external_dispatch_head']), 'native dispatch anchors unavailable while implementation/dispatch pending')
    require(type(packet_bytes) is dict and set(packet_bytes)=={'config','runtime','registration'},'exact saved packet byte population')
    for name,raw in packet_bytes.items():require(type(raw) is bytes and raw==(canonical(bundle[name])+'\n').encode(),'actual saved packet bytes equal constructed bodies')
    paths=bundle['packet_paths'];return {'expected_config_file':hashlib.sha256(packet_bytes['config']).hexdigest(),'expected_config_body':bundle['config']['sha256'],'registration_path':paths['registration'],'expected_registration_file':hashlib.sha256(packet_bytes['registration']).hexdigest(),'expected_registration_body':bundle['registration']['sha256'],'runtime_path':paths['runtime'],'expected_runtime_file':hashlib.sha256(packet_bytes['runtime']).hexdigest(),'expected_runtime_body':bundle['runtime']['sha256'],'expected_head':bundle['external_dispatch_head']}
def validate(bundle,*,expected_selection=None):
    for name in ('config','runtime','registration'):sealed(bundle[name])
    c,r,g=bundle['config'],bundle['runtime'],bundle['registration'];require(c['scope']==g['scope']==SCOPE and c['permission']==PERMISSION and c['resources']==r.get('resources',c['resources']),'exact separate import-only scope')
    require(c['native_launch_authorized'] is g['native_launch_authorized'] is bundle['native_launch_authorized'] is False and c['owner_approval'] is g['owner_approval'] is None,'closed owner/grant fields')
    require(c['runtime_files']==g['runtime_files']==r['files'] and c['source_files']==g['source_files'] and g['config_body_sha256']==c['sha256'] and g['runtime_body_sha256']==r['sha256'],'exact same packet source/runtime/body joins')
    require(c['source_commit']==r['source_commit']==g['source_commit'] and
        ((c['source_commit'] is bundle['external_dispatch_head'] is None) or
         (commit(c['source_commit']) and commit(bundle['external_dispatch_head']) and c['source_commit']!=bundle['external_dispatch_head'])), 'pending or distinct implementation/dispatch separation')
    require(r['cache_policy']['separate_main_source']=={'path':c['main_source'],'descriptor':c['source_files'][c['main_source']]} and r['native_runtime_qualified'] is False and r['exact_effects_or_order_prequalified'] is False,'separate own main and unobserved effects')
    expected=bundle['selected_inputs'] if expected_selection is None else expected_selection
    require(expected=={'runtime_map_sha256':digest(r['files']),'source_map_sha256':digest(c['source_files']),
        'rosters_sha256':digest(r['rosters']),'cache_policy_sha256':digest(r['cache_policy']),
        'initial35_sha256':digest(r['initial35_expectation'])},'external selected map/profile digests cannot rebind')
    return {'kind':'CPU_CLOSED_IMPORT_OBSERVATION_PACKET_VALIDATED','native_launch_authorized':False,'native_authority_issued':False,'scope':SCOPE}
def native_context(*args,**kwargs):raise ValueError('closed registration has no native context factory')
