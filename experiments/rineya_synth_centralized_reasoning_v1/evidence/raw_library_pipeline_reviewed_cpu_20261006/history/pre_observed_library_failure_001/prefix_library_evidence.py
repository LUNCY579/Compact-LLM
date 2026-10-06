"""Offline CPU validation of a required complete DLL evidence packet.

This prospective packet is not emitted by the current LibraryGuard summaries.
Integration must retain every checkpoint below, including complete cumulative
event batches, raw snapshots/scopes and directory identities. This module never
opens a library, observes a process, imports native APIs or grants authority.
Selected external byte maps are recipe identities, not proof of current bytes.
"""
from pathlib import Path
import hashlib,json,math

MAX_PACKET_BYTES=16*1024**2
MAX_OBSERVATIONS=4096
MAX_LIBRARIES=4096
MAX_EVENTS=16*MAX_LIBRARIES
MAX_DEPTH=64
ORIGINAL_SCOPE='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'

def require(value,message):
    if not value:raise ValueError(message)

def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def same(left,right):return canonical(left)==canonical(right)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def integer(value):return type(value) is int and 0<=value<2**64
def pairs(items):
    result={}
    for name,value in items:require(name not in result,'duplicate JSON field');result[name]=value
    return result
def number(text):
    value=float(text);require(math.isfinite(value),'nonfinite/overflow number');return value
def int_number(text):
    require(len(text)<=21,'bounded integer');value=int(text);require(-2**63<=value<2**64,'bounded integer');return value
def constant(text):raise ValueError('nonfinite JSON constant')
def parse(raw):
    require(type(raw) is bytes and 0<len(raw)<=MAX_PACKET_BYTES,'bounded actual packet bytes')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_int=int_number,parse_constant=constant)
    except (RecursionError,UnicodeError) as error:raise ValueError('bounded valid JSON nesting/encoding') from error

def descriptor(value,*,source=False):
    require(type(value) is dict and set(value)=={'bytes','sha256','role'} and integer(value['bytes']) and 0<value['bytes']<=2**31 and sha(value['sha256']),'typed selected byte descriptor')
    require(value['role']=='python_source' if source else value['role'] in {'extension_binary','native_library','interpreter_executable','torch_binary','interpreter_binary','windows_system_binary'},'selected descriptor role')

def path(text,suffixes):
    require(type(text) is str and 0<len(text)<=4096 and not any(c in text for c in '*?'),'bounded exact selected path')
    p=Path(text);require(p.is_absolute() and str(p)==text and '..' not in p.parts and p.suffix.lower() in suffixes,'canonical absolute selected path')
    return p

def selection(policy,identity,files,source_files,plan):
    require(type(policy) is dict and set(policy)=={'kind','search_directories','system_directory','library_files','initial_loaded_libraries','extension_transitions','framework_transition','provider_source'} and policy['kind'] in {'FINITE_WINDOWS_DLL_BOUNDARY_V1','FINITE_WINDOWS_DLL_BOUNDARY_V2'},'exact externally selected policy')
    require(type(files) is dict and 0<len(files)<=MAX_LIBRARIES and same(files,policy['library_files']),'exact external selected library byte map')
    directories=policy['search_directories'];require(type(directories) is list and 0<len(directories)<=128 and len(set(directories))==len(directories) and policy['system_directory'] in directories,'finite selected directory map')
    for text in directories:
        require(type(text) is str and 0<len(text)<=4096 and Path(text).is_absolute() and str(Path(text))==text and '..' not in Path(text).parts and not any(c in text for c in '*?'),'exact selected directory')
    for text,d in files.items():descriptor(d);require(str(path(text,{'.dll','.pyd','.exe'}).parent) in directories,'library exact selected parent')
    source=policy['provider_source'];require(type(source) is dict and set(source)=={'path','descriptor'} and source['path'] in source_files and same(source_files[source['path']],source['descriptor']),'external provider source byte join');path(source['path'],{'.py'});descriptor(source['descriptor'],source=True)
    require(type(identity) is dict and set(identity)=={'source_commit','original_scope','owned_birth','context_sha256','producer_source','fixture_only','native_PASS','native_authority_issued'},'exact externally selected original identity')
    require(type(identity['source_commit']) is str and len(identity['source_commit'])==40 and all(c in '0123456789abcdef' for c in identity['source_commit']) and identity['original_scope']==ORIGINAL_SCOPE and sha(identity['context_sha256']),'registered source/scope/context recipe identities')
    birth=identity['owned_birth'];require(type(birth) is dict and set(birth)=={'pid','creation_filetime_100ns'} and all(integer(x) and x>0 for x in birth.values()),'exact selected PID/FILETIME birth')
    require(identity['fixture_only'] is True and identity['native_PASS'] is identity['native_authority_issued'] is False,'incoming CPU identity cannot grant native authority')
    producer=identity['producer_source'];require(type(producer) is dict and set(producer)=={'path','descriptor'} and producer['path'] in source_files and same(source_files[producer['path']],producer['descriptor']),'external producer source byte join');path(producer['path'],{'.py'});descriptor(producer['descriptor'],source=True)
    def paths(rows):
        require(type(rows) is list and len(rows)<=MAX_LIBRARIES and len(set(rows))==len(rows) and all(type(p) is str and p in files for p in rows),'finite selected transition identities')
    paths(policy['initial_loaded_libraries']);require(policy['initial_loaded_libraries'],'nonempty initial module population');paths(policy['framework_transition'])
    transitions=policy['extension_transitions'];require(type(transitions) is dict and len(transitions)<=MAX_LIBRARIES,'bounded extension transitions')
    for text,row in transitions.items():
        require(text in files and Path(text).suffix.lower()=='.pyd' and type(row) is dict and set(row)=={'create','exec'},'exact selected extension plan');paths(row['create']);paths(row['exec'])
    require(type(plan) is list and len(plan)<=MAX_OBSERVATIONS//2,'externally selected finite ordered boundary plan')
    ids=set()
    for row in plan:
        require(type(row) is dict and set(row)=={'id','parent_id','name'} and integer(row['id']) and row['id']>0 and row['id'] not in ids and (row['parent_id'] is None or (integer(row['parent_id']) and row['parent_id'] in ids)),'selected ordered boundary topology');ids.add(row['id'])
        expected_new(policy,row['name'])

def expected_new(policy,name):
    if name=='framework_import':return set(policy['framework_transition'])
    require(type(name) is str and len(name)<=8192 and name.startswith('extension:'),'known source boundary name')
    selected,sep,operation=name[len('extension:'):].rpartition(':')
    require(sep and selected in policy['extension_transitions'] and operation in {'create','exec'},'exact selected extension operation')
    return set(policy['extension_transitions'][selected][operation])

def scope(row,policy):
    common={'directories':policy['search_directories'],'system_directory':policy['system_directory'],'default_flags':0xC00,'legacy_directory':'','closed':False}
    if policy['kind']=='FINITE_WINDOWS_DLL_BOUNDARY_V1':
        expected={**common,'kind':'OBSERVED_DLL_DIRECTORY_SCOPE','complete':True,'unmanaged_directory_registration':False,'completeness_basis':'explicit_injected_CPU_observation'}
    else:
        expected={**common,'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE','managed_setup_complete':True,'foreign_directory_registration':'UNOBSERVED','evidence_basis':'explicit_injected_CPU_managed_observation'}
    require(type(row) is dict and same(row,expected),'exact scope observation; V2 foreign directories remain UNOBSERVED')

def cleanup(row,policy):
    common={'remaining_managed_directories':[],'legacy_directory':'','closed':True}
    if policy['kind']=='FINITE_WINDOWS_DLL_BOUNDARY_V1':expected={**common,'kind':'OBSERVED_DLL_DIRECTORY_CLEANUP','complete':True,'all_cookies_removed':True}
    else:expected={**common,'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_CLEANUP','managed_cleanup_complete':True,'all_managed_cookies_removed':True,'foreign_directory_registration':'UNOBSERVED'}
    require(type(row) is dict and same(row,expected),'exact successful terminal cleanup; no foreign coverage claim')

def snapshot(value,files):
    require(type(value) is dict and set(value)=={'kind','complete','libraries'} and value['kind']=='OBSERVED_PROCESS_LIBRARY_SNAPSHOT' and value['complete'] is True and type(value['libraries']) is list and 0<len(value['libraries'])<=MAX_LIBRARIES,'complete raw module snapshot')
    state={};handles=set()
    for row in value['libraries']:
        require(type(row) is dict and set(row)=={'path','module_handle'} and type(row['path']) is str and row['path'] in files and integer(row['module_handle']) and row['module_handle']>0 and row['path'] not in state and row['module_handle'] not in handles,'unique registered snapshot path/handle')
        state[row['path']]=row['module_handle'];handles.add(row['module_handle'])
    return state

def validate_packet(raw,*,policy,expected_identity,selected_file_map,selected_source_map,expected_boundaries,expected_file_sha256):
    """Actual strict raw read + independent replay; always CPU classification.

    Source/runtime maps and packet-file SHA are external selections. They cannot
    certify current bytes, lossless OS observation or native producer integration.
    """
    require(sha(expected_file_sha256) and type(raw) is bytes and len(raw)<=MAX_PACKET_BYTES and hashlib.sha256(raw).hexdigest()==expected_file_sha256,'externally selected packet file bytes')
    value=parse(raw);selection(policy,expected_identity,selected_file_map,selected_source_map,expected_boundaries)
    require(type(value) is dict and set(value)=={'schema_version','kind','identity','policy_sha256','library_files','observations','producer_claims','fixture_only','native_PASS','native_authority_issued','sha256'} and type(value['schema_version']) is int and value['schema_version']==1 and value['kind']=='CPU_LIBRARY_BOUNDARY_RAW_PACKET_V1','exact prospective raw packet schema')
    require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'raw packet body checksum')
    incoming=value['identity'];require(type(incoming) is dict and incoming.get('fixture_only') is True and incoming.get('native_PASS') is incoming.get('native_authority_issued') is False,'incoming identity retains typed CPU flags')
    require(same(incoming,expected_identity) and value['policy_sha256']==digest(policy) and same(value['library_files'],selected_file_map),'exact external original identity/policy/byte-map joins')
    require(value['fixture_only'] is True and value['native_PASS'] is value['native_authority_issued'] is False,'raw CPU evidence cannot issue native authority')
    observations=value['observations'];require(type(observations) is list and 3<=len(observations)<=MAX_OBSERVATIONS,'bounded complete checkpoint population')
    state=None;events=[];stack=[];entered=[];returned=[];directory_identity=None;last_clock=None;first_clock=None;cleaned=False;precleanup=False
    for index,row in enumerate(observations):
        require(type(row) is dict and set(row)=={'ordinal','time_ns','event','boundary_id','scope','snapshot','event_trace','directory_identities','cleanup'} and type(row['ordinal']) is int and row['ordinal']==index+1 and integer(row['time_ns']),'typed ordered raw checkpoint')
        require(row['boundary_id'] is None or (integer(row['boundary_id']) and row['boundary_id']>0),'typed positive boundary identity')
        require(last_clock is None or row['time_ns']>=last_clock,'chronological checkpoint clock');last_clock=row['time_ns'];first_clock=last_clock if first_clock is None else first_clock;require(last_clock-first_clock<=600*10**9,'bounded recorded scope duration')
        name=row['event'];require(name in {'initial','enter','exit','precleanup','cleanup'},'known raw checkpoint event')
        observed=snapshot(row['snapshot'],selected_file_map)
        dirs=row['directory_identities'];require(type(dirs) is dict and set(dirs)==set(policy['search_directories']) and all(type(d) is dict and set(d)=={'device','inode'} and integer(d['device']) and integer(d['inode']) and d['inode']>0 for d in dirs.values()),'complete typed selected directory identities')
        if directory_identity is None:directory_identity=dirs
        require(dirs==directory_identity,'selected directory physical identity conservation')
        require(not cleaned,'no observations after final cleanup')
        if name=='initial':
            require(index==0 and row['boundary_id'] is None and set(observed)==set(policy['initial_loaded_libraries']),'exact initial snapshot population');state=dict(observed);allowed=set()
        elif name=='enter':
            require(not precleanup and len(entered)<len(expected_boundaries) and len(stack)<MAX_DEPTH,'bounded source boundary entry')
            plan=expected_boundaries[len(entered)];require(row['boundary_id']==plan['id'] and plan['parent_id']==(stack[-1]['id'] if stack else None),'exact source boundary order/parent')
            allowed=stack[-1]['expected'] if stack and stack[-1]['name']=='framework_import' else set()
            frame={'id':plan['id'],'name':plan['name'],'before':dict(observed),'expected':expected_new(policy,plan['name']),'nested':set()}
        elif name=='exit':
            require(stack and row['boundary_id']==stack[-1]['id'] and not precleanup,'balanced exact source boundary return');frame=stack[-1];allowed=frame['expected']
        elif name=='precleanup':
            require(not stack and not precleanup and entered==[r['id'] for r in expected_boundaries] and row['boundary_id'] is None and index==len(observations)-2,'all registered boundaries precede terminal observation');allowed=set();precleanup=True
        else:
            require(precleanup and not stack and row['boundary_id'] is None and index==len(observations)-1 and row['scope'] is None,'exact final cleanup checkpoint');allowed=set();cleanup(row['cleanup'],policy);cleaned=True
        if name!='cleanup':scope(row['scope'],policy);require(row['cleanup'] is None,'cleanup only at final checkpoint')
        trace=row['event_trace'];require(type(trace) is dict and set(trace)=={'kind','complete','events'} and trace['kind']=='OBSERVED_PROCESS_LIBRARY_EVENTS' and trace['complete'] is True and type(trace['events']) is list and len(trace['events'])<=MAX_EVENTS,'complete bounded raw event batch')
        rows=trace['events'];require(rows[:len(events)]==events,'append-only event trace; no dropped/reordered/rewritten prefix')
        for sequence,event in enumerate(rows):
            require(type(event) is dict and set(event)=={'sequence','event','path','module_handle'} and type(event['sequence']) is int and event['sequence']==sequence+1 and event['event'] in {'load','unload'} and type(event['path']) is str and event['path'] in selected_file_map and integer(event['module_handle']) and event['module_handle']>0,'ordered registered raw library event')
            if sequence<len(events):continue
            require(event['event']=='load','unload/transient lifetime violates selected conservation')
            require(event['path'] in allowed and event['path'] not in state and event['module_handle'] not in state.values(),'exact prospective load path/handle')
            state[event['path']]=event['module_handle']
        events=rows;require(state==observed,'independently replayed events equal complete snapshot')
        if name=='enter':entered.append(frame['id']);stack.append(frame)
        elif name=='exit':
            require(all(observed.get(p)==h for p,h in frame['before'].items()),'boundary existing identity conservation')
            new=set(observed)-set(frame['before']);require(new-frame['nested']==frame['expected'] and not(frame['expected']&set(frame['before'])),'exact own/nested source transition population')
            stack.pop();returned.append(frame['id'])
            if stack:stack[-1]['nested'].update(new)
    require(cleaned and not stack and len(returned)==len(expected_boundaries),'complete independently replayed terminal population')
    claims=value['producer_claims'];require(type(claims) is dict and set(claims)=={'completion_claimed','observation_count','event_count','boundary_count','complete_search_directory_coverage'} and claims['completion_claimed'] is True and all(type(claims[k]) is int for k in ('observation_count','event_count','boundary_count')) and type(claims['complete_search_directory_coverage']) is bool,'typed incoming producer claims')
    recomputed={'observation_count':len(observations),'event_count':len(events),'boundary_count':len(returned)}
    require(all(claims[k]==v for k,v in recomputed.items()),'producer counts differ from independently recomputed conservation')
    require(policy['kind']!='FINITE_WINDOWS_DLL_BOUNDARY_V2' or claims['complete_search_directory_coverage'] is False,'V2 foreign scope is UNOBSERVED, never complete')
    return {'kind':'CPU_OFFLINE_LIBRARY_EVIDENCE_REPLAY','status':'CPU_LIBRARY_CONSERVATION_VERIFIED','fixture_only':True,'native_PASS':False,'native_authority_issued':False,'native_producer_integration_verified':False,'current_runtime_bytes_qualified':False,'lossless_native_observer_qualified':False,'actual_search_directory_coverage_qualified':False,'incoming_producer_claims':claims,'independently_recomputed':recomputed,'original_identity':expected_identity,'policy_sha256':digest(policy),'final_libraries':state}
