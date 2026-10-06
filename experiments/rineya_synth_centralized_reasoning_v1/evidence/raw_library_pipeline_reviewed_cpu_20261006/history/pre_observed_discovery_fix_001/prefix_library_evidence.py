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

def selection(policy,identity,files,source_files,plan,*,_observed=False):
    require(type(policy) is dict and set(policy)=={'kind','search_directories','system_directory','library_files','initial_loaded_libraries','extension_transitions','framework_transition','provider_source'} and policy['kind'] in {'FINITE_WINDOWS_DLL_BOUNDARY_V1','FINITE_WINDOWS_DLL_BOUNDARY_V2'},'exact externally selected policy')
    require(type(files) is dict and 0<len(files)<=MAX_LIBRARIES and same(files,policy['library_files']),'exact external selected library byte map')
    directories=policy['search_directories'];require(type(directories) is list and 0<len(directories)<=128 and len(set(directories))==len(directories) and policy['system_directory'] in directories,'finite selected directory map')
    for text in directories:
        require(type(text) is str and 0<len(text)<=4096 and Path(text).is_absolute() and str(Path(text))==text and '..' not in Path(text).parts and not any(c in text for c in '*?'),'exact selected directory')
    for text,d in files.items():descriptor(d);require(str(path(text,{'.dll','.pyd','.exe'}).parent) in directories,'library exact selected parent')
    source=policy['provider_source'];require(type(source) is dict and set(source)=={'path','descriptor'} and source['path'] in source_files and same(source_files[source['path']],source['descriptor']),'external provider source byte join');path(source['path'],{'.py'});descriptor(source['descriptor'],source=True)
    identity_fields={'source_commit','original_scope','owned_birth','context_sha256','producer_source','fixture_only','native_PASS','native_authority_issued'}
    require(type(identity) is dict and set(identity)==(identity_fields|{'owned_birth_status','owner_receipt_sha256'} if _observed else identity_fields),'exact externally selected original identity')
    require(type(identity['source_commit']) is str and len(identity['source_commit'])==40 and all(c in '0123456789abcdef' for c in identity['source_commit']) and identity['original_scope']==ORIGINAL_SCOPE and sha(identity['context_sha256']),'registered source/scope/context recipe identities')
    birth=identity['owned_birth'];require((_observed and birth is None and identity['owned_birth_status']=='UNAVAILABLE') or (type(birth) is dict and set(birth)=={'pid','creation_filetime_100ns'} and all(integer(x) and x>0 for x in birth.values())),'exact selected PID/FILETIME birth')
    require((type(identity['fixture_only']) is bool if _observed else identity['fixture_only'] is True) and identity['native_PASS'] is identity['native_authority_issued'] is False,'incoming CPU identity cannot grant native authority')
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

# The historical CPU-only V1 interface above remains separate and unchanged.
# Observed failure recording must never infer a complete record from summaries.
OBSERVED_PACKET_KIND='OBSERVED_LIBRARY_GUARD_RAW_PACKET_V2'
OBSERVATION_COMPONENTS=frozenset(('scope','snapshot','event_trace','directory_identities','cleanup'))

def observed_error(value):
    require(type(value) is dict and set(value)=={'type','message','message_exact'} and type(value['type']) is str and 0<len(value['type'])<=128 and type(value['message']) is str and len(value['message'])<=4096 and type(value['message_exact']) is bool,'bounded original observed error fidelity')
    return value['message_exact']

def observed_identity(value,*,expected_identity):
    fields={'source_commit','original_scope','owned_birth','context_sha256','producer_source','fixture_only','native_PASS','native_authority_issued','owned_birth_status','owner_receipt_sha256'}
    require(type(value) is dict and type(expected_identity) is dict and set(value)==set(expected_identity)==fields and same(value,expected_identity),'exact external observed source/context/owner identity')
    require(type(value['fixture_only']) is bool and value['native_PASS'] is value['native_authority_issued'] is False and sha(value['owner_receipt_sha256']),'observed provenance never grants native authority')
    require(value['owned_birth_status'] in {'VALIDATED','UNAVAILABLE'} and ((value['owned_birth_status']=='UNAVAILABLE' and value['owned_birth'] is None) or (value['owned_birth_status']=='VALIDATED' and type(value['owned_birth']) is dict and set(value['owned_birth'])=={'pid','creation_filetime_100ns'} and all(integer(x) and x>0 for x in value['owned_birth'].values()))),'explicit validated or unavailable owned birth; never fabricated')

def observed_scope(value,policy,*,fixture_only):
    require(type(fixture_only) is bool,'typed observed provenance')
    projected=dict(value) if type(value) is dict else value
    if type(projected) is dict:
        key='completeness_basis' if policy['kind']=='FINITE_WINDOWS_DLL_BOUNDARY_V1' else 'evidence_basis'
        expected=('explicit_injected_CPU_observation' if fixture_only else 'Win32_calls_and_verified_fresh_bootstrap_trace') if key=='completeness_basis' else ('explicit_injected_CPU_managed_observation' if fixture_only else 'Win32_managed_directory_operations')
        require(projected.get(key)==expected,'exact recorded scope provenance basis')
        projected[key]='explicit_injected_CPU_observation' if key=='completeness_basis' else 'explicit_injected_CPU_managed_observation'
    # This projection validates the common fields; it does not reclassify the
    # incoming provider provenance or supply native directory coverage proof.
    scope(projected,policy)

def _observed_directories(value,policy):
    require(type(value) is dict and set(value)==set(policy['search_directories']) and all(type(d) is dict and set(d)=={'device','inode'} and integer(d['device']) and integer(d['inode']) and d['inode']>0 for d in value.values()),'complete typed observed directory identities')
    return value

def _observed_events(trace,previous,state,allowed,files):
    require(type(trace) is dict and set(trace)=={'kind','complete','events'} and trace['kind']=='OBSERVED_PROCESS_LIBRARY_EVENTS' and trace['complete'] is True and type(trace['events']) is list and len(trace['events'])<=MAX_EVENTS,'complete bounded observed event batch')
    rows=trace['events'];require(same(rows[:len(previous)],previous),'append-only observed event conservation')
    updated=dict(state)
    for index,event in enumerate(rows):
        require(type(event) is dict and set(event)=={'sequence','event','path','module_handle'} and type(event['sequence']) is int and event['sequence']==index+1 and event['event'] in {'load','unload'} and type(event['path']) is str and event['path'] in files and integer(event['module_handle']) and event['module_handle']>0,'typed selected observed event')
        if index<len(previous):continue
        require(event['event']=='load','observed unload/transient lifetime violates conservation')
        require(event['path'] in allowed and event['path'] not in updated and event['module_handle'] not in updated.values(),'exact observed prospective load identity')
        updated[event['path']]=event['module_handle']
    return rows,updated

def validate_observed_packet(raw,*,policy,expected_identity,selected_file_map,selected_source_map,expected_boundaries,expected_file_sha256):
    """Replay emitted V2 records, preserving partial failures and provenance.

    This parser supplies neither host-byte qualification nor observer efficacy.
    An incoming nonfixture claim remains a nonfixture claim, never NativePASS.
    """
    require(type(raw) is bytes and 0<len(raw)<=MAX_PACKET_BYTES and sha(expected_file_sha256) and hashlib.sha256(raw).hexdigest()==expected_file_sha256,'external observed packet file anchor')
    value=parse(raw)
    fields={'schema_version','kind','identity','policy_sha256','library_files','observations','failure_records','terminal','producer_claims','fixture_only','native_PASS','native_authority_issued','sha256'}
    require(type(value) is dict and set(value)==fields and type(value['schema_version']) is int and value['schema_version']==2 and value['kind']==OBSERVED_PACKET_KIND and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'exact sealed observed packet schema')
    observed_identity(value['identity'],expected_identity=expected_identity)
    selection(policy,expected_identity,selected_file_map,selected_source_map,expected_boundaries,_observed=True)
    require(value['policy_sha256']==digest(policy) and same(value['library_files'],selected_file_map),'external observed policy/byte-map join')
    require(type(value['fixture_only']) is bool and value['fixture_only'] is expected_identity['fixture_only'] and value['native_PASS'] is value['native_authority_issued'] is False,'typed observed provenance; no authority conversion')
    terminal=value['terminal'];require(type(terminal) is dict and set(terminal)=={'status','recording_complete','guard_failed','guard_closed'} and terminal['status'] in {'complete','failed','incomplete'} and all(type(terminal[k]) is bool for k in ('recording_complete','guard_failed','guard_closed')),'exact typed observed terminal state')
    success=terminal['status']=='complete'
    require((success and terminal['recording_complete'] is True and terminal['guard_failed'] is False and terminal['guard_closed'] is True and expected_identity['owned_birth_status']=='VALIDATED') or (not success and terminal['guard_failed'] is True),'complete recorded state or explicit failure; unavailable birth cannot complete')
    observations=value['observations'];failures=value['failure_records']
    require(type(observations) is list and len(observations)<=MAX_OBSERVATIONS and type(failures) is list and len(failures)<=6*MAX_OBSERVATIONS,'bounded observed checkpoint/failure population')
    require((success and len(observations)>=3 and not failures) or (not success and failures),'success has no failures; partial failures retain errors')
    exact_messages=True
    for index,record in enumerate(failures):
        require(type(record) is dict and set(record)=={'ordinal','stage','boundary_id','observation_ordinal','error'} and type(record['ordinal']) is int and record['ordinal']==index+1 and type(record['stage']) is str and 0<len(record['stage'])<=256,'ordered bounded original failure records')
        require(record['boundary_id'] is None or (integer(record['boundary_id']) and record['boundary_id']>0 and record['boundary_id'] in {p['id'] for p in expected_boundaries}),'selected failure boundary identity')
        require(record['observation_ordinal'] is None or (type(record['observation_ordinal']) is int and 1<=record['observation_ordinal']<=len(observations)),'failure links actual observation or explicitly unavailable stage')
        exact_messages=observed_error(record['error']) and exact_messages
    state=None;events=[];stack=[];entered=[];returned=[];failed=[];directories=None;last_clock=None;first_clock=None;cleanup_seen=False;precleanup=False;known=True;missing=[];violations=[]
    for index,row in enumerate(observations):
        row_fields={'ordinal','time_ns','event','boundary_id','boundary','scope','snapshot','event_trace','directory_identities','cleanup','observation_errors'}
        require(type(row) is dict and set(row)==row_fields and type(row['ordinal']) is int and row['ordinal']==index+1 and integer(row['time_ns']),'typed ordered emitted checkpoint')
        require(last_clock is None or row['time_ns']>=last_clock,'observed checkpoint chronological clock');last_clock=row['time_ns'];first_clock=last_clock if first_clock is None else first_clock;require(last_clock-first_clock<=600*10**9,'bounded observed recorded duration')
        name=row['event'];require(name in {'initial','enter','exit','boundary_error','precleanup','cleanup'} and not cleanup_seen,'exact observed checkpoint label/terminal order')
        require(row['boundary_id'] is None or (integer(row['boundary_id']) and row['boundary_id']>0),'typed observed boundary ID')
        errors=row['observation_errors'];require(type(errors) is dict and set(errors)<=OBSERVATION_COMPONENTS,'exact partial observation component errors')
        for component,error in errors.items():
            exact_messages=observed_error(error) and exact_messages
            require(not success and any(record['observation_ordinal']==row['ordinal'] and same(record['error'],error) for record in failures),'partial component error joins retained failure record')
            missing.append({'ordinal':row['ordinal'],'component':component})
        if name=='initial':
            require(index==0 and row['boundary_id'] is None and row['boundary'] is None,'initial emitted observation first');allowed=set()
        elif name=='enter':
            require(not precleanup and len(entered)<len(expected_boundaries) and len(stack)<MAX_DEPTH,'bounded reached emitted entry')
            plan=expected_boundaries[len(entered)];require(same(row['boundary'],plan) and row['boundary_id']==plan['id'] and plan['parent_id']==(stack[-1]['id'] if stack else None),'emitted name/parent/order equals external plan')
            allowed=stack[-1]['expected'] if stack and stack[-1]['name']=='framework_import' else set()
            frame={'id':plan['id'],'name':plan['name'],'before':None,'expected':expected_new(policy,plan['name']),'nested':set()}
        elif name in {'exit','boundary_error'}:
            require(stack and row['boundary_id']==stack[-1]['id'] and same(row['boundary'],expected_boundaries[len(entered)-1] if stack[-1]['id']==entered[-1] else next(p for p in expected_boundaries if p['id']==stack[-1]['id'])) and not precleanup,'balanced emitted boundary return/failure')
            frame=stack[-1];allowed=frame['expected']
            require(name!='boundary_error' or (not success and any(record['boundary_id']==frame['id'] for record in failures)),'failed boundary retains original error')
        else:
            require(not stack and row['boundary_id'] is None and row['boundary'] is None and entered==[p['id'] for p in expected_boundaries[:len(entered)]],'terminal follows reached external plan prefix')
            if name=='precleanup':require(not precleanup,'precleanup exactly once');precleanup=True;allowed=set()
            else:require(precleanup and index==len(observations)-1,'cleanup after explicit terminal observation');cleanup_seen=True;allowed=set()
        expected_present={'snapshot','event_trace','directory_identities'}|({'cleanup'} if name=='cleanup' else {'scope'})
        require(row['scope'] is None if name=='cleanup' else row['cleanup'] is None,'scope/cleanup exact stage placement')
        for component in expected_present:require(row[component] is not None or component in errors,'missing observed component must have explicit error')
        observed=None;observed_dirs=None;point_known=known and not errors
        for component in expected_present:
            if row[component] is None:continue
            try:
                if component=='snapshot':observed=snapshot(row[component],selected_file_map)
                elif component=='directory_identities':observed_dirs=_observed_directories(row[component],policy)
                elif component=='scope':observed_scope(row[component],policy,fixture_only=value['fixture_only'])
                elif component=='cleanup':cleanup(row[component],policy)
            except ValueError as error:
                require(not success and component in errors,'invalid raw component requires its explicit failure')
                violations.append({'ordinal':row['ordinal'],'component':component,'reason':str(error)});point_known=False
        if observed_dirs is not None:
            if directories is None:directories=observed_dirs
            require(same(observed_dirs,directories) or (not success and 'directory_identities' in errors),'observed directory lifetime conservation')
        if name=='initial' and observed is not None:
            require(set(observed)==set(policy['initial_loaded_libraries']) or (not success and 'snapshot' in errors),'observed exact initial identity population');state=dict(observed)
        if point_known and state is not None:
            try:
                next_events,next_state=_observed_events(row['event_trace'],events,state,allowed,selected_file_map)
                require(next_state==observed,'observed independent event/snapshot join')
                events=next_events;state=next_state
            except ValueError as error:
                require(not success and ('event_trace' in errors or 'snapshot' in errors),'invalid replay must retain explicit component failure')
                violations.append({'ordinal':row['ordinal'],'component':'event_trace','reason':str(error)});point_known=False
        elif row['event_trace'] is not None and 'event_trace' not in errors:
            trace=row['event_trace'];require(type(trace) is dict and set(trace)=={'kind','complete','events'} and trace['kind']=='OBSERVED_PROCESS_LIBRARY_EVENTS' and trace['complete'] is True and type(trace['events']) is list and len(trace['events'])<=MAX_EVENTS,'bounded subsequent trace after incomplete prefix')
            require(same(trace['events'][:len(events)],events),'known event prefix retained after incomplete observation')
        known=point_known
        if name=='enter':
            frame['before']=dict(observed) if point_known else None;entered.append(frame['id']);stack.append(frame)
        elif name in {'exit','boundary_error'}:
            new=set()
            if point_known and frame['before'] is not None:
                require(all(observed.get(p)==h for p,h in frame['before'].items()),'observed existing boundary identity conservation')
                new=set(observed)-set(frame['before']);own=new-frame['nested']
                require((own==frame['expected'] if name=='exit' else own<=frame['expected']) and not(frame['expected']&set(frame['before'])),'reached own/nested observed transitions')
            stack.pop();(returned if name=='exit' else failed).append(frame['id'])
            if stack:stack[-1]['nested'].update(new)
    if success:require(known and cleanup_seen and precleanup and not stack and entered==[p['id'] for p in expected_boundaries] and len(returned)==len(expected_boundaries),'complete independent observed replay')
    else:require(not stack or terminal['status']=='incomplete','unclosed partial boundary explicitly incomplete')
    claims=value['producer_claims'];claim_fields={'completion_claimed','observation_count','event_count','boundary_count','complete_search_directory_coverage'}
    require(type(claims) is dict and set(claims)==claim_fields and type(claims['completion_claimed']) is bool and claims['completion_claimed'] is success and all(type(claims[k]) is int and claims[k]>=0 for k in ('observation_count','event_count','boundary_count')) and type(claims['complete_search_directory_coverage']) is bool,'typed complete/failure producer claims')
    require(claims['observation_count']==len(observations) and claims['boundary_count']==len(entered),'independent reached checkpoint/entry counts')
    if known:require(claims['event_count']==len(events),'independent replayed event count')
    require(policy['kind']!='FINITE_WINDOWS_DLL_BOUNDARY_V2' or claims['complete_search_directory_coverage'] is False,'observed V2 foreign directory state remains UNOBSERVED')
    complete=success and known and exact_messages
    return {'kind':'OFFLINE_OBSERVED_LIBRARY_EVIDENCE_REPLAY','status':'OBSERVED_LIBRARY_COMPLETE_REPLAYED_UNQUALIFIED' if complete else 'OBSERVED_LIBRARY_FAILURE_PRESERVED' if terminal['status']=='failed' and exact_messages else 'OBSERVED_LIBRARY_INCOMPLETE_PRESERVED','fixture_only':value['fixture_only'],'native_PASS':False,'native_authority_issued':False,'native_producer_integration_verified':False,'current_runtime_bytes_qualified':False,'lossless_native_observer_qualified':False,'actual_search_directory_coverage_qualified':False,'conservation_complete':complete,'verified_prefix_available':known,'reached_counts':{'observations':len(observations),'entered':len(entered),'returned':len(returned),'failed':len(failed),'unreached':len(expected_boundaries)-len(entered),'verified_events':len(events)},'unknown_components':missing,'independent_violations':violations,'failure_records':failures,'incoming_producer_claims':claims,'original_identity':expected_identity,'final_verified_libraries':state if known else None}
