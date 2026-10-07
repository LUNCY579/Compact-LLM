"""Independent observer-only terminal byte auditor; no observer is loaded.

Native host facts belong to separately reviewed source-bound interfaces. CPU
fixture capabilities retain CPU provenance and cannot become native authority.
"""
from pathlib import Path
import hashlib,json,math,stat,time

MAX_JSON=16*1024**2
SCOPE='NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD'
OPS=('source_preflight','owned_identity','observer_binding','observer_start','initial_module_baseline','fixture_load','loaded_module_snapshot','fixture_unload','unloaded_module_snapshot','observer_stop','final_event_drain','final_source_preflight')
SNAPSHOT_FIELDS=('abi_version','row_bytes','snapshot_bytes','row_capacity','full_capacity','base_capacity','phase','faults','attempted','reserved','published_prefix','inflight','cookie_live','register_status','unregister_status','drained','complete','dropped')
ABI_FIXED={'abi_version':1,'row_bytes':2600,'snapshot_bytes':72,'row_capacity':4096,'full_capacity':1024,'base_capacity':256}

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def same(left,right):return canonical(left)==canonical(right)
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def uint(value):return type(value) is int and 0<=value<2**64
def fields(value,names,label):require(type(value) is dict and set(value)==set(names),'exact '+label+' schema')
def pairs(rows):
    result={}
    for key,value in rows:require(key not in result,'duplicate JSON key');result[key]=value
    return result
def floating(text):
    value=float(text);require(math.isfinite(value),'nonfinite/overflow JSON value');return value
def integer(text):
    require(len(text)<=21,'bounded JSON integer');value=int(text);require(-2**63<=value<2**64,'bounded JSON integer');return value
def parse(raw):
    require(type(raw) is bytes and len(raw)<=MAX_JSON,'bounded actual JSON bytes')
    try:return json.loads(raw,object_pairs_hook=pairs,parse_float=floating,parse_int=integer,parse_constant=lambda _:require(False,'nonfinite JSON value'))
    except (UnicodeError,RecursionError) as error:raise ValueError('bounded valid JSON') from error

class ReadSet:
    """Strict first-read descriptors plus complete flat directory rosters."""
    def __init__(self):self.files={};self.rosters={};self.total=0;self.started=time.monotonic()
    def check(self):require(time.monotonic()-self.started<=600,'bounded observer audit read deadline')
    def physical(self,path):
        path=Path(path);require(path.is_absolute() and path.resolve()==path,'exact physical selected path')
        for part in (path,*path.parents):require(not part.is_symlink() and not(getattr(part.lstat(),'st_file_attributes',0)&0x400),'observer audit refuses reparse aliases')
        return path
    def roster(self,path):
        self.check();path=self.physical(path);require(path.is_dir(),'selected directory')
        rows={p.name for p in path.iterdir()};require(len(rows)<=4096,'bounded complete roster')
        key=str(path);require(key not in self.rosters or rows==self.rosters[key],'original roster cannot rebase');self.rosters.setdefault(key,rows);return rows
    def read(self,path,descriptor,maximum=MAX_JSON):
        self.check();fields(descriptor,{'bytes','sha256'},'selected byte descriptor')
        require(uint(descriptor['bytes']) and descriptor['bytes']<=maximum and sha(descriptor['sha256']),'bounded external byte anchor')
        path=self.physical(path);before=path.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==descriptor['bytes'],'regular exact selected bytes')
        self.total+=before.st_size;require(self.total<=512*1024**2,'bounded whole observer audit bytes')
        with path.open('rb') as stream:raw=stream.read(before.st_size+1)
        after=path.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'stable exact observer audit bytes')
        key=str(path);require(key not in self.files or descriptor==self.files[key],'original byte anchor cannot rebase');self.files.setdefault(key,dict(descriptor));return raw
    def verify(self):
        for path,rows in list(self.rosters.items()):require(self.roster(Path(path))==rows,'final complete observer roster')
        for path,d in list(self.files.items()):self.read(Path(path),d)

def collector_snapshot(value,*,closed):
    fields(value,SNAPSHOT_FIELDS,'collector ABI snapshot')
    require(all(type(n) is int and (-2**31<=n<2**31 if k in ('register_status','unregister_status') else 0<=n<2**32) for k,n in value.items()),'typed collector ABI integers')
    require({k:value[k] for k in ABI_FIXED}==ABI_FIXED,'exact collector ABI sizes/capacities')
    require(value['faults']==value['dropped']==value['inflight']==0 and value['attempted']==value['reserved']==value['published_prefix']<=4096 and value['register_status']==0,'lossless published collector; no late fault/drop/inflight/truncation')
    if closed:require(value['phase']==5 and value['cookie_live']==0 and value['unregister_status']==0 and value['drained']==value['complete']==1,'native unregister/drain/seal terminal counters')
    else:require(value['phase']==2 and value['cookie_live']==1 and value['unregister_status']==-2**31 and value['drained']==value['complete']==0,'active registered collector counters')
    return value['published_prefix']

def modules(value,files):
    fields(value,{'kind','complete','libraries'},'complete module snapshot')
    require(value['kind']=='OBSERVED_PROCESS_LIBRARY_SNAPSHOT' and value['complete'] is True and type(value['libraries']) is list and 0<len(value['libraries'])<=4096,'bounded complete process baseline')
    result={};handles=set()
    for row in value['libraries']:
        fields(row,{'path','module_handle'},'module identity');p=row['path'];h=row['module_handle']
        require(type(p) is str and p in files and uint(h) and h>0 and p not in result and h not in handles,'unique selected module path/handle')
        result[p]=h;handles.add(h)
    return result

def replay_observer_records(result,*,library_files,observer_binding,fixture_path,owned_birth,source_commit,config_sha256):
    """Independent raw observer invariants; never imports producer/observer."""
    fields(result,{'schema_version','kind','scope','status','source_commit','config_sha256','owned_identity','abi_binding','fixture_handle','initial_modules','loaded_modules','unloaded_modules','events','final_collector_snapshot','phase_counts','errors','fixture_only','native_PASS','native_authority_issued','sha256'},'observer worker result')
    require(type(result['schema_version']) is int and result['schema_version']==1 and result['kind']=='OBSERVER_ONLY_WORKER_RESULT' and result['scope']==SCOPE and result['status']=='OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT','complete observer operations only')
    require(result['sha256']==digest({k:v for k,v in result.items() if k!='sha256'}),'worker result body checksum')
    require(result['source_commit']==source_commit and result['config_sha256']==config_sha256 and same(result['owned_identity'],owned_birth) and same(result['abi_binding'],observer_binding),'external source/config/owned/collector ABI identity joins')
    require(type(result['fixture_only']) is bool and result['native_PASS'] is result['native_authority_issued'] is False and result['errors']==[],'no producer failure/native authority claim')
    before=modules(result['initial_modules'],library_files);loaded=modules(result['loaded_modules'],library_files);after=modules(result['unloaded_modules'],library_files)
    handle=result['fixture_handle'];require(uint(handle) and handle>0 and fixture_path in library_files and fixture_path not in before and handle not in before.values(),'selected new fixture handle')
    require(loaded=={**before,fixture_path:handle} and after==before,'exact baseline/load/unload snapshot conservation')
    events=result['events'];fields(events,{'kind','complete','events'},'observer final event drain')
    require(events['kind']=='OBSERVED_PROCESS_LIBRARY_EVENTS' and events['complete'] is True and type(events['events']) is list and len(events['events'])==2,'exact lossless fixture event population')
    expected=[{'sequence':1,'event':'load','path':fixture_path,'module_handle':handle},{'sequence':2,'event':'unload','path':fixture_path,'module_handle':handle}]
    require(same(events['events'],expected),'ordered immutable fixture load/unload identities')
    require(collector_snapshot(result['final_collector_snapshot'],closed=True)==len(expected),'sealed native event count independently matches observed rows')
    return {'initial_libraries':before,'fixture_handle':handle,'events':expected,'observer_mechanics_recorded':True,'native_PASS':False,'native_authority_issued':False,'tensor_authority':False,'native_fit_established':False,'scoring_authorized':False,'study_authority':False}

def _raw_rows(rows,expected_events):
    require(type(rows) is list and len(rows)==2,'exact two raw callback rows')
    decoded=[];sizes=[]
    for index,row in enumerate(rows):
        fields(row,{'sequence','base_address','reason','image_size','full_units','base_units','flags','ready','full_name','base_name'},'raw collector row')
        require(all(type(row[k]) is int for k in ('sequence','base_address','reason','image_size','full_units','base_units','flags','ready')) and row['sequence']==index+1 and 0<row['base_address']<2**64 and row['reason']==index+1 and 0<row['image_size']<2**32 and row['flags']==0 and row['ready']==1,'exact ready lossless raw callback row')
        text={}
        for label,capacity in (('full',1024),('base',256)):
            value=row[label+'_name'];units=row[label+'_units']
            require(type(value) is list and 0<units<capacity and len(value)==units and all(type(n) is int and 0<=n<65536 for n in value),'nontruncated counted UTF16 callback names')
            try:name=b''.join(n.to_bytes(2,'little') for n in value).decode('utf-16-le','strict')
            except UnicodeError as error:raise ValueError('valid UTF16 callback name') from error
            require('\0' not in name,'no embedded callback path NUL');text[label]=name
        full=text['full']
        for prefix in ('\\??\\','\\\\?\\'):
            if full.startswith(prefix):full=full[len(prefix):];break
        require(str(Path(full))==full and Path(full).is_absolute() and Path(full).name==text['base'],'exact callback full/base path join')
        decoded.append({'sequence':index+1,'event':'load' if index==0 else 'unload','path':full,'module_handle':row['base_address']});sizes.append(row['image_size'])
    require(same(decoded,expected_events) and sizes[0]==sizes[1],'raw callback rows equal immutable load/unload events and image identity')

def replay_phases(raw,*,result,selected_context):
    require(type(raw) is bytes and 0<len(raw)<=4*1024**2 and raw.endswith(b'\n'),'bounded complete observer phase ledger')
    lines=raw.splitlines(keepends=True);require(len(lines)==2*len(OPS) and all(len(line)<=65536 for line in lines),'exact complete phase population and line bounds')
    counts={op:{'attempted':0,'returned':0,'failed':0,'pending':0} for op in OPS};returned={};clock=None
    for index,line in enumerate(lines):
        row=parse(line);fields(row,{'schema_version','ordinal','operation','event','monotonic','data','exception'},'observer phase row')
        require(type(row['schema_version']) is int and row['schema_version']==1 and type(row['ordinal']) is int and row['ordinal']==index+1 and row['operation']==OPS[index//2] and row['event']==('started' if index%2==0 else 'returned'),'source-defined exact operation order/cost')
        require(type(row['monotonic']) in (int,float) and math.isfinite(row['monotonic']) and row['monotonic']>=0 and (clock is None or row['monotonic']>=clock),'ordered finite observer phase clocks');clock=row['monotonic']
        require(row['exception'] is None and (row['data'] is None if index%2==0 else type(row['data']) is dict),'complete phase has ordinary return and no hidden exception')
        c=counts[row['operation']]
        if index%2==0:c['attempted']+=1;c['pending']+=1
        else:c['returned']+=1;c['pending']-=1;returned[row['operation']]=row['data']
    require(same(counts,result['phase_counts']),'independently recomputed operation counts')
    value=selected_context;source_map=value['config']['source_files'];runtime_map=value['runtime']['files']
    projection={'context_sha256':digest(value),'config_sha256':value['anchors']['expected_config_file'],'source_count':len(source_map),'runtime_count':len(runtime_map),'verified_bytes':sum(d['bytes'] for d in source_map.values())+sum(d['bytes'] for d in runtime_map.values())}
    require(same(returned['source_preflight'],projection) and same(returned['final_source_preflight'],projection),'same exact original source/runtime preflight projection')
    for op,key in (('owned_identity','owned_identity'),('observer_binding','abi_binding'),('initial_module_baseline','initial_modules'),('loaded_module_snapshot','loaded_modules'),('unloaded_module_snapshot','unloaded_modules')):require(same(returned[op],result[key]),'actual returned '+op+' matches result')
    start=returned['observer_start'];fields(start,{'status','snapshot'},'collector start');require(type(start['status']) is int and start['status']==0 and collector_snapshot(start['snapshot'],closed=False)==0,'successful empty native registration prefix')
    fixture=value['config']['binary_fixture']['path'];handle=result['fixture_handle']
    require(same(returned['fixture_load'],{'path':fixture,'module_handle':handle}) and same(returned['fixture_unload'],{'path':fixture,'module_handle':handle,'returned':True}),'exact counted fixture load/unload primitives')
    stop=returned['observer_stop'];fields(stop,{'status','snapshot'},'collector stop');require(type(stop['status']) is int and stop['status']==0 and same(stop['snapshot'],result['final_collector_snapshot']),'native unregister/drain recorded before final immutable read')
    drain=returned['final_event_drain'];fields(drain,{'events','snapshot','raw_rows'},'final drain')
    require(same(drain['events'],result['events']) and same(drain['snapshot'],result['final_collector_snapshot']),'final drain/result immutable event-counter joins')
    _raw_rows(drain['raw_rows'],result['events']['events'])
    return {'counts':counts,'first_clock':parse(lines[0])['monotonic'],'last_clock':clock,'returned':returned}

def replay_bootstrap_modules(bootstrap,runtime,*,fixture_only):
    """Replay exact source origins or separately observed finite namespace paths."""
    base={'kind','imports','modules','unknown_modules','fixture_only','namespace_execution_proven'}
    no_site=runtime.get('cache_policy',{}).get('mode')=='MEASURE_BYTE_VERIFIED_STARTUP_NO_SITE_V3'
    fields(bootstrap,base|{'interpreter_profile'} if no_site else base,'measured bootstrap')
    if no_site:require(same(bootstrap['interpreter_profile'],runtime['cache_policy']['interpreter_profile']) and bootstrap['interpreter_profile'].get('source_execution_history_proven') is False,'original measured no-site interpreter profile joins exact admission')
    require(bootstrap['kind']=='MEASURED_BOOTSTRAP_IMPORTS_V1' and bootstrap['imports']==['ctypes','ctypes.wintypes','_ctypes'] and bootstrap['fixture_only'] is fixture_only and bootstrap['namespace_execution_proven'] is False and bootstrap['unknown_modules']==[] and type(bootstrap['modules']) is list and len(bootstrap['modules'])<=4096,'bounded original bootstrap measurement, no unselected namespaces')
    selected=runtime.get('cache_policy',{}).get('allowed_bootstrap_modules',{});namespaces=runtime.get('cache_policy',{}).get('allowed_namespace_packages',{});names=set()
    def normalized(text):
        p=Path(text);layout=runtime.get('stdlib_layout',{});logical=layout.get('logical_base_prefix')
        return str(Path(layout['physical_base_prefix'])/p.relative_to(logical)) if logical is not None and p.is_relative_to(logical) else str(p)
    for row in bootstrap['modules']:
        base={'name','origin','classification','source_sha256','error'}
        fields(row,base|{'spec_search_locations','module_search_locations','namespace_files','namespace_files_sha256'} if row.get('classification')=='namespace_package' else base,'original namespace origin')
        require(type(row['name']) is str and row['name'] not in names,'unique original namespace names');names.add(row['name'])
        require(row['error'] is None,'no hidden original namespace origin/byte verification error')
        if row['classification']=='builtin_or_frozen':require(row['origin'] in ('built-in','frozen') and row['source_sha256'] is None,'interpreter-reported builtin/frozen origin; no execution proof')
        elif row['classification']=='namespace_package':
            entry=namespaces.get(row['name'])
            require(type(entry) is dict and set(entry)=={'search_locations','files','files_sha256'} and row['origin'] is row['source_sha256'] is None,'selected namespace distinct from source origin')
            require(type(row['spec_search_locations']) is type(row['module_search_locations']) is list and row['spec_search_locations']==row['module_search_locations']==entry['search_locations'] and row['spec_search_locations'] and all(type(p) is str and p==normalized(p) for p in row['spec_search_locations']),'two exact ordered original namespace search path observations')
            require(type(row['namespace_files']) is dict and row['namespace_files']==entry['files'] and row['namespace_files_sha256']==entry['files_sha256']==digest(row['namespace_files']) and all(runtime['files'].get(path)==d for path,d in row['namespace_files'].items()),'fresh measured namespace roster joins selected runtime bytes')
        else:
            entry=selected.get(row['name']);require(row['classification']=='source_or_extension' and type(entry) is dict and row['origin']==normalized(entry['path']) and row['source_sha256']==entry['source_sha256'],'measured namespace source/extension exact prospective admission')
    if not fixture_only:require(set(bootstrap['imports'])<=names,'actual explicit bootstrap imports measured')
    return names

def replay_startup(startup,*,result,selected_context):
    """Replay measured originals against a prospective admission, not a prior fact."""
    fields(startup,{'schema_version','kind','scope','source_commit','implementation_commit','config_sha256','context_sha256','bootstrap_policy_sha256','bootstrap','initial_module_snapshots','unknown_libraries','fixture_only','namespace_execution_proven','native_PASS','native_authority_issued','sha256'},'measured startup inventory')
    value=selected_context;config=value['config'];runtime=value['runtime']
    require(type(startup['schema_version']) is int and startup['schema_version']==1 and startup['kind']=='OBSERVER_STARTUP_INVENTORY_V1' and startup['scope']==SCOPE and startup['sha256']==digest({k:v for k,v in startup.items() if k!='sha256'}),'original measured startup schema/body')
    require(startup['source_commit']==value['expected_head'] and startup['implementation_commit']==config['implementation_commit'] and startup['config_sha256']==value['anchors']['expected_config_file'] and startup['context_sha256']==digest(value) and startup['bootstrap_policy_sha256']==digest(runtime.get('cache_policy',{})),'external dispatch/implementation/context/import-policy joins')
    require(startup['fixture_only'] is value['fixture_only'] and startup['namespace_execution_proven'] is startup['native_PASS'] is startup['native_authority_issued'] is False,'startup observation grants no namespace/native authority')
    observed=startup['initial_module_snapshots'];require(type(observed) is list and len(observed)==2 and same(observed[0],result['initial_modules']),'original first complete baseline retained, no admitted-map substitution')
    first=modules(observed[0],runtime['files']);second=modules(observed[1],runtime['files'])
    require(first==second and set(first)<=set(runtime['allowed_initial_loaded_libraries']) and startup['unknown_libraries']==[],'paired original baseline inside prospective admission; no unknown modules')
    replay_bootstrap_modules(startup['bootstrap'],runtime,fixture_only=value['fixture_only'])
    return {'initial_libraries':first,'namespace_execution_proven':False,'native_authority_issued':False}

def _sealed(value,pin):require(type(value) is dict and sha(pin) and value.get('sha256')==pin==digest({k:v for k,v in value.items() if k!='sha256'}),'external sealed body identity')
def _birth(value):
    fields(value,{'pid','creation_filetime_100ns'},'PID/FILETIME birth');require(all(uint(n) and n>0 for n in value.values()),'positive original birth');return value

def _resources(sup,samples,first,last):
    admissions=sup['admission_observations'];dispatch=sup['dispatch_observation']
    require(type(admissions) is list and len(admissions)==2,'two original observer admissions')
    uuid=None
    for row in admissions+[dispatch]:
        require(all(uint(row[k]) for k in ('host_available_bytes','worker_tree_rss_bytes','disk_free_bytes','gpu_free_bytes','gpu_total_bytes')),'typed admission byte amounts')
        require(row['host_available_bytes']>=3.3125*1024**3 and row['worker_tree_rss_bytes']<=2*1024**3 and row['disk_free_bytes']>=4*1024**3 and 4*1024**3<=row['gpu_free_bytes']<=row['gpu_total_bytes'],'original observer admission RAM/disk/GPU floors')
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in ('sample_started_monotonic','sample_completed_monotonic','gpu_sample_started_monotonic','gpu_sample_completed_monotonic')) and row['sample_started_monotonic']<=row['sample_completed_monotonic'] and row['gpu_sample_started_monotonic']<=row['gpu_sample_completed_monotonic'],'ordered finite admission times')
        require(type(row['GPU_index']) is int and row['GPU_index']==0 and type(row['gpu_uuid']) is str and row['gpu_uuid'] and (uuid is None or row['gpu_uuid']==uuid),'same selected GPU0 telemetry identity');uuid=row['gpu_uuid']
    require(admissions[1]['sample_started_monotonic']-admissions[0]['sample_started_monotonic']>=2 and admissions[1]['gpu_observation_id']!=admissions[0]['gpu_observation_id'] and admissions[1]['gpu_sample_started_monotonic']>admissions[0]['gpu_sample_started_monotonic'] and dispatch['sample_started_monotonic']-admissions[0]['sample_started_monotonic']<=30,'distinct two-second observer admissions within30s')
    resume=sup['worker_resume_monotonic'];require(dispatch['sample_completed_monotonic']<=resume and dispatch['gpu_sample_completed_monotonic']<=resume and resume-dispatch['sample_started_monotonic']<=.2 and resume-dispatch['gpu_sample_started_monotonic']<=2,'fresh original before-resume telemetry')
    require(type(samples) is list and 0<len(samples)<=32768,'bounded observer resource samples');clocks={'CPU':[],'GPU':[]}
    for row in samples:
        kind=row.get('kind');require(kind in clocks,'known CPU/GPU resource sample');clock=row['sample_started_monotonic'] if kind=='CPU' else row['gpu_sample_started_monotonic']
        require(type(clock) in (int,float) and math.isfinite(clock) and clock>=0 and (not clocks[kind] or clock>=clocks[kind][-1]),'chronological observer telemetry');clocks[kind].append(clock)
        if kind=='CPU':require(uint(row['host_available_bytes']) and row['host_available_bytes']>=1.25*1024**3 and uint(row['worker_tree_rss_bytes']) and row['worker_tree_rss_bytes']<=2*1024**3,'original observer work RAM/RSS floor')
        else:require(uint(row['gpu_free_bytes']) and uint(row['gpu_total_bytes']) and .5*1024**3<=row['gpu_free_bytes']<=row['gpu_total_bytes'] and row['gpu_uuid']==uuid,'same-device during-scope GPU floor')
    for kind,age in (('CPU',.2),('GPU',2.0)):
        rows=clocks[kind];require(rows and rows[0]<=resume+age and rows[-1]>=last-age and any(first<=t<=last for t in rows) and all(b-a<=age for a,b in zip(rows,rows[1:])),'observer whole recorded work telemetry coverage/cadence')

def replay_worker_command(sup,value,original):
    if value['runtime']['cache_policy']['mode']=='MEASURE_BYTE_VERIFIED_STARTUP_NO_SITE_V3':
        expected=[value['runtime']['cache_policy']['interpreter_profile']['executable'],'-I','-S','-B',value['config']['worker_path'],'--config',value['config_path'],'--output-dir',str(original)]
        require(sup.get('attempted_worker_argv')==expected,'actual no-site worker command exact registered interpreter/flags/worker/config/output')

def audit_terminal(preserved_directory,*,original_directory,selected_context,expected_preservation_file_sha256,expected_preservation_body_sha256,revalidate,fresh_death):
    """Full original/snapshot/record replay; host interfaces externally bound.

    Returns observer-only recorded mechanics, never tensor/fit/study authority.
    CPU classification comes from the original context and remains immutable.
    """
    require(callable(revalidate) and callable(fresh_death),'separately reviewed read-only context/death interfaces')
    value=parse(canonical(selected_context).encode('ascii'));projection=canonical(value);require(same(revalidate(),value),'same original externally selected observer context')
    config=value['config'];runtime=value['runtime'];registration=value['registration'];fixture=value['fixture_only']
    require(type(fixture) is bool and value['native_PASS'] is value['native_authority_issued'] is False and config['scope']==SCOPE,'observer-only original scope; no authority conversion')
    permission=config['permission'];require(type(permission) is dict and permission.get('allocations')==0 and type(permission.get('allocations')) is int and all(permission.get(k) is False for k in ('model','torch_import','CUDA_context','tensor_allocation','forward','backward','optimizer','decoder','scoring','protected_bank_access','M0_membership','paid_services')),'no model/framework/device/tensor/study permission')
    _sealed(config,value['anchors']['expected_config_body']);_sealed(runtime,value['anchors']['expected_runtime_body']);_sealed(registration,value['anchors']['expected_registration_body'])
    require(same(config['runtime_files'],runtime['files']) and same(config['source_files'],registration['source_files']) and same(runtime['files'],registration['runtime_files']) and registration['config_body_sha256']==config['sha256'] and registration['runtime_body_sha256']==runtime['sha256'] and config['implementation_commit']==runtime['implementation_commit']==registration['implementation_commit'] and config['implementation_commit']!=value['expected_head'],'original source/runtime/run registration closure')
    reader=ReadSet();root=Path(preserved_directory).absolute();original=Path(original_directory).absolute();require(root!=original,'independent original and preserved observer leaves')
    p=root/'PRESERVATION.json';pres=parse(reader.read(p,{'bytes':p.lstat().st_size,'sha256':expected_preservation_file_sha256}));_sealed(pres,expected_preservation_body_sha256)
    fields(pres,{'kind','scope','manifest_sha256','raw_files','source_files','source_snapshot_files','lease_files','current_death_observation','fixture_only','native_authority_issued','sha256'},'observer-only preservation')
    require(pres['kind']=='PRESERVED_OBSERVER_ONLY_QUALIFICATION_V1' and pres['scope']==SCOPE and pres['manifest_sha256']==config['sha256'] and pres['fixture_only'] is fixture and pres['native_authority_issued'] is False and same(pres['source_files'],config['source_files']),'original observer preservation/context/source joins')
    needed={'OBSERVER_STARTUP_INVENTORY.json','OBSERVER_PHASES.jsonl','OBSERVER_WORKER_RESULT.json','SUPERVISOR_RECEIPT.json','RESOURCE_OBSERVATIONS.json','supervisor_ownership.json','OWNED_EXTERNAL_ANCHORS.json','worker.stdout.log','worker.stderr.log','supervisor_heartbeat.jsonl'}
    names=set(pres['raw_files']);require(needed<=names and len(names)<=256 and all(type(n) is str and Path(n).name==n and Path(n).suffix in ('.json','.jsonl','.log','.txt') for n in names),'finite full flat observer raw population')
    require(reader.roster(original)==names,'original whole observer roster');raw={}
    for n,d in pres['raw_files'].items():raw[n]=reader.read(root/n,d);require(reader.read(original/n,d)==raw[n],'all original/preserved observer raw bytes')
    require(sum(len(x) for x in raw.values())<=64*1024**2,'bounded whole observer raw bytes')
    data={n:parse(b) for n,b in raw.items() if n.endswith('.json')}
    require(same(data['OWNED_EXTERNAL_ANCHORS.json'],{'config_path':value['config_path'],'anchors':value['anchors']}),'original config/runtime/run external anchors')
    sources=config['source_files'];snapshots=pres['source_snapshot_files'];require(0<len(sources)<=4096 and set(snapshots)==set(sources),'complete selected source snapshots');snapshot_names=set()
    for text,d in sources.items():
        require(type(text) is str and Path(text).suffix in ('.py','.c','.h','.json','.md'),'source-only original/snapshot inventory')
        copy=snapshots[text];name=hashlib.sha256(text.encode('utf-8')).hexdigest()+Path(text).suffix
        fields(copy,{'path','bytes','sha256','role'},'source snapshot');require(copy['path']=='source_snapshots/'+name and same({k:copy[k] for k in ('bytes','sha256','role')},d),'source snapshot original descriptor join')
        expected={k:d[k] for k in ('bytes','sha256')};require(reader.read(Path(text),expected)==reader.read(root/'source_snapshots'/name,expected),'original selected source/snapshot bytes');snapshot_names.add(name)
    require(reader.roster(root/'source_snapshots')==snapshot_names,'exact source copy roster')
    observer=runtime['library_event_observer'];require(same(observer,config['observer_registration']) and observer['abi']=={'version':1,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,'row_bytes':2600,'snapshot_bytes':72,'architecture':'x64'},'same selected native observer registration/ABI')
    # Only the two selected observer/fixture binary identities are read here.
    # Complete current runtime/host/cache qualification is the external context
    # interface; this audit never loads any binary to validate itself.
    for binary in (observer['binary'],config['binary_fixture']):
        require(runtime['files'].get(binary['path'])==binary['descriptor'],'selected binary exact runtime map identity')
        reader.read(Path(binary['path']),{k:binary['descriptor'][k] for k in ('bytes','sha256')})
    build=observer['build_receipt'];fields(build,{'path','file_sha256','body_sha256'},'external observer build selection')
    bp=Path(build['path']);build_data=parse(reader.read(bp,{'bytes':bp.lstat().st_size,'sha256':build['file_sha256']}));_sealed(build_data,build['body_sha256'])
    # The qualified contract performs exact compiler/source/binary build joins;
    # retaining its original independently selected receipt prevents rebasing.
    sup=data['SUPERVISOR_RECEIPT.json'];owner=data['supervisor_ownership.json'];launcher=_birth({k:sup['worker_identity'][k] for k in ('pid','creation_filetime_100ns')})
    replay_worker_command(sup,value,original)
    require(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and type(sup['returncode']) is int and sup['returncode']==0 and all(sup[k] is True for k in ('worker_attempted','worker_launched','worker_resumed','preflight_revalidated_before_current_observations','job_closed_kill_on_close','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified','partial_outputs_frozen')) and sup['owned_worker_identity_currently_alive'] is False and sup['unjoined_sampler_threads']==[],'actual successful owned terminal/Job/samplers/telemetry cleanup')
    require(sup['output_dir']==str(original) and sup['config_sha256']==value['anchors']['expected_config_file'] and sup['preflight']['source_commit']==value['expected_head'] and sup['preflight']['authorization']['scope']==SCOPE and sup['preflight']['profile_sha256']==registration['profile_sha256'] and same(sup['preflight']['resources'],config['resources']),'exact observer-only supervisor/profile/resources')
    require(owner['worker_pid']==launcher['pid'] and owner['config_sha256']==sup['config_sha256'] and owner['job_name']==sup['worker_identity']['job_name'] and sup['worker_identity']['assigned_before_resume'] is sup['worker_identity']['job_kill_on_close'] is True and sup['worker_identity']['breakaway_allowed'] is False,'original suspended launcher/Job/config ownership')
    partial={}
    for row in sup['partial_outputs']:
        fields(row,{'path','bytes','sha256'},'partial raw file');name=Path(row['path']).name
        require(type(row['path']) is str and row['path'] in (name,str(original/name)) and name not in partial,'exact unique supervisor raw path');partial[name]={k:row[k] for k in ('bytes','sha256')}
    require(partial=={n:d for n,d in pres['raw_files'].items() if n!='SUPERVISOR_RECEIPT.json'},'all frozen observer raw partial denominators')
    result=data['OBSERVER_WORKER_RESULT.json'];born=_birth(result['owned_identity']);births={(launcher['pid'],launcher['creation_filetime_100ns']):launcher}
    samples=data['RESOURCE_OBSERVATIONS.json'];require(type(samples) is list and samples,'original recorded resource population')
    for sample in samples:
        require(sample.get('kind') in ('CPU','GPU'),'known original resource sampler')
        if sample['kind']=='CPU':
            require(uint(sample['host_available_bytes']) and sample['host_available_bytes']>=1.25*1024**3 and uint(sample['worker_tree_rss_bytes']) and sample['worker_tree_rss_bytes']<=2*1024**3,'original CPU reserve/RSS')
            for row in sample['owned_process_identities']:
                b=_birth({k:row[k] for k in ('pid','creation_filetime_100ns')});births[(b['pid'],b['creation_filetime_100ns'])]=b
        else:require(uint(sample['gpu_free_bytes']) and sample['gpu_free_bytes']>=.5*1024**3,'original recorded during-scope GPU floor')
    require((born['pid'],born['creation_filetime_100ns']) in births,'actual executing observer worker birth in independent owned population')
    leases=pres['lease_files'];require(set(leases)=={'reservation.json','worker.json','resume.json','terminal.json'},'complete original four lease files');lease_names=set();lease_data={}
    for n,d in leases.items():
        fields(d,{'source_path','path','bytes','sha256'},'original lease copy');require(d['source_path']==str(Path(sup['lease_path'])/n) and d['path']=='lease_'+n,'original lease source/copy pointer')
        b=reader.read(root/d['path'],{k:d[k] for k in ('bytes','sha256')});require(reader.read(Path(d['source_path']),{k:d[k] for k in ('bytes','sha256')})==b,'all original/copy lease bytes');lease_data[n]=parse(b);lease_names.add(d['path'])
        if n=='terminal.json':require(b==raw['SUPERVISOR_RECEIPT.json'],'lease terminal exact supervisor bytes')
    require(reader.roster(Path(sup['lease_path']))==set(leases) and same(lease_data['worker.json'],sup['worker_identity']) and same(lease_data['resume.json']['worker_identity'],sup['worker_identity']) and lease_data['resume.json']['external_monitor_ready_before_resume'] is True and lease_data['resume.json']['resumed_monotonic']==sup['worker_resume_monotonic'] and lease_data['reservation.json']['config_sha256']==sup['config_sha256'],'original lease birth/resume/reservation joins')
    binding=result['abi_binding'];require(type(binding) is dict and set(binding)=={'path','descriptor','abi','dll_handle'} and binding['path']==observer['binary']['path'] and binding['descriptor']==observer['binary']['descriptor'] and binding['abi']==observer['abi'] and uint(binding['dll_handle']) and binding['dll_handle']>0,'actual selected observer handle/bytes/ABI binding')
    libraries={p:d for p,d in runtime['files'].items() if Path(p).suffix.lower() in ('.dll','.pyd','.exe')}
    replay=replay_observer_records(result,library_files=libraries,observer_binding=binding,fixture_path=config['binary_fixture']['path'],owned_birth=born,source_commit=value['expected_head'],config_sha256=value['anchors']['expected_config_file'])
    startup=replay_startup(data['OBSERVER_STARTUP_INVENTORY.json'],result=result,selected_context=value)
    require(startup['initial_libraries']==replay['initial_libraries'],'independent startup/worker baseline joins')
    require(set(replay['initial_libraries'])<=set(runtime['allowed_initial_loaded_libraries']) and replay['initial_libraries'].get(observer['binary']['path'])==binding['dll_handle'],'complete externally selected initial collector/module baseline')
    phases=replay_phases(raw['OBSERVER_PHASES.jsonl'],result=result,selected_context=value)
    require(type(sup['worker_resume_monotonic']) in (int,float) and type(sup['deadline_monotonic']) in (int,float) and sup['worker_resume_monotonic']<=phases['first_clock']<=phases['last_clock']<=sup['deadline_monotonic'] and 0<=sup['deadline_monotonic']-sup['worker_resume_monotonic']<=600,'all observer work inside original owned600s window')
    _resources(sup,samples,phases['first_clock'],phases['last_clock'])
    death=pres['current_death_observation'];fields(death,{'observed_monotonic','observed_utc','supervisor_file_sha256','config_file_sha256','observations','death_verifier_source_sha256'},'current all-birth death receipt')
    from datetime import datetime
    verifier=str(Path(value['config_path']).parent.parent/'native_supervisor.py')
    require(death['supervisor_file_sha256']==pres['raw_files']['SUPERVISOR_RECEIPT.json']['sha256'] and death['config_file_sha256']==sup['config_sha256'] and verifier in sources and death['death_verifier_source_sha256']==sources[verifier]['sha256'],'current death/SUP/config/pinned original verifier anchors')
    require(type(death['observed_monotonic']) in (int,float) and math.isfinite(death['observed_monotonic']) and death['observed_monotonic']>=phases['last_clock'] and type(death['observed_utc']) is str and len(death['observed_utc'])<=64 and datetime.fromisoformat(death['observed_utc']).utcoffset() is not None,'current terminal death after observed work/timezone')
    observed={}
    for row in death['observations']:
        fields(row,{'identity','currently_alive'},'dead original birth');b=_birth(row['identity']);key=(b['pid'],b['creation_filetime_100ns']);require(key not in observed and row['currently_alive'] is False,'unique preserved dead birth');observed[key]=b
    require(observed==births,'complete actual original owned birth death roster')
    for b in births.values():require(fresh_death(dict(b)) is False,'fresh exact same birth remains dead')
    require(reader.roster(root)==names|lease_names|{'PRESERVATION.json','source_snapshots'},'whole preserved observer raw/source/lease roster')
    reader.verify();require(canonical(selected_context)==projection and same(revalidate(),value),'same original external observer context after callbacks')
    return {'kind':'INDEPENDENT_OBSERVER_ONLY_TERMINAL_AUDIT','scope':SCOPE,'status':'CPU_OBSERVER_MECHANICS_RECORDED_UNQUALIFIED' if fixture else 'OBSERVER_MECHANICS_RECORDED_PENDING_CURRENT_QUALIFICATION','fixture_only':fixture,'observer_mechanics_recorded':True,'observed_window':{'first':phases['first_clock'],'last':phases['last_clock']},'guarantee_beyond_observed_window':False,'native_observer_qualified':False,'native_PASS':False,'native_authority_issued':False,'tensor_authority':False,'native_fit_established':False,'scoring_authorized':False,'study_authority':False,'owned_process_identities':list(births.values()),'phase_counts':phases['counts'],'events':replay['events']}

def _CPU_audit_fixture(preserved_directory,*,original_directory,context,contract,_capability,expected_preservation_file_sha256,expected_preservation_body_sha256,fresh_death):
    contract._check_CPU_capability(_capability);value=contract.context_value(context)
    require(value.get('fixture_only') is True and value.get('native_PASS') is value.get('native_authority_issued') is False,'CPU observer audit has no native context authority')
    def revalidate():return contract._CPU_revalidate_context(context,_capability=_capability)
    return audit_terminal(preserved_directory,original_directory=original_directory,selected_context=value,
        expected_preservation_file_sha256=expected_preservation_file_sha256,expected_preservation_body_sha256=expected_preservation_body_sha256,revalidate=revalidate,fresh_death=fresh_death)

def audit_registered(preserved_directory,*,original_directory,context,contract,expected_preservation_file_sha256,expected_preservation_body_sha256,fresh_death):
    """Read-only integrated scope; supplied death interface needs outer review.

    This does not import/load the collector and does not self-certify its ABI.
    Current-runtime/live-interface qualification is external to recorded replay.
    """
    here=Path(__file__).resolve().parent
    package=next(p for p in here.parents if p.name=='rineya_synth_centralized_reasoning_v1')
    require(here==package/'observer_startup_measurement_v1','staged/preserved observer auditor has no native authority')
    contract.native_gate();value=contract.context_value(context)
    require(value.get('fixture_only') is False,'CPU observer context cannot enter registered audit')
    return audit_terminal(preserved_directory,original_directory=original_directory,selected_context=value,
        expected_preservation_file_sha256=expected_preservation_file_sha256,expected_preservation_body_sha256=expected_preservation_body_sha256,
        revalidate=lambda:contract.revalidate_context(context),fresh_death=fresh_death)
