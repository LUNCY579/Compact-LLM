"""Separate pre-core install/ownership failure preservation audit.

This route has no success result and never certifies an executing interpreter
whose birth is unavailable. Context and live-death suppliers are externally
bound obligations; CPU fixtures inject them explicitly. No native API imports.
"""
from pathlib import Path
from types import ModuleType
from datetime import datetime
import hashlib,json,math,stat,time

SCOPE='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
CORE={'PHASE_OBSERVATIONS.jsonl','ownership_verified.json','input_check_summary.json','worker_result.json','PHASE_ACCOUNTING_FINAL.json','OWNED_WORKER_RETURN.json'}
REQUIRED={'SUPERVISOR_RECEIPT.json','RESOURCE_OBSERVATIONS.json','supervisor_ownership.json','OWNED_EXTERNAL_ANCHORS.json','LIBRARY_BOUNDARY_EVIDENCE.json','worker.stdout.log','worker.stderr.log','supervisor_heartbeat.jsonl'}
LEASES={'reservation.json','worker.json','resume.json','terminal.json'}
GIB=1024**3
RESOURCES={'CPU_sample_max_age_seconds':.2,'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'GPU_probe_interval_seconds':.5,'GPU_probe_timeout_seconds':2.0,'GPU_sample_max_age_seconds':2.0,'admission_poll_seconds':2.0,'admission_seconds':30,'disk_free_GiB':4.0,'external_watchdog_milliseconds':100,'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'whole_scope_seconds':600,'worker_process_tree_RSS_GiB':2.0}
PERMISSION={'allocations':2,'dtype':'int64','shape':[1,422],'logical_bytes':6752,'model':False,'forward':False,'backward':False,'optimizer':False,'decoder':False,'scoring':False,'M0_membership':False}

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def same(left,right):return canonical(left)==canonical(right)
def fields(value,names,label):require(type(value) is dict and set(value)==set(names),'exact '+label+' schema')
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def integer(value):return type(value) is int and 0<=value<2**64
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
def pairs(rows):
    result={}
    for k,v in rows:require(k not in result,'duplicate JSON key');result[k]=v
    return result
def number(text):
    value=float(text);require(math.isfinite(value),'nonfinite/overflow JSON');return value
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_constant=lambda _:require(False,'nonfinite JSON'))
def seal(value,pin=None):
    require(type(value) is dict and sha(value.get('sha256')) and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}) and (pin is None or value['sha256']==pin),'external body checksum')
def birth(value):
    fields(value,{'pid','creation_filetime_100ns'},'owned birth');require(all(integer(n) and n>0 for n in value.values()),'typed positive FILETIME birth');return value

class ReadSet:
    def __init__(self):self.files={};self.rosters={};self.total=0;self.started=time.monotonic()
    def check(self):require(time.monotonic()-self.started<=600,'bounded failure-audit read deadline')
    def physical(self,p):
        p=Path(p);require(p.is_absolute() and p.resolve()==p,'literal physical selected path')
        for part in (p,*p.parents):require(not part.is_symlink() and not(getattr(part.lstat(),'st_file_attributes',0)&0x400),'no selected reparse aliases')
        return p
    def roster(self,p):
        p=self.physical(p);require(p.is_dir(),'selected directory');rows={x.name for x in p.iterdir()};require(len(rows)<=4096,'finite exact directory roster')
        key=str(p);require(key not in self.rosters or rows==self.rosters[key],'first roster cannot rebase');self.rosters.setdefault(key,rows);return rows
    def read(self,p,d,maximum=16*1024**2):
        self.check();fields(d,{'bytes','sha256'},'byte descriptor');require(integer(d['bytes']) and d['bytes']<=maximum and sha(d['sha256']),'bounded exact byte descriptor')
        p=self.physical(p);before=p.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size==d['bytes'],'selected regular exact file size')
        self.total+=before.st_size;require(self.total<=512*1024**2,'finite total failure audit reads')
        with p.open('rb') as stream:raw=stream.read(before.st_size+1)
        after=p.lstat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and hashlib.sha256(raw).hexdigest()==d['sha256'],'exact stable selected file bytes')
        key=str(p);require(key not in self.files or self.files[key]==d,'first byte anchor cannot rebase');self.files.setdefault(key,dict(d));return raw
    def verify(self):
        for p,rows in list(self.rosters.items()):require(self.roster(Path(p))==rows,'final whole roster conservation')
        for p,d in list(self.files.items()):self.read(Path(p),d)

def audit_failure(preserved_directory,*,original_directory,selected_context,expected_preservation_file_sha256,expected_preservation_body_sha256,revalidate,fresh_death):
    """Failure-only byte/record audit; callbacks never confer native authority."""
    require(callable(revalidate) and callable(fresh_death),'externally bound context/death interfaces')
    value=parse(canonical(selected_context));projection=canonical(value)
    require(same(revalidate(),value),'exact original external context before failure reads')
    config=value['config'];runtime=value['runtime'];sources=config['source_files'];package=Path(value['config_path']).parent.parent
    seal(config);require(config['sha256']==value['anchors']['expected_config_body'],'external selected config body')
    require(config['scope']==SCOPE and same(config['permission'],PERMISSION) and same(config['resources'],RESOURCES),'unchanged purpose/permission/resource gates')
    require(type(value.get('fixture_only',False)) is bool and value.get('native_PASS',False) is value.get('native_authority_issued',False) is False,'context cannot supply native authority')
    require(type(runtime.get('library_boundary_plan')) is list and runtime['library_boundary_plan'],'external selected boundary plan required')
    policy=runtime['library_search_policy']['enforced_boundary'];fixture=value.get('fixture_only',False)
    original=Path(original_directory).absolute();root=Path(preserved_directory).absolute();require(root!=original and original.parent==package/'artifacts' and root.parent==package/'evidence','separate exact owned original/preserved failure leaf')
    reader=ReadSet();pres_path=root/'PRESERVATION.json'
    preservation=parse(reader.read(pres_path,{'bytes':pres_path.lstat().st_size,'sha256':expected_preservation_file_sha256}));seal(preservation,expected_preservation_body_sha256)
    fields(preservation,{'kind','scope','manifest_sha256','raw_files','source_files','source_snapshot_files','lease_files','current_death_observation','fixture_only','native_authority_issued','sha256'},'failure preservation')
    require(preservation['kind']=='PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC' and preservation['scope']==SCOPE and preservation['manifest_sha256']==config['sha256'] and preservation['fixture_only'] is fixture and preservation['native_authority_issued'] is False and same(preservation['source_files'],sources),'original selected failure preservation/context source joins')
    names=set(preservation['raw_files']);require(REQUIRED<=names and not(names&CORE) and len(names)<=4096 and all(type(n) is str and Path(n).name==n and n not in ('.','..') and Path(n).suffix in ('.json','.jsonl','.log','.txt','.snapshot') for n in names),'pre-core failure exact flat population; core/container never fabricated')
    require(reader.roster(original)==names,'whole original raw failure roster')
    raw={}
    for n,d in preservation['raw_files'].items():
        raw[n]=reader.read(root/n,d);require(reader.read(original/n,d)==raw[n],'all original/preserved raw byte pairs')
    require(sum(len(x) for x in raw.values())<=128*1024**2,'bounded failure raw denominator')
    data={n:parse(b) for n,b in raw.items() if n.endswith('.json')}
    require(same(data['OWNED_EXTERNAL_ANCHORS.json'],{'config_path':value['config_path'],'anchors':value['anchors']}),'exact original external anchor file')
    snapshots=preservation['source_snapshot_files'];require(set(snapshots)==set(sources) and 0<len(sources)<=4096,'exact failure source snapshot population')
    copied=set()
    for text,d in sources.items():
        require(type(text) is str and d['role']=={'.py':'python_source','.json':'source_json','.md':'source_markdown'}.get(Path(text).suffix),'failure source roles only; no model/runtime payload reads')
        entry=snapshots[text];name=hashlib.sha256(text.encode('utf-8')).hexdigest()+Path(text).suffix
        fields(entry,{'path','bytes','sha256','role'},'source copy descriptor');require(entry['path']=='source_snapshots/'+name and same({k:entry[k] for k in ('bytes','sha256','role')},d),'exact original/source snapshot map')
        expected={k:d[k] for k in ('bytes','sha256')};require(reader.read(Path(text),expected)==reader.read(root/'source_snapshots'/name,expected),'original/source snapshot bytes');copied.add(name)
    require(reader.roster(root/'source_snapshots')==copied,'complete source copy roster')
    sup=data['SUPERVISOR_RECEIPT.json'];owner=data['supervisor_ownership.json'];owned=sup['worker_identity'];launcher=birth({k:owned[k] for k in ('pid','creation_filetime_100ns')})
    require(type(sup['returncode']) is int and sup['returncode']!=0 and sup['status']!='SUPERVISED_WORKER_EXITED_ZERO','pre-core nonzero supervisor failure required')
    require(all(sup[k] is True for k in ('worker_attempted','worker_launched','worker_resumed','preflight_revalidated_before_current_observations','job_closed_kill_on_close','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified','partial_outputs_frozen')) and sup['owned_worker_identity_currently_alive'] is False and sup['unjoined_sampler_threads']==[],'launched terminal containment/sampler/telemetry cleanup required')
    require(sup['output_dir']==str(original) and sup['config_sha256']==value['anchors']['expected_config_file'] and sup['preflight']['source_commit']==value['expected_head'] and same(sup['preflight']['resources'],RESOURCES),'original supervisor/output/config/source/resource joins')
    require(finite(sup['elapsed_seconds']) and sup['elapsed_seconds']<=600 and finite(sup['worker_resume_monotonic']) and finite(sup['deadline_monotonic']) and 0<=sup['deadline_monotonic']-sup['worker_resume_monotonic']<=600,'unchanged whole scope and owned clock window')
    require(owned['assigned_before_resume'] is owned['job_kill_on_close'] is True and owned['breakaway_allowed'] is False and owner['worker_pid']==launcher['pid'] and owner['root_worker_pid']==launcher['pid'] and owner['job_name']==owned['job_name'] and owner['config_sha256']==sup['config_sha256'],'exact assigned launcher receipt; no interpreter assumption')
    partial=sup['partial_outputs'];require(type(partial) is list and len(partial)==len(names)-1,'exact supervisor partial denominator')
    pmap={}
    for row in partial:
        fields(row,{'path','bytes','sha256'},'partial raw descriptor');p=Path(row['path']);require(type(row['path']) is str and row['path'] in (p.name,str(original/p.name)) and p.name not in pmap,'exact unique partial raw leaves');pmap[p.name]={k:row[k] for k in ('bytes','sha256')}
    require(pmap=={n:d for n,d in preservation['raw_files'].items() if n!='SUPERVISOR_RECEIPT.json'},'partial map exact raw bytes')
    leases=preservation['lease_files'];require(set(leases)==LEASES,'four original lease files required')
    lease_root=Path(sup['lease_path']);require(lease_root.parent==package/'artifacts/_native_gpu_leases' and reader.roster(lease_root)==LEASES,'exact owned lease path/roster')
    lease_data={};lease_copies=set()
    for n,d in leases.items():
        fields(d,{'source_path','path','bytes','sha256'},'lease source/copy');require(d['source_path']==str(lease_root/n) and d['path']=='lease_'+n,'exact lease source/copy pointers')
        b=reader.read(root/d['path'],{k:d[k] for k in ('bytes','sha256')});require(reader.read(lease_root/n,{k:d[k] for k in ('bytes','sha256')})==b,'exact original/lease copy bytes');lease_data[n]=parse(b);lease_copies.add(d['path'])
        if n=='terminal.json':require(b==raw['SUPERVISOR_RECEIPT.json'],'lease terminal exact SUP bytes')
    require(same(lease_data['worker.json'],owned) and same(lease_data['resume.json']['worker_identity'],owned) and lease_data['resume.json']['external_monitor_ready_before_resume'] is True and lease_data['resume.json']['resumed_monotonic']==sup['worker_resume_monotonic'] and lease_data['reservation.json']['config_sha256']==sup['config_sha256'],'original worker/resume/reservation joins')
    births=_resources(sup,data['RESOURCE_OBSERVATIONS.json'],launcher)
    death=preservation['current_death_observation'];fields(death,{'observed_monotonic','observed_utc','supervisor_file_sha256','config_file_sha256','observations','death_verifier_source_sha256'},'all-birth current death proof')
    require(finite(death['observed_monotonic']) and death['observed_monotonic']>=sup['worker_resume_monotonic'] and type(death['observed_utc']) is str and len(death['observed_utc'])<=64 and datetime.fromisoformat(death['observed_utc']).utcoffset() is not None,'finite timezone-qualified terminal death evidence')
    require(death['supervisor_file_sha256']==preservation['raw_files']['SUPERVISOR_RECEIPT.json']['sha256'] and death['config_file_sha256']==sup['config_sha256'] and death['death_verifier_source_sha256']==config['helper_pins']['native_supervisor.py'],'selected death verifier/SUP/config pins')
    actual={}
    for row in death['observations']:
        fields(row,{'identity','currently_alive'},'current dead birth');b=birth(row['identity']);key=(b['pid'],b['creation_filetime_100ns']);require(key not in actual and row['currently_alive'] is False,'unique preserved dead births');actual[key]=b
    require(actual==births,'complete all-birth current terminal proof')
    for b in births.values():require(fresh_death(dict(b)) is False,'fresh externally bound same FILETIME birth must be dead')
    evidence=_library(raw['LIBRARY_BOUNDARY_EVIDENCE.json'],value,owner,reader)
    low_n,low_d=math.nextafter(float(sup['worker_resume_monotonic']),-math.inf).as_integer_ratio()
    high_n,high_d=math.nextafter(float(sup['deadline_monotonic']),math.inf).as_integer_ratio()
    low=(low_n*10**9)//low_d;high=-((-high_n*10**9)//high_d)
    require(all(low<=clock<=high for clock in evidence['checkpoint_clocks_ns']),'known pre-core raw clocks inside original owned window')
    require(reader.roster(root)==names|lease_copies|{'source_snapshots','PRESERVATION.json'},'whole preserved failure roster')
    reader.verify();require(canonical(selected_context)==projection and same(revalidate(),value),'same original external context after failure reads')
    return {'kind':'INDEPENDENT_PRECORE_LIBRARY_FAILURE_BYTE_AUDIT','status':'INCOMPLETE_FAILURE_PRESERVED','fixture_only':fixture,'native_PASS':False,'native_authority_issued':False,'completion_established':False,'native_fit_established':False,'scoring_authorized':False,'M0_membership':False,'executing_birth':None,'executing_birth_certified':False,'failure_byte_closure_verified':True,'owned_process_identities':list(births.values()),'library_evidence':evidence,'current_native_runtime_or_observer_qualified':False}

def _resources(sup,samples,launcher):
    admissions=sup['admission_observations'];dispatch=sup['dispatch_observation'];require(type(admissions) is list and len(admissions)==2,'two original admission samples')
    uuid=None
    for row in admissions+[dispatch]:
        require(all(integer(row[k]) for k in ('host_available_bytes','worker_tree_rss_bytes','disk_free_bytes','gpu_free_bytes','gpu_total_bytes')),'typed admission bytes')
        require(row['host_available_bytes']>=3.3125*GIB and row['worker_tree_rss_bytes']<=2*GIB and row['disk_free_bytes']>=4*GIB and 4*GIB<=row['gpu_free_bytes']<=row['gpu_total_bytes'],'unchanged admission floors')
        require(all(finite(row[k]) for k in ('sample_started_monotonic','sample_completed_monotonic','gpu_sample_started_monotonic','gpu_sample_completed_monotonic')) and row['sample_started_monotonic']<=row['sample_completed_monotonic'] and row['gpu_sample_started_monotonic']<=row['gpu_sample_completed_monotonic'],'finite ordered admission clocks')
        require(type(row['GPU_index']) is int and row['GPU_index']==0 and type(row['gpu_uuid']) is str and row['gpu_uuid'] and (uuid is None or row['gpu_uuid']==uuid),'same GPU0 identity');uuid=row['gpu_uuid']
    require(admissions[1]['sample_started_monotonic']-admissions[0]['sample_started_monotonic']>=2 and admissions[1]['gpu_observation_id']!=admissions[0]['gpu_observation_id'] and admissions[1]['gpu_sample_started_monotonic']>admissions[0]['gpu_sample_started_monotonic'] and dispatch['sample_started_monotonic']-admissions[0]['sample_started_monotonic']<=30,'distinct two-second admissions within30 seconds')
    births={(launcher['pid'],launcher['creation_filetime_100ns']):launcher};clocks={'CPU':[],'GPU':[]}
    require(type(samples) is list and 0<len(samples)<=32768,'bounded terminal resource population')
    for row in samples:
        domain=row.get('kind');require(domain in clocks,'known resource domain');clock=row['sample_started_monotonic'] if domain=='CPU' else row['gpu_sample_started_monotonic']
        require(finite(clock) and (not clocks[domain] or clock>=clocks[domain][-1]),'chronological resource samples');clocks[domain].append(clock)
        if domain=='CPU':
            require(all(integer(row[k]) for k in ('host_available_bytes','worker_tree_rss_bytes')) and row['host_available_bytes']>=1.25*GIB and row['worker_tree_rss_bytes']<=2*GIB,'unchanged during-work CPU floors')
            require(type(row['owned_process_identities']) is list and len(row['owned_process_identities'])<=256,'finite complete recorded birth roster')
            for found in row['owned_process_identities']:
                b=birth({k:found[k] for k in ('pid','creation_filetime_100ns')});births[(b['pid'],b['creation_filetime_100ns'])]=b
        else:require(integer(row['gpu_free_bytes']) and integer(row['gpu_total_bytes']) and .5*GIB<=row['gpu_free_bytes']<=row['gpu_total_bytes'] and row['gpu_uuid']==uuid,'unchanged same-GPU work floor')
    end=max((t for ts in clocks.values() for t in ts),default=sup['worker_resume_monotonic'])
    require(finite(end) and sup['worker_resume_monotonic']<=end<=sup['deadline_monotonic'],'recorded resource interval inside deadline; no death time inferred')
    for domain,age in (('CPU',.2),('GPU',2.0)):
        rows=clocks[domain];require(rows and rows[0]<=sup['worker_resume_monotonic']+age and rows[-1]>=end-age and all(b-a<=age for a,b in zip(rows,rows[1:])),'resource cadence/first/final edges')
    return births

def _library(raw,value,owner,reader):
    envelope=parse(raw);seal(envelope)
    fields(envelope,{'schema_version','kind','source_commit','original_scope','context_sha256','external_anchors','policy_sha256','provider_source','owner_receipt_sha256','raw_packet','raw_packet_sha256','main_errors','fixture_only','native_PASS','native_authority_issued','sha256'},'pre-core library envelope')
    policy=value['runtime']['library_search_policy']['enforced_boundary'];fixture=value.get('fixture_only',False)
    require(type(envelope['schema_version']) is int and envelope['schema_version']==1 and envelope['kind']=='OWNED_LIBRARY_BOUNDARY_EVIDENCE_V1' and envelope['fixture_only'] is fixture and envelope['native_PASS'] is envelope['native_authority_issued'] is False,'actual library envelope/no authority')
    require(envelope['source_commit']==value['expected_head'] and envelope['original_scope']==SCOPE and envelope['context_sha256']==digest(value) and same(envelope['external_anchors'],value['anchors']) and envelope['policy_sha256']==digest(policy) and same(envelope['provider_source'],policy['provider_source']) and envelope['owner_receipt_sha256']==digest(owner),'external pre-core context/policy/owner recipe joins')
    errors=envelope['main_errors'];require(type(errors) is list and 0<len(errors)<=32,'original pre-core main failure required')
    for row in errors:
        fields(row,{'stage','error'},'pre-core main failure');require(row['stage'] in {'install','ownership','cleanup','export'},'only pre-core install/ownership and terminal secondary stages')
    require(errors[0]['stage'] in {'install','ownership'},'specific first pre-core failure stage')
    packet=envelope['raw_packet'];require(type(packet) is dict and envelope['raw_packet_sha256']==digest(packet),'actual embedded packet value digest')
    package=Path(value['config_path']).parent.parent;source_files=value['config']['source_files'];path=package/'prefix_library_evidence.py';d=source_files[str(path)]
    require(d['role']=='python_source','selected independent replay source role');source=reader.read(path,{k:d[k] for k in ('bytes','sha256')})
    module=ModuleType('selected_precore_library_replay');module.__file__=str(path);exec(compile(source,str(path),'exec'),module.__dict__)
    worker=str(package/'prefix_input_owned_worker.py');identity={'source_commit':value['expected_head'],'original_scope':SCOPE,'owned_birth':None,'owned_birth_status':'UNAVAILABLE','owner_receipt_sha256':digest(owner),'context_sha256':digest(value),'producer_source':{'path':worker,'descriptor':source_files[worker]},'fixture_only':fixture,'native_PASS':False,'native_authority_issued':False}
    b=(canonical(packet)+'\n').encode('ascii');replay=module.validate_observed_packet(b,policy=policy,expected_identity=identity,selected_file_map=policy['library_files'],selected_source_map=source_files,expected_boundaries=value['runtime']['library_boundary_plan'],expected_file_sha256=hashlib.sha256(b).hexdigest())
    require(replay['native_PASS'] is replay['native_authority_issued'] is False and replay['conservation_complete'] is False and replay['reached_counts']['entered']==0,'pre-core raw prefix cannot claim execution or complete conservation')
    for row in errors:module.observed_error(row['error'])
    return {'replay':replay,'main_errors':errors,'checkpoint_clocks_ns':[row['time_ns'] for row in packet['observations'] if type(row['time_ns']) is int],'actual_packet_bytes_consumed':True,'native_observer_qualified':False,'executing_birth':None}
