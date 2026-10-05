from pathlib import Path
from datetime import datetime,timezone
import collections,json,math,sys,hashlib,ast,inspect,importlib.metadata
root=Path.cwd();package=root/'experiments/rineya_synth_centralized_reasoning_v1';sys.path.insert(0,str(package))
import context_phase_contract as c
from rs_contract import digest,canonical
leaf=package/'artifacts/context_phase_no_model_D001_attempt1'
saved=package/'evidence/context_phase_no_model_D001_attempt1_20261005'
config_path=package/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json'
target=package/'evidence/context_phase_no_model_D001_attempt1_independent_audit_20261005.json'
checks=0
def check(value,message):
 global checks
 checks+=1
 if not value:raise ValueError(message)
def read(path):return c.native.read_json(path,16*1024**2)
def info(path):
 path=Path(path);check(path.is_file() and not path.is_symlink(),'regular immutable file')
 return {'bytes':path.stat().st_size,'sha256':c.native.filehash(path)}
check(not target.exists(),'new write-once independent diagnostic audit')
scope=read(config_path);pres=read(saved/'preservation_receipt.json');sup=read(leaf/'SUPERVISOR_RECEIPT.json');worker=read(leaf/'worker_result.json');lease=Path(sup['lease_path'])
check(info(config_path)['sha256']=='53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030'
 and scope['sha256']==digest({k:v for k,v in scope.items() if k!='sha256'})=='944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62','trusted config byte/body anchors')
check(pres['sha256']==digest({k:v for k,v in pres.items() if k!='sha256'})=='a1b376c42fae68301230e0453c84d87911536b5fa39be9d9c746f895d7d0eb60','trusted preservation checksum')
check(info(leaf/'SUPERVISOR_RECEIPT.json')['sha256']=='a348bb43b5de0ebf592885e404043000608da928d9a7725a2b34cd2cb812b7fc'
 and info(leaf/'worker_result.json')['sha256']=='a57474b27a5e70e7cc46c719aeb1cdd86b9a3d324e8a5b285ba89b883e590a8d','trusted raw worker/supervisor bytes')
c.validate_manifest(scope)
check(len(scope['source_pins'])==107 and set(scope['source_pins'])==c.source_roster()
 and len(scope['runtime_pins'])==65 and len(scope['helper_pins'])==7,'exact own/transitive/source107 runtime65 helpers7 inventory')
check(pres['source_commit']==worker['source_commit']==sup['preflight']['source_commit']=='7e40a01538335d45442b9a63df371fdbeffde535','fixed launch source barrier')
check(pres['raw_directory']==sup['output_dir']==str(leaf) and pres['preserved_directory']==str(saved),'original/preserved owned leaf paths')
check(set(x.name for x in leaf.iterdir())==set(pres['raw_files']) and len(pres['raw_files'])==297
 and all(x.is_file() and not x.is_symlink() for x in leaf.iterdir()),'exact297 frozen raw files only')
def maps():
 return {'original_raw':{name:info(leaf/name) for name in pres['raw_files']},
 'preserved_raw':{name:info(saved/'raw'/name) for name in pres['raw_files']},
 'source':{name:info(Path(name)) for name in scope['source_pins']},
 'runtime':{name:info(Path(name)) for name in scope['runtime_pins']},
 'helper':{name:info(package/name) for name in scope['helper_pins']},
 'source_snapshots':{name:info(saved/'source_snapshots'/name) for name in pres['source_snapshots']},
 'config':info(config_path),'config_snapshot':info(saved/'config_snapshot.json'),
 'preservation_receipt':info(saved/'preservation_receipt.json'),
 'lease':{name:info(lease/name) for name in ('reservation.json','worker.json','resume.json','terminal.json')}}
before=maps()
check(before['original_raw']==before['preserved_raw']==pres['raw_files'],'every297 original/preserved raw byte')
check(before['config']==before['config_snapshot']==pres['config_snapshot'],'config snapshot exact')
check(before['source_snapshots']==pres['source_snapshots'] and len(before['source_snapshots'])==107,'every107 source snapshot byte')
for key,pins in [('source',scope['source_pins']),('runtime',scope['runtime_pins']),('helper',scope['helper_pins'])]:
 for name,item in before[key].items():check(item['sha256']==pins[name],'every current '+key+' bytepin')
for name,item in before['source'].items():check(before['source_snapshots'][Path(name).relative_to(root).as_posix()]==item,'source snapshot joins executing source')
for name,pin in scope['helper_pins'].items():check(scope['source_pins'][str((package/name).resolve())]==pin,'helper source cohesion')
# Enumerate exact primary runtime files without loading framework or native libraries.
tree=ast.parse(inspect.getsource(c.native.runtime_files))
names=next(ast.literal_eval(node.value) for node in tree.body[0].body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in node.targets))
actual=set()
for name,relative in names.items():
 distribution=importlib.metadata.distribution(name);check(distribution.version==scope['runtime_versions'][name],'actual framework distribution version')
 pkg=Path(distribution.locate_file(name))
 actual.update(str(Path(distribution.locate_file(path)).resolve()) for path in relative)
 actual.update(str(path.resolve()) for path in pkg.glob('*.pyd'))
 if name=='torch':actual.update(str(path.resolve()) for path in (pkg/'lib').glob('*.dll'))
actual.add(str(Path(sys.executable).resolve()))
remaining=set(scope['runtime_pins'])-actual
check(len(actual)==62 and actual<=set(scope['runtime_pins']) and {Path(path).name.lower() for path in remaining}=={'ast.py','nvml.dll','kernel32.dll'},'complete current62 primary runtime plus3 fixed files')
import sysconfig
ast_path=str((Path(sysconfig.get_path('stdlib'))/'ast.py').resolve());prior=read(package/'configs/SYNTH_NQ001_PROC7.json')
check({k:v for k,v in scope['runtime_pins'].items() if k!=ast_path}==prior['runtime_pins'],'unchanged qualified64 plus executedAST')
check(sys.version.split()[0]==scope['runtime_versions']['python'],'actual interpreter version')
check(sup['preflight']=={**scope,'worker_path':str(package/'context_phase_worker.py'),'config_path':str(config_path.resolve()),
 'config_sha256':before['config']['sha256'],'source_commit':pres['source_commit'],
 'bindings':{'source':scope['source_pins'],'runtime':scope['runtime_pins'],'ENV':scope['offline_environment'],'helpers':scope['helper_pins']}},'entire frozen actual preflight')
check(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode']==0 and worker['status']=='NO_MODEL_CONTEXT_PHASES_COMPLETE_UNQUALIFIED','only complete owned diagnostic')
check(all(sup[key] is True for key in ('worker_attempted','worker_launched','worker_resumed','owned_handle_alive_verified_after_launch',
 'preflight_revalidated_before_current_observations','job_closed_kill_on_close','partial_outputs_frozen','sampler_threads_terminal_verified','gpu_telemetry_cleanup_verified'))
 and sup['owned_worker_identity_currently_alive'] is False and sup['unjoined_sampler_threads']==[],'all owned launch and terminal flags')
check(sup['mechanical_qualification_established'] is False and sup['production_training_authorized'] is False and sup['protected_bank_access'] is False,'no qualification/training/bank grant')
check(sup['short_RAM_monitor_diagnostic']=={'attempted':False,'persisted':False,'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','write_error':None,'original_abort_reason_preserved':None},'success no contradictory abort')
check(0<worker['wall_seconds']<=sup['elapsed_seconds']<600 and sup['elapsed_seconds']==32.57800000000134,'worker/whole600 unchanged')
check(worker['config_sha256']==sup['config_sha256']==before['config']['sha256'],'worker/SUP/config bytes join')
check(all(worker[key] is False for key in ('model_loaded','model_constructed','actual_grading','usable_model_fit_established','historical_phase_or_cause_established',
 'historical_cause_established','native_fit_established','scoring_authorized','forward_authorized'))
 and all(worker[key]==0 and type(worker[key]) is int for key in ('forwards','backwards','optimizer_steps')),'no model/math/scoring/cause claims')
check(all(pres[key] is False for key in ('model_loaded','model_constructed','actual_grading','usable_model_fit_established','historical_phase_or_cause_established'))
 and all(pres[key]==0 for key in ('forwards','backwards','optimizer_steps')),'preserved scope no claim expansion')
check(not set(scope)&c.FORBIDDEN_FIELDS and scope['authorization']==c.AUTHORIZATION,'no model/row/loader/cache/device-reset authority fields')
check({item['path']:{k:item[k] for k in ('bytes','sha256')} for item in sup['partial_outputs']}==
 {name:item for name,item in before['original_raw'].items() if name!='SUPERVISOR_RECEIPT.json'},'exact frozen supervisor raw inventory')
start_fields={'ordinal','operation','event','monotonic','utc','historical_cause_established'}
end_fields=start_fields|{'error','result'}
pairs={};counts=collections.Counter()
for ordinal in range(1,146):
 begin=read(leaf/('phase_%04d_start.json'%ordinal));end=read(leaf/('phase_%04d_end.json'%ordinal))
 check(set(begin)==start_fields and set(end)==end_fields and begin['ordinal']==end['ordinal']==ordinal
  and type(begin['ordinal']) is int and begin['operation']==end['operation'] and begin['event']=='attempt' and end['event']=='returned'
  and end['error'] is None and begin['historical_cause_established'] is end['historical_cause_established'] is False,'closed same145 phase start/return/error pair')
 check(type(begin['monotonic']) in (int,float) and type(end['monotonic']) in (int,float)
  and math.isfinite(begin['monotonic']) and math.isfinite(end['monotonic']) and begin['monotonic']<=end['monotonic'],'finite phase monotonic chronology')
 bdt=datetime.fromisoformat(begin['utc']);edt=datetime.fromisoformat(end['utc'])
 check(bdt.tzinfo is not None and edt.tzinfo is not None and bdt<=edt,'aware UTC phase chronology')
 pairs[ordinal]=(begin,end);counts[begin['operation']]+=1
 if ordinal>1:check(pairs[ordinal-1][0]['monotonic']<=begin['monotonic'] and datetime.fromisoformat(pairs[ordinal-1][0]['utc'])<=bdt,'contiguous attempt chronology')
expected_files={('phase_%04d_%s.json'%(ordinal,suffix)) for ordinal in range(1,146) for suffix in ('start','end')}
check({name for name in pres['raw_files'] if name.startswith('phase_')}==expected_files,'exact290 complete phase files, none empty/pending/unadvertised')
account=worker['phase_accounting'];expected_counts={name:{'attempted':n,'returned':n,'failed':0,'pending':0} for name,n in counts.items()}
check(account['operations']==expected_counts and account['last_ordinal']==145 and account['publication_errors']==[],'all145 returned phase ledger conservation')
main_names=['torch_import','Qwen3Config_symbol_import','Qwen3ForCausalLM_symbol_import','framework_version_read','post_import_physical_growth_observation',
 'cuda_context_init','direct_CUDA_preload_mem_get_info','direct_CUDA_memory_allocated','direct_CUDA_memory_reserved','direct_CUDA_max_memory_allocated',
 'direct_CUDA_max_memory_reserved','precision_and_thread_configuration','Python_seed_101','Torch_CPU_seed_101','Torch_CUDA_seed_all_101',
 'CUDA_reset_peak_memory_stats','final_direct_CUDA_mem_get_info','final_direct_CUDA_memory_allocated','final_direct_CUDA_memory_reserved',
 'final_direct_CUDA_max_memory_allocated','final_direct_CUDA_max_memory_reserved','optional_Windows_PID_counters']
cursor=1;loaded=False;initialized=False;main=[];guard_rows=[];query_rows=[]
def consume(name):
 global cursor
 begin,end=pairs[cursor];check(begin['operation']==name,'exact source-defined phase order '+name);cursor+=1
 return begin,end
def guard(name):
 parent=consume('guard_'+name);children=[]
 if loaded:
  init=consume('guard_cuda_is_initialized');check(type(init[1]['result']) is bool and init[1]['result'] is initialized,'actual initialization status exact expected false/true')
  children.append(init);query_rows.append(init)
  if init[1]['result']:
   memory=consume('guard_cuda_mem_get_info');value=memory[1]['result']
   check(type(value) is list and len(value)==2 and all(type(v) is int and v>=0 for v in value) and value[0]<=value[1] and value[0]>=.5*1024**3,'original halfGiB guard query exact typed returned values')
   children.append(memory);query_rows.append(memory)
 check(parent[1]['result'] is None,'original guard returnedNone')
 for child in children:check(parent[0]['monotonic']<=child[0]['monotonic']<=child[1]['monotonic']<=parent[1]['monotonic'],'nonrecursive guard/query nested envelope')
 guard_rows.append(parent);return parent
for name in main_names:
 before_guard=guard('before_'+name);phase=consume(name)
 check(before_guard[1]['monotonic']<=phase[0]['monotonic'],'guard completed before actual registered phase')
 if name=='torch_import':loaded=True
 if name=='cuda_context_init':initialized=True
 after_guard=guard('after_'+name)
 check(phase[1]['monotonic']<=after_guard[0]['monotonic'],'returned actual phase precedes after guard')
 main.append(phase)
completed=guard('complete')
check(cursor==146 and account['last_phase']==completed[1] and completed[0]['ordinal']==143,'last outer guard result differs from latest issued nested ordinal, exact source semantics')
check(counts['guard_cuda_is_initialized']==44 and counts['guard_cuda_mem_get_info']==34 and len(guard_rows)==45 and len(main)==22,'exact22 phase+45guard+44init+34memory=145 denominator')
by_name={pair[0]['operation']:pair for pair in main}
check(by_name['torch_import'][1]['result']=={'opaque_return_type':'builtins.module'}
 and all(by_name[name][1]['result']=={'opaque_return_type':'builtins.type'} for name in ('Qwen3Config_symbol_import','Qwen3ForCausalLM_symbol_import')),'lazy module/class symbols only, not instantiated model')
check(by_name['framework_version_read'][1]['result']=={name:scope['runtime_versions'][name] for name in ('torch','transformers')},'actual imported runtime versions')
growth=by_name['post_import_physical_growth_observation'][1]['result']
check(set(growth)=={'available_bytes','worker_RSS_bytes'} and all(type(v) is int and v>=0 for v in growth.values())
 and growth['worker_RSS_bytes']<=2*1024**3 and growth['available_bytes']>=1.25*1024**3+max(0,2*1024**3-growth['worker_RSS_bytes']),'original postimport full growth reserve')
direct=[]
for name in ('direct_CUDA_preload_mem_get_info','final_direct_CUDA_mem_get_info'):
 value=by_name[name][1]['result'];check(type(value) is list and len(value)==2 and all(type(v) is int and v>=0 for v in value)
  and value[0]<=value[1] and value[0]>= (4 if name=='direct_CUDA_preload_mem_get_info' else .5)*1024**3,'direct typed GPU values/floors')
 direct.append(value)
allocator={name:pair[1]['result'] for name,pair in by_name.items() if name.endswith(('memory_allocated','memory_reserved'))}
check(len(allocator)==8 and all(type(value) is int and value==0 for value in allocator.values()),'eight actual current-process Torch allocator observations0; not total/perapp GPU memory')
check(by_name['Torch_CPU_seed_101'][1]['result']=={'opaque_return_type':'torch._C.Generator'}
 and all(by_name[name][1]['result'] is None for name in ('cuda_context_init','precision_and_thread_configuration','Python_seed_101','Torch_CUDA_seed_all_101','CUDA_reset_peak_memory_stats')),'recorded init/config/seed/processcounter operations exact')
check(by_name['optional_Windows_PID_counters'][1]['result']=={'supported':False,'reason':'Optional counter provider not registered in this scope','authoritative':False},'unsupported counters explicit no zero-use/authority inference')
resources=read(leaf/'RESOURCE_OBSERVATIONS.json');cpu=[row for row in resources if row['kind']=='CPU'];gpu=[row for row in resources if row['kind']=='GPU']
check(len(resources)==282 and len(cpu)==236 and len(gpu)==46,'complete returned CPU/GPU observation counts')
check(sup['bounds']=={'GPU_free_at_admission_GiB':4.0,'GPU_free_during_work_GiB':.5,'disk_free_GiB':4.0,'external_watchdog_milliseconds':100,
 'preworker_available_GiB':3.3125,'system_available_reserve_GiB':1.25,'whole_scope_seconds':600,'worker_process_tree_RSS_GiB':2.0}
 and sup['cadence']=={'CPU_freshness_seconds':.2,'CPU_watchdog_seconds':.1,'GPU_freshness_seconds':2.,'GPU_query_interval_seconds':.5,'GPU_query_timeout_seconds':2.,
 'admission_deadline_seconds':30.,'admission_interval_seconds':2.,'admission_samples':2},'unchanged exact amounts/cadence/deadline')
ad=sup['admission_observations'];check(len(ad)==2,'two successful fresh admissions')
uuid=ad[0]['gpu_uuid']
for row in ad+[sup['dispatch_observation']]:
 check(row['gpu_free_bytes']>=4*1024**3 and row['host_available_bytes']>=3.3125*1024**3 and row['disk_free_bytes']>=4*1024**3
  and row['worker_tree_rss_bytes']==0 and row['owned_process_identities']==row['owned_process_samples']==[],'preworker amount floors')
 now=max(row['sample_completed_monotonic'],row['gpu_sample_completed_monotonic'])
 check(0<=now-row['sample_started_monotonic']<=.2 and row['sample_started_monotonic']<=row['sample_completed_monotonic']<=now
  and 0<=now-row['gpu_sample_started_monotonic']<=2 and row['gpu_sample_started_monotonic']<=row['gpu_sample_completed_monotonic']<=now,'fresh admission chronology')
check(ad[1]['sample_started_monotonic']-ad[0]['sample_started_monotonic']>=2 and sup['dispatch_observation']['gpu_observation_id']==ad[-1]['gpu_observation_id'],'admission interval/same fresh dispatch')
owner=read(leaf/'supervisor_ownership.json');identity=sup['worker_identity'];resumed=read(lease/'resume.json');reservation=read(lease/'reservation.json')
check(owner['worker_pid']==owner['root_worker_pid']==identity['pid'] and owner['job_name']==identity['job_name'] and owner['config_sha256']==sup['config_sha256'],'exact ownership/config/Job identity')
check(identity['assigned_before_resume'] is True and identity['job_kill_on_close'] is True and identity['breakaway_allowed'] is False,'same owned containment parameters')
check(read(lease/'worker.json')==identity==resumed['worker_identity'] and resumed['external_monitor_ready_before_resume'] is True
 and resumed['resumed_monotonic']==sup['worker_resume_monotonic'] and resumed['deadline_monotonic']==sup['deadline_monotonic'],'assigned/ready before resume lease')
check(cpu[0]['phase']=='worker_suspended_startup' and cpu[0]['sample_completed_monotonic']<sup['worker_resume_monotonic'],'external initial monitor before worker execution')
check(reservation['config_sha256']==sup['config_sha256'] and reservation['output_dir']==str(leaf) and reservation['lease_id']==lease.name and reservation['GPU_index']==0
 and reservation['supervisor']=={'pid':owner['supervisor_pid'],'create_time':owner['supervisor_create_time']},'durable reservation identity')
check(before['lease']['terminal.json']==before['original_raw']['SUPERVISOR_RECEIPT.json'],'persistent lease terminal equals supervisor bytes')
seen={}
for row in cpu:
 check(all(type(row[key]) is int and row[key]>=0 for key in ('host_available_bytes','worker_tree_rss_bytes','disk_free_bytes'))
  and row['host_available_bytes']>=1.25*1024**3 and row['worker_tree_rss_bytes']<=2*1024**3 and row['disk_free_bytes']>=4*1024**3,'all observed CPU/RSS/disk floors')
 check(row['sample_started_monotonic']<=row['sample_completed_monotonic'] and row['sample_completed_monotonic']-row['sample_started_monotonic']<=.2,'fresh returned CPU service envelope')
 keys=('pid','create_time','creation_filetime_100ns')
 check([{key:item[key] for key in keys} for item in row['owned_process_samples']]==row['owned_process_identities']
  and row['worker_tree_rss_bytes']==sum(item['rss_bytes'] for item in row['owned_process_samples']),'exact sampled owned PID/createTime/FILETIME/RSS sum')
 for item in row['owned_process_samples']:
  check(row['sample_started_monotonic']<=item['sample_started_monotonic']<=item['sample_completed_monotonic']<=row['sample_completed_monotonic']
   and type(item['rss_bytes']) is int and item['rss_bytes']>=0,'individual process measurement envelope')
  previous=seen.setdefault(item['pid'],item['create_time']);check(previous==item['create_time'],'same live identity not reused')
 check(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in row['phase_latency_seconds'].values()),'finite phase service observations')
for row in gpu:
 check(row['GPU_index']==0 and row['gpu_uuid']==uuid and type(row['gpu_free_bytes']) is int and type(row['gpu_total_bytes']) is int
  and .5*1024**3<=row['gpu_free_bytes']<=row['gpu_total_bytes'] and bool(row['gpu_observation_id']),'all sampled GPU floors/identity')
 check(0<=row['gpu_sample_completed_monotonic']-row['gpu_sample_started_monotonic']<=2,'fresh returned GPU service envelope')
for rows,beg,end,maxage in ((cpu,'sample_started_monotonic','sample_completed_monotonic',.2),(gpu,'gpu_sample_started_monotonic','gpu_sample_completed_monotonic',2.)):
 for a,b in zip(rows,rows[1:]):check(0<=b[beg]-a[beg] and b[end]-a[beg]<=maxage,'cadence/publication gaps within registered freshness')
hb=[json.loads(line) for line in (leaf/'supervisor_heartbeat.jsonl').read_text().splitlines()]
check(len(hb)==236 and all(row['supervisor_pid']==owner['supervisor_pid'] for row in hb)
 and all(0<=b['unix_seconds']-a['unix_seconds']<=.2 for a,b in zip(hb,hb[1:])),'heartbeat ownership/chronology/freshness')
scope_start=sup['deadline_monotonic']-600;scope_finish=scope_start+sup['elapsed_seconds']
check(scope_start<=sup['worker_resume_monotonic']<=pairs[1][0]['monotonic']<=completed[1]['monotonic']<=scope_finish,'phase/resume/wholedeadline chronology')
# Read-only process identity checks; no process controls.
import psutil
seen[owner['supervisor_pid']]=owner['supervisor_create_time'];live={}
for pid,created in seen.items():
 try:live[str(pid)]=abs(psutil.Process(pid).create_time()-created)<.001
 except psutil.NoSuchProcess:live[str(pid)]=False
check(not any(live.values()),'all actually observed owned/supervisor identities independently terminal')
check(not any(name in sys.modules for name in ('torch','transformers','tokenizers')),'audit no framework/model/GPU execution')
after=maps();check(before==after,'all raw/preserved/source107/runtime65/helper7/config/snapshot/lease bytes unchanged')
def compact(values):return {key:({'entries':len(value),'manifest_sha256':digest(value)} if key in (
 'original_raw','preserved_raw','source','runtime','helper','source_snapshots','lease') else value) for key,value in values.items()}
phase_summary=[{'ordinal':begin['ordinal'],'operation':begin['operation'],'start_monotonic':begin['monotonic'],'end_monotonic':end['monotonic'],
 'start_utc':begin['utc'],'end_utc':end['utc'],'recorded_envelope_seconds':end['monotonic']-begin['monotonic'],'result':end['result']} for begin,end in main]
def timing(rows,begin,end):return {'maximum_returned_envelope_seconds':max(row[end]-row[begin] for row in rows),
 'maximum_start_gap_seconds':max(b[begin]-a[begin] for a,b in zip(rows,rows[1:])),
 'maximum_preceding_sample_age_at_next_publication_seconds':max(b[end]-a[begin] for a,b in zip(rows,rows[1:]))}
audit={'schema_version':1,'id':'SYNTH-CONTEXT-PHASE-NO-MODEL-D001-INDEPENDENT-AUDIT','status':'PASS_SCOPED_CURRENT_NO_MODEL_CONTEXT_PHASE_AND_GUARD_OBSERVATION_ONLY',
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':checks,'source_commit':pres['source_commit'],'attempt_path':str(leaf),'preserved_path':str(saved),
 'config_path':str(config_path),'config_file_sha256':before['config']['sha256'],'config_body_sha256':scope['sha256'],'purpose':c.PURPOSE,
 'source_runtime_helper_counts':{'source':107,'runtime':65,'helpers':7},'terminal':{'supervisor_status':sup['status'],'worker_status':worker['status'],
 'returncode':0,'whole_scope_seconds':sup['elapsed_seconds'],'worker_seconds':worker['wall_seconds'],'limit_seconds':600},
 'accounting':{'phase_ordinals':145,'phase_start_files':145,'phase_end_files':145,'registered_main_phases':22,'guard_boundaries':45,
 'guard_is_initialized_actual_calls':44,'guard_is_initialized_false':10,'guard_is_initialized_true':34,'guard_mem_get_info_actual_calls':34,
 'direct_mem_get_info_calls':2,'attempted':145,'returned':145,'failed':0,'pending':0,'publication_errors':[],
 'same_ordinal_operation_source_order_nested_envelope_and_ledger_verified':True,'last_completed_outer_guard_ordinal':143,'last_issued_nested_ordinal':145,
 'scope':'Explicit diagnostic phases plus individually proxied original guard CUDA reads; internal Torch/driver calls are not individually enumerated.'},
 'main_phases':phase_summary,'direct_observations':{'framework_versions':by_name['framework_version_read'][1]['result'],
 'post_import_growth':growth,'preload_free_total':direct[0],'final_free_total':direct[1],'all34_guard_free_total_values_same':True,
 'torch_current_process_allocator8_readings':allocator,'peak_statistics_reset_scope':'Original per-process allocator counter reset; no device reset/cache reclamation',
 'Windows_PID_counters':by_name['optional_Windows_PID_counters'][1]['result'],
 'limits':'Eight zero Torch allocator readings are not proof of zero CUDA-context/driver memory or per-app GPU usage. NVML and direct CUDA measurements occur at different times and expose different totals; no allocation attribution or historical causal comparison is made.'},
 'resource_observations':{'CPU_samples':236,'GPU_samples':46,'admission_samples':2,'minimum_host_available_bytes':min(row['host_available_bytes'] for row in cpu),
 'maximum_owned_tree_RSS_bytes':max(row['worker_tree_rss_bytes'] for row in cpu),'minimum_disk_free_bytes':min(row['disk_free_bytes'] for row in cpu),
 'minimum_external_GPU_free_bytes':min(row['gpu_free_bytes'] for row in gpu),'maximum_external_GPU_free_bytes':max(row['gpu_free_bytes'] for row in gpu),
 'CPU':timing(cpu,'sample_started_monotonic','sample_completed_monotonic'),'GPU':timing(gpu,'gpu_sample_started_monotonic','gpu_sample_completed_monotonic'),
 'maximum_heartbeat_gap_seconds':max(b['unix_seconds']-a['unix_seconds'] for a,b in zip(hb,hb[1:])),
 'maximum_returned_CPU_phase_latency_seconds':{key:max(row['phase_latency_seconds'][key] for row in cpu) for key in cpu[0]['phase_latency_seconds']},
 'all_current_observed_amount_cadence_freshness_identity_and600_gates_passed':True},
 'cleanup':{'owned_Job_assigned_before_resume':True,'external_monitor_ready_before_resume':True,'job_closed_kill_on_close':True,
 'exact_observed_identities_currently_alive':live,'sampler_threads_terminal_verified':True,'NVML_cleanup_verified':True,'unjoined_sampler_threads':[],
 'lease_path':str(lease),'durable_reservation_worker_resume_terminal_links_verified':True,'all297_raw_outputs_frozen':True,
 'actual_child_membership_limit':'Worker invokes the unchanged ownership verifier before its first recorded phase, but does not persist that verifier return. Owned sampled PID/createTime/FILETIME tree and pinned source prove containment; no missing child-verifier receipt is invented.'},
 'hash_closure':{'before':compact(before),'after':compact(after),'all_equal':True,'preservation_receipt_path':str(saved/'preservation_receipt.json'),
 'preservation_receipt_file_sha256':before['preservation_receipt']['sha256'],'preservation_receipt_body_sha256':pres['sha256']},
 'measurement_limits':{'phase_duration_scope':'Recorded start-to-end monotonic envelopes include start-marker fsync, actual callable work and scheduling; they are not isolated service/OS-query durations.',
 'submillisecond_clock_precision_not_inferred':True,'proxy_exact_object_and_exception_delivery':'Established by byte-pinned implementation and prior CPU identity/exception controls; actual JSON records preserve values/type projections, not live object identity.',
 'historical_phase_or_cause_established':False,'intrinsic_collector_or_model_fit_established':False,'current_observations_do_not_cure_or_reclassify_prior_abort':True},
 'model_loaded':False,'model_constructed':False,'explicit_model_or_tensor_allocation_requested':False,'loader_or_weights_opened':False,
 'forwards':0,'backwards':0,'optimizer_steps':0,'actual_grading':False,'scoring_authorized':False,'training_authorized':False,
 'native_model_mechanics_qualified':False,'raw_M0_collection_qualified':False,'study_warmstart_allowed':False,
 'transfer_or_retention_established':False,'historical_cause_established':False,'native_fit_established':False,
 'audit_framework_model_GPU_probe_execution':False,'protected_bank_access':False,'paid_services':False,
 'conclusion':'This one current owned NO_MODEL diagnostic completed all145 phase boundaries with the original guard CUDA queries individually recorded and all terminal byte/resource/cleanup links closed. It provides current import/context/query/config observations only; no model loading, generation, training, grading, historical cause or collector-fit conclusion is established.'}
audit['sha256']=digest(audit);raw=(canonical(audit)+'\n').encode()
import os
with target.open('xb') as stream:check(stream.write(raw)==len(raw),'complete independent receipt write');stream.flush();os.fsync(stream.fileno())
verified=read(target);check(verified==audit and verified['sha256']==digest({k:v for k,v in verified.items() if k!='sha256'}),'reread receipt body checksum')
print(json.dumps({'status':audit['status'],'checks':audit['checks_passed'],'file_sha256':info(target)['sha256'],'body_sha256':audit['sha256'],'bytes':len(raw),
 'phase_ordinals':145,'CPU_samples':236,'GPU_samples':46,'source_before_after_equal':True,'path':str(target)}))

