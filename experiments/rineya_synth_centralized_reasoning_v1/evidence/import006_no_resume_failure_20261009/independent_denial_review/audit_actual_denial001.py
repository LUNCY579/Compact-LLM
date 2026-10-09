import json,pathlib,hashlib,datetime
p=pathlib.Path(__file__).resolve().parents[3]
a=p/'artifacts/import_approved_dispatch006'
out=p/'artifacts/prefix_input_owned_import_observation_001'
c=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
d=lambda x:hashlib.sha256(c(x).encode()).hexdigest()
D=json.loads((a/'DISPATCH_006.json').read_bytes())
S=json.loads((out/'SUPERVISOR_RECEIPT.json').read_bytes())
X=json.loads((out/'OWNED_EXTERNAL_ANCHORS.json').read_bytes())
result=json.loads((a/'dispatcher.stdout.log').read_bytes())
process=json.loads((a/'DISPATCH_PROCESS_RESULT006.json').read_bytes())
assert sorted(x.name for x in out.iterdir())==['OWNED_EXTERNAL_ANCHORS.json','SUPERVISOR_RECEIPT.json']
assert X=={'schema_version':2,**D['binding'],'grant_reference':D['grant_reference'],'owner_receipt_reference':D['owner_receipt_reference']}
assert S['config_sha256']==D['binding']['anchors']['expected_config_file'] and S['output_dir']==str(out)
assert S['status']=='ADMISSION_DENIED' and S['failure']=="ModuleNotFoundError: No module named 'psutil'"
assert S['worker_attempted'] is S['worker_launched'] is S['worker_resumed'] is False and S['admission_observations']==[]
assert S['partial_outputs']==[{'bytes':(out/'OWNED_EXTERNAL_ANCHORS.json').stat().st_size,'path':'OWNED_EXTERNAL_ANCHORS.json','sha256':hashlib.sha256((out/'OWNED_EXTERNAL_ANCHORS.json').read_bytes()).hexdigest()}]
assert S['partial_outputs_frozen'] is S['sampler_threads_terminal_verified'] is S['gpu_telemetry_cleanup_verified'] is True and S['unjoined_sampler_threads']==[]
assert result['terminal_status']=='IMPORT_OBSERVATION_NOT_RESUMED_FAILURE_RETAINED' and result['terminal_audit'] is result['preserved_directory'] is None and result['native_PASS'] is result['native_authority_issued'] is False
assert process['exit_code']==2 and process['dispatcher_stdout_bytes']==573 and process['dispatcher_stderr_bytes']==0
assert process['sha256']==d({k:v for k,v in process.items() if k!='sha256'})
print('ACTUAL006_NO_RESUME_DENIAL_AUDIT_PASS')
pins={}
for f in (out/'SUPERVISOR_RECEIPT.json',out/'OWNED_EXTERNAL_ANCHORS.json',a/'dispatcher.stdout.log',a/'dispatcher.stderr.log',a/'DISPATCH_PROCESS_RESULT006.json',a/'DISPATCH_006.json',a/'independent_review/INDEPENDENT_APPROVED_IMPORT006_REVIEW002.json',p/'native_supervisor.py',p/'import_observation_supervisor.py',p/'short_ram_monitor.py',pathlib.Path(__file__).resolve()):
    raw=f.read_bytes();pins[str(f)]={'bytes':len(raw),'file_sha256':hashlib.sha256(raw).hexdigest()}
R={'kind':'INDEPENDENT_ACTUAL_IMPORT006_NO_RESUME_DENIAL_AUDIT','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_RETAINED_FAILURE_SCOPE','reviewer':'/root/launch_packet_review','pins':pins,'observed_raw_output_names':sorted(x.name for x in out.iterdir()),'anchors_match_registered_dispatch':True,'supervisor_status':S['status'],'failure':S['failure'],'worker_attempted':False,'worker_launched':False,'worker_resumed':False,'admission_observation_count':0,'postrun_terminal_status':result['terminal_status'],'semantic_terminal_audit':None,'native_PASS':False,'native_authority_issued':False,'model_or_transfer_result':False,'framework_import_observed':False,'GPU_resource_admission_proven':False,'worker_dead_birth_audit':'NOT_APPLICABLE_NO_CREATED_WORKER','dispatcher_terminal_observation':{'source':'Root actual unified exec handle99559 terminal exit2; Root subsequent matching-process snapshot found none','independently_live_process_inspected':False,'recorded_process_result':process},'source_diagnosis':{'supervise_call':'original _acquire_lease after preflight and before admission','psutil_call':'_pid_alive unconditionally _psutil before raw-birth Win32 branch','missing_resolution':'parent -I -S has no SITE path transition; short_ram_monitor inserts PACKAGE only; selected worker finite psutil bootstrap never reached','historical_first_worker_lease':'0748e2cffeb04c4ca780bbf844d709b1','historical_pid':442476,'historical_creation_filetime_100ns':134356607225469230,'scope':'Static selected source plus historical lease metadata diagnosis; no original traceback exists to prove individual stack frame.'},'limitations':'No worker exists, so no fabricated birth/death or11operation/measurement terminal audit is claimed. Supervisor cleanup booleans are actual receipt facts; GPU memory/admission was not sampled. Dispatcher terminality is Root tool evidence, not independently resampled by reviewer.'}
R['sha256']=d(R)
path=a/'independent_review/INDEPENDENT_ACTUAL_IMPORT006_DENIAL001.json'
b=(c(R)+'\n').encode();path.open('xb').write(b)
print('receipt',str(path),hashlib.sha256(b).hexdigest(),R['sha256'])
