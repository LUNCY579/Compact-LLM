from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent.parent
D=P/'evidence/arithmetic_measurement_stage_cpu_20261009'
R=P/'artifacts/arithmetic_raw_audit_work';M=P/'artifacts/arithmetic_measurement_completion_work'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def selected(path,pin):
 raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
 v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
selected(R/'frozen_arithmetic_raw_audit002/HANDOFF002.json','e73dd97ad3c995e2eea6c8a6ceab32c7e572a5b624ae5c7ca108e765df20a5b5')
rr=selected(R/'independent_review002/INDEPENDENT_RAW_REVIEW002.json','538459d919315e730bafc06da8ac0e4264294479eee81316ea5ad1985a4cc435')
selected(M/'frozen_arithmetic_measurement001/HANDOFF001.json','de53e5c612ab66aad9f0f306dcbcc9fab1dd7fa11f32758a60a94c8da03adcb8')
mr=selected(M/'independent_review001/INDEPENDENT_MEASUREMENT_WIRE_REVIEW001.json','9e0c8da75218834316d471264032f3c444503b860d0f1ca71ef509c13113a420')
rootraw=json.loads((R/'root_replay002/CPU_CONTROLS001.json').read_bytes())
rootmeasurement=json.loads((M/'root_replay001/CPU_CONTROLS001.json').read_bytes())
assert rootraw['success'] is True and rootraw['tests']==6
assert rootmeasurement['success'] is True and rootmeasurement['tests']==5
assert not D.exists();D.mkdir()
for src,name in ((R/'frozen_arithmetic_raw_audit002','raw_auditor/current002'),(R/'root_replay002','raw_auditor/root_replay002'),(R/'independent_review002','raw_auditor/independent_review002'),(M/'frozen_arithmetic_measurement001','measurement/candidate001'),(M/'root_replay001','measurement/root_replay001'),(M/'independent_review001','measurement/independent_review001')):shutil.copytree(src,D/name)
for src in M.iterdir():
 if src.is_dir() and src.name.startswith('pre_'):shutil.copytree(src,D/'measurement/history'/src.name)
shutil.copy2(Path(__file__).resolve(),D/'preserve_arithmetic_measurement_stage001.py')
v={'kind':'ROOT_PRESERVED_ARITHMETIC_MEASUREMENT_CPU_STAGE001','source_commit':'11ba5f5268a8e6a09affe4dd890184178877b3b7','raw_auditor_controls':6,'measurement_wire_controls':5,'raw_independent_review_body_sha256':rr['sha256'],'measurement_independent_review_body_sha256':mr['sha256'],'raw_readonly_replay_conditionally_implemented':True,'measurement_CPU_wire_replay_passed':True,'measurement_native_source_qualification_or_promotion_accepted':False,'package_native_paths_promoted':False,'actual_model_capture_scoring_or_enrollment_performed':False,'pending':['Authenticated original-source lineage and real INFQ/native/failure schema projections','Complete failed/partial causal-prefix occurrence manifests and source comparability','Authenticated original owner-state/chronology derivation','Fixed qualified task/dialogue/long-context grade providers and actual ratings/support','Fresh current native qualification, GPU admission and full matched M0/M1/M2 comparison'],'files':{f.relative_to(D).as_posix():{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(D.rglob('*')) if f.is_file()}}
v['sha256']=digest(v);(D/'ROOT_PRESERVATION.json').write_bytes((json.dumps(v,sort_keys=True,indent=2)+'\n').encode());print(json.dumps({'files':len(v['files']),'sha256':v['sha256']}))
