from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent.parent
W=P/'artifacts/arithmetic_raw_audit_work'
D=P/'evidence/arithmetic_raw_audit_reviewed_cpu_20261009'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def checked(path,pin):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==pin
 v=json.loads(b);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
h=checked(W/'frozen_arithmetic_raw_audit001/HANDOFF001.json','985f3a72e638a2ac8fedeef5d6722b956655e55be52c80573102f17fd4963cdf')
i=checked(W/'independent_review001/INDEPENDENT_RAW_REVIEW001.json','9d40a79234e577b77c499350822598247c9f6b8751fc1374b7c61018e6be26af')
r=json.loads((W/'root_replay001/CPU_CONTROLS001.json').read_bytes());assert r['success'] is True and r['tests']==5 and r['native_PASS'] is False
assert r['sha256']==digest({k:x for k,x in r.items() if k!='sha256'})
assert not D.exists();D.mkdir()
for src,dest in ((W/'frozen_arithmetic_raw_audit001','current'),(W/'independent_review001','independent_review'),(W/'root_replay001','root_replay')):shutil.copytree(src,D/dest)
for src in W.iterdir():
 if src.is_dir() and src.name.startswith('pre_'):shutil.copytree(src,D/'history'/src.name)
M=P/'artifacts/arithmetic_exposure_metadata_work'
assert hashlib.sha256((M/'INFQ_TRAIN_METADATA_JOIN001.json').read_bytes()).hexdigest()=='bc6092aa36b100a178a1894b918fe6ac14f6b9ffda6fd46ffd6d0b09a0666d9c'
assert hashlib.sha256((M/'project_INFQ_TRAIN_metadata001.py').read_bytes()).hexdigest()=='c13776e6ce8a241eeafec5cba4e57dc46a7dd02f638ed58782c0b1da28f58530'
shutil.copytree(M,D/'partial_exposure_metadata')
shutil.copy2(Path(__file__).resolve(),D/'preserve_arithmetic_raw_audit001.py')
v={'kind':'ROOT_PRESERVED_ARITHMETIC_RAW_AUDITOR_CPU001','source_commit':'11ba5f5268a8e6a09affe4dd890184178877b3b7','authored_controls':5,'root_controls':5,'independent_controls':5,'root_control_sha256':r['sha256'],'independent_review_sha256':i['sha256'],'actual_heavy_raw_audit_decoder_model_or_native_API_executed':False,'native_admission_or_scoring_qualified':False,'measurement_registration_still_under_review':True,'original384_768_sources_unchanged':True,'partial_INFQ_to_TRAIN_metadata_only':True,'complete_exposure_or_arithmetic_disjointness_established':False,'metadata_producer_source_postexecution_preservation_only':True,'metadata_seconds_not_resource_qualification':True,'files':{f.relative_to(D).as_posix():{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(D.rglob('*')) if f.is_file()}}
v['sha256']=digest(v);(D/'ROOT_PRESERVATION.json').write_bytes((json.dumps(v,sort_keys=True,indent=2)+'\n').encode());print(json.dumps({'files':len(v['files']),'sha256':v['sha256']}))
