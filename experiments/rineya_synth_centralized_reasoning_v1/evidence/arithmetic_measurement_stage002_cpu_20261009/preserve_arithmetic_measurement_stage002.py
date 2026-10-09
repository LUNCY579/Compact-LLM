from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent.parent;W=P/'artifacts/arithmetic_measurement_completion_work';D=P/'evidence/arithmetic_measurement_stage002_cpu_20261009'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def checked(path,pin):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==pin
 v=json.loads(b);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
checked(W/'frozen_arithmetic_measurement002/HANDOFF002.json','f4124e3f225f34a25eb8fe5129a6b36e772078343952fc40592e73ac3d9cc4a6')
i=checked(W/'independent_review002/INDEPENDENT_MEASUREMENT_WIRE_REVIEW002.json','58eacfe0c0bb2cbcd36c3dae52ae3cc44b1c7d80806ff09bcd4958f4f75dea21')
r=json.loads((W/'root_replay002/CPU_CONTROLS001.json').read_bytes());assert r['success'] is True and r['tests']==7
assert not D.exists();D.mkdir()
for src,n in ((W/'frozen_arithmetic_measurement002','candidate002'),(W/'root_replay002','root_replay002'),(W/'independent_review002','independent_review002')):shutil.copytree(src,D/n)
shutil.copy2(Path(__file__).resolve(),D/'preserve_arithmetic_measurement_stage002.py')
v={'kind':'ROOT_PRESERVED_MEASUREMENT_MEMBER_ANCHOR_CORRECTION_CPU002','source_commit':'11ba5f5268a8e6a09affe4dd890184178877b3b7','controls':7,'root_control_body_sha256':r['sha256'],'independent_review_body_sha256':i['sha256'],'predecessor_preservation_body_sha256':'e5c737b264b0235f5ae567c2db6095513f732767a5f19f90706bdbd41a4835e2','complete_member_order_and_capture_A_B_binding_CPU_seams_verified':True,'native_source_qualification_or_promotion_accepted':False,'prototype_five_domain_names_are_registered_roster_or_D':False,'actual_source_projection_capture_scoring_or_native_run_performed':False,'pending':['Authenticated original source and real native/INFQ schema support','Exhaustive observed causal-prefix occurrence providers and cross-schema comparability','Authenticated owner-state chronology','Qualified task/dialogue/long-context providers and actual source-correct support','Fresh resource/runtime/model qualification and full M0/M1/M2 result'],'files':{f.relative_to(D).as_posix():{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(D.rglob('*')) if f.is_file()}}
v['sha256']=digest(v);(D/'ROOT_PRESERVATION.json').write_bytes((json.dumps(v,sort_keys=True,indent=2)+'\n').encode());print(json.dumps({'files':len(v['files']),'sha256':v['sha256']}))
