from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent.parent;W=P/'artifacts/infq_exposure_projection_work'
D=P/'evidence/INFQ_exposure_projection_cpu_20261009'
O=Path('C:/Users/clclc/BusinessProject/synth-INFQ-exposure-projection-20261009-attempt003')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def selected(path,pin):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==pin
 v=json.loads(b);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
selected(W/'frozen_INFQ_projection005/HANDOFF005.json','392623f62e3e9c9ffe41b38c34423454ebcfd35cf05dfac77054af7eaaabea55')
review=selected(W/'independent_actual_review001/INDEPENDENT_ACTUAL_PROJECTION_AUDIT001.json','86e1f5baea216108b14b05a8c39b8e42ed6117f8fb25dd63c74daf6c6102ffaa')
out=selected(O/'INFQ_CAUSAL_INPUT_PROJECTION001.json','954230d581b30decccf1c8539e149fd6e7e40372c7d6eb0d1a013c7a573bd81d')
selected(O/'PROJECTION_RECEIPT001.json','af248fabd78219f1c35b2b54a3f3713eee04eb5723543327b24f3b7741376a69')
assert out['total_calls']==84 and out['total_input_positions']==28627
assert {c['parent_membership'] for c in out['calls']}=={'balanced_parent_023','balanced_parent_038'}
assert not D.exists();D.mkdir()
shutil.copytree(W/'frozen_INFQ_projection005',D/'selected_source005')
shutil.copytree(O,D/'actual_output_attempt003')
shutil.copytree(W/'independent_actual_review001',D/'independent_actual_review001')
for source in W.iterdir():
 if source.is_dir() and source.name!='frozen_INFQ_projection005' and source.name!='independent_actual_review001':
  shutil.copytree(source,D/'history'/source.name)
for name in ('READ_SCOPE_DRAFT001.json','READ_SCOPE001.json','READ_SCOPE002.json','READ_SCOPE003.json','READ_SCOPE004.json','ROOT_ACTUAL_EXECUTION001.json'):
 source=W/name
 if source.exists():shutil.copy2(source,D/name)
for name in ('run_INFQ_projection001.py','run_INFQ_projection002.py','run_INFQ_projection003.py'):
 shutil.copy2(P/'artifacts'/name,D/name)
shutil.copy2(Path(__file__).resolve(),D/'preserve_INFQ_projection001.py')
value={'kind':'ROOT_PRESERVED_ACTUAL_INFQ_SOURCE_RECIPE_PROJECTION001','source_commit':'47eeda299c137a76bc6370ab17e717fd41b2f07f','physical_forward_inputs':84,'input_positions':28627,'original_raw_files':95,'total_original_source_files':105,'selected_source_rows':4,'shared_source_parents':2,'actual_projection_accepted_within_named_scope':True,'independent_actual_audit_body_sha256':review['sha256'],'originals_unchanged':True,'failed_attempts_preserved':2,'input_vectors_independently_recorded_per_call':False,'global_exposure_coverage':False,'arithmetic_family_mapping':'UNDEFINED','native_admission_scoring_or_enrollment_qualified':False,'model_GPU_optimizer_tokenizer_or_scoring_performed':False,'files':{f.relative_to(D).as_posix():{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(D.rglob('*')) if f.is_file()}}
value['sha256']=digest(value);(D/'ROOT_PRESERVATION.json').write_bytes((json.dumps(value,sort_keys=True,indent=2)+'\n').encode());print(json.dumps({'files':len(value['files']),'sha256':value['sha256']}))
