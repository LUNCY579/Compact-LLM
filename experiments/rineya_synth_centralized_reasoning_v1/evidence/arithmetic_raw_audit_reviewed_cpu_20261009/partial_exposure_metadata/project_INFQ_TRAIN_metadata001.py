from pathlib import Path
from collections import Counter
import hashlib,json,os,time
P=Path('experiments/rineya_synth_centralized_reasoning_v1').resolve();D=P/'artifacts/arithmetic_exposure_metadata_work'
INPUTS={
 'INFQ_independent_audit':(P/'evidence/infq001_attempt1_independent_audit_20261004.json','b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e'),
 'INFQ_preserved_config':(P/'evidence/infq001_attempt1_20261004/SYNTH_INFQ001_TRAIN4.json','a26897bfbe36ce284221c6f4182c487d0d76287e708fe84de75f4e57fc2a9a79'),
 'TRAIN_enrollment':(P/'data/crossed_source2_v1_20261003/enrollment.json','2730f96e1e7b9b23bab6f5bfc869400d16fd42eca2071d004c68aacd2bb1defc'),
 'planned_occurrences':(P/'data/crossed_source2_v1_20261003/training_occurrences.json','9fb20d227e4b72d30dfe44be47e0764f9939544f7de34992c6ebce7a0b5d39dc'),
 'planned_schedule':(P/'data/crossed_source2_v1_20261003/schedule_result.json','2664eabfa4cfb9c4e28742cb58f4c6649b24eb80a16df3d5f2cf1a195a058744'),
}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def pairs(rows):
 value={}
 for k,v in rows:require(k not in value,'duplicate metadata key');value[k]=v
 return value
def readall():
 docs={};refs={}
 for name,(path,pin) in INPUTS.items():
  a=path.stat();raw=path.read_bytes();b=path.stat();require(len(raw)<=1024**2 and (a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino) and hashlib.sha256(raw).hexdigest()==pin,'exact bounded stable metadata '+name)
  refs[name]={'path':str(path),'bytes':len(raw),'sha256':pin};docs[name]=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
 return docs,refs
start=time.monotonic();docs,before=readall();a=docs['INFQ_independent_audit'];c=docs['INFQ_preserved_config'];e=docs['TRAIN_enrollment'];o=docs['planned_occurrences'];s=docs['planned_schedule']
require(a['status']=='PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS' and a['supervisor_status']=='SUPERVISED_WORKER_EXITED_ZERO' and a['worker_status']=='PASS_SCOPED_NATIVE_INFERENCE_MECHANICS' and a['answer_grading_performed'] is False,'actual preserved scoped observation metadata, no task grade')
require(a['hash_closure']['config_file_sha256']==before['INFQ_preserved_config']['sha256'] and a['hash_closure']['config_body_sha256']==c['sha256'],'audit-to-executing-config bytes/body reference')
require(e['sha256']==digest({k:v for k,v in e.items() if k!='sha256'}) and s['sha256']==digest({k:v for k,v in s.items() if k!='sha256'}) and o['schedule_sha256']==s['sha256'] and s['enrollment_sha256']==e['sha256'],'original CPU enrollment/schedule/occurrence lineage')
require(len(e['parents'])==20 and set(e['record_ids'])==set(o['row_ids'])=={'M1','M2'} and all(len(e['record_ids'][arm])==320 and len(set(e['record_ids'][arm]))==320 and len(o['row_ids'][arm])==540 for arm in ('M1','M2')),'exact registered20/320-perarm/540 planned dose metadata')
# Only selection metadata and independent audit row identity/count summaries are used.
# The full config was parsed; no prompt-array values or model responses are extracted.
choices=c['selection_bindings'];observed=a['numerics']['row_controls'];require(len(choices)==len(observed)==4 and [r['row_id'] for r in choices]==[r['row_id'] for r in observed] and len({r['row_id'] for r in observed})==4,'all four observed independent audit identities match config selection')
require(a['numerics']['two_exact_repeats_each'] is True and a['accounting']['generated_only_bulk_forwards']==4 and a['accounting']['parity_forward_calls']==2 and a['accounting']['actual_forward_calls']==84,'observed scoped operation totals')
rows=[]
for choice,witness in zip(choices,observed):
 arm=choice['arm'];rid=choice['row_id'];require(choice['split']=='train' and choice['skill'] in ('R','S') and rid in e['record_ids'][arm] and rid not in e['record_ids']['M2' if arm=='M1' else 'M1'] and choice['source_prompt_tokens']==witness['prompt_tokens'],'original arm/TRAIN identity and prompt-length metadata')
 n=witness['generated_tokens_per_repeat'];require(type(n) is int and 0<n<=32,'observed repeat length metadata')
 count=o['row_ids'][arm].count(rid)
 rows.append({'row_id':rid,'arm':arm,'skill':choice['skill'],'source_split':'train','source_prompt_tokens':choice['source_prompt_tokens'],'recorded_prompt_ids_sha256':choice['prompt_ids_sha256'],'enrolled_source_record_member':True,'observed_INFQ_repeat_capture_forwards':2*n,'observed_INFQ_bulk_forwards':1,'observed_INFQ_zero_parity_forwards':2 if not rows else 0,'planned_STUDY_occurrences':count,'planned_STUDY_occurrences_are_observed_training':False})
require(sum(r['observed_INFQ_repeat_capture_forwards'] for r in rows)==a['accounting']['individual_durable_token_stages']==a['accounting']['registered_operation_counts']['inference_qualification_forward']==78,'source-qualified observed individual forward denominator')
require([(r['arm'],r['planned_STUDY_occurrences']) for r in rows]==[('M1',1),('M1',6),('M2',2),('M2',5)],'exact planned counts distinct from observed native scope')
_,after=readall();require(before==after,'actual metadata byte references stable through projection')
v={'kind':'SCOPED_PRESERVED_INFQ_TO_TRAIN_METADATA_JOIN001','input_references_before':before,'input_references_after':after,'bytes_unchanged':True,'INFQ_source_commit':a['source_commit'],'observation_basis':'Previously independently audited successful four-row inference qualification; current action projects metadata, not a new observation or native rerun.','rows':rows,'TRAIN_source_population':{'parents':20,'record_ids_per_arm':320,'planned_occurrences_per_arm':540,'training_admitted':e['training_admitted']},'observed_INFQ_totals':{'distinct_prompt_rows':4,'repeat_capture_forwards':78,'bulk_forwards':4,'zero_parity_forwards':2,'all_forwards':84},'scope_incomplete':True,'all_available_or_consumed_exposure_coverage_established':False,'arithmetic_overlap_checked':False,'arithmetic_lineage_disjointness_qualified':False,'model_outputs_or_generated_token_IDs_extracted':False,'whole_INFQ_config_bytes_read':True,'whole_INFQ_config_JSON_parsed_including_embedded_prompt_arrays':True,'embedded_prompt_array_values_used_or_extracted':False,'TRAIN_token_pool_or_student_array_files_opened':False,'M0_or_scorer_run':False,'model_framework_or_native_API_run':False,'protected_bank_access':False,'status':'PARTIAL_OBSERVED_SOURCE_CONTACT_METADATA_ONLY','seconds':time.monotonic()-start,'limitations':['This is one actual audited scope; Native7, backend/Q003, later successful/failed capture prefixes and other exposures are not counted here.','Four native source prompt contacts do not mean four independent full input-occurrence records; the existing audit counts84 actual forwards while this receipt does not reconstruct their full input prefixes.','Planned SFT counts1/6 and2/5 are not observed extra training. Source/namespace inequality and this partial join cannot qualify arithmetic disjointness.']}
v['sha256']=digest(v);out=D/'INFQ_TRAIN_METADATA_JOIN001.json';raw=(canonical(v)+'\n').encode()
with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
require(out.read_bytes()==raw,'write-once metadata receipt readback');print(json.dumps({'path':str(out),'bytes':len(raw),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256'],'actual_observed_INFQ_source_rows':4,'planned_counts':[r['planned_STUDY_occurrences'] for r in rows],'scope_incomplete':True}))
