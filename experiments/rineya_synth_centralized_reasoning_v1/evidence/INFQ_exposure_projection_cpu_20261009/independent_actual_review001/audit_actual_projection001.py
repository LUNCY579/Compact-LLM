"""Independent actual INFQ recipe/prefix/source audit; no projector or model runs."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,os,sys,time

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
SOURCE=WORK/'frozen_INFQ_projection005'
SCOPE=SOURCE/'READ_SCOPE004.json'
OUTPUT=Path('C:/Users/clclc/BusinessProject/synth-INFQ-exposure-projection-20261009-attempt003')
PINS={'infq_projection.py':'cb121eac78a61e8634a17a48b61416e103b5ffeb5dd68bb7dab8db3b943d74ce',
 'test_infq_projection.py':'633473c8f49f028d85edf2bcf64eaaf9168fc025eee9785318f07a9adfdf0867',
 'READ_SCOPE004.json':'c6cabdae832bb844a7ee3bf66e30fb6e7515b8eef87d645571b563fad608c0ed'}
RESULT_PIN='954230d581b30decccf1c8539e149fd6e7e40372c7d6eb0d1a013c7a573bd81d'
RECEIPT_PIN='af248fabd78219f1c35b2b54a3f3713eee04eb5723543327b24f3b7741376a69'
FORBIDDEN=[];READS=[];START=time.monotonic()
def require(ok,message):
 if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def pairs(items):
 out={}
 for key,value in items:require(key not in out,'duplicate JSON key');out[key]=value
 return out
def finite(text):
 value=float(text);require(math.isfinite(value),'nonfinite JSON float');return value
def parse(raw):return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def body(value):require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'complete body seal')
def raw_ref(path,ref=None):
 path=Path(path);require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path,'exact regular original path')
 a=path.stat();require(a.st_size<=32*1024**2,'bounded file')
 with path.open('rb') as stream:
  first=os.fstat(stream.fileno());raw=stream.read(a.st_size+1);last=os.fstat(stream.fileno())
 z=path.stat();require((a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)==(first.st_size,first.st_mtime_ns,first.st_ino)==(last.st_size,last.st_mtime_ns,last.st_ino) and len(raw)==a.st_size,'original path/handle stable')
 actual={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if ref is not None:require(all(actual[k]==ref[k] for k in ('bytes','sha256')),'externally selected byte identity')
 require(time.monotonic()-START<=600,'independent CPU audit600s');return raw,actual

def protect(event,args):
 if event in ('ctypes.dlopen','ctypes.dlsym','subprocess.Popen','os.system'):FORBIDDEN.append(event);raise AssertionError('native/process call forbidden')
 if event=='import' and args and args[0].split('.',1)[0] in {'torch','transformers','tokenizers','safetensors','ctypes','psutil','numpy','jinja2','markupsafe'}:FORBIDDEN.append(args[0]);raise AssertionError('framework/tokenizer/native import forbidden')
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0]));text=path.as_posix().lower()
  require(path.suffix.lower() not in {'.safetensors','.pt','.bin','.dll','.pyd'} and path.name.lower() not in {'config.json','generation_config.json'} and '/check/' not in text and '/final/' not in text and 'token_pool_' not in text and 'student_pool_' not in text,'weights/model config/TRAIN array/bank forbidden')
  READS.append(str(path))
sys.addaudithook(protect)
source_before={}
for name,pin in PINS.items():
 _,ref=raw_ref(SOURCE/name);require(ref['sha256']==pin,'frozen source/scope exact byte pin');source_before[name]=ref
scope=parse(raw_ref(SCOPE)[0]);body(scope)
selected={**scope['raw_references'],**scope['public_source_references'],**scope['metadata_references'],
 scope['audit_reference']['path']:scope['audit_reference'],scope['config_reference']['path']:scope['config_reference'],scope['preservation_reference']['path']:scope['preservation_reference']}
require(len(selected)==105 and sum(ref['bytes'] for ref in selected.values())==4365088,'same original105/4365088 scope')
raws={};before={}
for path,ref in selected.items():raws[path],before[path]=raw_ref(path,ref)
def doc(ref):return parse(raws[ref['path']])
audit=doc(scope['audit_reference']);config=doc(scope['config_reference']);pres=doc(scope['preservation_reference'])
require(scope['audit_reference']['sha256']=='b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e' and audit['status']=='PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS','actual historic independently audited scope')
require(scope['preservation_reference']['sha256']=='fe49928806b6edd62b5c004afa5dde94e9d9d217f1026d6ecbee739db7644f67' and 'sha256' not in pres and pres['schema_version']==1 and pres['config_file_sha256']==audit['hash_closure']['config_file_sha256']==scope['config_reference']['sha256'],'real legacy unsealed preservation/config join')
body(config);require(config['sha256']==audit['hash_closure']['config_body_sha256'],'original config body')
rawmap={Path(path).name:{k:ref[k] for k in ('bytes','sha256')} for path,ref in scope['raw_references'].items()}
require(len(rawmap)==95 and rawmap==audit['hash_closure']['raw_preserved_hashes']==pres['files'],'all95 exact original byte map')
for path,ref in scope['public_source_references'].items():
 root=Path(scope['config_reference']['path']).parent.parent.parents[1]
 require(pres['source_snapshot_sha256'][Path(path).relative_to(root).as_posix()]==ref['sha256'],'original source recipe pin in preservation')
raw_docs={Path(path).name:parse(raws[path]) for path,ref in scope['raw_references'].items() if ref['role']=='INFQ_original_JSON'}
events=[parse(line) for line in raws[next(path for path in scope['raw_references'] if Path(path).name=='events.jsonl')].splitlines() if line]
worker=raw_docs['worker_result.json'];controls=[raw_docs['stage_inference_qualification_%04d.json'%n] for n in range(1,5)]
require(worker['numerical_controls']==controls,'original worker/control publication identity')
# Parse all original call outcomes independently using strict LIFO and monotonic IDs.
stack=[];attempts=[];returned={};counts=Counter();a_pos={};r_pos={};m_pos={};nested=[]
for position,event in enumerate(events):
 kind=event.get('event')
 if kind=='attempt':
  require(type(event['id']) is int and event['id']==len(attempts)+1,'contiguous global original IDs')
  if stack:
   require(len(stack)==1 and stack[-1]['id']==3 and stack[-1]['operation']=='checkpoint_load' and event['operation']=='tensor_copy' and 4<=event['id']<=313,'only exact original loader nesting')
   nested.append(event['id'])
  stack.append(event);attempts.append(event);counts[event['operation']]+=1;a_pos[event['id']]=position
 elif kind in ('returned','failed'):
  require(kind=='returned' and stack and event['id']==stack[-1]['id'] and event['operation']==stack[-1]['operation'] and event['id'] not in returned,'unique LIFO outcomes/no failures')
  stack.pop();returned[event['id']]=event;r_pos[event['id']]=position
 elif kind=='measurement_preserved':
  require(event['path'] not in m_pos and event['path'] in rawmap and event['sha256']==rawmap[event['path']]['sha256'],'unique exact original publication bytes')
  m_pos[event['path']]=position
require(not stack and nested==list(range(4,314)) and len(attempts)==len(returned)==403 and dict(counts)==audit['accounting']['registered_operation_counts'],'all403 original API/14 operation closure')
for ref in scope['metadata_references'].values():
 if Path(ref.get('path','')).name=='never':raise AssertionError('unused')
metadata={Path(path).name:raws[path] for path in scope['metadata_references']}
prep=parse(metadata['preparation_receipt.json']);enrollment=parse(metadata['enrollment.json'])
for name in ('candidate_records.jsonl','enrollment.json'):
 ref=next(ref for path,ref in scope['metadata_references'].items() if Path(path).name==name)
 require(prep['files'][name]=={k:ref[k] for k in ('bytes','sha256')},'actual source metadata preparation byte commitment')
fields={'row_id','parent_id','component_id','source_hash','arm','skill','split'}
identities={}
for line in metadata['candidate_records.jsonl'].splitlines():
 if line:
  record=parse(line);projection={key:record[key] for key in fields};require(projection['row_id'] not in identities,'unique candidate identity');identities[projection['row_id']]=projection
require(len(identities)==640 and set(record['parent_id'] for record in identities.values())==set(enrollment['parents']) and len(enrollment['parents'])==20,'original640/20 source cohort')
for arm in ('M1','M2'):require({key for key,value in identities.items() if value['arm']==arm}==set(enrollment['record_ids'][arm]) and len(enrollment['record_ids'][arm])==320,'all320 original perarm IDs')
result_raw,result_ref=raw_ref(OUTPUT/'INFQ_CAUSAL_INPUT_PROJECTION001.json');require(result_ref['sha256']==RESULT_PIN and result_ref['bytes']==250384,'Root actual result file')
result=parse(result_raw);body(result)
receipt_raw,receipt_ref=raw_ref(OUTPUT/'PROJECTION_RECEIPT001.json');require(receipt_ref['sha256']==RECEIPT_PIN,'actual projection receipt byte pin');receipt=parse(receipt_raw);body(receipt)
require(receipt['read_scope_sha256']==PINS['READ_SCOPE004.json'] and receipt['original_file_count']==105 and receipt['original_inputs']==selected and receipt['result_reference']=={**result_ref,'role':'new_projection_output'} and receipt['result_body_sha256']==result['sha256'],'actual output/input/scope/source receipt joins')
rows=config['rows'];require(len(rows)==4 and config['source_packet_sha256']==digest(rows),'exact original four input rows')
recipes=[]
def add(operation,row,ids,repeat,step,stage,origin):
 require(ids and all(type(x) is int and 0<=x<151936 for x in ids) and len(ids)<=512,'original causal IDs/vocabulary/context')
 member=identities[row['row_id']]
 recipes.append({'operation':operation,'row_id':row['row_id'],'repeat':repeat,'step':step,'input_ids':list(ids),'origin':origin,'durable_stage':stage,
 'input_positions':len(ids),'attention_mask':[1]*len(ids),'model_input_ids_sha256':digest(ids),'arithmetic_family_mapping':'UNDEFINED',
 'arm_membership':member['arm'],'parent_membership':member['parent_id'],'original_source_membership':member})
for op in ('zero_parity_base_forward','zero_parity_adapter_forward'):add(op,rows[0],rows[0]['prompt_ids'],None,None,None,'original_worker_config_row0_parity')
for row_ordinal,(row,control) in enumerate(zip(rows,controls),1):
 require(row['attention_mask']==[1]*len(row['prompt_ids']) and row['prompt_tokens']==len(row['prompt_ids']) and control['row_id']==row['row_id'] and control['prompt_ids_sha256']==digest(row['prompt_ids']) and control['source_packet_sha256']==config['source_packet_sha256'] and control['traces'][0]==control['traces'][1],'original source row and equal two traces')
 for repeat,trace in enumerate(control['traces']):
  prefix=list(row['prompt_ids'])
  for step,value in enumerate(trace):
   name='stage_inference_q%04d_r%d_t%02d.json'%(row_ordinal,repeat,step);saved=raw_docs[name]
   require(value['prefix_length']==len(prefix) and saved['row_id']==row['row_id'] and saved['repeat']==repeat and saved['step']==step and saved['token']==value['token'] and saved['logprob']==value['logprob'] and saved['prefix_length']==len(prefix) and saved['input_row_sha256']==digest(row) and saved['prompt_ids_sha256']==digest(row['prompt_ids']) and saved['source_packet_sha256']==config['source_packet_sha256'],'actual durable token/source/trace join')
   require(not any(t['token'] in (151643,151645) for t in trace[:step]),'first EOS boundary')
   add('inference_qualification_forward',row,prefix,repeat,step,name,'original_trace_prompt_plus_same_repeat_prior_generated_tokens');prefix.append(value['token'])
 add('inference_qualification_bulk_forward',row,row['prompt_ids']+[v['token'] for v in control['traces'][0]][:-1],0,None,'stage_inference_qualification_%04d.json'%row_ordinal,'original_bulk_prompt_plus_repeat0_excluding_last_selected_token')
# Bind recipes to actual global call IDs, then verify every cumulative stage cursor.
physical=[event for event in attempts if event['operation'] in {'zero_parity_base_forward','zero_parity_adapter_forward','inference_qualification_forward','inference_qualification_bulk_forward'}]
require(len(physical)==len(recipes)==84,'all84 physical calls')
component=Counter();positions=0
for number,(recipe,event) in enumerate(zip(recipes,physical),1):
 require(event['operation']==recipe['operation'],'physical source recipe operation order')
 recipe.update(forward_ordinal=number,journal_call_id=event['id'])
 if recipe['durable_stage'] is not None:
  component[event['operation']]+=1;positions+=recipe['input_positions'];stage=raw_docs[recipe['durable_stage']];cost=stage['costs'];next_pos=a_pos[physical[number]['id']] if number<len(physical) else len(events)
  require(cost['operations']=={op:{'attempted':n,'returned':n,'failed':0,'pending':0} for op,n in component.items()} and cost['attempted_forwards']==sum(component.values()) and cost['input_positions']=={'attempted':positions,'returned':positions,'failed':0,'pending':0},'exact cumulative component call/input cursor')
  require(r_pos[event['id']]<m_pos[recipe['durable_stage']]<next_pos,'actual return/publication/next-forward timeline')
require(result['calls']==recipes,'all84 independently rebuilt exact prefixes/masks/identity/ID recipes match actual output')
operation_counts=dict(Counter(row['operation'] for row in recipes));operation_positions={op:sum(row['input_positions'] for row in recipes if row['operation']==op) for op in operation_counts}
require(result['operation_counts']==operation_counts and result['operation_input_positions']==operation_positions and result['total_calls']==84 and result['total_input_positions']==sum(row['input_positions'] for row in recipes)==28627,'all84/28627 exact total and operation roles')
require(operation_counts=={'zero_parity_base_forward':1,'zero_parity_adapter_forward':1,'inference_qualification_forward':78,'inference_qualification_bulk_forward':4} and operation_positions=={'zero_parity_base_forward':419,'zero_parity_adapter_forward':419,'inference_qualification_forward':26544,'inference_qualification_bulk_forward':1245},'exact actual audited parity/repeat/bulk scope')
require(worker['forward_dose']['attempted_calls']==84 and worker['forward_dose']['input_positions']==28627 and audit['accounting']['actual_forward_calls']==84 and audit['accounting']['actual_input_positions']==28627,'actual audit/worker original scope totals')
require(result['input_values_are_independently_recorded_per_call'] is False and result['source_recipe_and_recorded_trace_reconstruction'] is True and result['global_exposure_coverage'] is False and result['R_S_to_arithmetic_family_mapping']=='UNDEFINED' and result['model_GPU_or_scoring_performed'] is False,'correct limitations and no new model/scoring claim')
require(set(result)=={'kind','calls','operation_counts','operation_input_positions','total_calls','total_input_positions','input_values_are_independently_recorded_per_call','source_recipe_and_recorded_trace_reconstruction','global_exposure_coverage','R_S_to_arithmetic_family_mapping','model_GPU_or_scoring_performed','sha256'},'no target/spec/messages emitted')
after={path:raw_ref(path,ref)[1] for path,ref in selected.items()};require(after==before,'all105 original byte identities stable through independent review')
source_after={name:raw_ref(SOURCE/name)[1] for name in PINS};require(source_after==source_before,'all frozen producer/source/scope bytes stable')
require(raw_ref(OUTPUT/'INFQ_CAUSAL_INPUT_PROJECTION001.json')[0]==result_raw and raw_ref(OUTPUT/'PROJECTION_RECEIPT001.json')[0]==receipt_raw,'actual output bytes stable through independent review')
require(not FORBIDDEN and time.monotonic()-START<=600,'no forbidden imports/APIs and CPU audit budget')
value={'kind':'INDEPENDENT_ACTUAL_INFQ84_SOURCE_RECIPE_PROJECTION_AUDIT001','accepted_scoped_projection':True,'original_inputs_before':before,'original_inputs_after':after,'all105_original_bytes_stable':True,'frozen_producer_source_before':source_before,'frozen_producer_source_after':source_after,'actual_result_reference':result_ref,'actual_receipt_reference':receipt_ref,'result_body_sha256':result['sha256'],'receipt_body_sha256':receipt['sha256'],'global_API_calls':403,'physical_forward_calls':84,'physical_input_positions':28627,'role_calls':operation_counts,'role_positions':operation_positions,'all84_prefixes_masks_hashes_globalIDs_timeline_costs_directly_rebuilt':True,'source_rows':4,'source_parents':sorted({r['parent_membership'] for r in recipes}),'candidate_lineage_rows':640,'enrolled_source_parents':20,'projection_program_rerun':False,'new_model_tokenizer_GPU_optimizer_or_scoring_action':False,'independent_pre_hook_input_telemetry_established':False,'global_exposure_coverage':False,'arithmetic_family_mapping':'UNDEFINED','complete_retention_disjointness_or_membership_qualified':False,'forbidden_imports_or_APIs':FORBIDDEN,'read_paths':sorted(set(READS)),'audit_wall_seconds':time.monotonic()-START,'prior_failures_preserved':['actual_failure001 legacy preservation schema assumption','actual_failure002 serial-only loader journal assumption'],'limitations':['Every reconstructed vector is joined to original config, durable token traces, source recipe and actual journal/cost ordering; no new independent per-call pre-hook observation is created.','Only historical INFQ84 source contacts are covered; Native7/Q003/later failed or successful attempts and global exposure lineage remain incomplete.','Whole candidate JSONL was parsed but only seven identity fields extracted/emitted; ignored prompt/target/spec contents are not processed, graded or emitted.','Independent CPU projection audit is not current model/runtime/resource qualification, arithmetic family mapping, native grading or source-correct retention enrollment.']};value['sha256']=digest(value)
out=HERE/'INDEPENDENT_ACTUAL_PROJECTION_AUDIT001.json';encoded=(canonical(value)+'\n').encode()
with out.open('xb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
require(out.read_bytes()==encoded,'independent actual audit durable readback');print(json.dumps({'path':str(out),'file_sha256':hashlib.sha256(encoded).hexdigest(),'body_sha256':value['sha256'],'all84_rebuilt':True,'original105_stable':True,'seconds':value['audit_wall_seconds']}))
