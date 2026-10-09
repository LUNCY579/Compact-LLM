"""Independent actual arithmetic capture-plan joins; pure bounded JSON/source reads."""
from collections import Counter
from pathlib import Path
import ast, hashlib, json, os, sys, time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
ACTUAL=PACKAGE/'artifacts/arithmetic_capture_actual_plan_work'
CPU=Path('C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001')
INPUTS={
 'plan':(ACTUAL/'ARITHMETIC_CAPTURE_PLAN001.json','81cf5f729e232710343a103cec1c34f4efb07acf5dca1ed32def76f9b5838c95'),
 'root_receipt':(ACTUAL/'ROOT_ACTUAL_PLAN001.json','1c0feaa8b1bb1c096ffaa1c48b21056bd691ab35013f73b617e0568cfecddcc2'),
 'contract_source':(PACKAGE/'artifacts/arithmetic_capture_preparation_work/arithmetic_capture_contract.py','81bd92c0323ef6c7414fd4a98812c281b573e3b85f3b862fd11dc6313e17a90d'),
 'root_writer_source':(PACKAGE/'artifacts/build_arithmetic_capture_plan001.py',None),
 'prompt':(CPU/'policy_prompt_tokens.json','7e2ec66fd08c0a7c5fd56444e0d3fa86cbd0b93b40c3cf8ffad75512ff6d7d66'),
 'oracle':(CPU/'oracle_CPU_only.json','0bee1da54cb5d08edf808f75251a26e0f401bbb00dfee3abaafede14f447058a'),
 'measurement':(Path(str(CPU)+'_projection_corrected')/'measurement_projected.json','e9a17b5fbca93c62684c5ddaa0380f721bd0e718158766c01f88ff55f626521e'),
 'population':(PACKAGE/'evidence/arithmetic_guard_reviewed_cpu_20261009/current/ARITHMETIC_GUARD_POPULATION001.json','2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3'),
 'policy':(PACKAGE/'evidence/arithmetic_guard_reviewed_cpu_20261009/current/SYNTH_ARITHMETIC_GUARD_POLICY001.json','42ee9c26f0fe1ab42c368618a5c25385fe09dac0f5777bab92df5c34c5fa05b7'),
 'ledger':(PACKAGE/'evidence/arithmetic_source_ledger_reviewed_cpu_20261009/current/ARITHMETIC_SOURCE_LEDGER001.json','c46f9ca13871e9bd9ab7695f0b79e729a9300d208b4d2bc9fce83ba92e56a252'),
 'cpu_tokenizer_source':(PACKAGE/'cpu_tokenizer.py',None),
 'pilot':(PACKAGE/'PILOT_DECISIONS.json','711336dbb5c6c68be8097f29159fcef934253292524bfe20936a1b2f32dc0ddd'),
 'tokenizer_receipt':(PACKAGE.parent/'post_training_mimo_centralized_v1/receipts/titan_static_v1/TOKENIZER_RECEIPT.json','b7994ddfb9b66dd1be491fcfdcf6c6b904f88d1b77c6615bf34a846d5714c086'),
}
EXPECTED_MAX_POSITIONS=[208184,208310,237054,237212,294186,293930,407552,409404]
SYSTEM='Compute the displayed arithmetic expression. Return exactly one JSON object with key "answer" and an integer value. No explanations or extra keys.'
FORBIDDEN=[];READS=[]

def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def pairs(items):
    value={}
    for key,item in items:require(key not in value,'duplicate JSON field');value[key]=item
    return value
def constant(value):raise ValueError('nonfinite JSON')
def ref(path,raw):return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def audit(event,args):
    if event in ('ctypes.dlopen','ctypes.dlsym','subprocess.Popen','os.system'):
        FORBIDDEN.append(event);raise AssertionError('no native/process API')
    if event=='import' and args and args[0].split('.',1)[0] in {'tokenizers','jinja2','markupsafe','torch','transformers','safetensors','psutil','ctypes','tensorflow','jax','flax'}:
        FORBIDDEN.append('import:'+args[0]);raise AssertionError('no tokenizer/framework/native import')
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0]));low=path.as_posix().lower()
        require(path.suffix.lower() not in ('.safetensors','.pt','.bin','.dll','.pyd','.exe') and path.name.lower() not in ('config.json','generation_config.json') and '/check/' not in low and '/final/' not in low,'no models/banks/native binaries')
        READS.append(str(path))
sys.addaudithook(audit)

def read_inputs():
    values={};refs={};raws={}
    for name,(path,pin) in INPUTS.items():
        require(path.is_file() and path.is_absolute() and not path.is_symlink(),'exact regular independent input')
        before=path.stat();raw=path.read_bytes();after=path.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and len(raw)<=16*1024**2,'bounded stable independent input')
        r=ref(path,raw);require(pin is None or r['sha256']==pin,'externally selected original input pin '+name);refs[name]=r;raws[name]=raw
        if path.suffix=='.json':values[name]=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    return values,refs,raws

def checksum(value):require(value.get('sha256')==digest({key:item for key,item in value.items() if key!='sha256'}),'canonical embedded body checksum')
def render(messages):return ''.join('<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>\n' for message in messages)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'

def validate(values,refs,raws):
    plan=values['plan'];root=values['root_receipt'];prompts=values['prompt'];measurement=values['measurement'];population=values['population'];policy=values['policy'];ledger=values['ledger'];oracle=values['oracle']
    for value in (plan,root,measurement,population,policy,ledger):checksum(value)
    require(root['plan_reference']=={**refs['plan'],'value_sha256':digest(plan)} and root['sha256']=='9f502fe028820beeb0273cbb4ac924e1e3163438252a084334944c990e54b159','actual Root receipt/file/whole-sealed-value joins')
    require(root['builder_source']=={'path':refs['contract_source']['path'],'sha256':refs['contract_source']['sha256']},'actual selected Root builder source')
    for name in ('population','prompt','measurement','policy','ledger'):require(root[name+'_reference']==refs[name],'actual Root original input reference')
    require(root['rows']==512 and root['parents']==128 and root['batches']==8 and root['maximum_forwards']==16400 and root['maximum_input_positions']==2295832 and all(root[k] is False for k in ('native_launch_performed','scoring_performed','M0_enrollment_performed','native_qualification_established')),'Root claims limited to exact CPU plan')
    closed=('native_launch_authorized','scoring_authorized','production_training_authorized','language_qualification_established','long_context_qualification_established','prior_correct_enrollment_performed','protected_bank_access','confirmation_eligible')
    fields={'kind','fixture_only','rows','parent_components','rows_per_batch','context_cap','maximum_generated_tokens','batches','metadata','origins','row_order_sha256','measurement_reference','prompt_reference','policy_reference','ledger_reference','source_pins','sha256',*closed}
    require(set(plan)==fields and plan['kind']=='SYNTH_ARITHMETIC_CAPTURE_PLAN001' and plan['fixture_only'] is False and all(plan[key] is False for key in closed),'exact closed actual capture-plan schema; no native/scoring authority')
    require(all(type(plan[key]) is int and plan[key]==want for key,want in [('rows',512),('parent_components',128),('rows_per_batch',64),('context_cap',512),('maximum_generated_tokens',32)]),'typed original capture denominators/bounds')
    for name in ('measurement','prompt','policy','ledger'):require(plan[name+'_reference']==refs[name],'plan exact original CPU file joins')
    require(plan['source_pins']=={'arithmetic_guard_DEV':[refs['prompt']['path'],refs['prompt']['sha256'],512]},'single exact arithmetic prompt source pin')
    require(prompts['native_scoring_authorized'] is False and oracle['oracle_exposed_CPU_only'] is True,'prompt-only policy and exposed CPU oracle separation')
    require(measurement['projection_kind']=='SYNTH_ARITHMETIC_TOKEN_PROJECTION001' and measurement['projection_schema_version']==1 and measurement['status']=='CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED' and measurement['expected_rows']==measurement['measured_rows']==512 and measurement['unmeasured_rows']==0 and measurement['overflows']==[] and measurement['failure'] is None and measurement['prompt_cap_tokens']==480 and measurement['context_cap_tokens']==512 and measurement['generation_reserve_tokens']==32,'corrected actual whole512 context measurement')
    require(measurement['policy_file']==refs['prompt'] and measurement['oracle_file']==refs['oracle'],'corrected projection original arrays')
    original={key:item for key,item in measurement.items() if key not in {'sha256','full_result_sha256','policy_file','oracle_file','projection_kind','projection_schema_version'}};original.update(policy_rows=prompts['rows'],oracle_rows_CPU_only=oracle['rows']);require(digest(original)==measurement['full_result_sha256']=='a01ca81b5174606b5e8e191f22ebc772ea48efda64f9c952776f1d02f2f45f6b','full original CPU result reconstructed from original arrays')
    require(all(measurement[key] is False for key in ('native_launch_authorized','native_scoring_authorized','training_admitted','M0_enrollment_performed','model_quality_established','population_changed')),'measurement preserves closed native/model claims')
    constants={}
    for node in ast.parse(raws['cpu_tokenizer_source']).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in {'ASSET_PINS','CHAT_TEMPLATE_PIN','EOS_ID','PAD_ID','DEPENDENCIES'}:constants[target.id]=ast.literal_eval(node.value)
    binding=measurement['binding'];recorded=values['tokenizer_receipt'];student=values['pilot']['student']
    require(binding['assets']==recorded['tokenizer_files'] and {key:item['sha256'] for key,item in binding['assets'].items()}==constants['ASSET_PINS'] and binding['chat_template_sha256']==recorded['chat_template_sha256']==constants['CHAT_TEMPLATE_PIN'],'exact recorded/source-derived fourasset/template binding')
    require(binding['model_dir']==recorded['checkpoint_path'] and binding['dependency_versions']==constants['DEPENDENCIES'] and binding['eos_token_id']==constants['EOS_ID']==151645 and binding['pad_token_id']==constants['PAD_ID']==151643 and binding['template_kwargs']=={'add_generation_prompt':True,'enable_thinking':False} and binding['context_cap']==512 and binding['framework_modules_imported']==[] and binding['local_only'] is True and binding['native_qualification'] is binding['training_admitted'] is False,'original measured tokenizer metadata, no live tokenizer/model claim')
    require(binding['binding_sources']=={refs[name]['path']:refs[name]['sha256'] for name in ('pilot','tokenizer_receipt')} and student['tokenizer_sha256']==constants['ASSET_PINS']['tokenizer.json'] and student['tokenizer_config_sha256']==constants['ASSET_PINS']['tokenizer_config.json'] and student['chat_template_sha256_recorded']==constants['CHAT_TEMPLATE_PIN'],'pilot/historical receipt source joins')
    require(len(prompts['rows'])==len(oracle['rows'])==len(population['rows'])==len(plan['metadata'])==len(plan['origins'])==512 and len(population['parents'])==128,'every original row and parent denominator')
    require(ledger['selected_parents']==population['parents'] and ledger['original_population_file_sha256']==refs['population']['sha256'],'ledger original selected parent population')
    parent_ids=[parent['parent_id'] for parent in population['parents']];row_ids=[row['row_id'] for row in population['rows']]
    require(len(set(parent_ids))==128 and len(set(row_ids))==512 and parent_ids==list(dict.fromkeys(row['parent_id'] for row in population['rows'])) and digest(parent_ids)==population['parent_order_sha256']==ledger['selected_parent_order_sha256']==measurement['parent_order_sha256'] and digest(row_ids)==population['row_order_sha256']==measurement['row_order_sha256']==plan['row_order_sha256'],'exact original row/parent order')
    require(measurement['population_sha256']==population['sha256'] and measurement['policy_sha256']==population['policy_sha256']==ledger['policy_sha256']==policy['sha256'],'single original policy/cohort')
    groups={};projected=[];origins=[]
    for index,(tokens,gold,row,meta,origin) in enumerate(zip(prompts['rows'],oracle['rows'],population['rows'],plan['metadata'],plan['origins'])):
        require(set(tokens)=={'row_id','parent_id','messages','prompt_sha256','input_ids','attention_mask','prompt_tokens','prompt_token_sha256'} and tokens['row_id']==gold['row_id']==row['row_id'] and tokens['parent_id']==gold['parent_id']==row['parent_id'],'complete current row token/oracle/population identity')
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt']}];require(tokens['messages']==messages,'original arithmetic system/user messages')
        chat=render(messages);require(hashlib.sha256(chat.encode()).hexdigest()==tokens['prompt_sha256'],'independently rendered nonthinking chat SHA')
        ids=tokens['input_ids'];require(type(ids) is list and ids and all(type(value) is int and value>=0 for value in ids) and tokens['prompt_tokens']==len(ids)<=480 and tokens['attention_mask']==[1]*len(ids) and tokens['prompt_token_sha256']==digest(ids),'actual original token arrays/dose/hash')
        want_meta={'row_id':row['row_id'],'component_id':row['parent_id'],'mechanism':'bounded_integer_arithmetic','variant':row['condition'],'rendering':row['rendering']};require(meta==want_meta,'source-derived arithmetic condition/rendering metadata')
        want_origin={'row_id':row['row_id'],'component_id':row['parent_id'],'message_sha256':digest(messages),'rendered_chat_sha256':tokens['prompt_sha256'],'prompt_token_sha256':digest(ids)};require(origin==want_origin,'exact measured message/render/token lineage')
        want_row={'row_id':row['row_id'],'prompt_sha256':tokens['prompt_sha256'],'prompt_ids':ids,'prompt_tokens':len(ids),'attention_mask':[1]*len(ids)};projected.append(want_row);origins.append(want_origin)
        groups.setdefault(row['parent_id'],[]).append((row['condition'],row['rendering'],digest(messages),tokens['prompt_sha256'],digest(ids)))
        completion=gold['completion_ids'];require(completion==gold['target_ids']+[151645] and 0<len(completion)<=32 and len(ids)+len(completion)<=512 and gold['input_ids']==ids+completion and gold['labels']==[-100]*len(ids)+completion and gold['attention_mask']==[1]*(len(ids)+len(completion)) and gold['total_tokens']==len(ids)+len(completion) and gold['completion_including_EOS_tokens']==len(completion),'CPU oracle prompt/completion/EOS positions only')
        require(gold['target']==canonical({'answer':row['answer']}) and gold['full_template_sha256']==hashlib.sha256((chat+gold['target']+'<|im_end|>\n').encode()).hexdigest(),'original CPU candidate oracle/template join; no model output scoring')
    require(Counter(row['parent_id'] for row in population['rows'])==Counter({parent:4 for parent in parent_ids}),'all128 original four-sibling parent groups')
    for parent,group in groups.items():require(len(group)==4 and {(item[0],item[1]) for item in group}=={(condition,rendering) for condition in (0,1) for rendering in ('expression_json','fully_parenthesized_infix')} and all(len({item[index] for item in group})==4 for index in (2,3,4)),'distinct original four-sibling messages/renderings/token arrays')
    require(set(plan['batches'])=={'arithmetic_guard_DEV'} and len(plan['batches']['arithmetic_guard_DEV'])==8,'exact eight batches')
    summaries=[]
    for number,batch in enumerate(plan['batches']['arithmetic_guard_DEV'],1):
        selected=projected[(number-1)*64:number*64];positions=sum(sum(len(row['prompt_ids'])+step for step in range(32)) for row in selected);zero=2*len(selected[0]['prompt_ids'])
        expected={'ordinal':number,'source_kind':'arithmetic_guard_DEV','start':(number-1)*64,'stop':number*64,'rows':selected,'source_packet_sha256':digest(selected),'maximum_capture_forwards':64*32,'maximum_capture_input_positions':positions,'maximum_zero_parity_forwards':2,'maximum_zero_parity_input_positions':zero,'maximum_total_forwards':64*32+2,'maximum_total_input_positions':positions+zero,'native_launch_authorized':False,'scoring_authorized':False}
        require(batch==expected and positions+zero==EXPECTED_MAX_POSITIONS[number-1],'exact source-order batch and independently summed greedy full-prefix dose')
        require(len({origin['message_sha256'] for origin in origins[(number-1)*64:number*64]})==len({origin['rendered_chat_sha256'] for origin in origins[(number-1)*64:number*64]})==len({origin['prompt_token_sha256'] for origin in origins[(number-1)*64:number*64]})==64,'all64 current prompts/messages/tokens distinct')
        summaries.append({key:expected[key] for key in ('ordinal','start','stop','source_packet_sha256','maximum_capture_forwards','maximum_capture_input_positions','maximum_zero_parity_forwards','maximum_zero_parity_input_positions','maximum_total_forwards','maximum_total_input_positions')})
    require(sum(row['maximum_total_forwards'] for row in summaries)==16400 and sum(row['maximum_total_input_positions'] for row in summaries)==2295832,'complete original512 maximum work denominator')
    return {'embedded_plan_body_sha256':plan['sha256'],'whole_sealed_plan_value_sha256':digest(plan),'full_CPU_measurement_result_sha256':measurement['full_result_sha256'],'rows':512,'parents':128,'batches':summaries,'maximum_forwards':16400,'maximum_input_positions':2295832,'all512_independent_chat_token_message_origin_joins':True,'all128_distinct_four_sibling_groups':True,'exact_8x64_original_order':True,'recorded_tokenizer_model_binding_metadata_joined':True,'live_tokenizer_or_model_binding_qualified':False}

def main():
    started=time.monotonic();values,before,raws=read_inputs();result=validate(values,before,raws)
    # Four bounded semantic counterexamples use no compiler imports or input mutations.
    rejected=[]
    for label in ('batch_dose','row_order','origin','authority'):
        candidate=json.loads(canonical(values));plan=candidate['plan']
        if label=='batch_dose':plan['batches']['arithmetic_guard_DEV'][0]['maximum_total_input_positions']-=1
        elif label=='row_order':plan['batches']['arithmetic_guard_DEV'][0]['rows'][0],plan['batches']['arithmetic_guard_DEV'][0]['rows'][1]=plan['batches']['arithmetic_guard_DEV'][0]['rows'][1],plan['batches']['arithmetic_guard_DEV'][0]['rows'][0]
        elif label=='origin':plan['origins'][0]['rendered_chat_sha256']='0'*64
        else:plan['native_launch_authorized']=True
        plan['sha256']=digest({key:item for key,item in plan.items() if key!='sha256'})
        candidate['root_receipt']['plan_reference']['value_sha256']=digest(plan);candidate['root_receipt']['sha256']=digest({key:item for key,item in candidate['root_receipt'].items() if key!='sha256'})
        try:validate(candidate,before,raws)
        except ValueError:rejected.append(label)
        else:raise AssertionError('semantic counterexample not rejected '+label)
    _,after,_=read_inputs();require(before==after,'all original input/source byte pins unchanged after independent audit')
    require(time.monotonic()-started<=600 and not FORBIDDEN,'bounded pure CPU audit')
    value=seal({'kind':'INDEPENDENT_ACTUAL_ARITHMETIC_CAPTURE_PLAN_CPU_AUDIT001','accepted':True,'input_references_before':before,'input_references_after':after,'input_bytes_unchanged':True,'audit_source':ref(Path(__file__).resolve(),Path(__file__).read_bytes()),'audit_seconds':time.monotonic()-started,'results':result,'authored_semantic_counterexamples_rejected':rejected,'forbidden_imports_or_APIs':FORBIDDEN,'read_paths':sorted(set(READS)),'tokenizer_framework_or_native_API_run':False,'M0_capture_or_scoring_performed':False,'prior_correct_enrollment_performed':False,'native_qualification_established':False,'quality_gain_established':False,'limitations':['Exact actual CPU prompt/token/source lineage and prospective eight-batch dose only; no native capture, resource admission, model parity or quality result.','All four tokenassets/template/model binding are recorded metadata/source-pin joins, not a fresh asset or installed dependency runtime rehash.','CPU oracle arrays are used only to verify original full-result and token/template boundaries; no model outputs are read or scored.']})
    output=HERE/'INDEPENDENT_ACTUAL_PLAN_REVIEW001.json'
    with output.open('xb') as stream:stream.write((canonical(value)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'path':str(output),'accepted':True,'body_sha256':value['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':512,'parents':128,'maximum_forwards':16400,'maximum_input_positions':2295832,'seconds':value['audit_seconds']}))
if __name__=='__main__':main()
