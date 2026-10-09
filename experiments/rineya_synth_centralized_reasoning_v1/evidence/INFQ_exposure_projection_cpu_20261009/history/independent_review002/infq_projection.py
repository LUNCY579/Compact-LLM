"""Original INFQ causal-input reconstruction; stdlib only, no live inference.

Actual raw reads require an explicitly selected reviewed scope SHA. Prefixes
derive from pinned source recipe and durable traces, not independent per-call
token telemetry. No arithmetic family mapping or global exclusion is supplied.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,os,re,sys,time

AUDIT_SHA='b655ec5d883d398714564a0d72e2f987ec061489eaec803e49bb234513f7965e'
OUTPUT=Path('C:/Users/clclc/BusinessProject/synth-INFQ-exposure-projection-20261009-attempt001')
MAX_FILE=32*1024**2
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def parse(raw):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON field');out[k]=v
        return out
    def finite(s):
        v=float(s);require(math.isfinite(v),'nonfinite JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def read_bound(ref):
    require(type(ref['bytes']) is int and 0<=ref['bytes']<=MAX_FILE,'bounded declared original file')
    if ref['bytes']==0:require(ref.get('role')=='hash_only' and ref['sha256']==hashlib.sha256(b'').hexdigest(),'zero bytes only original hash-only empty log')
    p=Path(ref['path']);require(p.is_absolute() and p.is_file() and not p.is_symlink(),'regular selected original file')
    with p.open('rb') as f:
        require(os.fstat(f.fileno()).st_size==ref['bytes'],'physical original size before read')
        raw=f.read(ref['bytes']+1)
        require(os.fstat(f.fileno()).st_size==ref['bytes'],'physical original size after read')
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'original bytes changed')
    return raw
def token_ids(ids):
    require(type(ids) is list and ids and all(type(x) is int and 0<=x<151936 for x in ids),'typed original causal input IDs')
    return ids
def parent_membership(candidate_bytes,enrollment):
    fields={'row_id','parent_id','component_id','source_hash','arm','skill','split'}
    records=[{key:record[key] for key in fields} for record in (parse(line) for line in candidate_bytes.splitlines() if line)]
    require(len(records)==640 and len({r['row_id'] for r in records})==640,'complete640 candidate identity metadata')
    require({r['parent_id'] for r in records}==set(enrollment['parents']) and len(enrollment['parents'])==len(set(enrollment['parents']))==20,'actual20 enrolled source parents')
    for arm in ('M1','M2'):
        selected={r['row_id'] for r in records if r['arm']==arm}
        require(type(enrollment['record_ids'][arm]) is list and len(enrollment['record_ids'][arm])==len(set(enrollment['record_ids'][arm]))==320
            and all(type(rid) is str for rid in enrollment['record_ids'][arm])
            and len(selected)==320 and selected==set(enrollment['record_ids'][arm]),'actual320 per-arm source identity/enrollment')
    require(all(r['split']=='train' and r['skill'] in ('R','S') for r in records),'original atomic TRAIN roles')
    return {r['row_id']:r for r in records}
def journal_calls(events,audit):
    attempts=[];returned={};counts={};active=None;attempt_positions={};return_positions={};measurements={}
    for position,event in enumerate(events):
        kind=event.get('event')
        if kind=='attempt':
            require(type(event.get('id')) is int and event['id']==len(attempts)+1 and active is None,
                'typed strictly ordered global API call IDs, no nested attempts')
            require(type(event.get('operation')) is str,'typed original API operation')
            active=event;attempts.append(event);attempt_positions[event['id']]=position
            counts[event['operation']]=counts.get(event['operation'],0)+1
        elif kind in ('returned','failed'):
            require(kind=='returned' and active is not None and type(event.get('id')) is int
                and event['id']==active['id'] and event.get('operation')==active['operation'] and event['id'] not in returned,
                'one matching original API return, no duplicates/failures')
            returned[event['id']]=event;return_positions[event['id']]=position;active=None
        elif kind=='measurement_preserved':
            require(type(event.get('path')) is str and event['path'] not in measurements,'unique original measurement publication')
            measurements[event['path']]=position
    require(active is None and len(attempts)==len(returned)==audit['accounting']['returned_API_calls']==403
        and counts==audit['accounting']['registered_operation_counts'],'full original403-call/14-operation journal closure')
    return attempts,returned,attempt_positions,return_positions,measurements
def component_costs(saved,counts,positions):
    costs=saved['costs'];expected={operation:{'attempted':n,'returned':n,'failed':0,'pending':0} for operation,n in counts.items()}
    require(costs['operations']==expected and costs['attempted_forwards']==sum(counts.values())
        and costs['input_positions']=={'attempted':positions,'returned':positions,'failed':0,'pending':0},
        'exact cumulative component call/position cursor at durable publication')

def reconstruct(config,controls,events,stages,parity,worker,audit):
    rows=config['rows'];require(len(rows)==len(controls)==4,'four original TRAIN rows')
    require(config['source_packet_sha256']==digest(rows),'original four-row packet')
    for row in rows:
        token_ids(row['prompt_ids']);require(row['prompt_tokens']==len(row['prompt_ids']) and row['attention_mask']==[1]*len(row['prompt_ids']),'original full prompt/mask')
    intended=[]
    for operation in ('zero_parity_base_forward','zero_parity_adapter_forward'):
        intended.append({'operation':operation,'row_id':rows[0]['row_id'],'repeat':None,'step':None,'input_ids':list(rows[0]['prompt_ids']),
            'origin':'original_worker_config_row0_parity','durable_stage':None})
    for ordinal,(row,control) in enumerate(zip(rows,controls),1):
        require(control['row_id']==row['row_id'] and control['source_packet_sha256']==config['source_packet_sha256']
            and control['prompt_ids_sha256']==digest(row['prompt_ids']) and len(control['traces'])==2,'original row/trace/prompt identity')
        traces=control['traces'];require(traces[0]==traces[1] and control['repeat_exact'] is True,'actual exact repeat traces')
        for repeat,trace in enumerate(traces):
            require(trace and len(trace)<=32,'bounded actual trace')
            for step,value in enumerate(trace):
                prefix=row['prompt_ids']+[x['token'] for x in trace[:step]];token_ids(prefix)
                require(type(value['token']) is int and 0<=value['token']<151936
                    and value['prefix_length']==len(prefix),'source-recipe prefix length/token')
                require(type(value['logprob']) in (int,float) and math.isfinite(value['logprob']) and value['logprob']<=0
                    and not any(x['token'] in (151643,151645) for x in trace[:step]),'finite original selected LP and first EOS boundary')
                name='stage_inference_q%04d_r%d_t%02d.json'%(ordinal,repeat,step);saved=stages[name]
                require(saved['row_id']==row['row_id'] and saved['repeat']==repeat and saved['step']==step
                    and saved['token']==value['token'] and saved['prefix_length']==len(prefix)
                    and saved['logprob']==value['logprob'] and saved['prompt_ids_sha256']==digest(row['prompt_ids'])
                    and saved['input_row_sha256']==digest(row) and saved['source_packet_sha256']==config['source_packet_sha256'],
                    'exact original durable step/trace/source identity join')
                intended.append({'operation':'inference_qualification_forward','row_id':row['row_id'],'repeat':repeat,'step':step,
                    'input_ids':prefix,'origin':'original_trace_prompt_plus_same_repeat_prior_generated_tokens','durable_stage':name})
        prefix=row['prompt_ids']+[x['token'] for x in traces[0][:-1]]
        intended.append({'operation':'inference_qualification_bulk_forward','row_id':row['row_id'],'repeat':0,'step':None,
            'input_ids':prefix,'origin':'original_bulk_prompt_plus_repeat0_excluding_last_selected_token','durable_stage':'stage_inference_qualification_%04d.json'%ordinal})
    operations={'zero_parity_base_forward','zero_parity_adapter_forward','inference_qualification_forward','inference_qualification_bulk_forward'}
    all_attempted,all_returned,attempt_positions,return_positions,measurements=journal_calls(events,audit)
    attempted=[e for e in all_attempted if e['operation'] in operations]
    require(len(attempted)==len(intended)==84,'every original84 forward attempt/return')
    projection=[];component_counts={};component_positions=0
    for ordinal,(recipe,event) in enumerate(zip(intended,attempted),1):
        require(event['operation']==recipe['operation'] and all_returned[event['id']]['operation']==recipe['operation'],
            'exact actual journal call ordinal/operation return join')
        if not recipe['operation'].startswith('zero_parity'):
            component_counts[recipe['operation']]=component_counts.get(recipe['operation'],0)+1
            component_positions+=len(recipe['input_ids'])
            name=recipe['durable_stage'];saved=stages[name]
            component_costs(saved,component_counts,component_positions)
            next_position=attempt_positions[attempted[ordinal]['id']] if ordinal<len(attempted) else len(events)
            require(name in measurements and return_positions[event['id']]<measurements[name]<next_position,
                'durable token/control publication after matching return and before next forward')
        projection.append({**recipe,'forward_ordinal':ordinal,'journal_call_id':event['id'],'input_positions':len(recipe['input_ids']),
            'attention_mask':[1]*len(recipe['input_ids']),'model_input_ids_sha256':digest(recipe['input_ids']),
            'arm_membership':'UNKNOWN_NOT_INFERRED_FROM_ID','parent_membership':None,'arithmetic_family_mapping':'UNDEFINED'})
    counts={op:sum(r['operation']==op for r in projection) for op in operations}
    positions={op:sum(r['input_positions'] for r in projection if r['operation']==op) for op in operations}
    accounting=audit['accounting'];dose=worker['forward_dose']
    require(len(projection)==accounting['actual_forward_calls']==dose['attempted_calls']
        and sum(r['input_positions'] for r in projection)==accounting['actual_input_positions']==dose['input_positions'],'actual audit/worker physical dose closure')
    for op in operations:
        require(dose['operations'][op]=={'attempted_calls':counts[op],'input_positions':positions[op]},'original per-operation physical dose')
    require(parity['zero_adapter_output_equal'] is True and parity['forward_dose']['attempted_calls']==2,'original parity receipt')
    require((len(projection),sum(r['input_positions'] for r in projection))==(84,28627)
        and counts['inference_qualification_forward']==78 and positions['inference_qualification_forward']==26544
        and counts['inference_qualification_bulk_forward']==4 and positions['inference_qualification_bulk_forward']==1245,
        'fixed actual INFQ84/28627 role totals')
    value={'kind':'ORIGINAL_INFQ_SOURCE_RECIPE_CAUSAL_INPUT_PROJECTION001','calls':projection,'operation_counts':counts,
        'operation_input_positions':positions,'total_calls':len(projection),'total_input_positions':sum(r['input_positions'] for r in projection),
        'input_values_are_independently_recorded_per_call':False,'source_recipe_and_recorded_trace_reconstruction':True,
        'global_exposure_coverage':False,'R_S_to_arithmetic_family_mapping':'UNDEFINED','model_GPU_or_scoring_performed':False}
    value['sha256']=digest(value);return value

def run(scope_path,expected_scope_sha256):
    before=Path(scope_path).read_bytes();require(hashlib.sha256(before).hexdigest()==expected_scope_sha256,'independently selected reviewed read scope')
    scope=parse(before);require(scope['sha256']==digest({k:v for k,v in scope.items() if k!='sha256'}),'scope seal')
    require(scope['audit_reference']['sha256']==AUDIT_SHA and Path(scope['output']).is_absolute()
        and Path(scope['output']).resolve()==OUTPUT.resolve(),'fixed original audit/output')
    require(OUTPUT.parent.is_dir() and not OUTPUT.exists() and not OUTPUT.is_symlink(),'fresh owned projection output')
    started=time.monotonic();allowed={**scope['raw_references'],**scope['public_source_references'],**scope.get('metadata_references',{}),
        scope['audit_reference']['path']:scope['audit_reference'],scope['config_reference']['path']:scope['config_reference'],
        scope['preservation_reference']['path']:scope['preservation_reference']}
    require(sum(ref['bytes'] for ref in allowed.values())<=scope['limits']['total_payload_bytes'],'whole bounded read allowance')
    audit=parse(read_bound(scope['audit_reference']))
    declared={Path(path).name:{k:ref[k] for k in ('bytes','sha256')} for path,ref in scope['raw_references'].items()}
    require(declared==audit['hash_closure']['raw_preserved_hashes'],'exact all95 actual-audit raw filename/byte/hash commitments')
    raw={}
    for path,ref in allowed.items():
        require(time.monotonic()-started<=600,'CPU projection wall budget');raw[path]=read_bound({'path':path,**ref})
    audit=parse(raw[scope['audit_reference']['path']]);config=parse(raw[scope['config_reference']['path']]);pres=parse(raw[scope['preservation_reference']['path']])
    require(audit['hash_closure']['config_body_sha256']==config['sha256'] and config['sha256']==digest({k:v for k,v in config.items() if k!='sha256'}),'actual config/audit body join')
    require(pres['sha256']==digest({k:v for k,v in pres.items() if k!='sha256'}),'actual preservation body seal')
    require(len(scope['raw_references'])==audit['hash_closure']['raw_file_count']==95,'all95 byte commitments')
    docs={Path(path).name:(parse(data) if scope['raw_references'][path]['role']=='INFQ_original_JSON'
        else [parse(line) for line in data.splitlines() if line] if scope['raw_references'][path]['role']=='INFQ_original_JSONL' else None)
        for path,data in raw.items() if path in scope['raw_references']}
    require(all(pres['raw_files'][name]==entry for name,entry in audit['hash_closure']['raw_preserved_hashes'].items()),'original preservation/raw full byte/hash closure')
    worker=docs['worker_result.json'];controls=[docs['stage_inference_qualification_%04d.json'%i] for i in range(1,5)]
    require(worker['numerical_controls']==controls,'actual worker/four durable control stages')
    for event in docs['events.jsonl']:
        if event.get('event')=='measurement_preserved':
            require(event['path'] in declared and event['sha256']==declared[event['path']]['sha256'],'journal measurement actual byte link')
    result=reconstruct(config,controls,docs['events.jsonl'],docs,docs['stage_zero_adapter_parity.json'],worker,audit)
    if scope.get('metadata_references'):
        metadata_refs=scope['metadata_references'];candidate_path=next(path for path in metadata_refs if Path(path).name=='candidate_records.jsonl')
        enrollment_path=next(path for path in metadata_refs if Path(path).name=='enrollment.json')
        prep_path=next(path for path in metadata_refs if Path(path).name=='preparation_receipt.json')
        prep=parse(raw[prep_path]);enrollment=parse(raw[enrollment_path])
        for path in (candidate_path,enrollment_path):
            require(prep['files'][Path(path).name]=={k:metadata_refs[path][k] for k in ('bytes','sha256')},'preparation exact source metadata byte commitments')
        identities=parent_membership(raw[candidate_path],enrollment)
        for call in result['calls']:
            require(call['row_id'] in identities,'observed original TRAIN row identity in enrolled source metadata')
            call['original_source_membership']=identities[call['row_id']]
            call['arm_membership']=identities[call['row_id']]['arm'];call['parent_membership']=identities[call['row_id']]['parent_id']
        result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'})
    for path,ref in allowed.items():require(read_bound({'path':path,**ref})==raw[path],'original byte identities after projection')
    require(Path(scope_path).read_bytes()==before,'scope bytes unchanged')
    OUTPUT.mkdir();result_raw=(canonical(result)+'\n').encode();result_path=OUTPUT/'INFQ_CAUSAL_INPUT_PROJECTION001.json'
    with result_path.open('xb') as f:require(f.write(result_raw)==len(result_raw),'complete projection write');f.flush();os.fsync(f.fileno())
    result_ref={'path':str(result_path),'bytes':len(result_raw),'sha256':hashlib.sha256(result_raw).hexdigest(),'role':'new_projection_output'}
    require(read_bound(result_ref)==result_raw and time.monotonic()-started<=600,'projection durable readback and whole CPU write budget')
    receipt={'read_scope_sha256':expected_scope_sha256,'original_file_count':len(allowed),'original_inputs':allowed,
        'result_reference':result_ref,'result_body_sha256':result['sha256'],
        'all95_original_byte_links_verified':True,'source_recipe_reconstruction_verified':True,'wall_seconds':time.monotonic()-started,
        'global_exposure_coverage':False,'model_GPU_or_scoring_performed':False}
    for name,value in (('PROJECTION_RECEIPT001.json',receipt),):
        with (OUTPUT/name).open('xb') as f:f.write((canonical(value)+'\n').encode());f.flush();os.fsync(f.fileno())
    require(time.monotonic()-started<=600 and Path(scope_path).read_bytes()==before,'whole CPU projection/write budget and immutable scope')
    for path,ref in allowed.items():require(read_bound({'path':path,**ref})==raw[path],'original source/raw/metadata bytes remain stable after outputs')
    return receipt

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scope',required=True);parser.add_argument('--scope-sha256',required=True)
    args=parser.parse_args();print(canonical(run(args.scope,args.scope_sha256)))
