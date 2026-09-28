from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,time
from datetime import datetime,timezone
started=time.perf_counter();root=Path.cwd();base=root/'artifacts/mimo_central/dev_binding_v1'
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def identity(p):
    raw=p.read_bytes();return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def read(p):return json.loads(p.read_bytes())
def rows(p):return [json.loads(x) for x in p.read_bytes().splitlines()]
r=read(base/'BINDING_RECEIPT.json');m=read(base/'STATIC_BINDING.json')
file_checks=0
for label,expected in r['files'].items():assert identity(base/label)==expected;file_checks+=1
for label,expected in r['source_files'].items():assert identity(root/label)==expected;file_checks+=1
for label,expected in m['bindings'].items():assert identity(root/label)==expected;file_checks+=1
p=rows(base/'policy/DEV_INPUTS.jsonl');e=rows(base/'evaluation/DEV_ROWS.jsonl')
assert len(p)==len(e)==832 and len({x['row_id'] for x in p})==832
full={**m,'policy_rows':p,'evaluator_rows':e};assert digest({k:v for k,v in full.items() if k!='data_sha256'})==m['data_sha256']==r['data_sha256']
sources={
 'acquisition':{x['row_id']:x for x in rows(root/'artifacts/mimo_central/bridge_tokenized_v1/DEV.tokenized.jsonl')},
 'deterministic_retention':{x['row_id']:x for x in rows(root/'artifacts/mimo_central/retention_tokenized_v1/DEV.tokenized.jsonl')},
 'long_context':{x['row_id']:x for x in rows(root/'artifacts/mimo_central/long_context_dev_v1/DEV.jsonl')},
 'language':{x['parent_id']+'-dev':x for x in rows(root/'artifacts/mimo_central/language_eval_tokenized_v1/DEV.prompts.jsonl')}}
counts=Counter();tokens=Counter();parents=defaultdict(list);target_positions_excluded=0
for policy,evaluator in zip(p,e):
    assert set(policy)=={'row_id','messages','input_ids','attention_mask','prompt_tokens','prompt_token_hash'}
    assert policy['row_id']==evaluator['row_id'] and evaluator['role']=='DEV'
    panel=evaluator['panel'];original=sources[panel][policy['row_id']];src=evaluator['scoring_payload']['source_row']
    assert digest(src)==evaluator['scoring_payload']['source_row_sha256']
    if panel in ('acquisition','deterministic_retention'):
        expected=original['input_ids'][:original['prompt_tokens']]
        assert len(original['input_ids'])>len(expected)
        target_positions_excluded+=len(original['input_ids'])-len(expected)
        assert digest(src)==original['source_row_sha256']
        assert digest(expected)==original['tokenized_prompt_sha256']
    elif panel=='long_context':
        expected=original['input_ids'];assert src==original
    else:
        expected=original['prompt_token_ids'];assert digest(src)==original['source_row_sha256']
    assert all(type(x) is int for x in policy['input_ids']+policy['attention_mask'])
    assert policy['input_ids']==expected and policy['prompt_tokens']==len(expected)
    assert policy['attention_mask']==[1]*len(expected)
    assert digest(expected)==policy['prompt_token_hash']==evaluator['prompt_token_hash']
    assert policy['messages']==(src['messages'] if panel=='language' else [{'role':'user','content':src['prompt']}])
    counts[panel]+=1;tokens[panel]+=len(expected);parents[evaluator['parent_id']].append(evaluator)
assert dict(counts)=={'acquisition':256,'deterministic_retention':256,'long_context':256,'language':64}
assert dict(tokens)=={'acquisition':22914,'deterministic_retention':80773,'long_context':780288,'language':6899}
assert len(parents)==544 and sum(tokens.values())==890874
for group in parents.values():
    assert len({(x['panel'],x['family'],x['stratum']) for x in group})==1
    panel=group[0]['panel'];variants={x['variant'] for x in group}
    if panel=='long_context':assert variants=={f'{n}-{a}_{d}' for n in (2024,4072) for a in ('answerable','unanswerable') for d in ('neutral','adversarial')} and len(group)==8
    elif panel=='deterministic_retention':assert variants==({'answerable','missing_evidence'} if group[0]['family']=='answerability_abstention' else {'standard'}) and len(group)==len(variants)
    else:assert len(group)==1
assert all(r[k] is False for k in ('native_launch_authorized','native_qualified','grading_qualified','language_scoring','training_eligibility','source_correct_membership'))
assert all(v is None for v in r['unresolved_native_registration'].values())
assert r['actual_response_records']==r['native_model_loads']==r['generated_model_tokens']==r['optimizer_steps']==0
out={'status':'PASS_SEPARATE_ARTIFACT_CHECKS_NO_NATIVE_CLAIM','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
 'rows_compared':832,'parent_groups':544,'file_identity_checks':file_checks,'prompt_tokens_by_panel':dict(tokens),
 'prompt_tokens_total':sum(tokens.values()),'short_reference_positions_excluded':target_positions_excluded,
 'findings':[],'binding_receipt':identity(base/'BINDING_RECEIPT.json'),'data_sha256':r['data_sha256'],
 'audit_code':identity(Path(__file__)),'audit_implementation':'Separate stdlib checks; no binder/scorer calls. Root also authored publisher; not independent human review.',
 'native_calls':0,'tokenizer_calls':0,'held_out_payloads_read':False,'native_qualification':False,
 'limits':'Checks stored prompt mappings and bytes, not fresh tokenization, source model behavior or qualified language grading.',
 'wall_seconds':time.perf_counter()-started}
target=root/'experiments/post_training_mimo_centralized_v1/receipts/DEV_BINDING_ARTIFACT_AUDIT.json'
with target.open('xb') as f:f.write(canon(out)+b'\n')
print(json.dumps(out))
