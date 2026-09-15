import collections,json
from pathlib import Path
B=Path('C:/Users/clclc/BusinessProject'); R=B/'ChatBubble-ced-native-loader-qualification'
def rows(p):
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f]
train=rows(B/'mq-data-003/train.jsonl'); dev=rows(B/'mq-data-003/dev.jsonl')
wanted={r['id'] for r in train+dev}; meta={}
for name in ('rineya_v3_reasoning_train_50000.jsonl','rineya_v3_natural_communication_train_30000.jsonl','rineya_v3_integrated_reasoning_dialogue_train_30000.jsonl'):
    with (R/'Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft'/name).open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in wanted:meta[r['id']]=r['metadata']
out={}
for key in ('source_base_template_family','source_index'):
    by=collections.defaultdict(list)
    for r in train:
        m=meta[r['id']];p=m.get('semantic_payload',{});v=p.get(key)
        if v is not None:by[(m['reply_class'],str(v))].append(r['id'])
    hits=[]
    for r in dev:
        m=meta[r['id']];p=m.get('semantic_payload',{});v=p.get(key); k=(m['reply_class'],str(v))
        if v is not None and k in by:hits.append({'dev_id':r['id'],'class':m['reply_class'],'value':v,'train_ids':by[k]})
    out[key]={'dev_with_shared_source_metadata':len(hits),'pairs':hits}
out['interpretation']='Declared v3 family IDs are disjoint; these shared underlying generator identities show that this is not a generator-family-disjoint transfer instrument. Shared source index is a diagnostic of reused base generators, not proof of identical token sequences.'
with (B/'model-quality-audit-20260915/corpus-payload-audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps({k:{a:b for a,b in v.items() if a!='pairs'} if isinstance(v,dict) else v for k,v in out.items()},indent=2))
