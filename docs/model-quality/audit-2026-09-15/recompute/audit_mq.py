"""Read-only offline MQ historical evidence audit; never loads models."""
import collections, hashlib, json, math, re, statistics
from pathlib import Path

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
BASE = ROOT.parent
OUT = BASE / 'model-quality-audit-20260915'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def seal_ok(x): return x['sha256']==hashlib.sha256(canonical({k:v for k,v in x.items() if k!='sha256'})).hexdigest()
inventory=[]
for p in sorted((BASE/'mq-runs').rglob('*')):
    if p.is_file(): inventory.append({'path':str(p.relative_to(BASE)), 'bytes':p.stat().st_size,'sha256':sha(p)})
plans={}
for d in sorted((BASE/'mq-runs').glob('prepared-*')):
    if not (d/'plan.json').exists(): continue
    p=read(d/'plan.json'); train=p['train']; dev=p['dev']
    plans[d.name]={'sha256':p['sha256'],'seal_valid':seal_ok(p),'selection_seed':p['design']['selection_seed'],
        'order_seed':p['design']['order_seed'],'train_rows':len(train),'processed_tokens':p['processed_tokens'],
        'whole_targets':sum(len(r['ids'])-1 for r in train),'response_targets':sum(r['answer_end']-r['answer_start']+1 for r in train),
        'groups':len(p['groups']),'group_sizes':[len(g) for g in p['groups']],
        'train_id_set_sha256':hashlib.sha256(canonical(sorted(r['id'] for r in train))).hexdigest(),
        'ordered_id_sha256':hashlib.sha256(canonical([r['id'] for r in train])).hexdigest(),
        'review_binds':read(d/'DATA_REVIEW_APPROVED.json')['plan_sha256']==p['sha256'],
        'raw_input_hashes_valid':all(sha(Path(v['path']))==v['sha256'] for v in p['inputs'].values()),
        'dev_rows':len(dev),'dev_scored':sum(r['evaluation'] is not None for r in dev),
        'answerability_false':sum(r['evaluation'] is not None and not r['evaluation']['answerable'] for r in dev),
        'mask_bounds_valid':all(0<r['answer_start']<r['answer_end']<len(r['ids']) for r in train+dev)}
cells={}
for d in sorted((BASE/'mq-runs').glob('cell-*')):
    if not (d/'result.json').exists():
        cells[d.name]={'failure':read(d/'failure.json')}; continue
    r=read(d/'result.json'); b=read(d/'baseline.json'); f=read(d/'final_dev.json'); steps=rows(d/'steps.jsonl')
    pairs=list(zip(b['records'], f['records'],strict=True))
    correct=lambda x:x['scores']['correct'] is True
    resource=[s['resource_before_update'] for s in steps]
    cells[d.name]={'arm':r['arm'],'checkout':r['binding']['checkout'],'plan_sha256':r['binding']['plan_sha256'],
        'code_sha256':r['binding']['code_sha256'],'result_hashes_valid':all(sha(d/n)==r[k] for n,k in [('baseline.json','baseline_sha256'),('final_dev.json','final_dev_sha256'),('steps.jsonl','steps_sha256')]),
        'progress':r['progress'],'baseline_correct':sum(correct(a) for a,b in pairs),'final_correct':sum(correct(b) for a,b in pairs),
        'scored':sum(a['scores']['correct'] is not None for a,b in pairs),
        'retained':sum(correct(a) and correct(b) for a,b in pairs),'gained':sum(not correct(a) and correct(b) for a,b in pairs),'lost':sum(correct(a) and not correct(b) for a,b in pairs),
        'baseline_ce':b['response_ce'],'final_ce':f['response_ce'],'generated_mean':statistics.mean(len(x['generated_ids']) for x in f['records']),
        'termination':sum(x['terminated'] for x in f['records']),'literal_unknown':sum(x['scores']['literal_unknown'] for x in f['records']),
        'finite_gradients_and_losses':all(math.isfinite(s['objective_ce']) and math.isfinite(s['grad_norm']) for s in steps),
        'lr_trace':[s['lr'] for s in steps],'row_order_sha256':hashlib.sha256(canonical([i for s in steps for i in s['row_ids']])).hexdigest(),
        'checkpoint_present':(d/r['checkpoint']).exists(), 'launch_present':(d/'launch.json').exists(),'supervisor_present':(d/'supervisor.json').exists(),
        'update_snapshot_max_gpu_allocated_bytes':max(x['gpu0']['peak'] for x in resource),
        'update_snapshot_max_process_rss_bytes':max(x['rss_bytes'] for x in resource),
        'observed_update_interval_seconds':resource[-1]['time_unix']-resource[0]['time_unix'],
        'cost_caveat':'Snapshots are not whole-run host peak or full wall time; inference latency absent.'}
quals={d.name:{'seal_valid':seal_ok(q),'checkout':q['binding']['checkout'],'plan_sha256':q['binding']['plan_sha256'],'checks':q['checks'],'metrics':q['metrics'],'optimizer_steps':q['optimizer_steps']} for d in sorted((BASE/'mq-runs').glob('qualification-*')) if (d/'qualification.json').exists() for q in [read(d/'qualification.json')]}
train=rows(BASE/'mq-data-003/train.jsonl'); dev=rows(BASE/'mq-data-003/dev.jsonl'); wanted={r['id'] for r in train+dev}
meta={}
source=ROOT/'Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft'
for name in ('rineya_v3_reasoning_train_50000.jsonl','rineya_v3_natural_communication_train_30000.jsonl','rineya_v3_integrated_reasoning_dialogue_train_30000.jsonl'):
    with (source/name).open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['id'] in wanted: meta[r['id']]=r['metadata']
metadata_keys=sorted(set().union(*(m.keys() for m in meta.values())))
def values(bank,key):return {str(meta[r['id']].get(key)) for r in bank if meta[r['id']].get(key) is not None}
overlap={k:{'train_unique':len(values(train,k)),'dev_unique':len(values(dev,k)),'shared':sorted(values(train,k)&values(dev,k))} for k in metadata_keys if 'template' in k or 'family' in k or 'scenario' in k}
def user(r):return next(x['content'] for x in r['messages'] if x['role']=='user')
def grams(r):
    words=re.findall(r'\w+',user(r).casefold());return set(zip(words,words[1:],words[2:]))
tg=[grams(r) for r in train]
near=[]
for d in dev:
    g=grams(d)
    similarity,index=max((len(g&t)/max(1,len(g|t)),i) for i,t in enumerate(tg))
    near.append({'dev_id':d['id'],'train_id':train[index]['id'],'word_trigram_jaccard':similarity,'dev_prompt':user(d),'train_prompt':user(train[index]),'dev_answer':d['messages'][-1]['content'],'train_answer':train[index]['messages'][-1]['content']})
out={'scope':'OFFLINE_RECOMPUTATION_NO_MODEL_CALLS','inventory':inventory,'plans':plans,'cells':cells,'qualifications':quals,
    'corpus':{'metadata_rows_found':len(meta),'metadata_keys':metadata_keys,'family_metadata_overlap':overlap,
        'train_reply_classes':dict(collections.Counter(meta[r['id']]['reply_class'] for r in train)),
        'dev_reply_classes':dict(collections.Counter(meta[r['id']]['reply_class'] for r in dev)),
        'near_duplicate_diagnostics':sorted(near,key=lambda x:-x['word_trigram_jaccard']),
        'near_duplicate_caveat':'Candidate attack, not automatic semantic leakage verdict; no model outcomes used for subset selection.'}}
with (OUT/'mq-audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
print(json.dumps({'plans':plans,'cells':cells,'family_overlap':overlap,'near_top':out['corpus']['near_duplicate_diagnostics'][:3]},indent=2,ensure_ascii=False))
