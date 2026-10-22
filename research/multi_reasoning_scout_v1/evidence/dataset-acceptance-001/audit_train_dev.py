"""P0 audit restricted to explicit train/dev payloads; never a whole-package verifier."""
import collections, hashlib, importlib.metadata, itertools, json, math, os, re, sys, tempfile, time, unicodedata
from pathlib import Path

ROOT=Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\Rineya_LCR_Full_v1_6000')
OUT=Path(__file__).resolve().parent
TOK=Path(r'C:\Users\clclc\BusinessProject\SupaExperiment\Rineya-Supa-A1.1')
SPLITS=('train','dev')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
tempfile.tempdir=str(OUT)
import check
import audit
import tokenizer_preflight

readlog=[]
def read(rel):
    allowed={'MANIFEST.json','README.md','QUARANTINE_AND_CLAIMS.json','STATUS.json','DATA_CARD.md','REPRODUCE.md',
      'provenance/ATTACHED_CONTRACT.json','provenance/ATTACHED_SPEC.md','provenance/TARGET_TOKENIZER_LOCK.json','provenance/PARTITIONS.json',
      'reports/acceptance.json','reports/leakage.json','reports/token_profile.json','reports/RUN_BINDING.json',
      'src/check.py','src/audit.py','src/tokenizer_preflight.py','src/native_census.py','src/generate.py'}
    allowed|={f'{folder}/{split}.jsonl' for folder in ('data','sft','worlds') for split in SPLITS}
    if rel not in allowed:raise ValueError('Nonallowlisted input: '+rel)
    p=ROOT/rel
    if p.is_symlink():raise ValueError('Symlink input')
    raw=p.read_bytes();readlog.append({'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    return raw

def save(name,value):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def rows(rel):return [json.loads(line) for line in read(rel).decode('utf-8').splitlines() if line]
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())
def profile(v):return {'min':min(v),'max':max(v),'mean':sum(v)/len(v),'p50':sorted(v)[len(v)//2]}

started=time.time()
manifest=json.loads(read('MANIFEST.json'))
metadata={p:json.loads(read(p)) for p in ('provenance/ATTACHED_CONTRACT.json','provenance/TARGET_TOKENIZER_LOCK.json','reports/acceptance.json','reports/leakage.json','reports/token_profile.json','reports/RUN_BINDING.json','QUARANTINE_AND_CLAIMS.json')}
for p in ('README.md','STATUS.json','DATA_CARD.md','REPRODUCE.md','provenance/ATTACHED_SPEC.md','provenance/PARTITIONS.json','src/check.py','src/audit.py','src/tokenizer_preflight.py','src/native_census.py','src/generate.py'):read(p)
data={s:rows(f'data/{s}.jsonl') for s in SPLITS}
sft={s:rows(f'sft/{s}.jsonl') for s in SPLITS}
worlds={s:rows(f'worlds/{s}.jsonl') for s in SPLITS}
allrows=sum(data.values(),[])
integrity=[{**item,'manifest_match':item['sha256']==manifest['files'][item['path']]['sha256'] and item['bytes']==manifest['files'][item['path']]['bytes']} for item in readlog if item['path']!='MANIFEST.json']
if not all(x['manifest_match'] for x in integrity):raise ValueError('Manifest mismatch; do not continue')
print('Explicit train/dev payload integrity passed.',flush=True)
verification=[];errors=[];semantic={};question_tokens={};world_by_id={w['id']:w for ws in worlds.values() for w in ws}
for r in allrows:
    try:
        v=check.validate_row(r);q=check.query_from_text(r['messages'][1]['content']);d=check.parse_record(q['record'])
        roots=d['selectors'].get((q['color'],q['site']),[])
        semantic[r['id']]=int(not roots or (len(roots)==1 and (roots[0] in d['withdrawn'] or roots[0] in d['nonissued'])))
        question_tokens[r['id']]=re.findall(r'[a-z]+',q['question'].lower())
        verification.append({'id':r['id'],'split':r['split'],'parsed_facts':v['parsed_facts'],'answer_match':True,'reason_match':True})
    except Exception as e:errors.append({'id':r['id'],'error':str(e)})
save('train_dev_verification.json',{'rows':len(allrows),'checked':len(verification),'errors':errors,'parsed_facts':sum(x['parsed_facts'] for x in verification),'independence':'Separate supplied text parser/solver; shared author and grammar with generator; not independent human certification'})
print('Text-only rederivation completed.',flush=True)
splitstats={};sets={};forms={};entitysets={};layouts={}
for s in SPLITS:
    rs=data[s];ids=[r['id'] for r in rs];groups=[r['group_id'] for r in rs]
    splitstats[s]={'rows':len(rs),'unique_ids':len(set(ids)),'unique_groups':len(set(groups)),
      'role_sequence_errors':sum([m['role'] for m in r['messages']]!=['system','user','assistant'] for r in rs),
      'sft_projection_exact':sft[s]==[{'messages':r['messages']} for r in rs],
      'world_rows':len(worlds[s]),'world_ids_match':{w['id'] for w in worlds[s]}==set(ids),
      'marginals':{d:dict(collections.Counter(r['metadata'][d] for r in rs)) for d in ('task_type','variant','difficulty','needle_position','length_band')},
      'unknown_count':sum(r['messages'][-1]['content']=='UNKNOWN' for r in rs),
      'unanswerability_reasons':dict(collections.Counter(r['metadata']['unanswerable_reason'] for r in rs if r['messages'][-1]['content']=='UNKNOWN')),
      'max_answer_words':max(len(r['messages'][-1]['content'].split()) for r in rs)}
    sets[s]={'ids':set(ids),'groups':set(groups),'templates':{r['metadata']['template_id'] for r in rs}}
    for field in ('context','question'):
        vals=[check.query_from_text(r['messages'][1]['content'])['record' if field=='context' else 'question'] for r in rs]
        for mode,func in [('exact',lambda x:x),('nfkc',norm),('digit_blinded',lambda x:re.sub(r'\d','#',norm(x)))]:sets[s][field+'_'+mode]={sha_text(func(v)) for v in vals}
    entitysets[s]=set(itertools.chain.from_iterable(w['entity_names'] for w in worlds[s]));forms[s]=set();layouts[s]=set()
    for r in rs:
        q=check.query_from_text(r['messages'][1]['content']);ent=set(world_by_id[r['id']]['entity_names'])
        for line in q['record'].splitlines():forms[s].add(re.sub(r'\b\d+\b','<NUMBER>',' '.join('<ENTITY>' if w in ent else w.lower() for w in re.findall(r'[A-Za-z]+|\d+',line))))
    for w in worlds[s]:
        g=w['dependency_graph'];node={n['id']:n['kind'] for n in g['nodes']}
        layouts[s].add(json.dumps({'nodes':sorted(collections.Counter(node.values()).items()),'edges':sorted(collections.Counter((node[a],node[b]) for a,b in g['edges']).items())},sort_keys=True))
overlap={k:len(sets['train'][k]&sets['dev'][k]) for k in sets['train']}
overlap.update({'entity_names':len(entitysets['train']&entitysets['dev']),'name_blind_sentence_forms':len(forms['train']&forms['dev']),'typed_graph_count_signature':len(layouts['train']&layouts['dev'])})
save('train_dev_structure.json',{'splits':splitstats,'train_dev_overlap':overlap,'scope':'Train/dev only; typed graph-count signatures do not prove structural isomorphism exclusion. Shared local forms violate a blanket template-disjointness claim.'})

ys={s:[int(r['messages'][-1]['content']=='UNKNOWN') for r in data[s]] for s in SPLITS}
counts=[collections.Counter(),collections.Counter()];totals=[0,0]
for r,y in zip(data['train'],ys['train']):counts[y].update(question_tokens[r['id']]);totals[y]+=len(question_tokens[r['id']])
vocab=set(counts[0])|set(counts[1]);V=len(vocab)
def nb(r):
    terms=question_tokens[r['id']]
    score=[sum(math.log((counts[c][w]+1)/(totals[c]+V)) for w in terms if w in vocab) for c in (0,1)]
    return int(score[1]>score[0])
question_ba={s:audit.ba(ys[s],[nb(r) for r in data[s]]) for s in SPLITS}
semantic_ba={s:audit.ba(ys[s],[semantic[r['id']] for r in data[s]]) for s in SPLITS}
surface={s:[audit.surface_features(r) for r in data[s]] for s in SPLITS};thresholds={}
for feature in surface['train'][0]:
    spec=audit.best_threshold([x[feature] for x in surface['train']],ys['train'])
    spec['balanced_accuracy_by_split']={s:audit.ba(ys[s],audit.threshold_predict([x[feature] for x in surface[s]],spec)) for s in SPLITS};thresholds[feature]=spec
winner=max(thresholds,key=lambda k:thresholds[k]['balanced_accuracy'])
majority=collections.Counter(r['messages'][-1]['content'] for r in data['train']).most_common(1)[0][0]
baselines={}
for s in SPLITS:
    gold=[r['messages'][-1]['content'] for r in data[s]];pred={'majority':[majority]*len(gold),'always_unknown':['UNKNOWN']*len(gold),'first_entity':[],'last_entity':[]}
    for r in data[s]:
        ent=set(world_by_id[r['id']]['entity_names']);ordered=[w for w in re.findall(r'\b[A-Z][a-z]+\b',check.query_from_text(r['messages'][1]['content'])['record']) if w in ent]
        pred['first_entity'].append(ordered[0] if ordered else 'UNKNOWN');pred['last_entity'].append(ordered[-1] if ordered else 'UNKNOWN')
    pred['copy_most_recent_entity']=pred['last_entity']
    baselines[s]={k:{'overall_accuracy':sum(a==b for a,b in zip(p,gold))/len(gold),'answerable_accuracy':sum(a==b and b!='UNKNOWN' for a,b in zip(p,gold))/sum(b!='UNKNOWN' for b in gold)} for k,p in pred.items()}
save('train_dev_shortcuts.json',{'question_only':{'classifier':'Laplace-smoothed multinomial naive Bayes; unigram question tokens; equal class priors; train fit; dev scored without refit','balanced_accuracy_by_split':question_ba},'semantic_selector_plus_validity':semantic_ba,'surface_thresholds':thresholds,'train_selected_feature':winner,'train_selected_feature_scores':thresholds[winner]['balanced_accuracy_by_split'],'exact_answer_baselines':baselines,'limitations':['Finite attacks only. Train scores are in-sample. Producer logistic/nearest-neighbor baselines not reexecuted because sklearn absent.','Cheap semantic screen solves answerability only, not exact answer strings.']})
print('Shortcut and split checks completed.',flush=True)
pre=tokenizer_preflight.qualify(TOK);save('tokenizer_preflight.json',pre)
measurements=[]
if pre['status']=='ASSETS_AND_TOKENIZER_SMOKE_PASS_ONLY':
    from transformers import Qwen2TokenizerFast
    with tempfile.TemporaryDirectory(prefix='tokenizer-',dir=OUT) as td:
        for name,expected in tokenizer_preflight.ASSETS.items():
            raw=tokenizer_preflight.read_regular_file(TOK/name)
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Tokenizer changed')
            (Path(td)/name).write_bytes(raw)
        tok=Qwen2TokenizerFast.from_pretrained(td,local_files_only=True,trust_remote_code=False)
        tokenizer_preflight.measure_smoke(tok)
        for i,r in enumerate(allrows):
            msgs=r['messages'];context=check.query_from_text(msgs[1]['content'])['record']
            c=tok.encode(context,add_special_tokens=False);p=tok.apply_chat_template(msgs[:2],tokenize=True,add_generation_prompt=True,enable_thinking=False);f=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=False)
            if f[:len(p)]!=p or f[len(p):].count(tok.eos_token_id)!=1:raise ValueError('Native template contract failed')
            band=r['metadata']['length_band'];lo,hi={'short':(300,600),'medium':(600,1100),'long':(1100,1600)}[band]
            measurements.append({'id':r['id'],'split':r['split'],'band':band,'context_tokens':len(c),'prompt_tokens':len(p),'total_tokens':len(f),'completion_tokens':len(f)-len(p),'prompt_le_1920':len(p)<=1920,'total_le_2048':len(f)<=2048,'band_fit':lo<=len(c)<=hi,'at_shared_endpoint':len(c) in (600,1100)})
            if (i+1)%1000==0:print(f'Tokenized {i+1} permitted rows.',flush=True)
save('native_train_dev_rows.json',measurements)
native={'status':'MEASURED_TRAIN_DEV_ONLY' if measurements else 'BLOCKED','rows':len(measurements),'rows_truncated':0,'rows_filtered':0,'holdout_rows_read':0,'band_policy':'Literal inclusive endpoints; shared-endpoint ambiguity separately counted, no new policy adopted.'}
if measurements:
    native.update({'prompt_tokens':profile([m['prompt_tokens'] for m in measurements]),'total_tokens':profile([m['total_tokens'] for m in measurements]),'prompt_over_1920':sum(not m['prompt_le_1920'] for m in measurements),'total_over_2048':sum(not m['total_le_2048'] for m in measurements),'outside_intended_band':sum(not m['band_fit'] for m in measurements),'at_shared_endpoint':sum(m['at_shared_endpoint'] for m in measurements),'by_split':{s:{'rows':sum(m['split']==s for m in measurements),'prompt_over_1920':sum(m['split']==s and not m['prompt_le_1920'] for m in measurements),'total_over_2048':sum(m['split']==s and not m['total_le_2048'] for m in measurements),'outside_intended_band':sum(m['split']==s and not m['band_fit'] for m in measurements)} for s in SPLITS}})
save('native_train_dev_profile.json',native)
save('partial_audit_receipt.json',{'status':'P0_STOP_NOT_ACCEPTED','dataset_root':str(ROOT),'output':str(OUT),'run_seconds':time.time()-started,'manifest_sha256':next(x['sha256'] for x in readlog if x['path']=='MANIFEST.json'),'explicit_input_inventory':readlog,'integrity':integrity,'source_audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'runtime':{p:importlib.metadata.version(p) for p in ('transformers','tokenizers','numpy')},'payload_splits_read':['train','dev'],'forbidden_payload_reads':0,'holdout_payload_reads':0,'holdout_payload_hashes_recomputed':0,'all_jsonl_read':False,'full_manifest_verified':False,'model_calls':0,'training_runs':0,'corpus_accepted':False,'semantic_errors':len(errors),'native':native,'question_only':question_ba,'semantic_shortcut':semantic_ba,'surface_train_selected':thresholds[winner],'overlap':overlap,'producer_acceptance_status':metadata['reports/acceptance.json']['status']})
print(json.dumps({'status':'P0_STOP_NOT_ACCEPTED','native':native,'semantic_errors':len(errors),'semantic_shortcut':semantic_ba,'surface_winner':winner,'surface_scores':thresholds[winner]['balanced_accuracy_by_split'],'overlap':overlap},indent=2),flush=True)
