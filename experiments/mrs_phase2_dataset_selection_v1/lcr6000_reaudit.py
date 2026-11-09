"""Fresh all-6000 open-development audit; never reads protected evaluation banks."""
import collections, hashlib, importlib.metadata, itertools, json, math, os, re, sys, tempfile, time, unicodedata
from pathlib import Path

ROOT=Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\Rineya_LCR_Full_v1_6000')
OUT=Path(__file__).resolve().parents[2]/'research/mrs_phase2_dataset_selection_v1/evidence/lcr6000-reaudit-001'
OUT.mkdir(parents=True,exist_ok=False)
TOK=Path(r'C:\Users\clclc\BusinessProject\SupaExperiment\Rineya-Supa-A1.1')
SPLITS=('train','dev','holdout')
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
print('Explicit all-6000 open payload integrity passed.',flush=True)
verification=[];errors=[];semantic={};question_tokens={};world_by_id={w['id']:w for ws in worlds.values() for w in ws}
for r in allrows:
    try:
        v=check.validate_row(r);q=check.query_from_text(r['messages'][1]['content']);d=check.parse_record(q['record'])
        roots=d['selectors'].get((q['color'],q['site']),[])
        semantic[r['id']]=int(not roots or (len(roots)==1 and (roots[0] in d['withdrawn'] or roots[0] in d['nonissued'])))
        question_tokens[r['id']]=re.findall(r'[a-z]+',q['question'].lower())
        verification.append({'id':r['id'],'split':r['split'],'parsed_facts':v['parsed_facts'],'answer_match':True,'reason_match':True})
    except Exception as e:errors.append({'id':r['id'],'error':str(e)})
save('all_splits_verification.json',{'rows':len(allrows),'checked':len(verification),'errors':errors,'parsed_facts':sum(x['parsed_facts'] for x in verification),'independence':'Separate supplied text parser/solver; shared author and grammar with generator; not independent human certification'})
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
save('all_splits_structure.json',{'splits':splitstats,'all_splits_overlap':overlap,'scope':'All three open splits are loaded. Pairwise extension below distinguishes full typed graphs from shared local operator grammar.'})

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
save('all_splits_shortcuts.json',{'question_only':{'classifier':'Laplace-smoothed multinomial naive Bayes; unigram question tokens; equal class priors; train fit; dev and open holdout scored without refit','balanced_accuracy_by_split':question_ba},'semantic_selector_plus_validity':semantic_ba,'surface_thresholds':thresholds,'train_selected_feature':winner,'train_selected_feature_scores':thresholds[winner]['balanced_accuracy_by_split'],'exact_answer_baselines':baselines,'limitations':['Finite attacks only. Train scores are in-sample. Producer logistic/nearest-neighbor baselines not reexecuted because sklearn absent.','Cheap semantic screen solves answerability only, not exact answer strings.']})
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
save('native_all_splits_rows.json',measurements)
native={'status':'MEASURED_ALL_6000_OPEN_ROWS' if measurements else 'BLOCKED','rows':len(measurements),'rows_truncated':0,'rows_filtered':0,'holdout_rows_read':600,'band_policy':'Literal inclusive endpoints; shared-endpoint ambiguity separately counted, no new policy adopted.'}
if measurements:
    native.update({'prompt_tokens':profile([m['prompt_tokens'] for m in measurements]),'total_tokens':profile([m['total_tokens'] for m in measurements]),'prompt_over_1920':sum(not m['prompt_le_1920'] for m in measurements),'total_over_2048':sum(not m['total_le_2048'] for m in measurements),'outside_intended_band':sum(not m['band_fit'] for m in measurements),'at_shared_endpoint':sum(m['at_shared_endpoint'] for m in measurements),'by_split':{s:{'rows':sum(m['split']==s for m in measurements),'prompt_over_1920':sum(m['split']==s and not m['prompt_le_1920'] for m in measurements),'total_over_2048':sum(m['split']==s and not m['total_le_2048'] for m in measurements),'outside_intended_band':sum(m['split']==s and not m['band_fit'] for m in measurements)} for s in SPLITS}})
save('native_all_splits_profile.json',native)

# Independent re-audit extensions: graph invariants, full templates, feasibility, and open scout exclusion.
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(value):return sha_text(canonical(value))
def renamed(value,entities,mapping=None):
    mapping={} if mapping is None else mapping
    if isinstance(value,dict):return {k:renamed(value[k],entities,mapping) for k in sorted(value) if k not in ('depth','id','split','group_id','allocation_id','generator_version')}
    if isinstance(value,list):return [renamed(v,entities,mapping) for v in value]
    if isinstance(value,str) and value in entities:
        if value not in mapping:mapping[value]='ENTITY'+str(len(mapping))
        return mapping[value]
    return value

def shape_counts(value,key='root',out=None):
    out=collections.Counter() if out is None else out
    if isinstance(value,dict):
        out['object:'+key]+=1
        for k,v in value.items():
            if k not in ('depth','root','task','color','site','reason'):shape_counts(v,k,out)
    elif isinstance(value,list):
        out['list:'+key]+=1
        for v in value:shape_counts(v,key,out)
    elif value is None:out['leaf:'+key+':missing']+=1
    elif isinstance(value,bool):out['leaf:'+key+':bool']+=1
    else:out['leaf:'+key+':value']+=1
    return out

graph_by_split={};whole_by_split={};whole_template={};reduced_counts={};graph_rows=[]
for s in SPLITS:
    graph_by_split[s]=set();whole_by_split[s]=set();whole_template[s]=set();reduced_counts[s]=set()
    for w in worlds[s]:
        g=w['dependency_graph'];kind={n['id']:n['kind'] for n in g['nodes']}
        # Different typed node/edge multisets PROVE non-isomorphism; matching invariants alone would not prove isomorphism.
        invariant={'nodes':sorted(collections.Counter(kind.values()).items()),'edges':sorted(collections.Counter((kind[a],kind[b]) for a,b in g['edges']).items())}
        graph_by_split[s].add(canonical(invariant))
        # Actual complete cases, without depth annotations, plus actual query: structural-count nonmatches prove
        # non-isomorphism for this key-labelled symbolic-world schema, independent of entity names.
        counts=shape_counts({'query':w['query'],'cases':w['cases']})
        whole_invariant={'task':w['query']['task'],'schema_counts':sorted(counts.items())}
        whole_by_split[s].add(canonical(whole_invariant))
        red=digest(renamed({'query':w['query'],'cases':w['cases']},set(w['entity_names'])))
        reduced_counts[s].add(red)
        graph_rows.append({'id':w['id'],'split':s,'registered_graph_invariant':digest(invariant),'whole_symbolic_invariant':digest(whole_invariant),'full_symbolic_entity_renamed_hash':red})
    for r in data[s]:
        ent=set(world_by_id[r['id']]['entity_names']);mapping={}
        text=r['messages'][1]['content']
        def convert(m):
            token=m.group()
            if token in ent:
                if token not in mapping:mapping[token]='ENTITY'+str(len(mapping))
                return mapping[token]
            return token.lower()
        whole_template[s].add(sha_text(re.sub(r'\b[A-Za-z]+\b',convert,norm(text))))
structure_pairs={}
for a,b in itertools.combinations(SPLITS,2):
    structure_pairs[a+'__'+b]={**{k:len(sets[a][k]&sets[b][k]) for k in sets[a]},
        'entity_name_intersection':len(entitysets[a]&entitysets[b]),'local_sentence_form_intersection':len(forms[a]&forms[b]),
        'registered_typed_graph_invariant_intersection':len(graph_by_split[a]&graph_by_split[b]),
        'whole_actual_symbolic_case_invariant_intersection':len(whole_by_split[a]&whole_by_split[b]),
        'full_symbolic_entity_renamed_hash_intersection':len(reduced_counts[a]&reduced_counts[b]),
        'full_entity_renamed_user_template_intersection':len(whole_template[a]&whole_template[b])}
structure={'status':'MEASURED_ALL_OPEN_SPLITS','pairwise':structure_pairs,
    'method':'Registered typed graphs: node-kind and typed-edge multiplicities. Full actual case structures: key-labelled object/list/leaf multiplicities excluding IDs, names and depth annotations. Zero intersection of necessary graph invariants proves no isomorphism under the defined typed/key-labelled representation; this does not require a canonical-labelling library.',
    'template_interpretation':'Split-assigned selectors/domain/filler templates and full entity-renamed user texts are distinct. Shared local task sentences and operator grammar are explicitly allowed by PARTITIONS.shared; 392 train/dev local forms alone are not proof that full scenario templates collide.',
    'limits':'Shared mathematical operators and reduced solution motifs remain. Full-text template hashes do not prove that every possible semantic abstraction is distinct. No claim of wholly novel operators or external protected-bank exclusion.',
    'invariant_counts':{s:{'registered':len(graph_by_split[s]),'whole_symbolic':len(whole_by_split[s]),'entity_renamed_worlds':len(reduced_counts[s]),'full_user_templates':len(whole_template[s])} for s in SPLITS}}
save('structural_reaudit.json',structure);save('structural_row_hashes.json',graph_rows)

balance=audit.balance(allrows)
feasibility=[]
for band,tables in balance['within_band_pair_tables'].items():
    for pair,cells in tables.items():
        for c in cells:
            e=c['proportional_expectation'];low=math.ceil(.97*e-1e-10);high=math.floor(1.03*e+1e-10)
            feasible=low<=high
            best=min(abs(math.floor(e)-e),abs(math.ceil(e)-e))/e
            if c['relative_error']>.03+1e-12:
                feasibility.append({'band':band,'pair':pair,**c,'integer_interval':[low,high],
                    'integer_feasible_at_fixed_realized_marginals':feasible,'minimum_integer_relative_error':best})
balance['literal_relative_failures']=feasibility
balance['number_impossible_cells_at_fixed_marginals']=sum(not x['integer_feasible_at_fixed_realized_marginals'] for x in feasibility)
balance['scope']='Recomputed global pairwise and all within-band pairwise tables. Literal +/-3 percent RELATIVE-count interpretation is one interpretation; original v1 does not resolve relative vs percentage-point or integer-rounding convention. No convention is silently weakened or selected here.'
save('balance_reaudit.json',balance)

SCOUT=Path(r'C:\Users\clclc\BusinessProject\ChatBubble-multi-reasoning-frozen-scout-v1\research\multi_reasoning_scout_v1')
scout_public=[json.loads(l) for l in (SCOUT/'bank/public.jsonl').read_text().splitlines()]
scout_worlds=[json.loads(l) for l in (SCOUT/'bank/worlds.jsonl').read_text().splitlines()]
scout_ids={r['id'] for r in scout_public}|{r['scenario_id'] for r in scout_public}
dataset_ids={r['id'] for r in allrows}|{r['group_id'] for r in allrows}
scout_context={norm(r['problem']) for r in scout_public};scout_questions={norm(r['problem'].splitlines()[-1]) for r in scout_public}
data_context={norm(check.query_from_text(r['messages'][1]['content'])['record']) for r in allrows}
data_questions={norm(check.query_from_text(r['messages'][1]['content'])['question']) for r in allrows}
scout_canon={digest(renamed(w['world'],set(w['world']['names']))) for w in scout_worlds}
data_canon=set().union(*reduced_counts.values())
scout_exclusion={'status':'PASS_EXACT_IDS_TEXT_AND_FULL_SCHEMA_CANONICAL_COMPARISON_ONLY',
    'bank_manifest_sha256':hashlib.sha256((SCOUT/'BANK_MANIFEST.json').read_bytes()).hexdigest(),
    'public_sha256':hashlib.sha256((SCOUT/'bank/public.jsonl').read_bytes()).hexdigest(),
    'worlds_sha256':hashlib.sha256((SCOUT/'bank/worlds.jsonl').read_bytes()).hexdigest(),
    'scout_rows':len(scout_public),'dataset_rows':len(allrows),'id_group_overlap':len(dataset_ids&scout_ids),
    'normalized_context_problem_overlap':len(data_context&scout_context),'normalized_question_overlap':len(data_questions&scout_questions),
    'full_entity_renamed_world_overlap':len(data_canon&scout_canon),
    'canonicalization':'Complete declared world for scout versus complete query/cases for LCR; remove lexical entity identities and ID/split/depth annotations, preserve semantic fields and list order.',
    'limits':'Different world schemas mean zero full-canonical collisions is not universal semantic graph-isomorphism proof. Both contain shared generic reasoning primitives. Protected evaluation banks remain unread and unverified.'}
save('frozen_scout_exclusion.json',scout_exclusion)

failures=[{'requirement':'native intended context bands','status':'FAIL_MEASURED','violations':native['outside_intended_band'],'action':'Issue new immutable dataset version that regenerates offending contexts within actual native bands; no truncation or relabelling after acceptance.'}]
if errors:failures.append({'requirement':'independent text answer check','status':'FAIL','errors':len(errors)})
if not balance['all_within_band_pairs_within_3_percent']:
    failures.append({'requirement':'within-band pairwise balance under literal relative-count +/-3%','status':'FAIL_UNDER_LITERAL_RELATIVE_INTERPRETATION','max_relative_error':balance['max_within_band_relative_error'],'integer_infeasible_cells_fixed_marginals':balance['number_impossible_cells_at_fixed_marginals'],'action':'Prospectively resolve integer rounding/tolerance semantics or regenerate allocations satisfying a feasible explicit contract; no retroactive acceptance.'})
receipt={'status':'P0_STOP_MEASURED_BAND_FAILURES_AND_UNRESOLVED_EXTERNAL_EXCLUSION','corpus_accepted':False,'dataset_root':str(ROOT),'manifest_sha256':sha_text(read('MANIFEST.json').decode()),'source_audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_repository':'LUNCY579/ChatBubble','source_branch':'codex/mrs-phase2-dataset-selection-v1','evidence_class':'OPEN_DATA_DEVELOPMENT_REAUDIT','rows':len(allrows),'payload_splits_read':list(SPLITS),'holdout_payload_reads':600,'all_jsonl_read':False,
    'access_basis':{'README':'data/dev and data/holdout are for data development; NOT model benchmarks','QUARANTINE_AND_CLAIMS':metadata['QUARANTINE_AND_CLAIMS.json'],'original_v1_spec_sha256':metadata['provenance/ATTACHED_CONTRACT.json']['source_sha256']},
    'forbidden_evaluation_payload_reads':0,'model_weights_loaded':0,'model_calls':0,'training_runs':0,'dataset_files_modified':0,'run_seconds':time.time()-started,'runtime':{p:importlib.metadata.version(p) for p in ('transformers','tokenizers','numpy')},
    'explicit_input_inventory':readlog,'native':native,'semantic_errors':errors,'independent_solver_verified_rows':len(verification),'parsed_facts':sum(x['parsed_facts'] for x in verification),
    'question_only_balanced_accuracy':question_ba,'surface_train_selected_feature':winner,'surface_train_selected_balanced_accuracy':thresholds[winner]['balanced_accuracy_by_split'],
    'semantic_selector_plus_validity':{'balanced_accuracy':semantic_ba,'category':'TASK_AWARE_SEMANTIC_ALGORITHM','binding_surface_threshold_failure':False,'interpretation':'Requires record facts, a relational color/location join, and explicit withdrawal/nonissuance checks. It does not by itself demonstrate forbidden question/surface leakage or violate a 0.55 question/surface gate. It reveals shallow semantic answerability for three of four absence mechanisms.'},
    'binding_failures_and_interpretation_dependent_failures':failures,'structural_reaudit':structure,'scout_exclusion':scout_exclusion,
    'external_protected_exclusion':'UNVERIFIED; obtain authorized blind hash-bound inventory/aggregate receipt without reading protected payloads',
    'remaining_limitations':['Finite question/surface attacks; producer sklearn exact-answer adversaries not rerun in this script.','Independent text checker is a separate algorithm sharing author/grammar with generator, not independent human review.','Whole scenario templates and typed full graphs may be disjoint while local operator primitives and reduced motifs are shared; no stronger semantic novelty claim.','All open splits are permanently data-design-exposed and cannot serve as model confirmation.'],
    'earlier_receipt_corrections':['Earlier holdout-unverified limitation is closed by explicit open-data authorization and all6000 census.','392 shared generic sentence forms do not alone establish full-scenario template or full typed-graph overlap.','87.5% semantic checker success is not automatically a forbidden question/surface shortcut threshold failure.']}
save('reaudit_receipt.json',receipt)
print(json.dumps({'status':receipt['status'],'rows':len(allrows),'seconds':receipt['run_seconds'],'native':native,'semantic_errors':len(errors),'structural_pairs':structure_pairs,'balance_max':balance['max_within_band_relative_error'],'integer_infeasible':balance['number_impossible_cells_at_fixed_marginals'],'semantic_checker_category':'not automatic surface violation','scout_overlap':scout_exclusion},indent=2),flush=True)
