"""All-row data audit. Fits ONLY declared non-neural shortcut adversaries.

No target language model, no weights, no remote service, no frozen evaluation bank.
A failed check aborts semantic validation; rows are never silently dropped.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, re, sys, unicodedata, warnings
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from check import validate_row,solve,parse_record,query_from_text,quantity,VERSION as CHECKER

SPLITS=('train','dev','holdout')
DIMS=('split','length_band','task_type','variant','needle_position','difficulty')

def save(p:Path,x:object)->None:
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def load(root):
    return [json.loads(l) for split in SPLITS for l in (root/'data'/(split+'.jsonl')).read_text().splitlines()]

def stat(a):
    a=np.array(a,dtype=float)
    return {k:float(v) for k,v in zip(('min','p50','p90','p95','max'),np.percentile(a,[0,50,90,95,100]))} if len(a) else None

def extract(x):
    return {'split':x['split'],**{d:x['metadata'][d] for d in DIMS if d!='split'}}

def balance(rows):
    vals=[extract(x) for x in rows];n=len(rows)
    marg={d:Counter(x[d] for x in vals) for d in DIMS}
    result={'rows':n,'marginals':{d:dict(v) for d,v in marg.items()},'pair_tables':{},'within_band_pair_tables':{}}
    def tables(cohort,dimensions):
        nn=len(cohort);ms={d:Counter(x[d] for x in cohort) for d in dimensions};out={}
        for d,e in itertools.combinations(dimensions,2):
            actual=Counter((x[d],x[e]) for x in cohort);cells=[]
            for a,b in itertools.product(sorted(ms[d]),sorted(ms[e])):
                exp=ms[d][a]*ms[e][b]/nn;obs=actual[(a,b)]
                cells.append({'left':a,'right':b,'actual':obs,'proportional_expectation':exp,
                  'absolute_error':abs(obs-exp),'relative_error':abs(obs-exp)/exp if exp else None})
            out[d+'__'+e]=cells
        return out
    result['pair_tables']=tables(vals,DIMS)
    for band in ('short','medium','long'):
        result['within_band_pair_tables'][band]=tables([v for v in vals if v['length_band']==band],[d for d in DIMS if d!='length_band'])
    allglobal=[c for v in result['pair_tables'].values() for c in v]
    allwithin=[c for tables_ in result['within_band_pair_tables'].values() for v in tables_.values() for c in v]
    result['max_global_relative_error']=max(c['relative_error'] for c in allglobal)
    result['max_within_band_relative_error']=max(c['relative_error'] for c in allwithin)
    result['all_global_pairs_within_3_percent']=result['max_global_relative_error']<=.03+1e-12
    result['all_within_band_pairs_within_3_percent']=result['max_within_band_relative_error']<=.03+1e-12
    result['rounding_note']='Integer counts cannot always be within 3% of small fractional targets; no rounding convention is silently substituted.'
    return result


def normalized(s,mode):
    if mode=='exact':return s
    s=' '.join(unicodedata.normalize('NFKC',s).split())
    return re.sub(r'\d','#',s) if mode=='digit_blinded' else s


def leakage(rows,worlds):
    records={s:[r for r in rows if r['split']==s] for s in SPLITS};out={'text_overlap':{},'entities':{},'template_ids':{},'case_layouts':{}}
    for mode in ('exact','nfkc_whitespace','digit_blinded'):
        out['text_overlap'][mode]={}
        for field in ('context','question','answer'):
            sets={}
            for split,rs in records.items():
                vals=[]
                for r in rs:
                    q=query_from_text(r['messages'][1]['content'])
                    v=q['record'] if field=='context' else q['question'] if field=='question' else r['messages'][-1]['content']
                    vals.append(normalized(v,mode))
                sets[split]=set(vals)
            out['text_overlap'][mode][field]={a+'__'+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(SPLITS,2)}
    entitysets={s:set(itertools.chain.from_iterable(w['entity_names'] for w in worlds if w['split']==s)) for s in SPLITS}
    out['entities']['counts']={s:len(v) for s,v in entitysets.items()}
    out['entities']['intersection']={a+'__'+b:len(entitysets[a]&entitysets[b]) for a,b in itertools.combinations(SPLITS,2)}
    temp={s:{r['metadata']['template_id'] for r in records[s]} for s in SPLITS}
    out['template_ids']['intersection']={a+'__'+b:len(temp[a]&temp[b]) for a,b in itertools.combinations(SPLITS,2)}
    # Count genuine name-blind shared local sentence forms, rather than using IDs as proof.
    patterns={s:set() for s in SPLITS};grams={s:set() for s in SPLITS}
    for r in rows:
        q=query_from_text(r['messages'][1]['content']);world=worlds_by_id[ r['id'] ]
        ent=set(world['entity_names'])
        for line in q['record'].splitlines():
            line=' '.join('<ENTITY>' if word in ent else word.lower() for word in re.findall(r'[A-Za-z]+|\d+',line))
            patterns[r['split']].add(re.sub(r'\b\d+\b','<NUMBER>',line))
        ws=re.findall(r'[a-z]+',q['record'].lower())
        grams[r['split']].update(' '.join(ws[i:i+8]) for i in range(len(ws)-7))
    out['shared_name_blind_sentence_forms']={a+'__'+b:len(patterns[a]&patterns[b]) for a,b in itertools.combinations(SPLITS,2)}
    out['shared_exact_context_8grams']={a+'__'+b:len(grams[a]&grams[b]) for a,b in itertools.combinations(SPLITS,2)}
    # Canonical typed-layout census. These are actual case graphs, not evidence of new operator families.
    layouts={s:set() for s in SPLITS}
    for w in worlds:
        graph=w['dependency_graph'];nodes={n['id']:n['kind'] for n in graph['nodes']}
        serial={'nodes':sorted(Counter(nodes.values()).items()),
                'edges':sorted(Counter((nodes[a],nodes[b]) for a,b in graph['edges']).items())}
        layouts[w['split']].add(json.dumps(serial,sort_keys=True))
    out['case_layouts']['typed_degree_signature_intersections']={a+'__'+b:len(layouts[a]&layouts[b]) for a,b in itertools.combinations(SPLITS,2)}
    out['case_layouts']['scope']='Typed graph census with split-specific operator width/depth. Not a proof of disjoint reduced solution motifs; temporal successor lookup is shared.'
    out['structural_disjointness_status']='NOT_FULLY_ESTABLISHED: operator primitives and local sentence forms are shared, despite disjoint names, IDs and registered case layouts.'
    out['external_evaluation_comparison']={'status':'NOT_RUN_NO_BANK_ACCESS','rows_opened':0,
       'exact_overlap':None,'nfkc_overlap':None,'digit_blinded_overlap':None,'entity_overlap':None,
       'template_overlap':None,'graph_overlap':None,'eightgram_overlap':None}
    out['answer_overlap_note']='Shared integer answers and UNKNOWN are expected and explicitly counted, not hidden as zero.'
    return out


def surface_features(row):
    q=query_from_text(row['messages'][1]['content']);c=q['record'];question=q['question']
    words=re.findall(r'[A-Za-z]+',c);qw=re.findall(r'[A-Za-z]+',question)
    # Do not count sentence-leading instruction verbs as queried entity names.
    proper=[w for w in re.findall(r'\b[A-Z][a-z]+\b',c) if w not in {'This','Bell','Inspection','Withdrawals','Forecasts','Capitalized','An','North','South','East','West','Signatures','Beside','During','For','After','With','Spare','Lighting','Props','The','A'}]
    punct=sum(c.count(ch) for ch in ';=:|/')
    return {
      'context_characters':len(c),'context_words':len(words),'question_characters':len(question),
      'question_words':len(qw),'sentence_count':c.count('.'),'line_count':c.count('\n')+1,
      'capitalized_name_mentions':len(proper),'unique_capitalized_names':len(set(proper)),
      'mean_context_word_length':sum(map(len,words))/len(words),
      'mean_question_word_length':sum(map(len,qw))/len(qw),
      'punctuation_density':punct/len(c),'digit_density':sum(ch.isdigit() for ch in c)/len(c),
      'comma_count':c.count(','),'question_context_exact_word_overlap':len(set(w.lower() for w in qw)&set(w.lower() for w in words)),
      'query_colour_word_mentions':len(re.findall(r'\b'+q['color']+r'\b',c)),
      'query_direction_word_mentions':len(re.findall(r'\b'+q['site']+r'\b',c,re.I)),
      'withdrawal_word_count':len(re.findall(r'\bwithdrew\b',c)),
      'negative_issuance_word_count':len(re.findall(r'\bnever issued\b',c)),
      'unrecorded_word_count':len(re.findall(r'\bunrecorded\b',c)),
      'forecast_word_count':len(re.findall(r'\bforecast\b',c)),
      'unique_lowercase_words':len(set(w.lower() for w in words)),
    }


def ba(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    return float(((p[y==1]==1).mean()+(p[y==0]==0).mean())/2)


def best_threshold(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=int)
    order=np.argsort(x,kind='stable');xs=x[order];ys=y[order];n=len(y)
    cuts=np.r_[0,np.flatnonzero(xs[1:]!=xs[:-1])+1,n]
    ones=np.r_[0,np.cumsum(ys)];zeros=np.r_[0,np.cumsum(1-ys)];P=int(y.sum());N=n-P
    # Positive = UNKNOWN. Enumerate both signs and EVERY attainable unique-value cut.
    right=((P-ones[cuts])/P+zeros[cuts]/N)/2
    left=(ones[cuts]/P+(N-zeros[cuts])/N)/2
    a=int(np.argmax(right));b=int(np.argmax(left));direction='ge' if right[a]>=left[b] else 'lt';idx=int(cuts[a if direction=='ge' else b])
    threshold=float(xs[0]-1) if idx==0 else float(xs[-1]+1) if idx==n else float((xs[idx-1]+xs[idx])/2)
    return {'balanced_accuracy':float(max(right[a],left[b])),'direction':direction,'threshold':threshold,'unique_thresholds':len(cuts)}


def threshold_predict(values,spec):
    values=np.asarray(values)
    return (values>=spec['threshold']).astype(int) if spec['direction']=='ge' else (values<spec['threshold']).astype(int)


def shortcuts(rows,record_maps=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack,csr_matrix
    n=len(rows);answers=np.array([r['messages'][-1]['content'] for r in rows],dtype=object)
    task=np.array([r['metadata']['task_type'] for r in rows]);split=np.array([r['split'] for r in rows]);y=(answers=='UNKNOWN').astype(int)
    parsed=[query_from_text(r['messages'][1]['content']) for r in rows]
    questions=[q['question'] for q in parsed];contexts=[q['record'] for q in parsed]
    train=np.flatnonzero(split=='train');other=np.flatnonzero(split!='train')
    answerable=y==0
    # Fixed majority baselines: TRAIN fit, evaluated on all splits; TRAIN scores explicitly in-sample.
    counts=Counter(answers[train]);majority=sorted(counts,key=lambda v:(-counts[v],v))[0]
    allmaj=np.full(n,majority,dtype=object)
    per_task={t:Counter(answers[train[task[train]==t]]).most_common(1)[0][0] for t in sorted(set(task))}
    taskmaj=np.array([per_task[t] for t in task],dtype=object)
    # Also include stronger majority among answerable training rows.
    ansmaj={t:Counter(answers[train[(task[train]==t)&(y[train]==0)]]).most_common(1)[0][0] for t in sorted(set(task))}
    ansmajpred=np.array([ansmaj[t] for t in task],dtype=object)
    first=[];last=[];copylast=[]
    for r,q in zip(rows,parsed):
        names=set(worlds_by_id[r['id']]['entity_names'])
        ordered=[w for w in re.findall(r'\b[A-Z][a-z]+\b',q['record']) if w in names]
        first.append(ordered[0] if ordered else 'UNKNOWN');last.append(ordered[-1] if ordered else 'UNKNOWN');copylast.append(last[-1])
    first_task=[];last_task=[]
    for r,q in zip(rows,parsed):
        # Task-valued candidate assertions, not arbitrary owner names. These extra
        # controls prevent a vacuous first-owner baseline being called first answer candidate.
        name=r'[A-Z][a-z]+'
        patterns={
         'alias_resolution':[fr'(?:entrusted|entrusts) ({name}) with custody'],
         'causal_disambiguation':[fr'({name}) exhibits exactly',fr'({name}) has documented',fr'identifies ({name}) as its sole qualifying cause'],
         'multi_condition':[fr'({name}) exhibits exactly',fr'({name}) has documented',fr'selects ({name}) as its sole fully qualified applicant'],
         'rule_exception':[fr'ordinarily requires ({name})',fr'requiring ({name})',fr'requires ({name}) after applying'],
         'count_with_exclusion':[r'retains ([a-z ]+) eligible pieces after all exclusions'],
         'temporal_correlation':[fr'({name}) immediately followed',fr'schedules ({name}) immediately after'],
         'object_state_tracking':[fr'became ({name}) at observation',fr'records ({name}) as its checkpoint state'],
         'latest_verified':[fr'logged ({name}) at edition',fr'endorses ({name}) as its newest verified entry']}
        candidates=[]
        for pat in patterns[q['task']]:
            for match in re.finditer(pat,q['record']):
                value=str(quantity(match[1])) if q['task']=='count_with_exclusion' else match[1]
                candidates.append((match.start(1),value))
        candidates.sort()
        first_task.append(candidates[0][1] if candidates else 'UNKNOWN')
        last_task.append(candidates[-1][1] if candidates else 'UNKNOWN')
    pred={'majority':allmaj,'majority_per_task':taskmaj,'majority_answerable_per_task':ansmajpred,
          'always_unknown':np.full(n,'UNKNOWN',dtype=object),'copy_most_recent_entity':np.array(copylast,dtype=object),
          'first_entity':np.array(first,dtype=object),'last_entity':np.array(last,dtype=object),
          'first_candidate':np.array(first_task,dtype=object),'last_candidate':np.array(last_task,dtype=object)}
    # These answer-prediction baselines get no context or no question respectively.
    # TRAIN is fivefold OOF; DEV/holdout are fitted on TRAIN only. Group IDs are unique per row.
    folds=list(StratifiedKFold(5,shuffle=True,random_state=20260916).split(train,task[train]))
    for label,texts in [('question_only',questions),('context_only',contexts)]:
        out=np.full(n,'UNKNOWN',dtype=object)
        for fitloc,testloc in folds+[(np.arange(len(train)),None)]:
            fit=train[fitloc];test=other if testloc is None else train[testloc]
            vec=TfidfVectorizer(ngram_range=(1,1),min_df=1,sublinear_tf=True,max_features=40000,dtype=np.float32)
            X=vec.fit_transform([texts[i] for i in fit]);Y=vec.transform([texts[i] for i in test])
            # TF-IDF is L2-normalized: maximum dot product is exact cosine NN.
            # Dense batched BLAS avoids repeated sparse all-pairs overhead. Ties
            # select the first TRAIN/fit row deterministically, never a test label.
            Xdense=X.toarray();Ydense=Y.toarray();chosen=[]
            for offset in range(0,len(test),128):
                similarity=Ydense[offset:offset+128]@Xdense.T
                chosen.extend(np.argmax(similarity,axis=1).tolist())
            idx=np.asarray(chosen,dtype=int);out[test]=answers[fit[idx]]
            print(f'Completed {label} fit={len(fit)} scored={len(test)}',flush=True)
        pred[label]=out
    result={'definitions':{
      'first_candidate':'First task-valued assertion or candidate in the visible prose; typed extraction is explicit in audit.py. Counts copy the first explicitly summarized retained quantity, not a batch operand.',
      'first_entity':'First capitalized entity name; the published entity lexicon excludes sentence-leading ordinary words.',
      'last_candidate':'Last task-valued assertion or candidate in the visible prose.',
      'last_entity':'Last capitalized entity name in model-visible record.',
      'most_recent':'Textually most recent, not an oracle chronological winner.',
      'question_only_context_only':'Unigram TF-IDF exact cosine nearest neighbor; fivefold OOF TRAIN, TRAIN-fitted DEV and holdout; equal-score ties use first fit row.',
      'limitation':'Finite weak baselines. Low exact-answer scores do not certify shortcut-free answerability or broad difficulty.'},
      'results':{},'threshold_answerable':.25}
    for label,pp in pred.items():
        pp=np.asarray(pp);mask=answerable
        result['results'][label]={'overall_accuracy':float((pp==answers).mean()),
          'answerable_correct':int((pp[mask]==answers[mask]).sum()),'answerable_total':int(mask.sum()),
          'answerable_accuracy':float((pp[mask]==answers[mask]).mean()),'by_task':{}}
        for t in sorted(set(task)):
            mask=(task==t)&answerable
            result['results'][label]['by_task'][t]={'correct':int((pp[mask]==answers[mask]).sum()),'n':int(mask.sum()),'accuracy':float((pp[mask]==answers[mask]).mean())}
    result['all_named_answer_baselines_le_25pct']=all(v['answerable_accuracy']<=.25 and all(z['accuracy']<=.25 for z in v['by_task'].values()) for v in result['results'].values())
    # Question-only ANSWERABILITY classifier is different from predicting exact answer strings.
    vectorizer=TfidfVectorizer(ngram_range=(1,1),lowercase=True,min_df=1,sublinear_tf=True)
    X=vectorizer.fit_transform([questions[i] for i in train]);Y=vectorizer.transform(questions)
    lengths=np.array([[len(questions[i]),len(questions[i].split())] for i in range(n)],dtype=float)
    scaler=StandardScaler().fit(lengths[train]);L=scaler.transform(lengths)
    clf=LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=20260916)
    clf.fit(hstack([X,csr_matrix(L[train])]),y[train]);predy=clf.predict(hstack([Y,csr_matrix(L)]))
    print('Question-only answerability classifier complete.',flush=True)
    result['question_only_answerability']={'fit':'TRAIN only','features':'unigram TF-IDF + question character/word counts',
      'balanced_accuracy_by_split':{s:ba(y[split==s],predy[split==s]) for s in SPLITS}}
    fs=[surface_features(r) for r in rows];names=list(fs[0]);matrix=np.array([[f[k] for k in names] for f in fs])
    print('Surface feature extraction complete.',flush=True)
    specs={};optimistic={}
    for j,k in enumerate(names):
        spec=best_threshold(matrix[train,j],y[train]);specs[k]=spec
        predk=threshold_predict(matrix[:,j],spec)
        spec['test_balanced_accuracy']={s:ba(y[split==s],predk[split==s]) for s in SPLITS}
        optimistic[k]=best_threshold(matrix[:,j],y)
    # No best feature is chosen using DEV/holdout. Report train-selected single feature separately.
    winner=max(names,key=lambda k:specs[k]['balanced_accuracy'])
    result['surface_threshold_search']={'feature_count':len(names),'features':specs,
       'train_selected_feature':winner,'train_selected_test_balanced_accuracy':specs[winner]['test_balanced_accuracy'],
       'all_cohort_optimistic_diagnostic':optimistic,
       'best_all_cohort_balanced_accuracy':max(v['balanced_accuracy'] for v in optimistic.values()),
       'scope':'exhaustive unique thresholds in both directions over the listed finite features only'}
    # Stronger cheap SEMANTIC screen: join color and location and check issuance/withdrawal.
    # This is explicitly not hidden just because it falls outside the simple surface matrix.
    print('Exhaustive threshold search complete.',flush=True)
    joined_pred=[]
    for index,(r,q) in enumerate(zip(rows,parsed)):
        d=record_maps[index] if record_maps is not None else parse_record(q['record']);rr=d['selectors'].get((q['color'],q['site']),[])
        joined_pred.append(int(not rr or (len(rr)==1 and (rr[0] in d['withdrawn'] or rr[0] in d['nonissued']))))
    result['selector_plus_status_answerability']={'balanced_accuracy':ba(y,joined_pred),
       'scope':'cheap task-aware relational join plus explicit validity notices; stronger than surface features',
       'interpretation':'High performance means most UNKNOWN mechanisms are easy once the right case is located, not that every abstention requires deep reasoning.'}
    return result,fs


def main(root:Path,skip_shortcuts=False):
    global worlds_by_id
    rows=load(root);worlds=[json.loads(l) for s in SPLITS for l in (root/'worlds'/(s+'.jsonl')).read_text().splitlines()]
    worlds_by_id={w['id']:w for w in worlds}
    if len(worlds_by_id)!=len(rows):raise ValueError('World count mismatch')
    receipts=[];facts=0;hard=[];positions=Counter();dproof=[];record_maps=[]
    for r in rows:
        check=validate_row(r);facts+=check['parsed_facts'];q=query_from_text(r['messages'][1]['content']);d=parse_record(q['record']);record_maps.append(d);m=r['metadata']
        # Verify both named near misses using actual parsed selector bindings and semantic summary facts.
        near_ok=True
        for name in m['near_miss_entities']:
            pairs=[k for k,v in d['selectors'].items() if name in v]
            if len(pairs)!=1:near_ok=False;continue
            c,s=pairs[0];matches=int(c==q['color'])+int(s==q['site'])
            if matches!=1 or (name,q['task']) not in d['summaries']:near_ok=False
        evidence=' '.join(r['messages'][1]['content'][a:b] for a,b in m['needle_char_spans'])
        ov=set(re.findall(r'[a-z]+',q['question'].lower()))&set(re.findall(r'[a-z]+',evidence.lower()))
        nhard={'distractor_count_pass':m['distractor_entities']>=8,'two_near_misses_pass':near_ok,
               'literal_positive_evidence_overlap':sorted(ov),'lexical_test_applicable':bool(m['needle_char_spans'])}
        m['hardness_checks']=nhard
        if m['difficulty']=='hard':hard.append(nhard)
        p=m['needle_position'];spans=m['needle_char_spans'];length=len(q['record'])
        if spans:
            if p=='early':ok=all(b/length<=.25 for a,b in spans)
            elif p=='late':ok=all(a/length>=.75 for a,b in spans)
            elif p=='middle':ok=all(.25<=a/length<=b/length<=.75 for a,b in spans)
            else:ok=2<=len(spans)<=3 and spans[-1][0]-spans[0][1]>=length*.2
            positions[f'{p}_applicable']+=1;positions[f'{p}_pass']+=int(ok);m['position_check_pass']=ok
            if p=='distributed':
                changed=[]
                for a,b in spans:
                    t=r['messages'][1]['content'];ablated=t[:a]+t[b:]
                    try:out=solve(ablated)['answer']
                    except Exception:out='PARSER_REJECTED'
                    changed.append(out!=r['messages'][-1]['content'])
                dproof.append({'id':r['id'],'fragments':len(spans),'necessary_fragments_under_deletion':sum(changed),'changes':changed})
                m['distributed_required_fragments']=sum(changed)
        else:m['position_check_pass']=None
        m['verified_by']=CHECKER
        receipts.append({'id':r['id'],'derived_answer':check['answer'],'derived_reason':check['reason'],
                         'parsed_facts':check['parsed_facts'],'agree':True})
    report=root/'reports'
    save(report/'verification.json',{'status':'PASS','rows_total':len(rows),'rows_independently_rederived':len(rows),
      'parsed_facts':facts,'answer_disagreements':0,'reason_disagreements':0,'checker':CHECKER,
      'checker_input':'model-visible user text only; metadata and assistant answer consulted only AFTER solving',
      'independence_limit':'Separate parser and solving implementation, same author and designed grammar. Not independent human certification.'})
    with (report/'verification_rows.jsonl').open('w') as f:
        for x in receipts:f.write(json.dumps(x,separators=(',',':'))+'\n')
    save(report/'evidence_and_difficulty.json',{'position_rules':{'early':'all positive-span ends <= 0.25','middle':'all positive spans in central half [0.25,0.75]','late':'all positive-span starts >= 0.75','distributed':'2-3 separated fragments with deletion audit'},
      'position_results':dict(positions),'unanswerable_positive_spans':0,'unanswerable_position_quota_scope':'related account or absence neighborhood; never claimed positive evidence',
      'hard_rows':len(hard),'hard_distractor_failures':sum(not x['distractor_count_pass'] for x in hard),
      'hard_near_miss_failures':sum(not x['two_near_misses_pass'] for x in hard),
      'hard_lexical_failures_answerable':sum(bool(x['literal_positive_evidence_overlap']) for x in hard),
      'hard_lexical_inapplicable_unknown':sum(not x['lexical_test_applicable'] for x in hard),
      'distributed_rows':len(dproof),'distributed_with_at_least_two_necessary_fragments':sum(x['necessary_fragments_under_deletion']>=2 for x in dproof)})
    save(report/'distributed_deletion_audit.json',dproof)
    save(report/'balance.json',balance(rows))
    save(report/'leakage.json',leakage(rows,worlds))
    tokens={'status':'BLOCKED_MISSING_HASH_LOCKED_TARGET_TOKENIZER','rows_total':len(rows),'rows_natively_measured':0,
       'tokenizer_reference':'provenance/TARGET_TOKENIZER_LOCK.json','native_total_tokens':None,'native_prompt_tokens':None,
       'rows_total_le_2048':None,'rows_prompt_le_1920':None,'native_band_fit':None,'rejected_for_native_fit':None,
       'no_truncation':True,'word_measurements_are_NOT_token_measurements':True,
       'context_word_profile_by_intended_band':{b:stat([x['metadata']['context_words'] for x in rows if x['metadata']['length_band']==b]) for b in ('short','medium','long')},
       'context_character_profile_by_task':{t:stat([x['metadata']['context_characters'] for x in rows if x['metadata']['task_type']==t]) for t in sorted({x['metadata']['task_type'] for x in rows})},
       'complete_sentences_added_to_meet_position':sum(x['metadata']['complete_sentences_added_for_position'] for x in rows),
       'note':'The native 2048-window and 128-token reserve are NOT verified. No zero-failure native claim is made.'}
    save(report/'token_profile.json',tokens)
    # Save the semantically verified row contents before expensive adversary work.
    for split in SPLITS:
        with (root/'data'/(split+'.jsonl')).open('w',encoding='utf-8',newline='\n') as f:
            for r in rows:
                if r['split']==split:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    print('All-row semantic, balance, leakage and evidence audits written.',flush=True)
    if skip_shortcuts:return len(rows)
    sh,fs=shortcuts(rows,record_maps=record_maps);save(report/'shortcut_floors.json',sh)
    with (report/'surface_features.jsonl').open('w') as f:
        for r,v in zip(rows,fs):f.write(json.dumps({'id':r['id'],'split':r['split'],'unanswerable':r['messages'][-1]['content']=='UNKNOWN','features':v},separators=(',',':'))+'\n')
    print('Shortcut audits written.',flush=True)
    return len(rows)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a.add_argument('--skip-shortcuts',action='store_true');args=a.parse_args();main(args.root,args.skip_shortcuts)
