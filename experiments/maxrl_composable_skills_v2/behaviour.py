"""Enroll actual source-correct examples and score preservation without hiding losses.

Reads actual response JSONL; never generates or fabricates native predictions.
Responses must contain exactly id, prompt_hash, response, generated_tokens.
Run metadata must bind model_sha256, tokenizer_sha256 and decoding_sha256.
Use enrol before inspecting candidate results; receipt is a fixed baseline.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import random
from .contracts import require,digest,filehash
from .corpus import readrows,reward,EVAL_ROLES


def bank(data:Path,role:str):
    require(role in EVAL_ROLES,'only exposed evaluation/guard banks may be enrolled')
    rows=list(readrows(data/'canonical'/(role+'.jsonl')))
    require(rows and len({r['id'] for r in rows})==len(rows),'empty or duplicated bank')
    return rows


def validate_meta(meta):
    require(type(meta) is dict and set(meta)=={'model_sha256','tokenizer_sha256','decoding_sha256'},'run metadata schema')
    require(all(type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None for x in meta.values()),'unbound run hashes')


def score_responses(rows,responses):
    require(type(responses) is list and len(responses)==len(rows),'missing/extra predictions')
    indexed={}
    for p in responses:
        require(type(p) is dict and set(p)=={'id','prompt_hash','response','generated_tokens'},'response schema')
        require(type(p['id']) is str and p['id'] not in indexed,'duplicate prediction ID')
        require(type(p['response']) is str and type(p['generated_tokens']) is int and p['generated_tokens']>0,'response/token cost')
        indexed[p['id']]=p
    require(set(indexed)=={r['id'] for r in rows},'prediction row-set mismatch')
    scores={}
    for r in rows:
        p=indexed[r['id']];require(p['prompt_hash']==r['prompt_hash'],'wrong rendered prompt')
        scores[r['id']]=reward(r,p['response'])
    return scores


def enroll(data:Path,role:str,responses:list,metadata:dict):
    validate_meta(metadata);rows=bank(data,role);scores=score_responses(rows,responses)
    return {'schema_version':'2.0.0','kind':'source_correct_enrollment','role':role,
            'bank_sha256':filehash(data/'canonical'/(role+'.jsonl')),'run':metadata,
            'responses_sha256':digest(responses),'row_scores':scores,
            'source_correct_ids':sorted(k for k,v in scores.items() if v),
            'source_correct_count':sum(scores.values()),'scoped_rows':len(rows),
            'evaluation_status':'exposed bank, not release qualification'}


def exact_mcnemar(gains:int,losses:int):
    n=gains+losses
    if n==0:return 1.0
    # Log-space summation avoids a giant 2**n float overflow.
    terms=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2) for k in range(min(gains,losses)+1)]
    peak=max(terms)
    return min(1.,2*math.exp(peak)*sum(math.exp(t-peak) for t in terms))


def summary(ids,before,after):
    n=len(ids);correct=sum(before[i] for i in ids)
    lost=sum(before[i]==1 and after[i]==0 for i in ids)
    gained=sum(before[i]==0 and after[i]==1 for i in ids)
    return {'rows':n,'before_correct':correct,'after_correct':sum(after[i] for i in ids),
            'gains':gained,'losses':lost,'net':gained-lost,
            'retained_source_correct':correct-lost,'source_correct_denominator':correct,
            'conditional_loss_rate':lost/correct if correct else None,
            'paired_p_descriptive_rowwise':exact_mcnemar(gained,lost),
            'p_warning':'rowwise diagnostic only; related children are clustered; no automatic equivalence, noninferiority, or promotion'}


def evaluate(data:Path,enrollment:dict,candidate:list,metadata:dict,bootstrap_draws=1000):
    validate_meta(metadata)
    require(type(enrollment) is dict and enrollment.get('kind')=='source_correct_enrollment','wrong enrollment')
    validate_meta(enrollment['run'])
    for k in ('tokenizer_sha256','decoding_sha256'):require(metadata[k]==enrollment['run'][k],'unmatched tokenizer/decoding')
    rows=bank(data,enrollment['role'])
    require(filehash(data/'canonical'/(enrollment['role']+'.jsonl'))==enrollment['bank_sha256'],'bank changed after enrollment')
    before=enrollment['row_scores'];ids=[r['id'] for r in rows]
    require(set(before)==set(ids) and all(type(x) is int and x in (0,1) for x in before.values()),'source scores invalid')
    require(enrollment['source_correct_ids']==sorted(i for i in ids if before[i]) and enrollment['source_correct_count']==sum(before.values()) and enrollment['scoped_rows']==len(rows),'source membership changed')
    after=score_responses(rows,candidate)
    groups=defaultdict(list);clusters=defaultdict(list)
    for r in rows:
        groups[r['tag']+':'+''.join(r['skills'])].append(r['id']);clusters[r['parent_id']].append(r['id'])
    # Macro-parent score difference: parents, NOT row x seed, are resampling units.
    parent_deltas=[sum(after[i]-before[i] for i in c)/len(c) for c in clusters.values()]
    rng=random.Random(2002);boot=[]
    require(type(bootstrap_draws) is int and 0<=bootstrap_draws<=10000,'bootstrap draw bound')
    for _ in range(bootstrap_draws):
        boot.append(sum(rng.choice(parent_deltas) for _ in parent_deltas)/len(parent_deltas))
    boot.sort()
    interval=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None
    pred_by={p['id']:p for p in candidate}
    boundaries={};twins=[]
    for t in readrows(data/'diagnostics/answerability_twins.jsonl'):
        if t['role']==enrollment['role']:twins.append(t)
    if twins:
        answerable=[t['answerable_id'] for t in twins];unanswerable=[t['unanswerable_id'] for t in twins]
        def is_unknown(response):
            from .contracts import strict_json
            try:return strict_json(response,4096)=={'answer':'UNKNOWN'}
            except (ValueError,UnicodeError,RecursionError):return False
        boundaries={'answerable_rows':len(answerable),'unanswerable_rows':len(unanswerable),
                    'false_abstentions':sum(is_unknown(pred_by[i]['response']) for i in answerable),
                    'correct_unknown':sum(after[i] for i in unanswerable),
                    'both_twins_correct':sum(after[t['answerable_id']] and after[t['unanswerable_id']] for t in twins)}
    return {'status':'DESCRIPTIVE_EVALUATION_NOT_QUALIFICATION','source_run':enrollment['run'],'candidate_run':metadata,
            'bank_sha256':enrollment['bank_sha256'],'source_enrollment_sha256':digest(enrollment),
            'overall':summary(ids,before,after),'by_family':{k:summary(v,before,after) for k,v in sorted(groups.items())},
            'parent_cluster_bootstrap':{'parents':len(parent_deltas),'draws':bootstrap_draws,'seed':2002,'estimand':'macro-parent mean accuracy change','percentile_95_interval':interval},
            'answerability':boundaries,'candidate_generated_tokens':sum(p['generated_tokens'] for p in candidate),
            'no_forgetting_established':False,'margin_and_power':'NOT_BOUND'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('enroll','compare'));p.add_argument('--data',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--role');p.add_argument('--enrollment',type=Path)
    a=p.parse_args();require(not a.out.exists(),'receipt path must be new')
    responses=list(readrows(a.responses));metadata=json.loads(a.metadata.read_text())
    if a.command=='enroll':result=enroll(a.data,a.role,responses,metadata)
    else:require(a.enrollment is not None,'need frozen source enrollment');result=evaluate(a.data,json.loads(a.enrollment.read_text()),responses,metadata)
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(result.get('status',result['kind'] if 'kind' in result else ''))
if __name__=='__main__':main()
