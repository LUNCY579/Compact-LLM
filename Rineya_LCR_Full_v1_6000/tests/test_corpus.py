import sys,copy,json,re,hashlib
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate import make_row,TASKS,SPLITS,REASONS,PARTITIONS,names_for
from check import validate_row,solve,query_from_text,parse_record,quantity,DataError
from tokenizer_preflight import inspect_assets,measure_smoke
from audit import best_threshold,threshold_predict,ba


def plan(task,split,reason,difficulty='hard',position='early',band='short',seed=94120561):
    return {'scenario_seed':seed,'split':split,'task_type':task,'unanswerable_reason':reason,
      'variant':'unanswerable' if reason else 'answerable','difficulty':difficulty,'needle_position':position,
      'length_band':band,'allocation_id':'unit_test_only'}

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('split',SPLITS)
@pytest.mark.parametrize('reason',[None,*REASONS])
def test_independent_semantics(task,split,reason):
    r,w=make_row(plan(task,split,reason),0)
    out=validate_row(r)
    assert out['answer']==w['gold']
    assert out['reason']==reason
    assert (out['answer']=='UNKNOWN')==(reason is not None)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('position',['early','middle','late','distributed'])
def test_all_positions_and_offsets(task,position):
    r,w=make_row(plan(task,'train',None,position=position,band='long'),0)
    validate_row(r)
    m=r['metadata'];sp=m['needle_char_spans'];n=m['context_characters']
    if position=='early':assert max(b for a,b in sp)<=n*.25
    elif position=='late':assert min(a for a,b in sp)>=n*.75
    elif position=='middle':assert min(a for a,b in sp)>=n*.25 and max(b for a,b in sp)<=n*.75
    else:
        assert len(sp)==3
        changed=0
        for a,b in sp:
            text=r['messages'][1]['content'];out=solve(text[:a]+text[b:])
            changed+=out['answer']!=w['gold']
        assert changed>=2

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('split',SPLITS)
def test_hard_no_literal_overlap(task,split):
    r,w=make_row(plan(task,split,None),0)
    assert r['metadata']['lexical_question_positive_evidence_overlap']==[]
    assert r['metadata']['distractor_entities']>=8
    assert r['metadata']['near_miss_candidates']==2

@pytest.mark.parametrize('task',TASKS)
def test_medium_real_lexical_easing(task):
    r,w=make_row(plan(task,'train',None,difficulty='medium'),0)
    validate_row(r)
    assert len(r['metadata']['lexical_question_positive_evidence_overlap'])>=2

@pytest.mark.parametrize('task',TASKS)
def test_metadata_not_used_to_solve(task):
    r,w=make_row(plan(task,'train',None),0)
    expected=solve(r['messages'][1]['content'])
    r['metadata']={'gold':'INVENTED','task_type':'FAKE'};r['messages'][-1]['content']='INVENTED'
    assert solve(r['messages'][1]['content'])==expected

@pytest.mark.parametrize('task',TASKS)
def test_wrong_gold_rejected(task):
    r,w=make_row(plan(task,'train',None),0);r['messages'][-1]['content']='WRONG'
    with pytest.raises(DataError):validate_row(r)

@pytest.mark.parametrize('fault',['role','span','offset','empty','reason'])
def test_invalid_structure(fault):
    r,w=make_row(plan('alias_resolution','train',None),0)
    if fault=='role':r['messages'][0]['role']='user'
    elif fault=='span':r['metadata']['needle_char_spans']=[[-1,50]]
    elif fault=='offset':r['metadata']['question_char_span']=[0,7]
    elif fault=='empty':r['messages'][1]['content']=''
    else:r['metadata']['unanswerable_reason']='entity_absent'
    with pytest.raises(DataError):validate_row(r)

@pytest.mark.parametrize('reason',REASONS)
def test_unknown_does_not_invent_positive_spans(reason):
    r,w=make_row(plan('rule_exception','dev',reason),0);validate_row(r)
    assert r['metadata']['needle_char_spans']==[]
    if reason=='entity_absent':
        q=query_from_text(r['messages'][1]['content']);d=parse_record(q['record'])
        assert (q['color'],q['site']) not in d['selectors']

@pytest.mark.parametrize('split',SPLITS)
def test_counterfactual_forecast_differs(split):
    p=plan('count_with_exclusion',split,None);p['variant']='counterfactual'
    r,w=make_row(p,0);assert r['metadata']['forecast']!=w['gold']

@pytest.mark.parametrize('split',SPLITS)
def test_deterministic_build(split):
    p=plan('temporal_correlation',split,None,position='distributed')
    assert make_row(p,0)==make_row(p,0)

@pytest.mark.parametrize('a,b',[('train','dev'),('dev','holdout'),('train','holdout')])
def test_prepartition_entity_spaces_disjoint(a,b):
    left=set(names_for(124,a,180));right=set(names_for(125,b,180))
    assert not left&right
    assert PARTITIONS[a]['depth']!=PARTITIONS[b]['depth']

@pytest.mark.parametrize('text,num',[('zero',0),('nineteen',19),('twenty',20),('twenty one',21),('ninety nine',99)])
def test_written_number_parser(text,num):assert quantity(text)==num

@pytest.mark.parametrize('text',['one hundred','twenty nineteen','twenty zero','-1','999','a few'])
def test_invalid_numbers(text):
    with pytest.raises(DataError):quantity(text)

@pytest.mark.parametrize('x,y',[([1,2,3,4],[0,0,1,1]),([1,2,3,4],[1,1,0,0]),([2,2,2,2],[0,1,0,1])])
def test_exhaustive_threshold_matches_bruteforce(x,y):
    spec=best_threshold(x,y)
    brute=[]
    cuts=[min(x)-1]+[(a+b)/2 for a,b in zip(sorted(set(x))[:-1],sorted(set(x))[1:])]+[max(x)+1]
    for t in cuts:
        for sign in ('lt','ge'):brute.append(ba(y,threshold_predict(x,{'threshold':t,'direction':sign})))
    assert abs(spec['balanced_accuracy']-max(brute))<1e-12


def test_missing_tokenizer_fails_closed(tmp_path):
    entries,errors=inspect_assets(tmp_path)
    assert len(errors)==4 and all(not e['hash_match'] for e in entries)

def test_wrong_tokenizer_hash_rejected(tmp_path):
    (tmp_path/'fixture').write_bytes(b'wrong')
    entries,errors=inspect_assets(tmp_path,{'fixture':hashlib.sha256(b'right').hexdigest()})
    assert errors and not entries[0]['hash_match']

def test_no_native_counts_claimed_without_assets():
    r,w=make_row(plan('alias_resolution','train',None),0)
    for k in ('context_tokens','total_tokens','rendered_total_tokens','rendered_prompt_tokens','completion_tokens'):
        assert r['metadata'][k] is None
