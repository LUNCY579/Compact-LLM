from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile
import pytest
from ..contracts import *
from ..generator import world,view,controls,boundary_pair
from ..oracle import answers,derive,reachable_sums
from ..verifier import score,concrete,compute_score
from ..corpus import build,audit,BAD_RESPONSES,readrows,verl
from ..behaviour import enroll,evaluate,exact_mcnemar,summary,score_responses
from experiments.maxrl_verifiable_data_v1.maxrl import advantages,checked_group

@pytest.fixture(params=[(sig,d) for sig in SIGNATURES for d in range(4)])
def case(request):
    sig,d=request.param;rng=random.Random(int(digest([sig,d])[:12],16))
    s=view(world(rng,d),sig,rng,d)
    return s,answers(s)


def test_positive_semantic_certificates(case):
    s,refs=case
    assert refs and '"UNKNOWN"' not in refs[0]
    assert all(score(s,r)==1 for r in refs)
    assert 1 <= len(refs) <= 3
    assert concrete(s)==derive(s)


def test_visible_specs_are_complete(case):
    s,_=case
    for t in range(4):assert parse_visible(render(s,t))==s
    prompt=render(s,0);prompt[1]['content']=prompt[1]['content'].replace('zero-based','one-based')
    with pytest.raises(ValueError):parse_visible(prompt)


def test_every_stage_has_a_live_control(case):
    s,refs=case;c=controls(s,refs)
    assert c is not None
    _,items=c
    assert {i['skill'] for i in items}==set(signature(s))
    for i in items:
        assert score(s,i['transferred_response'])==1
        assert score(i['spec'],i['transferred_response'])==0
        assert score(i['spec'],i['positive_response'])==1
        assert i['spec'][FIELDS[i['skill']]]!=s[FIELDS[i['skill']]]
        assert all(i['spec'][k]==v for k,v in s.items() if k!=FIELDS[i['skill']])


def test_bad_answers_and_wrong_verifier_bindings(case):
    s,refs=case
    for text in (*BAD_RESPONSES,'x'*4097,'{"answer":"unknown"}','{"answer":'+('['*2000)+'0'+(']'*2000)+'}'):
        assert score(s,text)==0
    p={'version':VERSION,'kind':'composition','spec':s}
    assert compute_score('rineya_maxrl_comp_v2',refs[0],canonical(p))==1
    with pytest.raises(ValueError):compute_score('wrong',refs[0],canonical(p))
    p['version']='1.0.0'
    with pytest.raises(ValueError):compute_score('rineya_maxrl_comp_v2',refs[0],canonical(p))

@pytest.mark.parametrize('sig',[s for s in SIGNATURES if 'S' in s])
def test_inside_range_answerability_twins(sig):
    rng=random.Random(888)
    pair=None
    for _ in range(100):
        pair=boundary_pair(view(world(rng,1),sig,rng,1))
        if pair:break
    assert pair
    yes,no,ys,ns=pair
    assert score(yes,ns[0])==0 and score(no,ns[0])==1 and score(no,ys[0])==0
    sums=reachable_sums(no);assert min(sums)<no['selection']['target']<max(sums)
    assert no['selection']['target'] not in sums
    a=copy.deepcopy(yes);b=copy.deepcopy(no);a['selection'].pop('target');b['selection'].pop('target');assert a==b


def test_nonfinite_and_corrupt_metadata_are_host_errors():
    rng=random.Random(2);s=view(world(rng,0),'RUSP',rng,0)
    for key in ('initial','lookup','updates','selection','precedence'):
        q=copy.deepcopy(s);q[key]=None
        with pytest.raises(ValueError):score(q,'{}')
    for response in ('{"answer":[true]}','{"answer":Infinity}','{"answer":[0,0]}','{"answer":"\\ud800"}'):
        assert score(s,response)==0
    q=copy.deepcopy(s);q['precedence']=[[0,1],[1,0]]
    with pytest.raises(ValueError):score(q,'{}')


def test_lookup_unknown_is_closed_world_not_false_metadata():
    s={'initial':[1,2,3,4,5,6],'lookup':{'requested':['a'],'links':{'a':'b','b':'a'}}}
    assert answers(s)==['{"answer":"UNKNOWN"}'] and score(s,'{"answer":"UNKNOWN"}')==1
    s['lookup']['links']['b']=2
    assert score(s,'{"answer":"UNKNOWN"}')==0 and score(s,'{"answer":[2]}')==1


def test_state_event_order_changes_semantics():
    s={'initial':[1,2,3,4,5,6],'updates':{'operations':[['copy',0,1],['add',1,10]],'cutoff':3}}
    t=copy.deepcopy(s);t['updates']['operations'].reverse()
    r=answers(s)[0];assert score(s,r)==1 and score(t,r)==0


def test_accepts_unlisted_order_and_distinct_selection():
    s={'initial':[1,2,1,2,1,2],'selection':{'k':2,'target':3}}
    refs=answers(s,3)
    candidate='{"answer":[5,4]}'
    assert candidate not in refs and score(s,candidate)==1
    s['precedence']=[[4,5]]
    assert score(s,candidate)==0 and score(s,'{"answer":[4,5]}')==1

@pytest.mark.parametrize('n',[2,3,4,8,16])
@pytest.mark.parametrize('p',[.001,.05,.5,.9])
def test_maxrl_finite_group_identity(n,p):
    expected=0.
    for k in range(n+1):
        rs=[1]*k+[0]*(n-k);a=advantages(rs,epsilon=0)
        expected+=math.comb(n,k)*p**k*(1-p)**(n-k)*sum(x*(r-p) for x,r in zip(a,rs))/n
    assert expected==pytest.approx((1-p)*(1-(1-p)**(n-1)),abs=1e-11)


def test_mastered_replay_does_not_invent_signal():
    assert advantages([1]*8)==[0.]*8 and advantages([0]*8)==[0.]*8
    assert advantages([1]+[0]*7,epsilon=0)==[7.]+[-1.]*7
    group=[{'prompt_id':'x','policy_id':'a','rollout_id':str(i),'reward':int(i==0),'generated_tokens':1} for i in range(8)]
    assert len(checked_group(group,8,'a'))==8
    group[1]['policy_id']='old'
    with pytest.raises(ValueError):checked_group(group,8,'a')

@pytest.fixture(scope='module')
def pilot(tmp_path_factory):
    root=tmp_path_factory.mktemp('v2')
    # A self-contained toy predecessor for tests only. Full builds bind actual v1.
    inventory={'problem_hashes':[digest('fixture/problem')],'prompt_hashes':[digest('fixture/prompt')],
               'provenance':{'archive_sha256':digest('fixture'),'manifest_sha256':digest('fixture manifest'),'canonical_instances':1,'scope':'synthetic fixture only'}}
    src=root/'exclusions.json';src.write_text(json.dumps(inventory))
    data=root/'data';manifest=build(data,src,'pilot',123)
    return data,src,manifest


def test_persisted_audit_and_reproducibility(pilot,tmp_path):
    data,source,m=pilot
    assert audit(data)['status']=='PASS_DATA_MECHANICS_ONLY'
    out=tmp_path/'repeat';assert build(out,source,'pilot',123)==m
    with pytest.raises(ValueError):build(data,source,'pilot',123)
    with pytest.raises(ValueError):build(tmp_path/'full',source,'full',123)


def test_withheld_signatures_and_guard_disjointness(pilot):
    data,_,_=pilot
    banks={}
    for path in (data/'canonical').glob('*.jsonl'):
        rows=list(readrows(path));banks[path.stem]={r['id'] for r in rows}
        if path.stem in ('train','calibration','warmstart'):
            for r in rows:assert ''.join(r['skills']) in (SKILLS if path.stem=='warmstart' else TRAIN_SIGNATURES)
    for a,b in combinations(banks,2):assert banks[a].isdisjoint(banks[b])
    eval_sigs={''.join(r['skills']) for r in readrows(data/'canonical/test.jsonl')}
    assert eval_sigs==set(SIGNATURES)


def test_corrupt_export_and_missing_stage_controls_fail(pilot,tmp_path):
    import shutil
    data,_,_=pilot;copydir=tmp_path/'copy';shutil.copytree(data,copydir)
    p=copydir/'policy/train.jsonl';p.write_text(p.read_text()+'{}\n')
    with pytest.raises(ValueError,match='artifact hash'):audit(copydir)


def fixture_responses(data,role,correct=True):
    rows=list(readrows(data/'canonical'/(role+'.jsonl')));ors={x['id']:x for x in readrows(data/'oracles'/(role+'.jsonl'))}
    # These are deliberately named SYNTHETIC fixtures, never native output files.
    return [{'id':r['id'],'prompt_hash':r['prompt_hash'],'response':ors[r['id']]['references'][0] if correct else '{}','generated_tokens':8} for r in rows]

META={'model_sha256':'0'*64,'tokenizer_sha256':'1'*64,'decoding_sha256':'2'*64}

def test_source_correct_enrollment_identity_and_real_loss(pilot):
    data,_,_=pilot;responses=fixture_responses(data,'guard_test');e=enroll(data,'guard_test',responses,META)
    report=evaluate(data,e,responses,META,20)
    assert report['overall']['losses']==0 and report['overall']['gains']==0
    damaged=copy.deepcopy(responses);damaged[0]['response']='{}'
    report=evaluate(data,e,damaged,{**META,'model_sha256':'3'*64},20)
    assert report['overall']['losses']==1 and report['overall']['conditional_loss_rate']>0
    assert report['no_forgetting_established'] is False
    for altered in (responses[:-1],responses+[responses[0]]):
        with pytest.raises(ValueError):evaluate(data,e,altered,META,0)
    with pytest.raises(ValueError):evaluate(data,e,responses,{**META,'decoding_sha256':'3'*64},0)
    with pytest.raises(ValueError):enroll(data,'replay_train',responses,META)
    damaged=copy.deepcopy(e);damaged['source_correct_ids']=[]
    with pytest.raises(ValueError):evaluate(data,damaged,responses,META,0)


def test_zero_source_correct_is_undefined(pilot):
    data,_,_=pilot;p=fixture_responses(data,'guard_test',False);e=enroll(data,'guard_test',p,META)
    result=evaluate(data,e,p,META,0)
    assert result['overall']['source_correct_denominator']==0 and result['overall']['conditional_loss_rate'] is None


def test_abstention_endpoint_and_parent_cluster(pilot):
    data,_,_=pilot;p=fixture_responses(data,'test');e=enroll(data,'test',p,META)
    report=evaluate(data,e,p,META,20)
    assert report['answerability']['correct_unknown']>0 and report['answerability']['false_abstentions']==0
    assert report['parent_cluster_bootstrap']['parents']<report['overall']['rows']


def test_exact_paired_statistics_do_not_promote_net_score():
    assert exact_mcnemar(20,4)==pytest.approx(.001543879508972168,abs=1e-10)
    assert exact_mcnemar(0,0)==1.
    r=summary(['a','b'],{'a':1,'b':0},{'a':0,'b':1})
    assert r['net']==0 and r['losses']==1 and r['conditional_loss_rate']==1


def test_weight_orthogonality_is_not_functional_additivity():
    # f(theta)=theta[0]*theta[1]; two orthogonal changes have nonzero cross effect.
    a=(1.,0.);b=(0.,1.)
    assert sum(x*y for x,y in zip(a,b))==0
    f=lambda x:x[0]*x[1]
    assert f((1.,1.))-f(a)-f(b)+f((0.,0.))==1


def rehash_manifest(data,path):
    m=json.loads((data/'manifest.json').read_text());r=str(path.relative_to(data))
    m['files'][r]={'bytes':path.stat().st_size,'sha256':filehash(path)}
    (data/'manifest.json').write_text(json.dumps(m))


def test_rehashed_atomic_bank_contamination_still_fails(pilot,tmp_path):
    import shutil
    data,_,_=pilot;out=tmp_path/'copy';shutil.copytree(data,out)
    p=out/'manifests/atomic_R.json';plan=json.loads(p.read_text())
    testid=next(readrows(out/'canonical/test.jsonl'))['id'];plan['ids'][0]=testid
    p.write_text(json.dumps(plan));rehash_manifest(out,p)
    with pytest.raises(ValueError,match='atomic contamination'):audit(out)


def test_rehashed_vacuous_dependency_still_fails(pilot,tmp_path):
    import shutil
    data,_,_=pilot;out=tmp_path/'copy';shutil.copytree(data,out)
    p=out/'diagnostics/stage_necessity.jsonl';rows=list(readrows(p))
    rows[0]['transferred_response']=rows[0]['positive_response']
    p.write_text(''.join(canonical(r)+'\n' for r in rows));rehash_manifest(out,p)
    with pytest.raises(ValueError,match='vacuous stage control'):audit(out)


def test_rehashed_false_unknown_twin_fails(pilot,tmp_path):
    import shutil
    data,_,_=pilot;out=tmp_path/'copy';shutil.copytree(data,out)
    p=out/'diagnostics/answerability_twins.jsonl';rows=list(readrows(p))
    rows[0]['unanswerable_id']=rows[0]['answerable_id']
    p.write_text(''.join(canonical(r)+'\n' for r in rows));rehash_manifest(out,p)
    with pytest.raises(ValueError,match='unmatched twin'):audit(out)


def test_parquet_missing_dependency_fails_without_output(pilot,tmp_path,monkeypatch):
    import builtins
    from ..export import parquet
    real=builtins.__import__
    def intercepted(name,*a,**k):
        if name.startswith('pyarrow'):raise ImportError('controlled missing engine')
        return real(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',intercepted)
    out=tmp_path/'parquet'
    with pytest.raises(RuntimeError,match='BLOCKED'):parquet(pilot[0],out)
    assert not out.exists()
