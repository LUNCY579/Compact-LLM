from __future__ import annotations
import copy
import json
import math
import random
import pytest
from ..specs import FAMILIES, canonical, digest, fingerprint, partition, render, parse_visible
from ..generators import make
from ..oracle import solutions
from ..verifier import score, compute_score
from ..corpus import BAD_FORMAT, build, audit, wrong_answer
from ..maxrl import advantages, checked_group, pass_at_k, signal_probability

@pytest.fixture(params=[(f,d) for f in FAMILIES for d in range(4)])
def case(request):
    f,d=request.param; rng=random.Random(int(digest([f,d])[:12],16))
    for _ in range(1000):
        s=make(f,d,rng)
        if s is not None:
            refs=solutions(f,s)
            assert refs
            return f,s,refs
    raise AssertionError("no instance")

def test_independent_certificates(case):
    f,s,refs=case
    assert all(score(f,s,x) == 1 for x in refs)
    assert score(f,s,wrong_answer(f,s,refs[0])) == 0

def test_public_only_roundtrip(case):
    f,s,_=case
    for template in range(6): assert parse_visible(render(f,s,template)) == (f,s)

def test_strict_response_attack_surface(case):
    f,s,refs=case
    for text in (*BAD_FORMAT, "x"*4097, '{"answer":Infinity}', '{"answer":'+"["*1200+"0"+"]"*1200+"}"):
        assert score(f,s,text) == 0
    v=json.loads(refs[0])["answer"]
    if isinstance(v,list): assert score(f,s,canonical({"answer":[float(x) for x in v]})) == 0

def test_reward_adapter_binding(case):
    f,s,refs=case
    payload=canonical({"version":"1.0.0","family":f,"spec":s})
    assert compute_score("rineya_maxrl_v1/"+f,refs[0],payload) == 1.0
    with pytest.raises(ValueError): compute_score("wrong",refs[0],payload)
    bad=json.loads(payload); bad["version"]="2.0"
    with pytest.raises(ValueError): compute_score("rineya_maxrl_v1/"+f,refs[0],canonical(bad))
    bad=json.loads(payload); bad["spec"]["extra"]=123
    with pytest.raises(ValueError): compute_score("rineya_maxrl_v1/"+f,refs[0],canonical(bad))

def test_oracles_not_canonical_targets():
    fixtures=[
      ("subset_sum",{"values":[1,2,1,2,7,9],"k":2,"target":3},[[0,1],[3,2]]),
      ("linear_constraints",{"n":3,"upper":3,"equations":[{"coefficients":[1,1,0],"rhs":2}]},[[0,2,0],[2,0,3]]),
      ("cnf_assignment",{"n":3,"clauses":[[1,2]]},[[1,0,0],[0,1,1]]),
      ("shortest_path",{"n":4,"edges":[[0,1,1],[1,3,1],[0,2,1],[2,3,1],[0,3,3]],"start":0,"goal":3},[[0,1,3],[0,2,3]]),
      ("precedence_order",{"n":3,"before":[[0,2]]},[[0,1,2],[1,0,2]]),
      ("finite_program",{"modulus":7,"operations":[[1,1],[1,2]],"length":2,"cases":[[0,3],[1,4]]},[[0,1],[1,0]]),
      ("countdown",{"numbers":[2,3,4],"target":14},["(2*(3+4))","((4+3)*2)"]),
    ]
    for f,s,answers in fixtures:
        assert answers[0] != answers[1]
        for a in answers: assert score(f,s,canonical({"answer":a})) == 1
    f,s,_=fixtures[3]
    assert score(f,s,'{"answer":[0,3]}') == 0

@pytest.mark.parametrize("text",["__import__('os').system('echo bad')","2**(3**4)","234","2+3+4+0","2*True+3+4","-2+3+4","(2+3+4)/0"])
def test_no_eval_or_expression_shortcuts(text):
    assert score("countdown",{"numbers":[2,3,4],"target":9},canonical({"answer":text})) == 0

def test_fingerprint_is_not_seed_or_wording():
    s={"numbers":[2,3,4],"target":14}
    assert fingerprint("countdown",s) == fingerprint("countdown",{"numbers":[4,2,3],"target":14})
    assert digest(render("countdown",s,0)) != digest(render("countdown",s,1))
    for i in range(100): assert partition(digest(i)) in {"train","dev","test","challenge","warmstart"}

@pytest.mark.parametrize("n",[2,3,4,8,16])
def test_rare_success_weights(n):
    a=advantages([1]+[0]*(n-1),epsilon=0)
    assert a == pytest.approx([float(n-1)]+[-1.0]*(n-1), abs=1e-12)
    assert advantages([0]*n) == [0.0]*n
    assert advantages([1]*n) == [0.0]*n
    assert abs(sum(a)) < 1e-12

@pytest.mark.parametrize("n",[2,3,4,6,8])
@pytest.mark.parametrize("p",[.001,.05,.2,.5,.9])
def test_exact_expected_gradient_is_order_n_minus_one(n,p):
    expected=0.
    for successes in range(n+1):
        rewards=[1]*successes+[0]*(n-successes)
        a=advantages(rewards,epsilon=0)
        estimate=sum(v*(r-p) for v,r in zip(a,rewards))/n
        probability=math.comb(n,successes)*p**successes*(1-p)**(n-successes)
        expected += probability*estimate
    target=(1-p)*(1-(1-p)**(n-1))
    assert math.isclose(expected,target,rel_tol=1e-10,abs_tol=1e-12)

@pytest.mark.parametrize("bad",[[],[1],[0,.5],[0,True],[0,2]])
def test_advantages_reject_nonbinary_groups(bad):
    with pytest.raises(ValueError): advantages(bad)

def test_group_identity_positive_and_negative_controls():
    rows=[{"prompt_id":"x","rollout_id":str(i),"policy_id":"p0","reward":int(i==0),"generated_tokens":10} for i in range(8)]
    assert len(checked_group(rows,8,"p0")) == 8
    for field,value in (("prompt_id","wrong"),("rollout_id","0"),("policy_id","stale"),("generated_tokens",0)):
        bad=copy.deepcopy(rows); bad[1][field]=value
        with pytest.raises(ValueError): checked_group(bad,8,"p0")
    with pytest.raises(ValueError): checked_group(rows[:-1],8,"p0")

def test_passk_is_coverage_not_selection():
    assert pass_at_k(16,1,1) == 1/16
    assert pass_at_k(16,1,16) == 1
    with pytest.raises(ValueError): pass_at_k(2,1,4)
    for p in (0,.01,.5,1): assert math.isclose(sum(signal_probability(p,16).values()),1)

def test_persisted_pilot_reproducible_and_leakage_controls(tmp_path):
    a=tmp_path/"a"; b=tmp_path/"b"
    ma=build(a,"pilot",1234); mb=build(b,"pilot",1234)
    assert ma == mb
    receipt=audit(a)
    assert receipt["status"] == "PASS" and receipt["checks"]["rows"] == 528
    with pytest.raises(ValueError): build(a,"pilot",1234)
    p=a/"policy/train.jsonl"; p.write_text(p.read_text()+"{}\n")
    with pytest.raises(ValueError,match="hash/size"): audit(a)


def test_specification_changes_are_not_ignored():
    fixtures = [
      ("countdown", {"numbers":[2,3,4],"target":14}, {"numbers":[2,3,4],"target":15}, "(2*(3+4))"),
      ("subset_sum", {"values":[1,2,3,4,5],"k":2,"target":3}, {"values":[1,2,3,4,5],"k":2,"target":4}, [0,1]),
      ("linear_constraints", {"n":3,"upper":3,"equations":[{"coefficients":[1,1,0],"rhs":2}]}, {"n":3,"upper":3,"equations":[{"coefficients":[1,1,0],"rhs":3}]}, [1,1,0]),
      ("cnf_assignment", {"n":3,"clauses":[[1,2]]}, {"n":3,"clauses":[[-1,-2]]}, [1,1,0]),
      ("shortest_path", {"n":3,"edges":[[0,1,1],[1,2,1],[0,2,3]],"start":0,"goal":2}, {"n":3,"edges":[[0,1,3],[1,2,3],[0,2,2]],"start":0,"goal":2}, [0,1,2]),
      ("precedence_order", {"n":3,"before":[[0,2]]}, {"n":3,"before":[[2,0]]}, [0,1,2]),
      ("finite_program", {"modulus":7,"operations":[[1,1],[1,2]],"length":2,"cases":[[0,3],[1,4]]}, {"modulus":7,"operations":[[1,1],[1,3]],"length":2,"cases":[[0,3],[1,4]]}, [0,1]),
      ("state_tracking", {"initial":[1,2],"operations":[["copy",0,1],["add",1,1]]}, {"initial":[1,2],"operations":[["add",1,1],["copy",0,1]]}, [2,3]),
    ]
    for f, original, changed, answer in fixtures:
        text=canonical({"answer":answer})
        assert score(f,original,text) == 1
        assert score(f,changed,text) == 0


def test_invalid_metadata_is_not_learner_failure():
    for f in FAMILIES:
        with pytest.raises(ValueError): score(f,{},'{"answer":[]}')
    with pytest.raises(ValueError): score("unknown",{},'{"answer":[]}')
