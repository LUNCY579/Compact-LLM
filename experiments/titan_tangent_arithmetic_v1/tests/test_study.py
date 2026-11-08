from dataclasses import asdict, replace
import json
import pytest
import torch
from experiments.titan_tangent_arithmetic_v1 import study
from experiments.titan_tangent_arithmetic_v1.data import load_rows, require_design
from experiments.titan_tangent_arithmetic_v1.smoke import fixture_rows, TinyCausalLM
from experiments.titan_tangent_arithmetic_v1.tangent import FrozenAnchor
from experiments.titan_tangent_arithmetic_v1.diagnostics import qualify, historical_diagnostics


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(asdict(r))+'\n' for r in rows))
    return path


def test_fixture_bank_is_actually_sequence_and_parent_disjoint(tmp_path):
    rows=load_rows(write_rows(tmp_path/'r.jsonl',fixture_rows()),vocab_size=16,max_length=4)
    require_design(rows,('A','B'))
    assert len(rows)==18
    assert len({r.input_ids for r in rows})==18


@pytest.mark.parametrize('bad',['duplicate','parent','sequence','vocab','label','length','empty','first_label','forbidden'])
def test_bad_data_refused(tmp_path,bad):
    rows=fixture_rows()
    if bad=='duplicate': rows.append(rows[0])
    if bad=='parent': rows[3]=replace(rows[3],parent_id=rows[0].parent_id)
    if bad=='sequence': rows[3]=replace(rows[3],input_ids=rows[0].input_ids,labels=rows[0].labels)
    if bad=='vocab': rows[0]=replace(rows[0],input_ids=(99,3,6,0))
    if bad=='label': rows[0]=replace(rows[0],labels=(-100,-100,7,0))
    if bad=='length': rows[0]=replace(rows[0],labels=(-100,))
    if bad=='first_label': rows[0]=replace(rows[0],labels=(1,-100,6,0))
    if bad=='empty': rows=[]
    with pytest.raises(ValueError):
        load_rows(write_rows(tmp_path/'r.jsonl',rows),vocab_size=16,max_length=4,
                  forbidden_parents={rows[0].parent_id} if bad=='forbidden' else set())


@pytest.mark.parametrize('changes',[{'steps':0},{'lr':float('nan')},{'weight_decay':.1},{'sampling':'guess'},{'warmup_fraction':.9,'decay_fraction':.5}])
def test_bad_recipe(changes):
    with pytest.raises(ValueError): replace(study.TrainSpec(),**changes).validate()


def test_tied_parameter_scope_and_nonzero_jvp_qualification():
    torch.manual_seed(17); torch.set_num_threads(1)
    a=FrozenAnchor(TinyCausalLM().double())
    assert 'head.weight' not in a.params
    q=qualify(a,fixture_rows()[0])
    assert q['status']=='QUALIFICATION_PASS' and q['optimizer_steps']==0
    assert all(e<1e-5 for e in q['finite_difference_relative_errors'])


@pytest.mark.parametrize('mode',['nonlinear','linear'])
def test_training_uses_only_response_and_keeps_anchor(mode):
    torch.manual_seed(17); torch.set_num_threads(1)
    a=FrozenAnchor(TinyCausalLM().double())
    rows=[r for r in fixture_rows() if r.task=='A' and r.split=='train']
    d,r=study.train(a,rows,mode,study.TrainSpec(steps=2,batch_size=2,lr=.01,sampling='one_pass'))
    assert r['updates']==2 and r['processed_tokens']==12 and r['supervised_tokens']==6
    assert any(t.count_nonzero() for t in d.values())
    a.assert_unchanged()
    with pytest.raises(ValueError): study.train(a,fixture_rows()[3:4],mode,study.TrainSpec())


def test_budget_refusal_is_before_optimizer():
    a=FrozenAnchor(TinyCausalLM().double())
    rows=[r for r in fixture_rows() if r.task=='A' and r.split=='train']
    with pytest.raises(ValueError): study.train(a,rows,'linear',study.TrainSpec(max_processed_tokens=1))
    assert a.budget.jvp==a.budget.nonlinear==0


def test_all_three_methods_use_only_four_training_runs(monkeypatch):
    a=FrozenAnchor(TinyCausalLM().double()); calls=[]; captured=[]
    def fake_train(anchor,rows,mode,spec):
        calls.append((rows[0].task,mode)); return anchor.zero(),{'updates':0}
    def fake_add(anchor,deltas,rows,mode,alphas):
        captured.append((mode,{k:id(v) for k,v in deltas.items()})); return {'status':'FIXTURE'}
    monkeypatch.setattr(study,'train',fake_train);monkeypatch.setattr(study,'addition',fake_add)
    saved=[]
    r=study.run_study(a,fixture_rows(),('A','B'),study.TrainSpec(),on_delta=lambda *x:saved.append(x[:2]))
    assert len(calls)==len(saved)==4
    assert captured[0][1]==captured[1][1] # post-hoc reuses exact nonlinear-trained deltas
    assert captured[1][1]!=captured[2][1]
    assert set(r['methods'])=={'ordinary','posthoc','tangent_trained'}


def test_coefficient_is_selected_on_dev_before_any_test_score(monkeypatch):
    a=FrozenAnchor(TinyCausalLM().double()); rows=fixture_rows(); sequence=[]
    def fake_score(anchor,delta,rs,mode):
        sequence.append(rs[0].split)
        return {'accuracy':1.,'predictions':[(1,)]*len(rs),'correct':[True]*len(rs),'ids':[r.id for r in rs]}
    monkeypatch.setattr(study,'score',fake_score)
    r=study.addition(a,{'A':a.zero(),'B':a.zero()},rows,'linear',alphas=(0.,.5,1.))
    assert r['selected_alpha']==0. # deterministic tie rule
    first=sequence.index('test')
    assert first==8 and 'dev' not in sequence[first:]


def test_zero_denominator_never_opens_test(monkeypatch):
    a=FrozenAnchor(TinyCausalLM().double()); sequence=[]
    def fake_score(anchor,delta,rs,mode):
        sequence.append(rs[0].split);return {'accuracy':0.}
    monkeypatch.setattr(study,'score',fake_score)
    r=study.addition(a,{'A':a.zero(),'B':a.zero()},fixture_rows(),'linear',alphas=(0.,1.))
    assert r['status']=='UNDEFINED_NORMALIZATION' and not r['test_read']
    assert 'test' not in sequence


def test_negation_zero_control_is_not_success(monkeypatch):
    a=FrozenAnchor(TinyCausalLM().double())
    monkeypatch.setattr(study,'score',lambda *args:{'accuracy':0.})
    r=study.negation(a,a.zero(),fixture_rows()[3:4],fixture_rows()[4:5],'linear')
    assert r['status']=='UNINFORMATIVE_ZERO_CONTROL'


def test_historical_diagnostics_are_no_training():
    torch.manual_seed(17);torch.set_num_threads(1)
    a=FrozenAnchor(TinyCausalLM().double()); d={k:torch.ones_like(v)*.001 for k,v in a.params.items()}
    row=replace(fixture_rows()[0],split='diagnostic')
    r=historical_diagnostics(a,d,[row],{k:t*2 for k,t in d.items()})
    assert r['optimizer_steps']==0 and not r['mechanism_established']
    assert 'matched_sum_vs_attenuated_a_rms' in r['rows'][0]
    with pytest.raises(ValueError): historical_diagnostics(a,d,fixture_rows()[:1])


def test_disentanglement_surface_has_expected_axes(monkeypatch):
    a=FrozenAnchor(TinyCausalLM().double())
    rows=[replace(r,split='diagnostic') for r in fixture_rows()]
    monkeypatch.setattr(study,'score',lambda anchor,d,rs,mode:{'predictions':[(1,)]*len(rs)})
    r=study.pair_surface(a,a.zero(),a.zero(),rows[:1],rows[9:10],'linear',grid=(-1,0,1))
    assert len(r)==9 and all(x['xi_sum']==0 for x in r)
    with pytest.raises(ValueError):study.pair_surface(a,a.zero(),a.zero(),rows[:1],rows[:1],'linear')
