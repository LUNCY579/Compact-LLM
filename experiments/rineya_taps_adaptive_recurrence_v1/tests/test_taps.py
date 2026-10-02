from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import TAPSConfig, TAPSScheduler, apply_update, scalar_events
from experiments.rineya_taps_adaptive_recurrence_v1.experiment import (GRID, SEEDS, Case, fixtures, rollout,
    select_on_dev, evaluate_test, add_oracles, run_fixture_experiment)
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, memory_guard, write_json, native_preflight


def test_disabled_is_unit_recurrence():
    c=fixtures('DEV')[1]
    a,_=rollout(c,TAPSConfig(enabled=False),4,17)
    b,_=rollout(c,TAPSConfig(scheduler_mode='unit'),4,17)
    assert a['proxy_loss']==b['proxy_loss']


def test_alpha_one_returns_original_proposal_exactly():
    old=torch.tensor([[1e10,1e-12],[1e10,-1e10]])
    proposal=torch.tensor([[.125,9.],[-.75,10.]])
    assert apply_update(old,proposal,torch.ones(2)) is proposal


@pytest.mark.parametrize('alpha',GRID)
def test_fixed_update_analytical(alpha):
    old=torch.tensor([[2.,-3.]])
    proposal=torch.tensor([[4.,1.]])
    s=TAPSScheduler(TAPSConfig(scheduler_mode='fixed',fixed_alpha=alpha))
    a,_=s.observe(proposal-old,('a',))
    assert torch.allclose(apply_update(old,proposal,a),old+alpha*(proposal-old))


def test_ema_and_bias_correction_independent_weighted_formula():
    s=TAPSScheduler(TAPSConfig(warmup_loops=0))
    history=[]
    for v in ([1.,2.],[-2.,1.],[3.,-4.],[0.,1.]):
        history.append(v)
        a,e=s.observe(torch.tensor([v]),('a',))
        w=torch.tensor([.8**j for j in reversed(range(len(history)))],dtype=torch.float64)
        w=w/w.sum(); h=torch.tensor(history,dtype=torch.float64)
        mean=(w[:,None]*h).sum(0)
        p=float(mean.square().mean()); energy=float((w*h.square().mean(1)).sum())
        f=max(energy-p,0.)
        expected=max(.8,min(1.2,1+.275*(p-1.5*f)/(p+1.5*f+1e-12)))
        assert float(e['progress'][0])==pytest.approx(p,abs=2e-6)
        assert float(e['fluctuation'][0])==pytest.approx(f,abs=2e-6)
        assert float(a[0])==pytest.approx(expected,abs=2e-6)
    assert torch.allclose(s.state.mean,(1-.8**4)*mean.float().unsqueeze(0),atol=1e-6)
    assert s.state.step==4


def test_zero_updates_finite_and_undefined_cosine_is_null():
    s=TAPSScheduler(TAPSConfig())
    for _ in range(8):
        a,e=s.observe(torch.zeros(2,3,4),('a','b'))
        assert torch.equal(a,torch.ones(2))
        for k in ('progress','fluctuation','progress_fluctuation_ratio','balance'):
            assert torch.isfinite(e[k]).all()
        assert scalar_events(e)[0]['delta_cosine'] is None


def test_alpha_bounds_on_adversarial_trajectories():
    s=TAPSScheduler(TAPSConfig(warmup_loops=0))
    g=torch.Generator().manual_seed(9)
    for k in range(64):
        a,_=s.observe(torch.randn(3,2,7,generator=g)*(-1)**k,('a','b','c'))
        assert bool(((a>=.8)&(a<=1.2)).all())


def test_refresh_observes_every_update_but_holds_alpha():
    s=TAPSScheduler(TAPSConfig(refresh_interval=3))
    seen=[]; alphas=[]
    for k in range(8):
        a,e=s.observe(torch.full((1,3),(-1.)**k),('a',))
        seen.append(e['refreshed']); alphas.append(float(a[0]))
    assert seen==[False,True,False,False,True,False,False,True]
    assert alphas[1]==alphas[2]==alphas[3]
    assert s.state.step==8


def test_reset_reproduces_independent_trajectory():
    s=TAPSScheduler(TAPSConfig()); values=[]
    for _ in range(2):
        s.reset(); run=[]
        for d in (1.,-1.,2.):
            a,_=s.observe(torch.full((1,2),d),('a',)); run.append(float(a[0]))
        values.append(run)
    assert values[0]==values[1]


@pytest.mark.parametrize('mode',['taps','random'])
def test_batch_isolation_and_permutation(mode):
    c=TAPSConfig(scheduler_mode=mode,random_pool=(.8,1.,1.2))
    batch=TAPSScheduler(c); individuals=[TAPSScheduler(c),TAPSScheduler(c)]; perm=TAPSScheduler(c)
    for k in range(6):
        d=torch.tensor([[1.,1.],[(-1.)**k,0.]])
        a,_=batch.observe(d,('a','b')); reverse,_=perm.observe(d.flip(0),('b','a'))
        independent=torch.cat([s.observe(d[i:i+1],(name,))[0] for i,(name,s) in enumerate(zip(('a','b'),individuals))])
        assert torch.equal(a,independent) and torch.equal(a,reverse.flip(0))


def test_changed_identity_requires_reset():
    s=TAPSScheduler(TAPSConfig()); s.observe(torch.ones(1,2),('a',))
    with pytest.raises(ValueError,match='reset'): s.observe(torch.ones(1,2),('b',))


def test_masked_padding_does_not_change_moments():
    a=TAPSScheduler(TAPSConfig(normalization_mode='masked_mean'))
    b=TAPSScheduler(TAPSConfig())
    for k in range(4):
        v=torch.tensor([[[1.,float(k)],[1000.,-1000.]]])
        aa,ae=a.observe(v,('a',),mask=torch.tensor([[True,False]]))
        ba,be=b.observe(v[:,:1],('a',))
        assert torch.equal(aa,ba)
        assert torch.equal(ae['progress'],be['progress'])


def test_logging_does_not_change_outputs_or_statistics():
    c=fixtures('DEV')[1]; log=[]
    r1,e1=rollout(c,TAPSConfig(),4,17)
    r2,e2=rollout(c,TAPSConfig(),4,17,sink=log.append)
    assert r1['proxy_loss']==r2['proxy_loss'] and len(log)==5
    for a,b in zip(e1,e2):
        assert {k:v for k,v in a.items() if 'seconds' not in k}=={k:v for k,v in b.items() if 'seconds' not in k}


def test_labels_do_not_influence_controller():
    c=fixtures('DEV')[1]
    _,a=rollout(c,TAPSConfig(),4,17)
    _,b=rollout(replace(c,label=(999.,999.)),TAPSConfig(),4,17)
    assert [e['alpha'] for e in a]==[e['alpha'] for e in b]


def calibration_fixture():
    cases=fixtures('DEV')[:1]; rows=[]; traces=[]
    for seed in SEEDS:
        for alpha in GRID:
            r,_=rollout(cases[0],TAPSConfig(scheduler_mode='fixed',fixed_alpha=alpha),1,seed,arm='A2');rows.append(r)
        _,e=rollout(cases[0],TAPSConfig(),1,seed);traces+=e
    return cases,rows,traces


def test_test_parent_rejected_before_scores_are_accessed():
    c=fixtures('TEST')[0]
    class Poison(dict):
        def __getitem__(self,k): raise AssertionError('calibration touched a protected score')
    with pytest.raises(PermissionError): select_on_dev([c],[Poison()],[],TAPSConfig(),[c.parent_id])


def test_test_alpha_pool_rejected():
    c,r,t=calibration_fixture();t[0]={**t[0],'split':'TEST'}
    with pytest.raises(PermissionError): select_on_dev(c,r,t,TAPSConfig(),['TEST-parent-monotone'])


def test_duplicate_calibration_cells_rejected():
    c,r,t=calibration_fixture()
    with pytest.raises(ValueError): select_on_dev(c,r+[r[0]],t,TAPSConfig(),[])


def test_oracle_cannot_access_test_before_freeze():
    with pytest.raises(PermissionError): add_oracles(fixtures('TEST'),[],TAPSConfig(),(1,),lambda x:None)


def test_frozen_configuration_and_parent_binding():
    c,r,t=calibration_fixture(); protected=['TEST-parent-monotone']
    plan=select_on_dev(c,r,t,TAPSConfig(),protected)
    with pytest.raises(PermissionError): evaluate_test(fixtures('TEST')[:1],TAPSConfig(beta=.5),(1,),plan,lambda x:None,lambda x:None)
    with pytest.raises(PermissionError): evaluate_test(fixtures('TEST')[1:2],TAPSConfig(),(1,),plan,lambda x:None,lambda x:None)


def test_random_schedule_is_seeded_and_nonconstant():
    c=TAPSConfig(scheduler_mode='random',random_pool=(.8,.8,1.2))
    def sequence():
        s=TAPSScheduler(c)
        return [float(s.observe(torch.ones(1,2),('a',))[0][0]) for _ in range(80)]
    assert sequence()==sequence() and len(set(sequence()))==2


@pytest.mark.parametrize('bad',[{'beta':1.},{'beta':-1.},{'gamma':0.},{'rho':-1.}, {'eps':0.},
    {'alpha_min':0.},{'alpha_max':.5},{'alpha_init':.9},{'fixed_alpha':3.},
    {'refresh_interval':0},{'refresh_interval':True},{'warmup_loops':-1},
    {'scheduler_mode':'adam'},{'normalization_mode':'sum'},{'seed':True},
    {'enabled':1},{'rho':float('nan')},{'scheduler_mode':'random'}, {'random_pool':(.1,)}])
def test_malformed_config_fails(bad):
    with pytest.raises(ValueError): TAPSConfig(**bad)


def test_nonfinite_delta_and_overflow_fail_closed():
    for d in (float('nan'),float('inf'),1e30):
        with pytest.raises(FloatingPointError): TAPSScheduler(TAPSConfig()).observe(torch.full((1,2),d),('a',))


def test_memory_guard_boundaries():
    memory_guard(100,80,reserve_bytes=20)
    with pytest.raises(MemoryError): memory_guard(99,80,reserve_bytes=20)
    with pytest.raises(ValueError): memory_guard(True,80)


def test_append_only_outputs(tmp_path):
    path=tmp_path/'r.json';write_json(path,{'status':'FAIL'})
    with pytest.raises(FileExistsError):write_json(path,{'status':'PASS'})
    with Journal(tmp_path/'r.jsonl') as log: log.append({'a':1});log.append({'a':2})
    with pytest.raises(FileExistsError): Journal(tmp_path/'r.jsonl')
    assert len((tmp_path/'r.jsonl').read_text().splitlines())==2


class Layer(torch.nn.Module):
    def __init__(self,i): super().__init__();self.i=i;self.seen=[]
    def forward(self,hidden_states,**kw):
        self.seen.append(kw)
        return hidden_states*(.9+.01*self.i)+.01*self.i


class Host(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(17)
        self.embed=torch.nn.Embedding(32,4)
        self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Layer(i) for i in range(6)])
        self.config=SimpleNamespace(layer_types=['full_attention']*6)
        self.eval().requires_grad_(False)
    def forward(self,input_ids,attention_mask=None,use_cache=False,past_key_values=None):
        h=self.embed(input_ids);pos=torch.arange(input_ids.shape[1])
        for l in self.model.layers:h=l(h,position_ids=pos,attention_mask=attention_mask,past_key_values=past_key_values)
        return h


@pytest.mark.parametrize('layers',[(3,),(3,4)])
@pytest.mark.parametrize('extra',[0,1])
def test_host_unit_matches_independent_manual_repeat(layers,extra):
    m=Host();ids=torch.tensor([[1,4,6]])
    with torch.inference_mode():
        expected=m.embed(ids)
        for i,l in enumerate(m.model.layers):
            expected=l(expected)
            if i==layers[-1] and extra:
                for j in layers:expected=m.model.layers[j](expected)
        with BlockRecurrence(m,layers,extra,config=TAPSConfig(scheduler_mode='unit')) as w:
            actual=m(ids,use_cache=False)
            assert torch.equal(actual,expected)
            assert w.extra_layer_applications==extra*len(layers)
        assert all('_taps_owner' not in b.__dict__ for b in m.model.layers)
        assert not m._forward_pre_hooks


def test_host_disabled_exact_native_baseline_and_state_reset():
    m=Host();ids=torch.tensor([[1,4]])
    with torch.inference_mode():
        base=m(ids,use_cache=False)
        with BlockRecurrence(m,(3,4),1,enabled=False):assert torch.equal(base,m(ids,use_cache=False))
        with BlockRecurrence(m,(3,4),1) as w:
            a=m(ids,use_cache=False);b=m(ids,use_cache=False)
            assert torch.equal(a,b)
            assert [r['iteration'] for r in w.records]==[0,1,0,1]


def test_host_first_pass_seeds_whole_block_delta():
    m=Host();ids=torch.tensor([[1,4]])
    with torch.inference_mode():
        h=m.embed(ids)
        for i in range(3):h=m.model.layers[i](h)
        expected=m.model.layers[4](m.model.layers[3](h))-h
        with BlockRecurrence(m,(3,4),1) as w:m(ids,use_cache=False)
        assert w.records[0]['delta_norm']==pytest.approx(float(expected.norm()))
        assert w.records[1]['refreshed'] is True


@pytest.mark.parametrize('kwargs',[{}, {'use_cache':True}, {'use_cache':False,'past_key_values':object()}])
def test_host_cache_rejected_before_forward(kwargs):
    m=Host()
    with torch.inference_mode(),BlockRecurrence(m,(3,4),1) as w:
        with pytest.raises(RuntimeError):m(torch.tensor([[1,4]]),**kwargs)
        assert w.calls==0


def test_host_no_training_overlap_or_extra_unsafe_loops():
    m=Host()
    with pytest.raises(ValueError):BlockRecurrence(m,(3,4),2)
    with BlockRecurrence(m,(3,4),1):
        with pytest.raises(RuntimeError):m(torch.tensor([[1,4]]),use_cache=False)
        with pytest.raises(RuntimeError):
            with BlockRecurrence(m,(4,5),1):pass
    assert not m._forward_pre_hooks


def test_exception_restores_original_forward():
    m=Host();b=m.model.layers[4];before=b.forward
    with pytest.raises(RuntimeError):
        with BlockRecurrence(m,(3,4),1):raise RuntimeError('injected')
    assert b.forward==before and not m._forward_pre_hooks


def test_host_reuses_identical_positions_and_masks():
    m=Host();mask=torch.ones(1,2,dtype=torch.long)
    with torch.inference_mode(),BlockRecurrence(m,(3,4),1):m(torch.tensor([[1,4]]),attention_mask=mask,use_cache=False)
    for i in (3,4):
        seen=m.model.layers[i].seen
        assert len(seen)==2 and seen[0]['position_ids'] is seen[1]['position_ids']
        assert seen[0]['attention_mask'] is seen[1]['attention_mask']


def test_native_preflight_never_loads_missing_model(tmp_path):
    r=native_preflight(tmp_path/'absent')
    assert r['status']=='BLOCKED' and r['native_runs']==0 and not r['training_attempted']


def test_complete_factorial_compute_ledger_and_no_native_claim(tmp_path):
    out=run_fixture_experiment(tmp_path/'fixture')
    summary=json.loads((out/'summary.json').read_text()); ledger=json.loads((out/'compute_ledger.json').read_text())
    assert summary['fixture_only'] and summary['native_runs']==0
    assert ledger['oracle_search_rollouts']==400
    assert ledger['oracle_search_core_calls']==1200
    assert (out/'frozen_plan.json').stat().st_mtime_ns < (out/'summary.json').stat().st_mtime_ns
    assert all(r['scheduler_capture_ratio'] is None for r in summary['headroom'] if not r['adaptive_headroom'])
    assert any(r['status']=='NOT_ENUMERATED_CAP' for r in summary['headroom'])
    with pytest.raises(FileExistsError):run_fixture_experiment(out)
