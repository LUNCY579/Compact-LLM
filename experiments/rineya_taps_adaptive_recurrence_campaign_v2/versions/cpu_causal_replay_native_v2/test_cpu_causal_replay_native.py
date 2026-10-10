from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import cpu_causal_replay_native as replay
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_sensitivity import FixtureModel


def test_exact_past_only_prefix_sequence_and_EOS_not_token_membership():
    prompt=[3,5];answer=[5,7,5]
    assert replay.causal_prefix(prompt,answer,0)==prompt
    assert replay.causal_prefix(prompt,answer,2)==[3,5,5,7]
    assert replay.causal_prefix(prompt,answer,3,'EOS')==[3,5,5,7,5]
    with pytest.raises(ValueError):replay.causal_prefix(prompt,answer,3)
    with pytest.raises(ValueError):replay.causal_prefix([1]*256,answer,1)
    with pytest.raises(ValueError):replay.causal_prefix(prompt,answer,2,'EOS')


def test_accepted_EOS_mass_differs_from_single_EOT_CE_and_matches_probability():
    logits=torch.tensor([1.,2.,3.],requires_grad=True)
    mass,eot=replay.objective(logits,kind='EOS',eos_ids=(1,2))
    expected=-(logits.softmax(0)[1]+logits.softmax(0)[2]).log()
    assert torch.allclose(mass,expected) and mass<eot
    gradient,=torch.autograd.grad(mass,logits)
    assert gradient.abs().sum()>0


@pytest.mark.parametrize('kind,target',[('answer',0),('EOS',None)])
def test_full_nonlinear_suffix_offset_grad_dot_and_three_fresh_rollout_FD(kind,target,tmp_path):
    model=FixtureModel();ids=torch.tensor([[3,5,7]]);values=[]
    for alpha,grad in ((1.,True),(1.01,False),(.99,False)):
        values.append(replay.rollout(model,ids,target,alpha,kind=kind,gradient=grad,guard=lambda:None,
            eos_ids=(1,2),vector_path=tmp_path/'G_d2.npz' if grad else None))
    nominal,plus,minus=values;gate=replay.discrepancy_gate(nominal,plus,minus)
    assert gate['absolute_error']<=gate['registered_tolerance'] and model.forward_calls==3
    assert all(torch.equal(x,ids) for x in model.seen_inputs)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert not any('_sensitivity_owner' in layer.__dict__ for layer in model.model.layers)
    import numpy as np
    with np.load(tmp_path/'G_d2.npz') as vectors:
        actual=float((vectors['G']*vectors['d2']).sum())
    assert actual==pytest.approx(nominal['sensitivity'],abs=1e-6)
    assert nominal['qualification_vectors']['file_sha256']==replay.sha(tmp_path/'G_d2.npz')
    if kind=='EOS':assert nominal['canonical_EOT_CE'] is not None


def test_feature_sink_is_label_independent_prompt_unit_and_precedes_loss(monkeypatch):
    model=FixtureModel();ids=torch.tensor([[3,5,7]]);order=[];features=[]
    original=replay.objective
    def objective(*args,**kwargs):order.append('loss');return original(*args,**kwargs)
    monkeypatch.setattr(replay,'objective',objective)
    def sink(value):order.append('feature');features.append(value)
    replay.rollout(model,ids,0,1.,guard=lambda:None,input_feature_sink=sink)
    replay.rollout(model,ids,1,1.,guard=lambda:None,input_feature_sink=sink)
    assert order==['feature','loss','feature','loss'] and features[0]==features[1]
    assert set(features[0])=={'progress','fluctuation','balance'}
    with pytest.raises(RuntimeError,match='predictor freeze'):
        replay.rollout(model,ids,0,.99,guard=lambda:None,input_feature_sink=sink)
    frozen={'progress':.2,'balance':.1,'case_id':'case','target_independent':True}
    expected=replay.digest(frozen)
    assert replay.digest(json.loads(json.dumps(frozen,sort_keys=True)))==expected


def test_case_mean_endpoint_derivative_and_EOS_remain_separate():
    def target(loss,sensitivity):
        return {'nominal':{'loss':loss,'sensitivity':sensitivity},'plus':{'loss':loss+.01*sensitivity},'minus':{'loss':loss-.01*sensitivity}}
    result=replay.aggregate_case([target(1.,.2),target(3.,-.1)],{'secondary':9},{'balance':.4})
    assert result['mean_answer_CE']['nominal']==2 and result['sum_answer_CE']['nominal']==4
    assert result['mean_answer_sensitivity']==pytest.approx(.05)
    assert result['EOS_secondary']=={'secondary':9} and result['mean_level_FD']['absolute_error']<1e-10


def test_bound_complete192_DEV_metadata_detects_future_or_missing_targets():
    package=Path(replay.__file__).resolve().parent
    packet=json.loads((package/'CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json').read_text())
    export=package/'runs/causal-replay-DEV-export-001'
    bank=json.loads((export/'DEVELOPMENT_BANK.json').read_text());tokens=json.loads((export/'DEVELOPMENT_TOKEN_BINDING.json').read_text())
    replay.verify_packet(packet,bank,tokens)
    bad=deepcopy(packet);bad['rows'][0]['answer_targets'][0]['prefix_token_ids'].append(bad['rows'][0]['canonical_answer_token_ids'][0])
    with pytest.raises(RuntimeError,match='future target'):replay.verify_packet(bad,bank,tokens)
    bad=deepcopy(packet);bad['rows'][0]['answer_targets'].pop()
    with pytest.raises(RuntimeError,match='omitted'):replay.verify_packet(bad,bank,tokens)


def test_full_forecast_refuses_missing_EOS_or_unaffordable_complete632targets():
    events=[{'event':'REPLAY_FORWARD_COMPLETED','label':label,'alpha':alpha,'state_gradient_evaluations':grad,'elapsed_seconds':1. if grad else .5}
        for label in ('qualification_maximum256_answer','qualification_maximum256_EOS') for alpha,grad in ((1.,1),(1.01,0),(.99,0))]
    assert replay.forecast(events,10)['remaining_forecast_seconds']==632*2*1.5
    assert replay.forecast(events,10)['admitted']
    assert not replay.forecast(events,3500)['admitted']
    with pytest.raises(RuntimeError,match='incomplete'):replay.forecast(events[:3],0)


def test_scalar_failure_retains_partial_and_restores_native_hooks():
    model=FixtureModel();ids=torch.tensor([[3,5,7]]);before=[x.forward for x in model.model.layers];model.fail_suffix=True
    events=[]
    class Journal:
        def append(self,value):events.append(value)
    rec=replay.Recorder(Journal())
    with pytest.raises(RuntimeError,match='suffix'):rec.triplet(model,ids,0,kind='answer',guard=lambda:None,label='failure')
    assert events[-1]['event']=='REPLAY_FORWARD_FAILED' and events[-1]['partial_compute']=='UNKNOWN'
    assert 'Traceback' in events[-1]['traceback']
    assert [x.forward for x in model.model.layers]==before
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in model.model.layers)


def test_exact1925_API_635_adjoint_252696_ledger_refuses_missing_or_reweighted_matrix():
    events=[]
    for index in range(18):
        events.append({'event':'VALUE_FORWARD_STARTED'})
        events.append({'event':'VALUE_FORWARD_FAILED' if index==16 else 'VALUE_FORWARD_COMPLETED','prefix_tokens':84 if index<8 else 256,
            'partial_compute':'UNKNOWN','error':'EXPECTED_CAUSAL_SUFFIX_EXCEPTION','state_gradient_evaluations':0})
    lengths=[84]*3+[256]*6
    package=Path(replay.__file__).resolve().parent
    packet=json.loads((package/'CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json').read_text())
    for row in packet['rows']:
        for target in row['answer_targets']:lengths.extend([target['prefix_tokens']]*3)
        lengths.extend([row['EOS']['prefix_tokens']]*3)
    assert len(lengths)==1905
    for index,length in enumerate(lengths):
        events.extend([{'event':'REPLAY_FORWARD_STARTED'},{'event':'REPLAY_FORWARD_COMPLETED','prefix_tokens':length,'state_gradient_evaluations':int(index%3==0)}])
    for use_cache,present in ((True,False),(False,True)):
        events.extend([{'event':'CACHE_REJECTION_STARTED','use_cache':use_cache,'cache_object_present':present},
                       {'event':'CACHE_REJECTED','model_compute_started':False,'wrapper_forward_count':0}])
    actual=replay.ledger(events,192)
    assert actual['model_API_attempts']==1925 and actual['state_adjoints']==635 and actual['actual_completed_prefix_positions']==252696
    assert actual['partial_compute_positions'] is None
    with pytest.raises(RuntimeError,match='ledger'):replay.ledger(events[:-1],192)
    with pytest.raises(RuntimeError,match='ledger'):replay.ledger(events,191)


def test_premodel_exact_qualification_DEV_allowlist_and_hash_guard_without_loader(monkeypatch):
    exact=replay.EXACT_QUALIFICATION_DEV
    expected='b75df9368e27cfc7de58e55370090e55bd5bcb4d1bd86aeb079b486555925da0'
    replay.verify_source_closure({exact:expected})
    with pytest.raises(RuntimeError,match='source closure'):replay.verify_source_closure({exact:'wrong'})
    smoke='Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl'
    with pytest.raises(RuntimeError,match='source closure'):
        replay.verify_source_closure({smoke:replay.sha(replay.REPO/smoke)})
    # Classification is tested on an existing unrelated source path, avoiding
    # any protected/TEST file access or synthetic native model construction.
    other='MODEL_QUALITY_STATUS.md'
    with pytest.raises(RuntimeError,match='source closure'):
        replay.verify_source_closure({other:replay.sha(replay.REPO/other)})
    def forbidden_hash(_path):raise AssertionError('forbidden data must not be read or hashed')
    monkeypatch.setattr(replay,'sha',forbidden_hash)
    for forbidden in ('Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/TEST.jsonl',
                      'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/other.jsonl'):
        with pytest.raises(RuntimeError,match='source closure'):
            replay.verify_source_closure({forbidden:'unused'})


def test_worker_missing_actual_grant_after_source_review_stops_before_source_loader(monkeypatch,tmp_path):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import causal_replay_authority as authority
    # The fabricated fixture has no grant, and cannot authorize a native run.
    fake_authority=tmp_path/'FABRICATED_NO_NATIVE_AUTHORITY';fake_authority.mkdir()
    monkeypatch.setattr(authority,'PACKAGE',fake_authority)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setenv('CPU_CAUSAL_REPLAY_PARENT_STARTED_PERF_COUNTER',str(replay.time.perf_counter()))
    admission=json.loads((replay.PACKAGE/'runs/native-cpu-causal-replay-001/admission.json').read_text())
    admission.update(requires_additional_human_authorization=True,
        human_extension_path=str(fake_authority/'HUMAN_CAUSAL_REPLAY_ALLOWANCE_001.json'),human_extension_sha256='ABSENT')
    admission['dispatch_hashes']={replay.EXACT_QUALIFICATION_DEV:'b75df9368e27cfc7de58e55370090e55bd5bcb4d1bd86aeb079b486555925da0'}
    review=tmp_path/'review.json';admission['review_receipt']=str(review)
    path=tmp_path/'admission.json';path.write_text(json.dumps(admission))
    review.write_text(json.dumps({'status':'PASS_CPU_CAUSAL_REPLAY_DISPATCH','admission_sha256':replay.sha(path)}))
    order=[];closure=replay.verify_source_closure
    def checked_closure(values):order.append('source');return closure(values)
    def checked_scope(value):order.append('grant');return authority.validate_human_scope(value)
    def forbidden_loader(*_args,**_kwargs):raise AssertionError('missing grant reached source/model loader')
    monkeypatch.setattr(replay,'verify_source_closure',checked_closure)
    monkeypatch.setattr(replay,'validate_human_scope',checked_scope)
    monkeypatch.setattr(replay,'admitted_source',forbidden_loader)
    out=tmp_path/'result';code=replay.run(admission['source'],out,path)
    result=json.loads((out/'receipt.json').read_text())
    assert order==['source','grant']
    assert code==2 and result['status']=='FAIL' and 'EXPLICIT_ADDITIONAL_ATTEMPT_REQUIRED' in result['error']
    assert result['native_model_loads']==result['model_load_attempts']==0
    assert not (out/'population.json').exists() and not (out/'hf_cache').exists()
