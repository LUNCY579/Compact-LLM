import json
import math
from pathlib import Path

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import cpu_h3_native as cpu
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.sensitivity import DiagnosticBlock
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_sensitivity import FixtureModel
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import TAPSConfig


def test_explicit_CPU_forward_and_diagnostic_values_match_at_all_registered_scales():
    model=FixtureModel();ids=torch.tensor([[3,5,7]])
    baseline=[layer.forward for layer in model.model.layers]
    for alpha,extra in ((1.,0),(.99,1),(1.,1),(1.01,1)):
        with torch.inference_mode(),BlockRecurrence(model,extra_loops=extra,config=TAPSConfig(scheduler_mode='fixed',fixed_alpha=alpha)) as block:
            cpu.install_cpu_conditioning(block);expected=cpu.forward_logits(model,ids)
        records=[]
        with torch.enable_grad(),cpu.diagnostic_rms(model,records),DiagnosticBlock(model,alpha=alpha,extra_loops=extra,device='cpu',guard=lambda:None):
            actual=cpu.forward_logits(model,ids).detach()
        assert torch.equal(expected,actual) and len(records)==1
        assert .25<=records[0]['rms_ratio_to_normal']<=4
        assert [layer.forward for layer in model.model.layers]==baseline
        assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.model.layers)


def test_CPU_fullsuffix_customhead_adjoint_matches_three_fresh_rollout_FD_no_param_grad():
    model=FixtureModel();ids=torch.tensor([[3,5,7]]);events=[]
    result=cpu.loss_diagnostic(model,ids,0,device='cpu',guard=lambda:None,synchronize=None,sink=events.append,label='fixture')
    assert model.forward_calls==3 and all(torch.equal(x,ids) for x in model.seen_inputs)
    assert result['full_model_calls']==3 and result['state_gradient_evaluations']==1
    assert result['absolute_error']<=result['registered_tolerance']
    assert result['nominal']['actual_block_RMS'] and result['nominal']['state_gradient_norm']>0
    assert sum(e.get('state_gradient_evaluations',0) for e in events)==1
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not result['gold_enters_forward_input'] and not result['generation_pool_access']
    assert all(e['target_enters_forward'] is False for e in events if e['event'].endswith('STARTED'))


def test_CPU_cache_position_device_and_forced_suffix_restoration_contracts():
    model=FixtureModel();ids=torch.tensor([[3,5,7]]);before=[layer.forward for layer in model.model.layers]
    for kwargs in ({'use_cache':True},{'use_cache':False,'past_key_values':object()}):
        with torch.enable_grad(),DiagnosticBlock(model,device='cpu',guard=lambda:None) as block:
            with pytest.raises(RuntimeError,match='cache'):model(input_ids=ids,attention_mask=torch.ones_like(ids),**kwargs)
            assert block.forwards==0
    model.bad_position=True
    with torch.enable_grad(),DiagnosticBlock(model,device='cpu',guard=lambda:None):
        with pytest.raises(RuntimeError,match='positions'):cpu.forward_logits(model,ids)
    model.bad_position=False;model.fail_suffix=True;events=[]
    with pytest.raises(RuntimeError,match='suffix'):
        cpu.loss_diagnostic(model,ids,0,device='cpu',guard=lambda:None,synchronize=None,sink=events.append,label='failure')
    assert events[-1]['completed'] is False and events[-1]['partial_compute']=='UNKNOWN'
    assert events[-1]['state_gradient_evaluations'] is None
    assert [layer.forward for layer in model.model.layers]==before
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.model.layers)
    with pytest.raises(RuntimeError,match='CPU native'):cpu.require_cpu_tensor(torch.empty(2,device='meta'))
    with pytest.raises(RuntimeError,match='batch1'):cpu.forward_logits(model,torch.ones(1,257,dtype=torch.long))


def test_fresh_CPU_maximum_whole_adjoint_cost_forecast_includes_all_seven_or_refuses():
    events=[{'event':'DIAGNOSTIC_FORWARD_COMPLETED','label':'qualification_maximum256','alpha':alpha,'state_gradient_evaluations':gradient,'elapsed_seconds':cost}
            for alpha,gradient,cost in ((1.,1,10.),(1.01,0,5.),(.99,0,6.))]
    forecast=cpu.complete_forecast(50.,events)
    assert forecast['remaining_forecast_seconds']==7*(10+2*6)*1.5
    assert forecast['admitted'] and forecast['device']=='cpu' and not forecast['GPU_timing_used']
    assert not cpu.complete_forecast(1799.,events)['admitted']
    with pytest.raises(RuntimeError,match='incomplete'):cpu.complete_forecast(0.,events[:2])
    events[0]['elapsed_seconds']=float('nan')
    with pytest.raises(RuntimeError,match='nonfinite'):cpu.complete_forecast(0.,events)


def test_CPU_host_guards_keep_historical_floor_peak_RSS_and_reserve(monkeypatch):
    assert cpu.CPU_INITIAL_FLOOR==7945689498
    from experiments.rineya_taps_adaptive_recurrence_v1.safety import memory_guard
    with pytest.raises(MemoryError):memory_guard(cpu.CPU_INITIAL_FLOOR-1,cpu.CPU_INITIAL_FLOOR,reserve_bytes=0)
    sample={'rss_bytes':1,'peak_rss_bytes':6*cpu.GIB+1,'host_available_bytes':2*cpu.GIB,'device':'cpu'}
    monkeypatch.setattr(cpu,'cpu_resources',lambda:sample)
    with pytest.raises(MemoryError,match='RSS'):cpu.resource_guard()
    sample['peak_rss_bytes']=1;sample['host_available_bytes']=cpu.GIB-1
    with pytest.raises(MemoryError,match='reserve'):cpu.resource_guard()


def test_original_all16_population_same_7_targets_no_choice_keyword_invention():
    package=Path(cpu.__file__).resolve().parent
    metadata=json.loads((package/'runs/binding-001/PROMPT_AUDIT.json').read_text())
    dev=[x for x in metadata if x['split']=='DEV']
    cases=[{'case_id':x['case_id'],'family':x['family'],'scenario_id':x['independent_cluster'],'scorer':x['scorer']} for x in dev]
    class Tokenizer:
        def encode(self,_answer,add_special_tokens=False):return [1,2]
    def canonical(case):
        scorer=case['scorer']
        if scorer['type']=='exact' and len(scorer['answers'])==1:return str(scorer['answers'][0])
        if scorer['type']=='numeric':return str(scorer['value'])
        if scorer['type']=='json_exact':return json.dumps(scorer['value'],separators=(',',':'),ensure_ascii=False)
        return None
    result=cpu.bind_population(cases,metadata,Tokenizer(),canonical)
    assert len(result)==16
    eligible=[x for x in result if x['status']=='ELIGIBLE']
    assert len(eligible)==7 and sum(x['input_tokens'] for x in eligible)==274
    assert sum(x['status']=='EXCLUDED_SHAPE' for x in result)==2
    assert all(x['target_token_id'] is None for x in result if x['status']!='ELIGIBLE')
    assert all(x['input_ids_sha256'] for x in result) and all(x['canonical_target_sha256'] for x in eligible)
    assert 11*84+12*256+3*274==4818


def test_CPU_module_has_no_GPU_forward_testing_escape_or_cuda_API():
    source=Path(cpu.__file__).read_text(encoding='utf-8')
    assert 'gpu_native' not in source and 'enforce_device' not in source and 'torch.cuda.' not in source
    with pytest.raises(RuntimeError,match='cannot select'):cpu.encode_prefix(None,{},'meta')


def test_completed_ledger_charges_expected_partial_and_cache_without_phantom_gradients():
    events=[]
    for index in range(18):
        events.append({'event':'VALUE_FORWARD_STARTED'})
        events.append({'event':'VALUE_FORWARD_FAILED' if index==16 else 'VALUE_FORWARD_COMPLETED',
                       'prefix_tokens':84 if index<8 else 256,'partial_compute':'UNKNOWN','error':'EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION'})
    for index in range(27):
        events.append({'event':'DIAGNOSTIC_FORWARD_STARTED'})
        length=84 if index<3 else 256 if index<6 else [84,37,30,34,30,27,32][(index-6)//3]
        events.append({'event':'DIAGNOSTIC_FORWARD_COMPLETED','prefix_tokens':length,'state_gradient_evaluations':int(index%3==0)})
    for use_cache,present in ((True,False),(False,True)):
        events.extend([{'event':'CACHE_REJECTION_STARTED','use_cache':use_cache,'cache_object_present':present},
                       {'event':'CACHE_REJECTED','model_compute_started':False,'wrapper_forward_count':0}])
    ledger=cpu.complete_ledger(events,16)
    assert ledger['model_API_attempts']==47 and ledger['completed_model_forwards']==44 and ledger['state_adjoints']==9
    assert ledger['partial_prefix_compute_positions'] is None
    assert ledger['completed_prefix_positions']==4818
    with pytest.raises(RuntimeError,match='ledger'):cpu.complete_ledger(events[:-1],16)
    with pytest.raises(RuntimeError,match='ledger'):cpu.complete_ledger(events,15)
