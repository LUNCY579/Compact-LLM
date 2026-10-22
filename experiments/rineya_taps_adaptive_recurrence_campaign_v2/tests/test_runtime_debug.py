from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug import (
    ALLOWANCE_NAME, CASE_IDS, SMOKE_IDS, DebugRecorder, LANE, count_forecast,
    generate, layer_journal, remaining_forecast, require_human_allowance,
    validate_ids, validate_positions, snapshot_bindings, check_restored,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_native import evaluate_case, install_gpu_timing
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.control import BlockRecurrence, TAPSConfig
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_gpu_semantics import (
    FakeModel, Tokenizer, CONFIG, case, flags, metrics, score, wrapper,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.binding import sha


def recorder(guard=None):
    operations=[]; forwards=[]
    value=DebugRecorder(operations.append,forwards.append,vocab_size=10,synchronize=lambda:None,
                       guard=guard or (lambda:None),enforce_device=False)
    return value,operations,forwards


@pytest.mark.parametrize('cap,sequence', [(24,(3,4,9)),(2,(3,4,5))])
def test_instrumented_greedy_tokens_scores_EOS_caps_and_prefixes_match_frozen_helper(cap,sequence):
    reference=FakeModel(sequence=sequence); target=FakeModel(sequence=sequence)
    with torch.inference_mode():
        original,_=evaluate_case(reference,Tokenizer(),case(cap),CONFIG,wrapper(),score,flags,metrics,enforce_device=False)
        rec,operations,forwards=recorder()
        observed,_=generate(target,Tokenizer(),case(cap),CONFIG,wrapper(),score,flags,rec,expected_ids=[1,2])
    assert observed['token_ids']==original['token_ids'] and observed['response']==original['response']
    assert observed['score']==original['score'] and observed['termination']==original['termination']
    assert [x.tolist() for x in target.seen]==[x.tolist() for x in reference.seen]
    assert observed['full_model_calls']==len(observed['token_ids']) and observed['gold_nll'] is None
    assert rec.model_attempts==rec.model_completed==len(observed['token_ids'])
    assert rec.generation_attempts==len(observed['token_ids'])
    assert len([e for e in forwards if e['event']=='FORWARD_COMPLETED'])==len(observed['token_ids'])
    names={e['operation'] for e in operations}
    assert {'MODEL_FORWARD','LOGITS_CLONE','FINITE_LOGITS','ARGMAX','ARGMAX_ITEM','CPU_DECODE','CPU_OUTPUT_FLAGS','RESOURCE_GUARD','TOKEN_TENSOR_CREATE','PREFIX_CAT','SYNC_AFTER_CAT','SYNC_END_DECODE'}<=names
    assert all(e.get('prefix_ids') is not None for e in operations)


@pytest.mark.parametrize('stage', ['MODEL_FORWARD','ARGMAX','ARGMAX_ITEM','RESOURCE_GUARD','TOKEN_TENSOR_CREATE','PREFIX_CAT','SYNC_AFTER_CAT'])
def test_stage_failure_preserves_CPU_prefix_full_traceback_and_stops_without_CUDA_recovery(stage):
    rec,ops,forwards=recorder();original=rec.step;failed=False
    def step(name,action,**kwargs):
        nonlocal failed
        if failed: raise AssertionError('postfault operation/recovery attempted')
        if name==stage:
            def failure(): raise RuntimeError('CUDA error: an illegal memory access was encountered')
            try: return original(name,failure,**kwargs)
            finally: failed=True
        return original(name,action,**kwargs)
    rec.step=step
    with torch.inference_mode(),pytest.raises(RuntimeError,match='illegal memory'):
        generate(FakeModel(),Tokenizer(),case(),CONFIG,wrapper(),score,flags,rec)
    assert rec.cuda_faulted and rec.first_unexpected_failure['operation']==stage
    assert 'Traceback' in rec.first_unexpected_failure['traceback']
    assert rec.first_unexpected_failure['prefix_ids']==[1,2]
    assert rec.first_unexpected_failure['cuda_context_usable'] is False
    if stage=='MODEL_FORWARD':
        assert rec.model_attempts==1 and rec.model_completed==0
        assert forwards[-1]['partial_compute']=='UNKNOWN'
    elif stage!='RESOURCE_GUARD': assert rec.model_completed==1
    assert ops[-1]['event']=='OP_FAILED'


def test_header_row_bounds_positions_and_prompt_binding_reject_invalid_input_before_model():
    assert validate_ids([151900],151936)==[151900]  # tokenizer length is not the embedding bound
    for values in ([-1],[151936],[True],[1.2],[],[1]*358):
        with pytest.raises(ValueError): validate_ids(values,151936)
    validate_positions([[0,1,2]],3)
    with pytest.raises(ValueError):validate_positions([[1,2,3]],3)
    rec,_,_=recorder();model=FakeModel()
    with pytest.raises(RuntimeError,match='prompt IDs differ'):
        generate(model,Tokenizer(),case(),CONFIG,wrapper(),score,flags,rec,expected_ids=[2,1])
    assert not model.seen and rec.model_attempts==0


def test_native_device_cache_and_model_API_ceiling_are_fail_closed_without_gpu_probe():
    rec,_,_=recorder();rec.enforce_device=True;rec.prefix_ids=[1,2]
    with pytest.raises(RuntimeError,match='cuda:0'):rec.forward(FakeModel(),torch.tensor([[1,2]]))
    rec,_,events=recorder();rec.prefix_ids=[1,2]
    with torch.inference_mode(),pytest.raises(RuntimeError,match='cache forbidden'):
        rec.forward(FakeModel(cache=True),torch.tensor([[1,2]]))
    assert rec.model_attempts==1 and rec.model_completed==0 and events[-1]['partial_compute']=='UNKNOWN'
    rec,_,_=recorder();rec.prefix_ids=[1,2];rec.model_attempts=588;model=FakeModel()
    with pytest.raises(RuntimeError,match='cap exhausted'):rec.forward(model,torch.tensor([[1,2]]))
    assert not model.seen and rec.model_attempts==588


def test_inherited_timer_finally_skips_sync_when_an_original_CUDA_exception_is_active():
    calls=[];events=[]
    rec=DebugRecorder(events.append,lambda _e:None,vocab_size=10,synchronize=lambda:calls.append('API'),guard=lambda:None,enforce_device=False)
    def original():
        try: raise RuntimeError('CUDA error: an illegal memory access was encountered')
        finally: rec.synchronize()
    with pytest.raises(RuntimeError,match='illegal memory'):original()
    assert rec.cuda_faulted and calls==[] and events[-1]['event']=='CUDA_SYNC_SKIPPED_AFTER_FAULT'
    rec.synchronize()
    assert calls==[]


def test_original_per_case_guard_refuses_late_forward_even_before_EOS_completion(monkeypatch):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import runtime_debug as module
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.binding import HISTORY
    inherited=json.loads(HISTORY.read_text(encoding='utf-8'))['config']['max_prompt_seconds']
    now=[10.];rec,_,_=recorder();rec.case_started=10.;rec.case_max_seconds=inherited
    monkeypatch.setattr(module.time,'perf_counter',lambda:now[0])
    assert rec.guard() is None
    now[0]=10.+inherited+.01
    with pytest.raises(TimeoutError,match='per-case'):rec.guard()


def test_actual_inherited_summarize_schema_and_A0_readout_gate_on_CPU_fakes():
    import sys
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.binding import ARMR, LAB
    sys.path[:0]=[str(ARMR/'src'),str(LAB/'src')]
    from armr.metrics import logit_metrics
    from armr.reporting import summarize
    config=json.loads((ARMR/'configs/reapplication_screen.json').read_text())
    with torch.inference_mode():
        rec,_,_=recorder();baseline,first=generate(FakeModel(),Tokenizer(),case(),config,wrapper(),score,flags,rec,logit_metric_fn=logit_metrics)
        rec,_,_=recorder();same,_=generate(FakeModel(),Tokenizer(),case(),config,wrapper(),score,flags,rec,base=first,logit_metric_fn=logit_metrics)
        assert summarize([same],[baseline],config)['go_fixed_smoke']
        assert same['readout_comparison_scope']=='actual same-case A0 first logits'
        rec,_,_=recorder();changed,_=generate(FakeModel(sequence=(4,4,9)),Tokenizer(),case(),config,wrapper(),score,flags,rec,base=first,logit_metric_fn=logit_metrics)
        gate=summarize([changed],[baseline],config)
        assert not gate['go_fixed_smoke'] and not gate['gate_checks']['logit_preservation']
        assert not gate['gate_checks']['kl_preservation'] and changed['top1_agreement']==0
        assert rec.model_attempts==len(changed['token_ids'])  # metrics add no model call


class Layer(torch.nn.Module):
    def forward(self,hidden_states,**kwargs): return hidden_states+.01*torch.sin(hidden_states)


class HookModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Layer() for _ in range(28)])
        self.config=SimpleNamespace(layer_types=['full_attention']*28)
        self.parameter=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.eval()
    def forward(self,input_ids,use_cache=False,**kwargs):
        hidden=input_ids.float().unsqueeze(-1).repeat(1,1,4)
        conditioning={'position_ids':torch.arange(input_ids.shape[1]).unsqueeze(0),
            'attention_mask':torch.ones(1,1,input_ids.shape[1],input_ids.shape[1]),'position_embeddings':None,'use_cache':False}
        for block in self.model.layers:hidden=block(hidden,**conditioning)
        return hidden


@pytest.mark.parametrize('forced', [False,True])
def test_metadata_hooks_and_recurrence_forward_bindings_restore_under_normal_and_forced_exception(forced):
    model=HookModel();rec,ops,_=recorder();rec.prefix_ids=[1,2]
    baseline=snapshot_bindings(model);before_hooks=[(len(b._forward_pre_hooks),len(b._forward_hooks)) for b in model.model.layers]
    def sink(_event):
        if forced:raise RuntimeError('FORCED_CPU_FIXTURE')
    with torch.inference_mode():
        original=model(input_ids=torch.tensor([[1,2]]),use_cache=False)
        context=pytest.raises(RuntimeError,match='FORCED_CPU_FIXTURE') if forced else __import__('contextlib').nullcontext()
        with context,layer_journal(model,rec),BlockRecurrence(model,extra_loops=0,config=replace(TAPSConfig(),scheduler_mode='unit'),sink=sink) as w:
            install_gpu_timing(w,lambda:None,require_cuda=False)
            value=model(input_ids=torch.tensor([[1,2]]),use_cache=False)
            assert torch.equal(value,original)
        check_restored(model,baseline)
        assert torch.equal(model(input_ids=torch.tensor([[1,2]]),use_cache=False),original)
    assert [(len(b._forward_pre_hooks),len(b._forward_hooks)) for b in model.model.layers]==before_hooks
    assert all(e.get('hidden_metadata') is not None for e in ops if e['operation'] in ('LAYER_ENTER','LAYER_EXIT'))


def test_exact_population_full_forecast_refusal_and_actual_allowance_file_cannot_be_template(tmp_path):
    counts=count_forecast()
    assert len(CASE_IDS)==3 and len(SMOKE_IDS)==2 and counts['physical_evaluations']==12+20==32
    assert counts['model_API_attempt_cap']==12*8+20*24+10+2==588
    assert remaining_forecast(20,[.1,.2])['admitted']
    assert not remaining_forecast(270,[.1])['admitted']
    with pytest.raises(ValueError):remaining_forecast(0,[])
    proposed=tmp_path/'PROPOSED_TEMPLATE_NOT_AUTHORIZED.json';proposed.write_text('{}')
    with pytest.raises(PermissionError,match='NEEDS_EXPLICIT'):
        require_human_allowance({'human_extension_path':str(proposed),'human_extension_sha256':sha(proposed)},scope=tmp_path,worker_sha='worker')
    target=tmp_path/ALLOWANCE_NAME
    admission={'human_extension_path':str(target),'human_extension_sha256':'missing'}
    with pytest.raises(PermissionError,match='missing'):require_human_allowance(admission,scope=tmp_path,worker_sha='worker')
    grant={'status':'EXPLICIT_ADDITIONAL_ATTEMPT_AUTHORIZED','lane':LANE,'attempts':1,'max_native_worker_seconds':300,
           'parent_run':'native-bulk-001','worker_sha256':'worker','training':False,'full_matrix_retry':False}
    target.write_text(json.dumps(grant));admission['human_extension_sha256']=sha(target)
    assert require_human_allowance(admission,scope=tmp_path,worker_sha='worker')==target
    grant['attempts']=2;target.write_text(json.dumps(grant));admission['human_extension_sha256']=sha(target)
    with pytest.raises(PermissionError,match='contract differs'):require_human_allowance(admission,scope=tmp_path,worker_sha='worker')
