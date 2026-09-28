"""Small synthetic tests only; no pretrained efficacy or GPU qualification."""
from dataclasses import replace
import copy
import math
import pytest
import torch
from experiments.architecture_benchmark_001.budget import Config
from experiments.architecture_benchmark_001.model import DenseReference, RMSNorm, rope, dense_attention
from experiments.architecture_benchmark_001.losses import recovery_loss, recovery_loss_from_hidden
from experiments.architecture_benchmark_001.numerics import compare_logits, compare_paths, continuation_probe


def model(ced=False, dtype=torch.float32):
    torch.manual_seed(31)
    c=Config(vocab_size=37, hidden_size=24, intermediate_size=48, num_attention_heads=6,
             num_key_value_heads=3, head_dim=8, encoder_layers=2 if ced else 4,
             decoder_layers=2 if ced else 0, architecture='ced' if ced else 'plain')
    return DenseReference(c).to(dtype=dtype).eval()


@pytest.mark.parametrize('chunk', [1, 3, 32])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_hidden_loss_exact_objective_and_gradients(chunk, checkpoint):
    torch.manual_seed(17)
    sh=torch.randn(2,6,7,requires_grad=True); wh=torch.randn(37,7,requires_grad=True)
    th=torch.randn(2,6,11,requires_grad=True); tw=torch.randn(37,11,requires_grad=True)
    ids=torch.tensor([[1,2,3,4,5,6],[7,8,9,10,-100,12]])
    valid=torch.tensor([[0,1,1,1,1,1],[1,1,1,1,1,0]])
    seg=torch.tensor([[0,0,0,1,1,1],[0,0,0,0,1,1]])
    a=recovery_loss(sh@wh.T,th@tw.T,ids,valid=valid,segments=seg,alpha=.7,temperature=2,chunk_tokens=chunk)
    ga=torch.autograd.grad(a,(sh,wh),retain_graph=True)
    work={}
    b=recovery_loss_from_hidden(sh,wh,th,tw,ids,valid=valid,segments=seg,alpha=.7,
                               temperature=2,chunk_tokens=chunk,checkpoint_chunks=checkpoint,work_counter=work)
    gb=torch.autograd.grad(b,(sh,wh))
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
    assert th.grad is None and tw.grad is None
    assert work['teacher_head_positions'] >= 6
    assert work['student_head_positions'] == work['teacher_head_positions']
    assert work['teacher_head_positions'] == (12 if checkpoint else 6)


@pytest.mark.parametrize('ced',[False,True])
def test_backbone_and_head_checkpointing_preserve_all_parameter_gradients(ced):
    a=model(ced).train(); b=copy.deepcopy(a).train(); b.gradient_checkpointing=True
    ids=torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
    torch.manual_seed(2)
    th=torch.randn(2,5,13); tw=torch.randn(37,13)
    old=recovery_loss(a(ids).logits,th@tw.T,ids)
    hidden=b(ids,compute_logits=False,return_hidden=True)
    assert hidden.logits is None and hidden.cache is None
    new=recovery_loss_from_hidden(hidden.hidden,b.embed_tokens.weight,th,tw,ids,chunk_tokens=2)
    old.backward(); new.backward()
    torch.testing.assert_close(old,new,atol=2e-6,rtol=2e-6)
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb and pa.grad is not None and pb.grad is not None
        torch.testing.assert_close(pa.grad,pb.grad,atol=5e-6,rtol=1e-4)
    assert not ced or b.memory.v_proj.weight.grad.abs().sum()>0


@pytest.mark.parametrize('kwargs',[{'alpha':2},{'temperature':0},{'chunk_tokens':0}])
def test_hidden_loss_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        recovery_loss_from_hidden(torch.randn(1,3,5),torch.randn(37,5),torch.randn(1,3,9),torch.randn(37,9),torch.tensor([[1,2,3]]),**kwargs)


def test_hidden_loss_rejects_nonfinite_head_and_empty_targets():
    sh=torch.randn(1,3,5,requires_grad=True); wh=torch.randn(37,5); wh[0,0]=float('inf')
    th=torch.randn(1,3,9); tw=torch.randn(37,9); ids=torch.tensor([[1,2,3]])
    with pytest.raises(ValueError,match='nonfinite'):
        recovery_loss_from_hidden(sh,wh,th,tw,ids)
    with pytest.raises(ValueError,match='no usable'):
        recovery_loss_from_hidden(sh,torch.randn(37,5),th,tw,ids,valid=torch.zeros_like(ids))


def test_global_scale_and_argmax_can_hide_strict_elementwise_failure():
    expected=torch.tensor([[20.,0.,-2.]])
    actual=torch.tensor([[20.,.0003,-2.]])
    r=compare_logits(expected,actual,atol=1e-4,rtol=1e-4)
    assert r['argmax_exact'] and r['within_global_scaled_tolerance_diagnostic']
    assert not r['strict_elementwise_pass'] and r['elementwise_violations']==1
    assert r['margin_certified_positions']==1 and r['mean_total_variation']>=0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_comparison_never_looks_like_zero_violations(bad):
    r=compare_logits(torch.tensor([[2.,0.]]),torch.tensor([[2.,bad]]),atol=1e-4,rtol=1e-4)
    assert not r['finite'] and not r['strict_elementwise_pass'] and not r['argmax_exact']
    assert r['elementwise_violations'] is None


@pytest.mark.parametrize('ced',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_multi_chunk_prefix_and_cached_continuations(ced,dtype):
    m=model(ced,dtype)
    ids=torch.tensor([[1,2,3,4,5,6,7,8]])
    tol=1e-10 if dtype==torch.float64 else 1e-4
    with torch.inference_mode():
        r=compare_paths(m,ids,chunk_sizes=(1,3,16),atol=tol,rtol=tol)
        c=continuation_probe(m,ids,steps=3,atol=tol,rtol=tol)
    assert r['all_strict_paths_pass'] and r['all_decode_choices_exact']
    assert r['causal_suffix_isolation_exact'] and len(r['chunks'])==4
    assert c['all_strict_paths_pass'] and c['first_choice_divergence_step'] is None


def test_fp64_norm_and_rope_are_really_double_and_native_frequency_order():
    torch.manual_seed(17)
    x=torch.randn(2,3,4,8,dtype=torch.float64)
    n=RMSNorm(8,1e-6).double()
    expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    assert torch.equal(n(x),expected)
    positions=torch.tensor([[10000,10001,10002,10003],[20000,20001,20002,20003]])
    inv=1.0/(1e6**(torch.arange(0,8,2,dtype=torch.float64)/8))
    angles=positions.double()[...,None]*inv
    angles=torch.cat([angles,angles],-1)[:,None]
    a,b=x.chunk(2,-1)
    expected=x*angles.cos()+torch.cat([-b,a],-1)*angles.sin()
    assert torch.equal(rope(x,positions,1e6),expected)
    assert not torch.equal(rope(x,positions,1e6),rope(x.float(),positions,1e6).double())


def test_metadata_diagnostic_left_padding_and_segments():
    m=model(True)
    ids=torch.tensor([[0,1,2,3,4,5]])
    valid=torch.tensor([[0,1,1,1,1,1]])
    seg=torch.tensor([[0,0,0,1,1,1]])
    pos=torch.tensor([[0,0,1,0,1,2]])
    with torch.inference_mode():
        r=compare_paths(m,ids,attention_mask=valid,segment_ids=seg,position_ids=pos)
    assert r['all_strict_paths_pass']


def test_hidden_only_requires_observable_output():
    m=model()
    with pytest.raises(ValueError): m(torch.tensor([[1,2]]),compute_logits=False)
