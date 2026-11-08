"""Mechanical tests only: synthetic tensors and a Qwen3-shaped local fixture."""
from dataclasses import replace
import copy
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.native_sparse_attention_v1.core import NSAConfig, BlockCompressor, masked_softmax, sparse_attention
from experiments.native_sparse_attention_v1.qwen3 import Qwen3NSAAttention, convert_qwen3, restore_qwen3, rotary, valid_from_mask
from experiments.native_sparse_attention_v1.budget import parameter_budget, access_budget

torch.set_num_threads(1)


def setup(t=17, *, dtype=torch.float32, seed=19, **changes):
    torch.manual_seed(seed)
    c = NSAConfig(block_size=4, selected_blocks=4, window_size=5, query_chunk_size=3,
                  compression_hidden=3, gate_hidden=5)
    c = replace(c, **changes)
    q = torch.randn(2, 4, t, 8, dtype=dtype)
    k, v = torch.randn(2, 2, t, 8, dtype=dtype), torch.randn(2, 2, t, 8, dtype=dtype)
    gates = torch.rand(2, 4, t, 3, dtype=dtype)
    valid = torch.ones(2, t, dtype=torch.bool)
    return q, k, v, gates, valid, c


def call(q, k, v, gates, valid, c, **kwargs):
    n = k.shape[2] // c.block_size
    shape = (*k.shape[:2], n, c.block_size, k.shape[-1])
    ck = k[:, :, :n*c.block_size].reshape(shape).mean(-2)
    cv = v[:, :, :n*c.block_size].reshape(shape).mean(-2)
    return sparse_attention(q, k, v, k, v, ck, cv, gates, valid, c, **kwargs)


def dense(q, k, v, valid):
    k, v = (x.repeat_interleave(q.shape[1] // x.shape[1], 1) for x in (k, v))
    t = q.shape[2]
    mask = torch.ones(t, t, dtype=torch.bool).tril()[None, None] & valid[:, None, None]
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    p = masked_softmax((q.to(acc) @ k.to(acc).transpose(-1, -2)) / math.sqrt(q.shape[-1]), mask)
    return (p @ v.to(acc)).to(q.dtype) * valid[:, None, :, None]


@pytest.mark.parametrize('field,value', [
    ('block_size',0), ('selected_blocks',0), ('window_size',-1), ('compression_hidden',0),
    ('gate_hidden',True), ('query_chunk_size',0), ('max_reference_tokens',0),
    ('max_scratch_bytes',0), ('force_initial_blocks',-1), ('force_recent_blocks',-1),
    ('control_seed',-1), ('kv_mode','bad'), ('selection_policy','oracle')])
def test_bad_config(field,value):
    with pytest.raises(ValueError):
        NSAConfig(**{field:value})


def test_mandatory_inside_budget():
    with pytest.raises(ValueError):
        NSAConfig(selected_blocks=1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize('t', [1,3,4,5,17])
def test_selected_full_coverage_matches_dense(dtype,t):
    q,k,v,gates,valid,c=setup(t,dtype=dtype,selected_blocks=20)
    gates.zero_(); gates[...,1]=1
    out,_=call(q,k,v,gates,valid,c)
    tol=0.025 if dtype==torch.bfloat16 else 2e-6
    torch.testing.assert_close(out,dense(q,k,v,valid),atol=tol,rtol=tol)


def test_selected_full_coverage_gradient_oracle():
    q,k,v,gates,valid,c=setup(selected_blocks=20,dtype=torch.float64)
    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
    gates.zero_(); gates[...,1]=1
    out,_=call(q,k,v,gates,valid,c)
    expected=dense(q,k,v,valid)
    a=torch.autograd.grad(out.square().sum(),(q,k,v),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(q,k,v))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('mode',['nsa','local','compressed_local'])
@pytest.mark.parametrize('cut',[1,3,4,5,8,9,16])
def test_causal_prefix_equivalence(mode,cut):
    q,k,v,gates,valid,c=setup()
    full,_=call(q,k,v,gates,valid,c,mode=mode)
    pref,_=call(q[:,:,:cut],k[:,:,:cut],v[:,:,:cut],gates[:,:,:cut],valid[:,:cut],c,mode=mode)
    torch.testing.assert_close(full[:,:,:cut],pref,atol=1e-6,rtol=1e-6)


def test_future_perturbation():
    q,k,v,gates,valid,c=setup()
    out,_=call(q,k,v,gates,valid,c)
    k[:,:,9:]+=1000; v[:,:,9:]-=1000; q[:,:,9:]*=100
    changed,_=call(q,k,v,gates,valid,c)
    torch.testing.assert_close(out[:,:,:9],changed[:,:,:9],atol=0,rtol=0)


def test_incomplete_summary_is_invisible():
    q,k,v,gates,valid,c=setup(11)
    gates.zero_(); gates[...,0]=1
    out,_=call(q,k,v,gates,valid,c)
    assert torch.equal(out[:,:,:3],torch.zeros_like(out[:,:,:3]))
    assert out[:,:,3].abs().sum()>0


def test_local_dense_and_independent_gates():
    q,k,v,gates,valid,c=setup(selected_blocks=20,window_size=32)
    gates.zero_(); gates[...,1]=1; gates[...,2]=1
    out,_=call(q,k,v,gates,valid,c)
    torch.testing.assert_close(out,2*dense(q,k,v,valid),atol=2e-6,rtol=2e-6)
    local,_=call(q,k,v,gates,valid,c,mode='local')
    torch.testing.assert_close(local,dense(q,k,v,valid),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('policy',['learned','hash_control'])
def test_selection_budget_unique_shared_and_visible(policy):
    q,k,v,gates,valid,c=setup(41,selected_blocks=3,selection_policy=policy)
    _,debug=call(q,k,v,gates,valid,c,return_debug=True)
    chosen,okay=debug['selected_blocks'],debug['selected_valid']
    assert chosen.shape==(2,2,41,3)  # KV groups, NOT query heads.
    for t in range(41):
        for b in range(2):
            for g in range(2):
                ids=chosen[b,g,t][okay[b,g,t]].tolist()
                assert len(ids)==len(set(ids))<=3
                assert 0 in ids and t//4 in ids
                assert all(i*4<=t for i in ids)
    pref=call(q[:,:,:23],k[:,:,:23],v[:,:,:23],gates[:,:,:23],valid[:,:23],c)[0]
    torch.testing.assert_close(pref,call(q,k,v,gates,valid,c)[0][:,:,:23],atol=1e-6,rtol=1e-6)


def test_selection_follows_compressed_relevance():
    q,k,v,gates,valid,c=setup(16,selected_blocks=1,force_initial_blocks=0,force_recent_blocks=0)
    q.zero_(); q[...,0]=1
    k.zero_(); k[:,:,8:12,0]=100
    _,debug=call(q,k,v,gates,valid,c,return_debug=True)
    assert (debug['selected_blocks'][:,:,15,0]==2).all()


@pytest.mark.parametrize('chunk',[1,4,9,32])
def test_chunk_equivalence(chunk):
    q,k,v,gates,valid,c=setup()
    out,_=call(q,k,v,gates,valid,c)
    other,_=call(q,k,v,gates,valid,replace(c,query_chunk_size=chunk))
    torch.testing.assert_close(out,other,atol=1e-6,rtol=1e-6)


def test_padding_and_all_masked_finite_backward():
    q,k,v,gates,valid,c=setup()
    valid[0,7:]=False; valid[1]=False
    k.requires_grad_(); v.requires_grad_()
    out,_=call(q,k,v,gates,valid,c)
    assert torch.isfinite(out).all() and (out[1]==0).all() and (out[0,:,7:]==0).all()
    out.square().sum().backward()
    assert torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()
    with torch.no_grad(): k[0,:,7:]+=100; v[0,:,7:]-=100
    changed,_=call(q,k,v,gates,valid,c)
    torch.testing.assert_close(out,changed,atol=0,rtol=0)


@pytest.mark.parametrize('failure',['left_padding','gate_nan','gate_range','gate_dtype','tokens','scratch','kv_shape','gqa','mode'])
def test_core_fails_closed(failure):
    q,k,v,gates,valid,c=setup()
    kwargs={}
    if failure=='left_padding': valid[0,0]=False
    if failure=='gate_nan': gates[0,0,0,0]=float('nan')
    if failure=='gate_range': gates[0,0,0,0]=2
    if failure=='gate_dtype': gates=gates.double()
    if failure=='tokens': c=replace(c,max_reference_tokens=16,max_grad_reference_tokens=16)
    if failure=='scratch': c=replace(c,max_scratch_bytes=1)
    if failure=='kv_shape': v=v[:,:,:-1]
    if failure=='gqa': q=q[:,:3]
    if failure=='mode': kwargs['mode']='unknown'
    with pytest.raises((ValueError,RuntimeError)):
        call(q,k,v,gates,valid,c,**kwargs)


def test_compressor_gradients_and_empty():
    torch.manual_seed(17)
    module=BlockCompressor(4,8,3)
    x=torch.randn(2,2,17,8,requires_grad=True)
    out=module(x)
    assert out.shape==(2,2,4,8)
    out.square().sum().backward()
    for p in module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert module(x[:,:,:3]).shape==(2,2,0,8)
    assert (x.grad[:,:,16]==0).all()


class Norm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*self.weight


class NativeFixture(nn.Module):
    """Not Transformers and NOT a Rineya checkpoint. Mathematical interface fixture."""
    def __init__(self,w=24,h=4,g=2,d=8):
        super().__init__()
        self.config=SimpleNamespace(model_type='qwen3')
        self.head_dim=d; self.h=h; self.g=g; self.scaling=d**-0.5
        self.attention_dropout=0.; self.sliding_window=None
        self.q_proj=nn.Linear(w,h*d,bias=False); self.k_proj=nn.Linear(w,g*d,bias=False)
        self.v_proj=nn.Linear(w,g*d,bias=False); self.o_proj=nn.Linear(h*d,w,bias=False)
        self.q_norm=Norm(d); self.k_norm=Norm(d); self.calls=0
    def forward(self,hidden_states,position_embeddings,attention_mask,**kwargs):
        self.calls+=1
        b,t,w=hidden_states.shape; cos,sin=position_embeddings
        def project(layer,heads): return layer(hidden_states).view(b,t,heads,self.head_dim).transpose(1,2)
        q=rotary(self.q_norm(project(self.q_proj,self.h)),cos,sin)
        k=rotary(self.k_norm(project(self.k_proj,self.g)),cos,sin)
        v=project(self.v_proj,self.g)
        valid=valid_from_mask(attention_mask,b,t,hidden_states.device)
        out=dense(q,k,v,valid).transpose(1,2).reshape(b,t,-1)
        return self.o_proj(out),None


def native_input(w=24,d=8,t=13):
    torch.manual_seed(33)
    x=torch.randn(2,t,w)
    angles=torch.arange(t)[:,None]*torch.exp(-torch.arange(d//2)/max(d,1))
    angles=torch.cat((angles,angles),-1)[None].expand(2,-1,-1)
    return dict(hidden_states=x,position_embeddings=(angles.cos(),angles.sin()),attention_mask=None)


def test_baseline_hard_delegation_and_restoration():
    native=NativeFixture().eval(); data=native_input()
    out=native(**data)[0]; wrapper=Qwen3NSAAttention(native,NSAConfig())
    # In baseline mode cache/backend kwargs go untouched to the original.
    got=wrapper(**data,past_key_value=object(),anything='native')[0]
    torch.testing.assert_close(out,got,atol=0,rtol=0)
    assert native.calls==2 and wrapper.native.q_proj is native.q_proj
    def broken(*args): raise AssertionError('NSA must not execute')
    wrapper.key_compressor.forward=broken
    torch.testing.assert_close(wrapper(**data)[0],out,atol=0,rtol=0)


@pytest.mark.parametrize('kv_mode',['shared','independent'])
def test_active_local_preserves_rope_and_native_geometry(kv_mode):
    native=NativeFixture().eval(); data=native_input()
    c=NSAConfig(block_size=4,window_size=32,compression_hidden=4,kv_mode=kv_mode)
    wrapper=Qwen3NSAAttention(native,c,mode='local')
    torch.testing.assert_close(wrapper(**data)[0],native(**data)[0],atol=2e-6,rtol=2e-6)
    if kv_mode=='independent':
        assert wrapper.k_win.weight.data_ptr()!=native.k_proj.weight.data_ptr()
        torch.testing.assert_close(wrapper.k_win.weight,native.k_proj.weight,atol=0,rtol=0)


@pytest.mark.parametrize('kv_mode',['independent','shared'])
def test_native_full_width_budget_not_hidden_div_heads(kv_mode):
    native=NativeFixture(w=1024,h=16,g=8,d=128).eval()
    c=NSAConfig(kv_mode=kv_mode)
    wrapper=Qwen3NSAAttention(native,c,mode='nsa')
    data=native_input(w=1024,d=128,t=33)
    assert wrapper(**data)[0].shape==(2,33,1024)
    mc=dict(hidden_size=1024,num_attention_heads=16,num_key_value_heads=8,head_dim=128,
            num_hidden_layers=28,model_type='qwen3',attention_bias=False)
    actual=sum(p.numel() for p in wrapper.parameters())-sum(p.numel() for p in native.parameters())
    assert parameter_budget(mc,c,4)['added_parameters']==actual*4
    assert wrapper.head_dim==128 and wrapper.heads==16 and wrapper.groups==8


def test_active_wrapper_backward_and_causality():
    native=NativeFixture().eval(); data=native_input()
    c=NSAConfig(block_size=4,window_size=5,selected_blocks=3,compression_hidden=4)
    wrapper=Qwen3NSAAttention(native,c,mode='nsa')
    full=wrapper(**data)[0]
    prefix=dict(hidden_states=data['hidden_states'][:,:7],
                position_embeddings=tuple(x[:,:7] for x in data['position_embeddings']),attention_mask=None)
    torch.testing.assert_close(full[:,:7],wrapper(**prefix)[0],atol=2e-6,rtol=2e-6)
    assert not torch.allclose(full,native(**data)[0])
    full.square().sum().backward()
    for name,p in wrapper.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum()>0, name


@pytest.mark.parametrize('extra',[{'past_key_value':object()},{'past_key_values':object()},
    {'use_cache':True},{'output_attentions':True},{'unknown':False},
    {'cache_position':torch.arange(1,14)},{'position_ids':torch.ones(2,13,dtype=torch.long)}])
def test_active_unsupported_arguments_rejected_before_native_call(extra):
    native=NativeFixture(); wrapper=Qwen3NSAAttention(native,NSAConfig(block_size=4),mode='nsa')
    with pytest.raises(ValueError): wrapper(**native_input(),**extra)
    assert native.calls==0


def model_fixture():
    return SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=NativeFixture()) for _ in range(3)]))


@pytest.mark.parametrize('indices',[[0,0],[0,3],[0,-1],[0,'1'],[]])
def test_transactional_conversion(indices):
    model=model_fixture(); original=[x.self_attn for x in model.model.layers]
    with pytest.raises(ValueError): convert_qwen3(model,NSAConfig(),layers=indices)
    assert [x.self_attn for x in model.model.layers]==original


def test_conversion_invalid_later_geometry_and_restore():
    model=model_fixture(); original=[x.self_attn for x in model.model.layers]
    original[2].config.model_type='other'
    with pytest.raises(ValueError): convert_qwen3(model,NSAConfig(),layers=[0,2])
    assert model.model.layers[0].self_attn is original[0]
    original[2].config.model_type='qwen3'
    receipt=convert_qwen3(model,NSAConfig(),layers=[0,2])
    assert receipt['added_parameters']>0 and not receipt['native_checkpoint_qualified']
    assert model.model.layers[1].self_attn is original[1]
    restore_qwen3(model,layers=[0,2])
    assert [x.self_attn for x in model.model.layers]==original


@pytest.mark.parametrize('form',['none','binary','bool4','additive4','extra_column'])
def test_canonical_masks(form):
    b,t=2,7; valid=torch.ones(b,t,dtype=torch.bool); valid[0,5:]=False
    expected=valid
    allowed=torch.ones(t,t,dtype=torch.bool).tril()[None,None]&valid[:,None,None]
    if form=='none': mask=None; expected=torch.ones_like(valid)
    if form=='binary': mask=valid.long()
    if form=='bool4': mask=allowed
    if form=='additive4': mask=torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,torch.finfo(torch.float32).min)
    if form=='extra_column': mask=torch.cat((allowed,torch.zeros(b,1,t,1,dtype=torch.bool)),-1)
    assert torch.equal(valid_from_mask(mask,b,t,torch.device('cpu')),expected)


@pytest.mark.parametrize('form',['leftpad','bias','noncausal','head_specific','nan'])
def test_noncanonical_masks_rejected(form):
    b,t=2,7
    mask=torch.zeros(b,1,t,t).masked_fill(~torch.ones(t,t,dtype=torch.bool).tril(),-torch.inf)
    if form=='leftpad': mask=torch.tensor([[0,1,1,1,1,1,1]]*2)
    if form=='bias': mask[:,:,4,0]=-0.2
    if form=='noncausal': mask[:,:,0,6]=0
    if form=='head_specific': mask=mask.expand(b,2,t,t)
    if form=='nan': mask[:,:,4,0]=torch.nan
    with pytest.raises(ValueError): valid_from_mask(mask,b,t,torch.device('cpu'))


def test_access_budget_not_speed_or_cache_claim():
    result=access_budget(4096,NSAConfig())
    assert result['three_path_position_sum_upper_bound']==512
    assert not result['resident_kv_evicted']
    with pytest.raises(ValueError): access_budget(0,NSAConfig())


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None,reason='Transformers is not installed; native integration NOT_RUN')
def test_optional_native_transformers_tiny_random_model():
    # A random tiny Qwen3 is only an API mechanical test, never checkpoint efficacy.
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=8,use_cache=False)
    cfg._attn_implementation='eager'
    torch.manual_seed(27)
    model=Qwen3ForCausalLM(cfg).eval()
    tokens=torch.randint(0,64,(1,13))
    baseline=model(tokens,use_cache=False).logits
    convert_qwen3(model,NSAConfig(block_size=4,window_size=8),layers=[0,1])
    torch.testing.assert_close(model(tokens,use_cache=False).logits,baseline,atol=0,rtol=0)
    for layer in model.model.layers: layer.self_attn.set_mode('nsa')
    out=model(tokens,use_cache=False).logits
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:,:7],model(tokens[:,:7],use_cache=False).logits,atol=2e-5,rtol=2e-5)
    out.square().sum().backward()


def test_autograd_ceiling_separate_from_prefill():
    q,k,v,gates,valid,c=setup(257)
    q.requires_grad_()
    with pytest.raises(ValueError,match='autograd'): call(q,k,v,gates,valid,c)
    with torch.no_grad(): assert torch.isfinite(call(q,k,v,gates,valid,c)[0]).all()
    wrapper=Qwen3NSAAttention(NativeFixture(),c,mode='nsa')
    with pytest.raises(ValueError,match='autograd'): wrapper(**native_input(t=257))


def test_hash_control_seed_has_effect():
    data=setup(93,selected_blocks=3,selection_policy='hash_control')
    a=call(*data,return_debug=True)[1]['selected_blocks']
    other=(*data[:-1],replace(data[-1],control_seed=9999))
    b=call(*other,return_debug=True)[1]['selected_blocks']
    assert not torch.equal(a,b)


def test_resource_config_invalid_interaction():
    with pytest.raises(ValueError): NSAConfig(max_reference_tokens=128)
    with pytest.raises(ValueError): NSAConfig(control_seed=2147483648)


def test_zero_head_dimension_is_rejected_cleanly():
    native=NativeFixture(); native.head_dim=0
    with pytest.raises(ValueError): Qwen3NSAAttention(native,NSAConfig())
