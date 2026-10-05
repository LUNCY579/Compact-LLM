"""Injected nonreentrant decoder checkpoint experiment; CPU/unqualified only.

Preserves evaluation mode, frozen BF16 parameter storage, FP32 compute and all
adapter factors. No framework import, model construction or native permission.
Recomputation is explicit additional work, never an omitted cost or dose change.
"""
from __future__ import annotations
from types import MethodType


def require(value,message):
    if not value:raise ValueError(message)


def install(model,torch,checkpoint,*,expected_layers=28,call=lambda name,fn:fn(),guard=lambda name:None):
    require(callable(checkpoint) and callable(call) and callable(guard),"injected checkpoint/journal/guard")
    require(type(expected_layers) is int and expected_layers==28 and model.training is False,"exact28 layers and preserved eval mode")
    layers=list(model.model.layers)
    require(len(layers)==expected_layers and all(layer.training is False and not getattr(layer,'gradient_checkpointing',False) for layer in layers),"unwrapped eval decoder layer inventory")
    require(float(model.config.attention_dropout)==0.,"zero dropout required")
    before={name:(p,p.data_ptr(),tuple(p.shape),p.dtype,p.requires_grad) for name,p in model.named_parameters(remove_duplicate=False)}
    saved=[(layer,'forward' in layer.__dict__,layer.__dict__.get('forward'),layer.forward) for layer in layers]
    counters={'forward_regions':0,'recompute_regions':0}
    from contextlib import contextmanager
    @contextmanager
    def region(kind):
        counters[kind]+=1;guard('checkpoint_'+kind+'_before')
        try:yield
        finally:guard('checkpoint_'+kind+'_after')
    for index,(layer,had,previous,original) in enumerate(saved):
        def wrapped(module,*args,_original=original,**kwargs):
            require(not kwargs.get('use_cache',False) and kwargs.get('past_key_values') is None,"no cache mutation inside recomputation")
            if not torch.is_grad_enabled():return _original(*args,**kwargs)
            return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,
                context_fn=lambda:(region('forward_regions'),region('recompute_regions')),**kwargs)
        layer.forward=MethodType(wrapped,layer)
    def verify():
        actual=dict(model.named_parameters(remove_duplicate=False))
        require(set(actual)==set(before) and all(actual[n] is p and p.data_ptr()==ptr and tuple(p.shape)==shape
                and p.dtype==dtype and p.requires_grad==grad for n,(p,ptr,shape,dtype,grad) in before.items()),"checkpoint wrapper changed parameter inventory/storage/flags")
        require(model.training is False and all(x.training is False for x in layers),"checkpoint wrapper changed eval mode")
    def remove():
        for layer,had,previous,original in saved:
            if had:layer.forward=previous
            elif 'forward' in layer.__dict__:del layer.forward
        verify()
    verify()
    return {'scope':'CPU_RANDOM_TINY_QWEN_CHECKPOINT_PROFILE_NATIVE_UNQUALIFIED','use_reentrant':False,
            'preserve_rng_state':True,'evaluation_mode_preserved':True,'decoder_layers':28,
            'counters':counters,'verify':verify,'remove':remove,'native_launch_authorized':False}
