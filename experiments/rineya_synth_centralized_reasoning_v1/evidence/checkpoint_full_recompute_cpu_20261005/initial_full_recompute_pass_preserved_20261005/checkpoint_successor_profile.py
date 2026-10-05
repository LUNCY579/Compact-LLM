"""Injected eval28 full nonreentrant checkpoint candidate; no launch authority.

No framework import or model construction. Native recipe/dose/resource/numerical
gates are unchanged. The injected early-stop context must be PyTorch's pinned
set_checkpoint_early_stop(False): recomputation finishes normally and is actual
additional work. No-grad uses the exact original forward without checkpointing.
"""
from contextlib import contextmanager
from copy import deepcopy
from time import monotonic
from types import MethodType


def require(value,message):
    if not value: raise ValueError(message)


def install(model,torch,checkpoint,*,early_stop_context,expected_layers=28,
            call=lambda name,fn:fn(),guard=lambda name:None,clock=monotonic):
    require(all(callable(value) for value in (checkpoint,early_stop_context,call,guard,clock)),"explicit checkpoint/full-recompute context/journal/guard/clock")
    require(type(expected_layers) is int and expected_layers==28,"exact28 decoder layers")
    layers=list(model.model.layers);modules=list(model.named_modules())
    require(len(layers)==28 and len({id(layer) for layer in layers})==28 and all(module.training is False for _,module in modules),"all model/decoder modules remain eval")
    require(float(model.config.attention_dropout)==0.,"zero attention dropout")
    for _,module in modules:
        for field in ("dropout","attention_dropout","p"):
            value=getattr(module,field,None)
            if type(value) in (int,float): require(value==0.,"all dropout probabilities must be zero")
    require(all(not getattr(layer,"gradient_checkpointing",False) and not getattr(getattr(layer.forward,"__func__",layer.forward),"_synth_checkpoint_successor_wrapper",False) for layer in layers),"uncheckpointed layer inventory")
    saved=[(layer,"forward" in layer.__dict__,layer.__dict__.get("forward"),layer.forward) for layer in layers]
    flags=[(module,module.training,hasattr(module,"gradient_checkpointing"),getattr(module,"gradient_checkpointing",None)) for _,module in modules]
    config_flags={name:(hasattr(model.config,name),getattr(model.config,name,None)) for name in ("use_cache","attention_dropout")}
    def parameters():
        result={}
        for name,parameter in model.named_parameters(remove_duplicate=False):
            storage=parameter.untyped_storage()
            result[name]=(parameter,parameter.data_ptr(),storage.data_ptr(),storage.nbytes(),tuple(parameter.shape),tuple(parameter.stride()),parameter.dtype,str(parameter.device),parameter.requires_grad)
        return result
    before=parameters()
    stats={str(index):{phase:{key:0 for key in ("attempted","returned","failed","pending","elapsed_seconds")} for phase in ("forward","recompute","no_grad")} for index in range(28)}
    regions={};next_region=0;poisoned=False;removed=False
    counters={"forward_regions":0,"recompute_regions":0,"no_grad_regions":0}
    def invoke(index,phase,identity,function):
        nonlocal poisoned
        require(not poisoned,"failed profile cannot continue")
        item=stats[str(index)][phase];item["attempted"]+=1;item["pending"]+=1
        key=(identity,phase)
        require(key not in regions,"one actual phase per layer region")
        record={"region_id":identity,"layer":index,"phase":phase,"status":"pending","started_monotonic":clock(),"elapsed_seconds":0.0}
        regions[key]=record
        executed=False
        def actual():
            nonlocal executed
            require(not executed,"journal must invoke exactly once")
            executed=True;started=clock()
            try: value=function()
            except BaseException:
                item["pending"]-=1;item["failed"]+=1;record["status"]="failed"
                raise
            else:
                item["pending"]-=1;item["returned"]+=1;record["status"]="returned"
                return value
            finally:
                elapsed=clock()-started;require(elapsed>=0,"monotonic duration")
                item["elapsed_seconds"]+=elapsed;record["elapsed_seconds"]=elapsed
        try:
            guard("checkpoint_layer%d_%s_before"%(index,phase))
            value=call("checkpoint_layer%02d_%s"%(index,phase),actual)
            require(executed and item["pending"]==0,"executed primitive return required")
            guard("checkpoint_layer%d_%s_after"%(index,phase))
            return value
        except BaseException:
            poisoned=True
            raise
    def verify(*,installed=False):
        actual=parameters()
        require(set(actual)==set(before) and all(actual[name]==value for name,value in before.items()),"exact parameter object/storage/shape/stride/dtype/device/flags preserved")
        require(all(module.training is training and hasattr(module,"gradient_checkpointing") is had and getattr(module,"gradient_checkpointing",None)==value for module,training,had,value in flags),"eval/checkpoint flags unchanged")
        require(all((hasattr(model.config,name),getattr(model.config,name,None))==value for name,value in config_flags.items()),"config/cache flags unchanged")
        if installed: require(all(layer.__dict__.get("forward") is wrappers[index] for index,layer in enumerate(layers)),"installed forward attributes changed")
    wrappers=[]
    def restore():
        nonlocal removed
        errors=[]
        for layer,had,value,original in saved:
            try:
                if had: layer.forward=value
                elif "forward" in layer.__dict__: del layer.forward
                require(("forward" in layer.__dict__) is had and (layer.__dict__.get("forward") is value if had else layer.forward.__func__ is original.__func__ and layer.forward.__self__ is original.__self__),"exact original forward restoration")
            except BaseException as error: errors.append(error)
        removed=not errors
        if errors: raise RuntimeError("forward restoration failed") from errors[0]
    for index,(_,_,_,original) in enumerate(saved):
        def wrapped(module,*args,_index=index,_original=original,**kwargs):
            nonlocal next_region,poisoned
            require(not removed and not poisoned,"active unpoisoned candidate required")
            require(kwargs.get("use_cache",False) is False and all(kwargs.get(key) is None for key in ("past_key_value","past_key_values","cache")),"all cache use/mutation forbidden")
            next_region+=1;identity=next_region
            if not torch.is_grad_enabled():
                counters["no_grad_regions"]+=1
                return invoke(_index,"no_grad",identity,lambda:_original(*args,**kwargs))
            mode={"phase":None}
            @contextmanager
            def region(phase):
                require(mode["phase"] is None,"non-overlapping forward/recompute contexts")
                mode["phase"]=phase
                try: yield
                finally: mode["phase"]=None
            def compute(*values,**options):
                phase=mode["phase"];require(phase in ("forward","recompute"),"declared checkpoint context required")
                counters[phase+"_regions"]+=1
                if phase=="recompute": require(regions.get((identity,"forward"),{}).get("status")=="returned","recompute paired with returned original region")
                return invoke(_index,phase,identity,lambda:_original(*values,**options))
            try:
                with early_stop_context(False):
                    return checkpoint(compute,*args,use_reentrant=False,preserve_rng_state=True,
                                      context_fn=lambda:(region("forward"),region("recompute")),**kwargs)
            except BaseException:
                poisoned=True
                raise
        wrapped._synth_checkpoint_successor_wrapper=True
        wrappers.append(MethodType(wrapped,layers[index]))
    try:
        guard("checkpoint_install_before")
        def assign():
            for layer,wrapper in zip(layers,wrappers): layer.forward=wrapper
        call("checkpoint_install_attributes",assign)
        verify(installed=True);guard("checkpoint_install_after")
    except BaseException:
        restore();verify()
        raise
    def remove():
        try:
            guard("checkpoint_remove_before");call("checkpoint_remove_attributes",restore);guard("checkpoint_remove_after")
        finally:
            restore();verify()
    def snapshot():
        return {"layers":deepcopy(stats),"regions":[deepcopy(value) for value in regions.values()],"counters":deepcopy(counters),
                "failed_or_pending":poisoned,"removed":removed,"use_reentrant":False,"preserve_rng_state":True,"checkpoint_early_stop":False,
                "duration_scope":"actual original decoder function calls only; model/controller wall time separate",
                "native_launch_authorized":False,"study_training_authorized":False,"native_profile_qualified":False}
    return {"scope":"CHECKPOINT_SUCCESSOR_CANDIDATE_NATIVE_UNQUALIFIED","decoder_layers":28,"use_reentrant":False,
            "preserve_rng_state":True,"checkpoint_early_stop":False,"counters":counters,"snapshot":snapshot,
            "verify":verify,"remove":remove,"native_launch_authorized":False,"study_training_authorized":False}
