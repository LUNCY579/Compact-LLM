"""Injected final-study adapter restore for raw inference; no loader or launch CLI.

Production requires the complete source-bound27/540 three-chunk publication and
cost chain. CPU_RANDOM_TINY fixtures are separately declared and cannot become
real-study authority. Only adapter parameter values are copied. Optimizer and
training RNG are validated as archive components but never restored/reused.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='adapter_inference_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE))
import native_loader as loader
import native_mechanics as mechanics
import native_precision as precision
import study_native_chunk as chunks
import study_schedule as schedule
import study_torch_backend as backend
from inference_component import validate_row

FIXTURE_SCOPE='CPU_RANDOM_TINY_FINAL_ADAPTER_INFERENCE_FIXTURE'
NATIVE_SCOPE='UNQUALIFIED_FINAL_STUDY_ADAPTER_INFERENCE'


def require(ok,message):
    if not ok:raise ValueError(message)


class Calls:
    def __init__(self,call,guard):self.call,self.guard,self.counts=call,guard,{}
    def invoke(self,name,function):
        c=self.counts.setdefault(name,{k:0 for k in ('attempted','returned','failed','pending')});c['attempted']+=1;c['pending']+=1
        self.guard('adapter_inference_before_'+name)
        executed=False
        def actual():
            nonlocal executed
            require(not executed,'one actual primitive per journal boundary');executed=True
            try:value=function()
            except BaseException:c['pending']-=1;c['failed']+=1;raise
            c['pending']-=1;c['returned']+=1;return value
        value=self.call(name,actual);require(executed and c['pending']==0,'journal must execute primitive');self.guard('adapter_inference_after_'+name)
        return value
    def cleanup(self,name,function):
        """Restore caller state once even if the journal/guard has failed.

        Cleanup may only undo this helper's transient local RNG consumption.
        Any rejected journal/guard still fails the restore; actual cleanup
        returns remain visible in its primitive counters.
        """
        c=self.counts.setdefault(name,{k:0 for k in ('attempted','returned','failed','pending')});c['attempted']+=1;c['pending']+=1
        executed=False
        def actual():
            nonlocal executed
            require(not executed,'one actual cleanup primitive');executed=True
            try:value=function()
            except BaseException:c['pending']-=1;c['failed']+=1;raise
            c['pending']-=1;c['returned']+=1;return value
        try:
            self.guard('adapter_inference_before_'+name)
            value=self.call(name,actual);require(executed and c['pending']==0,'journal must execute cleanup')
            self.guard('adapter_inference_after_'+name)
            return value
        finally:
            if not executed:actual()


def verify_final_chain(registration,arm,chain,latest):
    require(arm in ('M1','M2') and type(chain) is list and len(chain)==3,'all three successful final-study chunk descriptors required')
    schedule.validate_registration(registration)
    recipe=chunks.recipe(registration,arm,chunks.NATIVE_SCOPE)
    previous=None
    for index,cp in enumerate(chain,1):
        envelope=chunks._validate_resume(cp,registration,arm,index+1,recipe,cp['terminal_costs'])
        if previous is not None:
            for name,counts in previous['primitive_operations'].items():
                require(name in cp['terminal_costs']['primitive_operations'] and all(cp['terminal_costs']['primitive_operations'][name][k]>=n for k,n in counts.items()),'cumulative primitive costs cannot disappear')
        previous=cp['terminal_costs']
    require(latest==chain[-1]['terminal_costs'],'authoritative final terminal costs/hash binding; no rollback')
    final=chain[-1]['envelope']
    require(final['binding']==chunks.binding(registration,arm,27) and final['binding']['consumed_occurrences']==540,'exact final27/540 dose/order/globalLR')
    return deepcopy(final['backend_descriptor']),deepcopy(final['binding']),deepcopy(recipe)


def _validate_archive(torch,payload,descriptor,binding,shapes,parameter_order,rng_keys):
    require(type(payload) is dict and set(payload)=={'metadata','state','occurrence_order'},'complete study archive schema; no qualification STATE1')
    require(backend.canonical(payload['metadata'])==backend.canonical(descriptor['metadata']),'same verified metadata bytes/binding')
    order=payload['occurrence_order']
    require(type(order) is list and len(order)==540 and all(type(x) is str and x for x in order)
            and backend.digest(order)==binding['consumed_order_sha256'],'exact final540 occurrence order')
    state=payload['state']
    require(type(state) is dict and set(state)=={'schema_version','adapters','optimizer','gradients','rng'}
            and type(state['schema_version']) is int and state['schema_version']==1,'complete saved tensor state')
    require(set(state['adapters'])==set(shapes) and set(state['gradients'])==set(shapes)
            and all(g is None for g in state['gradients'].values()),'all exact adapter factors and empty accumulation')
    for name,shape in shapes.items():
        t=state['adapters'][name]
        require(torch.is_tensor(t) and list(t.shape)==shape and t.dtype==torch.float32 and t.device.type=='cpu'
                and bool(torch.isfinite(t).all()),'finite exact CPU FP32 adapter tensor '+name)
    opt=state['optimizer'];require(type(opt) is dict and set(opt)=={'state','param_groups'} and type(opt['param_groups']) is list and len(opt['param_groups'])==1,'complete archived optimizer schema')
    group=opt['param_groups'][0]
    expected={'lr':0.0,'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'maximize':False,
              'foreach':False,'capturable':False,'differentiable':False,'fused':False,'params':list(range(len(shapes)))}
    require(type(group) is dict and set(group)==set(expected) and mechanics.equal_tree(torch,group,expected),'final globalLR0 fixed AdamW metadata and parameter order')
    require(set(opt['state'])==set(range(len(shapes))) and type(parameter_order) is list and set(parameter_order)==set(shapes) and len(parameter_order)==len(shapes),'all archived moments with bound parameter order')
    for index,name in enumerate(parameter_order):
        item=opt['state'][index];require(set(item)=={'step','exp_avg','exp_avg_sq'},'complete moment fields')
        for key in ('exp_avg','exp_avg_sq'):
            t=item[key];require(torch.is_tensor(t) and list(t.shape)==shapes[name] and t.dtype==torch.float32 and t.device.type=='cpu' and bool(torch.isfinite(t).all()),'finite full FP32 moment '+name)
        require(bool((item['exp_avg_sq']>=0).all()),'nonnegative second moment')
        t=item['step'];require(torch.is_tensor(t) and t.ndim==0 and t.dtype==torch.float32 and t.device.type=='cpu' and float(t)==27.,'all exact global step27 counters')
    rng=state['rng'];require(type(rng) is dict and set(rng)==rng_keys,'complete archived RNG inventory (not reused)')
    try:random.Random().setstate(rng['python'])
    except (ValueError,TypeError,IndexError,OverflowError) as e:raise ValueError('invalid archived Python RNG') from e
    for key in rng_keys-{'python'}:
        # Same encoded-state policy as native_mechanics._validate_state;
        # probing a private generator never applies the archived global RNG.
        t=rng[key];current=torch.get_rng_state() if key=='cpu' else torch.cuda.get_rng_state(0)
        require(torch.is_tensor(t) and t.dtype==torch.uint8 and t.device.type=='cpu'
                and t.ndim==1 and t.shape==current.shape,'exact archived RNG state shape '+key)
        try:torch.Generator(device='cpu' if key=='cpu' else 'cuda:0').set_state(t)
        except (RuntimeError,ValueError,TypeError) as e:raise ValueError('invalid encoded archived RNG state '+key) from e
    return state['adapters']


def restore_final(torch,model,reader,call,guard,*,arm,registration=None,chunk_chain=None,latest_terminal_costs=None,
                  source_hashes=None,fixture=None):
    """Return receipt only; any failed copy poisons the model for collection.

    reader(descriptor) must return immutable bounded archive bytes. Native model
    loading/admission is external and no permit is supplied here. A CPU fixture
    provides descriptor/binding/profile/expected_recipe_sha256/cohort_sha256.
    """
    require(all(callable(x) for x in (reader,call,guard)) and arm in ('M1','M2'),'injected reader/journal/guard and adapted arm')
    ledger=Calls(call,guard);scope=FIXTURE_SCOPE if fixture is not None else NATIVE_SCOPE
    result={'status':'RESTORE_PENDING','scope':scope,'arm':arm,'inference_allowed':False,'native_launch_authorized':False,
            'training_authorized':False,'study_checkpoint_qualification_established':False,'optimizer_restored':False,
            'training_RNG_restored':False,'optimizer_policy':'archive validated but never instantiated/restored',
            'RNG_policy':'caller CPU/CUDA0 RNG snapshot restored in finally around transient fresh q/v initialization; private archive validation only; training RNG never applied',
            'caller_RNG_preserved':False,'confirmation_eligible':False}
    try:
        if fixture is None:
            require(source_hashes is not None,'pinned native base tensor hashes required')
            descriptor,binding,recipe=ledger.invoke('final_study_chain_verify',lambda:verify_final_chain(registration,arm,deepcopy(chunk_chain),deepcopy(latest_terminal_costs)))
            shapes=loader._adapter_shapes();layers=28;linears=197;expected_device='cuda:0';rng_keys={'python','cpu','cuda0'}
            metadata=descriptor['metadata'];require(metadata['scope']==chunks.NATIVE_SCOPE,'real final study archive scope; no CPU fixture laundering')
        else:
            require(registration is None and chunk_chain is None and latest_terminal_costs is None and source_hashes is None
                    and fixture.get('kind')=='CPU_RANDOM_TINY_FINAL27_RECIPE_FIXTURE_ONLY','explicit disjoint CPU fixture profile')
            require(not torch.cuda.is_initialized(),'CPU fixture must not initialize CUDA')
            descriptor,binding=deepcopy(fixture['descriptor']),deepcopy(fixture['binding']);profile=fixture['profile']
            metadata=descriptor['metadata'];require(metadata['scope']=='CPU_RANDOM_TOY','CPU archive must never claim native/study qualification')
            require(metadata['recipe_sha256']==fixture['expected_recipe_sha256'] and metadata['cohort_sha256']==fixture['cohort_sha256'],'fixture recipe/cohort binding')
            require(metadata['backend_source_sha256']==fixture['expected_backend_source_sha256'], 'fixture archive exact backend source binding')
            shapes=profile['expected_adapter_shapes'];layers=profile['expected_layers'];linears=profile['expected_linear_sites'];expected_device='cpu';rng_keys={'python','cpu'}
        require(type(binding) is dict and set(binding)==backend.BINDING_FIELDS and binding['completed_updates']==27
                and binding['consumed_occurrences']==540 and binding['next_update_ordinal']==28 and type(binding['learning_rate']) is float and binding['learning_rate']==0.0,'only complete final27/540 recipe; reject C9/partial')
        require(type(descriptor) is dict and set(descriptor)=={'schema_version','kind','path','bytes','sha256','metadata','metadata_sha256'}
                and descriptor['kind']==backend.KIND and descriptor['metadata_sha256']==backend.digest(metadata),'exact archive descriptor; qualification STATE1 rejected')
        require(metadata['arm']==arm and metadata['seed']==101 and backend.canonical(metadata['binding'])==backend.canonical(binding)
                and all(metadata.get(k) is False for k in ('training_authorized','native_launch_authorized','native_resume_qualified','qualification_STATE1_used')),'arm/seed/final binding and closed archive permissions')
        require(type(descriptor['bytes']) is int and 0<descriptor['bytes']<=backend.MAX_CHECKPOINT_BYTES,'bounded checkpoint32MiB')
        def read():
            value=reader(deepcopy(descriptor));require(type(value) is bytes and len(value)==descriptor['bytes']
                and hashlib.sha256(value).hexdigest()==descriptor['sha256'],'same immutable verified checkpoint bytes before deserialization')
            return value
        raw=ledger.invoke('final_adapter_checkpoint_bytes_read',read)
        payload=ledger.invoke('final_adapter_weights_only_load',lambda:torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
        parameter_order=metadata['adapter_parameter_order']
        adapters=ledger.invoke('final_adapter_archive_validate',lambda:_validate_archive(torch,payload,descriptor,binding,shapes,parameter_order,rng_keys))
        require(model.training is False and not getattr(model,'_synth_adapter_inference_restore_failed',False)
                and not any(n.endswith(('.lora_A','.lora_B')) for n,p in model.named_parameters()),'fresh raw eval base before inference-only factors')
        base=ledger.invoke('inference_frozen_base_inventory',lambda:mechanics._base_inventory(model,torch))
        require(all(str(item[0].device)==expected_device for item in base.values()),'fresh base device inventory')
        copies=None
        if fixture is None:
            ledger.invoke('inference_native_base_verify_before',lambda:loader.verify_frozen_base(torch,model,source_hashes,call,guard))
        else:copies=ledger.invoke('inference_CPU_base_snapshot',lambda:{n:v[0].detach().clone() for n,v in base.items()})
        ledger.invoke('inference_FP32_compute_install',lambda:precision.install_fp32_compute(model,torch,expected_linear_sites=linears))
        def snapshot_caller_rng():
            saved={'cpu':torch.get_rng_state().clone()}
            if 'cuda0' in rng_keys:saved['cuda0']=torch.cuda.get_rng_state(0).clone()
            return saved
        local_rng=ledger.invoke('inference_caller_RNG_snapshot',snapshot_caller_rng)
        def restore_caller_rng():
            torch.set_rng_state(local_rng['cpu'])
            if 'cuda0' in local_rng:torch.cuda.set_rng_state(local_rng['cuda0'],device=0)
            require(torch.equal(torch.get_rng_state(),local_rng['cpu'])
                    and ('cuda0' not in local_rng or torch.equal(torch.cuda.get_rng_state(0),local_rng['cuda0'])),'caller local RNG restored exactly')
            result['caller_RNG_preserved']=True
        try:
            inv=ledger.invoke('inference_fresh_zeroB_qv_install',lambda:precision.install_qv_lora(model,torch,rank=8,alpha=16,expected_layers=layers))
        finally:ledger.cleanup('inference_caller_RNG_restore',restore_caller_rng)
        require({x['name']:x['shape'] for x in inv['inventory']}==shapes and len(inv['inventory'])==(112 if fixture is None else len(shapes)),'same exact FP32 q/v inference inventory')
        require(metadata['adapter_inventory_sha256']==backend.digest(inv['inventory'])
                and parameter_order==[x['name'] for x in inv['inventory']], 'bound archived adapter inventory/order equals fresh inference model')
        zero=ledger.invoke('inference_fresh_zeroB_verify',lambda:precision.zero_lora_status(model,torch));require(zero['all_B_zero'] is True,'fresh zeroB before final-factor copy')
        entries=dict(mechanics._adapters(model,torch))
        for name in parameter_order:
            def copy(name=name):
                with torch.no_grad():entries[name].copy_(adapters[name])
            ledger.invoke('final_adapter_parameter_copy',copy)
        ledger.invoke('final_adapter_bitwise_copy_verify',lambda:require(all(mechanics.bitwise_equal_tensor(torch,adapters[n],entries[n]) for n in shapes),'all restored factor bytes exact'))
        ledger.invoke('inference_base_storage_verify_after',lambda:mechanics._verify_base(model,torch,base))
        if fixture is None:ledger.invoke('inference_native_base_verify_after',lambda:loader.verify_frozen_base(torch,model,source_hashes,call,guard))
        else:ledger.invoke('inference_CPU_base_bytes_verify_after',lambda:require(all(mechanics.bitwise_equal_tensor(torch,copies[n],dict(model.named_parameters(remove_duplicate=False))[n]) for n in copies),'no base parameter mutation'))
        require(all(p.grad is None for n,p in model.named_parameters()),'inference restore must not create gradients')
        result.update({'status':'RESTORED_FINAL_ADAPTER_INFERENCE_UNQUALIFIED','inference_allowed':True,
            'CPU_fixture':fixture is not None,'real_final_study_chain_verified':fixture is None,
            'checkpoint_sha256':descriptor['sha256'],'final_binding':binding,'adapter_tensors_copied':len(shapes),
            'base_unchanged':True,'evaluation_mode_preserved':model.training is False,'no_grad_required_for_all_raw_forwards':True,
            'prompt_boundary':'existing inference_component.validate_row exact five fields; no oracle/labels/teacher'})
    except BaseException as error:
        setattr(model,'_synth_adapter_inference_restore_failed',True)
        result.update({'status':'FAILED_RESTORE_MODEL_INVALID_DO_NOT_COLLECT','error':{'type':type(error).__name__,'message':str(error)},'inference_allowed':False})
    finally:result['operation_costs']=deepcopy(ledger.counts)
    return result


def validate_raw_input(model,row,restore_receipt):
    require(restore_receipt.get('inference_allowed') is True and not getattr(model,'_synth_adapter_inference_restore_failed',False)
            and model.training is False,'only successfully restored eval model may enter raw inference')
    return validate_row(row)
