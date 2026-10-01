"""Eight-row supervised CE boundary mechanics; no loader, grant or experiment driver.

This prospective core admits explicit synthetic CPU fixtures only. Native calls
fail before touching the model until a separately reviewed SFT scope exists.
The caller supplies prepared unpadded prompt and target-plus-EOS token arrays.
"""
from __future__ import annotations

import math

from . import update

SCHEMA = 'SKILL-ACQUISITION-CE-UPDATE-RECIPE-1'
REDUCTION = 'mean_labeled_tokens_per_row_then_equal_rows'
ROWS = 8
RECIPE_DYNAMIC_FIELDS = frozenset(('boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'))


def require(condition,message):
    if not condition:raise ValueError(message)


def static_recipe_sha256(recipe):
    """Pure canonical snapshot digest excluding exactly the five live fields."""
    require(type(recipe) is dict, 'recipe mapping required for static identity')
    return update.digest({key:value for key,value in recipe.items() if key not in RECIPE_DYNAMIC_FIELDS})


def prepared_row_binding(row):
    """Pure identity of exact unpadded prompt and prepared target-plus-EOS IDs."""
    require(type(row) is dict and type(row.get('id')) is str and row['id'], 'prepared row identity')
    for key in ('prompt_ids', 'completion_ids'):
        require(type(row.get(key)) is list and row[key]
                and all(type(token) is int and token >= 0 for token in row[key]),
                'prepared nonempty finite integer ' + key)
    prepared = {key: row[key] for key in ('id', 'prompt_ids', 'completion_ids')}
    P, C = len(row['prompt_ids']), len(row['completion_ids'])
    return {'id': row['id'], 'prompt_tokens': P, 'labeled_tokens': C, 'full_sequence_tokens': P + C,
            'prompt_ids_sha256': update.digest(row['prompt_ids']),
            'completion_ids_sha256': update.digest(row['completion_ids']),
            'prepared_row_sha256': update.digest(prepared)}


def validate_recipe(recipe):
    require(type(recipe) is dict and recipe.get('schema')==SCHEMA,'explicit CE recipe schema')
    require(type(recipe.get('synthetic_cpu')) is bool,'explicit synthetic CPU flag')
    require(recipe['synthetic_cpu'],'native CE has no separately admitted SFT scope; execution denied before model access')
    require(recipe.get('accumulation_rows')==ROWS and type(recipe['accumulation_rows']) is int
            and recipe.get('microbatch_rows')==1 and type(recipe['microbatch_rows']) is int
            and recipe.get('loss_reduction')==REDUCTION,'fixed eight-row mean CE contract')
    for key in ('boundary_id','generation','U_before'):
        require(type(recipe.get(key)) is int and recipe[key]>=0,'explicit nonnegative '+key)
    require(all(update._hex(recipe.get(key)) for key in ('symbolic_snapshot','start_policy_id')),'actual policy and symbolic fence hashes')
    names=recipe.get('trainable_names')
    require(type(names) is list and names and all(type(name) is str and name for name in names)
            and len(set(names))==len(names),'explicit selected trainable names')
    eos=recipe.get('eos_ids')
    require(type(eos) is list and eos and all(type(i) is int and i>=0 for i in eos)
            and len(set(eos))==len(eos),'explicit EOS inventory')
    require(type(recipe.get('target_eos_id')) is int and recipe['target_eos_id'] in eos
            and type(recipe.get('pad_id')) is int and recipe['pad_id']>=0
            and type(recipe.get('max_target_tokens')) is int and 1<=recipe['max_target_tokens']<=128,'explicit target EOS/PAD/cap')
    gc=recipe.get('gradient_checkpointing')
    require(type(gc) is dict and set(gc)=={'use_reentrant'} and gc['use_reentrant'] is False,
            'explicit non-reentrant decoder/head checkpointing')
    clip=recipe.get('gradient_clipping')
    require(type(clip) is dict and set(clip)=={'max_norm','norm_type','foreach','error_if_nonfinite'}
            and type(clip['max_norm']) in (int,float) and clip['max_norm']==1
            and type(clip['norm_type']) in (int,float) and clip['norm_type']==2
            and clip['foreach'] is False and clip['error_if_nonfinite'] is True,
            'explicit global gradient clip1/norm2/error/foreach contract')
    require(recipe.get('optimizer_origin')=='FRESH_PER_ARM','fresh per-arm Adam required; never inherit warm/pilot state')
    opt=recipe.get('optimizer')
    require(type(opt) is dict and set(opt)=={'class','lr','betas','eps','weight_decay','foreach','capturable'}
            and opt['class']=='torch.optim.AdamW','complete explicit AdamW recipe')
    require(all(type(opt.get(k)) in (int,float) and math.isfinite(opt[k]) for k in ('lr','eps','weight_decay'))
            and opt['lr']>0 and opt['eps']>0 and opt['weight_decay']==0,'finite positive Adam lr/eps and zero decay')
    require(type(opt['betas']) is list and len(opt['betas'])==2 and all(type(v) in (int,float)
            and math.isfinite(v) and 0<=v<1 for v in opt['betas'])
            and opt['foreach'] is False and opt['capturable'] is False,'explicit supported CPU Adam beta/variant')
    return recipe


def validate_rows(rows,recipe):
    require(type(rows) is list and len(rows)==ROWS,'one complete eight-row boundary; no partial batch')
    ids=[]
    for row in rows:
        require(type(row) is dict and type(row.get('id')) is str and row['id'],'prepared row identity')
        ids.append(row['id'])
        for key in ('prompt_ids','completion_ids'):
            require(type(row.get(key)) is list and row[key] and all(type(i) is int and i>=0 for i in row[key]),'prepared unpadded '+key)
        completion=row['completion_ids']
        require(len(completion)<=recipe['max_target_tokens'] and completion[-1]==recipe['target_eos_id']
                and [i for i,t in enumerate(completion) if t in recipe['eos_ids']]==[len(completion)-1],
                'target includes exactly one terminal registered EOS; no truncation/internal EOS')
        require(not any(key in row for key in ('attention_mask','labels','position_ids','padding_side')),
                'core accepts unpadded prepared IDs, not caller masks/padding')
    require(len(set(ids))==ROWS,'duplicate row identity within boundary')
    return rows


def loss_from_logprobs(values):
    """Pure scalar reference: equal row weights after each labeled-token mean."""
    require(type(values) is list and len(values)==ROWS,'eight CE rows')
    require(all(type(row) is list and row and all(type(v) in (int,float) and math.isfinite(v)
            and v<=0 for v in row) for row in values),'finite nonpositive selected token logprobs')
    return math.fsum(-math.fsum(row)/len(row) for row in values)/ROWS


def _finite_inventory(named,optimizer,torch):
    require(set(optimizer.state)<=set(p for p in named.values() if p.requires_grad),'foreign/frozen Adam state')
    for parameter in {id(p):p for p in named.values()}.values():
        flat=parameter.detach().reshape(-1)
        for start in range(0,flat.numel(),update.CHUNK_BYTES//4):
            require(bool(torch.isfinite(flat[start:start+update.CHUNK_BYTES//4]).all()),'nonfinite parameter')
        state=optimizer.state.get(parameter,{})
        if state:
            require(state['step'].dtype==torch.float32 and state['step'].shape==torch.Size([]),
                    'Adam U requires an actual FP32 scalar counter')
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:
                value=state[key]
                require(value.is_contiguous(),'noncontiguous Adam moment')
                for start in range(0,value.numel(),update.CHUNK_BYTES//4):
                    piece=value.reshape(-1)[start:start+update.CHUNK_BYTES//4]
                    require(bool(torch.isfinite(piece).all()),'nonfinite Adam moment')
                    if key=='exp_avg_sq':require(bool((piece>=0).all()),'negative Adam second moment')


def _mark_restore(error,attempted):
    try:error.requires_restore=attempted
    except (AttributeError,TypeError):pass


def _cleanup(model,optimizer,torch):
    """Attempt mode cleanup even when gradient clearing fails; retain both errors."""
    clear_error=None
    try:
        if isinstance(optimizer,torch.optim.Optimizer):optimizer.zero_grad(set_to_none=True)
    except BaseException as error:clear_error=error
    try:model.eval()
    except BaseException as mode_error:
        if clear_error is not None:
            clear_error.mode_cleanup_failure={'type':type(mode_error).__name__,'error':str(mode_error)[:1000]}
            raise clear_error from mode_error
        raise
    if clear_error is not None:raise clear_error


def boundary_update(model,optimizer,rows,recipe,operation):
    """One CE optimizer boundary. Caller commits progress only after this returns.

    Every row's backward returns before the next forward. A physical Adam start
    makes any subsequent exception require restoring/discarding live state.
    Failed attempts remain observable; this core never retries or rewinds costs.
    """
    validate_recipe(recipe)
    validate_rows(rows,recipe)
    recipe_sha256=static_recipe_sha256(recipe)
    require(callable(operation),'physical operation observer required')
    devices={p.device for p in model.parameters()}
    require(len(devices)==1 and next(iter(devices)).type=='cpu','synthetic CPU CE fixtures only')
    device=next(iter(devices))
    import torch
    from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime,s0_generate as generate,s0_pilot as pilot
    from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
    metadata={key:recipe[key] for key in ('boundary_id','generation','symbolic_snapshot','start_policy_id')}
    metadata['static_recipe_sha256']=recipe_sha256
    metadata['objective']='SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'
    optimizer_attempted=False
    primary_error=None
    try:
        state=dict(model.named_parameters(remove_duplicate=False));state.update(dict(model.named_buffers(remove_duplicate=False)))
        state_bytes=sum(v.numel()*v.element_size() for v in {id(v):v for v in state.values()}.values())
        with update._span(operation,'fingerprint',dict(metadata,reason='ce_pre_boundary_identity',state_bytes=state_bytes),device) as pre:
            named,trainable,U=update._actual_optimizer(model,optimizer,recipe,torch)
            require(U==recipe['U_before'],'Adam applied U differs from caller proof')
            require(U>0 or not optimizer.state,'U0 requires genuinely fresh empty Adam; no inherited state')
            require(all(p.grad is None for p in named.values()),'stale gradients at CE boundary')
            _finite_inventory(named,optimizer,torch)
            before=runtime.state_digests(model)
            require(runtime.combine(before)==recipe['start_policy_id'],'CE starting policy identity mismatch')
            frozen={n:h for n,h in before.items() if n not in recipe['trainable_names']}
            require(frozen,'CE fixture must include frozen state')
            frozen_sha=runtime.combine(frozen)
            pre.update(policy_sha256=runtime.combine(before),U=U,frozen_state_sha256=frozen_sha)
        vocab=model.get_output_embeddings().weight.shape[0]
        require(vocab>0 and all(t<vocab for row in rows for t in row['prompt_ids']+row['completion_ids'])
                and all(t<vocab for t in recipe['eos_ids']+[recipe['pad_id']]),'token outside actual vocabulary')
        with update._span(operation,'checkpoint_configuration',metadata,device):
            checkpointing=pilot.enable_checkpointing(model)
        with update._span(operation,'gradient_reset',metadata,device):
            optimizer.zero_grad(set_to_none=True)
            model.train()
        result={'schema':'SKILL-ACQUISITION-CE-BOUNDARY-RESULT-1',**metadata,'synthetic_cpu':True,
                'accumulation_rows':ROWS,'loss_reduction':REDUCTION,'U_before':U,'U_after':U,
                'frozen_state_sha256':frozen_sha,'identity_operation':pre,'rows':[],
                'checkpointing':checkpointing,'optimizer_recipe':dict(recipe['optimizer']),
                'gradient_clipping_recipe':dict(recipe['gradient_clipping']),'updated':False}
        losses=[]
        for row in rows:
            # Freeze the contents used by both the pre-forward observer and the
            # actual graph; caller list mutation cannot change the priced row.
            prompt,completion=list(row['prompt_ids']),list(row['completion_ids'])
            binding=prepared_row_binding({'id':row['id'],'prompt_ids':prompt,'completion_ids':completion})
            P,C=len(prompt),len(completion)
            row_meta=dict(metadata,prompt_id=binding['id'],selected_tokens=C,full_sequence_tokens=P+C,
                          **{key:binding[key] for key in ('prompt_ids_sha256','completion_ids_sha256','prepared_row_sha256')})
            with update._span(operation,'loss_forward',row_meta,device) as forward:
                sequence=torch.tensor([prompt+completion],dtype=torch.long,device=device)
                labels=completion_labels(sequence,torch.ones_like(sequence,dtype=torch.bool),[P],[C],
                                         recipe['eos_ids'],['eos'],recipe['max_target_tokens'])
                require(bool(labels[0,:P].eq(-100).all()) and torch.equal(labels[0,P:],sequence[0,P:])
                        and int((labels!=-100).sum())==C,'prompt/target/EOS mask drift')
                values=generate.teacher_forced_logprobs(model,[prompt],[completion],recipe['pad_id'],
                         padding_side='none',checkpoint_head=True)[0]
                require(values.requires_grad and values.dtype==torch.float32 and values.shape==(C,),
                        'differentiable FP32 target likelihood shape')
                require(bool(torch.isfinite(values).all()) and bool((values<=0).all()),'nonfinite/positive CE likelihood')
                selected=values.detach().tolist()
                loss=-values.mean()/ROWS
                scalar=float(loss.detach())
                expected=-math.fsum(selected)/C/ROWS
                u=2**-24;gamma=C*u/(1-C*u)
                bound=(gamma+2*u)*(-math.fsum(selected)/C/ROWS)+4*math.ulp(max(1,abs(expected)))
                require(math.isfinite(scalar) and abs(scalar-expected)<=bound,'row-token CE reduction drift')
            with update._span(operation,'backward',row_meta,device) as backward:
                loss.backward()
            losses.append(scalar)
            result['rows'].append({**binding,
                'selected_predictor_positions':list(range(P-1,P+C-1)),'target_positions':list(range(P,P+C)),
                'prompt_positions_selected':0,'pad_positions_selected':0,'eos_positions_selected':1,
                'row_mean_nll':-math.fsum(selected)/C,'weighted_loss':scalar,
                'forward_operation':forward,'backward_operation':backward})
            del values,loss,sequence,labels
        with update._span(operation,'gradient_validation',metadata,device) as raw_proof:
            raw_norm=update._gradient_proof(named,torch)
            raw_proof.update(raw_gradient_norm_float64=raw_norm,finite_confined=True)
        with update._span(operation,'gradient_clipping',dict(metadata,**recipe['gradient_clipping']),device) as clipped:
            returned_norm=torch.nn.utils.clip_grad_norm_(trainable,**recipe['gradient_clipping'])
            require(bool(torch.isfinite(returned_norm)),'nonfinite clip norm')
            clipped_norm=update._gradient_proof(named,torch)
            require(clipped_norm<=1.0+32*torch.finfo(torch.float32).eps,'clipped norm exceeds FP32 rounding allowance')
            clipped.update(raw_gradient_norm_float64=raw_norm,clip_returned_norm=float(returned_norm),
                           clipped_gradient_norm_float64=clipped_norm,rounding_allowance=32*torch.finfo(torch.float32).eps)
        optimizer_attempted=True
        with update._span(operation,'optimizer',dict(metadata,U_before=U,expected_U_after=U+1),device) as step:
            optimizer.step()
        with update._span(operation,'fingerprint',dict(metadata,reason='ce_post_boundary_identity',state_bytes=state_bytes),device) as post:
            _,_,U_after=update._actual_optimizer(model,optimizer,recipe,torch)
            require(U_after==U+1,'Adam U did not advance once')
            _finite_inventory(named,optimizer,torch)
            after=runtime.state_digests(model)
            require(runtime.combine({n:h for n,h in after.items() if n not in recipe['trainable_names']})==frozen_sha,
                    'frozen CE parameters/buffers changed')
            post.update(policy_sha256=runtime.combine(after),U=U_after,frozen_state_sha256=frozen_sha)
        result.update(updated=True,U_after=U_after,loss=math.fsum(losses),raw_gradient_norm_float64=raw_norm,
            clipped_gradient_norm_float64=clipped_norm,gradient_validation_operation=raw_proof,
            clipping_operation=clipped,optimizer_operation=step,final_identity_operation=post,
            final_policy_id=runtime.combine(after),requires_restore=False)
        return result
    except BaseException as error:
        primary_error=error
        _mark_restore(error,optimizer_attempted)
        raise
    finally:
        cleanup_entered=False
        cleanup_error_saved=None
        try:
            with update._span(operation,'gradient_cleanup',dict(metadata,optimizer_attempted=optimizer_attempted),device):
                cleanup_entered=True
                _cleanup(model,optimizer,torch)
        except BaseException as cleanup_error:
            cleanup_error_saved=cleanup_error
            _mark_restore(cleanup_error,optimizer_attempted)
            if primary_error is not None:
                primary_error.cleanup_failure={'type':type(cleanup_error).__name__,'error':str(cleanup_error)[:1000]}
                raise primary_error from cleanup_error
            raise
        finally:
            # A rejected observer start precedes cleanup work. Mode restoration
            # is still attempted; its exception retains the discard obligation.
            if not cleanup_entered:
                try:model.eval()
                except BaseException as mode_error:
                    _mark_restore(mode_error,optimizer_attempted)
                    original=primary_error if primary_error is not None else cleanup_error_saved
                    if original is not None:
                        original.mode_cleanup_failure={'type':type(mode_error).__name__,'error':str(mode_error)[:1000]}
                        raise original from mode_error
                    raise
