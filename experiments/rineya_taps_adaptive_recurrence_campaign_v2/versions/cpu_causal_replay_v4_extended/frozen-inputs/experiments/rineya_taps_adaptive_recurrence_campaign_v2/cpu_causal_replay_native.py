"""Prospective CPU causal canonical replay; no generation, fit or GPU fallback.

The coordinator alone launches this worker after source-bound review. Every
forward receives exactly the original prompt plus preceding canonical tokens.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import psutil
import torch
import torch.nn.functional as F

from .binding import ARMR, LAB, REPO, admitted_source, sha
from .causal_replay_extended_authority import validate_human_scope
from .cpu_h3_native import (CPU_INITIAL_FLOOR, GIB, cpu_resources, require_cpu_tensor, resource_guard,
    forward_logits, one_shot_anchors, diagnostic_rms, install_cpu_conditioning, discrepancy_gate)
from .sensitivity import DiagnosticBlock, moment_energies, finite
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import TAPSConfig
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, write_json, memory_guard

PACKAGE=Path(__file__).resolve().parent
LANE='CPU_CAUSAL_REPLAY_FP32'
PARENT_DEADLINE_SECONDS=4500
INTERNAL_RESERVE_SECONDS=180
BODY_DEADLINE_SECONDS=PARENT_DEADLINE_SECONDS-INTERNAL_RESERVE_SECONDS
CPU_HELPER_SHA='f242676e0ec0f019dd94ac27e5f3cb5eb8f3e311beb64cc0355102492396eaca'
MATH_SHA='5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d'
DESIGN_SHA='d586b68c0e43f8cbccaee0593be9356072a3987b66793ccea6f25abd736c6657'
PACKET_SHA='021ea720f4b644d0e0fff0b975ab460d80e6c02cb401400fd0be58a0ba4cb359'
EOS=(151643,151645)
EXPORT=PACKAGE/'runs/causal-replay-DEV-export-001'
EXACT_QUALIFICATION_DEV='Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def tensor_digest(value):
    require_cpu_tensor(value,'hashed vector');array=value.detach().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def disk_guard(extra_bytes=0):
    if shutil.disk_usage(PACKAGE).free<15*GIB:raise MemoryError('unchanged host disk reserve unavailable')
    used=sum(x.stat().st_size for x in PACKAGE.rglob('*') if x.is_file())
    if used+extra_bytes>1536*1024**2:raise MemoryError('campaign2GiB must retain512MiB evidence reserve')


def verify_source_closure(dispatch_hashes):
    """Pre-model exact dependency admission; no dataset parsing or loaders."""
    for relative,value in dispatch_hashes.items():
        path=(REPO/relative).resolve()
        allowed=(PACKAGE in path.parents or relative.startswith((
            'experiments/rineya_taps_adaptive_recurrence_v1/',
            'experiments/anchor_routed_middle_recurrence_v1/',
            'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/')) or relative==EXACT_QUALIFICATION_DEV)
        if not allowed or not path.is_file() or sha(path)!=value:raise RuntimeError('source closure differs:'+relative)


def count_contract():
    return {'cases':192,'parents':96,'answer_targets':440,'EOS_targets':192,'task_targets':632,
        'qualification_API_attempts':29,'qualification_adjoints':3,'model_API_attempts':1925,
        'full_forward_attempts':1923,'completed_forwards':1922,'expected_partial':1,
        'cache_rejections':2,'state_adjoints':635,'completed_prefix_positions':252696}


def causal_prefix(prompt,answer,index,kind='answer'):
    if not isinstance(prompt,list) or not isinstance(answer,list) or not prompt or not answer:
        raise ValueError('bound nonempty prompt/answer IDs required')
    if kind=='answer':
        if type(index) is not int or not 0<=index<len(answer):raise ValueError('answer position outside bound sequence')
        prefix=prompt+answer[:index]
    elif kind=='EOS':
        if index!=len(answer):raise ValueError('EOS must follow the complete canonical answer')
        prefix=prompt+answer
    else:raise ValueError('unregistered prefix kind')
    if len(prefix)>256 or any(type(x) is not int or not 0<=x<151936 for x in prefix):raise ValueError('causal CPU shape/embedding range differs')
    return prefix


def verify_packet(packet,bank,tokens):
    by={x['case_id']:x for x in tokens};raw={x['id']:x for x in bank}
    if len(bank)!=192 or len(tokens)!=192 or len(packet['rows'])!=192 or len(by)!=192 or len(raw)!=192 or {x['case_id'] for x in packet['rows']}!=set(raw) or set(raw)!=set(by):raise RuntimeError('complete192 DEV ID coverage differs')
    if any(x['split']!='DEVELOPMENT' for x in bank+tokens):raise PermissionError('validation/TEST surface forbidden')
    positions=0;answer_sum=eos_sum=0
    for row in packet['rows']:
        original=by[row['case_id']]['input_token_ids'];answer=row['canonical_answer_token_ids'];source=raw[row['case_id']]
        if row['canonical_answer']!=source['gold'] or by[row['case_id']]['raw_prompt']!=source['prompt'] or row['original_prompt_ids_sha256']!=digest(original):raise RuntimeError('canonical/prompt source binding differs')
        if len(row['answer_targets'])!=len(answer):raise RuntimeError('answer targets omitted/duplicated')
        for index,target in enumerate(row['answer_targets']):
            expected=causal_prefix(original,answer,index)
            if target['answer_position']!=index or target['target_token_id']!=answer[index] or target['prefix_token_ids']!=expected or target['prefix_sha256']!=digest(expected):raise RuntimeError('current/future target entered causal prefix')
            positions+=1;answer_sum+=len(expected)
        expected=causal_prefix(original,answer,len(answer),'EOS')
        if row['EOS']['prefix_token_ids']!=expected or row['EOS']['prefix_sha256']!=digest(expected):raise RuntimeError('EOS causal prefix differs')
        eos_sum+=len(expected)
    if positions!=440 or answer_sum!=57596 or eos_sum!=25048 or len({x['parent_id'] for x in bank})!=96:raise RuntimeError('exact target/parent/position totals differ')
    return by,raw


def objective(logits,target=None,*,kind='answer',eos_ids=EOS):
    finite(logits)
    if kind=='answer':
        if type(target) is not int or not 0<=target<logits.numel():raise ValueError('offline target outside actual head')
        return F.cross_entropy(logits.unsqueeze(0),torch.tensor([target],dtype=torch.long)),None
    if kind!='EOS' or len(set(eos_ids))!=2 or any(type(x) is not int or not 0<=x<logits.numel() for x in eos_ids):raise ValueError('fixed accepted EOS set differs')
    allowed=torch.tensor(sorted(eos_ids),dtype=torch.long)
    mass=torch.logsumexp(logits,0)-torch.logsumexp(logits[allowed],0)
    canonical=F.cross_entropy(logits.unsqueeze(0),torch.tensor([max(eos_ids)],dtype=torch.long))
    return mass,canonical


def rollout(model,ids,target,alpha,*,kind='answer',guard,gradient=True,vector_path=None,eos_ids=EOS,input_feature_sink=None):
    require_cpu_tensor(ids,'causal prefix');records=[]
    with torch.enable_grad(),diagnostic_rms(model,records),DiagnosticBlock(model,alpha=alpha,device='cpu',guard=guard) as block:
        logits=forward_logits(model,ids)
        progress,fluctuation,balance=(float(x[0]) for x in moment_energies(block.d1,block.d2))
        if input_feature_sink is not None:
            if not gradient or alpha!=1. or kind!='answer':raise RuntimeError('predictor freeze requires original unit answer-position0')
            input_feature_sink({'progress':progress,'fluctuation':fluctuation,'balance':balance})
        loss,eot=objective(logits,target,kind=kind,eos_ids=eos_ids);finite(loss)
        g=None;direction=None;vectors=None
        if gradient:
            g,=torch.autograd.grad(loss,block.leaf);finite(g,block.d2)
            direction=float((g.float()*block.d2.float()).sum())
            if vector_path is not None:
                path=Path(vector_path)
                import numpy as np
                if g.numel()>256*1024 or block.d2.shape!=g.shape:raise RuntimeError('qualification vector disk bound exceeded')
                disk_guard(g.numel()*8+65536)
                with path.open('xb') as stream:np.savez(stream,G=g.detach().numpy(),d2=block.d2.detach().numpy())
                vectors={'path':str(path),'file_sha256':sha(path),'shape':list(g.shape),'dtype':'float32',
                    'G_sha256':tensor_digest(g),'d2_sha256':tensor_digest(block.d2),'dot_recomputed':direction}
        result={'alpha':alpha,'loss':float(loss.detach()),'sensitivity':direction,
            'canonical_EOT_CE':float(eot.detach()) if eot is not None else None,
            'progress':progress,'fluctuation':fluctuation,'balance':balance,
            'state_gradient_norm':float(g.norm()) if g is not None else None,'raw_update_norm':float(block.d2.norm()),
            'state_gradient_evaluations':int(gradient),'anchor_counts':list(block.anchors),
            'block_local_calls':block.calls,'RMS':records,'qualification_vectors':vectors}
        guard()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('parameter gradient/training firewall failed')
    return result


class Recorder:
    def __init__(self,attempts):self.attempts=attempts;self.events=[]
    def append(self,event):
        value={**event,'device':'cpu'};self.attempts.append(value);self.events.append(value)
    def triplet(self,model,ids,target,*,kind,guard,label,context=None,vector_path=None,input_feature_sink=None):
        values=[]
        for alpha,gradient in ((1.,True),(1.01,False),(.99,False)):
            start=time.perf_counter();base={'label':label,'kind':kind,'alpha':alpha,'prefix_tokens':ids.shape[1],
                'prefix_ids_sha256':digest(ids[0].tolist()),'state_gradient_requested':gradient,**(context or {})}
            self.append({'event':'REPLAY_FORWARD_STARTED',**base,'completed':False})
            try:value=rollout(model,ids,target,alpha,kind=kind,guard=guard,gradient=gradient,vector_path=vector_path if gradient else None,input_feature_sink=input_feature_sink if gradient else None)
            except Exception as error:
                self.append({'event':'REPLAY_FORWARD_FAILED',**base,'completed':False,'partial_compute':'UNKNOWN',
                    'elapsed_seconds':time.perf_counter()-start,'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()});raise
            self.append({'event':'REPLAY_FORWARD_COMPLETED',**base,'completed':True,'elapsed_seconds':time.perf_counter()-start,
                'state_gradient_evaluations':int(gradient),'initial_block_applications':1,'extra_block_applications':1,'block_width':2,
                'resources':guard(),'scalar_result':value})
            values.append(value)
        nominal,plus,minus=values
        gate=discrepancy_gate(nominal,plus,minus)
        return {'nominal':nominal,'plus':plus,'minus':minus,**gate,'kind':kind,'full_model_calls':3,'state_adjoints':1}


def aggregate_case(answer_results,eos_result,predictor):
    if not answer_results:raise ValueError('all canonical answer targets required')
    n=len(answer_results)
    loss={key:sum(x[key]['loss'] for x in answer_results)/n for key in ('nominal','plus','minus')}
    derivative=sum(x['nominal']['sensitivity'] for x in answer_results)/n
    gate=discrepancy_gate({'sensitivity':derivative},{'loss':loss['plus']},{'loss':loss['minus']})
    return {'answer_target_count':n,'mean_answer_CE':loss,'mean_answer_sensitivity':derivative,
        'sum_answer_CE':{key:value*n for key,value in loss.items()},'sum_answer_sensitivity':derivative*n,
        'mean_level_FD':gate,'input_only_predictor':predictor,'EOS_secondary':eos_result,
        'direction_neutral':abs(derivative)<=.001,
        'mean_sign_numerically_uncertain':(1 if derivative>0 else -1 if derivative<0 else 0)!=(1 if gate['finite_difference']>0 else -1 if gate['finite_difference']<0 else 0),
        'answer_likelihood_scope':'canonical answer prefix likelihood, not terminated-response probability',
        'semantic_efficacy':'NOT_ASSESSED','independent_units':'parent worlds, not targets'}


def forecast(events,elapsed):
    observed=[e for e in events if e.get('event')=='REPLAY_FORWARD_COMPLETED' and e.get('label') in ('qualification_maximum256_answer','qualification_maximum256_EOS')]
    if len(observed)!=6:raise RuntimeError('fresh CPU answer/EOS maximum timings incomplete')
    if any(not math.isfinite(e['elapsed_seconds']) or e['elapsed_seconds']<=0 for e in observed):raise RuntimeError('fresh CPU timing missing/nonfinite')
    g=max(e['elapsed_seconds'] for e in observed if e['state_gradient_evaluations']==1)
    v=max(e['elapsed_seconds'] for e in observed if e['state_gradient_evaluations']==0)
    remaining=632*(g+2*v)*1.5
    return {'admitted':elapsed+remaining+INTERNAL_RESERVE_SECONDS<=PARENT_DEADLINE_SECONDS,'remaining_task_targets':632,'remaining_task_forwards':1896,
        'elapsed_seconds':elapsed,'maximum_whole_forward_adjoint_seconds':g,'maximum_value_forward_seconds':v,
        'remaining_forecast_seconds':remaining,'factor':1.5,'internal_reserve_seconds':INTERNAL_RESERVE_SECONDS,
        'parent_deadline_seconds':PARENT_DEADLINE_SECONDS,'body_deadline_seconds':BODY_DEADLINE_SECONDS,'device':'cpu'}


def ledger(events,case_count):
    counts=Counter(e['event'] for e in events)
    expected={'VALUE_FORWARD_STARTED':18,'VALUE_FORWARD_COMPLETED':17,'VALUE_FORWARD_FAILED':1,
        'REPLAY_FORWARD_STARTED':1905,'REPLAY_FORWARD_COMPLETED':1905,'CACHE_REJECTION_STARTED':2,'CACHE_REJECTED':2}
    if dict(counts)!=expected or case_count!=192:raise RuntimeError('complete192 causal replay attempted/completed ledger differs')
    complete=[e for e in events if e['event'] in ('VALUE_FORWARD_COMPLETED','REPLAY_FORWARD_COMPLETED')]
    adjoints=sum(e.get('state_gradient_evaluations',0) for e in complete);positions=sum(e['prefix_tokens'] for e in complete)
    if adjoints!=635 or positions!=252696:raise RuntimeError('causal replay exact635 adjoints/252696 positions differ')
    failure=next(e for e in events if e['event']=='VALUE_FORWARD_FAILED')
    if 'EXPECTED_CAUSAL_SUFFIX_EXCEPTION' not in failure['error'] or failure['partial_compute']!='UNKNOWN':raise RuntimeError('causal expected partial differs')
    if {(e['use_cache'],e['cache_object_present']) for e in events if e['event']=='CACHE_REJECTION_STARTED'}!={(True,False),(False,True)}:raise RuntimeError('both cache variants required')
    if any(e['model_compute_started'] is not False or e['wrapper_forward_count']!=0 for e in events if e['event']=='CACHE_REJECTED'):raise RuntimeError('cache rejection entered model')
    return {**count_contract(),'partial_compute_positions':None,'actual_completed_prefix_positions':positions}


def run(source,out,admission_path):
    source,out,admission_path=Path(source).resolve(),Path(out).resolve(),Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents:raise PermissionError('causal replay artifacts must be campaign-local')
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();model=None
    receipt={'status':'FAIL','lane':LANE,'native_model_loads':0,'model_load_attempts':0,'training':False,'optimizer_steps':0,
        'generation':False,'generation_pool_access':False,'native_test':False,'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        a=json.loads(admission_path.read_text())
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only contained child environment required')
        started=float(os.environ['CPU_CAUSAL_REPLAY_PARENT_STARTED_PERF_COUNTER'])
        if not math.isfinite(started) or not 0<=time.perf_counter()-started<BODY_DEADLINE_SECONDS:raise RuntimeError('contained parent clock missing/expired')
        expected={'status':'READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH','lane':LANE,'device':'cpu','dtype':'float32','attn_implementation':'eager',
            'hard_deadline_seconds':PARENT_DEADLINE_SECONDS,'finalization_reserve_seconds':INTERNAL_RESERVE_SECONDS,'CPU_threads':4,'maximum_prefix_tokens':256,
            'worker_host_available_floor_bytes':CPU_INITIAL_FLOOR,'worker_rss_ceiling_bytes':6*GIB,'host_reserve_bytes':GIB,
            'parent_host_available_floor_bytes':math.ceil(8.4*GIB),'fd_step':.01,'fd_absolute_tolerance':.001,'fd_relative_tolerance':.03,
            'training':False,'generation_pool_access':False,'native_test':False,'count_contract':count_contract()}
        if any(a.get(key)!=value for key,value in expected.items()) or a.get('source')!=str(source):raise RuntimeError('causal replay admission contract differs')
        if sha(PACKAGE/'cpu_h3_native.py')!=CPU_HELPER_SHA or sha(PACKAGE/'sensitivity.py')!=MATH_SHA or sha(PACKAGE/'CPU_CAUSAL_REPLAY_DESIGN_DRAFT_003.md')!=DESIGN_SHA:raise RuntimeError('frozen helper/math/design source differs')
        review_path=(PACKAGE/a['review_receipt']).resolve(strict=True)
        if PACKAGE not in review_path.parents:raise PermissionError('causal replay review unscoped')
        review=json.loads(review_path.read_text())
        if review['status']!='PASS_CPU_CAUSAL_REPLAY_DISPATCH' or review['admission_sha256']!=sha(admission_path):raise RuntimeError('causal replay review differs')
        verify_source_closure(a['dispatch_hashes'])
        validate_human_scope(a)
        packet_path=PACKAGE/'CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json'
        if sha(packet_path)!=PACKET_SHA:raise RuntimeError('complete causal metadata packet differs')
        packet=json.loads(packet_path.read_text());bank_path=EXPORT/'DEVELOPMENT_BANK.json';tokens_path=EXPORT/'DEVELOPMENT_TOKEN_BINDING.json'
        if sha(bank_path)!=a['development_bank_sha256'] or sha(tokens_path)!=a['development_token_binding_sha256']:raise RuntimeError('DEV-only exports differ')
        bank=json.loads(bank_path.read_text());tokens=json.loads(tokens_path.read_text());by,raw=verify_packet(packet,bank,tokens)
        write_json(out/'population.json',packet['rows'])
        historical,inherited=admitted_source(source);binding_path=PACKAGE/'runs/binding-001/MODEL_DATA_BINDING.json';binding=json.loads(binding_path.read_text())
        if a['checkpoint_files']!=binding['checkpoint_files'] or a['binding_sha256']!=sha(binding_path):raise RuntimeError('canonical binding differs')
        cache=out/'hf_cache';os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HF_MODULES_CACHE=str(cache/'modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if platform.python_version()!=historical['python'] or torch.__version__!=historical['torch'] or transformers.__version__!=historical['transformers'] or sha(inspect.getfile(Qwen3DecoderLayer))!=historical['qwen_implementation_sha256']:raise RuntimeError('exact CPU runtime differs')
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False,cache_dir=cache/'hub')
        for row in packet['rows']:
            if tokenizer.encode(row['canonical_answer'],add_special_tokens=False)!=row['canonical_answer_token_ids']:raise RuntimeError('canonical answer tokenizer differs')
            text=by[row['case_id']]['rendered_prompt']
            if tokenizer(text)['input_ids']!=by[row['case_id']]['input_token_ids']:raise RuntimeError('original rendered prompt IDs differ')
        memory_guard(psutil.virtual_memory().available,CPU_INITIAL_FLOOR,reserve_bytes=0)
        write_json(out/'preflight.json',{'status':'METADATA_READY_NOT_QUALIFIED','resources':cpu_resources(),'device':'cpu','initial_floor_bytes':CPU_INITIAL_FLOOR})
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
        receipt['model_load_attempts']=1
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True,device_map={'':'cpu'},cache_dir=cache/'hub').eval().requires_grad_(False)
        receipt['native_model_loads']=1;receipt['after_load_resources']=resource_guard()
        code=Path(inspect.getfile(type(model))).resolve()
        if type(model).__name__!='Qwen3AbstainForCausalLM' or cache not in code.parents or sha(code)!=historical['checkpoint_files_before']['modeling_supa_a11.py'] or model.config.abstain_token_id!=51378 or model.config.abstain_logit_bias!=5.156249985098839:raise RuntimeError('custom head/code differs')
        for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
            require_cpu_tensor(tensor,name)
            if tensor.is_floating_point() and tensor.dtype!=torch.float32:raise RuntimeError('CPU model floating tensor notFP32')
        sys.path[:0]=[str(ARMR/'src'),str(LAB/'src')]
        from armr.provenance import checkpoint_files,state_hash
        from armr.hooks import Reapplication
        from armr.recurrence import FixedBlock
        before_files=checkpoint_files(source);before_state=state_hash(model)
        if before_state!=historical['state_hash_before']:raise RuntimeError('actual loaded CPU state differs')
        receipt.update(loaded_state_sha256=before_state,loaded_custom_code_path=str(code),loaded_custom_code_sha256=sha(code))
        guard=lambda:resource_guard(started,float(BODY_DEADLINE_SECONDS))
        disk_guard()
        write_json(out/'registration.json',{'lane':LANE,'admission_sha256':sha(admission_path),'worker_sha256':sha(__file__),'design_sha256':DESIGN_SHA,
            'parent_deadline_seconds':PARENT_DEADLINE_SECONDS,'body_deadline_seconds':BODY_DEADLINE_SECONDS,'internal_reserve_seconds':INTERNAL_RESERVE_SECONDS,
            'metadata_packet_sha256':PACKET_SHA,'CPU_helper_sha256':CPU_HELPER_SHA,'math_sha256':MATH_SHA,'count_contract':count_contract(),
            'sources':inherited,'device':'cpu','dtype':'float32','threads':4,'primary':'per-case mean canonical answer CE and derivative',
            'secondary':'accepted EOS mass; canonical EOT CE descriptive','predictor':'input-only first unit forward balance, frozen before gold','training':False,'generation':False,
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
                'Qwen_implementation_sha256':sha(inspect.getfile(Qwen3DecoderLayer)),'CPU_threads':torch.get_num_threads(),'dtype':'float32','attn_implementation':'eager'},
            'target_independent_predictors':True,'parameter_gradients':False,'only_state_leaf_grad':True})
        vectors=out/'qualification_vectors';vectors.mkdir()
        with Journal(out/'attempts.jsonl') as attempts,Journal(out/'targets.jsonl') as targets,Journal(out/'outcomes.jsonl') as outcomes,Journal(out/'features.jsonl') as features,one_shot_anchors(model):
            rec=Recorder(attempts)
            original_pop=json.loads((PACKAGE/'runs/binding-001/PROMPT_AUDIT.json').read_text());representative=next(x for x in original_pop if x['case_id']=='mergedev_grounded_qa_rag_10414013c88bdaac')
            rep_ids=representative['input_token_ids'];rep_target=tokenizer.encode(str(representative['scorer']['value']),add_special_tokens=False)[0]
            filler=tokenizer.encode('.',add_special_tokens=False)[0];maximum=rep_ids+[filler]*(256-len(rep_ids));prefixes=[('representative',rep_ids),('maximum256',maximum)]
            parity=[]
            def value_forward(ids,alpha,extra,diagnostic,label):
                begin=time.perf_counter();base={'label':label,'alpha':alpha,'extra_loops':extra,'diagnostic':diagnostic,'prefix_tokens':ids.shape[1],'prefix_ids_sha256':digest(ids[0].tolist())}
                rec.append({'event':'VALUE_FORWARD_STARTED',**base,'completed':False})
                try:
                    if diagnostic:
                        with torch.enable_grad(),DiagnosticBlock(model,alpha=alpha,extra_loops=extra,device='cpu',guard=guard):value=forward_logits(model,ids).detach().clone()
                    else:
                        wrapper=Reapplication(model) if extra==0 else FixedBlock(model,[12,13],applications=2) if alpha==1. else BlockRecurrence(model,extra_loops=extra,config=TAPSConfig(scheduler_mode='fixed',fixed_alpha=alpha))
                        with torch.inference_mode(),wrapper:
                            if isinstance(wrapper,BlockRecurrence):install_cpu_conditioning(wrapper)
                            value=forward_logits(model,ids)
                except Exception as error:
                    rec.append({'event':'VALUE_FORWARD_FAILED',**base,'completed':False,'elapsed_seconds':time.perf_counter()-begin,'partial_compute':'UNKNOWN','error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()});raise
                rec.append({'event':'VALUE_FORWARD_COMPLETED',**base,'completed':True,'state_gradient_evaluations':0,'elapsed_seconds':time.perf_counter()-begin,'logits_sha256':tensor_digest(value),'resources':guard()})
                return value
            scalar_qual=[];maximum_intact=None
            for label,prefix in prefixes:
                ids=torch.tensor([prefix],dtype=torch.long)
                for alpha,extra in ((1.,0),(.99,1),(1.,1),(1.01,1)):
                    expected=value_forward(ids,alpha,extra,False,label);actual=value_forward(ids,alpha,extra,True,label)
                    equal=torch.equal(expected,actual)
                    parity.append({'label':label,'alpha':alpha,'extra':extra,'exact_equal':equal,'reference_sha256':tensor_digest(expected),'diagnostic_sha256':tensor_digest(actual)})
                    if not equal:raise RuntimeError('exact CPU replay value identity failed')
                    if label=='maximum256' and extra==0:maximum_intact=expected
                scalar_qual.append(rec.triplet(model,ids,rep_target,kind='answer',guard=guard,label='qualification_'+label+'_answer',vector_path=vectors/(label+'_G_d2.npz')))
            methods=[x.forward for x in model.model.layers]
            def forced(_module,_args):raise RuntimeError('EXPECTED_CAUSAL_SUFFIX_EXCEPTION')
            handle=model.model.layers[14].register_forward_pre_hook(forced)
            try:
                try:value_forward(torch.tensor([maximum]),1.,1,True,'forced_exception')
                except RuntimeError as error:
                    if str(error)!='EXPECTED_CAUSAL_SUFFIX_EXCEPTION':raise
                else:raise RuntimeError('forced suffix exception absent')
            finally:handle.remove()
            restored=value_forward(torch.tensor([maximum]),1.,0,False,'restored_after_exception')
            if not torch.equal(restored,maximum_intact) or any(x.forward!=m for x,m in zip(model.model.layers,methods)):raise RuntimeError('normal/forced restoration differs')
            for use_cache,present in ((True,False),(False,True)):
                rec.append({'event':'CACHE_REJECTION_STARTED','use_cache':use_cache,'cache_object_present':present})
                with torch.enable_grad(),DiagnosticBlock(model,device='cpu',guard=guard) as block:
                    try:model(input_ids=torch.tensor([maximum]),attention_mask=torch.ones(1,256,dtype=torch.long),use_cache=use_cache,logits_to_keep=1,**({'past_key_values':object()} if present else {}))
                    except RuntimeError as error:
                        if 'cache' not in str(error):raise
                        rec.append({'event':'CACHE_REJECTED','completed':False,'model_compute_started':False,'wrapper_forward_count':block.forwards})
                    else:raise RuntimeError('cache reject absent')
            eos_qual=rec.triplet(model,torch.tensor([maximum]),None,kind='EOS',guard=guard,label='qualification_maximum256_EOS',vector_path=vectors/'EOS_G_d2.npz')
            if state_hash(model)!=before_state or checkpoint_files(source)!=before_files or any(sha(REPO/x)!=value for x,value in a['dispatch_hashes'].items()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('qualification model/source/gradient mutation')
            write_json(out/'qualification.json',{'status':'PASS_CPU_CAUSAL_REPLAY_MECHANICS','device':'cpu','parity':parity,'scalar_CE_FD':scalar_qual,'EOS_mass_FD':eos_qual,
                'maximum_prefix_tokens':256,'cache_rejections':2,'forced_restoration':True,'before_state_sha256':before_state,'after_state_sha256':state_hash(model),'resources':guard(),
                'parameter_gradients':'NONE','source_unchanged':True,'checkpoint_files_unchanged':True,'qualified_state_adjoints':3,'qualified_vector_pairs':3})
            projected=forecast(rec.events,time.perf_counter()-started);write_json(out/'matrix_forecast.json',projected)
            if not projected['admitted']:raise TimeoutError(f'complete192-case causal replay does not fit{PARENT_DEADLINE_SECONDS}s/reserve')
            for row in packet['rows']:
                prompt=by[row['case_id']]['input_token_ids'];answer=row['canonical_answer_token_ids'];answer_results=[];predictor=None
                for index,target in enumerate(answer):
                    prefix=causal_prefix(prompt,answer,index)
                    context={'case_id':row['case_id'],'parent_id':row['parent_id'],'target_index':index,'preceding_gold_count':index,'teacher_conditioned':index>0,
                             'original_prompt_ids_sha256':digest(prompt),'current_future_target_positions_in_prefix':False}
                    capture=[]
                    def freeze_input(values):
                        frozen={**values,'case_id':row['case_id'],'input_ids_sha256':digest(prompt),'teacher_conditioned':False,
                            'target_independent':True,'preceding_gold_tokens':0,'preceding_gold_count':0,'phase':'INPUT_ONLY','prefix_kind':'INPUT_ONLY',
                            'source_prefix_sha256':digest(prompt),'frozen_before_loss':True,'loss_or_gradient_predictors':False}
                        frozen['freeze_sha256']=digest(frozen);features.append(frozen);capture.append(frozen)
                    result=rec.triplet(model,torch.tensor([prefix]),target,kind='answer',guard=guard,label=row['case_id']+':answer:'+str(index),context=context,
                        input_feature_sink=freeze_input if index==0 else None)
                    if index==0:predictor=capture[0]
                    targets.append({**context,'kind':'answer','target_kind':'answer','answer_position':index,'prefix_tokens':len(prefix),'prefix_sha256':digest(prefix),
                        'target_token_id':target,'result':result,'feature_use':'input-only frozen predictor; later quantities diagnostic-only'})
                    answer_results.append(result)
                prefix=causal_prefix(prompt,answer,len(answer),'EOS')
                eos_result=rec.triplet(model,torch.tensor([prefix]),None,kind='EOS',guard=guard,label=row['case_id']+':EOS',context={'case_id':row['case_id'],'target_index':len(answer),'preceding_gold_count':len(answer),'teacher_conditioned':True,'current_future_target_positions_in_prefix':False})
                targets.append({'case_id':row['case_id'],'parent_id':row['parent_id'],'kind':'EOS','target_kind':'EOS','answer_position':len(answer),
                    'preceding_gold_count':len(answer),'teacher_conditioned':True,'prefix_tokens':len(prefix),'prefix_sha256':digest(prefix),'accepted_EOS_ids':EOS,'result':eos_result,
                    'current_future_target_positions_in_prefix':False})
                outcomes.append({key:row[key] for key in ('case_id','parent_id','category','difficulty_stratum','answerability','source_status')}|
                    {'variant_id':raw[row['case_id']]['variant_id'],'pair_kind':raw[row['case_id']]['pair_kind']}|aggregate_case(answer_results,eos_result,predictor))
            complete=ledger(rec.events,outcomes.count);write_json(out/'compute_ledger.json',complete)
        after=state_hash(model)
        if after!=before_state or checkpoint_files(source)!=before_files or any(sha(REPO/x)!=value for x,value in a['dispatch_hashes'].items()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('final source/state/gradient preservation failed')
        receipt.update(status='CPU_CAUSAL_REPLAY_COMPLETE_SURROGATE_ONLY',before_state_sha256=before_state,after_state_sha256=after,source_unchanged=True,checkpoint_files_unchanged=True,
            parameter_gradients='NONE',trainable_parameters=0,only_state_leaf_grad=True,compute_ledger=complete,cases=192,final_resources=guard(),target_counts={'answer':440,'EOS':192},qualified_vector_pairs=3)
    except Exception as error:receipt.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        receipt.update(wall_seconds=time.perf_counter()-started,ended_utc=datetime.now(timezone.utc).isoformat(),final_resources_on_stop=cpu_resources())
        write_json(out/'receipt.json',receipt)
    return 0 if receipt['status']=='CPU_CAUSAL_REPLAY_COMPLETE_SURROGATE_ONLY' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.source,args.out,args.admission))
