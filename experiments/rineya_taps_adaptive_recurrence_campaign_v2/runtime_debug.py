"""Source-bound, separately authorized CUDA failure localization only.

Import loads no model. The coordinator owns the only contained launch. Native
execution requires launch blocking before this module/Torch import, and the
actual human allowance file; a proposed template cannot authorize execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

IMPORT_ENV = {name: os.environ.get(name) for name in ('CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG')}
TORCH_ALREADY_IMPORTED = 'torch' in sys.modules

import psutil
import torch

from .binding import ARMR, LAB, REPO, admitted_source, header_only, sha
from .bulk_native import adapt_bulk_case, read_bulk_packet, strict_score
from .control import BlockRecurrence, TAPSConfig
from .gpu_native import encode_prefix, gpu_resources, install_gpu_timing, one_shot_anchors, require_cuda_tensor, resource_guard
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence as ReferenceBlock
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, write_json

PACKAGE = Path(__file__).resolve().parent
GIB = 1024**3
LANE = 'GPU_FP32_RUNTIME_DEBUG'
GPU_HELPER_SHA = '4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f'
BULK_HELPER_SHA = '1ed52ccbb546d826caec2bb48b3c7fa76dd5a136f1a1d7e3fae421b4871c46aa'
FIXED_ALPHA = 1.1473425965417514
CASE_IDS = tuple('BULK-TAPS-001:9463250d48f1:dialogue_memory:'+suffix for suffix in ('21:v0', '20:v1', '13:v1'))
SMOKE_IDS = ('calibration_mathematics_429ca310e753b947', 'calibration_unanswerable_abstention_3006ee0f901c7173')
ALLOWANCE_NAME = 'HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json'


def count_forecast():
    return {'debug_cases':3, 'debug_policies':4, 'debug_physical_evaluations':12,
            'SMOKE_physical_evaluations':20, 'physical_evaluations':32,
            'generation_call_cap':576, 'maximum_shape_forward_attempts':10,
            'maximum_completed_forwards':9, 'maximum_expected_partial':1,
            'cache_rejection_attempts':2, 'model_API_attempt_cap':588,
            'parent_deadline_seconds':300, 'evidence_reserve_seconds':25}


def require_human_allowance(admission, *, scope=PACKAGE, worker_sha=None):
    path = Path(admission.get('human_extension_path', '__MISSING__')).resolve()
    if path != (Path(scope)/ALLOWANCE_NAME).resolve() or not path.is_file():
        raise PermissionError('NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT: actual human allowance missing')
    if sha(path) != admission.get('human_extension_sha256'):
        raise PermissionError('NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT: human allowance hash differs')
    value = json.loads(path.read_text(encoding='utf-8'))
    expected = {'status':'EXPLICIT_ADDITIONAL_ATTEMPT_AUTHORIZED', 'lane':LANE,
                'attempts':1, 'max_native_worker_seconds':300, 'parent_run':'native-bulk-001',
                'worker_sha256':worker_sha or sha(__file__), 'training':False, 'full_matrix_retry':False}
    if value != expected:
        raise PermissionError('NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT: exact human allowance contract differs')
    return path


def validate_admission(admission, binding, source):
    expected = {'status':'READY_FOR_RUNTIME_DEBUG_FP32_DISPATCH', 'lane':LANE,
        'device':'cuda:0', 'dtype':'float32', 'attn_implementation':'eager',
        'training':False, 'native_test':False, 'hard_deadline_seconds':300,
        'evidence_reserve_seconds':25, 'gpu_allocated_ceiling_bytes':5*GIB,
        'gpu_global_free_reserve_bytes':GIB, 'gpu_initial_free_floor_bytes':6*GIB,
        'worker_host_available_floor_bytes':6*GIB, 'worker_rss_ceiling_bytes':6*GIB,
        'parent_host_available_floor_bytes':7*GIB, 'debug_case_ids':list(CASE_IDS),
        'smoke_case_ids':list(SMOKE_IDS), 'max_debug_new_tokens':8,
        'max_smoke_new_tokens':24, 'maximum_prefix_tokens':357,
        'fixed_alpha':FIXED_ALPHA, 'seed':17, 'CUDA_LAUNCH_BLOCKING':'1',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'TF32':False,
        'count_forecast':count_forecast()}
    if any(admission.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('runtime debug admission/count/resource/decoding contract differs')
    if admission.get('source') != str(source) or admission.get('checkpoint_files') != binding['checkpoint_files']:
        raise RuntimeError('runtime debug canonical assets differ')
    if sha(PACKAGE/'gpu_native.py') != GPU_HELPER_SHA or sha(PACKAGE/'bulk_native.py') != BULK_HELPER_SHA:
        raise RuntimeError('runtime debug immutable helper source differs')
    require_human_allowance(admission)


def validate_ids(values, vocab_size, *, max_prefix=357):
    if type(vocab_size) is not int or vocab_size<=0 or not isinstance(values,list) or not 1<=len(values)<=max_prefix:
        raise ValueError('nonempty registered prefix and embedding row count required')
    if any(type(value) is not int or not 0<=value<vocab_size for value in values):
        raise ValueError('token ID must be an integer inside the canonical embedding rows')
    return values


def validate_positions(values, prefix_length):
    if values != [list(range(prefix_length))]:
        raise ValueError('automatic batch1 prefix positions differ')


def metadata(tensor):
    return {'shape':list(tensor.shape), 'dtype':str(tensor.dtype), 'device':str(tensor.device),
            'stride':list(tensor.stride()), 'storage_offset':tensor.storage_offset(),
            'contiguous':tensor.is_contiguous()}


class DebugRecorder:
    """CPU-held context/tracebacks; no CUDA API is invoked by failure logging."""
    def __init__(self, operations, forwards, *, vocab_size, synchronize, guard, enforce_device=True):
        self.operations, self.forwards = operations, forwards
        self.vocab_size, self.raw_synchronize, self.raw_guard = vocab_size, synchronize, guard
        self.enforce_device = enforce_device
        self.context, self.prefix_ids = {}, []
        self.operation_count = self.model_attempts = self.model_completed = 0
        self.generation_attempts = self.maximum_attempts = self.cache_attempts = 0
        self.cuda_faulted = False; self.first_unexpected_failure = None
        self.forward_seconds = []; self.last_operation = None
        self.case_started=None; self.case_max_seconds=None

    def guard(self):
        if self.cuda_faulted: raise RuntimeError('CUDA fault recorded; resource recovery forbidden')
        if self.case_started is not None and time.perf_counter()-self.case_started>self.case_max_seconds:
            raise TimeoutError('unchanged per-case native wall limit exceeded')
        return self.raw_guard()

    @staticmethod
    def cuda_error(error):
        return error is not None and ('cuda' in str(error).lower() or type(error).__name__=='AcceleratorError')

    def synchronize(self):
        # Inherited timing uses finally:synchronize(). An active original-layer
        # CUDA exception must be noticed before that finally issues another API.
        active=sys.exception()
        if self.cuda_faulted or self.cuda_error(active):
            self.cuda_faulted=True
            self.operations({'event':'CUDA_SYNC_SKIPPED_AFTER_FAULT',**self.context,'prefix_ids':list(self.prefix_ids),
                             'completed':False,'reason':'CPU-held fault/active exception; no CUDA API issued'})
            return None
        return self.step('CUDA_SYNCHRONIZE',self.raw_synchronize)

    def step(self, name, action, *, extra=None, expected_marker=None):
        self.operation_count += 1; index=self.operation_count; begin=time.perf_counter()
        base={**self.context, 'operation':name, 'operation_index':index,
              'prefix_ids':list(self.prefix_ids), 'prefix_tokens':len(self.prefix_ids), **(extra or {})}
        self.operations({'event':'OP_STARTED', **base, 'completed':False, 'elapsed_seconds':None})
        self.last_operation=name
        try:
            value=action()
        except Exception as error:
            description=f'{type(error).__name__}: {error}'
            expected=expected_marker is not None and expected_marker in str(error)
            self.cuda_faulted |= self.cuda_error(error)
            failure={'event':'OP_FAILED', **base, 'completed':False, 'elapsed_seconds':time.perf_counter()-begin,
                     'error':description, 'traceback':''.join(traceback.format_exception(error)),
                     'expected_exception':expected, 'cuda_context_usable':False if self.cuda_faulted else None}
            self.operations(failure)
            if not expected and self.first_unexpected_failure is None: self.first_unexpected_failure=failure
            raise
        self.operations({'event':'OP_COMPLETED', **base, 'completed':True, 'elapsed_seconds':time.perf_counter()-begin,
                         **({'resources':value} if name=='RESOURCE_GUARD' and isinstance(value,dict) else {})})
        return value

    def check_tensor(self, ids):
        if ids.dtype != torch.long or tuple(ids.shape)!=(1,len(self.prefix_ids)):
            raise ValueError('batch1 long input IDs/CPU prefix shape differs')
        if self.enforce_device: require_cuda_tensor(ids,'debug prefix')
        validate_ids(self.prefix_ids,self.vocab_size)
        actual=self.step('INPUT_ID_HOST_VERIFY',lambda:ids.detach().cpu().tolist())
        if actual != [self.prefix_ids]: raise ValueError('actual tensor IDs differ from CPU-held prefix')

    def forward(self, model, ids, *, kind='generation', expected_marker=None):
        self.check_tensor(ids)
        self.step('RESOURCE_GUARD',self.guard)
        mask=self.step('ATTENTION_MASK_CREATE',lambda:torch.ones_like(ids))
        self.step('SYNC_BEFORE_FORWARD',self.synchronize)
        if self.model_attempts>=588: raise RuntimeError('debug588 model-API cap exhausted')
        self.model_attempts+=1
        if kind=='maximum': self.maximum_attempts+=1
        else: self.generation_attempts+=1
        begin=time.perf_counter()
        event={**self.context,'model_API_attempt':self.model_attempts,'kind':kind,
               'prefix_ids':list(self.prefix_ids),'prefix_tokens':len(self.prefix_ids),'input_metadata':metadata(ids)}
        self.forwards({'event':'FORWARD_STARTED',**event,'completed':False,'forward_seconds':None})
        try:
            result=self.step('MODEL_FORWARD',lambda:model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1),expected_marker=expected_marker)
            self.step('SYNC_AFTER_FORWARD',self.synchronize)
            if getattr(result,'past_key_values',None) is not None: raise RuntimeError('debug returned cache forbidden')
            logits=self.step('LOGITS_CLONE',lambda:result.logits[0,-1].float().clone())
            if tuple(logits.shape)!=(self.vocab_size,) or logits.dtype!=torch.float32: raise RuntimeError('canonical FP32 output vocabulary shape differs')
            if self.enforce_device: require_cuda_tensor(logits,'debug logits')
            if not self.step('FINITE_LOGITS',lambda:bool(torch.isfinite(logits).all())): raise FloatingPointError('nonfinite debug logits')
            self.step('SYNC_AFTER_LOGIT_CHECK',self.synchronize)
            sample=self.step('RESOURCE_GUARD',self.guard)
        except Exception as error:
            self.forwards({'event':'FORWARD_FAILED',**event,'completed':False,'partial_compute':'UNKNOWN',
                           'forward_seconds':time.perf_counter()-begin,'error':f'{type(error).__name__}: {error}'})
            raise
        elapsed=time.perf_counter()-begin; self.model_completed+=1; self.forward_seconds.append(elapsed)
        self.forwards({'event':'FORWARD_COMPLETED',**event,'completed':True,'forward_seconds':elapsed,'resources':sample})
        return logits


@contextmanager
def layer_journal(model, recorder):
    """Owned metadata hooks return None; original layer values are untouched."""
    handles=[]
    try:
        for index,layer in enumerate(model.model.layers):
            def before(_module,args,kwargs,i=index):
                hidden=args[0] if args else kwargs.get('hidden_states')
                position=kwargs.get('position_ids')
                if position is not None:
                    actual=recorder.step('POSITION_ID_HOST_VERIFY',lambda:position.detach().cpu().tolist(),extra={'layer':i})
                    validate_positions(actual,len(recorder.prefix_ids))
                recorder.step('LAYER_ENTER',lambda:None,extra={'layer':i,'hidden_metadata':metadata(hidden)})
            def after(_module,args,kwargs,output,i=index):
                recorder.step('LAYER_EXIT',lambda:None,extra={'layer':i,'hidden_metadata':metadata(output)})
            handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
            handles.append(layer.register_forward_hook(after,with_kwargs=True))
        yield
    finally:
        for handle in handles: handle.remove()


def generate(model, tokenizer, case, config, wrapper, scorer, flags, recorder, *, expected_ids=None, base=None, logit_metric_fn=None):
    """Exact greedy full-prefix semantics with added diagnostic stage logging."""
    begin=time.perf_counter(); recorder.context={**recorder.context,'case_id':case['case_id'],'phase':'generation'};recorder.prefix_ids=[]
    recorder.case_started=begin;recorder.case_max_seconds=config['max_prompt_seconds']
    cpu=recorder.step('PROMPT_ENCODE_CPU',lambda:encode_prefix(tokenizer,case,torch.device('cpu')))
    host=cpu[0].tolist(); validate_ids(host,recorder.vocab_size)
    if expected_ids is not None and host!=expected_ids: raise RuntimeError('source-rendered prompt IDs differ')
    cap=min(config['max_new_tokens'],case.get('max_new_tokens',config['max_new_tokens']))
    if type(cap) is not int or not 1<=cap<=24 or len(host)+cap-1>357: raise RuntimeError('debug registered generation shape/cap differs')
    recorder.prefix_ids=list(host); device=next(model.parameters()).device
    ids=recorder.step('INPUT_TO_DEVICE',lambda:cpu.to(device))
    wrapper.records.clear(); wrapper.calls=0; wrapper.capture=True
    calls=positions=0; first=None; hidden=[]
    eos=model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos])
    if not eos or any(type(x) is not int or not 0<=x<recorder.vocab_size for x in eos): raise RuntimeError('canonical EOS IDs invalid')
    tokens=[]; text=''; finish='token_cap'
    for index in range(cap):
        recorder.context['full_model_call']=index+1
        logits=recorder.forward(model,ids); calls+=1; positions+=len(recorder.prefix_ids)
        if first is None:
            first=recorder.step('FIRST_LOGITS_CLONE',lambda:logits.clone()); hidden=list(wrapper.records); wrapper.capture=False
        selected=recorder.step('ARGMAX',lambda:logits.argmax())
        token=recorder.step('ARGMAX_ITEM',lambda:int(selected.item()))
        validate_ids([token],recorder.vocab_size); tokens.append(token)
        text=recorder.step('CPU_DECODE',lambda:tokenizer.decode(tokens,skip_special_tokens=True),extra={'generated_ids':list(tokens)})
        checked=recorder.step('CPU_OUTPUT_FLAGS',lambda:flags(text,tokens))
        if token in eos: finish='eos'; break
        if checked['repetition_loop']: finish='repetition_loop'; break
        if checked['script_corruption'] or checked['control_count'] or checked['replacement_count']>=3: finish='script_or_unicode_corruption'; break
        if time.perf_counter()-begin>config['max_prompt_seconds']: finish='timeout'; break
        recorder.step('RESOURCE_GUARD',recorder.guard)
        if index+1<cap:
            next_prefix=recorder.prefix_ids+[token]; validate_ids(next_prefix,recorder.vocab_size)
            addition=recorder.step('TOKEN_TENSOR_CREATE',lambda:torch.tensor([[token]],dtype=ids.dtype,device=ids.device),extra={'next_prefix_ids':next_prefix})
            ids=recorder.step('PREFIX_CAT',lambda:torch.cat((ids,addition),dim=1),extra={'next_prefix_ids':next_prefix})
            recorder.step('SYNC_AFTER_CAT',recorder.synchronize)
            recorder.prefix_ids=next_prefix
    recorder.step('SYNC_END_DECODE',recorder.synchronize)
    row={'case_id':case['case_id'],'response':text,'token_ids':tokens,'termination':finish,'eos_emitted':finish=='eos',
        'truncated':finish=='token_cap','safety_stopped':finish not in ('eos','token_cap'),'physical_run':True,
        'score':scorer(text,case['scorer'])['score'], 'format_score':scorer(text,case['format_scorer'])['score'] if case.get('format_scorer') else None,
        'full_model_calls':calls,'processed_token_positions':positions,'initial_block_applications':calls,
        'extra_layer_applications':calls*wrapper.extra_layer_applications,'wall_seconds':time.perf_counter()-begin,
        'hidden':hidden,'trajectory_min_rms_ratio':wrapper.min_rms_ratio,'trajectory_max_rms_ratio':wrapper.max_rms_ratio,
        'gold_nll':None,'gold_answer':None,'target_input':'NONE','instrumentation':'blocking/metadata/CPU-prefix journals; no production-speed measurement',**flags(text,tokens)}
    if logit_metric_fn is not None:
        row.update(recorder.step('FIRST_LOGIT_METRICS',lambda:logit_metric_fn(first if base is None else base,first,config['top_k'],None)))
        row['readout_comparison_scope']='actual same-case A0 first logits' if base is not None else 'self-reference; value parity evaluated separately'
    recorder.case_started=None;recorder.case_max_seconds=None
    return row,first


def remaining_forecast(elapsed, warm_seconds, *, remaining_calls=96):
    if not warm_seconds or any(not math.isfinite(x) or x<=0 for x in warm_seconds): raise ValueError('fresh completed blocking maximum/SMOKE timing required')
    forecast=remaining_calls*max(warm_seconds)*1.5
    return {'admitted':elapsed+forecast+25<=300,'elapsed_seconds':elapsed,'remaining_calls_cap':remaining_calls,
            'largest_completed_blocking_forward_seconds':max(warm_seconds),'factor':1.5,
            'remaining_forecast_seconds':forecast,'evidence_reserve_seconds':25,'parent_deadline_seconds':300}


def run(source, out, admission_path):
    source,out,admission_path=Path(source).resolve(),Path(out).resolve(),Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents: raise PermissionError('debug outputs/admission must remain campaign-local')
    out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'FAIL','lane':LANE,'native_model_loads':0,'model_load_attempts':0,'training':False,'native_test':False,
             'controller_candidate_revision':0,'started_utc':datetime.now(timezone.utc).isoformat()}
    recorder=None; model=None; parent_started=None
    try:
        admission=json.loads(admission_path.read_text(encoding='utf-8'))
        require_human_allowance(admission)
        if IMPORT_ENV!={'CUDA_LAUNCH_BLOCKING':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'} or TORCH_ALREADY_IMPORTED:
            raise RuntimeError('blocking/workspace must be bound before worker/Torch import')
        parent_started=float(os.environ['RUNTIME_DEBUG_PARENT_STARTED_MONOTONIC'])
        if not math.isfinite(parent_started) or not 0<=time.perf_counter()-parent_started<275: raise RuntimeError('contained parent creation clock missing/expired')
        binding=json.loads((PACKAGE/'runs/binding-001/MODEL_DATA_BINDING.json').read_text())
        validate_admission(admission,binding,source)
        review_path=(PACKAGE/admission['review_receipt']).resolve(strict=True)
        if PACKAGE not in review_path.parents: raise PermissionError('debug review path unscoped')
        review=json.loads(review_path.read_text())
        if review.get('status')!='PASS_RUNTIME_DEBUG_FP32_DISPATCH' or review.get('admission_sha256')!=sha(admission_path): raise RuntimeError('debug independent review/admission differs')
        for relative,expected in admission['dispatch_hashes'].items():
            path=(REPO/relative).resolve(strict=True)
            allowed=PACKAGE in path.parents or relative.startswith(('experiments/rineya_taps_adaptive_recurrence_v1/','experiments/anchor_routed_middle_recurrence_v1/','Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/')) or relative=='Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl'
            if not allowed or sha(path)!=expected: raise RuntimeError('debug admitted source differs:'+relative)
        raw_cases,_=read_bulk_packet(admission); case_map={x['id']:x for x in raw_cases}; debug=[case_map[x] for x in CASE_IDS]
        token_path=Path(admission['bulk_token_binding_path']).resolve(strict=True)
        if PACKAGE not in token_path.parents or sha(token_path)!=admission['bulk_token_binding_sha256']: raise RuntimeError('debug exact token binding differs')
        token=json.loads(token_path.read_text()); token_rows={x['case_id']:x for x in token['rows']}
        if token.get('bank_sha256')!=admission['bulk_bank_sha256']: raise RuntimeError('debug token bank identity differs')
        history,inherited=admitted_source(source)
        header=header_only(source/'model.safetensors')
        if header['embedding_metadata']['shape']!=[151936,1024]: raise RuntimeError('debug source-bound embedding header differs')
        cache=out/'hf_cache'
        os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HF_MODULES_CACHE=str(cache/'modules'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if platform.python_version()!=history['python'] or torch.__version__!=history['torch'] or transformers.__version__!=history['transformers'] or sha(inspect.getfile(Qwen3DecoderLayer))!=history['qwen_implementation_sha256']: raise RuntimeError('debug exact pinned runtime differs')
        if not torch.cuda.is_available() or psutil.virtual_memory().available<6*GIB: raise MemoryError('debug CUDA/host initial floor unavailable')
        torch.cuda.set_device(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        if gpu_resources()['gpu_free_bytes']<6*GIB: raise MemoryError('debug initial GPUfree6GiB floor unavailable')
        sys.path[:0]=[str(ARMR/'src'),str(LAB/'src')]
        from armr.provenance import cases_from, checkpoint_files, state_hash
        from armr.recurrence import FixedBlock
        from armr.hooks import Reapplication
        from armr.metrics import output_flags, logit_metrics
        from armr.reporting import summarize as smoke_gate
        from rineya_cross_scale.evaluation.scoring import score_response as old_score
        config={**history['config'],'device':'cuda:0','dtype':'float32','max_rss_gib':6}
        smoke_all=cases_from(LAB/'data/evaluation'/config['smoke_file']);smoke_map={x['case_id']:x for x in smoke_all};smoke=[smoke_map[x] for x in SMOKE_IDS]
        before_files=checkpoint_files(source)
        with Journal(out/'operations.jsonl') as operations,Journal(out/'forwards.jsonl') as forwards,Journal(out/'trajectories.jsonl') as traces,Journal(out/'outcomes.jsonl') as outcomes:
            def guard():
                if recorder is not None and recorder.cuda_faulted: raise RuntimeError('CUDA fault already recorded; resource recovery forbidden')
                if time.perf_counter()-parent_started>275: raise TimeoutError('debug300s parent budget must retain25s evidence reserve')
                return resource_guard()
            recorder=DebugRecorder(operations.append,forwards.append,vocab_size=151936,synchronize=lambda:torch.cuda.synchronize(0),guard=guard)
            tokenizer=recorder.step('TOKENIZER_LOCAL_LOAD',lambda:AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False,cache_dir=cache/'hub'))
            for raw in debug:
                ids=encode_prefix(tokenizer,adapt_bulk_case(raw),torch.device('cpu'))[0].tolist();validate_ids(ids,151936)
                if ids!=token_rows[raw['id']]['input_token_ids'] or len(ids)>350: raise RuntimeError('debug initial source-bound IDs differ')
            receipt['model_load_attempts']=1
            model=recorder.step('MODEL_LOCAL_LOAD',lambda:AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True,device_map={'':'cuda:0'},cache_dir=cache/'hub').eval().requires_grad_(False))
            receipt['native_model_loads']=1; receipt['after_load_resources']=recorder.step('RESOURCE_GUARD',guard)
            code=Path(inspect.getfile(type(model))).resolve()
            if type(model).__name__!='Qwen3AbstainForCausalLM' or cache not in code.parents or sha(code)!=history['checkpoint_files_before']['modeling_supa_a11.py'] or model.config.abstain_token_id!=51378 or model.config.abstain_logit_bias!=5.156249985098839 or model.config.vocab_size!=151936 or model.get_input_embeddings().weight.shape[0]!=151936: raise RuntimeError('debug exact head/embedding/code differs')
            for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
                require_cuda_tensor(tensor,name)
                if tensor.is_floating_point() and tensor.dtype!=torch.float32: raise RuntimeError('debug floating tensor is notFP32')
            before_state=recorder.step('LOADED_STATE_HASH',lambda:state_hash(model))
            if before_state!=history['state_hash_before']: raise RuntimeError('debug loaded state differs')
            receipt.update(loaded_state_sha256=before_state,loaded_custom_code_path=str(code),loaded_custom_code_sha256=sha(code))
            baseline_bindings=snapshot_bindings(model)
            taps=TAPSConfig();unit=replace(taps,scheduler_mode='unit');fixed=replace(taps,scheduler_mode='fixed',fixed_alpha=FIXED_ALPHA)
            policies=[('A0',unit,0),('A1',unit,1),('A2',fixed,1),('A3',taps,1)]
            baselines={}
            write_json(out/'registration.json',{'lane':LANE,'admission_sha256':sha(admission_path),'worker_sha256':sha(__file__),
                'case_ids':CASE_IDS,'smoke_ids':SMOKE_IDS,'source_hashes':inherited,'count_forecast':count_forecast(),
                'runtime':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'qwen_sha256':sha(inspect.getfile(Qwen3DecoderLayer))},
                'config':config,'debug_config':{**config,'max_new_tokens':8},'taps':asdict(taps),'import_environment':IMPORT_ENV,'source_embedding_header':header['embedding_metadata'],
                'full_matrix_retry':False,'efficacy_analysis':False,'training':False,'gold_nll_status':'NOT_COLLECTED_CAUSAL_GUARD'})
            def make_wrapper(kind,cfg,extra,sample,sink=None):
                if kind=='legacy': return FixedBlock(model,[12,13],applications=2) if extra else Reapplication(model)
                cls=ReferenceBlock if kind=='reference' else BlockRecurrence
                return cls(model,extra_loops=extra,config=cfg,sample_id=sample,sink=sink,guard=lambda:recorder.step('RESOURCE_GUARD',recorder.guard))
            def evaluate(case,arm,cfg,extra,split,kind='new',expected_ids=None):
                recorder.context={'case_id':case['case_id'],'arm':arm,'seed':17,'split':split,'extra_loops':extra,'fixed_alpha':cfg.fixed_alpha if cfg.scheduler_mode=='fixed' else None}
                recorder.prefix_ids=[]
                recorder.step('SEED17',lambda:(torch.manual_seed(17),torch.cuda.manual_seed_all(17)))
                def sink(event): traces.append({**event,**recorder.context,'configured_seed':17})
                wrapper=make_wrapper(kind,cfg,extra,case['case_id'],sink)
                with torch.inference_mode(),wrapper:
                    install_gpu_timing(wrapper,recorder.synchronize)
                    row,first=generate(model,tokenizer,case,config if split=='SMOKE' else {**config,'max_new_tokens':8},wrapper,old_score if split=='SMOKE' else strict_score,output_flags,recorder,expected_ids=expected_ids,
                        base=baselines.get((split,case['case_id'])),logit_metric_fn=logit_metrics)
                    if arm=='A0': baselines[(split,case['case_id'])]=first
                    row.update(**recorder.context);outcomes.append(row)
                    if row['safety_stopped']: raise RuntimeError('debug generation safety stop')
                recorder.step('WRAPPER_RESTORED',lambda:check_restored(model,baseline_bindings))
                return row,first
            def equal(a,b):
                if not recorder.step('VALUE_PARITY',lambda:torch.equal(a,b)): raise RuntimeError('debug native value parity differs')
            with one_shot_anchors(model),layer_journal(model,recorder):
                for case in smoke:
                    for extra in (0,1):
                        a,la=evaluate(case,'qualification_legacy',unit,extra,'SMOKE','legacy'); b,lb=evaluate(case,'qualification_new',unit,extra,'SMOKE')
                        if a['token_ids']!=b['token_ids']: raise RuntimeError('debug noextra/unit greedy parity differs')
                        equal(la,lb)
                    a,la=evaluate(case,'qualification_reference_A3',taps,1,'SMOKE','reference'); b,lb=evaluate(case,'qualification_successor_A3',taps,1,'SMOKE')
                    if a['token_ids']!=b['token_ids']: raise RuntimeError('debug original A3 greedy parity differs')
                    equal(la,lb)
                baseline=None
                for arm,cfg,extra in policies:
                    current=[evaluate(case,arm,cfg,extra,'SMOKE')[0] for case in smoke]
                    if arm=='A0': baseline=current
                    elif not smoke_gate(current,baseline,config)['go_fixed_smoke']: raise RuntimeError('debug representative smoke failed')
                initial=token_rows[CASE_IDS[2]]['input_token_ids'];filler=tokenizer.encode('.',add_special_tokens=False)[0]
                maximum=initial+[filler]*(357-len(initial));validate_ids(maximum,151936)
                def maximum_forward(kind=None,cfg=unit,extra=0,forced=False):
                    recorder.context={'phase':'maximum','arm':kind or 'bare','seed':17,'extra_loops':extra,'fixed_alpha':cfg.fixed_alpha if cfg.scheduler_mode=='fixed' else None}
                    recorder.prefix_ids=list(maximum)
                    ids=recorder.step('MAXIMUM_INPUT_CREATE',lambda:torch.tensor([maximum],dtype=torch.long,device='cuda:0'))
                    if kind is None: return recorder.forward(model,ids,kind='maximum')
                    def maximum_sink(event):
                        traces.append({**event,**recorder.context,'configured_seed':17})
                        if forced: raise RuntimeError('EXPECTED_DEBUG_RESTORATION_EXCEPTION')
                    wrapper=make_wrapper(kind,cfg,extra,'maximum',maximum_sink)
                    with wrapper:
                        install_gpu_timing(wrapper,recorder.synchronize)
                        return recorder.forward(model,ids,kind='maximum',expected_marker='EXPECTED_DEBUG_RESTORATION_EXCEPTION' if forced else None)
                with torch.inference_mode():
                    intact=maximum_forward();equal(intact,maximum_forward('legacy',unit,0));equal(intact,maximum_forward('new',unit,0))
                    legacy=maximum_forward('legacy',unit,1);equal(legacy,maximum_forward('new',unit,1))
                    maximum_forward('new',fixed,1)
                    old=maximum_forward('reference',taps,1);equal(old,maximum_forward('new',taps,1))
                    try: maximum_forward('new',fixed,1,forced=True)
                    except RuntimeError as error:
                        if str(error)!='EXPECTED_DEBUG_RESTORATION_EXCEPTION': raise
                    else: raise RuntimeError('debug forced exception did not occur')
                    check_restored(model,baseline_bindings);equal(intact,maximum_forward())
                    for use_cache,present in ((True,False),(False,True)):
                        recorder.context={'phase':'cache_rejection','use_cache':use_cache,'cache_object_present':present};recorder.prefix_ids=[maximum[0]]
                        ids=recorder.step('CACHE_INPUT_CREATE',lambda:torch.tensor([[maximum[0]]],dtype=torch.long,device='cuda:0'))
                        if recorder.model_attempts>=588: raise RuntimeError('debug588 model-API cap exhausted')
                        recorder.model_attempts+=1;recorder.cache_attempts+=1
                        forwards.append({'event':'CACHE_REJECTION_STARTED',**recorder.context,'model_API_attempt':recorder.model_attempts})
                        kwargs={'use_cache':use_cache,**({'past_key_values':object()} if present else {})}
                        with BlockRecurrence(model) as w:
                            try: recorder.step('CACHE_MODEL_INVOCATION',lambda:model(input_ids=ids,**kwargs),expected_marker='cache')
                            except RuntimeError as error:
                                if 'cache' not in str(error).lower() or w.forward_count!=0: raise
                                forwards.append({'event':'CACHE_REJECTED',**recorder.context,'completed':False,'model_compute_started':False,'partial_compute_positions':0})
                            else: raise RuntimeError('debug cache rejection absent')
                after_qual=recorder.step('QUALIFICATION_STATE_HASH',lambda:state_hash(model))
                if after_qual!=before_state or checkpoint_files(source)!=before_files or any(sha(REPO/relative)!=expected for relative,expected in admission['dispatch_hashes'].items()): raise RuntimeError('debug qualification mutation')
                check_restored(model,baseline_bindings)
                if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise RuntimeError('debug gradients/weight training forbidden')
                write_json(out/'qualification.json',{'status':'PASS_RUNTIME_DEBUG_REPRESENTATIVE_NOT_EFFICACY','maximum_prefix_tokens':357,'maximum_attempts':10,
                    'before_state_sha256':before_state,'after_state_sha256':after_qual,'forced_restoration':True,'cache_rejections':2,
                    'noextra_unit_originalA3_value_and_greedy_parity':True,'source_unchanged':True,'representative_SMOKE_only':True})
                forecast=remaining_forecast(time.perf_counter()-parent_started,recorder.forward_seconds[1:]);write_json(out/'matrix_forecast.json',forecast)
                if not forecast['admitted']: raise TimeoutError('complete registered debug population does not fit300s/25s reserve')
                for raw in debug:
                    for arm,cfg,extra in policies: evaluate(adapt_bulk_case(raw),arm,cfg,extra,'RUNTIME_DEBUG',expected_ids=token_rows[raw['id']]['input_token_ids'])
            check_restored(model,baseline_bindings)
            after_state=recorder.step('FINAL_STATE_HASH',lambda:state_hash(model))
            if after_state!=before_state or checkpoint_files(source)!=before_files or any(sha(REPO/relative)!=expected for relative,expected in admission['dispatch_hashes'].items()): raise RuntimeError('debug final mutation')
            if outcomes.count!=32 or recorder.generation_attempts>576 or recorder.maximum_attempts!=10 or recorder.cache_attempts!=2 or recorder.model_attempts>588 or recorder.model_completed!=recorder.generation_attempts+9: raise RuntimeError('debug complete registered counts differ')
            receipt.update(status='RUNTIME_DEBUG_COMPLETE_NOT_EFFICACY',after_state_sha256=after_state,source_unchanged=True,checkpoint_files_unchanged=True,
                           final_resources=recorder.step('RESOURCE_GUARD',guard),trainable_parameters=0,physical_evaluations=outcomes.count,count_forecast=count_forecast())
    except Exception as error:
        receipt.update(error=f'{type(error).__name__}: {error}',traceback=''.join(traceback.format_exception(error)))
        if 'NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT' in str(error): receipt['status']='NEEDS_EXPLICIT_ADDITIONAL_ATTEMPT'
    finally:
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),parent_elapsed_seconds=time.perf_counter()-parent_started if parent_started is not None else None,
            final_cuda_resources_on_stop=None,resident_state_after_fault=None,
            postfailure_cuda_policy='no CUDA recovery/resource/state queries after fault; CPU traceback/held IDs only')
        if recorder is not None:
            receipt.update(model_API_attempts=recorder.model_attempts,completed_model_forwards=recorder.model_completed,
                generation_attempts=recorder.generation_attempts,maximum_attempts=recorder.maximum_attempts,cache_attempts=recorder.cache_attempts,
                cuda_faulted=recorder.cuda_faulted,first_unexpected_failure=recorder.first_unexpected_failure,last_operation=recorder.last_operation)
        write_json(out/'receipt.json',receipt)
    return 0 if receipt['status']=='RUNTIME_DEBUG_COMPLETE_NOT_EFFICACY' else 2


def snapshot_bindings(model):
    return [(('forward' in layer.__dict__),layer.__dict__.get('forward')) for layer in model.model.layers]


def check_restored(model, before=None):
    if any('_taps_owner' in layer.__dict__ or '_armr_owner' in layer.__dict__ for layer in model.model.layers):
        raise RuntimeError('debug wrapper owner survived restoration')
    if before is not None and snapshot_bindings(model)!=before:
        raise RuntimeError('debug bound-forward ownership/restoration differs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.source,args.out,args.admission))
