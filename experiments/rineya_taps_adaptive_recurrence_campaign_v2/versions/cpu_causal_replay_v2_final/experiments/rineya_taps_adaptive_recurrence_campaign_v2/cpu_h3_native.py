"""Offline canonical first-token state sensitivity; explicit CPU native contract.

No generation, adaptive alpha, teacher continuation, pool access or weight grad.
Only the coordinator's reviewed contained worker may call this native entrypoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import psutil
import torch

from .binding import ARMR, LAB, REPO, admitted_source, sha
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import TAPSConfig
from .binding import header_only
from experiments.rineya_taps_adaptive_recurrence_v1.safety import memory_guard, runtime_guard
from .sensitivity import DiagnosticBlock, fixed_rollout
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, write_json

PACKAGE = Path(__file__).resolve().parent
GIB = 1024**3
CPU_INITIAL_FLOOR = math.ceil(7.4*GIB)
LANE = "CPU_H3_STATE_SENSITIVITY_FP32"
SENSITIVITY_SHA256 = "5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d"


def require_cpu_tensor(tensor, name='tensor'):
    if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu':
        raise RuntimeError(name+' must reside on the explicit CPU native device')


def cpu_resources():
    info=psutil.Process().memory_info()
    return {'rss_bytes':info.rss,'peak_rss_bytes':getattr(info,'peak_wset',None),
            'host_available_bytes':psutil.virtual_memory().available,'device':'cpu','GPU_memory':None}


def resource_guard(started=None,max_seconds=1800.):
    if started is not None: runtime_guard(started,max_seconds,max_rss_bytes=6*GIB)
    sample=cpu_resources()
    if sample['rss_bytes']>6*GIB or sample['peak_rss_bytes'] is not None and sample['peak_rss_bytes']>6*GIB:
        raise MemoryError('unchanged CPU worker RSS6GiB ceiling exceeded')
    if sample['host_available_bytes']<GIB: raise MemoryError('CPU host1GiB running reserve unavailable')
    return sample


def encode_prefix(tokenizer,case,device='cpu'):
    if torch.device(device).type!='cpu': raise RuntimeError('CPU H3 encoding cannot select another device')
    rendered=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer(rendered,return_tensors='pt')['input_ids']
    require_cpu_tensor(ids,'rendered input IDs')
    return ids


def forward_logits(model,ids):
    require_cpu_tensor(ids,'CPU prefix')
    if ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1 or not 1<=ids.shape[1]<=256:
        raise RuntimeError('CPU H3 requires batch1 long prefix<=256')
    if any(p.device.type!='cpu' for p in model.parameters()): raise RuntimeError('CPU model parameter device differs')
    result=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1)
    if getattr(result,'past_key_values',None) is not None: raise RuntimeError('CPU model returned forbidden cache')
    logits=result.logits[0,-1].float().clone()
    require_cpu_tensor(logits,'CPU custom-head logits')
    if not bool(torch.isfinite(logits).all()): raise FloatingPointError('nonfinite CPU logits')
    return logits


@contextmanager
def one_shot_anchors(model):
    counts=[0,0,0];handles=[]
    def start(_module,_args,_kwargs): counts[:]=[0,0,0]
    def finish(_module,_args,_kwargs,_output):
        if counts!=[1,1,1]: raise RuntimeError('CPU H3 L0-L2 must remain one-shot')
    try:
        handles.append(model.register_forward_pre_hook(start,with_kwargs=True))
        for index in range(3):
            def visit(_module,_args,i=index): counts[i]+=1
            handles.append(model.model.layers[index].register_forward_pre_hook(visit))
        handles.append(model.register_forward_hook(finish,with_kwargs=True))
        yield
    finally:
        for handle in handles:handle.remove()


def install_cpu_conditioning(wrapper):
    """CPU-native mask/position/replay checks; no GPU helper or timing flag."""
    keys=('position_ids','position_embeddings','attention_mask','cache_position')
    conditioning={}
    def capture(_module,args,kwargs):
        hidden=args[0] if args else kwargs['hidden_states']; require_cpu_tensor(hidden,'CPU block hidden')
        if hidden.dtype!=torch.float32:raise RuntimeError('CPU hidden notFP32')
        positions=kwargs.get('position_ids');mask=kwargs.get('attention_mask')
        require_cpu_tensor(positions,'CPU positions');require_cpu_tensor(mask,'CPU causal mask')
        if positions.dtype!=torch.long or not torch.equal(positions,torch.arange(hidden.shape[1]).unsqueeze(0)):
            raise RuntimeError('CPU positions are not full-prefix arange')
        if mask.dtype!=torch.float32 or tuple(mask.shape[-2:])!=(hidden.shape[1],hidden.shape[1]):raise RuntimeError('CPU causal mask differs')
        conditioning.clear();conditioning.update({key:kwargs.get(key) for key in keys})
    wrapper.handles.append(wrapper.model.model.layers[12].register_forward_pre_hook(capture,with_kwargs=True))
    def checked(original):
        def call(*args,**kwargs):
            if not conditioning or any(kwargs.get(key) is not conditioning[key] for key in keys):raise RuntimeError('CPU replay conditioning identity differs')
            return original(*args,**kwargs)
        return call
    wrapper.originals=[checked(original) for original in wrapper.originals]


@contextmanager
def diagnostic_rms(model,records):
    """Observe actual ordinary/controlled states; don't reconstruct a saved normal."""
    measured={};handles=[]
    def before(module,_args,_kwargs):
        owner=module.__dict__.get('_sensitivity_owner')
        if owner is None:raise RuntimeError('CPU diagnostic owner absent')
        original=owner.originals[-1];calls=0
        def normal(*args,**kwargs):
            nonlocal calls
            output=original(*args,**kwargs)
            if calls==0:measured['normal_rms']=float(output.detach().square().mean().sqrt())
            calls+=1
            return output
        owner.originals[-1]=normal
    def after(_module,_args,_kwargs,output):
        require_cpu_tensor(output,'CPU controlled state')
        controlled=float(output.detach().square().mean().sqrt());normal=measured['normal_rms']
        ratio=1. if controlled==0 and normal==0 else controlled/max(normal,1e-12)
        if not math.isfinite(ratio) or not .25<=ratio<=4.:raise FloatingPointError('unchanged CPU ARMR RMS gate exceeded')
        records.append({'normal_rms':normal,'controlled_rms':controlled,'rms_ratio_to_normal':ratio})
    try:
        handles.append(model.model.layers[13].register_forward_pre_hook(before,with_kwargs=True))
        handles.append(model.model.layers[13].register_forward_hook(after,with_kwargs=True))
        yield
    finally:
        for handle in handles:handle.remove()


def complete_forecast(elapsed,events,*,deadline=1800,reserve=180):
    selected=[e for e in events if e.get('event')=='DIAGNOSTIC_FORWARD_COMPLETED' and e.get('label')=='qualification_maximum256']
    if len(selected)!=3 or {(e.get('alpha'),e.get('state_gradient_evaluations')) for e in selected}!={(1.,1),(1.01,0),(.99,0)}:
        raise RuntimeError('fresh CPU maximum256 forward/adjoint timing matrix incomplete')
    if any(not isinstance(e.get('elapsed_seconds'),(int,float)) or not math.isfinite(e['elapsed_seconds']) or e['elapsed_seconds']<=0 for e in selected):raise RuntimeError('fresh CPU timing missing/nonfinite')
    gradient=max(e['elapsed_seconds'] for e in selected if e['state_gradient_evaluations']==1)
    value=max(e['elapsed_seconds'] for e in selected if e['state_gradient_evaluations']==0)
    remaining=7*(gradient+2*value)*1.5
    return {'admitted':elapsed+remaining+reserve<=deadline,'elapsed_seconds':elapsed,'remaining_rows':7,
            'maximum256_whole_forward_and_adjoint_seconds':gradient,'maximum256_value_forward_seconds':value,
            'factor':1.5,'remaining_forecast_seconds':remaining,'finalization_reserve_seconds':reserve,
            'parent_deadline_seconds':deadline,'device':'cpu','GPU_timing_used':False}


def complete_ledger(events,outcome_count):
    counts=Counter(e.get('event') for e in events)
    expected={'VALUE_FORWARD_STARTED':18,'VALUE_FORWARD_COMPLETED':17,'VALUE_FORWARD_FAILED':1,
              'DIAGNOSTIC_FORWARD_STARTED':27,'DIAGNOSTIC_FORWARD_COMPLETED':27,
              'CACHE_REJECTION_STARTED':2,'CACHE_REJECTED':2}
    if dict(counts)!=expected or outcome_count!=16:raise RuntimeError('CPU H3 complete47 API/population ledger differs')
    failures=[e for e in events if e['event']=='VALUE_FORWARD_FAILED']
    if 'EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION' not in failures[0].get('error','') or failures[0].get('partial_compute')!='UNKNOWN':raise RuntimeError('CPU H3 expected partial exception differs')
    cache=[e for e in events if e['event']=='CACHE_REJECTION_STARTED']
    if {(e.get('use_cache'),e.get('cache_object_present')) for e in cache}!={(True,False),(False,True)}:raise RuntimeError('CPU H3 cache attempt variants differ')
    if any(e.get('model_compute_started') is not False or e.get('wrapper_forward_count')!=0 for e in events if e['event']=='CACHE_REJECTED'):raise RuntimeError('CPU H3 cache reject entered model')
    adjoints=sum(e['state_gradient_evaluations'] for e in events if e['event']=='DIAGNOSTIC_FORWARD_COMPLETED')
    if adjoints!=9:raise RuntimeError('CPU H3 state adjoint count differs')
    positions=sum(e['prefix_tokens'] for e in events if e['event'] in ('VALUE_FORWARD_COMPLETED','DIAGNOSTIC_FORWARD_COMPLETED'))
    if positions!=4818:raise RuntimeError('CPU H3 completed4818 prefix positions differ')
    return {'full_forward_attempts':45,'completed_model_forwards':44,'expected_partial':1,
            'cache_rejected_before_compute':2,'model_API_attempts':47,'state_adjoints':9,
            'completed_prefix_positions':positions,
            'partial_prefix_compute_positions':None,'CPU_only':True}


def bind_population(cases, metadata, tokenizer, canonical_answer):
    """Declare every input/exclusion before native outcomes; no target invention."""
    index = {item["case_id"]: item for item in metadata if item["split"] == "DEV"}
    if len(cases) != 16 or set(index) != {case["case_id"] for case in cases}:
        raise RuntimeError("diagnostic population must bind all16 original allowlisted DEV cases")
    result = []
    for case in cases:
        bound = index[case["case_id"]]
        row = {"case_id": case["case_id"], "family": case["family"], "scenario_id": case["scenario_id"],
               "input_token_ids": bound["input_token_ids"], "input_tokens": bound["input_tokens"],
               "prefix_sha256": bound["rendered_prompt_sha256"], "target_token_id": None,
               "target_convention": "inherited exact(single answer)/numeric/JSON first native token only",
               "gold_enters_forward_input": False, "generation_pool_access": False}
        row['input_ids_sha256']=hashlib.sha256(json.dumps(row['input_token_ids'],separators=(',',':')).encode()).hexdigest()
        if row["input_tokens"] > 256:
            row.update(status="EXCLUDED_SHAPE", diagnostic_forward_calls=0)
        else:
            answer = canonical_answer(case)
            if answer is None:
                row.update(status="NOT_APPLICABLE_INHERITED_TARGET", diagnostic_forward_calls=0)
            else:
                targets = tokenizer.encode(answer, add_special_tokens=False)
                if not targets:
                    raise RuntimeError("inherited canonical target encoded empty")
                row.update(status="ELIGIBLE", target_token_id=int(targets[0]), canonical_answer=answer,
                           canonical_target_token_ids=targets, diagnostic_forward_calls=3,
                           diagnostic_backward_calls=1)
                row['canonical_target_sha256']=hashlib.sha256(answer.encode()).hexdigest()
        result.append(row)
    if (sum(row["status"] == "ELIGIBLE" for row in result) != 7
            or sum(row["status"] == "NOT_APPLICABLE_INHERITED_TARGET" for row in result) != 7
            or sum(row["status"] == "EXCLUDED_SHAPE" for row in result) != 2):
        raise RuntimeError("prospective diagnostic eligibility counts differ; stop before loading")
    return result


def discrepancy_gate(nominal, plus, minus, step=.01):
    derivative = nominal["sensitivity"]
    fd = (plus["loss"] - minus["loss"]) / (2 * step)
    if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (derivative, fd)):
        raise FloatingPointError("nonfinite native adjoint/finite difference")
    error = abs(fd - derivative)
    tolerance = 1e-3 + .03 * abs(derivative)
    if error > tolerance:
        raise FloatingPointError(f"native finite-difference discrepancy {error} exceeds registered {tolerance}")
    return {"finite_difference": fd, "absolute_error": error, "registered_tolerance": tolerance,
            "status": "PASS_NATIVE_LOCAL_SURROGATE_CHECK"}


def loss_diagnostic(model, ids, target, *, device, guard, synchronize, sink, label):
    """Three whole forwards, one suffix-state adjoint; target never conditions F."""
    results = []
    for alpha, gradient in ((1., True), (1.01, False), (.99, False)):
        start = time.perf_counter()
        sink({"event": "DIAGNOSTIC_FORWARD_STARTED", "label": label, "alpha": alpha,
              "prefix_tokens": ids.shape[1], "target_enters_forward": False,
              "state_gradient_requested": gradient, "completed": False})
        try:
            if torch.device(device).type!='cpu' or synchronize is not None:raise RuntimeError('CPU H3 cannot use another device/synchronization backend')
            rms=[]
            with diagnostic_rms(model,rms):
                result = fixed_rollout(model, ids, target, alpha=alpha, device='cpu', guard=guard,
                                       synchronize=None, compute_gradient=gradient)
        except Exception as error:
            sink({"event": "DIAGNOSTIC_FORWARD_FAILED", "label": label, "alpha": alpha,
                  "elapsed_seconds": time.perf_counter() - start, "completed": False,
                  "partial_compute": "UNKNOWN", "state_gradient_evaluations": None,
                  "error": f"{type(error).__name__}: {error}"}); raise
        sink({"event": "DIAGNOSTIC_FORWARD_COMPLETED", "label": label, "alpha": alpha,
              "prefix_tokens": ids.shape[1], "completed": True, "elapsed_seconds": time.perf_counter() - start,
              "full_model_calls": 1, "state_gradient_evaluations": int(gradient),
              "initial_block_applications": 1, "extra_block_applications": 1,
              "block_width": 2, "anchor_counts": result["anchor_counts"], "resources": guard(),'actual_block_RMS':rms})
        result['actual_block_RMS']=rms
        results.append(result)
    nominal, plus, minus = results
    return {"nominal": nominal, "plus": plus, "minus": minus,
            **discrepancy_gate(nominal, plus, minus), "full_model_calls": 3,
            "state_gradient_evaluations": 1, "trajectory_reuse": False,
            "gold_enters_forward_input": False, "generation_pool_access": False,
            "interpretation": "canonical first-token CE surrogate; not semantic answer correctness"}


def run(source, out, admission_path):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents:
        raise PermissionError("diagnostic output/admission must be campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    started, model = time.perf_counter(), None
    receipt = {"status": "FAIL", "lane": LANE, "native_model_loads": 0,
               "training": False, "native_test": False, "generation_pool_access": False,
               "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU H3 child must hide GPU devices before import')
        started=float(os.environ['CPU_H3_PARENT_STARTED_PERF_COUNTER'])
        if not math.isfinite(started) or not 0<=time.perf_counter()-started<1620:raise RuntimeError('CPU parent creation clock missing/expired')
        required = {"status": "READY_FOR_CPU_H3_FP32_DISPATCH", "device": "cpu", "dtype": "float32",
                    "attn_implementation": "eager", "maximum_prefix_tokens": 256, "fd_step": .01,
                    "fd_absolute_tolerance": 1e-3, "fd_relative_tolerance": .03, "training": False,
                    "native_test": False, "generation_pool_access": False,
                    "worker_host_available_floor_bytes": CPU_INITIAL_FLOOR,
                    "worker_rss_ceiling_bytes": 6*GIB, "host_reserve_bytes":GIB,
                    "parent_host_available_floor_bytes": math.ceil(8.4*GIB),
                    "hard_deadline_seconds":1800, "CPU_threads":4, "finalization_reserve_seconds":180}
        if any(admission.get(key) != value for key, value in required.items()) or admission.get("source") != str(source):
            raise RuntimeError("diagnostic source/gradient/device/resource/FD admission differs")
        if sha(PACKAGE / "sensitivity.py") != SENSITIVITY_SHA256:
            raise RuntimeError("frozen math apparatus bytes changed")
        review_path = (PACKAGE / admission["review_receipt"]).resolve(strict=True)
        if PACKAGE not in review_path.parents:
            raise PermissionError("diagnostic review must be campaign-local")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "PASS_CPU_H3_FP32_DISPATCH" or review.get("admission_sha256") != sha(admission_path):
            raise RuntimeError("diagnostic review does not bind this admission")
        for relative, expected in admission["dispatch_hashes"].items():
            path = (REPO / relative).resolve(strict=True)
            allowed = PACKAGE in path.parents or relative.endswith(".py") and relative.startswith((
                "experiments/rineya_taps_adaptive_recurrence_v1/", "experiments/anchor_routed_middle_recurrence_v1/src/",
                "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/")) or relative == "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl"
            if not allowed or sha(path) != expected:
                raise RuntimeError("diagnostic reviewed source differs: " + relative)
        historical, inherited_hashes = admitted_source(source)
        binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if admission.get("binding_sha256") != sha(binding_path) or admission.get("checkpoint_files") != binding["checkpoint_files"]:
            raise RuntimeError("diagnostic model/data metadata binding differs")
        header=header_only(source/'model.safetensors')
        if header['embedding_metadata']['shape']!=[151936,1024]:raise RuntimeError('CPU H3 embedding header differs')
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"] or torch.__version__ != historical["torch"]
                or transformers.__version__ != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("diagnostic exact frozen runtime differs")
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import cases_from, checkpoint_files, state_hash
        from armr.metrics import canonical_answer
        from armr.hooks import Reapplication
        from armr.recurrence import FixedBlock
        path = LAB / "data/evaluation/mergedev_fast.jsonl"
        if sha(path) != binding["datasets"]["DEV"]["sha256"]:
            raise RuntimeError("diagnostic allowlisted DEV differs")
        cases = cases_from(path)
        metadata = json.loads((binding_path.parent / "PROMPT_AUDIT.json").read_text(encoding="utf-8"))
        population = bind_population(cases, metadata, tokenizer, canonical_answer)
        if any(not 0<=entry['target_token_id']<151936 for entry in population if entry['status']=='ELIGIBLE'):raise RuntimeError('CPU H3 canonical target outside embedding/readout rows')
        for entry in population:
            case = next(case for case in cases if case["case_id"] == entry["case_id"])
            if encode_prefix(tokenizer, case, torch.device("cpu"))[0].tolist() != entry["input_token_ids"]:
                raise RuntimeError("diagnostic rendered prefix differs from native token binding")
        write_json(out / "population.json", population)
        write_json(out / "registration.json", {"lane": LANE, "admission_sha256": sha(admission_path),
                   "binding_sha256": sha(binding_path), "population_sha256": sha(out / "population.json"),
                   "sources": inherited_hashes, "sensitivity_sha256": SENSITIVITY_SHA256, "CPU_worker_sha256": sha(__file__),
                   "maximum_prefix_tokens": 256, "fd_step": .01, "fd_tolerance": "1e-3+.03*abs(sensitivity)",
                   "training": False, "generation": False, "generation_pool_access": False, "parameter_gradients": False,
                   "state_gradient": "post-L13 controlled clone leaf, full native suffix+custom readout",
                   "eligible_rows": 7, "not_applicable_rows": 7, "excluded_shape_rows": 2,
                   'CPU_threads':historical['config']['threads'],'device':'cpu','dtype':'float32','attn_implementation':'eager',
                   'CPU_initial_floor_bytes':CPU_INITIAL_FLOOR,'RSS_ceiling_bytes':6*GIB,'host_reserve_bytes':GIB,
                   'runtime':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'Qwen_implementation_sha256':sha(inspect.getfile(Qwen3DecoderLayer))},
                   'embedding_header':header['embedding_metadata'],'GPU_qualification_borrowed':False,
                   "forecast": {"forward_attempts": 45, "completed_forwards_expected": 44, "expected_forced_partial": 1,
                                "cache_rejections": 2, "model_API_invocations": 47, "state_gradient_evaluations": 9}})
        memory_guard(psutil.virtual_memory().available, CPU_INITIAL_FLOOR, reserve_bytes=0)
        write_json(out/'preflight.json',{'status':'METADATA_READY_NOT_QUALIFIED','device':'cpu','resources':cpu_resources(),'worker_before_weights_floor_bytes':CPU_INITIAL_FLOOR,'runtime_source_sha256':sha(inspect.getfile(Qwen3DecoderLayer)),'model_loads':0})
        torch.set_num_threads(historical["config"]["threads"])
        torch.use_deterministic_algorithms(True); torch.manual_seed(17)
        if historical["config"]["threads"] != 4: raise RuntimeError("historical CPU thread contract differs")
        receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cpu"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1; receipt["after_load_resources"] = resource_guard()
        code = Path(inspect.getfile(type(model))).resolve()
        if (type(model).__name__ != "Qwen3AbstainForCausalLM" or cache not in code.parents
                or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]
                or model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839):
            raise RuntimeError("diagnostic actual custom head/code differs")
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            require_cpu_tensor(tensor, name)
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError("diagnostic floating model tensor notFP32")
        before_files, before_state = checkpoint_files(source), state_hash(model)
        if before_state != historical["state_hash_before"]:
            raise RuntimeError("diagnostic loaded state differs")
        receipt.update(loaded_state_sha256=before_state, loaded_custom_code_path=str(code), loaded_custom_code_sha256=sha(code))
        eligible = [entry for entry in population if entry["status"] == "ELIGIBLE"]
        representative = eligible[0]
        ids = torch.tensor([representative["input_token_ids"]], dtype=torch.long, device="cpu")
        filler = tokenizer.encode(".", add_special_tokens=False)[0]
        max_ids = torch.cat((ids, torch.full((1, 256 - ids.shape[1]), filler, dtype=torch.long, device="cpu")), dim=1)
        sync = None
        guard = lambda: resource_guard(started, admission['hard_deadline_seconds']-admission['finalization_reserve_seconds'])
        with Journal(out / "attempts.jsonl") as attempts, Journal(out / "outcomes.jsonl") as outcomes, one_shot_anchors(model):
            timing_events=[]
            def record_attempt(event):
                event={**event,'device':'cpu','parameter_gradients':'NONE','CPU_threads':4}
                attempts.append(event);timing_events.append(event)
            def value_forward(prefix, alpha, extra, diagnostic, label):
                begin = time.perf_counter()
                record_attempt({"event": "VALUE_FORWARD_STARTED", "label": label, "prefix_tokens": prefix.shape[1],
                                 "alpha": alpha, "extra_loops": extra, "diagnostic": diagnostic, "completed": False})
                try:
                    if diagnostic:
                        rms=[]
                        with torch.enable_grad(), diagnostic_rms(model,rms), DiagnosticBlock(model, alpha=alpha, extra_loops=extra, device="cpu", guard=guard):
                            value = forward_logits(model, prefix).detach().clone()
                    else:
                        cfg = TAPSConfig(scheduler_mode="fixed", fixed_alpha=alpha)
                        wrapper=Reapplication(model) if extra==0 else FixedBlock(model,[12,13],applications=2) if alpha==1. else BlockRecurrence(model,extra_loops=extra,config=cfg)
                        with torch.inference_mode(), wrapper:
                            if isinstance(wrapper,BlockRecurrence):install_cpu_conditioning(wrapper)
                            value = forward_logits(model,prefix)
                except Exception as error:
                    record_attempt({"event": "VALUE_FORWARD_FAILED", "label": label, "completed": False,
                                     "elapsed_seconds": time.perf_counter() - begin, "partial_compute": "UNKNOWN",
                                     "error": f"{type(error).__name__}: {error}"}); raise
                record_attempt({"event": "VALUE_FORWARD_COMPLETED", "label": label, "completed": True,
                                 "prefix_tokens": prefix.shape[1], "alpha": alpha, "extra_loops": extra,
                                 "full_model_calls": 1, "state_gradient_evaluations": 0,
                                 "elapsed_seconds": time.perf_counter() - begin, "resources": guard(),'device':'cpu',
                                 'actual_block_RMS':rms if diagnostic else None,
                                 'initial_block_applications':1,'extra_block_applications':extra,'block_width':2})
                return value
            maximum_intact = None
            for label, prefix in (("representative", ids), ("maximum256", max_ids)):
                for alpha, extra in ((1., 0), (.99, 1), (1., 1), (1.01, 1)):
                    expected = value_forward(prefix, alpha, extra, False, label)
                    actual = value_forward(prefix, alpha, extra, True, label)
                    if label == "maximum256" and extra == 0:
                        maximum_intact = expected
                    if not torch.equal(expected, actual):
                        raise RuntimeError("native diagnostic value identity failed")
                loss_diagnostic(model, prefix, representative["target_token_id"], device="cpu", guard=guard,
                                synchronize=sync, sink=record_attempt, label="qualification_" + label)
            original_methods = [layer.forward for layer in model.model.layers]
            original = maximum_intact  # reuse already-charged maximum noextra value
            def forced(_module, _args): raise RuntimeError("EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION")
            handle = model.model.layers[14].register_forward_pre_hook(forced)
            try:
                try: value_forward(max_ids, 1., 1, True, "forced_exception")
                except RuntimeError as error:
                    if str(error) != "EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION": raise
                else: raise RuntimeError("diagnostic forced exception not exercised")
            finally: handle.remove()
            restored = value_forward(max_ids, 1., 0, False, "restored_after_exception")
            if not torch.equal(original, restored) or any(layer.forward != method for layer, method in zip(model.model.layers, original_methods)):
                raise RuntimeError("native diagnostic methods/values did not restore")
            for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                begin = time.perf_counter(); record_attempt({"event": "CACHE_REJECTION_STARTED", "use_cache": kwargs["use_cache"], "cache_object_present": "past_key_values" in kwargs})
                with torch.enable_grad(), DiagnosticBlock(model, device="cpu", guard=guard) as block:
                    try: model(input_ids=max_ids, attention_mask=torch.ones_like(max_ids), logits_to_keep=1, **kwargs)
                    except RuntimeError as error:
                        if "cache" not in str(error): raise
                        record_attempt({"event": "CACHE_REJECTED", "completed": False, "model_compute_started": False,
                                         "wrapper_forward_count": block.forwards, "elapsed_seconds": time.perf_counter() - begin})
                    else: raise RuntimeError("native diagnostic cache rejection failed")
            if state_hash(model) != before_state or checkpoint_files(source) != before_files or any(sha(REPO/relative)!=expected for relative,expected in admission['dispatch_hashes'].items()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError("diagnostic qualification mutated model/source or produced parameter gradients")
            write_json(out / "qualification.json", {"status": "PASS_CPU_H3_NATIVE_MECHANICS", 'lane':LANE,'device':'cpu',"maximum_prefix_tokens": 256,
                       "noextra_unit_and_099_101_value_identity": True, "full_suffix_state_adjoint_FD": True,
                       "exception_cache_restoration": True, "parameter_gradients": "NONE", "source_unchanged": True,
                       "before_state_sha256": before_state, "after_state_sha256": before_state,
                       "resources": guard(), "target_convention": "first inherited canonical token; not semantic correctness"})
            forecast=complete_forecast(time.perf_counter()-started,timing_events,deadline=admission['hard_deadline_seconds'],reserve=admission['finalization_reserve_seconds'])
            write_json(out/'matrix_forecast.json',forecast)
            if not forecast['admitted']:raise TimeoutError('complete CPU seven-row H3 matrix does not fit remaining deadline/reserve')
            for entry in population:
                if entry["status"] != "ELIGIBLE":
                    outcomes.append({**entry, "physical_run": False, "measured_loss": None,
                                     "state_gradient_evaluations": 0, "forward_calls": 0}); continue
                prefix = torch.tensor([entry["input_token_ids"]], dtype=torch.long, device="cpu")
                row = loss_diagnostic(model, prefix, entry["target_token_id"], device="cpu", guard=guard,
                                      synchronize=sync, sink=record_attempt, label=entry["case_id"])
                outcomes.append({**entry, **row, "physical_run": True, "status": "NATIVE_LOCAL_SURROGATE_MEASURED"})
            ledger=complete_ledger(timing_events,outcomes.count)
            write_json(out/'compute_ledger.json',ledger)
        after_state=state_hash(model)
        if after_state != before_state or checkpoint_files(source) != before_files or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"] or any(sha(REPO/relative)!=expected for relative,expected in admission['dispatch_hashes'].items()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("diagnostic final model/source/gradient preservation failed")
        receipt.update(status="CPU_H3_COMPLETE_NOT_SEMANTIC_EFFICACY", source_unchanged=True,
                       checkpoint_files_unchanged=True, parameter_gradients="NONE", optimizer_steps=0,
                       measured_rows=7, not_applicable=7, excluded_shape=2, final_resources=guard(),compute_ledger=ledger,
                       before_state_sha256=before_state,after_state_sha256=after_state)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt['traceback']=__import__('traceback').format_exc()
    finally:
        receipt.update(wall_seconds=time.perf_counter() - started, ended_utc=datetime.now(timezone.utc).isoformat(),
                       final_resources_on_stop=cpu_resources())
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "CPU_H3_COMPLETE_NOT_SEMANTIC_EFFICACY" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission))
