"""Separate exact BF16 CPU load-only fit experiment; no GPU qualification."""
from __future__ import annotations
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import native, owned_runtime_identity_probe as identity
from .operations import OperationsStore, canonical, host_state_path, TERMINAL
from experiments.opallo_native_collector_v1.supervisor import WindowsJob
from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint

PACKAGE, ROOT = native.PACKAGE, native.ROOT
CONTRACT = PACKAGE / "contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json"
KEY, SCOPE = "CRS-CPU-LOAD-FIT-001", "EXACT_BF16_CPU_LOAD_ONLY_FIT"
AUTHORITY_SHA = "e66d181f01a2346617b5433623859e71b518eefe45c896e117d097c2ddeeeb86"
CORRECTION_SHA = "af6f1d6506899a328845709fac2e9ee42f484effe00cf0c17f9e1d6ec1806229"
GIB, CHUNK, FATAL_EXIT = 2**30, 2*1024*1024, os._exit
require = native.require

class Basic(c.Structure):
    _fields_=[("PerProcessUserTimeLimit",c.c_longlong),("PerJobUserTimeLimit",c.c_longlong),
              ("LimitFlags",w.DWORD),("MinimumWorkingSetSize",c.c_size_t),("MaximumWorkingSetSize",c.c_size_t),
              ("ActiveProcessLimit",w.DWORD),("Affinity",c.c_size_t),("PriorityClass",w.DWORD),("SchedulingClass",w.DWORD)]
class IO(c.Structure):
    _fields_=[(name,c.c_ulonglong) for name in ("ReadOperationCount","WriteOperationCount","OtherOperationCount",
             "ReadTransferCount","WriteTransferCount","OtherTransferCount")]
class Extended(c.Structure):
    _fields_=[("BasicLimitInformation",Basic),("IoInfo",IO),("ProcessMemoryLimit",c.c_size_t),
              ("JobMemoryLimit",c.c_size_t),("PeakProcessMemoryUsed",c.c_size_t),("PeakJobMemoryUsed",c.c_size_t)]

def validate_job_limits(value):
    require(value["flags"] == 0x2200 and value["job_commit_limit_bytes"] == 2*GIB,
            "exact CPU kill-on-close/no-breakaway/2GiB Job commit flags required")

def query_limits(kernel,handle):
    kernel.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p]
    kernel.QueryInformationJobObject.restype=w.BOOL
    limits=Extended()
    require(kernel.QueryInformationJobObject(handle,9,c.byref(limits),c.sizeof(limits),None),"CPU explicit Job limit query failed")
    value={"flags":int(limits.BasicLimitInformation.LimitFlags),"job_commit_limit_bytes":int(limits.JobMemoryLimit),
           "peak_job_committed_bytes":int(limits.PeakJobMemoryUsed),"RSS_is_distinct_from_commit":True}
    validate_job_limits(value)
    return value

class CPUJob(WindowsJob):
    def __init__(self,process):
        super().__init__(process)
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2200;limits.JobMemoryLimit=2*GIB
        try:
            require(self.kernel.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits)),"CPU Job memory limit setup failed")
            self.limit_witness=query_limits(self.kernel,self.handle)
        except BaseException:self.close();raise

def cpu_query_witness(handle,pid,created):
    import psutil
    kernel=c.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)];kernel.IsProcessInJob.restype=w.BOOL
    kernel.GetHandleInformation.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];kernel.GetHandleInformation.restype=w.BOOL
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    try:
        require(pid==os.getpid() and psutil.Process().create_time()==created,"CPU query target identity drift")
        flags=w.DWORD();member=w.BOOL()
        require(kernel.GetHandleInformation(handle,c.byref(flags)) and not flags.value&1,"CPU query handle inheritable")
        require(kernel.IsProcessInJob(kernel.GetCurrentProcess(),handle,c.byref(member)) and member.value,"CPU worker outside exact Job")
        result={**query_limits(kernel,handle),"pid":pid,"creation_time":created,"exact_owned_job_member":True}
    finally:require(kernel.CloseHandle(handle),"CPU query handle did not close")
    result["query_handle_closed_before_framework"]=True
    return result

def checked_contract(expected):
    require(native.sha(CONTRACT)==expected,"CPU contract drift")
    cfg=native.read_json(CONTRACT)
    require(cfg["scope"]==SCOPE and cfg["job_key"]==KEY and cfg["device"]=="cpu" and cfg["dtype"]=="bfloat16","wrong CPU experiment")
    require(cfg["limits"]=={"wall_seconds":120,"runs":1,"automatic_retry":False,"tree_rss_gib":2.0,
            "job_commit_gib":2.0,"host_reserve_gib":1.25,"preworker_host_gib":3.3125,"disk_free_gib":4.0,
            "freshness_seconds":5,"threads":2},"CPU limits changed")
    require(cfg["dose"]=={"meta_constructions":1,"CPU_allocations":1,"checkpoint_loads":1,"serialized_tensors":310,
            "unique_parameters":596049920,"forwards":0,"tokenizers":0,"optimizer_calls":0},"CPU load dose changed")
    require(native.sha(PACKAGE/cfg["authority_path"])==AUTHORITY_SHA,"actual CPU delegated scope drift")
    require(cfg["source_round2_authority_sha256"]==CORRECTION_SHA
            and native.sha(PACKAGE/"authorization/DELEGATED_CPU_NATIVE_SOURCE_CORRECTION_002.json")==CORRECTION_SHA,
            "finite CPU source round2 authority drift")
    native.verify_sources(cfg)
    require(hashlib.sha256(canonical(identity.registry(cfg))).hexdigest()==cfg["registry_sha256"],"original four executable registry drift")
    return cfg

def cpu_telemetry(output):
    import psutil
    return {"sampled_at":time.time(),"device":"cpu","host_available_gib":psutil.virtual_memory().available/GIB,
            "disk_free_gib":psutil.disk_usage(str(output)).free/GIB,"scope":SCOPE}

def admission_reasons(value,*,now=None,rss=None):
    now=time.time() if now is None else now
    reasons=[]
    if not isinstance(value,dict) or value.get("device")!="cpu" or value.get("scope")!=SCOPE:return ["wrong/missing CPU resource scope"]
    for name in ("sampled_at","host_available_gib","disk_free_gib"):
        if type(value.get(name)) not in (int,float) or not math.isfinite(value[name]):return ["invalid CPU telemetry"]
    if not 0<=now-value["sampled_at"]<=5:reasons.append("stale/future CPU telemetry")
    if rss is not None and (not math.isfinite(rss) or rss>2):reasons.append("owned-tree RSS limit")
    floor=3.3125 if rss is None else 1.25+max(0,2-rss)
    if value["host_available_gib"]<floor:reasons.append("CPU host floor")
    if value["disk_free_gib"]<4:reasons.append("CPU disk floor")
    return reasons

def observe_guard(job,cfg,output,stage,budget):
    value=identity.inventory(job,cfg,budget)
    identity.append(output/"OWNED_IMAGES.jsonl",{"stage":stage,"inventory":value})
    require(all(not m.get("query_error") and not m.get("close_error") and m.get("exact_owned_job_member")
                and m.get("hash_matches") for m in value["members"]),"CPU owned executable/membership refusal; raw inventory retained")
    require(value["observed_rss_bytes"]<=2*GIB,"CPU tree RSS ceiling")
    return value

def owned_gpu_observation(job):
    raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True,timeout=3)
    pids=[]
    for line in raw.splitlines():
        require(line.strip().isdigit(),"required owned GPU PID attribution unavailable")
        pids.append(int(line.strip()))
    owned=set(native.job_process_ids(job))
    require(not owned.intersection(pids),"owned CPU Job observed on GPU")
    return {"sampled_at":time.time(),"owned_pids":sorted(owned),"GPU_pids":pids,"owned_GPU_participation":False,
            "unrelated_GPU_resource_applicability":False,"hidden_internal_C_activity":"UNMEASURED"}

def forbidden_call(module,name,event):
    if module.startswith("torch.cuda") or (module.startswith("torch") and name.startswith("_cuda_")):return True
    if module.startswith("torch.backends.cuda") or module.startswith("torch.backends.cudnn"):return True
    if module.startswith("torch.serialization") and name in {"load","save","_load","_save"}:return True
    if module.startswith("transformers.tokenization") and name in {"__init__","from_pretrained"}:return True
    if module.startswith(("torch.nn","transformers.models")) and name in {"forward","_call_impl","generate"}:return True
    if module.startswith(("torch.optim","torch.autograd")) and name in {"step","backward","grad"}:return True
    return False

def allowed_cuda_import_call(module,name,event,*,state,filename,code_flags,callback,context):
    if context["import_phase"] is not True or filename not in context["definition_source_files"]:return False
    if event=="call" and not code_flags&2:
        return module.startswith("torch.cuda") and (state is False or (name=="<module>" and module=="torch.cuda" and state is None))
    if module!="torch.cuda" or state is not False or filename!=context["cuda_main_source"]:return False
    if name in {"_is_compiled","is_initialized"}:return True
    return name=="_lazy_call" and callback in {"torch.cuda._check_capability","torch.cuda._check_cubins","torch.cuda._register_triton_kernels"}

def arm_boundary(output,cfg):
    fatal=FATAL_EXIT
    context={"import_phase":True,"definition_source_files":set(cfg["CUDA_definition_source_sha256"]),
             "cuda_main_source":cfg["CUDA_main_source"]}
    def profile(frame,event,arg):
        if event in {"call","return"}:module,name=frame.f_globals.get("__name__",""),frame.f_code.co_name
        elif event=="c_call":module,name=getattr(arg,"__module__","") or "",getattr(arg,"__name__","")
        else:return
        cuda_module=sys.modules.get("torch.cuda");state=vars(cuda_module).get("_initialized") if cuda_module is not None else None
        callback=frame.f_locals.get("callable")
        callback_name=(getattr(callback,"__module__","")+"."+getattr(callback,"__name__","")) if callback is not None else None
        permitted=allowed_cuda_import_call(module,name,event,state=state,filename=frame.f_code.co_filename,
                    code_flags=frame.f_code.co_flags,callback=callback_name,context=context) if event=="call" else False
        invalid_return=event=="return" and module.startswith("torch.cuda") and state is not False
        if invalid_return or (event!="return" and forbidden_call(module,name,event) and not permitted):
            try:identity.append(output/"BOUNDARY_REFUSALS.jsonl",{"module":module,"name":name,"before_body":event!="return"})
            finally:fatal(86)
    sys.setprofile(profile);threading.setprofile(profile)
    return profile,context

def check_tensor(torch,tensor):
    require(tensor.device.type=="cpu" and tensor.dtype==torch.bfloat16 and tensor.is_contiguous(),"CPU BF16 contiguous tensor required")

def tensor_digest(torch,tensor):
    check_tensor(torch,tensor)
    pointer,total=tensor.data_ptr(),tensor.numel()*2
    digest=hashlib.sha256()
    for offset in range(0,total,CHUNK):
        span=min(CHUNK,total-offset)
        digest.update(memoryview((c.c_ubyte*span).from_address(pointer+offset)).cast("B"))
    return digest.hexdigest()

def allocate_cpu_once(torch,model):
    # A reference dictionary is not a second state-dict storage allocation.
    references=list(model.named_parameters(remove_duplicate=False));groups={}
    for name,parameter in references:
        require(parameter.device.type=="meta" and parameter.dtype==torch.bfloat16,"BF16 meta construction required")
        groups.setdefault(id(parameter),[]).append((name,parameter))
    require(model.lm_head.weight is model.model.embed_tokens.weight
            and sum(v[0][1].numel() for v in groups.values())==596049920,"meta tie/count mismatch")
    for members in groups.values():
        original=members[0][1]
        replacement=torch.nn.Parameter(torch.empty(tuple(original.shape),dtype=torch.bfloat16,device="cpu"),requires_grad=False)
        for name,_ in members:
            path,leaf=name.rsplit(".",1);model.get_submodule(path)._parameters[leaf]=replacement
    buffers=list(model.named_buffers(remove_duplicate=False))
    require([name for name,_ in buffers]==["model.rotary_emb.inv_freq"],"unexpected derived CPU buffer")
    name,buffer=buffers[0]
    require(buffer.device.type=="meta" and buffer.numel()==64 and buffer.dtype==torch.float32,"unexpected rotary meta buffer")
    replacement=torch.empty(tuple(buffer.shape),dtype=buffer.dtype,device="cpu")
    model.model.rotary_emb._buffers["inv_freq"]=replacement
    model.model.rotary_emb.original_inv_freq=replacement
    require(model.lm_head.weight is model.model.embed_tokens.weight
            and model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr(),"CPU tied storage duplicated")
    return {"unique_parameters":sum(p.numel() for p in model.parameters()),"derived_nonpersistent_buffers":
            [{"name":name,"device":"cpu","dtype":"float32","bytes":256,"values_initialized_or_qualified":False}]}

def load_streamed_cpu(torch,model,path,event,schedule):
    states=model.state_dict();buffer=bytearray(CHUNK);hashes={};chunks=0
    for tensor in states.values():check_tensor(torch,tensor)
    before=Path(path).stat()
    with Path(path).open("rb",buffering=0) as stream:
        header,header_bytes,payload=stream_checkpoint._header(stream,before.st_size,states,torch)
        require(sorted(header)==schedule,"canonical CPU tensor schedule mismatch")
        for name in sorted(header):
            event("tensor_copy_begin",name=name)
            begin,end=header[name]["data_offsets"];stream.seek(8+header_bytes+begin)
            target=states[name].detach().reshape(-1);digest=hashlib.sha256();offset=0
            while offset<end-begin:
                count=min(CHUNK,end-begin-offset);view=memoryview(buffer)[:count]
                stream_checkpoint._read_exact_into(stream,view);digest.update(view)
                source=torch.frombuffer(view,dtype=torch.bfloat16)
                target[offset//2:(offset+count)//2].copy_(source,non_blocking=False)
                del source,view;offset+=count;chunks+=1
            expected=digest.hexdigest();require(tensor_digest(torch,states[name])==expected,"CPU loaded tensor bytes mismatch: "+name)
            hashes[name]=expected;event("tensor_copy_end",name=name,sha256=expected)
    after=Path(path).stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),"canonical checkpoint identity drift")
    return hashes,{"cpu_model_allocated":True,"GPU_copy_requested":False,"read_buffer_bytes":CHUNK,
                   "digest_span_bytes_max":CHUNK,"payload_bytes":payload,"serialized_tensors":len(hashes),"chunks":chunks}

def binding(expected,source):
    return {"scope":SCOPE,"source_commit":source["commit"],"contract_sha256":expected,"CPU_source_sha256":native.sha(__file__)}

def require_job(store,go,args):
    store._require_lease(go["owner"],go["fence"])
    row=store.get_job(KEY)
    require(row["state"]=="RUNNING" and row["reserved_seconds"]==120 and row["epoch"]==args.epoch
            and row["payload"]==go["binding"] and row["request_hash"]==go["request_hash"]
            and row["worker"]==go["worker_identity"] and row["owner"]==go["owner"] and row["fence"]==go["fence"]
            and go["binding"]==binding(args.contract_sha256,go["source"]),"exact CPU admitted RUNNING row required")

def validate_completion(output,go,cfg):
    result=native.read_json(output/"CPU_LOAD_RESULT.json");events=identity.read_jsonl(output/"EVENTS.jsonl")
    require(result["binding"]==go["binding"] and result["status"]=="CPU_LOAD_ONLY_COMPLETED"
            and all(type(result.get(field)) is int and result[field]==value for field,value in
                    (("forwards",0),("tokenizers",0),("optimizer_calls",0),("serialized_tensors",310),("unique_parameters",596049920))),
            "CPU completion identity/dose mismatch")
    schedule=cfg["tensor_schedule"]
    require(type(schedule) is list and len(schedule)==310 and len(set(schedule))==310 and schedule==sorted(schedule)
            and sorted(result["tensor_sha256"])==schedule,"exact complete unique CPU tensor manifest required")
    require(all(type(digest) is str and len(digest)==64 and set(digest)<=set("0123456789abcdef") for digest in result["tensor_sha256"].values()),"CPU tensor digest schema mismatch")
    require(result["device"]=="cpu" and result["dtype"]=="bfloat16" and result["CUDA_initialized"] is False
            and result["asset_sha256"]==cfg["asset_sha256"] and result["tied_storage_identity"] is True
            and result["loader"]["cpu_model_allocated"] is True and result["loader"]["GPU_copy_requested"] is False
            and type(result["loader"]["serialized_tensors"]) is int and result["loader"]["serialized_tensors"]==310
            and result["loader"]["read_buffer_bytes"]==CHUNK and result["loader"]["digest_span_bytes_max"]==CHUNK
            and result["allocation"]["unique_parameters"]==596049920
            and result["allocation"]["derived_nonpersistent_buffers"]==[{"name":"model.rotary_emb.inv_freq","device":"cpu","dtype":"float32","bytes":256,"values_initialized_or_qualified":False}],
            "CPU truthful allocation/device/asset/byte verification predicates missing")
    expected=["import_begin","import_end","model_construct_begin","model_construct_end","cpu_allocate_begin","cpu_allocate_end","checkpoint_load_begin"]
    for name in schedule:expected.extend(["tensor_copy_begin","tensor_copy_end"])
    expected.extend(["checkpoint_load_end","loaded_verification_end"])
    require([e["phase"] for e in events]==expected and len(result["tensor_sha256"])==310,"CPU completion journal missing/pending/reordered")
    require(all(e["binding"]==go["binding"] for e in events),"CPU journal source binding mismatch")
    for name,begin,end in zip(schedule,events[7:-2:2],events[8:-2:2]):
        require(begin["name"]==end["name"]==name and end["sha256"]==result["tensor_sha256"][name],"CPU tensor journal mismatch")
    require(not (output/"BOUNDARY_REFUSALS.jsonl").exists(),"CPU fatal boundary refusal")
    return result

def child(args,cfg):
    import psutil
    deadline=time.monotonic()+5
    while not (args.output/"GO.json").exists() and time.monotonic()<deadline:time.sleep(.02)
    go=native.read_json(args.output/"GO.json")
    store=OperationsStore(host_state_path(),args.output/"receipts")
    try:
        require_job(store,go,args)
        require(native.committed_snapshot()==go["source"] and native.runtime_metadata()==cfg["runtime"],"CPU source/runtime drift")
        native.verify_assets(cfg)
        witness=cpu_query_witness(go["own_job_query_handle"],go["query_handle_target_pid"],go["query_handle_target_creation_time"])
        native.write_new(args.output/"CPU_CHILD_CONTAINMENT.json",witness)
        def event(phase,**detail):
            require_job(store,go,args)
            identity.append(args.output/"EVENTS.jsonl",{"phase":phase,"binding":go["binding"],**detail})
        event("import_begin");profile,boundary_context=arm_boundary(args.output,cfg)
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM
        boundary_context["import_phase"]=False
        def guard():
            require(vars(torch.cuda).get("_initialized") is False and sys.getprofile() is profile
                    and threading.getprofile() is profile,"CPU CUDA initialized or fatal boundary disabled")
            sample=cpu_telemetry(args.output);require(not admission_reasons(sample,rss=psutil.Process().memory_info().rss/GIB),"CPU child resource gate")
        guard();event("import_end",CUDA_initialized=False,imported_CUDA_modules=[m for m in sys.modules if m.startswith("torch.cuda")],hidden_C_effects="UNMEASURED")
        torch.set_num_threads(2)
        config=Qwen3Config(**native.read_json(Path(cfg["model_dir"])/"config.json"));config._attn_implementation="eager"
        event("model_construct_begin");prior=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):model=Qwen3ForCausalLM(config)
        finally:torch.set_default_dtype(prior)
        event("model_construct_end");guard();event("cpu_allocate_begin")
        allocation=allocate_cpu_once(torch,model);event("cpu_allocate_end",allocation=allocation);guard()
        event("checkpoint_load_begin")
        hashes,loader=load_streamed_cpu(torch,model,Path(cfg["model_dir"])/"model.safetensors",event,cfg["tensor_schedule"])
        event("checkpoint_load_end");guard()
        require(all(p.device.type=="cpu" and p.dtype==torch.bfloat16 for p in model.parameters())
                and all(b.device.type=="cpu" for b in model.buffers()) and allocation["unique_parameters"]==596049920,"loaded CPU devices/count mismatch")
        require(native.sha(Path(cfg["model_dir"])/"model.safetensors")==cfg["asset_sha256"]["model.safetensors"],"loaded canonical checkpoint binding drift")
        event("loaded_verification_end")
        native.write_new(args.output/"CPU_LOAD_RESULT.json",{"status":"CPU_LOAD_ONLY_COMPLETED","binding":go["binding"],
            "tensor_sha256":hashes,"loader":loader,"allocation":allocation,"serialized_tensors":310,"unique_parameters":596049920,
            "forwards":0,"tokenizers":0,"optimizer_calls":0,"device":"cpu","dtype":"bfloat16",
            "asset_sha256":cfg["asset_sha256"],"tied_storage_identity":model.lm_head.weight is model.model.embed_tokens.weight
                and model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr(),
            "native_GPU_qualified":False,"reader_qualified":False,"training_qualified":False,"CUDA_initialized":False,
            "hidden_C_activity":"UNMEASURED","accounting_finalized":False})
    finally:store.close()

def known_launcher(process):
    """Identity from the exact original Popen process handle, never a PID search."""
    kernel=c.WinDLL("kernel32",use_last_error=True)
    kernel.GetProcessId.argtypes=[w.HANDLE];kernel.GetProcessId.restype=w.DWORD
    kernel.GetProcessTimes.argtypes=[w.HANDLE,c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME),c.POINTER(w.FILETIME)];kernel.GetProcessTimes.restype=w.BOOL
    kernel.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,c.POINTER(w.DWORD)];kernel.QueryFullProcessImageNameW.restype=w.BOOL
    handle=int(process._handle);created,exited,kernel_time,user_time=w.FILETIME(),w.FILETIME(),w.FILETIME(),w.FILETIME()
    require(kernel.GetProcessId(handle)==process.pid and kernel.GetProcessTimes(handle,c.byref(created),c.byref(exited),c.byref(kernel_time),c.byref(user_time)),"known launcher handle identity unavailable")
    image=c.create_unicode_buffer(32768);size=w.DWORD(len(image))
    require(kernel.QueryFullProcessImageNameW(handle,0,image,c.byref(size)),"known launcher handle image unavailable")
    return {"pid":process.pid,"creation_FILETIME":(created.dwHighDateTime<<32)|created.dwLowDateTime,
            "executable":image.value,"original_Popen_handle":handle}

def cleanup_suspended_launcher(process,known,*,resumed):
    import psutil
    require(known is not None and resumed is False and known_launcher(process)==known,"known suspended Popen handle/image/time proof required")
    children=psutil.Process(known["pid"]).children(recursive=True)
    require(not children,"preJob launcher has uncontained descendants; outcome unknown")
    process.kill();process.wait(timeout=5)
    require(process.poll() is not None,"exact known suspended launcher exit unavailable")
    return {"owned_tree_exited":True,"remaining_owned_pids":[],"suspended_never_resumed_launcher":True,
            "known_handle_identity":known,"observed_at":time.time()}

def final_success_gate(outcome,*,completion,exit_proof,elapsed,expired,resource_reason):
    if exit_proof is None:return "OUTCOME_UNKNOWN"
    if outcome=="SUCCEEDED" and (completion is None or expired or resource_reason is not None
                                 or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed>120):return "FAILED"
    return outcome

def supervise(args,cfg):
    import psutil
    source=native.committed_snapshot();require(not source["dirty"],"clean exact CPU source required")
    require(native.runtime_metadata()==cfg["runtime"],"CPU pinned runtime drift")
    expected=binding(args.contract_sha256,source)
    require(len(args.review)==2,"CPU SCI/OPS approvals required")
    roles=set()
    for path in args.review:
        review=native.read_json(path)
        require(review.get("status")=="APPROVED_FOR_CPU_LOAD_ONLY" and review.get("binding")==expected
                and identity.reviewer_identity(review.get("reviewer"))=={"science":"exception_science","operations":"exception_ops"}.get(review.get("role")),"CPU scoped source review missing")
        roles.add(review["role"])
    require(roles=={"science","operations"},"CPU independent roles required")
    for candidate in psutil.process_iter(["pid","name"]):
        if candidate.pid==os.getpid() or "python" not in (candidate.info["name"] or "").lower():continue
        try:
            command=candidate.cmdline()
            require(not (any("experiments.rineya_centralized_reasoning_system_v1" in part for part in command)
                         and any(flag in command for flag in ("--child","--worker"))),"actual CRS worker already present")
        except psutil.NoSuchProcess:continue
    native.verify_assets(cfg)
    output=args.output.resolve();require(output.is_relative_to(PACKAGE/"results") and not output.exists(),"new CPU-only output required")
    first=cpu_telemetry(PACKAGE);require(not admission_reasons(first),"CPU admission1 refused")
    time.sleep(2);second=cpu_telemetry(PACKAGE);require(not admission_reasons(second),"CPU admission2 refused")
    output.mkdir(parents=True)
    store=OperationsStore(host_state_path(),output/"receipts")
    process=job=watch=None;lease=None;attempted=False;reserved=False;worker=None;started=None
    owner="CRS-CPU-"+uuid.uuid4().hex;outcome="FAILED";reason=None;exit_proof=None;completion=None
    stdout=stderr=None;budget=identity.HashBudget();absolute=None;expired=threading.Event();launcher=None;resumed=False
    try:
        row=store.register_job(KEY,expected,args.epoch);require(row["state"]=="PENDING","fixed CPU key already used")
        store.ready(KEY);lease=store.acquire_lease(owner,180);store.reserve(KEY,owner,lease["fence"],120);reserved=True;started=time.monotonic()
        command=[sys.executable,"-B","-m","experiments.rineya_centralized_reasoning_system_v1.cpu_native_load","--child",
                 "--contract-sha256",args.contract_sha256,"--output",str(output),"--epoch",args.epoch]
        native.write_new(output/"LAUNCH_INTENT.json",{"binding":expected,"command":command,"first":first,"second":second,"lease":lease})
        stdout=(output/"stdout.log").open("xb");stderr=(output/"stderr.log").open("xb")
        native.write_new(output/"LAUNCH_ATTEMPT.json",{"attempted_at":time.time(),"binding":expected});attempted=True
        def absolute_deadline():
            expired.set()
            try:identity.append(output/"ABSOLUTE_DEADLINE.jsonl",{"reason":"CPU_WALL120","creation_response_known":process is not None,"exact_job_known":job is not None})
            finally:
                if job is not None:job.kill()
        absolute=threading.Timer(max(0,120-(time.monotonic()-started)),absolute_deadline);absolute.daemon=True;absolute.start()
        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1"),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|4)
        launcher=known_launcher(process)
        worker={"pid":process.pid,"creation_time":str(psutil.Process(process.pid).create_time()),"command_hash":hashlib.sha256(canonical(command)).hexdigest()}
        store.mark_running(KEY,owner,lease["fence"],worker)
        job=CPUJob(process);watch=identity.ResourceWatch(job,started=started,output=output,seconds=120);watch.start()
        require(not expired.is_set(),"CPU absolute deadline elapsed during creation response")
        native.write_new(output/"CPU_JOB_LIMITS.json",job.limit_witness)
        observe_guard(job,cfg,output,"before_resume",budget);watch.check()
        require(native.job_process_ids(job)==[process.pid] and not psutil.Process(process.pid).children(recursive=True),"CPU launcher not suspended before owned assignment")
        native.resume_suspended_primary_thread(process)
        resumed=True
        until=time.monotonic()+3;targets=[]
        while time.monotonic()<until:
            current=observe_guard(job,cfg,output,"runtime_target",budget);watch.check()
            targets=[m for m in current["members"] if Path(m["executable"]).resolve()==Path(sys._base_executable).resolve()]
            if len(targets)==1:break
            time.sleep(.02)
        require(len(targets)==1,"one exact CPU CPython target required")
        go={"binding":expected,"source":source,"owner":owner,"fence":lease["fence"],"request_hash":row["request_hash"],
            "worker_identity":worker,**native.duplicate_owned_job_query_handle(job,targets[0])}
        native.write_new(output/"GO.json",go)
        next_gpu=0
        while True:
            store._require_lease(owner,lease["fence"]);watch.check();require(not expired.is_set(),"CPU absolute wall deadline")
            current=observe_guard(job,cfg,output,"runtime",budget)
            require(not admission_reasons(cpu_telemetry(output),rss=current["observed_rss_bytes"]/GIB),"CPU live resource gate")
            if time.monotonic()>=next_gpu:
                identity.append(output/"GPU_ATTRIBUTION.jsonl",owned_gpu_observation(job));next_gpu=time.monotonic()+.5
            require(not (output/"BOUNDARY_REFUSALS.jsonl").exists(),"CPU fatal API boundary refused")
            if process.poll() is not None and not native.job_process_ids(job):
                require(process.returncode==0,"CPU worker failed")
                completion=validate_completion(output,go,cfg);watch.check();outcome="SUCCEEDED";break
            time.sleep(.05)
    except BaseException as error:reason=type(error).__name__+": "+str(error)
    finally:
        try:
            if job:exit_proof=native.drain_owned_job(job)
            elif process is not None and launcher is not None and not resumed:
                exit_proof=cleanup_suspended_launcher(process,launcher,resumed=resumed)
                if worker is None:
                    # Identified preJob exit remains charged; bind this known handle observation to the existing reservation.
                    worker={"pid":launcher["pid"],"creation_time":str(launcher["creation_FILETIME"]),
                            "command_hash":hashlib.sha256(canonical(command)).hexdigest()}
                    store.mark_running(KEY,owner,lease["fence"],worker)
            if process is not None:process.wait(timeout=5)
        except BaseException as error:reason=(reason or "")+"; exit unknown: "+str(error);exit_proof=None
        if absolute:absolute.cancel()
        if watch:watch.close()
        if job:job.close()
        for pipe in (stdout,stderr):
            if pipe:pipe.close()
        try:
            if reserved:
                if attempted and (worker is None or exit_proof is None):
                    store.mark_unknown(KEY,reason or "CPU dispatch/exit uncertain; full120 reservation held")
                    outcome="OUTCOME_UNKNOWN"
                    native.write_new(output/"ACCOUNTING_UNKNOWN.json",{"state":"OUTCOME_UNKNOWN","reserved_seconds":120,"reason":reason})
                else:
                    elapsed=0 if not attempted else max(.000001,time.monotonic()-started)
                    outcome=final_success_gate(outcome,completion=completion,exit_proof=exit_proof if attempted else {"never_dispatched":True},
                        elapsed=elapsed,expired=expired.is_set(),resource_reason=watch.reason if watch else None)
                    if elapsed>120 or expired.is_set() or (watch and watch.reason):reason=(reason or "")+"; final CPU wall/resource refusal"
                    proof={"observer":"CPU_LOAD_SUPERVISOR","request_hash":row["request_hash"],"owner":owner,"fence":lease["fence"],
                           "elapsed_native_seconds":elapsed,"worker_identity":worker,"process_exited":bool(attempted),
                           "exit_observed_at":time.time(),"owned_tree_exited":bool(exit_proof),"owned_tree_exit":exit_proof,
                           "never_dispatched":not attempted,"launch_attempted":attempted}
                    ref=store.write_receipt("native_outcome_observation",KEY,proof)
                    store.reconcile(KEY,outcome,elapsed,{"native_outcome_proof_ref":ref,"reason":reason},owner=owner,fence=lease["fence"])
                    store.release_lease(owner,lease["fence"])
                    native.write_new(output/"FINALIZATION_RECEIPT.json",{"status":outcome,"reason":reason,"actual_seconds":elapsed,"proof_ref":ref,
                        "CPU_load_only_observed":completion is not None,"native_GPU_reader_training_qualified":False})
        except BaseException as error:
            row_state=store.get_job(KEY)["state"] if reserved else None
            outcome="OUTCOME_UNKNOWN" if row_state not in TERMINAL else "PUBLICATION_FAILED_KNOWN_SCHEDULER_OUTCOME"
            native.write_new(output/"FINALIZATION_ERROR.json",{"outward_status":outcome,"scheduler_state":row_state,
                "error":type(error).__name__+": "+str(error),"reservation_held":row_state not in TERMINAL})
        finally:store.close()
    return {"outcome":outcome,"reason":reason,"CPU_load_only_observed":completion is not None}

def recover_finalization(output,store,expected):
    """Terminal publication/lease recovery only; never reserve or dispatch."""
    row=store.get_job(KEY)
    require(row["state"] in TERMINAL and row["actual_seconds"] is not None and row["payload"]==expected,
            "CPU terminal same-binding known-cost row required")
    ref=row["result"]["evidence"]["native_outcome_proof_ref"]
    proof=store.verify_receipt(ref)["payload"]
    require(proof["request_hash"]==row["request_hash"] and proof["elapsed_native_seconds"]==row["actual_seconds"]
            and ((proof.get("process_exited") is True and proof.get("owned_tree_exited") is True
                  and proof["owned_tree_exit"]["owned_tree_exited"] is True) or
                 (proof.get("never_dispatched") is True and proof.get("launch_attempted") is False)),
            "CPU immutable observed owned-exit/no-dispatch proof required")
    lease=store.db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
    if lease is not None and lease["status"]=="LIVE":
        require((lease["owner"],lease["fence"])==(row["owner"],row["fence"]),"CPU recovery cannot release another lease")
        store.release_lease(row["owner"],row["fence"])
    result={"status":row["state"],"actual_seconds":row["actual_seconds"],"proof_ref":ref,"recovery_dispatches":0,
            "native_GPU_reader_training_qualified":False}
    target=output/"RECOVERED_FINALIZATION_RECEIPT.json"
    if target.exists():require(native.read_json(target)==result,"CPU recovered publication drift")
    else:native.write_new(target,result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--epoch",default="CRS-EPOCH-001")
    parser.add_argument("--child",action="store_true");parser.add_argument("--review",type=Path,action="append",default=[])
    parser.add_argument("--recover-finalization",action="store_true")
    args=parser.parse_args();cfg=checked_contract(args.contract_sha256)
    if args.recover_finalization:
        store=OperationsStore(host_state_path(),args.output/"receipts")
        try:print(json.dumps(recover_finalization(args.output,store,binding(args.contract_sha256,native.committed_snapshot())),indent=2))
        finally:store.close()
    elif args.child:child(args,cfg)
    else:print(json.dumps(supervise(args,cfg),indent=2))

if __name__=="__main__":main()
