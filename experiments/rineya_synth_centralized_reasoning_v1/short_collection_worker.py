"""Staged owned one-fixed-batch M0 raw collector; no launch on import.

Native run is blocked while staged. Collection records generated-only strings,
tokens and likelihoods; it has no grader, oracle sidecar, training or checkpoint
restore path. Completion does not certify correctness or prior-correct retention.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import short_collection_contract as contract
sys.path.insert(0,str(contract.PACKAGE))
from inference_component import InferenceComponent, validate_row
from inference_qual_worker import make_guard
import worker_ram_probe
from native_loader import load_titan, verify_frozen_base
from native_precision import install_fp32_compute, install_qv_lora, install_fp32_activation_assertions, zero_lora_status
from native_worker import Journal, verify_worker_ownership, preserve_terminal
from rs_contract import digest, require

GIB=1024**3


class ForwardDose:
    def __init__(self,torch,config):
        self.torch,self.maximum_calls,self.maximum_positions=torch,config["dose"]["maximum_forward_calls"],config["dose"]["maximum_forward_input_positions"]
        require(self.maximum_calls==2050 and type(self.maximum_positions) is int and self.maximum_positions>0,"new explicit batch2050 dose; no inherited262 cap")
        self.current=None;self.calls=self.positions=0;self.operations={}
    def before(self,model,args,kwargs):
        ids=kwargs.get("input_ids")
        require(self.torch.is_tensor(ids) and tuple(ids.shape[:1])==(1,) and len(ids.shape)==2 and ids.dtype==self.torch.long
                and 0<ids.shape[1]<=512 and self.current in ("zero_parity_base_forward","zero_parity_adapter_forward","inference_forward"),"registered bounded model forward")
        length=int(ids.shape[1]);require(self.calls<self.maximum_calls and self.positions+length<=self.maximum_positions,"actual batch forward/position cap exhausted")
        self.calls+=1;self.positions+=length
        entry=self.operations.setdefault(self.current,{"attempted_calls":0,"input_positions":0});entry["attempted_calls"]+=1;entry["input_positions"]+=length
    def wrap(self,call):
        def invoke(operation,function):
            previous=self.current;self.current=operation
            try:return call(operation,function)
            finally:self.current=previous
        return invoke
    def snapshot(self):
        return deepcopy({"attempted_calls":self.calls,"input_positions":self.positions,"operations":self.operations,
                         "maximum_forward_calls":self.maximum_calls,"maximum_forward_input_positions":self.maximum_positions})


class LocalGeneratedDecoder:
    """Only the byte-bound tokenizers JSON decoder; no AutoTokenizer/fallback."""
    def __init__(self,config):
        self.config=config
        self.verify_binding()
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        from tokenizers import Tokenizer
        self.engine=Tokenizer.from_file(str(Path(config["model_dir"])/"tokenizer.json"))
        require(self.engine.token_to_id("<|im_end|>")==151645 and self.engine.token_to_id("<|endoftext|>")==151643,"decoder exact special-token identities")
    def verify_binding(self):
        require(importlib.metadata.version("tokenizers")==self.config["runtime_versions"]["tokenizers"]=="0.22.2","actual decoder runtime version")
        for name in ("tokenizer.json","tokenizer_config.json"):
            require(contract.native.filehash(Path(self.config["model_dir"])/name)==self.config["model_pins"][name],"local decoder asset drift")
        for filename,pin in self.config["decoder_runtime_pins"].items():require(contract.native.filehash(Path(filename))==pin,"decoder source/binary drift")
    def decode(self,ids,*,skip_special_tokens):
        require(skip_special_tokens is True and type(ids) is list and len(ids)<=32 and all(type(x) is int and 0<=x<151936 for x in ids),"bounded generated-only decoder input")
        return self.engine.decode(list(ids),skip_special_tokens=True)


class DurableRawCollection(InferenceComponent):
    """Use collect unchanged, adding synchronous per-token partial preservation."""
    def __init__(self,*args,source_packet_sha256,**kwargs):
        self.packet_sha=source_packet_sha256
        super().__init__(*args,**kwargs)
    def _trace(self,row,operation,trace,step_preserve=None):
        def preserve_token(step):
            receipt=self._receipt(row,self.packet_sha,trace,"RAW_TOKEN_PARTIAL_UNSCORED")
            if receipt["finish_reason"] != "eos" and len(trace) < self.generation_cap:
                receipt["finish_reason"]="in_progress"
                receipt["truncated"]=False
            receipt["row_ordinal"]=self.row_cursor;receipt["step_ordinal"]=len(trace)
            self._persist("raw_row%04d_token%02d"%(self.row_cursor,len(trace)),receipt)
        return super()._trace(row,operation,trace,preserve_token)


def _logits(torch,value,length,device):
    require(torch.is_tensor(value) and tuple(value.shape)==(1,length,151936) and value.dtype==torch.float32
            and str(value.device)==device and not value.requires_grad and bool(torch.isfinite(value).all()),"full finite FP32 no-grad parity logits")
    return value


def collect_loaded(torch,model,decoder,config,source_hashes,journal,guard,report,dose,*,
        compute=install_fp32_compute,adapters=install_qv_lora,activation_hooks=install_fp32_activation_assertions,
        zero=zero_lora_status,base_verify=verify_frozen_base,device="cuda:0"):
    """Injected CPU mocks can check protocol; native run uses exact helpers."""
    rows=[validate_row(r) for r in config["rows"]]
    require(len(rows)==64 and len({r["row_id"] for r in rows})==64 and digest(rows)==config["source_packet_sha256"],"whole64 prompt-only packet before model work")
    require(not any(n.endswith((".lora_A",".lora_B")) for n,_ in model.named_parameters()),"fresh dense M0; no STATE1/study/adapter warmstart")
    call=dose.wrap(journal.call);report["compute"]=call("install_fp32_compute",lambda:compute(model,torch))
    hooks=call("install_activation_hooks",lambda:activation_hooks(model,torch));forward_hook=model.register_forward_pre_hook(dose.before,with_kwargs=True)
    component=None
    try:
        ids=torch.tensor([rows[0]["prompt_ids"]],dtype=torch.long,device=device)
        with torch.no_grad():
            baseline=_logits(torch,call("zero_parity_base_forward",lambda:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits),len(rows[0]["prompt_ids"]),device)
        report["adapters"]=call("install_qv_lora",lambda:adapters(model,torch))
        report["zero_adapter_before"]=call("zero_lora_query_before",lambda:zero(model,torch))
        require(report["zero_adapter_before"]["all_B_zero"] is True,"fresh exact zero B")
        with torch.no_grad():
            adapted=_logits(torch,call("zero_parity_adapter_forward",lambda:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits),len(rows[0]["prompt_ids"]),device)
            equal=bool(torch.equal(baseline,adapted));maximum=0.0 if equal else float((baseline-adapted).abs().max().item())
        report["zero_adapter_parity"]={"equal":equal,"maximum_error":maximum,"capability_scored":False,"dose":dose.snapshot()}
        journal.evidence("zero_adapter_parity",report["zero_adapter_parity"]);require(equal and maximum==0.,"physical zero-B output parity")
        del baseline,adapted,ids
        component=DurableRawCollection(torch,model,decoder,call,guard,journal.evidence,
            source_packet_sha256=config["source_packet_sha256"],max_attempted_forwards=2048,
            max_input_positions=config["dose"]["maximum_capture_input_positions"],context_cap=512,generation_cap=32,device=device)
        collected=component.collect(rows,source_packet_sha256=config["source_packet_sha256"])
        require(len(collected["rows"])==64 and [r["row_id"] for r in collected["rows"]]==[r["row_id"] for r in rows],"complete fixed batch/order; no reduction after outputs")
        report["collection"]=collected
        report["parameter_preservation"]=component.verify_unchanged()
        report["zero_adapter_after"]=call("zero_lora_query_after",lambda:zero(model,torch));require(report["zero_adapter_after"]["all_B_zero"] is True,"zero B changed without training authority")
        report["base_postcheck"]=call("verify_frozen_base",lambda:base_verify(torch,model,source_hashes,call,guard))
        require(report["base_postcheck"]["verified_source_tensors"]==310 and report["base_postcheck"]["frozen_base_unchanged"] is True,"all310 native frozen base bytes unchanged")
        require(hooks["observations"] and all(n>0 for n in hooks["observations"].values()),"all activation sites observed")
        report["activation_observations"]=deepcopy(hooks["observations"])
        if callable(getattr(decoder,"verify_binding",None)):decoder.verify_binding()
        journal.evidence("short_M0_collection_completed".lower(),{"rows":64,"source_packet_sha256":config["source_packet_sha256"],"dose":dose.snapshot(),"correctness_certified":False,"retention_certified":False})
    finally:
        report["forward_dose"]=dose.snapshot()
        if component is not None:
            report["collection_costs"]=component.costs();report["last_partial_receipt"]=deepcopy(component.last_receipt)
        forward_hook.remove()
        for handle in hooks["handles"]:handle.remove()
    return report


def run(config_path,output):
    contract.integrated()  # No staged escape, even with a fabricated ownership file.
    output,config_path=Path(output).absolute(),Path(config_path).absolute()
    require(output.resolve().parent==(contract.PACKAGE/"artifacts").resolve() and output.is_dir() and not output.is_symlink(),"owned new output")
    journal=Journal(output);start=time.monotonic();torch=model=dose=None
    status,error,report="SHORT_M0_RAW_COLLECTION_FAILED",None,{}
    try:
        require((output/"supervisor_ownership.json").is_file(),"ownership receipt before framework required")
        owner=contract.native.read_json(output/"supervisor_ownership.json");ownership=verify_worker_ownership(owner,config_path);journal.evidence("worker_ownership_verified",ownership)
        guard=make_guard(output,owner,start,lambda:torch,available=worker_ram_probe.available_bytes);journal.guard=guard
        config=contract.preflight(config_path);report["source_commit"],report["config_sha256"]=config["source_commit"],config["config_sha256"]
        report["batch_selector"]=deepcopy(config["batch_selector"])
        for k,v in config["offline_environment"].items():os.environ[k]=v
        os.environ["CUDA_VISIBLE_DEVICES"]="0";guard("pre_import");require(shutil.disk_usage(contract.PACKAGE).free>=4*GIB,"disk admission floor")
        import torch as loaded_torch
        torch=loaded_torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        require(str(torch.__version__)==config["runtime_versions"]["torch"] and importlib.metadata.version("transformers")==config["runtime_versions"]["transformers"],"framework runtime versions")
        import psutil
        rss=psutil.Process().memory_info().rss;require(psutil.virtual_memory().available>=1.25*GIB+max(0,2*GIB-rss),"post-import full growth reserve")
        journal.call("cuda_context_init",torch.cuda.init);require(torch.cuda.mem_get_info()[0]>=4*GIB,"preload GPU admission")
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
        random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101);torch.cuda.reset_peak_memory_stats()
        model,loader,source_hashes=load_titan(torch,Qwen3Config,Qwen3ForCausalLM,config["model_dir"],journal.call,guard)
        report["loader"]=loader;journal.evidence("loader",loader)
        decoder=journal.call("generated_decoder_construct",lambda:LocalGeneratedDecoder(config))
        dose=ForwardDose(torch,config);collect_loaded(torch,model,decoder,config,source_hashes,journal,guard,report,dose)
        contract.preflight(config_path);guard("completed");status="COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED"
    except BaseException as exc:
        error={"type":type(exc).__name__,"message":str(exc)}
        try:journal.emit({"event":"worker_failed","error":error})
        except BaseException as fail:report["journal_failure"]={"type":type(fail).__name__,"message":str(fail)}
    finally:
        report.update({"status":status,"error":error,"wall_seconds":time.monotonic()-start,"operation_accounting":journal.snapshot(),
            "scope":contract.PURPOSE,"correctness_certified":False,"retention_certified":False,"scoring_authorized":False,
            "production_training_authorized":False,"language_qualification_established":False,"long_context_qualification_established":False,
            "protected_bank_access":False,"paid_services":False,"prior_correct_enrollment_performed":False,"confirmation_eligible":False,
            "quality_gain_established":False,"optimizer_steps":0,"backwards":0,"requires_independent_supervised_audit":True})
        if dose is not None:report["forward_dose"]=dose.snapshot()
        preserve_terminal(output,report,journal,torch)
    return 0 if status=="COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED" else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();return run(args.config,args.output_dir)


if __name__=="__main__":raise SystemExit(main())
