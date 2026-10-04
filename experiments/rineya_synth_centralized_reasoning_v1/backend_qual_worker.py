"""Owned A9+10/BfreshC9+10 worker. Staging has no native dispatch permission."""
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
PACKAGE=HERE.parents[1] if HERE.name=="backend_qual_work" and HERE.parent.name=="artifacts" else HERE
for location in (PACKAGE,HERE):
    if str(location) not in sys.path: sys.path.insert(0,str(location))
import backend_qual_contract as contract
from backend_qual_core import run_case
from backend_qual_native import NativeAdapter
from inference_qual_worker import make_guard
from native_worker import Journal,verify_worker_ownership,preserve_terminal
from native_loader import load_titan
from study_torch_backend import StudyTorchBackend
from rs_contract import canonical,digest,require

GIB=1024**3


class DurableJournal(Journal):
    def emit(self,value):
        super().emit(value)
        os.fsync(self.stream.fileno())


def remove_hooks(hooks):
    errors=[]
    for hook in hooks:
        try: hook.remove()
        except BaseException as error: errors.append({"type":type(error).__name__,"message":str(error)})
    return errors


def run(config_path,output):
    """Only an integrated, committed, explicit qualification scope can proceed."""
    # Stage/native authority rejection precedes any owned-Job, framework or GPU API.
    config=contract.preflight(config_path)
    output=Path(output).absolute()
    require(output.resolve().parent==(PACKAGE/"artifacts").resolve() and output.is_dir() and not output.is_symlink(),"owned new output leaf")
    require((output/"supervisor_ownership.json").is_file(),"owned supervisor dispatch required")
    owner=contract.native.read_json(output/"supervisor_ownership.json")
    ownership=verify_worker_ownership(owner,config_path)
    journal=DurableJournal(output);start=time.monotonic();torch=None;hooks=[]
    guard=make_guard(output,owner,start,lambda:torch)
    journal.guard=guard;journal.evidence("worker_ownership_verified",ownership)
    report={"status":"FAILED_SCOPED_BACKEND_QUALIFICATION","phase":config["phase"],"config_sha256":config["config_sha256"],
            "production_training_authorized":False,"protected_bank_access":False,"study_warmstart_allowed":False,"quality_gain_established":False}
    def preserve(name,value):
        if name=="checkpoint_path":
            require(value["filename"]=="qualification_C9.pt","only private C9 checkpoint filename")
            path=output/value["filename"];require(not path.exists() and not path.is_symlink(),"write-once C9")
            return str(path)
        journal.evidence(name,value)
    def admission(request):
        guard("before_framework_model_load_and_constructor")
        return {"binding":request,"authorized":True,"owned_supervision_verified":True,"observed_monotonic":time.monotonic()}
    def factory(scope,call,resource_guard,dose):
        nonlocal torch
        for key,value in scope["offline_environment"].items(): os.environ[key]=value
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        require(shutil.disk_usage(PACKAGE).free>=4*GIB,"disk admission floor")
        def frameworks():
            import torch as imported
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            return imported,Qwen3Config,Qwen3ForCausalLM
        torch,config_class,model_class=call("qualification_framework_import",frameworks)
        require(str(torch.__version__)==scope["runtime_versions"]["torch"]
                and importlib.metadata.version("transformers")==scope["runtime_versions"]["transformers"],"executing framework versions")
        import psutil
        rss=psutil.Process().memory_info().rss
        require(psutil.virtual_memory().available>=1.25*GIB+max(0,2*GIB-rss),"post-import whole growth budget")
        call("cuda_context_init",torch.cuda.init)
        require(call("qualification_GPU_preload_admission",lambda:torch.cuda.mem_get_info()[0])>=4*GIB,"fresh GPU preload admission")
        def configure():
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
            random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101);torch.cuda.reset_peak_memory_stats()
        call("qualification_precision_seed_configure",configure)
        model,loader_receipt,source_hashes=load_titan(torch,config_class,model_class,scope["model_dir"],call,resource_guard)
        journal.evidence("loader",loader_receipt)
        # All actual forwards, including any constructor surprise, are metered.
        hook=model.register_forward_pre_hook(dose.before,with_kwargs=True);hooks.append(hook)
        backend=StudyTorchBackend(torch,model,call,resource_guard,arm="M1",seed=101,
                    recipe_sha256=scope["qualification_recipe"]["sha256"],cohort_sha256=scope["qualification_recipe"]["cohort_sha256"],
                    execution_scope="UNQUALIFIED_NATIVE_IMPLEMENTATION",source_hashes=source_hashes)
        return NativeAdapter(torch,backend,call,resource_guard,dose)
    try:
        report=run_case(config,factory,journal.call,guard,preserve,admission=admission)
        report["source_commit"]=config["source_commit"]
        if report["status"].startswith("PASS_SCOPED"):
            contract.preflight(config_path);guard("qualification_completed")
    except BaseException as error:
        report["status"]="FAILED_SCOPED_BACKEND_QUALIFICATION";report["error"]={"type":type(error).__name__,"message":str(error)}
        try: journal.emit({"event":"worker_failed","error":report["error"]})
        except BaseException as failure: report["journal_shutdown_error"]={"type":type(failure).__name__,"message":str(failure)}
    finally:
        cleanup_errors=remove_hooks(hooks)
        if cleanup_errors:
            report["status"]="FAILED_SCOPED_BACKEND_QUALIFICATION";report["hook_cleanup_errors"]=cleanup_errors
        report.update({"wall_seconds":time.monotonic()-start,"operation_accounting":journal.snapshot(),"config_sha256":config["config_sha256"],
                       "production_training_authorized":False,"study_training_authorized":False,"study_warmstart_allowed":False,
                       "DEV_collection_authorized":False,"scoring_authorized":False,"protected_bank_access":False,"quality_gain_established":False,
                       "full_resumed9_qualified":False,"full27_qualified":False,"automatic_retry_authorized":False})
        report["sha256"]=digest({k:v for k,v in report.items() if k!="sha256"})
        preserve_terminal(output,report,journal,torch)
    return 0 if report["status"].startswith("PASS_SCOPED") else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    return run(args.config,args.output_dir)


if __name__=="__main__": raise SystemExit(main())
