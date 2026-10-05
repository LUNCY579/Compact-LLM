"""Injected bounded qualification controller; no framework/model imports or CLI."""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=="backend_qual_work" and HERE.parent.name=="artifacts" else HERE
for location in (PACKAGE,HERE):
    if str(location) not in sys.path: sys.path.insert(0,str(location))
import backend_qual_contract as contract
from study_native_chunk import _PrimitiveCollector
from rs_contract import digest,require


class ForwardDose:
    """Physical model-prehook attempts, including all zero/reference forwards."""
    OPERATIONS={"training_forward":"train","reference_forward":"reference",
                "zero_parity_base_forward":"zero_parity","zero_parity_adapter_forward":"zero_parity"}
    def __init__(self,scope):
        self.scope=scope;self.stack=[];self.attempted_calls=self.input_positions=0
        self.operations={name:{"attempted_calls":0,"input_positions":0} for name in ("train","reference","zero_parity")}
    def wrap(self,call):
        def counted(operation,function):
            self.stack.append(operation)
            try: return call(operation,function)
            finally: self.stack.pop()
        return counted
    def charge(self,length):
        require(self.stack and self.stack[-1] in self.OPERATIONS,"unregistered physical forward")
        require(type(length) is int and 0<length<=512,"one complete bounded sequence")
        require(self.attempted_calls<self.scope["dose"]["maximum_forward_calls"] and self.input_positions+length<=102400,"qualification forward/position cap")
        category=self.OPERATIONS[self.stack[-1]]
        limit=self.scope["dose"][("train" if category=="train" else "reference" if category=="reference" else "zero_parity")+"_forwards"]
        require(self.operations[category]["attempted_calls"]<limit,"forward category cap")
        self.attempted_calls+=1;self.input_positions+=length
        self.operations[category]["attempted_calls"]+=1;self.operations[category]["input_positions"]+=length
    def before(self,model,args,kwargs):
        ids=kwargs.get("input_ids")
        require(hasattr(ids,"shape") and len(ids.shape)==2 and ids.shape[0]==1 and str(ids.dtype)=="torch.int64","single typed model input")
        self.charge(int(ids.shape[1]))
    def snapshot(self):
        return {"attempted_calls":self.attempted_calls,"input_positions":self.input_positions,"operations":deepcopy(self.operations),
                "maximum_forward_calls":self.scope["dose"]["maximum_forward_calls"],"maximum_forward_input_positions":102400}
    def complete(self):
        snapshot=self.snapshot();spec=self.scope["dose"]
        require(snapshot["attempted_calls"]==spec["maximum_forward_calls"] and snapshot["input_positions"]==spec["expected_forward_input_positions"],"all registered forwards/positions must complete")
        require(all(snapshot["operations"][name]["attempted_calls"]==spec[key] for name,key in
                    (("train","train_forwards"),("reference","reference_forwards"),("zero_parity","zero_parity_forwards"))),"complete forward factor counts")
        return snapshot


def _finite(value,label):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,"finite nonnegative "+label)
    return float(value)


def _protocol_counts():
    return {name:{key:0 for key in ("attempted","returned","failed","pending")} for name in
            ("gradient_reset","forward","loss","backward","clip","step","checkpoint_save","checkpoint_restore","reference_control")}


def run_case(scope,factory,call,guard,preserve,*,cpu_control=False,admission=None):
    """A:200 TRAIN+2parity; B:20 TRAIN+20reference+2parity. Never study.

    Native factory must include import/model loading/constructor and receive the
    shared journal/dose hook. CPU controls require no framework modules anywhere.
    preserve(name,JSON) is durable, exclusive, and receives completed fragments.
    """
    require(not any(name.split(".",1)[0] in {"torch","transformers","jax","tensorflow","flax"} for name in sys.modules),"no framework before owned qualification admission")
    require(type(cpu_control) is bool and all(callable(value) for value in (factory,call,guard,preserve)),"explicit qualified factory/journal/guard/preservation")
    frozen=deepcopy(scope);registration,rows=contract.validate_packet(frozen)
    phase=frozen["phase"];prior=None
    if phase=="B": prior=contract.verify_predecessor(frozen["predecessor"],frozen,native_supervised=not cpu_control)
    if not cpu_control:
        require(contract.HERE==contract.PACKAGE and frozen["owned_qualification_authorized"] is True and frozen["staged_native_dispatch_forbidden"] is False and callable(admission),"owned reviewed qualification only; default/staged native denied")
        request={"qualification_recipe_sha256":frozen["qualification_recipe"]["sha256"],"source_packet_sha256":frozen["source_packet_sha256"],"phase":phase,"stage":"before_framework_model_load_and_constructor"}
        accepted=admission(deepcopy(request));now=time.monotonic()
        require(type(accepted) is dict and accepted.get("binding")==request and accepted.get("authorized") is True and accepted.get("owned_supervision_verified") is True
                and type(accepted.get("observed_monotonic")) in (int,float) and 0<=now-accepted["observed_monotonic"]<=.2,"fresh source/phase-bound owned admission")
    collector=_PrimitiveCollector(call,guard);dose=ForwardDose(frozen);shared=dose.wrap(collector)
    counts=_protocol_counts();adapter=None
    result={"phase":phase,"source_packet_sha256":frozen["source_packet_sha256"],"qualification_recipe_sha256":frozen["qualification_recipe"]["sha256"],
            "config_sha256":frozen.get("config_sha256",digest(frozen)),"status":"FAILED_SCOPED_BACKEND_QUALIFICATION","update_summaries":[],
            "checkpoint_c9":None,"witness10":None,"native_execution_performed":not cpu_control,"CPU_fake_control":cpu_control,
            "production_training_authorized":False,"study_training_authorized":False,"study_warmstart_allowed":False,
            "DEV_collection_authorized":False,"scoring_authorized":False,"protected_bank_access":False,"quality_gain_established":False,
            "full_resumed9_qualified":False,"full27_qualified":False,"automatic_retry_authorized":False,
            "qualification_scope":frozen["qualification_scope"],"whole_rows_prevalidated":200}
    def invoke(name,function):
        item=counts[name];item["attempted"]+=1;item["pending"]+=1
        guard("qualification_before_"+name)
        try: value=function()
        except BaseException: item["pending"]-=1;item["failed"]+=1;raise
        item["pending"]-=1;item["returned"]+=1
        guard("qualification_after_"+name);return value
    try:
        adapter=factory(frozen,shared,guard,dose)
        backend=adapter.backend
        require(backend.call is shared and backend.guard is guard and backend.arm=="M1" and backend.seed==101
                and backend.recipe_sha256==frozen["qualification_recipe"]["sha256"]
                and backend.cohort_sha256==frozen["qualification_recipe"]["cohort_sha256"],"qualified factory must use shared bound journal/source recipe")
        zero=adapter.zero_parity(deepcopy(rows[0]));preserve("zero_parity",zero)
        require(zero.get("equal") is True and zero.get("max_error")==0.0 and zero.get("all_B_zero") is True,"physical exact zero-adapter parity")
        if phase=="B":
            invoke("checkpoint_restore",lambda:backend.restore_checkpoint(deepcopy(prior["checkpoint_c9"]),deepcopy(frozen["c9_binding"])))
            restored=adapter.witness(deepcopy(frozen["c9_binding"]));preserve("restored_c9",restored)
            require(restored==prior["witness9"],"fresh-process C9 state must match independently detached A witness bitwise")
            reference=invoke("reference_control",lambda:adapter.prepare_reference(deepcopy(rows[180:200])))
        updates=registration["arms"]["M1"]["updates"][:10] if phase=="A" else registration["arms"]["M1"]["updates"][9:10]
        for update in updates:
            ordinal=update["update_ordinal"];invoke("gradient_reset",lambda:backend.reset_gradients(ordinal))
            losses=[]
            for slot,frozen_row in enumerate(rows[(ordinal-1)*20:ordinal*20],1):
                row=deepcopy(frozen_row);output=invoke("forward",lambda:backend.forward(row))
                require(row==frozen_row,"backend/reader row mutation")
                loss=invoke("loss",lambda:backend.completion_loss(output,row))
                losses.append(_finite(backend.last_loss_check["ce"],"completion-plus-EOS sequence mean loss"))
                invoke("backward",lambda:backend.backward(loss,scale=.05))
                del output,loss
                preserve("ce_u%02d_r%02d"%(ordinal,slot),deepcopy(backend.last_loss_check))
            if phase=="B":
                comparison=adapter.compare_reference(reference);preserve("reference_gradient_comparison",comparison)
                require(comparison.get("passed") is True and comparison.get("atol")==1e-6 and comparison.get("rtol")==1e-5
                        and comparison.get("count")==adapter.expected_gradient_count,"all registered reference gradients within unchanged tolerances")
            norm=_finite(invoke("clip",lambda:backend.finite_clip_gradients(max_norm=1.0)),"preclip gradient norm")
            invoke("step",lambda:backend.optimizer_step(lr=update["learning_rate"],update_ordinal=ordinal))
            result["update_summaries"].append({"update_ordinal":ordinal,"learning_rate":update["learning_rate"],"whole_sequences":20,"sequence_mean_loss":sum(losses)/20,"preclip_norm":norm})
            if ordinal==9:
                result["witness9"]=adapter.witness(deepcopy(frozen["c9_binding"]));preserve("detached_c9_witness",result["witness9"])
                path=preserve("checkpoint_path",{"filename":"qualification_C9.pt","qualification_recipe_sha256":frozen["qualification_recipe"]["sha256"]})
                require(type(path) is str and Path(path).is_absolute() and not Path(path).exists() and "state1" not in path.lower(),"private new C9 path; no STATE1/warmstart")
                result["checkpoint_c9"]=invoke("checkpoint_save",lambda:backend.save_checkpoint(path,deepcopy(frozen["c9_binding"])))
                contract.verify_descriptor_bytes(result["checkpoint_c9"])
                preserve("checkpoint_c9_descriptor",result["checkpoint_c9"])
        result["witness10"]=adapter.witness(deepcopy(frozen["c10_binding"]));preserve("detached_witness10",result["witness10"])
        require(result["witness10"].get("kind")==contract.WITNESS_KIND and result["witness10"].get("binding")==frozen["c10_binding"],"complete non-checkpoint W10")
        if phase=="B": require(result["witness10"]==prior["witness10"],"fresh-process update10 continuation must be bitwise identical to continuous A")
        dose.complete()
        expected_checks=frozen["resource_feasibility"]["A_total_full_base_checks" if phase=="A" else "B_total_full_base_checks"]
        require(collector.snapshot().get("native_base_content_verify")=={"attempted":expected_checks,"returned":expected_checks,"failed":0,"pending":0},"exact full-base content-check cost count")
        require(all(item["failed"]==item["pending"]==0 for item in collector.snapshot().values()),"failed/pending primitive calls forbid success")
        result["status"]=("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL" if phase=="A" else "PASS_CPU_FAKE_BACKEND_FRESH_CONT10_CONTROL") if cpu_control else ("PASS_SCOPED_BACKEND_A9_CONT10_MECHANICS" if phase=="A" else "PASS_SCOPED_BACKEND_FRESH_CONT10_MECHANICS")
    except BaseException as error:
        result["error"]={"type":type(error).__name__,"message":str(error)}
    finally:
        if adapter is not None:
            try: adapter.close()
            except BaseException as error:
                result["status"]="FAILED_SCOPED_BACKEND_QUALIFICATION";result["cleanup_error"]={"type":type(error).__name__,"message":str(error)}
        result["primitive_operations"]=collector.snapshot();result["protocol_operations"]=deepcopy(counts);result["dose"]=dose.snapshot()
    result["sha256"]=digest(result)
    return result
