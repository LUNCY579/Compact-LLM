"""Injected nine-update protocol; CPU proof only, never a launch permission.

Native input must be a lazy factory, including model loading and construction.
Its caller must separately complete authorization, owned-Job containment and
current resource preflight before framework import. A fresh bound admission is
mandatory; this module has no loader, CLI, framework import or default permit.

Backend primitives remain journaled by backend.call; wrapper calls are counted
separately. preserve(event) supplies a new checkpoint path or publishes an exact
JSON payload with flush/fsync and returns {path,bytes,sha256,payload_sha256,durable}.
Self-recording terminal/promotion costs are projected, then reconciled against
actual returned operations before a resumable descriptor can be emitted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import study_schedule as schedule

PACKAGE = Path(__file__).resolve().parent
FAKE_SCOPE = "CPU_PROTOCOL_FAKE"
NATIVE_SCOPE = "UNQUALIFIED_NATIVE_IMPLEMENTATION"
KIND = "SYNTH_STUDY_CHUNK_PROTOCOL_V1"
BACKEND_KIND = "SYNTH_STUDY_TORCH_BOUNDARY_V1"
BINDING_FIELDS = {"completed_updates", "consumed_occurrences", "input_tokens", "supervised_tokens",
                  "consumed_order_sha256", "next_update_ordinal", "learning_rate"}
METHODS = ("reset_gradients", "forward", "completion_loss", "backward", "finite_clip_gradients",
           "optimizer_step", "save_checkpoint", "restore_checkpoint", "primitive_operation_snapshot")
OPERATIONS = {"gradient_reset":27, "forward":540, "completion_loss":540, "backward":540,
              "finite_gradient_clip":27, "optimizer_step":27, "checkpoint_save":3,
              "checkpoint_reload":2, "checkpoint_publish":3, "terminal_costs_publish":3,
              "checkpoint_promote":3}
FRAMEWORKS = {"torch", "transformers", "jax", "tensorflow", "flax"}
MAX_BYTES = 32 * 1024**2


def require(value, message):
    if not value: raise ValueError(message)


def finite(value, label):
    require(type(value) in (int,float) and math.isfinite(value) and value >= 0, "finite nonnegative " + label)
    return float(value)


def _hash(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}",value), "SHA256 required")
    return value


def _framework_free():
    require(not any(name.split(".",1)[0] in FRAMEWORKS for name in sys.modules), "no framework before owned native admission")


def recipe(registration, arm, scope):
    require(arm in ("M1","M2") and scope in (FAKE_SCOPE,NATIVE_SCOPE), "adapted arm and explicit protocol scope")
    value = {"schema_version":1, "registration_sha256":registration["sha256"], "arm":arm, "seed":101,
             "cohort_sha256":registration["arms"][arm]["occurrence_order_sha256"],
             "source_pins":deepcopy(registration["source_and_decision_pins"]), "execution_scope":scope,
             "chunk_source_sha256":schedule.filehash(__file__),
             "backend_source_sha256":schedule.filehash(PACKAGE/"study_torch_backend.py")}
    value["sha256"] = schedule.digest(value)
    return value


def binding(registration, arm, completed):
    require(type(completed) is int and completed in (9,18,27), "complete nine-update boundary")
    updates=registration["arms"][arm]["updates"][:completed]
    ids=[identity for update in updates for identity in update["occurrence_row_ids"]]
    return {"completed_updates":completed, "consumed_occurrences":completed*20,
            "input_tokens":sum(update["input_tokens"] for update in updates),
            "supervised_tokens":sum(update["supervised_tokens"] for update in updates),
            "consumed_order_sha256":schedule.digest(ids), "next_update_ordinal":completed+1,
            "learning_rate":schedule.learning_rates()[completed-1]}


def _counts(value):
    require(type(value) is dict, "primitive operation counts")
    for name,item in value.items():
        require(type(name) is str and type(item) is dict and set(item)=={"attempted","returned","failed","pending"}
                and all(type(n) is int and n>=0 for n in item.values())
                and item["attempted"]==item["returned"]+item["failed"]+item["pending"], "conserved primitive counts")
    return deepcopy(value)


def _merge(*inventories):
    value={}
    for inventory in inventories:
        for name,item in _counts(inventory).items():
            result=value.setdefault(name,{key:0 for key in item})
            for key,n in item.items(): result[key]+=n
    return value


def validate_costs(value, *, successful=False):
    require(type(value) is dict and set(value)=={"schema_version","scope","protocol","positions","primitive_operations"}
            and type(value["schema_version"]) is int and value["schema_version"]==1 and value["scope"]==KIND, "exact terminal cost schema")
    require(set(value["protocol"])==set(OPERATIONS), "protocol cost closure")
    _counts(value["protocol"]); _counts(value["primitive_operations"])
    require(all(value["protocol"][name]["attempted"]<=limit for name,limit in OPERATIONS.items()), "protocol budget")
    positions=value["positions"]
    require(type(positions) is dict and set(positions)=={"input_attempted","input_returned","supervised_attempted","supervised_returned"}
            and all(type(n) is int and n>=0 for n in positions.values())
            and positions["input_returned"]<=positions["input_attempted"]<=131072
            and positions["supervised_returned"]<=positions["supervised_attempted"], "position cost conservation")
    if successful:
        require(all(not item["failed"] and not item["pending"] for inventory in
                    (value["protocol"],value["primitive_operations"]) for item in inventory.values()), "failed/pending costs forbid resume")
    return deepcopy(value)


class _PrimitiveCollector:
    """Collect every shared call, including loading before backend construction."""
    def __init__(self, call, guard): self.external,self.guard,self.counts=call,guard,{}

    def snapshot(self): return deepcopy(self.counts)

    def __call__(self, operation, function):
        require(type(operation) is str and operation and callable(function),"named primitive boundary")
        counts=self.counts.setdefault(operation,{key:0 for key in ("attempted","returned","failed","pending")})
        counts["attempted"]+=1;counts["pending"]+=1
        executed=False
        def actual():
            nonlocal executed
            require(not executed,"shared journal must invoke primitive once")
            executed=True
            try: value=function()
            except BaseException:
                counts["pending"]-=1;counts["failed"]+=1
                raise
            counts["pending"]-=1;counts["returned"]+=1
            return value
        self.guard("chunk_primitive_before_"+operation)
        value=self.external(operation,actual)
        require(executed and counts["pending"]==0,"shared journal must return executed primitive")
        self.guard("chunk_primitive_after_"+operation)
        return value


class _Ledger:
    def __init__(self, backend, call, guard, previous=None):
        self.backend,self.call,self.guard=backend,call,guard
        self.protocol={name:{key:0 for key in ("attempted","returned","failed","pending")} for name in OPERATIONS}
        self.positions={key:0 for key in ("input_attempted","input_returned","supervised_attempted","supervised_returned")}
        self.primitives,self.previous_primitives={},{}
        if previous is not None:
            validate_costs(previous,successful=True)
            self.protocol,self.positions,self.previous_primitives=deepcopy(previous["protocol"]),deepcopy(previous["positions"]),deepcopy(previous["primitive_operations"])

    def snapshot(self):
        current=self.call.snapshot()
        if self.backend is not None:
            # Backend counters are a subset witness, never added a second time.
            for name,item in _counts(self.backend.primitive_operation_snapshot()).items():
                require(name in current and all(current[name][key]>=n for key,n in item.items()),
                        "backend bypassed shared primitive collector")
        return {"schema_version":1,"scope":KIND,"protocol":deepcopy(self.protocol),"positions":deepcopy(self.positions),
                "primitive_operations":_merge(self.previous_primitives,current)}

    def invoke(self, operation, function, *, row=None):
        counts=self.protocol[operation]
        require(counts["attempted"]<OPERATIONS[operation], "operation budget "+operation)
        counts["attempted"]+=1;counts["pending"]+=1
        if row is not None:
            self.positions["input_attempted"]+=len(row["input_ids"])
            self.positions["supervised_attempted"]+=len(row["completion_ids"])
        self.guard("chunk_before_"+operation)
        try: value=function()
        except BaseException:
            counts["pending"]-=1;counts["failed"]+=1
            raise
        counts["pending"]-=1;counts["returned"]+=1
        if row is not None:
            self.positions["input_returned"]+=len(row["input_ids"])
            self.positions["supervised_returned"]+=len(row["completion_ids"])
        self.guard("chunk_after_"+operation)
        return value

    def primitive(self, operation, function):
        counts=self.primitives.setdefault(operation,{key:0 for key in ("attempted","returned","failed","pending")})
        counts["attempted"]+=1;counts["pending"]+=1
        executed=False
        def actual():
            nonlocal executed
            require(not executed,"journal must invoke primitive once")
            executed=True
            try: value=function()
            except BaseException:
                counts["pending"]-=1;counts["failed"]+=1
                raise
            counts["pending"]-=1;counts["returned"]+=1
            return value
        value=self.call(operation,actual)
        require(executed and counts["pending"]==0,"journal did not return executed primitive")
        return value


def _read_bytes(path, size, expected):
    path=Path(path)
    require(path.is_absolute() and not path.is_symlink() and "state1" not in str(path).lower(), "absolute study checkpoint; STATE1 forbidden")
    require(type(size) is int and 0<size<=MAX_BYTES,"bounded descriptor bytes")
    _hash(expected)
    with path.open("rb") as stream: raw=stream.read(size+1)
    require(len(raw)==size and hashlib.sha256(raw).hexdigest()==expected,"published bytes/hash changed")
    return raw


def _verify_publication(publication, payload):
    require(type(publication) is dict and set(publication)=={"path","bytes","sha256","payload_sha256","durable"}
            and publication["durable"] is True and publication["payload_sha256"]==schedule.digest(payload), "durable exact publication schema")
    raw=_read_bytes(publication["path"],publication["bytes"],publication["sha256"])
    parsed=json.loads(raw.decode(),object_pairs_hook=schedule._pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite publication")))
    require(schedule.canonical(parsed)==schedule.canonical(payload),"published payload changed")
    return deepcopy(publication)


def _descriptor(descriptor, expected, current_recipe):
    require(type(descriptor) is dict and set(descriptor)=={"schema_version","kind","path","bytes","sha256","metadata","metadata_sha256"}
            and type(descriptor["schema_version"]) is int and descriptor["schema_version"]==1 and descriptor["kind"]==BACKEND_KIND,"exact study backend descriptor; STATE1 forbidden")
    metadata=descriptor["metadata"]
    require(type(metadata) is dict and descriptor["metadata_sha256"]==schedule.digest(metadata)
            and schedule.canonical(metadata.get("binding"))==schedule.canonical(expected) and set(expected)==BINDING_FIELDS,"backend metadata/cursor checksum")
    for name,want in (("arm",current_recipe["arm"]),("seed",101),("scope",current_recipe["execution_scope"]),
                      ("recipe_sha256",current_recipe["sha256"]),("cohort_sha256",current_recipe["cohort_sha256"]),
                      ("backend_source_sha256",current_recipe["backend_source_sha256"])):
        require(metadata.get(name)==want,"backend arm/source/recipe/cohort binding: "+name)
    require(all(metadata.get(name) is False for name in ("native_launch_authorized","training_authorized","native_resume_qualified","qualification_STATE1_used")),"backend permission/warmstart claim")
    _hash(metadata.get("fresh_start_attestation_sha256"))
    _read_bytes(descriptor["path"],descriptor["bytes"],descriptor["sha256"])
    return deepcopy(descriptor)


def _project(costs, operation):
    projected=deepcopy(costs)
    item=projected["protocol"][operation];item["attempted"]+=1;item["returned"]+=1
    for suffix in ("write","verify"):
        name="chunk_"+operation+"_"+suffix
        item=projected["primitive_operations"].setdefault(name,{key:0 for key in ("attempted","returned","failed","pending")})
        item["attempted"]+=1;item["returned"]+=1
    validate_costs(projected,successful=True)
    return projected


def _complete_costs(costs, expected, chunk, *, posted=False):
    validate_costs(costs,successful=True)
    counts={"gradient_reset":chunk*9,"forward":chunk*180,"completion_loss":chunk*180,"backward":chunk*180,
            "finite_gradient_clip":chunk*9,"optimizer_step":chunk*9,"checkpoint_save":chunk,
            "checkpoint_reload":chunk-1,"checkpoint_publish":chunk-int(posted),
            "terminal_costs_publish":chunk-int(posted),"checkpoint_promote":chunk-int(posted)}
    require(all(costs["protocol"][name]=={"attempted":n,"returned":n,"failed":0,"pending":0}
                for name,n in counts.items()),"complete returned chunk protocol; partial costs forbidden")
    require(costs["positions"]=={"input_attempted":expected["input_tokens"],"input_returned":expected["input_tokens"],
                                  "supervised_attempted":expected["supervised_tokens"],"supervised_returned":expected["supervised_tokens"]},
            "exact global dose costs")


def _publish(ledger, preserve, operation, payload):
    def write():
        publication=ledger.primitive("chunk_"+operation+"_write",lambda:preserve({"event":operation,"payload":deepcopy(payload)}))
        return ledger.primitive("chunk_"+operation+"_verify",lambda:_verify_publication(publication,payload))
    return ledger.invoke(operation,write)


def _validate_resume(resume, registration, arm, chunk, current_recipe, latest):
    require(type(resume) is dict and set(resume)=={"schema_version","kind","status","envelope","publication","durable_terminal","terminal_publication","promotion","promotion_publication","terminal_costs","sha256"}
            and resume["kind"]==KIND and resume["status"]=="SUCCESSFUL_STUDY_CHUNK_PROTOCOL_ONLY"
            and resume["sha256"]==schedule.digest({k:v for k,v in resume.items() if k!="sha256"}),"successful prior chunk descriptor only; STATE1 forbidden")
    envelope=resume["envelope"]
    require(type(envelope) is dict and set(envelope)=={"schema_version","kind","arm","seed","chunk_ordinal","recipe","binding","backend_descriptor","posted_save_costs","native_resume_qualified","training_authorized","sha256"}
            and envelope["kind"]==KIND and envelope["seed"]==101 and envelope["native_resume_qualified"] is False and envelope["training_authorized"] is False
            and envelope["sha256"]==schedule.digest({k:v for k,v in envelope.items() if k!="sha256"})
            and envelope["recipe"]==current_recipe and envelope["arm"]==arm and envelope["chunk_ordinal"]==chunk-1,
            "resume same previous chunk/arm/source/recipe/cohort")
    require(envelope["binding"]==binding(registration,arm,(chunk-1)*9),"resume full cursor/order/dose/globalLR")
    _descriptor(envelope["backend_descriptor"],envelope["binding"],current_recipe)
    _verify_publication(resume["publication"],envelope)
    terminal=resume["durable_terminal"]
    require(terminal["envelope_sha256"]==envelope["sha256"] and terminal["costs_sha256"]==schedule.digest(terminal["costs"]),"durable terminal binding")
    require(terminal["costs"]==_project(_project(envelope["posted_save_costs"],"checkpoint_publish"),"terminal_costs_publish"),
            "posted-save to durable terminal primitive cost projection chain")
    _verify_publication(resume["terminal_publication"],terminal)
    promotion=resume["promotion"]
    require(promotion["envelope_sha256"]==envelope["sha256"] and promotion["durable_terminal_sha256"]==schedule.digest(terminal)
            and promotion["terminal_costs"]==resume["terminal_costs"],"promotion exact terminal cost binding")
    _verify_publication(resume["promotion_publication"],promotion)
    validate_costs(latest,successful=True)
    require(latest==resume["terminal_costs"]==_project(terminal["costs"],"checkpoint_promote"),"latest terminal costs required; rollback forbidden")
    _complete_costs(latest,envelope["binding"],chunk-1)
    _complete_costs(envelope["posted_save_costs"],envelope["binding"],chunk-1,posted=True)
    return envelope


def run_chunk(registration, arm, chunk, backend, row_get, call, guard, preserve, *,
              resume_descriptor=None, latest_terminal_costs=None, admission=None):
    """Run exactly nine full updates; injected native factory needs fresh admission.

    A tagged native factory accepts call/guard/arm/seed/recipe_sha256/cohort_sha256
    keyword arguments and includes model loading. CPU_PROTOCOL_FAKE is an eager
    stdlib fixture. No already loaded native model/backend is accepted.
    """
    _framework_free()
    frozen=deepcopy(registration);schedule.validate_registration(frozen)
    require(arm in ("M1","M2") and type(chunk) is int and chunk in (1,2,3),"adapted arm and chunk1/2/3")
    require(all(callable(value) for value in (row_get,call,guard,preserve)),"explicit reader/journal/guard/preservation callbacks")
    scope=getattr(backend,"execution_scope",None)
    current_recipe=recipe(frozen,arm,scope)
    if scope==NATIVE_SCOPE: require(callable(backend),"eager native backend forbidden; admission must precede model loading/constructor")
    start,stop=(chunk-1)*9,chunk*9
    previous=None
    if chunk==1:
        require(resume_descriptor is None and latest_terminal_costs is None,"fresh chunk cannot warmstart/reset consumed costs")
    else:
        previous=_validate_resume(deepcopy(resume_descriptor),frozen,arm,chunk,current_recipe,deepcopy(latest_terminal_costs))
    # Read every occurrence, including repeats, before construction/restore/forward.
    _,_,pools,_=schedule._inputs()
    rows=[]
    for update in frozen["arms"][arm]["updates"][start:stop]:
        for identity in update["occurrence_row_ids"]:
            row=deepcopy(row_get(identity));schedule.validate_token_row(row)
            require(row==pools[arm][identity],"reader mutation or row differs from frozen pool")
            rows.append(row)
    require(len(rows)==180,"all180 whole occurrences required")
    _framework_free()
    collector=_PrimitiveCollector(call,guard)
    if scope==NATIVE_SCOPE:
        require(callable(admission),"fresh bound native admission callback required before factory")
        request={"schema_version":1,"stage":"study_chunk_before_model_load_and_backend_constructor","arm":arm,"chunk_ordinal":chunk,
                 "seed":101,"recipe_sha256":current_recipe["sha256"],"cohort_sha256":current_recipe["cohort_sha256"],
                 "registration_sha256":frozen["sha256"],"source_binding_sha256":schedule.digest(current_recipe),"execution_scope":scope}
        evidence=admission(deepcopy(request));now=time.monotonic()
        require(type(evidence) is dict and set(evidence)=={"binding","authorized","owned_supervision_verified","model_load_and_constructor_in_scope","observed_monotonic","expires_monotonic"}
                and evidence["binding"]==request and all(evidence[key] is True for key in ("authorized","owned_supervision_verified","model_load_and_constructor_in_scope")),"exact authorized native admission stage/source binding")
        observed,expires=evidence["observed_monotonic"],evidence["expires_monotonic"]
        require(type(observed) in (int,float) and type(expires) in (int,float) and math.isfinite(observed) and math.isfinite(expires)
                and observed<=now<=expires<=observed+.2,"fresh admission within unchanged200ms CPU window")
        _framework_free()
    if scope==FAKE_SCOPE:
        require(getattr(backend,"call",None) is call and getattr(backend,"guard",None) is guard,"bound CPU fake journal/guard")
        require(all(getattr(backend,name,None)==want for name,want in (("arm",arm),("seed",101),("recipe_sha256",current_recipe["sha256"]),("cohort_sha256",current_recipe["cohort_sha256"]))),"backend fresh source/arm/recipe identity")
        backend.call=collector
    ledger=_Ledger(None,collector,guard,latest_terminal_costs)
    result={"status":"RUNNING_PROTOCOL_ONLY","scope":scope,"arm":arm,"chunk_ordinal":chunk,"recipe":current_recipe,
            "completed_updates":start,"update_summaries":[],"resume_descriptor":None,
            "native_resume_status":"NATIVE_RESUME_UNQUALIFIED","study_status":"STUDY_UNAUTHORIZED","study_unauthorized":True,
            "native_launch_authorized":False,"training_authorized":False,"native_resume_qualified":False,"automatic_resume_authorized":False}
    result["native_admission"]=deepcopy(evidence) if scope==NATIVE_SCOPE else None
    result["prevalidated_whole_occurrences"]=len(rows)
    try:
        if scope==NATIVE_SCOPE:
            guard("chunk_before_native_model_load_and_constructor")
            backend=backend(call=collector,guard=guard,arm=arm,seed=101,recipe_sha256=current_recipe["sha256"],cohort_sha256=current_recipe["cohort_sha256"])
        require(getattr(backend,"execution_scope",None)==scope and all(callable(getattr(backend,name,None)) for name in METHODS),"exact actual backend protocol")
        require(all(getattr(backend,name,None)==want for name,want in (("arm",arm),("seed",101),("recipe_sha256",current_recipe["sha256"]),("cohort_sha256",current_recipe["cohort_sha256"]))),"backend fresh source/arm/recipe identity")
        require(getattr(backend,"call",None) is collector and getattr(backend,"guard",None) is guard,"one shared primitive collector/resource guard; wrapper double counting forbidden")
        ledger.backend=backend
        attestation=backend.fresh_start_attestation
        require(type(attestation) is dict and attestation.get("qualification_STATE1_used") is False and attestation.get("optimizer_state_empty") is True
                and attestation.get("sha256")==schedule.digest({k:v for k,v in attestation.items() if k!="sha256"}),"fresh seeded zeroB start; STATE1 forbidden")
        require(getattr(backend,"completed_updates",None)==0,"fresh backend required; continuation comes only from previous successful chunk")
        if previous is not None:
            ledger.invoke("checkpoint_reload",lambda:backend.restore_checkpoint(deepcopy(previous["backend_descriptor"]),deepcopy(previous["binding"])))
        for offset,update in enumerate(frozen["arms"][arm]["updates"][start:stop]):
            ordinal=update["update_ordinal"]
            ledger.invoke("gradient_reset",lambda:backend.reset_gradients(ordinal))
            losses=[]
            for frozen_row in rows[offset*20:(offset+1)*20]:
                row=deepcopy(frozen_row)
                output=ledger.invoke("forward",lambda:backend.forward(row),row=frozen_row)
                require(row==frozen_row,"backend forward mutated frozen whole row")
                loss=ledger.invoke("completion_loss",lambda:backend.completion_loss(output,row))
                # CE was scalarized inside the backend's already journaled check.
                losses.append(finite(backend.last_loss_check["ce"],"checked sequence loss"))
                ledger.invoke("backward",lambda:backend.backward(loss,scale=.05))
                # Release full logits and the consumed graph before the next
                # forward or checkpoint; retain only the checked scalar loss.
                del output, loss
            norm=finite(ledger.invoke("finite_gradient_clip",lambda:backend.finite_clip_gradients(max_norm=1.0)),"preclip norm")
            ledger.invoke("optimizer_step",lambda:backend.optimizer_step(lr=update["learning_rate"],update_ordinal=ordinal))
            result["completed_updates"]=ordinal
            result["update_summaries"].append({"update_ordinal":ordinal,"learning_rate":update["learning_rate"],"mean_per_sequence_loss":sum(losses)/20,"gradient_norm":norm})
        end=binding(frozen,arm,stop)
        path=preserve({"event":"checkpoint_path","arm":arm,"chunk_ordinal":chunk,"recipe_sha256":current_recipe["sha256"]})
        require(type(path) is str and Path(path).is_absolute() and not Path(path).exists() and not Path(path).is_symlink()
                and "state1" not in path.lower(),"new write-once study checkpoint path")
        descriptor=ledger.invoke("checkpoint_save",lambda:backend.save_checkpoint(path,deepcopy(end)))
        ledger.primitive("chunk_saved_checkpoint_verify",lambda:_descriptor(descriptor,end,current_recipe))
        require(descriptor["metadata"]["fresh_start_attestation_sha256"]==attestation["sha256"],"fresh start attestation changed")
        posted=ledger.snapshot();validate_costs(posted,successful=True)
        _complete_costs(posted,end,chunk,posted=True)
        envelope={"schema_version":1,"kind":KIND,"arm":arm,"seed":101,"chunk_ordinal":chunk,"recipe":current_recipe,
                  "binding":end,"backend_descriptor":descriptor,"posted_save_costs":posted,"native_resume_qualified":False,"training_authorized":False}
        envelope["sha256"]=schedule.digest(envelope)
        publication=_publish(ledger,preserve,"checkpoint_publish",envelope)
        terminal_costs=_project(ledger.snapshot(),"terminal_costs_publish")
        terminal={"envelope_sha256":envelope["sha256"],"costs":terminal_costs,"costs_sha256":schedule.digest(terminal_costs)}
        terminal_publication=_publish(ledger,preserve,"terminal_costs_publish",terminal)
        require(ledger.snapshot()==terminal_costs,"terminal projected costs differ from actual returns")
        promoted_costs=_project(terminal_costs,"checkpoint_promote")
        promotion={"envelope_sha256":envelope["sha256"],"durable_terminal_sha256":schedule.digest(terminal),"terminal_costs":promoted_costs}
        promotion_publication=_publish(ledger,preserve,"checkpoint_promote",promotion)
        require(ledger.snapshot()==promoted_costs,"promotion projected costs differ from actual returns")
        _complete_costs(promoted_costs,end,chunk)
        resume={"schema_version":1,"kind":KIND,"status":"SUCCESSFUL_STUDY_CHUNK_PROTOCOL_ONLY","envelope":envelope,"publication":publication,
                "durable_terminal":terminal,"terminal_publication":terminal_publication,"promotion":promotion,"promotion_publication":promotion_publication,"terminal_costs":promoted_costs}
        resume["sha256"]=schedule.digest(resume)
        result["resume_descriptor"]=resume
        result["status"]="COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED"
    except BaseException as error:
        result["status"]="FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED"
        result["error_type"],result["error"]=type(error).__name__,str(error)
    finally:
        result["costs"]=ledger.snapshot()
    return result
