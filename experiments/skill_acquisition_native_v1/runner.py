"""Prospective admitted native S1 runner. Importing this module imports no ML runtime.

The CLI accepts only a separately authorized production packet with completed
S0 prerequisites, native qualification and measured budgets. This source grants
no execution. Pure orchestration spies are accounting tests, never qualification.
One invocation executes one named arm; later H1 decisions remain external.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
from importlib.machinery import NamespaceLoader
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import uuid

from . import data, journal, plan, protocol, update
from .ce_operation_contracts import FILE_HASH_LEAF, MODEL_FINGERPRINT, OPAQUE_INPUT_FRAME

SCHEMA = "SKILL-ACQUISITION-NATIVE-RESULT-1"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGES = ("skill_acquisition_native_v1", "maxrl_comp002_pretraining_v1",
                   "maxrl_composable_skills_v2", "maxrl_verifiable_data_v1")
IO_KINDS = frozenset(("module_import", "model_load", "tokenizer_load", "dataset_read", "canary",
    "checkpoint_construct", "checkpoint_adapter_io", "checkpoint_save", "checkpoint_restore",
    "donor_reference", "donor_write", "donor_roundtrip", "code_verification", "receipt_write"))
PROFILE_FIELDS = ("read_bytes_upper", "write_bytes_upper", "identity_hash_attempts_upper", "identity_hash_bytes_upper")
HOST_STAGES = IO_KINDS | frozenset(("input_io","process_rng_seed","configured_runtime","inventory_configuration",
    "fingerprint","score","loss_forward","backward","gradient_validation","optimizer","rollout","donor_chunk",
    "constructor","metadata_write","journal_write","save_validate","rolling_retirement","payload_write","verify",
    "restore_copy","cleanup","wall_watchdog"))


def require(value, message):
    if not value:
        raise ValueError(message)


class SetupObserver:
    """Bounded bootstrap observations; persist and charge before framework work.

    Admission cannot know its plan/caps until the signed packet references have
    been checked. These records explicitly precede the durable plan ledger.
    """
    def __init__(self):
        self.events = []
        self.bytes = 0

    def __call__(self, kind, phase, operation_id, **metadata):
        require(len(self.events) < 20000, "bootstrap operation record bound")
        record = {"kind":kind,"phase":phase,"operation_id":operation_id,"metadata":metadata}
        require(len(journal.canonical(record)) < 16384, "bootstrap metadata byte bound")
        self.bytes+=len(journal.canonical(record))
        require(self.bytes<=1024**2,"bounded bootstrap evidence cannot exceed1MiB")
        self.events.append(record)


def observed_file(path, observer, maximum=None):
    """One <=1MiB streamed file hash, with reservation before the first read."""
    path = Path(path).resolve(strict=True)
    size = path.stat().st_size
    require(maximum is None or size <= maximum, "input file byte bound")
    identifier, begin, count = uuid.uuid4().hex, time.perf_counter(), 0
    fields = dict(path=str(path),read_bytes_upper=size+1,write_bytes_upper=0,
                  identity_hash_attempts_upper=1,identity_hash_bytes_upper=size+1,wall_contract=FILE_HASH_LEAF)
    observer("input_io","started",operation_id=identifier,**fields)
    h=hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block=stream.read(min(1024**2,size+1-count))
                if not block:break
                count+=len(block)
                require(count<=size,"input file grew during hash")
                h.update(block)
        require(count==size,"input file shrank during hash")
    except BaseException:
        observer("input_io","failed",operation_id=identifier,known_bytes_read=count,
                 wall_s=time.perf_counter()-begin,wall_contract=FILE_HASH_LEAF)
        raise
    observer("input_io","returned",operation_id=identifier,bytes_read=count,
             wall_s=time.perf_counter()-begin,wall_contract=FILE_HASH_LEAF)
    return {"path":str(path),"bytes":size,"sha256":h.hexdigest()}


def bounded_bytes(path,maximum):
    pieces,count=[],0
    with Path(path).open("rb") as stream:
        while True:
            block=stream.read(min(1024**2,maximum+1-count))
            if not block:break
            count+=len(block)
            require(count<=maximum,"bounded JSON input grew")
            pieces.append(block)
    return b"".join(pieces)


def pinned_helper(path,load,observer):
    """Price the source read/hash performed by s0_runtime.load_pinned before lease."""
    path=Path(path).resolve(strict=True)
    size=path.stat().st_size
    require(size<=1024**2,"pinned resource helper source byte bound")
    identifier=uuid.uuid4().hex
    observer("input_io","started",operation_id=identifier,path=str(path),read_bytes_upper=size,
             write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=size,wall_contract=OPAQUE_INPUT_FRAME)
    try:
        module=load()
        require(Path(module.__file__).resolve()==path,"pinned helper origin drift")
    except BaseException:
        observer("input_io","failed",operation_id=identifier,known_bytes_read=size,wall_contract=OPAQUE_INPUT_FRAME)
        raise
    observer("input_io","returned",operation_id=identifier,bytes_read=size,
             scope="retained source-read/hash upper even when pinned helper cache already exists",wall_contract=OPAQUE_INPUT_FRAME)
    return module


def code_bundle(observer, root=ROOT):
    result={}
    for package in SOURCE_PACKAGES:
        directory=Path(root)/"experiments"/package
        require(directory.is_dir(),"source package missing: "+package)
        for path in sorted(directory.rglob("*.py")):
            name=path.relative_to(root).as_posix()
            result[name]=observed_file(path,observer,4*1024**2)["sha256"]
    require("experiments/skill_acquisition_native_v1/runner.py" in result,"runner source absent")
    # The root namespace can affect package resolution even when it is empty.
    initializer=Path(root)/"experiments"/"__init__.py"
    if initializer.exists():
        result["experiments/__init__.py"]=observed_file(initializer,observer,1024**2)["sha256"]
    return result


def validate_profiles(io):
    profiles=io.get("operation_bounds")
    require(type(profiles) is dict and set(profiles)>=IO_KINDS,
            "measured per-operation IO reservation profiles required before framework")
    for kind, fields in profiles.items():
        require(type(kind) is str and type(fields) is dict and set(fields)==set(PROFILE_FIELDS)
                and all(type(fields[k]) is int and fields[k]>=0 for k in PROFILE_FIELDS),
                "explicit measured IO profile fields/types: "+str(kind))
        require((fields["identity_hash_attempts_upper"]==0)==(fields["identity_hash_bytes_upper"]==0),
                "hash profile count/bytes incoherent")
    require(profiles["model_load"]["identity_hash_attempts_upper"]>=3
            and profiles["model_load"]["read_bytes_upper"]>0,"model load implicit verification work must be priced")
    return profiles


def model_identity_lower_bound(weights,warm,observer):
    """Analytical payload lower bound; extra measured helper reads remain required."""
    def payload(path,dtype):
        path=Path(path).resolve(strict=True)
        identifier=uuid.uuid4().hex
        upper=min(path.stat().st_size,1024**2+8)
        observer("input_io","started",operation_id=identifier,path=str(path),read_bytes_upper=upper,
                 write_bytes_upper=0,identity_hash_attempts_upper=0,identity_hash_bytes_upper=0,wall_contract=OPAQUE_INPUT_FRAME)
        try:
            reader=update._TensorFile(path)
            try:
                require(all(e["dtype"]==dtype for e in reader.header.values()),"bound model/delta dtype")
                amount=sum(e["data_offsets"][1]-e["data_offsets"][0] for e in reader.header.values())
                read=reader.start
            finally:reader.close()
        except BaseException:
            observer("input_io","failed",operation_id=identifier,known_bytes_read=upper,wall_contract=OPAQUE_INPUT_FRAME)
            raise
        observer("input_io","returned",operation_id=identifier,bytes_read=read,wall_contract=OPAQUE_INPUT_FRAME)
        return amount,path.stat().st_size
    S,file_bytes=payload(weights,"BF16")
    W,_=payload(warm,"F32")
    passes=[{"name":"model_BF16_file_pass","bytes_lower":file_bytes},
            {"name":"cold_FP32_state_pass","bytes_lower":2*S},
            {"name":"warmed_FP32_state_pass","bytes_lower":2*S},
            {"name":"stored_BF16_tensor_verification_pass","bytes_lower":S},
            {"name":"warm_delta_data_pass","bytes_lower":W}]
    return {"passes":passes,"identity_hash_attempts_lower":len(passes),
            "identity_hash_bytes_lower":sum(item["bytes_lower"] for item in passes),
            "scope":"analytical lower bound; metadata/config/hash and extra assertion reads require measured profile"}


def metadata(packet, observer):
    # Fail closed before source/evidence IO for a plainly unapproved packet.
    protocol.require(type(packet) is dict and packet.get("approved") is True
                     and packet.get("training_authorized") is True,"packet is unapproved; no framework import")
    sources=code_bundle(observer)
    verify_module_origins(sources)
    protocol.admit_packet(packet,code_sha256=sources,io_observer=observer)
    require(Path(packet["source_checkout"]).resolve()==ROOT,"packet does not name this running source checkout")
    context=protocol.admitted_context(packet,io_observer=observer)
    context["profiles"]=validate_profiles(context["io"])
    transients=context["io"].get("host_transient_upper_bytes")
    require(type(transients) is dict and set(transients)>=HOST_STAGES
            and all(type(value) is int and value>=0 for value in transients.values()),
            "bound per-operation host RAM transients required before framework")
    context["host_transients"]=transients
    assets=context["prerequisites"]["assets"]
    lower=model_identity_lower_bound(Path(assets["model_path"])/"model.safetensors",packet["anchor"]["delta_path"],observer)
    profile=context["profiles"]["model_load"]
    require(profile["identity_hash_attempts_upper"]>=lower["identity_hash_attempts_lower"]
            and profile["identity_hash_bytes_upper"]>=lower["identity_hash_bytes_lower"],"model-load measured hash profile below analytical work lower bound")
    context["model_load_identity_lower_bound"]=lower
    context["code"]=sources
    return context


class Operations:
    """Resource check plus measured reservation before every native physical start."""
    def __init__(self, ledger, profiles, gate):
        self.ledger,self.profiles,self.gate=ledger,profiles,gate

    @property
    def generation(self):return self.ledger.generation
    def snapshot(self):return self.ledger.snapshot()

    def __call__(self,kind,phase,operation_id,**metadata):
        fields=dict(metadata)
        if phase=="started":
            self.gate(kind)
            if kind in IO_KINDS:
                profile=self.profiles[kind]
                for stem in ("identity_hash_attempts","identity_hash_bytes"):
                    key=stem+"_lower"
                    if key in fields:
                        require(type(fields[key]) is int and fields[key]>=0
                                and fields[key]<=profile[stem+"_upper"],"opaque helper hash profile below analytical lower: "+kind+"/"+key)
                for key in PROFILE_FIELDS:
                    require(fields.get(key,0)<=profile[key],"declared operation exceeds measured IO upper: "+kind+"/"+key)
                    fields[key]=profile[key]
            fields.setdefault("generation",self.ledger.generation)
            fields.setdefault("symbolic_snapshot",self.ledger.symbolic_snapshot)
            fields.setdefault("invocation_id",self.ledger.current_invocation)
        self.ledger.operation(kind,phase,operation_id=operation_id,**fields)

    operation=__call__

    @contextlib.contextmanager
    def span(self,kind,**metadata):
        identifier,begin=uuid.uuid4().hex,time.perf_counter()
        self(kind,"started",operation_id=identifier,**metadata)
        record={"operation_id":identifier}
        try:yield record
        except BaseException as error:
            self(kind,"failed",operation_id=identifier,wall_s=time.perf_counter()-begin,
                 error={"type":type(error).__name__,"message":str(error)[:1000]})
            raise
        self(kind,"returned",operation_id=identifier,wall_s=time.perf_counter()-begin,**{
             k:v for k,v in record.items() if k!="operation_id"})


class ResourceGate:
    def __init__(self, packet, runtime, host, output, ledger,transients):
        self.packet,self.runtime,self.host,self.output,self.ledger=packet,runtime,host,Path(output),ledger
        self.transients=transients
        self.framework=None

    def _disk(self):
        limits=self.packet["limits"]
        used=sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file()) if self.output.exists() else 0
        parent=self.output
        while not parent.exists():parent=parent.parent
        free=shutil.disk_usage(parent).free
        remaining=max(0,limits["active_storage_reserve_bytes"]-used)
        require(free>=limits["disk_floor_bytes"]+remaining,"native disk floor/remaining active reservation")
        return {"free_bytes":free,"owned_bytes":used,"remaining_reserve_bytes":remaining}

    def launch(self):
        self.runtime.host_require(self.packet["limits"]["host_floor_gib"],"S1 native launch",module=self.host)
        return self._disk()

    def __call__(self,stage):
        self.ledger.tick_wall()
        # This is the registered S0-Q1 unit guard; the measured launch floor is
        # checked separately before model allocation and is never weakened.
        require(stage in self.transients,"unpriced host transient: "+stage)
        self.runtime.host_guard(1.0+self.transients[stage]/1024**3,"S1 "+stage,module=self.host)
        disk=self._disk()
        if self.framework is not None:
            torch=self.framework
            free,total=torch.cuda.mem_get_info(0)
            reserved=torch.cuda.memory_reserved(0)
            require(free>=.5*1024**3 and reserved<=.95*.78*total,"native GPU headroom/reserved ceiling")
        return disk


class WallDeadline:
    """Hard wall watchdog. A killed process leaves pending starts as cost evidence.

    Kernel operations cannot safely unwind Python at an arbitrary CUDA point.
    The watchdog terminates this admitted worker at its remaining cumulative
    deadline; a documented same-boot resume must charge the interruption.
    """
    def __init__(self,ledger,timer_factory=threading.Timer,terminate=os._exit):
        self.ledger,self.timer_factory,self.terminate=ledger,timer_factory,terminate
        self.timer=None

    def __enter__(self):
        self.ledger.tick_wall()
        remaining=self.ledger.core.caps["wall_charged_s"]-self.ledger.core.counters["wall_charged_s"]
        require(remaining>0,"no cumulative wall allowance for worker")
        self.identifier=uuid.uuid4().hex
        deadline=self.ledger._wall_last+remaining
        self.ledger.operation("wall_watchdog","started",operation_id=self.identifier,remaining_seconds=remaining,
                              absolute_monotonic_deadline=deadline)
        delay=deadline-self.ledger.clock()
        require(delay>0,"wall deadline reached before watchdog activation")
        self.timer=self.timer_factory(delay,lambda:self.terminate(124))
        self.timer.daemon=True
        self.timer.start()
        return self

    def __exit__(self,typ,value,traceback):
        if self.timer is not None:self.timer.cancel()
        if not self.ledger.fatal:
            self.ledger.operation("wall_watchdog","returned",operation_id=self.identifier)


def checkpoint_due(C,U,previous_U,cadence=10):
    return C%cadence==0 or (previous_U==0 and U==1)


def execute_boundaries(planned,rendered,ledger,backend,operations,checkpoints,recipe,cadence=10):
    """Actual C/U orchestration; pure test spies must label themselves synthetic.

    A boundary collects all 16 independently seeded samples before invoking the
    update. The update returns after parity, finite scope and Adam/frozen proofs.
    No replay/refill or signal-conditioned exposure schedule is introduced.
    """
    while ledger.progress.C<len(planned.boundaries):
        C=ledger.progress.C
        boundary=planned.boundaries[C]
        policy=backend.fingerprint(operations,"before_collection")
        prompts=[rendered[item.row_id] for item in boundary.prompts]
        groups,attempts=[],[]
        for i,prompt in enumerate(prompts):
            group=[]
            for k in range(8):
                operations.gate("rollout")
                trajectory=planned.trajectory(C,i,k)
                attempt=ledger.rollout_start(trajectory,len(prompt["prompt_ids"]),policy)
                try:
                    sample=backend.sample(prompt,trajectory["sample_seed"],operations)
                    sample.update(rollout_id=attempt,prompt_id=prompt["id"],policy_id=policy)
                    require(sample["generated_tokens"]==len(sample["completion_ids"]),"sample token count drift")
                    update.validate_trajectory(sample["completion_ids"],sample["stop_reason"],sample["raw_step_logprobs"],
                                               recipe["eos_ids"],recipe["max_new_tokens"])
                    sample.update(backend.score(prompt["row"],sample,operations))
                except BaseException as error:
                    ledger.rollout_failed(attempt,error)
                    raise
                ledger.rollout_return(attempt,sample)
                group.append(sample);attempts.append(attempt)
            groups.append(group)
        current=dict(recipe,boundary_id=C,generation=ledger.generation,symbolic_snapshot=ledger.symbolic_snapshot,
                     start_policy_id=policy,trainable_names=backend.trainable_names)
        prior_U=ledger.progress.U
        record=backend.update(prompts,groups,current,operations)
        ledger.commit_boundary(attempts,record)
        if checkpoint_due(ledger.progress.C,ledger.progress.U,prior_U,cadence):
            backend.save(checkpoints,ledger,operations)
    require(ledger.progress.complete(),"final count/history completion failed")


class NativeBackend:
    """Delayed framework/model implementation, reachable only after strong admission."""
    def __init__(self,packet,context,runtime,operations,gate):
        self.packet,self.context,self.runtime,self.ops,self.gate=packet,context,runtime,operations,gate
        self.model=self.optimizer=self.tokenizer=None
        self.trainable_names=[]

    def initialize(self):
        with self.ops.span("module_import"):
            self.runtime.assert_runtime()
            import torch
            from experiments.maxrl_comp002_pretraining_v1 import s0_generate, s0_competence, s0_pilot
            self.torch,self.generate,self.competence,self.pilot=torch,s0_generate,s0_competence,s0_pilot
            verify_module_origins(self.context["code"])
        self.runtime.configure("cuda:0")
        self.gate.framework=self.torch
        with self.ops.span("process_rng_seed"):
            import numpy as np
            seed=self.context["planned"].seed
            random.seed(seed)
            np.random.seed(seed % 2**32)
            self.torch.manual_seed(seed)
            self.torch.cuda.manual_seed_all(seed)
            self.process_rng_seed={"run_seed":seed,"python":seed,"numpy":seed % 2**32,
                                   "torch_cpu":seed,"torch_cuda_all":seed,
                                   "restore_rule":"checkpoint replaces all process RNG after fresh initialization"}
        actual=self.runtime.decoding_preimage("cuda:0")
        protocol.validate_current_runtime(self.packet,actual,io_observer=self.ops)
        self.gate("configured_runtime")
        with self.ops.span("code_verification"):
            require(self.competence.live_bundle()==self.context["prerequisites"]["adopted"]["scoring_code_bundle_sha256"],
                    "loaded scorer bundle drift")
        require(Path(self.runtime.MODEL_DIR).resolve()==Path(self.context["prerequisites"]["assets"]["model_path"]).resolve(),
                "registered model path drift")
        require(observed_file(self.runtime.ASSET_RECEIPT,self.ops)["sha256"]==self.runtime.ASSET_RECEIPT_SHA,
                "pinned asset receipt drift")
        require(observed_file(self.runtime.TOKENIZER_RECEIPT,self.ops)["sha256"]==self.runtime.TOKENIZER_RECEIPT_SHA
                ==self.context["census"]["tokenizer_receipt_sha256"],"pinned tokenizer receipt drift")
        with self.ops.span("tokenizer_load"):
            self.tokenizer=self.runtime.load_tokenizer(self.runtime.MODEL_DIR,self.runtime.TOKENIZER_RECEIPT,
                                                       asset_receipt=self.runtime.ASSET_RECEIPT)
        with self.ops.span("model_load"):
            # The lease wait and framework imports can invalidate a previous
            # host snapshot. Recheck the adopted frozen launch floor here.
            self.gate.launch()
            a=self.packet["anchor"]
            self.model,self.identity=self.runtime.load_anchor(self.runtime.MODEL_DIR,"cuda:0",self.runtime.ASSET_RECEIPT,
               delta_path=a["delta_path"],delta_sha256=a["delta_sha256"],arm_sha256=a["arm_sha256"])
            require(self.identity["anchor"]==a and self.identity["state_fingerprint"]==a["arm_sha256"],"rebuilt warm identity drift")
        with self.ops.span("inventory_configuration"):
            chosen=self.pilot.set_inventory(self.model,*self.packet["inventory"]["blocks"])
            require(chosen["trainable_numel"]==self.packet["inventory"]["numel"],"adopted parameter count drift")
            self.trainable_names=chosen["trainable_names"]
            self.pilot.enable_checkpointing(self.model)
            settings=self.packet["update_recipe"]["optimizer"]
            self.optimizer=self.torch.optim.AdamW([p for n,p in self.model.named_parameters() if n in set(self.trainable_names)],
                lr=settings["lr"],betas=tuple(settings["betas"]),eps=settings["eps"],weight_decay=settings["weight_decay"],
                foreach=settings["foreach"],capturable=settings["capturable"])
            require(not self.optimizer.state,"fresh native Adam must be lazy; no warm/pilot optimizer inheritance")
        self.model.eval()
        with self.ops.span("fingerprint",state_bytes=self.state_bytes,reason="initial_inventory_frozen_proof",wall_contract=MODEL_FINGERPRINT) as proof:
            fingerprints=self.runtime.state_digests(self.model)
            require(self.runtime.combine(fingerprints)==self.packet["anchor"]["arm_sha256"],"inventory changed initial model")
            self.frozen_sha256=self.runtime.combine({n:h for n,h in fingerprints.items() if n not in self.trainable_names})
            proof.update(policy_sha256=self.runtime.combine(fingerprints),U=0,frozen_state_sha256=self.frozen_sha256)
        self.inventory=[{"name":n,"shape":list(dict(self.model.named_parameters())[n].shape),"dtype":"torch.float32"}
                        for n in self.trainable_names]
        self.guard_canary()

    @property
    def state_bytes(self):
        state=dict(self.model.named_parameters(remove_duplicate=False))
        state.update(dict(self.model.named_buffers(remove_duplicate=False)))
        return sum(v.numel()*v.element_size() for v in {id(v):v for v in state.values()}.values())

    def guard_canary(self):
        with self.ops.span("canary"):
            cal=self.competence.load_calibration(self.runtime.DATA)
            sections=self.competence.canary_sections(self.runtime.DATA,cal["bank"])
            receipt=self.competence.canary.run_canary_on(sections,role="calibration",signatures=("R","S","P"),
                expected_code_bundle=self.context["prerequisites"]["adopted"]["scoring_code_bundle_sha256"],
                scope_receipts={"calibration":cal["receipt"]})
            require(receipt["overall"]=="PASS","registered scorer canary failed")
            self.canary={"receipt_sha256":protocol.digest(receipt),"overall":"PASS",
                         "model_outputs_read":False,"roles":["calibration","guard_dev"],
                         "code_bundle_sha256":receipt["code_bundle_sha256"]}

    def fingerprint(self,operations,reason):
        with operations.span("fingerprint",state_bytes=self.state_bytes,reason=reason,wall_contract=MODEL_FINGERPRINT) as proof:
            digests=self.runtime.state_digests(self.model)
            policy=self.runtime.combine(digests)
            _,_,U=update._actual_optimizer(self.model,self.optimizer,dict(self.packet["update_recipe"],trainable_names=self.trainable_names),self.torch)
            frozen=self.runtime.combine({n:h for n,h in digests.items() if n not in self.trainable_names})
            require(frozen==self.frozen_sha256,"native frozen state drift")
            proof.update(policy_sha256=policy,U=U,frozen_state_sha256=frozen)
            return policy

    def sample(self,prompt,seed,operations):
        self.model.eval()
        sampled=self.generate.sample_group(self.model,prompt["prompt_ids"],1,seed,
                    max_new_tokens=128,eos_ids=self.packet["update_recipe"]["eos_ids"],pad_id=self.packet["update_recipe"]["pad_id"])
        require(sampled["group_size"]==1 and len(sampled["completions"])==1,"native single-sample collection drift")
        result=dict(sampled["completions"][0],sample_seed=seed,sampler=sampled["sampler"])
        return result

    def score(self,row,sample,operations):
        with operations.span("score"):
            text=self.competence.response_text(self.tokenizer,sample["completion_ids"],sample["stop_reason"])
            scored=self.competence.score_completion(row,sample,text)
            require(scored["reward"]==self.competence.canary.direct_strict(row,text["response"]),"TRAIN independent strict parity failed")
            return dict(scored,response=text["response"],body_special_ids=text["body_special_ids"],
                        body_unknown_ids=text["body_unknown_ids"])

    def update(self,prompts,groups,recipe,operations):
        return update.boundary_update(self.model,self.optimizer,prompts,groups,recipe,operations)

    def save(self,checkpoints,ledger,operations):
        with operations.span("checkpoint_adapter_io"):
            progress=ledger.progress_record()
            manifest=checkpoints.save(self.model,self.optimizer,progress,operations)
            require(manifest["progress"]==progress,"verified saved checkpoint progress drift")
            self.last_checkpoint_progress=progress
            return manifest

    def restore(self,checkpoints,ledger,operations):
        with operations.span("checkpoint_adapter_io"):
            restored=checkpoints.restore(self.model,self.optimizer)
        ledger.install_restore(restored)
        self.last_checkpoint_progress=None # new fence has not yet been saved
        return restored

    def cleanup(self):
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        self.optimizer=self.model=self.tokenizer=None
        gc.collect()
        if hasattr(self,"torch"):
            self.torch.cuda.synchronize(0)
            self.torch.cuda.empty_cache()


def verify_module_origins(sources):
    for name,module in list(sys.modules.items()):
        if not any(name=="experiments."+p or name.startswith("experiments."+p+".") for p in SOURCE_PACKAGES):continue
        path=getattr(module,"__file__",None)
        if path is None:
            spec=getattr(module,"__spec__",None)
            package=name.removeprefix("experiments.")
            expected=ROOT/"experiments"/package
            paths=list(getattr(module,"__path__",()))
            locations=list(getattr(spec,"submodule_search_locations",()) or ())
            require(package in SOURCE_PACKAGES and getattr(module,"__name__",None)==name
                    and getattr(spec,"name",None)==name and getattr(spec,"origin",None) is None
                    and type(getattr(spec,"loader",None)) is NamespaceLoader
                    and getattr(module,"__loader__",None) is spec.loader
                    and len(paths)==len(locations)==1
                    and Path(paths[0]).resolve()==expected and Path(locations[0]).resolve()==expected
                    and any(key.startswith("experiments/"+package+"/") and key.endswith(".py") for key in sources),
                    "unbound runtime namespace module: "+name)
            continue
        location=Path(path).resolve()
        require(location.is_relative_to(ROOT) and location.relative_to(ROOT).as_posix() in sources,
                "module loaded outside hashed source bundle: "+name)


def finish_checkpoint(backend,checkpoints,ledger,operations):
    if getattr(backend,"last_checkpoint_progress",None)==ledger.progress_record():
        return "COINCIDENT_FINAL_PERIODIC_ALREADY_VERIFIED"
    backend.save(checkpoints,ledger,operations)
    return "FINAL_VERIFIED_CHECKPOINT_SAVED"


def create_json(path,value,operations):
    raw=journal.canonical(value)+b"\n"
    with operations.span("receipt_write",path=str(path),write_bytes_upper=len(raw),
                         identity_hash_attempts_upper=1,identity_hash_bytes_upper=len(raw)) as op:
        bound=operations.profiles["receipt_write"]
        require(len(raw)<=bound["write_bytes_upper"],"receipt exceeds measured write reservation")
        operations.ledger.reserve_metadata("runner_receipt",len(raw),write_id=uuid.uuid4().hex,path=str(Path(path).resolve()))
        with Path(path).open("xb") as stream:
            for start in range(0,len(raw),1024**2):
                piece=raw[start:start+1024**2]
                require(stream.write(piece)==len(piece),"short result publication")
            stream.flush();os.fsync(stream.fileno())
        actual=observed_file(path,operations)
        require(actual["sha256"]==hashlib.sha256(raw).hexdigest(),"result readback mismatch")
        op.update(bytes_written=len(raw),sha256=actual["sha256"])
        return actual


def inspect_results(output,ledger,operations,packet_sha256):
    """Require a native count proof, verified file and final durable cost seal."""
    preserved=[]
    for path in sorted(Path(output).glob("RESULT*.json")):
        require(not path.is_symlink() and path.resolve().parent==Path(output).resolve(),"owned result path")
        actual=observed_file(path,operations,32*1024**2)
        # Parse under another reserved, durable read (the file hash above does
        # not supply bytes or authorize an unpriced reread).
        with operations.span("input_io",path=str(path),read_bytes_upper=actual["bytes"]+1,write_bytes_upper=0,
                             identity_hash_attempts_upper=1,identity_hash_bytes_upper=actual["bytes"]+1,wall_contract=OPAQUE_INPUT_FRAME) as record:
            raw=bounded_bytes(path,actual["bytes"]);record["bytes_read"]=len(raw)
            require(len(raw)==actual["bytes"] and hashlib.sha256(raw).hexdigest()==actual["sha256"],"result changed while read")
            try:value=journal._load(raw)
            except ValueError:value=None
        candidates=[op for op in ledger.external_ops.values() if op["operation"]=="receipt_write"
                    and op["phase"]=="returned" and op["metadata"].get("path")==str(path)
                    and op["result"].get("sha256")==actual["sha256"]]
        sealed=(type(value) is dict and value.get("schema")==SCHEMA and value.get("completed") is True
                and value.get("packet_sha256")==packet_sha256 and ledger.progress.complete()
                and any(op["metadata"].get("invocation_id")==value.get("invocation_id")
                        and ledger.invocations.get(value["invocation_id"],{}).get("outcome")=="DONOR_COMPLETE"
                        for op in candidates))
        if sealed:return True,preserved
        preserved.append(dict(actual,promotion="UNSEALED_PRESERVED"))
    return False,preserved


def windows_boot_identity(query=None):
    """Read the OS BootIdentifier GUID; unavailable/ambiguous epochs fail closed.

    NtQuerySystemInformation interface: Microsoft Learn winternl documentation.
    Class90/structure: original phnt ntexapi.h declarations maintained by
    System Informer, https://github.com/winsiderss/systeminformer/blob/master/phnt/include/ntexapi.h
    No rounded wall-time estimate or model/framework import is used.
    """
    import ctypes
    class BootEnvironment(ctypes.Structure):
        _fields_=[("identifier",ctypes.c_ubyte*16),("firmware",ctypes.c_uint32),("flags",ctypes.c_uint64)]
    information=BootEnvironment()
    returned=ctypes.c_uint32()
    if query is None:
        require(os.name=="nt","Windows boot identity is required for this registered worker")
        query=ctypes.WinDLL("ntdll").NtQuerySystemInformation
        query.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
        query.restype=ctypes.c_long
    status=query(90,ctypes.byref(information),ctypes.sizeof(information),ctypes.byref(returned))
    require(status==0 and returned.value==ctypes.sizeof(information),"OS boot identity unavailable; epoch cannot be certified")
    identifier=uuid.UUID(bytes_le=bytes(information.identifier))
    require(identifier.int!=0 and information.firmware in (1,2) and not information.flags&2,
            "OS boot identity/hibernation epoch is ambiguous; fail closed")
    return "windows-boot-guid-"+str(identifier)


def run(packet,*,resume=False,reason=None,started=None,setup=None):
    """One real production invocation. No synthetic/native bypass is exposed."""
    begin=time.perf_counter() if started is None else started
    bootstrap=setup or SetupObserver()
    context=metadata(packet,bootstrap)
    # Pure runtime import contains no Torch; pinned host and shared lease helpers
    # are imported only after the complete metadata/resource contract exists.
    from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime
    verify_module_origins(context["code"])
    require(Path(runtime.DATA).resolve()==Path(packet["data_manifest"]["path"]).resolve().parent,"registered TRAIN root drift")
    source=runtime.assert_clean_checkout(ROOT)
    host=pinned_helper(runtime.HOSTBUDGET_PATH,runtime.hostbudget,bootstrap)
    lease=pinned_helper(runtime.GPU_LEASE_PATH,runtime.gpu_lease,bootstrap)
    runtime.host_require(packet["limits"]["host_floor_gib"],"S1 before shared lease",module=host)
    output=Path(packet["output_dir"]).resolve()
    label="comp-rs-001-s1-"+packet["arm"].lower().replace("_","-")
    with runtime.lease(label,900,module=lease):
        ledger=journal.DurableLedger(output,context["planned"],context["caps"],
            {"packet_sha256":protocol.digest(packet),"code":context["code"],"source":source},
            {k:packet["limits"][k] for k in journal.IO_FIELDS},resume=resume)
        # Include metadata/lease waiting in this invocation's elapsed charge.
        boot_id=windows_boot_identity()
        ledger.start_invocation(resume=resume,reason=reason,started=begin,boot_id=boot_id)
        ledger.charge_setup(bootstrap.events)
        gate=ResourceGate(packet,runtime,host,output,ledger,context["host_transients"])
        operations=Operations(ledger,context["profiles"],gate)
        gate.launch()
        sealed,preserved_results=inspect_results(output,ledger,operations,protocol.digest(packet))
        if sealed:
            ledger.end_invocation("REFUSED_ALREADY_COMPLETE")
            raise ValueError("verified completed donor/cost seal already exists; no native relaunch")
        backend=NativeBackend(packet,context,runtime,operations,gate)
        watchdog=WallDeadline(ledger)
        watchdog.__enter__()
        completed=False
        try:
            backend.initialize()
            verify_module_origins(context["code"])
            with operations.span("dataset_read"):
                rows,manifests,data_binding=data.load_atomic_train(runtime.DATA,plan.ARMS[packet["arm"]])
                require(all(protocol.digest(manifests[k])==context["planned"].manifests_sha256[k] for k in manifests),"live atomic plan drift")
                rendered=data.render_training_prompts(backend.tokenizer,rows,runtime.render_prompt)
                require({k:len(v["prompt_ids"]) for k,v in rendered.items()}==context["census"]["prompt_lengths"],"live tokenizer census drift")
            from .checkpoint import NativeCheckpoints,constructor_hash_lower_bound
            bindings={"code":context["code"],"data":data_binding,"runtime":context["prerequisites"]["adopted"]["decoding_preimage"],
                "recipe":packet["update_recipe"],"inventory":backend.inventory,"plan":context["planned"].export(),
                "own_start_sha256":packet["start"]["arm_sha256"],"common_anchor_sha256":packet["anchor"]["arm_sha256"],
                "packet_sha256":protocol.digest(packet)}
            bindings["process_rng_seed"]=backend.process_rng_seed
            with operations.span("checkpoint_construct",**constructor_hash_lower_bound(bindings)):
                checkpoints=NativeCheckpoints(output/"checkpoints",bindings,packet["inventory"]["numel"],gate,
                    metadata_reserve=ledger.reserve_metadata).attach_ledger(operations)
            if resume:backend.restore(checkpoints,ledger,operations)
            else:backend.save(checkpoints,ledger,operations)
            execute_boundaries(context["planned"],rendered,ledger,backend,operations,checkpoints,packet["update_recipe"],10)
            final_checkpoint=finish_checkpoint(backend,checkpoints,ledger,operations)
            final=backend.fingerprint(operations,"final_completed_master")
            warm=context["prerequisites"]["warm"]
            delta=warm["delta"]
            require(protocol.hex64(delta.get("state_sha256")),"registered name-bound warm delta identity missing")
            donor_bind={"schema":update.DONOR_SCHEMA,"synthetic_cpu":False,"approved":True,"training_authorized":True,
                "base_weights_sha256":packet["anchor"]["base_weights_sha256"],"warm_delta_file_sha256":delta["file_sha256"],
                "warm_delta_data_sha256":packet["anchor"]["delta_sha256"],"warm_delta_state_sha256":delta["state_sha256"],
                "common_anchor_sha256":packet["anchor"]["arm_sha256"],"own_start_sha256":packet["start"]["arm_sha256"],
                "own_start_kind":"independent","own_start_components":[],"final_master_sha256":final,
                "frozen_state_sha256":backend.frozen_sha256,"trainable_names_sha256":update.digest(sorted(backend.trainable_names)),
                "operation":operations,"gate":gate}
            donor_path=output/("DONOR."+ledger.symbolic_snapshot+".safetensors")
            donor=update.extract_donor(backend.model,runtime.MODEL_DIR/"model.safetensors",packet["anchor"]["delta_path"],
                                      backend.trainable_names,donor_path,donor_bind)
            # Readback source and current native master, after donor proof.
            require(code_bundle(operations)==context["code"],"source changed during native invocation")
            require(runtime.assert_clean_checkout(ROOT)==source,"checkout changed during native invocation")
            require(backend.fingerprint(operations,"post_donor_final_identity")==final,"completed donor model drift")
            with operations.span("cleanup"):backend.cleanup()
            ledger.tick_wall()
            result={"schema":SCHEMA,"stage":"S1_NATIVE_DONOR","mode":"production","test_mode":False,"registered_run":True,
                "completed":True,"outcome":"DONOR_COMPLETE","arm":packet["arm"],"packet_sha256":protocol.digest(packet),
                "source":source,"code_sha256":context["code"],"anchor":packet["anchor"],"own_start":packet["start"],
                "plan_sha256":context["planned"].sha256,"units":context["planned"].units,"data":data_binding,
                "inventory":packet["inventory"],"optimizer":packet["update_recipe"]["optimizer"],"update_recipe_sha256":protocol.digest(packet["update_recipe"]),
                "qualification":packet["qualification"],"canary":backend.canary,"donor":donor,
                "model_load_identity_lower_bound":context["model_load_identity_lower_bound"],
                "process_rng_seed":backend.process_rng_seed,
                "final_checkpoint":final_checkpoint,
                "progress":ledger.progress_record(),"costs_before_result_publication":ledger.snapshot(),
                "invocation_id":ledger.current_invocation,"completion_requires_final_cost_seal":True,
                "preserved_unsealed_results":preserved_results,
                "required_cost_seal":"same invocation has returned receipt_write for this result file and invocation_end DONOR_COMPLETE",
                "cost_authority":"PHYSICAL_COSTS.jsonl includes publication/readback and invocation-end work",
                "later_arm_launch_authorized":False,"H1_required_external":True}
            result_path=output/"RESULT.json"
            if os.path.lexists(result_path):result_path=output/("RESULT."+ledger.symbolic_snapshot+"."+ledger.current_invocation+".json")
            result["result_path"]=str(result_path)
            create_json(result_path,result,operations)
            watchdog.__exit__(None,None,None)
            ledger.end_invocation("DONOR_COMPLETE")
            completed=True
            return result
        except BaseException as error:
            # A partial final donor/result stays unpromoted evidence. Resuming
            # uses a fresh fence and create-only snapshot-specific donor path.
            try:
                with operations.span("cleanup"):backend.cleanup()
            except BaseException:pass
            if not ledger.fatal:
                try:
                    failure={"schema":SCHEMA,"stage":"S1_NATIVE_DONOR","mode":"production","completed":False,
                        "outcome":"INCOMPLETE","arm":packet["arm"],"packet_sha256":protocol.digest(packet),
                        "error":{"type":type(error).__name__,"message":str(error)[:2000]},"costs":ledger.snapshot()}
                    create_json(output/("ATTEMPT."+ledger.current_invocation+".json"),failure,operations)
                    ledger.end_invocation("INCOMPLETE")
                except BaseException:pass
            raise
        finally:
            if watchdog.timer is not None:watchdog.timer.cancel()
            if not completed:
                backend.model=backend.optimizer=backend.tokenizer=None


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet",required=True)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--resume-reason")
    args=ap.parse_args(argv)
    begin=time.perf_counter()
    setup=SetupObserver()
    path=Path(args.packet).resolve(strict=True)
    require(path.stat().st_size<=32*1024**2,"packet byte limit")
    # This one command-line input necessarily precedes approval determination.
    # It is bounded, then included in setup evidence if approval is admitted.
    raw=bounded_bytes(path,32*1024**2)
    packet=journal._load(raw)
    setup("input_io","started",operation_id="command-line-packet",path=str(path),read_bytes_upper=len(raw),
          write_bytes_upper=0,identity_hash_attempts_upper=0,identity_hash_bytes_upper=0,wall_contract=OPAQUE_INPUT_FRAME)
    setup("input_io","returned",operation_id="command-line-packet",bytes_read=len(raw))
    try:
        run(packet,resume=args.resume,reason=args.resume_reason,started=begin,setup=setup)
    except (ValueError,OSError,RuntimeError) as error:
        print("S1 execution refused or incomplete: "+str(error),file=sys.stderr)
        return 3
    return 0


if __name__=="__main__":raise SystemExit(main())
