"""Stdlib scalar quadratic/Adam/RNG proof; no Torch, model, GPU or child."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
import weakref
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import study_native_chunk as chunk
import study_schedule as schedule


class Journal:
    def __init__(self): self.events=[];self.fail_after=None
    def __call__(self,name,function):
        self.events.append((name,"attempt"))
        value=function()
        self.events.append((name,"returned"))
        if self.fail_after==name: raise RuntimeError("journal failed after actual return")
        return value


class Preserve:
    def __init__(self,directory): self.directory=Path(directory);self.events=[];self.fail=None;self.corrupt=None
    def __call__(self,event):
        self.events.append(deepcopy(event))
        name=event["event"]
        if self.fail==name: raise RuntimeError("injected persistence failure "+name)
        if name=="checkpoint_path":
            return str(self.directory/("study_"+event["arm"]+"_chunk"+str(event["chunk_ordinal"])+".weights"))
        payload=event["payload"]
        path=self.directory/(name+"_"+str(len(self.events))+".json")
        raw=(schedule.canonical(payload)+"\n").encode()
        with path.open("xb") as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
        publication={"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"payload_sha256":schedule.digest(payload),"durable":True}
        if self.corrupt==name: publication["sha256"]="0"*64
        return publication


class QuadraticAdam:
    execution_scope=chunk.FAKE_SCOPE
    def __init__(self,arm,current_recipe,call,guard,*,scope=chunk.FAKE_SCOPE):
        self.execution_scope,self.arm,self.seed=scope,arm,101
        self.recipe_sha256,self.cohort_sha256=current_recipe["sha256"],current_recipe["cohort_sha256"]
        self.backend_source_sha256=current_recipe["backend_source_sha256"]
        self.call,self.guard=call,guard
        self.rng=random.Random(101);self.weight=self.rng.random()
        self.gradient=self.m=self.v=0.0;self.completed_updates=self.input_tokens=self.supervised_tokens=0
        self.order=[];self.ordinals=[];self.lr=None;self.primitives={};self.phase="fresh";self.n=0
        self._attestation={"kind":"FRESH_ZERO_B_SEEDED_STUDY_START_NOT_NATIVE_PARITY","seed":101,
                           "initial_weight":self.weight,"qualification_STATE1_used":False,"optimizer_state_empty":True,"scope":scope}
        self._attestation["sha256"]=schedule.digest(self._attestation)
    @property
    def fresh_start_attestation(self): return deepcopy(self._attestation)
    def primitive_operation_snapshot(self): return deepcopy(self.primitives)
    def invoke(self,name,function):
        self.guard("fake_before_"+name)
        counts=self.primitives.setdefault(name,{k:0 for k in ("attempted","returned","failed","pending")})
        counts["attempted"]+=1;counts["pending"]+=1
        def actual():
            try: value=function()
            except BaseException: counts["pending"]-=1;counts["failed"]+=1;raise
            counts["pending"]-=1;counts["returned"]+=1;return value
        value=self.call(name,actual);self.guard("fake_after_"+name);return value
    def reset_gradients(self,update_ordinal):
        if self.phase not in ("fresh","complete") or update_ordinal!=self.completed_updates+1: raise ValueError("partial/global reset")
        self.invoke("gradient_reset",lambda:setattr(self,"gradient",0.0))
        self.n=0;self.active=update_ordinal;self.phase="forward"
    def forward(self,row):
        if self.phase!="forward": raise ValueError("forward phase")
        schedule.validate_token_row(row)
        def forward():
            target=sum(row["completion_ids"][:-1])/max(1,len(row["completion_ids"])-1)/100000
            return {"derivative":self.weight+self.rng.random()*.01-target,"row":deepcopy(row)}
        self.pending=self.invoke("training_forward",forward);self.phase="loss";return self.pending
    def completion_loss(self,output,row):
        if self.phase!="loss" or output is not self.pending or row!=output["row"]: raise ValueError("loss row")
        self.loss=self.invoke("completion_CE_selected_NLL_check",lambda:output["derivative"]**2/2)
        self.last_loss_check={"ce":self.loss};self.phase="backward";return self.loss
    def backward(self,loss,*,scale):
        if self.phase!="backward" or loss!=self.loss or type(scale) is not float or scale!=.05: raise ValueError("scale/order")
        self.invoke("training_backward",lambda:setattr(self,"gradient",self.gradient+self.pending["derivative"]*scale))
        row=self.pending["row"];self.order.append(row["row_id"])
        self.input_tokens+=len(row["input_ids"]);self.supervised_tokens+=len(row["completion_ids"])
        self.n+=1;self.pending=None;self.phase="clip" if self.n==20 else "forward"
    def finite_clip_gradients(self,*,max_norm):
        if self.phase!="clip" or max_norm!=1.0: raise ValueError("full20 clip")
        def clip():
            if not math.isfinite(self.gradient): raise ValueError("finite gradient")
            norm=abs(self.gradient);self.gradient=max(-1.,min(1.,self.gradient));return norm
        value=self.invoke("finite_gradient_clip",clip);self.phase="step";return value
    def optimizer_step(self,*,lr,update_ordinal):
        if self.phase!="step" or update_ordinal!=self.active or lr!=schedule.learning_rates()[update_ordinal-1]: raise ValueError("globalLR/full update")
        def step():
            self.m=.9*self.m+.1*self.gradient;self.v=.999*self.v+.001*self.gradient**2
            self.weight-=lr*(self.m/(1-.9**update_ordinal))/(math.sqrt(self.v/(1-.999**update_ordinal))+1e-8)
        self.invoke("optimizer_step",step)
        self.invoke("consumed_gradient_clear",lambda:setattr(self,"gradient",0.0))
        self.completed_updates=update_ordinal;self.lr=lr;self.ordinals.append(update_ordinal);self.phase="complete"
    def state(self):
        state=self.rng.getstate()
        return {"weight":self.weight,"m":self.m,"v":self.v,"gradient":self.gradient,"completed_updates":self.completed_updates,
                "input_tokens":self.input_tokens,"supervised_tokens":self.supervised_tokens,"order":list(self.order),"ordinals":list(self.ordinals),
                "lr":self.lr,"rng":[state[0],list(state[1]),state[2]]}
    def metadata(self,binding):
        return {"schema_version":1,"kind":chunk.BACKEND_KIND,"experiment":"SYNTH-CRS-001","scope":self.execution_scope,
                "arm":self.arm,"seed":101,"recipe_sha256":self.recipe_sha256,"cohort_sha256":self.cohort_sha256,
                "backend_source_sha256":self.backend_source_sha256,"fresh_start_attestation_sha256":self._attestation["sha256"],
                "binding":deepcopy(binding),"native_launch_authorized":False,"training_authorized":False,"native_resume_qualified":False,"qualification_STATE1_used":False}
    def save_checkpoint(self,path,binding):
        expected={"completed_updates":self.completed_updates,"consumed_occurrences":len(self.order),"input_tokens":self.input_tokens,
                  "supervised_tokens":self.supervised_tokens,"consumed_order_sha256":schedule.digest(self.order),"next_update_ordinal":self.completed_updates+1,"learning_rate":self.lr}
        if self.phase!="complete" or self.gradient!=0.0 or binding!=expected: raise ValueError("partial accumulation/binding")
        metadata=self.metadata(binding);raw=schedule.canonical({"metadata":metadata,"state":self.state()}).encode()
        def write():
            with Path(path).open("xb") as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
        self.invoke("checkpoint_write",write)
        return {"schema_version":1,"kind":chunk.BACKEND_KIND,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"metadata":metadata,"metadata_sha256":schedule.digest(metadata)}
    def restore_checkpoint(self,descriptor,expected_binding):
        if self.phase!="fresh" or descriptor["metadata"]!=self.metadata(expected_binding): raise ValueError("restore arm/source/recipe/partial")
        raw=self.invoke("checkpoint_read_verified_bytes",lambda:chunk._read_bytes(descriptor["path"],descriptor["bytes"],descriptor["sha256"]))
        state=json.loads(raw)["state"]
        if state["gradient"]!=0.0 or state["completed_updates"]!=expected_binding["completed_updates"]: raise ValueError("partial saved state")
        def restore():
            for name in ("weight","m","v","gradient","completed_updates","input_tokens","supervised_tokens","order","ordinals","lr"): setattr(self,name,deepcopy(state[name]))
            self.rng.setstate((state["rng"][0],tuple(state["rng"][1]),state["rng"][2]));self.phase="complete"
        self.invoke("checkpoint_state_restore",restore)
        return {"status":"RESTORED_UNQUALIFIED_BACKEND_COMPONENT","binding":deepcopy(expected_binding),"native_resume_qualified":False,"training_authorized":False}


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration=schedule.read_json(PACKAGE/"STUDY_DEV_REGISTRATION_DRAFT.json")
        cls.pools={arm:{row["row_id"]:row for row in schedule.read_jsonl(schedule.BUNDLE/("token_pool_"+arm+".jsonl"))} for arm in ("M1","M2")}
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.journal=Journal();self.guard=lambda stage:None
    def fixture(self,arm="M1",name="fake",scope=chunk.FAKE_SCOPE):
        directory=self.directory/name;directory.mkdir()
        recipe=chunk.recipe(self.registration,arm,scope)
        return QuadraticAdam(arm,recipe,self.journal,self.guard,scope=scope),Preserve(directory)
    def execute(self,backend,preserve,arm="M1",ordinal=1,**kwargs):
        return chunk.run_chunk(self.registration,arm,ordinal,backend,self.pools.get(arm,self.pools["M1"]).__getitem__,self.journal,self.guard,preserve,**kwargs)
    def test_consumed_output_and_loss_released_before_next_forward_and_clip(self):
        class Output(dict): pass
        class Loss: pass
        backend,preserve=self.fixture()
        old_forward,old_loss,old_backward,old_clip=backend.forward,backend.completion_loss,backend.backward,backend.finite_clip_gradients
        previous={"output":None,"loss":None}
        def released():
            for reference in previous.values():
                if reference is not None and reference() is not None:
                    raise RuntimeError("consumed forward graph still retained")
        def forward(row):
            released()
            value=Output(old_forward(row));backend.pending=value
            previous["output"]=weakref.ref(value)
            return value
        def loss(output,row):
            old_loss(output,row)
            value=Loss();backend.loss=value
            previous["loss"]=weakref.ref(value)
            return value
        def backward(value,**kwargs):
            old_backward(value,**kwargs);backend.loss=None
        def clip(**kwargs):
            released();return old_clip(**kwargs)
        backend.forward,backend.completion_loss,backend.backward,backend.finite_clip_gradients=forward,loss,backward,clip
        result=self.execute(backend,preserve)
        self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"],result.get("error"))
        self.assertEqual(180,result["costs"]["protocol"]["backward"]["returned"])
        released()
    def test_continuous27_equals_three9_chunks_bitwise_Adam_RNG_order_dose_both_arms(self):
        for arm in ("M1","M2"):
            full,_=self.fixture(arm,arm+"continuous")
            for update in self.registration["arms"][arm]["updates"]:
                ordinal=update["update_ordinal"];full.reset_gradients(ordinal)
                for identity in update["occurrence_row_ids"]:
                    row=deepcopy(self.pools[arm][identity]);output=full.forward(row);loss=full.completion_loss(output,row);full.backward(loss,scale=.05)
                full.finite_clip_gradients(max_norm=1.);full.optimizer_step(lr=update["learning_rate"],update_ordinal=ordinal)
            resume=None;costs=None
            for ordinal in (1,2,3):
                part,preserve=self.fixture(arm,arm+str(ordinal))
                result=self.execute(part,preserve,arm,ordinal,resume_descriptor=resume,latest_terminal_costs=costs)
                self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"],result.get("error"))
                resume,costs=result["resume_descriptor"],result["costs"]
            self.assertEqual(full.state(),part.state())
            self.assertEqual(list(range(1,28)),part.ordinals)
            self.assertEqual(schedule.EXPECTED_TOTALS[arm],(part.input_tokens,part.supervised_tokens))
            self.assertEqual(540,len(part.order));self.assertEqual(0.,part.lr)
            self.assertEqual(27,costs["protocol"]["optimizer_step"]["returned"])
            self.assertEqual(540,costs["protocol"]["backward"]["returned"])
            self.assertEqual(27,costs["primitive_operations"]["consumed_gradient_clear"]["returned"])
            self.assertEqual(2,costs["protocol"]["checkpoint_reload"]["returned"])
            self.assertEqual("NATIVE_RESUME_UNQUALIFIED",result["native_resume_status"])
            self.assertTrue(result["study_unauthorized"]);self.assertFalse(result["training_authorized"])
    def test_all180_occurrences_prevalidate_before_any_backend_operation(self):
        backend,preserve=self.fixture();reads=[]
        def reader(identity):
            reads.append(identity);row=deepcopy(self.pools["M1"][identity])
            if len(reads)==180: row["labels"][0]=0
            return row
        with self.assertRaises(ValueError): chunk.run_chunk(self.registration,"M1",1,backend,reader,self.journal,self.guard,preserve)
        self.assertEqual(180,len(reads));self.assertEqual({},backend.primitives);self.assertEqual([],preserve.events)
    def test_reader_content_mutation_and_wrong_arm_are_denied_before_backend(self):
        for arm in ("M0","M2"):
            backend,preserve=self.fixture(name=arm)
            with self.assertRaises(ValueError): self.execute(backend,preserve,arm)
            self.assertEqual({},backend.primitives)
        backend,preserve=self.fixture(name="mutated")
        def altered(identity):
            row=deepcopy(self.pools["M1"][identity]);row["input_ids"][0]+=1;row["prompt_ids"][0]+=1;return row
        with self.assertRaisesRegex(ValueError,"frozen pool"):
            chunk.run_chunk(self.registration,"M1",1,backend,altered,self.journal,self.guard,preserve)
        self.assertEqual({},backend.primitives)
    def test_forward_row_mutation_stops_before_loss_and_does_not_mutate_pool(self):
        backend,preserve=self.fixture();original=backend.forward
        def bad(row):
            output=original(row);row["labels"][0]=0;return output
        backend.forward=bad;result=self.execute(backend,preserve)
        self.assertIn("mutated",result["error"]);self.assertIsNone(result["resume_descriptor"])
        self.assertEqual(1,result["costs"]["protocol"]["forward"]["returned"])
        self.assertEqual(0,result["costs"]["protocol"]["completion_loss"]["attempted"])
    def test_partial_accumulation_cannot_be_saved_or_promoted(self):
        backend,preserve=self.fixture();original=backend.optimizer_step
        def uncleared(**kwargs): original(**kwargs);backend.gradient=.25
        backend.optimizer_step=uncleared
        result=self.execute(backend,preserve)
        self.assertEqual("FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"])
        self.assertIn("partial",result["error"]);self.assertIsNone(result["resume_descriptor"])
        self.assertEqual(0,result["costs"]["protocol"]["checkpoint_publish"]["attempted"])
    def test_persistence_failures_keep_actual_costs_and_forbid_promotion_resume(self):
        for event in ("checkpoint_publish","terminal_costs_publish","checkpoint_promote"):
            backend,preserve=self.fixture(name=event);preserve.fail=event
            result=self.execute(backend,preserve)
            self.assertEqual("FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"])
            self.assertIsNone(result["resume_descriptor"])
            self.assertEqual(9,result["costs"]["protocol"]["optimizer_step"]["returned"])
            self.assertEqual(1,result["costs"]["protocol"][event]["failed"])
            if event!="checkpoint_promote": self.assertEqual(0,result["costs"]["protocol"]["checkpoint_promote"]["attempted"])
    def test_unverified_publication_hash_never_promotes(self):
        backend,preserve=self.fixture();preserve.corrupt="checkpoint_publish"
        result=self.execute(backend,preserve)
        self.assertIn("hash",result["error"]);self.assertIsNone(result["resume_descriptor"])
        self.assertEqual(0,result["costs"]["protocol"]["checkpoint_promote"]["attempted"])
    def test_journal_failure_after_actual_return_does_not_erase_primitive_return(self):
        backend,preserve=self.fixture();self.journal.fail_after="chunk_terminal_costs_publish_write"
        result=self.execute(backend,preserve)
        costs=result["costs"]["primitive_operations"]["chunk_terminal_costs_publish_write"]
        self.assertEqual({"attempted":1,"returned":1,"failed":0,"pending":0},costs)
        self.assertEqual(1,result["costs"]["protocol"]["terminal_costs_publish"]["failed"])
        self.assertIsNone(result["resume_descriptor"])
    def test_partial_failed_latest_costs_and_changed_recipe_or_bytes_refuse_restore(self):
        first,preserve=self.fixture(name="first");result=self.execute(first,preserve)
        resume=result["resume_descriptor"]
        for index,mutation in enumerate((lambda r:r["envelope"]["binding"].update(completed_updates=8),
                                         lambda r:r["envelope"]["recipe"].update(seed=102),
                                         lambda r:r["envelope"].update(arm="M2"))):
            altered=deepcopy(resume);mutation(altered)
            altered["sha256"]=schedule.digest({k:v for k,v in altered.items() if k!="sha256"})
            backend,sink=self.fixture(name="resume_bad"+str(index))
            with self.assertRaises(ValueError): self.execute(backend,sink,ordinal=2,resume_descriptor=altered,latest_terminal_costs=resume["terminal_costs"])
            self.assertEqual({},backend.primitives)
        backend,sink=self.fixture(name="latest_bad")
        latest=deepcopy(resume["terminal_costs"]);latest["protocol"]["forward"]["attempted"]+=1;latest["protocol"]["forward"]["pending"]+=1
        with self.assertRaisesRegex(ValueError,"failed/pending"):
            self.execute(backend,sink,ordinal=2,resume_descriptor=resume,latest_terminal_costs=latest)
        self.assertEqual({},backend.primitives)
        Path(resume["envelope"]["backend_descriptor"]["path"]).write_bytes(b"changed")
        backend,sink=self.fixture(name="bytes_bad")
        with self.assertRaisesRegex(ValueError,"bytes/hash"):
            self.execute(backend,sink,ordinal=2,resume_descriptor=resume,latest_terminal_costs=resume["terminal_costs"])
        self.assertEqual({},backend.primitives)
    def test_qualification_STATE1_eager_native_and_default_no_admission_denied(self):
        backend,preserve=self.fixture();backend.execution_scope=chunk.NATIVE_SCOPE
        with self.assertRaisesRegex(ValueError,"eager native"): self.execute(backend,preserve)
        calls=[]
        def factory(**kwargs): calls.append(kwargs);raise AssertionError("constructor forbidden")
        factory.execution_scope=chunk.NATIVE_SCOPE
        with self.assertRaisesRegex(ValueError,"admission callback"): self.execute(factory,preserve)
        self.assertEqual([],calls)
        for resume in ({"kind":"STATE1"},None):
            backend,sink=self.fixture(name="state"+str(resume is None))
            with self.assertRaises(ValueError): self.execute(backend,sink,ordinal=2,resume_descriptor=resume,latest_terminal_costs={})
            self.assertEqual({},backend.primitives)
    def test_native_tagged_factory_is_CPU_fake_admitted_before_constructor_with_bound_freshness(self):
        order=[];holder={};directory=self.directory/"lazy";directory.mkdir();preserve=Preserve(directory)
        def factory(**kwargs):
            order.append("factory")
            kwargs["call"]("factory_model_load",lambda:"CPU_FAKE_ONLY")
            recipe=chunk.recipe(self.registration,"M1",chunk.NATIVE_SCOPE)
            holder["backend"]=QuadraticAdam("M1",recipe,kwargs["call"],kwargs["guard"],scope=chunk.NATIVE_SCOPE)
            return holder["backend"]
        factory.execution_scope=chunk.NATIVE_SCOPE
        def admission(request):
            order.append("admission");now=time.monotonic()
            return {"binding":request,"authorized":True,"owned_supervision_verified":True,"model_load_and_constructor_in_scope":True,"observed_monotonic":now,"expires_monotonic":now+.19}
        result=self.execute(factory,preserve,admission=admission)
        self.assertEqual(["admission","factory"],order)
        self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"],result.get("error"))
        for costs in (result["costs"],result["resume_descriptor"]["envelope"]["posted_save_costs"],result["resume_descriptor"]["durable_terminal"]["costs"]):
            self.assertEqual({"attempted":1,"returned":1,"failed":0,"pending":0},costs["primitive_operations"]["factory_model_load"])
        self.assertTrue(result["study_unauthorized"]);self.assertNotIn("torch",sys.modules)
        for index,change in enumerate((lambda e:e["binding"].update(arm="M2"),lambda e:e.update(observed_monotonic=0.,expires_monotonic=.2),lambda e:e.update(model_load_and_constructor_in_scope=False))):
            sink=Preserve(self.directory);order.clear()
            def bad(request):
                value=admission(request);change(value);return value
            with self.assertRaises(ValueError): self.execute(factory,sink,admission=bad)
            self.assertEqual(["admission"],order)
    def test_factory_load_and_failed_constructor_costs_survive_without_backend(self):
        def factory(**kwargs):
            kwargs["call"]("factory_model_load",lambda:"CPU_FAKE_ONLY")
            def fail(): raise RuntimeError("CPU fake constructor failure")
            kwargs["call"]("factory_constructor",fail)
        factory.execution_scope=chunk.NATIVE_SCOPE
        def admission(request):
            now=time.monotonic()
            return {"binding":request,"authorized":True,"owned_supervision_verified":True,"model_load_and_constructor_in_scope":True,"observed_monotonic":now,"expires_monotonic":now+.19}
        preserve=Preserve(self.directory)
        result=self.execute(factory,preserve,admission=admission)
        self.assertEqual("FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED",result["status"])
        self.assertEqual({"attempted":1,"returned":1,"failed":0,"pending":0},result["costs"]["primitive_operations"]["factory_model_load"])
        self.assertEqual({"attempted":1,"returned":0,"failed":1,"pending":0},result["costs"]["primitive_operations"]["factory_constructor"])
        self.assertIsNone(result["resume_descriptor"]);self.assertEqual([],preserve.events)
    def test_resigned_posted_save_primitive_erasure_fails_exact_projection_chain(self):
        backend,preserve=self.fixture(name="truthful");result=self.execute(backend,preserve)
        original=result["resume_descriptor"];altered=deepcopy(original)
        envelope=altered["envelope"];envelope["posted_save_costs"]["primitive_operations"]={}
        envelope["sha256"]=schedule.digest({k:v for k,v in envelope.items() if k!="sha256"})
        directory=self.directory/"forged";directory.mkdir();writer=Preserve(directory)
        altered["publication"]=writer({"event":"checkpoint_publish","payload":envelope})
        terminal=altered["durable_terminal"];terminal["envelope_sha256"]=envelope["sha256"]
        altered["terminal_publication"]=writer({"event":"terminal_costs_publish","payload":terminal})
        promotion=altered["promotion"];promotion["envelope_sha256"]=envelope["sha256"];promotion["durable_terminal_sha256"]=schedule.digest(terminal)
        altered["promotion_publication"]=writer({"event":"checkpoint_promote","payload":promotion})
        altered["sha256"]=schedule.digest({k:v for k,v in altered.items() if k!="sha256"})
        later,sink=self.fixture(name="later")
        with self.assertRaisesRegex(ValueError,"projection chain"):
            self.execute(later,sink,ordinal=2,resume_descriptor=altered,latest_terminal_costs=original["terminal_costs"])
        self.assertEqual({},later.primitives)


if __name__=="__main__": unittest.main()
