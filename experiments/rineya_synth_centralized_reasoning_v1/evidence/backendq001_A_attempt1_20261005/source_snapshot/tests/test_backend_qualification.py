"""Stdlib scalar/Fake protocol only; CLI help never imports frameworks/models."""
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

STAGE=Path(__file__).resolve().parents[1]
PACKAGE=STAGE.parents[1] if STAGE.name=="backend_qual_work" else STAGE
for location in (PACKAGE,PACKAGE/"tests",STAGE): sys.path.insert(0,str(location))
import backend_qual_contract as contract
import backend_qual_core as core
import backend_qual_worker as worker
import backend_qual_supervisor as supervisor
from test_study_native_chunk import QuadraticAdam,Journal


class Sink:
    def __init__(self,directory): self.directory=Path(directory);self.events=[];self.fail=None
    def __call__(self,name,value):
        self.events.append((name,deepcopy(value)))
        if self.fail==name: raise RuntimeError("injected durable evidence failure")
        if name=="checkpoint_path": return str(self.directory/value["filename"])
        with (self.directory/(name+".json")).open("x") as stream: json.dump(value,stream)


class FakeAdapter:
    expected_gradient_count=1
    def __init__(self,scope,call,guard,dose):
        self.scope,self.call,self.guard,self.dose=scope,call,guard,dose
        call("factory_model_load",lambda:"STD_LIB_FAKE_ONLY")
        self.backend=QuadraticAdam("M1",scope["qualification_recipe"],call,guard)
        self.check();self.check()
        original_forward=self.backend.forward
        def forward(row):
            old_call=self.backend.call
            def counted(name,function):
                if name=="training_forward":
                    def actual(): self.dose.charge(len(row["input_ids"]));return function()
                    return old_call(name,actual)
                return old_call(name,function)
            self.backend.call=counted
            try: return original_forward(row)
            finally: self.backend.call=old_call
        self.backend.forward=forward
        original_step=self.backend.optimizer_step
        def step(**kwargs): original_step(**kwargs);self.check()
        self.backend.optimizer_step=step
        original_save=self.backend.save_checkpoint
        def save(path,binding): self.check();return original_save(path,binding)
        self.backend.save_checkpoint=save
        original_restore=self.backend.restore_checkpoint
        def restore(descriptor,binding): self.check();result=original_restore(descriptor,binding);self.check();return result
        self.backend.restore_checkpoint=restore
    def check(self): return self.call("native_base_content_verify",lambda:True)
    def zero_parity(self,row):
        for operation in ("zero_parity_base_forward","zero_parity_adapter_forward"):
            self.call(operation,lambda:self.dose.charge(len(row["input_ids"])))
        self.check();return {"equal":True,"max_error":0.0,"all_B_zero":True,"CPU_fake_only":True}
    def witness(self,binding):
        self.check();state=self.backend.state()
        return {"kind":contract.WITNESS_KIND,"binding":deepcopy(binding),"state":state,"state_sha256":contract.digest(state),"CPU_fake_only":True}
    def prepare_reference(self,rows):
        before=self.backend.state();rng=random.Random();rng.setstate(self.backend.rng.getstate());gradient=0.
        for row in rows:
            def forward():
                self.dose.charge(len(row["input_ids"]))
                target=sum(row["completion_ids"][:-1])/max(1,len(row["completion_ids"])-1)/100000
                return self.backend.weight+rng.random()*.01-target
            derivative=self.call("reference_forward",forward)
            gradient+=self.call("reference_autograd_grad",lambda:derivative*.05)
        self.check()
        if self.backend.state()!=before: raise AssertionError("reference pass mutated state")
        return gradient
    def compare_reference(self,reference):
        error=abs(self.backend.gradient-reference)
        return {"passed":error<=1e-6+1e-5*abs(reference),"atol":1e-6,"rtol":1e-5,"count":1,"error":error,"CPU_fake_only":True}
    def close(self): pass


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.scopeA=contract.build_scope("A")
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.journal=Journal();self.guard=lambda stage:None
    def execute(self,scope,name,**kwargs):
        directory=self.path/name;directory.mkdir();sink=Sink(directory);holder={}
        def factory(*args): holder["adapter"]=FakeAdapter(*args);return holder["adapter"]
        if "factory" in kwargs: factory=kwargs.pop("factory")
        result=core.run_case(scope,factory,self.journal,self.guard,sink,cpu_control=True,**kwargs)
        return result,sink,holder
    def predecessor(self,result,sink):
        path=sink.directory/"worker_result.json";path.write_text(contract.canonical(result))
        return {"result_path":str(path),"result_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"supervisor_path":str(sink.directory/"fake_unused.json"),"supervisor_sha256":"0"*64}
    def scopeB(self,predecessor): return contract.build_scope("B",predecessor=predecessor)
    def test_exact200_preselected202A42B_and_unchanged_resource_numerical_caps(self):
        scope=self.scopeA
        self.assertEqual(200,len(scope["rows"]));self.assertEqual(431,scope["dose"]["maximum_row_tokens"])
        self.assertEqual(48112,scope["dose"]["expected_forward_input_positions"])
        self.assertEqual(47722,scope["c10_binding"]["input_tokens"]);self.assertEqual(2127,scope["c10_binding"]["supervised_tokens"])
        self.assertEqual(42592,scope["c9_binding"]["input_tokens"])
        self.assertEqual(contract.native.RESOURCE_BOUNDS,scope["resources"]);self.assertEqual(contract.native.TOLERANCES,scope["tolerances"])
        self.assertEqual((14,16,9),tuple(scope["resource_feasibility"][k] for k in ("first9_plus_C9_including_parity_and_witness_checks","A_total_full_base_checks","B_total_full_base_checks")))
        self.assertFalse(scope["resource_feasibility"]["scope_fit_qualified"]);self.assertFalse(scope["owned_qualification_authorized"])
        self.assertEqual([6.874,45.9],[round(x,3) for x in scope["resource_feasibility"]["observed_check_seconds"]])
    def test_A9continuous10_and_BfreshC9next20_reference_are_bitwise_same(self):
        a,sink,holder=self.execute(self.scopeA,"A")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL",a["status"],a.get("error"))
        self.assertEqual(202,a["dose"]["attempted_calls"])
        self.assertEqual(16,a["primitive_operations"]["native_base_content_verify"]["returned"])
        self.assertEqual(1,a["primitive_operations"]["factory_model_load"]["returned"])
        bscope=self.scopeB(self.predecessor(a,sink));b,_,_=self.execute(bscope,"B")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_FRESH_CONT10_CONTROL",b["status"],b.get("error"))
        self.assertEqual(a["witness10"],b["witness10"])
        self.assertEqual(42,b["dose"]["attempted_calls"]);self.assertEqual(10650,b["dose"]["input_positions"])
        self.assertEqual(9,b["primitive_operations"]["native_base_content_verify"]["returned"])
        self.assertEqual(20,b["primitive_operations"]["reference_autograd_grad"]["returned"])
        self.assertFalse(a["native_execution_performed"]);self.assertFalse(a["full27_qualified"]);self.assertFalse(b["full_resumed9_qualified"])
    def test_last_whole_row_mutation_partial_or_reordering_denied_before_factory(self):
        for change in (lambda s:s["rows"][-1]["labels"].__setitem__(0,0),lambda s:s["rows"].pop(),lambda s:s["rows"].reverse()):
            scope=deepcopy(self.scopeA);change(scope);calls=[]
            with self.assertRaises(ValueError): core.run_case(scope,lambda *a:calls.append(a),self.journal,self.guard,lambda *a:None,cpu_control=True)
            self.assertFalse(calls)
    def test_false_authority_resource_numerics_or_dose_cannot_be_resigned(self):
        for change in (lambda s:s["authorization"].update(production_training=True),lambda s:s["resources"].update(whole_scope_seconds=601),
                       lambda s:s["tolerances"].update(CE_absolute_error=1e-3),lambda s:s["dose"].update(maximum_forward_calls=203),lambda s:s.update(study_warmstart_allowed=True)):
            scope=deepcopy(self.scopeA);change(scope)
            with self.assertRaises(ValueError): contract.validate_packet(scope)
    def test_stage_native_default_and_eager_preflight_denied_before_factory(self):
        with patch.object(contract,"HERE",PACKAGE/"artifacts/backend_qual_work"):
            calls=[]
            with self.assertRaisesRegex(ValueError,"staged native"):
                core.run_case(self.scopeA,lambda *a:calls.append(a),self.journal,self.guard,lambda *a:None)
            self.assertFalse(calls)
            with self.assertRaisesRegex(ValueError,"staged native dispatch"): contract.preflight(self.path/"fake.json")
            with self.assertRaisesRegex(ValueError,"staging cannot"): contract.write_scope(self.path/"fake.json","A",authorize_owned_qualification=True)
            forged=deepcopy(self.scopeA);forged.update(owned_qualification_authorized=True,staged_native_dispatch_forbidden=False)
            with self.assertRaisesRegex(ValueError,"staged native"):
                core.run_case(forged,lambda *a:calls.append(a),self.journal,self.guard,lambda *a:None,admission=lambda request:{})
            self.assertFalse(calls)
    def test_pending_or_actual_returned_costs_survive_journal_failure(self):
        self.journal.fail_after="training_forward"
        result,_,_=self.execute(self.scopeA,"after_return")
        self.assertEqual("FAILED_SCOPED_BACKEND_QUALIFICATION",result["status"])
        self.assertEqual({"attempted":1,"returned":1,"failed":0,"pending":0},result["primitive_operations"]["training_forward"])
        self.assertIsNone(result["checkpoint_c9"])
    def test_partial_A_C9_or_failed10_cannot_be_B_predecessor(self):
        directory=self.path/"partial";directory.mkdir();sink=Sink(directory);sink.fail="ce_u10_r01"
        partial=core.run_case(self.scopeA,lambda *args:FakeAdapter(*args),self.journal,self.guard,sink,cpu_control=True)
        self.assertEqual("FAILED_SCOPED_BACKEND_QUALIFICATION",partial["status"]);self.assertIsNotNone(partial["checkpoint_c9"])
        bscope=self.scopeB(self.predecessor(partial,sink))
        with self.assertRaisesRegex(ValueError,"successful"): contract.verify_predecessor(bscope["predecessor"],bscope,native_supervised=False)
    def test_A_actual_dose_cost_recipe_and_checkpoint_byte_forgery_rejected(self):
        a,sink,_=self.execute(self.scopeA,"original")
        for index,change in enumerate((lambda r:r["dose"].update(attempted_calls=201),lambda r:r["protocol_operations"]["backward"].update(attempted=199,returned=199),
                                       lambda r:r.update(qualification_recipe_sha256="0"*64),lambda r:r["checkpoint_c9"]["metadata"]["binding"].update(completed_updates=8))):
            altered=deepcopy(a);change(altered);altered["sha256"]=contract.digest({k:v for k,v in altered.items() if k!="sha256"})
            directory=self.path/("forged"+str(index));directory.mkdir();writer=Sink(directory)
            predecessor=self.predecessor(altered,writer);bscope=self.scopeB(predecessor)
            with self.assertRaises(ValueError): contract.verify_predecessor(predecessor,bscope,native_supervised=False)
        Path(a["checkpoint_c9"]["path"]).write_bytes(b"changed")
        predecessor=self.predecessor(a,sink);bscope=self.scopeB(predecessor)
        with self.assertRaisesRegex(ValueError,"C9 bytes"): contract.verify_predecessor(predecessor,bscope,native_supervised=False)
    def test_output_ownership_released_before_next_forward_or_checkpoint(self):
        holder={};references=[]
        class Output:
            def __init__(self,value): self.value=value
            def __getitem__(self,key): return self.value[key]
        def factory(*args):
            adapter=FakeAdapter(*args);original=adapter.backend.forward
            def forward(row):
                gc.collect()
                if references and references[-1]() is not None: raise AssertionError("previous output retained")
                value=original(row);output=Output(value);adapter.backend.pending=output;references.append(weakref.ref(output));return output
            adapter.backend.forward=forward;holder["adapter"]=adapter;return adapter
        result,_,_=self.execute(self.scopeA,"lifetime",factory=factory)
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL",result["status"],result.get("error"))
        self.assertTrue(all(reference() is None for reference in references))
    def test_exact_dispatch_clone_qualified_helpers_and_startup_CLI_are_isolated(self):
        import inspect
        import native_supervisor
        source=inspect.getsource(native_supervisor.supervise)
        expected=source.replace("def supervise(","def supervise_backend_qualification(").replace("native_worker.py","backend_qual_worker.py").replace("mechanical_native_qualification_only",contract.SCOPE).replace("mechanical qualification only","backend qualification only")
        self.assertEqual(expected,inspect.getsource(supervisor.supervise_backend_qualification))
        result=subprocess.run([sys.executable,"-I","-B",str(STAGE/"backend_qual_worker.py"),"--help"],text=True,capture_output=True,timeout=30)
        self.assertEqual(0,result.returncode,result.stderr);self.assertIn("--output-dir",result.stdout)
        code="import sys;sys.path.insert(0,sys.argv[1]);import backend_qual_worker;assert 'torch' not in sys.modules and 'transformers' not in sys.modules;print('FRAMEWORK_FREE')"
        result=subprocess.run([sys.executable,"-I","-B","-c",code,str(STAGE)],text=True,capture_output=True,timeout=30,check=True)
        self.assertEqual("FRAMEWORK_FREE",result.stdout.strip())
    def test_transitive_current_source_closure_and_interpreter_gate_are_present(self):
        pins=contract.source_pins()
        for name in ("inference_contract.py","inference_component.py","study_native_chunk.py","study_torch_backend.py"):
            self.assertIn(str((PACKAGE/name).resolve()),pins)
        import inspect
        self.assertIn('native.runtime_files()==body["runtime_pins"]',inspect.getsource(contract.preflight))
        self.assertEqual("96f361052c34750313731c1fad0fe07d5ed48d376ee59850370cfeae99f59047",contract.native.filehash(PACKAGE/"study_native_chunk.py"))
    def test_hook_cleanup_failure_marks_failed_and_still_preserves_terminal(self):
        failures=worker.remove_hooks([type("BadHook",(),{"remove":lambda self:(_ for _ in ()).throw(RuntimeError("injected remove failure"))})()])
        self.assertEqual("RuntimeError",failures[0]["type"])
        package=self.path/"package";output=package/"artifacts/owned";output.mkdir(parents=True)
        (output/"supervisor_ownership.json").write_text('{}')
        scope=deepcopy(self.scopeA);scope.update(config_sha256="f"*64,source_commit="a"*40)
        captured=[]
        actual_preserve=worker.preserve_terminal
        def terminal(out,value,journal,torch):
            self.assertIsNone(torch)
            captured.append(deepcopy(value))
            actual_preserve(out,value,journal,torch)
            self.assertTrue(journal.stream.closed)
        report={"status":"PASS_SCOPED_BACKEND_A9_CONT10_MECHANICS","phase":"A"}
        with patch.object(worker,"PACKAGE",package),patch.object(contract,"preflight",return_value=scope),\
             patch.object(worker,"verify_worker_ownership",return_value={"mock":True}),patch.object(worker,"make_guard",return_value=lambda stage:None),\
             patch.object(worker,"run_case",return_value=report),patch.object(worker,"remove_hooks",return_value=failures),\
             patch.object(worker,"preserve_terminal",side_effect=terminal):
            self.assertEqual(2,worker.run(self.path/"config.json",output))
        self.assertEqual("FAILED_SCOPED_BACKEND_QUALIFICATION",captured[0]["status"])
        self.assertEqual(failures,captured[0]["hook_cleanup_errors"])


if __name__=="__main__": unittest.main()
