"""Bound framework-free amendment, count, failure and cleanup controls."""
from copy import deepcopy
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import backend_qual_contract as contract
import backend_qual_worker as worker


def snapshot(phase):
    counts=contract.checkpoint_binding(phase)["counts"]
    value={"use_reentrant":False,"checkpoint_early_stop":False,"preserve_rng_state":True,"failed_or_pending":False,"removed":False,
           "counters":{"forward_regions":counts["forward"],"recompute_regions":counts["recompute"],"no_grad_regions":56},"layers":{},"regions":[]}
    identity=0
    for index in range(28):
        value["layers"][str(index)]={}
        for kind in ("forward","recompute","no_grad"):
            n=counts["per_layer_"+kind]
            value["layers"][str(index)][kind]={"attempted":n,"returned":n,"failed":0,"pending":0,"elapsed_seconds":n*.001}
        for slot in range(counts["per_layer_forward"]):
            identity+=1
            for kind in ("forward","recompute"):value["regions"].append({"region_id":identity,"layer":index,"phase":kind,"status":"returned","elapsed_seconds":.001})
        for slot in range(2):
            identity+=1;value["regions"].append({"region_id":identity,"layer":index,"phase":"no_grad","status":"returned","elapsed_seconds":.001})
    return value


class Controls(unittest.TestCase):
    def test_actual_bound_contract_preserves_dose_base_recipe_limits_and_separate_profile(self):
        scope=contract.build_scope("A")
        self.assertEqual("SYNTH-BACKENDQ-002-FULLRECOMPUTE",scope["id"])
        self.assertEqual(contract.PROFILE,scope["base_compute_profile"])
        self.assertEqual(contract.CHECKPOINT_PROFILE,scope["profile"])
        self.assertEqual((202,48112,200,10),tuple(scope["dose"][key] for key in ("maximum_forward_calls","expected_forward_input_positions","native_backwards","native_optimizer_steps")))
        self.assertEqual((16,9),tuple(scope["resource_feasibility"][key] for key in ("A_total_full_base_checks","B_total_full_base_checks")))
        self.assertEqual(contract.native.RESOURCE_BOUNDS,scope["resources"]);self.assertEqual(contract.native.TOLERANCES,scope["tolerances"])
        self.assertFalse(scope["owned_qualification_authorized"]);self.assertTrue(scope["staged_native_dispatch_forbidden"])
        self.assertEqual(contract.CPU_PROOF_SHA,scope["checkpoint_amendment"]["CPU_proof_file_sha256"])
        self.assertEqual(contract.INSTALLER_SHA,scope["checkpoint_amendment"]["installer_sha256"])
        self.assertTrue(scope["checkpoint_runtime_pins"])
        self.assertEqual(21,len(scope["checkpoint_runtime_pins"]))
        self.assertTrue(any(path.endswith("python311.dll") for path in scope["checkpoint_runtime_pins"]))
        self.assertEqual({**contract.native.read_json(contract.PACKAGE/'configs/SYNTH_NQ001_PROC7.json')["runtime_pins"],**contract.checkpoint_runtime_files()},scope["runtime_pins"])
        self.assertFalse(any(name in sys.modules for name in ("torch","transformers")))
        self.assertEqual(HERE,Path(worker.__file__).resolve().parent)
        self.assertIn(str(HERE/"test_backend_recompute_scope.py"),scope["source_pins"])
    def test_A5600_B1120_additional_actual_regions_valid_only_when_complete_paired(self):
        for phase,total in (("A",5600),("B",1120)):
            amendment=contract.checkpoint_binding(phase);actual=snapshot(phase)
            self.assertEqual(total,actual["counters"]["recompute_regions"])
            self.assertTrue(worker.validate_decoder_work(actual,amendment))
            for mutate in (lambda x:x["regions"].pop(),lambda x:x["regions"].append(deepcopy(x["regions"][0])),
                           lambda x:x["regions"][0].update(status="pending"),lambda x:x["regions"][1].update(layer=27),
                           lambda x:x["layers"]["0"]["recompute"].update(failed=1,returned=x["layers"]["0"]["recompute"]["returned"]-1),
                           lambda x:x.update(checkpoint_early_stop=True),lambda x:x["counters"].update(recompute_regions=0),
                           lambda x:x["regions"][0].update(elapsed_seconds=float("inf")),
                           lambda x:x["layers"]["0"]["forward"].update(elapsed_seconds=0),
                           lambda x:(x["regions"][0].update(layer=27),x["regions"][1].update(layer=27))):
                changed=deepcopy(actual);mutate(changed)
                with self.assertRaises(ValueError):worker.validate_decoder_work(changed,amendment)
    def test_partial_work_persisted_and_cleanup_attempted_without_shrinking_denominator(self):
        partial=snapshot("B");partial["regions"]=partial["regions"][:3];partial["failed_or_pending"]=True
        removed=[]
        installed={"snapshot":lambda:deepcopy(partial),"remove":lambda:removed.append(True)}
        with tempfile.TemporaryDirectory() as folder:
            report={};self.assertEqual([],worker.close_profiles([installed],folder,report))
            self.assertEqual(3,report["decoder_work"][0]["region_file"]["rows"])
            self.assertTrue(report["decoder_work"][0]["failed_or_pending"]);self.assertEqual([True],removed)
    def test_install_after_return_failure_retains_cleanup_ownership_and_shared_hooks(self):
        partial=snapshot("B");partial["regions"]=[];partial["failed_or_pending"]=True
        removed=[];profiles=[];model=object();torch=object();checkpoint=object();early_stop=object()
        guard=lambda name:None
        installed={"snapshot":lambda:deepcopy(partial),"remove":lambda:removed.append(True)}
        def call(name,fn):
            value=fn()
            if name=="qualification_full_checkpoint_install":raise RuntimeError("after-return journal failure")
            return value
        with patch.object(worker.checkpoint_profile,"install",return_value=installed) as injected:
            with self.assertRaisesRegex(RuntimeError,"after-return"):
                worker.install_profile(model,torch,checkpoint,early_stop,call,guard,profiles)
            self.assertEqual([installed],profiles)
            injected.assert_called_once_with(model,torch,checkpoint,early_stop_context=early_stop,expected_layers=28,call=call,guard=guard)
        with tempfile.TemporaryDirectory() as folder:
            report={};self.assertEqual([],worker.close_profiles(profiles,folder,report));self.assertEqual([True],removed)
    def test_isolated_cli_bootstrap_help_and_dispatch_denial(self):
        argv=[sys.executable,"-I","-B",str(HERE/"backend_qual_worker.py")]
        help_result=subprocess.run(argv+["--help"],text=True,capture_output=True,timeout=30)
        self.assertEqual(0,help_result.returncode,help_result.stderr)
        self.assertIn("--config",help_result.stdout)
        denied=subprocess.run(argv+["--config",str(HERE/"does_not_exist.json"),"--output-dir",str(HERE)],text=True,capture_output=True,timeout=30)
        self.assertNotEqual(0,denied.returncode);self.assertIn("staged native dispatch",denied.stderr)
    def test_bad_installer_proof_bytes_and_staged_dispatch_fail_before_models(self):
        with patch.object(contract,"INSTALLER_SHA","0"*64):
            with self.assertRaisesRegex(ValueError,"bytes changed"):contract.checkpoint_binding("A")
        with patch.object(contract,"CPU_PROOF_SHA","0"*64):
            with self.assertRaisesRegex(ValueError,"bytes changed"):contract.checkpoint_binding("A")
        with self.assertRaisesRegex(ValueError,"staged native dispatch"):contract.preflight(HERE/'fake.json')
        with self.assertRaisesRegex(ValueError,"staging cannot"):contract.write_scope(HERE/'fake.json',"A",authorize_owned_qualification=True)


if __name__=="__main__":unittest.main()
