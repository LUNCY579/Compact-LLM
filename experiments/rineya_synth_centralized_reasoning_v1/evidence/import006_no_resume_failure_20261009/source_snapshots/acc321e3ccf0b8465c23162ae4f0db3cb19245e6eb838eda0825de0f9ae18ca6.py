"""Staged stdlib/mock controls only; no model/framework/GPU/native launch."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=HERE.parent if HERE.name=="tests" else HERE
sys.path.insert(0,str(WORK))
import short_collection_contract as contract
import short_collection_worker as worker
import short_collection_supervisor as supervisor
for target in (contract,worker,supervisor):
    assert Path(target.__file__).resolve().parent==WORK.resolve(), "controls must exercise actual root/stage targets"
sys.path.insert(0,str(contract.PACKAGE/"tests"))
from test_inference_component import FakeTorch,Parameter,Decoder,Model,row
from test_inference_qualification import HookedModel
from native_worker import Journal
import native_supervisor as qualified_supervisor
from rs_contract import digest


class ContractTests(unittest.TestCase):
    def test_staged_write_preflight_worker_dispatch_refuse_before_artifacts_or_native(self):
        with patch.object(contract,"STAGED",True):
            for action in (lambda:contract.write_scope(WORK/"not_created.json","atomic_RS_DEV",1),
                           lambda:contract.preflight(WORK/"not_created.json"),
                           lambda:worker.run(WORK/"not_created.json",WORK/"not_created"),
                           lambda:supervisor.dispatch(WORK/"not_created.json",WORK/"not_created")):
                with self.assertRaisesRegex(ValueError,"staged preparation"):action()
            self.assertFalse((WORK/"not_created.json").exists());self.assertFalse((WORK/"not_created").exists())

    def test_actual_cpu_plan_fixed18batches_complete1152_and_new2050caps(self):
        body=contract.plan()
        self.assertEqual(1152,body["rows"])
        self.assertEqual([6,12],[len(body["batches"][k]) for k in ("atomic_RS_DEV","task_guard_DEV")])
        for kind,batches in body["batches"].items():
            for ordinal,batch in enumerate(batches,1):
                self.assertEqual(batch,contract.select_batch(body,kind,ordinal))
                self.assertEqual(64,len(batch["rows"]));self.assertEqual(2050,batch["maximum_total_forwards"])
                expected=sum(32*r["prompt_tokens"]+496 for r in batch["rows"])+2*batch["rows"][0]["prompt_tokens"]
                self.assertEqual(expected,batch["maximum_total_input_positions"])
                self.assertTrue(all(set(r)==set(contract.ROW_FIELDS) for r in batch["rows"]))
        for kind,ordinal in (("language",1),("long_context",1),("atomic_RS_DEV",7),("task_guard_DEV",13),("atomic_RS_DEV",True)):
            with self.assertRaises(ValueError):contract.select_batch(body,kind,ordinal)

    def test_public_plan_source_tampering_and_resigned_expanded_permissions_rejected(self):
        original=contract.native.read_json
        for field,value in (("rows",1151),("scoring_authorized",True),("language_qualification_established",True)):
            body=original(contract.PLAN_PATH);body[field]=value;body["sha256"]=digest({k:v for k,v in body.items() if k!="sha256"})
            with patch.object(contract.native,"read_json",side_effect=lambda path:body if Path(path)==contract.PLAN_PATH else original(path)):
                with self.assertRaises(ValueError):contract.plan()

    def test_contract_has_no_scoring_language_long_or_training_authority(self):
        self.assertIs(True,contract.AUTHORIZATION["raw_DEV_collection"])
        for key in ("production_training","scoring","language","long_context","prior_correct_enrollment","protected_bank_access","paid_services"):
            self.assertIs(False,contract.AUTHORIZATION[key])
        self.assertEqual(2050,contract.FORWARD_CAP)

    def test_local_generated_only_decoder_uses_pinned_json_CPU_without_AutoTokenizer(self):
        config={"model_dir":str(contract.MODEL_DIR),"model_pins":dict(contract.ASSET_PINS),
                "runtime_versions":{"tokenizers":"0.22.2"},"decoder_runtime_pins":contract._tokenizer_runtime()}
        decoder=worker.LocalGeneratedDecoder(config)
        self.assertEqual('',decoder.decode([151645,151643],skip_special_tokens=True))
        with self.assertRaises(ValueError):decoder.decode([151645],skip_special_tokens=False)
        with self.assertRaises(ValueError):decoder.decode([True],skip_special_tokens=True)
        bad=deepcopy(config);bad['model_pins']['tokenizer.json']='0'*64
        with self.assertRaisesRegex(ValueError,'asset drift'):worker.LocalGeneratedDecoder(bad)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_executed_decoder_python_closure_and_implementation_drift_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            package=Path(directory)/'tokenizers';(package/'implementations').mkdir(parents=True)
            (package/'__init__.py').write_text('from .implementations import *\n')
            (package/'implementations/__init__.py').write_text('from .nested import decoder\n')
            (package/'implementations/nested.py').write_text('decoder = "CPU_FIXTURE"\n')
            (package/'tokenizers.pyd').write_bytes(b'CPU_FIXTURE_BINARY_NOT_LOADED')
            dist=SimpleNamespace(version='0.22.2',locate_file=lambda name:package)
            with patch.object(contract.importlib.metadata,'distribution',return_value=dist):
                closure=contract._tokenizer_runtime()
            self.assertEqual(4,len(closure));self.assertIn(str((package/'implementations/nested.py').resolve()),closure)
            decoder=object.__new__(worker.LocalGeneratedDecoder)
            decoder.config={"model_dir":str(contract.MODEL_DIR),"model_pins":dict(contract.ASSET_PINS),
                "runtime_versions":{"tokenizers":"0.22.2"},"decoder_runtime_pins":closure}
            decoder.verify_binding()
            (package/'implementations/nested.py').write_text('decoder = "DRIFT"\n')
            with self.assertRaisesRegex(ValueError,'source/binary drift'):decoder.verify_binding()
            self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_isolated_cli_help_imports_no_framework_or_native_worker_dispatch(self):
        result=subprocess.run([sys.executable,"-I","-B",str(WORK/"short_collection_worker.py"),"--help"],capture_output=True,text=True,timeout=20)
        self.assertEqual(0,result.returncode,result.stderr)
        code="import sys;sys.path.insert(0,sys.argv[1]);import short_collection_contract,short_collection_worker,short_collection_supervisor;assert not any(x in sys.modules for x in ['torch','transformers','rs_verifier','cpu_tokenizer']);print('FRAMEWORK_FREE')"
        result=subprocess.run([sys.executable,"-I","-B","-c",code,str(WORK)],capture_output=True,text=True,timeout=20)
        self.assertEqual(0,result.returncode,result.stderr);self.assertEqual("FRAMEWORK_FREE",result.stdout.strip())

    def test_dispatch_algorithm_exact_qualified_copy_with_fixed_declared_purpose(self):
        expected=inspect.getsource(qualified_supervisor.supervise)
        for old,new in (("def supervise(","def _supervise_registered_batch("),("native_worker.py","short_collection_worker.py"),
                        ("mechanical_native_qualification_only",contract.PURPOSE),("mechanical qualification only","one fixed short M0 raw batch only")):
            expected=expected.replace(old,new)
        actual=inspect.getsource(supervisor._supervise_registered_batch).replace('        if monitor is not None:\n            receipt["short_RAM_monitor_diagnostic"] = monitor.diagnostic_receipt()\n','')
        self.assertEqual(expected,actual)
        self.assertEqual(supervisor.ADAPTED_SUPERVISOR_SOURCE_SHA256,hashlib.sha256((contract.PACKAGE/"native_supervisor.py").read_bytes()).hexdigest())
        self.assertTrue(issubclass(supervisor._Monitor,qualified_supervisor._Monitor))
        for name in ("_WindowsOwnedWorker","_bounds_valid","resource_snapshot","_shutdown_gpu_probe"):
            self.assertIs(getattr(supervisor,name),getattr(qualified_supervisor,name))


class WorkerTests(unittest.TestCase):
    def fixture(self,directory):
        torch=FakeTorch();torch.equal=lambda x,y:x.shape==y.shape and [(r.token,r.selected,r.baseline) for r in x.rows]==[(r.token,r.selected,r.baseline) for r in y.rows]
        model=HookedModel(torch);decoder=Decoder();journal=Journal(Path(directory));rows=[row(i) for i in range(64)]
        config={"rows":rows,"source_packet_sha256":digest(rows),"dose":{"maximum_forward_calls":2050,
                "maximum_forward_input_positions":35840,"maximum_capture_input_positions":35836}}
        dose=worker.ForwardDose(torch,config);report={}
        def adapters(model,torch):
            for factor in ('A','B'):
                model.parameters['fake.lora_'+factor]=Parameter(requires_grad=True)
                model.parameters['fake.lora_'+factor].dtype='float32'
            return {"scope":"CPU_FAKE_COMPONENT","rank":8}
        helpers={"compute":lambda *args:{"scope":"CPU_FAKE_COMPONENT"},"adapters":adapters,
                 "activation_hooks":lambda *args:{"observations":model.observations,"handles":[]},
                 "zero":lambda *args:{"all_B_zero":True},
                 "base_verify":lambda *args:{"verified_source_tensors":310,"frozen_base_unchanged":True,"scope":"CPU_FAKE_COMPONENT"}}
        return torch,model,decoder,config,journal,dose,report,helpers
    def execute(self,fixture):
        torch,model,decoder,config,journal,dose,report,helpers=fixture
        return worker.collect_loaded(torch,model,decoder,config,{},journal,lambda stage:None,report,dose,device="cpu",**helpers)

    def test_full64_mock_collection_decode_generated_only_durable_tokens_no_scoring(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            f=self.fixture(directory);report=self.execute(f)
            self.assertEqual(64,len(report["collection"]["rows"]));self.assertEqual(194,report["forward_dose"]["attempted_calls"])
            self.assertEqual(580,report["forward_dose"]["input_positions"])
            self.assertEqual([[9,10,151645]]*64,f[2].calls)
            self.assertEqual(192,len(list(Path(directory).glob('stage_raw_row*_token*.json'))))
            first=json.loads((Path(directory)/'stage_raw_row0001_token01.json').read_text())
            second=json.loads((Path(directory)/'stage_raw_row0001_token02.json').read_text())
            eos=json.loads((Path(directory)/'stage_raw_row0001_token03.json').read_text())
            self.assertEqual('in_progress',first['finish_reason']);self.assertFalse(first['truncated'])
            self.assertEqual('in_progress',second['finish_reason']);self.assertFalse(second['truncated'])
            self.assertEqual('eos',eos['finish_reason']);self.assertFalse(eos['truncated'])
            self.assertEqual([],f[1].prehooks)
            self.assertTrue(all(not r["capability_correctness_scored"] and not r["confirmation_eligible"] for r in report["collection"]["rows"]))
            self.assertFalse(any('optimizer' in k or 'backward' in k for k in f[4].snapshot()["operations"]))
            f[4].stream.close()

    def test_token_partial_before_actual32cap_is_not_marked_cap_truncated(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            torch=FakeTorch();model=Model(torch,tokens=(9,));journal=Journal(Path(directory));decoder=Decoder()
            collector=worker.DurableRawCollection(torch,model,decoder,journal.call,lambda stage:None,journal.evidence,
                source_packet_sha256=digest([row()]),max_attempted_forwards=32,max_input_positions=560)
            result=collector.collect([row()],source_packet_sha256=digest([row()]))
            partial=json.loads((Path(directory)/'stage_raw_row0001_token31.json').read_text())
            capped=json.loads((Path(directory)/'stage_raw_row0001_token32.json').read_text())
            self.assertEqual('in_progress',partial['finish_reason']);self.assertFalse(partial['truncated'])
            self.assertEqual('generation_cap',capped['finish_reason']);self.assertTrue(capped['truncated'])
            self.assertEqual('generation_cap',result['rows'][0]['finish_reason']);self.assertTrue(result['rows'][0]['truncated'])
            self.assertEqual(32,collector.costs()['attempted_forwards']);journal.stream.close()

    def test_all64_prevalidate_last_oracle_or_overflow_before_any_forward(self):
        for field in ('oracle','overflow'):
            with tempfile.TemporaryDirectory(dir=WORK) as directory:
                f=self.fixture(directory)
                if field=='oracle':f[3]['rows'][-1]['target']='PRIVATE'
                else:f[3]['rows'][-1].update(prompt_ids=[5]*481,prompt_tokens=481,attention_mask=[1]*481)
                with self.assertRaises(ValueError):self.execute(f)
                self.assertEqual([],f[1].calls);self.assertEqual({},f[4].snapshot()['operations']);f[4].stream.close()

    def test_forward_failure_preserves_row_tokens_costs_and_no_next_row(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            f=self.fixture(directory)
            f[1].fault=lambda model,logits:(_ for _ in ()).throw(RuntimeError('injected forward failure')) if len(model.calls)==5 else None
            with self.assertRaisesRegex(RuntimeError,'injected forward failure'):self.execute(f)
            self.assertEqual(5,f[6]['forward_dose']['attempted_calls'])
            self.assertEqual([9,10],[x['token'] for x in f[6]['last_partial_receipt']['trace']])
            self.assertEqual(2,len(list(Path(directory).glob('stage_raw_row0001_token*.json'))))
            self.assertEqual(1,f[4].snapshot()['operations']['inference_forward']['failed']);self.assertFalse(f[1].prehooks);f[4].stream.close()

    def test_return_sink_failure_keeps_actual_model_return_costs_and_stops(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            f=self.fixture(directory);emit=f[4].emit
            def fail(event):
                if event.get('event')=='returned' and event.get('operation')=='inference_forward':raise OSError('returned sink failure')
                emit(event)
            f[4].emit=fail
            with self.assertRaisesRegex(OSError,'returned sink failure'):self.execute(f)
            self.assertEqual({'attempted':1,'returned':1,'failed':0,'pending':0},f[4].snapshot()['operations']['inference_forward'])
            self.assertEqual(3,f[6]['forward_dose']['attempted_calls']);self.assertEqual('COLLECTION_FAILED',f[6]['last_partial_receipt']['status']);f[4].stream.close()

    def test_partial_sink_failure_sticky_halt_with_no_next_forward(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            f=self.fixture(directory);save=f[4].evidence
            def fail(name,value):
                if name.startswith('raw_row0001_token02'):raise OSError('token evidence failed')
                save(name,value)
            f[4].evidence=fail
            with self.assertRaisesRegex(OSError,'token evidence failed'):self.execute(f)
            self.assertEqual(4,f[6]['forward_dose']['attempted_calls']);self.assertEqual([9,10],[x['token'] for x in f[6]['last_partial_receipt']['trace']]);f[4].stream.close()

    def test_zero_B_or_base_byte_failure_cannot_complete_collection(self):
        for stage in ('zero','base'):
            with tempfile.TemporaryDirectory(dir=WORK) as directory:
                f=self.fixture(directory)
                if stage=='zero':f[-1]['zero']=lambda *args:{'all_B_zero':False}
                else:f[-1]['base_verify']=lambda *args:{'verified_source_tensors':309,'frozen_base_unchanged':True}
                with self.assertRaises(ValueError):self.execute(f)
                self.assertEqual(1 if stage=='zero' else 194,f[5].calls);self.assertFalse(f[1].prehooks);f[4].stream.close()

    def test_actual_perhook2050_globalcaps_refuse_before_next_model_attempt(self):
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            f=self.fixture(directory);dose=f[5];dose.current='inference_forward';dose.calls=2050
            ids=f[0].tensor([[5,6]],dtype=f[0].long,device='cpu')
            with self.assertRaisesRegex(ValueError,'cap exhausted'):dose.before(None,(),{'input_ids':ids})
            self.assertEqual(2050,dose.calls);dose.current='training_forward'
            with self.assertRaisesRegex(ValueError,'registered'):dose.before(None,(),{'input_ids':ids})
            f[4].stream.close()


if __name__=='__main__':unittest.main()
