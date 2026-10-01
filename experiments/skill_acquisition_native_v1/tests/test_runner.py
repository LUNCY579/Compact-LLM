"""Pure orchestration spies; no native trainer qualification or ML imports."""
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import journal, plan, runner
from .test_journal import tiny_plan


class SpyBackend:
    synthetic_accounting_spy=True
    trainable_names=["toy.weight"]
    def __init__(self,ledger,fail_update=False):
        self.ledger,self.fail_update=ledger,fail_update
        self.policy="e"*64
        self.calls=[]
        self.saved=[]
        self.ordinal=0

    def fingerprint(self,operations,reason):
        self.calls.append(("fingerprint",self.ledger.progress.C))
        with operations.span("fingerprint",state_bytes=8,reason=reason):return self.policy

    def sample(self,prompt,seed,operations):
        self.calls.append(("sample",self.ledger.progress.C,prompt["id"],seed))
        self.ordinal+=1
        return {"completion_ids":[2],"stop_reason":"eos","generated_tokens":1,
                "raw_step_logprobs":[-1.0],"sample_seed":seed,"ordinal":self.ordinal}

    def score(self,row,sample,operations):
        return {"reward":int(sample["ordinal"]==1)}

    def update(self,prompts,groups,recipe,operations):
        self.calls.append(("update",self.ledger.progress.C))
        self.recipe=recipe
        meta={k:recipe[k] for k in ("boundary_id","generation","symbolic_snapshot","start_policy_id")}
        with operations.span("fingerprint",state_bytes=8,reason="pre_update",**meta) as pre:
            pre.update(policy_sha256=self.policy,U=self.ledger.progress.U,frozen_state_sha256="0"*64)
        signal=any(0<sum(item["reward"] for item in group)<8 for group in groups)
        U=self.ledger.progress.U
        result=dict(meta,updated=signal,U_before=U,U_after=U+int(signal),identity_operation=pre,optimizer_operation=None,
                    frozen_state_sha256="0"*64)
        if signal:
            with operations.span("optimizer",U_before=U,U_after=U+1,**meta) as op:pass
            if self.fail_update:raise ValueError("post-update proof failed")
            with operations.span("fingerprint",state_bytes=8,reason="post_update",**meta) as post:
                post.update(policy_sha256="f"*64,U=U+1,frozen_state_sha256="0"*64)
            self.policy="f"*64
            result.update(optimizer_operation=op,final_identity_operation=post,final_policy_id=self.policy,parity="PASS")
        return result

    def save(self,checkpoints,ledger,operations):
        self.saved.append((ledger.progress.C,ledger.progress.U))
        with operations.span("checkpoint_save",C=ledger.progress.C,U=ledger.progress.U):pass
        self.last_checkpoint_progress=ledger.progress_record()


class RunnerPureTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.root=Path(temp.name)
        self.addCleanup(temp.cleanup)
        self.planned=tiny_plan()
        prices={k:1 for k in ("t_gen95","t_update95","t_ckpt_upper","t_init_upper","t_restore_upper","t_cleanup_upper","t_finalize_upper")}
        caps=plan.derive_caps(self.planned,{"row%d"%i:3 for i in range(4)},prices,10)
        self.ledger=journal.DurableLedger(self.root,self.planned,caps,{"code":"d"*64},{k:10**8 for k in journal.IO_FIELDS})
        self.ledger.start_invocation(boot_id="pure-spy-boot")
        self.profiles={kind:{key:0 for key in runner.PROFILE_FIELDS} for kind in runner.IO_KINDS}
        self.profiles["model_load"].update(read_bytes_upper=1,identity_hash_attempts_upper=3,identity_hash_bytes_upper=1)
        self.profiles["receipt_write"].update(write_bytes_upper=1000000,identity_hash_attempts_upper=1,identity_hash_bytes_upper=1000000)
        self.gates=[]
        self.ops=runner.Operations(self.ledger,self.profiles,lambda stage:self.gates.append(stage))
        self.rendered={"row%d"%i:{"id":"row%d"%i,"prompt_ids":[3,4,5],"row":{"id":"row%d"%i}} for i in range(4)}
        self.recipe={"eos_ids":[2],"pad_id":0,"max_new_tokens":4}

    def test_unapproved_fail_before_source_and_framework(self):
        with patch.object(runner,"code_bundle",side_effect=AssertionError("source read forbidden")):
            with self.assertRaisesRegex(ValueError,"unapproved"):
                runner.metadata({"approved":False,"training_authorized":False},runner.SetupObserver())

    def test_import_and_pure_controls_need_no_framework(self):
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("transformers",sys.modules)
        self.assertNotIn("numpy",sys.modules)
        source=Path(runner.__file__).read_text()
        ast.parse(source)
        root=ast.parse(source)
        imports=[name.name for node in root.body if isinstance(node,ast.Import) for name in node.names]
        self.assertFalse(set(imports)&{"torch","numpy","transformers","safetensors"})

    def test_complete_collected_G16_before_update_C_U_separate(self):
        spy=SpyBackend(self.ledger)
        runner.execute_boundaries(self.planned,self.rendered,self.ledger,spy,self.ops,object(),self.recipe)
        self.assertEqual((self.ledger.progress.C,self.ledger.progress.U),(2,1))
        self.assertEqual(self.ledger.core.counters["rollouts_started"],32)
        self.assertEqual(self.ledger.core.counters["optimizer_started"],1)
        self.assertEqual(spy.saved,[(1,1)])
        for C in (0,1):
            part=[call for call in spy.calls if call[1]==C]
            self.assertEqual([call[0] for call in part],["fingerprint"]+["sample"]*16+["update"])
            expected=[self.planned.trajectory(C,i,k)["sample_seed"] for i in range(2) for k in range(8)]
            self.assertEqual([call[3] for call in part if call[0]=="sample"],expected)
        self.assertEqual(spy.recipe["boundary_id"],1)
        self.assertEqual(spy.recipe["trainable_names"],["toy.weight"])

    def test_returned_optimizer_without_postproof_never_commits_C_U(self):
        spy=SpyBackend(self.ledger,fail_update=True)
        with self.assertRaisesRegex(ValueError,"post-update proof"):
            runner.execute_boundaries(self.planned,self.rendered,self.ledger,spy,self.ops,object(),self.recipe)
        self.assertEqual((self.ledger.progress.C,self.ledger.progress.U),(0,0))
        self.assertEqual(self.ledger.core.counters["optimizer_returned"],1)
        self.assertEqual(self.ledger.core.counters["rollouts_returned"],16)
        self.assertEqual(spy.saved,[])

    def test_IO_profiles_must_be_explicit_measured_fields(self):
        runner.validate_profiles({"operation_bounds":self.profiles})
        bad=copy.deepcopy(self.profiles)
        del bad["model_load"]
        with self.assertRaisesRegex(ValueError,"profiles"):
            runner.validate_profiles({"operation_bounds":bad})
        bad=copy.deepcopy(self.profiles)
        bad["model_load"]["identity_hash_bytes_upper"]=False
        with self.assertRaisesRegex(ValueError,"types"):
            runner.validate_profiles({"operation_bounds":bad})

    def test_analytical_load_hash_lower_bound_includes_extra_native_assertion_passes(self):
        def write(name,dtype,payload):
            path=self.root/name
            header=json.dumps({"x":{"dtype":dtype,"shape":[2],"data_offsets":[0,len(payload)]}}).encode()
            path.write_bytes(struct.pack("<Q",len(header))+header+payload)
            return path
        base=write("base.safetensors","BF16",struct.pack("<HH",0x3f80,0x4000))
        warm=write("warm.safetensors","F32",struct.pack("<ff",.01,.02))
        observer=runner.SetupObserver()
        lower=runner.model_identity_lower_bound(base,warm,observer)
        self.assertEqual(lower["identity_hash_attempts_lower"],5)
        self.assertEqual(lower["identity_hash_bytes_lower"],base.stat().st_size+5*4+8)
        self.assertEqual([e["phase"] for e in observer.events],["started","returned"]*2)
        self.assertEqual(observer.events[1]["metadata"]["bytes_read"],base.stat().st_size-4)

    def test_gate_and_reservation_precede_work_and_profile_mismatch(self):
        with self.ops.span("model_load") as result:
            self.assertEqual(self.gates[-1],"model_load")
            self.assertEqual(self.ledger.extra["model_load_attempts"],1)
            result["bytes_read"]=1
        self.assertEqual(self.ledger.extra["chunk_io_read_bytes"],1)
        with self.assertRaisesRegex(ValueError,"measured IO"):
            with self.ops.span("model_load",read_bytes_upper=2):
                self.fail("physical work cannot begin")
        with self.assertRaisesRegex(ValueError,"analytical lower"):
            with self.ops.span("model_load",identity_hash_attempts_lower=4):
                self.fail("unpriced hash work cannot begin")

    def test_create_only_verified_result_IO_is_charged(self):
        path=self.root/"sample-result.json"
        actual=runner.create_json(path,{"test":"accounting-spy"},self.ops)
        self.assertTrue(path.exists())
        self.assertEqual(len(actual["sha256"]),64)
        self.assertGreater(self.ledger.extra["chunk_io_read_bytes"],0)
        with self.assertRaises(FileExistsError):runner.create_json(path,{"test":"overwrite"},self.ops)

    def test_checkpoint_cadence_and_first_actual_U(self):
        self.assertTrue(runner.checkpoint_due(10,0,0))
        self.assertTrue(runner.checkpoint_due(7,1,0))
        self.assertFalse(runner.checkpoint_due(7,0,0))
        self.assertFalse(runner.checkpoint_due(9,2,1))

    def test_coincident_final_periodic_checkpoint_counts_once(self):
        spy=SpyBackend(self.ledger)
        runner.execute_boundaries(self.planned,self.rendered,self.ledger,spy,self.ops,object(),self.recipe,cadence=2)
        self.assertEqual(spy.saved,[(1,1),(2,1)])
        self.assertEqual(runner.finish_checkpoint(spy,object(),self.ledger,self.ops),"COINCIDENT_FINAL_PERIODIC_ALREADY_VERIFIED")
        self.assertEqual(spy.saved,[(1,1),(2,1)])

    def test_partial_unsealed_result_is_preserved_and_completed_requires_cost_seal(self):
        path=self.root/"RESULT.json"
        path.write_bytes(b'{"partial":')
        sealed,preserved=runner.inspect_results(self.root,self.ledger,self.ops,"a"*64)
        self.assertFalse(sealed)
        self.assertEqual(preserved[0]["promotion"],"UNSEALED_PRESERVED")
        self.assertEqual(path.read_bytes(),b'{"partial":')
        spy=SpyBackend(self.ledger)
        runner.execute_boundaries(self.planned,self.rendered,self.ledger,spy,self.ops,object(),self.recipe)
        successor=self.root/"RESULT.fresh.json"
        value={"schema":runner.SCHEMA,"completed":True,"packet_sha256":"a"*64,"invocation_id":self.ledger.current_invocation}
        runner.create_json(successor,value,self.ops)
        self.assertFalse(runner.inspect_results(self.root,self.ledger,self.ops,"a"*64)[0])
        self.ledger.end_invocation("DONOR_COMPLETE")
        self.assertTrue(runner.inspect_results(self.root,self.ledger,self.ops,"a"*64)[0])
        self.assertTrue(path.exists())

    def test_exact_boot_GUID_and_failure_hibernation_ambiguity(self):
        import ctypes
        import uuid
        wanted=uuid.UUID("6b8d34d1-1c37-409a-8dd7-7c6df8e88c47")
        def query(number,information,size,returned):
            self.assertEqual(number,90)
            information._obj.identifier[:]=wanted.bytes_le
            information._obj.firmware=1
            returned._obj.value=size
            return 0
        self.assertEqual(runner.windows_boot_identity(query),"windows-boot-guid-"+str(wanted))
        def unavailable(*args):return -1
        with self.assertRaisesRegex(ValueError,"unavailable"):
            runner.windows_boot_identity(unavailable)
        def hibernation(number,information,size,returned):
            query(number,information,size,returned);information._obj.flags=2;return 0
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            runner.windows_boot_identity(hibernation)

    def test_deadline_uses_only_remaining_cumulative_wall(self):
        timers=[]
        class Timer:
            def __init__(self,seconds,callback):self.seconds,self.callback=seconds,callback;timers.append(self)
            def start(self):self.started=True
            def cancel(self):self.cancelled=True
        exits=[]
        with runner.WallDeadline(self.ledger,timer_factory=Timer,terminate=exits.append):
            self.assertTrue(timers[0].started)
            self.assertLessEqual(timers[0].seconds,self.ledger.core.caps["wall_charged_s"])
            timers[0].callback()
        self.assertEqual(exits,[124])
        self.assertTrue(timers[0].cancelled)


if __name__=="__main__":unittest.main()
