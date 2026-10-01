"""Pure durable-cost controls. No framework/model/native qualification is performed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import journal as j, plan


def tiny_plan():
    boundaries = tuple(plan.Boundary(i, tuple(plan.Prompt("R", "row%d" % (2*i+p),
                        tuple(1000+100*i+10*p+k for k in range(8))) for p in range(2))) for i in range(2))
    return plan.Plan("R", 1301, True, boundaries, "a"*64, {"R":"b"*64}, "c"*64)


class JournalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.planned = tiny_plan()
        self.caps = plan.derive_caps(self.planned, {"row%d" % i: 3 for i in range(4)},
                    {k:1 for k in ("t_gen95","t_update95","t_ckpt_upper","t_init_upper",
                                  "t_restore_upper","t_cleanup_upper","t_finalize_upper")}, 10)
        self.extra = {key:10**8 for key in j.IO_FIELDS}
        self.now = 100.0
        self.ledger = self.new()
        self.ledger.start_invocation(boot_id="same-boot")

    def new(self, resume=False):
        return j.DurableLedger(self.root, self.planned, self.caps, {"code":"d"*64}, self.extra,
                               resume=resume, clock=lambda:self.now)

    def returned(self, ledger=None, mixed=False):
        ledger = ledger or self.ledger
        attempts=[]
        for p in range(2):
            for k in range(8):
                trajectory=self.planned.trajectory(ledger.progress.C,p,k)
                attempt=ledger.rollout_start(trajectory,3,"e"*64)
                sample={"rollout_id":attempt,"prompt_id":trajectory["row_id"],"policy_id":"e"*64,
                        "generated_tokens":1,"completion_ids":[9],"reward":int(mixed and p==0 and k==0)}
                ledger.rollout_return(attempt,sample)
                attempts.append(attempt)
        return attempts

    def proof(self, ledger=None, mixed=False):
        ledger=ledger or self.ledger
        meta={"boundary_id":ledger.progress.C,"generation":ledger.generation,
              "symbolic_snapshot":ledger.symbolic_snapshot,"start_policy_id":"e"*64}
        def op(kind,identifier,**extra):
            fields=dict(meta,**extra)
            ledger.operation(kind,"started",identifier,**fields)
            ledger.operation(kind,"returned",identifier,**fields)
            return {"operation_id":identifier,**fields}
        pre=op("fingerprint","pre%d" % ledger.progress.C,state_bytes=8,policy_sha256="e"*64,
               U=ledger.progress.U,frozen_state_sha256="0"*64)
        result=dict(meta,updated=mixed,U_before=ledger.progress.U,U_after=ledger.progress.U+int(mixed),
                    identity_operation=pre,optimizer_operation=None,frozen_state_sha256="0"*64)
        if mixed:
            result.update(parity="PASS",final_policy_id="f"*64,
                optimizer_operation=op("optimizer","step%d" % ledger.progress.C,
                                       U_before=ledger.progress.U,U_after=ledger.progress.U+1),
                final_identity_operation=op("fingerprint","post%d" % ledger.progress.C,state_bytes=8,
                                            policy_sha256="f"*64,U=ledger.progress.U+1,frozen_state_sha256="0"*64))
        return result

    def restored(self, progress):
        target=copy.deepcopy(progress)
        target.update(generation=self.ledger.generation+1,snapshot="f"*64)
        self.ledger.operation("checkpoint_restore","started","restore",read_bytes_upper=12)
        self.ledger.operation("checkpoint_restore","returned","restore",C=target["C"],U=target["U"],
                 generation=target["generation"],symbolic_snapshot=target["snapshot"],bytes_read=12,manifest_sha256="1"*64)
        return {"progress":target,"manifest_sha256":"1"*64}

    def test_fsync_start_and_exact_replay_costs(self):
        a=self.ledger.rollout_start(self.planned.trajectory(0,0,0),3,"e"*64)
        text=self.ledger.path.read_text()
        self.assertIn(a,text)
        self.ledger.end_invocation("interrupted")
        original=self.ledger.snapshot()
        resumed=self.new(resume=True)
        self.assertEqual(resumed.core.counters,original["cumulative"])
        self.assertEqual(len(resumed.core.pending_rollouts),1)
        self.assertEqual(resumed.core.counters["generated_tokens_charged"],128)
        self.assertGreater(resumed.extra["chunk_io_read_bytes"],0)
        resumed.start_invocation(resume=True,reason="documented failure",boot_id="same-boot")
        self.assertEqual(resumed.core.counters["resumes"],1)

    def test_uniform_C_advances_without_Adam_U(self):
        self.ledger.commit_boundary(self.returned(),self.proof())
        self.assertEqual((self.ledger.progress.C,self.ledger.progress.U),(1,0))
        self.assertEqual(self.ledger.core.counters["optimizer_started"],0)

    def test_mixed_commit_requires_actual_returned_step_and_order(self):
        attempts=self.returned(mixed=True)
        self.ledger.commit_boundary(attempts,self.proof(mixed=True))
        self.assertEqual((self.ledger.progress.C,self.ledger.progress.U),(1,1))
        self.assertEqual(self.ledger.core.counters["optimizer_returned"],1)

    def test_permuting_prompt_groups_refused(self):
        attempts=self.returned()
        proof=self.proof()
        with self.assertRaisesRegex(ValueError,"order"):
            self.ledger.commit_boundary(attempts[8:]+attempts[:8],proof)
        self.assertEqual(self.ledger.progress.C,0)

    def test_restore_retains_costs_and_fences_all_old_samples(self):
        checkpoint=self.ledger.progress_record()
        attempts=self.returned(mixed=True)
        result=self.proof(mixed=True)
        self.ledger.commit_boundary(attempts,result)
        counters=dict(self.ledger.core.counters)
        restored=self.restored(checkpoint)
        self.ledger.install_restore(restored)
        self.assertEqual((self.ledger.progress.C,self.ledger.progress.U),(0,0))
        self.assertEqual(self.ledger.core.counters,counters)
        self.assertEqual(self.ledger.extra["checkpoint_restore_attempts"],1)
        with self.assertRaisesRegex(ValueError,"fence"):
            self.ledger.commit_boundary(attempts,result)

    def test_restore_invalid_history_leaves_progress_unchanged(self):
        self.ledger.commit_boundary(self.returned(),self.proof())
        before=self.ledger.progress_record()
        restored=self.restored(before)
        restored["progress"]["history"][0]["updated"]=True
        with self.assertRaisesRegex(ValueError,"updated-history"):
            self.ledger.install_restore(restored)
        self.assertEqual(self.ledger.progress_record(),before)

    def test_returned_unused_work_is_stale_after_restore(self):
        checkpoint=self.ledger.progress_record()
        attempts=self.returned(mixed=True)
        result=self.proof(mixed=True)
        self.ledger.install_restore(self.restored(checkpoint))
        result=dict(result,generation=self.ledger.generation,symbolic_snapshot=self.ledger.symbolic_snapshot)
        with self.assertRaisesRegex(ValueError,"stale"):
            self.ledger.commit_boundary(attempts,result)
        self.assertEqual(self.ledger.progress.C,0)

    def test_interrupted_restore_can_reinitialize_but_not_sample_or_update(self):
        self.ledger.operation("checkpoint_restore","started","oldrestore",read_bytes_upper=12)
        self.ledger.end_invocation("interrupted_restore")
        resumed=self.new(resume=True)
        resumed.start_invocation(resume=True,reason="restore interrupted",boot_id="same-boot")
        resumed.operation("module_import","started","freshmodules")
        resumed.operation("module_import","returned","freshmodules")
        resumed.operation("model_load","started","freshmodel")
        resumed.operation("model_load","returned","freshmodel")
        self.assertTrue(resumed.restoration_pending)
        self.assertEqual(resumed.extra["checkpoint_restore_attempts"],1)
        with self.assertRaisesRegex(ValueError,"fence"):
            resumed.rollout_start(self.planned.trajectory(0,0,0),3,"e"*64)

    def test_IO_upper_reserved_and_no_refund(self):
        self.ledger.operation("input_io","started","input",read_bytes_upper=100,
                              identity_hash_attempts_upper=1,identity_hash_bytes_upper=100)
        self.ledger.operation("input_io","failed","input",known_bytes_read=2)
        self.assertEqual(self.ledger.extra["chunk_io_read_bytes"],100)
        self.assertEqual(self.ledger.extra["identity_hash_bytes"],100)
        self.assertEqual(self.ledger.extra["identity_hash_attempts"],1)

    def test_rehashed_bootstrap_charge_laundering_is_refused(self):
        observations=[{"kind":"input_io","phase":"started","operation_id":"setup",
            "metadata":{"read_bytes_upper":100,"write_bytes_upper":0,"identity_hash_attempts_upper":1,"identity_hash_bytes_upper":100}},
            {"kind":"input_io","phase":"returned","operation_id":"setup","metadata":{"bytes_read":99}}]
        self.ledger.charge_setup(observations)
        rows=self.ledger.path.read_bytes().splitlines()
        last=j._load(rows[-1])
        last["payload"]["charges"]["chunk_io_read_bytes"]=0
        last["sha256"]=j.digest({key:value for key,value in last.items() if key!="sha256"})
        rows[-1]=j.canonical(last)
        self.ledger.path.write_bytes(b"\n".join(rows)+b"\n")
        with self.assertRaisesRegex(ValueError,"bootstrap.*charges"):
            self.new(resume=True)

    def test_restore_manifest_identity_and_one_use_proof(self):
        original=self.ledger.progress_record()
        restored=self.restored(original)
        wrong=copy.deepcopy(restored);wrong["manifest_sha256"]="2"*64
        with self.assertRaisesRegex(ValueError,"unused returned restore"):
            self.ledger.install_restore(wrong)
        self.assertEqual(self.ledger.progress_record(),original)
        self.ledger.install_restore(restored)
        after=self.ledger.progress_record()
        with self.assertRaisesRegex(ValueError,"unused returned restore"):
            self.ledger.install_restore(restored)
        self.assertEqual(self.ledger.progress_record(),after)

    def test_final_policy_cannot_be_changed_without_returned_hash_proof(self):
        attempts=self.returned(mixed=True)
        proof=self.proof(mixed=True)
        proof["final_policy_id"]="1"*64
        with self.assertRaisesRegex(ValueError,"post-update native hash"):
            self.ledger.commit_boundary(attempts,proof)
        self.assertEqual(self.ledger.progress.C,0)

    def test_three_hash_pass_span_actuals_bound_by_builtin_plus_extra(self):
        self.ledger.operation("fingerprint","started","hashwork",state_bytes=300,
                              identity_hash_attempts_upper=2,identity_hash_bytes_upper=0)
        self.ledger.operation("fingerprint","returned","hashwork",identity_hash_attempts_actual=3,
                              identity_hash_bytes_actual=300)
        self.assertEqual(self.ledger.extra["identity_hash_attempts"],3)
        self.assertEqual(self.ledger.extra["identity_hash_bytes"],300)
        self.ledger.operation("fingerprint","started","excess",state_bytes=10)
        with self.assertRaisesRegex(ValueError,"hash work exceeded"):
            self.ledger.operation("fingerprint","returned","excess",identity_hash_attempts_actual=1,
                                  identity_hash_bytes_actual=11)

    def test_failed_checkpoint_hash_uncertainty_retains_whole_bound(self):
        self.ledger.operation("fingerprint","started","uncertain",state_bytes=300,identity_hash_attempts_upper=2)
        self.ledger.operation("fingerprint","failed","uncertain",identity_hash_actual_known=False,
                              uncertainty_rule="entire admitted hash work remains charged")
        self.assertEqual(self.ledger.extra["identity_hash_attempts"],3)
        self.assertEqual(self.ledger.extra["identity_hash_bytes"],300)
        self.assertEqual(self.ledger.external_ops["uncertain"]["phase"],"failed")

    def test_boot_change_unknown_wall_fails_closed(self):
        self.ledger.rollout_start(self.planned.trajectory(0,0,0),3,"e"*64)
        resumed=self.new(resume=True)
        with self.assertRaisesRegex(plan.UnresolvedBudget,"wall upper"):
            resumed.start_invocation(resume=True,reason="boot interrupted",boot_id="different-boot")
        self.assertTrue(resumed.core.wall_unknown)

    def test_partial_or_rehashed_semantic_corruption_refused(self):
        self.ledger.end_invocation("stopped")
        with self.ledger.path.open("ab") as f:f.write(b'{')
        with self.assertRaisesRegex(ValueError,"partial"):
            self.new(resume=True)

    def test_cap_rejection_precedes_work(self):
        self.ledger.extra_limits["model_load_attempts"]=1
        self.ledger.operation("model_load","started","load1")
        self.ledger.operation("model_load","returned","load1")
        with self.assertRaisesRegex(plan.CeilingRejected,"model_load"):
            self.ledger.operation("model_load","started","load2")
        self.assertNotIn("load2",self.ledger.external_ops)


if __name__=="__main__": unittest.main()
