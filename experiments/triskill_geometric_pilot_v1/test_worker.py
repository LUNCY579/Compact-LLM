import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import worker
from worker import (ARMS, Budget, MAX_FORWARDS, MAX_GENERATIONS, MAX_TOKEN_POSITIONS, directional_rotation,
                    native_allocation, preflight, prompt_system, screen_schedule, validate_freeze)
from benchmark import build_development, build_screen
from geometry import SOURCES, sha256

# Observed on this host on 2026-10-01 with no research job running: WDDM lists the
# compositor and shell, two of them without attribution.
WDDM_HOLDERS = "2608, [Insufficient Permissions]\n7148, C:\\Windows\\explorer.exe\n26156, [Insufficient Permissions]"


def gpu(free_mib=6900, holders=WDDM_HOLDERS):
    return lambda: (f"NVIDIA GeForce RTX 4070 Laptop GPU, 8188, {free_mib}, 3", holders)


def proc(pid, name, rss=None):
    memory = None if rss is None else types.SimpleNamespace(rss=rss)
    return types.SimpleNamespace(info={"pid": pid, "name": name, "memory_info": memory})


class BudgetTests(unittest.TestCase):
    def test_prospective_limits_can_fail(self):
        clock=[0.]
        b=Budget(now=lambda:clock[0])
        b.forward(288)
        self.assertEqual(b.receipt()["model_forwards"],1)
        for attr,value,action in (
            ("forwards",MAX_FORWARDS,lambda:b.forward(1)),
            ("token_positions",MAX_TOKEN_POSITIONS,lambda:b.forward(1)),
            ("generations",MAX_GENERATIONS,b.generation),
        ):
            before=getattr(b,attr)
            setattr(b,attr,value)
            with self.assertRaises(RuntimeError):action()
            setattr(b,attr,before)
        clock[0]=worker.WALL_SECONDS
        with self.assertRaises(RuntimeError):b.check()
        self.assertLess(worker.WALL_SECONDS, 600)

    def test_registered_allocation_fits_with_exact_generation_count(self):
        allocation=native_allocation(build_screen(),build_development())
        self.assertTrue(allocation["fits"])
        self.assertEqual(allocation["generations"],MAX_GENERATIONS)
        parts=allocation["parts"]
        self.assertEqual(parts["generation"],[154*32,154*(288+31)])
        self.assertEqual(parts["teacher_forced_readout"],[18*4,18*4*319])
        self.assertEqual(parts["hook_liveness"],[2*4,2*4*288])
        self.assertEqual(allocation["forwards"],sum(part[0] for part in parts.values()))
        self.assertLessEqual(allocation["forwards"],MAX_FORWARDS)
        self.assertLessEqual(allocation["token_positions"],MAX_TOKEN_POSITIONS)


class ScheduleTests(unittest.TestCase):
    def test_every_arm_covers_every_row_except_prompt(self):
        rows=build_screen()
        schedule=screen_schedule(rows)
        self.assertEqual(sum(len(unit_rows) for _,unit_rows in schedule),MAX_GENERATIONS)
        self.assertEqual([arm for arm,_ in schedule[:3]],["BASE","T","H"])
        self.assertTrue(all(row.task_kind.endswith("parent") for _,unit_rows in schedule[:3] for row in unit_rows))
        coverage={arm:set() for arm in ARMS}
        for arm,unit_rows in schedule:
            coverage[arm].update(row.row_id for row in unit_rows)
        every={row.row_id for row in rows}
        chains={row.row_id for row in rows if "rules" in row.metadata}
        for arm in ARMS:
            self.assertEqual(coverage[arm], chains if arm == "PROMPT" else every, arm)
        # Weight reloads after the gate: H (already loaded) -> T -> BASE -> TH-hosted arms.
        targets=[worker.weights_target(arm) for arm,_ in schedule]
        self.assertEqual(sum(1 for a,b in zip(targets,targets[1:]) if a!=b),5)

    def test_prompt_control_changes_exactly_chain_systems(self):
        rows=build_screen()
        self.assertEqual(sum(prompt_system(row,"PROMPT") != row.system for row in rows),10)
        for row in rows:
            self.assertEqual(prompt_system(row,"TH"),row.system)

    def test_reverse_uses_transpose_without_choosing_from_outcomes(self):
        import numpy as np
        r=np.array([[0.,-1.],[1.,0.]])
        np.testing.assert_array_equal(directional_rotation(r,"REVERSE"),r.T)
        np.testing.assert_array_equal(directional_rotation(r,"ROTATION"),r)
        np.testing.assert_array_equal(directional_rotation(r,"RANDOM"),r)
        self.assertIsNone(directional_rotation(r,"IDENTITY"))
        self.assertIsNone(directional_rotation(r,"TH"))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock=Path(self.temp.name)/"rineya-gpu-0.lock"

    def run_preflight(self, token=None, processes=(), free_mib=6900, holders=WDDM_HOLDERS, ancestors=(4242,)):
        return preflight(token, lock=self.lock, processes=list(processes), gpu_query=gpu(free_mib, holders),
                         ancestors=set(ancestors))

    def gpu_blockers(self, record):
        return [reason for reason in record["blockers"] if "GPU" in reason or "lease" in reason or "research" in reason]

    def test_desktop_gpu_holders_do_not_block(self):
        record=self.run_preflight(processes=[proc(11,"dwm.exe",90<<20),proc(12,"explorer.exe",200<<20),
                                             proc(13,"python.exe",20<<20)])
        self.assertEqual(self.gpu_blockers(record),[])
        self.assertIn("Insufficient Permissions",record["gpu_holders_raw_information_only"])

    def test_resource_and_ownership_rules_still_fail_closed(self):
        self.assertIn("GPU below 6 GiB preload threshold",self.run_preflight(free_mib=998)["blockers"])
        self.assertIn("foreign research/model process has >=512MiB RSS",
                      self.run_preflight(processes=[proc(99,"python.exe",3<<30)])["blockers"])
        self.assertIn("research process memory not visible",
                      self.run_preflight(processes=[proc(98,"python.exe")])["blockers"])
        self.lock.write_text(json.dumps({"pid":77,"token":"other","label":"someone-else"}),encoding="utf-8")
        self.assertIn("shared GPU lease held/unreadable",self.run_preflight()["blockers"])
        self.lock.write_text("{not json",encoding="utf-8")
        self.assertIn("shared GPU lease held/unreadable",self.run_preflight()["blockers"])
        failing=lambda:(_ for _ in ()).throw(OSError("nvidia-smi missing"))
        record=preflight(None,lock=Path(self.temp.name)/"absent.lock",processes=[],gpu_query=failing,ancestors=set())
        self.assertIn("GPU query failed",record["blockers"])

    def test_supervisor_lease_is_recognized_through_launcher_ancestry(self):
        lease={"pid":4242,"token":"t0k","label":worker.LEASE_LABEL}
        self.lock.write_text(json.dumps(lease),encoding="utf-8")
        self.assertTrue(self.run_preflight("t0k",ancestors=(1,4242))["lease_owned_by_supervisor"])
        self.assertEqual(self.gpu_blockers(self.run_preflight("t0k",ancestors=(1,4242))),[])
        for token,ancestors,label in (("bad",(4242,),worker.LEASE_LABEL),("t0k",(1,),worker.LEASE_LABEL),
                                      ("t0k",(4242,),"other-label")):
            self.lock.write_text(json.dumps({**lease,"label":label}),encoding="utf-8")
            record=self.run_preflight(token,ancestors=ancestors)
            self.assertIn("shared GPU lease not held by this worker's supervisor",record["blockers"])
        self.lock.unlink()
        self.assertIn("shared GPU lease not held by this worker's supervisor",self.run_preflight("t0k")["blockers"])

    def test_own_supervisor_python_is_not_foreign(self):
        record=self.run_preflight(processes=[proc(4242,"python.exe",3<<30)],ancestors=(4242,))
        self.assertNotIn("foreign research/model process has >=512MiB RSS",record["blockers"])


class FreezeTests(unittest.TestCase):
    def test_empty_freeze_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"freeze.json"
            path.write_text(json.dumps({"files":{}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                validate_freeze(path,sha256(path))
            with self.assertRaisesRegex(ValueError,"digest required"):
                validate_freeze(path,"0"*64)

    def test_eos_ids_match_the_pinned_generation_config(self):
        config=json.loads((SOURCES["BASE"][0].parent/"generation_config.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["eos_token_id"]),set(worker.EOS_TOKEN_IDS))

    def test_lease_labels_agree(self):
        import supervise
        self.assertEqual(supervise.LEASE_LABEL,worker.LEASE_LABEL)


if __name__=="__main__":unittest.main()
