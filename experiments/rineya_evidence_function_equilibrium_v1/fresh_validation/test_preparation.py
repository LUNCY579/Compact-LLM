import math
from pathlib import Path
import tempfile
import unittest
from .resources import LIMITS, blockers, dual_lease
from .statistics import binomial_cdf, cp_lower, cp_upper, damage_power, paired, verdict, registered_gates


class StatisticsTest(unittest.TestCase):
    def test_exact_known_tail(self):
        self.assertAlmostEqual(binomial_cdf(2, 4, .5), 11 / 16)
        self.assertAlmostEqual(cp_upper(0, 100, .05), 1 - .05 ** .01)
        self.assertAlmostEqual(cp_lower(100, 100, .05), .05 ** .01)

    def test_equal_totals_do_not_prove_preservation(self):
        value = paired([True] * 20 + [False] * 20,
                       [True] * 19 + [False, True] + [False] * 19)
        self.assertEqual((value["gains"], value["losses"], value["difference"]), (1, 1, 0))
        self.assertLess(value["difference_lower"], -.03)
        self.assertGreater(value["conditional_damage_upper"], .03)

    def test_missing_pairs_and_nonbooleans_refused(self):
        for a, b in (([], []), ([True], []), ([1], [True])):
            with self.assertRaises(ValueError):
                paired(a, b)

    def test_all_gains_and_damage_are_detectable(self):
        value = paired([False] * 100, [True] * 100)
        self.assertGreater(value["difference_lower"], .05)
        damaged = paired([True] * 100, [False] * 100)
        self.assertLess(damaged["difference_lower"], -.9)
        self.assertEqual(damaged["conditional_damage_upper"], 1)

    def test_conditional_denominator_sensitivity(self):
        value = damage_power(900)
        self.assertGreater(value["power"], .8)
        self.assertLess(value["zero_loss_upper"], .03)
        self.assertGreater(cp_upper(0, 47, .005), .03)

    def test_unresolved_limb_never_becomes_a_negative_clause(self):
        gates = registered_gates()
        passing = {"directional_family_alpha": .025, "claims": 10, "n": 20160,
                   "difference_lower": .06, "difference_upper": .1, "comparator_correct": 1000,
                   "conditional_damage_upper": .02, "conditional_damage_lower": 0}
        values = {g["id"]: dict(passing) for g in gates}
        self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "PROCEED")
        values["G1"].update(difference_lower=.04, difference_upper=.06)
        self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "HOLD")
        values["G1"].update(difference_upper=.049)
        self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "REJECT")
        values.pop("G2")
        self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "HOLD")
        self.assertEqual(verdict(values, gates, prerequisites_complete=False)["decision"], "HOLD")

    def test_duplicate_gates_changed_thresholds_and_truthy_strings_refused(self):
        gates = registered_gates()
        passing = {"directional_family_alpha": .025, "claims": 10, "n": 20160,
                   "difference_lower": .06, "difference_upper": .1, "comparator_correct": 1000,
                   "conditional_damage_upper": .02, "conditional_damage_lower": 0}
        values = {g["id"]: dict(passing) for g in gates}
        self.assertEqual(verdict({"G1": passing}, [gates[0]] * 10,
                                 prerequisites_complete=True)["decision"], "HOLD")
        self.assertEqual(verdict(values, gates, prerequisites_complete="false")["decision"], "HOLD")
        for invalid_upper in (.01, float("nan"), float("inf"), 1.01, True):
            values["G1"] = {**passing, "difference_upper": invalid_upper}
            self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "HOLD")
        values["G1"] = dict(passing)
        gates[0] = {**gates[0], "threshold": -.03}
        self.assertEqual(verdict(values, gates, prerequisites_complete=True)["decision"], "HOLD")
        import json
        from .preflight import PACKAGE
        protocol = json.loads((PACKAGE / "protocol.json").read_text())
        self.assertEqual(protocol["primary_gates"], registered_gates())


class ResourceTest(unittest.TestCase):
    def test_second_lease_failure_keeps_foreign_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = (Path(directory) / "central.lock", Path(directory) / "temp.lock")
            paths[1].write_text("foreign")
            with self.assertRaises(FileExistsError):
                with dual_lease(paths):
                    self.fail("must not enter")
            self.assertFalse(paths[0].exists())
            self.assertEqual(paths[1].read_text(), "foreign")

    def test_release_checks_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = (Path(directory) / "central.lock", Path(directory) / "temp.lock")
            with dual_lease(paths):
                paths[1].write_text("new-owner")
            self.assertFalse(paths[0].exists())
            self.assertEqual(paths[1].read_text(), "new-owner")

    def test_guards_refuse_low_memory_and_other_lock(self):
        value = {"host_available_bytes": LIMITS["host_admission_bytes"] - 1,
                 "gpu": {"free_bytes": LIMITS["gpu_admission_bytes"]},
                 "disk_free_bytes": LIMITS["disk_floor_bytes"], "process_rss_bytes": 1,
                 "locks": [{"present": True, "holder_text": "foreign", "path": "lock"}],
                 "possible_model_workers": [], "gpu_process_inventory": []}
        reasons = blockers(value)
        self.assertTrue(any(v.startswith("HOST_RAM") for v in reasons))
        self.assertTrue(any(v.startswith("FOREIGN_LOCK") for v in reasons))

    def test_own_gpu_process_is_exempt_foreign_worker_is_not(self):
        value = {"host_available_bytes": LIMITS["host_admission_bytes"],
                 "gpu": {"free_bytes": LIMITS["gpu_admission_bytes"]},
                 "disk_free_bytes": LIMITS["disk_floor_bytes"], "process_rss_bytes": 1,
                 "locks": [], "possible_model_workers": [], "own_pid": 123,
                 "gpu_process_inventory": ["123, python.exe"]}
        self.assertEqual(blockers(value), [])
        value["gpu_process_inventory"].append("456, python.exe")
        self.assertIn("FOREIGN_GPU_MODEL_PROCESS", blockers(value))
        value["gpu_error"] = "process query failed"
        self.assertTrue(any(v.startswith("GPU_INVENTORY_UNAVAILABLE") for v in blockers(value)))


if __name__ == "__main__":
    unittest.main()
