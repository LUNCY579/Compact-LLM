import copy
import unittest

from measurement_enrollment import check_enrollment, digest, freeze_enrollment


def fixture():
    rows = []
    outcomes = {}
    for source in ("arithmetic", "retrieval", "abstention", "dialogue", "long_context"):
        for ordinal in range(2):
            key = f"fixture:{source}:{ordinal}"
            rows.append({"key": key, "parent_id": key + ":parent", "source": source,
                "role": "retention", "stratum": "fixture", "rendering": "fixture",
                "prompt_hash": digest(key)})
            outcomes[key] = {"correct": ordinal == 0, "format_correct": True, "abstained": False, "capped": False}
    return {"rows": rows}, outcomes, ["arithmetic", "retrieval", "abstention", "dialogue", "long_context"]


class EnrollmentTests(unittest.TestCase):
    def test_keeps_all_rows_and_correct_subset_separately(self):
        cohort, baseline, sources = fixture()
        enrollment = freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)
        result = check_enrollment(enrollment, cohort, baseline, baseline_id="fixture:M0", required_sources=sources)
        self.assertEqual(len(result["complete_row_ids"]), 10)
        for group in result["guards"].values():
            self.assertEqual(len(group["all_guard_row_ids"]), 2)
            self.assertEqual(len(group["prior_correct_row_ids"]), 1)
        self.assertFalse(result["native_membership_admitted"])

    def test_zero_support_remains_undefined(self):
        cohort, baseline, sources = fixture()
        for key in baseline:
            if key.startswith("fixture:dialogue:"):
                baseline[key]["correct"] = False
        result = freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)["payload"]
        self.assertEqual(result["guards"]["dialogue"]["status"], "UNDEFINED_ZERO_SOURCE_CORRECT_SUPPORT")

    def test_missing_baseline_row_refuses(self):
        cohort, baseline, sources = fixture()
        del baseline[next(iter(baseline))]
        with self.assertRaisesRegex(ValueError, "exact complete"):
            freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)

    def test_no_retroactive_guard_subset(self):
        cohort, baseline, sources = fixture()
        enrollment = freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)
        changed = copy.deepcopy(cohort)
        changed["rows"] = changed["rows"][1:]
        smaller = {k: v for k, v in baseline.items() if k != cohort["rows"][0]["key"]}
        with self.assertRaisesRegex(ValueError, "membership drift"):
            check_enrollment(enrollment, changed, smaller, baseline_id="fixture:M0", required_sources=sources)

    def test_source_success_change_refuses(self):
        cohort, baseline, sources = fixture()
        enrollment = freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)
        baseline[next(iter(baseline))]["correct"] = False
        with self.assertRaisesRegex(ValueError, "membership drift"):
            check_enrollment(enrollment, cohort, baseline, baseline_id="fixture:M0", required_sources=sources)

    def test_bad_boolean_and_duplicate_key_refuse(self):
        cohort, baseline, sources = fixture()
        baseline[next(iter(baseline))]["correct"] = 1
        with self.assertRaisesRegex(ValueError, "typed"):
            freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)
        cohort, baseline, sources = fixture()
        cohort["rows"].append(dict(cohort["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)

    def test_parent_role_collision_refuses(self):
        cohort, baseline, sources = fixture()
        cohort["rows"][2]["parent_id"] = cohort["rows"][0]["parent_id"]
        with self.assertRaisesRegex(ValueError, "parent crosses"):
            freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources)

    def test_no_undeclared_guard_dropping(self):
        cohort, baseline, sources = fixture()
        with self.assertRaisesRegex(ValueError, "silently excluded"):
            freeze_enrollment(cohort, baseline, baseline_id="fixture:M0", required_sources=sources[:-1])


if __name__ == "__main__":
    unittest.main()
