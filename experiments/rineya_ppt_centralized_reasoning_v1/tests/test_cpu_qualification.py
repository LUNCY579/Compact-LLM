"""Regression controls for incomplete or misleading CPU qualification receipts."""
from __future__ import annotations

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qualify_cpu as qualification


def successful_result(executed_ids, **overrides):
    fields = dict(executed_test_ids=list(executed_ids), testsRun=len(executed_ids),
                  skipped=[], expectedFailures=[], wasSuccessful=lambda: True)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class QualificationGateTests(unittest.TestCase):
    def test_complete_inventory_allows_extra_tests(self):
        ids = sorted(qualification.REQUIRED_TEST_IDS) + ["additional.independent.check"]
        self.assertEqual(qualification.qualification_issues(ids, successful_result(ids)), [])

    def test_missing_mandatory_checks_reject_unrelated_replacements(self):
        required = qualification.REQUIRED_TEST_IDS
        # Same total number of green checks must not hide lost coverage.
        for missing in sorted(required):
            with self.subTest(missing=missing):
                ids = sorted(required - {missing}) + ["additional.independent.check"]
                self.assertEqual(len(ids), len(required))
                issues = qualification.qualification_issues(ids, successful_result(ids))
                self.assertTrue(any("not loaded" in issue and missing in issue for issue in issues))
                self.assertTrue(any("not executed" in issue and missing in issue for issue in issues))

    def test_loaded_but_unexecuted_checks_are_rejected(self):
        loaded = sorted(qualification.REQUIRED_TEST_IDS)
        executed = loaded[1:]
        issues = qualification.qualification_issues(loaded, successful_result(executed))
        self.assertTrue(any("not executed" in issue and loaded[0] in issue for issue in issues))
        self.assertFalse(any("not loaded" in issue for issue in issues))

    def test_skips_and_expected_failures_are_rejected(self):
        ids = sorted(qualification.REQUIRED_TEST_IDS)
        for kind, reason in (("skipped", "skipped tests"),
                             ("expectedFailures", "expected failures")):
            # unittest considers these outcomes successful; qualification cannot.
            result = successful_result(ids, **{kind: [(ids[0], "deliberate control")]})
            self.assertTrue(result.wasSuccessful())
            issues = qualification.qualification_issues(ids, result)
            self.assertTrue(any(reason in issue for issue in issues))

    def test_result_records_executed_test_ids(self):
        check = unittest.FunctionTestCase(lambda: None)
        suite = unittest.TestSuite([unittest.TestSuite([check])])
        self.assertEqual(tuple(qualification.test_ids(suite)), (check.id(),))
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0,
                                        resultclass=qualification.QualificationResult).run(suite)
        self.assertEqual(result.executed_test_ids, [check.id()])
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
