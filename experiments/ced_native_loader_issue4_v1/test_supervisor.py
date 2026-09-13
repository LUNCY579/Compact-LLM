"""Standard-library checks of receipt rejection. Not native-model evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from . import qualify as q


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'junit.xml'

    def xml(self, keys=None):
        keys = sorted(q.EXPECTED) if keys is None else keys
        root = ET.Element('testsuites')
        suite = ET.SubElement(root, 'testsuite', tests=str(len(keys)), failures='0', errors='0', skipped='0')
        for classname, name in keys:
            ET.SubElement(suite, 'testcase', classname=classname, name=name)
        return root, suite

    def check(self, root):
        ET.ElementTree(root).write(self.path)
        return q.inspect_junit(self.path)

    def test_exact_suite_is_accepted(self):
        root, _ = self.xml()
        result = self.check(root)
        self.assertEqual(result['passed'], 32)
        self.assertEqual(result['skipped'], 0)

    def test_skip_is_rejected(self):
        root, suite = self.xml()
        ET.SubElement(suite[0], 'skipped')
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_failure_is_rejected(self):
        root, suite = self.xml()
        ET.SubElement(suite[0], 'failure')
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_error_is_rejected(self):
        root, suite = self.xml()
        ET.SubElement(suite[0], 'error')
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_aggregate_failure_is_rejected(self):
        root, suite = self.xml()
        suite.set('errors', '1')
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_wrong_count_is_rejected(self):
        root, suite = self.xml()
        suite.set('tests', '33')
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_missing_case_is_rejected(self):
        root, _ = self.xml(sorted(q.EXPECTED)[1:])
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_empty_suite_is_rejected(self):
        root, _ = self.xml([])
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_duplicate_case_is_rejected(self):
        keys = sorted(q.EXPECTED)
        root, _ = self.xml(keys + [keys[0]])
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_unknown_case_is_rejected(self):
        root, _ = self.xml(sorted(q.EXPECTED) + [('campaign', 'optimizer')])
        with self.assertRaises(RuntimeError):
            self.check(root)

    def test_malformed_xml_is_rejected(self):
        self.path.write_text('<broken')
        with self.assertRaises(ET.ParseError):
            q.inspect_junit(self.path)

    def test_missing_junit_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            q.inspect_junit(self.path)

    def test_pinned_runtime_is_accepted(self):
        q.require_runtime({'packages': {**q.PINS, 'torch': '2.10.0+cpu'}})

    def test_every_missing_pin_is_blocked(self):
        for name in q.PINS:
            with self.subTest(package=name):
                versions = {**q.PINS, name: None}
                with self.assertRaises(q.Blocked):
                    q.require_runtime({'packages': versions})

    def test_transformers_upgrade_is_blocked(self):
        with self.assertRaises(q.Blocked):
            q.require_runtime({'packages': {**q.PINS, 'transformers': '4.51.4'}})

    def test_cuda_build_is_not_cpu_evidence(self):
        with self.assertRaises(q.Blocked):
            q.require_runtime({'packages': {**q.PINS, 'torch': '2.10.0+cu128'}})

    def test_receipt_cannot_be_overwritten(self):
        path = Path(self.temp.name) / 'receipt.json'
        q.write_once(path, {'status': 'FAIL'})
        with self.assertRaises(FileExistsError):
            q.write_once(path, {'status': 'PASS'})

    def test_output_inside_checkout_is_rejected(self):
        with self.assertRaises(q.Blocked):
            q.qualify('0' * 40, q.ROOT / 'forbidden-test-output')

    def test_invalid_commit_is_rejected_before_git(self):
        with patch.object(q, 'git') as call:
            with self.assertRaises(q.Blocked):
                q.source_identity('shortsha')
            call.assert_not_called()

    def test_all_claim_permissions_stay_false(self):
        self.assertTrue(q.PERMISSIONS)
        self.assertTrue(all(value is False for value in q.PERMISSIONS.values()))


if __name__ == '__main__':
    unittest.main()
