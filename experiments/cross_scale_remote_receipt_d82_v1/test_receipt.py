"""Stdlib self-tests for receipt rejection, not XSC model tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location("xsc_d82_receipt", Path(__file__).with_name("receipt.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"junit.xml"
        self.nodes = sorted(r.NATIVE)+[f"{r.SCOPE}/test_xsc.py::test_fixture[{i}]" for i in range(132)]
        self.root = ET.Element("testsuites")
        self.suite = ET.SubElement(self.root, "testsuite", tests="134", failures="0", errors="0", skipped="0")
        for n in self.nodes:
            cls, name = r.junit_identity(n)
            ET.SubElement(self.suite, "testcase", classname=cls, name=name)

    def result(self, code=0):
        ET.ElementTree(self.root).write(self.path)
        return r.assess_junit(self.path, self.nodes, code)

    def test_exact_inventory_pass(self):
        self.assertTrue(self.result()["passed"])

    def test_nonzero_exit_rejected(self):
        self.assertFalse(self.result(1)["passed"])

    def test_native_skip_rejected(self):
        ET.SubElement(self.suite[0], "skipped")
        self.suite.set("skipped", "1")
        self.assertFalse(self.result()["passed"])

    def test_ordinary_skip_rejected(self):
        ET.SubElement(self.suite[3], "skipped")
        self.suite.set("skipped", "1")
        self.assertFalse(self.result()["passed"])

    def test_failure_rejected(self):
        ET.SubElement(self.suite[0], "failure")
        self.suite.set("failures", "1")
        self.assertFalse(self.result()["passed"])

    def test_error_rejected(self):
        ET.SubElement(self.suite[0], "error")
        self.suite.set("errors", "1")
        self.assertFalse(self.result()["passed"])

    def test_duplicate_case_rejected(self):
        self.suite[1].attrib.update(self.suite[0].attrib)
        self.assertFalse(self.result()["passed"])

    def test_missing_case_rejected(self):
        self.suite.remove(self.suite[-1])
        self.assertFalse(self.result()["passed"])

    def test_extra_case_rejected(self):
        ET.SubElement(self.suite, "testcase", classname="outside", name="extra")
        self.assertFalse(self.result()["passed"])

    def test_summary_forgery_rejected(self):
        self.suite.set("failures", "1")
        self.assertFalse(self.result()["passed"])

    def test_wrong_identity_rejected(self):
        self.suite[8].set("classname", "other.tests")
        self.assertFalse(self.result()["passed"])

    def test_malformed_xml_rejected(self):
        self.path.write_text("<broken>")
        with self.assertRaises(ET.ParseError):
            r.assess_junit(self.path, self.nodes, 0)

    def test_collection_pass(self):
        self.assertEqual(r.validate_collection("\n".join(self.nodes)), self.nodes)

    def test_collection_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            r.validate_collection("\n".join(self.nodes[:-1]+[self.nodes[0]]))

    def test_collection_missing_rejected(self):
        with self.assertRaises(ValueError):
            r.validate_collection("\n".join(self.nodes[:-1]))

    def test_collection_scope_rejected(self):
        with self.assertRaises(ValueError):
            r.validate_collection("\n".join(self.nodes[:-1]+["experiments/other/test.py::test_extra"]))

    def test_native_name_required(self):
        with self.assertRaises(ValueError):
            r.validate_collection("\n".join(self.nodes).replace("test_native_random_qwen[encoder]", "test_fake[encoder]"))

    def test_create_only(self):
        p = Path(self.tmp.name)/"report.json"
        r.write_new(p, {"a": 1})
        with self.assertRaises(FileExistsError):
            r.write_new(p, {"a": 2})
        self.assertIn('"a": 1', p.read_text())

    def test_permissions_closed(self):
        self.assertTrue(r.PERMISSIONS)
        self.assertTrue(all(v is False for v in r.PERMISSIONS.values()))

    def test_wrong_checkout_head_rejected(self):
        p = Path(self.tmp.name).resolve()
        with patch.object(r, "git", side_effect=[str(p), "0"*40]):
            with self.assertRaisesRegex(ValueError, "wrong target HEAD"):
                r.checkout(p)

    def test_dirty_checkout_rejected(self):
        p = Path(self.tmp.name).resolve()
        with patch.object(r, "git", side_effect=[str(p), r.TARGET, " M source.py"]):
            with self.assertRaisesRegex(ValueError, "not clean"):
                r.checkout(p)

    def test_existing_output_rejected(self):
        out = Path(self.tmp.name)/"existing"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            r.run(Path(self.tmp.name)/"subject", out)

    def test_inside_subject_output_rejected(self):
        p = Path(self.tmp.name)
        with self.assertRaisesRegex(ValueError, "outside"):
            r.run(p, p/"output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
