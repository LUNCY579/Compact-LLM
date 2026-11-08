from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
import torch

from ..alignment import fit_bridge
from ..artifacts import (ROOT, file_sha256, load_bridge, load_capture, new_directory,
                         save_bridge, save_capture, strict_json, write_json)
from ..contracts import digest
from ..evaluation import Outcome, Panel, paired_report
from ..fixtures import identity
from .test_alignment import bank, affine_banks


def panels():
    rows = tuple(Outcome(str(i), str(i // 2), "a" if i < 4 else "b", digest(i), digest([i, "ids"]), i < 4,
                         True, i == 7, True) for i in range(8))
    other = tuple(replace(r, correct=(i < 5)) for i, r in enumerate(rows))
    shared = dict(registration_sha256=digest("registered"), scorer_sha256=digest("scorer"),
                  decoding_sha256=digest("decode"), role="SELECT", expected_ids=tuple(str(i) for i in range(8)))
    return (Panel(identity("a"), **shared, outcomes=rows),
            Panel(identity("b"), **shared, outcomes=other))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_capture_roundtrip(self):
        original = bank()
        sha = save_capture(original, self.root / "bank")
        restored = load_capture(self.root / "bank", sha)
        self.assertEqual(original.contract, restored.contract)
        self.assertTrue(torch.equal(original.values, restored.values))

    def test_bridge_roundtrip(self):
        a, b, c, d = affine_banks()
        fit, report = fit_bridge(a, b, c, d)
        sha = save_bridge(fit, report, a, b, self.root / "bridge")
        loaded = load_bridge(self.root / "bridge", sha, source=a.identity, target=b.identity,
                             source_site=a.site, target_site=b.site, registration_sha256=a.registration_sha256)
        self.assertTrue(torch.equal(loaded(a.values), fit(a.values)))
        self.assertEqual(loaded.count, fit.count)

    def test_bridge_wrong_target_refused(self):
        a, b, c, d = affine_banks()
        fit, report = fit_bridge(a, b, c, d)
        sha = save_bridge(fit, report, a, b, self.root / "bridge")
        with self.assertRaisesRegex(ValueError, "target_identity"):
            load_bridge(self.root / "bridge", sha, source=a.identity, target=identity("wrong"),
                        source_site=a.site, target_site=b.site, registration_sha256=a.registration_sha256)

    def test_capture_payload_tampering(self):
        sha = save_capture(bank(), self.root / "bank")
        path = self.root / "bank" / "values.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "payload digest"):
            load_capture(self.root / "bank", sha)

    def test_capture_manifest_tampering(self):
        sha = save_capture(bank(), self.root / "bank")
        path = self.root / "bank" / "manifest.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            load_capture(self.root / "bank", sha)

    def test_payload_budget(self):
        sha = save_capture(bank(), self.root / "bank")
        with self.assertRaisesRegex(ValueError, "budget"):
            load_capture(self.root / "bank", sha, max_bytes=1)

    def test_manifest_shape_lie(self):
        save_capture(bank(), self.root / "bank")
        path = self.root / "bank" / "manifest.json"
        data = json.loads(path.read_text())
        data["shape"] = [24, 4]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            load_capture(self.root / "bank", file_sha256(path))

    def test_manifest_unknown_field(self):
        save_capture(bank(), self.root / "bank")
        path = self.root / "bank" / "manifest.json"
        data = json.loads(path.read_text())
        data["promote"] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_capture(self.root / "bank", file_sha256(path))

    def test_partial_artifact_refused(self):
        (self.root / "partial").mkdir()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_capture(self.root / "partial", "0" * 64)

    def test_overwrite_refused(self):
        sha = save_capture(bank(), self.root / "bank")
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_capture(bank(), self.root / "bank")
        load_capture(self.root / "bank", sha)

    def test_source_directory_output_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            new_directory(ROOT / "must-not-exist-test-output")

    def test_duplicate_json_refused(self):
        path = self.root / "bad.json"
        path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            strict_json(path)

    def test_nonfinite_json_refused(self):
        path = self.root / "bad.json"
        path.write_text('{"a": NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            strict_json(path)

    def test_write_json_no_overwrite(self):
        path = self.root / "data.json"
        write_json(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            write_json(path, {"a": 2})
        self.assertEqual(strict_json(path), {"a": 1})

    def test_symlink_payload_refused(self):
        sha = save_capture(bank(), self.root / "bank")
        payload = self.root / "bank" / "values.safetensors"
        moved = self.root / "elsewhere"
        payload.rename(moved)
        payload.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "symlink"):
            load_capture(self.root / "bank", sha)


class EvaluationTests(unittest.TestCase):
    def test_headroom_counts(self):
        a, b = panels()
        report = paired_report(a, b)
        self.assertEqual(report["paired_counts"], {"both_correct": 4, "candidate_only": 1, "receiver_only": 0, "both_wrong": 3})
        self.assertEqual(report["paired_gain"], 1 / 8)
        self.assertFalse(report["automatic_promotion"])
        self.assertFalse(report["functional_transfer_established"])

    def test_row_order_not_assumed(self):
        a, b = panels()
        self.assertEqual(paired_report(a, b), paired_report(a, replace(b, outcomes=tuple(reversed(b.outcomes)))))

    def test_bootstrap_seed_reproducible(self):
        a, b = panels()
        self.assertEqual(paired_report(a, b, seed=12), paired_report(a, b, seed=12))

    def test_single_group_ci_undefined(self):
        a, b = panels()
        a = replace(a, outcomes=tuple(replace(r, group_id="one") for r in a.outcomes))
        b = replace(b, outcomes=tuple(replace(r, group_id="one") for r in b.outcomes))
        self.assertIsNone(paired_report(a, b)["cluster_bootstrap_95pct"])

    def test_null_has_no_discordance(self):
        a, _ = panels()
        r = paired_report(a, a)
        self.assertEqual(r["paired_gain"], 0)
        self.assertEqual(r["row_mcnemar_exact_p_descriptive_only"], 1)

    def test_unknown_and_format_regression_separate(self):
        a, b = panels()
        b = replace(b, outcomes=b.outcomes[:-1] + (replace(b.outcomes[-1], format_ok=False, literal_unknown_ok=False),))
        r = paired_report(a, b)
        self.assertEqual(r["unknown_cases"], 1)
        self.assertEqual(r["literal_unknown_regressions"], 1)
        self.assertEqual(r["format_regressions"], 1)

    def test_zero_unknown_denominator_explicit(self):
        a, b = panels()
        a = replace(a, outcomes=tuple(replace(r, unknown_expected=False) for r in a.outcomes))
        b = replace(b, outcomes=tuple(replace(r, unknown_expected=False) for r in b.outcomes))
        self.assertEqual(paired_report(a, b)["unknown_cases"], 0)

    def test_family_results_retained(self):
        a, b = panels()
        self.assertEqual(paired_report(a, b)["family_paired_gain"], {"a": 0, "b": 0.25})


def bad_panel_test(change):
    def test(self):
        a, b = panels()
        with self.assertRaises((ValueError, TypeError)):
            paired_report(a, change(b))
    return test


for name, change in {
    "missing_row": lambda p: replace(p, outcomes=p.outcomes[:-1]),
    "duplicate_row": lambda p: replace(p, outcomes=p.outcomes + (p.outcomes[0],)),
    "missing_expected_id": lambda p: replace(p, expected_ids=p.expected_ids[:-1]),
    "empty": lambda p: replace(p, expected_ids=(), outcomes=()),
    "error_row": lambda p: replace(p, outcomes=(replace(p.outcomes[0], status="error"),) + p.outcomes[1:]),
    "integer_bool": lambda p: replace(p, outcomes=(replace(p.outcomes[0], correct=1),) + p.outcomes[1:]),
    "scorer": lambda p: replace(p, scorer_sha256=digest("different")),
    "decode": lambda p: replace(p, decoding_sha256=digest("different")),
    "registration": lambda p: replace(p, registration_sha256=digest("different")),
    "role": lambda p: replace(p, role="CHECK"),
    "sealed_role": lambda p: replace(p, role="SEALED"),
    "precision": lambda p: replace(p, model=replace(p.model, precision="float16")),
    "template": lambda p: replace(p, model=replace(p.model, template_sha256=digest("different"))),
    "group_metadata": lambda p: replace(p, outcomes=(replace(p.outcomes[0], group_id="different"),) + p.outcomes[1:]),
    "token_ids_metadata": lambda p: replace(p, outcomes=(replace(p.outcomes[0], input_ids_sha256=digest("different")),) + p.outcomes[1:]),
    "prompt_metadata": lambda p: replace(p, outcomes=(replace(p.outcomes[0], prompt_sha256=digest("different")),) + p.outcomes[1:]),
    "unknown_metadata": lambda p: replace(p, outcomes=(replace(p.outcomes[0], unknown_expected=True),) + p.outcomes[1:]),
}.items():
    setattr(EvaluationTests, "test_refuses_" + name, bad_panel_test(change))


class ResourceGuardTests(unittest.TestCase):
    def test_low_ram_refused_before_payload_loading(self):
        from ..artifacts import preflight_fit_request
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = save_capture(bank(), root / "capture")
            spec = {k: {"directory": "capture", "manifest_sha256": sha}
                    for k in ("source", "target", "check_source", "check_target")}
            with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=1)):
                with self.assertRaisesRegex(ValueError, "insufficient"):
                    preflight_fit_request(root / "request.json", spec)

    def test_oversized_capture_refused_before_materialization(self):
        from ..artifacts import preflight_fit_request
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = save_capture(bank(), root / "capture")
            spec = {k: {"directory": "capture", "manifest_sha256": sha}
                    for k in ("source", "target", "check_source", "check_target")}
            # Sparse file: do not allocate 70 MiB of real payload data in this test.
            with (root / "capture" / "values.safetensors").open("r+b") as f:
                f.truncate(73400320)
            with self.assertRaisesRegex(ValueError, "combined capture"):
                preflight_fit_request(root / "request.json", spec)
