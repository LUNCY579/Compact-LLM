"""Adversarial provenance tests; no model, tokenizer or sealed endpoints."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import receipts as r


def arm():
    value = {k: r.digest(k) for k in r.COMMON_KEYS + r.ARM_KEYS if k.endswith("sha256") or k == "implementation_digest"}
    value.update(basis_run_id="protected-A", rank_budget=4, parameter_budget=128,
                 optimizer_steps=8, training_run_id="fixture-A", alpha=1.0, completed=True)
    return value


def calibration():
    rows = [{"id": "fit-1", "parent_id": "parent-fit", "split": "fit", "open": True},
            {"id": "check-1", "parent_id": "parent-check", "split": "map_check", "open": True}]
    return {"dataset_sha256": r.digest("open-source"), "calibration_sha256": r.digest(rows),
            "role": "open_calibration", "rows": rows}


def source_fixture(root):
    root.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (root / "source.txt").write_text("fixed", encoding="utf-8")
    git("add", "source.txt")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "fixture")
    return git("rev-parse", "HEAD")


def endpoint_expected(metadata, path, ids, commit):
    return {**metadata, "endpoint_sha256": r.file_digest(path), "endpoint_row_ids": ids,
            "endpoint_row_ids_sha256": r.digest(ids), "source_commit": commit}


class ReceiptTests(unittest.TestCase):
    def test_required_mismatch_refusals(self):
        # Explicit coverage of wrong base, donor, tokenizer, layer map, calibration,
        # skill dataset, evaluation bank, delta receipt, probe weights and row identity.
        fields = {"wrong_base": "base_checkpoint_sha256", "wrong_donor": "donor_checkpoint_sha256",
                  "wrong_tokenizer": "tokenizer_sha256", "wrong_layer_map": "layer_map_sha256",
                  "missing_calibration_digest": "calibration_sha256", "mismatched_skill_dataset": "dataset_sha256",
                  "mismatched_evaluation_bank": "evaluation_bank_sha256", "wrong_delta_receipt": "skill_delta_sha256",
                  "wrong_probe_weight": "probe_weight_sha256", "wrong_training_rows": "training_rows_sha256"}
        for name, key in fields.items():
            with self.subTest(required_negative=name):
                expected, actual = arm(), arm()
                actual[key] = None if name.startswith("missing") else r.digest("wrong")
                with self.assertRaisesRegex(r.Blocked, "BLOCKED_ON_COMPARABILITY"):
                    r.validate_arm(actual, expected)
        self.assertTrue(r.validate_arm(arm(), arm()))

    def test_gate_before_endpoint_open(self):
        metadata = arm()
        metadata["views_sha256"] = r.digest("another-bank-same-row-count")
        spec = {"metadata": metadata, "path": "must-never-open.json", "sha256": r.digest("payload")}
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoints were opened")):
            with self.assertRaises(r.Blocked):
                r.compare_endpoint_files(spec, spec, arm(), arm(), cross_training_draw=True, source_root="never-open")

    def test_payload_bytes_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            commit = source_fixture(source)
            p = Path(tmp) / "endpoints.json"
            p.write_text('{"row": true}', encoding="utf-8")
            spec = {"metadata": arm(), "path": p, "sha256": r.file_digest(p)}
            expected = endpoint_expected(arm(), p, ["row"], commit)
            report = r.compare_endpoint_files(spec, spec, expected, expected,
                                              cross_training_draw=False, source_root=source)
            self.assertEqual(report["exact_p"], "1")
            p.write_text('{"row": false}', encoding="utf-8")
            with self.assertRaises(r.Blocked):
                r.compare_endpoint_files(spec, spec, expected, expected,
                                         cross_training_draw=False, source_root=source)
            # Rewriting the untrusted sidecar as well must not bypass the frozen output receipt.
            spec["sha256"] = r.file_digest(p)
            with patch.object(Path, "read_text", side_effect=AssertionError("endpoints were opened")):
                with self.assertRaises(r.Blocked):
                    r.compare_endpoint_files(spec, spec, expected, expected,
                                             cross_training_draw=False, source_root=source)

    def test_exact_population_order_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            commit = source_fixture(source)
            p = Path(tmp) / "endpoints.json"
            for content, ids in (('{"a": true}', ["a", "b"]),
                                 ('{"b": false, "a": true}', ["a", "b"]),
                                 ('{"a": true, "a": false}', ["a"])):
                with self.subTest(content=content):
                    p.write_text(content, encoding="utf-8")
                    expected = endpoint_expected(arm(), p, ids, commit)
                    spec = {"metadata": arm(), "path": p, "sha256": r.file_digest(p)}
                    with self.assertRaises(r.Blocked):
                        r.compare_endpoint_files(spec, spec, expected, expected,
                                                 cross_training_draw=False, source_root=source)

    def test_cross_draw_flag_and_source_required_before_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            commit = source_fixture(source)
            p = Path(tmp) / "endpoints.json"
            p.write_text('{"row": true}', encoding="utf-8")
            left_meta, right_meta = arm(), {**arm(), "training_run_id": "other-training-draw"}
            left_expected = endpoint_expected(left_meta, p, ["row"], commit)
            right_expected = endpoint_expected(right_meta, p, ["row"], commit)
            left = {"metadata": left_meta, "path": p, "sha256": r.file_digest(p)}
            right = {"metadata": right_meta, "path": p, "sha256": r.file_digest(p)}
            with patch.object(Path, "read_text", side_effect=AssertionError("endpoints were opened")):
                with self.assertRaises(r.Blocked):
                    r.compare_endpoint_files(left, right, left_expected, right_expected,
                                             cross_training_draw=False, source_root=source)
                missing = {**left_expected, "source_commit": None}
                with self.assertRaises(r.Blocked):
                    r.compare_endpoint_files(left, left, missing, missing,
                                             cross_training_draw=False, source_root=source)

    def test_projection_basis_from_another_run(self):
        basis = {"sha256": r.digest("basis"), "run_id": "run-A", "rank": 1,
                 "scope": ["layer.0"], "normalization": "fixed-base-module-norm",
                 "source_kind": "independent_skill_delta"}
        r.validate_basis(basis, "run-A", basis["sha256"])
        with self.assertRaises(r.Blocked):
            r.validate_basis(basis, "run-B", basis["sha256"])

    def test_norm_control_mismatch(self):
        value = dict(rank_budget=2, parameter_budget=32, target_modules=["q", "v"],
                     optimizer_steps=8, training_rows_sha256=r.digest("rows"), norm=0.5,
                     effective_ranks={"q": 2, "v": 2})
        r.validate_norm_control(value, value)
        for key, bad in (("norm", 0.51), ("rank_budget", 1), ("parameter_budget", 64),
                         ("effective_ranks", {"q": 3, "v": 2})):
            with self.subTest(key=key), self.assertRaises(r.Blocked):
                r.validate_norm_control(value, {**value, key: bad})

    def test_sealed_calibration_refused_even_if_relabelled(self):
        good = calibration()
        r.validate_calibration(good)
        for key, value in (("role", "sealed_evaluation"), ("calibration_sha256", None)):
            with self.assertRaises(r.Blocked):
                r.validate_calibration({**good, key: value})
        with self.assertRaises(r.Blocked):
            r.validate_calibration(good, forbidden_row_ids={"fit-1"})
        with self.assertRaises(r.Blocked):
            r.validate_calibration(good, forbidden_parent_ids={"parent-fit"})
        bad = copy.deepcopy(good)
        bad["rows"][1]["parent_id"] = "parent-fit"
        bad["calibration_sha256"] = r.digest(bad["rows"])
        with self.assertRaises(r.Blocked):
            r.validate_calibration(bad)

    def test_incompatible_base_composition(self):
        a = {**arm(), "target_modules": ["q"], "skill_id": "A", "independent_training": True}
        b = {**a, "skill_id": "B", "training_run_id": "fixture-B"}
        r.validate_composition(a, b)
        b["base_checkpoint_sha256"] = r.digest("other-base")
        with self.assertRaises(r.Blocked):
            r.validate_composition(a, b)

    def test_stale_missing_dirty_and_untracked_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", tmp, *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            (root / "source.txt").write_text("fixed", encoding="utf-8")
            git("add", "source.txt")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "fixture")
            commit = git("rev-parse", "HEAD")
            r.verify_source(root, commit)
            for value in (None, "0" * 40):
                with self.assertRaises(r.Blocked):
                    r.verify_source(root, value)
            (root / "evidence.json").write_text("untracked", encoding="utf-8")
            with self.assertRaises(r.Blocked):
                r.verify_source(root, commit)
            (root / "evidence.json").unlink()
            (root / "source.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaises(r.Blocked):
                r.verify_source(root, commit)

    def test_wrapper_complete_after_failed_child(self):
        with self.assertRaises(r.Blocked):
            r.complete_child(1, {"status": "COMPLETE", "completed": True})
        with self.assertRaises(r.Blocked):
            r.complete_child(0, {"status": "FAILED", "completed": False})
        self.assertEqual(r.complete_child(0, {"status": "COMPLETE", "completed": True})["status"], "COMPLETE")

    def test_capture_schema(self):
        cap = {k: r.digest(k) for k in r.CAPTURE_KEYS}
        cap.update(role="donor_base", layer_ids=[0, 18, 35], pooling="last_nonpad_prompt_token",
                   eval_mode=True, optimization_steps=0, dtype="float32", padding_excluded=True,
                   anchor_checkpoint_sha256=cap["model_sha256"])
        r.validate_capture(cap, cap)
        with self.assertRaises(r.Blocked):
            r.validate_capture({**cap, "model_sha256": r.digest("skilled-not-base")}, cap)

    def test_exact_statistics_and_composition_definition(self):
        a, b = {str(k): True for k in range(19)}, {str(k): False for k in range(19)}
        report = r.paired_report(a, b, cross_training_draw=True)
        self.assertEqual((report["net"], report["nd"]), (19, 19))
        self.assertEqual(report["exact_p"], "1/262144")
        self.assertFalse(report["directional_claim_floor_met"])
        self.assertTrue(r.paired_report(a, b, cross_training_draw=False)["directional_claim_floor_met"])
        self.assertEqual(r.composition_losses(a, b, b, b)["I_A_pp"], 100)

    def test_write_once_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r.write_once(root / "receipt.json", {"finite": 1})
            with self.assertRaises(FileExistsError):
                r.write_once(root / "receipt.json", {"finite": 2})
            self.assertEqual(r.implementation_digest(root, ["receipt.json"])["files"]["receipt.json"],
                             r.file_digest(root / "receipt.json"))
            with self.assertRaises(r.Blocked):
                r.implementation_digest(root, ["../outside"])
            with self.assertRaises(ValueError):
                r.digest({"nonfinite": float("nan")})


if __name__ == "__main__":
    unittest.main()
