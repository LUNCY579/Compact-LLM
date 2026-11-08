from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ..__main__ import main
from ..artifacts import ROOT, strict_json, write_json
from ..demo import run
from ..planning import SMALL, Q4, Q8, Shape, bridge_parameters, graft_budget, plan, stitch_budget
from ..qualification import verify_source
from .test_artifacts_evaluation import panels
from dataclasses import asdict


class PlanningTests(unittest.TestCase):
    def test_original_parameter_counts(self):
        self.assertEqual(SMALL.model_parameters, 596049920)
        self.assertEqual(Q4.model_parameters, 4022468096)
        self.assertEqual(Q8.model_parameters, 8190735360)

    def test_no_hidden_width_inference_from_heads(self):
        self.assertNotEqual(SMALL.width // SMALL.q_heads, SMALL.head_dim)
        self.assertEqual(SMALL.head_dim, 128)

    def test_graft_keeps_no_extra_kv_bank(self):
        b = graft_budget(SMALL, Q4)
        self.assertEqual(b["extra_attention_kv_banks"], 0)
        self.assertEqual(b["added_frozen_parameters"], Q4.ffn_parameters + Q4.width)
        self.assertFalse(b["allocation_or_speed_measured"])

    def test_return_delta_map_has_no_bias_cost(self):
        b = graft_budget(SMALL, Q4)
        self.assertEqual(b["interface_parameters"], 2 * 1024 * 2560 + 2560)

    def test_cross_stitch_retains_two_tables(self):
        b = stitch_budget(Q4, SMALL, 18, 14)
        expected = (151936 * (2560 + 1024) + 18 * Q4.block_parameters + 14 * SMALL.block_parameters +
                    1024 + bridge_parameters(2560, 1024))
        self.assertEqual(b["total_parameters"], expected)
        self.assertFalse(b["cache_implemented"])

    def test_identity_control_tied_table_deduplicated(self):
        b = stitch_budget(SMALL, SMALL, 14, 14, same_source_identity_control=True)
        self.assertEqual(b["total_parameters"], SMALL.model_parameters + bridge_parameters(1024, 1024))

    def test_different_sources_cannot_claim_identity_deduplication(self):
        with self.assertRaises(ValueError):
            stitch_budget(Q4, SMALL, 18, 14, same_source_identity_control=True)

    def test_all_execution_gates_closed(self):
        p = plan()
        for k in ("native_ready", "backbone_recovery_authorized", "donor_download_authorized", "sealed_access_authorized"):
            self.assertFalse(p[k])

    def test_shape_bool_rejected(self):
        with self.assertRaises(ValueError):
            replace(SMALL, width=True)

    def test_gqa_geometry_rejected(self):
        with self.assertRaises(ValueError):
            replace(SMALL, q_heads=15)


class WorkflowTests(unittest.TestCase):
    def test_source_manifest_matches(self):
        self.assertGreater(len(verify_source()["test_ids"]), 100)

    def test_tampered_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "evidence"))
            (root / "planning.py").write_text("# tampered\n")
            with self.assertRaisesRegex(ValueError, "inventory/hash"):
                verify_source(root)

    def test_unlisted_source_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "evidence"))
            (root / "unexpected.py").write_text("# extra\n")
            with self.assertRaisesRegex(ValueError, "inventory/hash"):
                verify_source(root)

    def test_cli_plan_is_nonexecuting(self):
        out = io.StringIO()
        with redirect_stdout(out): main(["plan"])
        self.assertFalse(json.loads(out.getvalue())["native_ready"])

    def test_demo_fit_and_score_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run(root / "demo")
            self.assertEqual(result["pretrained_model_calls"], 0)
            self.assertFalse(result["efficacy_claim"])
            self.assertTrue(result["backbone_tensors_unchanged"])
            self.assertEqual(result["fixture_optimizer_updates"], 4)
            output = io.StringIO()
            with redirect_stdout(output):
                main(["fit", "--request", str(root / "demo" / "fit_request.json"), "--output", str(root / "refit")])
            self.assertFalse(json.loads(output.getvalue())["native_qualification"])
            a, b = panels()
            write_json(root / "score-request.json", {"receiver": asdict(a), "candidate": asdict(b)})
            with redirect_stdout(io.StringIO()):
                main(["score", "--request", str(root / "score-request.json"), "--output", str(root / "score")])
            self.assertEqual(strict_json(root / "score" / "SCORE.json")["paired_gain"], 0.125)

    def test_native_cli_command_absent(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["train-native"])
