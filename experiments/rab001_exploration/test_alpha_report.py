"""Synthetic alpha curves from saved endpoint fixtures; no new model evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.rab001_exploration import alpha_report as report
from experiments.rab001_exploration import frontier_report as common


class AlphaReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel = [json.loads(line) for line in (common.BASE / "data/monitor_probe.jsonl").read_text().splitlines()]
        cls.context = common.make_context(panel,
            common.read_json(common.BASE / f"outputs/retention-{common.DIGEST[:8]}/B28/retention.json"),
            common.read_json(common.EVIDENCE / "HEADROOM_AUDIT.json"),
            common.read_json(common.EVIDENCE / "PROBE_LABEL_AUDIT.json"),
            common.read_json(common.BASE / "protocol.json")["G1_RETENTION"])
        cls.dose = common.read_json(common.EVIDENCE / f"ARM_DOSE_{report.SPECS['P'][0]}.json")

    def fixture(self):
        dose = self.dose
        body = {"completed": True, "tag": report.SPECS["P"][0], "kind": "p-output-gate", "arm": "P", "mode": "P",
                "endpoint_controls_passed": True, "registered_bare_policy": "permissive", "reference_n": 47, "panel_rows": 150,
                "grid": list(report.GRID), "numerics": {**dose["numerics"], "probe_device": "cpu", "build_device": "cuda"},
                "provenance": {"implementation_digest": common.DIGEST, "seed": 17,
                               "saved_after_tokens": 5233, "saved_after_steps": 41, "checkpoint_sha256": dose["checkpoint"]["sha256"]},
                "endpoint_controls": {"0.0": {"passed": True}, "1.0": {"passed": True}}, "results": {}}
        for alpha in report.GRID:
            point = copy.deepcopy(dose["checkpoints"][0 if alpha == 0.0 else -1])
            summary = common.summarize_checkpoint(point, self.context, True)
            gains = summary["gained_ids"]
            body["results"][str(alpha)] = {**point["retention"]["permissive"], "per_row": point["retention"]["per_row"],
                                           "outputs": point["outputs"], "alpha": alpha, "kind": "p-output-gate",
                                           "gain_strata": summary["gained_by_stratum"],
                                           "gain_ids_by_stratum": {name: [rid for rid in gains if self.context["strata"][rid] == name]
                                                                   for name in ("wrong_answer", "format_only", "runaway")}}
        return body

    def test_valid_synthetic_grid_rescored(self):
        result = report.validate_sweep(self.fixture(), "P", self.context, self.dose)
        self.assertEqual(result["points"]["0.0"]["correct_total"], 47)
        self.assertEqual(result["points"]["1.0"]["correct_total"], 52)

    def test_endpoint_controls_are_independently_verified(self):
        body = self.fixture()
        rid = next(iter(body["results"]["0.0"]["outputs"]))
        out = body["results"]["0.0"]["outputs"][rid]
        out["terminated"] = not out["terminated"]
        with self.assertRaisesRegex(common.InvalidArtifact, "alpha zero"):
            report.validate_sweep(body, "P", self.context, self.dose)

    def test_grid_and_checkpoint_recipe_are_fixed(self):
        body = self.fixture()
        body["grid"].append(.6)
        with self.assertRaisesRegex(common.InvalidArtifact, "grid"):
            report.validate_sweep(body, "P", self.context, self.dose)
        body = self.fixture()
        body["tag"] = "P_other_seed"
        with self.assertRaisesRegex(common.InvalidArtifact, "tag"):
            report.validate_sweep(body, "P", self.context, self.dose)

    def test_wrong_stratum_counts_rejected(self):
        body = self.fixture()
        body["results"]["0.25"]["gain_strata"]["wrong_answer"] += 1
        with self.assertRaisesRegex(common.InvalidArtifact, "strata"):
            report.validate_sweep(body, "P", self.context, self.dose)

    def test_missing_architecture_preserves_all_twelve_tests(self):
        sweep = report.validate_sweep(self.fixture(), "P", self.context, self.dose)
        pairs = report.fixed_pair_report({"P": sweep})
        self.assertEqual(len(pairs), 12)
        self.assertEqual(sum(pair["p_is_missing_placeholder"] for pair in pairs.values()), 6)
        self.assertTrue(all(pair["p_holm"] >= pair["p_raw"] for pair in pairs.values()))

    def test_create_only_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.json"
            path.write_text("preserved")
            with self.assertRaisesRegex(common.InvalidArtifact, "already exists"):
                report.main(["--out", str(path)])
            self.assertEqual(path.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
