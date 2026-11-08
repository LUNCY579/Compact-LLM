"""Independent toy effects and invalid-pair controls; standard library only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.post_training_mimo_centralized_v1.scripts.paired_analysis import analyze, main


def row(parent, seed, base, candidate, family="f", **extra):
    return dict(parent_id=parent, family=family, seed=seed, base_correct=base,
                candidate_correct=candidate, **extra)


def document(rows, role="DEV"):
    return {"role": role, "rows": rows}


class PairedAnalysisTests(unittest.TestCase):
    def test_repeated_seeds_are_one_parent_cluster_not_independent_rows(self):
        single = document([row("a", 101, False, True), row("b", 101, True, False),
                           row("c", 101, False, True), row("d", 101, False, False)])
        repeated = document([dict(r, seed=seed) for r in single["rows"] for seed in (101, 211, 307)])
        left = analyze(single, bootstrap_samples=400, bootstrap_seed=29)
        right = analyze(repeated, bootstrap_samples=400, bootstrap_seed=29)
        self.assertEqual(left["aggregate"]["accuracy_delta"], .25)
        self.assertEqual(right["aggregate"]["independent_parent_count"], 4)
        self.assertEqual(right["aggregate"]["paired_seed_observations"], 12)
        self.assertEqual(left["aggregate"]["descriptive_intervals"], right["aggregate"]["descriptive_intervals"])
        self.assertFalse(right["noninferiority_certificate"])

    def test_parent_vector_keeps_opposite_seed_outcomes_together(self):
        data = document([row("a", 101, False, True), row("a", 211, True, False),
                         row("b", 101, True, False), row("b", 211, False, True)])
        result = analyze(data, bootstrap_samples=200)
        interval = result["aggregate"]["descriptive_intervals"]["accuracy_delta"]
        self.assertEqual((interval["low"], interval["high"]), (0, 0))
        self.assertEqual(result["aggregate"]["source_correct_losses"], 2)
        self.assertEqual(result["aggregate"]["source_incorrect_repairs"], 2)

    def test_retention_denominators_and_zero_opportunities_are_explicit(self):
        gained = analyze(document([row("a", 101, False, True)]), bootstrap_samples=30)["aggregate"]
        self.assertEqual(gained["source_correct_denominator"], 0)
        self.assertIsNone(gained["source_correct_loss_rate"])
        self.assertIsNone(gained["source_correct_retention_rate"])
        self.assertIsNone(gained["descriptive_intervals"]["source_correct_loss_rate"])
        self.assertEqual(gained["source_incorrect_repair_rate"], 1)
        lost = analyze(document([row("b", 101, True, False)]), bootstrap_samples=30)["aggregate"]
        self.assertEqual(lost["source_incorrect_denominator"], 0)
        self.assertIsNone(lost["source_incorrect_repair_rate"])
        self.assertEqual(lost["source_correct_loss_rate"], 1)

    def test_positive_toy_effect_and_family_regression_remain_visible(self):
        data = document([row(f"gain{i}", 101, False, True, family="gain") for i in range(3)] +
                        [row("loss", 101, True, False, family="loss")])
        result = analyze(data, bootstrap_samples=50)
        self.assertEqual(result["aggregate"]["accuracy_delta"], .5)
        self.assertGreater(result["aggregate"]["descriptive_intervals"]["accuracy_delta"]["low"], 0)
        loss = next(f for f in result["per_family"] if f["family"] == "loss")
        self.assertEqual(loss["accuracy_delta"], -1)
        self.assertEqual(loss["source_correct_losses"], 1)
        self.assertEqual(loss["per_seed"][0]["accuracy_delta"], -1)

    def test_duplicate_parent_seed_and_missing_pair_rejected(self):
        item = row("a", 101, False, True)
        with self.assertRaisesRegex(ValueError, "duplicate parent"):
            analyze(document([item, copy.deepcopy(item)]), bootstrap_samples=10)
        missing = dict(item)
        del missing["candidate_correct"]
        with self.assertRaisesRegex(ValueError, "missing paired"):
            analyze(document([missing]), bootstrap_samples=10)

    def test_unequal_seed_coverage_or_changed_family_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            analyze(document([row("a", 101, False, True), row("a", 211, False, True),
                              row("b", 101, False, True)]), bootstrap_samples=10)
        with self.assertRaisesRegex(ValueError, "family changed"):
            analyze(document([row("a", 101, False, True, family="f"),
                              row("a", 211, False, True, family="g")]), bootstrap_samples=10)

    def test_strict_bool_nonfinite_and_nondev_roles_rejected(self):
        for value in (0, 1, "false", None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze(document([row("a", 101, value, True)]), bootstrap_samples=10)
        for role in ("CHECK", "FINAL", "TRAIN", "DEV/CAL", None):
            with self.subTest(role=role), self.assertRaises(ValueError):
                analyze(document([row("a", 101, False, True)], role=role), bootstrap_samples=10)
        for confidence in (float("nan"), float("inf"), True, 0, 1):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                analyze(document([row("a", 101, False, True)]), confidence=confidence)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analyze({**document([row("a", 101, False, True)]), "metadata": {"cost": float("inf")}})

    def test_missing_optional_panels_are_null_and_partial_panels_rejected(self):
        data = document([row("a", 101, False, True), row("b", 101, True, True)])
        result = analyze(data, bootstrap_samples=10)
        self.assertEqual(result["aggregate"]["optional_panels"], {"malformed": None, "eos_failure": None})
        data["rows"][0].update(base_malformed=False, candidate_malformed=True)
        with self.assertRaisesRegex(ValueError, "partial panel"):
            analyze(data, bootstrap_samples=10)
        del data["rows"][0]["candidate_malformed"]
        with self.assertRaisesRegex(ValueError, "missing pair"):
            analyze(data, bootstrap_samples=10)

    def test_measured_optional_panel_does_not_disappear(self):
        data = document([row("a", 101, False, True, base_malformed=True, candidate_malformed=False,
                              base_eos_failure=False, candidate_eos_failure=True)])
        panels = analyze(data, bootstrap_samples=10)["aggregate"]["optional_panels"]
        self.assertEqual(panels["malformed"]["candidate_minus_base"], -1)
        self.assertEqual(panels["eos_failure"]["candidate_minus_base"], 1)

    def test_cli_is_write_once_and_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            source, target = Path(directory)/"input.json", Path(directory)/"result.json"
            source.write_text(json.dumps(document([row("a", 101, False, True)])))
            args = ["--input", str(source), "--output", str(target), "--bootstrap-samples", "10"]
            self.assertEqual(main(args), 0)
            old = target.read_bytes()
            self.assertEqual(main(args), 2)
            self.assertEqual(old, target.read_bytes())
            self.assertEqual(len(json.loads(old)["input_sha256"]), 64)
            source.write_text('{"role":"DEV","rows":[],"unexpected":NaN}')
            self.assertEqual(main(["--input", str(source), "--output", str(Path(directory)/"bad.json")]), 2)
            self.assertFalse((Path(directory)/"bad.json").exists())


if __name__ == "__main__":
    unittest.main()
