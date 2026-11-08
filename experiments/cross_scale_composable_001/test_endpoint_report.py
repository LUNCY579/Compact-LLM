"""Synthetic endpoint custody and relation tests; never reads a real model score."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import endpoint_report as e
from . import receipts as r
from .test_receipts import source_fixture


class EndpointReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.commit = source_fixture(self.source)
        self.paths = {side: self.root / (side + ".json") for side in ("left", "right")}
        self.ids = [f"row-{i}" for i in range(20)]
        for side in self.paths:
            self.paths[side].write_text(json.dumps({row: side == "left" for row in self.ids}), encoding="utf-8")
        self.evaluation = {key: r.digest(key) for key in e.EVALUATION_HASHES}
        self.evaluation.update(endpoint_population_sha256=r.digest(self.ids), views_sha256=e.SELECT_VIEWS_SHA256,
                               use="DEV", instrument_role="ADMITTED_FOR_SELECTION", instrument_id="SELECT-VIEWS-001",
                               endpoint_domain="acquisition", endpoint_name="canonical_correct")
        self.roots = {}
        self.root_ids = {}
        for run in ("S0", "X0", "X_shuffle"):
            value = {key: r.digest(key) for key in e.ROOT_HASHES}
            value.update(base_checkpoint_sha256=self.evaluation["base_checkpoint_sha256"],
                         delta_sha256=r.digest("learned-" + run), run_id=run, source_commit=self.commit,
                         optimizer_steps=64, rank_budget=8, parameter_budget=163840,
                         target_modules=["layer24.q", "layer24.v"], completed=True)
            key = r.digest(value)
            self.roots[key] = value
            self.root_ids[run] = key

    def tearDown(self):
        self.tmp.cleanup()

    def subject(self, side, kind="trained", run="S0"):
        root = self.root_ids[run]
        value = {"kind": kind, "source_commit": self.commit, "evaluation": copy.deepcopy(self.evaluation),
                 "probe_weight_sha256": r.digest("weights-" + side),
                 "subject_receipt_sha256": r.digest("custody-" + side),
                 "endpoint_sha256": r.file_digest(self.paths[side]), "completed": True,
                 "training_roots": [root], "delta_sha256": self.roots[root]["delta_sha256"],
                 "transform_receipt_sha256": None, "terms": []}
        if kind == "base":
            value.update(training_roots=[], delta_sha256=None,
                         probe_weight_sha256=self.evaluation["base_probe_weight_sha256"])
        elif kind == "transform":
            value.update(delta_sha256=r.digest("scaled-" + run), transform_receipt_sha256=r.digest("transform-" + run))
        return value

    def registry(self, relation="independent_training", left=None, right=None):
        return {"schema": 1, "source_commit": self.commit, "evaluation": copy.deepcopy(self.evaluation),
                "endpoint_row_ids": self.ids[:], "training_roots": copy.deepcopy(self.roots),
                "subjects": {"left": left or self.subject("left", run="X0"),
                             "right": right or self.subject("right", run="S0")},
                "contrast": {"relation": relation}}

    def artifacts(self, registry):
        return [{"manifest": copy.deepcopy(registry["subjects"][side]), "path": self.paths[side]}
                for side in ("left", "right")]

    def compare(self, registry, artifacts=None):
        left, right = artifacts or self.artifacts(registry)
        return e.compare_endpoint_files(left, right, registry, r.digest(registry), source_root=self.source)

    def composition_registry(self, anchor_kind="trained"):
        anchor = self.subject("right", kind=anchor_kind, run="S0")
        other = self.subject("left", run="X0")
        combined = self.subject("left")
        combined.update(kind="composition", training_roots=sorted([*anchor["training_roots"], *other["training_roots"]]),
                        delta_sha256=r.digest("composed-delta"),
                        terms=[e.subject_term(anchor), e.subject_term(other)])
        registry = self.registry("fixed_component_composition", combined, anchor)
        registry["contrast"]["fixed_component_sha256"] = r.digest(e.subject_term(anchor))
        return registry

    def test_three_training_draws_and_20_row_floor(self):
        for run in ("S0", "X_shuffle"):
            with self.subTest(control=run):
                registry = self.registry(left=self.subject("left", kind="transform", run="X0"),
                                         right=self.subject("right", kind="transform", run=run))
                registry["contrast"]["matched_root_fields"] = sorted(e.MATCHABLE_ROOT_FIELDS)
                result = self.compare(registry)
                self.assertEqual((result["net"], result["nd"], result["exact_p"]), (20, 20, "1/524288"))
                self.assertTrue(result["acquisition_20_row_floor_applies"])
                self.assertTrue(result["paired"]["cross_draw_floor_cleared"])
                self.assertFalse(result["improvement_established"])
        self.paths["left"].write_text(json.dumps({row: i != 0 for i, row in enumerate(self.ids)}), encoding="utf-8")
        result = self.compare(self.registry())
        self.assertEqual(result["net"], 19)
        self.assertFalse(result["paired"]["cross_draw_floor_cleared"])

    def test_honest_base_anchor_and_same_checkpoint_transform(self):
        base = self.registry("base_anchor", self.subject("left", kind="transform"), self.subject("right", kind="base"))
        self.assertFalse(self.compare(base)["acquisition_20_row_floor_applies"])
        transform = self.registry("same_checkpoint_transform", self.subject("left", kind="transform"), self.subject("right"))
        self.assertFalse(self.compare(transform)["independent_training_roots"])

    def test_fixed_component_composition_keeps_paired_interpretation(self):
        for kind in ("trained", "transform"):
            with self.subTest(kind=kind):
                result = self.compare(self.composition_registry(kind))
                self.assertEqual(result["net"], 20)
                self.assertFalse(result["acquisition_20_row_floor_applies"])
        registry = self.composition_registry()
        registry["subjects"]["left"], registry["subjects"]["right"] = registry["subjects"]["right"], registry["subjects"]["left"]
        for side in ("left", "right"):
            registry["subjects"][side]["endpoint_sha256"] = r.file_digest(self.paths[side])
        self.compare(registry)

    def paired_compositions_registry(self):
        registry = self.composition_registry()
        naive = registry["subjects"]["left"]
        projected = copy.deepcopy(naive)
        projected.update(delta_sha256=r.digest("orthogonal-composition"),
                         probe_weight_sha256=r.digest("orthogonal-composition-weights"),
                         subject_receipt_sha256=r.digest("orthogonal-composition-receipt"),
                         endpoint_sha256=r.file_digest(self.paths["right"]))
        projected["terms"][1].update(delta_sha256=r.digest("projected-B"),
                                     transform_receipt_sha256=r.digest("project-B-against-A"))
        registry["subjects"]["right"] = projected
        registry["contrast"]["relation"] = "paired_compositions"
        return registry

    def test_paired_naive_and_projected_compositions(self):
        result = self.compare(self.paired_compositions_registry())
        self.assertEqual((result["net"], result["nd"]), (20, 20))
        self.assertEqual(result["relation"], "paired_compositions")
        self.assertFalse(result["acquisition_20_row_floor_applies"])

    def test_paired_compositions_reject_changed_fixed_component_and_roots(self):
        for mutation in ("fixed_digest", "fixed_delta", "coefficient", "different_roots"):
            registry = self.paired_compositions_registry()
            right = registry["subjects"]["right"]
            if mutation == "fixed_digest":
                registry["contrast"]["fixed_component_sha256"] = r.digest("unrelated-fixed-component")
            elif mutation == "fixed_delta":
                right["terms"][0].update(delta_sha256=r.digest("changed-A"),
                                         transform_receipt_sha256=r.digest("changed-A-transform"))
            elif mutation == "coefficient":
                right["terms"][0]["coefficient"] = .5
            else:
                replacement = self.root_ids["X_shuffle"]
                right["terms"][1].update(root_sha256=replacement,
                                         delta_sha256=self.roots[replacement]["delta_sha256"],
                                         transform_receipt_sha256=None)
                right["training_roots"] = sorted([self.root_ids["S0"], replacement])
            with self.subTest(mutation=mutation), patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
                with self.assertRaises(r.Blocked):
                    self.compare(registry)

    def test_relation_misclassifications_refuse_before_payload_open(self):
        registries = []
        for relation in ("same_checkpoint_transform", "base_anchor", "fixed_component_composition"):
            registries.append(self.registry(relation))
        registries.append(self.registry("independent_training", self.subject("left", kind="transform"), self.subject("right")))
        registries.append(self.registry("base_anchor", self.subject("left", kind="base"), self.subject("right", kind="base")))
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            for registry in registries:
                with self.subTest(relation=registry["contrast"]["relation"]), self.assertRaises(r.Blocked):
                    self.compare(registry)

    def test_composition_scaled_missing_or_wrong_constituent(self):
        for mutation in ("coefficient", "delta", "omit_root", "single_run", "fixed_digest"):
            registry = self.composition_registry()
            combined = registry["subjects"]["left"]
            if mutation == "coefficient":
                combined["terms"][0]["coefficient"] = .5
            elif mutation == "delta":
                combined["terms"][0].update(delta_sha256=r.digest("other"), transform_receipt_sha256=r.digest("other-transform"))
            elif mutation == "omit_root":
                combined["training_roots"] = combined["training_roots"][:1]
            elif mutation == "single_run":
                combined["training_run_id"] = "invented-composite"
            else:
                registry["contrast"]["fixed_component_sha256"] = r.digest("wrong-fixed-component")
            with self.subTest(mutation=mutation), self.assertRaises(r.Blocked):
                self.compare(registry)

    def test_base_fabricated_training_metadata_refused(self):
        base = self.subject("right", kind="base")
        base["training_rows_sha256"] = r.digest("fake-matching-rows")
        with self.assertRaises(r.Blocked):
            self.compare(self.registry("base_anchor", right=base))

    def test_base_cannot_claim_arbitrary_probe_weights(self):
        base = self.subject("right", kind="base")
        base["probe_weight_sha256"] = r.digest("trained-model-mislabelled-base")
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            with self.assertRaisesRegex(r.Blocked, "qualified base realization"):
                self.compare(self.registry("base_anchor", right=base))

    def test_payload_and_weight_custody_is_frozen(self):
        registry = self.registry()
        observed = self.artifacts(registry)
        observed[0]["manifest"]["probe_weight_sha256"] = r.digest("different-model")
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            with self.assertRaises(r.Blocked):
                self.compare(registry, observed)
        self.paths["left"].write_text(json.dumps({row: False for row in self.ids}), encoding="utf-8")
        with self.assertRaises(r.Blocked):
            self.compare(registry)

    def test_registry_digest_and_source_commit_refused(self):
        registry = self.registry()
        artifacts = self.artifacts(registry)
        with self.assertRaises(r.Blocked):
            e.compare_endpoint_files(*artifacts, registry, r.digest("different-registry"), source_root=self.source)
        bad = copy.deepcopy(registry)
        bad["source_commit"] = "0" * 40
        for subject in bad["subjects"].values():
            subject["source_commit"] = bad["source_commit"]
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            with self.assertRaises(r.Blocked):
                self.compare(bad)
        (self.source / "unexpected.json").write_text("untracked", encoding="utf-8")
        with self.assertRaises(r.Blocked):
            self.compare(registry)

    def test_training_sources_resolve_but_may_be_historical(self):
        registry = self.registry()
        # Move only the report checkout forward; both training roots retain the real older commit.
        (self.source / "analysis.txt").write_text("later report implementation", encoding="utf-8")
        def git(*args):
            return subprocess.run(["git", "-C", str(self.source), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git("add", "analysis.txt")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "report")
        report_commit = git("rev-parse", "HEAD")
        registry["source_commit"] = report_commit
        for subject in registry["subjects"].values():
            subject["source_commit"] = report_commit
        self.compare(registry)
        old_key = self.root_ids["X0"]
        root = registry["training_roots"].pop(old_key)
        root["source_commit"] = "f" * 40
        new_key = r.digest(root)
        registry["training_roots"][new_key] = root
        registry["subjects"]["left"]["training_roots"] = [new_key]
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            with self.assertRaisesRegex(r.Blocked, "actual exact Git commit"):
                self.compare(registry)

    def test_historical_probe_source_is_not_relabelled_as_report_source(self):
        registry = self.registry()
        (self.source / "documentation.txt").write_text("later documentation", encoding="utf-8")
        def git(*args):
            return subprocess.run(["git", "-C", str(self.source), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git("add", "documentation.txt")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "docs")
        registry["source_commit"] = git("rev-parse", "HEAD")
        self.assertNotEqual(registry["source_commit"], registry["subjects"]["left"]["source_commit"])
        self.compare(registry)
        registry["subjects"]["left"]["source_commit"] = "e" * 40
        with patch.object(Path, "read_text", side_effect=AssertionError("endpoint opened")):
            with self.assertRaisesRegex(r.Blocked, "actual exact Git commit"):
                self.compare(registry)

    def test_duplicate_partial_reordered_or_nonboolean_endpoints_refused(self):
        for content in ('{"row-0":true,"row-0":false}', '{"row-0":true}',
                        json.dumps({row: True for row in reversed(self.ids)}),
                        json.dumps({row: 1 for row in self.ids})):
            self.paths["left"].write_text(content, encoding="utf-8")
            with self.subTest(content=content[:40]), self.assertRaises(r.Blocked):
                self.compare(self.registry())

    def test_confirmation_wrong_bank_and_spent_role_refused(self):
        for field, value in (("use", "CONFIRMATION"), ("instrument_id", "SELECT-CONFIRM"),
                             ("views_sha256", r.digest("same-size-wrong-bank")),
                             ("instrument_role", "HELD_FOR_CONFIRMATION")):
            registry = self.registry()
            registry["evaluation"][field] = value
            for subject in registry["subjects"].values():
                subject["evaluation"] = copy.deepcopy(registry["evaluation"])
            with self.subTest(field=field), self.assertRaises(r.Blocked):
                self.compare(registry)
        registry = self.registry()
        registry["evaluation"].update(endpoint_domain="long_context", instrument_id="LC256")
        for subject in registry["subjects"].values():
            subject["evaluation"] = copy.deepcopy(registry["evaluation"])
        with self.assertRaises(r.Blocked):
            self.compare(registry)

    def test_training_fields_are_not_forged_common_evaluation_fields(self):
        registry = self.registry()
        root_id = self.root_ids["X0"]
        altered = registry["training_roots"].pop(root_id)
        altered["training_rows_sha256"] = r.digest("different-legitimate-draw")
        new_root = r.digest(altered)
        registry["training_roots"][new_root] = altered
        registry["subjects"]["left"]["training_roots"] = [new_root]
        self.assertTrue(self.compare(registry)["independent_training_roots"])
        registry["contrast"]["matched_root_fields"] = ["training_rows_sha256"]
        with self.assertRaisesRegex(r.Blocked, "matched training field"):
            self.compare(registry)


if __name__ == "__main__":
    unittest.main()
