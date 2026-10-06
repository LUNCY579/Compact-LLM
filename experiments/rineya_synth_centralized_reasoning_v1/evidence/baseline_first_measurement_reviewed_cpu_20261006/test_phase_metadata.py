"""Pure metadata checks; no model, response, oracle, scorer or native execution."""
import hashlib
import json
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[2]
AMENDMENT = PACKAGE / "BASELINE_FIRST_MEASUREMENT_AMENDMENT_001.json"

def read(path):
    return json.loads(path.read_bytes())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()

class MetadataPhaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amendment = read(AMENDMENT)
        cls.old = read(PACKAGE / "DEV_FINISH_POLICY_001.json")

    def test_only_future_treatment_requirements_change_phase(self):
        a, old = self.amendment, self.old
        requirements = old["required_before_any_native_scoring"]
        self.assertEqual(a["predecessor"]["historical_requirements_preserved"], requirements)
        m0 = a["phase_preconditions"]["M0_ONLY_SCORING_AND_GUARD_ENROLLMENT"]
        treatment = a["phase_preconditions"]["TREATMENT_IMPORT_AND_COMPARISON"]
        self.assertEqual(m0["retained_historical_preconditions"], [requirements[i] for i in (0, 3, 4)])
        self.assertFalse(m0["requires_treatment_checkpoints"])
        self.assertEqual(treatment["historical_preconditions"], requirements)
        self.assertTrue(treatment["requires_treatment_checkpoints"])
        self.assertIn("before first treatment update", m0["ordering"])
        self.assertIn("only after production", treatment["future_checkpoint_binding"])

    def test_frozen_finish_population_recipe_and_permission_invariants(self):
        a, old = self.amendment, self.old
        invariant = a["invariants"]
        for name in ("correctness", "generation", "timing_disclosure", "population",
                     "pilot_decisions", "statistics_policy"):
            self.assertEqual(invariant[name], old[name])
        self.assertEqual([invariant[n] for n in ("complete_DEV_rows", "complete_joined_parents",
            "final_treatment_updates", "final_treatment_sequences")], [384, 96, 27, 540])
        self.assertEqual(invariant["retention_domain_classes"], ["task", "language", "long_context"])
        self.assertEqual(invariant["supplemental_retention_sources"],
                         ["arithmetic", "retrieval", "abstention", "dialogue"])
        for name in ("historical_5pp_CI_policy_unchanged", "supplemental_3pp_engineering_policy_unchanged",
                     "student_recipe_optimizer_tokens_source_parents_answer_format_unchanged"):
            self.assertIs(invariant[name], True)
        for name in ("owner_native_launch_authorized", "native_scoring_authorized",
                     "training_authorized", "protected_bank_access", "confirmation_authorized", "promotion_authorized"):
            self.assertIs(a["authority"][name], False)
        for phase in a["phase_preconditions"].values():
            self.assertIs(phase["execution_authorized"], False)
        self.assertTrue(all(v is None for v in a["unresolved_bindings"].values()))
        self.assertIs(a["adoption_limits"]["native_grading_api_added"], False)
        self.assertIs(a["adoption_limits"]["score_import_native_gate_changed"], False)

    def test_exact_preserved_source_and_external_review_anchors(self):
        a = self.amendment
        self.assertEqual(a["sha256"], digest({k: v for k, v in a.items() if k != "sha256"}))
        for name, ref in a["source_anchors"].items():
            raw = (PACKAGE / name).read_bytes()
            self.assertEqual(len(raw), ref["bytes"], name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ref["sha256"], name)
        for name in ("external_draft_copy", "external_independent_review_copy"):
            ref = a["reviewed_inputs"][name]
            raw = Path(ref["path"]).read_bytes()
            self.assertEqual(len(raw), ref["bytes"], name)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ref["sha256"], name)
        draft = read(Path(a["reviewed_inputs"]["external_draft_copy"]["path"]))
        review = read(Path(a["reviewed_inputs"]["external_independent_review_copy"]["path"]))
        self.assertIs(draft["authority"]["owner_adopted"], False)
        self.assertEqual(review["status"], "PASS_CLOSED_PROSPECTIVE_DRAFT_ONLY")
        self.assertEqual(a["source_anchors"]["DEV_FINISH_POLICY_001.json"]["sha256"],
            "db808f1b53b1ebd8c716a246e80a4eb13a8275aad643b13a05017c9162990e7d")
        self.assertEqual(a["source_anchors"]["score_import.py"]["sha256"],
            "f859973eabc5ff0d04cd9befe0b41f97c7474abf8b1d46e768ebf4abc6ff551a")

if __name__ == "__main__":
    unittest.main(verbosity=2)
