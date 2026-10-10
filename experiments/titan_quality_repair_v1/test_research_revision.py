"""Research specification integrity; no issued operator permit or model calls."""
import copy
import unittest
from . import custody, registry, schema
from .research_revision import build_qualification_proposal
from .native_qualification import proposal_payload_digest, validate_proposal


class ResearchRevisionTests(unittest.TestCase):
    def test_additive_spec_keeps_science_and_has_no_run_authority(self):
        proposal = build_qualification_proposal()
        self.assertEqual(proposal["proposal_version"], 2)
        self.assertTrue(proposal["scientific_recipe_choices_unchanged"])
        self.assertEqual(proposal["registry_sha256"], registry.build_registry()["registry_sha256"])
        self.assertFalse(proposal["native_execution_authorized"])
        self.assertFalse(proposal["training_sweep_authorized"])
        self.assertFalse(proposal["final_evaluation_authorized"])
        self.assertEqual(proposal["attempt"]["padding_input_slots"], 1984)
        self.assertEqual(proposal["attempt"]["declared_attempt_cost_seconds"], 720)
        self.assertEqual(proposal["attempt"]["total_model_constructions_max"], 2)
        self.assertEqual(proposal["proposal_payload_sha256"], proposal_payload_digest(proposal))
        self.assertEqual(proposal["source_manifest_sha256"], schema.digest(custody.source_manifest()))
        self.assertNotIn("operator_grant_descriptor", proposal)

    def test_changed_workload_cannot_borrow_review_digest(self):
        proposal = build_qualification_proposal()
        altered = copy.deepcopy(proposal)
        altered["attempt"]["generation_rows"] = 9
        self.assertNotEqual(proposal_payload_digest(altered), proposal["proposal_payload_sha256"])
        with self.assertRaisesRegex(ValueError, "EXACT_EIGHT|EXACT_WORKLOAD"):
            validate_proposal(altered)

    def test_runtime_and_original_source_assets_are_pinned(self):
        proposal = build_qualification_proposal()
        self.assertEqual(proposal["runtime_site_packages"],
                         ["C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages"])
        self.assertEqual(proposal["source_sidecar_sha256"], custody.source_identity()["sidecar_sha256"])
        self.assertEqual(proposal["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(proposal["environment"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
