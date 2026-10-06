"""Authored CPU fixtures only; native capture paths must never be read."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import importlib.util
import m0_scoring as scoring
import m0_score_audit as audit


class ScoringControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qualification = scoring.qualify_frozen_verifier()
        cls.fixture = scoring.authored_fixture()
        cls.packet = scoring.score_authored_fixture(cls.fixture)

    def test_authored_semantic_vectors_qualify_without_authority(self):
        self.assertGreaterEqual(self.qualification["checks_count"], 60)
        self.assertEqual(self.qualification["actual_responses_read"], 0)
        self.assertFalse(self.qualification["native_scoring_authorized"])
        self.assertEqual(self.qualification, scoring.qualify_frozen_verifier())

    def test_preserved_copy_location_resolves_package_and_qualifies(self):
        with tempfile.TemporaryDirectory(dir=scoring.HERE) as directory:
            source = Path(directory) / "m0_scoring.py"
            source.write_bytes(Path(scoring.__file__).read_bytes())
            spec = importlib.util.spec_from_file_location("preserved_m0_scoring_control", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.PACKAGE, scoring.PACKAGE)
            self.assertEqual(module.qualify_frozen_verifier(), self.qualification)
            with self.assertRaisesRegex(ValueError, "staging cannot"):
                module.validate_m0_scoring_registration({}, approval_anchor={}, review_anchor={})

    def _reject_registered_batches_before_import(self, batches, pattern, registration_updates=None):
        registration = {"kind": "REGISTERED_M0_ONLY_SCORING_V1", "scope": scoring.SCOPE,
            "native_scoring_authorized": True, "sha256": "f" * 64,
            "scorer_sources": scoring.SCORER_PINS, "finish_policy_file_sha256": scoring.FINISH_FILE,
            "importer_file_sha256": scoring.IMPORTER_PIN,
            "implementation_files": {"m0_scoring.py": "a" * 64, "m0_score_audit.py": "b" * 64},
            "baseline_first_amendment_anchor": "CPU_AMENDMENT", "qualification_anchor": "CPU_QUALIFICATION",
            "batches": batches, "requires_treatment_checkpoints": False,
            "panel_registration_sha256": "c" * 64, "row_order_sha256": "d" * 64,
            "output_path": str(scoring.HERE / "NEVER_CREATED.json")}
        if registration_updates is not None:
            registration.update(registration_updates)
        approval = {"kind": "OWNER_M0_ONLY_SCORING_AUTHORIZATION", "scope": scoring.SCOPE,
            "approved": True, "human_evidence_reference": "CPU_SIMULATION_ONLY",
            "registration_body_sha256": registration["sha256"]}
        review = {"kind": "INDEPENDENT_M0_SCORER_QUALIFICATION_REVIEW", "scope": scoring.SCOPE,
            "qualified": True, "registration_body_sha256": registration["sha256"],
            "qualification_body_sha256": self.qualification["sha256"]}
        amendment = {"id": "SYNTH-BASELINE-FIRST-MEASUREMENT-AMENDMENT-001",
            "predecessor": {"finish_policy": {"sha256": scoring.FINISH_FILE}}}
        with patch.object(scoring, "_public_location_gate"):
            with patch.object(scoring, "read_anchor", side_effect=[approval, review, registration, amendment, self.qualification]):
                with patch.object(scoring, "read_pinned", return_value=b"CPU_SELECTED_SOURCE_ONLY"):
                    with patch.object(scoring, "qualify_frozen_verifier", return_value=self.qualification):
                        with patch.object(scoring, "_load_importer", side_effect=AssertionError("panel/capture import forbidden")):
                            with self.assertRaisesRegex(ValueError, pattern):
                                scoring.score_complete_m0({}, approval_anchor={}, review_anchor={})

    def test_null_batch_template_rejected_before_panel_or_capture_import(self):
        valid = {"preserved_directory": str(scoring.HERE), "audit_path": str(scoring.HERE / "NOT_READ.json"),
            "expected_audit_sha256": "a" * 64, "expected_config_sha256": "b" * 64}
        self._reject_registered_batches_before_import([deepcopy(valid) for _ in range(4)] + [None, None],
            "nonnull exact M0 batch")

    def test_malformed_batch_paths_hashes_and_fields_rejected_before_import(self):
        valid = {"preserved_directory": str(scoring.HERE), "audit_path": str(scoring.HERE / "NOT_READ.json"),
            "expected_audit_sha256": "a" * 64, "expected_config_sha256": "b" * 64}
        for change, pattern in [({"audit_path": "relative.json"}, "absolute selected"),
                               ({"expected_audit_sha256": "not-a-hash"}, "hash descriptors"),
                               ({"extra": True}, "nonnull exact")]:
            batches = [deepcopy(valid) for _ in range(6)]
            batches[-1].update(change)
            self._reject_registered_batches_before_import(batches, pattern)

    def test_null_panel_or_order_anchor_rejected_before_panel_and_capture_import(self):
        valid = {"preserved_directory": str(scoring.HERE), "audit_path": str(scoring.HERE / "NOT_READ.json"),
            "expected_audit_sha256": "a" * 64, "expected_config_sha256": "b" * 64}
        batches = [deepcopy(valid) for _ in range(6)]
        for key in ("panel_registration_sha256", "row_order_sha256"):
            for value in (None, "not-a-hash", True):
                self._reject_registered_batches_before_import(batches, "panel/order SHA256",
                    registration_updates={key: value})

    def test_complete384_96_and_equal_finish_rule(self):
        self.assertEqual(len(self.packet["rows"]), 384)
        self.assertEqual(len({r["component_id"] for r in self.packet["rows"]}), 96)
        self.assertEqual(self.packet["verifier_calls"], 384)
        for finish in ("eos", "generation_cap"):
            selected = [r for r in self.packet["rows"] if r["finish_reason"] == finish]
            self.assertEqual(len(selected), 192)
            self.assertTrue(any(r["accepted"] for r in selected))
            self.assertTrue(any(not r["accepted"] for r in selected))
        self.assertFalse(self.packet["comparison_authorized"])
        self.assertFalse(self.packet["retention_certified"])

    def test_independent_consumer_replays_without_producer_scoring_loop(self):
        with patch.object(scoring, "_score_rows", side_effect=AssertionError("producer loop forbidden")):
            result = audit.audit_authored_fixture(self.fixture, self.packet)
        self.assertEqual(result["rows_replayed"], 384)
        self.assertFalse(result["native_scoring_authorized"])

    def test_score_change_resealed_is_rejected_by_consumer(self):
        packet = deepcopy(self.packet)
        packet["rows"][0]["accepted"] = not packet["rows"][0]["accepted"]
        packet = scoring.seal({k: v for k, v in packet.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "score row replay"):
            audit.audit_authored_fixture(self.fixture, packet)

    def test_partial_population_resealed_is_rejected(self):
        packet = deepcopy(self.packet)
        packet["rows"].pop()
        packet = scoring.seal({k: v for k, v in packet.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "whole384"):
            audit.audit_authored_fixture(self.fixture, packet)

    def test_order_lineage_finish_and_claim_forgery_rejected(self):
        mutations = [lambda p: p["rows"].reverse(),
            lambda p: p["lineage"].update(fixture_factory_sha256="f" * 64),
            lambda p: p["rows"][0].update(finish_reason="generation_cap"),
            lambda p: p.update(retention_certified=True)]
        for mutate in mutations:
            packet = deepcopy(self.packet)
            mutate(packet)
            packet = scoring.seal({k: v for k, v in packet.items() if k != "sha256"})
            with self.assertRaises(ValueError):
                audit.audit_authored_fixture(self.fixture, packet)

    def test_fixture_data_cannot_be_mutated_and_resealed(self):
        fixture = deepcopy(self.fixture)
        fixture.rows[0]["raw"]["decoded_response"] = '{"answer":[1]}'
        fixture.sha256 = scoring.digest(fixture.rows)
        with self.assertRaisesRegex(ValueError, "unmodified authored"):
            scoring.score_authored_fixture(fixture)

    def test_fixture_factory_does_not_accept_paths_or_captures(self):
        with self.assertRaises(TypeError):
            scoring.authored_fixture("DO_NOT_READ_NATIVE_CAPTURE")
        with self.assertRaisesRegex(ValueError, "authored fixture factory"):
            scoring._Fixture([], object())
        with self.assertRaisesRegex(ValueError, "fixture cannot audit native"):
            packet = deepcopy(self.packet)
            packet["scope"] = scoring.SCOPE
            audit.audit_authored_fixture(self.fixture, packet)

    def test_native_public_producer_gate_precedes_any_read(self):
        with patch.object(scoring, "read_pinned", side_effect=AssertionError("file read forbidden")):
            with patch.object(scoring, "_load_importer", side_effect=AssertionError("importer forbidden")):
                with self.assertRaisesRegex(ValueError, "staging cannot"):
                    scoring.score_complete_m0({"path": "DO_NOT_READ_CAPTURE"},
                        approval_anchor={"approved": True}, review_anchor={"qualified": True})

    def test_native_public_audit_gate_precedes_any_read(self):
        with patch.object(scoring, "read_pinned", side_effect=AssertionError("file read forbidden")):
            with self.assertRaisesRegex(ValueError, "staging cannot"):
                audit.audit_complete_m0_score_packet({}, {}, approval_anchor={}, review_anchor={})

    def test_write_once_exact_bytes_anchored_consumer_replay(self):
        with tempfile.TemporaryDirectory(dir=scoring.HERE) as directory:
            path = Path(directory) / "authored_packet.json"
            published = scoring.publish_authored_fixture(self.fixture, self.packet, path)
            original = path.read_bytes()
            self.assertEqual(original, scoring.canonical(self.packet) + b"\n")
            self.assertEqual(scoring.read_anchor(published.anchor), self.packet)
            with patch.object(scoring, "_score_rows", side_effect=AssertionError("producer loop forbidden")):
                result = audit.audit_published_authored_fixture(self.fixture, published)
            self.assertEqual(result["replay"]["rows_replayed"], 384)
            with self.assertRaises(FileExistsError):
                scoring.publish_authored_fixture(self.fixture, self.packet, path)
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "selected byte identity"):
                audit.audit_published_authored_fixture(self.fixture, published)

    def test_publication_scope_and_native_saved_read_refused_before_io(self):
        with patch.object(scoring, "read_pinned", side_effect=AssertionError("input read forbidden")):
            with self.assertRaisesRegex(ValueError, "staging cannot"):
                scoring.publish_complete_m0({}, approval_anchor={}, review_anchor={})
            with self.assertRaisesRegex(ValueError, "staging cannot"):
                audit.audit_registered_m0_score_packet({"path": "NATIVE_NOT_READ"}, {},
                    approval_anchor={}, review_anchor={})
            with self.assertRaisesRegex(ValueError, "original authored publication"):
                audit.audit_published_authored_fixture(self.fixture, {"path": "NATIVE_NOT_READ"})
        altered = deepcopy(self.packet)
        altered["scope"] = scoring.SCOPE
        with patch.object(scoring, "_write_once_score_packet", side_effect=AssertionError("write forbidden")):
            with self.assertRaisesRegex(ValueError, "synthetic score"):
                scoring.publish_authored_fixture(self.fixture, altered, "NATIVE_NOT_WRITTEN")

    def test_denied_permission_precedes_registration_and_importer(self):
        calls = []
        def read(anchor):
            calls.append(anchor)
            return {"kind": "OWNER_M0_ONLY_SCORING_AUTHORIZATION", "scope": scoring.SCOPE,
                    "approved": False, "human_evidence_reference": "CPU_ONLY_NOT_PERMISSION"}
        with patch.object(scoring, "_public_location_gate"):
            with patch.object(scoring, "read_anchor", side_effect=read):
                with patch.object(scoring, "_load_importer", side_effect=AssertionError("capture importer forbidden")):
                    with self.assertRaisesRegex(ValueError, "explicit M0-only"):
                        scoring.score_complete_m0("REGISTRATION_NOT_READ",
                            approval_anchor="CPU_DENIAL", review_anchor="REVIEW_NOT_READ")
        self.assertEqual(calls, ["CPU_DENIAL"])

    def test_self_signed_qualification_without_review_is_not_authority(self):
        values = [{"kind": "OWNER_M0_ONLY_SCORING_AUTHORIZATION", "scope": scoring.SCOPE,
                   "approved": True, "human_evidence_reference": "CPU_SIMULATION_ONLY"},
                  self.qualification]
        with patch.object(scoring, "_public_location_gate"):
            with patch.object(scoring, "read_anchor", side_effect=values):
                with patch.object(scoring, "_load_importer", side_effect=AssertionError("capture importer forbidden")):
                    with self.assertRaisesRegex(ValueError, "trusted independently"):
                        scoring.score_complete_m0({}, approval_anchor={}, review_anchor={})

    def test_independent_reference_missing_cycle_and_irrelevant_distractor(self):
        with scoring.frozen_verifier() as verifier:
            skill, spec = scoring.authored_specs()[0]
            for change in ("missing", "cycle", "distractor"):
                altered = deepcopy(spec)
                if change == "missing":
                    del altered["lookup"]["links"]["b"]
                elif change == "cycle":
                    altered["lookup"]["links"]["b"] = "a"
                else:
                    altered["lookup"]["links"]["unused"] = "unused"
                messages = verifier.render_ledger(altered)
                expected = verifier.primary.solve(altered)["solutions"]
                self.assertEqual(expected, verifier.solve_visible(messages)["solutions"])
                answer = '{"answer":"UNKNOWN"}' if not expected else verifier.contract.format_answer(expected[0])
                self.assertIs(verifier.verify_response(messages, answer)["accepted"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
