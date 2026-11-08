"""Synthetic, local fixtures for TRAIN candidate provenance and selection."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import select_language_replay as s


def original(row_id, family="family_01", cls="acknowledgment_update", surface="surface_00", counterfactual=None):
    metadata = {"split": "train", "reply_class": cls, "capability_group": "communication",
                "base_template_family": family, "template_family": family + "__" + surface,
                "source_provenance": "synthetic_test_only", "human_review_status": "pending"}
    if counterfactual is not None:
        metadata["counterfactual_group_id"] = counterfactual
    return {"id": row_id, "system_policy_version": "synthetic_fixture",
            "metadata": metadata, "messages": [
                {"role": "system", "content": "Synthetic fixture."},
                {"role": "user", "content": "Fixture request " + row_id},
                {"role": "assistant", "content": "Fixture answer."}]}


def preserved(row):
    row = copy.deepcopy(row)
    row["metadata"].update({"preservation_replay_for": "rineya_v3", "preservation_source_version": "v2.0"})
    return row


def write_source(root, name, rows):
    # Noncanonical spacing and original field order test byte-preserving exports.
    raw = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
    (root / name).write_bytes(raw)
    return {"path": name, "sha256": s.sha256(raw), "bytes": len(raw), "rows": len(rows), "all_split_train": True}


def fixture_candidate(row_id, family="family_01", cls="acknowledgment_update", surface="surface_00"):
    row = original(row_id, family, cls, surface)
    raw = s.json_bytes(row) + b"\n"
    origin = s.Origin(row_id, 1, s.sha256(s.json_bytes(row)), s.sha256(raw), family, cls, "communication", None)
    p = preserved(row)
    return s.Candidate(p, s.json_bytes(p) + b"\n", 1, origin)


def production_shape_fixture(root):
    rows = []
    for cls, quota in sorted(s.CLASS_QUOTAS.items()):
        for group in range(quota):
            for variant in range(5):
                family = cls + "_base_" + str(group)
                rows.append(original(f"v2_{cls}_{group}_{variant}", family, cls, "surface_" + str(variant), family + "_pair"))
    original_binding = write_source(root, "original_v2_train.jsonl", rows)
    original_binding.update({"version": "v2.0", "preservation_members": len(rows)})
    v1_spoof = original("rineya_srb_v1_fake", "v1_family_with_misleading_v2_label")
    source_binding = write_source(root, "preservation_train.jsonl", [preserved(row) for row in rows] + [preserved(v1_spoof)])
    audit = {"audit_id": "SYNTHETIC_FIXTURE", "source": source_binding,
             "original_train_sources": [original_binding], "origin_comparison_removed_metadata_fields": list(s.ADDED_FIELDS),
             "counterfactual_base_group_conflicts": {}, "quality_flags": [], "license": "synthetic test fixture only"}
    audit_raw = s.json_bytes(audit)
    (root / "audit.json").write_bytes(audit_raw)
    registration = {
        "status": "FROZEN_TRAIN_LANGUAGE_CANDIDATES_ONLY", "selection_seed": 42,
        "source_audit_path": "audit.json", "source_audit_sha256": s.sha256(audit_raw),
        "source": source_binding, "original_v2_source": original_binding,
        "class_quotas": dict(s.CLASS_QUOTAS), "parent_cluster": "v2.0:metadata.base_template_family",
        "one_primary_row_per_cluster": True, "bounded_alternatives_per_selected_cluster": 3,
        "original_rows_immutable": True, "maximum_candidate_rows_read": len(rows) + 1,
        "expected_eligible_rows": len(rows), "expected_eligible_clusters": 256,
        "native_model_outcome_selection": False, "source_correct_training_filter": False,
        "authorization": {"read_bound_TRAIN_sources": True, "candidate_export": True, "native_execution": False,
                          "training": False, "CHECK": False, "FINAL": False},
        "training_eligibility": False, "e1_launch_ready": False,
    }
    (root / "registration.json").write_bytes(s.json_bytes(registration))
    return registration, audit, rows


class ProvenanceTests(unittest.TestCase):
    def test_wrong_v2_label_does_not_admit_v1_and_exact_member_is_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = original("v2_known")
            v1 = original("rineya_srb_v1_unknown", "unqualified_v1")
            original_binding = write_source(root, "original.jsonl", [row])
            inventory = s.scan_original(root / "original.jsonl", original_binding)
            source_binding = write_source(root, "source.jsonl", [preserved(v1), preserved(row)])
            candidates, counts = s.scan_candidates(root / "source.jsonl", source_binding, inventory, {"quality_flags": []})
            self.assertEqual([item.origin.row_id for item in candidates], ["v2_known"])
            self.assertEqual(counts["not_original_v2_members"], 1)
            self.assertEqual(counts["exact_original_v2_members"], 1)

    def test_structural_mismatch_after_only_two_removed_fields_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = original("v2_known")
            binding = write_source(root, "original.jsonl", [row])
            inventory = s.scan_original(root / "original.jsonl", binding)
            for mutation in ("target", "extra_metadata", "missing_addition"):
                changed = preserved(row)
                if mutation == "target":
                    changed["messages"][-1]["content"] = "Altered target"
                elif mutation == "extra_metadata":
                    changed["metadata"]["extra_ignored_if_buggy"] = "must not ignore"
                else:
                    del changed["metadata"][s.ADDED_FIELDS[0]]
                binding = write_source(root, "source.jsonl", [changed])
                with self.assertRaises(ValueError):
                    s.scan_candidates(root / "source.jsonl", binding, inventory, {"quality_flags": []})

    def test_flagged_id_entire_family_and_roleplay_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [original("ok", "good_family"), original("flagged_id", "good_family"),
                    original("family_not_listed_id", "bad_family"),
                    original("roleplay", "role_family", "roleplay_reasoning")]
            binding = write_source(root, "original.jsonl", rows)
            inventory = s.scan_original(root / "original.jsonl", binding)
            binding = write_source(root, "source.jsonl", [preserved(row) for row in rows])
            audit = {"quality_flags": [
                {"scope": "target_predicate", "family": "good_family", "affected_row_ids": ["flagged_id"]},
                {"scope": "entire_family", "family": "bad_family", "affected_row_ids": []}]}
            candidates, counts = s.scan_candidates(root / "source.jsonl", binding, inventory, audit)
            self.assertEqual([candidate.origin.row_id for candidate in candidates], ["ok"])
            self.assertEqual(counts["excluded_audit_quality_flags"], 2)
            self.assertEqual(counts["excluded_roleplay_reasoning"], 1)

    def test_cross_base_counterfactual_group_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [original("a", "base_a", counterfactual="pair"), original("b", "base_b", counterfactual="pair")]
            binding = write_source(root, "original.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "counterfactual"):
                s.scan_original(root / "original.jsonl", binding)

    def test_source_hash_mismatch_duplicate_id_nontrain_and_role_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = original("a")
            binding = write_source(root, "source.jsonl", [row])
            binding["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                s.scan_original(root / "source.jsonl", binding)
            binding = write_source(root, "source.jsonl", [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                s.scan_original(root / "source.jsonl", binding)
            for mutate in ("nontrain", "role"):
                bad = copy.deepcopy(row)
                if mutate == "nontrain":
                    bad["metadata"]["split"] = "dev"
                else:
                    bad["messages"][-1]["role"] = "user"
                binding = write_source(root, "source.jsonl", [bad])
                with self.assertRaises(ValueError):
                    s.scan_original(root / "source.jsonl", binding)


class SelectionTests(unittest.TestCase):
    def test_surface_variants_one_primary_and_three_ranked_alternatives(self):
        candidates = [fixture_candidate("row_" + str(i), surface="surface_" + str(i)) for i in range(7)]
        primary, alternatives, ranks = s.select_candidates(candidates, {"acknowledgment_update": 1}, 17)
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(alternatives), 3)
        self.assertEqual({item.cluster for item in primary + alternatives}, {"v2.0:family_01"})
        self.assertEqual({ranks[item.origin.row_id]["within_family_row_rank"] for item in primary + alternatives}, {1, 2, 3, 4})
        self.assertEqual(ranks[primary[0].origin.row_id]["within_family_row_rank"], 1)

    def test_order_independent_and_matches_exact_registered_hashes(self):
        candidates = [fixture_candidate(f"row_{family}_{variant}", "base_" + str(family), surface=str(variant))
                      for family in range(5) for variant in range(5)]
        first = s.select_candidates(candidates, {"acknowledgment_update": 3}, 19)
        shuffled = list(candidates)
        random.Random(888).shuffle(shuffled)
        second = s.select_candidates(shuffled, {"acknowledgment_update": 3}, 19)
        self.assertEqual(first, second)
        clusters = sorted({item.cluster for item in candidates}, key=lambda cluster:
                          hashlib.sha256(json.dumps([19, "acknowledgment_update", cluster], separators=(",", ":")).encode()).hexdigest())
        self.assertEqual([item.cluster for item in first[0]], clusters[:3])
        for item in first[0]:
            expected = min((candidate for candidate in candidates if candidate.cluster == item.cluster), key=lambda candidate:
                           hashlib.sha256(json.dumps([19, candidate.origin.row_id, candidate.origin.canonical_sha256], separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(item.origin.row_id, expected.origin.row_id)

    def test_quota_shortage_cannot_substitute_extra_rows_or_other_classes(self):
        candidates = [fixture_candidate("surface_" + str(i)) for i in range(20)]
        with self.assertRaisesRegex(ValueError, "insufficient distinct clusters"):
            s.select_candidates(candidates, {"acknowledgment_update": 2}, 17)
        candidates.extend(fixture_candidate("other_" + str(i), "other_base_" + str(i), "apology_repair") for i in range(5))
        with self.assertRaisesRegex(ValueError, "insufficient distinct clusters"):
            s.select_candidates(candidates, {"acknowledgment_update": 2, "apology_repair": 1}, 17)
        with self.assertRaisesRegex(ValueError, "no frozen quota"):
            s.select_candidates(candidates, {"acknowledgment_update": 1}, 17)


class ExportTests(unittest.TestCase):
    def test_exact_quota_byte_preserving_lineage_deterministic_and_writeonce(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registration, audit, originals = production_shape_fixture(root)
            registration_path = root / "registration.json"
            receipt = s.build_selection(registration_path, root, root / "out", "a" * 40)
            self.assertEqual(receipt["primary_rows"], 256)
            self.assertEqual(receipt["alternative_rows"], 768)
            self.assertEqual(receipt["class_counts"], s.CLASS_QUOTAS)
            source_lines = {(json.loads(raw)["id"]): raw for raw in (root / "preservation_train.jsonl").read_bytes().splitlines(keepends=True)}
            out = root / "out"
            primary = (out / "TRAIN_CANDIDATES.jsonl").read_bytes().splitlines(keepends=True)
            alternate = (out / "ALTERNATIVE_CANDIDATES.jsonl").read_bytes().splitlines(keepends=True)
            self.assertEqual(len({json.loads(raw)["metadata"]["base_template_family"] for raw in primary}), 256)
            for raw in primary + alternate:
                self.assertEqual(raw, source_lines[json.loads(raw)["id"]])
            lineage = [json.loads(line) for line in (out / "LINEAGE.jsonl").read_text().splitlines()]
            self.assertEqual(len(lineage), 1024)
            original_lookup = {row["id"]: row for row in originals}
            for item in lineage:
                self.assertEqual(item["canonical_original_record_sha256"], s.sha256(s.json_bytes(original_lookup[item["row_id"]])))
                self.assertEqual(item["source_raw_line_sha256"], s.sha256(source_lines[item["row_id"]]))
                self.assertEqual(item["human_review_status"], "pending")
                self.assertEqual(item["semantic_target_review"], "PENDING")
                self.assertFalse(item["training_eligibility"])
                self.assertEqual(item["role"], "TRAIN")
                self.assertEqual(item["recovered_source_version"], "v2.0")
            before = {path.name: path.read_bytes() for path in out.iterdir()}
            with self.assertRaises(FileExistsError):
                s.build_selection(registration_path, root, out, "a" * 40)
            self.assertEqual(before, {path.name: path.read_bytes() for path in out.iterdir()})
            s.build_selection(registration_path, root, root / "second", "a" * 40)
            for name in ("TRAIN_CANDIDATES.jsonl", "ALTERNATIVE_CANDIDATES.jsonl", "LINEAGE.jsonl"):
                self.assertEqual((out / name).read_bytes(), (root / "second" / name).read_bytes())
            for name, descriptor in receipt["files"].items():
                self.assertEqual(s.sha256((out / name).read_bytes()), descriptor["sha256"])
            self.assertFalse(receipt["training_eligibility"])
            self.assertFalse(receipt["source_version_label_used_for_membership"])
            self.assertFalse(receipt["statistical_independence_claim"])
            self.assertEqual(receipt["cost"]["model_calls"], 0)

    def test_binding_and_eligibility_gate_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registration, audit, _ = production_shape_fixture(root)
            for key, value in (("source_audit_sha256", "0" * 64), ("native_model_outcome_selection", True),
                               ("source_correct_training_filter", True), ("expected_eligible_rows", 1281)):
                changed = copy.deepcopy(registration)
                changed[key] = value
                (root / "registration.json").write_bytes(s.json_bytes(changed))
                with self.assertRaises(ValueError):
                    s.build_selection(root / "registration.json", root, root / "out", "a" * 40)
                self.assertFalse((root / "out").exists())
            altered = copy.deepcopy(registration)
            altered["class_quotas"]["apology_repair"] -= 1
            with self.assertRaisesRegex(ValueError, "exact frozen"):
                s.validate_registration(altered, audit)

    def test_source_paths_cannot_escape_bound_root(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in ("../other/train.jsonl", "safe/../../other.jsonl"):
                with self.assertRaises(ValueError):
                    s.resolve_bound_path(Path(temp), path)


if __name__ == "__main__":
    unittest.main()
