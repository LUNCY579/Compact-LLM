"""Adversarial registry checks; stdlib only and no sealed-bank/model access."""
from __future__ import annotations

import copy
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from registry_tools.registry import (
    COVERAGE, NORMALIZER, PROSPECTIVE, RETROSPECTIVE, Registry,
    RegistryError, RegistryLocked, CommittedRegistrationError, canonical, digest, empty_batch,
    fingerprints, make_exclusion, make_parent, make_release, make_row,
    make_source, make_structure, pilot_v3_batch, strict_json,
)


def _race_registration(path, batch, ready, start, output):
    """Fresh process has independent memory and competes for the same file lock."""
    ready.put(True)
    start.wait(10)
    try:
        result = Registry(path).register(batch, cycle_id="race-cycle")
        output.put(("committed", result["sequence"]))
    except (RegistryLocked, RegistryError) as error:
        output.put(("rejected", type(error).__name__))


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = Registry(Path(self.temp.name) / "registry")
        self.source = make_source("source-a", "family-a", "v1", digest("content-a"),
                                  origin="synthetic fixture", owner="tests",
                                  licence_evidence="Original synthetic fixture",
                                  use_permission="Test only")
        self.structure = make_structure({"operation": "ordered-sum", "inputs": 2},
                                        description="Synthetic fixture")
        self.parent = make_parent({"operands": [4, 8], "rule": "sum in order"},
                                  construction_schema="test-construction-v1", split="TRAIN",
                                  source_ids=["source-a"],
                                  structural_cluster_id=self.structure["structural_cluster_id"],
                                  aliases=["legacy-parent-a"])
        self.payload = {"messages": [{"role": "user", "content": "Add 4 and 8."}], "target": "12"}

    def batch(self, **items):
        batch = empty_batch()
        for key, records in items.items():
            batch[key] = records
        return batch

    def reserve(self, parent=None, sources=None, structures=None):
        return self.registry.register(self.batch(sources=sources or [self.source],
                                                structures=structures or [self.structure],
                                                parents=[parent or self.parent]), cycle_id="reserve-test")

    def row(self, **changes):
        row = make_row("row-a", self.parent["semantic_parent_id"], "TRAIN", copy.deepcopy(self.payload),
                       source_ids=["source-a"])
        row.update(changes)
        return row

    def register_row(self):
        self.reserve()
        return self.registry.register(self.batch(rows=[self.row()]), cycle_id="row-test")

    def assert_rejected_without_commit(self, batch):
        before = self.registry.snapshot()
        with self.assertRaises(RegistryError):
            self.registry.register(batch, cycle_id="rejection-test")
        self.assertEqual(before, self.registry.snapshot())
        self.assertFalse((self.registry.path / ".registry-cycle.lock").exists())

    def different_parent(self, specification=None, split="TRAIN", source_ids=None, **kwargs):
        return make_parent(specification or {"operands": [7, 9], "rule": "sum in order"},
                           construction_schema="test-construction-v1", split=split,
                           source_ids=source_ids or ["source-a"],
                           structural_cluster_id=self.structure["structural_cluster_id"], **kwargs)

    def test_reservation_precedes_row_and_chain_replays(self):
        first = self.reserve()
        second = self.registry.register(self.batch(rows=[self.row()]), cycle_id="second-cycle")
        self.assertEqual((first["sequence"], second["sequence"]), (1, 2))
        snapshot = self.registry.snapshot()
        self.assertEqual(snapshot["tail_sha256"], second["snapshot_sha256"])
        self.assertEqual(snapshot["summary"]["unique_semantic_parents"], 1)
        self.assertEqual(snapshot["summary"]["prospective_parents"], 1)
        files = sorted((self.registry.path / "snapshots").glob("*.json"))
        self.assertEqual(strict_json(files[1].read_text())["previous_snapshot_sha256"], first["snapshot_sha256"])

    def test_same_batch_prospective_generation_rejected(self):
        self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure],
                                                      parents=[self.parent], rows=[self.row()]))

    def test_retrospective_registration_has_no_prospective_claim(self):
        parent = copy.deepcopy(self.parent)
        parent.update(registration_mode=RETROSPECTIVE, prospective=False)
        self.registry.register(self.batch(sources=[self.source], structures=[self.structure],
                                         parents=[parent], rows=[self.row()]), cycle_id="retro-test")
        summary = self.registry.summary()
        self.assertEqual(summary["prospective_parents"], 0)
        self.assertEqual(summary["retrospective_parents"], 1)
        self.assertEqual(summary["model_gain"], "UNMEASURED")
        self.assertEqual(summary["historical_semantic_overlap"], "NOT_AUDITED")

    def test_same_parent_cross_release_role_rejected(self):
        self.register_row()
        drift = copy.deepcopy(self.parent)
        drift["split"] = "DEV"
        self.assert_rejected_without_commit(self.batch(parents=[drift]))
        self.assert_rejected_without_commit(self.batch(rows=[self.row(split="CAL", row_id="row-other")]))

    def test_source_revision_keeps_family_split(self):
        self.reserve()
        revision = make_source("source-a-v2", "family-a", "v2", digest("changed content"),
                               origin="new revision", owner="tests", licence_evidence="fixture",
                               use_permission="test")
        parent = self.different_parent(split="DEV", source_ids=[revision["source_id"]])
        self.assert_rejected_without_commit(self.batch(sources=[revision], parents=[parent]))
        parent["split"] = "TRAIN"
        self.registry.register(self.batch(sources=[revision], parents=[parent]), cycle_id="revision-safe")
        self.assertEqual(self.registry.summary()["source_families"], 1)

    def test_same_source_revision_content_drift(self):
        self.reserve()
        revision = copy.deepcopy(self.source)
        revision.update(source_id="new-id", content_sha256=digest("changed"))
        self.assert_rejected_without_commit(self.batch(sources=[revision]))
        revision = copy.deepcopy(self.source)
        revision["origin"] = "different alleged origin"
        self.assert_rejected_without_commit(self.batch(sources=[revision]))

    def test_source_content_family_laundering(self):
        self.reserve()
        alias = copy.deepcopy(self.source)
        alias.update(source_id="fake-new-source", source_family_id="fake-new-family")
        self.assert_rejected_without_commit(self.batch(sources=[alias]))

    def test_source_scope_revision_laundering(self):
        self.reserve()
        alias = copy.deepcopy(self.source)
        alias.update(source_id="fake-producer", revision="v2", content_sha256=digest("v2"),
                     split_scope="PRODUCER_ONLY")
        self.assert_rejected_without_commit(self.batch(sources=[alias]))

    def test_source_ancestry_inherits_content_role(self):
        self.reserve()
        derivative = make_source("source-derivative", "new-family", "v1", digest("paraphrased source"),
                                 origin="paraphrase", owner="tests", licence_evidence="fixture", use_permission="test",
                                 ancestor_source_ids=["source-a"])
        parent = self.different_parent(split="CAL", source_ids=[derivative["source_id"]])
        self.assert_rejected_without_commit(self.batch(sources=[derivative], parents=[parent]))

    def test_source_ancestry_cannot_become_producer_only(self):
        self.reserve()
        derivative = make_source("fake-code", "fake-code-family", "v1", digest("code"), origin="fake",
                                 owner="tests", licence_evidence="fixture", use_permission="test",
                                 split_scope="PRODUCER_ONLY", ancestor_source_ids=["source-a"])
        self.assert_rejected_without_commit(self.batch(sources=[derivative]))

    def test_unsafe_source_cannot_be_laundered(self):
        unsafe = copy.deepcopy(self.source)
        unsafe["safety"] = "EXCLUDED"
        self.registry.register(self.batch(sources=[unsafe]), cycle_id="exclusion-source")
        derivative = make_source("safe-alleged", "new-family", "v1", digest("derivative"), origin="fake",
                                 owner="tests", licence_evidence="fixture", use_permission="test",
                                 ancestor_source_ids=["source-a"])
        self.assert_rejected_without_commit(self.batch(sources=[derivative]))

    def test_unsafe_same_content_alias_cannot_omit_ancestry_to_become_safe(self):
        for unsafe_status in ("EXCLUDED", "UNQUALIFIED"):
            with self.subTest(safety=unsafe_status):
                registry = Registry(Path(self.temp.name) / unsafe_status)
                unsafe = copy.deepcopy(self.source)
                unsafe["safety"] = unsafe_status
                registry.register(self.batch(sources=[unsafe]), cycle_id="unsafe-first")
                safe_alias = copy.deepcopy(self.source)
                safe_alias["source_id"] = "alleged-new-safe-id"
                with self.assertRaises(RegistryError):
                    registry.register(self.batch(sources=[safe_alias]), cycle_id="alias-safe")
                self.assertEqual(registry.snapshot()["sequence"], 1)

    def test_unsafe_family_revision_cannot_discard_safety(self):
        unsafe = copy.deepcopy(self.source)
        unsafe["safety"] = "EXCLUDED"
        self.registry.register(self.batch(sources=[unsafe]), cycle_id="unsafe-family")
        new_revision = copy.deepcopy(self.source)
        new_revision.update(source_id="source-v2", revision="v2", content_sha256=digest("different revision"))
        self.assert_rejected_without_commit(self.batch(sources=[new_revision]))

    def test_unqualified_sources_not_admitted(self):
        self.source["safety"] = "UNQUALIFIED"
        self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[self.parent]))

    def test_declared_renaming_and_rewording_inherit_parent(self):
        self.reserve()
        renamed = self.different_parent({"operands": [4, 8], "rule": "renamed operands still summed"},
                                        split="DEV", ancestor_parent_ids=[self.parent["semantic_parent_id"]])
        self.assert_rejected_without_commit(self.batch(parents=[renamed]))
        renamed["split"] = "TRAIN"
        self.registry.register(self.batch(parents=[renamed]), cycle_id="renamed-safe")
        self.assertEqual(self.registry.summary()["declared_parent_lineage_groups"], 1)
        self.assertEqual(self.registry.summary()["derivative_parent_specifications"], 1)

    def test_parent_identity_is_independent_of_legacy_label(self):
        specification = {"family": "test-family", "world": {"operands": [4, 8]},
                         "full_public_rules": "Add the two operands."}
        original = self.different_parent(specification, aliases=["legacy-original"])
        relabeled = self.different_parent(copy.deepcopy(specification), aliases=["legacy-renamed"])
        self.assertEqual(original["semantic_parent_id"], relabeled["semantic_parent_id"])
        changed = copy.deepcopy(specification)
        changed["world"]["operands"][1] = 9
        substantive_change = self.different_parent(changed, aliases=["legacy-original"])
        self.assertNotEqual(original["semantic_parent_id"], substantive_change["semantic_parent_id"])

    def test_parent_derivative_cannot_drop_source_ancestry(self):
        self.reserve()
        new_source = make_source("source-b", "family-b", "v1", digest("different material"),
                                 origin="other", owner="tests", licence_evidence="fixture", use_permission="test")
        derivative = self.different_parent(source_ids=["source-b"],
                                           ancestor_parent_ids=[self.parent["semantic_parent_id"]])
        self.assert_rejected_without_commit(self.batch(sources=[new_source], parents=[derivative]))
        new_source["ancestor_source_ids"] = ["source-a"]
        self.registry.register(self.batch(sources=[new_source], parents=[derivative]), cycle_id="derivative-safe")

    def test_mixed_siblings_rejected_and_atomic(self):
        self.reserve()
        valid = self.row()
        invalid = make_row("bad-sibling", self.parent["semantic_parent_id"], "DEV",
                           {"messages": [{"role": "user", "content": "Sum four and eight."}], "target": "12"},
                           source_ids=["source-a"])
        self.assert_rejected_without_commit(self.batch(rows=[valid, invalid]))
        self.assertEqual(self.registry.summary()["unique_rows"], 0)

    def test_rewording_rows_do_not_inflate_parent_counts(self):
        self.register_row()
        sibling = make_row("row-b", self.parent["semantic_parent_id"], "TRAIN",
                           {"messages": [{"role": "user", "content": "Sum four and eight."}], "target": "12"},
                           source_ids=["source-a"], ancestor_row_ids=["row-a"])
        self.registry.register(self.batch(rows=[sibling]), cycle_id="sibling-safe")
        self.assertEqual(self.registry.summary()["unique_rows"], 2)
        self.assertEqual(self.registry.summary()["unique_semantic_parents"], 1)

    def test_row_ancestry_cannot_cross_undeclared_parent(self):
        self.register_row()
        other = self.different_parent()
        self.registry.register(self.batch(parents=[other]), cycle_id="another-parent")
        row = make_row("row-b", other["semantic_parent_id"], "TRAIN",
                       {"messages": [{"role": "user", "content": "Sum 7 and 9."}], "target": "16"},
                       source_ids=["source-a"], ancestor_row_ids=["row-a"])
        self.assert_rejected_without_commit(self.batch(rows=[row]))

    def test_duplicate_exact_row_with_new_id_rejected(self):
        self.register_row()
        self.assert_rejected_without_commit(self.batch(rows=[self.row(row_id="row-duplicate")]))

    def test_normalized_row_duplicate_rejected(self):
        self.register_row()
        payload = {"messages": [{"role": "user", "content": " ＡＤＤ  4\nAND 8. "}], "target": "12"}
        duplicate = make_row("row-normalized", self.parent["semantic_parent_id"], "TRAIN", payload, source_ids=["source-a"])
        self.assert_rejected_without_commit(self.batch(rows=[duplicate]))

    def test_lexical_row_duplicate_rejected(self):
        self.register_row()
        payload = {"messages": [{"role": "user", "content": "Add 4 and 8 ."}], "target": "12"}
        duplicate = make_row("row-lexical", self.parent["semantic_parent_id"], "TRAIN", payload, source_ids=["source-a"])
        self.assertNotEqual(fingerprints(payload)["normalized_sha256"], fingerprints(self.payload)["normalized_sha256"])
        self.assert_rejected_without_commit(self.batch(rows=[duplicate]))

    def test_same_prompt_cannot_be_reparented_with_changed_target(self):
        self.register_row()
        other = self.different_parent()
        self.registry.register(self.batch(parents=[other]), cycle_id="other-parent")
        payload = copy.deepcopy(self.payload)
        payload["target"] = "16"
        row = make_row("row-reparented", other["semantic_parent_id"], "TRAIN", payload, source_ids=["source-a"])
        self.assert_rejected_without_commit(self.batch(rows=[row]))

    def test_duplicate_normalized_parent_rejected(self):
        self.reserve()
        parent = self.different_parent({"operands": [4, 8], "rule": "SUM   IN ORDER"})
        self.assertNotEqual(parent["semantic_parent_id"], self.parent["semantic_parent_id"])
        self.assert_rejected_without_commit(self.batch(parents=[parent]))

    def test_scoped_instance_duplicate_detects_renaming(self):
        self.parent["instance_fingerprints"] = [{"algorithm": "FIXTURE_ENTITY_NEUTRAL_V1", "sha256": digest("same instance"),
                                                 "coverage": "Fixture-only declared entity normalization"}]
        self.reserve()
        other = self.different_parent({"renamed": [4, 8]}, instance_fingerprints=self.parent["instance_fingerprints"])
        self.assert_rejected_without_commit(self.batch(parents=[other]))

    def test_immutable_alias_mapping(self):
        self.reserve()
        other = self.different_parent(aliases=["legacy-parent-a"])
        self.assert_rejected_without_commit(self.batch(parents=[other]))
        drift = copy.deepcopy(self.parent)
        drift["aliases"] = ["renamed-original"]
        self.assert_rejected_without_commit(self.batch(parents=[drift]))

    def test_shared_structure_allows_instance_dev_and_discloses(self):
        self.source["split_scope"] = "PRODUCER_ONLY"
        self.reserve()
        other = self.different_parent(split="DEV")
        self.registry.register(self.batch(parents=[other]), cycle_id="shared-structure")
        summary = self.registry.summary()
        self.assertEqual(summary["structural_clusters"], 1)
        self.assertEqual(summary["structural_roles"][self.structure["structural_cluster_id"]], ["DEV", "TRAIN"])
        self.assertIn("NOT_UNSEEN_STRUCTURE", summary["shared_structure_disclosure"])

    def test_structural_identity_drift_rejected(self):
        self.reserve()
        drift = copy.deepcopy(self.structure)
        drift["specification"]["inputs"] = 3
        self.assert_rejected_without_commit(self.batch(structures=[drift]))

    def test_byte_identical_release_versions_reuse_without_new_parents(self):
        self.register_row()
        release = make_release("release-v1", ["row-a"], manifest_sha256=digest("manifest1"),
                               canonical_sha256=digest("canonical"), registration_mode=PROSPECTIVE)
        self.registry.register(self.batch(releases=[release]), cycle_id="release-one")
        version = copy.deepcopy(release)
        version.update(release_id="release-v2", manifest_sha256=digest("manifest2"))
        self.registry.register(self.batch(parents=[self.parent], rows=[self.row()], releases=[version]), cycle_id="release-reuse")
        summary = self.registry.summary()
        self.assertEqual((summary["unique_semantic_parents"], summary["unique_rows"], summary["release_versions"]), (1, 1, 2))
        self.assert_rejected_without_commit(self.batch(releases=[version]))

    def test_release_cannot_relabel_retrospective_evidence(self):
        self.parent.update(registration_mode=RETROSPECTIVE, prospective=False)
        self.register_row()
        release = make_release("release", ["row-a"], manifest_sha256=digest("m"), canonical_sha256=digest("c"), registration_mode=PROSPECTIVE)
        self.assert_rejected_without_commit(self.batch(releases=[release]))

    def test_known_exclusion_hash_rejects_and_batch_is_atomic(self):
        excluded = make_exclusion("known-exclusion", [{"domain": "parent_specification", "sha256": self.parent["specification_sha256"]}],
                                  evidence_reference="Authorized hash-only fixture", coverage_statement="Known exact fixture only")
        self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[self.parent], exclusions=[excluded]))

    def test_exclusion_cannot_be_added_after_matching_admission(self):
        self.register_row()
        excluded = make_exclusion("known-exclusion", [{"domain": "row_normalized", "sha256": fingerprints(self.payload)["normalized_sha256"]}],
                                  evidence_reference="Authorized hash-only fixture", coverage_statement="Known normalized fixture only")
        self.assert_rejected_without_commit(self.batch(exclusions=[excluded]))

    def test_unqualified_inventory_keeps_unknown_unknown(self):
        unknown = make_exclusion("historical-unqualified", [], evidence_reference="Fixture inventory without hashes",
                                 coverage_statement="Historical corpus not audited", qualification="UNQUALIFIED_HISTORICAL_INVENTORY")
        self.registry.register(self.batch(exclusions=[unknown]), cycle_id="unknown-coverage")
        self.reserve()
        self.assertEqual(self.registry.summary()["unqualified_historical_inventories"], 1)
        self.assertEqual(self.registry.summary()["historical_semantic_overlap"], "NOT_AUDITED")
        self.assertEqual(self.registry.summary()["coverage"]["semantic_decontamination"], "NOT_PROVEN")

    def test_exclusion_normalizer_namespace_mismatch_rejected(self):
        excluded = make_exclusion("wrong-normalizer", [{"domain": "row_normalized", "sha256": digest("external"),
                                                        "identity_space": "LEGACY_UNKNOWN_NORMALIZER"}],
                                  evidence_reference="Fixture manifest", coverage_statement="Unqualified external normalization")
        self.assert_rejected_without_commit(self.batch(exclusions=[excluded]))
        excluded["qualification"] = "UNQUALIFIED_HISTORICAL_INVENTORY"
        self.assert_rejected_without_commit(self.batch(exclusions=[excluded]))

    def test_parent_id_specification_collision_is_rejected(self):
        self.reserve()
        drift = copy.deepcopy(self.parent)
        drift["specification"]["operands"] = [3, 9]
        self.assert_rejected_without_commit(self.batch(parents=[drift]))
        collision = {"collision": "injected"}
        def injected_digest(value):
            return self.parent["specification_sha256"] if value == collision else digest(value)
        with patch("registry_tools.registry.digest", side_effect=injected_digest):
            other = self.different_parent(collision)
            self.assert_rejected_without_commit(self.batch(parents=[other]))

    def test_fingerprint_drift_rejected(self):
        self.reserve()
        row = self.row()
        row["payload"]["target"] = "13"
        self.assert_rejected_without_commit(self.batch(rows=[row]))

    def test_empty_target_is_valid_exact_string_but_none_and_false_rejected(self):
        self.reserve()
        payload = copy.deepcopy(self.payload)
        payload["target"] = ""
        row = make_row("row-eos-only", self.parent["semantic_parent_id"], "TRAIN", payload, source_ids=["source-a"])
        self.registry.register(self.batch(rows=[row]), cycle_id="empty-target")
        self.assertEqual(self.registry.snapshot()["state"]["rows"]["row-eos-only"]["payload"]["target"], "")
        for invalid in (None, False):
            with self.subTest(invalid=invalid):
                payload["target"] = invalid
                row = make_row("row-invalid", self.parent["semantic_parent_id"], "TRAIN", payload, source_ids=["source-a"])
                self.assert_rejected_without_commit(self.batch(rows=[row]))

    def test_closed_splits_and_boolean_integer_aliases_rejected(self):
        for split in ("CHECK", "FINAL", "train", " TRAIN", True, 1, None, ["TRAIN"]):
            with self.subTest(split=split):
                parent = copy.deepcopy(self.parent)
                parent["split"] = split
                self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[parent]))
        batch = empty_batch()
        batch["schema_version"] = True
        self.assert_rejected_without_commit(batch)

    def test_malformed_metadata_rejected(self):
        mutations = [
            ("prospective", 1), ("prospective", "true"), ("registration_mode", []),
            ("historical_overlap", "CLEAN"), ("aliases", "legacy"),
            ("source_ids", ["source-a", "source-a"]), ("ancestor_parent_ids", None),
            ("construction_schema", "../path"), ("specification_sha256", "A" * 64),
            ("instance_fingerprints", [{"algorithm": "f", "sha256": "0", "coverage": "x"}]),
        ]
        for key, value in mutations:
            with self.subTest(field=key):
                parent = copy.deepcopy(self.parent)
                parent[key] = value
                self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[parent]))
        source = copy.deepcopy(self.source)
        source["safety"] = []
        self.assert_rejected_without_commit(self.batch(sources=[source]))
        source = copy.deepcopy(self.source)
        source["extra_unreviewed_metadata"] = True
        self.assert_rejected_without_commit(self.batch(sources=[source]))

    def test_target_fields_cannot_enter_construction_spec(self):
        for key in ("target", "messages", "solution", "completion", "rendered", "model_response"):
            with self.subTest(key=key):
                parent = self.different_parent({"facts": [1, 2], "hidden": {key: "already generated"}})
                self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[parent]))

    def test_json_duplicate_keys_nonfinite_and_python_types_rejected(self):
        with self.assertRaises(RegistryError):
            strict_json('{"split":"TRAIN","split":"DEV"}')
        for value in (float("nan"), float("inf"), {1: "nonstring key"}, (1, 2)):
            with self.subTest(value=value), self.assertRaises(RegistryError):
                canonical(value)

    def test_unknown_ancestry_and_cycles_rejected(self):
        source = copy.deepcopy(self.source)
        source["ancestor_source_ids"] = ["missing-source"]
        self.assert_rejected_without_commit(self.batch(sources=[source]))
        source["ancestor_source_ids"] = [source["source_id"]]
        self.assert_rejected_without_commit(self.batch(sources=[source]))
        parent = copy.deepcopy(self.parent)
        parent["ancestor_parent_ids"] = [parent["semantic_parent_id"]]
        self.assert_rejected_without_commit(self.batch(sources=[self.source], structures=[self.structure], parents=[parent]))

    def test_duplicate_ids_in_batch_rejected(self):
        self.assert_rejected_without_commit(self.batch(sources=[self.source, copy.deepcopy(self.source)]))

    def test_snapshot_drift_and_sequence_gap_detected(self):
        result = self.reserve()
        path = Path(result["snapshot_path"])
        original = path.read_bytes()
        path.write_bytes(original.replace(b"reserve-test", b"reserve-evil"))
        with self.assertRaises(RegistryError):
            self.registry.snapshot()
        path.write_bytes(original)
        new_path = path.with_name(path.name.replace("00000001", "00000002", 1))
        path.rename(new_path)
        with self.assertRaises(RegistryError):
            self.registry.snapshot()

    def test_hash_chain_rejects_replaced_predecessor(self):
        self.register_row()
        paths = sorted((self.registry.path / "snapshots").glob("*.json"))
        first = strict_json(paths[0].read_text())
        first["cycle_id"] = "different-valid-cycle"
        changed = (canonical(first) + "\n").encode()
        import hashlib
        renamed = paths[0].with_name("00000001-" + hashlib.sha256(changed).hexdigest() + ".json")
        paths[0].unlink()
        renamed.write_bytes(changed)
        with self.assertRaises(RegistryError):
            self.registry.snapshot()

    def test_snapshot_summary_boolean_count_rejected(self):
        result = self.reserve()
        path = Path(result["snapshot_path"])
        snapshot = strict_json(path.read_text())
        snapshot["summary"]["unique_semantic_parents"] = True
        changed = (canonical(snapshot) + "\n").encode()
        import hashlib
        replacement = path.with_name("00000001-" + hashlib.sha256(changed).hexdigest() + ".json")
        path.unlink()
        replacement.write_bytes(changed)
        with self.assertRaises(RegistryError):
            self.registry.snapshot()

    def test_orphan_pending_file_is_not_committed(self):
        self.reserve()
        pending = self.registry.path / "snapshots" / ".pending-crash"
        pending.write_text("half written payload")
        self.assertEqual(self.registry.snapshot()["sequence"], 1)

    def test_publication_failure_leaves_no_partial_registration(self):
        batch = self.batch(sources=[self.source], structures=[self.structure], parents=[self.parent])
        with patch("registry_tools.registry.os.link", side_effect=OSError("injected unsupported publication")):
            with self.assertRaises(OSError):
                self.registry.register(batch, cycle_id="failed-publish")
        self.assertEqual(self.registry.snapshot()["sequence"], 0)
        self.assertEqual(list((self.registry.path / "snapshots").iterdir()), [])
        self.reserve()

    def test_postpublication_pending_cleanup_failure_returns_committed_receipt(self):
        original_unlink = Path.unlink
        def fail_pending(path, *args, **kwargs):
            if path.name.startswith(".pending-"):
                raise PermissionError("injected pending cleanup denial")
            return original_unlink(path, *args, **kwargs)
        with patch("registry_tools.registry.Path.unlink", new=fail_pending):
            result = self.reserve()
        self.assertEqual(result["sequence"], 1)
        self.assertEqual(self.registry.snapshot()["tail_sha256"], result["snapshot_sha256"])
        self.assertTrue(result["cleanup_warnings"])
        self.assertEqual(len(list((self.registry.path / "snapshots").glob(".pending-*"))), 1)

    def test_postpublication_lock_cleanup_failure_returns_committed_receipt(self):
        original_unlink = Path.unlink
        def fail_lock(path, *args, **kwargs):
            if path.name == ".registry-cycle.lock":
                raise PermissionError("injected lock cleanup denial")
            return original_unlink(path, *args, **kwargs)
        with patch("registry_tools.registry.Path.unlink", new=fail_lock):
            result = self.reserve()
        self.assertEqual(self.registry.snapshot()["sequence"], 1)
        self.assertTrue(result["cleanup_warnings"])
        self.assertTrue((self.registry.path / ".registry-cycle.lock").exists())

    def test_postpublication_lock_change_error_carries_committed_receipt(self):
        original_link = os.link
        def change_owner_after_link(source, destination):
            original_link(source, destination)
            (self.registry.path / ".registry-cycle.lock").write_text('{"owner":"changed-after-publish"}')
        with patch("registry_tools.registry.os.link", side_effect=change_owner_after_link):
            with self.assertRaises(CommittedRegistrationError) as caught:
                self.reserve()
        receipt = caught.exception.receipt
        self.assertEqual(self.registry.snapshot()["tail_sha256"], receipt["snapshot_sha256"])
        self.assertEqual(receipt["sequence"], 1)
        self.assertTrue((self.registry.path / ".registry-cycle.lock").exists())

    def test_existing_snapshot_is_never_overwritten(self):
        first = self.reserve()
        original = Path(first["snapshot_path"]).read_bytes()
        self.registry.register(self.batch(rows=[self.row()]), cycle_id="second-write")
        self.assertEqual(Path(first["snapshot_path"]).read_bytes(), original)

    def test_lock_contention_no_stale_takeover(self):
        with self.registry._lock("cycle-a"):
            with self.assertRaises(RegistryLocked):
                self.registry.register(self.batch(sources=[self.source]), cycle_id="cycle-b")
        self.assertFalse((self.registry.path / ".registry-cycle.lock").exists())
        lock = self.registry.path / ".registry-cycle.lock"
        lock.write_text('{"old":"lock"}')
        with self.assertRaises(RegistryLocked):
            self.registry.register(self.batch(sources=[self.source]), cycle_id="cycle-c")
        self.assertEqual(lock.read_text(), '{"old":"lock"}')

    def test_changed_lock_owner_is_retained(self):
        with self.assertRaises(RegistryLocked):
            with self.registry._lock("cycle-a"):
                lock = self.registry.path / ".registry-cycle.lock"
                lock.write_text('{"owner":"changed"}')
        self.assertTrue(lock.exists())

    def test_lock_owner_guard_prevents_publication_after_ownership_change(self):
        original_fsync = os.fsync
        calls = 0
        def replace_after_pending_fsync(descriptor):
            nonlocal calls
            original_fsync(descriptor)
            calls += 1
            if calls == 2:  # Lock fsync, then complete pending snapshot fsync.
                (self.registry.path / ".registry-cycle.lock").write_text('{"owner":"changed"}')
        with patch("registry_tools.registry.os.fsync", side_effect=replace_after_pending_fsync):
            with self.assertRaises(RegistryLocked):
                self.reserve()
        self.assertEqual(self.registry.snapshot()["sequence"], 0)
        self.assertEqual(list((self.registry.path / "snapshots").iterdir()), [])
        self.assertTrue((self.registry.path / ".registry-cycle.lock").exists())

    def test_multiprocess_race_only_one_commit(self):
        context = multiprocessing.get_context("spawn")
        ready, output = context.Queue(), context.Queue()
        start = context.Event()
        batch = self.batch(sources=[self.source], structures=[self.structure], parents=[self.parent])
        processes = [context.Process(target=_race_registration,
                                     args=(str(self.registry.path), batch, ready, start, output)) for _ in range(2)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertTrue(ready.get(timeout=15))
            start.set()
            results = [output.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sum(result[0] == "committed" for result in results), 1)
            self.assertEqual(self.registry.snapshot()["sequence"], 1)
            self.assertEqual(self.registry.summary()["unique_semantic_parents"], 1)
        finally:
            start.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            ready.close()
            output.close()

    def test_native_prompt_token_hash_binding_and_reidentification(self):
        self.source["split_scope"] = "PRODUCER_ONLY"
        self.reserve()
        native = {"kind": "NATIVE_TITAN_PROMPT_TOKEN_IDS", "prompt_ids_sha256": digest([1, 2, 3]),
                  "tokenizer_identity_sha256": digest("tokenizer"), "chat_template_sha256": digest("template"),
                  "template_args_sha256": digest("args"), "profile_rows_sha256": digest("rows"),
                  "profile_summary_sha256": digest("summary"), "row_payload_sha256": digest(self.payload)}
        row = self.row(native_prompt_tokens=native)
        self.registry.register(self.batch(rows=[row]), cycle_id="native-first")
        other = self.different_parent(split="DEV")
        self.registry.register(self.batch(parents=[other]), cycle_id="native-parent")
        payload = {"messages": [{"role": "user", "content": "Different rendered string same alleged token IDs"}], "target": "16"}
        native = copy.deepcopy(native)
        native["row_payload_sha256"] = digest(payload)
        duplicate = make_row("native-duplicate", other["semantic_parent_id"], "DEV", payload,
                             source_ids=["source-a"], native_prompt_tokens=native)
        self.assert_rejected_without_commit(self.batch(rows=[duplicate]))
        native["row_payload_sha256"] = digest("incorrect binding")
        self.assert_rejected_without_commit(self.batch(rows=[duplicate | {"native_prompt_tokens": native}]))
        self.assertEqual(self.registry.summary()["native_prompt_token_fingerprints"], 1)

    def test_native_hash_exclusion(self):
        self.reserve()
        native = {"kind": "NATIVE_TITAN_PROMPT_TOKEN_IDS", "prompt_ids_sha256": digest([1, 2]),
                  "tokenizer_identity_sha256": digest("t"), "chat_template_sha256": digest("c"),
                  "template_args_sha256": digest("a"), "profile_rows_sha256": digest("r"),
                  "profile_summary_sha256": digest("s"), "row_payload_sha256": digest(self.payload)}
        excluded = make_exclusion("native-known", [{"domain": "native_titan_prompt_token_ids", "sha256": native["prompt_ids_sha256"]}],
                                  evidence_reference="Hash-only fixture", coverage_statement="Only known native prompt token hash")
        self.assert_rejected_without_commit(self.batch(rows=[self.row(native_prompt_tokens=native)], exclusions=[excluded]))

    def test_pilot_import_retrospective_with_native_profile_hashes(self):
        root = Path(__file__).resolve().parents[1]
        batch = pilot_v3_batch(root)
        self.registry.register(batch, cycle_id="pilot-v3-retrospective")
        summary = self.registry.summary()
        self.assertEqual((summary["unique_semantic_parents"], summary["unique_rows"], summary["structural_clusters"]), (72, 72, 6))
        self.assertEqual(summary["parents_by_split"], {"CAL": 12, "DEV": 12, "TRAIN": 48, "TRAIN_SELECTION": 0})
        self.assertEqual((summary["prospective_parents"], summary["retrospective_parents"]), (0, 72))
        self.assertEqual(summary["native_prompt_token_fingerprints"], 72)
        self.assertEqual(summary["immutable_parent_aliases"], 72)
        for parent in batch["parents"]:
            self.assertNotIn("legacy_parent_id", parent["specification"])
            self.assertEqual(len(parent["aliases"]), 1)
        self.assertIn("entity-renaming isomorphism", summary["coverage"]["not_covered"])


if __name__ == "__main__":
    unittest.main()
