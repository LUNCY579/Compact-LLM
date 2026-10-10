"""EXPOSED exclusion mechanics. These inputs/derivatives can never become TRAIN.

No renderer, registry, query solver, model, or dataset producer is invoked.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mechanisms.distributed_dossier_v1 import fixture_exclusions as firewall

ROOT = Path(__file__).resolve().parents[1]


def fixture(fixture_id):
    """Only rehydrate a source-bound EXCLUDED spec, never a dataset world."""
    item = next(f for f in firewall.load_inventory()["fixtures"] if f["fixture_id"] == fixture_id)
    return firewall.fixture_world(item["spec"])


def rename_and_represent(world):
    """EXPOSED label/order/format control with several independent name spaces."""
    w = copy.deepcopy(world)
    entities = {e["entity_id"]: "Opaque" + str(100 - i) for i, e in enumerate(w["entities"])}
    variables = {v["variable_id"]: "Choice" + str(20 - i) for i, v in enumerate(w["variables"])}
    def transform(value, field=None):
        if type(value) is dict:
            if set(value) == {"var"}: return {"var": variables[value["var"]]}
            return {key: transform(child, key) for key, child in value.items()}
        if type(value) is list: return [transform(child, field) for child in value]
        if type(value) is str:
            if field in {"event_id", "predecessor"}: return "EventName" + value
            if field in {"policy_id", "base_policy_id"}: return "PolicyName" + value
            if field == "program_id": return "ProgramName" + value
            if field == "leg_id": return "LegName" + value
            if field == "exception_id": return "ExceptionName" + value
            if field == "variable_id": return variables[value]
            if field not in {"kind", "entity_kind", "scalar_type", "guard", "action", "predicate", "code", "operation", "comparison", "scope", "attribute", "value", "source_register", "destination_register", "encoding", "combine", "mechanism"} and value in entities:
                return entities[value]
        return value
    # Entity-ID variable choices require explicit typing instead of generic names.
    for v in w["variables"]:
        if v["scalar_type"] == "EntityId": v["choices"] = [entities[e] for e in v["choices"]]
    w = transform(w)
    for i, e in enumerate(w["entities"]): e["public_label"] = "New Label " + str(i)
    for i, r in enumerate(w["records"]):
        r["record_id"] = "RenamedRecord" + str(i); r["document_id"] = "RenamedDocument" + str(i % 3)
        if r["kind"] == "operator_menu": r["candidates"].reverse()
        if r["kind"] == "session_plan":
            for leg in r["legs"]: leg["people"].sort()
        if r["kind"] == "bundle": r["plans"].sort()
        if r["kind"] == "trace_join": r["sources"].sort()
    for i, d in enumerate(w["documents"]):
        d.update(document_id="RenamedDocument" + str(i), title="Paraphrased notes " + ["One", "Two", "Three"][i],
                 purpose="booking_minutes", dated_day=17, style="small_table",
                 record_ids=[r["record_id"] for r in w["records"] if r["document_id"] == "RenamedDocument" + str(i)])
        d["record_ids"].reverse()
    for i, rule in enumerate(w["framework"]["rule_clauses"]):
        rule["rule_id"] = "OtherRule" + str(i); rule["public_variant"] = 1
    for v in w["variables"]: v["choices"].sort()
    if w["mechanism"] == "policy_feasibility": w["query"]["candidate_ids"].sort()
    w["entities"].reverse(); w["variables"].reverse(); w["records"].reverse(); w["documents"].reverse()
    w["framework"]["rule_clauses"].reverse(); w["format"] = "TOKEN"
    return w


class FixtureExclusionTests(unittest.TestCase):
    def test_original_and_corrected_design_all_excluded(self):
        for revision in ("original", "corrected"):
            for letter in "ABC":
                fid = f"design-{revision}-{letter}"
                outcome = firewall.check_world(fixture(fid))
                self.assertEqual(outcome["decision"], "EXCLUDED", outcome)
                self.assertTrue(outcome["must_reject"])
                self.assertIn(fid, outcome["matched_fixture_ids"])
                self.assertFalse(outcome["admission_authorized"])

    def test_exact_source_hashes_separate_from_semantic_hashes(self):
        inventory = firewall.load_inventory()
        sources = {s["source_id"]: s for s in inventory["sources"]}
        self.assertEqual(sources["design-original"]["source_sha256"], "f403f79619eea8269a229402cfd0720b7af1f293032e62065add04dcf7ed7abf")
        self.assertEqual(sources["design-corrected"]["source_sha256"], "104b59b3d7cc823db2678f20b8f58d94f439e7bacd1ab87b4417ef4b42e53de9")
        before = firewall.semantic_signatures(fixture("design-original-B"))
        after = firewall.semantic_signatures(fixture("design-corrected-B"))
        self.assertNotEqual(before["alpha_semantic_sha256"], after["alpha_semantic_sha256"])
        self.assertNotEqual(after["alpha_semantic_sha256"], sources["design-corrected"]["source_sha256"])

    def test_all_explicit_mechanics_seeds_are_inventory_bound_excluded(self):
        inventory = firewall.load_inventory()
        self.assertEqual(len(inventory["fixtures"]), 21)
        for item in inventory["fixtures"]:
            self.assertEqual(item["role"], "EXCLUDED")
            self.assertFalse(item["admission_allowed"])
            outcome = firewall.check_world(firewall.fixture_world(item["spec"]))
            self.assertEqual(outcome["decision"], "EXCLUDED", item["fixture_id"])
            self.assertIn(item["fixture_id"], outcome["matched_fixture_ids"])

    def test_renaming_order_paraphrase_table_and_format_match(self):
        for fid in ["design-corrected-A", "design-original-B", "design-corrected-B", "design-corrected-C",
                    "author-helper-commitment_ledger", "author-helper-guarded_transition_learning", "author-helper-policy_feasibility",
                    "public-text-helper-A", "public-text-helper-B", "public-text-helper-C",
                    "firewall-exposed-variable-control", "firewall-exposed-ordered-leg-control",
                    "firewall-exposed-repeated-menu-control", "firewall-exposed-repeated-exception-control"]:
            with self.subTest(fixture=fid):
                before = fixture(fid); after = rename_and_represent(before)
                a, b = firewall.semantic_signatures(before), firewall.semantic_signatures(after)
                self.assertEqual(a["alpha_semantic_sha256"], b["alpha_semantic_sha256"])
                self.assertNotEqual(a["exact_semantic_sha256"], b["exact_semantic_sha256"])
                self.assertIn(fid, firewall.check_world(after)["matched_fixture_ids"])

    def test_variable_domains_and_shared_variable_name_map(self):
        w = fixture("firewall-exposed-variable-control")
        renamed = rename_and_represent(w)
        signed = firewall.semantic_signatures(renamed)
        self.assertEqual({n["namespace"] for n in signed["name_map"] if n["namespace"].startswith("variable:")},
                         {"variable:EntityId", "variable:Quantity"})
        changed = copy.deepcopy(renamed)
        next(v for v in changed["variables"] if v["scalar_type"] == "Quantity")["choices"] = [7, 9]
        outcome = firewall.check_world(changed)
        self.assertEqual(outcome["decision"], "NO_KNOWN_MATCH")
        self.assertEqual(outcome["derivative_coverage"], "NOT_PROVEN")
        self.assertFalse(outcome["admission_authorized"])

    def test_semantic_order_retained_presentation_order_removed(self):
        w = fixture("firewall-exposed-ordered-leg-control")
        presentation = copy.deepcopy(w); presentation["records"].reverse(); presentation["documents"].reverse()
        self.assertEqual(firewall.semantic_signatures(w)["alpha_semantic_sha256"], firewall.semantic_signatures(presentation)["alpha_semantic_sha256"])
        semantic = copy.deepcopy(w)
        for r in semantic["records"]:
            if r["kind"] == "session_plan": r["legs"].reverse()
        self.assertNotEqual(firewall.semantic_signatures(w)["alpha_semantic_sha256"], firewall.semantic_signatures(semantic)["alpha_semantic_sha256"])
        b = fixture("author-helper-guarded_transition_learning")
        # Mark one branch by the retained final-station query before reversing.
        b["query"].update(final_stations=["ST1"], combine="single")
        reversed_fork = copy.deepcopy(b)
        next(r for r in reversed_fork["records"] if r["kind"] == "trace_fork")["destinations"].reverse()
        self.assertNotEqual(firewall.semantic_signatures(b)["alpha_semantic_sha256"], firewall.semantic_signatures(reversed_fork)["alpha_semantic_sha256"])

    def test_entity_kinds_literals_event_and_policy_names_are_typed(self):
        w = fixture("design-corrected-A")
        # Identical spelling across independent entity/event namespaces is legal.
        for r in w["records"]:
            if r.get("event_id") == "E1": r["event_id"] = "S1"
            if r.get("predecessor") == "E1": r["predecessor"] = "S1"
        self.assertEqual(firewall.semantic_signatures(w)["alpha_semantic_sha256"],
                         firewall.semantic_signatures(rename_and_represent(w))["alpha_semantic_sha256"])
        changed = copy.deepcopy(w)
        next(r for r in changed["records"] if r["kind"] == "membership")["from_day"] = 1
        self.assertNotEqual(firewall.semantic_signatures(w)["alpha_semantic_sha256"], firewall.semantic_signatures(changed)["alpha_semantic_sha256"])
        # Two identical-looking incidental values are literals, never entity refs.
        c = fixture("author-helper-policy_feasibility")
        next(r for r in c["records"] if r["kind"] == "incidental")["value"] = "coffee"
        self.assertEqual(firewall.check_world(c)["decision"], "NO_KNOWN_MATCH")

    def test_declared_mutation_ancestry_excludes_and_unknown_coverage_not_proven(self):
        w = fixture("design-corrected-A")
        next(r for r in w["records"] if r.get("event_id") == "E4")["quantity"] = 8
        self.assertEqual(firewall.check_world(w)["decision"], "NO_KNOWN_MATCH")
        for ancestry in [["design-corrected-A"], ["design-original"], ["104b59b3d7cc823db2678f20b8f58d94f439e7bacd1ab87b4417ef4b42e53de9"]]:
            result = firewall.check_world(w, ancestry=ancestry)
            self.assertEqual(result["decision"], "EXCLUDED")
            self.assertTrue(result["must_reject"])
            self.assertEqual(result["matched_ancestry"], ancestry)
        result = firewall.check_world(w, ancestry=["unknown-source"])
        self.assertEqual(result["decision"], "NO_KNOWN_MATCH")
        self.assertEqual(result["unknown_ancestry_ids"], ["unknown-source"])
        self.assertEqual(result["derivative_coverage"], "NOT_PROVEN")

    def test_nontrivial_symmetry_enumeration_and_budget_defer(self):
        # Repeated equal room/plan specifications leave nontrivial symbol cells.
        w = fixture("firewall-exposed-symmetric-control")
        signed = firewall.semantic_signatures(w)
        self.assertGreater(signed["canonicalization"]["permutations_enumerated"], 1)
        self.assertEqual(signed["alpha_semantic_sha256"], firewall.semantic_signatures(rename_and_represent(w))["alpha_semantic_sha256"])
        self.assertEqual(firewall.check_world(w, max_permutations=1)["decision"], "DEFER")
        self.assertEqual(firewall.check_world(w, max_graph_nodes=1)["decision"], "DEFER")

    def test_no_answer_oracle_fields_and_no_solver_call(self):
        forbidden = {"answer", "target", "possible_answers", "proof", "oracle", "solution", "certificate", "status"}
        def keys(value):
            if type(value) is dict:
                for key, child in value.items(): yield key; yield from keys(child)
            elif type(value) is list:
                for child in value: yield from keys(child)
        for item in firewall.load_inventory()["fixtures"]:
            self.assertFalse(forbidden & set(keys(item["spec"])))
        # Guard dynamic import, rather than importing a solver to test its absence.
        with patch("importlib.import_module", side_effect=AssertionError("production module import forbidden")):
            result = firewall.check_world(fixture("design-corrected-A"))
        self.assertEqual(result["decision"], "EXCLUDED")
        self.assertFalse(result["solver_called"])
        for field in forbidden:
            w = fixture("design-corrected-A"); w[field] = 12
            self.assertEqual(firewall.check_world(w)["decision"], "DEFER")

    def test_inventory_source_or_signature_tamper_fails_closed(self):
        original = json.loads(firewall.INVENTORY_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inventory.json"
            for mutation in [lambda m: m["sources"][0].update(source_sha256="0" * 64),
                             lambda m: m["fixtures"][0].update(alpha_semantic_sha256="0" * 64),
                             lambda m: m["fixtures"][0].update(role="TRAIN"),
                             lambda m: m.update(comparison_version="unreviewed")]:
                manifest = copy.deepcopy(original); mutation(manifest); target.write_text(json.dumps(manifest))
                outcome = firewall.check_world(fixture("design-corrected-A"), inventory_path=target)
                self.assertEqual(outcome["decision"], "DEFER")
                self.assertTrue(outcome["must_reject"])

    def test_input_not_mutated_and_bad_ancestry_deferred(self):
        w = fixture("design-corrected-A"); before = copy.deepcopy(w)
        firewall.check_world(w)
        self.assertEqual(w, before)
        for ancestry in ["design-corrected-A", {"fixture": "design-corrected-A"}, [1]]:
            self.assertEqual(firewall.check_world(w, ancestry=ancestry)["decision"], "DEFER")

    def test_raw_bookkeeping_removed_and_environment_drift_deferred(self):
        public = fixture("design-corrected-A")
        raw = copy.deepcopy(public)
        raw.update(parent_id="ExposedOnly", reservation_id="NoReceipt", content_family_id="exposed.content.only",
                   source_domain_id="ExposedDomain", producer_family_id="exposed.producer.only", structure_id="A1", level=1)
        a, b = firewall.semantic_signatures(public), firewall.semantic_signatures(raw)
        self.assertEqual(a["alpha_semantic_sha256"], b["alpha_semantic_sha256"])
        self.assertEqual(firewall.check_world(raw)["decision"], "EXCLUDED")
        with tempfile.TemporaryDirectory() as directory:
            manifest = json.loads(firewall.INVENTORY_PATH.read_text())
            manifest["comparison_environment"][0]["sha256"] = "0" * 64
            path = Path(directory) / "drift.json"; path.write_text(json.dumps(manifest))
            result = firewall.check_world(raw, inventory_path=path)
            self.assertEqual(result["decision"], "DEFER")
            self.assertTrue("comparison implementation/schema/catalogue changed" in result["reason"] or
                            "externally reviewed inventory pin identity mismatch" in result["reason"])

    def test_repeated_program_and_exception_identity_constraints_retained(self):
        for fid, kind, field, namespace in [("firewall-exposed-repeated-menu-control", "operator_menu", "program_id", "program"),
                                             ("firewall-exposed-repeated-exception-control", "exception", "exception_id", "exception")]:
            w = fixture(fid); signed = firewall.semantic_signatures(w)
            self.assertIn(namespace, {n["namespace"] for n in signed["name_map"]})
            variant = copy.deepcopy(w)
            duplicated = [r for r in variant["records"] if r["kind"] == kind][-1]
            if kind == "operator_menu":
                for candidate in duplicated["candidates"]: candidate[field] = "Distinct" + candidate[field]
            else: duplicated[field] = "Distinct" + duplicated[field]
            self.assertNotEqual(signed["alpha_semantic_sha256"], firewall.semantic_signatures(variant)["alpha_semantic_sha256"])
            self.assertEqual(signed["alpha_semantic_sha256"], firewall.semantic_signatures(rename_and_represent(w))["alpha_semantic_sha256"])

    def test_fixture_and_source_completeness_seal_rejects_forged_inventories(self):
        world = fixture("design-original-A")
        original = json.loads(firewall.INVENTORY_PATH.read_text())
        def replace_fixture(m):
            m["fixtures"][0]["fixture_id"] = "forged-fixture"
        def replace_binding(m):
            for key in ("spec_path", "spec_source_sha256", "exact_semantic_sha256", "alpha_semantic_sha256"):
                m["fixtures"][0][key] = m["fixtures"][1][key]
        def replace_source(m):
            m["sources"][0]["source_id"] = "forged-source"
        def duplicate_fixture(m):
            m["fixtures"][-1] = copy.deepcopy(m["fixtures"][0])
        def duplicate_source(m):
            m["sources"][-1] = copy.deepcopy(m["sources"][0])
        mutations = [lambda m: m.update(fixtures=[]), lambda m: m["fixtures"].pop(), replace_fixture, replace_binding,
                     duplicate_fixture, lambda m: m["fixtures"].append(copy.deepcopy(m["fixtures"][0])),
                     lambda m: m.update(sources=[]), lambda m: m["sources"].pop(), replace_source, duplicate_source,
                     lambda m: m["sources"].append(copy.deepcopy(m["sources"][0])),
                     lambda m: m["fixtures"][0].update(source_ids=[]),
                     lambda m: m["sources"][0].update(snapshot_path=m["sources"][1]["snapshot_path"], source_sha256=m["sources"][1]["source_sha256"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged.json"
            for mutate in mutations:
                manifest = copy.deepcopy(original); mutate(manifest)
                # Inventing accurate-looking totals cannot replace sealed members.
                manifest.update(fixture_count=21, source_count=12, complete=True, completeness="PASS")
                path.write_text(json.dumps(manifest))
                outcome = firewall.check_world(world, inventory_path=path)
                self.assertEqual(outcome["decision"], "DEFER", mutate)
                self.assertTrue(outcome["must_reject"])
                self.assertFalse(outcome["admission_authorized"])
            empty = copy.deepcopy(original); empty["fixtures"] = []; path.write_text(json.dumps(empty))
            with patch.object(firewall, "INVENTORY_PATH", path):
                self.assertEqual(firewall.check_world(world)["decision"], "DEFER")

    def test_separate_seal_tampering_and_external_review_pin_binding(self):
        world = fixture("design-original-A")
        manifest_raw = firewall.INVENTORY_PATH.read_bytes()
        manifest = json.loads(manifest_raw)
        with tempfile.TemporaryDirectory() as directory:
            seal_path = Path(directory) / "fake-seal.json"
            seal_path.write_text(json.dumps(dict(protected_fixtures={}, protected_sources={})))
            with patch.object(firewall, "COMPLETENESS_SEAL_PATH", seal_path):
                result = firewall.check_world(world)
                self.assertEqual(result["decision"], "DEFER")
                self.assertIn("trusted completeness seal byte identity mismatch", result["reason"])
            pin_path = Path(directory) / "root-pin.json"
            pin = dict(schema_version=1, status="REVIEWED_EXCLUSION_PIN", comparison_algorithm=firewall.ALGORITHM,
                       comparison_version=firewall.VERSION, inventory_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                       completeness_seal_sha256=firewall.COMPLETENESS_SEAL_SHA256,
                       comparison_environment=manifest["comparison_environment"], review_receipt_sha256="1" * 64)
            pin_path.write_text(json.dumps(pin))
            with patch.object(firewall, "REVIEWED_INVENTORY_PIN_PATH", pin_path):
                result = firewall.check_world(world)
                self.assertEqual(result["decision"], "EXCLUDED")
                self.assertEqual(result["inventory_review"], "REVIEWED_EXTERNAL_PIN")
                custom = Path(directory) / "other-identity.json"
                custom.write_text(json.dumps(manifest, separators=(",", ":")))
                result = firewall.check_world(world, inventory_path=custom)
                self.assertEqual(result["decision"], "DEFER")
                self.assertIn("externally reviewed inventory pin identity mismatch", result["reason"])

    def test_schema_versions_require_exact_integer_and_historical_sources_stay_excluded(self):
        world = fixture("design-original-A")
        manifest = json.loads(firewall.INVENTORY_PATH.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "typed-manifest.json"
            pin_path = Path(directory) / "typed-pin.json"
            seal_path = Path(directory) / "typed-seal.json"
            pin = dict(schema_version=1, status="REVIEWED_EXCLUSION_PIN", comparison_algorithm=firewall.ALGORITHM,
                       comparison_version=firewall.VERSION, inventory_sha256=hashlib.sha256(firewall.INVENTORY_PATH.read_bytes()).hexdigest(),
                       completeness_seal_sha256=firewall.COMPLETENESS_SEAL_SHA256,
                       comparison_environment=manifest["comparison_environment"])
            for value in (True, 1.0, "1", None):
                altered = copy.deepcopy(manifest); altered["schema_version"] = value; target.write_text(json.dumps(altered))
                self.assertEqual(firewall.check_world(world, inventory_path=target)["decision"], "DEFER")
                altered_pin = dict(pin, schema_version=value); pin_path.write_text(json.dumps(altered_pin))
                with patch.object(firewall, "REVIEWED_INVENTORY_PIN_PATH", pin_path):
                    self.assertEqual(firewall.check_world(world)["decision"], "DEFER")
                seal = json.loads(firewall.COMPLETENESS_SEAL_PATH.read_bytes()); seal["schema_version"] = value
                seal_raw = json.dumps(seal).encode(); seal_path.write_bytes(seal_raw)
                with patch.object(firewall, "COMPLETENESS_SEAL_PATH", seal_path), patch.object(firewall, "COMPLETENESS_SEAL_SHA256", hashlib.sha256(seal_raw).hexdigest()):
                    self.assertEqual(firewall.check_world(world)["decision"], "DEFER")
        history = [h for s in manifest["sources"] for h in s["historical_bindings"]]
        self.assertTrue(history)
        for previous in history:
            result = firewall.check_world(world, ancestry=[previous["source_sha256"]])
            self.assertEqual(result["decision"], "EXCLUDED")
            self.assertEqual(result["matched_ancestry"], [previous["source_sha256"]])


if __name__ == "__main__": unittest.main()
