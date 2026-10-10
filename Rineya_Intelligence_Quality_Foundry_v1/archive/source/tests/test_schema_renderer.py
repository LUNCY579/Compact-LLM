"""EXPOSED_MECHANICS_ONLY fixtures. Never parents, rows, or quota evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "mechanisms/distributed_dossier_v1"


def load(name):
    spec = importlib.util.spec_from_file_location("test_dossier_" + name, MODULE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema, renderer = load("schema"), load("renderer")


def mechanics_world(mechanism):
    """Author small test inputs only; do not invoke production/export APIs."""
    cat = schema.load_catalogue()
    kinds = {"commitment_ledger": ["commitment", "sponsor", "signer", "group"],
             "guarded_transition_learning": ["station", "operator"],
             "policy_feasibility": ["person", "room", "kit", "credential", "group", "plan", "bundle"]}[mechanism]
    entities = [{"entity_id": kind[:2].upper() + str(n), "kind": kind, "public_label": kind.title() + " " + str(n)} for kind in kinds for n in range(4)]
    def eid(kind, n=0): return next(e["entity_id"] for e in entities if e["kind"] == kind and e["public_label"].endswith(" " + str(n)))
    if mechanism == "commitment_ledger":
        records = [
            {"kind": "alias", "entity_kind": "sponsor", "left": eid("sponsor"), "right": eid("sponsor", 1)},
            {"kind": "membership", "child": eid("sponsor"), "parent": eid("group"), "from_day": 0, "through_day": 31},
            {"kind": "delegation", "signer": eid("signer"), "owner": eid("sponsor"), "from_day": 0, "through_day": 31},
            {"kind": "ledger_start", "event_id": "E0", "day": 0, "commitment": eid("commitment"), "owner": eid("sponsor"), "quantity": 2, "signer": eid("signer")},
            {"kind": "ledger_amend", "event_id": "E1", "day": 1, "commitment": eid("commitment"), "predecessor": "E0", "quantity": 3, "signer": eid("signer")},
            {"kind": "ledger_transfer", "event_id": "E2", "day": 2, "commitment": eid("commitment"), "predecessor": "E1", "owner": eid("sponsor", 1), "signer": eid("signer")},
            {"kind": "ledger_cancel", "event_id": "E3", "day": 3, "commitment": eid("commitment"), "predecessor": "E2", "signer": eid("signer")},
            {"kind": "ledger_split", "event_id": "E4", "day": 4, "commitment": eid("commitment"), "predecessor": "E3", "children": [{"commitment": eid("commitment", n), "owner": eid("sponsor"), "quantity": n} for n in (1, 2)], "signer": eid("signer")},
        ]
        query = {"mechanism": mechanism, "cutoff": 10, "group_label": eid("group")}
        structure = "A2"
    elif mechanism == "guarded_transition_learning":
        state = {"m": 0, "x": 1, "y": 0}
        records = [{"kind": "operator_menu", "operator": eid("operator", n), "candidates": [{"program_id": "PR" + str(n) + str(k), "guard": "always", "action": action} for k, action in enumerate(("flip_m", "flip_x", "flip_y"))]} for n in range(2)]
        records += [
            {"kind": "bench_example", "operator": eid("operator"), "input": dict(state), "output": {"m": 1, "x": 1, "y": 0}},
            {"kind": "station_initial", "station": eid("station"), "state": dict(state)},
            {"kind": "trace_apply", "tick": 0, "station": eid("station"), "operator": eid("operator")},
            {"kind": "trace_splice", "tick": 1, "source": eid("station"), "source_register": "m", "destination": eid("station"), "destination_register": "x"},
            {"kind": "trace_fork", "tick": 2, "source": eid("station"), "destinations": [eid("station", 1), eid("station", 2)]},
            {"kind": "trace_join", "tick": 3, "sources": [eid("station", 1), eid("station", 2)], "destination": eid("station", 3), "operation": "xor"},
        ]
        query = {"mechanism": mechanism, "final_stations": [eid("station", 3)], "combine": "single", "encoding": "4m+2x+y"}
        structure = "B2"
    else:
        def leg(n):
            return {"leg_id": "L" + str(n), "slot": n, "room": eid("room"), "kit": eid("kit"), "leader": eid("person"), "people": [eid("person")], "group": eid("group"), "kit_demand": 1}
        records = [{"kind": "fact_bool", "predicate": "room_available", "arguments": [eid("room"), 0], "value": 1},
                   {"kind": "fact_count", "predicate": "capacity", "arguments": [eid("room")], "value": 4}]
        records += [{"kind": "session_plan", "plan": eid("plan", n), "legs": [leg(n)]} for n in range(3)]
        records += [{"kind": "bundle", "bundle": eid("bundle", n), "plans": [eid("plan"), eid("plan", n+1)]} for n in range(2)]
        records += [{"kind": "bundle", "bundle": eid("bundle", 2), "plans": [eid("plan", 1), eid("plan", 2)]},
                    {"kind": "base_policy", "policy_id": "POL0", "scope": "each_leg", "predicate": {"code": "capacity_ok"}},
                    {"kind": "exception", "exception_id": "EX0", "base_policy_id": "POL0", "selector": {"kind": "leg", "plan": eid("plan"), "leg_id": "L0"}, "condition": {"predicate": "capacity", "arguments": [eid("room")], "comparison": "ge", "value": 4}}]
        query = {"mechanism": mechanism, "candidate_ids": [eid("bundle", n) for n in range(3)], "candidate_kind": "bundle", "operation": "count_feasible"}
        structure = "C2"
    records.append({"kind": "incidental", "subject": entities[0]["entity_id"], "attribute": "meeting_refreshment", "value": "tea"})
    for n, record in enumerate(records): record.update(record_id="R" + str(n), document_id="D" + str(n % 3))
    docs = [{"document_id": "D" + str(n), "purpose": "maintenance_log", "title": "Workshop notes " + ["One", "Two", "Three"][n], "dated_day": n, "record_ids": [r["record_id"] for r in records if r["document_id"] == "D" + str(n)], "style": "controlled_prose"} for n in range(3)]
    return {"world_schema": cat["world_schema"], "parent_id": "MechanicsFixture", "reservation_id": "MechanicsReservation", "mechanism": mechanism, "structure_id": structure, "level": 1, "entities": entities, "variables": [], "documents": docs, "records": records, "framework": {"semantic_version": cat["contract_id"], "rule_clauses": [{"rule_id": "FW" + str(n), "code": code, "public_variant": 0} for n, code in enumerate(cat["framework_required"][mechanism])], "closed_candidate_list": True, "unknown_defaults": "NONE", "source_assertions_are_conjunctive": True}, "query": query, "format": "JSON", "content_family_id": "mechanics-only", "source_domain_id": "Workshop", "producer_family_id": "mechanics-producer"}


def leaves(value, path=""):
    if type(value) is dict and set(value) == {"var"}:
        yield path
    elif type(value) is dict:
        for key, item in value.items():
            if key not in ("record_id", "document_id"): yield from leaves(item, path + "." + key if path else key)
    elif type(value) is list:
        for n, item in enumerate(value): yield from leaves(item, path + f"[{n}]")
    else:
        yield path


class SchemaRendererTests(unittest.TestCase):
    def test_all_22_kinds_two_prose_and_table_variants(self):
        seen = set()
        for mechanism in schema.load_catalogue()["mechanisms"]:
            world = mechanics_world(mechanism)
            schema.validate_world(world)
            seen |= {r["kind"] for r in world["records"]}
            for style in ("controlled_prose", "small_table"):
                for variant in (0, 1):
                    w = copy.deepcopy(world)
                    for d in w["documents"]: d["style"] = style
                    rendered = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY", view_plan={"record_variants": {r["record_id"]: variant for r in w["records"]}, "query_variant": variant})
                    self.assertFalse(rendered["export_allowed"])
                    self.assertNotIn("target", rendered)
                    self.assertEqual([m["role"] for m in rendered["messages"]], ["system", "user"])
                    self.assertFalse(rendered["messages"][1]["content"].endswith("\n"))
                    for record in w["records"]:
                        actual = {a["field_path"] for a in rendered["atom_spans"] if a["record_id"] == record["record_id"]}
                        self.assertEqual(actual, set(leaves(record)), record["kind"])
                    raw = rendered["messages"][1]["content"].encode("utf-8")
                    for block in ("record_spans", "atom_spans", "source_spans", "framing_spans"):
                        for item in rendered[block]:
                            sp = item["span"]
                            self.assertTrue(raw[sp["utf8_start"]:sp["utf8_end"]].decode("utf-8"))
        self.assertEqual(seen, set(schema.load_catalogue()["productions"]))

    def test_duplicate_json_keys_and_numeric_types(self):
        with self.assertRaises(schema.SchemaError): schema.loads_world('{"world_schema":1,"world_schema":2}')
        for value in (True, 1.0, "1", None):
            w = mechanics_world("commitment_ledger")
            next(r for r in w["records"] if r["kind"] == "ledger_start")["quantity"] = value
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)

    def test_unknown_fields_duplicate_declarations_and_dangling_refs(self):
        mutations = [lambda w: w.update(answer=3), lambda w: w["entities"].append(copy.deepcopy(w["entities"][0])), lambda w: w["records"].append(copy.deepcopy(w["records"][0])), lambda w: w["records"][0].update(extra=True), lambda w: w["records"][3].update(commitment="Missing"), lambda w: w["records"][4].update(predecessor="Missing"), lambda w: w["records"][0].update(entity_kind="group")]
        for mutate in mutations:
            w = mechanics_world("commitment_ledger"); mutate(w)
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)

    def test_variable_domains_and_references(self):
        w = mechanics_world("commitment_ledger")
        w["variables"] = [{"variable_id": "V", "scalar_type": "Quantity", "choices": [2, 3]}]
        w["records"][3]["quantity"] = {"var": "V"}
        schema.validate_world(w)
        for choices in ([2, 2], [2, 3.0], [3, 2], [2, 3, 4, 5]):
            v = copy.deepcopy(w); v["variables"][0]["choices"] = choices
            with self.assertRaises(schema.SchemaError): schema.validate_world(v)
        v = copy.deepcopy(w); v["variables"][0]["scalar_type"] = "Unsupported"
        with self.assertRaises(schema.SchemaError): schema.validate_world(v)
        v = copy.deepcopy(w); v["records"][3]["quantity"] = {"var": "Missing"}
        with self.assertRaises(schema.SchemaError): schema.validate_world(v)

    def test_invalid_policy_ast_and_scope_selector(self):
        for scope in ("cross_leg", "each_bundle", "unknown"):
            w = mechanics_world("policy_feasibility")
            next(r for r in w["records"] if r["kind"] == "base_policy")["scope"] = scope
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)
        for mutation in (lambda r: r["predicate"].update(extra=0), lambda r: r["predicate"].update(code="unknown")):
            w = mechanics_world("policy_feasibility"); mutation(next(r for r in w["records"] if r["kind"] == "base_policy"))
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)
        w = mechanics_world("policy_feasibility")
        next(r for r in w["records"] if r["kind"] == "exception")["selector"]["leg_id"] = "Missing"
        with self.assertRaises(schema.SchemaError): schema.validate_world(w)

    def test_candidate_language_and_truth_table_distinctness(self):
        for field, value in (("guard", "m2"), ("action", "copy_my"), ("guard", {"code": "always"})):
            w = mechanics_world("guarded_transition_learning"); w["records"][0]["candidates"][0][field] = value
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)
        w = mechanics_world("guarded_transition_learning")
        w["records"][0]["candidates"][1].update(guard="always", action="flip_m")
        with self.assertRaises(schema.SchemaError): schema.validate_world(w)

    def test_redundant_and_conflicting_source_assertions_are_not_schema_errors(self):
        w = mechanics_world("commitment_ledger")
        repeated = copy.deepcopy(w["records"][3]); repeated["record_id"] = "Repeat"
        w["records"].append(repeated); w["documents"][0]["record_ids"].append("Repeat")
        schema.validate_world(w)
        repeated["quantity"] = 8
        schema.validate_world(w)  # Compatible-world solver must decide this contradiction.
        c = mechanics_world("policy_feasibility")
        policy = copy.deepcopy(next(r for r in c["records"] if r["kind"] == "base_policy"))
        policy.update(record_id="ConflictingPolicy", document_id="D0", scope="each_bundle", predicate={"code": "shared_kit_quota"})
        c["records"].append(policy); c["documents"][0]["record_ids"].append(policy["record_id"])
        schema.validate_world(c)  # Existing local assertion provides the exception's required scope.
        plan = copy.deepcopy(next(r for r in c["records"] if r["kind"] == "session_plan"))
        plan.update(record_id="ConflictingPlan", document_id="D0")
        plan["legs"][0]["leg_id"] = "OtherLeg"
        c["records"].append(plan); c["documents"][0]["record_ids"].append(plan["record_id"])
        schema.validate_world(c)  # Existing definition provides the named leg reference.

    def test_utf8_and_title_escaping(self):
        w = mechanics_world("policy_feasibility")
        w["documents"][0]["title"] = "R\u00e9cords One"
        r = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
        raw = r["messages"][1]["content"].encode("utf-8")
        for item in r["atom_spans"]:
            sp = item["span"]; raw[sp["utf8_start"]:sp["utf8_end"]].decode("utf-8")
        w["documents"][0]["title"] = "Bad " + chr(0xD800)
        with self.assertRaises(schema.SchemaError): schema.validate_world(w)
        for title in ('Notes "doors"', "Notes | east", "Workshop notes 0", "Final notes", "Assistant instructions", " Answer words", "Notes-", "Notes\nEast"):
            w["documents"][0]["title"] = title
            with self.assertRaises(schema.SchemaError): schema.validate_world(w)

    def test_view_plan_cannot_change_semantic_inventory_or_export(self):
        w = mechanics_world("commitment_ledger")
        for plan in ({"document_order": ["D0"]}, {"record_orders": {"D0": []}}, {"record_variants": {"R0": True}}, {"labels": {}}, {"query_variant": 2}):
            with self.assertRaises(renderer.RenderError): renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY", view_plan=plan)
        rendered = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
        with self.assertRaises(renderer.RenderError): renderer.export_public_packet(rendered, row_id="R", parent_id="P", view_id="V", revision=1)
        with self.assertRaises(renderer.RenderError): renderer.render_public(w, None, "DD-CATALOGUE-001")

    def test_every_guard_action_realization_has_leaf_spans(self):
        cat = schema.load_catalogue()
        base = mechanics_world("guarded_transition_learning")
        for guard in cat["guard_realizations"]:
            for action in cat["action_realizations"]:
                candidate = {"program_id": "CheckProgram", "guard": guard, "action": action}
                table = tuple(schema._program(candidate, s) for s in __import__("itertools").product((0, 1), repeat=3))
                alternatives = []
                for other_action in cat["action_realizations"]:
                    other = {"program_id": "Alternative" + str(len(alternatives)), "guard": "always", "action": other_action}
                    other_table = tuple(schema._program(other, s) for s in __import__("itertools").product((0, 1), repeat=3))
                    if other_table != table and other_table not in [t for _, t in alternatives]: alternatives.append((other, other_table))
                    if len(alternatives) == 2: break
                w = copy.deepcopy(base)
                w["records"][0]["candidates"] = [candidate] + [a for a, _ in alternatives]
                schema.validate_world(w)
                for style in ("controlled_prose", "small_table"):
                    w["documents"][0]["style"] = style
                    r = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
                    actual = {a["field_path"] for a in r["atom_spans"] if a["record_id"] == "R0"}
                    self.assertEqual(actual, set(leaves(w["records"][0])))

    def test_all_fact_predicates_policy_predicates_and_selectors(self):
        cat = schema.load_catalogue()
        base = mechanics_world("policy_feasibility")
        ids = {kind: next(e["entity_id"] for e in base["entities"] if e["kind"] == kind) for kind in cat["entity_kinds"] if any(e["kind"] == kind for e in base["entities"])}
        for predicate, definition in cat["predicates"].items():
            w = copy.deepcopy(base)
            record = w["records"][0]
            record["kind"] = "fact_bool" if definition["result"] == "Bit" else "fact_count"
            record["predicate"] = predicate
            record["arguments"] = [0 if typ.startswith("slot ") else ids[typ.split()[0]] for typ in definition["arguments"]]
            record["value"] = 1
            schema.validate_world(w)
            for style in ("controlled_prose", "small_table"):
                for d in w["documents"]: d["style"] = style
                r = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
                self.assertEqual({a["field_path"] for a in r["atom_spans"] if a["record_id"] == "R0"}, set(leaves(record)))
        for scope, codes in cat["compatibility"]["policy_scope_predicates"].items():
            for code in codes:
                for selector_kind in cat["compatibility"]["exception_scope_selector_pairs"][scope]:
                    w = copy.deepcopy(base)
                    policy = next(r for r in w["records"] if r["kind"] == "base_policy")
                    policy["scope"] = scope; policy["predicate"] = {"code": code}
                    if code == "leader_credential": policy["predicate"]["credential"] = ids["credential"]
                    if scope == "cross_leg":
                        for r in w["records"]:
                            if r["kind"] == "session_plan":
                                leg = copy.deepcopy(r["legs"][0]); leg["leg_id"] += "b"; leg["slot"] += 1; r["legs"].append(leg)
                    exception = next(r for r in w["records"] if r["kind"] == "exception")
                    selector = {"kind": selector_kind}
                    if selector_kind == "leg": selector.update(plan=ids["plan"], leg_id="L0")
                    else: selector[selector_kind] = ids[selector_kind]
                    exception["selector"] = selector
                    schema.validate_world(w)
                    r = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
                    for record in (policy, exception):
                        self.assertEqual({a["field_path"] for a in r["atom_spans"] if a["record_id"] == record["record_id"]}, set(leaves(record)))

    def test_every_variable_type_and_wrong_entity_domain(self):
        base = mechanics_world("commitment_ledger")
        cases = [("Quantity", [2, 3], "quantity", 3), ("Day", [0, 1], "day", 3), ("EntityId", ["SP0", "SP1"], "owner", 3)]
        for typ, choices, field, index in cases:
            w = copy.deepcopy(base); w["variables"] = [{"variable_id": "V", "scalar_type": typ, "choices": choices}]; w["records"][index][field] = {"var": "V"}
            schema.validate_world(w)
            rendered = renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
            self.assertIn("{unreported V}", rendered["messages"][1]["content"])
        w = copy.deepcopy(base); w["variables"] = [{"variable_id": "V", "scalar_type": "EntityId", "choices": ["GR0", "SP1"]}]; w["records"][3]["owner"] = {"var": "V"}
        with self.assertRaises(schema.SchemaError): schema.validate_world(w)
        b = mechanics_world("guarded_transition_learning")
        for typ, choices, path in (("Bit", [0, 1], "m"), ("Tick", [0, 1], "tick"), ("Register", ["m", "x"], "source_register")):
            w = copy.deepcopy(b); w["variables"] = [{"variable_id": "V", "scalar_type": typ, "choices": choices}]
            record = next(r for r in w["records"] if r["kind"] == ("station_initial" if typ == "Bit" else "trace_apply" if typ == "Tick" else "trace_splice"))
            (record["state"] if typ == "Bit" else record)[path] = {"var": "V"}
            schema.validate_world(w); renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")
        w = mechanics_world("policy_feasibility"); w["variables"] = [{"variable_id": "V", "scalar_type": "SmallCount", "choices": [3, 4]}]; w["records"][1]["value"] = {"var": "V"}
        schema.validate_world(w); renderer.render_public(w, None, "DD-CATALOGUE-001", mode="EXPOSED_MECHANICS_ONLY")

    def test_reservation_closed_types_registry_identity_and_world_binding(self):
        w = mechanics_world("commitment_ledger")
        w["format"] = "TOKEN"  # A2, level1, case0 allocation format.
        h = "0" * 64
        reservation = {"reservation_id": w["reservation_id"], "namespace": "riqf-c002-distributed-dossier-v1", "parent_id": w["parent_id"], "semantic_parent_id": "sp-" + h, "mechanism": w["mechanism"], "structure_id": w["structure_id"], "structure_index": 2, "level": 1, "case_index": 0, "role": "EXPOSED_DEV", "format": "TOKEN", "content_family_id": w["content_family_id"], "content_family_sha256": h, "source_domain_id": w["source_domain_id"], "producer_family_id": w["producer_family_id"], "producer_version_sha256": h, "semantic_recipe_sha256": h, "normalized_structure_sha256": h, "structure_cluster_id": "sc-" + h, "ancestry": [], "unsolved_recipe_sha256": schema.digest(w), "reserved_before_render_solve": True, "reservation_sequence": 1, "registry_previous_sha256": None, "registry_receipt_sha256": h, "registry_adapter_receipt_sha256": h}
        schema.validate_reservation(reservation, w)
        for field, value in (("semantic_parent_id", "ShortID"), ("structure_cluster_id", "A2"), ("parent_id", "sp-" + h), ("reserved_before_render_solve", 1), ("reservation_sequence", True), ("level", 1.0), ("unsolved_recipe_sha256", "1" * 64), ("role", "PROSPECTIVE_TRAIN"), ("format", "JSON")):
            r = copy.deepcopy(reservation); r[field] = value
            with self.assertRaises(schema.SchemaError): schema.validate_reservation(r, w)
        r = copy.deepcopy(reservation); r["extra"] = 1
        with self.assertRaises(schema.SchemaError): schema.validate_reservation(r, w)

        r = copy.deepcopy(reservation); r["ancestry"] = [{"ancestor_identity_type": "author_parent", "ancestor_id": "sp-" + h, "relation": "view", "ancestor_sha256": h}]
        with self.assertRaises(schema.SchemaError): schema.validate_reservation(r, w)

    def test_nonfinite_json_exponent_rejected(self):
        for raw in ('[1e999]', '{"value":-1e999}', 'NaN', 'Infinity'):
            with self.assertRaises(schema.SchemaError): schema.loads_json(raw)

    def test_exception_condition_requires_asserted_key(self):
        w = mechanics_world("policy_feasibility")
        exception = next(r for r in w["records"] if r["kind"] == "exception")
        exception["condition"]["arguments"] = ["RO1"]
        with self.assertRaises(schema.SchemaError): schema.validate_world(w)


if __name__ == "__main__": unittest.main()
