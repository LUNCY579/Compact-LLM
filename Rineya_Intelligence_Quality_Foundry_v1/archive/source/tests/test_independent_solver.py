"""Exposed-design mechanics only; no generated dataset world or admission claim."""
import copy
import hashlib
import itertools
import json
import unittest

from mechanisms.distributed_dossier_v1.independent_solver import (
    SchemaError, apply_program, exact_target, program_truth_table,
    solve_parsed_public, solve_public_world, validate_answer,
)
from mechanisms.distributed_dossier_v1.public_parser import parse_public
from tests.test_public_parser import _a as handwritten_a_source


def state(m=0, x=0, y=0):
    return dict(m=m, x=x, y=y)


def record(kind, record_id, **fields):
    return dict(kind=kind, record_id=record_id, document_id="D0", **fields)


def mechanics_world(mechanism, entities, records, query):
    """Typed public mechanics input derived only from the exposed DESIGN fixtures."""
    world = dict(world_schema="rineya.distributed_dossier_world.v1", mechanism=mechanism,
                 entities=[dict(entity_id=e, kind=k, public_label=e) for e, k in entities],
                 variables=[], documents=[], records=records,
                 framework=dict(semantic_version="RIQF-C002-DD-001", rule_clauses=[],
                                closed_candidate_list=True, unknown_defaults="NONE",
                                source_assertions_are_conjunctive=True),
                 query=dict(mechanism=mechanism, **query), format="JSON")
    return attach(world)


def attach(world):
    # Public source attachments survive controls; this is not an author renderer.
    documents = []
    for i in range(3):
        ids = [r["record_id"] for index, r in enumerate(world["records"]) if index % 3 == i]
        for r in world["records"]:
            if r["record_id"] in ids:
                r["document_id"] = f"D{i}"
        documents.append(dict(document_id=f"D{i}", purpose="booking_minutes", title=f"Fixture notes {i}",
                              dated_day=0, record_ids=ids, style="controlled_prose"))
    world["documents"] = documents
    return world


def a_fixture():
    entities = [(e, k) for k, es in {"sponsor": ["S0", "SX", "S1", "O"],
                "signer": ["W", "V"], "group": ["G1", "G2"], "commitment": ["C1", "C2"]}.items() for e in es]
    return mechanics_world("commitment_ledger", entities, [
        record("alias", "A01", entity_kind="sponsor", left="S0", right="SX"),
        record("alias", "A02", entity_kind="sponsor", left="SX", right="S1"),
        record("membership", "A03", child="S1", parent="G1", from_day=0, through_day=20),
        record("membership", "A04", child="G1", parent="G2", from_day=0, through_day=20),
        record("delegation", "A05", signer="W", owner="S1", from_day=0, through_day=20),
        record("delegation", "A06", signer="V", owner="O", from_day=0, through_day=20),
        record("ledger_start", "A07", event_id="E1", day=1, commitment="C1", owner="O", quantity=4, signer="V"),
        record("ledger_start", "A08", event_id="E2", day=1, commitment="C2", owner="S1", quantity=3, signer="W"),
        record("ledger_transfer", "A09", event_id="E3", day=3, commitment="C1", predecessor="E1", owner="S0", signer="V"),
        record("ledger_amend", "A10", event_id="E4", day=5, commitment="C1", predecessor="E3", quantity=7, signer="W"),
        record("ledger_amend", "A11", event_id="E5", day=4, commitment="C2", predecessor="E2", quantity=5, signer="W"),
        record("ledger_amend", "A12", event_id="E6", day=11, commitment="C2", predecessor="E5", quantity=12, signer="W"),
        record("ledger_amend", "A13", event_id="E7", day=6, commitment="C1", predecessor="E4", quantity=12, signer="V"),
    ], dict(cutoff=10, group_label="G2"))


def b_fixture(original=False):
    programs = lambda prefix, guards, action: [dict(program_id=f"{prefix}{i+1}", guard=g, action=action) for i, g in enumerate(guards)]
    return mechanics_world("guarded_transition_learning",
        [("Nava", "operator"), ("Loru", "operator"), ("Q", "station")], [
            record("operator_menu", "B01", operator="Nava", candidates=programs("N", ["m0", "m1" if original else "y1", "always"], "flip_x")),
            record("operator_menu", "B02", operator="Loru", candidates=programs("L", ["x1", "always", "y1"], "flip_m")),
            record("bench_example", "B03", operator="Nava", input=state(), output=state(x=1)),
            record("bench_example", "B04", operator="Nava", input=state(m=1), output=state(m=1)),
            record("bench_example", "B05", operator="Loru", input=state(), output=state()),
            record("bench_example", "B06", operator="Loru", input=state(x=1), output=state(m=1, x=1)),
            record("station_initial", "B07", station="Q", state=state()),
            record("trace_apply", "B08", tick=1, station="Q", operator="Loru"),
            record("trace_apply", "B09", tick=2, station="Q", operator="Nava"),
            record("trace_apply", "B10", tick=3, station="Q", operator="Loru"),
            record("trace_apply", "B11", tick=4, station="Q", operator="Nava"),
        ], dict(final_stations=["Q"], combine="single", encoding="4m+2x+y"))


def c_fixture():
    entities = [(e, k) for k, es in {"room": ["Elm", "Pine", "Cedar"], "person": ["Ada", "Bo"],
               "kit": ["K"], "group": ["G"], "credential": ["Q"], "plan": [f"P{i}" for i in range(1, 7)]}.items() for e in es]
    facts = []
    def fact(predicate, arguments, value):
        facts.append(record("fact_bool" if predicate in {"credential", "member", "room_available", "kit_available"} else "fact_count",
                            f"F{len(facts)+1:02d}", predicate=predicate, arguments=arguments, value=value))
    for room, capacity in [("Elm", 2), ("Pine", 1), ("Cedar", 3)]:
        fact("capacity", [room], capacity)
    fact("credential", ["Ada", "Q"], 1)
    fact("credential", ["Bo", "Q"], 0)
    fact("member", ["Bo", "G"], 1)
    for room, slot, value in [("Elm", 1, 1), ("Pine", 1, 1), ("Cedar", 2, 0), ("Pine", 3, 1), ("Elm", 3, 1), ("Elm", 4, 1)]:
        fact("room_available", [room, slot], value)
    for slot, units in [(1, 2), (2, 1), (3, 1), (4, 1)]:
        fact("kit_units", ["K", slot], units)
    policies = [record("base_policy", f"CPol{i}", policy_id=f"Policy{i}", scope="each_leg", predicate=pred)
                for i, pred in enumerate([dict(code="capacity_ok"), dict(code="room_available"),
                                          dict(code="leader_credential", credential="Q"), dict(code="kit_units_ok")])]
    plans = []
    for i, (room, slot, leader, people, demand) in enumerate([
            ("Elm", 1, "Ada", ["Ada", "Bo"], 2), ("Pine", 1, "Bo", ["Bo"], 1),
            ("Cedar", 2, "Ada", ["Ada"], 1), ("Pine", 3, "Ada", ["Ada", "Bo"], 1),
            ("Elm", 3, "Bo", ["Bo"], 1), ("Elm", 4, "Ada", ["Ada"], 2)], 1):
        plans.append(record("session_plan", f"CPlan{i}", plan=f"P{i}", legs=[dict(leg_id=f"L{i}", slot=slot,
                     room=room, kit="K", leader=leader, people=people, group="G", kit_demand=demand)]))
    exception = record("exception", "CEx", exception_id="Ex", base_policy_id="Policy2",
                       selector=dict(kind="plan", plan="P2"), condition=dict(predicate="member", arguments=["Bo", "G"], comparison="eq", value=1))
    return mechanics_world("policy_feasibility", entities, facts + policies + plans + [exception],
                           dict(candidate_ids=[f"P{i}" for i in range(1, 7)], candidate_kind="plan", operation="count_feasible"))


def by_id(world, record_id):
    return next(r for r in world["records"] if r["record_id"] == record_id)


def remove(world, record_id):
    world["records"] = [r for r in world["records"] if r["record_id"] != record_id]
    return attach(world)


def add_entity(world, entity_id, kind):
    world["entities"].append(dict(entity_id=entity_id, kind=kind, public_label=entity_id))


def metadata_fixture(world):
    """Truthful field/span mechanics source; it makes no catalogue parsing claim."""
    def leaves(value, path=""):
        if isinstance(value, dict) and set(value) != {"var"}:
            return [(p, v) for key, item in value.items() for p, v in leaves(item, f"{path}.{key}".lstrip("."))]
        if isinstance(value, list):
            return [(p, v) for i, item in enumerate(value) for p, v in leaves(item, f"{path}[{i}]")]
        return [(path, value)]

    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()

    text = "É mechanics source\n"
    atoms, framing, record_spans = [], [], []
    def write(path, value):
        nonlocal text
        text += path + " = "
        start = len(text.encode("utf-8"))
        text += json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        end = len(text.encode("utf-8"))
        text += "\n"
        return dict(message_index=1, utf8_start=start, utf8_end=end)

    frame_values = [("mechanism", world["mechanism"]), ("format", world["format"])]
    for field in ["entities", "variables", "documents", "query"]:
        frame_values += leaves(world[field], field)
    frame_values += leaves(world["framework"]["rule_clauses"], "framework.rule_clauses")
    for path, value in frame_values:
        framing.append(dict(field_path=path, span=write(path, value), typed_value_sha256=digest(value)))
    for record in world["records"]:
        start = len(text.encode("utf-8"))
        for path, value in leaves({k: v for k, v in record.items() if k not in {"record_id", "document_id"}}):
            atoms.append(dict(atom_id=f"Atom{len(atoms)+1}", record_id=record["record_id"], field_path=path,
                              span=write(record["record_id"] + "." + path, value), typed_value_sha256=digest(value)))
        record_spans.append(dict(record_id=record["record_id"], span=dict(message_index=1,
                                 utf8_start=start, utf8_end=len(text.encode("utf-8")))))
    messages = [dict(role="system", content="Exposed mechanics instruction"), dict(role="user", content=text)]
    parsed = dict(world=copy.deepcopy(world), atom_spans=atoms, record_spans=record_spans, framing_spans=framing,
                  messages_sha256=digest(messages), unparsed_assertions=[],
                  coverage_spans=[dict(category="mechanics_source", span=dict(message_index=i, utf8_start=0,
                                  utf8_end=len(message["content"].encode("utf-8")))) for i, message in enumerate(messages)])
    return parsed, messages


class ExposedFixtureChecks(unittest.TestCase):
    def assert_answers(self, world, answers, compatible=None):
        solved = solve_public_world(attach(world))
        self.assertEqual(solved["possible_answers"], answers)
        self.assertEqual(solved["status"], "INCONSISTENT" if not answers else "DETERMINATE" if len(answers) == 1 else "UNDERDETERMINED")
        if compatible is not None:
            self.assertEqual(solved["compatible_world_count"], compatible)
        return solved

    def test_A12_and_exposed_field_controls(self):
        self.assert_answers(a_fixture(), [12], 1)
        cases = [("A10", "quantity", 8, 13), ("A12", "day", 9, 19), ("A08", "quantity", 9, 12)]
        for rid, field, value, expected in cases:
            world = a_fixture()
            by_id(world, rid)[field] = value
            self.assert_answers(world, [expected])
        world = a_fixture()
        world["records"].append(record("delegation", "Extra", signer="V", owner="S1", from_day=0, through_day=20))
        self.assert_answers(world, [17])
        world = a_fixture()
        world["records"].reverse()
        self.assert_answers(world, [12])

    def test_A_typed_alias_ablation_actual_alternatives(self):
        for rid, field, choices in [("A01", "right", ["SX", "O"]), ("A02", "right", ["S1", "O"])]:
            world = a_fixture()
            world["variables"] = [dict(variable_id="Identity", scalar_type="EntityId", choices=choices)]
            by_id(world, rid)[field] = dict(var="Identity")
            solved = self.assert_answers(world, [5, 12], 2)
            self.assertEqual(len(solved["certificate"]["compatible_witnesses"]), 2)

    def test_A_forks_stale_and_unauthorized_predecessors(self):
        world = a_fixture()
        world["records"].append(record("ledger_amend", "Fork", event_id="EF", day=5, commitment="C1", predecessor="E3", quantity=8, signer="W"))
        self.assert_answers(world, [], 0)
        by_id(world, "Fork")["signer"] = "V"
        self.assert_answers(world, [12])
        by_id(world, "Fork").update(day=8, signer="W", predecessor="E1")
        self.assert_answers(world, [12])
        by_id(world, "Fork").update(predecessor="E7")  # E7 is ignored, never an accepted head.
        self.assert_answers(world, [12])

    def test_A_split_child_head_no_conservation_and_cancel(self):
        world = a_fixture()
        add_entity(world, "C3", "commitment")
        add_entity(world, "C4", "commitment")
        old = by_id(world, "A09")
        old.pop("owner")
        old.update(kind="ledger_split", children=[dict(commitment="C3", owner="S0", quantity=7), dict(commitment="C4", owner="S1", quantity=1)])
        by_id(world, "A10").update(commitment="C3", quantity=9)
        by_id(world, "A13").update(commitment="C3")
        self.assert_answers(world, [15])
        by_id(world, "A07")["quantity"] = 12
        self.assert_answers(world, [15])
        world["records"].append(record("ledger_cancel", "Cancel", event_id="EC", day=7, commitment="C4", predecessor="E3", signer="W"))
        self.assert_answers(world, [14])

    def test_A_duplicate_assertions_alias_creation_and_cycles(self):
        world = a_fixture()
        duplicate = dict(by_id(world, "A10"), record_id="Duplicate")
        world["records"].append(duplicate)
        self.assert_answers(world, [12], 1)
        duplicate["quantity"] = 8
        self.assert_answers(world, [], 0)
        world = a_fixture()
        world["records"].append(record("alias", "AC", entity_kind="commitment", left="C1", right="C2"))
        self.assert_answers(world, [])
        world = a_fixture()
        world["records"].append(record("membership", "Cycle", child="G2", parent="G1", from_day=0, through_day=20))
        self.assert_answers(world, [])

    def test_A_finite_interval_constraints_retain_valid_bindings(self):
        world = a_fixture()
        world["variables"] = [dict(variable_id="DayChoice", scalar_type="Day", choices=[0, 22])]
        by_id(world, "A05")["from_day"] = dict(var="DayChoice")
        solved = self.assert_answers(world, [12], 1)
        self.assertEqual(solved["enumeration"]["source_binding_count"], 2)

    def test_B6_all_four_actual_demonstration_ablations(self):
        solved = self.assert_answers(b_fixture(), [6], 1)
        self.assertTrue(solved["enumeration"]["programs_uniquely_identified"])
        for rid, answers in [("B03", [0, 6]), ("B04", [4, 6]), ("B05", [2, 6]), ("B06", [0, 6])]:
            self.assert_answers(remove(b_fixture(), rid), answers, 2)
        world = b_fixture()
        by_id(world, "B07")["state"]["y"] = 1
        self.assert_answers(world, [7])
        world = b_fixture()
        by_id(world, "B08")["operator"] = "Nava"
        self.assert_answers(world, [2])

    def test_B_original_redundant_observation_regression(self):
        self.assert_answers(b_fixture(original=True), [6])
        solved = self.assert_answers(remove(b_fixture(original=True), "B03"), [6], 1)
        self.assertTrue(solved["enumeration"]["programs_uniquely_identified"])

    def test_B_query_determinacy_separate_from_program_identification(self):
        world = remove(b_fixture(), "B03")
        world["records"] = [r for r in world["records"] if not r["kind"].startswith("trace_")]
        solved = self.assert_answers(world, [0], 2)
        self.assertFalse(solved["enumeration"]["programs_uniquely_identified"])

    def test_B_bounded_complete_576_source_program_combinations(self):
        # Mechanics-only maximum-domain intervention on the exposed B fixture.
        world = b_fixture()
        world["records"] = [r for r in world["records"] if r["kind"] != "bench_example"]
        by_id(world, "B01")["candidates"].append(dict(program_id="N4", guard="always", action="flip_y"))
        by_id(world, "B02")["candidates"].append(dict(program_id="L4", guard="always", action="swap_xy"))
        add_entity(world, "Third", "operator")
        world["records"].append(record("operator_menu", "ThirdMenu", operator="Third", candidates=[
            dict(program_id=f"Third{i}", guard=g, action=a) for i, (g, a) in enumerate([
                ("always", "flip_m"), ("always", "flip_x"), ("always", "flip_y"), ("always", "swap_mx")])]))
        world["variables"] = [dict(variable_id="FirstTick", scalar_type="Tick", choices=[0, 1, 5]),
                              dict(variable_id="SecondTick", scalar_type="Tick", choices=[2, 6, 7])]
        by_id(world, "B08")["tick"] = dict(var="FirstTick")
        by_id(world, "B09")["tick"] = dict(var="SecondTick")
        solved = solve_public_world(attach(world))
        self.assertEqual(solved["compatible_world_count"], 576)
        self.assertEqual(solved["enumeration"]["source_binding_count"], 9)
        self.assertEqual(solved["enumeration"]["joint_program_assignment_count_before_calibration"], 64)
        self.assertEqual(len(solved["enumeration"]["assignments"]), 576)

    def test_repeated_identity_definitions_conjoin_after_bindings(self):
        world = b_fixture()
        world["records"].append(dict(by_id(world, "B01"), record_id="RepeatedMenu"))
        self.assert_answers(world, [6], 1)
        by_id(world, "RepeatedMenu")["candidates"] = copy.deepcopy(by_id(world, "B01")["candidates"])
        by_id(world, "RepeatedMenu")["candidates"][1]["guard"] = "m1"
        self.assert_answers(world, [], 0)
        world = c_fixture()
        original = by_id(world, "F05")
        world["records"].append(dict(original, record_id="RepeatedFact"))
        world["variables"] = [dict(variable_id="FactValue", scalar_type="Bit", choices=[0, 1])]
        original["value"] = dict(var="FactValue")
        self.assert_answers(world, [2], 1)
        world = a_fixture()
        world["records"].append(dict(by_id(world, "A09"), record_id="RepeatedTransfer", owner="S1"))
        self.assert_answers(world, [12], 1)  # Alias-equivalent owner is the same asserted event.

    def test_B_incompatible_calibration_and_finite_state_bindings(self):
        world = b_fixture()
        world["records"].append(record("bench_example", "Conflict", operator="Nava", input=state(), output=state()))
        self.assert_answers(world, [], 0)
        world = b_fixture()
        world["variables"] = [dict(variable_id="InitialY", scalar_type="Bit", choices=[0, 1])]
        by_id(world, "B07")["state"]["y"] = dict(var="InitialY")
        self.assert_answers(world, [6, 7], 2)

    def test_B_splice_fork_join_complete_world_execution(self):
        world = b_fixture()
        for entity in ["Other", "Left", "Right", "Merged"]:
            add_entity(world, entity, "station")
        world["records"] = [r for r in world["records"] if not r["kind"].startswith("trace_")]
        world["records"] += [
            record("station_initial", "InitOther", station="Other", state=state(y=1)),
            record("trace_apply", "T1", tick=1, station="Q", operator="Nava"),
            record("trace_splice", "T2", tick=2, source="Q", source_register="x", destination="Other", destination_register="m"),
            record("trace_fork", "T3", tick=3, source="Other", destinations=["Left", "Right"]),
            record("trace_apply", "T4", tick=4, station="Left", operator="Nava"),
            record("trace_apply", "T5", tick=5, station="Right", operator="Loru"),
            record("trace_join", "T6", tick=6, sources=["Left", "Right"], destination="Merged", operation="xor"),
            record("trace_apply", "T7", tick=7, station="Merged", operator="Nava"),
        ]
        # Other=(1,0,1); both branches remain unchanged; XOR=000; Nava ->010.
        world["query"].update(final_stations=["Q", "Merged"], combine="xor")
        self.assert_answers(world, [0], 1)
        by_id(world, "T3")["source"] = "Q"
        # Q=010; Left=000, Right=110; XOR=110; Nava identity; Q xor110=100.
        self.assert_answers(world, [4])
        by_id(world, "T6")["destination"] = "Other"
        world["query"]["final_stations"] = ["Q", "Other"]
        self.assert_answers(world, [])

    def test_B_old_state_actions_truth_tables_and_tick_collisions(self):
        self.assertEqual(apply_program(dict(program_id="P", guard="m0", action="swap_mx"), state(x=1, y=1)), state(m=1, y=1))
        self.assertEqual(apply_program(dict(program_id="P", guard="always", action="copy_xy"), state(x=1)), state(x=1, y=1))
        for guard, action in itertools.product(["always", "m0", "m1", "x0", "x1", "y0", "y1"],
                                             ["flip_m", "flip_x", "flip_y", "swap_mx", "swap_xy", "copy_mx", "copy_xm", "copy_xy", "copy_yx"]):
            self.assertEqual(len(program_truth_table(dict(program_id="P", guard=guard, action=action))), 8)
        world = b_fixture()
        by_id(world, "B11")["tick"] = 3
        self.assert_answers(world, [])

    def test_C2_and_all_twelve_exposed_controls(self):
        self.assert_answers(c_fixture(), [2], 1)
        for policy in ["CPol0", "CPol1", "CPol3"]:
            self.assert_answers(remove(c_fixture(), policy), [3])
        world = c_fixture()
        by_id(world, "CPol2")["predicate"] = dict(code="capacity_ok")
        self.assert_answers(world, [3])
        self.assert_answers(remove(c_fixture(), "CEx"), [1])
        for predicate, arguments, value, expected in [
            ("room_available", ["Cedar", 2], 1, 3), ("capacity", ["Pine"], 2, 3),
            ("credential", ["Bo", "Q"], 1, 3), ("kit_units", ["K", 4], 2, 3),
            ("kit_units", ["K", 1], 1, 1), ("capacity", ["Cedar"], 2, 2)]:
            world = c_fixture()
            next(r for r in world["records"] if r.get("predicate") == predicate and r.get("arguments") == arguments)["value"] = value
            self.assert_answers(world, [expected])
        with self.assertRaises(SchemaError):
            solve_public_world(remove(c_fixture(), "CPol2"))

    def test_C_zero_is_determinate_fact_contradiction_is_inconsistent(self):
        world = c_fixture()
        for r in world["records"]:
            if r["kind"] == "fact_bool" and r["predicate"] == "room_available":
                r["value"] = 0
        self.assert_answers(world, [0], 1)
        world = c_fixture()
        world["records"].append(record("fact_count", "Conflict", predicate="capacity", arguments=["Elm"], value=3))
        self.assert_answers(world, [], 0)
        world["records"][-1]["value"] = 2
        self.assert_answers(world, [2], 1)

    def test_C_finite_source_domain_and_no_default_missing_fact(self):
        world = c_fixture()
        world["variables"] = [dict(variable_id="CredentialChoice", scalar_type="Bit", choices=[0, 1])]
        by_id(world, "F05")["value"] = dict(var="CredentialChoice")
        self.assert_answers(world, [2, 3], 2)
        # P5 still needs Bo's credential even though P2 is exempt.
        with self.assertRaises(SchemaError):
            solve_public_world(remove(c_fixture(), "F05"))

    def test_C_true_exemption_skips_waived_missing_fact(self):
        world = c_fixture()
        world["query"]["candidate_ids"] = ["P1", "P2", "P3"]
        remove(world, "F05")
        self.assert_answers(world, [2], 1)
        by_id(world, "F06")["value"] = 0  # The exemption no longer applies.
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))
        world["variables"] = [dict(variable_id="ExemptionChoice", scalar_type="Bit", choices=[0, 1])]
        by_id(world, "F06")["value"] = dict(var="ExemptionChoice")
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_decisive_false_masks_missing_fact_in_all_policy_orders(self):
        world = remove(c_fixture(), "F03")  # P3 lacks capacity, but its room is unavailable.
        policies = [r for r in world["records"] if r["kind"] == "base_policy"]
        other = [r for r in world["records"] if r["kind"] != "base_policy"]
        for order in itertools.permutations(policies):
            control = copy.deepcopy(world)
            control["records"] = copy.deepcopy(list(order) + other)
            self.assert_answers(control, [2], 1)

    def test_C_unmasked_missing_on_any_source_compatible_binding_rejects_world(self):
        world = remove(c_fixture(), "F03")
        world["variables"] = [dict(variable_id="RoomChoice", scalar_type="Bit", choices=[0, 1])]
        by_id(world, "F09")["value"] = dict(var="RoomChoice")
        for choices in [[0, 1], [1, 0]]:
            world["variables"][0]["choices"] = choices
            with self.assertRaises(SchemaError):
                solve_public_world(attach(world))
        # The true-room binding is source-incompatible, so it needs no policy evaluation.
        world["records"].append(record("fact_bool", "RepeatedRoom", predicate="room_available",
                                       arguments=["Cedar", 2], value=0))
        solved = self.assert_answers(world, [2], 1)
        self.assertEqual(solved["enumeration"]["source_binding_count"], 2)
        self.assertEqual(len(solved["enumeration"]["assignments"]), 2)

    def test_C_exception_condition_keys_mandatory_even_when_unused_or_conflicted(self):
        world = remove(c_fixture(), "F06")
        by_id(world, "CEx")["selector"] = dict(kind="plan", plan="P6")
        for r in world["records"]:
            if r["kind"] == "fact_bool" and r["predicate"] == "room_available":
                r["value"] = 0
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))
        world["records"].append(record("fact_count", "SourceConflict", predicate="capacity", arguments=["Elm"], value=8))
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_all_members_false_masks_missing_member_without_person_order(self):
        world = c_fixture()
        world["query"]["candidate_ids"] = ["P1", "P2", "P3"]
        world["records"] = [r for r in world["records"]
                            if r["kind"] not in {"base_policy", "exception"} or r["record_id"] == "CPol1"]
        world["records"].append(record("base_policy", "Membership", policy_id="MembershipRule",
                                       scope="each_leg", predicate=dict(code="all_members")))
        by_id(world, "F06")["value"] = 0  # Bo is known not to be a member; Ada is unreported.
        for people in [["Ada", "Bo"], ["Bo", "Ada"]]:
            by_id(world, "CPlan1")["legs"][0]["people"] = people
            self.assert_answers(world, [0], 1)
        world["query"]["candidate_ids"] = ["P1", "P2", "P6"]
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_bundle_local_false_masks_unused_missing_quota(self):
        world = c_fixture()
        world["records"] = [r for r in world["records"]
                            if r["record_id"] != "CPol3" and r.get("predicate") != "kit_units"]
        for bundle, plans in [("BU1", ["P1", "P3"]), ("BU2", ["P2", "P3"]), ("BU3", ["P3", "P4"])]:
            add_entity(world, bundle, "bundle")
            world["records"].append(record("bundle", f"R{bundle}", bundle=bundle, plans=plans))
        world["records"].append(record("base_policy", "Quota", policy_id="QuotaRule", scope="each_bundle",
                                       predicate=dict(code="shared_kit_quota")))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [0], 1)
        world["records"].reverse()
        self.assert_answers(world, [0], 1)

    def test_C_bundle_false_quota_masks_missing_local_and_true_quota_requires_it(self):
        world = remove(c_fixture(), "F04")  # P1's Ada credential is absent.
        remove(world, "CPol3")
        for i in range(1, 4):
            add_entity(world, f"BU{i}", "bundle")
            world["records"].append(record("bundle", f"RBU{i}", bundle=f"BU{i}", plans=["P1", "P2"]))
        world["records"].append(record("base_policy", "Quota", policy_id="QuotaRule", scope="each_bundle",
                                       predicate=dict(code="shared_kit_quota")))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [0], 1)  # Demand3 exceeds known quota2.
        by_id(world, "F13")["value"] = 3
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_bundle_true_exemption_skips_missing_quota(self):
        world = c_fixture()
        world["records"] = [r for r in world["records"] if r["kind"] not in {"base_policy", "exception"}
                            and r.get("predicate") != "kit_units"]
        for i in range(1, 4):
            add_entity(world, f"BU{i}", "bundle")
            world["records"].append(record("bundle", f"RBU{i}", bundle=f"BU{i}", plans=["P1", "P2"]))
        world["records"].append(record("base_policy", "Quota", policy_id="QuotaRule", scope="each_bundle",
                                       predicate=dict(code="shared_kit_quota")))
        for i in range(1, 4):
            world["records"].append(record("exception", f"Waive{i}", exception_id=f"WaiveRule{i}", base_policy_id="QuotaRule",
                                           selector=dict(kind="bundle", bundle=f"BU{i}"),
                                           condition=dict(predicate="capacity", arguments=["Elm"], comparison="ge", value=2)))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [3], 1)
        by_id(world, "F01")["value"] = 1
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_bundle_false_quota_masks_other_missing_quota_in_both_orders(self):
        world = remove(c_fixture(), "F13")
        world["records"] = [r for r in world["records"] if r["kind"] not in {"base_policy", "exception"}]
        by_id(world, "CPlan2")["legs"][0].update(slot=2, kit_demand=2)
        for i in range(1, 4):
            add_entity(world, f"BU{i}", "bundle")
            world["records"].append(record("bundle", f"RBU{i}", bundle=f"BU{i}", plans=["P1", "P2"]))
        world["records"].append(record("base_policy", "Quota", policy_id="QuotaRule", scope="each_bundle",
                                       predicate=dict(code="shared_kit_quota")))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [0], 1)  # Slot2 exceeds quota1; slot1 has no asserted quota.
        world["records"].reverse()
        self.assert_answers(world, [0], 1)
        by_id(world, "F14")["value"] = 2
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_repeated_definitions_need_one_valid_scope_or_leg_before_contradiction(self):
        world = c_fixture()
        world["records"].append(record("base_policy", "ConflictingScope", policy_id="Policy2", scope="each_bundle",
                                       predicate=dict(code="shared_kit_quota")))
        # Use a bundle query so this primitive record itself remains valid.
        for i in range(1, 4):
            add_entity(world, f"BU{i}", "bundle")
            world["records"].append(record("bundle", f"RBU{i}", bundle=f"BU{i}", plans=["P1", "P2"]))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [], 0)
        world = c_fixture()
        world["query"]["candidate_ids"] = ["P1", "P2", "P3"]
        world["records"] = [r for r in world["records"] if r["kind"] not in {"base_policy", "exception"}]
        for rid in ["CPlan1", "CPlan2", "CPlan3"]:
            plan = by_id(world, rid)
            one_leg = copy.deepcopy(plan)
            one_leg["record_id"] = f"Repeat{rid}"
            plan["legs"].append(dict(plan["legs"][0], leg_id=f"Second{rid}", slot=4, room="Elm"))
            world["records"].append(one_leg)
        world["records"].append(record("base_policy", "Cross", policy_id="CrossRule", scope="cross_leg",
                                       predicate=dict(code="slots_increase")))
        self.assert_answers(world, [], 0)

    def test_global_incidental_source_contradictions_and_finite_vocabulary(self):
        for fixture, subject, answer in [(a_fixture, "S0", 12), (b_fixture, "Q", 6), (c_fixture, "Elm", 2)]:
            world = fixture()
            world["records"] += [record("incidental", "ColorA", subject=subject, attribute="curtain_color", value="red"),
                                 record("incidental", "ColorB", subject=subject, attribute="curtain_color", value="red")]
            solved = self.assert_answers(world, [answer], 1)
            self.assertGreaterEqual(len(solved["certificate"]["sufficient_atom_ids"]), 6)
            by_id(world, "ColorB")["value"] = "blue"
            self.assert_answers(world, [], 0)
            by_id(world, "ColorB")["value"] = "magenta"
            with self.assertRaises(SchemaError):
                solve_public_world(attach(world))
        world = a_fixture()
        world["records"] += [record("incidental", "AliasColorA", subject="S0", attribute="curtain_color", value="red"),
                             record("incidental", "AliasColorB", subject="S1", attribute="curtain_color", value="blue")]
        self.assert_answers(world, [], 0)

    def test_C_two_leg_scopes_selectors_and_extra_one_leg_candidate(self):
        world = c_fixture()
        # Keep P1..P3 as queried, each now a two-leg mechanics control.
        world["query"]["candidate_ids"] = ["P1", "P2", "P3"]
        for rid in ["CPlan1", "CPlan2", "CPlan3"]:
            plan = by_id(world, rid)
            second = dict(plan["legs"][0], leg_id=f"Second{rid}", slot=4, room="Elm")
            plan["legs"].append(second)
        world["records"] = [r for r in world["records"] if r["kind"] != "base_policy"]
        remove(world, "CEx")
        world["records"] += [record("base_policy", "Cross", policy_id="CrossRule", scope="cross_leg", predicate=dict(code="slots_increase")),
                             record("base_policy", "Diff", policy_id="DiffRule", scope="cross_leg", predicate=dict(code="rooms_differ"))]
        # P1 same room fails; P2/P3 room differs and increasing slots pass.
        self.assert_answers(world, [2])
        world["records"].append(record("exception", "X", exception_id="XRule", base_policy_id="DiffRule",
                                      selector=dict(kind="plan", plan="P1"), condition=dict(predicate="member", arguments=["Bo", "G"], comparison="eq", value=1)))
        self.assert_answers(world, [3])
        by_id(world, "X")["selector"] = dict(kind="group", group="G")
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_bundle_quota_shared_resources_and_exact_exception_scope(self):
        world = c_fixture()
        world["records"] = [r for r in world["records"] if r["kind"] not in {"base_policy", "exception"}]
        for bundle, plans in [("BU1", ["P1", "P2"]), ("BU2", ["P1", "P3"]), ("BU3", ["P2", "P3"])]:
            add_entity(world, bundle, "bundle")
            world["records"].append(record("bundle", f"R{bundle}", bundle=bundle, plans=plans))
        world["records"].append(record("base_policy", "Quota", policy_id="QuotaRule", scope="each_bundle", predicate=dict(code="shared_kit_quota")))
        world["query"].update(candidate_kind="bundle", candidate_ids=["BU1", "BU2", "BU3"])
        self.assert_answers(world, [2])  # BU1 demand3 at slot1 exceeds2; others fit independently.
        world["records"].append(record("exception", "Waive", exception_id="WaiveRule", base_policy_id="QuotaRule",
                                      selector=dict(kind="bundle", bundle="BU1"), condition=dict(predicate="capacity", arguments=["Elm"], comparison="ge", value=2)))
        self.assert_answers(world, [3])
        by_id(world, "Waive")["base_policy_id"] = "Missing"
        with self.assertRaises(SchemaError):
            solve_public_world(attach(world))

    def test_C_leg_selector_does_not_waive_other_leg(self):
        world = c_fixture()
        p2 = by_id(world, "CPlan2")
        p2["legs"].append(dict(p2["legs"][0], leg_id="L2second"))
        by_id(world, "CEx")["selector"] = dict(kind="leg", plan="P2", leg_id="L2")
        self.assert_answers(world, [1])
        by_id(world, "CEx")["selector"] = dict(kind="group", group="G")
        self.assert_answers(world, [3])  # P2 and P5 waived alongside P1.

    def test_typed_validation_before_status_and_no_integer_coercion(self):
        for fixture, rid, field, bad in [(a_fixture, "A10", "quantity", True), (a_fixture, "A12", "day", 9.0),
                                         (c_fixture, "F01", "value", False), (b_fixture, "B08", "tick", True)]:
            world = fixture()
            by_id(world, rid)[field] = bad
            with self.assertRaises(SchemaError):
                solve_public_world(world)
        world = b_fixture()
        by_id(world, "B01")["candidates"][1] = dict(by_id(world, "B01")["candidates"][0], program_id="OtherProgram")
        with self.assertRaises(SchemaError):
            solve_public_world(world)
        world = a_fixture()
        by_id(world, "A01")["entity_kind"] = "group"
        with self.assertRaises(SchemaError):
            solve_public_world(world)
        world["answer"] = 12
        with self.assertRaises(SchemaError):
            solve_public_world(world)

    def test_public_atom_support_and_exhaustive_witness_ledger(self):
        world = a_fixture()
        world["_public"] = dict(atom_spans=[dict(atom_id="PublicAtom", record_id="A10", field_path="quantity")], messages_sha256="abc")
        with self.assertRaises(SchemaError):
            solve_public_world(world)
        with self.assertRaises(SchemaError):
            solve_parsed_public(dict(world=a_fixture(), atom_spans=world["_public"]["atom_spans"],
                                     public_messages_sha256="abc", unparsed_assertions=[]))

    def test_certificate_prior_observation_dependencies_and_final_rules(self):
        world = b_fixture()
        add_entity(world, "Other", "station")
        world["records"].append(record("station_initial", "InitOther", station="Other", state=state(y=1)))
        world["query"].update(final_stations=["Q", "Other"], combine="xor")
        rule_codes = ["FINITE_WORLDS", "STATUS", "B_PROGRAM", "B_INFER", "B_TRACE", "B_XOR", "B_RESULT"]
        world["framework"]["rule_clauses"] = [dict(rule_id=f"Rule{i}", code=code, public_variant=0)
                                               for i, code in enumerate(rule_codes)]
        solved = self.assert_answers(world, [7], 1)
        certificate = solved["certificate"]
        self.assertEqual(set(certificate["operative_rule_ids"]), {f"Rule{i}" for i in range(len(rule_codes))})
        inferred_loru = [step for step in certificate["step_values"]
                         if step["operation"] == "candidate_elimination" and step["value"] == ["L1"]][0]
        # Mechanics IDs are deterministically scoped to the visible record field.
        earlier_observation = "a" + hashlib.sha256(json.dumps(["B05", "input.m"], sort_keys=True,
                                separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()[:24]
        self.assertIn(earlier_observation, inferred_loru["inputs"])
        known = set(certificate["sufficient_atom_ids"]) | set(certificate["operative_rule_ids"])
        for step in certificate["step_values"]:
            self.assertTrue(set(step["inputs"]) <= known)
            self.assertNotIn(step["step_id"], known)
            known.add(step["step_id"])

    def test_certificate_executed_rules_include_rejected_bindings_without_unused_background(self):
        world = c_fixture()
        world["variables"] = [dict(variable_id="CredentialChoice", scalar_type="Bit", choices=[0, 1])]
        by_id(world, "F05")["value"] = dict(var="CredentialChoice")
        world["records"].append(record("fact_bool", "RepeatedCredential", predicate="credential",
                                       arguments=["Bo", "Q"], value=1))
        codes = ["FINITE_WORLDS", "STATUS", "C_FACTS", "C_LOCAL", "C_CROSS_LEG", "C_BUNDLE", "C_EXEMPTION", "C_RESULT"]
        world["framework"]["rule_clauses"] = [dict(rule_id=f"Rule{i}", code=code, public_variant=0)
                                               for i, code in enumerate(codes)]
        solved = self.assert_answers(world, [3], 1)
        operative = set(solved["certificate"]["operative_rule_ids"])
        self.assertIn("Rule2", operative)  # Source contradiction tested on the rejected binding.
        self.assertIn("Rule6", operative)  # Exemption condition is evaluated on the compatible binding.
        self.assertNotIn("Rule4", operative)  # No cross-leg policy is evaluated.
        self.assertNotIn("Rule5", operative)  # No bundle policy is evaluated.
        self.assertEqual(set(solved["fixed_background_rule_ids"]), {f"Rule{i}" for i in range(len(codes))})
        conflicting = copy.deepcopy(world)
        conflicting["records"].append(record("fact_count", "ImpossibleCapacity", predicate="capacity", arguments=["Elm"], value=8))
        conflict = self.assert_answers(conflicting, [], 0)
        self.assertEqual(set(conflict["certificate"]["operative_rule_ids"]), {"Rule0", "Rule1", "Rule2"})
        self.assertEqual(conflict["certificate"]["inconsistency_core_atom_ids"], [])

    def test_certificate_step_ids_avoid_public_rule_ids(self):
        world = a_fixture()
        world["framework"]["rule_clauses"] = [dict(rule_id="s0001", code="ENTITY_IDENTITY", public_variant=0)]
        solved = self.assert_answers(world, [12], 1)
        ids = [step["step_id"] for step in solved["certificate"]["step_values"]]
        self.assertNotIn("s0001", ids)
        self.assertEqual(len(ids), len(set(ids)))
        known = set(solved["certificate"]["sufficient_atom_ids"]) | {"s0001"}
        for step in solved["certificate"]["step_values"]:
            self.assertNotIn(step["step_id"], step["inputs"])
            self.assertTrue(set(step["inputs"]) <= known)
            known.add(step["step_id"])

    def test_public_framing_ids_avoid_visible_rule_and_record_identities(self):
        messages = handwritten_a_source()
        parsed = parse_public(messages)
        baseline = solve_parsed_public(parsed, messages=messages, require_source_bytes=True)
        old_query_atom = next(a["atom_id"] for a in baseline["public_framing_atom_map"]
                              if a["field_path"] == "query.cutoff")
        for citation in ["F2", "R0"]:
            control_messages = copy.deepcopy(messages)
            control_messages[1]["content"] = control_messages[1]["content"].replace(f"[{citation}]", f"[{old_query_atom}]")
            control = parse_public(control_messages)
            solved = solve_parsed_public(control, messages=control_messages, require_source_bytes=True)
            new_query_atom = next(a["atom_id"] for a in solved["public_framing_atom_map"]
                                  if a["field_path"] == "query.cutoff")
            self.assertNotEqual(new_query_atom, old_query_atom)
            framing_ids = {a["atom_id"] for a in solved["public_framing_atom_map"]}
            rules = {r["rule_id"] for r in control["world"]["framework"]["rule_clauses"]}
            self.assertFalse(framing_ids & rules)
            self.assertFalse(framing_ids & {r["record_id"] for r in control["world"]["records"]})
        atom_rule_messages = copy.deepcopy(messages)
        atom_rule_messages[1]["content"] = atom_rule_messages[1]["content"].replace("[F2]", "[A000001]")
        atom_rule_parsed = parse_public(atom_rule_messages)
        source_ids = {a["atom_id"] for a in atom_rule_parsed["atom_spans"]}
        self.assertNotIn("A000001", source_ids)
        solved = solve_parsed_public(atom_rule_parsed, messages=atom_rule_messages, require_source_bytes=True)
        self.assertIn("A000001", solved["certificate"]["operative_rule_ids"])
        self.assertFalse(source_ids & set(solved["fixed_background_rule_ids"]))

    def test_complete_public_metadata_and_bytes_are_separately_grounded(self):
        world = a_fixture()
        world["variables"] = [dict(variable_id="DayChoice", scalar_type="Day", choices=[0, 22])]
        by_id(world, "A05")["from_day"] = dict(var="DayChoice")
        parsed, messages = metadata_fixture(world)
        self.assertIn("from_day", [a["field_path"] for a in parsed["atom_spans"] if a["record_id"] == "A05"])
        self.assertNotIn("from_day.var", [a["field_path"] for a in parsed["atom_spans"] if a["record_id"] == "A05"])
        structural = solve_parsed_public(parsed)
        self.assertEqual(structural["possible_answers"], [12])
        self.assertFalse(structural["public_grounding"]["admission_grounding"])
        self.assertEqual(structural["public_grounding"]["mode"], "STRUCTURAL_METADATA_ONLY_NOUTF8REREAD")
        with self.assertRaises(SchemaError):
            solve_parsed_public(parsed, require_source_bytes=True)
        with self.assertRaises(SchemaError):
            solve_parsed_public(parsed, messages=messages, require_source_bytes=True)
        # Actual public clauses, authored independently of the renderer, ground the certificate.
        messages = handwritten_a_source(variables=['Unreported U, a Quantity, is one of [4, 8].'])
        messages[1]["content"] = messages[1]["content"].replace("with 4 units", "with {unreported U} units")
        parsed = parse_public(messages)
        grounded = solve_parsed_public(parsed, messages=messages, require_source_bytes=True)
        self.assertEqual(grounded["possible_answers"], [0])
        self.assertTrue(grounded["public_grounding"]["admission_grounding"])
        self.assertFalse(grounded["public_grounding"]["admission_ready"])
        self.assertEqual(grounded["public_grounding"]["mode"], "REPARSED_PUBLIC_SOURCE_WITH_INDEPENDENT_PARSER")
        self.assertEqual(grounded["public_messages_sha256"], parsed["messages_sha256"])
        real_ids = {a["atom_id"] for a in parsed["atom_spans"]}
        framing_ids = {a["atom_id"] for a in grounded["public_framing_atom_map"]}
        self.assertEqual(set(grounded["certificate"]["sufficient_atom_ids"]), real_ids | framing_ids)
        self.assertTrue(any(a.startswith("Q") for a in framing_ids))
        for step in grounded["certificate"]["step_values"]:
            self.assertTrue(set(step["inputs"]) <= real_ids | framing_ids |
                            set(grounded["certificate"]["operative_rule_ids"]) |
                            {s["step_id"] for s in grounded["certificate"]["step_values"]})

    def test_public_metadata_rejects_incomplete_duplicate_wrong_field_hash_and_span(self):
        messages = handwritten_a_source()
        parsed = parse_public(messages)
        controls = []
        missing = copy.deepcopy(parsed)
        missing["atom_spans"] = [a for a in missing["atom_spans"] if not (a["record_id"] == "R4" and a["field_path"] == "kind")]
        controls.append(missing)
        duplicate = copy.deepcopy(parsed)
        duplicate["atom_spans"].append(copy.deepcopy(duplicate["atom_spans"][0]))
        controls.append(duplicate)
        duplicate_id = copy.deepcopy(parsed)
        duplicate_id["atom_spans"][1]["atom_id"] = duplicate_id["atom_spans"][0]["atom_id"]
        controls.append(duplicate_id)
        wrong_hash = copy.deepcopy(parsed)
        wrong_hash["atom_spans"][0]["typed_value_sha256"] = "0" * 64
        controls.append(wrong_hash)
        wrong_field = copy.deepcopy(parsed)
        wrong_field["atom_spans"][0]["field_path"] = "document_id"
        controls.append(wrong_field)
        wrong_span = copy.deepcopy(parsed)
        wrong_span["atom_spans"][0]["span"].update(utf8_start=0, utf8_end=1)
        controls.append(wrong_span)
        missing_record = copy.deepcopy(parsed)
        missing_record["record_spans"].pop()
        controls.append(missing_record)
        missing_frame = copy.deepcopy(parsed)
        missing_frame["framing_spans"].pop()
        controls.append(missing_frame)
        for value in [None, (), ["assertion"]]:
            malformed = copy.deepcopy(parsed)
            malformed["unparsed_assertions"] = value
            controls.append(malformed)
        for control in controls:
            with self.subTest(control=controls.index(control)):
                with self.assertRaises(SchemaError):
                    solve_parsed_public(control, messages=messages, require_source_bytes=True)

    def test_public_source_bytes_reject_hash_coverage_overflow_and_utf8_split(self):
        messages = handwritten_a_source(titles=["Récords One", "Records Two", "Records Three"])
        parsed = parse_public(messages)
        wrong_messages = copy.deepcopy(messages)
        wrong_messages[1]["content"] += "extra assertion"
        with self.assertRaises(SchemaError):
            solve_parsed_public(parsed, messages=wrong_messages, require_source_bytes=True)
        wrong_role = copy.deepcopy(messages)
        wrong_role[0]["role"] = "user"
        with self.assertRaises(SchemaError):
            solve_parsed_public(parsed, messages=wrong_role, require_source_bytes=True)
        for kind in ["gap", "overlap", "overflow", "utf8"]:
            control = copy.deepcopy(parsed)
            if kind == "gap":
                control["coverage_spans"][1]["span"]["utf8_start"] = 1
            elif kind == "overlap":
                control["coverage_spans"].append(copy.deepcopy(control["coverage_spans"][1]))
            elif kind == "overflow":
                control["framing_spans"][0]["span"]["utf8_end"] = len(messages[1]["content"].encode("utf-8")) + 1
            else:
                mid_character = messages[1]["content"].encode("utf-8").find("é".encode("utf-8")) + 1
                control["framing_spans"][0]["span"].update(utf8_start=mid_character, utf8_end=mid_character+1)
            with self.subTest(kind=kind):
                with self.assertRaises(SchemaError):
                    solve_parsed_public(control, messages=messages, require_source_bytes=True)

    def test_public_source_reread_rejects_rehashed_query_and_record_imposture(self):
        messages = handwritten_a_source()
        parsed = parse_public(messages)
        original = solve_parsed_public(parsed, messages=messages, require_source_bytes=True)
        self.assertEqual(original["possible_answers"], [0])
        tampered = copy.deepcopy(parsed)
        tampered["world"]["query"]["cutoff"] = 2
        frame = next(f for f in tampered["framing_spans"] if f["field_path"] == "query.cutoff")
        frame["typed_value_sha256"] = hashlib.sha256(b"2").hexdigest()
        self.assertEqual(solve_parsed_public(tampered)["possible_answers"], [5])
        with self.assertRaises(SchemaError):
            solve_parsed_public(tampered, messages=messages, require_source_bytes=True)
        tampered = copy.deepcopy(parsed)
        by_id(tampered["world"], "R6")["day"] = 12  # Cancel changes from before cutoff to after it.
        atom = next(a for a in tampered["atom_spans"] if a["record_id"] == "R6" and a["field_path"] == "day")
        atom["typed_value_sha256"] = hashlib.sha256(b"12").hexdigest()
        self.assertEqual(solve_parsed_public(tampered)["possible_answers"], [11])
        with self.assertRaises(SchemaError):
            solve_parsed_public(tampered, messages=messages, require_source_bytes=True)


class StrictOutputChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution = solve_public_world(a_fixture())

    def test_exact_formats_and_wrong_scalar_status(self):
        for fmt, target in [("JSON", '{"status":"DETERMINATE","answer":12}'), ("LINE", "DETERMINATE|12"), ("TOKEN", "12")]:
            self.assertEqual(exact_target(fmt, self.solution), target)
            self.assertTrue(validate_answer(fmt, target.encode(), self.solution)["exact_correct"])
            self.assertFalse(validate_answer(fmt, target + "\n", self.solution)["exact_correct"])
        wrong = validate_answer("TOKEN", "13", self.solution)
        self.assertTrue(wrong["format_correct"])
        self.assertFalse(wrong["semantic_correct"])
        self.assertFalse(validate_answer("TOKEN", "UNKNOWN", self.solution)["semantic_correct"])

    def test_json_duplicate_extra_missing_wrong_types_rejected(self):
        for text in ['{"status":"DETERMINATE","answer":12,"answer":12}',
                     '{"status":"DETERMINATE","answer":12,"extra":0}', '{"status":"DETERMINATE"}',
                     '{"status":"DETERMINATE","answer":true}', '{"status":"DETERMINATE","answer":12.0}',
                     '{"status":"DETERMINATE","answer":"12"}', '{"status":"DETERMINATE","answer":NaN}',
                     '{"status":"DETERMINATE","answer":012}', '["DETERMINATE",12]']:
            result = validate_answer("JSON", text, self.solution)
            self.assertFalse(result["format_correct"], text)
            self.assertFalse(result["semantic_correct"], text)

    def test_canonical_json_whitespace_key_order_and_eos_text(self):
        for text in ['{"answer":12,"status":"DETERMINATE"}', '{"status": "DETERMINATE","answer":12}',
                     ' {"status":"DETERMINATE","answer":12}', '{"status":"DETERMINATE","answer":12}\n']:
            result = validate_answer("JSON", text, self.solution)
            self.assertTrue(result["semantic_correct"])
            self.assertFalse(result["format_correct"])
        for fmt, text in [("TOKEN", "12<eos>"), ("TOKEN", "12 explanation"), ("TOKEN", "012"),
                          ("TOKEN", "+12"), ("TOKEN", "12.0"), ("LINE", "DETERMINATE |12"),
                          ("LINE", "DETERMINATE|012"), ("JSON", exact_target("JSON", self.solution) + "<eos>")]:
            self.assertFalse(validate_answer(fmt, text, self.solution)["format_correct"])
        self.assertFalse(validate_answer("TOKEN", b"\xff", self.solution)["exact_correct"])
        for fmt, text in [("TOKEN", "9" * 5000), ("LINE", "DETERMINATE|" + "9" * 5000)]:
            self.assertFalse(validate_answer(fmt, text, self.solution)["format_correct"])

    def test_diagnostic_outputs_null_and_no_model_eos_assumption(self):
        unknown = dict(status="UNDERDETERMINED", possible_answers=[5, 12])
        conflict = dict(status="INCONSISTENT", possible_answers=[])
        for solution, tokens in [(unknown, ['{"status":"UNDERDETERMINED","answer":null}', "UNDERDETERMINED|-", "UNKNOWN"]),
                                 (conflict, ['{"status":"INCONSISTENT","answer":null}', "INCONSISTENT|-", "CONFLICT"])]:
            for fmt, text in zip(["JSON", "LINE", "TOKEN"], tokens):
                result = validate_answer(fmt, text, solution)
                self.assertTrue(result["exact_correct"])
                self.assertEqual(result["native_eos"], "NOT_MEASURED")
        self.assertFalse(validate_answer("JSON", '{"status":"INCONSISTENT","answer":0}', conflict)["format_correct"])


if __name__ == "__main__":
    unittest.main()
