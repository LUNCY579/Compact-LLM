"""Fresh, query-unsolved dossier authoring; construction requires two birth gates.

This module never renders, solves a query, writes a registry, or exports rows.
Allocation and primitive one-step source arithmetic are safe before birth. The
only full-world entry point is construct_worlds, which first requires externally
pinned catalogue and fixture review identities and explicit orchestration consent.
Intended diagnostic cells are recipes, never observed answerability classifications.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import itertools
from pathlib import Path
import random

from . import fixture_exclusions, schema

ROOT = schema.ROOT
MODULE = Path(__file__).resolve()
MECHANISMS = ("commitment_ledger", "guarded_transition_learning", "policy_feasibility")
FORMATS = ("JSON", "LINE", "TOKEN")
BIRTH_CONSENT = "ROOT_CONFIRMED_CATALOGUE_AND_FIXTURE_GATES"
PIN_FIELDS = frozenset({"catalogue_freeze_receipt_sha256", "fixture_reviewed_inventory_pin_sha256",
                        "fixture_inventory_sha256", "fixture_exclusions_sha256",
                        "fixture_completeness_seal_sha256"})
LIMITATIONS = (
    "Fresh original fact draws are declared provenance, not proof of historical novelty.",
    "NO_KNOWN_MATCH covers only the fixture firewall's stated complete-spec comparisons.",
    "Dependency, actual status, guard activation, and structural influence remain untested until reservation.",
    "No rejection replenishment, outcome-based resampling, or unallocated repair is performed.",
)


class BirthGateError(ValueError):
    """A complete frozen/reviewed identity or explicit birth consent is absent."""


@dataclass(frozen=True)
class Attempt:
    parent_id: str
    reservation_id: str
    mechanism: str
    mechanism_index: int
    structure_id: str
    structure_index: int
    level: int
    case_index: int
    role: str
    format: str
    intended_status: str
    content_family_id: str
    source_domain_id: str
    producer_family_id: str
    stream_key: str

    def cell_metadata(self):
        return {key: getattr(self, key) for key in (
            "mechanism", "structure_id", "structure_index", "level", "case_index", "role", "format")}


def isolated_rng(seed, *stream_parts):
    """A local stream independent of global RNG and unrelated authoring choices."""
    if type(seed) is not int or any(type(p) is not str for p in stream_parts):
        raise ValueError("seed must be an integer and stream labels must be strings")
    raw = schema.canonical_bytes(["DD-FRESH-WORLD-STREAM-v1", seed, *stream_parts])
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def allocation(seed):
    """Return all 54 slots, roles, formats, and planned status recipes; no worlds."""
    isolated_rng(seed, "allocation")  # strict primitive seed validation only
    namespace = hashlib.sha256(schema.canonical_bytes(["DD-FRESH-ALLOCATION-v1", seed])).hexdigest()[:12]
    result = []
    for mi, mechanism in enumerate(MECHANISMS):
        for si in range(3):
            for level in (1, 2):
                for case in range(3):
                    ordinal = len(result)
                    status = "DETERMINATE" if case < 2 else (
                        "UNDERDETERMINED" if (si + level - 1) % 2 == 0 else "INCONSISTENT")
                    result.append(Attempt(
                        parent_id=f"Dw{namespace}{ordinal:02d}", reservation_id=f"Dr{namespace}{ordinal:02d}",
                        mechanism=mechanism, mechanism_index=mi, structure_id="ABC"[mi] + str(si),
                        structure_index=si, level=level, case_index=case,
                        role="EXPOSED_DEV" if si == 2 else "PROSPECTIVE_TRAIN",
                        format=FORMATS[(mi + si + (si + 2) * (level - 1) + case) % 3],
                        intended_status=status, content_family_id=f"dd-content-{namespace}-{ordinal:02d}",
                        source_domain_id="CommunityWorkshop", producer_family_id="dd-original-author-v1",
                        stream_key=f"{mi}/{si}/{level}/{case}"))
    return tuple(result)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound_read(path, expected):
    if type(expected) is not str or not schema.HASH.fullmatch(expected):
        raise BirthGateError("external pin is not a complete lowercase SHA256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise BirthGateError("external byte pin mismatch: " + str(path))
    return schema.loads_json(raw)


def verify_birth_gates(expected_pins):
    """Read-only gate check; expected pin bytes must come from the root reviewer.

    The caller cannot substitute an unreviewed producer completeness seal for the
    independently reviewed fixture pin. Neither a missing pin nor DEFER opens birth.
    """
    if type(expected_pins) is not dict or set(expected_pins) != PIN_FIELDS:
        raise BirthGateError("requires all five externally reviewed birth pins")
    try:
        base = MODULE.parent
        cat = schema.load_catalogue()
        receipt = _bound_read(base / "catalogue_freeze_receipt.json", expected_pins["catalogue_freeze_receipt_sha256"])
        fields = cat["gate_receipt_schema"]["catalogue_freeze_receipt.json"]["exact_fields"]
        schema._closed(receipt, fields, "catalogue freeze receipt")
        if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1 or receipt["status"] != "INDEPENDENTLY_REVIEWED_FROZEN":
            raise BirthGateError("catalogue is not independently reviewed and frozen")
        live = {"catalogue_sha256": _hash(schema.CATALOGUE_PATH),
                "design_sha256": _hash(schema.BLUEPRINT / "DESIGN.md"),
                "contract_sha256": _hash(schema.BLUEPRINT / "CONTRACT.json")}
        if receipt["catalogue_id"] != cat["catalogue_version"] or any(receipt[k] != v for k, v in live.items()):
            raise BirthGateError("catalogue or reviewed blueprint byte identity changed")
        names = {"schema.py", "renderer.py", "public_parser.py", "independent_solver.py"}
        if type(receipt["code_identity"]) is not dict or set(receipt["code_identity"]) != names:
            raise BirthGateError("frozen catalogue must bind all four exact core code pins")
        identities = {name: _hash(base / name) for name in names}
        if identities != receipt["code_identity"]:
            raise BirthGateError("frozen core code identity changed")
        review_path = (ROOT / receipt["review_path"]).resolve()
        review_path.relative_to(ROOT)
        review = _bound_read(review_path, receipt["independent_review_sha256"])
        if (review.get("status") != "PASS_INDEPENDENT_CATALOGUE_AND_IMPLEMENTATION_REVIEW"
                or review.get("unresolved_critical_or_major_findings") != []
                or review.get("producer_identity") != dict(identities, **{"catalogue.json": live["catalogue_sha256"]})
                or any(review.get(k) != live[k] for k in ("design_sha256", "contract_sha256"))):
            raise BirthGateError("catalogue independent review does not bind the live complete implementation")
        fixture_pin = _bound_read(fixture_exclusions.REVIEWED_INVENTORY_PIN_PATH,
                                 expected_pins["fixture_reviewed_inventory_pin_sha256"])
        _bound_read(fixture_exclusions.INVENTORY_PATH, expected_pins["fixture_inventory_sha256"])
        if _hash(Path(fixture_exclusions.__file__)) != expected_pins["fixture_exclusions_sha256"]:
            raise BirthGateError("externally reviewed fixture firewall code pin changed")
        seal = expected_pins["fixture_completeness_seal_sha256"]
        if seal != fixture_exclusions.COMPLETENESS_SEAL_SHA256 or _hash(fixture_exclusions.COMPLETENESS_SEAL_PATH) != seal:
            raise BirthGateError("fixture completeness seal pin changed")
        if (fixture_pin.get("status") != "REVIEWED_EXCLUSION_PIN"
                or fixture_pin.get("inventory_sha256") != expected_pins["fixture_inventory_sha256"]
                or fixture_pin.get("completeness_seal_sha256") != seal):
            raise BirthGateError("independently reviewed fixture inventory identity is absent or changed")
        inventory = fixture_exclusions.load_inventory()
        if inventory.get("_runtime_inventory_review") != "REVIEWED_EXTERNAL_PIN":
            raise BirthGateError("fixture inventory has producer completeness only")
        return {"schema_version": 1, "decision": "BIRTH_PIN_GATES_VERIFIED",
                "external_pins": copy.deepcopy(expected_pins), **live,
                "catalogue_id": cat["catalogue_version"], "core_code_identity": identities,
                "constructor_sha256": _hash(MODULE), "fixture_scope": copy.deepcopy(fixture_exclusions.COVERAGE),
                "solver_called": False, "registry_written": False}
    except BirthGateError:
        raise
    except (OSError, KeyError, TypeError, ValueError, schema.SchemaError,
            fixture_exclusions.ComparisonDeferred) as exc:
        raise BirthGateError("birth gate unavailable or invalid: " + str(exc)) from exc


def apply_program(program, state):
    """Apply one sampled source program to one bench state; never a test trace."""
    if type(program) is not dict or set(program) != {"program_id", "guard", "action"}:
        raise ValueError("program must have exactly program_id, guard, action")
    if any(type(v) is not str for v in program.values()) or not schema.ID.fullmatch(program["program_id"]):
        raise ValueError("invalid opaque program ID")
    if program["guard"] not in ("always", "m0", "m1", "x0", "x1", "y0", "y1"):
        raise ValueError("unknown guard")
    if program["action"] not in ("flip_m", "flip_x", "flip_y", "swap_mx", "swap_xy", "copy_mx", "copy_xm", "copy_xy", "copy_yx"):
        raise ValueError("unknown action")
    if type(state) is not dict or set(state) != set("mxy") or any(type(v) is not int or v not in (0, 1) for v in state.values()):
        raise ValueError("bench state must contain three integer bits")
    before, after = dict(state), dict(state)
    guard = program["guard"]
    if guard != "always" and before[guard[0]] != int(guard[1]):
        return after
    opcode, operands = program["action"].split("_")
    if opcode == "flip": after[operands] = 1 - before[operands]
    elif opcode == "swap": after[operands[0]], after[operands[1]] = before[operands[1]], before[operands[0]]
    else: after[operands[1]] = before[operands[0]]
    return after


def truth_table(program):
    return tuple(tuple(apply_program(program, dict(zip("mxy", bits)))[r] for r in "mxy")
                 for bits in itertools.product((0, 1), repeat=3))


def sample_operator_menu(rng, *, count=3, latent_program=None):
    """Sample distinct source functions; no trace/output scoring or world retries."""
    if type(count) is not int or count not in (3, 4) or not isinstance(rng, random.Random):
        raise ValueError("menu requires a local Random and three or four programs")
    catalogue = schema.load_catalogue()
    pool = [{"program_id": "P", "guard": g, "action": a}
            for g in catalogue["guard_realizations"] for a in catalogue["action_realizations"]]
    representatives = {}
    for p in pool: representatives.setdefault(truth_table(p), p)
    values = list(representatives.values())
    if latent_program is None:
        values = rng.sample(values, count)
        selected_index = rng.randrange(count)
    else:
        target = truth_table(latent_program)
        if target not in representatives: raise ValueError("latent source function outside language")
        values = [copy.deepcopy(latent_program), *rng.sample([p for p in values if truth_table(p) != target], count - 1)]
        rng.shuffle(values)
        selected_index = next(i for i, p in enumerate(values) if truth_table(p) == target)
    ids = rng.sample(range(100, 1000), count)
    for p, n in zip(values, ids): p["program_id"] = "Program" + str(n)
    return copy.deepcopy(values), values[selected_index]["program_id"]


def author_bench_examples(candidates, latent_program_id, rng):
    """Author a minimal distinguishing one-step calibration set from source tables.

    Only the sampled operator's eight-state table is used. This does not classify
    a world, execute its later test trace, or measure answer sensitivity.
    """
    if not isinstance(rng, random.Random) or type(candidates) is not list or len(candidates) not in (3, 4):
        raise ValueError("invalid source calibration menu")
    tables = [truth_table(p) for p in candidates]
    ids = [p["program_id"] for p in candidates]
    if len(set(tables)) != len(tables) or len(set(ids)) != len(ids) or latent_program_id not in ids:
        raise ValueError("source menu must have distinct function/ID members and a latent member")
    target = ids.index(latent_program_id)
    state_order = rng.sample(range(8), 8)
    chosen = None
    for width in range(1, 9):
        choices = [indices for indices in itertools.combinations(state_order, width)
                   if all(any(tables[target][i] != table[i] for i in indices)
                          for n, table in enumerate(tables) if n != target)]
        if choices:
            chosen = rng.choice(choices)
            break
    states = list(itertools.product((0, 1), repeat=3))
    return [{"input": dict(zip("mxy", states[i])), "output": dict(zip("mxy", tables[target][i]))} for i in chosen]


_FIRST = ("Alder", "Briar", "Cedar", "Dahlia", "Elm", "Fern", "Hazel", "Iris", "Juniper", "Laurel", "Maple", "Orchid", "Pine", "Reed", "Sage", "Willow")
_LAST = ("Brook", "Cove", "Field", "Grove", "Hill", "Lane", "Meadow", "Point", "Ridge", "Shore", "Vale", "Wood")


class _Source:
    """Private authoring accumulator, created only beyond verified birth gates."""
    def __init__(self, attempt, seed, cat):
        self.attempt, self.cat = attempt, cat
        self.facts = isolated_rng(seed, attempt.stream_key, "factual-recipe")
        self.names = isolated_rng(seed, attempt.stream_key, "labels-identifiers")
        self.presentation = isolated_rng(seed, attempt.stream_key, "source-placement")
        self.entities, self.variables, self.records = [], [], []
        self.labels, self.ids, self.event_ids = set(), set(), set()
        self.documents = []

    def entity(self, kind):
        while True:
            label = self.names.choice(_FIRST) + " " + self.names.choice(_LAST)
            if label not in self.labels: break
        self.labels.add(label)
        eid = self.symbol("E")
        self.entities.append({"entity_id": eid, "kind": kind, "public_label": label})
        return eid

    def symbol(self, prefix):
        while True:
            value = prefix + str(self.names.randrange(100000, 10000000))
            if value not in self.ids: break
        self.ids.add(value)
        return value

    def event(self): return self.symbol("Event")

    def source_documents(self, descriptors):
        for purpose, title in descriptors:
            self.documents.append({"document_id": self.symbol("Doc"), "purpose": purpose, "title": title,
                                   "dated_day": self.facts.randint(1, 28), "record_ids": [],
                                   "style": "controlled_prose"})

    def record(self, kind, document, **fields):
        # Citation spelling and printed positions use an isolated presentation RNG.
        rid = self.symbol("R")
        doc = self.documents[document % len(self.documents)]
        record = {"record_id": rid, "document_id": doc["document_id"], "kind": kind, **copy.deepcopy(fields)}
        self.records.append(record); doc["record_ids"].append(rid)
        return record

    def variable(self, typ, choices):
        name = self.symbol("Choice")
        self.variables.append({"variable_id": name, "scalar_type": typ, "choices": sorted(set(choices))})
        return {"var": name}

    def finish(self, query):
        a = self.attempt
        for doc in self.documents: self.presentation.shuffle(doc["record_ids"])
        self.presentation.shuffle(self.records)
        self.presentation.shuffle(self.entities)
        self.presentation.shuffle(self.documents)
        rules = [{"rule_id": self.symbol("Rule"), "code": code, "public_variant": self.presentation.randrange(2)}
                 for code in self.cat["framework_required"][a.mechanism]]
        return {"world_schema": self.cat["world_schema"], "parent_id": a.parent_id,
                "reservation_id": a.reservation_id, "mechanism": a.mechanism, "structure_id": a.structure_id,
                "level": a.level, "entities": self.entities, "variables": self.variables,
                "documents": self.documents, "records": self.records,
                "framework": {"semantic_version": self.cat["contract_id"], "rule_clauses": rules,
                              "closed_candidate_list": True, "unknown_defaults": "NONE",
                              "source_assertions_are_conjunctive": True}, "query": query, "format": a.format,
                "content_family_id": a.content_family_id, "source_domain_id": a.source_domain_id,
                "producer_family_id": a.producer_family_id}


def _ledger_source(s):
    a, rng = s.attempt, s.facts
    descriptors = [("alias_crosswalk", "Workshop name register"), ("delegation_circular", "Equipment editing circular"),
                   ("booking_minutes", "Workshop booking minutes"), ("delivery_slips", "Equipment delivery slips")]
    if a.level == 2: descriptors.append(("maintenance_log", "Equipment continuation log"))
    s.source_documents(descriptors)
    cutoff = rng.randint(22, 29)
    group, group_middle, query_group = [s.entity("group") for _ in range(3)]
    s.record("alias", 0, entity_kind="group", left=group, right=group_middle)
    s.record("alias", 2, entity_kind="group", left=group_middle, right=query_group)
    nested = s.entity("group") if a.structure_index == 1 else group
    if nested != group:
        s.record("membership", 3, child=nested, parent=group, from_day=0, through_day=31)
    owners, signers = [], []
    n = 2 if a.level == 1 else 3
    if a.structure_index == 2: n += 1
    for i in range(n):
        owner, mid, used = [s.entity("sponsor") for _ in range(3)]
        signer = s.entity("signer")
        owners.append(used); signers.append(signer)
        s.record("alias", i, entity_kind="sponsor", left=owner, right=mid)
        s.record("alias", i + 2, entity_kind="sponsor", left=mid, right=used)
        s.record("membership", i + 1, child=owner, parent=nested, from_day=rng.randint(0, 2), through_day=31)
        s.record("delegation", i + 3, signer=signer, owner=used, from_day=0, through_day=31)

    def quantity(i):
        # Disjoint factual ranges for the first contributing term across fresh cases.
        return 2 + 3 * a.case_index + rng.randrange(2) if i == 0 else rng.randint(2, 10)

    def reported(q, i):
        if a.intended_status == "UNDERDETERMINED" and i == 0:
            return s.variable("Quantity", [q, min(12, q + 1)])
        return q

    if a.structure_index in (0, 1):
        for i in range(n):
            commitment = s.entity("commitment")
            start, amend = s.event(), s.event()
            day = rng.randint(3, 7)
            owner, signer = owners[i], signers[i]
            if a.structure_index == 1:
                outside = s.entity("sponsor")
                s.record("delegation", i + 2, signer=signer, owner=outside, from_day=0, through_day=31)
                owner = outside
            s.record("ledger_start", i + 2, event_id=start, day=day, commitment=commitment,
                     owner=owner, quantity=rng.randint(0, 12), signer=signer)
            q = quantity(i)
            s.record("ledger_amend", i + 3, event_id=amend, day=day + 2, commitment=commitment,
                     predecessor=start, quantity=reported(q, i), signer=signer)
            if a.structure_index == 1:
                s.record("ledger_transfer", i + 1, event_id=s.event(), day=day + 5, commitment=commitment,
                         predecessor=amend, owner=owners[i], signer=signer)
            if a.intended_status == "INCONSISTENT" and i == 0:
                s.record("ledger_amend", i + 1, event_id=s.event(), day=day + 2, commitment=commitment,
                         predecessor=start, quantity=min(12, q + 2), signer=signer)
        if a.structure_index == 0:
            cancelled = s.entity("commitment")
            start = s.event()
            s.record("ledger_start", 2, event_id=start, day=8, commitment=cancelled,
                     owner=owners[-1], quantity=rng.randint(2, 9), signer=signers[-1])
            s.record("ledger_cancel", 3, event_id=s.event(), day=15, commitment=cancelled,
                     predecessor=start, signer=signers[-1])
    else:
        parent, child_a, child_b = [s.entity("commitment") for _ in range(3)]
        start, split, amend = s.event(), s.event(), s.event()
        day = rng.randint(3, 6)
        # Positive parent amount below either child's allowed amount makes parent
        # retirement meaningful without a conservation shortcut in any birth cell.
        s.record("ledger_start", 2, event_id=start, day=day, commitment=parent, owner=owners[0], quantity=1, signer=signers[0])
        s.record("ledger_split", 3, event_id=split, day=day + 3, commitment=parent, predecessor=start,
                 children=[{"commitment": child_a, "owner": owners[0], "quantity": rng.randint(2, 4)},
                           {"commitment": child_b, "owner": owners[1], "quantity": rng.randint(2, 4)}], signer=signers[0])
        q = quantity(0)
        s.record("ledger_amend", 1, event_id=amend, day=day + 6, commitment=child_a,
                 predecessor=split, quantity=reported(q, 0), signer=signers[0])
        if a.level == 2:
            s.record("ledger_amend", 4, event_id=s.event(), day=day + 7, commitment=child_b,
                     predecessor=split, quantity=rng.randint(5, 7), signer=signers[1])
        if a.intended_status == "INCONSISTENT":
            s.record("ledger_amend", 0, event_id=s.event(), day=day + 6, commitment=child_a,
                     predecessor=split, quantity=min(12, q + 2), signer=signers[0])
        for i in range(2, n):
            s.record("ledger_start", i, event_id=s.event(), day=day + 1, commitment=s.entity("commitment"),
                     owner=owners[i], quantity=rng.randint(2, 6), signer=signers[i])
    return {"mechanism": a.mechanism, "cutoff": cutoff, "group_label": query_group}


def _transition_source(s):
    a, rng = s.attempt, s.facts
    docs = [("calibration_notes", "Sorting station program notes"), ("calibration_notes", "Sorting station bench notes"),
            ("maintenance_log", "Sorting station event log"), ("booking_minutes", "Sorting station shift minutes")]
    if a.level == 2: docs.append(("delivery_slips", "Sorting station continuation slips"))
    s.source_documents(docs)
    operators = [s.entity("operator") for _ in range(2 if a.level == 1 else 3)]
    # Guarded swaps avoid imposing linear fork symmetry; their actual influence is
    # intentionally left for the registered independent dependency verifier.
    archetypes = [{"program_id": "Latent", "guard": "y" + str(rng.randrange(2)), "action": "swap_mx"},
                  {"program_id": "Latent", "guard": "m" + str(rng.randrange(2)), "action": "flip_y"},
                  {"program_id": "Latent", "guard": "x" + str(rng.randrange(2)), "action": "copy_yx"}]
    rng.shuffle(archetypes)
    for i, operator in enumerate(operators):
        menu, latent = sample_operator_menu(rng, count=3 + ((a.case_index + i) % 2), latent_program=archetypes[i])
        s.record("operator_menu", i, operator=operator, candidates=menu)
        examples = author_bench_examples(menu, latent, rng)
        if a.intended_status == "UNDERDETERMINED" and i == 0:
            # Born with an unconstrained operator; not a damaged determinate parent.
            examples = []
        for j, example in enumerate(examples):
            s.record("bench_example", i + j + 1, operator=operator, **example)
        if a.intended_status == "INCONSISTENT" and i == 0:
            observation = examples[0]
            conflicting = copy.deepcopy(observation["output"])
            conflicting["m"] = 1 - conflicting["m"]
            s.record("bench_example", 3, operator=operator, input=observation["input"], output=conflicting)
    ticks = iter(sorted(rng.sample(range(1, 31), 10)))
    def apply(station, operator, document):
        s.record("trace_apply", document, tick=next(ticks), station=station, operator=operator)
    def initialized():
        station = s.entity("station")
        s.record("station_initial", 2, station=station, state={r: rng.randrange(2) for r in "mxy"})
        return station
    if a.structure_index == 0:
        station = initialized()
        schedule = [0, 1, 0] if a.level == 1 else [0, 1, 2, 0, 2, 1, 0, 1]
        for j, index in enumerate(schedule): apply(station, operators[index], j + 1)
        finals, combine = [station], "single"
    elif a.structure_index == 1:
        left, right = initialized(), initialized()
        apply(left, operators[0], 2); apply(right, operators[1], 3)
        s.record("trace_splice", 0, tick=next(ticks), source=left, source_register=rng.choice("mxy"),
                 destination=right, destination_register=rng.choice("mxy"))
        apply(right, operators[0], 1)
        if a.level == 2:
            for j, (station, index) in enumerate(((left, 2), (right, 1), (left, 0), (right, 2))):
                apply(station, operators[index], j + 1)
        finals, combine = [left, right], "xor"
    else:
        origin = initialized()
        left, right, merged = [s.entity("station") for _ in range(3)]
        s.record("trace_fork", 3, tick=next(ticks), source=origin, destinations=[left, right])
        if a.level == 1:
            apply(left, operators[0], 1); apply(right, operators[1], 2)
        else:
            for j, (station, index) in enumerate(((left, 0), (right, 1), (left, 2), (right, 0), (left, 1), (right, 2))):
                apply(station, operators[index], j)
        s.record("trace_join", 0, tick=next(ticks), sources=sorted([left, right]), destination=merged, operation="xor")
        apply(merged, operators[-1], 1)
        finals, combine = [merged], "single"
    return {"mechanism": a.mechanism, "final_stations": finals, "combine": combine, "encoding": "4m+2x+y"}


def _policy_source(s):
    a, rng = s.attempt, s.facts
    docs = [("session_proposals", "Workshop session proposals"), ("room_roster", "Workshop room register"),
            ("policy_amendments", "Workshop session policies"), ("booking_minutes", "Workshop resource minutes")]
    if a.level == 2: docs.append(("delegation_circular", "Workshop credential circular"))
    s.source_documents(docs)
    credential, group = s.entity("credential"), s.entity("group")
    plan_count = 3 if a.level == 1 or a.structure_index == 1 else 4
    shared_kit = s.entity("kit") if a.structure_index == 2 else None
    common_slot = rng.randint(0, 7)
    plans, details = [], []
    demands = rng.sample([1, 2, 3] if plan_count == 3 else [1, 2, 3, 4], plan_count)
    # The low bundle diagnostic recipe places its unreported source on the
    # demand-two member. This is a birth choice, not a solved-outcome selection.
    unreported_plan = demands.index(2) if shared_kit and a.level == 1 else 1
    def fact(predicate, args, value, document):
        return s.record("fact_count" if predicate in ("capacity", "kit_units") else "fact_bool",
                        document, predicate=predicate, arguments=args, value=value)
    for i in range(plan_count):
        plan = s.entity("plan"); plans.append(plan)
        legs = []
        first_slot = rng.randint(0, 4)
        for j in range(2 if a.structure_index == 1 else 1):
            room, leader = s.entity("room"), s.entity("person")
            kit = shared_kit or s.entity("kit")
            slot = common_slot if shared_kit else first_slot + 2 * j
            leg = {"leg_id": s.symbol("Leg"), "slot": slot, "room": room, "kit": kit,
                   "leader": leader, "people": [leader], "group": group,
                   "kit_demand": demands[i] if shared_kit else rng.randint(1, 3)}
            legs.append(leg)
            cap = rng.randint(2 + 2 * a.case_index, 3 + 2 * a.case_index) if i == j == 0 else rng.randint(1, 8)
            fact("capacity", [room], cap, i + j + 1)
            available = 1
            if a.intended_status == "UNDERDETERMINED" and i == unreported_plan and j == 0:
                available = s.variable("Bit", [0, 1])
            fact("room_available", [room, slot], available, i + j + 3)
            # The narrow exception binds the first leg; the other leg still needs
            # its own credential. No missing fact is supplied by the exception.
            fact("credential", [leader, credential], 0 if i == j == 0 else 1, i + j + 2)
            if a.level == 2 and shared_kit is None:
                fact("kit_available", [kit, slot], 1, i + j + 4)
            if a.structure_index == 0 and a.level == 2:
                fact("member", [leader, group], 1, i + 3)
            elif i == j == 0:
                fact("member", [leader, group], 1, 3)
            if a.intended_status == "INCONSISTENT" and i == 1 and j == 0:
                fact("room_available", [room, slot], 0, 0)
        s.record("session_plan", i, plan=plan, legs=legs)
        details.append(legs)
    policies = {}
    codes = [("each_leg", "capacity_ok"), ("each_leg", "room_available"), ("each_leg", "leader_credential")]
    if a.level == 2: codes.append(("each_leg", "kit_available"))
    if a.structure_index == 0 and a.level == 2: codes.append(("each_leg", "all_members"))
    if a.structure_index == 1:
        codes.append(("cross_leg", "slots_increase"))
        if a.level == 2: codes.append(("cross_leg", "rooms_differ"))
    if shared_kit: codes.append(("each_bundle", "shared_kit_quota"))
    for i, (scope, code) in enumerate(codes):
        policy = s.symbol("Policy"); policies[code] = policy
        predicate = {"code": code, **({"credential": credential} if code == "leader_credential" else {})}
        s.record("base_policy", i + 2, policy_id=policy, scope=scope, predicate=predicate)
    first = details[0][0]
    selector = {"kind": "leg", "plan": plans[0], "leg_id": first["leg_id"]} if a.structure_index == 1 else {"kind": "plan", "plan": plans[0]}
    s.record("exception", 2, exception_id=s.symbol("Exception"), base_policy_id=policies["leader_credential"],
             selector=selector, condition={"predicate": "member", "arguments": [first["leader"], group], "comparison": "eq", "value": 1})
    if shared_kit:
        if a.level == 2: fact("kit_available", [shared_kit, common_slot], 1, 4)
        # Quota is an authored resource fact; no bundle decision/count is computed.
        fact("kit_units", [shared_kit, common_slot], 3 if a.level == 1 else 5, 3)
        candidates = []
        for i, pair in enumerate(itertools.combinations(plans, 2)):
            bundle = s.entity("bundle"); candidates.append(bundle)
            s.record("bundle", i + 1, bundle=bundle, plans=sorted(pair))
        kind = "bundle"
    else:
        candidates, kind = plans, "plan"
    return {"mechanism": a.mechanism, "candidate_ids": sorted(candidates), "candidate_kind": kind, "operation": "count_feasible"}


def construct_worlds(*, seed, expected_pins, birth_authorization=None):
    """Return unsolved source objects and the complete 54-attempt ledger.

    The root must first independently confirm both review gates and pass the exact
    authorization string. Returned complete {world, source_cell_metadata} pairs
    must be reserved before any final-query solve, rendering, or dependency test.
    Failed schema/firewall attempts remain quarantined in this ledger. A known
    fixture match, DEFER, or unreviewed inventory never yields a usable world.
    """
    if birth_authorization != BIRTH_CONSENT:
        raise BirthGateError("root must explicitly confirm both gates before full-world birth")
    gates = verify_birth_gates(expected_pins)
    slots, cat = allocation(seed), schema.load_catalogue()
    worlds, attempts = [], []
    for attempt in slots:
        ledger = {**asdict(attempt), "revision": 1, "disposition": "RAW_UNSOLVED_FOR_RESERVATION",
                  "actual_status": None, "query_solved": False, "reserved": False,
                  "retry_count": 0, "input_content_source_ids": [], "input_semantic_parent_ids": [],
                  "ancestry": [], "training_ready": False}
        try:
            source = _Source(attempt, seed, cat)
            query = (_ledger_source if attempt.mechanism_index == 0 else
                     _transition_source if attempt.mechanism_index == 1 else _policy_source)(source)
            world = schema.validate_world(source.finish(query), catalogue=cat)
            exclusion = fixture_exclusions.check_world(world, ancestry=[])
            ledger["fixture_exclusion"] = exclusion
            if (exclusion.get("decision") != "NO_KNOWN_MATCH" or exclusion.get("must_reject") is not False
                    or exclusion.get("inventory_review") != "REVIEWED_EXTERNAL_PIN"):
                raise BirthGateError("fixture firewall rejected/deferred original attempt: " + exclusion.get("reason", "unknown"))
            digest = schema.digest(world)
            metadata = {"schema_version": 1, "catalogue_id": gates["catalogue_id"],
                        "catalogue_sha256": gates["catalogue_sha256"], "design_sha256": gates["design_sha256"],
                        "contract_sha256": gates["contract_sha256"], "allocated_cell": attempt.cell_metadata(),
                        "content_family_id": attempt.content_family_id, "source_domain_id": attempt.source_domain_id,
                        "producer_family_id": attempt.producer_family_id, "producer_version_sha256": gates["constructor_sha256"],
                        "ancestry": []}
            family = {"content_family_id": attempt.content_family_id, "family_kind": "SOURCE_CONTENT_FAMILY",
                      "assigned_role": attempt.role, "source_domain_id": attempt.source_domain_id,
                      "producer_family_id": attempt.producer_family_id, "origin": "ORIGINAL_AUTHOR_CREATED",
                      "setting": "COMMUNITY_WORKSHOP", "owner": "Rineya original controlled dossier author",
                      "revision": "BirthOne", "origin_recipe_sha256": digest,
                      "rights_declaration": "CC0-1.0 proposed authored facts/text", "imported_text": False,
                      "benchmark_rows": False, "existing_train_rows": False, "private_eval_derivation": False,
                      "document_inventory": sorted({d["purpose"] for d in world["documents"]}),
                      "content_shared_across_roles": False, "historical_semantic_overlap": "UNRESOLVED",
                      "rights_receipt_sha256": None}
            lineage = {"author_parent_id": attempt.parent_id,
                       "allocated_cell": [attempt.mechanism_index, attempt.structure_index, attempt.level, attempt.case_index],
                       "fresh_origin_recipe_sha256": digest, "input_content_source_ids": [], "input_semantic_parent_ids": [],
                       "producer_sha256": gates["constructor_sha256"], "complete_unsolved_specification_sha256": digest,
                       "alpha_semantic_sha256": exclusion["alpha_semantic_sha256"], "declared_fresh": True,
                       "detected_ancestry_findings": [], "canonical_lineage_group_id": attempt.content_family_id,
                       "method_and_limitations": "Separate original factual-recipe stream per slot; no parent inputs. " + " ".join(LIMITATIONS)}
            # Only this exact pair belongs in the registry's complete specification.
            worlds.append({"world": world, "source_cell_metadata": metadata,
                           "source_family": family, "construction_lineage": lineage})
            ledger.update(unsolved_recipe_sha256=digest, source_family_sha256=schema.digest(family),
                          registry_full_specification_sha256=schema.digest({"world": world, "source_cell_metadata": metadata}))
        except (BirthGateError, schema.SchemaError, KeyError, TypeError, ValueError) as exc:
            ledger.update(disposition="QUARANTINED_BIRTH_ATTEMPT", failure_type=type(exc).__name__, reason=str(exc))
        attempts.append(ledger)
    return {"worlds": worlds, "attempts": attempts, "birth_gate_receipt": gates,
            "allocated_attempt_slots": 54, "registered_specifications": 0,
            "query_solved": False, "training_ready": False, "limitations": list(LIMITATIONS)}
