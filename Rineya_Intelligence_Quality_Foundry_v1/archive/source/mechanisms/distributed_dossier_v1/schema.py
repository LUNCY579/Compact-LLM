"""Closed-record validation only: no query execution or compatible-world solving.

Repeated source assertions remain evidence. Primitive/reference failures raise
SchemaError; typed contradictory evidence is deliberately left to the solver.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import re

CATALOGUE_PATH = Path(__file__).with_name("catalogue.json")
ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT = ROOT / "cycles/002-source-registry/reviewed_blueprint_v1"
ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
REGISTRY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[A-Za-z0-9 -]{1,40}\Z")


class SchemaError(ValueError):
    """Invalid bytes, closed fields, primitive types, or typed references."""


def canonical_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SchemaError("value is not supported finite JSON/UTF-8") from exc


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fail(path, detail):
    raise SchemaError(f"{path}: {detail}")


def _closed(obj, fields, path):
    if type(obj) is not dict or set(obj) != set(fields):
        _fail(path, "requires exactly fields " + ", ".join(fields))


def _str(value, path, pattern=None):
    if type(value) is not str or (pattern is not None and not pattern.fullmatch(value)):
        _fail(path, "invalid string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(path, "string cannot be represented as valid UTF-8")
    return value


def _integer(value, low, high, path):
    if type(value) is not int or not low <= value <= high:
        _fail(path, f"requires integer {low}..{high}, without type coercion")


def _enum(value, choices, path):
    if type(value) is not str or value not in choices:
        _fail(path, "unsupported code")


def _list(value, low, high, path, unique=False, ordered=True):
    if type(value) is not list or not low <= len(value) <= high:
        _fail(path, f"requires array size {low}..{high}")
    keys = [canonical_bytes(v) for v in value]
    if unique and len(keys) != len(set(keys)):
        _fail(path, "duplicate array member")
    if not ordered:
        try:
            ordered_value = sorted(value)
        except TypeError:
            _fail(path, "set elements require one supported scalar type")
        if value != ordered_value:
            _fail(path, "set must be canonically sorted")


def loads_json(raw):
    """Decode JSON without losing duplicate object keys or nonfinite numbers."""
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                _fail("JSON", f"duplicate key {key}")
            obj[key] = value
        return obj
    def finite_float(token):
        number = float(token)
        if not math.isfinite(number): _fail("JSON", "nonfinite numeric exponent")
        return number
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_float=finite_float,
                          parse_constant=lambda x: _fail("JSON", f"invalid number {x}"))
    except SchemaError:
        raise
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SchemaError("malformed JSON") from exc


def load_catalogue(path=None):
    return loads_json(Path(path or CATALOGUE_PATH).read_bytes())


def validate_title(title, catalogue=None):
    cat = catalogue or load_catalogue()
    codec = cat["title_codec"]
    _str(title, "document.title")
    if not codec["min_length"] <= len(title) <= codec["max_length"] or not title[0].isalpha() or not title[-1].isalpha() or any(not (ch.isalpha() or ch in " -") for ch in title):
        _fail("document.title", "requires ordinary alphabetic words/spaces/hyphens")
    words = set(re.split(r"[ -]+", title.casefold()))
    if words & set(codec["reserved_casefold_words"]):
        _fail("document.title", "reserved instruction/answer/role word")
    return title


def loads_world(raw, *, public=False):
    return validate_world(loads_json(raw), public=public)


class _Validator:
    def __init__(self, world, catalogue):
        self.world = world
        self.cat = catalogue
        self.entities = {}
        self.variables = {}
        self.used_variables = set()

    def entity(self, value, kinds, path):
        _str(value, path, ID)
        if value not in self.entities or (kinds and self.entities[value]["kind"] not in kinds):
            _fail(path, "unknown entity or wrong entity kind")

    def scalar(self, value, typ, path, kinds=()):
        if typ == "EntityId":
            self.entity(value, kinds, path)
        elif typ == "Register":
            _enum(value, ["m", "x", "y"], path)
        else:
            low, high = {"Bit": (0, 1), "Quantity": (0, 12), "SmallCount": (0, 8),
                         "Day": (0, 31), "Tick": (0, 31), "Slot": (0, 7)}[typ]
            _integer(value, low, high, path)

    def value(self, value, typ, path, kinds=()):
        if type(value) is dict:
            _closed(value, ["var"], path)
            _str(value["var"], path + ".var", ID)
            variable = self.variables.get(value["var"])
            if variable is None or variable["scalar_type"] != typ:
                _fail(path, "unknown variable or incompatible variable domain")
            self.used_variables.add(value["var"])
            for item in variable["choices"]:
                self.scalar(item, typ, path, kinds)
        else:
            self.scalar(value, typ, path, kinds)

    def ref(self, value, kinds, path, variable=False):
        (self.value if variable else self.scalar)(value, "EntityId", path, kinds)

    def state(self, state, path):
        _closed(state, ["m", "x", "y"], path)
        for reg in state:
            self.value(state[reg], "Bit", f"{path}.{reg}")

    def arguments(self, predicate, args, path):
        definitions = self.cat["predicates"][predicate]["arguments"]
        _list(args, len(definitions), len(definitions), path)
        for n, (value, definition) in enumerate(zip(args, definitions)):
            if definition.startswith("slot "):
                # Predicate slots and entity arguments have no Value_ prefix in the contract.
                self.scalar(value, "Slot", f"{path}[{n}]")
            else:
                self.ref(value, [definition.split()[0]], f"{path}[{n}]")

    def record(self, record):
        path = "records." + str(record.get("record_id", "?"))
        kind = record.get("kind")
        _enum(kind, self.cat["productions"], path + ".kind")
        fields = self.cat["productions"][kind]["fields"]
        _closed(record, ["record_id", "document_id", "kind", *fields], path)
        for key in ("record_id", "document_id"):
            _str(record[key], path + "." + key, ID)
        v = lambda key, typ: self.value(record[key], typ, path + "." + key)
        r = lambda key, kinds, variable=False: self.ref(record[key], kinds, path + "." + key, variable)
        if kind == "alias":
            _enum(record["entity_kind"], ["commitment", "sponsor", "signer", "group"], path)
            for key in ("left", "right"):
                r(key, [record["entity_kind"]], True)
        elif kind in ("membership", "delegation"):
            if kind == "membership":
                r("child", ["sponsor", "group"], True); r("parent", ["group"], True)
            else:
                r("signer", ["signer"], True); r("owner", ["sponsor"], True)
            v("from_day", "Day"); v("through_day", "Day")
            # Reversed resolved intervals invalidate compatible worlds, not the primitive schema.
        elif kind.startswith("ledger_"):
            _str(record["event_id"], path + ".event_id", ID)
            v("day", "Day"); r("commitment", ["commitment"]); r("signer", ["signer"], True)
            if "predecessor" in record:
                _str(record["predecessor"], path + ".predecessor", ID)
            if "owner" in record:
                r("owner", ["sponsor"], True)
            if "quantity" in record:
                v("quantity", "Quantity")
            if "children" in record:
                _list(record["children"], 2, 2, path + ".children")
                for n, child in enumerate(record["children"]):
                    cp = f"{path}.children[{n}]"
                    _closed(child, ["commitment", "owner", "quantity"], cp)
                    self.ref(child["commitment"], ["commitment"], cp + ".commitment")
                    self.ref(child["owner"], ["sponsor"], cp + ".owner", True)
                    self.value(child["quantity"], "Quantity", cp + ".quantity")
        elif kind == "operator_menu":
            r("operator", ["operator"])
            _list(record["candidates"], 3, 4, path + ".candidates")
            ids, tables = [], []
            for candidate in record["candidates"]:
                _closed(candidate, ["program_id", "guard", "action"], path)
                _str(candidate["program_id"], path + ".program_id", ID)
                _enum(candidate["guard"], self.cat["guard_realizations"], path)
                _enum(candidate["action"], self.cat["action_realizations"], path)
                ids.append(candidate["program_id"])
                tables.append(tuple(_program(candidate, state) for state in itertools.product((0, 1), repeat=3)))
            if len(set(ids)) != len(ids) or len(set(tables)) != len(tables):
                _fail(path, "candidate IDs and complete truth tables must be distinct")
        elif kind == "bench_example":
            r("operator", ["operator"]); self.state(record["input"], path + ".input"); self.state(record["output"], path + ".output")
        elif kind == "station_initial":
            r("station", ["station"]); self.state(record["state"], path + ".state")
        elif kind.startswith("trace_"):
            v("tick", "Tick")
            for key in ("station", "source", "destination"):
                if key in record:
                    r(key, ["station"])
            if kind == "trace_apply":
                r("operator", ["operator"], True)
            elif kind == "trace_splice":
                v("source_register", "Register"); v("destination_register", "Register")
            else:
                key = "destinations" if kind == "trace_fork" else "sources"
                _list(record[key], 2, 2, path + "." + key, unique=True, ordered=kind == "trace_fork")
                for item in record[key]:
                    self.ref(item, ["station"], path + "." + key)
                if kind == "trace_join" and record["operation"] != "xor":
                    _fail(path, "join operation must be xor")
        elif kind in ("fact_bool", "fact_count"):
            predicate = record["predicate"]
            permitted = [k for k, p in self.cat["predicates"].items() if p["result"] == ("Bit" if kind == "fact_bool" else "SmallCount")]
            _enum(predicate, permitted, path)
            self.arguments(predicate, record["arguments"], path + ".arguments")
            v("value", self.cat["predicates"][predicate]["result"])
        elif kind == "session_plan":
            r("plan", ["plan"]); _list(record["legs"], 1, 2, path + ".legs")
            leg_ids = []
            for n, leg in enumerate(record["legs"]):
                lp = f"{path}.legs[{n}]"
                _closed(leg, ["leg_id", "slot", "room", "kit", "leader", "people", "group", "kit_demand"], lp)
                _str(leg["leg_id"], lp + ".leg_id", ID); leg_ids.append(leg["leg_id"])
                self.scalar(leg["slot"], "Slot", lp + ".slot")
                for key, kind2 in (("room", "room"), ("kit", "kit"), ("leader", "person"), ("group", "group")):
                    self.ref(leg[key], [kind2], lp + "." + key)
                _list(leg["people"], 1, 4, lp + ".people", unique=True, ordered=False)
                for person in leg["people"]:
                    self.ref(person, ["person"], lp + ".people")
                _integer(leg["kit_demand"], 1, 4, lp + ".kit_demand")
            if len(set(leg_ids)) != len(leg_ids):
                _fail(path, "duplicate leg ID within a plan definition")
        elif kind == "bundle":
            r("bundle", ["bundle"]); _list(record["plans"], 2, 2, path + ".plans", unique=True, ordered=False)
            for plan in record["plans"]:
                self.ref(plan, ["plan"], path + ".plans")
        elif kind == "base_policy":
            _str(record["policy_id"], path + ".policy_id", ID)
            scopes = self.cat["compatibility"]["policy_scope_predicates"]
            _enum(record["scope"], scopes, path + ".scope")
            predicate = record["predicate"]
            if type(predicate) is not dict:
                _fail(path, "policy predicate must be a closed tagged object")
            _enum(predicate.get("code"), scopes[record["scope"]], path + ".predicate.code")
            keys = ["code", "credential"] if predicate["code"] == "leader_credential" else ["code"]
            _closed(predicate, keys, path + ".predicate")
            if "credential" in predicate:
                self.ref(predicate["credential"], ["credential"], path + ".predicate.credential")
        elif kind == "exception":
            for key in ("exception_id", "base_policy_id"):
                _str(record[key], path + "." + key, ID)
            selector = record["selector"]
            if type(selector) is not dict:
                _fail(path, "selector must be tagged closed object")
            kind2 = selector.get("kind")
            mapping = {"plan": ["plan"], "leg": ["plan", "leg_id"], "group": ["group"], "bundle": ["bundle"]}
            _enum(kind2, mapping, path + ".selector.kind")
            _closed(selector, ["kind", *mapping[kind2]], path + ".selector")
            for key in mapping[kind2]:
                if key == "leg_id":
                    _str(selector[key], path + ".selector.leg_id", ID)
                else:
                    self.ref(selector[key], [key], path + ".selector." + key)
            condition = record["condition"]
            _closed(condition, ["predicate", "arguments", "comparison", "value"], path + ".condition")
            _enum(condition["predicate"], self.cat["predicates"], path)
            self.arguments(condition["predicate"], condition["arguments"], path + ".condition.arguments")
            typ = self.cat["predicates"][condition["predicate"]]["result"]
            _enum(condition["comparison"], ["eq"] if typ == "Bit" else ["eq", "ge"], path)
            self.scalar(condition["value"], typ, path + ".condition.value")
        elif kind == "incidental":
            r("subject", [])
            _enum(record["attribute"], self.cat["incidental_vocabulary"], path)
            _enum(record["value"], self.cat["incidental_vocabulary"][record["attribute"]], path)


def _program(program, state):
    """Source-language truth table arithmetic, never a test-trace/query solver."""
    regs = dict(zip("mxy", state))
    guard, action = program["guard"], program["action"]
    if guard != "always" and regs[guard[0]] != int(guard[1]):
        return state
    result = dict(regs)
    opcode, operands = action.split("_")
    if opcode == "flip":
        result[operands] = 1 - regs[operands]
    elif opcode == "swap":
        result[operands[0]], result[operands[1]] = regs[operands[1]], regs[operands[0]]
    else:
        result[operands[1]] = regs[operands[0]]
    return tuple(result[x] for x in "mxy")


def validate_world(world, *, public=False, catalogue=None):
    """Validate all 22 record kinds without inferring status or final answers."""
    cat = catalogue or load_catalogue()
    _closed(world, cat["world_fields" if public else "raw_world_fields"], "world")
    if world["world_schema"] != cat["world_schema"]:
        _fail("world_schema", "wrong version")
    _enum(world["mechanism"], cat["mechanisms"], "mechanism")
    _enum(world["format"], ["JSON", "LINE", "TOKEN"], "format")
    if not public:
        for key in ("parent_id", "reservation_id", "source_domain_id"):
            _str(world[key], key, ID)
        for key in ("content_family_id", "producer_family_id"):
            _str(world[key], key, REGISTRY_ID)
        _enum(world["structure_id"], cat["structures"], "structure_id")
        if cat["structures"][world["structure_id"]]["mechanism"] != world["mechanism"]:
            _fail("structure_id", "mechanism mismatch")
        _integer(world["level"], 1, 2, "level")
    validator = _Validator(world, cat)
    _list(world["entities"], 0, 2**31 - 1, "entities")
    labels = set()
    for entity in world["entities"]:
        _closed(entity, ["entity_id", "kind", "public_label"], "entity")
        _str(entity["entity_id"], "entity.entity_id", ID)
        _enum(entity["kind"], cat["entity_kinds"], "entity.kind")
        _str(entity["public_label"], "entity.public_label", LABEL)
        if entity["entity_id"] in validator.entities or entity["public_label"] in labels:
            _fail("entities", "duplicate entity ID or public label")
        validator.entities[entity["entity_id"]] = entity; labels.add(entity["public_label"])
    if world["mechanism"] == "commitment_ledger" and sum(e["kind"] == "commitment" for e in world["entities"]) > 12:
        _fail("entities", "more than 12 declared commitment entities")
    _list(world["variables"], 0, 2, "variables")
    for variable in world["variables"]:
        _closed(variable, ["variable_id", "scalar_type", "choices"], "variable")
        _str(variable["variable_id"], "variable.variable_id", ID)
        _enum(variable["scalar_type"], ["Bit", "Quantity", "SmallCount", "EntityId", "Day", "Tick", "Register"], "variable.scalar_type")
        if variable["variable_id"] in validator.variables or variable["variable_id"] in validator.entities or variable["variable_id"] in labels:
            _fail("variable", "duplicate/colliding variable ID")
        _list(variable["choices"], 2, 3, "variable.choices", unique=True, ordered=False)
        for choice in variable["choices"]:
            validator.scalar(choice, variable["scalar_type"], "variable.choices")
        validator.variables[variable["variable_id"]] = variable
    _list(world["documents"], 3, 6, "documents")
    docs, assigned = {}, []
    purposes = cat["document_purposes"]
    for doc in world["documents"]:
        _closed(doc, ["document_id", "purpose", "title", "dated_day", "record_ids", "style"], "document")
        _str(doc["document_id"], "document.document_id", ID)
        validate_title(doc["title"], cat)
        _enum(doc["purpose"], purposes, "document.purpose")
        _integer(doc["dated_day"], 0, 31, "document.dated_day")
        _enum(doc["style"], ["controlled_prose", "small_table"], "document.style")
        _list(doc["record_ids"], 0, 40, "document.record_ids", unique=True)
        for rid in doc["record_ids"]:
            _str(rid, "document.record_ids", ID)
        if doc["document_id"] in docs:
            _fail("documents", "duplicate document ID")
        docs[doc["document_id"]] = doc; assigned.extend(doc["record_ids"])
    _list(world["records"], 0, 40, "records")
    records = {}
    permitted = cat["record_mechanisms"][world["mechanism"]]
    for record in world["records"]:
        if type(record) is not dict:
            _fail("records", "record must be object")
        validator.record(record)
        if record["kind"] not in permitted:
            _fail("records", "record kind incompatible with mechanism")
        if record["record_id"] in records:
            _fail("records", "duplicate record ID")
        if record["document_id"] not in docs or record["record_id"] not in docs[record["document_id"]]["record_ids"]:
            _fail("record.document_id", "dangling or mismatched source document")
        records[record["record_id"]] = record
    if len(assigned) != len(set(assigned)) or set(assigned) != set(records):
        _fail("documents", "source assignment must cover each record exactly once")
    framework = world["framework"]
    _closed(framework, ["semantic_version", "rule_clauses", "closed_candidate_list", "unknown_defaults", "source_assertions_are_conjunctive"], "framework")
    if framework["semantic_version"] != cat["contract_id"] or framework["closed_candidate_list"] is not True or framework["unknown_defaults"] != "NONE" or framework["source_assertions_are_conjunctive"] is not True:
        _fail("framework", "fixed framework field mismatch")
    _list(framework["rule_clauses"], 1, len(cat["semantic_rules"]), "framework.rule_clauses")
    ids, codes = [], []
    for clause in framework["rule_clauses"]:
        _closed(clause, ["rule_id", "code", "public_variant"], "framework.clause")
        _str(clause["rule_id"], "framework.rule_id", ID)
        _enum(clause["code"], cat["framework_variants"], "framework.code")
        _integer(clause["public_variant"], 0, len(cat["framework_variants"][clause["code"]]) - 1, "framework.public_variant")
        ids.append(clause["rule_id"]); codes.append(clause["code"])
    if len(set(ids)) != len(ids) or len(set(codes)) != len(codes):
        _fail("framework", "duplicate rule declaration")
    if not set(cat["framework_required"][world["mechanism"]]) <= set(codes):
        _fail("framework", "missing operative semantic rule")
    query = world["query"]
    mechanism = world["mechanism"]
    if mechanism == "commitment_ledger":
        _closed(query, ["mechanism", "cutoff", "group_label"], "query")
        validator.scalar(query["cutoff"], "Day", "query.cutoff")
        validator.ref(query["group_label"], ["group"], "query.group_label", True)
    elif mechanism == "guarded_transition_learning":
        _closed(query, ["mechanism", "final_stations", "combine", "encoding"], "query")
        _enum(query["combine"], ["single", "xor"], "query.combine")
        n = 1 if query["combine"] == "single" else 2
        _list(query["final_stations"], n, n, "query.final_stations", unique=True)
        for item in query["final_stations"]:
            validator.ref(item, ["station"], "query.final_stations")
        if query["encoding"] != "4m+2x+y":
            _fail("query.encoding", "unsupported encoding")
    else:
        _closed(query, ["mechanism", "candidate_ids", "candidate_kind", "operation"], "query")
        _enum(query["candidate_kind"], ["plan", "bundle"], "query.candidate_kind")
        _list(query["candidate_ids"], 3, 12, "query.candidate_ids", unique=True, ordered=False)
        for item in query["candidate_ids"]:
            validator.ref(item, [query["candidate_kind"]], "query.candidate_ids")
        if query["operation"] != "count_feasible":
            _fail("query.operation", "unsupported operation")
    if query["mechanism"] != mechanism:
        _fail("query.mechanism", "world mismatch")
    # Reference checks preserve repeated source definitions for compatible-world semantics.
    events = {r["event_id"] for r in records.values() if r["kind"].startswith("ledger_")}
    menus = {r["operator"] for r in records.values() if r["kind"] == "operator_menu"}
    if world["mechanism"] == "guarded_transition_learning":
        if public and len(menus) not in (2, 3):
            _fail("operator_menu", "requires two or three learned operators")
        if not public and len(menus) != (2 if world["level"] == 1 else 3):
            _fail("operator_menu", "operator count differs from declared level")
        if sum(r["kind"].startswith("trace_") for r in records.values()) > 10:
            _fail("records", "more than ten test trace events")
    plans = {}
    bundles = {}
    policies = {}
    fact_keys = {(r["predicate"], tuple(r["arguments"])) for r in records.values() if r["kind"] in ("fact_bool", "fact_count")}
    for record in records.values():
        if record["kind"] == "session_plan": plans.setdefault(record["plan"], []).append(record)
        if record["kind"] == "bundle": bundles.setdefault(record["bundle"], []).append(record)
        if record["kind"] == "base_policy": policies.setdefault(record["policy_id"], []).append(record)
    for record in records.values():
        kind = record["kind"]
        if "predecessor" in record and record["predecessor"] not in events:
            _fail(record["record_id"], "dangling predecessor event")
        if kind in ("bench_example", "trace_apply"):
            operator = record["operator"]
            choices = validator.variables[operator["var"]]["choices"] if type(operator) is dict else [operator]
            if any(item not in menus for item in choices): _fail(record["record_id"], "operator has no menu")
        if kind == "bundle" and any(item not in plans for item in record["plans"]):
            _fail(record["record_id"], "bundle member lacks plan definition")
        if kind == "exception":
            condition = record["condition"]
            if (condition["predicate"], tuple(condition["arguments"])) not in fact_keys:
                _fail(record["record_id"], "exception condition source key is unasserted")
            if record["base_policy_id"] not in policies: _fail(record["record_id"], "dangling policy reference")
            sel = record["selector"]
            if not any(sel["kind"] in cat["compatibility"]["exception_scope_selector_pairs"][policy["scope"]] for policy in policies[record["base_policy_id"]]):
                _fail(record["record_id"], "selector incompatible with every named policy assertion")
            if "plan" in sel and sel["plan"] not in plans: _fail(record["record_id"], "selector plan undefined")
            if "bundle" in sel and sel["bundle"] not in bundles: _fail(record["record_id"], "selector bundle undefined")
            if sel["kind"] == "leg" and not any(sel["leg_id"] in [x["leg_id"] for x in p["legs"]] for p in plans[sel["plan"]]):
                _fail(record["record_id"], "selector leg not in named plan")
    if mechanism == "policy_feasibility":
        definitions = plans if query["candidate_kind"] == "plan" else bundles
        if any(item not in definitions for item in query["candidate_ids"]): _fail("query", "candidate undefined")
        affected = set(query["candidate_ids"]) if query["candidate_kind"] == "plan" else {p for b in query["candidate_ids"] for definition in bundles[b] for p in definition["plans"]}
        if any(all(p["scope"] == "each_bundle" for p in values) for values in policies.values()) and query["candidate_kind"] != "bundle":
            _fail("policies", "bundle policy requires bundle query")
        if any(all(p["scope"] == "cross_leg" for p in values) for values in policies.values()):
            if any(not any(len(p["legs"]) == 2 for p in plans[key]) for key in affected):
                _fail("policies", "cross-leg policy requires two ordered candidate legs")
    return world


def public_projection(world):
    """Copy only the complete public fields; retains all incidental assertions."""
    import copy
    return copy.deepcopy({key: world[key] for key in load_catalogue()["world_fields"]})


def validate_reservation(reservation, world):
    contract = loads_json((BLUEPRINT / "CONTRACT.json").read_bytes())
    fields = contract["types"]["Reservation"]["fields"]
    _closed(reservation, fields, "reservation")
    for key, descriptor in fields.items():
        value = reservation[key]
        if descriptor == "Hash": _str(value, key, HASH)
        elif descriptor == "Hash|null":
            if value is not None: _str(value, key, HASH)
        elif descriptor == "AuthorId": _str(value, key, ID)
        elif descriptor == "RegistryId": _str(value, key, REGISTRY_ID)
        elif descriptor in ("SemanticParentId", "StructuralClusterId"):
            prefix = "sp-" if descriptor == "SemanticParentId" else "sc-"
            _str(value, key, re.compile(prefix + r"[0-9a-f]{64}\Z"))
    for key in ("parent_id", "reservation_id", "mechanism", "structure_id", "level", "format", "content_family_id", "source_domain_id", "producer_family_id"):
        if reservation[key] != world[key]: _fail(key, "reservation/world mismatch")
    if reservation["namespace"] != contract["parent_namespace"] or reservation["reserved_before_render_solve"] is not True:
        _fail("reservation", "wrong namespace or reservation timing")
    _integer(reservation["structure_index"], 0, 2, "structure_index")
    _integer(reservation["level"], 1, 2, "level")
    _integer(reservation["case_index"], 0, 2, "case_index")
    _integer(reservation["reservation_sequence"], 0, 2**63 - 1, "reservation_sequence")
    structure_index = int(world["structure_id"][1])
    if reservation["structure_index"] != structure_index: _fail("structure_index", "wrong cell")
    role = "EXPOSED_DEV" if structure_index == 2 else "PROSPECTIVE_TRAIN"
    if reservation["role"] != role: _fail("role", "wrong allocated role")
    mi = contract["allocation"]["mechanisms"].index(world["mechanism"])
    fi = (mi + structure_index + (structure_index + 2) * (world["level"] - 1) + reservation["case_index"]) % 3
    if reservation["format"] != ["JSON", "LINE", "TOKEN"][fi]: _fail("format", "wrong allocated format")
    if reservation["unsolved_recipe_sha256"] != digest(world): _fail("unsolved_recipe_sha256", "world binding mismatch")
    _list(reservation["ancestry"], 0, 1000, "ancestry")
    for edge in reservation["ancestry"]:
        _closed(edge, ["ancestor_identity_type", "ancestor_id", "relation", "ancestor_sha256"], "ancestry.edge")
        _enum(edge["ancestor_identity_type"], ["author_parent", "registry_parent", "registry_source", "author_row"], "ancestor_identity_type")
        _enum(edge["relation"], ["source", "view", "repair", "counterfactual", "ablation", "relabel", "reorder"], "relation")
        pattern = re.compile(r"sp-[0-9a-f]{64}\Z") if edge["ancestor_identity_type"] == "registry_parent" else REGISTRY_ID if edge["ancestor_identity_type"] == "registry_source" else ID
        _str(edge["ancestor_id"], "ancestor_id", pattern); _str(edge["ancestor_sha256"], "ancestor_sha256", HASH)
    return reservation
