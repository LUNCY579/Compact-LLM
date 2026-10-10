"""Independent, total public-text parser for the distributed-dossier catalogue.

Only public message bytes and the data catalogue are inputs.  This module never
imports the author schema, renderer, generator, or either solving implementation.
Syntax/schema errors raise PublicParseError; source contradictions remain facts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path


class PublicParseError(ValueError):
    def __init__(self, message, *, code="PARSE_ERROR", message_index=None, utf8_start=None):
        self.code = code
        self.message_index = message_index
        self.utf8_start = utf8_start
        super().__init__(f"{code}: {message}")


_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z", re.ASCII)
_LABEL = re.compile(r"[A-Za-z0-9 -]{1,40}\Z", re.ASCII)
_KINDS = {"commitment", "sponsor", "signer", "group", "station", "operator", "person", "room", "kit", "credential", "plan", "bundle"}
_MECHANISMS = {"commitment_ledger", "guarded_transition_learning", "policy_feasibility"}
_PURPOSES = {"delegation_circular", "booking_minutes", "delivery_slips", "alias_crosswalk", "calibration_notes", "maintenance_log", "room_roster", "policy_amendments", "session_proposals"}
_RULES = {
    "commitment_ledger": {"FINITE_WORLDS", "STATUS", "ENTITY_IDENTITY", "A_MEMBERSHIP", "A_AUTHORITY", "A_TIME", "A_SINGLE_SUCCESSOR", "A_TRANSITIONS", "A_RESULT"},
    "guarded_transition_learning": {"FINITE_WORLDS", "STATUS", "ENTITY_IDENTITY", "B_PROGRAM", "B_INFER", "B_TRACE", "B_XOR", "B_RESULT"},
    "policy_feasibility": {"FINITE_WORLDS", "STATUS", "ENTITY_IDENTITY", "C_FACTS", "C_LOCAL", "C_CROSS_LEG", "C_BUNDLE", "C_EXEMPTION", "C_RESULT"},
}
_FIELDS = {
    "alias": {"entity_kind", "left", "right"},
    "membership": {"child", "parent", "from_day", "through_day"},
    "delegation": {"signer", "owner", "from_day", "through_day"},
    "ledger_start": {"event_id", "day", "commitment", "owner", "quantity", "signer"},
    "ledger_amend": {"event_id", "day", "commitment", "predecessor", "quantity", "signer"},
    "ledger_transfer": {"event_id", "day", "commitment", "predecessor", "owner", "signer"},
    "ledger_cancel": {"event_id", "day", "commitment", "predecessor", "signer"},
    "ledger_split": {"event_id", "day", "commitment", "predecessor", "children", "signer"},
    "operator_menu": {"operator", "candidates"},
    "bench_example": {"operator", "input", "output"},
    "station_initial": {"station", "state"},
    "trace_apply": {"tick", "station", "operator"},
    "trace_splice": {"tick", "source", "source_register", "destination", "destination_register"},
    "trace_fork": {"tick", "source", "destinations"},
    "trace_join": {"tick", "sources", "destination", "operation"},
    "fact_bool": {"predicate", "arguments", "value"},
    "fact_count": {"predicate", "arguments", "value"},
    "session_plan": {"plan", "legs"},
    "bundle": {"bundle", "plans"},
    "base_policy": {"policy_id", "scope", "predicate"},
    "exception": {"exception_id", "base_policy_id", "selector", "condition"},
    "incidental": {"subject", "attribute", "value"},
}
_PREDICATES = {
    "room_available": ("room", "Slot"), "kit_available": ("kit", "Slot"),
    "credential": ("person", "credential"), "member": ("person", "group"),
    "capacity": ("room",), "kit_units": ("kit", "Slot"),
}
_SCOPES = {
    "each_leg": {"capacity_ok", "room_available", "kit_available", "kit_units_ok", "leader_credential", "all_members"},
    "cross_leg": {"slots_increase", "rooms_differ", "groups_equal"},
    "each_bundle": {"shared_kit_quota"},
}
_SELECTORS = {"each_leg": {"plan", "leg", "group"}, "cross_leg": {"plan"}, "each_bundle": {"bundle"}}
_GUARDS = {"always", "m0", "m1", "x0", "x1", "y0", "y1"}
_ACTIONS = {"flip_m", "flip_x", "flip_y", "swap_mx", "swap_xy", "copy_mx", "copy_xm", "copy_xy", "copy_yx"}
_SCALAR_BOUNDS = {"Bit": (0, 1), "Quantity": (0, 12), "SmallCount": (0, 8), "Day": (0, 31), "Tick": (0, 31), "Slot": (0, 7), "Demand": (1, 4)}
_ENCODED = re.compile(r"(?<![A-Za-z])(?:answer|gold|target|winning|winner|necessary|distractor|authoritative|active_scope|determinate|underdetermined|inconsistent|train|dev)(?=$|[^A-Za-z])", re.I)
_TITLE_RESERVED = {"answer", "target", "status", "unknown", "conflict", "determinate", "underdetermined", "inconsistent", "ignore", "output", "print", "assistant", "system", "prompt", "user", "tool", "train", "dev", "test", "check", "final", "case", "level", "winning", "authoritative", "distractor", "oracle"}


def _valid_title(title):
    return (type(title) is str and 4 <= len(title) <= 80 and title[0].isalpha() and title[-1].isalpha() and all(c.isalpha() or c in " -" for c in title) and not (_TITLE_RESERVED & set(re.split(r"[ -]+", title.casefold()))))


def _schema(condition, message):
    if not condition:
        raise PublicParseError(message, code="SCHEMA_ERROR")


def _keys(value, keys, where):
    _schema(type(value) is dict and set(value) == set(keys), f"closed fields at {where}")


def _id(value, where):
    _schema(type(value) is str and _ID.fullmatch(value) is not None, f"invalid ID at {where}")
    _schema(not _ENCODED.search(value), f"encoded or reserved identity at {where}")


def _array(value, lo, hi, where, *, unique=False, canonical=False):
    _schema(type(value) is list and lo <= len(value) <= hi, f"invalid array size at {where}")
    if unique:
        _schema(len({_canonical(x) for x in value}) == len(value), f"duplicate set member at {where}")
    if canonical:
        _schema(all(type(x) in {str, int} for x in value) and len({type(x) for x in value}) <= 1 and value == sorted(value), f"noncanonical typed set order at {where}")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _program_table(guard, action):
    """Independent eight-state evaluator used only to type-check menu distinctness."""
    result = []
    for state in itertools.product((0, 1), repeat=3):
        bits = dict(zip("mxy", state))
        out = bits.copy()
        if guard == "always" or bits[guard[0]] == int(guard[1]):
            verb, regs = action.split("_")
            if verb == "flip":
                out[regs] = 1 - bits[regs]
            elif verb == "swap":
                out[regs[0]], out[regs[1]] = bits[regs[1]], bits[regs[0]]
            else:
                out[regs[1]] = bits[regs[0]]
        result.append(tuple(out[r] for r in "mxy"))
    return tuple(result)


def _validate_world(world, catalogue):
    """Validate syntax and references without deciding compatible-world status."""
    _keys(world, {"world_schema", "mechanism", "entities", "variables", "documents", "records", "framework", "query", "format"}, "world")
    _schema(world["world_schema"] == "rineya.distributed_dossier_world.v1", "world version")
    mechanism = world["mechanism"]
    _schema(mechanism in _MECHANISMS and world["format"] in {"JSON", "LINE", "TOKEN"}, "task mechanism/format")
    entities = {}
    labels = set()
    _schema(type(world["entities"]) is list, "entities must be an array")
    for entity in world["entities"]:
        _keys(entity, {"entity_id", "kind", "public_label"}, "entity")
        eid, label = entity["entity_id"], entity["public_label"]
        _id(eid, "entity_id")
        _schema(entity["kind"] in _KINDS, "unknown entity kind")
        _schema(type(label) is str and _LABEL.fullmatch(label) and not _ENCODED.search(label), "invalid or encoded public label")
        _schema(eid not in entities and label not in labels, "duplicate entity identity/label")
        entities[eid] = entity["kind"]
        labels.add(label)
    variables = {}
    _array(world["variables"], 0, 2, "variables")
    for var in world["variables"]:
        _keys(var, {"variable_id", "scalar_type", "choices"}, "variable")
        vid = var["variable_id"]
        _id(vid, "variable_id")
        _schema(vid not in variables and vid not in entities and vid not in labels, "variable identity collision")
        _schema(var["scalar_type"] in {"Bit", "Quantity", "SmallCount", "EntityId", "Day", "Tick", "Register"}, "unknown finite scalar type")
        _array(var["choices"], 2, 3, "variable choices", unique=True, canonical=True)
        variables[vid] = var

    def scalar(value, typ, where, *, variable=False):
        if type(value) is dict:
            _schema(variable, f"variable forbidden at {where}")
            _keys(value, {"var"}, where)
            _schema(value["var"] in variables, f"undeclared variable at {where}")
            v = variables[value["var"]]
            _schema(v["scalar_type"] == typ, f"variable type confusion at {where}")
            for choice in v["choices"]:
                scalar(choice, typ, where)
        elif typ == "Register":
            _schema(type(value) is str and value in "mxy" and len(value) == 1, f"register at {where}")
        elif typ == "EntityId":
            _schema(type(value) is str and value in entities, f"entity at {where}")
        else:
            lo, hi = _SCALAR_BOUNDS[typ]
            _schema(type(value) is int and lo <= value <= hi, f"integer range/type at {where}")

    def ref(value, kinds, where, *, variable=False):
        kinds = {kinds} if type(kinds) is str else set(kinds)
        if type(value) is dict:
            scalar(value, "EntityId", where, variable=variable)
            choices = variables[value["var"]]["choices"]
        else:
            scalar(value, "EntityId", where)
            choices = [value]
        _schema(all(entities[x] in kinds for x in choices), f"wrong entity kind at {where}")

    for var in world["variables"]:
        for choice in var["choices"]:
            scalar(choice, var["scalar_type"], "finite choice")
    docs = {}
    _array(world["documents"], 3, 6, "documents")
    for doc in world["documents"]:
        _keys(doc, {"document_id", "purpose", "title", "dated_day", "record_ids", "style"}, "document")
        _id(doc["document_id"], "document_id")
        _schema(doc["document_id"] not in docs, "duplicate document")
        _schema(doc["purpose"] in _PURPOSES and doc["style"] in {"controlled_prose", "small_table"}, "document purpose/style")
        title = doc["title"]
        _schema(_valid_title(title), "document title codec or forbidden word")
        scalar(doc["dated_day"], "Day", "document date")
        _array(doc["record_ids"], 0, 40, "document record_ids", unique=True)
        docs[doc["document_id"]] = doc
    _array(world["records"], 0, 40, "records")
    records = {}
    events, policies, plan_records, menu_records, bundle_records = {}, {}, {}, {}, {}
    for record in world["records"]:
        _schema(type(record) is dict and record.get("kind") in _FIELDS, "unknown record kind")
        _keys(record, _FIELDS[record["kind"]] | {"kind", "record_id", "document_id"}, "record")
        _id(record["record_id"], "record_id")
        _schema(record["record_id"] not in records, "duplicate public record ID")
        _schema(record["document_id"] in docs, "unknown record document")
        records[record["record_id"]] = record
        if record["kind"].startswith("ledger_"):
            _id(record["event_id"], "event_id")
            events.setdefault(record["event_id"], []).append(record)
        if record["kind"] == "base_policy":
            policies.setdefault(record["policy_id"], []).append(record)
        if record["kind"] == "session_plan":
            plan_records.setdefault(record["plan"], []).append(record)
        if record["kind"] == "operator_menu":
            menu_records.setdefault(record["operator"], []).append(record)
        if record["kind"] == "bundle":
            bundle_records.setdefault(record["bundle"], []).append(record)
    owners = [rid for d in docs.values() for rid in d["record_ids"]]
    _schema(len(owners) == len(set(owners)) and set(owners) == set(records), "document record ownership/inventory")
    for did, doc in docs.items():
        _schema(all(records[rid]["document_id"] == did for rid in doc["record_ids"]), "wrong record ownership")

    def state(value, where):
        _keys(value, {"m", "x", "y"}, where)
        for reg, bit in value.items():
            scalar(bit, "Bit", f"{where}.{reg}", variable=True)

    def fact(value, where, *, condition=False):
        pred = value["predicate"]
        _schema(pred in _PREDICATES, f"unknown predicate at {where}")
        args = value["arguments"]
        _array(args, len(_PREDICATES[pred]), len(_PREDICATES[pred]), f"{where}.arguments")
        for i, (arg, typ) in enumerate(zip(args, _PREDICATES[pred])):
            if typ == "Slot":
                scalar(arg, typ, f"{where}.arguments[{i}]")
            else:
                ref(arg, typ, f"{where}.arguments[{i}]")
        typ = "SmallCount" if pred in {"capacity", "kit_units"} else "Bit"
        scalar(value["value"], typ, f"{where}.value", variable=not condition)
        if condition:
            _schema(value["comparison"] in ({"eq", "ge"} if typ == "SmallCount" else {"eq"}), "condition comparison/type mismatch")

    permitted = (
        {"alias", "membership", "delegation", "incidental"} | {x for x in _FIELDS if x.startswith("ledger_")} if mechanism == "commitment_ledger" else
        {"operator_menu", "bench_example", "station_initial", "trace_apply", "trace_splice", "trace_fork", "trace_join", "incidental"} if mechanism == "guarded_transition_learning" else
        {"fact_bool", "fact_count", "session_plan", "bundle", "base_policy", "exception", "incidental"}
    )
    for record in world["records"]:
        kind, rid = record["kind"], record["record_id"]
        _schema(kind in permitted, "record kind is incompatible with mechanism")
        if kind == "alias":
            _schema(record["entity_kind"] in {"commitment", "sponsor", "signer", "group"}, "alias kind")
            ref(record["left"], record["entity_kind"], rid + ".left", variable=True)
            ref(record["right"], record["entity_kind"], rid + ".right", variable=True)
        elif kind in {"membership", "delegation"}:
            if kind == "membership":
                ref(record["child"], {"sponsor", "group"}, rid + ".child", variable=True)
                ref(record["parent"], "group", rid + ".parent", variable=True)
            else:
                ref(record["signer"], "signer", rid + ".signer", variable=True)
                ref(record["owner"], "sponsor", rid + ".owner", variable=True)
            for f in ("from_day", "through_day"):
                scalar(record[f], "Day", rid + "." + f, variable=True)
        elif kind.startswith("ledger_"):
            scalar(record["day"], "Day", rid + ".day", variable=True)
            ref(record["commitment"], "commitment", rid + ".commitment")
            ref(record["signer"], "signer", rid + ".signer", variable=True)
            if "predecessor" in record:
                _id(record["predecessor"], "predecessor")
                _schema(record["predecessor"] in events, "dangling event predecessor")
            if "owner" in record:
                ref(record["owner"], "sponsor", rid + ".owner", variable=True)
            if "quantity" in record:
                scalar(record["quantity"], "Quantity", rid + ".quantity", variable=True)
            if "children" in record:
                _array(record["children"], 2, 2, rid + ".children")
                for child in record["children"]:
                    _keys(child, {"commitment", "owner", "quantity"}, "split child")
                    ref(child["commitment"], "commitment", "child.commitment")
                    ref(child["owner"], "sponsor", "child.owner", variable=True)
                    scalar(child["quantity"], "Quantity", "child.quantity", variable=True)
                # Alias-equivalent or reused children are legitimate source contradictions.
        elif kind == "operator_menu":
            ref(record["operator"], "operator", rid + ".operator")
            _array(record["candidates"], 3, 4, rid + ".candidates")
            pids, tables = set(), set()
            for p in record["candidates"]:
                _keys(p, {"program_id", "guard", "action"}, "guarded program")
                _id(p["program_id"], "program_id")
                _schema(p["program_id"] not in pids and p["guard"] in _GUARDS and p["action"] in _ACTIONS, "program identity/AST")
                pids.add(p["program_id"])
                tables.add(_program_table(p["guard"], p["action"]))
            _schema(len(tables) == len(pids), "menu candidates are not distinct functions")
        elif kind in {"bench_example", "station_initial"}:
            if kind == "bench_example":
                ref(record["operator"], "operator", rid + ".operator")
                _schema(record["operator"] in menu_records, "operator missing public menu")
                state(record["input"], rid + ".input")
                state(record["output"], rid + ".output")
            else:
                ref(record["station"], "station", rid + ".station")
                state(record["state"], rid + ".state")
        elif kind.startswith("trace_"):
            scalar(record["tick"], "Tick", rid + ".tick", variable=True)
            if kind == "trace_apply":
                ref(record["station"], "station", rid + ".station")
                ref(record["operator"], "operator", rid + ".operator", variable=True)
                ops = variables[record["operator"]["var"]]["choices"] if type(record["operator"]) is dict else [record["operator"]]
                _schema(all(op in menu_records for op in ops), "applied operator missing public menu")
            elif kind == "trace_splice":
                for f in ("source", "destination"):
                    ref(record[f], "station", rid + "." + f)
                    scalar(record[f + "_register"], "Register", rid + "." + f + "_register", variable=True)
            else:
                array_field = "destinations" if kind == "trace_fork" else "sources"
                _array(record[array_field], 2, 2, rid + "." + array_field, unique=True, canonical=kind == "trace_join")
                for s in record[array_field]:
                    ref(s, "station", rid + "." + array_field)
                ref(record["source" if kind == "trace_fork" else "destination"], "station", rid)
                if kind == "trace_join":
                    _schema(record["operation"] == "xor", "join AST operation")
        elif kind in {"fact_bool", "fact_count"}:
            _schema(record["predicate"] in ({"capacity", "kit_units"} if kind == "fact_count" else {"room_available", "kit_available", "credential", "member"}), "fact result type confusion")
            fact(record, rid)
        elif kind == "session_plan":
            ref(record["plan"], "plan", rid + ".plan")
            _array(record["legs"], 1, 2, rid + ".legs")
            leg_ids = set()
            for leg in record["legs"]:
                _keys(leg, {"leg_id", "slot", "room", "kit", "leader", "people", "group", "kit_demand"}, "leg")
                _id(leg["leg_id"], "leg_id")
                _schema(leg["leg_id"] not in leg_ids, "duplicate leg identity")
                leg_ids.add(leg["leg_id"])
                scalar(leg["slot"], "Slot", "leg.slot")
                scalar(leg["kit_demand"], "Demand", "leg.kit_demand")
                for f, typ in (("room", "room"), ("kit", "kit"), ("leader", "person"), ("group", "group")):
                    ref(leg[f], typ, "leg." + f)
                _array(leg["people"], 1, 4, "leg.people", unique=True, canonical=True)
                for person in leg["people"]:
                    ref(person, "person", "leg.people")
        elif kind == "bundle":
            ref(record["bundle"], "bundle", rid + ".bundle")
            _array(record["plans"], 2, 2, rid + ".plans", unique=True, canonical=True)
            for p in record["plans"]:
                ref(p, "plan", rid + ".plans")
                _schema(p in plan_records, "bundle member lacks plan definition")
        elif kind == "base_policy":
            _id(record["policy_id"], "policy_id")
            predicate = record["predicate"]
            _schema(type(predicate) is dict and predicate.get("code") in _SCOPES.get(record["scope"], set()), "policy scope/predicate incompatibility")
            _keys(predicate, {"code", "credential"} if predicate["code"] == "leader_credential" else {"code"}, "policy AST")
            if "credential" in predicate:
                ref(predicate["credential"], "credential", "policy credential")
        elif kind == "exception":
            _id(record["exception_id"], "exception_id")
            _id(record["base_policy_id"], "base_policy_id")
            _schema(record["base_policy_id"] in policies, "dangling policy exception")
            selector = record["selector"]
            _schema(type(selector) is dict and selector.get("kind") in {"plan", "leg", "group", "bundle"}, "selector AST")
            sel = selector["kind"]
            _keys(selector, {"kind", "plan", "leg_id"} if sel == "leg" else {"kind", sel}, "selector AST")
            _schema(any(sel in _SELECTORS.get(p["scope"], set()) for p in policies[record["base_policy_id"]]), "policy scope/selector incompatibility")
            ref(selector["plan" if sel == "leg" else sel], "plan" if sel == "leg" else sel, "selector reference")
            if sel in {"plan", "leg"}:
                _schema(selector["plan"] in plan_records, "selector missing plan")
            if sel == "bundle":
                _schema(selector["bundle"] in bundle_records, "selector missing bundle")
            if sel == "leg":
                _id(selector["leg_id"], "selector leg ID")
                _schema(any(leg["leg_id"] == selector["leg_id"] for p in plan_records[selector["plan"]] for leg in p["legs"]), "selector missing leg")
            _keys(record["condition"], {"predicate", "arguments", "comparison", "value"}, "condition AST")
            fact(record["condition"], "condition", condition=True)
        else:
            ref(record["subject"], _KINDS, rid + ".subject")
            vocab = catalogue["incidental_vocabulary"]
            _schema(record["attribute"] in vocab and record["value"] in vocab[record["attribute"]], "uncatalogued incidental vocabulary")

    if mechanism == "policy_feasibility":
        # Conditions always consult their named reported fact.  Policy premise
        # necessity depends on conjunction/exemption evaluation and belongs to
        # the independent solver, which must never supply missing defaults.
        fact_keys = {(r["predicate"], tuple(r["arguments"])) for r in world["records"] if r["kind"] in {"fact_bool", "fact_count"}}
        for record in world["records"]:
            if record["kind"] == "exception":
                condition = record["condition"]
                _schema((condition["predicate"], tuple(condition["arguments"])) in fact_keys, "exception condition lacks asserted fact key")

    framework = world["framework"]
    _keys(framework, {"semantic_version", "rule_clauses", "closed_candidate_list", "unknown_defaults", "source_assertions_are_conjunctive"}, "framework")
    _schema(framework["semantic_version"] == "RIQF-C002-DD-001" and framework["closed_candidate_list"] is True and framework["unknown_defaults"] == "NONE" and framework["source_assertions_are_conjunctive"] is True, "framework meanings")
    rules, codes = set(), set()
    for clause in framework["rule_clauses"]:
        _keys(clause, {"rule_id", "code", "public_variant"}, "framework clause")
        _id(clause["rule_id"], "rule_id")
        _schema(clause["rule_id"] not in rules and clause["code"] not in codes, "duplicate framework framing")
        rules.add(clause["rule_id"])
        codes.add(clause["code"])
        _schema(clause["code"] in _RULES[mechanism] and type(clause["public_variant"]) is int and 0 <= clause["public_variant"] < len(catalogue["framework_variants"][clause["code"]]), "framework variant/code")
    _schema(codes == _RULES[mechanism], "missing or extra semantic framework clause")
    query = world["query"]
    _schema(type(query) is dict and query.get("mechanism") == mechanism, "query mechanism")
    if mechanism == "commitment_ledger":
        _keys(query, {"mechanism", "cutoff", "group_label"}, "query")
        scalar(query["cutoff"], "Day", "query cutoff")
        ref(query["group_label"], "group", "query group", variable=True)
        _schema(sum(kind == "commitment" for kind in entities.values()) <= 12, "declared commitment entity bound")
    elif mechanism == "guarded_transition_learning":
        _keys(query, {"mechanism", "final_stations", "combine", "encoding"}, "query")
        _array(query["final_stations"], 1, 2, "final stations", unique=True)
        for station in query["final_stations"]:
            ref(station, "station", "query final station")
        _schema(query["combine"] == ("single" if len(query["final_stations"]) == 1 else "xor") and query["encoding"] == "4m+2x+y", "query combine/encoding")
        _schema(len({r["operator"] for r in world["records"] if r["kind"] == "operator_menu"}) in {2, 3}, "operator cardinality")
        _schema(sum(r["kind"].startswith("trace_") for r in world["records"]) <= 10, "trace event bound")
    else:
        _keys(query, {"mechanism", "candidate_ids", "candidate_kind", "operation"}, "query")
        _schema(query["candidate_kind"] in {"plan", "bundle"} and query["operation"] == "count_feasible", "candidate query shape")
        _array(query["candidate_ids"], 3, 12, "query candidate IDs", unique=True, canonical=True)
        for cid in query["candidate_ids"]:
            ref(cid, query["candidate_kind"], "query candidate")
            _schema(any(r.get(query["candidate_kind"]) == cid and r["kind"] == ("session_plan" if query["candidate_kind"] == "plan" else "bundle") for r in world["records"]), "candidate missing definition")
        candidate_plans = set(query["candidate_ids"]) if query["candidate_kind"] == "plan" else {p for r in world["records"] if r["kind"] == "bundle" and r["bundle"] in query["candidate_ids"] for p in r["plans"]}
        if any(p["scope"] == "cross_leg" for rs in policies.values() for p in rs):
            # One individually valid definition resolves the typed reference.
            # Disagreement with another definition is a source conjunction with
            # no compatible world, not malformed topology at the parser boundary.
            _schema(all(any(len(p["legs"]) == 2 for p in plan_records[pid]) for pid in candidate_plans), "cross-leg policy requires two-leg candidate plans")
        if any(p["scope"] == "each_bundle" for rs in policies.values() for p in rs):
            _schema(query["candidate_kind"] == "bundle", "bundle policy requires bundle candidates")
    used = set()

    def walk(value):
        if type(value) is dict:
            if set(value) == {"var"}:
                used.add(value["var"])
            else:
                for child in value.values():
                    walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)

    walk(world["records"])
    walk(world["query"])
    _schema(used == set(variables), "unused declared finite variable")
    return world


def _semantic_leaves(value, prefix=""):
    """Finite variable references are single typed fields, not '.var' leaves."""
    if type(value) is dict and set(value) != {"var"}:
        return {path: leaf for key, child in value.items()
                for path, leaf in _semantic_leaves(child, f"{prefix}.{key}".lstrip(".")).items()}
    if type(value) is list:
        return {path: leaf for i, child in enumerate(value)
                for path, leaf in _semantic_leaves(child, f"{prefix}[{i}]").items()}
    return {prefix: value}


def _framing_values(world):
    values = {"mechanism": world["mechanism"], "format": world["format"]}
    for key in ("entities", "variables", "documents", "query"):
        values.update(_semantic_leaves(world[key], key))
    values.update(_semantic_leaves(world["framework"]["rule_clauses"], "framework.rule_clauses"))
    return values


def _allocate_atom_ids(result):
    """Reserve visible identity tokens before allocating parser-only citations."""
    identity_fields = {"entity_id", "variable_id", "document_id", "record_id", "rule_id",
                       "event_id", "program_id", "policy_id", "exception_id", "leg_id"}
    reserved = set()

    def visit(value):
        if type(value) is dict:
            for key, child in value.items():
                if key in identity_fields and type(child) is str:
                    reserved.add(child)
                visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)

    visit(result["world"])
    counter = 0
    for item in result["atom_spans"]:
        while True:
            counter += 1
            identifier = f"A{counter:06d}"
            if identifier not in reserved:
                break
        item["atom_id"] = identifier


def _validate_correspondence(result, messages):
    """Require one grounded span for every semantic and framing field."""
    world = result["world"]
    raw = [message["content"].encode("utf-8") for message in messages]

    def span(value):
        _keys(value, {"message_index", "utf8_start", "utf8_end"}, "source span")
        mi, start, end = (value[k] for k in ("message_index", "utf8_start", "utf8_end"))
        _schema(type(mi) is int and mi in (0, 1) and type(start) is int and type(end) is int
                and 0 <= start < end <= len(raw[mi]), "invalid visible source span")
        try:
            raw[mi][start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PublicParseError("source span splits a UTF-8 scalar", code="SCHEMA_ERROR") from exc
        return mi, start, end

    document_sources = {}
    for item in result["document_spans"]:
        _keys(item, {"document_id", "span"}, "document source correspondence")
        did = item["document_id"]
        _schema(did not in document_sources, "duplicate document source correspondence")
        document_sources[did] = span(item["span"])
    _schema(set(document_sources) == {d["document_id"] for d in world["documents"]}
            and result["source_spans"] == result["document_spans"], "incomplete document source correspondence")
    records = {r["record_id"]: r for r in world["records"]}
    record_sources = {}
    for item in result["record_spans"]:
        _keys(item, {"record_id", "span"}, "record source correspondence")
        rid = item["record_id"]
        _schema(rid in records and rid not in record_sources, "unknown or duplicate record source correspondence")
        observed = span(item["span"])
        owner = document_sources[records[rid]["document_id"]]
        _schema(observed[0] == owner[0] == 1 and owner[1] <= observed[1] < observed[2] <= owner[2], "record lies outside its source document")
        record_sources[rid] = observed
    _schema(set(record_sources) == set(records), "incomplete record source correspondence")
    expected = {(rid, path): leaf for rid, record in records.items()
                for path, leaf in _semantic_leaves({k: v for k, v in record.items() if k not in {"record_id", "document_id"}}).items()}
    seen, identifiers = set(), set()
    for item in result["atom_spans"]:
        _keys(item, {"atom_id", "record_id", "field_path", "span", "typed_value_sha256"}, "semantic source correspondence")
        pair = item["record_id"], item["field_path"]
        _id(item["atom_id"], "semantic atom ID")
        _schema(pair in expected and pair not in seen and item["atom_id"] not in identifiers, "unknown or duplicate semantic source correspondence")
        observed, owner = span(item["span"]), record_sources[pair[0]]
        _schema(observed[0] == owner[0] and owner[1] <= observed[1] < observed[2] <= owner[2], "semantic atom lies outside its source record")
        _schema(item["typed_value_sha256"] == _hash(expected[pair]), "semantic source value mismatch")
        seen.add(pair)
        identifiers.add(item["atom_id"])
    _schema(seen == set(expected), "incomplete semantic source correspondence")
    expected_framing, seen = _framing_values(world), set()
    for item in result["framing_spans"]:
        _keys(item, {"field_path", "span", "typed_value_sha256"}, "framing source correspondence")
        path = item["field_path"]
        _schema(path in expected_framing and path not in seen, "unknown or duplicate framing source correspondence")
        _schema(span(item["span"])[0] == 1, "framing field outside user dossier")
        _schema(item["typed_value_sha256"] == _hash(expected_framing[path]), "framing source value mismatch")
        seen.add(path)
    _schema(seen == set(expected_framing), "incomplete framing source correspondence")


_QUOTED = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))*"'
_INTEGER = r"(?:0|[1-9][0-9]*)"
_VARIABLE = r"\{unreported [A-Za-z][A-Za-z0-9_-]{0,31}\}"
_FIELD = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*(?:\[\d+\])?)\}")
_LIST_PATTERN = r'\[(?:[^\[\]"\n]|' + _QUOTED + r')*\]'
for _depth in range(3):
    _LIST_PATTERN = r'\[(?:[^\[\]"\n]|' + _QUOTED + "|" + _LIST_PATTERN + r')*\]'


def _strict_json(raw, label):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PublicParseError("duplicate " + label + " key", code="SCHEMA_ERROR")
            result[key] = value
        return result

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicParseError("invalid UTF-8 " + label + " JSON", code="SCHEMA_ERROR") from exc
    _schema(type(result) is dict, label + " must be an object")
    return result


def _load_catalogue(path):
    path = Path(path) if path is not None else Path(__file__).with_name("catalogue.json")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PublicParseError("unreadable catalogue", code="SCHEMA_ERROR") from exc
    cat = _strict_json(raw, "catalogue")
    _schema(cat.get("schema_version") == "rineya.distributed_dossier_catalogue.v1" and cat.get("contract_id") == "RIQF-C002-DD-001", "catalogue identity")
    required = {"catalogue_version", "system_message", "envelope", "reading_guide_lines", "framework_variants", "framework_required", "incidental_vocabulary", "guard_realizations", "action_realizations", "policy_predicate_realizations", "selector_realizations", "comparison_realizations", "scope_realizations", "combine_realizations", "operation_realizations", "attribute_realizations", "query_variants", "answer_instructions", "nested_productions", "productions", "fact_templates", "fact_bool_realizations", "condition", "lexical"}
    _schema(required <= set(cat), "incomplete executable public catalogue")
    _schema(all(type(cat[key]) is dict for key in required - {"catalogue_version", "system_message", "reading_guide_lines", "condition"}), "invalid catalogue map types")
    _schema(type(cat["catalogue_version"]) is str and type(cat["condition"]) is str, "invalid catalogue scalar types")
    _schema(type(cat["system_message"]) is str and bool(cat["system_message"]), "catalogue system instruction")
    _schema(type(cat["reading_guide_lines"]) is list and bool(cat["reading_guide_lines"]) and all(type(x) is str and x and "\n" not in x for x in cat["reading_guide_lines"]), "catalogue reading guide")
    _keys(cat["envelope"], {"title", "guide_header", "task", "vocabulary_header", "variable_header", "no_variables", "framework_header", "documents_header", "document", "end_document", "question_header", "entity", "variable", "framework", "record"}, "catalogue envelope")
    _schema(all(type(x) is str and x and "\n" not in x for x in cat["envelope"].values()), "invalid envelope literal")
    _schema(set(cat.get("productions", {})) == set(_FIELDS), "incomplete public catalogue record kinds")
    _schema(set(cat.get("framework_variants", {})) == set.union(*_RULES.values()), "incomplete framework catalogue")
    _schema(cat.get("lexical", {}).get("line_ending") == "LF", "unsupported line convention")
    _schema(cat["lexical"].get("scalar_separator") == ", " and cat["lexical"].get("compound_separator") == " / " and cat["lexical"].get("table_separator") == " | ", "unsupported separator convention")
    _schema(set(cat["query_variants"]) == _MECHANISMS and all(type(v) is list and len(v) == 2 and all(type(x) is str for x in v) for v in cat["query_variants"].values()), "incomplete query catalogue")
    _schema(set(cat["answer_instructions"]) == {"JSON", "LINE", "TOKEN"} and all(type(x) is str for x in cat["answer_instructions"].values()), "incomplete answer-format catalogue")
    _schema(set(cat["fact_templates"]) == set(_PREDICATES) and set(cat["nested_productions"]) == {"state", "program_always", "program_conditional", "child", "leg"}, "incomplete compound/fact catalogue")
    _schema(set(cat["guard_realizations"]) == _GUARDS and set(cat["action_realizations"]) == _ACTIONS and set(cat["operation_realizations"]) == {"xor"}, "unsupported guarded AST catalogue")
    _schema(set(cat["framework_required"]) == _MECHANISMS and all(type(cat["framework_required"][m]) is list and all(type(x) is str for x in cat["framework_required"][m]) and set(cat["framework_required"][m]) == _RULES[m] for m in _MECHANISMS), "missing framework requirements")
    _schema(all(type(v) is list and len(v) == 2 and all(type(x) is str for x in v) and len(set(v)) == 2 for v in cat["framework_variants"].values()), "ambiguous/incomplete framework variants")
    for kind, fields in _FIELDS.items():
        prod = cat["productions"][kind]
        _schema(type(prod) is dict and type(prod.get("fields")) is dict and type(prod.get("variants")) is list and type(prod.get("table_headers")) is list, "invalid production shape")
        _schema(set(prod.get("fields", {})) == fields and len(prod.get("variants", [])) == 2 and len(prod.get("table_headers", [])) == 2, "incomplete production " + kind)
        _schema(all(type(v) is str for v in prod["fields"].values()), "invalid field type descriptor")
        _schema(all(type(x) is str for x in prod["variants"]) and prod.get("table_separator") == ["---"] * (len(fields) + 2), "malformed production/table convention")
        for i, header in enumerate(prod["table_headers"]):
            _schema(header == ["Citation" if i == 0 else "Record", kind, *prod["fields"]], "unknown table columns")
    return cat, hashlib.sha256(raw).hexdigest()


def _split_items(text, separator):
    """Split only unquoted top-level delimiters, retaining exact scalar offsets."""
    stack, quoted, escaped = [], False, False
    result, start, i = [], 0, 0
    pairs = {"[": "]", "(": ")", "{": "}"}
    while i < len(text):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
            i += 1
            continue
        if c == '"':
            quoted = True
        elif c in pairs:
            stack.append(pairs[c])
        elif c in "])}":
            if not stack or stack.pop() != c:
                raise PublicParseError("malformed compound/list delimiters")
        elif not stack and text.startswith(separator, i):
            result.append((text[start:i], start))
            i += len(separator)
            start = i
            continue
        i += 1
    if stack or quoted:
        raise PublicParseError("unterminated string/list")
    result.append((text[start:], start))
    return result


def _pattern(spec):
    if spec in {"String", "Label"}:
        return _QUOTED
    if spec.startswith("Value_"):
        return "(?:" + _VARIABLE + "|" + _pattern(spec[6:]) + ")"
    if "[]" in spec or spec.startswith("Set[") or spec == "Arguments":
        return _LIST_PATTERN
    if spec.startswith("EntityId"):
        return _QUOTED
    if spec.startswith("ID"):
        return r"[A-Za-z][A-Za-z0-9_-]{0,31}"
    if spec in _SCALAR_BOUNDS or spec.startswith("integer") or spec in {"Bit|SmallCount"}:
        return _INTEGER
    if spec == "State":
        return r"\(m=[^\n]*\)"
    return r"[^\n]+?"


def _match(template, text, specs=None):
    """Match all literal bytes; captures are lexically constrained where possible."""
    specs = specs or {}
    chunks, names, last = [], [], 0
    for i, found in enumerate(_FIELD.finditer(template)):
        name = found.group(1)
        chunks.append(re.escape(template[last:found.start()]))
        chunks.append(f"(?P<f{i}>" + _pattern(specs.get(name, "Text")) + ")")
        names.append((name, f"f{i}"))
        last = found.end()
    chunks.append(re.escape(template[last:]))
    match = re.fullmatch("".join(chunks), text)
    if match is None:
        return None
    result = {}
    for name, group in names:
        if name in result:
            raise PublicParseError("repeated placeholder in catalogue production")
        result[name] = (match.group(group), match.start(group), match.end(group))
    return result


def _set_field(obj, path, value):
    indexed = re.fullmatch(r"([A-Za-z_]+)\[(\d+)\]", path)
    if indexed:
        key, ix = indexed.group(1), int(indexed.group(2))
        obj.setdefault(key, [])
        while len(obj[key]) <= ix:
            obj[key].append(None)
        obj[key][ix] = value
    else:
        obj[path] = value


class _Decoder:
    def __init__(self, catalogue):
        self.cat = catalogue
        self.labels = {}
        self.variables = {}

    def mapped(self, text, mapping):
        matches = [key for key, literal in mapping.items() if literal == text]
        if len(matches) != 1:
            raise PublicParseError("unknown or ambiguous finite phrase")
        return matches[0]

    def fields(self, captures, specs):
        obj, atoms = {}, []
        for field, (text, start, end) in captures.items():
            value, inner = self.value(text, specs[field])
            _set_field(obj, field, value)
            atoms.extend((field + suffix, start + a, start + b, val) for suffix, a, b, val in inner)
        return obj, atoms

    def value(self, text, spec):
        variable = re.fullmatch(_VARIABLE, text)
        if variable:
            if not spec.startswith("Value_"):
                raise PublicParseError("finite variable in non-variable field")
            vid = text[len("{unreported "):-1]
            if vid not in self.variables:
                raise PublicParseError("undeclared finite variable")
            return {"var": vid}, [("", 0, len(text), {"var": vid})]
        spec = spec[6:] if spec.startswith("Value_") else spec
        if spec in {"String", "Label"} or spec.startswith("EntityId"):
            if not re.fullmatch(_QUOTED, text):
                raise PublicParseError("malformed quoted scalar")
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PublicParseError("malformed JSON string") from exc
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                raise PublicParseError("unpaired Unicode surrogate")
            if spec.startswith("EntityId"):
                if value not in self.labels:
                    raise PublicParseError("unknown quoted entity label")
                value = self.labels[value]
            return value, [("", 0, len(text), value)]
        if spec.startswith("ID"):
            _id(text, "public token")
            return text, [("", 0, len(text), text)]
        if spec in _SCALAR_BOUNDS or spec.startswith("integer") or spec == "Bit|SmallCount":
            if not re.fullmatch(_INTEGER, text):
                raise PublicParseError("noncanonical integer")
            # Every catalogue scalar is at most 31. Reject before Python's
            # decimal-conversion limit can escape this total public boundary.
            _schema(len(text) <= 2, "integer exceeds finite public scalar bound")
            value = int(text)
            return value, [("", 0, len(text), value)]
        if spec == "BoolWord":
            value = int(self.mapped(text, self.cat["fact_bool_realizations"]))
            return value, [("", 0, len(text), value)]
        if spec in {"Guard", "Action", "Scope", "Attribute", "Comparison", "Combine"}:
            key = {"Guard": "guard", "Action": "action", "Scope": "scope", "Attribute": "attribute", "Comparison": "comparison", "Combine": "combine"}[spec] + "_realizations"
            value = self.mapped(text, self.cat[key])
            return value, [("", 0, len(text), value)]
        if spec == "State":
            fields = {r: "Value_Bit" for r in "mxy"}
            captures = _match(self.cat["nested_productions"]["state"], text, fields)
            if captures is None:
                raise PublicParseError("malformed state")
            obj, atoms = self.fields(captures, fields)
            return obj, [("." + p, a, b, v) for p, a, b, v in atoms]
        if spec == "GuardedProgram":
            for which in ("program_always", "program_conditional"):
                fields = {"program_id": "ID", "guard": "Guard", "action": "Action"}
                captures = _match(self.cat["nested_productions"][which], text, fields)
                if captures is not None:
                    obj, atoms = self.fields(captures, fields)
                    if which == "program_always":
                        obj["guard"] = "always"
                        a = text.index("always")
                        atoms.append(("guard", a, a + len("always"), "always"))
                    return obj, [("." + p, a, b, v) for p, a, b, v in atoms]
            raise PublicParseError("unknown guarded-program AST")
        if spec in {"ChildCommitment", "Leg"}:
            fields = ({"commitment": "EntityId commitment", "owner": "Value_EntityId sponsor", "quantity": "Value_Quantity"} if spec == "ChildCommitment" else {"leg_id": "ID", "slot": "Slot", "room": "EntityId room", "kit": "EntityId kit", "leader": "EntityId person", "people": "Set[EntityId person]", "group": "EntityId group", "kit_demand": "Demand"})
            captures = _match(self.cat["nested_productions"]["child" if spec == "ChildCommitment" else "leg"], text, fields)
            if captures is None:
                raise PublicParseError("malformed child/leg AST")
            obj, atoms = self.fields(captures, fields)
            return obj, [("." + p, a, b, v) for p, a, b, v in atoms]
        if spec == "PolicyPredicate":
            for code, template in self.cat["policy_predicate_realizations"].items():
                template = template.replace("<credential>", "{credential}")
                captures = _match(template, text, {"credential": "EntityId credential"})
                if captures is not None:
                    obj, atoms = self.fields(captures, {"credential": "EntityId credential"})
                    obj["code"] = code
                    return obj, [(".code", 0, len(text), code)] + [("." + p, a, b, v) for p, a, b, v in atoms]
            raise PublicParseError("unknown policy AST")
        if spec == "Selector":
            for kind, template in self.cat["selector_realizations"].items():
                for old, new in (("<plan>", "{plan}"), ("<leg-id>", "{leg_id}"), ("<group>", "{group}"), ("<bundle>", "{bundle}")):
                    template = template.replace(old, new)
                fields = {"plan": "EntityId plan", "leg_id": "ID", "group": "EntityId group", "bundle": "EntityId bundle"}
                captures = _match(template, text, fields)
                if captures is not None:
                    obj, atoms = self.fields(captures, fields)
                    obj["kind"] = kind
                    return obj, [(".kind", 0, len(text), kind)] + [("." + p, a, b, v) for p, a, b, v in atoms]
            raise PublicParseError("unknown selector AST")
        if spec == "FactCondition":
            fields = {"predicate": "Predicate", "arguments": "Arguments", "comparison": "Comparison", "value": "Bit|SmallCount"}
            captures = _match(self.cat["condition"], text, fields)
            if captures is None:
                raise PublicParseError("malformed condition AST")
            obj, atoms = self.fields(captures, fields)
            return obj, [("." + p, a, b, v) for p, a, b, v in atoms]
        if "[]" in spec or spec.startswith("Set[") or spec == "Arguments":
            if not text.startswith("[") or not text.endswith("]"):
                raise PublicParseError("list requires square brackets")
            if spec == "Arguments":
                elem = "ScalarOrEntity"
            elif spec.startswith("Set["):
                elem = spec[4:spec.index("]")]
            else:
                elem = spec[:spec.index("[]")].removeprefix("ordered ")
            separator = self.cat["lexical"]["compound_separator" if elem in {"Leg", "ChildCommitment", "GuardedProgram"} else "scalar_separator"]
            parts = [] if text == "[]" else _split_items(text[1:-1], separator)
            result, atoms = [], []
            for i, (part, offset) in enumerate(parts):
                val, inner = self.value(part, elem)
                result.append(val)
                atoms.extend((f"[{i}]" + p, 1 + offset + a, 1 + offset + b, v) for p, a, b, v in inner)
            return result, atoms
        if spec == "ScalarOrEntity":
            return self.value(text, "EntityId" if text.startswith('"') else "SmallCount")
        if spec == "Predicate":
            if text not in _PREDICATES:
                raise PublicParseError("unknown predicate token")
            return text, [("", 0, len(text), text)]
        if spec == "Register" or "|" in spec:
            permitted = set(spec.split("|")) if "|" in spec else set("mxy")
            if text not in permitted:
                raise PublicParseError("unknown enum token")
            return text, [("", 0, len(text), text)]
        if spec == "constant xor":
            value = self.mapped(text, self.cat["operation_realizations"])
            return value, [("", 0, len(text), value)]
        if spec == "string from a frozen finite vocabulary":
            if not any(text in values for values in self.cat["incidental_vocabulary"].values()):
                raise PublicParseError("uncatalogued incidental string")
            return text, [("", 0, len(text), text)]
        raise PublicParseError("unsupported catalogue field descriptor " + spec)

    def fact_clause(self, text):
        for pred, template in self.cat["fact_templates"].items():
            fields = {f"arguments[{i}]": ("Slot" if typ == "Slot" else "EntityId " + typ) for i, typ in enumerate(_PREDICATES[pred])}
            fields["value"] = "Value_" + ("SmallCount" if pred in {"capacity", "kit_units"} else "BoolWord")
            captures = _match(template, text, fields)
            if captures is not None:
                obj, atoms = self.fields(captures, fields)
                obj["predicate"] = pred
                atoms.append(("predicate", 0, len(text), pred))
                return obj, atoms
        raise PublicParseError("unknown fact clause")

    def production(self, text):
        possibilities = []
        for kind, production in self.cat["productions"].items():
            for template in production["variants"]:
                fields = production["fields"].copy()
                fields.update({"scope": "Scope", "attribute": "Attribute", "arguments": "Arguments"})
                captures = _match(template, text, fields)
                if captures is None:
                    continue
                try:
                    if "fact_clause" in captures:
                        clause, offset, _ = captures["fact_clause"]
                        obj, atoms = self.fact_clause(clause)
                        if (obj["predicate"] in {"capacity", "kit_units"}) != (kind == "fact_count"):
                            continue
                        atoms = [(p, offset + a, offset + b, v) for p, a, b, v in atoms]
                    else:
                        obj, atoms = self.fields(captures, fields)
                    obj["kind"] = kind
                    atoms.append(("kind", 0, len(text), kind))
                    possibilities.append((obj, atoms))
                except PublicParseError:
                    continue
        if len(possibilities) != 1:
            raise PublicParseError("unknown or ambiguous source clause")
        return possibilities[0]


class _PublicReader:
    def __init__(self, messages, catalogue):
        self.messages, self.cat = messages, catalogue
        self.text = messages[1]["content"]
        self.decoder = _Decoder(catalogue)
        self.lines = self.text.split("\n")
        self.starts, pos = [], 0
        for line in self.lines:
            self.starts.append(pos)
            pos += len(line) + 1
        self.cursor = 0
        self.record_spans, self.atom_spans, self.framing_spans = [], [], []
        self.document_spans, self.coverage_spans = [], []
        self.coverage_spans.append({"category": "system_instruction", "span": self.span(0, len(messages[0]["content"]), 0)})

    def span(self, start, end, message_index=1):
        text = self.messages[message_index]["content"]
        return {"message_index": message_index, "utf8_start": len(text[:start].encode("utf-8")), "utf8_end": len(text[:end].encode("utf-8"))}

    def take(self, category):
        if self.cursor >= len(self.lines):
            raise PublicParseError("truncated dossier", message_index=1, utf8_start=len(self.text.encode("utf-8")))
        index = self.cursor
        line, start = self.lines[index], self.starts[index]
        self.cursor += 1
        end = start + len(line) + (index < len(self.lines) - 1)
        if end > start:
            self.coverage_spans.append({"category": category, "span": self.span(start, end)})
        return line, start

    def expect(self, literal, category="framing"):
        line, start = self.take(category)
        if line != literal:
            raise PublicParseError("missing, altered, or extra " + category, message_index=1, utf8_start=self.span(start, start)["utf8_start"])
        return start

    def peek(self):
        return self.lines[self.cursor] if self.cursor < len(self.lines) else None

    def frame(self, path, start, end, value):
        self.framing_spans.append({"field_path": path, "span": self.span(start, end), "typed_value_sha256": _hash(value)})

    def atoms(self, record_id, start, atoms):
        for path, a, b, value in atoms:
            _schema(b > a, "empty semantic atom span")
            self.atom_spans.append({"atom_id": f"A{len(self.atom_spans) + 1:06d}", "record_id": record_id, "field_path": path, "span": self.span(start + a, start + b), "typed_value_sha256": _hash(value)})

    def decode_line(self, template, specs, category):
        line, start = self.take(category)
        captures = _match(template, line, specs)
        if captures is None:
            raise PublicParseError("malformed " + category, message_index=1, utf8_start=self.span(start, start)["utf8_start"])
        obj, atoms = self.decoder.fields(captures, specs)
        return obj, atoms, line, start

    def parse(self):
        env = self.cat["envelope"]
        self.expect(env["title"])
        dossier_start = self.expect(env["guide_header"], "reading_guide_heading")
        for line in self.cat["reading_guide_lines"]:
            self.expect(line, "reading_guide_clause")
        self.expect("", "blank_line")
        task, atoms, _, start = self.decode_line(env["task"], {"mechanism": "|".join(sorted(_MECHANISMS)), "format": "JSON|LINE|TOKEN"}, "task")
        for path, a, b, value in atoms:
            self.frame(path, start + a, start + b, value)
        self.expect("", "blank_line")
        self.expect(env["vocabulary_header"], "vocabulary_heading")
        entities = []
        while self.peek() not in {"", None}:
            entity, atoms, _, start = self.decode_line(env["entity"], {"public_label": "Label", "kind": "|".join(sorted(_KINDS)), "entity_id": "ID"}, "entity_definition")
            for path, a, b, value in atoms:
                self.frame(f"entities[{len(entities)}]." + path, start + a, start + b, value)
            if entity["public_label"] in self.decoder.labels:
                raise PublicParseError("duplicate public label declaration", code="SCHEMA_ERROR")
            self.decoder.labels[entity["public_label"]] = entity["entity_id"]
            entities.append(entity)
        self.expect("", "blank_line")
        self.expect(env["variable_header"], "variable_heading")
        variables = []
        if self.peek() == env["no_variables"]:
            self.expect(env["no_variables"], "finite_domain_declaration")
        else:
            while self.peek() not in {"", None}:
                line, start = self.take("finite_domain_declaration")
                captures = _match(env["variable"], line, {"variable_id": "ID", "scalar_type": "Bit|Quantity|SmallCount|EntityId|Day|Tick|Register", "choices": "Arguments"})
                if captures is None:
                    raise PublicParseError("malformed finite domain")
                scalar_type = captures["scalar_type"][0]
                specs = {"variable_id": "ID", "scalar_type": "Bit|Quantity|SmallCount|EntityId|Day|Tick|Register", "choices": "Set[" + scalar_type + "]"}
                variable, atoms = self.decoder.fields(captures, specs)
                if variable["variable_id"] in self.decoder.variables:
                    raise PublicParseError("duplicate finite variable", code="SCHEMA_ERROR")
                self.decoder.variables[variable["variable_id"]] = variable
                for path, a, b, value in atoms:
                    self.frame(f"variables[{len(variables)}]." + path, start + a, start + b, value)
                variables.append(variable)
        self.expect("", "blank_line")
        self.expect(env["framework_header"], "framework_heading")
        framework = []
        while self.peek() not in {"", None}:
            line, start = self.take("framework_clause")
            captures = _match(env["framework"], line, {"rule_id": "ID"})
            if captures is None:
                raise PublicParseError("malformed framework citation")
            text, a, b = captures["clause"]
            matches = [(code, variant) for code, variants in self.cat["framework_variants"].items() for variant, literal in enumerate(variants) if literal == text]
            if len(matches) != 1:
                raise PublicParseError("unknown or conflicting framework text")
            code, variant = matches[0]
            rid = captures["rule_id"][0]
            _id(rid, "framework rule ID")
            obj = {"rule_id": rid, "code": code, "public_variant": variant}
            self.frame(f"framework.rule_clauses[{len(framework)}].rule_id", start + captures["rule_id"][1], start + captures["rule_id"][2], rid)
            self.frame(f"framework.rule_clauses[{len(framework)}].code", start + a, start + b, code)
            self.frame(f"framework.rule_clauses[{len(framework)}].public_variant", start + a, start + b, variant)
            framework.append(obj)
        self.expect("", "blank_line")
        self.expect(env["documents_header"], "documents_heading")
        documents, records, incidental_bytes = [], [], 0
        while self.peek() != env["question_header"]:
            doc, atoms, _, doc_start = self.decode_line(env["document"], {"document_id": "ID", "title": "String", "dated_day": "Day", "purpose": "|".join(sorted(_PURPOSES)), "style": "controlled_prose|small_table"}, "document_header")
            doc["record_ids"] = []
            di = len(documents)
            for path, a, b, value in atoms:
                self.frame(f"documents[{di}]." + path, doc_start + a, doc_start + b, value)
            ending = env["end_document"].replace("{document_id}", doc["document_id"])
            while self.peek() != ending:
                if self.peek() is None:
                    raise PublicParseError("missing document terminator")
                record_start = self.starts[self.cursor]
                if doc["style"] == "controlled_prose":
                    line, source_start = self.take("source_clause")
                    captures = _match(env["record"], line, {"record_id": "ID"})
                    if captures is None:
                        raise PublicParseError("malformed record citation")
                    rid = captures["record_id"][0]
                    body, offset, _ = captures["clause"]
                    obj, atoms = self.decoder.production(body)
                    self.atoms(rid, source_start + offset, atoms)
                    id_a, id_b = captures["record_id"][1:]
                    source_end = source_start + len(line)
                else:
                    header, header_start = self.take("table_header")
                    matches = [(kind, prod) for kind, prod in self.cat["productions"].items() if header in [self.cat["lexical"]["table_separator"].join(h) for h in prod["table_headers"]]]
                    if len(matches) != 1:
                        raise PublicParseError("unknown, extra, or ambiguous table columns")
                    kind, prod = matches[0]
                    self.expect(self.cat["lexical"]["table_separator"].join(prod["table_separator"]), "table_separator")
                    line, source_start = self.take("table_source_row")
                    cells = _split_items(line, self.cat["lexical"]["table_separator"])
                    field_names = list(prod["fields"])
                    if len(cells) != len(field_names) + 2:
                        raise PublicParseError("wrong table field count")
                    citation = re.fullmatch(r"\[([A-Za-z][A-Za-z0-9_-]{0,31})\]", cells[0][0])
                    if citation is None or cells[1][0] != kind:
                        raise PublicParseError("malformed table citation/kind")
                    rid, id_a, id_b = citation.group(1), cells[0][1] + 1, cells[0][1] + len(cells[0][0]) - 1
                    obj, atoms = {"kind": kind}, [("kind", cells[1][1], cells[1][1] + len(kind), kind)]
                    specs = prod["fields"].copy()
                    specs.update({"scope": "Scope", "attribute": "Attribute", "arguments": "Arguments"})
                    for field, (cell, offset) in zip(field_names, cells[2:]):
                        value, inner = self.decoder.value(cell, specs[field])
                        obj[field] = value
                        atoms.extend((field + p, offset + a, offset + b, v) for p, a, b, v in inner)
                    self.atoms(rid, source_start, atoms)
                    source_end = source_start + len(line)
                obj.update({"record_id": rid, "document_id": doc["document_id"]})
                self.frame(f"documents[{di}].record_ids[{len(doc['record_ids'])}]", source_start + id_a, source_start + id_b, rid)
                doc["record_ids"].append(rid)
                records.append(obj)
                self.record_spans.append({"record_id": rid, "span": self.span(record_start, source_end)})
                if obj["kind"] == "incidental":
                    incidental_bytes += self.span(record_start, source_end)["utf8_end"] - self.span(record_start, source_end)["utf8_start"]
            end_start = self.expect(ending, "document_terminator")
            self.document_spans.append({"document_id": doc["document_id"], "span": self.span(doc_start, end_start + len(ending))})
            documents.append(doc)
            self.expect("", "blank_line")
        dossier_end = self.starts[self.cursor]
        self.expect(env["question_header"], "question_heading")
        query_line, query_start = self.take("query_clause")
        mechanism = task["mechanism"]
        specs = ({"cutoff": "Day", "group_label": "Value_EntityId group"} if mechanism == "commitment_ledger" else {"final_stations": "ordered EntityId[]", "combine": "Combine"} if mechanism == "guarded_transition_learning" else {"candidate_kind": "plan|bundle", "candidate_ids": "Set[EntityId]"})
        captures = [m for template in self.cat["query_variants"][mechanism] if (m := _match(template, query_line, specs)) is not None]
        if len(captures) != 1:
            raise PublicParseError("unknown or duplicate query")
        query, atoms = self.decoder.fields(captures[0], specs)
        query["mechanism"] = mechanism
        if mechanism == "guarded_transition_learning":
            query["encoding"] = "4m+2x+y"
            atoms.append(("encoding", 0, len(query_line), query["encoding"]))
        elif mechanism == "policy_feasibility":
            query["operation"] = "count_feasible"
            atoms.append(("operation", 0, len(query_line), query["operation"]))
        for path, a, b, value in atoms:
            self.frame("query." + path, query_start + a, query_start + b, value)
        self.frame("query.mechanism", query_start, query_start + len(query_line), mechanism)
        self.expect(self.cat["answer_instructions"][task["format"]], "answer_format_instruction")
        if self.cursor != len(self.lines):
            raise PublicParseError("extra prose or terminal newline")
        world = {"world_schema": "rineya.distributed_dossier_world.v1", "mechanism": mechanism, "entities": entities, "variables": variables, "documents": documents, "records": records, "framework": {"semantic_version": "RIQF-C002-DD-001", "rule_clauses": framework, "closed_candidate_list": True, "unknown_defaults": "NONE", "source_assertions_are_conjunctive": True}, "query": query, "format": task["format"]}
        _validate_world(world, self.cat)
        # Counts final visible dossier bytes, including guide and source framing.
        dossier_span = self.span(dossier_start, dossier_end)
        _schema(incidental_bytes <= 0.15 * (dossier_span["utf8_end"] - dossier_span["utf8_start"]), "incidental byte fraction exceeds bound")
        return {"world": world, "record_spans": self.record_spans, "atom_spans": self.atom_spans, "framing_spans": self.framing_spans, "document_spans": self.document_spans, "source_spans": self.document_spans, "coverage_spans": self.coverage_spans, "dossier_span": dossier_span, "unparsed_assertions": []}


def _check_frozen_catalogue(catalogue_path, catalogue, digest):
    """Bind local catalogue, review and both implementations; never trust a flag."""
    path = Path(catalogue_path) if catalogue_path is not None else Path(__file__).with_name("catalogue.json")
    receipt_path = path.with_name("catalogue_freeze_receipt.json")
    if not receipt_path.is_file():
        raise PublicParseError("missing independent catalogue freeze receipt", code="SCHEMA_ERROR")
    root = Path(__file__).resolve().parents[2]
    try:
        receipt = _strict_json(receipt_path.read_bytes(), "freeze receipt")
    except OSError as exc:
        raise PublicParseError("unreadable freeze receipt", code="SCHEMA_ERROR") from exc
    _keys(receipt, {"schema_version", "status", "catalogue_id", "catalogue_sha256", "design_sha256", "contract_sha256", "independent_review_sha256", "code_identity", "review_path"}, "freeze receipt")
    _schema(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1 and receipt["status"] == "INDEPENDENTLY_REVIEWED_FROZEN", "unreviewed freeze receipt")
    _schema(receipt["catalogue_id"] == catalogue["catalogue_version"] and receipt["catalogue_sha256"] == digest, "stale catalogue pin")
    review_relative = "cycles/003-dossier-implementation/catalogue_review/REVIEW.json"
    _schema(receipt["review_path"] == review_relative, "unexpected independent review destination")
    fixed = {
        "design_sha256": root / "cycles/002-source-registry/reviewed_blueprint_v1/DESIGN.md",
        "contract_sha256": root / "cycles/002-source-registry/reviewed_blueprint_v1/CONTRACT.json",
        "independent_review_sha256": root / review_relative,
    }
    for key, target in fixed.items():
        _schema(type(receipt[key]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is not None and target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == receipt[key], "stale reviewed source pin " + key)
    blueprint_pin_path = fixed["design_sha256"].with_name("PIN_RECEIPT.json")
    _schema(blueprint_pin_path.is_file(), "missing reviewed blueprint pin receipt")
    try:
        blueprint_pin = _strict_json(blueprint_pin_path.read_bytes(), "reviewed blueprint pin")
    except OSError as exc:
        raise PublicParseError("unreadable reviewed blueprint pin receipt", code="SCHEMA_ERROR") from exc
    _schema(type(blueprint_pin.get("schema_version")) is int and blueprint_pin["schema_version"] == 1
            and blueprint_pin.get("status") == "HASH_PINNED_REVIEWED_DESIGN_ONLY"
            and blueprint_pin.get("design_id") == catalogue["contract_id"], "unreviewed blueprint pin identity")
    _keys(blueprint_pin.get("files"), {"DESIGN.md", "CONTRACT.json"}, "reviewed blueprint file pins")
    _schema(blueprint_pin["files"]["DESIGN.md"] == receipt["design_sha256"]
            and blueprint_pin["files"]["CONTRACT.json"] == receipt["contract_sha256"], "reviewed blueprint pin drift")
    _keys(receipt["code_identity"], {"schema.py", "renderer.py", "public_parser.py", "independent_solver.py"}, "code pin")
    for name, expected in receipt["code_identity"].items():
        target = Path(__file__).with_name(name)
        _schema(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected) is not None and target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected, "stale code pin " + name)
    review = _strict_json(fixed["independent_review_sha256"].read_bytes(), "independent review")
    _schema(type(review.get("schema_version")) is int and review["schema_version"] == 1 and review.get("status") == "PASS_INDEPENDENT_CATALOGUE_AND_IMPLEMENTATION_REVIEW", "independent review did not pass")
    _schema(type(review.get("unresolved_critical_or_major_findings")) is list and review["unresolved_critical_or_major_findings"] == [], "missing or unresolved independent review findings")
    _keys(review.get("producer_identity"), {"schema.py", "renderer.py", "public_parser.py", "independent_solver.py", "catalogue.json"}, "review producer identity")
    _schema(review["producer_identity"] == dict(receipt["code_identity"], **{"catalogue.json": digest}) and review.get("design_sha256") == receipt["design_sha256"] and review.get("contract_sha256") == receipt["contract_sha256"], "independent review source identity mismatch")
    # Exact bytes are pinned, but this is a local receipt, not an authenticated signature.
    return receipt


def parse_public(messages, catalogue_path=None, *, require_frozen=False):
    """Read exactly two visible role messages and reconstruct CONTRACT PublicWorld.

    Returns ParsedPublic plus UTF-8 byte coverage/framing/document source maps,
    public-input hashes and catalogue readiness.  Default development parsing is
    explicitly UNREVIEWED_MECHANICS_ONLY and cannot authorize row admission.
    Production callers MUST set require_frozen=True; the fixed sibling receipt
    binds catalogue, independent review, blueprint, and four code files.
    """
    _schema(type(require_frozen) is bool, "require_frozen must be Boolean")
    _schema(type(messages) is list and len(messages) == 2, "exactly one system and one user message required")
    for i, (message, role) in enumerate(zip(messages, ("system", "user"))):
        _keys(message, {"role", "content"}, "message")
        _schema(message["role"] == role and type(message["content"]) is str, "wrong actual message role/content")
        text = message["content"]
        _schema(bool(text) and "\r" not in text and "\x00" not in text and not any(0xD800 <= ord(c) <= 0xDFFF for c in text), "unsupported line/control/Unicode input")
        _schema(len(text.encode("utf-8")) <= 200000, "public message byte bound")
    catalogue, digest = _load_catalogue(catalogue_path)
    if messages[0]["content"] != catalogue["system_message"]:
        raise PublicParseError("altered actual system instruction", message_index=0, utf8_start=0)
    frozen = _check_frozen_catalogue(catalogue_path, catalogue, digest) if require_frozen else None
    result = _PublicReader(messages, catalogue).parse()
    _allocate_atom_ids(result)
    _validate_correspondence(result, messages)
    result.update({"parser_identity_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_identity_sha256": digest, "catalogue_status": "INDEPENDENTLY_REVIEWED_FROZEN" if frozen else "UNREVIEWED_MECHANICS_ONLY", "admission_ready": bool(frozen), "messages_sha256": _hash(messages)})
    # Exact ordered coverage partition is independently checked after parsing.
    for i, message in enumerate(messages):
        cursor = 0
        for item in result["coverage_spans"]:
            s = item["span"]
            if s["message_index"] == i:
                _schema(s["utf8_start"] == cursor and s["utf8_end"] > cursor, "non-total visible byte coverage")
                cursor = s["utf8_end"]
        _schema(cursor == len(message["content"].encode("utf-8")), "unconsumed visible bytes")
    return result
