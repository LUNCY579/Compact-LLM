"""Independent finite semantics for reconstructed distributed-dossier public facts.

No author generator, private world, renderer, target, or author solver is imported.
The entry point consumes the public parser's reconstruction. Typed fixture tests
are mechanics checks, never dataset construction/admission. Enumeration is finite
(at most nine source bindings times 64 learned-program assignments). Certificates
are deliberately nonminimal complete-source support, with no global proof claim.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

MECHANISMS = {"commitment_ledger", "guarded_transition_learning", "policy_feasibility"}
REGISTERS = ("m", "x", "y")
GUARDS = {"always", "m0", "m1", "x0", "x1", "y0", "y1"}
ACTIONS = {"flip_m", "flip_x", "flip_y", "swap_mx", "swap_xy",
           "copy_mx", "copy_xm", "copy_xy", "copy_yx"}
KINDS = {"commitment", "sponsor", "signer", "group", "station", "operator",
         "person", "room", "kit", "credential", "plan", "bundle"}
SCOPES = {
    "each_leg": {"capacity_ok", "room_available", "kit_available", "kit_units_ok",
                 "leader_credential", "all_members"},
    "cross_leg": {"slots_increase", "rooms_differ", "groups_equal"},
    "each_bundle": {"shared_kit_quota"},
}
SELECTORS = {"each_leg": {"plan", "leg", "group"}, "cross_leg": {"plan"},
             "each_bundle": {"bundle"}}
FACT_ARGS = {"room_available": ("room", "slot"), "kit_available": ("kit", "slot"),
             "credential": ("person", "credential"), "member": ("person", "group"),
             "capacity": ("room",), "kit_units": ("kit", "slot")}
BOOL_FACTS = {"room_available", "kit_available", "credential", "member"}
INCIDENTAL_VALUES = {"curtain_color": {"red", "blue", "green", "cream"},
                     "slip_paper_color": {"white", "blue", "yellow", "cream"},
                     "meeting_refreshment": {"tea", "coffee", "water", "juice"}}
ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
SCHEMA = "rineya.distributed_dossier_world.v1"
RULE_CODES = {"FINITE_WORLDS", "STATUS", "ENTITY_IDENTITY", "A_MEMBERSHIP", "A_AUTHORITY",
              "A_TIME", "A_SINGLE_SUCCESSOR", "A_TRANSITIONS", "A_RESULT", "B_PROGRAM",
              "B_INFER", "B_TRACE", "B_XOR", "B_RESULT", "C_FACTS", "C_LOCAL",
              "C_CROSS_LEG", "C_BUNDLE", "C_EXEMPTION", "C_RESULT"}


class SchemaError(ValueError):
    """Malformed public mechanics block solving; this is never a status."""


class _Incompatible(Exception):
    """Well-typed source conjunction has no world under this binding."""


class _MissingPremise(Exception):
    """Unasserted policy evidence, pending decisive-false conjunction evaluation."""
    def __init__(self, keys: set[tuple[Any, ...]]):
        self.keys = keys


def _conjunction(checks: list[bool | _MissingPremise]) -> bool | _MissingPremise:
    """False settles a conjunction; unresolved reports receive no default value."""
    if any(check is False for check in checks):
        return False
    missing = set().union(*(check.keys for check in checks if isinstance(check, _MissingPremise)))
    return _MissingPremise(missing) if missing else True


def _policy_check(evaluate) -> bool | _MissingPremise:
    try:
        return evaluate()
    except _MissingPremise as missing:
        return missing


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _int(value: Any, lo: int, hi: int, name: str) -> None:
    if type(value) is not int or not lo <= value <= hi:
        raise SchemaError(f"{name} requires exact integer {lo}..{hi}")


def _id(value: Any, name: str) -> None:
    if type(value) is not str or not ID.fullmatch(value):
        raise SchemaError(f"{name} requires a public ID")


def _shape(value: Any, fields: set[str], name: str) -> None:
    if type(value) is not dict or set(value) != fields:
        raise SchemaError(f"{name} requires exactly {sorted(fields)}")


def _public_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def _semantic_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Public Value_T references are one semantic field, never a '.var' atom."""
    if type(value) is dict and set(value) != {"var"}:
        return {path: leaf for key, item in value.items()
                for path, leaf in _semantic_leaves(item, f"{prefix}.{key}".lstrip(".")).items()}
    if type(value) is list:
        return {path: leaf for index, item in enumerate(value)
                for path, leaf in _semantic_leaves(item, f"{prefix}[{index}]").items()}
    return {prefix: value}


def _framing_values(world: dict[str, Any]) -> dict[str, Any]:
    values = {"mechanism": world["mechanism"], "format": world["format"]}
    for key in ("entities", "variables", "documents", "query"):
        values.update(_semantic_leaves(world[key], key))
    values.update(_semantic_leaves(world["framework"]["rule_clauses"], "framework.rule_clauses"))
    return values


def _framing_atom_id(path: str) -> str:
    prefix = "Q" if path.startswith("query.") else "F" if path.startswith("framework.") else "M"
    return prefix + _digest(path)[:24]


def _public_identity_ids(world: dict[str, Any]) -> set[str]:
    identities = set()
    def visit(value: Any) -> None:
        if type(value) is dict:
            for key, item in value.items():
                if key.endswith("_id") and type(item) is str:
                    identities.add(item)
                visit(item)
        elif type(value) is list:
            for item in value:
                visit(item)
    visit({k: v for k, v in world.items() if k != "_public"})
    return identities


def _framing_atom_ids(world: dict[str, Any]) -> dict[str, str]:
    occupied = _public_identity_ids(world)
    occupied.update(atom["atom_id"] for atom in world.get("_public", {}).get("atom_spans", []))
    result = {}
    for path in sorted(_framing_values(world)):
        candidate = _framing_atom_id(path)
        nonce = 0
        while candidate in occupied:
            nonce += 1
            candidate = candidate[0] + _digest([path, nonce])[:24]
        result[path] = candidate
        occupied.add(candidate)
    return result


def _validate_public_metadata(world: dict[str, Any]) -> None:
    if "_public" not in world:
        return
    public = world["_public"]
    required = {"atom_spans", "record_spans", "framing_spans", "messages_sha256"}
    optional = {"messages", "coverage_spans", "require_source_bytes", "parser_identity_sha256",
                "catalogue_identity_sha256", "admission_ready", "catalogue_status"}
    if type(public) is not dict or not required <= set(public) or set(public) - required - optional:
        raise SchemaError("public metadata requires complete record, atom, and framing correspondence")
    digest = public["messages_sha256"]
    if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SchemaError("public message hash requires lowercase SHA256")
    require_bytes = public.get("require_source_bytes", False)
    if type(require_bytes) is not bool:
        raise SchemaError("require_source_bytes must be Boolean")
    messages = public.get("messages")
    if require_bytes and messages is None:
        raise SchemaError("actual public message bytes required for grounded solving")
    encoded = None
    if messages is not None:
        if type(messages) is not list or len(messages) != 2:
            raise SchemaError("public grounding requires exactly system and user messages")
        for message, role in zip(messages, ("system", "user")):
            _shape(message, {"role", "content"}, "public grounding message")
            if message["role"] != role or type(message["content"]) is not str or not message["content"]:
                raise SchemaError("wrong public grounding message role/content")
        try:
            encoded = [message["content"].encode("utf-8") for message in messages]
        except UnicodeEncodeError as exc:
            raise SchemaError("public grounding message contains invalid Unicode") from exc
        if _public_digest(messages) != digest:
            raise SchemaError("public message bytes do not match parsed message hash")

    def span(value: Any) -> tuple[int, int, int]:
        _shape(value, {"message_index", "utf8_start", "utf8_end"}, "public UTF8 span")
        index, start, end = value["message_index"], value["utf8_start"], value["utf8_end"]
        _int(index, 0, 1, "span message index")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise SchemaError("public UTF8 span requires nonempty integer offsets")
        if encoded is not None:
            if end > len(encoded[index]):
                raise SchemaError("public UTF8 span exceeds actual message bytes")
            try:
                encoded[index][:start].decode("utf-8")
                encoded[index][start:end].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SchemaError("public span splits a UTF8 code point") from exc
        return index, start, end

    for field in ("atom_spans", "record_spans", "framing_spans"):
        if type(public[field]) is not list:
            raise SchemaError(f"public {field} must be an array")
    records = {r["record_id"]: r for r in world["records"]}
    source_spans = {}
    for item in public["record_spans"]:
        _shape(item, {"record_id", "span"}, "public record correspondence")
        identifier = item["record_id"]
        if type(identifier) is not str or identifier not in records or identifier in source_spans:
            raise SchemaError("unknown or duplicate public record span")
        source_spans[identifier] = span(item["span"])
        if source_spans[identifier][0] != 1:
            raise SchemaError("source record must be in the user message")
    if set(source_spans) != set(records):
        raise SchemaError("public record spans do not cover every source record")
    expected = {(identifier, path): value for identifier, record in records.items()
                for path, value in _semantic_leaves({k: v for k, v in record.items()
                                                   if k not in {"record_id", "document_id"}}).items()}
    seen, atom_ids = set(), set()
    rule_ids = {clause["rule_id"] for clause in world["framework"]["rule_clauses"]}
    for item in public["atom_spans"]:
        _shape(item, {"atom_id", "record_id", "field_path", "span", "typed_value_sha256"}, "public atom correspondence")
        _id(item["atom_id"], "public atom ID")
        if type(item["record_id"]) is not str or type(item["field_path"]) is not str:
            raise SchemaError("public atom record/path must be text")
        pair = (item["record_id"], item["field_path"])
        if pair not in expected or pair in seen or item["atom_id"] in atom_ids or item["atom_id"] in rule_ids:
            raise SchemaError("unknown, duplicated, or nonsemantic public atom correspondence")
        seen.add(pair)
        atom_ids.add(item["atom_id"])
        observed = span(item["span"])
        source = source_spans[pair[0]]
        if observed[0] != source[0] or not source[1] <= observed[1] < observed[2] <= source[2]:
            raise SchemaError("public atom lies outside its source record span")
        if item["typed_value_sha256"] != _public_digest(expected[pair]):
            raise SchemaError("public atom typed hash does not match reconstructed semantic field")
    if seen != set(expected):
        raise SchemaError("public atom spans do not exactly cover all semantic source fields")
    framing, framing_seen = _framing_values(world), set()
    for item in public["framing_spans"]:
        _shape(item, {"field_path", "span", "typed_value_sha256"}, "public framing correspondence")
        path = item["field_path"]
        if type(path) is not str or path not in framing or path in framing_seen:
            raise SchemaError("unknown or duplicate public framing field")
        framing_seen.add(path)
        span(item["span"])
        if item["typed_value_sha256"] != _public_digest(framing[path]):
            raise SchemaError("public framing typed hash does not match reconstructed field")
    if framing_seen != set(framing):
        raise SchemaError("public framing spans do not cover all declared query and metadata fields")
    coverage = public.get("coverage_spans")
    if encoded is not None:
        if type(coverage) is not list:
            raise SchemaError("source bytes require the parser's complete coverage partition")
        cursors = [0, 0]
        for item in coverage:
            _shape(item, {"category", "span"}, "public coverage span")
            if type(item["category"]) is not str or not item["category"]:
                raise SchemaError("public coverage requires a category")
            index, start, end = span(item["span"])
            if cursors[index] != start:
                raise SchemaError("public coverage has gaps or overlapping byte ranges")
            cursors[index] = end
        if cursors != [len(value) for value in encoded]:
            raise SchemaError("public coverage does not partition both actual messages")
        # A self-consistent typed hash is not evidence that the bytes assert its value.
        # Reconstruct independently from the actual public source before grounding it.
        from .public_parser import PublicParseError, parse_public
        if "admission_ready" in public and type(public["admission_ready"]) is not bool:
            raise SchemaError("parsed admission readiness must be Boolean")
        try:
            fresh = parse_public(messages, require_frozen=public.get("admission_ready", False))
        except PublicParseError as exc:
            raise SchemaError(f"independent public source reread failed: {exc}") from exc
        reconstructed = {k: v for k, v in world.items() if k != "_public"}
        if _canonical(fresh["world"]) != _canonical(reconstructed):
            raise SchemaError("reconstructed world differs from independent actual-source parse")
        for field in ("atom_spans", "record_spans", "framing_spans", "coverage_spans"):
            if _canonical(fresh[field]) != _canonical(public[field]):
                raise SchemaError(f"supplied {field} differs from independent actual-source correspondence")
        for field in ("parser_identity_sha256", "catalogue_identity_sha256", "catalogue_status", "admission_ready"):
            if field in public and public[field] != fresh[field]:
                raise SchemaError(f"supplied {field} differs from independent source verifier identity")


def apply_program(program: dict[str, Any], state: dict[str, int]) -> dict[str, int]:
    """Execute one guarded AST with guard/action inputs taken from old state."""
    _shape(program, {"program_id", "guard", "action"}, "program")
    _id(program["program_id"], "program_id")
    if program["guard"] not in GUARDS or program["action"] not in ACTIONS:
        raise SchemaError("unknown guarded-program guard/action")
    _shape(state, set(REGISTERS), "state")
    for register in REGISTERS:
        _int(state[register], 0, 1, register)
    old, new = state, dict(state)
    guard = program["guard"]
    if guard != "always" and old[guard[0]] != int(guard[1]):
        return new
    action = program["action"]
    if action.startswith("flip_"):
        register = action[-1]
        new[register] = 1 - old[register]
    elif action.startswith("swap_"):
        left, right = action[-2:]
        new[left], new[right] = old[right], old[left]
    else:
        source, destination = action[-2:]
        new[destination] = old[source]
    return new


def program_truth_table(program: dict[str, Any]) -> tuple[int, ...]:
    return tuple(_code(apply_program(program, dict(zip(REGISTERS, bits))))
                 for bits in itertools.product((0, 1), repeat=3))


def _code(state: dict[str, int]) -> int:
    return 4 * state["m"] + 2 * state["x"] + state["y"]


def _xor(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {register: left[register] ^ right[register] for register in REGISTERS}


class _Validator:
    """Independent mechanics types, bounds, references, and compatibility matrix."""
    def __init__(self, world: dict[str, Any]):
        required = {"world_schema", "mechanism", "entities", "variables", "documents",
                    "records", "framework", "query", "format"}
        if type(world) is not dict or set(world) - {"_public"} != required:
            raise SchemaError("expected reconstructed PublicWorld, no author/private fields")
        if world["world_schema"] != SCHEMA or world["mechanism"] not in MECHANISMS:
            raise SchemaError("unsupported public world schema/mechanism")
        if world["format"] not in {"JSON", "LINE", "TOKEN"}:
            raise SchemaError("unknown answer format")
        for field in ("entities", "variables", "documents", "records"):
            if type(world[field]) is not list:
                raise SchemaError(f"{field} must be an array")
        if len(world["records"]) > 40 or len(world["variables"]) > 2:
            raise SchemaError("finite source enumeration cap exceeded")
        self.entities: dict[str, str] = {}
        labels = set()
        for entity in world["entities"]:
            _shape(entity, {"entity_id", "kind", "public_label"}, "entity")
            _id(entity["entity_id"], "entity_id")
            label = entity["public_label"]
            if (entity["kind"] not in KINDS or type(label) is not str or
                    not re.fullmatch(r"[A-Za-z0-9 -]{1,40}", label)):
                raise SchemaError("invalid entity kind/label")
            if entity["entity_id"] in self.entities or label in labels:
                raise SchemaError("entity IDs and labels must be unique")
            self.entities[entity["entity_id"]] = entity["kind"]
            labels.add(label)
        if sum(kind == "commitment" for kind in self.entities.values()) > 12:
            raise SchemaError("commitment entity cap exceeded")
        self.variables: dict[str, dict[str, Any]] = {}
        for var in world["variables"]:
            _shape(var, {"variable_id", "scalar_type", "choices"}, "variable")
            _id(var["variable_id"], "variable_id")
            if (var["variable_id"] in self.variables or var["variable_id"] in self.entities
                    or var["variable_id"] in labels):
                raise SchemaError("variable identity collision")
            if type(var["choices"]) is not list or not 2 <= len(var["choices"]) <= 3:
                raise SchemaError("variable domains require 2..3 choices")
            if len({_canonical(v) for v in var["choices"]}) != len(var["choices"]):
                raise SchemaError("duplicate finite-domain choices")
            self.variables[var["variable_id"]] = var
            for value in var["choices"]:
                self.scalar(value, var["scalar_type"], "variable choice")
        self.world = world
        self.document_ids = set()
        document_record_ids = []
        for document in world["documents"]:
            _shape(document, {"document_id", "purpose", "title", "dated_day", "record_ids", "style"}, "document")
            _id(document["document_id"], "document_id")
            self.scalar(document["dated_day"], "Day", "document day")
            if document["document_id"] in self.document_ids:
                raise SchemaError("duplicate document declaration")
            self.document_ids.add(document["document_id"])
            if (type(document["title"]) is not str or not 4 <= len(document["title"]) <= 80
                    or document["style"] not in {"controlled_prose", "small_table"}
                    or document["purpose"] not in {"delegation_circular", "booking_minutes", "delivery_slips",
                       "alias_crosswalk", "calibration_notes", "maintenance_log", "room_roster",
                       "policy_amendments", "session_proposals"}):
                raise SchemaError("invalid document metadata")
            if type(document["record_ids"]) is not list:
                raise SchemaError("document record IDs must be an array")
            for record_id in document["record_ids"]:
                _id(record_id, "document record ID")
                document_record_ids.append(record_id)
        if len(set(document_record_ids)) != len(document_record_ids):
            raise SchemaError("duplicate record attachment in documents")
        framework = world["framework"]
        _shape(framework, {"semantic_version", "rule_clauses", "closed_candidate_list", "unknown_defaults",
                           "source_assertions_are_conjunctive"}, "framework")
        if (framework["semantic_version"] != "RIQF-C002-DD-001" or
                framework["closed_candidate_list"] is not True or framework["unknown_defaults"] != "NONE"
                or framework["source_assertions_are_conjunctive"] is not True
                or type(framework["rule_clauses"]) is not list):
            raise SchemaError("invalid finite semantic framework")
        rule_ids, rule_codes = set(), set()
        for clause in framework["rule_clauses"]:
            _shape(clause, {"rule_id", "code", "public_variant"}, "framework clause")
            _id(clause["rule_id"], "framework rule ID")
            _int(clause["public_variant"], 0, 999, "framework variant")
            if clause["code"] not in RULE_CODES or clause["rule_id"] in rule_ids or clause["code"] in rule_codes:
                raise SchemaError("unknown/duplicate framework rule")
            rule_ids.add(clause["rule_id"])
            rule_codes.add(clause["code"])
        self.record_ids: set[str] = set()
        self.events = {r.get("event_id") for r in world["records"] if type(r) is dict
                       and str(r.get("kind", "")).startswith("ledger_")}
        self.policies: dict[str, list[dict[str, Any]]] = {}
        self.plans: dict[str, list[dict[str, Any]]] = {}
        for record in world["records"]:
            if type(record) is dict and record.get("kind") == "base_policy":
                self.policies.setdefault(record.get("policy_id"), []).append(record)
            if type(record) is dict and record.get("kind") == "session_plan":
                self.plans.setdefault(record.get("plan"), []).append(record)
        for record in world["records"]:
            self.record(record)
        if set(document_record_ids) != self.record_ids:
            raise SchemaError("document record attachment does not match public inventory")
        for record in world["records"]:
            if not any(d["document_id"] == record["document_id"] and record["record_id"] in d["record_ids"]
                       for d in world["documents"]):
                raise SchemaError("source record attached to the wrong document")
        self.query(world["query"])
        if len([r for r in world["records"] if r["kind"].startswith("trace_")]) > 10:
            raise SchemaError("test trace event cap exceeded")
        operators = {r["operator"] for r in world["records"] if r["kind"] == "operator_menu"}
        if world["mechanism"] == "guarded_transition_learning" and not 2 <= len(operators) <= 3:
            raise SchemaError("learning mechanism requires two or three operators")
        for r in world["records"]:
            if r["kind"] == "bench_example" and r["operator"] not in operators:
                raise SchemaError("bench observation lacks a declared candidate menu")
            if r["kind"] == "trace_apply":
                for value in self.values(r["operator"]):
                    if value not in operators:
                        raise SchemaError("trace operator lacks a declared candidate menu")
        self.topology()
        _validate_public_metadata(world)

    def topology(self) -> None:
        """Check schema requirements before contradictions can mask a schema error."""
        records, query = self.world["records"], self.world["query"]
        if self.world["mechanism"] == "guarded_transition_learning":
            definitions = {r["station"] for r in records if r["kind"] == "station_initial"}
            for r in records:
                if r["kind"] == "trace_fork":
                    definitions.update(r["destinations"])
                elif r["kind"] == "trace_join":
                    definitions.add(r["destination"])
            if any(station not in definitions for station in query["final_stations"]):
                raise SchemaError("final query station lacks a source definition")
            return
        if self.world["mechanism"] != "policy_feasibility":
            return
        bundles = [r for r in records if r["kind"] == "bundle"]
        affected = set(query["candidate_ids"]) if query["candidate_kind"] == "plan" else {
            p for bundle in bundles if bundle["bundle"] in query["candidate_ids"] for p in bundle["plans"]}
        available = {(r["predicate"], *r["arguments"]) for r in records
                     if r["kind"] in {"fact_bool", "fact_count"}}
        # Conditions are mandatory even for selectors that cannot affect the query.
        # Policy resource keys are conditional and checked only on compatible bindings.
        needed = set()
        policies = [r for r in records if r["kind"] == "base_policy"]
        for r in records:
            if r["kind"] == "exception":
                needed.add((r["condition"]["predicate"], *r["condition"]["arguments"]))
                if r["selector"]["kind"] == "bundle" and not any(
                        b["bundle"] == r["selector"]["bundle"] for b in bundles):
                    raise SchemaError("selected bundle lacks a source definition")
        if any(p["scope"] == "cross_leg" for p in policies):
            for plan_id in affected:
                if not any(len(p["legs"]) == 2 for p in self.plans[plan_id]):
                    raise SchemaError("affected cross_leg candidate requires two ordered legs")
        if needed - available:
            raise SchemaError(f"missing asserted source facts: {sorted(needed - available)}")

    def scalar(self, value: Any, scalar_type: str, name: str) -> None:
        bounds = {"Bit": (0, 1), "Quantity": (0, 12), "SmallCount": (0, 8),
                  "Day": (0, 31), "Tick": (0, 31), "slot": (0, 7)}
        if scalar_type in bounds:
            _int(value, *bounds[scalar_type], name)
        elif scalar_type == "Register":
            if type(value) is not str or value not in REGISTERS:
                raise SchemaError("unknown register")
        elif scalar_type == "EntityId":
            self.entity(value, KINDS, name)
        else:
            raise SchemaError(f"unknown scalar type {scalar_type}")

    def values(self, value: Any) -> list[Any]:
        if type(value) is dict:
            _shape(value, {"var"}, "variable reference")
            if value["var"] not in self.variables:
                raise SchemaError("undeclared variable")
            return self.variables[value["var"]]["choices"]
        return [value]

    def value(self, value: Any, scalar_type: str, name: str) -> None:
        if type(value) is dict:
            self.values(value)
            if self.variables[value["var"]]["scalar_type"] != scalar_type:
                raise SchemaError("variable scalar type does not match its field")
        for choice in self.values(value):
            self.scalar(choice, scalar_type, name)

    def entity(self, value: Any, kinds: set[str], name: str, variable: bool = False) -> None:
        if variable and type(value) is dict:
            self.values(value)
            if self.variables[value["var"]]["scalar_type"] != "EntityId":
                raise SchemaError("entity reference requires EntityId variable")
        for choice in self.values(value) if variable else [value]:
            if type(choice) is not str or self.entities.get(choice) not in kinds:
                raise SchemaError(f"{name} has missing/wrong-kind entity reference")

    def state(self, value: Any) -> None:
        _shape(value, set(REGISTERS), "state")
        for register in REGISTERS:
            self.value(value[register], "Bit", register)

    def arguments(self, predicate: str, arguments: Any) -> None:
        if predicate not in FACT_ARGS or type(arguments) is not list:
            raise SchemaError("unknown fact predicate/arguments")
        signature = FACT_ARGS[predicate]
        if len(arguments) != len(signature):
            raise SchemaError("fact argument arity mismatch")
        for value, kind in zip(arguments, signature):
            if kind == "slot":
                self.scalar(value, "slot", "fact slot")
            else:
                self.entity(value, {kind}, "fact argument")

    def record(self, r: Any) -> None:
        if type(r) is not dict:
            raise SchemaError("record must be an object")
        for field in ("record_id", "document_id"):
            _id(r.get(field), field)
        if r["document_id"] not in self.document_ids:
            raise SchemaError("missing declared source document")
        if r["record_id"] in self.record_ids:
            raise SchemaError("duplicate public record ID")
        self.record_ids.add(r["record_id"])
        kind = r.get("kind")
        common = {"kind", "record_id", "document_id"}
        fields = {
            "alias": {"entity_kind", "left", "right"},
            "membership": {"child", "parent", "from_day", "through_day"},
            "delegation": {"signer", "owner", "from_day", "through_day"},
            "ledger_start": {"event_id", "day", "commitment", "owner", "quantity", "signer"},
            "ledger_amend": {"event_id", "day", "commitment", "predecessor", "quantity", "signer"},
            "ledger_transfer": {"event_id", "day", "commitment", "predecessor", "owner", "signer"},
            "ledger_cancel": {"event_id", "day", "commitment", "predecessor", "signer"},
            "ledger_split": {"event_id", "day", "commitment", "predecessor", "children", "signer"},
            "operator_menu": {"operator", "candidates"}, "bench_example": {"operator", "input", "output"},
            "station_initial": {"station", "state"}, "trace_apply": {"tick", "station", "operator"},
            "trace_splice": {"tick", "source", "source_register", "destination", "destination_register"},
            "trace_fork": {"tick", "source", "destinations"},
            "trace_join": {"tick", "sources", "destination", "operation"},
            "fact_bool": {"predicate", "arguments", "value"}, "fact_count": {"predicate", "arguments", "value"},
            "session_plan": {"plan", "legs"}, "bundle": {"bundle", "plans"},
            "base_policy": {"policy_id", "scope", "predicate"},
            "exception": {"exception_id", "base_policy_id", "selector", "condition"},
            "incidental": {"subject", "attribute", "value"},
        }
        if kind not in fields:
            raise SchemaError("unknown record kind")
        _shape(r, common | fields[kind], f"{kind} record")
        allowed = (kind in {"alias", "membership", "delegation"} or kind.startswith("ledger_"))
        if kind != "incidental":
            if self.world["mechanism"] == "commitment_ledger" and not allowed:
                raise SchemaError("record belongs to another mechanism")
            if self.world["mechanism"] == "guarded_transition_learning" and not (
                    kind in {"operator_menu", "bench_example", "station_initial"} or kind.startswith("trace_")):
                raise SchemaError("record belongs to another mechanism")
            if self.world["mechanism"] == "policy_feasibility" and kind not in {
                    "fact_bool", "fact_count", "session_plan", "bundle", "base_policy", "exception"}:
                raise SchemaError("record belongs to another mechanism")
        if kind == "alias":
            if r["entity_kind"] not in {"commitment", "sponsor", "signer", "group"}:
                raise SchemaError("invalid alias kind")
            for field in ("left", "right"):
                self.entity(r[field], {r["entity_kind"]}, field, True)
        elif kind in {"membership", "delegation"}:
            for field in ("from_day", "through_day"):
                self.value(r[field], "Day", field)
            if kind == "membership":
                self.entity(r["child"], {"sponsor", "group"}, "child", True)
                self.entity(r["parent"], {"group"}, "parent", True)
            else:
                self.entity(r["signer"], {"signer"}, "signer", True)
                self.entity(r["owner"], {"sponsor"}, "owner", True)
        elif kind.startswith("ledger_"):
            _id(r["event_id"], "event_id")
            self.value(r["day"], "Day", "event day")
            self.entity(r["commitment"], {"commitment"}, "commitment")
            self.entity(r["signer"], {"signer"}, "signer", True)
            if "predecessor" in r:
                _id(r["predecessor"], "predecessor")
                if r["predecessor"] not in self.events:
                    raise SchemaError("missing declared event predecessor")
            if "owner" in r:
                self.entity(r["owner"], {"sponsor"}, "owner", True)
            if "quantity" in r:
                self.value(r["quantity"], "Quantity", "quantity")
            if kind == "ledger_split":
                if type(r["children"]) is not list or len(r["children"]) != 2:
                    raise SchemaError("split requires exactly two children")
                for child in r["children"]:
                    _shape(child, {"commitment", "owner", "quantity"}, "child")
                    self.entity(child["commitment"], {"commitment"}, "child commitment")
                    self.entity(child["owner"], {"sponsor"}, "child owner", True)
                    self.value(child["quantity"], "Quantity", "child quantity")
        elif kind == "operator_menu":
            self.entity(r["operator"], {"operator"}, "operator")
            if type(r["candidates"]) is not list or not 3 <= len(r["candidates"]) <= 4:
                raise SchemaError("operator menu needs 3..4 candidates")
            tables = [program_truth_table(p) for p in r["candidates"]]
            if len(set(tables)) != len(tables) or len({p["program_id"] for p in r["candidates"]}) != len(tables):
                raise SchemaError("candidate programs must have distinct IDs and truth tables")
        elif kind == "bench_example":
            self.entity(r["operator"], {"operator"}, "operator")
            self.state(r["input"])
            self.state(r["output"])
        elif kind == "station_initial":
            self.entity(r["station"], {"station"}, "station")
            self.state(r["state"])
        elif kind.startswith("trace_"):
            self.value(r["tick"], "Tick", "trace tick")
            for field in ("station", "source", "destination"):
                if field in r:
                    self.entity(r[field], {"station"}, field)
            if kind == "trace_apply":
                self.entity(r["operator"], {"operator"}, "operator", True)
            if kind == "trace_splice":
                for field in ("source_register", "destination_register"):
                    self.value(r[field], "Register", field)
            if kind in {"trace_fork", "trace_join"}:
                field = "destinations" if kind == "trace_fork" else "sources"
                if type(r[field]) is not list or len(r[field]) != 2 or len(set(r[field])) != 2:
                    raise SchemaError("fork/join requires two distinct stations")
                for station in r[field]:
                    self.entity(station, {"station"}, field)
            if kind == "trace_join" and r["operation"] != "xor":
                raise SchemaError("join operation must be xor")
        elif kind in {"fact_bool", "fact_count"}:
            self.arguments(r["predicate"], r["arguments"])
            boolean = r["predicate"] in BOOL_FACTS
            if boolean != (kind == "fact_bool"):
                raise SchemaError("fact record kind/result type mismatch")
            self.value(r["value"], "Bit" if boolean else "SmallCount", "fact value")
        elif kind == "session_plan":
            self.entity(r["plan"], {"plan"}, "plan")
            if type(r["legs"]) is not list or not 1 <= len(r["legs"]) <= 2:
                raise SchemaError("plan requires one or two ordered legs")
            leg_ids = set()
            for leg in r["legs"]:
                _shape(leg, {"leg_id", "slot", "room", "kit", "leader", "people", "group", "kit_demand"}, "leg")
                _id(leg["leg_id"], "leg_id")
                if leg["leg_id"] in leg_ids:
                    raise SchemaError("duplicate leg ID in plan")
                leg_ids.add(leg["leg_id"])
                _int(leg["slot"], 0, 7, "slot")
                _int(leg["kit_demand"], 1, 4, "kit demand")
                for field, entity_kind in (("room", "room"), ("kit", "kit"), ("leader", "person"), ("group", "group")):
                    self.entity(leg[field], {entity_kind}, field)
                if type(leg["people"]) is not list or not 1 <= len(leg["people"]) <= 4 or len(set(leg["people"])) != len(leg["people"]):
                    raise SchemaError("people must be a duplicate-free 1..4 set")
                for person in leg["people"]:
                    self.entity(person, {"person"}, "person")
        elif kind == "bundle":
            self.entity(r["bundle"], {"bundle"}, "bundle")
            if type(r["plans"]) is not list or len(r["plans"]) != 2 or len(set(r["plans"])) != 2:
                raise SchemaError("bundle needs two distinct plans")
            for plan in r["plans"]:
                self.entity(plan, {"plan"}, "member plan")
                if plan not in self.plans:
                    raise SchemaError("bundle member lacks a source plan")
        elif kind == "base_policy":
            _id(r["policy_id"], "policy_id")
            scope, pred = r["scope"], r["predicate"]
            if type(pred) is not dict or pred.get("code") not in SCOPES.get(scope, set()):
                raise SchemaError("invalid policy scope/predicate combination")
            _shape(pred, {"code", "credential"} if pred["code"] == "leader_credential" else {"code"}, "policy predicate")
            if "credential" in pred:
                self.entity(pred["credential"], {"credential"}, "credential")
        elif kind == "exception":
            _id(r["exception_id"], "exception_id")
            _id(r["base_policy_id"], "base_policy_id")
            if r["base_policy_id"] not in self.policies:
                raise SchemaError("dangling exception base policy")
            selector, condition = r["selector"], r["condition"]
            if type(selector) is not dict or selector.get("kind") not in {"plan", "leg", "group", "bundle"}:
                raise SchemaError("invalid exception selector")
            selkind = selector["kind"]
            _shape(selector, {"kind", "plan", "leg_id"} if selkind == "leg" else {"kind", selkind}, "selector")
            if not any(selkind in SELECTORS.get(policy.get("scope"), set())
                       for policy in self.policies[r["base_policy_id"]]):
                raise SchemaError("invalid policy scope/exception selector pair")
            if selkind == "leg":
                self.entity(selector["plan"], {"plan"}, "selected plan")
                _id(selector["leg_id"], "selected leg")
                if selector["plan"] not in self.plans or not any(
                    leg.get("leg_id") == selector["leg_id"] for p in self.plans[selector["plan"]] for leg in p.get("legs", [])):
                    raise SchemaError("missing selected source leg")
            else:
                self.entity(selector[selkind], {selkind}, "selected entity")
                if selkind == "plan" and selector["plan"] not in self.plans:
                    raise SchemaError("missing selected source plan")
            _shape(condition, {"predicate", "arguments", "comparison", "value"}, "condition")
            self.arguments(condition["predicate"], condition["arguments"])
            boolean = condition["predicate"] in BOOL_FACTS
            if condition["comparison"] not in ({"eq"} if boolean else {"eq", "ge"}):
                raise SchemaError("invalid fact condition comparison")
            self.scalar(condition["value"], "Bit" if boolean else "SmallCount", "condition value")
        else:
            self.entity(r["subject"], KINDS, "incidental subject")
            if (r["attribute"] not in INCIDENTAL_VALUES or type(r["value"]) is not str
                    or r["value"] not in INCIDENTAL_VALUES[r["attribute"]]):
                raise SchemaError("invalid incidental fact")

    def query(self, q: Any) -> None:
        mechanism = self.world["mechanism"]
        if type(q) is not dict or q.get("mechanism") != mechanism:
            raise SchemaError("query/world mechanism mismatch")
        if mechanism == "commitment_ledger":
            _shape(q, {"mechanism", "cutoff", "group_label"}, "query")
            self.scalar(q["cutoff"], "Day", "cutoff")
            self.entity(q["group_label"], {"group"}, "queried group", True)
        elif mechanism == "guarded_transition_learning":
            _shape(q, {"mechanism", "final_stations", "combine", "encoding"}, "query")
            if q["combine"] not in {"single", "xor"} or q["encoding"] != "4m+2x+y":
                raise SchemaError("invalid state query encoding/combine")
            if type(q["final_stations"]) is not list or len(q["final_stations"]) != (1 if q["combine"] == "single" else 2):
                raise SchemaError("state query station count mismatch")
            if len(set(q["final_stations"])) != len(q["final_stations"]):
                raise SchemaError("duplicate final station")
            for station in q["final_stations"]:
                self.entity(station, {"station"}, "final station")
        else:
            _shape(q, {"mechanism", "candidate_ids", "candidate_kind", "operation"}, "query")
            if q["candidate_kind"] not in {"plan", "bundle"} or q["operation"] != "count_feasible":
                raise SchemaError("invalid feasibility query")
            if type(q["candidate_ids"]) is not list or not 3 <= len(q["candidate_ids"]) <= 12 or len(set(q["candidate_ids"])) != len(q["candidate_ids"]):
                raise SchemaError("query needs complete duplicate-free 3..12 candidate list")
            sources = {r[q["candidate_kind"]] for r in self.world["records"]
                       if r["kind"] == ("session_plan" if q["candidate_kind"] == "plan" else "bundle")}
            for candidate in q["candidate_ids"]:
                self.entity(candidate, {q["candidate_kind"]}, "query candidate")
                if candidate not in sources:
                    raise SchemaError("query candidate lacks source definition")
            if any(r["kind"] == "base_policy" and r["scope"] == "each_bundle" for r in self.world["records"]) and q["candidate_kind"] != "bundle":
                raise SchemaError("bundle policy requires bundle query")


def _resolve(value: Any, bindings: dict[str, Any]) -> Any:
    if type(value) is dict:
        if set(value) == {"var"}:
            return bindings[value["var"]]
        return {key: _resolve(item, bindings) for key, item in value.items()}
    if type(value) is list:
        return [_resolve(item, bindings) for item in value]
    return value


def _semantic(record: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in record.items() if k not in {"record_id", "document_id"}}
    if record["kind"] == "operator_menu":
        result["candidates"] = sorted(result["candidates"], key=lambda p: p["program_id"])
    elif record["kind"] == "bundle":
        result["plans"] = sorted(result["plans"])
    elif record["kind"] == "session_plan":
        result["legs"] = [dict(leg, people=sorted(leg["people"])) for leg in result["legs"]]
    elif record["kind"] == "trace_join":
        result["sources"] = sorted(result["sources"])
    return result


def _merge(records: list[dict[str, Any]], key) -> dict[Any, dict[str, Any]]:
    """Conjoin definitions under one source identity; identical support is redundant."""
    result: dict[Any, dict[str, Any]] = {}
    for record in records:
        identity = key(record)
        if identity in result and _semantic(result[identity]) != _semantic(record):
            raise _Incompatible(f"conflicting definitions for {identity}")
        result.setdefault(identity, record)
    return result


class _Ledger:
    def __init__(self, world: dict[str, Any]):
        public = world.get("_public", {})
        self.atom_map: dict[str, list[str]] = {}
        self.framing_atoms: list[dict[str, Any]] = []
        self.identity_ids = _public_identity_ids(world)
        atom_spans = public.get("atom_spans", []) if type(public) is dict else []
        for atom in atom_spans:
            if type(atom) is dict and "record_id" in atom and "atom_id" in atom:
                self.atom_map.setdefault(atom["record_id"], []).append(atom["atom_id"])
        framing_ids = _framing_atom_ids(world) if "_public" in world else {}
        for framing in public.get("framing_spans", []) if type(public) is dict else []:
            atom = dict(framing, atom_id=framing_ids[framing["field_path"]])
            self.framing_atoms.append(atom)
            self.atom_map.setdefault("@framing", []).append(atom["atom_id"])
        # Only mechanics fixtures without public metadata use synthetic identifiers.
        for record in world["records"]:
            if record["record_id"] not in self.atom_map:
                if "_public" in world:
                    raise SchemaError("public source correspondence cannot use synthetic fallback")
                self.atom_map[record["record_id"]] = ["a" + _digest([record["record_id"], path])[:24]
                    for path in _semantic_leaves({k: v for k, v in record.items()
                                                 if k not in {"record_id", "document_id"}})]
        self.rules = {r["code"]: r["rule_id"] for r in world["framework"].get("rule_clauses", [])}
        self.steps: list[dict[str, Any]] = []
        self.used_rule_ids: set[str] = set()

    def add(self, operation: str, value: Any, records=(), rules=(), extra_inputs=()) -> str:
        occupied = ({atom for atoms in self.atom_map.values() for atom in atoms} |
                    set(self.rules.values()) | self.identity_ids)
        occupied.update(step["step_id"] for step in self.steps)
        number = len(self.steps) + 1
        while f"s{number:04d}" in occupied:
            number += 1
        step_id = f"s{number:04d}"
        # Conservative transitive dependencies; no local-minimality claim is made.
        inputs = {step["step_id"] for step in self.steps}
        inputs.update(extra_inputs)
        for record in records:
            inputs.update(self.atom_map.get(record["record_id"], []))
        used_rules = {self.rules[rule] for rule in rules if rule in self.rules}
        inputs.update(used_rules)
        self.used_rule_ids.update(used_rules)
        self.steps.append({"step_id": step_id, "operation": operation,
                           "inputs": sorted(inputs), "value": value})
        return step_id


class _Union:
    def __init__(self, entities: list[dict[str, Any]]):
        self.parents = {e["entity_id"]: e["entity_id"] for e in entities}

    def root(self, entity: str) -> str:
        while self.parents[entity] != entity:
            entity = self.parents[entity]
        return entity

    def join(self, left: str, right: str) -> None:
        left, right = self.root(left), self.root(right)
        self.parents[max(left, right)] = min(left, right)


def _check_incidentals(world: dict[str, Any], union: _Union | None = None) -> None:
    records = [dict(r, subject=union.root(r["subject"]) if union else r["subject"])
               for r in world["records"] if r["kind"] == "incidental"]
    _merge(records, lambda r: (r["subject"], r["attribute"]))


def _solve_a(world: dict[str, Any], ledger: _Ledger) -> int:
    records, query = world["records"], world["query"]
    union = _Union(world["entities"])
    for record in records:
        if record["kind"] == "alias":
            union.join(record["left"], record["right"])
            ledger.add("alias_component", sorted({union.root(record["left"]), union.root(record["right"])}), [record], ["ENTITY_IDENTITY"])
    _check_incidentals(world, union)
    memberships = [r for r in records if r["kind"] == "membership"]
    delegations = [r for r in records if r["kind"] == "delegation"]
    for r in memberships + delegations:
        if r["from_day"] > r["through_day"]:
            raise _Incompatible("reversed dated interval")
    cutoff = query["cutoff"]
    edges: dict[str, set[str]] = {}
    for r in memberships:
        if r["from_day"] <= cutoff <= r["through_day"]:
            edges.setdefault(union.root(r["child"]), set()).add(union.root(r["parent"]))
    visiting, done = set(), set()

    def visit(node: str) -> None:
        if node in visiting:
            raise _Incompatible("active membership cycle")
        if node in done:
            return
        visiting.add(node)
        for parent in edges.get(node, set()):
            visit(parent)
        visiting.remove(node)
        done.add(node)
    for node in edges:
        visit(node)

    def reaches(owner: str) -> bool:
        target = union.root(query["group_label"])
        pending, seen = [union.root(owner)], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(edges.get(node, set()))
        return False

    def authorized(event: dict[str, Any], owner: str) -> bool:
        matches = [r for r in delegations if union.root(r["signer"]) == union.root(event["signer"])
                   and union.root(r["owner"]) == union.root(owner)
                   and r["from_day"] <= event["day"] <= r["through_day"]]
        ledger.add("authorize", bool(matches), [event] + matches, ["A_AUTHORITY"])
        return bool(matches)

    # Same asserted event ID is conjunctive, not a hidden printed-order choice.
    event_records = []
    for record in records:
        if not record["kind"].startswith("ledger_"):
            continue
        canonical = dict(record)
        for field in ("commitment", "owner", "signer"):
            if field in canonical:
                canonical[field] = union.root(canonical[field])
        if "children" in canonical:
            canonical["children"] = [dict(child, commitment=union.root(child["commitment"]), owner=union.root(child["owner"]))
                                     for child in canonical["children"]]
        event_records.append(canonical)
    events = list(_merge(event_records, lambda r: r["event_id"]).values())
    created: set[str] = set()
    current: dict[str, dict[str, Any]] = {}
    for day, group in itertools.groupby(sorted(events, key=lambda r: r["day"]), lambda r: r["day"]):
        if day > cutoff:
            break
        today = list(group)
        # All successors read heads at the beginning of this effective day.
        successors: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in today:
            commitment = union.root(event["commitment"])
            if event["kind"] == "ledger_start":
                continue
            head = current.get(commitment)
            if (head is None or not head["open"] or event["predecessor"] != head["event"]
                    or day <= head["day"]):
                ledger.add("ledger_transition", [], [event], ["A_TIME"])
                continue
            if authorized(event, head["owner"]):
                successors.setdefault((commitment, head["event"]), []).append(event)
        single_successor = not any(len(group) > 1 for group in successors.values())
        ledger.add("ledger_transition", single_successor, today, ["A_SINGLE_SUCCESSOR"])
        if not single_successor:
            raise _Incompatible("same-day authorized single-head fork")
        creations = [e for e in today if e["kind"] == "ledger_start" and authorized(e, e["owner"])]
        new_components = [union.root(e["commitment"]) for e in creations]
        for group in successors.values():
            event = group[0]
            if event["kind"] == "ledger_split":
                new_components.extend(union.root(child["commitment"]) for child in event["children"])
        if len(set(new_components)) != len(new_components) or any(c in created for c in new_components):
            raise _Incompatible("duplicate creation of alias-equivalent commitment")
        created.update(new_components)
        for event in creations:
            current[union.root(event["commitment"])] = {"event": event["event_id"], "day": day,
                "owner": union.root(event["owner"]), "quantity": event["quantity"], "open": True}
            ledger.add("ledger_transition", event["quantity"], [event], ["A_TRANSITIONS"])
        for (commitment, _), group in successors.items():
            event, head = group[0], dict(current[commitment])
            kind = event["kind"]
            head.update(event=event["event_id"], day=day)
            if kind == "ledger_amend":
                head["quantity"] = event["quantity"]
            elif kind == "ledger_transfer":
                head["owner"] = union.root(event["owner"])
            elif kind in {"ledger_cancel", "ledger_split"}:
                head["open"] = False
            current[commitment] = head
            if kind == "ledger_split":
                for child in event["children"]:
                    current[union.root(child["commitment"])] = {"event": event["event_id"], "day": day,
                        "owner": union.root(child["owner"]), "quantity": child["quantity"], "open": True}
            ledger.add("ledger_transition", head["quantity"] if head["open"] else 0, [event], ["A_TRANSITIONS"])
    result = 0
    for commitment, head in sorted(current.items()):
        if head["open"] and reaches(head["owner"]):
            result += head["quantity"]
            ledger.add("sum_term", head["quantity"], memberships, ["A_RESULT", "A_MEMBERSHIP"])
    _int(result, 0, 144, "ledger result")
    return result


def _b_candidates(world: dict[str, Any], ledger: _Ledger) -> dict[str, list[dict[str, Any]]]:
    menus = _merge([r for r in world["records"] if r["kind"] == "operator_menu"], lambda r: r["operator"])
    observations = [r for r in world["records"] if r["kind"] == "bench_example"]
    remaining = {}
    for operator, menu in sorted(menus.items()):
        examples = [r for r in observations if r["operator"] == operator]
        remaining[operator] = list(menu["candidates"])
        for index, example in enumerate(examples):
            remaining[operator] = [program for program in remaining[operator]
                                   if apply_program(program, example["input"]) == example["output"]]
            ledger.add("candidate_elimination", [p["program_id"] for p in remaining[operator]],
                       [menu] + examples[:index + 1], ["B_INFER"])
    return remaining


def _solve_b(world: dict[str, Any], programs: dict[str, dict[str, Any]], ledger: _Ledger) -> int:
    records, query = world["records"], world["query"]
    initial = _merge([r for r in records if r["kind"] == "station_initial"], lambda r: r["station"])
    stations = {station: dict(r["state"]) for station, r in initial.items()}
    for station, initial_record in sorted(initial.items()):
        ledger.add("station_transition", dict(stations[station]), [initial_record], ["B_TRACE"])
    # Distinct events at one resolved tick violate global unique chronology.
    trace = _merge([r for r in records if r["kind"].startswith("trace_")], lambda r: r["tick"])
    for tick, event in sorted(trace.items()):
        kind = event["kind"]
        try:
            if kind == "trace_apply":
                station = event["station"]
                stations[station] = apply_program(programs[event["operator"]], stations[station])
                ledger.add("station_transition", dict(stations[station]), [event], ["B_TRACE", "B_PROGRAM"])
            elif kind == "trace_splice":
                source_value = stations[event["source"]][event["source_register"]]
                stations[event["destination"]][event["destination_register"]] = source_value
                ledger.add("splice", dict(stations[event["destination"]]), [event], ["B_TRACE"])
            elif kind == "trace_fork":
                old = dict(stations[event["source"]])
                if any(destination in stations for destination in event["destinations"]):
                    raise _Incompatible("fork destination is not fresh")
                for destination in event["destinations"]:
                    stations[destination] = dict(old)
                ledger.add("station_transition", old, [event], ["B_TRACE"])
            else:
                if event["destination"] in stations:
                    raise _Incompatible("join destination is not fresh")
                left, right = event["sources"]
                stations[event["destination"]] = _xor(stations[left], stations[right])
                ledger.add("xor_merge", dict(stations[event["destination"]]), [event], ["B_TRACE", "B_XOR"])
        except KeyError as exc:
            raise _Incompatible("trace references station before definition") from exc
    try:
        result = dict(stations[query["final_stations"][0]])
        if query["combine"] == "xor":
            result = _xor(result, stations[query["final_stations"][1]])
            ledger.add("xor_merge", dict(result), rules=["B_XOR"])
    except KeyError as exc:
        raise SchemaError("final query station has no trace/initial definition") from exc
    encoded = _code(result)
    ledger.add("answer_set", [encoded], rules=["B_RESULT"])
    return encoded


def _solve_c(world: dict[str, Any], ledger: _Ledger) -> int:
    records, query = world["records"], world["query"]
    # Every source assertion is conjunctive, including candidate-associated facts.
    facts: dict[tuple[Any, ...], int] = {}
    fact_records: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in records:
        if r["kind"] in {"fact_bool", "fact_count"}:
            key = (r["predicate"], *r["arguments"])
            if key in facts and facts[key] != r["value"]:
                ledger.add("predicate", False, fact_records[key] + [r], ["C_FACTS"])
                raise _Incompatible("conflicting source fact values")
            facts[key] = r["value"]
            fact_records.setdefault(key, []).append(r)
    plans = _merge([r for r in records if r["kind"] == "session_plan"], lambda r: r["plan"])
    bundles = _merge([r for r in records if r["kind"] == "bundle"], lambda r: r["bundle"])
    policies = _merge([r for r in records if r["kind"] == "base_policy"], lambda r: r["policy_id"])
    exceptions = _merge([r for r in records if r["kind"] == "exception"], lambda r: r["exception_id"])
    affected = set(query["candidate_ids"]) if query["candidate_kind"] == "plan" else {
        plan for bundle_id in query["candidate_ids"] for plan in bundles[bundle_id]["plans"]}

    def fact(predicate: str, *args: Any) -> int:
        key = (predicate, *args)
        if key not in facts:
            raise _MissingPremise({key})
        ledger.add("predicate", facts[key], fact_records[key], ["C_FACTS"])
        return facts[key]

    conditions = {}
    for identity, exception in exceptions.items():
        condition = exception["condition"]
        observed = fact(condition["predicate"], *condition["arguments"])
        conditions[identity] = (observed == condition["value"] if condition["comparison"] == "eq"
                                else observed >= condition["value"])
        ledger.add("exemption", conditions[identity], [exception], ["C_EXEMPTION"])

    def exempt(policy: dict[str, Any], plan: str | None = None,
               leg: dict[str, Any] | None = None, bundle: str | None = None) -> bool:
        for identity, exception in exceptions.items():
            if exception["base_policy_id"] != policy["policy_id"] or not conditions[identity]:
                continue
            selector = exception["selector"]
            matches = ((selector["kind"] == "plan" and selector["plan"] == plan) or
                       (selector["kind"] == "leg" and selector["plan"] == plan and leg is not None
                        and selector["leg_id"] == leg["leg_id"]) or
                       (selector["kind"] == "group" and leg is not None and selector["group"] == leg["group"]) or
                       (selector["kind"] == "bundle" and selector["bundle"] == bundle))
            if matches:
                ledger.add("exemption", True, [exception, policy], ["C_EXEMPTION"])
                return True
        return False

    def leg_predicate(policy: dict[str, Any], leg: dict[str, Any]) -> bool | _MissingPremise:
        code = policy["predicate"]["code"]
        if code == "capacity_ok":
            return fact("capacity", leg["room"]) >= len(leg["people"])
        if code in {"room_available", "kit_available"}:
            return fact(code, leg["room"] if code == "room_available" else leg["kit"], leg["slot"]) == 1
        if code == "kit_units_ok":
            return fact("kit_units", leg["kit"], leg["slot"]) >= leg["kit_demand"]
        if code == "leader_credential":
            return fact("credential", leg["leader"], policy["predicate"]["credential"]) == 1
        return _conjunction([_policy_check(lambda person=person:
                            fact("member", person, leg["group"]) == 1)
                            for person in leg["people"]])

    feasibility: dict[str, bool | _MissingPremise] = {}
    for plan_id, plan in plans.items():
        if plan_id not in affected:
            continue
        checks = []
        for policy in policies.values():
            if policy["scope"] == "each_leg":
                for leg in plan["legs"]:
                    checks.append(True if exempt(policy, plan=plan_id, leg=leg) else
                                  _policy_check(lambda: leg_predicate(policy, leg)))
            elif policy["scope"] == "cross_leg":
                if len(plan["legs"]) != 2:
                    raise SchemaError("cross_leg policy requires two ordered legs")
                left, right = plan["legs"]
                code = policy["predicate"]["code"]
                checks.append(True if exempt(policy, plan=plan_id) else {
                    "slots_increase": left["slot"] < right["slot"],
                    "rooms_differ": left["room"] != right["room"],
                    "groups_equal": left["group"] == right["group"]}[code])
        feasibility[plan_id] = _conjunction(checks)
        if not isinstance(feasibility[plan_id], _MissingPremise):
            applied_rules = (["C_LOCAL"] if any(p["scope"] == "each_leg" for p in policies.values()) else [])
            if any(p["scope"] == "cross_leg" for p in policies.values()):
                applied_rules.append("C_CROSS_LEG")
            ledger.add("candidate_decision", feasibility[plan_id], [plan] + list(policies.values()), applied_rules)
    for bundle_id, bundle in bundles.items():
        if query["candidate_kind"] != "bundle" or bundle_id not in query["candidate_ids"]:
            continue
        checks = [feasibility[plan] for plan in bundle["plans"]]
        if any(check is False for check in checks):
            feasibility[bundle_id] = False
            ledger.add("candidate_decision", False, [bundle] + list(policies.values()), ["C_BUNDLE"])
            continue
        for policy in policies.values():
            if policy["scope"] != "each_bundle":
                continue
            if exempt(policy, bundle=bundle_id):
                checks.append(True)
                continue
            demands: dict[tuple[str, int], int] = {}
            for plan_id in bundle["plans"]:
                for leg in plans[plan_id]["legs"]:
                    key = (leg["kit"], leg["slot"])
                    demands[key] = demands.get(key, 0) + leg["kit_demand"]
            quota_checks = []
            for (kit, slot), demand in sorted(demands.items()):
                quota_checks.append(_policy_check(lambda kit=kit, slot=slot, demand=demand:
                                    demand <= fact("kit_units", kit, slot)))
                ledger.add("quota_sum", demand, [bundle, policy] + [plans[p] for p in bundle["plans"]], ["C_BUNDLE"])
            checks.append(_conjunction(quota_checks))
        feasibility[bundle_id] = _conjunction(checks)
        if not isinstance(feasibility[bundle_id], _MissingPremise):
            ledger.add("candidate_decision", feasibility[bundle_id], [bundle] + list(policies.values()), ["C_BUNDLE"])
    for candidate in query["candidate_ids"]:
        truth = feasibility[candidate]
        if isinstance(truth, _MissingPremise):
            raise SchemaError(f"unmasked missing asserted facts for {candidate}: {sorted(truth.keys)}; unknown defaults forbidden")
    result = sum(feasibility[candidate] is True for candidate in query["candidate_ids"])
    ledger.add("candidate_decision", result, [plans[c] if query["candidate_kind"] == "plan" else bundles[c]
               for c in query["candidate_ids"]], ["C_RESULT"])
    return result


def solve_public_world(world: dict[str, Any]) -> dict[str, Any]:
    """Enumerate all public finite source/program assignments and return Solution.

    Schema/parse failures raise SchemaError. They are never labeled INCONSISTENT.
    The witness ledger establishes finite compatible-world coverage; support is
    nonminimal and per-atom controls must independently call this API again.
    """
    try:
        validator = _Validator(world)
    except (KeyError, TypeError) as exc:
        raise SchemaError(f"malformed public mechanics: {exc}") from exc
    variable_ids = sorted(validator.variables)
    domains = [validator.variables[v]["choices"] for v in variable_ids]
    answers: set[int] = set()
    compatible_count = 0
    witnesses: dict[int, dict[str, Any]] = {}
    exemplar_steps: list[dict[str, Any]] = []
    learned_assignments: set[tuple[tuple[str, str], ...]] = set()
    operator_ids: list[str] = []
    rejected = []
    enumeration = []
    executed_rule_ids: set[str] = set()
    source_binding_count = 1
    for domain in domains:
        source_binding_count *= len(domain)
    raw_menu_sizes = {r["operator"]: len(r["candidates"]) for r in world["records"] if r["kind"] == "operator_menu"}
    raw_program_count = 1
    for size in raw_menu_sizes.values():
        raw_program_count *= size
    for choices in itertools.product(*domains):
        bindings = dict(zip(variable_ids, choices))
        resolved = _resolve({k: v for k, v in world.items() if k != "_public"}, bindings)
        if "_public" in world:
            resolved["_public"] = world["_public"]
        base_ledger = _Ledger(resolved)
        try:
            if world["mechanism"] != "commitment_ledger":
                _check_incidentals(resolved)
            if world["mechanism"] == "guarded_transition_learning":
                candidates = _b_candidates(resolved, base_ledger)
                operator_ids = sorted(candidates)
                assignments = itertools.product(*(candidates[o] for o in operator_ids))
                if not all(candidates.values()):
                    raise _Incompatible("no candidate matches every calibration observation")
            else:
                operator_ids, assignments = [], [()]
            for programs in assignments:
                ledger = _Ledger(resolved)
                ledger.steps = list(base_ledger.steps)
                ledger.used_rule_ids.update(base_ledger.used_rule_ids)
                learned = dict(zip(operator_ids, programs))
                assignment_key = tuple((operator, learned[operator]["program_id"]) for operator in operator_ids)
                try:
                    if world["mechanism"] == "commitment_ledger":
                        result = _solve_a(resolved, ledger)
                    elif world["mechanism"] == "guarded_transition_learning":
                        result = _solve_b(resolved, learned, ledger)
                    else:
                        result = _solve_c(resolved, ledger)
                    _int(result, 0, 7 if world["mechanism"] == "guarded_transition_learning" else
                         12 if world["mechanism"] == "policy_feasibility" else 144, "scalar answer")
                    compatible_count += 1
                    answers.add(result)
                    learned_assignments.add(assignment_key)
                    witness = {"bindings": [{"variable_id": v, "value": bindings[v]} for v in variable_ids],
                               "learned_programs": [{"operator_id": o, "program_id": p} for o, p in assignment_key],
                               "result": result, "semantic_world_sha256": _digest({"world": {k: v for k, v in resolved.items() if k != "_public"}, "programs": learned})}
                    if result not in witnesses:
                        witnesses[result] = witness
                        if not exemplar_steps:
                            exemplar_steps = ledger.steps
                    enumeration.append({"bindings": witness["bindings"], "learned_programs": witness["learned_programs"],
                                        "result": result, "compatible": True})
                except _Incompatible as exc:
                    rejected.append(str(exc))
                    enumeration.append({"bindings": [{"variable_id": v, "value": bindings[v]} for v in variable_ids],
                                        "learned_programs": [{"operator_id": o, "program_id": p} for o, p in assignment_key],
                                        "compatible": False, "reason": str(exc)})
                finally:
                    executed_rule_ids.update(ledger.used_rule_ids)
        except _Incompatible as exc:
            rejected.append(str(exc))
            enumeration.append({"bindings": [{"variable_id": v, "value": bindings[v]} for v in variable_ids],
                                "learned_programs": [], "compatible": False, "reason": str(exc)})
        finally:
            executed_rule_ids.update(base_ledger.used_rule_ids)
    status = "INCONSISTENT" if not compatible_count else "DETERMINATE" if len(answers) == 1 else "UNDERDETERMINED"
    support_ledger = _Ledger(world)
    support = sorted({atom for atoms in support_ledger.atom_map.values() for atom in atoms})
    support_ledger.steps = exemplar_steps
    support_ledger.add("answer_set", sorted(answers), world["records"], ["FINITE_WORLDS", "STATUS"],
                       support + sorted(executed_rule_ids))
    executed_rule_ids.update(support_ledger.used_rule_ids)
    exemplar_steps = support_ledger.steps
    operative_rules = sorted(executed_rule_ids)
    public = world.get("_public", {})
    source_bytes_verified = bool(public.get("messages")) if type(public) is dict else False
    grounding_mode = ("REPARSED_PUBLIC_SOURCE_WITH_INDEPENDENT_PARSER" if source_bytes_verified else
                      "STRUCTURAL_METADATA_ONLY_NOUTF8REREAD" if "_public" in world else
                      "EXPOSED_MECHANICS_ONLY_NO_PUBLIC_SPANS")
    return {
        "status": status, "possible_answers": sorted(answers), "compatible_world_count": compatible_count,
        "certificate": {"sufficient_atom_ids": support if compatible_count else [],
                        "operative_rule_ids": operative_rules, "step_values": exemplar_steps,
                        "compatible_witnesses": [witnesses[a] for a in sorted(witnesses)],
                        "inconsistency_core_atom_ids": [], "alternative_support_sets": [],
                        "minimality_scope": "LOCAL_SINGLE_ATOM_ABLATIONS_NOT_ALL_GLOBAL_PROOFS"},
        "solver_identity_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_messages_sha256": public.get("messages_sha256", "") if type(public) is dict else "",
        "public_grounding": {"mode": grounding_mode, "source_bytes_verified": source_bytes_verified,
                             "admission_grounding": source_bytes_verified,
                             "admission_ready": source_bytes_verified and public.get("admission_ready", False)},
        "public_framing_atom_map": support_ledger.framing_atoms,
        "fixed_background_rule_ids": sorted(support_ledger.rules.values()),
        "enumeration": {"source_binding_count": source_binding_count,
                        "joint_program_assignment_count_before_calibration": raw_program_count if raw_menu_sizes else None,
                        "source_program_combinations_before_calibration": source_binding_count * raw_program_count,
                        "assignments": enumeration, "rejection_reasons": sorted(set(rejected)),
                        "learned_program_assignment_count": len(learned_assignments) if raw_menu_sizes else None,
                        "programs_uniquely_identified": len(learned_assignments) == 1 if raw_menu_sizes else None},
        "limitations": ["Nonminimal full-source support and conservative transitive step dependencies; no minimal unsat core claimed.",
                        "Local atom/rule dependency checks require separate registered interventions.",
                        "Typed mechanics fixtures do not establish public-byte parsing or row admission.",
                        "Declared fixed background rules are not evidence of executed, answer-sensitive, or minimal rules.",
                        "Operative rules union traced execution across bindings; some primitive compatibility checks have no per-rule typed step.",
                        grounding_mode],
    }


def solve_parsed_public(parsed: dict[str, Any], *, messages: list[dict[str, str]] | None = None,
                        require_source_bytes: bool = False) -> dict[str, Any]:
    """Consume complete ParsedPublic; production grounding also requires source bytes."""
    if (type(parsed) is not dict or "world" not in parsed or
            type(parsed.get("unparsed_assertions")) is not list or parsed["unparsed_assertions"] != []):
        raise SchemaError("expected successful ParsedPublic")
    if type(parsed["world"]) is not dict or "_public" in parsed["world"]:
        raise SchemaError("ParsedPublic world cannot contain private grounding metadata")
    if ("public_messages_sha256" in parsed and "messages_sha256" in parsed and
            parsed["public_messages_sha256"] != parsed["messages_sha256"]):
        raise SchemaError("conflicting parsed public message hashes")
    world = dict(parsed["world"])
    world["_public"] = {"atom_spans": parsed.get("atom_spans"), "record_spans": parsed.get("record_spans"),
                        "framing_spans": parsed.get("framing_spans"),
                        "messages_sha256": parsed.get("public_messages_sha256", parsed.get("messages_sha256", "")),
                        "require_source_bytes": require_source_bytes}
    if messages is not None:
        world["_public"]["messages"] = messages
    if "coverage_spans" in parsed:
        world["_public"]["coverage_spans"] = parsed["coverage_spans"]
    for field in ("parser_identity_sha256", "catalogue_identity_sha256", "catalogue_status", "admission_ready"):
        if field in parsed:
            world["_public"][field] = parsed[field]
    return solve_public_world(world)


def exact_target(format_name: str, solution: dict[str, Any]) -> str:
    status = solution["status"]
    if status not in {"DETERMINATE", "UNDERDETERMINED", "INCONSISTENT"}:
        raise SchemaError("invalid solution status")
    answer = solution["possible_answers"][0] if status == "DETERMINATE" and len(solution["possible_answers"]) == 1 else None
    if status == "DETERMINATE" and type(answer) is not int:
        raise SchemaError("determinate solution needs one exact integer")
    if format_name == "JSON":
        return json.dumps({"status": status, "answer": answer}, separators=(",", ":"))
    if format_name == "LINE":
        return f"{status}|{answer if answer is not None else '-'}"
    if format_name == "TOKEN":
        return str(answer) if answer is not None else {"UNDERDETERMINED": "UNKNOWN", "INCONSISTENT": "CONFLICT"}[status]
    raise SchemaError("unknown answer format")


def validate_answer(format_name: str, submitted: bytes | str, solution: dict[str, Any]) -> dict[str, Any]:
    """Strict final bytes, semantic meaning, and canonical format checked separately.

    EOS is a tokenizer/template property; literal '<eos>' and similar trailing text
    are rejected. This function does not simulate native token IDs or model calls.
    """
    target = exact_target(format_name, solution)
    if type(submitted) is bytes:
        try:
            text = submitted.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return {"semantic_correct": False, "format_correct": False, "exact_correct": False,
                    "reason": "submitted bytes are not valid UTF-8", "native_eos": "NOT_MEASURED"}
    elif type(submitted) is str:
        text = submitted
    else:
        return {"semantic_correct": False, "format_correct": False, "exact_correct": False,
                "reason": "submitted answer must be bytes or text", "native_eos": "NOT_MEASURED"}
    status, answer, format_correct, reason = None, None, False, "invalid answer syntax"
    if format_name == "JSON":
        pairs = None
        try:
            def unique_pairs(items):
                nonlocal pairs
                if len({key for key, _ in items}) != len(items):
                    raise ValueError("duplicate JSON key")
                pairs = items
                return dict(items)
            decoded = json.loads(text, object_pairs_hook=unique_pairs,
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            if type(decoded) is not dict or set(decoded) != {"status", "answer"}:
                raise ValueError("missing/extra JSON keys")
            status, answer = decoded["status"], decoded["answer"]
            if status not in {"DETERMINATE", "UNDERDETERMINED", "INCONSISTENT"}:
                raise ValueError("invalid status")
            if status == "DETERMINATE":
                if type(answer) is not int or not 0 <= answer <= 144:
                    raise ValueError("answer requires exact bounded integer")
            elif answer is not None:
                raise ValueError("diagnostic answer requires null")
            canonical = json.dumps({"status": status, "answer": answer}, separators=(",", ":"))
            format_correct = text == canonical
            reason = "ok" if format_correct else "JSON bytes are not canonical (key order/whitespace/integer encoding)"
        except (ValueError, TypeError, KeyError) as exc:
            status, answer = None, None
            reason = str(exc)
    elif format_name == "LINE":
        match = re.fullmatch(r"(DETERMINATE)\|((?:0|[1-9][0-9]*))|(UNDERDETERMINED|INCONSISTENT)\|-", text)
        if match:
            status = match.group(1) or match.group(3)
            number = match.group(2)
            answer = int(number) if number is not None and len(number) <= 3 else None
            if number is not None and len(number) > 3:
                return {"semantic_correct": False, "format_correct": False, "exact_correct": False,
                        "reason": "answer outside scalar bounds", "native_eos": "NOT_MEASURED"}
            format_correct = answer is None or answer <= 144
            reason = "ok" if format_correct else "answer outside scalar bounds"
    else:
        if text in {"UNKNOWN", "CONFLICT"}:
            status = "UNDERDETERMINED" if text == "UNKNOWN" else "INCONSISTENT"
            format_correct, reason = True, "ok"
        elif INTEGER.fullmatch(text):
            if len(text) > 3:
                return {"semantic_correct": False, "format_correct": False, "exact_correct": False,
                        "reason": "answer outside scalar bounds", "native_eos": "NOT_MEASURED"}
            status, answer = "DETERMINATE", int(text)
            format_correct = answer <= 144
            reason = "ok" if format_correct else "answer outside scalar bounds"
    expected_answer = solution["possible_answers"][0] if solution["status"] == "DETERMINATE" else None
    semantic_correct = status == solution["status"] and answer == expected_answer
    return {"semantic_correct": semantic_correct, "format_correct": format_correct,
            "exact_correct": semantic_correct and format_correct and text == target,
            "reason": reason if not format_correct else "ok" if semantic_correct else "wrong semantic status/answer",
            "native_eos": "NOT_MEASURED"}
