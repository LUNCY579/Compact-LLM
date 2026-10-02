"""CRS CPU reference runtime. Supplied machinery; no neural discovery claims.

No scorer import, filesystem retrieval, eval, shell, network or model-generated code.
The controlled grammar reader is deliberately a fixture, not a native model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class ContractError(ValueError):
    pass


class BudgetExhausted(ContractError):
    pass


class Status(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"
    CONTRADICTORY = "CONTRADICTORY"


@dataclass(frozen=True)
class Limits:
    max_source_bytes: int = 65536
    max_events: int = 256
    max_graph_nodes: int = 32
    max_operations: int = 64
    max_wall_seconds: float = 10
    max_state_bytes: int = 262144

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1e7:
                raise ContractError(f"invalid limit {key}")
            if key != "max_wall_seconds" and type(value) is not int:
                raise ContractError(f"noninteger limit {key}")


@dataclass
class Budget:
    limits: Limits
    operations: int = 0
    scanned_bytes: int = 0
    verification_scanned_bytes: int = 0
    events: int = 0
    peak_state_bytes: int = 0
    start: float = field(default_factory=time.monotonic)

    def charge(self, state=None):
        self.operations += 1
        if self.operations > self.limits.max_operations:
            raise BudgetExhausted("operation_limit")
        if time.monotonic() - self.start > self.limits.max_wall_seconds:
            raise BudgetExhausted("wall_limit")
        if state is not None:
            n = len(canonical(state))
            self.peak_state_bytes = max(self.peak_state_bytes, n)
            if n > self.limits.max_state_bytes:
                raise BudgetExhausted("state_limit")

    def receipt(self):
        return {"operations": self.operations, "source_bytes_scanned": self.scanned_bytes,
                "source_bytes_verification_scanned": self.verification_scanned_bytes,
                "extracted_events": self.events, "peak_serialized_state_bytes": self.peak_state_bytes,
                "wall_seconds": time.monotonic() - self.start, "model_calls": 0,
                "input_tokens": None, "output_tokens": None,
                "token_accounting": "not_applicable_no_tokenizer_or_model_in_CPU_fixture"}


class Evidence:
    def __init__(self, source: str, limit: int = 65536):
        if not isinstance(source, str):
            raise ContractError("source must be text")
        self.raw = source.encode("utf-8")
        if len(self.raw) > limit:
            raise BudgetExhausted("source_byte_limit")
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        self._spans = []
        offset = 0
        for line in source.splitlines(keepends=True):
            raw = line.encode("utf-8")
            self._spans.append({"id": f"{self.sha256}:{offset}:{offset+len(raw)}",
                                "source_sha256": self.sha256, "start": offset,
                                "end": offset+len(raw), "offset_unit": "UTF8_BYTE",
                                "text": line.rstrip("\r\n")})
            offset += len(raw)

    def spans(self):
        return copy.deepcopy(self._spans)

    def resolve(self, ref):
        if not isinstance(ref, dict) or ref.get("source_sha256") != self.sha256 or ref.get("offset_unit") != "UTF8_BYTE":
            raise ContractError("foreign source reference")
        start, end = ref.get("start"), ref.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.raw):
            raise ContractError("invalid byte span")
        try:
            return self.raw[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError("span splits a UTF8 character") from exc


class WorkingState:
    def __init__(self):
        self.nodes = {}
        self.revision = 0

    def derive(self, key, value, dependencies=()):
        if key in dependencies or any(d not in self.nodes or not self.nodes[d]["valid"] for d in dependencies):
            raise ContractError("unknown/cyclic/stale dependency")
        pending = list(dependencies)
        seen = set()
        while pending:
            node = pending.pop()
            if node == key:
                raise ContractError("cyclic dependency replacement")
            if node not in seen:
                seen.add(node)
                pending.extend(self.nodes[node]["dependencies"])
        if key in self.nodes:
            self.invalidate(key)
        self.nodes[key] = {"value": copy.deepcopy(value), "dependencies": list(dependencies),
                           "valid": True, "revision": self.revision}

    def invalidate(self, key):
        if key not in self.nodes:
            raise ContractError("unknown node")
        self.revision += 1
        invalid = {key}
        while True:
            old = len(invalid)
            invalid.update(k for k, v in self.nodes.items() if invalid.intersection(v["dependencies"]))
            if len(invalid) == old:
                break
        for k in invalid:
            self.nodes[k]["valid"] = False


_GRAMMAR = re.compile(r"^([A-Za-z][A-Za-z0-9_]*) @(\d+) (INIT|TRANSFER|CANCEL|ALIAS|REQUIRE|EXCEPT|FACT|EMPTY) (.+)$")
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _name(v):
    if not isinstance(v, str) or not _IDENT.fullmatch(v):
        raise ContractError("invalid identifier")
    return v


class FixtureReader:
    origin = "CODEX_SUPPLIED_CONTROLLED_GRAMMAR_FIXTURE"

    def read(self, evidence: Evidence, budget: Budget):
        events, identities = [], {}
        budget.scanned_bytes += len(evidence.raw)
        for span in evidence.spans():
            match = _GRAMMAR.fullmatch(span["text"])
            if not match:
                if re.match(r"^[A-Za-z][A-Za-z0-9_]* @\d+ (INIT|TRANSFER|CANCEL|ALIAS|REQUIRE|EXCEPT|FACT|EMPTY)\b", span["text"]):
                    raise ContractError("malformed event line")
                continue  # embedded instructions and distractors remain data
            eid, tick, kind, args = match.groups()
            tick = int(tick)
            if tick > 1000000:
                raise ContractError("event time overflow")
            args = args.split()
            expected = {"INIT": 2, "TRANSFER": 3, "CANCEL": 1, "ALIAS": 2,
                        "REQUIRE": 2, "EXCEPT": 2, "FACT": 2, "EMPTY": 1}[kind]
            if len(args) != expected:
                raise ContractError("malformed event arguments")
            if kind == "TRANSFER" and args[2] not in {"CONFIRMED", "PROPOSED", "NEGATED"}:
                raise ContractError("invalid modality")
            if kind == "FACT" and args[1] not in {s.value for s in Status}:
                raise ContractError("invalid fact status")
            for a in args:
                _name(a)
            event = {"id": eid, "time": tick, "kind": kind, "args": args,
                     "reference": {k: span[k] for k in ("source_sha256", "start", "end", "offset_unit")}}
            signature = (tick, kind, tuple(args))
            if eid in identities:
                if signature != identities[eid]:
                    raise ContractError("conflicting duplicate event identity")
                continue
            identities[eid] = signature
            events.append(event)
            if len(events) > budget.limits.max_events:
                raise BudgetExhausted("event_limit")
        budget.events = len(events)
        return events


def _validate_events(events, evidence, budget=None):
    if not isinstance(events, list) or len(events) > 256:
        raise ContractError("invalid extraction list")
    signatures = {}
    for e in events:
        if not isinstance(e, dict) or set(e) != {"id", "time", "kind", "args", "reference"}:
            raise ContractError("invalid extracted event schema")
        _name(e["id"])
        if type(e["time"]) is not int or not 0 <= e["time"] <= 1000000 or not isinstance(e["args"], list):
            raise ContractError("invalid extracted time/arguments")
        arity = {"INIT": 2, "TRANSFER": 3, "CANCEL": 1, "ALIAS": 2, "REQUIRE": 2, "EXCEPT": 2, "FACT": 2, "EMPTY": 1}
        if e["kind"] not in arity or len(e["args"]) != arity[e["kind"]]:
            raise ContractError("invalid extracted kind/arity")
        for argument in e["args"]:
            _name(argument)
        if e["kind"] == "TRANSFER" and e["args"][2] not in {"CONFIRMED", "PROPOSED", "NEGATED"}:
            raise ContractError("invalid extracted modality")
        if e["kind"] == "FACT" and e["args"][1] not in {s.value for s in Status}:
            raise ContractError("invalid extracted fact status")
        if not isinstance(e["reference"], dict) or set(e["reference"]) != {"source_sha256", "start", "end", "offset_unit"}:
            raise ContractError("invalid extracted reference schema")
        text = evidence.resolve(e["reference"]).rstrip("\r\n")
        if budget is not None:
            budget.verification_scanned_bytes += e["reference"]["end"] - e["reference"]["start"]
        match = _GRAMMAR.fullmatch(text)
        if not match or (e["id"], e["time"], e["kind"], e["args"]) != (match.group(1), int(match.group(2)), match.group(3), match.group(4).split()):
            raise ContractError("extraction is not supported by its exact controlled-grammar source span")
        signature = (e["time"], e["kind"], tuple(e["args"]))
        if e["id"] in signatures and signatures[e["id"]] != signature:
            raise ContractError("conflicting extraction identities")
        signatures[e["id"]] = signature
    expected = {}
    for span in evidence.spans():
        if budget is not None:
            budget.verification_scanned_bytes += span["end"] - span["start"]
        match = _GRAMMAR.fullmatch(span["text"])
        if match:
            eid = match.group(1)
            signature = (int(match.group(2)), match.group(3), tuple(match.group(4).split()))
            if eid in expected and expected[eid] != signature:
                raise ContractError("conflicting source identity cannot be silently omitted")
            expected[eid] = signature
        elif re.match(r"^[A-Za-z][A-Za-z0-9_]* @\d+ (INIT|TRANSFER|CANCEL|ALIAS|REQUIRE|EXCEPT|FACT|EMPTY)\b", span["text"]):
            raise ContractError("malformed source event cannot be silently omitted")
    if signatures != expected:
        raise ContractError("incomplete extracted source coverage")
    by_id = {e["id"]: e for e in events}
    for e in events:
        if e["kind"] in {"CANCEL", "REQUIRE", "EXCEPT"}:
            target = by_id.get(e["args"][0])
            if not target or target["kind"] != "TRANSFER":
                raise ContractError("invalid transfer reference")
            if e["kind"] == "CANCEL" and e["time"] < target["time"]:
                raise ContractError("cancellation before target")


def _aliases(events, tick):
    edges = {}
    for e in events:
        if e["kind"] == "ALIAS" and e["time"] <= tick:
            edges.setdefault(e["args"][0], set()).add(e["args"][1])

    def resolve(value):
        seen = set()
        while value in edges:
            if value in seen or len(edges[value]) != 1:
                return None
            seen.add(value)
            value = next(iter(edges[value]))
        return value
    return resolve


def _fact(events, predicate, tick):
    facts = [e for e in events if e["kind"] == "FACT" and e["args"][0] == predicate and e["time"] <= tick]
    if not facts:
        return Status.UNKNOWN.value
    last = max(e["time"] for e in facts)
    states = {e["args"][1] for e in facts if e["time"] == last}
    if len(states) > 1:
        return Status.CONTRADICTORY.value
    return next(iter(states))


def trace_state(events, entity, tick, complete):
    """Supplied domain primitive, not an autonomous discovered procedure."""
    if not complete:
        return {"status": "UNKNOWN", "value": None, "evidence_ids": []}
    resolve = _aliases(events, tick)
    entity = resolve(entity)
    if entity is None:
        return {"status": "UNKNOWN", "value": None, "evidence_ids": []}
    cancellations = {e["args"][0] for e in events if e["kind"] == "CANCEL" and e["time"] <= tick}
    active = [e for e in events if e["kind"] in {"INIT", "TRANSFER", "EMPTY"} and e["time"] <= tick
              and resolve(e["args"][0]) == entity and e["id"] not in cancellations]
    groups = {}
    for e in active:
        if e["kind"] == "TRANSFER" and e["args"][2] != "CONFIRMED":
            continue
        groups.setdefault(e["time"], []).append(e)
    status, owner, refs = "UNKNOWN", None, []
    for event_time in sorted(groups):
        owners, uncertainty = set(), set()
        for e in groups[event_time]:
            eligible = True
            guards = [g for g in events if g["kind"] in {"REQUIRE", "EXCEPT"} and g["args"][0] == e["id"]]
            guard_states = []
            for g in guards:
                st = _fact(events, g["args"][1], e["time"])
                if g["kind"] == "REQUIRE" and st == "REFUTED" or g["kind"] == "EXCEPT" and st == "SUPPORTED":
                    eligible = False
                if st in {"UNKNOWN", "CONTRADICTORY"}:
                    guard_states.append(st)
            if not eligible:
                continue
            refs.append(e["id"])
            uncertainty.update(guard_states)
            if guard_states:
                continue  # uncertain applicability is not a definitely conflicting owner
            v = resolve(e["args"][1]) if e["kind"] != "EMPTY" else "__EMPTY__"
            if v is None:
                uncertainty.add("UNKNOWN")
            else:
                owners.add(v)
        if "CONTRADICTORY" in uncertainty or len(owners) > 1:
            status, owner = "CONTRADICTORY", None
        elif "UNKNOWN" in uncertainty:
            status, owner = "UNKNOWN", None
        elif owners:
            owner = next(iter(owners))
            status, owner = ("REFUTED", None) if owner == "__EMPTY__" else ("SUPPORTED", owner)
    return {"status": status, "value": owner, "evidence_ids": sorted(set(refs))}


DEFAULT_GRAPH = [
    {"id": "read", "op": "LOOKUP", "deps": [], "args": {}},
    {"id": "answer", "op": "TRACE_STATE", "deps": ["read"], "args": {}},
    {"id": "check", "op": "CHECK", "deps": ["answer"], "args": {}}
]


class ProcedureSnapshot:
    def __init__(self, entries):
        if not isinstance(entries, list) or len(entries) > 16:
            raise ContractError("invalid snapshot")
        for entry in entries:
            if set(entry) != {"id", "version", "origin", "status", "applicability", "graph", "validation"}:
                raise ContractError("invalid procedure schema")
            _name(entry["id"])
            if type(entry["version"]) is not int or entry["version"] < 1:
                raise ContractError("invalid procedure version")
            if entry["origin"] not in {"HUMAN_SUPPLIED", "CODEX_SUPPLIED", "MODEL_PROPOSED", "MODEL_REVISED", "EMPIRICALLY_SELECTED"}:
                raise ContractError("invalid origin")
            validate_graph(entry["graph"], Limits())
        keys = [(e["id"], e["version"]) for e in entries]
        if len(keys) != len(set(keys)):
            raise ContractError("duplicate procedure version")
        self._bytes = canonical(entries)
        self.sha256 = hashlib.sha256(self._bytes).hexdigest()

    def entries(self):
        return json.loads(self._bytes)

    def add(self, *_args, **_kwargs):
        raise ContractError("evaluation_memory_write_rejected")


def validate_graph(graph, limits):
    if not isinstance(graph, list) or not 0 < len(graph) <= limits.max_graph_nodes:
        raise ContractError("invalid graph size")
    prior = set()
    names = {"LOOKUP", "JOIN", "FILTER", "TRACE_STATE", "AGGREGATE", "CHECK"}
    allowed_args = {"LOOKUP": set(), "JOIN": set(), "FILTER": {"modality", "until"},
                    "TRACE_STATE": set(), "AGGREGATE": {"field"}, "CHECK": set()}
    counts = {"LOOKUP": 0, "JOIN": 1, "FILTER": 1, "TRACE_STATE": 1, "AGGREGATE": 1, "CHECK": 1}
    types = {}
    for node in graph:
        if not isinstance(node, dict) or set(node) != {"id", "op", "deps", "args"}:
            raise ContractError("malformed graph node")
        nid, op, deps, args = (node[k] for k in ("id", "op", "deps", "args"))
        _name(nid)
        if nid in prior or op not in names or not isinstance(deps, list) or any(type(d) is not str or d not in prior for d in deps):
            raise ContractError("duplicate/cyclic/unknown operation or dependency")
        if len(deps) != counts[op] or not isinstance(args, dict) or set(args) - allowed_args[op]:
            raise ContractError("invalid arity or arguments")
        if op == "FILTER":
            if args.get("modality", "CONFIRMED") not in {"CONFIRMED", "ALL"}:
                raise ContractError("invalid filter modality")
            if "until" in args and (type(args["until"]) is not int or not 0 <= args["until"] <= 1000000):
                raise ContractError("invalid filter time")
        if op == "AGGREGATE" and args.get("field", "entity") not in {"entity", "owner"}:
            raise ContractError("invalid aggregate field")
        if op != "LOOKUP":
            expected = "answer" if op == "CHECK" else "events"
            if types[deps[0]] != expected:
                raise ContractError("operation input type mismatch")
        types[nid] = "events" if op in {"LOOKUP", "JOIN", "FILTER"} else "answer"
        prior.add(nid)
    if types[graph[-1]["id"]] != "answer" or graph[-1]["op"] != "CHECK":
        raise ContractError("graph must end in independent-of-gold CHECK")


def validate_case(case):
    required = {"id", "parent_id", "family", "split", "source", "query", "rules"}
    if not isinstance(case, dict) or set(case) != required:
        raise ContractError("runtime case schema rejects gold and unknown fields")
    if case["split"] not in {"TRAIN", "DEV", "CONFIRM"}:
        raise ContractError("invalid CRS split; central banks inaccessible")
    query, rules = case["query"], case["rules"]
    if not isinstance(query, dict) or set(query) != {"entity", "time"}:
        raise ContractError("invalid query")
    _name(query["entity"])
    if type(query["time"]) is not int or not 0 <= query["time"] <= 1000000:
        raise ContractError("invalid query time")
    if not isinstance(rules, dict) or set(rules) != {"cancel_mode", "equal_time", "complete"} or rules["cancel_mode"] != "remove_event" or rules["equal_time"] != "contradictory" or type(rules["complete"]) is not bool:
        raise ContractError("unsupported domain rules")


def run_case(case, *, arm="B", snapshot=None, reader=None, limits=None, graph=None):
    """Fresh state and costs on every invocation; no scoring data accepted."""
    limits = limits or Limits()
    budget = Budget(limits)
    state = WorkingState()
    result = {"status": "UNKNOWN", "value": None, "evidence_ids": []}
    events = []
    stop = "FAILED"
    error = None
    source_sha = None
    procedure = None
    reader = reader or FixtureReader()
    try:
        validate_case(case)
        if arm not in {"B", "C", "D"}:
            raise ContractError("CPU fixture A unavailable; no model-only execution")
        evidence = Evidence(case["source"], limits.max_source_bytes)
        source_sha = evidence.sha256
        if arm == "C":
            if not isinstance(snapshot, ProcedureSnapshot):
                raise ContractError("C requires frozen snapshot")
            eligible = [e for e in snapshot.entries() if e["status"] == "FROZEN" and e["applicability"] == "ownership_remove_event_v1"]
            if len(eligible) != 1:
                raise ContractError("procedure applicability ambiguous or absent")
            procedure = {"id": eligible[0]["id"], "version": eligible[0]["version"], "origin": eligible[0]["origin"]}
            graph = eligible[0]["graph"]
        elif arm == "D":
            if not isinstance(snapshot, ProcedureSnapshot) or any(e["applicability"] == "ownership_remove_event_v1" for e in snapshot.entries()):
                raise ContractError("D requires frozen nonapplicable irrelevant snapshot")
            if len(snapshot._bytes) != len(supplied_snapshot()._bytes):
                raise ContractError("D requires byte-size match to registered CPU C snapshot")
        graph = copy.deepcopy(graph if graph is not None else DEFAULT_GRAPH)
        validate_graph(graph, limits)
        for node in graph:
            budget.charge(state.nodes)
            op, args = node["op"], node["args"]
            input_value = copy.deepcopy(state.nodes[node["deps"][0]]["value"]) if node["deps"] else None
            if op == "LOOKUP":
                value = reader.read(evidence, budget)
                _validate_events(value, evidence, budget)
                if len(value) > limits.max_events:
                    raise BudgetExhausted("event_limit")
                events = copy.deepcopy(value)
            elif op == "JOIN":
                resolve = _aliases(input_value, case["query"]["time"])
                value = copy.deepcopy(input_value)
                for e in value:
                    if e["kind"] in {"INIT", "TRANSFER"}:
                        for index in (0, 1):
                            resolved = resolve(e["args"][index])
                            if resolved:
                                e["args"][index] = resolved
            elif op == "FILTER":
                value = [e for e in input_value if e["time"] <= args.get("until", case["query"]["time"])
                         and (args.get("modality", "CONFIRMED") == "ALL" or e["kind"] != "TRANSFER" or e["args"][2] == "CONFIRMED")]
            elif op == "TRACE_STATE":
                value = trace_state(input_value, case["query"]["entity"], case["query"]["time"], case["rules"]["complete"])
            elif op == "AGGREGATE":
                if not case["rules"]["complete"]:
                    value = {"status": "UNKNOWN", "value": None, "evidence_ids": []}
                else:
                    entities = {e["args"][0] for e in input_value if e["kind"] in {"INIT", "TRANSFER", "EMPTY"}}
                    values = [trace_state(input_value, entity, case["query"]["time"], True) for entity in entities]
                    if any(v["status"] != "SUPPORTED" for v in values):
                        value = {"status": "UNKNOWN", "value": None, "evidence_ids": []}
                    else:
                        n = len(entities) if args.get("field", "entity") == "entity" else len({v["value"] for v in values})
                        value = {"status": "SUPPORTED", "value": n, "evidence_ids": sorted({eid for v in values for eid in v["evidence_ids"]})}
            else:  # CHECK validates structure/source references, never semantic gold
                if not isinstance(input_value, dict) or set(input_value) != {"status", "value", "evidence_ids"} or input_value["status"] not in {s.value for s in Status}:
                    raise ContractError("invalid answer state")
                if input_value["status"] != "SUPPORTED" and input_value["value"] is not None:
                    raise ContractError("unsupported answer value")
                ids = {e["id"] for e in events}
                if any(eid not in ids for eid in input_value["evidence_ids"]):
                    raise ContractError("answer has foreign evidence")
                value = input_value
            state.derive(node["id"], value, node["deps"])
            budget.charge(state.nodes)
        result = state.nodes[graph[-1]["id"]]["value"]
        stop = "ANSWER" if result["status"] == "SUPPORTED" else "EXPLICIT_STATUS_ABSTENTION_FIXTURE"
    except (ContractError, ValueError, TypeError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop = "BUDGET_EXHAUSTED" if isinstance(exc, BudgetExhausted) else "FAILED"
    return {"case_id": case.get("id") if isinstance(case, dict) else None, "arm": arm,
            "answer": copy.deepcopy(result), "stop_reason": stop, "error": error,
            "source_sha256": source_sha, "procedure": procedure,
            "procedure_snapshot_sha256": snapshot.sha256 if snapshot else None,
            "procedure_bytes": len(snapshot._bytes) if snapshot else 0,
            "reader_origin": reader.origin, "controller_origin": "CODEX_SUPPLIED",
            "primitive_origin": "CODEX_SUPPLIED", "evidence_class": "CPU_ENGINEERING_FIXTURE",
            "verifier_scope": "schema_and_source_reference_consistency_not_independent_truth",
            "cost": budget.receipt(), "provenance_graph": copy.deepcopy(state.nodes),
            "coverage": {"supplied_complete_flag": case.get("rules", {}).get("complete") if isinstance(case, dict) else None,
                         "full_source_scanned": budget.scanned_bytes == len(case.get("source", "").encode("utf-8")) if isinstance(case, dict) and isinstance(case.get("source"), str) else False}}


def supplied_snapshot():
    return ProcedureSnapshot([{"id": "ownership", "version": 1, "origin": "CODEX_SUPPLIED",
                               "status": "FROZEN", "applicability": "ownership_remove_event_v1",
                               "graph": DEFAULT_GRAPH, "validation": "CPU_ONLY_NO_NATIVE_DISCOVERY"}])
