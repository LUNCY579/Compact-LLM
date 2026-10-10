"""Answer-free, bounded semantic exclusion of exposed distributed-dossier inputs.

This is a pre-solve firewall, not a solver or a novelty certificate.  Source byte
hashes bind the inventory; semantic signatures compare typed specifications.
Graph refinement only partitions symbols.  Every remaining nontrivial symbol
permutation is enumerated, or the comparison DEFERs.  A refinement hash alone is
never treated as graph isomorphism.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

from . import schema

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "cycles/003-dossier-implementation/fixture_exclusion"
INVENTORY_PATH = EVIDENCE / "INVENTORY.json"
COMPLETENESS_SEAL_PATH = EVIDENCE / "COMPLETENESS_SEAL.json"
# This separate producer seal binds concrete protected membership and source/spec
# snapshot identities. A manifest cannot authorize changing its own seal.
COMPLETENESS_SEAL_SHA256 = "ba6d4027f7787648ae9a408f9e44ee30bc83ea2d1e3c8370f10c0f5c7c3915ef"
REVIEWED_INVENTORY_PIN_PATH = EVIDENCE / "REVIEWED_INVENTORY_PIN.json"
ALGORITHM = "DD-FIXTURE-TYPED-TREE-COLORED-DEPENDENCY-GRAPH"
VERSION = "1.0.3"
SPEC_SCHEMA = "rineya.exposed_fixture_spec.v1"
COVERAGE = {
    "scope": "complete typed distributed-dossier specification with fixed framework",
    "ignored": ["raw author/registry bookkeeping", "output format", "documents and source placement",
                "entity public labels", "record IDs",
                "framework rule IDs and sentence variants"],
    "alpha_names": ["typed entity IDs", "variable IDs", "event IDs", "policy IDs",
                    "leg IDs scoped by plan", "program IDs scoped by operator", "exception IDs"],
    "unordered": ["entity/variable declarations", "source record collection", "variable choices",
                  "operator menu candidates", "alias endpoints", "trace_join sources",
                  "leg people", "bundle plans", "query candidate IDs", "XOR final stations",
                  "framework rule codes"],
    "ordered": ["session_plan legs", "trace_fork destinations", "ledger_split children",
                "predicate arguments", "state registers and encoding"],
    "retained": ["all typed operative and incidental assertions, including repetitions",
                 "finite variable domains and shared-variable identity", "query", "framework codes",
                 "days/ticks/predecessor links", "policy/exception scope and conditions",
                 "repeated-definition symbol equality, including program/exception identities"],
    "not_covered": ["undeclared derivative ancestry", "partial context projection",
                    "semantic mutations", "alias-component collapse", "redundant-assertion insertion/removal",
                    "arbitrary logical equivalence or program-function equivalence",
                    "other world schemas and historical dataset overlap"],
}


class ComparisonDeferred(ValueError):
    """Unsupported typed input, unbound inventory, or exhausted canonical budget."""


def _bytes(value):
    return schema.canonical_bytes(value)


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def fixture_spec(world):
    """Project a validated public/raw world; no target/proof/oracle field is allowed."""
    if type(world) is not dict:
        raise ComparisonDeferred("world must be a closed typed object")
    try:
        schema.validate_world(world, public=set(world) == set(schema.load_catalogue()["world_fields"]))
    except (schema.SchemaError, KeyError, TypeError) as exc:
        raise ComparisonDeferred("unsupported/invalid typed world: " + str(exc)) from exc
    framework = world["framework"]
    return {
        "spec_schema": SPEC_SCHEMA,
        "world_schema": world["world_schema"],
        "mechanism": world["mechanism"],
        "entities": [{"entity_id": e["entity_id"], "kind": e["kind"]} for e in world["entities"]],
        "variables": copy.deepcopy(world["variables"]),
        "records": [{k: copy.deepcopy(v) for k, v in r.items() if k not in ("record_id", "document_id")}
                    for r in world["records"]],
        "framework": {"semantic_version": framework["semantic_version"],
                      "rule_codes": sorted(c["code"] for c in framework["rule_clauses"]),
                      "closed_candidate_list": framework["closed_candidate_list"],
                      "unknown_defaults": framework["unknown_defaults"],
                      "source_assertions_are_conjunctive": framework["source_assertions_are_conjunctive"]},
        "query": copy.deepcopy(world["query"]),
    }


def fixture_world(spec):
    """Rehydrate an EXPOSED fixture solely for closed schema checking/tests.

    Synthesized source attachments carry no admission/reservation/row identity.
    They are not included in the compared specification.
    """
    fields = {"spec_schema", "world_schema", "mechanism", "entities", "variables", "records", "framework", "query"}
    if type(spec) is not dict or set(spec) != fields or spec["spec_schema"] != SPEC_SCHEMA:
        raise ComparisonDeferred("unsupported or answer-bearing fixture specification")
    if type(spec["entities"]) is not list or any(type(e) is not dict or set(e) != {"entity_id", "kind"} for e in spec["entities"]):
        raise ComparisonDeferred("fixture entities must be closed typed declarations")
    fw = spec["framework"]
    if type(fw) is not dict or set(fw) != {"semantic_version", "rule_codes", "closed_candidate_list", "unknown_defaults", "source_assertions_are_conjunctive"}:
        raise ComparisonDeferred("fixture framework must be closed")
    if type(fw["rule_codes"]) is not list or any(type(c) is not str for c in fw["rule_codes"]):
        raise ComparisonDeferred("fixture rule codes must be a string array")
    if type(spec["records"]) is not list or any(type(r) is not dict or {"record_id", "document_id"} & set(r) for r in spec["records"]):
        raise ComparisonDeferred("fixture records must omit source bookkeeping")
    records = [dict(copy.deepcopy(r), record_id=f"FixtureRecord{i}", document_id=f"FixtureDocument{i % 3}") for i, r in enumerate(spec["records"])]
    world = {"world_schema": spec["world_schema"], "mechanism": spec["mechanism"],
             "entities": [dict(e, public_label=e["entity_id"]) for e in spec["entities"]],
             "variables": copy.deepcopy(spec["variables"]), "records": records,
             "documents": [dict(document_id=f"FixtureDocument{i}", purpose="maintenance_log", title="Exposed fixture " + word,
                                dated_day=0, record_ids=[r["record_id"] for r in records if r["document_id"] == f"FixtureDocument{i}"],
                                style="controlled_prose") for i, word in enumerate(["One", "Two", "Three"])],
             "framework": {k: copy.deepcopy(v) for k, v in fw.items() if k != "rule_codes"},
             "query": copy.deepcopy(spec["query"]), "format": "JSON"}
    world["framework"]["rule_clauses"] = [dict(rule_id=f"FixtureRule{i}", code=c, public_variant=0) for i, c in enumerate(fw["rule_codes"])]
    # Set-valued schema arrays use spelling order before the alpha comparison.
    for variable in world["variables"]: variable["choices"].sort()
    for record in records:
        if record["kind"] == "session_plan":
            for leg in record["legs"]: leg["people"].sort()
        if record["kind"] == "bundle": record["plans"].sort()
        if record["kind"] == "trace_join": record["sources"].sort()
    if world["mechanism"] == "policy_feasibility": world["query"]["candidate_ids"].sort()
    try:
        schema.validate_world(world, public=True)
    except (schema.SchemaError, KeyError, TypeError) as exc:
        raise ComparisonDeferred("invalid exposed fixture specification: " + str(exc)) from exc
    return world


def _typed_tree(spec):
    """Explicit field typing prevents coincidentally equal literals becoming refs."""
    entities = {e["entity_id"]: e["kind"] for e in spec["entities"]}
    variables = {v["variable_id"]: v["scalar_type"] for v in spec["variables"]}
    symbols = set()
    symbol_scopes = {}
    def ref(namespace, name):
        token = (namespace, name)
        symbols.add(token)
        return {"$ref": [namespace, name]}
    def entity(value):
        if type(value) is dict: return variable(value)
        return ref("entity:" + entities[value], value)
    def variable(value): return ref("variable:" + variables[value["var"]], value["var"])
    def scalar(value): return variable(value) if type(value) is dict else value
    def unordered(values): return {"$unordered": values}
    def arguments(predicate, values):
        definitions = schema.load_catalogue()["predicates"][predicate]["arguments"]
        return [value if definition.startswith("slot ") else entity(value) for definition, value in zip(definitions, values)]
    def legref(plan, leg_id):
        # Scope is represented by a graph edge, not by an unnormalized name color.
        token = ("leg", plan + "/" + leg_id)
        symbol_scopes[token] = ("plan_scope", ("entity:plan", plan))
        return ref(*token)
    records = []
    for original in spec["records"]:
        r = copy.deepcopy(original)
        kind = r["kind"]
        if kind == "alias":
            r["endpoints"] = unordered([entity(r.pop("left")), entity(r.pop("right"))])
        elif kind in ("membership", "delegation"):
            for key in (("child", "parent") if kind == "membership" else ("signer", "owner")): r[key] = entity(r[key])
            for key in ("from_day", "through_day"): r[key] = scalar(r[key])
        elif kind.startswith("ledger_"):
            for key in ("event_id", "predecessor"):
                if key in r: r[key] = ref("event", r[key])
            for key in ("commitment", "owner", "signer"):
                if key in r: r[key] = entity(r[key])
            for key in ("day", "quantity"):
                if key in r: r[key] = scalar(r[key])
            for child in r.get("children", []):
                child["commitment"] = entity(child["commitment"]); child["owner"] = entity(child["owner"]); child["quantity"] = scalar(child["quantity"])
        elif kind == "operator_menu":
            operator = r["operator"]; r["operator"] = entity(operator)
            for candidate in r["candidates"]:
                token = ("program", operator + "/" + candidate["program_id"])
                symbol_scopes[token] = ("operator_scope", ("entity:operator", operator))
                candidate["program_id"] = ref(*token)
            r["candidates"] = unordered(r["candidates"])
        elif kind in ("bench_example", "station_initial"):
            key = "operator" if kind == "bench_example" else "station"; r[key] = entity(r[key])
            for key in (("input", "output") if kind == "bench_example" else ("state",)):
                r[key] = {reg: scalar(value) for reg, value in r[key].items()}
        elif kind.startswith("trace_"):
            r["tick"] = scalar(r["tick"])
            for key in ("station", "operator", "source", "destination"):
                if key in r: r[key] = entity(r[key])
            for key in ("source_register", "destination_register"):
                if key in r: r[key] = scalar(r[key])
            if "destinations" in r: r["destinations"] = [entity(value) for value in r["destinations"]]
            if "sources" in r: r["sources"] = unordered([entity(value) for value in r["sources"]])
        elif kind in ("fact_bool", "fact_count"):
            r["arguments"] = arguments(r["predicate"], r["arguments"]); r["value"] = scalar(r["value"])
        elif kind == "session_plan":
            plan = r["plan"]; r["plan"] = entity(plan)
            for leg in r["legs"]:
                leg["leg_id"] = legref(plan, leg["leg_id"])
                for key in ("room", "kit", "leader", "group"): leg[key] = entity(leg[key])
                leg["people"] = unordered([entity(person) for person in leg["people"]])
        elif kind == "bundle":
            r["bundle"] = entity(r["bundle"]); r["plans"] = unordered([entity(plan) for plan in r["plans"]])
        elif kind == "base_policy":
            r["policy_id"] = ref("policy", r["policy_id"])
            if "credential" in r["predicate"]: r["predicate"]["credential"] = entity(r["predicate"]["credential"])
        elif kind == "exception":
            r["exception_id"] = ref("exception", r["exception_id"])
            r["base_policy_id"] = ref("policy", r["base_policy_id"])
            sel = r["selector"]; plan = sel.get("plan")
            if "leg_id" in sel: sel["leg_id"] = legref(plan, sel["leg_id"])
            for key in ("plan", "group", "bundle"):
                if key in sel: sel[key] = entity(sel[key])
            condition = r["condition"]; condition["arguments"] = arguments(condition["predicate"], condition["arguments"])
        elif kind == "incidental": r["subject"] = entity(r["subject"])
        else: raise ComparisonDeferred("record kind not covered: " + str(kind))
        records.append(r)
    query = copy.deepcopy(spec["query"])
    if spec["mechanism"] == "commitment_ledger": query["group_label"] = entity(query["group_label"])
    elif spec["mechanism"] == "guarded_transition_learning":
        finals = [entity(e) for e in query["final_stations"]]
        query["final_stations"] = unordered(finals) if query["combine"] == "xor" else finals
    else: query["candidate_ids"] = unordered([entity(e) for e in query["candidate_ids"]])
    variable_declarations = []
    for v in spec["variables"]:
        variable_declarations.append({"variable_id": ref("variable:" + v["scalar_type"], v["variable_id"]),
                                      "scalar_type": v["scalar_type"],
                                      "choices": unordered([entity(c) if v["scalar_type"] == "EntityId" else c for c in v["choices"]])})
    tree = {"spec_schema": spec["spec_schema"], "world_schema": spec["world_schema"], "mechanism": spec["mechanism"],
            "entities": unordered([entity(e["entity_id"]) for e in spec["entities"]]),
            "variables": unordered(variable_declarations), "records": unordered(records),
            "framework": dict(spec["framework"], rule_codes=unordered(spec["framework"]["rule_codes"])), "query": query}
    return tree, symbols, symbol_scopes


def _normalize(tree, mapping=None):
    if type(tree) is dict:
        if set(tree) == {"$ref"}:
            namespace, name = tree["$ref"]
            return {"$ref": [namespace, mapping[(namespace, name)] if mapping is not None else name]}
        if set(tree) == {"$unordered"}:
            values = [_normalize(v, mapping) for v in tree["$unordered"]]
            return {"$unordered": sorted(values, key=_bytes)}
        return {k: _normalize(v, mapping) for k, v in tree.items()}
    if type(tree) is list: return [_normalize(v, mapping) for v in tree]
    return tree


def _partitions(tree, symbols, symbol_scopes, max_graph_nodes):
    nodes, outgoing, incoming = [], defaultdict(list), defaultdict(list)
    symbol_nodes = {}
    counts = Counter()
    def node(color):
        if len(nodes) >= max_graph_nodes: raise ComparisonDeferred("colored dependency graph node budget exceeded")
        nodes.append(_bytes(color)); return len(nodes) - 1
    def edge(a, role, b): outgoing[a].append((role, b)); incoming[b].append((role, a))
    for symbol in sorted(symbols): symbol_nodes[symbol] = node(["symbol", symbol[0]])
    def visit(value):
        if type(value) is dict and set(value) == {"$ref"}:
            symbol = tuple(value["$ref"]); counts[symbol] += 1; return symbol_nodes[symbol]
        if type(value) is dict:
            if set(value) == {"$unordered"}:
                current = node(["unordered", len(value["$unordered"])])
                for child in value["$unordered"]: edge(current, "member", visit(child))
            else:
                current = node(["object", sorted(value)])
                for key, child in sorted(value.items()): edge(current, "field:" + key, visit(child))
        elif type(value) is list:
            current = node(["ordered", len(value)])
            for i, child in enumerate(value): edge(current, "index:" + str(i), visit(child))
        else: current = node(["literal", type(value).__name__, value])
        return current
    visit(tree)
    for symbol, (role, scope) in symbol_scopes.items(): edge(symbol_nodes[symbol], role, symbol_nodes[scope])
    ranks = {color: i for i, color in enumerate(sorted(set(nodes)))}
    colors = [ranks[color] for color in nodes]
    rounds = 0
    for _ in range(len(nodes)):
        signatures = [(colors[i], tuple(sorted((role, colors[j]) for role, j in outgoing[i])),
                       tuple(sorted((role, colors[j]) for role, j in incoming[i]))) for i in range(len(nodes))]
        ranks = {value: i for i, value in enumerate(sorted(set(signatures)))}
        refined = [ranks[value] for value in signatures]; rounds += 1
        old_partition_count, new_partition_count = len(set(colors)), len(set(refined))
        colors = refined
        if new_partition_count == old_partition_count: break
    cells = defaultdict(list)
    for symbol, i in symbol_nodes.items(): cells[(symbol[0], colors[i])].append(symbol)
    groups = []
    for (namespace, color), values in sorted(cells.items()):
        # A declaration-only entity has no semantic occurrence or scope edge.
        # Its arbitrary spelling permutation cannot change the normalized tree.
        trivial = all(v[0].startswith("entity:") and counts[v] == 1 and not outgoing[symbol_nodes[v]] for v in values)
        groups.append((namespace, color, sorted(values), trivial))
    return groups, {"graph_nodes": len(nodes), "refinement_rounds": rounds,
                    "unresolved_cells": [len(values) for _, _, values, trivial in groups if len(values) > 1 and not trivial],
                    "declaration_only_automorphism_cells": [len(values) for _, _, values, trivial in groups if len(values) > 1 and trivial]}


def semantic_signatures(value, *, is_spec=False, max_permutations=100000, max_graph_nodes=4096):
    """Exact typed spelling-sensitive and complete bounded alpha canonicalization."""
    if type(max_permutations) is not int or max_permutations < 1 or type(max_graph_nodes) is not int or max_graph_nodes < 1:
        raise ComparisonDeferred("canonicalization budgets must be positive integers")
    spec = fixture_spec(fixture_world(value)) if is_spec else fixture_spec(value)
    tree, symbols, symbol_scopes = _typed_tree(spec)
    groups, stats = _partitions(tree, symbols, symbol_scopes, max_graph_nodes)
    required = math.prod(math.factorial(len(values)) for _, _, values, trivial in groups if not trivial)
    if required > max_permutations:
        raise ComparisonDeferred(f"complete alpha canonicalization needs {required} permutations; budget {max_permutations}")
    possibilities, offsets = [], Counter()
    for namespace, _, values, trivial in groups:
        slots = list(range(offsets[namespace], offsets[namespace] + len(values))); offsets[namespace] += len(values)
        possibilities.append([(values, tuple(slots))] if trivial else [(values, p) for p in itertools.permutations(slots)])
    best, best_map = None, None
    for assignment in itertools.product(*possibilities):
        mapping = {symbol: slot for values, slots in assignment for symbol, slot in zip(values, slots)}
        raw = _bytes(_normalize(tree, mapping))
        if best is None or raw < best: best, best_map = raw, mapping
    # Empty symbol vocabularies still yield one assignment.
    stats["permutations_enumerated"] = required
    return {"comparison_algorithm": ALGORITHM, "comparison_version": VERSION,
            "exact_semantic_sha256": _sha(_normalize(tree)),
            "alpha_semantic_sha256": hashlib.sha256(best).hexdigest(),
            "canonicalization": stats,
            "name_map": [{"namespace": namespace, "source_name": name, "canonical_name": slot}
                         for (namespace, name), slot in sorted(best_map.items())],
            "coverage": copy.deepcopy(COVERAGE)}


def load_inventory(path=None):
    """Validate answer-free specs and immutable byte bindings before comparison."""
    manifest_path = Path(path or INVENTORY_PATH)
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = schema.loads_json(manifest_raw)
        seal_raw = COMPLETENESS_SEAL_PATH.read_bytes()
        if hashlib.sha256(seal_raw).hexdigest() != COMPLETENESS_SEAL_SHA256:
            raise ComparisonDeferred("trusted completeness seal byte identity mismatch")
        seal = schema.loads_json(seal_raw)
        if type(seal["schema_version"]) is not int or seal["schema_version"] != 1 or seal["comparison_algorithm"] != ALGORITHM or seal["comparison_version"] != VERSION or seal["role"] != "EXCLUDED":
            raise ComparisonDeferred("trusted completeness seal version/role mismatch")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ComparisonDeferred("inventory schema_version must be exact integer 1")
        if manifest["comparison_algorithm"] != ALGORITHM or manifest["comparison_version"] != VERSION:
            raise ComparisonDeferred("inventory comparison version mismatch")
        if manifest["role"] != "EXCLUDED" or manifest["registered_source_safety"] != "EXCLUDED_METADATA_ONLY":
            raise ComparisonDeferred("inventory has unsafe role")
        for collection, id_key, binding_name in [("sources", "source_id", "protected_sources"), ("fixtures", "fixture_id", "protected_fixtures")]:
            values = manifest[collection]
            if type(values) is not list or any(type(v) is not dict or type(v.get(id_key)) is not str for v in values):
                raise ComparisonDeferred("inventory requires a typed complete " + collection + " collection")
            actual_ids = [v[id_key] for v in values]
            expected = seal[binding_name]
            if len(actual_ids) != len(set(actual_ids)):
                raise ComparisonDeferred("duplicate protected " + collection + " identity")
            if set(actual_ids) != set(expected):
                raise ComparisonDeferred("protected " + collection + " membership differs from trusted completeness seal")
            for value in values:
                binding = expected[value[id_key]]
                # These fields are identity-bearing source/spec bindings, not
                # attacker-controlled totals, free text, or recomputed counts.
                actual_binding = {key: value.get(key, [] if key in ("ancestry", "historical_bindings") else None) for key in binding}
                for key in ("source_ids", "ancestry"):
                    if key in actual_binding:
                        if type(actual_binding[key]) is not list or any(type(x) is not str for x in actual_binding[key]):
                            raise ComparisonDeferred("invalid protected lineage binding")
                        actual_binding[key] = sorted(actual_binding[key])
                if _bytes(actual_binding) != _bytes(binding):
                    raise ComparisonDeferred("protected " + collection + " source/spec binding changed: " + value[id_key])
        environment = manifest["comparison_environment"]
        if type(environment) is not list or any(type(v) is not dict or set(v) != {"path", "sha256"} for v in environment):
            raise ComparisonDeferred("comparison environment requires closed identities")
        paths = [v["path"] for v in environment]
        if len(paths) != len(set(paths)) or set(paths) != set(seal["comparison_dependency_paths"]):
            raise ComparisonDeferred("comparison dependency membership changed")
        reviewed_pin = None
        if REVIEWED_INVENTORY_PIN_PATH.exists():
            reviewed_pin = schema.loads_json(REVIEWED_INVENTORY_PIN_PATH.read_bytes())
            if (type(reviewed_pin["schema_version"]) is not int or reviewed_pin["schema_version"] != 1 or reviewed_pin["status"] != "REVIEWED_EXCLUSION_PIN"
                    or reviewed_pin["comparison_algorithm"] != ALGORITHM or reviewed_pin["comparison_version"] != VERSION
                    or reviewed_pin["completeness_seal_sha256"] != COMPLETENESS_SEAL_SHA256
                    or reviewed_pin["inventory_sha256"] != hashlib.sha256(manifest_raw).hexdigest()
                    or _bytes(sorted(reviewed_pin["comparison_environment"], key=lambda v: v["path"])) != _bytes(sorted(environment, key=lambda v: v["path"]))):
                raise ComparisonDeferred("externally reviewed inventory pin identity mismatch")
        for dependency in manifest["comparison_environment"]:
            live_path = (ROOT / dependency["path"]).resolve(); live_path.relative_to(ROOT)
            if hashlib.sha256(live_path.read_bytes()).hexdigest() != dependency["sha256"]:
                raise ComparisonDeferred("comparison implementation/schema/catalogue changed: " + dependency["path"])
        for source in manifest["sources"]:
            snapshot = (ROOT / source["snapshot_path"]).resolve(); snapshot.relative_to(ROOT)
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source["source_sha256"]:
                raise ComparisonDeferred("source snapshot byte hash mismatch: " + source["source_id"])
            for historical in source["historical_bindings"]:
                snapshot = (ROOT / historical["snapshot_path"]).resolve(); snapshot.relative_to(ROOT)
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != historical["source_sha256"]:
                    raise ComparisonDeferred("historical source snapshot byte hash mismatch: " + source["source_id"])
        known_sources = {s["source_id"] for s in manifest["sources"]}
        fixture_ids = set()
        for entry in manifest["fixtures"]:
            if entry["fixture_id"] in fixture_ids: raise ComparisonDeferred("duplicate fixture ID")
            fixture_ids.add(entry["fixture_id"])
            if entry["role"] != "EXCLUDED" or entry["admission_allowed"] is not False or not set(entry["source_ids"]) <= known_sources:
                raise ComparisonDeferred("unsafe/unbound fixture entry")
            spec_path = (ROOT / entry["spec_path"]).resolve(); spec_path.relative_to(ROOT)
            raw = spec_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["spec_source_sha256"]: raise ComparisonDeferred("fixture specification bytes changed")
            spec = schema.loads_json(raw)
            signatures = _bound_spec_signatures(raw)
            for key in ("exact_semantic_sha256", "alpha_semantic_sha256"):
                if signatures[key] != entry[key]: raise ComparisonDeferred("fixture canonical signature mismatch")
            entry["spec"] = spec
        manifest["_runtime_inventory_review"] = "REVIEWED_EXTERNAL_PIN" if reviewed_pin else "PRODUCER_COMPLETENESS_SEAL_ONLY"
        return manifest
    except ComparisonDeferred: raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ComparisonDeferred("exclusion inventory unavailable/invalid: " + str(exc)) from exc


@lru_cache(maxsize=128)
def _bound_spec_signatures(raw):
    # Cache by verified exact immutable specification bytes, never by path/mtime.
    return semantic_signatures(schema.loads_json(raw), is_spec=True)


def check_world(world, *, ancestry=(), max_permutations=100000, max_graph_nodes=4096, inventory_path=None):
    """Check before solving/reservation. NO_KNOWN_MATCH is not TRAIN admission.

    ancestry is an explicit iterable of source IDs, fixture IDs, source SHA256s,
    or known exact/alpha semantic SHA256s. Unknown ancestry is reported and cannot
    certify freshness. The caller still owns provenance and every other gate.
    """
    base = {"comparison_algorithm": ALGORITHM, "comparison_version": VERSION,
            "coverage": copy.deepcopy(COVERAGE), "derivative_coverage": "NOT_PROVEN",
            "admission_authorized": False, "solver_called": False,
            "matched_fixture_ids": [], "matched_ancestry": []}
    try:
        if isinstance(ancestry, (str, bytes)) or not isinstance(ancestry, (list, tuple, set, frozenset)) or any(type(a) is not str for a in ancestry):
            raise ComparisonDeferred("ancestry must be an explicit collection of string identifiers")
        inventory = load_inventory(inventory_path)
        base["inventory_review"] = inventory["_runtime_inventory_review"]
        base["completeness_seal_sha256"] = COMPLETENESS_SEAL_SHA256
        identifiers = {s["source_id"] for s in inventory["sources"]} | {s["source_sha256"] for s in inventory["sources"]}
        identifiers.update(h["source_sha256"] for s in inventory["sources"] for h in s["historical_bindings"])
        for f in inventory["fixtures"]:
            identifiers.update((f["fixture_id"], f["exact_semantic_sha256"], f["alpha_semantic_sha256"]))
        hits = sorted(set(ancestry) & identifiers)
        base["unknown_ancestry_ids"] = sorted(set(ancestry) - identifiers)
        if hits:
            base.update(decision="EXCLUDED", must_reject=True, matched_ancestry=hits,
                        reason="declared derivative of an exposed excluded source/fixture")
            return base
        signatures = semantic_signatures(world, max_permutations=max_permutations, max_graph_nodes=max_graph_nodes)
        base.update(signatures)
        matches = [f for f in inventory["fixtures"] if f["alpha_semantic_sha256"] == signatures["alpha_semantic_sha256"]]
        base["matched_fixture_ids"] = [f["fixture_id"] for f in matches]
        base["match_kinds"] = sorted({"EXACT_TYPED_SPEC" if f["exact_semantic_sha256"] == signatures["exact_semantic_sha256"] else "ALPHA_TYPED_SPEC" for f in matches})
        base.update(decision="EXCLUDED" if matches else "NO_KNOWN_MATCH", must_reject=bool(matches),
                    reason="complete typed specification matches exposed fixture" if matches else "no complete-spec match within declared comparison coverage; derivative novelty not proven")
        return base
    except ComparisonDeferred as exc:
        base.update(decision="DEFER", must_reject=True, reason=str(exc))
        return base
