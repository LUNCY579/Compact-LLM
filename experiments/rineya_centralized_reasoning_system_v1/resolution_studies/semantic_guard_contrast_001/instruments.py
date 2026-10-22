"""Supplied exposed DEV instruments. No package/core/oracle imports or gold inputs.

F is a chronological state machine; L/P are separate subset decision rules.
Shared parser/grammar assumptions and prior builder gold exposure are explicit.
"""
import collections
import re

RULES = {"cancel_mode": "remove_event", "equal_time": "contradictory", "complete": True}
IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
KINDS = {"EMPTY": 1, "TRANSFER": 3, "FACT": 2, "REQUIRE": 2,
         "EXCEPT": 2, "ALIAS": 2, "CANCEL": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(public):
    require(type(public) is dict and set(public) == {"source", "query", "rules"}, "public fields only")
    require(public["rules"] == RULES and type(public["rules"]["complete"]) is bool, "exact rules")
    query = public["query"]
    require(type(query) is dict and set(query) == {"entity", "time"}, "query fields")
    require(type(query["entity"]) is str and IDENT.fullmatch(query["entity"]), "query identifier")
    require(type(query["time"]) is int and 0 <= query["time"] <= 1000000, "query cutoff")
    source = public["source"]
    require(type(source) is str and 0 < len(source) <= 8192 and len(source.encode()) <= 8192, "source bound")
    lines = source.splitlines()
    require(0 < len(lines) <= 64, "record bound")
    records, identities = [], set()
    for line in lines:
        tokens = line.split()
        require(len(tokens) >= 4 and IDENT.fullmatch(tokens[0]) and tokens[0] not in identities, "unique record ID")
        require(re.fullmatch(r"@[0-9]{1,7}", tokens[1]) is not None, "record time")
        tick, kind, args = int(tokens[1][1:]), tokens[2], tokens[3:]
        require(tick <= 1000000 and kind in KINDS and len(args) == KINDS[kind], "record kind/arity/time")
        require(all(IDENT.fullmatch(value) for value in args), "record identifiers")
        if kind == "FACT":
            require(args[1] in {"SUPPORTED", "REFUTED"}, "registered premise words")
        if kind == "TRANSFER":
            require(args[2] == "CONFIRMED", "registered confirmed modality")
        identities.add(tokens[0])
        records.append({"id": tokens[0], "time": tick, "kind": kind, "args": args})
    transfers = {row["id"]: row for row in records if row["kind"] == "TRANSFER"}
    for row in records:
        if row["kind"] in {"REQUIRE", "EXCEPT", "CANCEL"}:
            require(row["args"][0] in transfers, "typed transfer reference")
        if row["kind"] == "CANCEL":
            require(row["time"] >= transfers[row["args"][0]]["time"], "cancellation chronology")
    aliases = {}
    for row in records:
        if row["kind"] == "ALIAS" and row["time"] <= query["time"]:
            aliases.setdefault(row["args"][0], set()).add(row["args"][1])
    require(all(len(values) == 1 for values in aliases.values()), "unique aliases")
    mapping = {name: next(iter(values)) for name, values in aliases.items()}
    for name in mapping:
        resolve(name, mapping)
    return records, mapping


def resolve(name, mapping):
    visited = set()
    while name in mapping:
        require(name not in visited and len(visited) < 64, "alias cycle/limit")
        visited.add(name)
        name = mapping[name]
    return name


def public_structure(public):
    """Precall structural evidence extraction; produces no answer prediction."""
    records, aliases = parse(public)
    cutoff = public["query"]["time"]
    canceled = {r["args"][0] for r in records if r["kind"] == "CANCEL" and r["time"] <= cutoff}
    entity = resolve(public["query"]["entity"], aliases)
    candidates = sorted((r for r in records if r["kind"] == "TRANSFER" and r["id"] not in canceled
                         and r["time"] <= cutoff and resolve(r["args"][0], aliases) == entity), key=lambda r: r["time"])
    require(len(candidates) == 6 and [r["time"] for r in candidates] == [10, 20, 30, 40, 50, 60], "registered six positions")
    require(sum(r["kind"] == "EMPTY" and r["time"] == 0 and resolve(r["args"][0], aliases) == entity for r in records) == 1, "explicit empty fallback")
    observed = []
    for position in (0, 2, 4, 5):
        candidate = candidates[position]
        guards = []
        for row in records:
            if row["kind"] in {"REQUIRE", "EXCEPT"} and row["args"][0] == candidate["id"]:
                facts = [f for f in records if f["kind"] == "FACT" and f["args"][0] == row["args"][1] and f["time"] <= candidate["time"]]
                guards.append({"guard": row, "eligible_facts": facts})
        observed.append({"position": position, "candidate": candidate, "guards": guards})
    return {"records": records, "candidates": candidates, "inspected_records": observed,
            "token_bag": dict(collections.Counter(public["source"].split())), "aliases": aliases}


def full_semantic(public):
    """F: advance latest factual state to each chronological binding cutoff."""
    records, aliases = parse(public)
    cutoff = public["query"]["time"]
    target = resolve(public["query"]["entity"], aliases)
    revoked = {r["args"][0] for r in records if r["kind"] == "CANCEL" and r["time"] <= cutoff}
    bindings = sorted((r for r in records if r["kind"] in {"TRANSFER", "EMPTY"}
                       and r["time"] <= cutoff and resolve(r["args"][0], aliases) == target
                       and r["id"] not in revoked), key=lambda r: (r["time"], r["id"]))
    require(len({r["time"] for r in bindings}) == len(bindings), "registered unique binding times")
    facts = sorted((r for r in records if r["kind"] == "FACT"), key=lambda r: (r["time"], r["id"]))
    state, cursor, outcome, trace, operations = {}, 0, None, [], 0
    for binding in bindings:
        while cursor < len(facts) and facts[cursor]["time"] <= binding["time"]:
            fact = facts[cursor]
            predicate = fact["args"][0]
            previous = state.get(predicate)
            if previous is None or previous["time"] < fact["time"]:
                state[predicate] = {"time": fact["time"], "states": {fact["args"][1]}, "ids": [fact["id"]]}
            else:
                previous["states"].add(fact["args"][1]); previous["ids"].append(fact["id"])
            cursor += 1; operations += 1
        excluded, conflicting, absent, guards = False, False, False, []
        for condition in records:
            if condition["kind"] not in {"REQUIRE", "EXCEPT"} or condition["args"][0] != binding["id"]:
                continue
            premise = state.get(condition["args"][1])
            status = ("UNKNOWN" if premise is None else "CONTRADICTORY" if len(premise["states"]) > 1 else next(iter(premise["states"])))
            excluded |= condition["kind"] == "REQUIRE" and status == "REFUTED" or condition["kind"] == "EXCEPT" and status == "SUPPORTED"
            conflicting |= status == "CONTRADICTORY"; absent |= status == "UNKNOWN"
            guards.append({"id": condition["id"], "relation": condition["kind"], "predicate": condition["args"][1],
                           "cutoff": binding["time"], "state": status, "latest_fact_ids": [] if premise is None else list(premise["ids"])})
            operations += 1
        disposition = "EXCLUDED" if excluded else "CONTRADICTORY" if conflicting else "UNKNOWN" if absent else "REFUTED" if binding["kind"] == "EMPTY" else "SUPPORTED"
        if not excluded:
            outcome = {"status": disposition, "value": resolve(binding["args"][1], aliases) if disposition == "SUPPORTED" else None}
        trace.append({"binding": binding["id"], "time": binding["time"], "disposition": disposition, "guards": guards})
        operations += 1
    require(outcome is not None, "no applicable binding")
    return {"answer": outcome, "parsed_events": records, "guard_trace": trace, "operations": operations,
            "source_bytes_scanned": len(public["source"].encode()), "guard_evaluations": sum(len(t["guards"]) for t in trace)}


def watched(public, kind):
    """L/P: subset semantics independently implemented, never call F."""
    require(kind in {"L", "P"}, "registered control")
    records, aliases = parse(public)
    tick = public["query"]["time"]
    entity = resolve(public["query"]["entity"], aliases)
    canceled = {row["args"][0] for row in records if row["kind"] == "CANCEL" and row["time"] <= tick}
    candidates = sorted((r for r in records if r["kind"] == "TRANSFER" and r["time"] <= tick
                         and r["id"] not in canceled and resolve(r["args"][0], aliases) == entity), key=lambda r: r["time"])
    require(len(candidates) == 6, "registered N")
    N = len(candidates)
    positions = [N - 1, min(N - 2, max(1, N - 4))] if kind == "L" else sorted({N - 1, N - 2, 0}, reverse=True)
    trace, winner, operations = [], None, 0
    for position in positions:
        candidate, evidence = candidates[position], []
        for condition in records:
            if condition["kind"] in {"REQUIRE", "EXCEPT"} and condition["args"][0] == candidate["id"]:
                eligible = [f for f in records if f["kind"] == "FACT" and f["args"][0] == condition["args"][1] and f["time"] <= candidate["time"]]
                latest = max((f["time"] for f in eligible), default=None)
                used = [f for f in eligible if f["time"] == latest]
                states = {f["args"][1] for f in used}
                status = "UNKNOWN" if not states else next(iter(states)) if len(states) == 1 else "CONTRADICTORY"
                evidence.append({"id": condition["id"], "relation": condition["kind"], "predicate": condition["args"][1],
                                 "cutoff": candidate["time"], "state": status, "latest_fact_ids": [f["id"] for f in used]})
                operations += 1 + len(eligible)
        excluded = any(g["relation"] == "REQUIRE" and g["state"] == "REFUTED" or g["relation"] == "EXCEPT" and g["state"] == "SUPPORTED" for g in evidence)
        disposition = ("EXCLUDED" if excluded else "CONTRADICTORY" if any(g["state"] == "CONTRADICTORY" for g in evidence)
                       else "UNKNOWN" if any(g["state"] == "UNKNOWN" for g in evidence) else "SUPPORTED")
        trace.append({"position": position, "binding": candidate["id"], "disposition": disposition, "guards": evidence})
        operations += 1
        if position == N - 1 and disposition in {"UNKNOWN", "CONTRADICTORY"}:
            winner = {"status": disposition, "value": None}
            break
        if disposition == "SUPPORTED" and (kind == "P" or position != N - 1):
            winner = {"status": "SUPPORTED", "value": resolve(candidate["args"][1], aliases)}
            break
    return {"answer": winner if winner is not None else {"status": "REFUTED", "value": None},
            "parsed_events": records, "guard_trace": trace, "operations": operations,
            "candidate_count": N, "planned_positions": positions, "source_bytes_scanned": len(public["source"].encode()),
            "guard_evaluations": sum(len(t["guards"]) for t in trace)}
