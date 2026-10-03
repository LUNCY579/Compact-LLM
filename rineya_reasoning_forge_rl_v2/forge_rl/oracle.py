"""Separate audit algorithms, deliberately not importing the constructive solvers.

Routes: Bellman-Ford over augmented states rather than Dijkstra.
Workflow: reverse value iteration rather than forward Dijkstra.
Scheduling: exhaustive prefix DFS rather than subset-dominance DP.
Selection: recursive subset traversal rather than Cartesian-product enumeration.
Replay: independently implemented state transitions.
The shared task grammar and shared authorship remain common-mode risks.
"""
from __future__ import annotations
import math
from typing import Any
from .common import integer


def check_world(w: dict[str, Any]) -> None:
    """Reject malformed/unbounded input before executing the bounded oracles."""
    f, q, r = w["family"], w["query"], w["records"]
    if not isinstance(q, dict) or not isinstance(r, list) or not 1 <= len(r) <= 3000:
        raise ValueError("Invalid world shape")
    def ints(obj: dict[str, Any], fields: list[str], lo: int = -10**9, hi: int = 10**9) -> None:
        if any(not integer(obj.get(k)) or not lo <= obj[k] <= hi for k in fields):
            raise ValueError("Integer field outside supported range")
    def unique(values: list[Any]) -> None:
        if len(set(values)) != len(values):
            raise ValueError("Duplicate logical record identity")
    if f in ("ledger_replay", "register_machine"):
        field = "account" if f == "ledger_replay" else "register"
        names = set("ABCD" if f == "ledger_replay" else "XYZ")
        initial = [x for x in r if x.get("kind") == "initial"]
        if len(initial) != len(names) or {x.get(field) for x in initial} != names:
            raise ValueError("Initial state is incomplete")
        ints(q, ["cutoff"], 0, 2000)
        if q.get(field) not in names:
            raise ValueError("Invalid target")
        if f == "register_machine" and q.get("modulus") != 97:
            raise ValueError("Unexpected modulus")
        for x in initial:
            ints(x, ["amount" if f == "ledger_replay" else "value"])
        events = [x for x in r if x.get("kind") != "initial"]
        unique([x.get("tick") for x in events])
        for x in events:
            ints(x, ["tick"], 1, 2000)
            if x.get(field) not in names or x.get("other") not in names or x[field] == x["other"]:
                raise ValueError("Invalid operands")
            if f == "ledger_replay":
                ints(x, ["amount"], -100, 100)
                if x.get("kind") not in ("add", "scale", "transfer") or x.get("status") not in ("posted", "void"):
                    raise ValueError("Invalid ledger record")
            else:
                ints(x, ["value", "threshold"], -100, 100)
                if x.get("kind") not in ("add", "mix", "swap", "conditional"):
                    raise ValueError("Invalid instruction")
    elif f in ("temporal_route", "continuous_replan"):
        nodes = q.get("nodes")
        if not isinstance(nodes, list) or not 2 <= len(nodes) <= 12 or any(not isinstance(x, str) for x in nodes):
            raise ValueError("Invalid node list")
        unique(nodes)
        if q.get("start") not in nodes or q.get("goal") not in nodes or len(q.get("waypoints", [])) > 4 or any(x not in nodes for x in q["waypoints"]):
            raise ValueError("Invalid route query")
        ints(q, ["cutoff"], 0, 10000)
        unique([(x.get("src"), x.get("dst"), x.get("tick")) for x in r])
        for x in r:
            ints(x, ["cost"], 1, 1000); ints(x, ["tick"], 0, 10000)
            if x.get("kind") != "edge" or x.get("src") not in nodes or x.get("dst") not in nodes or type(x.get("open")) is not bool or x.get("status") not in ("approved", "draft"):
                raise ValueError("Invalid edge")
    elif f == "prerequisite_plan":
        ints(q, ["bits"], 1, 8)
        maximum = (1 << q["bits"]) - 1
        ints(q, ["initial", "goal_on", "goal_off"], 0, maximum)
        if q["goal_on"] & q["goal_off"] or len(r) > 24:
            raise ValueError("Invalid workflow")
        unique([x.get("name") for x in r])
        for x in r:
            ints(x, ["pre_on", "pre_off", "add", "delete"], 0, maximum)
            ints(x, ["cost"], 1, 100)
            if x.get("kind") != "action" or not isinstance(x.get("name"), str) or x["pre_on"] & x["pre_off"]:
                raise ValueError("Invalid action")
    elif f == "deadline_schedule":
        ints(q, ["start_time"], 0, 100)
        names = [x.get("name") for x in r]
        unique(names)
        if len(r) > 8 or any(not isinstance(n, str) for n in names):
            raise ValueError("Invalid jobs")
        for x in r:
            ints(x, ["duration"], 1, 100); ints(x, ["release", "deadline"], 0, 1000)
            if x.get("kind") != "job" or not isinstance(x.get("after"), list) or any(p not in names for p in x["after"]):
                raise ValueError("Invalid prerequisite")
    elif f == "budget_selection":
        ints(q, ["cutoff"], 0, 10000)
        items = [x for x in r if x.get("kind") == "item"]
        names = [x.get("name") for x in items]
        unique(names)
        if not 1 <= len(items) <= 12 or any(not isinstance(n, str) for n in names):
            raise ValueError("Invalid item set")
        for x in r:
            if x.get("kind") == "item":
                ints(x, ["cost", "weight", "value"], 1, 1000)
                if any(not isinstance(x.get(k), list) or any(n not in names for n in x[k]) for k in ("requires", "excludes")):
                    raise ValueError("Invalid item constraints")
            elif x.get("kind") == "limit":
                ints(x, ["tick", "value"], 0, 10000)
                if x.get("key") not in ("budget", "capacity") or x.get("status") not in ("approved", "draft"):
                    raise ValueError("Invalid limit")
            else:
                raise ValueError("Invalid selection record")
        unique([(x["key"], x["tick"]) for x in r if x["kind"] == "limit"])
    elif f == "evidence_chain":
        ints(q, ["cutoff"], 0, 10000)
        if not isinstance(q.get("start"), str) or not isinstance(q.get("path"), list) or not 1 <= len(q["path"]) <= 8 or any(x not in ("owner", "next", "site") for x in q["path"]):
            raise ValueError("Invalid relation query")
        unique([(x.get("entity"), x.get("key"), x.get("tick")) for x in r])
        for x in r:
            ints(x, ["tick"], 0, 10000)
            if x.get("kind") != "field" or any(not isinstance(x.get(k), str) for k in ("entity", "key", "value")) or x["key"] not in ("owner", "next", "site") or x.get("status") not in ("approved", "draft"):
                raise ValueError("Invalid evidence record")
    else:
        raise ValueError("Unrecognized task family")


def replay_transition(f: str, before: Any, event: dict[str, Any]) -> Any:
    """Independent transition checker also used by process-label auditing."""
    if f == "prerequisite_plan":
        flags = {i for i in range(8) if before & (2 ** i)}
        required = {i for i in range(8) if event["pre_on"] & (2 ** i)}
        forbidden = {i for i in range(8) if event["pre_off"] & (2 ** i)}
        if not required <= flags or forbidden & flags:
            raise ValueError("Inapplicable action")
        flags -= {i for i in range(8) if event["delete"] & (2 ** i)}
        flags |= {i for i in range(8) if event["add"] & (2 ** i)}
        return sum(2 ** i for i in flags)
    state = dict(before)
    if f == "ledger_replay":
        if event["status"] == "posted":
            a, b, n = event["account"], event["other"], event["amount"]
            if event["kind"] == "transfer":
                delta = {k: 0 for k in state}
                delta[a] -= n; delta[b] += n
                state = {k: state[k] + delta[k] for k in state}
            elif event["kind"] == "scale":
                state[a] = n * before[a]
            elif event["kind"] == "add":
                state[a] = n + before[a]
            else:
                raise ValueError("Unknown operation")
        return state
    if f == "register_machine":
        a, b, n = event["register"], event["other"], event["value"]
        op = event["kind"]
        if op == "swap":
            state[a] = before[b]; state[b] = before[a]
        else:
            increment = n if op == "add" else 2 * before[b] + n if op == "mix" else n if before[b] > event["threshold"] else -n
            state[a] = (before[a] + increment) - 97 * ((before[a] + increment) // 97)
        return state
    raise ValueError("No transition checker for this family")


def resolved_edges(w: dict[str, Any]) -> dict[tuple[str, str], int]:
    candidates = {}
    for e in w["records"]:
        if e["status"] != "approved" or e["tick"] > w["query"]["cutoff"]:
            continue
        k = (e["src"], e["dst"])
        if k not in candidates or e["tick"] > candidates[k]["tick"]:
            candidates[k] = e
    return {k: e["cost"] for k, e in candidates.items() if e["open"]}


def resolved_limits(w: dict[str, Any]) -> dict[str, int]:
    out = {}
    for key in ("budget", "capacity"):
        matches = [r for r in w["records"] if r["kind"] == "limit" and r["key"] == key and r["status"] == "approved" and r["tick"] <= w["query"]["cutoff"]]
        if matches:
            out[key] = max(matches, key=lambda r: r["tick"])["value"]
    return out


def audit_answer(w: dict[str, Any]) -> dict[str, Any]:
    """Return exact answer or objective certificate, without a canonical plan."""
    check_world(w)
    f, q = w["family"], w["query"]
    if f in ("ledger_replay", "register_machine"):
        name, val = ("account", "amount") if f == "ledger_replay" else ("register", "value")
        state = {x[name]: x[val] for x in w["records"] if x["kind"] == "initial"}
        events = {x["tick"]: x for x in w["records"] if x["kind"] != "initial"}
        for tick in sorted(events):
            if tick <= q["cutoff"]:
                state = replay_transition(f, state, events[tick])
        return {"value": state[q[name]]}
    if f in ("temporal_route", "continuous_replan"):
        way = q["waypoints"]
        states = [(node, k) for node in q["nodes"] for k in range(len(way) + 1)]
        initial = (q["start"], int(bool(way) and q["start"] == way[0]))
        distance = {s: math.inf for s in states}
        distance[initial] = 0
        edges = resolved_edges(w)
        for _ in range(len(states) - 1):
            previous = distance.copy()
            for (a, b), c in edges.items():
                for k in range(len(way) + 1):
                    k2 = k + 1 if k < len(way) and b == way[k] else k
                    distance[b, k2] = min(distance[b, k2], previous[a, k] + c)
            if distance == previous:
                break
        cost = distance[q["goal"], len(way)]
        return {"status": "IMPOSSIBLE"} if cost == math.inf else {"optimal_cost": int(cost)}
    if f == "prerequisite_plan":
        states = list(range(1 << q["bits"]))
        values = {s: (0 if (s | q["goal_on"]) == s and (s & q["goal_off"]) == 0 else math.inf) for s in states}
        # Precompute a transition graph with the independently coded set semantics.
        graph = {}
        for state in states:
            edges = []
            for action in w["records"]:
                try:
                    nxt = replay_transition(f, state, action)
                except ValueError:
                    continue
                edges.append((nxt, action["cost"]))
            graph[state] = edges
        for _ in range(len(states) - 1):
            old = values.copy()
            for s in states:
                for nxt, cost in graph[s]:
                    values[s] = min(values[s], cost + old[nxt])
            if values == old:
                break
        val = values[q["initial"]]
        return {"status": "IMPOSSIBLE"} if val == math.inf else {"optimal_cost": int(val)}
    if f == "deadline_schedule":
        best = math.inf
        jobs = {j["name"]: j for j in w["records"]}
        def visit(done: frozenset[str], time: int) -> None:
            nonlocal best
            if time >= best:
                return
            if len(done) == len(jobs):
                best = time
                return
            for name, job in jobs.items():
                if name in done or not set(job["after"]) <= done:
                    continue
                end = max(time, job["release"]) + job["duration"]
                if end <= job["deadline"]:
                    visit(done | {name}, end)
        visit(frozenset(), q["start_time"])
        return {"status": "IMPOSSIBLE"} if best == math.inf else {"optimal_finish": int(best)}
    if f == "budget_selection":
        limits = resolved_limits(w)
        if len(limits) != 2:
            return {"status": "UNKNOWN"}
        items = [r for r in w["records"] if r["kind"] == "item"]
        best = (0, 0)
        def visit(i: int, chosen: frozenset[str], cost: int, weight: int, value: int) -> None:
            nonlocal best
            if cost > limits["budget"] or weight > limits["capacity"]:
                return
            if i == len(items):
                included = [r for r in items if r["name"] in chosen]
                if all(set(r["requires"]) <= chosen and not (set(r["excludes"]) & chosen) for r in included):
                    best = min(best, (-value, cost))
                return
            visit(i + 1, chosen, cost, weight, value)
            item = items[i]
            visit(i + 1, chosen | {item["name"]}, cost + item["cost"], weight + item["weight"], value + item["value"])
        visit(0, frozenset(), 0, 0, 0)
        return {"optimal_value": -best[0], "optimal_cost": best[1]}
    if f == "evidence_chain":
        current = q["start"]
        for field in q["path"]:
            candidates = [x for x in w["records"] if x["entity"] == current and x["key"] == field and x["status"] == "approved" and x["tick"] <= q["cutoff"]]
            if not candidates:
                return {"status": "UNKNOWN"}
            current = max(candidates, key=lambda r: r["tick"])["value"]
        return {"answer": current}
    raise ValueError("Unknown family")


def verify_solution(w: dict[str, Any], obj: Any, cert: dict[str, Any] | None = None) -> dict[str, Any]:
    """Semantic witness checks accept alternative optimal plans, but never extra keys.

    `cert` is a trusted verifier-side certificate, never a model-provided value.
    Feasibility is diagnostic only; the default reward is exact outcome correctness.
    """
    cert = audit_answer(w) if cert is None else cert
    result = {"schema_valid": False, "feasible": False, "correct": False, "reason": "schema"}
    if not isinstance(obj, dict):
        return result
    f, q = w["family"], w["query"]
    if set(obj) == {"status"}:
        result["schema_valid"] = obj["status"] in ("UNKNOWN", "IMPOSSIBLE")
        result["correct"] = result["schema_valid"] and obj == cert
        result["reason"] = "correct_status" if result["correct"] else "incorrect_status"
        return result
    if f in ("ledger_replay", "register_machine", "evidence_chain"):
        key = "answer" if f == "evidence_chain" else "value"
        valid = set(obj) == {key} and (isinstance(obj[key], str) if key == "answer" else integer(obj[key]))
        result.update(schema_valid=valid, feasible=valid and obj == cert, correct=valid and obj == cert,
                      reason="exact_match" if valid and obj == cert else "incorrect_value")
        return result
    listkey = "path" if f in ("temporal_route", "continuous_replan") else "actions" if f == "prerequisite_plan" else "order" if f == "deadline_schedule" else "items"
    numberkeys = ["finish"] if f == "deadline_schedule" else ["value", "cost"] if f == "budget_selection" else ["cost"]
    if set(obj) != {listkey, *numberkeys} or not isinstance(obj.get(listkey), list) or len(obj[listkey]) > 512 or any(not isinstance(v, str) for v in obj[listkey]) or any(not integer(obj.get(k)) or obj[k] < 0 for k in numberkeys):
        return result
    result["schema_valid"] = True
    if "status" in cert:
        result["reason"] = "answer_requires_status"
        return result
    sequence = obj[listkey]
    if f in ("temporal_route", "continuous_replan"):
        edges = resolved_edges(w)
        if not sequence or sequence[0] != q["start"] or sequence[-1] != q["goal"]:
            return result
        cost = 0; k = 0
        for node in sequence:
            if k < len(q["waypoints"]) and node == q["waypoints"][k]:
                k += 1
        for edge in zip(sequence, sequence[1:]):
            if edge not in edges:
                return result
            cost += edges[edge]
        result["feasible"] = k == len(q["waypoints"]) and cost == obj["cost"]
        result["correct"] = result["feasible"] and cost == cert["optimal_cost"]
    elif f == "prerequisite_plan":
        actions = {a["name"]: a for a in w["records"]}
        state = q["initial"]; cost = 0
        for name in sequence:
            if name not in actions:
                return result
            try:
                state = replay_transition(f, state, actions[name])
            except ValueError:
                return result
            cost += actions[name]["cost"]
        result["feasible"] = (state | q["goal_on"]) == state and not state & q["goal_off"] and cost == obj["cost"]
        result["correct"] = result["feasible"] and cost == cert["optimal_cost"]
    elif f == "deadline_schedule":
        jobs = {j["name"]: j for j in w["records"]}
        if set(sequence) != set(jobs) or len(sequence) != len(jobs):
            return result
        done: set[str] = set(); now = q["start_time"]
        for name in sequence:
            job = jobs[name]
            if not set(job["after"]) <= done:
                return result
            now = max(now, job["release"]) + job["duration"]
            if now > job["deadline"]:
                return result
            done.add(name)
        result["feasible"] = now == obj["finish"]
        result["correct"] = result["feasible"] and now == cert["optimal_finish"]
    else:
        items = {i["name"]: i for i in w["records"] if i["kind"] == "item"}
        chosen = set(sequence)
        if len(chosen) != len(sequence) or not chosen <= set(items):
            return result
        selected = [items[n] for n in chosen]
        cost = sum(i["cost"] for i in selected); weight = sum(i["weight"] for i in selected); value = sum(i["value"] for i in selected)
        limits = resolved_limits(w)
        if cost > limits["budget"] or weight > limits["capacity"] or any(not set(i["requires"]) <= chosen or set(i["excludes"]) & chosen for i in selected):
            return result
        result["feasible"] = cost == obj["cost"] and value == obj["value"]
        result["correct"] = result["feasible"] and cost == cert["optimal_cost"] and value == cert["optimal_value"]
    result["reason"] = "optimal_verified" if result["correct"] else "feasible_suboptimal" if result["feasible"] else "invalid_witness"
    return result


def verify_trace(w: dict[str, Any], answer: dict[str, Any], steps: list[dict[str, Any]]) -> bool:
    """Reconstruct the complete reference witness trace from visible records.

    This validates structured checkpoints and arithmetic, not the faithfulness
    of an arbitrary model-written prose chain or an exhaustive search transcript.
    """
    f, q = w["family"], w["query"]
    expected = []
    if f in ("ledger_replay", "register_machine"):
        field, value = ("account", "amount") if f == "ledger_replay" else ("register", "value")
        state = {r[field]: r[value] for r in w["records"] if r["kind"] == "initial"}
        for e in sorted((r for r in w["records"] if r["kind"] != "initial" and r["tick"] <= q["cutoff"]), key=lambda r: r["tick"]):
            after = replay_transition(f, state, e)
            expected.append({"tick": e["tick"], "operation": e["kind"], "before": state.copy(), "after": after.copy(), "event": e.copy()})
            state = after
        if answer != {"value": state[q[field]]}:
            return False
    elif f in ("temporal_route", "continuous_replan") and "path" in answer:
        edges = resolved_edges(w); cost = 0
        for a, b in zip(answer["path"], answer["path"][1:]):
            if (a, b) not in edges:
                return False
            cost += edges[a, b]
            expected.append({"from": a, "to": b, "edge_cost": edges[a, b], "cumulative_cost": cost})
    elif f == "prerequisite_plan" and "actions" in answer:
        state = q["initial"]; cost = 0
        actions = {a["name"]: a for a in w["records"]}
        for name in answer["actions"]:
            if name not in actions:
                return False
            action = actions[name]
            try:
                after = replay_transition(f, state, action)
            except ValueError:
                return False
            cost += action["cost"]
            expected.append({"action": name, "before": state, "after": after, "cumulative_cost": cost, "event": action.copy()})
            state = after
    elif f == "deadline_schedule" and "order" in answer:
        jobs = {r["name"]: r for r in w["records"]}; time = q["start_time"]
        for name in answer["order"]:
            if name not in jobs:
                return False
            job = jobs[name]; start = max(time, job["release"]); time = start + job["duration"]
            expected.append({"job": name, "start": start, "finish": time, "deadline": job["deadline"], "after": job["after"]})
    elif f == "budget_selection" and "items" in answer:
        items = {r["name"]: r for r in w["records"] if r["kind"] == "item"}
        cost = weight = value = 0
        for name in answer["items"]:
            if name not in items:
                return False
            r = items[name]; cost += r["cost"]; weight += r["weight"]; value += r["value"]
            expected.append({"item": name, "cumulative_cost": cost, "cumulative_weight": weight, "cumulative_value": value})
    elif f == "evidence_chain":
        current = q["start"]
        for field in q["path"]:
            matching = [r for r in w["records"] if r["entity"] == current and r["key"] == field and r["status"] == "approved" and r["tick"] <= q["cutoff"]]
            if not matching:
                expected.append({"entity": current, "field": field, "status": "missing_approved_evidence"})
                break
            record = max(matching, key=lambda r: r["tick"])
            expected.append({"entity": current, "field": field, "value": record["value"], "tick": record["tick"]})
            current = record["value"]
    return expected == steps
