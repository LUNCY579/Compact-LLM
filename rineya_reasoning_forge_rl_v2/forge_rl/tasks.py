"""World generation and constructive solvers; no imports from the audit oracle.

All action/state spaces are finite. Costs and durations are positive. Traces are
solver-authored demonstrations, not transcripts of any language model's thoughts.
"""
from __future__ import annotations
import copy
import heapq
import itertools
import random
from typing import Any
from .common import FAMILIES, dumps


def make_world(family: str, rng: random.Random, tier: int, transfer: bool = False) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    w: dict[str, Any] = {"family": family, "query": {}, "records": []}
    r = w["records"]
    if family == "ledger_replay":
        names = list("ABCD")
        for a in names:
            r.append({"kind": "initial", "account": a, "amount": rng.randint(-20, 40)})
        n = (16, 32, 64, 128, 256, 512)[tier] * (2 if transfer else 1)
        for t in range(1, n + 1):
            a, b = rng.sample(names, 2)
            op = rng.choice(["add", "transfer", "transfer", "scale"])
            r.append({"kind": op, "tick": t, "account": a, "other": b,
                      "amount": rng.choice([-2, -1, 1, 2]) if op == "scale" else rng.randint(-9, 12),
                      "status": "void" if rng.random() < .12 else "posted"})
        w["query"] = {"account": rng.choice(names), "cutoff": n - rng.randint(0, min(n // 5, 12))}
    elif family == "register_machine":
        for a in "XYZ":
            r.append({"kind": "initial", "register": a, "value": rng.randrange(97)})
        n = (12, 24, 48, 96, 192, 384)[tier] * (2 if transfer else 1)
        ops = ["add", "mix", "swap", "conditional"]
        for t in range(1, n + 1):
            a, b = rng.sample(list("XYZ"), 2)
            r.append({"kind": rng.choice(ops), "tick": t, "register": a, "other": b,
                      "value": rng.randint(-8, 11), "threshold": rng.randrange(97)})
        w["query"] = {"register": rng.choice(list("XYZ")), "cutoff": n, "modulus": 97}
    elif family in ("temporal_route", "continuous_replan"):
        n = 9 if transfer else rng.choice([6, 7, 8])
        nodes = [f"N{i}" for i in range(n)]
        start, goal = rng.sample(nodes, 2)
        count = 0
        for a in nodes:
            for b in nodes:
                if a != b and rng.random() < (.25 if transfer else .31):
                    count += 1
                    r.append({"kind": "edge", "src": a, "dst": b, "cost": rng.randint(1, 12),
                              "tick": count, "status": "approved", "open": True})
        for _ in range(rng.randint(3, 7)):
            if not r:
                break
            old = rng.choice(r)
            count += 1
            r.append({"kind": "edge", "src": old["src"], "dst": old["dst"], "cost": rng.randint(1, 14),
                      "tick": count, "status": rng.choice(["draft", "approved", "approved"]),
                      "open": rng.random() > .35})
        waypoints = rng.sample([x for x in nodes if x not in (start, goal)], 2 if transfer else rng.randrange(2))
        cutoff = max(1, count - rng.randrange(3))
        w["query"] = {"nodes": nodes, "start": start, "goal": goal, "waypoints": waypoints, "cutoff": cutoff}
        if family == "continuous_replan":
            # Four causal snapshots. The tape is private until a stage is reached.
            # Teacher actions take one edge of the current optimal route (or WAIT).
            w["query"]["waypoints"] = []
            base = copy.deepcopy(w)
            base["family"] = "temporal_route"
            stages = []
            current = start
            for stage in range(4):
                base["query"]["start"] = current
                base["query"]["cutoff"] = count
                snapshot = copy.deepcopy(base)
                ans, _ = solve(snapshot)
                stages.append({"stage": stage, "world": snapshot, "reference": ans,
                               "observation": "Initial dispatch" if stage == 0 else "Use the new approved records; previous plans may be stale."})
                if ans.get("path") and len(ans["path"]) > 1:
                    current = ans["path"][1]
                else:
                    current = rng.choice(nodes)
                if stage < 3:
                    for _ in range(2):
                        a, b = rng.sample(nodes, 2)
                        count += 1
                        base["records"].append({"kind": "edge", "src": a, "dst": b, "cost": rng.randint(1, 14),
                                                "tick": count, "status": "approved", "open": rng.random() > .3})
            # Canonical task is the final decision, not a question containing future tape.
            w["records"] = copy.deepcopy(stages[-1]["world"]["records"])
            w["query"] = copy.deepcopy(stages[-1]["world"]["query"])
            w["episode"] = stages
    elif family == "prerequisite_plan":
        n = 8 if transfer else rng.choice([5, 6, 7])
        initial = rng.randrange(1 << n)
        goal_on = sum(1 << x for x in rng.sample(range(n), rng.randint(2, min(4, n))))
        goal_off = (1 << n) - 1 - goal_on if transfer else 0
        for i in range(n * 2 + 2):
            pre_on = sum(1 << b for b in rng.sample(range(n), rng.randrange(3)))
            pre_off = sum(1 << b for b in rng.sample(range(n), rng.randrange(2))) & ~pre_on
            add = 1 << rng.randrange(n)
            delete = (1 << rng.randrange(n)) & ~add if rng.random() < .6 else 0
            r.append({"kind": "action", "name": f"A{i}", "pre_on": pre_on, "pre_off": pre_off,
                      "add": add, "delete": delete, "cost": rng.randint(1, 7)})
        w["query"] = {"bits": n, "initial": initial, "goal_on": goal_on, "goal_off": goal_off}
    elif family == "deadline_schedule":
        n = 8 if transfer else rng.choice([5, 6, 7])
        for i in range(n):
            parents = rng.sample(range(i), min(i, rng.randrange(3)))
            r.append({"kind": "job", "name": f"J{i}", "duration": rng.randint(1, 5),
                      "release": rng.randint(0, 10), "deadline": rng.randint(12, n * 5 + 8),
                      "after": [f"J{p}" for p in parents]})
        w["query"] = {"start_time": rng.randrange(4)}
    elif family == "budget_selection":
        n = 12 if transfer else rng.choice([7, 8, 9])
        for i in range(n):
            r.append({"kind": "item", "name": f"I{i}", "cost": rng.randint(1, 10),
                      "weight": rng.randint(1, 8), "value": rng.randint(1, 20),
                      "requires": [f"I{rng.randrange(i)}"] if i and rng.random() < .3 else [],
                      "excludes": [f"I{rng.randrange(i)}"] if i and rng.random() < .25 else []})
        t = 0
        missing = rng.random() < .15
        for key in ("budget", "capacity"):
            for version in range(3):
                t += 1
                r.append({"kind": "limit", "key": key, "tick": t,
                          "value": rng.randint(n, n * 3),
                          "status": "draft" if (missing and key == "capacity") or version == 2 else "approved"})
        w["query"] = {"cutoff": 5}
    elif family == "evidence_chain":
        n = 9 if transfer else 6
        nodes = [f"E{i}" for i in range(n)]
        t = 0
        fields = ["owner", "next", "site"]
        for node in nodes:
            for key in fields:
                for v in range(rng.choice([1, 2, 3])):
                    t += 1
                    r.append({"kind": "field", "entity": node, "key": key, "tick": t,
                              "value": rng.choice(nodes) if key != "site" else f"S{rng.randrange(9)}",
                              "status": "draft" if rng.random() < .22 else "approved"})
        path = [rng.choice(["owner", "next"]) for _ in range(6 if transfer else rng.randint(2, 4))] + ["site"]
        w["query"] = {"start": rng.choice(nodes), "path": path, "cutoff": max(1, t - rng.randint(0, 10))}
    return w


def active_edges(w: dict[str, Any]) -> dict[tuple[str, str], int]:
    selected = {}
    for e in sorted(w["records"], key=lambda x: x["tick"]):
        if e["status"] == "approved" and e["tick"] <= w["query"]["cutoff"]:
            selected[e["src"], e["dst"]] = e
    return {key: e["cost"] for key, e in selected.items() if e["open"]}


def route(w: dict[str, Any]) -> dict[str, Any]:
    q = w["query"]
    edges = active_edges(w)
    way = q["waypoints"]
    def advance(node: str, k: int) -> int:
        return k + 1 if k < len(way) and node == way[k] else k
    initial = (q["start"], advance(q["start"], 0))
    heap = [(0, (q["start"],), initial)]
    best = {initial: (0, (q["start"],))}
    while heap:
        cost, path, state = heapq.heappop(heap)
        if best.get(state) != (cost, path):
            continue
        node, k = state
        if node == q["goal"] and k == len(way):
            return {"path": list(path), "cost": cost}
        for (a, b), c in sorted(edges.items()):
            if a != node:
                continue
            nxt = (b, advance(b, k))
            value = (cost + c, path + (b,))
            if nxt not in best or value < best[nxt]:
                best[nxt] = value
                heapq.heappush(heap, (value[0], value[1], nxt))
    return {"status": "IMPOSSIBLE"}


def applicable(state: int, a: dict[str, Any]) -> bool:
    return state & a["pre_on"] == a["pre_on"] and not state & a["pre_off"]


def apply_action(state: int, a: dict[str, Any]) -> int:
    if not applicable(state, a):
        raise ValueError("Action precondition is false")
    return (state & ~a["delete"]) | a["add"]


def is_goal(state: int, q: dict[str, Any]) -> bool:
    return state & q["goal_on"] == q["goal_on"] and not state & q["goal_off"]


def plan(w: dict[str, Any]) -> dict[str, Any]:
    q = w["query"]
    heap = [(0, (), q["initial"])]
    best = {q["initial"]: (0, ())}
    while heap:
        cost, path, state = heapq.heappop(heap)
        if best.get(state) != (cost, path):
            continue
        if is_goal(state, q):
            return {"actions": list(path), "cost": cost}
        for a in sorted(w["records"], key=lambda a: a["name"]):
            if applicable(state, a):
                nxt = apply_action(state, a)
                value = (cost + a["cost"], path + (a["name"],))
                if nxt not in best or value < best[nxt]:
                    best[nxt] = value
                    heapq.heappush(heap, (value[0], value[1], nxt))
    return {"status": "IMPOSSIBLE"}


def schedule(w: dict[str, Any]) -> dict[str, Any]:
    jobs = sorted(w["records"], key=lambda j: j["name"])
    ids = {j["name"]: i for i, j in enumerate(jobs)}
    dp = {0: (w["query"]["start_time"], ())}
    for mask in range(1 << len(jobs)):
        if mask not in dp:
            continue
        now, order = dp[mask]
        for i, j in enumerate(jobs):
            if mask & (1 << i) or any(not mask & (1 << ids[p]) for p in j["after"]):
                continue
            finish = max(now, j["release"]) + j["duration"]
            if finish > j["deadline"]:
                continue
            nxt = mask | (1 << i)
            val = (finish, order + (j["name"],))
            if nxt not in dp or val < dp[nxt]:
                dp[nxt] = val
    out = dp.get((1 << len(jobs)) - 1)
    return {"order": list(out[1]), "finish": out[0]} if out else {"status": "IMPOSSIBLE"}


def selection(w: dict[str, Any]) -> dict[str, Any]:
    limits = {}
    for r in sorted((r for r in w["records"] if r["kind"] == "limit"), key=lambda r: r["tick"]):
        if r["status"] == "approved" and r["tick"] <= w["query"]["cutoff"]:
            limits[r["key"]] = r["value"]
    if set(limits) != {"budget", "capacity"}:
        return {"status": "UNKNOWN"}
    items = sorted((r for r in w["records"] if r["kind"] == "item"), key=lambda r: r["name"])
    best = (0, 0, ())
    for flags in itertools.product([False, True], repeat=len(items)):
        chosen = [x for x, flag in zip(items, flags) if flag]
        names = {x["name"] for x in chosen}
        cost = sum(x["cost"] for x in chosen)
        weight = sum(x["weight"] for x in chosen)
        if cost > limits["budget"] or weight > limits["capacity"]:
            continue
        if any(not set(x["requires"]) <= names or set(x["excludes"]) & names for x in chosen):
            continue
        value = sum(x["value"] for x in chosen)
        rank = (-value, cost, tuple(sorted(names)))
        if rank < best:
            best = rank
    return {"items": list(best[2]), "value": -best[0], "cost": best[1]}


def ledger_step(state: dict[str, int], e: dict[str, Any]) -> dict[str, int]:
    s = state.copy()
    a, b, v = e["account"], e["other"], e["amount"]
    if e["status"] == "void":
        return s
    if e["kind"] == "add":
        s[a] += v
    elif e["kind"] == "scale":
        s[a] *= v
    elif e["kind"] == "transfer":
        s[a] -= v
        s[b] += v
    else:
        raise ValueError("Invalid ledger operation")
    return s


def register_step(state: dict[str, int], e: dict[str, Any], mod: int) -> dict[str, int]:
    s = state.copy()
    a, b, v = e["register"], e["other"], e["value"]
    if e["kind"] == "add":
        s[a] = (s[a] + v) % mod
    elif e["kind"] == "mix":
        s[a] = (s[a] + 2 * s[b] + v) % mod
    elif e["kind"] == "swap":
        s[a], s[b] = s[b], s[a]
    elif e["kind"] == "conditional":
        if s[b] > e["threshold"]:
            s[a] = (s[a] + v) % mod
        else:
            s[a] = (s[a] - v) % mod
    else:
        raise ValueError("Invalid register operation")
    return s


def solve(w: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    f, q = w["family"], w["query"]
    steps: list[dict[str, Any]] = []
    if f in ("ledger_replay", "register_machine"):
        field = "account" if f == "ledger_replay" else "register"
        val = "amount" if f == "ledger_replay" else "value"
        state = {r[field]: r[val] for r in w["records"] if r["kind"] == "initial"}
        for e in sorted((r for r in w["records"] if r["kind"] != "initial"), key=lambda e: e["tick"]):
            if e["tick"] > q["cutoff"]:
                continue
            after = ledger_step(state, e) if f == "ledger_replay" else register_step(state, e, q["modulus"])
            steps.append({"tick": e["tick"], "operation": e["kind"], "before": state.copy(), "after": after.copy(), "event": e.copy()})
            state = after
        return {"value": state[q[field]]}, steps
    if f in ("temporal_route", "continuous_replan"):
        ans = route(w)
        if "path" in ans:
            edges = active_edges(w)
            cost = 0
            for a, b in zip(ans["path"], ans["path"][1:]):
                cost += edges[a, b]
                steps.append({"from": a, "to": b, "edge_cost": edges[a, b], "cumulative_cost": cost})
        return ans, steps
    if f == "prerequisite_plan":
        ans = plan(w)
        if "actions" in ans:
            state = q["initial"]
            cost = 0
            actions = {a["name"]: a for a in w["records"]}
            for name in ans["actions"]:
                a = actions[name]
                after = apply_action(state, a)
                cost += a["cost"]
                steps.append({"action": name, "before": state, "after": after, "cumulative_cost": cost, "event": a.copy()})
                state = after
        return ans, steps
    if f == "deadline_schedule":
        ans = schedule(w)
        if "order" in ans:
            jobs = {j["name"]: j for j in w["records"]}
            now = q["start_time"]
            for name in ans["order"]:
                job = jobs[name]
                start = max(now, job["release"])
                now = start + job["duration"]
                steps.append({"job": name, "start": start, "finish": now, "deadline": job["deadline"], "after": job["after"]})
        return ans, steps
    if f == "budget_selection":
        ans = selection(w)
        if "items" in ans:
            cost = weight = value = 0
            byname = {r["name"]: r for r in w["records"] if r["kind"] == "item"}
            for name in ans["items"]:
                item = byname[name]
                cost += item["cost"]; weight += item["weight"]; value += item["value"]
                steps.append({"item": name, "cumulative_cost": cost, "cumulative_weight": weight, "cumulative_value": value})
        return ans, steps
    if f == "evidence_chain":
        selected = {}
        for r in sorted(w["records"], key=lambda r: r["tick"]):
            if r["status"] == "approved" and r["tick"] <= q["cutoff"]:
                selected[r["entity"], r["key"]] = r
        entity = q["start"]
        for key in q["path"]:
            r = selected.get((entity, key))
            if r is None:
                steps.append({"entity": entity, "field": key, "status": "missing_approved_evidence"})
                return {"status": "UNKNOWN"}, steps
            steps.append({"entity": entity, "field": key, "value": r["value"], "tick": r["tick"]})
            entity = r["value"]
        return {"answer": entity}, steps
    raise ValueError(f"Unknown family: {f}")


def trace_text(w: dict[str, Any], answer: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    """Task-specific structured solution, not search transcript or human mental states."""
    f = w["family"]
    intro = {
        "ledger_replay": "Identify the requested account and cutoff. Initialize every account, sort events by tick, and ignore void or future events. Replay signed transfers literally; negative balances are allowed.",
        "register_machine": "Initialize X, Y, and Z. Process the numbered instructions in order. Read conditions and both source registers from the pre-step state; use a nonnegative residue modulo 97.",
        "temporal_route": "For each directed edge, select the latest approved version at the cutoff and discard closed edges. Search states consisting of location and completed ordered waypoints; minimize total positive cost.",
        "continuous_replan": "Rebuild the current approved graph from the causal history. Earlier plans are not binding. Plan from the reported current location with the currently known costs, not unobserved future updates.",
        "prerequisite_plan": "Separate the goal flags from action preconditions. Search finite flag states using only applicable actions, applying deletes before adds. Minimize total positive action cost, then verify the goal.",
        "deadline_schedule": "List all jobs and precedence constraints. Search feasible subsets on the single machine, start each selected job at the earliest time allowed by its release, reject late completions, and minimize final finish.",
        "budget_selection": "Resolve the latest approved budget and capacity. Compare item subsets satisfying both limits, all dependencies, and all exclusions. Maximize total value, then minimize cost.",
        "evidence_chain": "Resolve only approved fields at the cutoff. Follow the requested relations in order; stop with UNKNOWN when a required field has no approved evidence. Drafts are not facts.",
    }[f]
    lines = ["1. " + intro]
    for i, step in enumerate(steps, 2):
        display = {k: v for k, v in step.items() if k != "event"}
        lines.append(f"{i}. Checkpoint: {dumps(display)}.")
    if answer.get("status") == "IMPOSSIBLE":
        lines.append(f"{len(lines)+1}. Exhaustive finite-state/subset search found no feasible goal; this is impossibility under the stated complete rules, not missing evidence.")
    elif answer.get("status") == "UNKNOWN":
        lines.append(f"{len(lines)+1}. A required approved record is absent. Do not substitute a draft or guess a value.")
    else:
        lines.append(f"{len(lines)+1}. The replay or witness satisfies the task. The exact solver checks the optimization objective where applicable. Emit only the required fields.")
    lines.append(f"{len(lines)+1}. Final result: {dumps(answer)}.")
    return "\n".join(lines)
