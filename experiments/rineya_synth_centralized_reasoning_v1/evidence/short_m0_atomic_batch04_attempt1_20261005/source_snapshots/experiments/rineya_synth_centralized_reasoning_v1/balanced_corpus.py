"""Prospective CPU corpus with crossed conditions/renderings and paired parents.

Separate successor to task_compiler v1. No native launch, token guess, teacher
call, or claim of learned skill. Every intervention is source-registered before
component allocation; surface item reindexing is invertible.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from itertools import combinations
import random

from rs_contract import canonical, digest, format_answer, isint, parse_visible, render, require, validate_spec
from rs_solver import solve
from rs_verifier import solve_visible, verify

VERSION = "synth-balanced-corpus-1"
CONDITIONS = {"R": ("depth_previous", "depth_next", "endpoint_rebound", "request_rebound"),
              "S": ("target_low", "target_high", "k_previous", "k_next")}


def choose_target(spec: dict, seed: int) -> list[int]:
    require(isint(seed, 0, 2**32 - 1), "target seed")
    result = solve(spec)
    require(result["status"] == "solved", "only solved atomic certificates")
    return copy.deepcopy(random.Random(seed).choice(result["solutions"]))


def _chain(spec: dict, name: str) -> tuple[list[str], int]:
    links = spec["lookup"]["links"]
    path = []
    while type(name) is str:
        require(name in links and name not in path, "unresolved chain")
        path.append(name)
        name = links[name]
    return path, name


def family_key(spec: dict) -> str:
    """Conservative isomorphism screen; collisions can over-group worlds.

    R ignores numeric values, alias spelling and item indices but retains
    terminal equivalence, path lengths and shared-node usage. S ignores item
    ordering. No representation change manufactures a fresh parent family.
    """
    sig = validate_spec(spec)
    shape = {"skill": sig, "n": len(spec["initial"])}
    if "R" in sig:
        groups = defaultdict(list)
        usage = Counter()
        distance = {}
        for name in set(spec["lookup"]["requested"]):
            path, terminal = _chain(spec, name)
            groups[terminal].append(len(path))
            usage.update(path)
            distance.update({node: len(path) - j for j, node in enumerate(path)})
        shape["terminal_groups"] = sorted(sorted(v) for v in groups.values())
        shape["shared_nodes"] = sorted((distance[node], count) for node, count in usage.items())
    if "S" in sig:
        shape["initial"] = sorted(spec["initial"])
        shape["selection"] = spec["selection"]
        if sig == "RS":
            eligible = solve({"initial": spec["initial"], "lookup": spec["lookup"]})["solutions"][0]
            shape["eligible_values"] = sorted(spec["initial"][j] for j in eligible)
    return digest(shape)


def condition_worlds(base: dict, seed: int, source_policy: int = 1) -> list[dict]:
    sig = validate_spec(base)
    require(sig in ("R", "S") and solve(base)["status"] == "solved", "solved atomic parent required")
    require(isint(seed, 0, 2**32 - 1), "condition seed")
    require(isint(source_policy, 1, 2), "registered source policy")
    rng = random.Random(seed)
    worlds = []
    if sig == "R":
        requested = base["lookup"]["requested"]
        path, terminal = _chain(base, requested[0])
        require(1 <= len(path) <= 5, "registered depth stratum")
        # Rebuild the first route without erasing nodes used by another route.
        # Generated source parents have private chains; this also handles shared
        # tails correctly for independent adversarial controls.
        others = {node for name in requested[1:] for node in _chain(base, name)[0]}
        incoming = {v for k, v in base["lookup"]["links"].items() if k not in path and type(v) is str}
        for name, depth in (("depth_previous", 1 + ((len(path) - 2) % 5)),
                            ("depth_next", 1 + (len(path) % 5))):
            changed = copy.deepcopy(base)
            links = changed["lookup"]["links"]
            for node in path:
                if node == requested[0] or node not in others | incoming:
                    del links[node]
            names = [requested[0]] + [f"d{i}_0" for i in range(1, depth)]
            require(all(node not in links for node in names), "depth alias collision")
            links.update({node: terminal if j == depth - 1 else names[j + 1]
                          for j, node in enumerate(names)})
            worlds.append({"name": name, "spec": changed, "changed_fields": ["lookup.links"], "effect": "preserve"})
        original = solve(base)["solutions"][0]
        alternatives = [j for j in range(len(base["initial"])) if j not in original]
        require(bool(alternatives), "unrequested endpoint required")
        replacement = rng.choice(alternatives)
        changed = copy.deepcopy(base)
        changed["lookup"]["links"][path[-1]] = replacement
        worlds.append({"name": "endpoint_rebound", "spec": changed,
                       "changed_fields": ["lookup.links"], "effect": "change"})
        unused = []
        for name in base["lookup"]["links"]:
            if name in requested:
                continue
            try:
                candidate_path, candidate_terminal = _chain(base, name)
                if candidate_terminal not in original and (source_policy == 1 or len(candidate_path) == len(path)):
                    unused.append(name)
            except ValueError:
                # Unrequested cycles/missing links remain irrelevant distractors.
                continue
        require(bool(unused), "source must contain an unrequested novel endpoint")
        changed = copy.deepcopy(base)
        changed["lookup"]["requested"][0] = rng.choice(unused)
        worlds.append({"name": "request_rebound", "spec": changed,
                       "changed_fields": ["lookup.requested"], "effect": "change"})
    else:
        values = base["initial"]
        k, target = base["selection"]["k"], base["selection"]["target"]
        all_targets = sorted({sum(values[j] for j in subset) for subset in combinations(range(len(values)), k)})
        lower, upper = [t for t in all_targets if t < target], [t for t in all_targets if t > target]
        require(bool(lower) and bool(upper), "interior feasible target required")
        for name, new_target in (("target_low", rng.choice(lower)), ("target_high", rng.choice(upper))):
            changed = copy.deepcopy(base)
            changed["selection"]["target"] = new_target
            worlds.append({"name": name, "spec": changed, "changed_fields": ["selection.target"], "effect": "change"})
        for name, new_k in (("k_previous", 1 + ((k - 2) % 5)), ("k_next", 1 + (k % 5))):
            selected = rng.sample(range(len(values)), new_k)
            changed = copy.deepcopy(base)
            changed["selection"] = {"k": new_k, "target": sum(values[j] for j in selected)}
            worlds.append({"name": name, "spec": changed,
                           "changed_fields": ["selection.k", "selection.target"], "effect": "change"})
    require(tuple(w["name"] for w in worlds) == CONDITIONS[sig], "condition policy drift")
    old = format_answer(choose_target(base, seed))
    for world in worlds:
        changed_fields = []
        if sig == "R":
            for field in ("links", "requested"):
                if canonical(base["lookup"][field]) != canonical(world["spec"]["lookup"][field]):
                    changed_fields.append("lookup." + field)
        else:
            for field in ("k", "target"):
                if base["selection"][field] != world["spec"]["selection"][field]:
                    changed_fields.append("selection." + field)
        world["changed_fields"] = changed_fields
        first, second = solve(world["spec"]), solve_visible(render(world["spec"]))
        require(first["status"] == "solved" and first["solutions"] == second["solutions"], "condition disagreement")
        require(verify(render(world["spec"]), old)["accepted"] == (world["effect"] == "preserve"), "intervention witness failure")
    return worlds


def surface_view(spec: dict, template: int, rotation: int) -> dict:
    validate_spec(spec)
    n = len(spec["initial"])
    require(isint(rotation, 0, n - 1), "surface rotation")
    item_map = [(j + rotation) % n for j in range(n)]
    changed = copy.deepcopy(spec)
    for j, value in enumerate(spec["initial"]):
        changed["initial"][item_map[j]] = value
    if "lookup" in spec:
        changed["lookup"]["links"] = {name: item_map[value] if type(value) is int else value
                                      for name, value in spec["lookup"]["links"].items()}
    return {"spec": changed, "messages": render(changed, template), "item_map": item_map}


def build_balanced_plan(seed: int = 20261003, parent_count: int = 64, source_policy: int = 1) -> dict:
    require(isint(seed, 0, 2**32 - 1) and isint(parent_count, 2, 256) and parent_count % 2 == 0, "bounded even plan size/seed")
    require(isint(source_policy, 1, 2), "registered source policy")
    rng = random.Random(seed)
    sources, parents, conditions, attempts = {}, {}, {}, []
    attempt = 0
    while len(parents) < parent_count:
        require(attempt < parent_count * 4, "semantic candidate budget exhausted")
        i = len(parents)
        spec = {"initial": rng.sample(range(-40, 41), rng.randint(6, 12))}
        n = len(spec["initial"])
        if i % 2 == 0:
            count = rng.randint(2, min(4, n - 1))
            endpoints = rng.sample(range(n), count + 1)
            links = {"x0_0": endpoints[-1]}
            if source_policy == 2:
                alternate_depth = 1 + ((i // 2) % 5)
                links = {f"x0_{hop}": endpoints[-1] if hop == alternate_depth - 1 else f"x0_{hop + 1}"
                         for hop in range(alternate_depth)}
            for branch in range(count):
                depth = 1 + ((i // 2) % 5) if branch == 0 else rng.randint(1, 5)
                for hop in range(depth):
                    name = f"a{branch}_{hop}"
                    links[name] = endpoints[branch] if hop == depth - 1 else f"a{branch}_{hop + 1}"
            spec["lookup"] = {"links": links, "requested": [f"a{j}_0" for j in range(count)]}
        else:
            k = 1 + ((i // 2) % 5)
            targets = sorted({sum(spec["initial"][j] for j in subset) for subset in combinations(range(n), k)})
            spec["selection"] = {"k": k, "target": rng.choice(targets[1:-1])}
        condition_seed = rng.randrange(2**32)
        try:
            worlds = condition_worlds(spec, condition_seed, source_policy)
        except ValueError as exc:
            attempts.append({"attempt": attempt, "spec_hash": digest(spec), "accepted": False, "reason": str(exc)})
            attempt += 1
            continue
        pid = f"balanced_parent_{i:03d}"
        sid = f"balanced_source_{i:03d}"
        content = {"base_spec": spec, "condition_seed": condition_seed, "conditions": worlds}
        if source_policy == 2:
            content["source_policy"] = source_policy
        sources[sid] = {"source_id": sid, "version": source_policy, "content": content, "sha256": digest(content),
                        "provenance": "deterministic_local_sources_and_preregistered_cyclic_condition_strata",
                        "license": "repository_terms_not_external_data", "attribution": "SYNTH-CRS-001"}
        parents[pid] = {"parent_id": pid, "family_id": pid, "source_ids": [sid], "spec": spec}
        conditions[pid] = worlds
        attempts.append({"attempt": attempt, "spec_hash": digest(spec), "accepted": True, "reason": "independent_conditions_verified"})
        attempt += 1
    leaders, ownership = {p: p for p in parents}, {}

    def root(pid: str) -> str:
        while leaders[pid] != pid:
            pid = leaders[pid]
        return pid

    for pid, parent in parents.items():
        tokens = [("family", parent["family_id"])]
        tokens.extend(("world", family_key(spec)) for spec in [parent["spec"]] + [w["spec"] for w in conditions[pid]])
        tokens.extend((kind, value) for sid in parent["source_ids"]
                      for kind, value in (("source_id", sid), ("source_hash", sources[sid]["sha256"])))
        for token in tokens:
            if token in ownership:
                a, b = sorted((root(pid), root(ownership[token])))
                leaders[b] = a
            else:
                ownership[token] = pid
    groups = defaultdict(list)
    for pid in parents:
        groups[root(pid)].append(pid)
    assignments = {}
    for group in groups.values():
        component = digest({"parents": sorted(group), "sources": sorted(sources[parents[p]["source_ids"][0]]["sha256"] for p in group)})
        bucket = int(digest({"seed": seed, "component": component})[:12], 16) % 100
        split = "train" if bucket < 60 else "dev" if bucket < 80 else "eval_candidate"
        assignments.update({pid: {"component_id": component, "split": split} for pid in group})
    plan = {"version": VERSION if source_policy == 1 else "synth-balanced-corpus-2",
            "seed": seed, "sources": sources, "parents": parents,
            "assignments": assignments, "condition_plans": conditions, "attempts": attempts}
    plan["sha256"] = digest(plan)
    return plan


def validate_plan(plan: dict) -> None:
    require(type(plan) is dict and set(plan) == {"version", "seed", "sources", "parents", "assignments",
            "condition_plans", "attempts", "sha256"} and type(plan["parents"]) is dict, "plan fields")
    require(plan.get("version") in (VERSION, "synth-balanced-corpus-2"), "source policy version")
    expected = build_balanced_plan(plan["seed"], len(plan["parents"]), 1 if plan["version"] == VERSION else 2)
    require(canonical(expected) == canonical(plan), "registered plan or source/hash drift")


def _records(plan: dict) -> list[dict]:
    rows = []
    for pid, parent in sorted(plan["parents"].items()):
        skill = validate_spec(parent["spec"])
        require(skill in ("R", "S"), "training corpus must remain atomic")
        for arm in ("M1", "M2"):
            for ci, world in enumerate(plan["condition_plans"][pid]):
                original = parent["spec"] if arm == "M1" else world["spec"]
                seed = int(digest({"seed": plan["seed"], "parent": pid, "condition_index": ci})[:8], 16)
                witness = choose_target(original, seed)
                for template in range(4):
                    rotation = (ci * 4 + template) % len(original["initial"])
                    view = surface_view(original, template, rotation)
                    target = format_answer([view["item_map"][j] for j in witness])
                    verdict = verify(view["messages"], target)
                    first = solve(view["spec"])
                    require(verdict["accepted"] and first["solutions"] == verdict["outcome"]["solutions"], "visible verifier disagreement")
                    row = {"parent_id": pid, **plan["assignments"][pid], "arm": arm, "skill": skill,
                           "condition": f"surface_repeat_{ci}" if arm == "M1" else world["name"],
                           "condition_index": ci, "template": template, "rotation": rotation,
                           "base_spec_hash": digest(parent["spec"]), "spec": view["spec"],
                           "messages": view["messages"], "target": target, "verification": verdict,
                           "source_hash": plan["sources"][parent["source_ids"][0]]["sha256"],
                           "item_map": view["item_map"]}
                    row["row_id"] = digest(row)
                    rows.append(row)
    return rows


def compile_balanced_records(plan: dict) -> list[dict]:
    validate_plan(plan)
    rows = _records(plan)
    ownership = {}
    for row in rows:
        token = family_key(row["spec"])
        require(token not in ownership or ownership[token] == row["split"], "surface/condition family crosses splits")
        ownership[token] = row["split"]
    return rows


def validate_balanced_records(rows: list[dict], plan: dict) -> None:
    require(type(rows) is list and 0 < len(rows) <= 8192, "bounded corpus rows")
    expected = {r["row_id"]: r for r in compile_balanced_records(plan)}
    require(len(rows) == len(expected), "incomplete registered corpus")
    seen = set()
    for row in rows:
        require(type(row) is dict and type(row.get("row_id")) is str and row["row_id"] in expected, "unregistered row")
        require(row["row_id"] not in seen and canonical(row) == canonical(expected[row["row_id"]]), "duplicate or rebound row")
        seen.add(row["row_id"])


def audit_balance(rows: list[dict]) -> dict:
    exposures = {arm: Counter() for arm in ("M1", "M2")}
    crossing = Counter()
    positions = defaultdict(Counter)
    strata = defaultdict(lambda: defaultdict(Counter))
    parents = {}
    for row in rows:
        arm, skill = row["arm"], row["skill"]
        exposures[arm][row["parent_id"]] += 1
        crossing[f"{arm}:{row['condition']}:{row['template']}"] += 1
        parents[row["parent_id"]] = row["split"]
        key = arm + ":" + skill
        answer = solve_visible(row["messages"])
        import json
        positions[key].update(str(j) for j in json.loads(row["target"])["answer"])
        strata[key]["n"][str(len(row["spec"]["initial"]))] += 1
        strata[key]["answer_size"][str(len(json.loads(row["target"])["answer"]))] += 1
        strata[key]["solution_multiplicity"][str(len(answer["solutions"]))] += 1
        if skill == "R":
            lookup = row["spec"]["lookup"]
            depths = [len(_chain(row["spec"], name)[0]) for name in lookup["requested"]]
            strata[key]["requested_count"][str(len(lookup["requested"]))] += 1
            strata[key]["first_requested_depth"][str(depths[0])] += 1
            strata[key]["max_requested_depth"][str(max(depths))] += 1
        else:
            strata[key]["k"][str(row["spec"]["selection"]["k"])] += 1
    cell_counts = Counter((r["parent_id"], r["arm"], r["condition_index"], r["template"]) for r in rows)
    return {"record_count": len(rows), "parents_by_split": dict(Counter(parents.values())),
            "rows_by_arm_skill": dict(Counter(r["arm"] + ":" + r["skill"] for r in rows)),
            "parent_exposures": {a: dict(c) for a, c in exposures.items()},
            "condition_template_counts": dict(crossing),
            "answer_position_counts_by_arm_skill": {k: dict(c) for k, c in positions.items()},
            "strata_by_arm_skill": {k: {s: dict(c) for s, c in v.items()} for k, v in strata.items()},
            "full_crossing": bool(rows) and len(cell_counts) == len(parents) * 32 and all(v == 1 for v in cell_counts.values()),
            "matched_parent_exposures": exposures["M1"] == exposures["M2"],
            "no_RS_in_training": all(validate_spec(parse_visible(r["messages"])) in ("R", "S")
                                     for r in rows if r["split"] == "train")}
