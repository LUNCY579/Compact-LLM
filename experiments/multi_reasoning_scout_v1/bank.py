"""Deterministic new worlds; independent forward and exhaustive answer solvers.

Never reads historical datasets. Registry metadata is the only external data input.
This is deliberately open DEV forever, including its smoke and calibration partitions.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict
import itertools
import random
import re

from .core import (ANSWERS, CLASSIFICATION, RECORDS, PublicItem, digest, dump_lines_new,
                   file_hash, new_dir, read_json, write_new)

SEED = 29170917
FAMILIES = ("register_machine", "counterfactual_state", "prerequisite_schedule", "evidence_chain")


def step(x, op):
    kind, n = op
    return (x + n) % 4 if kind == "add" else x ^ n


def solve_forward(w):
    family = w["family"]
    if family in ("register_machine", "counterfactual_state"):
        values = {w["initial"]} if w["initial"] is not None else set(range(4))
        for i, op in enumerate(w["ops"]):
            if i != w["skip"]:
                values = {step(x, op) for x in values}
    elif family == "prerequisite_schedule":
        possible = []
        unknown = w["missing"]
        alternatives = [w["deps"]]
        if unknown is not None:
            alternatives = []
            for bits in itertools.product((0, 1), repeat=unknown):
                ds = [list(x) for x in w["deps"]]
                ds[unknown] = [i for i, b in enumerate(bits) if b]
                alternatives.append(ds)
        for deps in alternatives:
            starts = []
            for row in deps:
                starts.append(max((starts[j] + 1 for j in row), default=0))
            possible.append(starts[w["query"]])
        values = set(possible)
    else:
        current = w["query"]
        links = dict(w["links"])
        while current in links:
            current = links[current]
        found = [(time, value) for node, time, value in w["facts"] if node == current]
        values = {max(found)[1]} if found else set(range(4))
    return str(next(iter(values))) if len(values) == 1 else "UNKNOWN"


def solve_exhaustive(w):
    """Different algorithm: backward state preimages / feasible schedules / path enumeration."""
    family = w["family"]
    outcomes = set()
    if family in ("register_machine", "counterfactual_state"):
        for result in range(4):
            predecessors = {result}
            for i in reversed(range(len(w["ops"]))):
                if i == w["skip"]:
                    continue
                kind, n = w["ops"][i]
                predecessors = {(v - n) % 4 if kind == "add" else v ^ n for v in predecessors}
            if w["initial"] is None or w["initial"] in predecessors:
                outcomes.add(result)
    elif family == "prerequisite_schedule":
        unknown = w["missing"]
        bits_options = [()] if unknown is None else itertools.product((0, 1), repeat=unknown)
        for bits in bits_options:
            deps = [list(d) for d in w["deps"]]
            if unknown is not None:
                deps[unknown] = [i for i, bit in enumerate(bits) if bit]
            legal = [s for s in itertools.product(range(4), repeat=4)
                     if all(s[i] >= s[j] + 1 for i in range(4) for j in deps[i])]
            outcomes.add(min(s[w["query"]] for s in legal))
    else:
        # Enumerate paths rather than mutate a current-state cursor.
        nodes = list(range(5))
        endpoints = set()
        edge_set = set(map(tuple, w["links"]))
        sources = {a for a, _ in edge_set}
        for length in range(1, 6):
            for tail in itertools.permutations([i for i in nodes if i != w["query"]], length - 1):
                path = (w["query"],) + tail
                if path[-1] not in sources and all((a, b) in edge_set for a, b in zip(path, path[1:])):
                    endpoints.add(path[-1])
        for end in endpoints:
            facts = [row for row in w["facts"] if row[0] == end]
            if not facts:
                outcomes.update(range(4))
            else:
                latest = sorted(facts, key=lambda f: f[1])[-1]
                outcomes.add(latest[2])
    return str(next(iter(outcomes))) if len(outcomes) == 1 else "UNKNOWN"


def world(rng, family):
    names = ["n" + ''.join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=6)) for _ in range(5)]
    w = {"family": family, "names": names}
    if family in ("register_machine", "counterfactual_state"):
        w.update(initial=None if rng.random() < .2 else rng.randrange(4),
                 ops=[[rng.choice(("add", "xor")), rng.randrange(4)] for _ in range(rng.randint(3, 6))],
                 skip=None)
        if family == "counterfactual_state":
            w["skip"] = rng.randrange(len(w["ops"]))
    elif family == "prerequisite_schedule":
        w.update(deps=[[j for j in range(i) if rng.random() < .55] for i in range(4)],
                 query=rng.randrange(4), missing=rng.choice((None, None, None, 1, 2, 3)))
    else:
        order = rng.sample(range(5), 5)
        w.update(links=list(zip(order[:3], order[1:4])), query=order[0],
                 facts=[[i, t, rng.randrange(4)] for i in range(5) for t in (1, 2)
                        if i != order[3] or rng.random() >= .45])
        rng.shuffle(w["facts"])
    return w


def render(w):
    family, names = w["family"], w["names"]
    if family in ("register_machine", "counterfactual_state"):
        initial = "not recorded" if w["initial"] is None else str(w["initial"])
        text = (f"Register {names[0]} holds one of 0,1,2,3. Initial value: {initial}. "
                "Apply numbered events in order. ADD k adds k modulo 4. XOR k is bitwise XOR on two bits.\n")
        text += '\n'.join(f"{i+1}. {kind.upper()} {n}" for i, (kind, n) in enumerate(w["ops"]))
        if w["skip"] is not None:
            text += f"\nCounterfactual: event {w['skip'] + 1} is cancelled; all other events stay in order."
        return text + f"\nWhat is the final value of {names[0]}?"
    if family == "prerequisite_schedule":
        text = ("Four unit-duration jobs may run in parallel. Times begin at 0. A job starts only after "
                "all prerequisites finish. All listed dependencies are complete except any explicitly "
                "unrecorded row. The only possible prerequisites of each job are jobs earlier in "
                "this list. Find the earliest possible start under the actual dependencies.\n")
        for i, ds in enumerate(w["deps"]):
            row = "not recorded" if i == w["missing"] else (', '.join(names[j] for j in ds) or "none")
            text += f"Job {names[i]}: prerequisites {row}.\n"
        return text + f"What is the earliest start time of {names[w['query']]}?"
    text = ("Follow directed links until a node has no outgoing link. Its value is the most recent "
            "dated value record for that node; older records are superseded. Values are 0,1,2,3. "
            "Links below are complete.\n")
    text += '\n'.join(f"{names[a]} -> {names[b]}" for a, b in w["links"])
    text += '\n' + '\n'.join(f"time {t}: {names[i]} value {v}" for i, t, v in w["facts"])
    return text + f"\nWhat value is reached from {names[w['query']]}?"


def generate():
    rng = random.Random(SEED)
    rows, gold, worlds = [], [], []
    for family in FAMILIES:
        targets = list(ANSWERS) * 6
        rng.shuffle(targets)
        for i, target in enumerate(targets):
            for _ in range(10000):
                w = world(rng, family)
                if solve_forward(w) == target:
                    break
            else:
                raise ValueError("bounded label balancing failed")
            if solve_exhaustive(w) != target:
                raise ValueError("independent solvers disagree")
            sid = "MRS1-world-" + digest(w)[:24]
            item = PublicItem("MRS1-" + digest([sid, family])[:24], sid, family,
                              "SMOKE" if i < 2 else "CAL" if i < 6 else "CHECK", render(w))
            rows.append(asdict(item))
            gold.append({"id": item.id, "scenario_id": sid, "family": family, "split": item.split,
                         "answer": target, "answerable": target != "UNKNOWN",
                         "difficulty": len(w.get("ops", w.get("links", w.get("deps", [])))),
                         "verification": "forward == independent exhaustive"})
            worlds.append({"id": item.id, "world": w})
    rng.shuffle(rows)
    order = {r["id"]: i for i, r in enumerate(rows)}
    return rows, sorted(gold, key=lambda g: order[g["id"]]), sorted(worlds, key=lambda w: order[w["id"]])


def shortcuts(rows, gold):
    labels = {g["id"]: g["answer"] for g in gold}
    # Honest CAL-fit/CHECK-score shallow classifier, never a selector or model input.
    def features(row):
        text = row["problem"]
        q = re.sub(r"n[a-z]{6}", "ENTITY", text.splitlines()[-1])
        return {"family": row["family"], "question_only": q, "length_bin": len(text) // 100,
                "missing_marker": "not recorded" in text, "line_count": len(text.splitlines()),
                "digit_sum_mod4": sum(map(int, re.findall(r"\d", text))) % 4,
                "last_digit": (re.findall(r"\d", text) or [""])[-1]}
    cal = [r for r in rows if r["split"] == "CAL"]
    check = [r for r in rows if r["split"] == "CHECK"]
    prior = Counter(labels[r["id"]] for r in cal)
    default = sorted(prior, key=lambda a: (-prior[a], a))[0]
    probes = {}
    for feature in features(rows[0]):
        votes = defaultdict(Counter)
        for row in cal:
            votes[features(row)[feature]][labels[row["id"]]] += 1
        correct = 0
        for row in check:
            counts = votes[features(row)[feature]]
            guess = sorted(counts, key=lambda a: (-counts[a], a))[0] if counts else default
            correct += guess == labels[row["id"]]
        probes[feature] = {"correct": correct, "n": len(check), "accuracy": correct / len(check)}
    groups = {"all": rows, "CHECK": check, **{f: [r for r in check if r["family"] == f] for f in FAMILIES}}
    return {"method": "CAL-fitted single-feature majority; deterministic lexical tie; held-out CHECK reporting",
            "probes": probes, "answer_priors": {key: dict(Counter(labels[r["id"]] for r in rs))
                                                   for key, rs in groups.items()},
            "constant_CHECK_floor": max(Counter(labels[r["id"]] for r in check).values()) / len(check),
            "limitation": "Limited shallow audit, not a proof against every shortcut. Missing-fact cues are measured."}


def build():
    rows, gold, worlds = generate()
    access = read_json(RECORDS / "ACCESS_MANIFEST.json")
    forbidden_ids = set(access["quarantined_ids"])
    if any(r["id"] in forbidden_ids or r["scenario_id"] in forbidden_ids for r in rows):
        raise ValueError("quarantined ID collision")
    out = new_dir(RECORDS / "bank")
    for name, values in (("public", rows), ("gold", gold), ("worlds", worlds)):
        dump_lines_new(out / f"{name}.jsonl", values)
    audit = shortcuts(rows, gold)
    write_new(RECORDS / "BANK_MANIFEST.json", {
        "version": "MRS-bank-1", "classification": CLASSIFICATION, "seed": SEED,
        "n": len(rows), "worlds": len({r["scenario_id"] for r in rows}),
        "splits": dict(Counter(r["split"] for r in rows)), "families": list(FAMILIES),
        "files": {p.name: file_hash(p) for p in sorted(out.iterdir())},
        "case_ids": [r["id"] for r in rows], "shortcut_audit": audit,
        "independent_solver_agreement": len(rows), "quarantined_id_overlap": 0,
        "protected_data_reads": 0,
        "disjointness": "New deterministic namespace/world construction; no historical rows are imported. Protected content was not inspected; semantic similarity is not excluded.",
    })
    return audit
