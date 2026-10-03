"""Deterministic, CPU-only R/S fixture compiler and fail-closed export boundary.

No tokenizer budget, model binding, or launch permission is supplied here.
All dependency-connected parents are split before variants are constructed.
"""
from __future__ import annotations

import copy
import random
from itertools import combinations

from rs_contract import (canonical, digest, format_answer, isint, parse_visible,
                         render, require, validate_spec)
from rs_solver import solve
from rs_verifier import solve_visible, verify

VERSION = "synth-rs-compiler-1"
SPLITS = ("train", "dev", "eval_fixture")
SOURCE_FIELDS = {"source_id", "version", "content", "sha256", "provenance", "license", "attribution"}
PARENT_FIELDS = {"parent_id", "family_id", "source_ids", "spec"}
RECORD_FIELDS = {"schema_version", "row_id", "parent_id", "family_id", "component_id", "split",
                 "source_ids", "source_hashes", "generator", "seed", "arm", "variant", "skill",
                 "spec", "messages", "candidate", "verification", "spec_hash", "parent_spec_hash"}


def identifier(value: object) -> bool:
    return type(value) is str and 1 <= len(value) <= 128 and value.isascii() and all(
        c.isalnum() or c in "_-.:" for c in value)


def structural_key(spec: dict) -> str:
    """Conservative near-duplicate screen, not a proof of contamination freedom.

    S fingerprints ignore item permutations. R fingerprints ignore alias names,
    initial numeric values, unrequested distractors, and repeated requests.
    RS includes concrete item/value binding because it affects selection.
    Over-grouping is preferable to declaring renamed parents fresh.
    """
    sig = validate_spec(spec)
    shape = {"skill": sig, "n": len(spec["initial"])}
    if "R" in sig:
        paths = set()
        links = spec["lookup"]["links"]
        for name in spec["lookup"]["requested"]:
            seen = set()
            depth = 0
            while type(name) is str and name in links and name not in seen:
                seen.add(name)
                name = links[name]
                depth += 1
            terminal = name if type(name) is int else ("cycle" if name in seen else "missing")
            paths.add((depth, terminal))
        shape["paths"] = sorted(paths, key=canonical)
    if "S" in sig:
        shape["initial"] = sorted(spec["initial"]) if sig == "S" else spec["initial"]
        shape["selection"] = spec["selection"]
    return digest(shape)


def register_parents(sources: list[dict], parents: list[dict], seed: int = 1003) -> dict:
    require(type(sources) is list and 1 <= len(sources) <= 256, "source count")
    require(type(parents) is list and 1 <= len(parents) <= 256, "parent count")
    require(isint(seed, 0, 2**32 - 1), "seed")
    source_map = {}
    for source in sources:
        require(type(source) is dict and set(source) == SOURCE_FIELDS, "source fields")
        require(identifier(source["source_id"]) and source["source_id"] not in source_map, "source identity")
        require(type(source["version"]) is int and source["version"] >= 1, "source version")
        require(type(source["content"]) is dict and len(canonical(source["content"])) <= 65536, "source content")
        require(source["sha256"] == digest(source["content"]), "source hash mismatch")
        require(all(type(source[k]) is str and 1 <= len(source[k]) <= 1024
                    for k in ("provenance", "license", "attribution")), "source provenance")
        source_map[source["source_id"]] = copy.deepcopy(source)
    parent_map = {}
    for parent in parents:
        require(type(parent) is dict and set(parent) == PARENT_FIELDS, "parent fields")
        require(identifier(parent["parent_id"]) and parent["parent_id"] not in parent_map and
                identifier(parent["family_id"]), "parent identity")
        refs = parent["source_ids"]
        require(type(refs) is list and 1 <= len(refs) <= 256 and
                all(identifier(s) and s in source_map for s in refs) and
                len(set(refs)) == len(refs), "source dependencies")
        validate_spec(parent["spec"])
        require(any(canonical(source_map[s]["content"].get("base_spec")) == canonical(parent["spec"]) for s in refs),
                "parent world lacks matching source content")
        parent_map[parent["parent_id"]] = copy.deepcopy(parent)
    leaders = {p: p for p in parent_map}

    def root(p: str) -> str:
        while leaders[p] != p:
            p = leaders[p]
        return p

    owners = {}
    for pid, parent in sorted(parent_map.items()):
        tokens = [("family", parent["family_id"]), ("structure", structural_key(parent["spec"]))]
        # Register the frozen intervention footprints before allocating splits.
        # These are planning fingerprints; no accepted examples exist yet.
        # A counterfactual collision connects parents just like a source collision.
        if validate_spec(parent["spec"]) in ("R", "S") and solve(parent["spec"])["status"] == "solved":
            tokens.extend(("structure", structural_key(v)) for _, v, _, _ in condition_variants(parent["spec"]))
        for sid in parent["source_ids"]:
            tokens += [("source_id", sid), ("source_hash", source_map[sid]["sha256"])]
        for token in tokens:
            if token in owners:
                a, b = sorted((root(pid), root(owners[token])))
                leaders[b] = a
            else:
                owners[token] = pid
    groups = {}
    for pid in parent_map:
        groups.setdefault(root(pid), []).append(pid)
    assignments = {}
    for group in groups.values():
        component = digest({"parents": sorted(group), "sources": sorted({source_map[s]["sha256"]
                            for pid in group for s in parent_map[pid]["source_ids"]})})
        bucket = int(digest({"component": component, "seed": seed})[:12], 16) % 100
        split = "train" if bucket < 60 else "dev" if bucket < 80 else "eval_fixture"
        for pid in group:
            assignments[pid] = {"component_id": component, "split": split}
    catalog = {"version": VERSION, "seed": seed, "sources": source_map,
               "parents": parent_map, "assignments": assignments}
    catalog["sha256"] = digest(catalog)
    return catalog


def fixture_catalog(seed: int = 1003, parent_count: int = 48) -> dict:
    require(isint(parent_count, 2, 256) and parent_count % 2 == 0, "even parent count 2..256")
    require(isint(seed, 0, 2**32 - 1), "seed")
    rng = random.Random(seed)
    sources, parents = [], []
    for i in range(parent_count):
        n = rng.randint(6, 12)
        spec = {"initial": rng.sample(range(-40, 41), n)}
        if i % 2 == 0:
            endpoints = rng.sample(range(n), 2)
            links = {"noise": (endpoints[0] + 1) % n}
            for branch, endpoint in enumerate(endpoints):
                depth = rng.randint(1, 4)
                for hop in range(depth):
                    name = f"a{branch}_{hop}"
                    links[name] = endpoint if hop == depth - 1 else f"a{branch}_{hop + 1}"
            spec["lookup"] = {"links": links, "requested": ["a0_0", "a1_0"]}
        else:
            k = rng.randint(1, 4)
            selected = rng.sample(range(n), k)
            spec["selection"] = {"k": k, "target": sum(spec["initial"][j] for j in selected)}
        content = {"base_spec": spec}
        sid = f"local_source_{i:03d}"
        sources.append({"source_id": sid, "version": 1, "content": content, "sha256": digest(content),
                        "provenance": "deterministically_constructed_cpu_fixture",
                        "license": "repository_terms_not_external_data",
                        "attribution": "SYNTH-CRS-001 CPU compiler"})
        parents.append({"parent_id": f"parent_{i:03d}", "family_id": f"local_family_{i:03d}",
                        "source_ids": [sid], "spec": spec})
    return register_parents(sources, parents, seed)


def condition_variants(base: dict) -> list[tuple[str, dict, list[str], str]]:
    sig = validate_spec(base)
    require(sig in ("R", "S"), "condition amplification requires atomic R/S")
    variants = [("baseline", copy.deepcopy(base), [], "preserve")]
    first = solve(base)
    require(first["status"] == "solved", "only solved parents amplified")
    if sig == "R":
        deep = copy.deepcopy(base)
        links = deep["lookup"]["links"]
        require("extension" not in links and len(links) < 80, "extension alias capacity")
        name = deep["lookup"]["requested"][0]
        links["extension"] = links[name]
        links[name] = "extension"
        variants.append(("depth_extended", deep, ["lookup.links"], "preserve"))
        changed = copy.deepcopy(base)
        node = changed["lookup"]["requested"][0]
        while type(changed["lookup"]["links"][node]) is str:
            node = changed["lookup"]["links"][node]
        new = next((j for j in range(len(base["initial"])) if j not in first["solutions"][0]), None)
        require(new is not None, "R amplification requires an unrequested endpoint")
        changed["lookup"]["links"][node] = new
        variants.append(("endpoint_changed", changed, ["lookup.links"], "change"))
        requested = copy.deepcopy(base)
        require("additional" not in requested["lookup"]["links"] and
                len(requested["lookup"]["requested"]) < len(base["initial"]), "additional request capacity")
        requested["lookup"]["links"]["additional"] = new
        requested["lookup"]["requested"].append("additional")
        variants.append(("request_added", requested, ["lookup.links", "lookup.requested"], "change"))
    else:
        k = base["selection"]["k"]
        target_changed = copy.deepcopy(base)
        target = next((sum(base["initial"][i] for i in subset)
                       for subset in combinations(range(len(base["initial"])), k)
                       if sum(base["initial"][i] for i in subset) != base["selection"]["target"]), None)
        require(target is not None, "no answer-sensitive target intervention available")
        target_changed["selection"]["target"] = target
        variants.append(("target_changed", target_changed, ["selection.target"], "change"))
        cardinality = copy.deepcopy(base)
        new_k = 2 if k == 1 else k - 1
        cardinality["selection"] = {"k": new_k, "target": sum(base["initial"][:new_k])}
        variants.append(("cardinality_changed", cardinality, ["selection.k", "selection.target"], "change"))
        expanded = copy.deepcopy(base)
        require(len(expanded["initial"]) < 15, "item capacity")
        expanded["initial"].append(0)
        variants.append(("item_added", expanded, ["initial"], "preserve"))
    for _, spec, _, _ in variants:
        validate_spec(spec)
    return variants


def _record(catalog: dict, pid: str, arm: str, variant: dict, spec: dict) -> dict:
    parent = catalog["parents"][pid]
    assignment = catalog["assignments"][pid]
    messages = render(spec, variant["template"])
    primary = solve(spec)
    independent = solve_visible(messages)
    require(primary["status"] == independent["status"] and
            primary["solutions"] == independent["solutions"], "independent solver disagreement")
    candidate = format_answer(primary["solutions"][0] if primary["solutions"] else "UNKNOWN")
    verification = verify(messages, candidate)
    require(verification["accepted"], "candidate verification failed")
    record = {"schema_version": 1, "parent_id": pid, "family_id": parent["family_id"],
              **assignment, "source_ids": sorted(parent["source_ids"]),
              "source_hashes": sorted(catalog["sources"][sid]["sha256"] for sid in parent["source_ids"]),
              "generator": VERSION, "seed": catalog["seed"], "arm": arm, "variant": variant,
              "skill": validate_spec(spec), "spec": copy.deepcopy(spec), "messages": messages,
              "candidate": candidate, "verification": verification, "spec_hash": digest(spec),
              "parent_spec_hash": digest(parent["spec"])}
    record["row_id"] = digest(record)
    return record


def compile_records(catalog: dict) -> list[dict]:
    validate_catalog(catalog)
    rows = []
    for pid, parent in sorted(catalog["parents"].items()):
        sig = validate_spec(parent["spec"])
        if sig == "RS":
            require(catalog["assignments"][pid]["split"] != "train", "RS parent in training split")
            rows.append(_record(catalog, pid, "EVAL_RS", {"kind": "heldout_composition", "name": "rs",
                        "template": 0, "changed_fields": [], "answer_effect": "preserve"}, parent["spec"]))
            continue
        for template in range(4):
            rows.append(_record(catalog, pid, "M1", {"kind": "surface", "name": f"surface_{template}",
                        "template": template, "changed_fields": [], "answer_effect": "preserve"}, parent["spec"]))
        for template, (name, spec, fields, effect) in enumerate(condition_variants(parent["spec"])):
            rows.append(_record(catalog, pid, "M2", {"kind": "condition", "name": name,
                        "template": template, "changed_fields": fields, "answer_effect": effect}, spec))
    validate_records(rows, catalog)
    return rows


def validate_catalog(catalog: dict) -> None:
    require(type(catalog) is dict and set(catalog) ==
            {"version", "seed", "sources", "parents", "assignments", "sha256"}, "catalog fields")
    require(type(catalog["sources"]) is dict and type(catalog["parents"]) is dict, "catalog maps")
    fresh = register_parents(list(catalog["sources"].values()), list(catalog["parents"].values()), catalog["seed"])
    require(canonical(catalog) == canonical(fresh), "catalog provenance, assignment, or hash drift")


def validate_records(records: list[dict], catalog: dict) -> None:
    validate_catalog(catalog)
    require(type(records) is list and 1 <= len(records) <= 2048, "record count")
    seen = set()
    ownership = {}
    for row in records:
        require(type(row) is dict and set(row) == RECORD_FIELDS, "record fields")
        require(type(row["schema_version"]) is int and row["schema_version"] == 1, "record version")
        pid = row["parent_id"]
        require(type(pid) is str and pid in catalog["parents"], "unregistered parent")
        parent = catalog["parents"][pid]
        variant = row["variant"]
        require(type(variant) is dict and set(variant) ==
                {"kind", "name", "template", "changed_fields", "answer_effect"}, "variant fields")
        require(row["arm"] in ("M1", "M2", "EVAL_RS"), "arm")
        if row["arm"] in ("M1", "M2"):
            require(validate_spec(parent["spec"]) in ("R", "S"), "adaptation records must be atomic")
        if row["arm"] == "M1":
            expected = parent["spec"]
            require(variant == {"kind": "surface", "name": f"surface_{variant['template']}",
                    "template": variant["template"], "changed_fields": [], "answer_effect": "preserve"}, "surface drift")
        elif row["arm"] == "M2":
            variants = condition_variants(parent["spec"])
            require(isint(variant["template"], 0, 3), "condition template")
            name, expected, fields, effect = variants[variant["template"]]
            require(variant == {"kind": "condition", "name": name, "template": variant["template"],
                    "changed_fields": fields, "answer_effect": effect}, "condition declaration drift")
        else:
            expected = parent["spec"]
            require(validate_spec(expected) == "RS" and row["split"] != "train" and
                    variant == {"kind": "heldout_composition", "name": "rs", "template": 0,
                                "changed_fields": [], "answer_effect": "preserve"}, "RS isolation")
        require(row["spec"] == expected and parse_visible(row["messages"]) == expected, "visible/world drift")
        expected_row = _record(catalog, pid, row["arm"], variant, expected)
        require(canonical(row) == canonical(expected_row), "record binding or verification drift")
        require(row["row_id"] not in seen, "duplicate record")
        seen.add(row["row_id"])
        for token in [row["component_id"], structural_key(row["spec"]), *row["source_hashes"]]:
            require(token not in ownership or ownership[token] == row["split"], "cross-split dependency or variant collision")
            ownership[token] = row["split"]


def export_training(records: list[dict], catalog: dict, arm: str) -> list[dict]:
    require(arm in ("M1", "M2"), "only registered adaptation arms")
    validate_records(records, catalog)
    exported = []
    for row in records:
        if row["split"] != "train" or row["arm"] != arm:
            continue
        require(validate_spec(parse_visible(row["messages"])) in ("R", "S"), "non-atomic training example")
        require(row["verification"]["outcome"]["status"] == "solved", "answerability training not registered")
        # Construct from an allowlist. No lineage, gold solutions, or verifier sidecar enters messages.
        exported.append({"messages": copy.deepcopy(row["messages"]), "target": row["candidate"]})
    return exported
