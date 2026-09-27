"""Development-only policy evolution, validation selection, frozen holdout reports."""
from __future__ import annotations

import random
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from .replay import Plan, Spec, evaluate
from .schema import (World, bytes_hash, digest, fields, load_partition, read_json,
                     require, validate_catalog, write_new)


def engine_hash() -> str:
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + [root / "PROTOCOL.md"]
    return digest({p.name: bytes_hash(p.read_bytes()) for p in paths})


def groups(entries: list[dict]) -> dict[str, str]:
    """Connected components over ALL declared leakage keys, not just one group id."""
    parent = {e["id"]: e["id"] for e in entries}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners: dict[str, str] = {}
    for entry in entries:
        for key in entry["leakage_keys"]:
            if key in owners:
                a, b = find(entry["id"]), find(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = entry["id"]
    return {key: find(key) for key in parent}


def world_groups(worlds: tuple[World, ...]) -> dict[str, str]:
    return groups([{"id": w.key, "leakage_keys": w.leakage_keys} for w in worlds])


def cluster_mean(rows: list[dict], membership: dict[str, str], key: str) -> float:
    """Seeds average within world, worlds within dependency cluster, clusters equally."""
    by_world: dict[str, list[float]] = {}
    for row in rows:
        by_world.setdefault(row["world_id"], []).append(row[key])
    by_group: dict[str, list[float]] = {}
    for world_id, values in by_world.items():
        by_group.setdefault(membership[world_id], []).append(mean(values))
    return mean(mean(values) for values in by_group.values())


def score(worlds: tuple[World, ...], spec: Spec, plan: Plan) -> tuple[tuple[float, float, float], list[dict]]:
    rows = []
    for world in worlds:
        for seed in plan.seeds:
            result = evaluate(world, spec, plan, seed)
            rows.append({key: result[key] for key in
                         ("world_id", "auc", "hypervolume", "attempts", "support_misses")}
                        | {"seed": seed})
    membership = world_groups(worlds)
    rank = (cluster_mean(rows, membership, "auc"),
            cluster_mean(rows, membership, "hypervolume"),
            -cluster_mean(rows, membership, "attempts"))
    return rank, rows


def mutate(spec: Spec, rng: random.Random, plan: Plan) -> Spec:
    """Bounded data-only DSL mutation. No eval/exec, imports, APIs, or generated code."""
    choices = {
        "explore": (0.0, 0.1, 0.25, 0.5, 1.0),
        "trend": (0.0, 0.25, 0.5, 1.0, 2.0),
        "repair": (0.0, 0.1, 0.25, 0.5),
        "depth_penalty": (0.0, 0.01, 0.02, 0.05, 0.1),
        "width": tuple(range(1, min(plan.workers, 8) + 1)),
        "patience": (2, 4, 8, 12, 24),
        "min_depth": (1, 2, 3, 4),
    }
    key = rng.choice(tuple(choices))
    return Spec.parse(asdict(replace(spec, **{key: rng.choice(choices[key])})))


def fit(catalog_path: Path, plan: Plan, output: Path) -> dict:
    """No test world is read here. Freeze before any test evaluation is possible."""
    plan = Plan.parse(asdict(plan))
    catalog = validate_catalog(read_json(catalog_path))
    for split, minimum in (("train", 3), ("validation", 2), ("test", 5)):
        membership = groups([e for e in catalog["worlds"] if e["split"] == split])
        require(len(set(membership.values())) >= minimum,
                f"need >= {minimum} independent declared {split} groups")
    train = load_partition(catalog_path.parent, catalog, "train")
    validation = load_partition(catalog_path.parent, catalog, "validation")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "catalog_commitment.json", catalog)
    write_new(output / "plan.json", asdict(plan))
    rng = random.Random(plan.search_seed)
    incumbent = Spec()
    initial_hash = digest(asdict(incumbent))
    candidates: dict[str, dict] = {}
    trajectory = []

    def register(spec: Spec, generation: int) -> str:
        key = digest(asdict(spec))
        if key not in candidates:
            rank, rows = score(train, spec, plan)
            candidates[key] = {"id": key, "spec": asdict(spec), "generation": generation,
                               "train_rank": rank, "train_feedback": rows}
        return key

    current = register(incumbent, 0)
    for generation in range(1, plan.generations + 1):
        previous = current
        candidates_this_round = [current]
        for _ in range(plan.mutations):
            candidates_this_round.append(register(mutate(incumbent, rng, plan), generation))
        # Incumbent first: equal scores never cause gratuitous policy replacement.
        current = max(candidates_this_round, key=lambda key: candidates[key]["train_rank"])
        incumbent = Spec.parse(candidates[current]["spec"])
        trajectory.append({"generation": generation, "previous": previous, "selected": current,
                           "previous_rank": candidates[previous]["train_rank"],
                           "selected_rank": candidates[current]["train_rank"],
                           "candidates": candidates_this_round})
    # Candidate generation is now over. Validation is read for one selection pass,
    # never fed back into mutation. Test outcomes have not been opened.
    for candidate in candidates.values():
        rank, rows = score(validation, Spec.parse(candidate["spec"]), plan)
        candidate["validation_rank"] = rank
        candidate["validation_feedback"] = rows
    selected = max(candidates, key=lambda key: candidates[key]["validation_rank"])
    payload = {
        "schema": "rineya-dream-freeze-v1", "kind": catalog["kind"],
        "catalog_hash": digest(catalog), "engine_hash": engine_hash(), "plan": asdict(plan),
        "selected_id": selected, "selected_spec": candidates[selected]["spec"],
        "initial_id": initial_hash, "search_seed": plan.search_seed,
        "candidate_count": len(candidates), "test_outcomes_read_during_fit": False,
        "native_execution_authorized": False,
        "baselines": {m: asdict(Spec(method=m, width=plan.workers)) for m in ("fixed", "greedy", "random")},
        "claim_scope": "synthetic mechanics only" if catalog["kind"] == "synthetic" else "historical replay only",
    }
    write_new(output / "search.json", {"candidates": list(candidates.values()), "trajectory": trajectory})
    envelope = {"payload": payload, "sha256": digest(payload)}
    write_new(output / "freeze.json", envelope)
    return envelope


def paired_comparison(results: list[dict], membership: dict[str, str], plan: Plan) -> dict:
    """Primary adaptive-minus-fixed comparison; resample dependency groups, NOT probes."""
    paired = {}
    for row in results:
        if row["arm"] in ("adaptive", "fixed"):
            paired.setdefault((row["world_id"], row["seed"]), {})[row["arm"]] = row["auc"]
    by_world: dict[str, list[float]] = {}
    for (world_id, _), arms in paired.items():
        require(set(arms) == {"adaptive", "fixed"}, "missing paired control")
        by_world.setdefault(world_id, []).append(arms["adaptive"] - arms["fixed"])
    by_cluster: dict[str, list[float]] = {}
    for world_id, values in by_world.items():
        by_cluster.setdefault(membership[world_id], []).append(mean(values))
    values = [mean(by_cluster[key]) for key in sorted(by_cluster)]
    require(bool(values), "empty paired comparison")
    ci = None
    if len(values) >= 5:
        rng = random.Random(plan.search_seed + 1)
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
        ci = [samples[49], samples[1949]]
    return {"metric": "attempt-normalized hypervolume AUC", "contrast": "adaptive - fixed",
            "mean_delta": mean(values), "cluster_count": len(values),
            "bootstrap_95_percentile": ci, "bootstrap_replicates": 2000 if ci is not None else 0,
            "scope": "descriptive cluster bootstrap; not a live-performance claim; small-cluster uncertainty"}


def evaluate_frozen(catalog_path: Path, run: Path) -> dict:
    """Exclusive holdout-open marker is written BEFORE loading test outcomes.

    The marker is an accidental-reuse guard, not access control against an owner
    who can delete files. A new research revision must retire any exposed holdout.
    """
    envelope = read_json(run / "freeze.json")
    fields(envelope, "payload sha256")
    payload = envelope["payload"]
    require(digest(payload) == envelope["sha256"], "freeze content mismatch")
    require(payload["schema"] == "rineya-dream-freeze-v1", "wrong freeze schema")
    require(payload["engine_hash"] == engine_hash(), "engine/protocol changed since policy freeze")
    require(payload["native_execution_authorized"] is False, "native execution not supported")
    catalog = validate_catalog(read_json(catalog_path))
    require(digest(catalog) == payload["catalog_hash"], "catalog changed since freeze")
    require(read_json(run / "catalog_commitment.json") == catalog, "catalog snapshot mismatch")
    plan = Plan.parse(payload["plan"])
    selected = Spec.parse(payload["selected_spec"])
    require(selected.method == "adaptive" and digest(asdict(selected)) == payload["selected_id"],
            "selected policy mismatch")
    require(set(payload["baselines"]) == {"fixed", "greedy", "random"}, "baseline arms changed")
    specs = {name: Spec.parse(raw) for name, raw in payload["baselines"].items()}
    require(all(spec.method == name for name, spec in specs.items()), "baseline identity mismatch")
    specs["adaptive"] = selected
    write_new(run / "holdout_opened.json", {"freeze_hash": envelope["sha256"],
                                           "catalog_hash": payload["catalog_hash"],
                                           "rule": "consumed even if evaluation subsequently fails"})
    worlds = load_partition(catalog_path.parent, catalog, "test")
    rows = []
    for arm, spec in specs.items():
        for world in worlds:
            for seed in plan.seeds:
                rows.append(evaluate(world, spec, plan, seed) | {"arm": arm, "seed": seed})
    membership = world_groups(worlds)
    summaries = {}
    for arm in specs:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {key: cluster_mean(arm_rows, membership, key)
                          for key in ("auc", "auc_gain", "hypervolume", "attempts", "revealed", "support_misses")}
    report = {
        "schema": "rineya-dream-report-v1", "kind": catalog["kind"],
        "status": "PASS_SYNTHETIC_MECHANICS_ONLY" if catalog["kind"] == "synthetic" else "HISTORICAL_REPLAY_COMPLETED",
        "freeze_hash": envelope["sha256"], "engine_hash": payload["engine_hash"],
        "catalog_hash": payload["catalog_hash"], "heldout_worlds": len(worlds),
        "independent_declared_groups": len(set(membership.values())), "episodes": len(rows),
        "candidate_count": payload["candidate_count"], "arms": summaries,
        "primary_comparison": paired_comparison(rows, membership, plan),
        "native_model_loads": 0, "native_training_steps": 0, "llm_calls": 0,
        "live_controller_deployments": 0, "live_efficiency_improvement_established": False,
        "cost_note": "Attempt budgets are matched. Unknown/missing costs remain null in traces. No GPU-budget matching claimed.",
    }
    write_new(run / "episodes.json", rows)
    write_new(run / "report.json", report)
    return report
