"""Strict paired evaluation joining; no evaluation feedback into training/allocation."""
from collections import defaultdict
import random
import torch

MATCH = ("dataset_sha256", "base_sha256", "tokenizer_sha256", "runtime_sha256", "scorer_sha256", "decoding_sha256", "bank_id", "split")


def indexed(rows):
    out = {}
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ("id", "parent_id", "domain")) or type(row.get("correct")) is not bool or row["id"] in out:
            raise ValueError("invalid/duplicate evaluation row")
        if not isinstance(row.get("prompt_sha256"), str) or len(row["prompt_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["prompt_sha256"]):
            raise ValueError("missing prompt hash")
        out[row["id"]] = row
    if not out:
        raise ValueError("empty evaluation bank")
    return out


def paired(reference, candidate, reference_meta, candidate_meta, *, seed=17, draws=1000):
    for key in MATCH:
        if key not in reference_meta or not reference_meta[key] or reference_meta[key] != candidate_meta.get(key):
            raise ValueError(f"unmatched evaluation metadata: {key}")
    for key in (k for k in MATCH if k.endswith("sha256")):
        v = reference_meta[key]
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"invalid evaluation hash: {key}")
    if type(draws) is not int or draws < 1:
        raise ValueError("positive bootstrap draws required")
    a, b = indexed(reference), indexed(candidate)
    if a.keys() != b.keys():
        raise ValueError("no row dropping/intersection allowed")
    domains, parents = defaultdict(list), defaultdict(list)
    for key in sorted(a):
        x, y = a[key], b[key]
        if (x["parent_id"], x["domain"], x["prompt_sha256"]) != (y["parent_id"], y["domain"], y["prompt_sha256"]):
            raise ValueError("row provenance mismatch")
        domains[x["domain"]].append((x["correct"], y["correct"]))
        parents[x["parent_id"]].append(int(y["correct"]) - int(x["correct"]))
    result = {}
    for domain, pairs in domains.items():
        source = sum(x for x, y in pairs)
        kept = sum(x and y for x, y in pairs)
        gains = sum(not x and y for x, y in pairs)
        damage = sum(x and not y for x, y in pairs)
        result[domain] = {"n": len(pairs), "acquisition_net": (gains-damage)/len(pairs),
                          "gains": gains, "damage": damage, "retention_fraction": kept/source if source else None}
    rng = random.Random(seed)
    clusters = [parents[p] for p in sorted(parents)]
    samples = []
    for _ in range(draws):
        draw = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        samples.append(sum(map(sum, draw)) / sum(map(len, draw)))
    ci = torch.quantile(torch.tensor(samples, dtype=torch.float64), torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"domains": result, "parent_clustered_net_accuracy_ci95": ci, "independent_parents": len(parents),
            "bootstrap_seed": seed, "bootstrap_draws": draws, "status": "DESCRIPTIVE_EXPOSED_EVALUATION",
            "quality_improvement_established": False}


def forgetting_matrix(order, snapshots):
    # snapshots[0] is base; all later entries score every atomic skill, including not-yet-trained.
    if len(snapshots) != len(order) + 1 or any(set(x) != set(order) for x in snapshots):
        raise ValueError("base plus one complete snapshot per skill required")
    if any(type(v) not in (int, float) or not 0 <= v <= 1 for x in snapshots for v in x.values()):
        raise ValueError("invalid atomic accuracy")
    return {s: {"base": snapshots[0][s], "after": [x[s] for x in snapshots[1:]],
                "acquisition": snapshots[i+1][s]-snapshots[0][s],
                "final_minus_at_acquisition": snapshots[-1][s]-snapshots[i+1][s],
                "forgetting_from_best": max(x[s] for x in snapshots[i+1:])-snapshots[-1][s]}
            for i, s in enumerate(order)}
