"""No-truncation greedy collection and paired, parent-clustered comparisons."""
from collections import defaultdict
import random
import time

import torch

from .data import validate_cases, digest


def collect(model, tokenizer, cases, *, seed, context_limit, max_new_tokens, guard=None):
    validate_cases(cases)
    if type(context_limit) is not int or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Positive integer context/generation limits required")
    if any(len(row["prompt_ids"]) + max_new_tokens > context_limit for row in cases):
        raise ValueError("Prompt plus reserved completion exceeds context; no truncation allowed")
    modes = [(m, m.training) for m in model.modules()]
    records = []
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.inference_mode():
            for row in cases:
                if guard is not None:
                    guard()
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if output.ndim != 2 or output.shape[0] != 1 or not torch.equal(output[:, :ids.shape[1]], ids):
                    raise ValueError("Generation changed/truncated the prompt")
                generated = output[0, ids.shape[1]:].tolist()
                if len(generated) > max_new_tokens:
                    raise ValueError("Generation exceeded its budget")
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                record = {k: row[k] for k in ("id", "parent_id", "split", "task", "role", "length", "depth",
                                               "answer", "answerable", "synthetic", "fixture")}
                record.update(seed=seed, prompt_sha256=digest(row["prompt_ids"]), prediction=text, correct=text == row["answer"].strip(),
                              abstained=text == "UNKNOWN", generated_tokens=len(generated),
                              seconds=time.perf_counter() - started,
                              hit_generation_cap=len(generated) == max_new_tokens,
                              truncated=False)
                records.append(record)
        return records
    finally:
        for m, mode in modes:
            m.training = mode


def _macro(rows, role):
    cells = defaultdict(list)
    for row in rows:
        if row["role"] == role:
            cells[(row["task"], row["length"], row["depth"])].append(row["delta"])
    if not cells:
        return None
    return sum(sum(v) / len(v) for v in cells.values()) / len(cells)


def compare(reference, candidate, *, resamples=2000, bootstrap_seed=733):
    """Two-way paired bootstrap: resample seeds AND semantic parents.

    The SAME parent draw is used across resampled seeds, preserving repeated
    length/depth views and repeated evaluation examples across training seeds.
    This is uncertainty estimation, not an automatic release certificate.
    """
    if type(resamples) is not int or resamples < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    def index(rows):
        result = {}
        for row in rows:
            key = (row["seed"], row["id"])
            if key in result or type(row["correct"]) is not bool or row.get("truncated", True):
                raise ValueError("Duplicate, invalid or truncated score record")
            result[key] = row
        if not result:
            raise ValueError("Empty predictions")
        return result
    ref, cand = index(reference), index(candidate)
    if ref.keys() != cand.keys():
        raise ValueError("Paired prediction sets must match exactly")
    pairs = []
    for key, before in ref.items():
        after = cand[key]
        fields = ("parent_id", "split", "task", "role", "length", "depth", "answer", "answerable",
                  "synthetic", "fixture", "prompt_sha256")
        if any(before[f] != after[f] for f in fields):
            raise ValueError("Mismatched paired metadata")
        pairs.append(dict(before, delta=int(after["correct"]) - int(before["correct"])))
    seeds = sorted({p["seed"] for p in pairs})
    parent_groups = defaultdict(set)
    indexed = defaultdict(list)
    for p in pairs:
        parent_groups[(p["role"], p["task"])].add(p["parent_id"])
        indexed[(p["seed"], p["parent_id"])].append(p)
    parents = set().union(*parent_groups.values())
    if sum(len(v) for v in parent_groups.values()) != len(parents):
        raise ValueError("A semantic parent must have one task and endpoint")
    for parent in parents:
        views = [{(r["id"], r["role"], r["task"], r["length"], r["depth"])
                  for r in indexed[(seed, parent)]} for seed in seeds]
        if not views[0] or any(v != views[0] for v in views):
            raise ValueError("All seeds must share the same complete parent/view grid")
    roles = sorted({p["role"] for p in pairs})
    distributions = {role: [] for role in roles}
    rng = random.Random(bootstrap_seed)
    for _ in range(resamples):
        drawn_seeds = rng.choices(seeds, k=len(seeds))
        drawn_parents = []
        for group in sorted(parent_groups):
            population = sorted(parent_groups[group])
            drawn_parents.extend(rng.choices(population, k=len(population)))
        sample = [row for seed in drawn_seeds for parent in drawn_parents for row in indexed[(seed, parent)]]
        for role in roles:
            distributions[role].append(_macro(sample, role))
    endpoints = {}
    for role in roles:
        values = sorted(distributions[role])
        endpoints[role] = {"delta": _macro(pairs, role),
                           "ci95": [values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]],
                           "per_seed": {str(s): _macro([p for p in pairs if p["seed"] == s], role) for s in seeds}}
    reasons = []
    if len(seeds) < 3:
        reasons.append("fewer_than_three_seeds")
    if any(p["fixture"] for p in pairs):
        reasons.append("fixture_evidence")
    if any(p["split"] != "confirmation" for p in pairs):
        reasons.append("not_untouched_confirmation")
    if not any(not p["synthetic"] and p["role"] == "long_context" for p in pairs):
        reasons.append("no_downstream_long_context_evidence")
    if len({p["task"] for p in pairs if p["role"] == "long_context"}) < 2:
        reasons.append("fewer_than_two_long_context_tasks")
    for role in ("long_context", "acquisition", "retention", "abstention"):
        independent = len({p["parent_id"] for p in pairs if p["role"] == role})
        if independent < 256:
            reasons.append(f"{role}_fewer_than_256_parents_power_review_required")
        if role not in endpoints:
            reasons.append(f"missing_{role}")
        elif role == "long_context":
            if endpoints[role]["delta"] < 0.05 or endpoints[role]["ci95"][0] <= 0:
                reasons.append("long_context_gain_not_established")
        elif endpoints[role]["ci95"][0] < -0.02:
            reasons.append(f"{role}_noninferiority_not_established")
    return {"status": "REVIEW_ELIGIBLE_NOT_RELEASED" if not reasons else "NOT_PROMOTABLE",
            "reasons": reasons, "endpoints": endpoints, "seeds": seeds,
            "unique_parents": len(parents), "paired_rows": len(pairs),
            "reference_correct_to_wrong": sum(p["delta"] < 0 for p in pairs),
            "wrong_to_correct": sum(p["delta"] > 0 for p in pairs),
            "candidate_unnecessary_abstentions": sum(r["answerable"] and r["abstained"] for r in candidate),
            "note": "Thresholds are proposed screening/review rules; provenance, power and export require human review."}
