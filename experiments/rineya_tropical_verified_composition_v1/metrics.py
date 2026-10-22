"""Paired outcomes with parent-clustered uncertainty; never auto-promotes models."""
from __future__ import annotations

import math
import random
import statistics


def pass_at_k(successes: int, attempts: int, k: int) -> float:
    if any(type(x) is not int for x in (successes, attempts, k)):
        raise ValueError("Counts must be integers")
    if not 0 <= successes <= attempts or not 1 <= k <= attempts:
        raise ValueError("Invalid pass@k counts")
    return 1.0 - math.comb(attempts - successes, k) / math.comb(attempts, k) if attempts - successes >= k else 1.0


def validate_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = row["task_id"]
        if not identity or identity in result or not row["parent_id"] or not row["task_key"] or not row.get("contract_id"):
            raise ValueError("Missing or duplicate evaluation identity")
        outcomes = row["successes"]
        if not outcomes or any(type(x) is not bool for x in outcomes):
            raise ValueError("Expected nonempty binary verifier outcomes")
        result[identity] = row
    if not result:
        raise ValueError("Empty evaluation")
    return result


def summarize(rows: list[dict]) -> dict:
    rows = list(validate_rows(rows).values())
    per_row = [sum(r["successes"]) / len(r["successes"]) for r in rows]
    return {"tasks": len(rows), "pass_at_1": statistics.mean(per_row),
            "all_draws_success_coverage": statistics.mean(any(r["successes"]) for r in rows),
            "attempts": sum(len(r["successes"]) for r in rows)}


def paired(before: list[dict], after: list[dict], seed: int = 13,
           bootstrap_draws: int = 1000) -> dict:
    left, right = validate_rows(before), validate_rows(after)
    if left.keys() != right.keys():
        raise ValueError("Unpaired evaluation bank")
    if type(bootstrap_draws) is not int or bootstrap_draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    clusters: dict[str, list[float]] = {}
    retained = lost = gained = enrolled = 0
    for identity, a in left.items():
        b = right[identity]
        if (a["task_key"], a["parent_id"], a["contract_id"], len(a["successes"])) != (
                b["task_key"], b["parent_id"], b["contract_id"], len(b["successes"])):
            raise ValueError("Evaluation identity or decoding-count mismatch")
        delta = statistics.mean(b["successes"]) - statistics.mean(a["successes"])
        clusters.setdefault(a["parent_id"], []).append(delta)
        # Enrollment is a fixed first draw, not whichever baseline draw succeeded.
        source_correct = a["successes"][0]
        enrolled += source_correct
        retained += source_correct and b["successes"][0]
        lost += source_correct and not b["successes"][0]
        gained += not source_correct and b["successes"][0]
    values = [statistics.mean(v) for _, v in sorted(clusters.items())]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values)))
                     for _ in range(bootstrap_draws))
    return {"parent_macro_delta": statistics.mean(values), "parents": len(values),
            "ci95_parent_bootstrap": [samples[int(0.025 * bootstrap_draws)],
                                      samples[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))]],
            "source_correct_first_draw": enrolled, "retained": retained, "lost": lost,
            "gained_first_draw": gained,
            "retention": retained / enrolled if enrolled else None,
            "decision": "DESCRIPTIVE_ONLY"}
