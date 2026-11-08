"""Selectors see completed public candidates only, never oracle records."""
from __future__ import annotations

from collections import Counter
import random

from .contracts import exact, parse_answer, require


def public_candidate(record: dict) -> dict:
    return {"slot": record["slot"], "text": record.get("text", ""),
            "stop": record.get("stop", "unavailable"), "state": record["state"]}


def normalized(candidate: dict) -> str | None:
    exact(candidate, {"slot", "text", "stop", "state"}, "public candidate")
    if candidate["state"] != "returned" or candidate["stop"] != "eos":
        return None
    return parse_answer(candidate["text"])


def select(candidates: list[dict], policy: str, seed: int) -> int | None:
    require(policy in {"first", "random", "vote"}, "unknown selector")
    require(len({c["slot"] for c in candidates}) == len(candidates), "duplicate candidate slots")
    values = [normalized(c) for c in candidates]  # Enforce allowlist even for random/first.
    if not candidates:
        return None
    if policy == "first":
        return candidates[0]["slot"]
    if policy == "random":
        return random.Random(seed).choice(candidates)["slot"]
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None  # No newly synthesized abstention or answer.
    maximum = max(counts.values())
    return next(c["slot"] for c, v in zip(candidates, values)
                if v is not None and counts[v] == maximum)


def pass_at_k(n: int, c: int, k: int) -> float:
    require(all(type(v) is int for v in (n, c, k)) and n >= 1 and 0 <= c <= n and 1 <= k <= n,
            "invalid pass@k counts")
    if n-c < k:
        return 1.0
    product = 1.0
    for j in range(k):
        product *= (n-c-j)/(n-j)
    return 1.0-product
