from __future__ import annotations

from collections import Counter

from consensus_lab.evaluation.sampling import deterministic_stratified_sample


def test_small_stratified_sample_balances_primary_decision() -> None:
    rows = [
        {
            "id": f"{decision}-{index}",
            "decision": decision,
            "difficulty": "easy",
            "reasoning_family": f"family-{index % 4}",
            "domain": f"domain-{index % 8}",
        }
        for decision in ("answer", "abstain")
        for index in range(40)
    ]
    first = deterministic_stratified_sample(rows, limit=16, seed=7)
    second = deterministic_stratified_sample(rows, limit=16, seed=7)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert Counter(row["decision"] for row in first) == {"answer": 8, "abstain": 8}
    assert len({row["domain"] for row in first}) >= 4
