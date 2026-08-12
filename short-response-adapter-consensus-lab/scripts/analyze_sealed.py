from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.analysis.statistics import holm_adjust, paired_binary_statistics
from consensus_lab.io import read_jsonl, sha256_file, write_json_atomic
from consensus_lab.paths import LabPaths


def _select(rows: list[dict[str, Any]], committee: str, method: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["example_id"]): row
        for row in rows
        if row["committee"] == committee and row["method"] == method
    }


def main() -> None:
    paths = LabPaths.discover(LAB_ROOT)
    source = paths.assert_output_path(LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl")
    rows = read_jsonl(source)
    conditions = {
        "backbone": _select(rows, "backbone_k1", "hard_plurality"),
        "track_a": _select(rows, "track_a_k1", "hard_plurality"),
        "abstention": _select(rows, "abstention_k1", "hard_plurality"),
        "hard_k7": _select(rows, "full_k7", "hard_plurality"),
        "probability_k7": _select(rows, "full_k7", "probability_mean"),
    }
    expected_ids = set(conditions["hard_k7"])
    if any(set(value) != expected_ids for value in conditions.values()):
        raise RuntimeError("Sealed conditions are not paired on identical example IDs")
    specs = [
        ("hard_k7 vs backbone", "hard_k7", "backbone"),
        ("hard_k7 vs Track A", "hard_k7", "track_a"),
        ("hard_k7 vs abstention specialist", "hard_k7", "abstention"),
        ("hard_k7 vs probability mean", "hard_k7", "probability_k7"),
        ("probability mean vs backbone", "probability_k7", "backbone"),
        ("probability mean vs Track A", "probability_k7", "track_a"),
        ("probability mean vs abstention specialist", "probability_k7", "abstention"),
    ]
    comparisons = []
    ids = sorted(expected_ids)
    for index, (label, left_name, right_name) in enumerate(specs):
        result = paired_binary_statistics(
            [bool(conditions[left_name][value]["correct"]) for value in ids],
            [bool(conditions[right_name][value]["correct"]) for value in ids],
            seed=20260900 + index,
        ).to_dict()
        comparisons.append({"label": label, "condition_a": left_name, "condition_b": right_name, **result})
    adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_mcnemar_p"] = adjusted_p
    rates = {
        name: sum(bool(row["correct"]) for row in condition.values()) / len(condition)
        for name, condition in conditions.items()
    }
    k1_invariants = {}
    for committee in ("backbone_k1", "track_a_k1", "abstention_k1"):
        hard = _select(rows, committee, "hard_plurality")
        probability = _select(rows, committee, "probability_mean")
        k1_invariants[committee] = {
            "paired_n": len(ids),
            "exact_token_sequence_match_rate": sum(
                hard[value]["output_token_ids"] == probability[value]["output_token_ids"] for value in ids
            )
            / len(ids),
        }
    packet = {
        "schema_version": "1.0",
        "status": "FROZEN_SEALED_FINAL_NO_POST_SEALED_SELECTION",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "frozen_manifest": str(LAB_ROOT / "artifacts/frozen_configs/frozen_final_config.json"),
        "paired_example_count": len(ids),
        "condition_correctness": rates,
        "paired_comparisons": comparisons,
        "k1_method_invariants": k1_invariants,
        "interpretation": {
            "hard_beats_base": rates["hard_k7"] > rates["backbone"],
            "hard_beats_track_a": rates["hard_k7"] > rates["track_a"],
            "hard_beats_best_individual": rates["hard_k7"] > max(rates["track_a"], rates["abstention"]),
            "hard_beats_probability_mean": rates["hard_k7"] > rates["probability_k7"],
            "warning": "The sealed packet is a preregistered 32-example resource-aware sample, not the full 4,050-example TEST split.",
        },
    }
    output = paths.assert_output_path(LAB_ROOT / "results/summaries/sealed_statistics.json")
    write_json_atomic(output, packet)
    print(json.dumps({"output": str(output), "rates": rates, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
