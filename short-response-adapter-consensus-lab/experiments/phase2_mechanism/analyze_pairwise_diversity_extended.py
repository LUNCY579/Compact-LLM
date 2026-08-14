from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/sealed/v22_frozen_finalists.jsonl"
DEV_DIVERSITY = ROOT / "results/phase2_mechanism/teacher_forced_expert_diversity.csv"
OUTPUT = ROOT / "results/phase2_mechanism/pairwise_diversity_extended.json"
CSV = ROOT / "results/phase2_mechanism/pairwise_diversity_extended.csv"
REPORT = ROOT / "reports/PHASE2_PAIRWISE_DIVERSITY_EXTENDED.md"
EXPECTED_RAW_SHA256 = "d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d"
EXPERTS = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def binary_mutual_information(counts: list[int]) -> float:
    total = sum(counts)
    joint = [value / total for value in counts]
    left = [joint[0] + joint[1], joint[2] + joint[3]]
    right = [joint[0] + joint[2], joint[1] + joint[3]]
    result = 0.0
    for index, probability in enumerate(joint):
        if probability:
            i, j = divmod(index, 2)
            result += probability * math.log(probability / (left[i] * right[j]))
    return result


def main() -> None:
    if sha256(RAW) != EXPECTED_RAW_SHA256:
        raise RuntimeError("Cornerstone hash differs from the verified artifact")
    dev_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with DEV_DIVERSITY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dev_rows[(row["expert_a"], row["expert_b"])] = row
    if len(dev_rows) != 21:
        raise RuntimeError("Expected all 21 expert pairs in the DEV contingency table")

    js_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    examples = 0
    tokens = 0
    for row in iter_jsonl(RAW):
        if row["committee"] != "full_k7" or row["method"] != "hard_plurality":
            continue
        examples += 1
        for step in row["steps"]:
            tokens += 1
            order = list(step["expert_order"])
            if order != EXPERTS:
                raise RuntimeError("Expert order changed in the frozen trace")
            matrix = step["pairwise_js_matrix"]
            for left in range(len(EXPERTS)):
                for right in range(left + 1, len(EXPERTS)):
                    pair = (EXPERTS[left], EXPERTS[right])
                    value = float(matrix[left][right])
                    js_values[pair]["all"].append(value)
                    js_values[pair]["correct" if row["correct"] else "wrong"].append(value)
    if examples != 4050 or tokens != 23204 or len(js_values) != 21:
        raise RuntimeError(f"Unexpected full trace extent: examples={examples}, tokens={tokens}, pairs={len(js_values)}")

    rows = []
    for pair in sorted(js_values):
        source = dev_rows[pair]
        a_only = int(source["a_only_correct"])
        b_only = int(source["b_only_correct"])
        both_correct = int(source["both_correct"])
        both_wrong = int(source["both_wrong"])
        contingency = [both_correct, a_only, b_only, both_wrong]
        values = js_values[pair]
        rows.append(
            {
                "expert_a": pair[0],
                "expert_b": pair[1],
                "full_test_token_count": len(values["all"]),
                "mean_pairwise_js_full_test": sum(values["all"]) / len(values["all"]),
                "mean_pairwise_js_on_correct_sequences": sum(values["correct"]) / len(values["correct"]),
                "mean_pairwise_js_on_wrong_sequences": sum(values["wrong"]) / len(values["wrong"]),
                "dev_gold_prefix_token_count": sum(contingency),
                "dev_correctness_mutual_information_nats": binary_mutual_information(contingency),
                "dev_success_jaccard": both_correct / (both_correct + a_only + b_only),
                "dev_failure_jaccard": both_wrong / (both_wrong + a_only + b_only),
                "dev_top1_disagreement_rate": float(source["top1_disagreement_rate"]),
                "dev_double_fault_rate": float(source["double_fault_rate"]),
                "dev_q_statistic": float(source["q_statistic"]),
                "dev_correctness_phi": float(source["correctness_phi"]),
                "analysis_role": "TEST DESCRIPTIVE JS + DEV MECHANISTIC CORRECTNESS ASSOCIATION",
            }
        )

    packet = {
        "schema_version": "1.0",
        "status": "PAIRWISE_DIVERSITY_EXTENDED_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(rows),
        "full_test_example_count": examples,
        "full_test_token_count": tokens,
        "dev_gold_prefix_token_count": 18760,
        "pairs": rows,
        "provenance": {
            "raw_path": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "raw_sha256": sha256(RAW),
            "dev_diversity_path": str(DEV_DIVERSITY.relative_to(ROOT)).replace("\\", "/"),
            "dev_diversity_sha256": sha256(DEV_DIVERSITY),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "new_model_inference": False,
        },
        "limitations": [
            "Full-TEST JS values are measured along the hard-vote trajectory and are associated with, not causal for, final sequence correctness.",
            "DEV mutual information and Jaccard values describe dependent gold-prefix token decisions.",
            "High distributional divergence may reflect incompetence rather than useful complementarity.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    leading = sorted(rows, key=lambda row: row["mean_pairwise_js_full_test"], reverse=True)
    report_rows = [
        f'| {row["expert_a"]} / {row["expert_b"]} | {row["mean_pairwise_js_full_test"]:.4f} | '
        f'{row["dev_top1_disagreement_rate"]:.4f} | {row["dev_double_fault_rate"]:.4f} | '
        f'{row["dev_correctness_mutual_information_nats"]:.4f} | {row["dev_success_jaccard"]:.4f} | '
        f'{row["dev_failure_jaccard"]:.4f} |'
        for row in leading
    ]
    REPORT.write_text(
        "\n".join(
            [
                "# Extended pairwise expert diversity",
                "",
                "Status: **PAIRWISE_DIVERSITY_EXTENDED_COMPLETE**",
                "",
                "| Expert pair | Mean JS (TEST hard path) | DEV disagreement | DEV double fault | DEV correctness MI (nats) | Success Jaccard | Failure Jaccard |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *report_rows,
                "",
                "Distributional and correctness diversity are descriptive. Large values are not automatically useful: low-competence experts can be diverse for the wrong reason.",
                "",
                f"Raw TEST SHA-256: `{EXPECTED_RAW_SHA256}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": packet["status"], "pairs": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
