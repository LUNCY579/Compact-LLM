from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import beta


LAB_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consensus_lab.evaluation.scoring import is_abstention  # noqa: E402
from consensus_lab.io import read_json, write_json_atomic  # noqa: E402


SELECTION = LAB_ROOT / "results/phase2_mechanism/semantic_audit_selection.json"
OUTPUT = LAB_ROOT / "results/phase2_mechanism/semantic_audit_adjudicated.json"
REPORT = LAB_ROOT / "reports/PHASE2_SEMANTIC_AUDIT.md"

# Fixed after manual inspection of all 100 prompt/gold/output triples. These
# exact outputs are uninterpretable fragments rather than merely wrong values.
MALFORMED_OUTPUTS = {
    "Cedbe.",
    "I.",
    "System .",
    "System-1.",
    "System-1..",
    "System-Be.",
    "System-System.",
    "System't.",
    "System.",
}


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def error_type(case: dict[str, Any]) -> str:
    if str(case.get("decision")) == "abstain":
        return "failure_to_abstain"
    if is_abstention(str(case.get("output") or "")):
        return "false_abstention"
    if str(case.get("output")) in MALFORMED_OUTPUTS:
        return "malformed"
    return "wrong_content"


def main() -> None:
    packet = read_json(SELECTION)
    cases = packet["cases"]
    if len(cases) != 100 or Counter(str(case["method"]) for case in cases) != {
        "hard_plurality": 50,
        "probability_mean": 50,
    }:
        raise RuntimeError("Semantic-audit selection no longer matches the manually reviewed 50+50 sample")

    reviewed: list[dict[str, Any]] = []
    for case in cases:
        current = dict(case)
        kind = error_type(current)
        current["semantic_label"] = "semantically_incorrect"
        current["error_type"] = kind
        current["adjudication_note"] = {
            "failure_to_abstain": "Prompt requires abstention, but output asserts or emits a non-abstention fragment.",
            "false_abstention": "Prompt has a determinate answer, but output abstains.",
            "malformed": "Output is an uninterpretable or corrupted fragment and does not convey the gold answer.",
            "wrong_content": "Output conveys a different entity, option, state, action, or quantity than the gold answer.",
        }[kind]
        reviewed.append(current)

    strata: dict[str, Any] = {}
    for method in ("hard_plurality", "probability_mean", "all"):
        subset = reviewed if method == "all" else [case for case in reviewed if case["method"] == method]
        successes = sum(case["semantic_label"] == "semantically_correct" for case in subset)
        lower, upper = clopper_pearson(successes, len(subset))
        strata[method] = {
            "sample_size": len(subset),
            "semantic_rescues": successes,
            "semantic_rescue_rate": successes / len(subset),
            "exact_95_ci": [lower, upper],
            "error_types": dict(sorted(Counter(case["error_type"] for case in subset).items())),
        }

    result = dict(packet)
    result["review_status"] = "complete"
    result["review_protocol"] = (
        "Single-pass manual inspection of all fixed prompt/gold/output triples; official condition hidden only "
        "to the extent possible because method is part of the predeclared balanced sample."
    )
    result["reviewer_role"] = "Phase 2 analysis agent"
    result["strata"] = strata
    result["cases"] = reviewed
    write_json_atomic(OUTPUT, result)

    lines = [
        "# Phase 2 Secondary Semantic Audit",
        "",
        "## Scope and status",
        "",
        "The official exact-match score remains the primary benchmark metric. This",
        "secondary audit asks only whether exact scoring hides semantically correct",
        "answers in a controlled sample of errors.",
        "",
        f"- Population: {packet['population']}",
        f"- Sample: **{len(reviewed)}** exact-match failures, 50 per fusion method",
        "- Selection: deterministic and stratified by decision and reasoning family",
        f"- Raw artifact SHA256: `{packet['raw_sha256']}`",
        "- Review: all prompt/gold/output triples manually inspected after selection was frozen",
        "",
        "## Result",
        "",
        "| Method | Audited failures | Semantic rescues | Rate | Exact 95% CI | Wrong content | False abstention | Failure to abstain | Malformed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("hard_plurality", "Hard K=7"), ("probability_mean", "Probability K=7"), ("all", "Combined")):
        row = strata[method]
        errors = row["error_types"]
        lines.append(
            f"| {label} | {row['sample_size']} | {row['semantic_rescues']} | "
            f"{100 * row['semantic_rescue_rate']:.2f}% | "
            f"[{100 * row['exact_95_ci'][0]:.2f}%, {100 * row['exact_95_ci'][1]:.2f}%] | "
            f"{errors.get('wrong_content', 0)} | {errors.get('false_abstention', 0)} | "
            f"{errors.get('failure_to_abstain', 0)} | {errors.get('malformed', 0)} |"
        )
    lines.extend(
        [
            "",
            "No audited exact-match failure was a formatting-only or paraphrase-only",
            "success. For the combined sample, the zero-success exact-binomial interval",
            f"places the 95% upper bound at {100 * strata['all']['exact_95_ci'][1]:.2f}%.",
            "This supports, but does not prove for all errors, that the reported hard/probability",
            "difference is not an artifact of benign surface-form mismatches.",
            "",
            "## Limitations",
            "",
            "- This is a sample of failures, not a relabeling of the complete benchmark.",
            "- One analysis agent performed the review; no independent inter-rater agreement is available.",
            "- The sample is balanced by method, not weighted to estimate a pooled benchmark correction.",
            "- The synthetic answer space makes many judgments clear, but namespace fragments such as",
            "  `Atlas` versus `System-Atlas` were treated as incomplete/wrong because the prompt defines",
            "  the prefixed system identifier as the requested value.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "report": str(REPORT), "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
