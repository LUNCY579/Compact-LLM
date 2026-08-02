from __future__ import annotations

"""Strict verification for Evaluator v4.1's supported response surface."""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (
    QUALITY_EVALUATION_VERSION,
    build_corpus_profile,
    evaluate_response,
    load_quality_config,
)

PACKET = ROOT / "data/evaluation/quality_v4_1_comprehensive_calibration.jsonl"
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_1.jsonl"
REPORT = ROOT / "reports/QUALITY_V4_1_COMPREHENSIVE_CALIBRATION_REPORT.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels(result: dict[str, Any]) -> set[str]:
    return set(result.get("critical_failures") or []) | set(result.get("major_issues") or []) | set(result.get("minor_issues") or [])


def main() -> None:
    if QUALITY_EVALUATION_VERSION != "4.1":
        raise SystemExit(f"Expected evaluator 4.1, found {QUALITY_EVALUATION_VERSION}")
    packet = read_jsonl(PACKET)
    benchmark = read_jsonl(BENCHMARK)
    corpus = build_corpus_profile([item["record"] for item in packet])
    config = load_quality_config()
    failures: list[str] = []
    evaluated: list[dict[str, Any]] = []
    detected_modes: Counter[str] = Counter()

    for item in packet:
        reply = str(item["reply"])
        token_count = int(item.get("visible_model_tokens_override", max(1, len(reply.split()))))
        result = evaluate_response(
            item["record"], reply, max(1, len(reply.split())),
            visible_model_tokens=token_count, config=config, corpus_profile=corpus,
        )
        actual = labels(result)
        lo = float(item.get("expected_score_min", 0.0))
        hi = float(item.get("expected_score_max", 1.0))
        include = set(item.get("expected_labels_include") or [])
        exclude = set(item.get("expected_labels_exclude") or [])
        errors: list[str] = []
        if not (lo - 1e-9 <= float(result["quality_score"]) <= hi + 1e-9):
            errors.append(f"score {result['quality_score']:.3f} not in [{lo:.3f}, {hi:.3f}]")
        missing = include - actual
        if missing:
            errors.append(f"missing labels {sorted(missing)}")
        forbidden_labels = {label for label in exclude if label not in {"critical", "major", "minor"}}
        present = forbidden_labels & actual
        if present:
            errors.append(f"forbidden labels {sorted(present)}")
        if "critical" in exclude and result.get("critical_failures"):
            errors.append(f"unexpected critical labels {result['critical_failures']}")
        if "major" in exclude and result.get("major_issues"):
            errors.append(f"unexpected major labels {result['major_issues']}")
        if "minor" in exclude and result.get("minor_issues"):
            errors.append(f"unexpected minor labels {result['minor_issues']}")
        for mode, score in (result.get("detected_response_mode_confidences") or {}).items():
            if float(score) >= 0.65:
                detected_modes[mode] += 1
        if errors:
            failures.append(f"{item['id']}: " + "; ".join(errors))
        evaluated.append({"item": item, "result": result, "errors": errors})

    official_categories = {str(row["category"]) for row in benchmark}
    good_categories = {
        tag.split(":", 1)[1]
        for item in packet for tag in item.get("coverage_tags", [])
        if tag.startswith("category:") and "category_good" in item.get("coverage_tags", [])
    }
    poor_categories = {
        tag.split(":", 1)[1]
        for item in packet for tag in item.get("coverage_tags", [])
        if tag.startswith("category:") and "category_poor" in item.get("coverage_tags", [])
    }
    if good_categories != official_categories:
        failures.append(f"category-good coverage mismatch: missing={sorted(official_categories-good_categories)} extra={sorted(good_categories-official_categories)}")
    if poor_categories != official_categories:
        failures.append(f"category-poor coverage mismatch: missing={sorted(official_categories-poor_categories)} extra={sorted(poor_categories-official_categories)}")

    taxonomy_labels = set(config.allowed_scored_labels)
    covered_labels = {
        label for item in packet if "label_coverage" in item.get("coverage_tags", [])
        for label in item.get("expected_labels_include", [])
    }
    if covered_labels != taxonomy_labels:
        failures.append(f"label coverage mismatch: missing={sorted(taxonomy_labels-covered_labels)} extra={sorted(covered_labels-taxonomy_labels)}")

    authored_modes = {
        str(mode) for row in benchmark
        for mode in (row.get("response_contract") or {}).get("expected_modes", [])
    }
    missing_modes = authored_modes - set(detected_modes)
    if missing_modes:
        failures.append(f"authored mode coverage missing: {sorted(missing_modes)}")

    lines = [
        "# Quality Evaluator v4.1 comprehensive calibration report",
        "",
        "This deterministic packet covers the complete response surface supported by the repository: every official benchmark category, every authored contract instance through its reference, every scored label, every functional response mode, real Phase 2B regressions, partial answers, and adversarial keyword traps.",
        "",
        f"- Evaluator version: **{QUALITY_EVALUATION_VERSION}**",
        f"- Total calibration cases: **{len(packet)}**",
        f"- Official reference-contract cases: **{sum(1 for item in packet if 'official_reference' in item.get('coverage_tags', []))}**",
        f"- Supported categories covered good + poor: **{len(official_categories)}/{len(official_categories)}**",
        f"- Scored labels covered: **{len(covered_labels)}/{len(taxonomy_labels)}**",
        f"- Authored response modes observed: **{len(authored_modes - missing_modes)}/{len(authored_modes)}**",
        f"- Exact expectation matches: **{len(packet)-len([row for row in evaluated if row['errors']])}/{len(packet)}**",
        f"- Verification failures: **{len(failures)}**",
        "",
        "## Functional-mode coverage",
        "",
    ]
    lines.extend(f"- `{mode}`: {detected_modes[mode]} strong calibration occurrence(s)" for mode in sorted(authored_modes))
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in failures[:100]) if failures else lines.append("- None")
    lines.extend(["", "## Calibration cases", "", "| ID | Score | Labels | Status | Reply |", "|---|---:|---|---|---|"])
    for row in evaluated:
        item, result = row["item"], row["result"]
        lab = ", ".join(sorted(labels(result))) or "none"
        status = "PASS" if not row["errors"] else "FAIL"
        reply = str(item["reply"]).replace("|", "\\|").replace("\n", " ↵ ")
        lines.append(f"| {item['id']} | {result['quality_score']:.3f} | {lab} | {status} | {reply} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        print("\n".join(failures[:100]))
        raise SystemExit(f"Evaluator v4.1 calibration failed: {len(failures)} issue(s). See {REPORT}")
    print(f"Evaluator v4.1 verification passed: {len(packet)}/{len(packet)} cases, {len(official_categories)} categories, {len(taxonomy_labels)} scored labels, {len(authored_modes)} authored modes.")
    print(REPORT)


if __name__ == "__main__":
    main()
