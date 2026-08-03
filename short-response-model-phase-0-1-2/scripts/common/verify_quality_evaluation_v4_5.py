from __future__ import annotations

"""Strict verification for Evaluator v4.5's supported response surface."""

import copy
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

from scripts.common.build_v4_5_response_contract_benchmark import build_contract

PACKET = ROOT / "data/evaluation/quality_v4_5_comprehensive_calibration.jsonl"
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_5.jsonl"
REPORT = ROOT / "reports/QUALITY_V4_5_COMPREHENSIVE_CALIBRATION_REPORT.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels(result: dict[str, Any]) -> set[str]:
    return set(result.get("critical_failures") or []) | set(result.get("major_issues") or []) | set(result.get("minor_issues") or [])


def main() -> None:
    if QUALITY_EVALUATION_VERSION != "4.5":
        raise SystemExit(f"Expected evaluator 4.5, found {QUALITY_EVALUATION_VERSION}")
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

    # Prompt-function coverage comes from actual messages, never category or reference.
    prompt_functions = {
        mode for row in evaluated
        for mode in (row["result"].get("prompt_function_expectations") or {}).get("expected_modes", [])
    }
    observed_functions = {
        mode for row in evaluated
        for mode, confidence in (row["result"].get("detected_response_mode_confidences") or {}).items()
        if float(confidence) >= 0.65
    }

    collapse_rows = [row for row in evaluated if "real_collapse_150" in row["item"].get("coverage_tags", [])]
    if len(collapse_rows) != 150:
        failures.append(f"real-collapse coverage expected 150, found {len(collapse_rows)}")
    for row in collapse_rows:
        result = row["result"]
        if float(result["quality_score"]) != 0.0 or not result.get("critical_failures"):
            failures.append(f"{row['item']['id']}: real collapse was not zero-quality critical")

    coherent_rows = [row for row in evaluated if "real_coherent_150" in row["item"].get("coverage_tags", [])]
    if len(coherent_rows) != 150:
        failures.append(f"real-coherent coverage expected 150, found {len(coherent_rows)}")

    targeted_rows = [
        row for row in evaluated
        if {"v4_4_targeted_regression", "v4_5_targeted_regression"}
        & set(row["item"].get("coverage_tags", []))
    ]
    if len(targeted_rows) != 41:
        failures.append(f"targeted v4.5 coverage expected 41, found {len(targeted_rows)}")

    # Category is reporting metadata only: changing or removing it cannot alter
    # any case-level scoring result.
    invariant_fields = (
        "quality_score", "quality_dimensions", "quality_weights",
        "quality_defects", "critical_failures", "major_issues", "minor_issues",
        "contract_fulfillment_score", "prompt_function_alignment_score",
    )
    for source in benchmark[:20]:
        reply = str(source["reference"])
        token_count = max(1, len(reply.split()))
        variants = []
        for category in (source.get("category"), "deliberately_wrong_category", None):
            candidate = copy.deepcopy(source)
            if category is None:
                candidate.pop("category", None)
            else:
                candidate["category"] = category
            variants.append(evaluate_response(candidate, reply, token_count, visible_model_tokens=token_count, config=config, corpus_profile=corpus))
        baseline = variants[0]
        for variant in variants[1:]:
            for field in invariant_fields:
                if variant.get(field) != baseline.get(field):
                    failures.append(f"category invariance failed for {source['id']} field {field}")

    # Strong contract-authoring invariance: rebuild from the same visible
    # messages after changing category, reference, and inherited contracts.
    # The resulting contract must be byte-for-byte identical.
    for source in benchmark:
        baseline_contract = build_contract(source)
        variants = []
        wrong = copy.deepcopy(source)
        wrong["category"] = "deliberately_wrong_category"
        wrong["reference"] = "Deliberately unrelated reference text."
        wrong["response_contract"] = {
            "schema_version": "0.0",
            "requirements": [{"id": "poison", "label": "explicit_required_content_missing"}],
        }
        variants.append(wrong)
        absent = copy.deepcopy(source)
        absent.pop("category", None)
        absent.pop("reference", None)
        absent.pop("response_contract", None)
        variants.append(absent)
        for variant in variants:
            if build_contract(variant) != baseline_contract:
                failures.append(f"fresh contract construction invariance failed for {source['id']}")
        if baseline_contract.get("contract_origin") != "freshly_authored_from_visible_messages_v4_5":
            failures.append(f"missing v4.5 contract origin for {source['id']}")
        if baseline_contract.get("contract_category_used") is not False:
            failures.append(f"category provenance is not false for {source['id']}")
        if baseline_contract.get("contract_reference_used") is not False:
            failures.append(f"reference provenance is not false for {source['id']}")
        if baseline_contract.get("contract_inherited_requirements_used") is not False:
            failures.append(f"inherited-contract provenance is not false for {source['id']}")

    # Review diagnostics are zero-cost: adding a legacy expected_modes warning
    # must not change score, labels, dimensions, or promotion evidence.
    # Legacy expected_modes are ignored in v4.5.
    probe = copy.deepcopy(benchmark[0])
    reply = str(probe["reference"])
    token_count = max(1, len(reply.split()))
    base = evaluate_response(probe, reply, token_count, visible_model_tokens=token_count, config=config, corpus_profile=corpus)
    probe.setdefault("response_contract", {})["expected_modes"] = ["privacy_assurance", "definition_or_explanation"]
    changed = evaluate_response(probe, reply, token_count, visible_model_tokens=token_count, config=config, corpus_profile=corpus)
    for field in invariant_fields:
        if changed.get(field) != base.get(field):
            failures.append(f"legacy expected_modes invariance failed for field {field}")

    lines = [
        "# Quality Evaluator v4.5 comprehensive calibration report",
        "",
        "This deterministic packet covers the complete response surface supported by the repository: every official benchmark category, every authored contract instance through its reference, every scored label, prompt-grounded conversational functions, real Phase 2B regressions, partial answers, and adversarial keyword traps.",
        "",
        f"- Evaluator version: **{QUALITY_EVALUATION_VERSION}**",
        f"- Total calibration cases: **{len(packet)}**",
        f"- Human-adjudicated coherent Phase 2B cases: **{len(coherent_rows)}/150**",
        f"- Targeted v4.5 regression cases: **{len(targeted_rows)}/41**",
        f"- Official reference-contract cases: **{sum(1 for item in packet if 'official_reference' in item.get('coverage_tags', []))}**",
        f"- Supported categories covered good + poor: **{len(official_categories)}/{len(official_categories)}**",
        f"- Scored labels covered: **{len(covered_labels)}/{len(taxonomy_labels)}**",
        f"- Prompt-grounded functions inferred: **{len(prompt_functions)}**",
        f"- Prompt functions strongly observed: **{len(prompt_functions & observed_functions)}/{len(prompt_functions)}**",
        f"- Real collapsed responses rejected: **{sum(1 for row in collapse_rows if row['result']['quality_score'] == 0.0 and row['result'].get('critical_failures'))}/150**",
        f"- Exact expectation matches: **{len(packet)-len([row for row in evaluated if row['errors']])}/{len(packet)}**",
        f"- Verification failures: **{len(failures)}**",
        "",
        "## Functional-mode coverage",
        "",
    ]
    lines.extend(f"- `{mode}`: {detected_modes[mode]} strong calibration occurrence(s)" for mode in sorted(prompt_functions))
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
        raise SystemExit(f"Evaluator v4.5 calibration failed: {len(failures)} issue(s). See {REPORT}")
    print(f"Evaluator v4.5 verification passed: {len(packet)}/{len(packet)} cases, {len(official_categories)} reporting categories, {len(taxonomy_labels)} scored labels, 150/150 real collapses.")
    print(REPORT)


if __name__ == "__main__":
    main()
