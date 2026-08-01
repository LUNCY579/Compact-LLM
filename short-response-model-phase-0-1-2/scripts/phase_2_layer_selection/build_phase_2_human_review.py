from __future__ import annotations

"""Build the Phase 2A human-review packet from saved layer-study artifacts.

The static study intentionally keeps raw case-level generations for every
variant.  This report surfaces the strongest selective candidate beside the
all-layer reference and the RANDOM4/BOTTOM4 controls so the human reviewer can
see whether an aggregate ranking reflects meaningful behavior.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 2 static-study human-review packet.")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json",
    )
    parser.add_argument(
        "--all-results",
        type=Path,
        default=ROOT / "artifacts/dense_sft_evaluation/results.jsonl",
        help="Case-level generations for the all-layer Phase 1 reference.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "reports/PHASE_2_HUMAN_REVIEW_PACKET.md",
    )
    parser.add_argument(
        "--max-cases-per-variant",
        type=int,
        default=12,
        help="Maximum low-scoring/flagged examples rendered for each variant.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ").strip()


def variant_results_path(row: dict[str, Any]) -> Path | None:
    artifact_dir = row.get("artifact_dir")
    if not artifact_dir:
        return None
    return Path(str(artifact_dir)) / "evaluation" / "results.jsonl"


def choose_control(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in payload.get("controls", []):
        if str(row.get("variant", "")).lower() == name.lower():
            return row
    return None


def unique_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("variant", row.get("artifact_dir", id(row))))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def case_priority(row: dict[str, Any]) -> tuple[int, int, float, str]:
    critical = len(row.get("critical_failures") or [])
    major = len(row.get("major_issues") or [])
    score = float(row.get("quality_score", row.get("heuristic_score", 1.0)))
    # Flagged cases first, then low scores, with stable ID ordering.
    return (-critical, -major, score, str(row.get("id", "")))


def case_table(rows: list[dict[str, Any]], limit: int) -> str:
    if not rows:
        return "_Raw case-level output was not found. Re-run the associated evaluation before human approval._"
    selected = sorted(rows, key=case_priority)[: max(1, limit)]
    lines = [
        "| ID | Category | Quality | Intelligibility | Pass | Reply | Critical/hard | Major | Warnings |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            "| {id} | {category} | {score:.3f} | {intelligibility:.3f} | {quality_pass} | {reply} | {critical} | {major} | {warnings} |".format(
                id=escape(row.get("id")),
                category=escape(row.get("category")),
                score=float(row.get("quality_score", row.get("heuristic_score", 0.0))),
                intelligibility=float(row.get("intelligibility_score", 0.0)),
                quality_pass="YES" if row.get("quality_pass") else "NO",
                reply=escape(row.get("reply")),
                critical=escape(", ".join(row.get("critical_failures") or []) or "-"),
                major=escape(", ".join(row.get("major_issues") or []) or "-"),
                warnings=escape(", ".join(row.get("semantic_warnings") or []) or "-"),
            )
        )
    return "\n".join(lines)


def summary_row(label: str, row: dict[str, Any]) -> str:
    layers = row.get("layers", "all")
    vram_reduction = row.get("vram_reduction_vs_all")
    vram_text = "reference" if vram_reduction is None else f"{100.0 * float(vram_reduction):.1f}%"
    return (
        f"| {escape(label)} | {escape(row.get('variant', label))} | {escape(layers)} | "
        f"{float(row.get('mean_quality_score', row.get('mean_heuristic_score', 0.0))):.3f} | "
        f"{float(row.get('worst_decile_mean_quality_score', 0.0)):.3f} | "
        f"{float(row.get('quality_pass_rate', 0.0)):.3f} | "
        f"{float(row.get('intelligibility_pass_rate', 0.0)):.3f} | "
        f"{int(row.get('hard_failure_case_count', 0))} | {int(row.get('critical_failure_count', 0))} | "
        f"{int(row.get('major_case_count', row.get('major_issue_count', 0)))} | "
        f"{float(row.get('semantic_pass_rate', 1.0)):.3f} | {vram_text} |"
    )


def main() -> None:
    args = parse_args()
    payload = read_json(args.results)

    strongest = payload.get("strongest_candidate") or {}
    ranked = payload.get("ranked_single_layers") or []
    if not strongest and ranked:
        strongest = ranked[0]

    random4 = choose_control(payload, "random4")
    bottom4 = choose_control(payload, "bottom4")
    all_reference = dict(payload.get("all_layer_reference") or {})
    all_reference.setdefault("variant", "all_layers_phase1_reference")
    all_reference.setdefault("layers", "all")

    selected_rows = unique_rows(
        row
        for row in (strongest, random4, bottom4)
        if isinstance(row, dict) and row
    )

    summary_lines = [
        "| Role | Variant | Layers | Mean quality | Worst decile | Quality pass | Intelligibility | Hard cases | Critical | Major cases | Semantic pass | Train VRAM reduction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("All-layer reference", all_reference),
    ]
    role_names = {
        str(strongest.get("variant", "")): "Strongest selective candidate",
        "random4": "Parameter-count control",
        "bottom4": "Low-ranked control",
    }
    for row in selected_rows:
        role = role_names.get(str(row.get("variant", "")), str(row.get("variant", "candidate")))
        summary_lines.append(summary_row(role, row))

    sections: list[str] = []
    sections.append("## All-layer Reference — Lowest/Flagged Cases\n\n" + case_table(read_jsonl(args.all_results), args.max_cases_per_variant))
    for row in selected_rows:
        path = variant_results_path(row)
        rows = read_jsonl(path) if path is not None else []
        sections.append(
            f"## {escape(row.get('variant', 'candidate'))} — Lowest/Flagged Cases\n\n"
            f"- Layers: `{escape(row.get('layers'))}`\n"
            f"- Raw results: `{escape(path or 'not recorded')}`\n\n"
            + case_table(rows, args.max_cases_per_variant)
        )

    report = """# PHASE 2 HUMAN REVIEW PACKET

## Purpose

Review the strongest static selective-layer candidate against the all-layer
reference and parameter-count controls. This packet does not overturn the
automated gate. It helps determine whether score differences correspond to
meaningful changes in correctness, context use, naturalness, completeness, and
safety.

## Important Boundary

Phase 2A measures **where adapters can learn the behavior**. It does not prove
that untrained transformer blocks can be removed during inference. Whole-layer
removal and recovery are evaluated separately in Phase 2B.

## Aggregate Comparison

""" + "\n".join(summary_lines) + "\n\n" + "\n\n".join(sections) + """

## Human Review Checklist

- Confirm the strongest candidate has zero hard failures and no low-tail collapse; do not rely on the mean alone.
- Inspect every critical or major semantic flag.
- Compare factual compression, clarification, context references, uncertainty, and safety.
- Confirm RANDOM4 and BOTTOM4 behave worse when the ranking says they should.
- Keep the negative result when no selective candidate satisfies the formal gate.

## Decision

The authoritative decision remains in `reports/PHASE_2_STOP_GATE.md` and, when
needed, `reports/PHASE_2_DEVIATION_APPROVAL.md`.
"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8", newline="\n")
    print(f"Phase 2 human review packet: {args.output_report}")


if __name__ == "__main__":
    main()
