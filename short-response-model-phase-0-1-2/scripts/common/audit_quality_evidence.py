from __future__ import annotations

"""Read-only audit for stale or incomplete Evaluation v3 evidence."""

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION

REQUIRED_FIELDS = {
    "quality_evaluation_version",
    "case_count",
    "mean_quality_score",
    "worst_decile_mean_quality_score",
    "mean_quality_dimensions",
    "category_quality_scores",
    "mean_issue_cost",
    "severe_issue_case_rate",
    "semantic_collapse_case_rate",
    "semantic_unresponsive_case_rate",
    "mean_semantic_coherence_score",
    "mean_prompt_relevance_score",
    "mean_intent_fulfillment_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit evaluation JSON evidence for quality schema v3.1.")
    parser.add_argument(
        "roots", nargs="*", type=Path,
        default=[Path("artifacts"), Path("playground_artifacts")],
        help="Directories or JSON files to scan.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/QUALITY_EVIDENCE_AUDIT.md"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def looks_like_evaluation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    indicators = {
        "mean_heuristic_score", "mean_quality_score", "quality_evaluation_version",
        "semantic_pass_rate", "complete_utterance_rate", "median_reply_tokens",
    }
    return len(indicators.intersection(payload)) >= 2


def iter_json_files(root: Path):
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
    elif root.is_dir():
        yield from sorted(root.rglob("*.json"))


def classify(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "status": "INVALID_JSON", "details": str(exc)}
    if not looks_like_evaluation(payload):
        return None
    version = str(payload.get("quality_evaluation_version", "legacy/absent"))
    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if version != QUALITY_EVALUATION_VERSION:
        status = "STALE_POLICY"
        details = f"quality_evaluation_version={version}"
    elif missing:
        status = "INCOMPLETE_CURRENT"
        details = "missing: " + ", ".join(missing)
    else:
        status = "VALID_CURRENT"
        details = (
            f"mean={float(payload.get('mean_quality_score', 0.0)):.3f}; "
            f"issue_cost={float(payload.get('mean_issue_cost', 0.0)):.3f}; "
            f"severe_rate={float(payload.get('severe_issue_case_rate', 0.0)):.3f}"
        )
    return {"path": str(path), "status": status, "details": details}


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in args.roots:
        for path in iter_json_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            row = classify(path)
            if row is not None:
                rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    table = ["| Status | File | Details |", "|---|---|---|"]
    for row in sorted(rows, key=lambda item: (item["status"], item["path"])):
        table.append(f"| {row['status']} | `{row['path']}` | {row['details'].replace('|', '/')} |")
    if not rows:
        table.append("| NONE | - | No evaluation-like JSON summaries were found. |")

    report = "\n".join([
        "# QUALITY EVIDENCE AUDIT", "",
        "This is a read-only inventory. Summaries from earlier policies cannot be mixed with the current evaluator in automated comparisons.",
        "", "## Counts", "",
        *[f"- {key}: `{value}`" for key, value in sorted(counts.items())],
        "" if counts else "- No evaluation-like JSON summaries found.",
        "", "## Files", "", *table, "", "## Interpretation", "",
        "- `VALID_CURRENT`: current continuous quality evidence with semantic integrity fields.",
        "- `STALE_POLICY`: rescore preserved generation results or rerun evaluation.",
        "- `INCOMPLETE_CURRENT`: producer did not write all opportunity-comparison fields.",
        "- `INVALID_JSON`: repair the evidence before any phase consumes it.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(args.report)
    problematic = sum(counts.get(key, 0) for key in ("STALE_POLICY", "INCOMPLETE_CURRENT", "INVALID_JSON"))
    if args.strict and problematic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
