from __future__ import annotations

"""Read-only audit for stale or incomplete evaluation evidence.

This tool never edits checkpoints, summaries, reports, or gates. It scans JSON
files and identifies summaries that were produced by the legacy evaluator or
that are missing the independent quality-v2 fields required by later phases.
"""

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "quality_evaluation_version",
    "mean_quality_score",
    "quality_pass_rate",
    "intelligibility_pass_rate",
    "hard_failure_case_count",
    "worst_decile_mean_quality_score",
    "semantic_pass_rate",
    "complete_utterance_rate",
    "category_quality_scores",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit evaluation JSON evidence for quality schema v2.0.")
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("artifacts"), Path("playground_artifacts")],
        help="Directories or JSON files to scan.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/QUALITY_EVIDENCE_AUDIT.md"))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when stale or invalid evidence is found.")
    return parser.parse_args()


def looks_like_evaluation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    indicators = {
        "mean_heuristic_score",
        "mean_quality_score",
        "quality_evaluation_version",
        "semantic_pass_rate",
        "complete_utterance_rate",
        "median_reply_tokens",
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
    except Exception as exc:  # audit must report malformed evidence instead of crashing
        return {"path": str(path), "status": "INVALID_JSON", "details": str(exc)}
    if not looks_like_evaluation(payload):
        return None
    version = str(payload.get("quality_evaluation_version", "legacy/absent"))
    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if version != "2.0":
        status = "STALE_LEGACY"
        details = f"quality_evaluation_version={version}"
    elif missing:
        status = "INCOMPLETE_V2"
        details = "missing: " + ", ".join(missing)
    else:
        status = "VALID_V2"
        details = (
            f"mean={float(payload.get('mean_quality_score', 0.0)):.3f}; "
            f"pass={float(payload.get('quality_pass_rate', 0.0)):.3f}; "
            f"hard={int(payload.get('hard_failure_case_count', 0))}"
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

    report = "\n".join(
        [
            "# QUALITY EVIDENCE AUDIT",
            "",
            "This is a read-only inventory. Legacy summaries are not safe for automated acceptance under quality evaluator v2.0.",
            "",
            "## Counts",
            "",
            *[f"- {key}: `{value}`" for key, value in sorted(counts.items())],
            "" if counts else "- No evaluation-like JSON summaries found.",
            "",
            "## Files",
            "",
            *table,
            "",
            "## Interpretation",
            "",
            "- `VALID_V2`: may be consumed by v2 gates.",
            "- `STALE_LEGACY`: re-run generation evaluation; do not copy the old aggregate score forward.",
            "- `INCOMPLETE_V2`: the producer did not write all independent quality fields and must be re-run or fixed.",
            "- `INVALID_JSON`: repair the evidence file before any gate uses it.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(args.report)
    problematic = sum(counts.get(key, 0) for key in ("STALE_LEGACY", "INCOMPLETE_V2", "INVALID_JSON"))
    if args.strict and problematic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
