from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, atomic_write_text, sha256_file


class PromotionError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    severe_family_drop: float = 0.10,
    maximum_invalid_rate: float = 0.02,
    maximum_completion_drop: float = 0.05,
    maximum_format_drop: float = 0.05,
    require_overall_improvement: bool = True,
) -> dict[str, Any]:
    if candidate.get("dataset_sha256") != baseline.get("dataset_sha256"):
        raise PromotionError("candidate and baseline dataset hashes differ")
    if candidate.get("items") != baseline.get("items"):
        raise PromotionError("candidate and baseline item counts differ")
    reasons = []
    family_deltas = {}
    for family, base_details in baseline["families"].items():
        if family not in candidate["families"]:
            reasons.append(f"missing_family:{family}")
            continue
        delta = float(candidate["families"][family]["score"]) - float(
            base_details["score"]
        )
        family_deltas[family] = delta
        if delta < -severe_family_drop:
            reasons.append(f"severe_family_drop:{family}:{delta:.6f}")
    invalid_rate = float(candidate.get("invalid_generation_rate", 0.0))
    if invalid_rate > maximum_invalid_rate:
        reasons.append(f"invalid_generation_rate:{invalid_rate:.6f}")
    completion_delta = float(candidate.get("completion_rate", 0.0)) - float(
        baseline.get("completion_rate", 0.0)
    )
    if completion_delta < -maximum_completion_drop:
        reasons.append(f"completion_drop:{completion_delta:.6f}")
    candidate_format = candidate.get("format_adherence", {}).get("overall")
    baseline_format = baseline.get("format_adherence", {}).get("overall")
    format_delta = None
    if candidate_format is not None and baseline_format is not None:
        format_delta = float(candidate_format) - float(baseline_format)
        if format_delta < -maximum_format_drop:
            reasons.append(f"format_drop:{format_delta:.6f}")
    overall_delta = float(candidate["overall_score"]) - float(
        baseline["overall_score"]
    )
    if require_overall_improvement and overall_delta <= 0:
        reasons.append(f"no_overall_improvement:{overall_delta:.6f}")
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "overall_delta": overall_delta,
        "scenario_macro_delta": float(candidate["scenario_macro_score"])
        - float(baseline["scenario_macro_score"]),
        "worst_family_score": float(candidate["worst_family_score"]),
        "format_delta": format_delta,
        "completion_delta": completion_delta,
        "invalid_generation_rate": invalid_rate,
        "family_deltas": family_deltas,
    }


def promote(
    *,
    baseline_path: Path,
    candidate_paths: dict[str, Path],
    output_json: Path,
    output_markdown: Path,
    top_n: int,
    method_complexity: dict[str, int] | None = None,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    complexity = method_complexity or {}
    rows = []
    for name, path in sorted(candidate_paths.items()):
        candidate = _load(path)
        decision = evaluate_candidate(baseline, candidate)
        rows.append(
            {
                "name": name,
                "summary_path": str(path.resolve()),
                "summary_sha256": sha256_file(path),
                "complexity": int(complexity.get(name, 100)),
                **decision,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["worst_family_score"],
            -row["scenario_macro_delta"],
            -row["overall_delta"],
            row["complexity"],
            row["name"],
        )
    )
    promoted = [row["name"] for row in eligible[: max(0, top_n)]]
    report = {
        "schema_version": "1.0",
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        },
        "top_n": top_n,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "promoted": promoted,
        "rows": rows,
    }
    atomic_write_json(output_json, report)
    lines = [
        "# Candidate Promotion",
        "",
        f"Promoted: {', '.join(promoted) if promoted else 'none'}",
        "",
        "| Candidate | Eligible | Overall Δ | Scenario Δ | Worst family | Reasons |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['eligible']} | {row['overall_delta']:+.4f} | "
            f"{row['scenario_macro_delta']:+.4f} | {row['worst_family_score']:.4f} | "
            f"{'; '.join(row['rejection_reasons'])} |"
        )
    atomic_write_text(output_markdown, "\n".join(lines))
    return report
