from __future__ import annotations

import collections
import html
import json
from pathlib import Path
from typing import Any

from ..utils import atomic_write_json, atomic_write_text, html_page, iter_jsonl, markdown_table, stable_int_hash, utc_now_iso
from .stats import exact_mcnemar, paired_bootstrap_difference, proportion_ci_wilson, sign_flip_randomization


def _select_rows(path: str | Path, protocol: str | None) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path, tolerate_truncated_tail=True))
    protocols = sorted({str(row.get("protocol")) for row in rows})
    if protocol is None:
        if len(protocols) != 1:
            raise ValueError(f"{path} contains protocols {protocols}; specify a protocol explicitly")
        protocol = protocols[0]
    selected = [row for row in rows if str(row.get("protocol")) == protocol]
    if not selected:
        raise ValueError(f"No rows for protocol {protocol!r} in {path}")
    return selected


def _metric_value(row: dict[str, Any], metric: str) -> float:
    current: Any = row
    for part in metric.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric {metric!r} is absent from {row.get('request_id')}")
        current = current[part]
    if isinstance(current, bool):
        return float(current)
    return float(current)


def _unique_by_problem(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row["problem_id"])
        if pid in result:
            raise ValueError(
                f"Protocol contains multiple rows for problem {pid}; compare an end-to-end protocol or pre-aggregate transitions"
            )
        result[pid] = row
    return result


def _group_effect(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_values = [_metric_value(left, metric) for left, _ in pairs]
    candidate_values = [_metric_value(right, metric) for _, right in pairs]
    effect = paired_bootstrap_difference(
        baseline_values,
        candidate_values,
        confidence=0.95,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "n": len(pairs),
        "baseline_mean": sum(baseline_values) / len(baseline_values),
        "candidate_mean": sum(candidate_values) / len(candidate_values),
        "difference": effect,
    }


def compare_runs(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
    *,
    baseline_protocol: str | None = None,
    candidate_protocol: str | None = None,
    metric: str = "score.correct_strict",
    noninferiority_margin: float = 0.02,
    bootstrap_samples: int = 10000,
    randomization_samples: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    baseline_rows = _select_rows(baseline_path, baseline_protocol)
    candidate_rows = _select_rows(candidate_path, candidate_protocol)
    baseline = _unique_by_problem(baseline_rows)
    candidate = _unique_by_problem(candidate_rows)
    common = sorted(set(baseline) & set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    missing_candidate = sorted(set(baseline) - set(candidate))
    if not common:
        raise ValueError("Baseline and candidate have no shared problem IDs")
    pairs = [(baseline[pid], candidate[pid]) for pid in common]
    baseline_values = [_metric_value(left, metric) for left, _ in pairs]
    candidate_values = [_metric_value(right, metric) for _, right in pairs]
    ci95 = paired_bootstrap_difference(
        baseline_values,
        candidate_values,
        confidence=0.95,
        samples=bootstrap_samples,
        seed=seed,
    )
    ci90 = paired_bootstrap_difference(
        baseline_values,
        candidate_values,
        confidence=0.90,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    baseline_binary = [bool(value) for value in baseline_values]
    candidate_binary = [bool(value) for value in candidate_values]
    mcnemar = exact_mcnemar(baseline_binary, candidate_binary)
    randomization = sign_flip_randomization(
        baseline_values,
        candidate_values,
        samples=randomization_samples,
        seed=seed,
    )
    margin = float(noninferiority_margin)
    superior = ci95["lower"] > 0.0
    noninferior = ci95["lower"] > -margin
    equivalent = ci90["lower"] > -margin and ci90["upper"] < margin
    inferior = ci95["upper"] < -margin
    if superior:
        verdict = "SUPERIOR"
    elif equivalent:
        verdict = "EQUIVALENT"
    elif noninferior:
        verdict = "NONINFERIOR"
    elif inferior:
        verdict = "INFERIOR"
    else:
        verdict = "INCONCLUSIVE"

    groups: dict[str, Any] = {}
    for field in ["category", "domain", "difficulty"]:
        grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
        for pair in pairs:
            grouped[str(pair[0].get(field))].append(pair)
        groups[field] = {
            key: _group_effect(
                values,
                metric,
                bootstrap_samples=min(bootstrap_samples, 3000),
                seed=seed + stable_int_hash(f"{field}:{key}") % 100000,
            )
            for key, values in sorted(grouped.items())
        }

    baseline_successes = sum(baseline_binary)
    candidate_successes = sum(candidate_binary)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "baseline_path": str(Path(baseline_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "baseline_protocol": baseline_protocol or baseline_rows[0]["protocol"],
        "candidate_protocol": candidate_protocol or candidate_rows[0]["protocol"],
        "metric": metric,
        "paired_problems": len(common),
        "missing_from_baseline": missing_baseline,
        "missing_from_candidate": missing_candidate,
        "baseline": {
            "mean": sum(baseline_values) / len(common),
            "confidence_interval": proportion_ci_wilson(baseline_successes, len(common)),
        },
        "candidate": {
            "mean": sum(candidate_values) / len(common),
            "confidence_interval": proportion_ci_wilson(candidate_successes, len(common)),
        },
        "paired_difference_candidate_minus_baseline": {"ci95": ci95, "ci90": ci90},
        "mcnemar": mcnemar,
        "randomization": randomization,
        "decision": {
            "noninferiority_margin": margin,
            "superior": superior,
            "noninferior": noninferior,
            "equivalent": equivalent,
            "inferior": inferior,
            "verdict": verdict,
            "interpretation": (
                "Candidate exceeds baseline under the paired 95% interval."
                if verdict == "SUPERIOR"
                else "Candidate is statistically equivalent within the preregistered margin."
                if verdict == "EQUIVALENT"
                else "Candidate remains above the preregistered non-inferiority boundary."
                if verdict == "NONINFERIOR"
                else "Candidate falls below the preregistered non-inferiority boundary."
                if verdict == "INFERIOR"
                else "The interval crosses the relevant decision boundaries."
            ),
        },
        "groups": groups,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "comparison.json", report)

    discordant = []
    for left, right in pairs:
        left_value = bool(_metric_value(left, metric))
        right_value = bool(_metric_value(right, metric))
        if left_value != right_value:
            discordant.append(
                {
                    "problem_id": left["problem_id"],
                    "category": left.get("category"),
                    "baseline_correct": left_value,
                    "candidate_correct": right_value,
                    "baseline_output": left.get("raw_output"),
                    "candidate_output": right.get("raw_output"),
                }
            )
    discordant.sort(key=lambda row: stable_int_hash(str(row["problem_id"])))
    atomic_write_json(output / "discordant_audit_sample.json", discordant[:200])

    md = [
        "# Paired Comparison Report",
        "",
        f"**Verdict: {verdict}**",
        "",
        report["decision"]["interpretation"],
        "",
        markdown_table(
            ["Quantity", "Value"],
            [
                ["Paired problems", len(common)],
                ["Baseline mean", f"{100 * report['baseline']['mean']:.3f}%"],
                ["Candidate mean", f"{100 * report['candidate']['mean']:.3f}%"],
                ["Difference", f"{100 * ci95['estimate']:+.3f} pp"],
                ["95% paired bootstrap CI", f"[{100 * ci95['lower']:+.3f}, {100 * ci95['upper']:+.3f}] pp"],
                ["90% paired bootstrap CI", f"[{100 * ci90['lower']:+.3f}, {100 * ci90['upper']:+.3f}] pp"],
                ["Non-inferiority margin", f"{100 * margin:.3f} pp"],
                ["Exact McNemar p", f"{mcnemar['p_value_two_sided']:.6g}"],
                ["Sign-flip p", f"{randomization['p_value_two_sided']:.6g}"],
            ],
        ),
        "",
        "## Discordant outcomes",
        "",
        markdown_table(
            ["Direction", "Count"],
            [
                ["Baseline only correct", mcnemar["baseline_only_correct"]],
                ["Candidate only correct", mcnemar["candidate_only_correct"]],
            ],
        ),
        "",
        "## Category effects",
        "",
        markdown_table(
            ["Category", "N", "Baseline", "Candidate", "Difference (95% CI)"],
            [
                [
                    key,
                    value["n"],
                    f"{100 * value['baseline_mean']:.2f}%",
                    f"{100 * value['candidate_mean']:.2f}%",
                    f"{100 * value['difference']['estimate']:+.2f} [{100 * value['difference']['lower']:+.2f}, {100 * value['difference']['upper']:+.2f}] pp",
                ]
                for key, value in report["groups"]["category"].items()
            ],
        ),
    ]
    atomic_write_text(output / "comparison.md", "\n".join(md) + "\n")
    body = "<h1>Paired Comparison Report</h1>" + f"<h2>{html.escape(verdict)}</h2>" + f"<p>{html.escape(report['decision']['interpretation'])}</p>"
    body += "<table><tr><th>Metric</th><th>Value</th></tr>"
    body += f"<tr><td>Paired problems</td><td>{len(common)}</td></tr>"
    body += f"<tr><td>Baseline</td><td>{100*report['baseline']['mean']:.3f}%</td></tr>"
    body += f"<tr><td>Candidate</td><td>{100*report['candidate']['mean']:.3f}%</td></tr>"
    body += f"<tr><td>Difference (95% CI)</td><td>{100*ci95['estimate']:+.3f} [{100*ci95['lower']:+.3f}, {100*ci95['upper']:+.3f}] pp</td></tr></table>"
    atomic_write_text(output / "comparison.html", html_page("Rineya Paired Comparison", body))
    return report
