from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, atomic_write_text, sha256_file


class ComparisonError(RuntimeError):
    pass


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row["case_id"]
            if case_id in result:
                raise ComparisonError(f"duplicate case ID at line {line_number}: {case_id}")
            result[case_id] = row
    return result


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("empty quantile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def scenario_bootstrap(
    rows: list[tuple[str, float]], *, seed: int = 17, replicates: int = 10_000
) -> dict[str, Any]:
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for scenario_id, delta in rows:
        by_scenario[scenario_id].append(delta)
    scenario_deltas = {
        scenario: sum(values) / len(values)
        for scenario, values in by_scenario.items()
    }
    keys = sorted(scenario_deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        draw = [scenario_deltas[rng.choice(keys)] for _ in keys]
        samples.append(sum(draw) / len(draw))
    observed = sum(scenario_deltas.values()) / len(scenario_deltas)
    return {
        "scenario_count": len(keys),
        "replicates": replicates,
        "seed": seed,
        "observed_macro_delta": observed,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
        "probability_delta_gt_zero": sum(value > 0 for value in samples) / replicates,
    }


def mcnemar_exact_two_sided(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = min(a_only, b_only)
    cumulative = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * cumulative / (2**discordant))


def compare_prediction_files(
    path_a: Path,
    path_b: Path,
    *,
    label_a: str,
    label_b: str,
    seed: int = 17,
    replicates: int = 10_000,
) -> dict[str, Any]:
    a = read_predictions(path_a)
    b = read_predictions(path_b)
    if set(a) != set(b):
        raise ComparisonError(
            f"case sets differ: a_only={len(set(a)-set(b))}, b_only={len(set(b)-set(a))}"
        )
    rows = []
    family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    a_only_cases = []
    b_only_cases = []
    both_correct = []
    both_wrong = []
    response_changes = []
    for case_id in sorted(a):
        row_a, row_b = a[case_id], b[case_id]
        for field in ("scenario_id", "family", "prompt_sha256"):
            if row_a[field] != row_b[field]:
                raise ComparisonError(f"paired metadata differs for {case_id}: {field}")
        score_a = float(row_a["score"]["score"])
        score_b = float(row_b["score"]["score"])
        delta = score_b - score_a
        record = {
            "case_id": case_id,
            "scenario_id": row_a["scenario_id"],
            "family": row_a["family"],
            "a": score_a,
            "b": score_b,
            "delta": delta,
        }
        rows.append(record)
        family[row_a["family"]].append(record)
        if score_a > 0 and score_b == 0:
            a_only_cases.append(case_id)
        elif score_b > 0 and score_a == 0:
            b_only_cases.append(case_id)
        elif score_a > 0 and score_b > 0:
            both_correct.append(case_id)
        else:
            both_wrong.append(case_id)
        if row_a["response_sha256"] != row_b["response_sha256"]:
            response_changes.append(case_id)

    overall_a = sum(row["a"] for row in rows) / len(rows)
    overall_b = sum(row["b"] for row in rows) / len(rows)
    family_report = {
        name: {
            "items": len(values),
            "a_score": sum(row["a"] for row in values) / len(values),
            "b_score": sum(row["b"] for row in values) / len(values),
            "delta": sum(row["delta"] for row in values) / len(values),
            "a_only": sum(row["a"] > 0 and row["b"] == 0 for row in values),
            "b_only": sum(row["b"] > 0 and row["a"] == 0 for row in values),
        }
        for name, values in sorted(family.items())
    }
    return {
        "schema_version": "1.0",
        "label_a": label_a,
        "label_b": label_b,
        "predictions_a": {"path": str(path_a.resolve()), "sha256": sha256_file(path_a)},
        "predictions_b": {"path": str(path_b.resolve()), "sha256": sha256_file(path_b)},
        "items": len(rows),
        "overall": {
            "a_score": overall_a,
            "b_score": overall_b,
            "delta_b_minus_a": overall_b - overall_a,
        },
        "families": family_report,
        "paired": {
            "a_only": len(a_only_cases),
            "b_only": len(b_only_cases),
            "both_correct": len(both_correct),
            "both_wrong": len(both_wrong),
            "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(
                len(a_only_cases), len(b_only_cases)
            ),
        },
        "scenario_bootstrap": scenario_bootstrap(
            [(row["scenario_id"], row["delta"]) for row in rows],
            seed=seed,
            replicates=replicates,
        ),
        "response_changes": len(response_changes),
        "sets": {
            "a_only": a_only_cases,
            "b_only": b_only_cases,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "response_changes": response_changes,
        },
    }


def render_comparison(report: dict[str, Any]) -> str:
    label_a, label_b = report["label_a"], report["label_b"]
    overall = report["overall"]
    paired = report["paired"]
    bootstrap = report["scenario_bootstrap"]
    lines = [
        f"# {label_b} vs {label_a}",
        "",
        f"- {label_a}: {overall['a_score']:.3f}",
        f"- {label_b}: {overall['b_score']:.3f}",
        f"- Delta ({label_b} - {label_a}): {overall['delta_b_minus_a']:+.3f}",
        f"- Paired discordance: {label_a}-only {paired['a_only']}, {label_b}-only {paired['b_only']}",
        f"- Exact McNemar p: {paired['mcnemar_exact_two_sided_p']:.4f}",
        f"- Scenario bootstrap 95% CI: [{bootstrap['ci95'][0]:+.3f}, {bootstrap['ci95'][1]:+.3f}]",
        f"- Responses changed: {report['response_changes']} / {report['items']}",
        "",
        "## Families",
        "",
        f"| Family | Items | {label_a} | {label_b} | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, details in report["families"].items():
        lines.append(
            f"| {family} | {details['items']} | {details['a_score']:.3f} | "
            f"{details['b_score']:.3f} | {details['delta']:+.3f} |"
        )
    return "\n".join(lines)


def write_comparison(
    report: dict[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, render_comparison(report))
