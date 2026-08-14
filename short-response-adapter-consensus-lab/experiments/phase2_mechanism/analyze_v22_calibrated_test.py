from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
CORNERSTONE = ROOT / "results/sealed/v22_frozen_finalists.jsonl"
RUN = ROOT / "results/runs/phase2_v22_calibrated_test_confirmation_dev_full.jsonl"
MANIFEST = ROOT / "artifacts/phase2_mechanism/v22_calibrated_test_confirmation_manifest.json"
OUTPUT = ROOT / "results/phase2_mechanism/v22_calibrated_test_analysis.json"
SUMMARY_CSV = ROOT / "results/phase2_mechanism/v22_calibrated_test_summary.csv"
PAIR_CSV = ROOT / "results/phase2_mechanism/v22_calibrated_test_paired.csv"
REPORT = ROOT / "reports/PHASE2_V22_CALIBRATED_TEST_CONFIRMATION.md"
EXPECTED = 4050
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {
        "hard_plurality": {},
        "probability_mean": {},
        "calibrated_probability_mean": {},
    }
    cornerstone_hash = manifest["cornerstone"]["sha256"]
    if sha256(CORNERSTONE) != cornerstone_hash:
        raise RuntimeError("Cornerstone hash differs from the frozen confirmation manifest")
    with CORNERSTONE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["committee"] == "full_k7" and row["method"] in {
                "hard_plurality",
                "probability_mean",
            }:
                method = row["method"]
                example_id = row["example_id"]
                if example_id in records[method]:
                    raise RuntimeError(f"Duplicate cornerstone key {method}/{example_id}")
                records[method][example_id] = row
    expected_config = manifest["config"]["canonical_sha256"]
    with RUN.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed calibrated run line {line_number}: {exc}") from exc
            if row["committee"] != "full_k7" or row["method"] != "calibrated_probability_mean":
                raise RuntimeError(f"Unexpected calibrated condition at line {line_number}")
            if row["config_sha256"] != expected_config:
                raise RuntimeError(f"Calibrated config mismatch at line {line_number}")
            example_id = row["example_id"]
            if example_id in records["calibrated_probability_mean"]:
                raise RuntimeError(f"Duplicate calibrated key {example_id}")
            records["calibrated_probability_mean"][example_id] = row
    counts = {method: len(rows) for method, rows in records.items()}
    if counts != {method: EXPECTED for method in records}:
        raise RuntimeError(f"Confirmation is incomplete: {counts}")
    id_sets = [set(rows) for rows in records.values()]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise RuntimeError("Condition example-ID sets differ")
    for example_id in id_sets[0]:
        rows = [records[method][example_id] for method in records]
        if len({row["gold"] for row in rows}) != 1:
            raise RuntimeError(f"Gold mismatch for {example_id}")
        if len({row["decision"] for row in rows}) != 1:
            raise RuntimeError(f"Decision-class mismatch for {example_id}")
        if len({row["prompt_hash"] for row in rows}) != 1:
            raise RuntimeError(f"Prompt mismatch for {example_id}")
    return records


def summarize(method: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[example_id] for example_id in sorted(rows)]
    answer = [row for row in ordered if row["decision"] == "answer"]
    abstain = [row for row in ordered if row["decision"] == "abstain"]
    return {
        "condition": method,
        "n": len(ordered),
        "correct": sum(bool(row["correct"]) for row in ordered),
        "accuracy": sum(bool(row["correct"]) for row in ordered) / len(ordered),
        "answerable_n": len(answer),
        "answerable_correct": sum(bool(row["correct"]) for row in answer),
        "answerable_accuracy": sum(bool(row["correct"]) for row in answer) / len(answer),
        "abstention_n": len(abstain),
        "abstention_correct": sum(bool(row["correct"]) for row in abstain),
        "abstention_accuracy": sum(bool(row["correct"]) for row in abstain) / len(abstain),
        "mean_output_tokens": float(np.mean([row["generated_tokens"] for row in ordered])),
        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in ordered])),
        "mean_forward_passes": float(np.mean([row["forward_passes"] for row in ordered])),
    }


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    estimates: list[np.ndarray] = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(differences), size=(batch, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
        remaining -= batch
    low, high = np.quantile(np.concatenate(estimates), [0.025, 0.975])
    return float(low), float(high)


def compare(
    calibrated: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    baseline_name: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = sorted(calibrated)
    calibrated_correct = np.asarray([bool(calibrated[value]["correct"]) for value in ids])
    baseline_correct = np.asarray([bool(baseline[value]["correct"]) for value in ids])
    differences = calibrated_correct.astype(float) - baseline_correct.astype(float)
    calibrated_only = int(np.sum(calibrated_correct & ~baseline_correct))
    baseline_only = int(np.sum(~calibrated_correct & baseline_correct))
    discordant = calibrated_only + baseline_only
    low, high = bootstrap_ci(differences, rng)
    p_value = (
        float(binomtest(min(calibrated_only, baseline_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    strata: dict[str, Any] = {}
    for decision in ("answer", "abstain"):
        local = [value for value in ids if calibrated[value]["decision"] == decision]
        local_diff = np.asarray(
            [int(calibrated[value]["correct"]) - int(baseline[value]["correct"]) for value in local],
            dtype=float,
        )
        strata[decision] = {
            "n": len(local),
            "calibrated_correct": sum(bool(calibrated[value]["correct"]) for value in local),
            "baseline_correct": sum(bool(baseline[value]["correct"]) for value in local),
            "difference_calibrated_minus_baseline": float(local_diff.mean()),
        }
    return {
        "comparison": f"calibrated_probability_mean_vs_{baseline_name}",
        "baseline": baseline_name,
        "n": len(ids),
        "both_correct": int(np.sum(calibrated_correct & baseline_correct)),
        "calibrated_only_correct": calibrated_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": int(np.sum(~calibrated_correct & ~baseline_correct)),
        "difference_calibrated_minus_baseline": float(differences.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "mcnemar_exact_p": p_value,
        "strata": strata,
    }


def holm(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    manifest = load_json(MANIFEST)
    if manifest.get("status") != "V22_CALIBRATED_TEST_CONFIRMATION_FROZEN":
        raise RuntimeError("v2.2 calibrated confirmation manifest is not frozen")
    if manifest.get("calibrated_confirmation_outcomes_read_before_freeze") is not False:
        raise RuntimeError("Manifest does not preserve the calibrated-outcome freeze boundary")
    records = load_records(manifest)
    summaries = [summarize(method, records[method]) for method in records]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [
        compare(records["calibrated_probability_mean"], records[baseline], baseline, rng)
        for baseline in ("hard_plurality", "probability_mean")
    ]
    for row, adjusted in zip(
        comparisons, holm([row["mcnemar_exact_p"] for row in comparisons]), strict=True
    ):
        row["holm_adjusted_mcnemar_p"] = adjusted
    analysis = {
        "schema_version": "1.0",
        "status": "V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": manifest["analysis_role"],
        "example_count": EXPECTED,
        "condition_summaries": summaries,
        "paired_family": {
            "definition": "calibrated K7 versus frozen hard K7 and raw probability K7; exact McNemar with two-test Holm correction",
            "comparisons": comparisons,
        },
        "provenance": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "cornerstone": str(CORNERSTONE),
            "cornerstone_sha256": sha256(CORNERSTONE),
            "calibrated_run": str(RUN),
            "calibrated_run_sha256": sha256(RUN),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "caveats": manifest["caveats"],
        "limitations": [
            "The hard/raw full-TEST outcomes were known before this one-condition confirmation was frozen.",
            "An archived n=32 pilot previously exposed examples from this TEST split.",
            "The calibrated method and temperatures were selected on DEV only and were not refit after TEST output.",
            "Subgroup effects are descriptive; the confirmatory family contains only the two global paired comparisons.",
        ],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(SUMMARY_CSV, summaries)
    write_csv(PAIR_CSV, comparisons)

    summary_lines = [
        f'| {row["condition"]} | {row["correct"]}/{row["n"]} | {100*row["accuracy"]:.3f}% | '
        f'{100*row["answerable_accuracy"]:.3f}% | {100*row["abstention_accuracy"]:.3f}% | '
        f'{row["mean_latency_seconds"]:.3f} | {row["mean_forward_passes"]:.2f} |'
        for row in summaries
    ]
    pair_lines = [
        f'| calibrated − {row["baseline"]} | {100*row["difference_calibrated_minus_baseline"]:+.3f} pp | '
        f'[{100*row["bootstrap_ci95_low"]:+.3f}, {100*row["bootstrap_ci95_high"]:+.3f}] | '
        f'{row["calibrated_only_correct"]}/{row["baseline_only_correct"]} | '
        f'{row["mcnemar_exact_p"]:.3e} | {row["holm_adjusted_mcnemar_p"]:.3e} |'
        for row in comparisons
    ]
    report = "\n".join(
        [
            "# v2.2 calibrated-probability TEST confirmation",
            "",
            "Status: **V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE**",
            "",
            "The calibrated K=7 method and all temperatures were fixed from DEV before this output existed. Hard/raw anchors come from the verified cornerstone; their outcomes were already known, and the archived n=32 exposure caveat remains.",
            "",
            "## Full TEST conditions",
            "",
            "| Condition | Correct | Exact accuracy | Answerable | Abstention | Latency (s) | Forwards |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "## Predeclared paired family",
            "",
            "| Comparison | Difference | Paired bootstrap 95% CI | Calibrated-only/Baseline-only | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
            *pair_lines,
            "",
            "The two global comparisons constitute this confirmation family. Answerable/abstention cuts are descriptive, and v2.2 results are not aggregated with v2.1.2.",
            "",
            f"Cornerstone SHA-256: `{analysis['provenance']['cornerstone_sha256']}`.",
            f"Calibrated run SHA-256: `{analysis['provenance']['calibrated_run_sha256']}`.",
        ]
    )
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
