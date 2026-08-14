from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "cross_v212_confirmation_manifest.json"
# scripts/run_dev.py uses the suffix ``dev_full`` for every unbounded run,
# including a config whose selected dataset is named v212_test.
FULL_RUN = ROOT / "results" / "runs" / "phase2_cross_v212_full_k7_dev_full.jsonl"
CONTROL_RUN = ROOT / "results" / "runs" / "phase2_cross_v212_controls_dev_full.jsonl"
OUTPUT = ROOT / "results" / "phase2_mechanism" / "cross_v212_analysis.json"
SUMMARY_CSV = ROOT / "results" / "phase2_mechanism" / "cross_v212_summary.csv"
PAIR_CSV = ROOT / "results" / "phase2_mechanism" / "cross_v212_paired.csv"
REPORT = ROOT / "reports" / "PHASE2_CROSS_V212_RESULTS.md"

N = 1320
BOOTSTRAPS = 20_000
SEED = 20260812
CONDITIONS = {
    "backbone": ("backbone_k1", "hard_plurality"),
    "track_a": ("track_a_k1", "hard_plurality"),
    "abstention": ("abstention_k1", "hard_plurality"),
    "hard_k7": ("full_k7", "hard_plurality"),
    "probability_k7": ("full_k7", "probability_mean"),
    "calibrated_probability_k7": ("full_k7", "calibrated_probability_mean"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        batch = min(1000, BOOTSTRAPS - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        samples[start : start + batch] = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def holm(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return adjusted


def read_runs(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    records = {condition: {} for condition in CONDITIONS}
    expected_hashes = {
        FULL_RUN: manifest["configs"]["full_k7"]["canonical_sha256"],
        CONTROL_RUN: manifest["configs"]["controls"]["canonical_sha256"],
    }
    reverse = {value: key for key, value in CONDITIONS.items()}
    for path, expected_hash in expected_hashes.items():
        if not path.is_file():
            raise RuntimeError(f"Missing frozen cross-benchmark output: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row["config_sha256"] != expected_hash:
                    raise RuntimeError(f"Config hash mismatch in {path} line {line_number}")
                key = (row["committee"], row["method"])
                if key not in reverse:
                    raise RuntimeError(f"Unexpected condition {key} in {path}")
                condition = reverse[key]
                example_id = row["example_id"]
                if example_id in records[condition]:
                    raise RuntimeError(f"Duplicate {condition}/{example_id}")
                records[condition][example_id] = row
    counts = {condition: len(rows) for condition, rows in records.items()}
    if any(count != N for count in counts.values()):
        raise RuntimeError(f"Cross-benchmark run incomplete: {counts}; expected {N} each")
    id_sets = {frozenset(rows) for rows in records.values()}
    if len(id_sets) != 1:
        raise RuntimeError("Cross-benchmark condition ID sets differ")
    for example_id in records["hard_k7"]:
        golds = {records[condition][example_id]["gold"] for condition in records}
        prompts = {records[condition][example_id]["prompt_hash"] for condition in records}
        if len(golds) != 1 or len(prompts) != 1:
            raise RuntimeError(f"Cross-condition identity mismatch for {example_id}")
    return records


def summarize(condition: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    answerable = [row for row in values if row["decision"] == "answer"]
    abstention = [row for row in values if row["decision"] == "abstain"]
    return {
        "condition": condition,
        "committee": values[0]["committee"],
        "method": values[0]["method"],
        "n": len(values),
        "correct": sum(bool(row["correct"]) for row in values),
        "accuracy": float(np.mean([bool(row["correct"]) for row in values])),
        "answerable_n": len(answerable),
        "answerable_accuracy": float(np.mean([bool(row["correct"]) for row in answerable])),
        "abstention_n": len(abstention),
        "abstention_accuracy": float(np.mean([bool(row["correct"]) for row in abstention])),
        "mean_output_tokens": float(np.mean([row["generated_tokens"] for row in values])),
        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in values])),
        "mean_forward_passes": float(np.mean([row["forward_passes"] for row in values])),
        "mean_ttft_seconds": float(np.mean([row["time_to_first_final_token_seconds"] for row in values])),
        "peak_gpu_vram_bytes": max(row["peak_gpu_vram_bytes"] or 0 for row in values),
        "peak_cpu_rss_bytes": max(row["peak_cpu_rss_bytes"] or 0 for row in values),
    }


def compare(
    hard: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], baseline_name: str, rng: np.random.Generator
) -> dict[str, Any]:
    ids = sorted(hard)
    differences = np.array(
        [int(hard[example_id]["correct"]) - int(baseline[example_id]["correct"]) for example_id in ids],
        dtype=float,
    )
    hard_correct = np.array([bool(hard[example_id]["correct"]) for example_id in ids])
    baseline_correct = np.array([bool(baseline[example_id]["correct"]) for example_id in ids])
    rescue = int(np.sum(hard_correct & ~baseline_correct))
    harm = int(np.sum(~hard_correct & baseline_correct))
    low, high = bootstrap_ci(differences, rng)
    discordant = rescue + harm
    p_value = float(binomtest(min(rescue, harm), discordant, 0.5).pvalue) if discordant else 1.0
    strata = {}
    for decision in ("answer", "abstain"):
        selected = [example_id for example_id in ids if hard[example_id]["decision"] == decision]
        local = [int(hard[example_id]["correct"]) - int(baseline[example_id]["correct"]) for example_id in selected]
        strata[decision] = {
            "n": len(selected),
            "hard_correct": sum(bool(hard[example_id]["correct"]) for example_id in selected),
            "baseline_correct": sum(bool(baseline[example_id]["correct"]) for example_id in selected),
            "difference_hard_minus_baseline": float(np.mean(local)),
        }
    return {
        "comparison": f"hard_k7_vs_{baseline_name}",
        "baseline": baseline_name,
        "n": N,
        "hard_accuracy": float(hard_correct.mean()),
        "baseline_accuracy": float(baseline_correct.mean()),
        "difference_hard_minus_baseline": float(differences.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "both_correct": int(np.sum(hard_correct & baseline_correct)),
        "rescue": rescue,
        "harm": harm,
        "both_wrong": int(np.sum(~hard_correct & ~baseline_correct)),
        "mcnemar_exact_p": p_value,
        "strata": strata,
    }


def main() -> None:
    manifest = load_json(MANIFEST)
    if manifest["status"] != "V212_CROSS_CORPUS_CONFIRMATION_FROZEN":
        raise RuntimeError("Cross-benchmark manifest is not frozen")
    records = read_runs(manifest)
    summaries = [summarize(condition, records[condition]) for condition in CONDITIONS]
    rng = np.random.default_rng(SEED)
    baseline_order = ["backbone", "track_a", "abstention", "probability_k7", "calibrated_probability_k7"]
    comparisons = [compare(records["hard_k7"], records[baseline], baseline, rng) for baseline in baseline_order]
    adjusted = holm([row["mcnemar_exact_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_mcnemar_p"] = value

    analysis = {
        "schema_version": "1.0",
        "status": "V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": manifest["analysis_role"],
        "claim_boundary": manifest["claim_boundary"],
        "example_count": N,
        "provenance": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "full_run": str(FULL_RUN),
            "full_run_sha256": sha256(FULL_RUN),
            "control_run": str(CONTROL_RUN),
            "control_run_sha256": sha256(CONTROL_RUN),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "dataset_sha256": manifest["v212_test_sha256"],
        },
        "condition_summaries": summaries,
        "paired_family": {
            "definition": manifest["statistics"],
            "comparisons": comparisons,
        },
        "caveats": manifest["caveats"],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    flat_pairs = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in comparisons]
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_pairs[0]))
        writer.writeheader()
        writer.writerows(flat_pairs)

    summary_lines = [
        f'| {row["condition"]} | {row["correct"]}/{row["n"]} | {100*row["accuracy"]:.2f}% | '
        f'{100*row["answerable_accuracy"]:.2f}% | {100*row["abstention_accuracy"]:.2f}% | '
        f'{row["mean_latency_seconds"]:.3f} | {row["mean_forward_passes"]:.2f} |'
        for row in summaries
    ]
    pair_lines = [
        f'| Hard − {row["baseline"]} | {100*row["difference_hard_minus_baseline"]:+.2f} pp | '
        f'[{100*row["bootstrap_ci95_low"]:+.2f}, {100*row["bootstrap_ci95_high"]:+.2f}] | '
        f'{row["rescue"]}/{row["harm"]} | {row["mcnemar_exact_p"]:.3e} | {row["holm_adjusted_mcnemar_p"]:.3e} |'
        for row in comparisons
    ]
    report = "\n".join(
        [
            "# Phase 2 v2.1.2 Fixed-Method Cross-Corpus Results",
            "",
            f'Role: **{manifest["analysis_role"]}**. {manifest["claim_boundary"]}',
            "",
            "| Condition | Correct | Exact acc. | Answerable | Abstention | Latency (s) | Forwards |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *summary_lines,
            "",
            "| Comparison | Effect | Paired bootstrap 95% CI | Rescue/harm | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
            *pair_lines,
            "",
            "## Mandatory caveats",
            "",
            *[f"- {caveat}" for caveat in manifest["caveats"]],
            "",
            "No v2.2 and v2.1.2 accuracy is pooled.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
