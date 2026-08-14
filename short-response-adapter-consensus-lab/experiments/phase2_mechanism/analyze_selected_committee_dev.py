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
RUN = ROOT / "results/runs/phase2_selected_committee_dev_free_run_dev_full.jsonl"
FULL_RUN = ROOT / "results/runs/phase2_selected_dev_free_run_dev_full.jsonl"
MANIFEST = ROOT / "artifacts/phase2_mechanism/selected_committee_dev_manifest.json"
OUTPUT = ROOT / "results/phase2_mechanism/selected_committee_dev_analysis.json"
SUMMARY_CSV = ROOT / "results/phase2_mechanism/selected_committee_dev_summary.csv"
LOO_CSV = ROOT / "results/phase2_mechanism/selected_committee_dev_leave_one_out.csv"
STRATA_CSV = ROOT / "results/phase2_mechanism/selected_committee_dev_leave_one_out_strata.csv"
REPORT = ROOT / "reports/PHASE2_SELECTED_COMMITTEE_DEV.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_PANEL_ROWS = 48600
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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    estimates = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        estimates.append(values[indices].mean(axis=1))
        remaining -= batch
    low, high = np.quantile(np.concatenate(estimates), [0.025, 0.975])
    return float(low), float(high)


def holm(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(values) - rank) * value))
        adjusted[index] = running
    return adjusted


def load_records(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    expected_committees = set(manifest["committees"])
    expected_hash = manifest["config"]["canonical_sha256"]
    records = {committee: {} for committee in expected_committees}
    rows = 0
    with RUN.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed panel row {line_number}: {exc}") from exc
            rows += 1
            committee = row["committee"]
            if committee not in records or row["method"] != "hard_plurality":
                raise RuntimeError(f"Unexpected panel condition at line {line_number}")
            if row["config_sha256"] != expected_hash:
                raise RuntimeError(f"Panel config mismatch at line {line_number}")
            example_id = row["example_id"]
            if example_id in records[committee]:
                raise RuntimeError(f"Duplicate panel key {committee}/{example_id}")
            records[committee][example_id] = row
    if rows != EXPECTED_PANEL_ROWS or any(len(group) != EXPECTED_EXAMPLES for group in records.values()):
        raise RuntimeError(
            f"Panel incomplete: rows={rows}, counts={ {key: len(value) for key, value in records.items()} }"
        )

    full: dict[str, dict[str, Any]] = {}
    with FULL_RUN.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["committee"] == "full_k7" and row["method"] == "hard_plurality":
                if row["example_id"] in full:
                    raise RuntimeError(f"Duplicate full-K7 key {row['example_id']}")
                full[row["example_id"]] = row
    if len(full) != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Full-K7 hard reference incomplete: {len(full)}")
    records["full_k7"] = full
    id_sets = [set(group) for group in records.values()]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise RuntimeError("Committee example-ID sets differ")
    for example_id in id_sets[0]:
        group = [condition[example_id] for condition in records.values()]
        if len({row["gold"] for row in group}) != 1:
            raise RuntimeError(f"Gold mismatch for {example_id}")
        if len({row["decision"] for row in group}) != 1:
            raise RuntimeError(f"Decision mismatch for {example_id}")
        if len({row["prompt_hash"] for row in group}) != 1:
            raise RuntimeError(f"Prompt mismatch for {example_id}")
    return records


def summarize(
    committee: str, rows: dict[str, dict[str, Any]], role: dict[str, Any]
) -> dict[str, Any]:
    ordered = [rows[example_id] for example_id in sorted(rows)]
    answer = [row for row in ordered if row["decision"] == "answer"]
    abstain = [row for row in ordered if row["decision"] == "abstain"]
    return {
        "committee": committee,
        "k": role["k"],
        "mask": role["mask"],
        "members": role["members"],
        "teacher_forced_best_at_k": role["teacher_forced_best_at_k"],
        "removed_expert": role["removed_expert"],
        "n": len(ordered),
        "correct": sum(bool(row["correct"]) for row in ordered),
        "accuracy": float(np.mean([bool(row["correct"]) for row in ordered])),
        "answerable_n": len(answer),
        "answerable_correct": sum(bool(row["correct"]) for row in answer),
        "answerable_accuracy": float(np.mean([bool(row["correct"]) for row in answer])),
        "abstention_n": len(abstain),
        "abstention_correct": sum(bool(row["correct"]) for row in abstain),
        "abstention_accuracy": float(np.mean([bool(row["correct"]) for row in abstain])),
        "balanced_answer_class_accuracy": 0.5
        * (
            float(np.mean([bool(row["correct"]) for row in answer]))
            + float(np.mean([bool(row["correct"]) for row in abstain]))
        ),
        "mean_output_tokens": float(np.mean([row["generated_tokens"] for row in ordered])),
        "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in ordered])),
        "mean_ttft_seconds": float(
            np.mean([row["time_to_first_final_token_seconds"] for row in ordered])
        ),
        "mean_forward_passes": float(np.mean([row["forward_passes"] for row in ordered])),
        "mean_adapter_evaluations": float(
            np.mean([row["adapter_evaluations"] for row in ordered])
        ),
        "accuracy_per_mean_forward_pass": float(
            np.mean([bool(row["correct"]) for row in ordered])
            / np.mean([row["forward_passes"] for row in ordered])
        ),
        "mean_ties_per_example": float(np.mean([row["tie_count"] for row in ordered])),
        "mean_vote_entropy": float(np.mean([row["mean_vote_entropy"] for row in ordered])),
        "peak_gpu_vram_bytes": max(row["peak_gpu_vram_bytes"] or 0 for row in ordered),
        "peak_cpu_rss_bytes": max(row["peak_cpu_rss_bytes"] or 0 for row in ordered),
        "analysis_role": "DEV MODEL-SELECTION / TARGETED FREE-RUNNING COMMITTEE PANEL",
    }


def compare_full_to_loo(
    full: dict[str, dict[str, Any]],
    loo: dict[str, dict[str, Any]],
    removed_expert: str,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = sorted(full)
    full_correct = np.asarray([bool(full[value]["correct"]) for value in ids])
    loo_correct = np.asarray([bool(loo[value]["correct"]) for value in ids])
    difference = full_correct.astype(float) - loo_correct.astype(float)
    full_only = int(np.sum(full_correct & ~loo_correct))
    loo_only = int(np.sum(~full_correct & loo_correct))
    discordant = full_only + loo_only
    low, high = bootstrap_ci(difference, rng)
    p_value = (
        float(binomtest(min(full_only, loo_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    strata_rows = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for example_id in ids:
        groups[("answer_class", str(full[example_id]["decision"]))].append(example_id)
        groups[("reasoning_family", str(full[example_id]["reasoning_family"]))].append(example_id)
    for (dimension, value), selected in sorted(groups.items()):
        local = np.asarray(
            [int(full[example_id]["correct"]) - int(loo[example_id]["correct"]) for example_id in selected],
            dtype=float,
        )
        strata_rows.append(
            {
                "removed_expert": removed_expert,
                "dimension": dimension,
                "value": value,
                "n": len(selected),
                "full_correct": sum(bool(full[example_id]["correct"]) for example_id in selected),
                "loo_correct": sum(bool(loo[example_id]["correct"]) for example_id in selected),
                "difference_full_minus_loo": float(local.mean()),
                "analysis_role": "descriptive DEV stratum; no multiplicity-adjusted inference",
            }
        )
    return (
        {
            "removed_expert": removed_expert,
            "n": len(ids),
            "full_accuracy": float(full_correct.mean()),
            "loo_accuracy": float(loo_correct.mean()),
            "difference_full_minus_loo": float(difference.mean()),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "both_correct": int(np.sum(full_correct & loo_correct)),
            "full_only_correct": full_only,
            "loo_only_correct": loo_only,
            "both_wrong": int(np.sum(~full_correct & ~loo_correct)),
            "mcnemar_exact_p": p_value,
        },
        strata_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [
        {key: ";".join(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main() -> None:
    manifest = load_json(MANIFEST)
    if manifest.get("status") != "DEV_SELECTED_COMMITTEE_PANEL_FROZEN":
        raise RuntimeError("Selected committee panel is not frozen")
    if manifest.get("selection_used_test_outcomes") is not False:
        raise RuntimeError("Selected committee panel is not DEV-only")
    records = load_records(manifest)
    roles = dict(manifest["committees"])
    roles["full_k7"] = {
        "mask": 127,
        "members": ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"],
        "k": 7,
        "teacher_forced_best_at_k": 7,
        "removed_expert": None,
    }
    summaries = [summarize(name, records[name], roles[name]) for name in sorted(records)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    strata = []
    for committee, role in manifest["committees"].items():
        if role["removed_expert"] is None:
            continue
        comparison, local = compare_full_to_loo(
            records["full_k7"], records[committee], role["removed_expert"], rng
        )
        comparison["committee"] = committee
        comparisons.append(comparison)
        strata.extend(local)
    for row, adjusted in zip(
        comparisons, holm([row["mcnemar_exact_p"] for row in comparisons]), strict=True
    ):
        row["holm_adjusted_mcnemar_p"] = adjusted

    summary_by_name = {row["committee"]: row for row in summaries}
    scaling = []
    for k in range(1, 8):
        candidate = next(
            row for row in summaries if int(row["teacher_forced_best_at_k"] or -1) == k
        )
        scaling.append(candidate)

    analysis = {
        "schema_version": "1.0",
        "status": "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": manifest["analysis_role"],
        "example_count": EXPECTED_EXAMPLES,
        "panel_row_count": EXPECTED_PANEL_ROWS,
        "condition_count_including_reused_full_k7": len(summaries),
        "condition_summaries": summaries,
        "selected_best_at_k_scaling": scaling,
        "leave_one_out_family": {
            "definition": "full K7 hard versus every K6 leave-one-out committee; exact McNemar with seven-test Holm correction",
            "comparisons": comparisons,
            "descriptive_strata": strata,
        },
        "provenance": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "panel_run": str(RUN),
            "panel_run_sha256": sha256(RUN),
            "full_k7_run": str(FULL_RUN),
            "full_k7_run_sha256": sha256(FULL_RUN),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "The best-at-K committees were selected on teacher-forced DEV and are evaluated on the same DEV examples; scaling is model-selection evidence.",
            "Same-split partial full-K7 DEV outputs already existed when this panel was frozen, although committee identities follow only the frozen teacher-forced rule and exhaustive leave-one-out structure.",
            "Category and answer-class leave-one-out rows are descriptive and are not additional confirmatory families.",
            "Latency, TTFT, VRAM, RSS, and accuracy-per-forward are implementation- and hardware-specific.",
        ],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(SUMMARY_CSV, summaries)
    write_csv(LOO_CSV, comparisons)
    write_csv(STRATA_CSV, strata)

    scaling_lines = [
        f'| {row["k"]} | {row["committee"]} | {row["correct"]}/{row["n"]} | '
        f'{100*row["accuracy"]:.3f}% | {100*row["balanced_answer_class_accuracy"]:.3f}% | '
        f'{row["mean_latency_seconds"]:.3f} | '
        f'{row["mean_forward_passes"]:.2f} | {1e3*row["accuracy_per_mean_forward_pass"]:.4f} |'
        for row in scaling
    ]
    loo_lines = [
        f'| {row["removed_expert"]} | {100*row["difference_full_minus_loo"]:+.3f} pp | '
        f'[{100*row["bootstrap_ci95_low"]:+.3f}, {100*row["bootstrap_ci95_high"]:+.3f}] | '
        f'{row["full_only_correct"]}/{row["loo_only_correct"]} | '
        f'{row["mcnemar_exact_p"]:.3e} | {row["holm_adjusted_mcnemar_p"]:.3e} |'
        for row in comparisons
    ]
    REPORT.write_text(
        "\n".join(
            [
                "# Selected free-running DEV committee panel",
                "",
                "Status: **DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE**",
                "",
                "This targeted panel measures the teacher-forced best committee at each K and every K=6 leave-one-out under free-running hard plurality. It is same-split DEV model-selection evidence, not TEST confirmation.",
                "",
                "## Teacher-forced-selected best-at-K trajectory and systems curve",
                "",
                "| K | Committee | Correct | Exact acc. | Balanced answer-class | Latency (s) | Forwards | 1000×accuracy/forward |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
                *scaling_lines,
                "",
                "## Seven-comparison leave-one-out family",
                "",
                "| Removed expert | Full minus K6 | Paired bootstrap 95% CI | Full-only/K6-only | McNemar p | Holm p |",
                "|---|---:|---:|---:|---:|---:|",
                *loo_lines,
                "",
                "Answer-class and reasoning-family deltas are descriptive and stored in the machine artifact/strata CSV. Systems measurements are reference-decoder and hardware specific.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
