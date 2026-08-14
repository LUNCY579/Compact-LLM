from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from consensus_lab.io import write_json_atomic  # noqa: E402


SOURCES = {
    "selected_dev": (
        ROOT / "results/phase2_mechanism/selected_dev_free_run_analysis.json",
        "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE",
    ),
    "adaptive_dev": (
        ROOT / "results/phase2_mechanism/adaptive_free_run_dev_analysis.json",
        "DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE",
    ),
    "selected_committee_dev": (
        ROOT / "results/phase2_mechanism/selected_committee_dev_analysis.json",
        "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE",
    ),
    "v22_calibrated_test": (
        ROOT / "results/phase2_mechanism/v22_calibrated_test_analysis.json",
        "V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE",
    ),
    "cross_v212": (
        ROOT / "results/phase2_mechanism/cross_v212_analysis.json",
        "V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE",
    ),
}

OUTPUT = ROOT / "results/phase2_mechanism/final_synthesis_facts.json"
REPORT = ROOT / "reports/PHASE2_FINAL_SYNTHESIS_FACTS.md"

QUESTION_EVIDENCE_MAP = {
    "6": ["selected_committee_dev.leave_one_out_family.comparisons"],
    "7": ["selected_committee_dev.leave_one_out_family.comparisons"],
    "10": ["selected_committee_dev.selected_best_at_k_scaling"],
    "11": ["selected_dev.teacher_forced_vs_free_running", "selected_dev.paired_family"],
    "12": ["selected_dev.teacher_forced_vs_free_running"],
    "13": ["selected_dev.paired_family", "v22_calibrated_test.paired_family"],
    "14": ["selected_dev.method_summaries", "selected_dev.paired_family"],
    "15": ["selected_dev.paired_family", "v22_calibrated_test.paired_family"],
    "16": ["cross_v212.condition_summaries", "cross_v212.paired_family"],
    "17": ["adaptive_dev.paired_accuracy", "adaptive_dev.efficiency"],
    "18": [
        "selected_committee_dev.selected_best_at_k_scaling",
        "adaptive_dev.summaries",
        "adaptive_dev.efficiency",
    ],
    "19": [
        "selected_dev.paired_family",
        "selected_committee_dev.selected_best_at_k_scaling",
        "v22_calibrated_test.paired_family",
        "cross_v212.paired_family",
    ],
    "20": [
        "selected_dev.teacher_forced_vs_free_running",
        "selected_committee_dev.leave_one_out_family",
        "v22_calibrated_test.paired_family",
        "cross_v212.paired_family",
    ],
}

HYPOTHESIS_EVIDENCE_MAP = {
    "H1_calibration_robust_consensus": [
        "selected_dev.paired_family",
        "v22_calibrated_test.paired_family",
    ],
    "H4_ordinal_bottleneck_rank_sufficiency": [
        "selected_dev.method_summaries",
        "selected_dev.paired_family",
    ],
    "H6_committee_size_threshold": [
        "selected_committee_dev.selected_best_at_k_scaling"
    ],
    "H7_trajectory_correction": ["selected_dev.teacher_forced_vs_free_running"],
    "H8_abstention_eos_composition": [
        "selected_committee_dev.leave_one_out_family.comparisons"
    ],
    "H10_specialization_matters": [
        "selected_committee_dev.leave_one_out_family.comparisons"
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.3f} pp"


def sci(value: float) -> str:
    return f"{value:.3e}"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    body = ["| " + " | ".join(headers) + " |"]
    body.append("|" + "|".join("---" for _ in headers) + "|")
    body.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(body)


def selected_section(packet: dict[str, Any]) -> list[str]:
    summaries = packet["method_summaries"]
    comparisons = packet["paired_family"]["comparisons"]
    teacher_free = packet["teacher_forced_vs_free_running"]
    runtime = packet.get("runtime_boundaries", {})
    oom = packet.get("oom_retry_telemetry", {})
    lines = [
        "## Same-split selected DEV methods",
        "",
        md_table(
            ["Method", "Correct/N", "Free exact", "Answerable", "Abstention", "Latency (s)", "Forwards"],
            (
                (
                    row["method"],
                    f'{row["correct"]}/{row["n"]}',
                    pct(row["accuracy"]),
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    f'{row["mean_latency_seconds"]:.4f}',
                    f'{row["mean_forward_passes"]:.3f}',
                )
                for row in summaries
            ),
        ),
        "",
        md_table(
            ["Comparator minus hard", "Difference", "Paired 95% CI", "Other-only/Hard-only", "Holm p"],
            (
                (
                    row["other_method"],
                    pp(row["difference_other_minus_hard"]),
                    f'{pp(row["bootstrap_ci95_low"])} to {pp(row["bootstrap_ci95_high"])}',
                    f'{row["other_only_correct"]}/{row["hard_only_correct"]}',
                    sci(row["holm_adjusted_mcnemar_p"]),
                )
                for row in comparisons
            ),
        ),
        "",
        md_table(
            ["Method", "Local token", "Local all-token example", "Free exact", "Free gap vs hard"],
            (
                (
                    row["method"],
                    pct(row["teacher_forced_token_accuracy"]),
                    pct(row["teacher_forced_all_token_example_accuracy"]),
                    pct(row["free_running_exact_accuracy"]),
                    pp(row["free_sequence_advantage_vs_hard"]),
                )
                for row in teacher_free
            ),
        ),
    ]
    if runtime or oom:
        lines.extend(
            [
                "",
                "### Runtime provenance",
                "",
                f'- Stable partial top-k first raw line: **{runtime.get("stable_partial_topk_first_line", "n/a")}**; '
                f'OOM retry telemetry first raw line: **{runtime.get("oom_retry_telemetry_first_line", "n/a")}**.',
                f'- Rows carrying retry telemetry: **{oom.get("rows_with_field", 0):,}**; '
                f'examples with retry: **{oom.get("examples_with_one_or_more_retries", 0):,}** '
                f'(**{oom.get("total_retries", 0):,}** total retries).',
            ]
        )
    return lines


def adaptive_section(packet: dict[str, Any]) -> list[str]:
    paired = packet["paired_accuracy"]
    efficiency = packet["efficiency"]
    return [
        "## Adaptive consultation",
        "",
        md_table(
            ["Condition", "Correct/N", "Exact", "Forwards", "Forwards/token", "Trigger/token", "Latency (s)"],
            (
                (
                    row["condition"],
                    f'{row["correct"]}/{row["n"]}',
                    pct(row["accuracy"]),
                    f'{row["mean_forward_passes"]:.3f}',
                    f'{row["mean_forwards_per_generated_token"]:.3f}',
                    pct(row["adaptive_trigger_token_rate"]),
                    f'{row["mean_latency_seconds"]:.4f}',
                )
                for row in packet["summaries"]
            ),
        ),
        "",
        f'Adaptive minus hard: **{pp(paired["difference_adaptive_minus_hard"])}** '
        f'(paired 95% CI {pp(paired["paired_bootstrap_ci95_low"])} to '
        f'{pp(paired["paired_bootstrap_ci95_high"])}; exact McNemar '
        f'p={sci(paired["mcnemar_exact_p"])}).',
        "",
        f'Expert-forward reduction versus hard: **{pct(efficiency["forward_pass_reduction_vs_hard"])}**; '
        f'latency ratio adaptive/hard: **{efficiency["latency_ratio_adaptive_over_hard"]:.4f}**.',
    ]


def committee_section(packet: dict[str, Any]) -> list[str]:
    family = packet["leave_one_out_family"]
    answer_class = {
        (row["removed_expert"], row["value"]): row["difference_full_minus_loo"]
        for row in family.get("descriptive_strata", [])
        if row["dimension"] == "answer_class"
    }
    return [
        "## Selected committee scaling and leave-one-out",
        "",
        md_table(
            ["K", "Committee", "Exact", "Balanced", "Latency (s)", "Forwards", "1000x acc./forward"],
            (
                (
                    row["k"],
                    row["committee"],
                    pct(row["accuracy"]),
                    pct(row["balanced_answer_class_accuracy"]),
                    f'{row["mean_latency_seconds"]:.4f}',
                    f'{row["mean_forward_passes"]:.3f}',
                    f'{1000 * row["accuracy_per_mean_forward_pass"]:.4f}',
                )
                for row in packet["selected_best_at_k_scaling"]
            ),
        ),
        "",
        md_table(
            [
                "Removed expert",
                "Full minus K6",
                "Answerable delta",
                "Abstention delta",
                "Paired 95% CI",
                "Full-only/K6-only",
                "Holm p",
            ],
            (
                (
                    row["removed_expert"],
                    pp(row["difference_full_minus_loo"]),
                    (
                        pp(answer_class[(row["removed_expert"], "answer")])
                        if (row["removed_expert"], "answer") in answer_class
                        else "not captured"
                    ),
                    (
                        pp(answer_class[(row["removed_expert"], "abstain")])
                        if (row["removed_expert"], "abstain") in answer_class
                        else "not captured"
                    ),
                    f'{pp(row["bootstrap_ci95_low"])} to {pp(row["bootstrap_ci95_high"])}',
                    f'{row["full_only_correct"]}/{row["loo_only_correct"]}',
                    sci(row["holm_adjusted_mcnemar_p"]),
                )
                for row in family["comparisons"]
            ),
        ),
    ]


def calibrated_section(packet: dict[str, Any]) -> list[str]:
    return [
        "## Selected calibrated probability on v2.2 TEST",
        "",
        md_table(
            ["Condition", "Correct/N", "Exact", "Answerable", "Abstention", "Latency (s)"],
            (
                (
                    row["condition"],
                    f'{row["correct"]}/{row["n"]}',
                    pct(row["accuracy"]),
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    f'{row["mean_latency_seconds"]:.4f}',
                )
                for row in packet["condition_summaries"]
            ),
        ),
        "",
        md_table(
            ["Comparison", "Difference", "Paired 95% CI", "Calibrated-only/Baseline-only", "Holm p"],
            (
                (
                    row["comparison"],
                    pp(row["difference_calibrated_minus_baseline"]),
                    f'{pp(row["bootstrap_ci95_low"])} to {pp(row["bootstrap_ci95_high"])}',
                    f'{row["calibrated_only_correct"]}/{row["baseline_only_correct"]}',
                    sci(row["holm_adjusted_mcnemar_p"]),
                )
                for row in packet["paired_family"]["comparisons"]
            ),
        ),
    ]


def cross_section(packet: dict[str, Any]) -> list[str]:
    return [
        "## Fixed-method v2.1.2 transfer",
        "",
        md_table(
            ["Condition", "Correct/N", "Exact", "Answerable", "Abstention", "Latency (s)"],
            (
                (
                    row["condition"],
                    f'{row["correct"]}/{row["n"]}',
                    pct(row["accuracy"]),
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    f'{row["mean_latency_seconds"]:.4f}',
                )
                for row in packet["condition_summaries"]
            ),
        ),
        "",
        md_table(
            ["Hard versus baseline", "Difference", "Paired 95% CI", "Rescue/Harm", "Holm p"],
            (
                (
                    row["baseline"],
                    pp(row["difference_hard_minus_baseline"]),
                    f'{pp(row["bootstrap_ci95_low"])} to {pp(row["bootstrap_ci95_high"])}',
                    f'{row["rescue"]}/{row["harm"]}',
                    sci(row["holm_adjusted_mcnemar_p"]),
                )
                for row in packet["paired_family"]["comparisons"]
            ),
        ),
        "",
        "Claim boundary: fixed-method cross-corpus transfer only; the v2.1.2 corpus is not a globally untouched holdout, and no v2.2/v2.1.2 accuracy is pooled.",
    ]


def main() -> None:
    packets: dict[str, dict[str, Any]] = {}
    for name, (path, expected_status) in SOURCES.items():
        if not path.is_file():
            raise RuntimeError(f"Missing final synthesis source: {path}")
        packet = load_json(path)
        if packet.get("status") != expected_status:
            raise RuntimeError(
                f"Incomplete final synthesis source {path}: "
                f"expected {expected_status!r}, found {packet.get('status')!r}"
            )
        packets[name] = packet

    facts = {
        "schema_version": "1.0",
        "status": "FINAL_SYNTHESIS_FACTS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_dev": {
            "method_summaries": packets["selected_dev"]["method_summaries"],
            "paired_family": packets["selected_dev"]["paired_family"],
            "teacher_forced_vs_free_running": packets["selected_dev"][
                "teacher_forced_vs_free_running"
            ],
            "runtime_boundaries": packets["selected_dev"]["runtime_boundaries"],
            "oom_retry_telemetry": packets["selected_dev"]["oom_retry_telemetry"],
        },
        "adaptive_dev": {
            "policy": packets["adaptive_dev"]["policy"],
            "summaries": packets["adaptive_dev"]["summaries"],
            "paired_accuracy": packets["adaptive_dev"]["paired_accuracy"],
            "efficiency": packets["adaptive_dev"]["efficiency"],
        },
        "selected_committee_dev": {
            "selected_best_at_k_scaling": packets["selected_committee_dev"][
                "selected_best_at_k_scaling"
            ],
            "leave_one_out_family": packets["selected_committee_dev"][
                "leave_one_out_family"
            ],
        },
        "v22_calibrated_test": {
            "condition_summaries": packets["v22_calibrated_test"]["condition_summaries"],
            "paired_family": packets["v22_calibrated_test"]["paired_family"],
            "caveats": packets["v22_calibrated_test"]["caveats"],
        },
        "cross_v212": {
            "condition_summaries": packets["cross_v212"]["condition_summaries"],
            "paired_family": packets["cross_v212"]["paired_family"],
            "claim_boundary": packets["cross_v212"]["claim_boundary"],
            "caveats": packets["cross_v212"]["caveats"],
        },
        "question_evidence_map": QUESTION_EVIDENCE_MAP,
        "hypothesis_evidence_map": HYPOTHESIS_EVIDENCE_MAP,
        "provenance": {
            "sources": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path, _ in SOURCES.values()
            },
            "generator": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "generator_sha256": sha256(Path(__file__).resolve()),
        },
    }
    report = "\n".join(
        [
            "# Phase 2 Final Synthesis Fact Packet",
            "",
            "All numbers below are generated from the five strict long-run analysis JSON files. This packet is a transcription-safe source for the final narrative, not an independent statistical family.",
            "",
            *selected_section(facts["selected_dev"]),
            "",
            *adaptive_section(facts["adaptive_dev"]),
            "",
            *committee_section(facts["selected_committee_dev"]),
            "",
            *calibrated_section(facts["v22_calibrated_test"]),
            "",
            *cross_section(facts["cross_v212"]),
            "",
            "## Required-question and hypothesis routing",
            "",
            "The machine-readable packet maps affected questions 6–7, 10–20 and hypotheses H1/H4/H6/H7/H8/H10 to exact nested fields. Questions not listed are adjudicated by the already-complete cornerstone and static mechanism analyses.",
            "",
            "Exact nested statistics, subgroup records, caveats, and hashes remain in `results/phase2_mechanism/final_synthesis_facts.json` and its five source analyses.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    facts["provenance"]["report"] = str(REPORT.relative_to(ROOT)).replace("\\", "/")
    facts["provenance"]["report_sha256"] = sha256(REPORT)
    write_json_atomic(OUTPUT, facts)
    print(
        json.dumps(
            {
                "status": facts["status"],
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
