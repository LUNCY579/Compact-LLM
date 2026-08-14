from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "paper" / "tables"

CORNERSTONE_AUDIT = ROOT / "reports" / "phase2_cornerstone_verification.json"
CROSSBENCH_GATE = ROOT / "results" / "phase2_mechanism" / "cross_benchmark_verification.json"
TEST_SUMMARY = ROOT / "results" / "summaries" / "v22_frozen_finalists_sealed.json"
TEST_STATS = ROOT / "results" / "summaries" / "sealed_statistics.json"
MECHANISM = ROOT / "results" / "phase2_mechanism" / "existing_results_analysis.json"
TEACHER = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_analysis.json"
ROBUST_SOFT = ROOT / "results" / "phase2_mechanism" / "robust_soft_controls_dev.json"
CONFIDENCE_CONDITIONING = (
    ROOT / "results" / "phase2_mechanism" / "confidence_conditioned_correctness_dev.json"
)
COMMITTEE_DIAGNOSTICS = (
    ROOT / "results" / "phase2_mechanism" / "teacher_forced_committee_diagnostics.json"
)
COALITION_STRATA = ROOT / "results" / "phase2_mechanism" / "coalition_strata_full_test.json"
DIVERSITY_EXTENDED = ROOT / "results" / "phase2_mechanism" / "pairwise_diversity_extended.json"
HARD_TIE_TEMP = ROOT / "results" / "phase2_mechanism" / "hard_temperature_tie_sensitivity_dev.json"
SOURCE_MANIFEST = ROOT / "artifacts" / "manifests" / "official_sources_before.json"
EXPERT_DIVERSITY = ROOT / "results" / "phase2_mechanism" / "teacher_forced_expert_diversity.csv"
ADAPTIVE = ROOT / "results" / "phase2_mechanism" / "adaptive_teacher_forced_dev.json"
FREE_RUN = ROOT / "results" / "phase2_mechanism" / "selected_dev_free_run_analysis.json"
ADAPTIVE_FREE_RUN = ROOT / "results" / "phase2_mechanism" / "adaptive_free_run_dev_analysis.json"
CROSS_RESULTS = ROOT / "results" / "phase2_mechanism" / "cross_v212_analysis.json"
V22_CALIBRATED_RESULTS = ROOT / "results" / "phase2_mechanism" / "v22_calibrated_test_analysis.json"
SELECTED_COMMITTEE_RESULTS = (
    ROOT / "results" / "phase2_mechanism" / "selected_committee_dev_analysis.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_complete(path: Path, expected_status: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    packet = load_json(path)
    if packet.get("status") != expected_status:
        raise RuntimeError(
            f"Refusing incomplete optional table source {path}: "
            f"expected status {expected_status!r}, found {packet.get('status')!r}"
        )
    return packet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def pct(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def sci(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.3e}"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    body = ["| " + " | ".join(clean(value) for value in headers) + " |"]
    body.append("|" + "|".join("---" for _ in headers) + "|")
    body.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(body)


def dataset_table(cornerstone: dict[str, Any], cross: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v22_roles = {
        "train": ("adapter training", "training corpus", "adapter weights trained on this split"),
        "dev": ("temperature and method selection", "DEV model-selection/calibration", "teacher-forced analysis and selected free-running confirmation"),
        "test": ("no post-full-result selection", "TEST confirmatory", "prior n=32 feasibility exposure"),
    }
    for split in ("train", "dev", "test"):
        info = cornerstone["dataset_splits"][split]
        selection, role, prior = v22_roles[split]
        rows.append(
            {
                "dataset": "multidomain_reasoning_v22",
                "version": "2.2",
                "split": split,
                "n": info["rows"],
                "sha256": info["sha256"],
                "path": info["path"],
                "training_exposure": "yes" if split == "train" else "no direct adapter training",
                "selection_exposure": selection,
                "evaluation_role": role,
                "task_families": 15,
                "prior_access_or_overlap": prior,
            }
        )
    v212_roles = {
        "train": ("none for v2.2 adapters", "cross-corpus provenance only", "1,008/1,320 TEST rows repeat exact prompt+gold from this TRAIN split"),
        "dev": ("none for frozen transfer", "unused by fixed transfer", "not used for method selection"),
        "test": ("fixed methods only", "cross-corpus transfer pending", "historical D_greedy evaluation; template repetition caveat"),
    }
    for split in ("train", "dev", "test"):
        info = cross["splits"][split]
        selection, role, prior = v212_roles[split]
        rows.append(
            {
                "dataset": "general_reasoning_v212",
                "version": "2.1.2",
                "split": split,
                "n": info["rows"],
                "sha256": info["sha256"],
                "path": info["path"],
                "training_exposure": "no v2.2 adapter training",
                "selection_exposure": selection,
                "evaluation_role": role,
                "task_families": 33,
                "prior_access_or_overlap": prior,
            }
        )
    return rows


def inventory_table(
    cornerstone: dict[str, Any], source_manifest: dict[str, Any], summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    manifest_entries = {entry["relative_path"].replace("\\", "/"): entry for entry in source_manifest["entries"]}
    anchor_prefix = "artifacts/phase_9_capability_composition_v22_lr1e4_s1200/frozen_anchor"
    dense_prefix = "artifacts/phase_9_capability_composition_v22_lr1e4_s1200/track_d_greedy_soup/final_model"
    anchor_weights = manifest_entries[f"{anchor_prefix}/model.safetensors"]["sha256"]
    tokenizer_hash = manifest_entries[f"{anchor_prefix}/tokenizer.json"]["sha256"]
    dense_weights = manifest_entries[f"{dense_prefix}/model.safetensors"]["sha256"]
    dense_tokenizer_hash = manifest_entries[f"{dense_prefix}/tokenizer.json"]["sha256"]
    hard_summary = next(row for row in summaries if row["committee"] == "full_k7" and row["method"] == "hard_plurality")
    adapter_parameters = hard_summary["committee_adapter_parameter_count"] // 7
    rows = [
        {
            "model_or_adapter": "frozen_anchor",
            "role": "shared frozen backbone",
            "parameters": hard_summary["backbone_parameter_count"],
            "adapter_type": "none",
            "rank": "",
            "alpha": "",
            "target_modules": "",
            "tokenizer_sha256": tokenizer_hash,
            "declared_base": "self",
            "config_sha256": manifest_entries[f"{anchor_prefix}/config.json"]["sha256"],
            "weights_sha256": anchor_weights,
            "training_lineage": "phase_9 capability composition frozen anchor",
        }
    ]
    for expert, packet in cornerstone["frozen_config"]["adapter_packets"].items():
        rows.append(
            {
                "model_or_adapter": expert,
                "role": "v2.2 voting expert",
                "parameters": adapter_parameters,
                "adapter_type": packet["peft_type"],
                "rank": packet["rank"],
                "alpha": packet["alpha"],
                "target_modules": ", ".join(packet["target_modules"]),
                "tokenizer_sha256": tokenizer_hash,
                "declared_base": packet["declared_base"],
                "config_sha256": packet["adapter_config_sha256"],
                "weights_sha256": packet["adapter_weights_sha256"],
                "training_lineage": "multidomain_reasoning_v22 TRAIN",
            }
        )
    rows.append(
        {
            "model_or_adapter": "track_d_greedy_soup/final_model",
            "role": "historical dense comparator; unmatched lineage",
            "parameters": hard_summary["backbone_parameter_count"],
            "adapter_type": "merged dense",
            "rank": "",
            "alpha": "",
            "target_modules": "",
            "tokenizer_sha256": dense_tokenizer_hash,
            "declared_base": "unmatched to direct-vote adapter packet",
            "config_sha256": manifest_entries[f"{dense_prefix}/config.json"]["sha256"],
            "weights_sha256": dense_weights,
            "training_lineage": "historical official v2.1.2 evaluation; excluded from matched committee claims",
        }
    )
    return rows


def primary_test_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [
        ("backbone_k1", "hard_plurality", "Backbone"),
        ("abstention_k1", "hard_plurality", "Abstention specialist"),
        ("track_a_k1", "hard_plurality", "Track A"),
        ("full_k7", "probability_mean", "Probability K=7"),
        ("full_k7", "hard_plurality", "Hard plurality K=7"),
    ]
    rows = []
    for committee, method, label in wanted:
        source = next(row for row in summaries if row["committee"] == committee and row["method"] == method)
        correct = round(source["correctness"] * source["case_count"])
        low, high = wilson_interval(correct, source["case_count"])
        rows.append(
            {
                "method": label,
                "committee": committee,
                "n": source["case_count"],
                "correct": correct,
                "exact_accuracy": source["correctness"],
                "ci95_low_wilson": low,
                "ci95_high_wilson": high,
                "answerable_accuracy": source["answerable_correctness"],
                "abstention_accuracy": source["abstention_correctness"],
                "false_abstention_rate": source["false_abstention_rate"],
                "mean_output_tokens": source["mean_output_tokens"],
                "mean_latency_seconds": source["mean_latency_seconds"],
                "mean_expert_forwards": source["mean_forward_passes"],
                "adapter_storage_bytes": source["adapter_storage_bytes"],
            }
        )
    return rows


def rescue_table(mechanism: dict[str, Any], stats: dict[str, Any]) -> list[dict[str, Any]]:
    holm = {(row["condition_a"], row["condition_b"]): row for row in stats["paired_comparisons"]}
    mappings = {
        "hard_vs_track_a": ("hard_k7", "track_a"),
        "hard_vs_probability": ("hard_k7", "probability_k7"),
    }
    rows = []
    for source in mechanism["rescue_harm"]:
        if source["dimension"] != "global":
            continue
        pair = mappings[source["comparison"]]
        paired = holm[pair]
        rows.append(
            {
                "comparison": source["comparison_label"],
                "candidate": source["candidate"],
                "baseline": source["baseline"],
                "n": source["n"],
                "both_correct": source["both_correct"],
                "rescue": source["rescue"],
                "harm": source["harm"],
                "both_wrong": source["both_wrong"],
                "net_gain": source["net_gain"],
                "ci95_low": source["net_gain_ci_low"],
                "ci95_high": source["net_gain_ci_high"],
                "mcnemar_exact_p": source["mcnemar_exact_p_unadjusted"],
                "holm_adjusted_p": paired["holm_adjusted_mcnemar_p"],
                "analysis_role": source["analysis_role"],
            }
        )
    return rows


def v22_calibrated_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = {row["condition"]: row for row in analysis["condition_summaries"]}
    calibrated = summaries["calibrated_probability_mean"]
    rows = []
    for comparison in analysis["paired_family"]["comparisons"]:
        baseline = summaries[comparison["baseline"]]
        rows.append(
            {
                "comparison": comparison["comparison"],
                "n": comparison["n"],
                "calibrated_accuracy": calibrated["accuracy"],
                "baseline_accuracy": baseline["accuracy"],
                "difference_calibrated_minus_baseline": comparison[
                    "difference_calibrated_minus_baseline"
                ],
                "bootstrap_ci95_low": comparison["bootstrap_ci95_low"],
                "bootstrap_ci95_high": comparison["bootstrap_ci95_high"],
                "calibrated_only_correct": comparison["calibrated_only_correct"],
                "baseline_only_correct": comparison["baseline_only_correct"],
                "mcnemar_exact_p": comparison["mcnemar_exact_p"],
                "holm_adjusted_mcnemar_p": comparison["holm_adjusted_mcnemar_p"],
                "analysis_role": analysis["analysis_role"],
            }
        )
    return rows


def calibration_table(
    teacher: dict[str, Any], confidence_conditioning: dict[str, Any]
) -> list[dict[str, Any]]:
    conditioned = {
        row["expert"]: row for row in confidence_conditioning["expert_summaries"]
    }
    rows = []
    for expert, source in teacher["calibration"]["experts"].items():
        condition = conditioned[expert]
        rows.append(
            {
                "expert": expert,
                "fitted_temperature": source["temperature"],
                "token_count": source["token_count"],
                "example_count": teacher["example_count"],
                "top1_accuracy": source["top1_accuracy"],
                "mean_confidence": source["mean_confidence"],
                "confidence_minus_accuracy": source["confidence_minus_accuracy"],
                "ece_15": source["ece_15"],
                "mean_nll": source["mean_nll"],
                "mean_brier": source["mean_brier"],
                "mean_entropy": source["mean_entropy"],
                "mean_margin": source["mean_margin"],
                "mean_confidence_given_correct": condition["mean_confidence_given_correct"],
                "mean_confidence_given_incorrect": condition["mean_confidence_given_incorrect"],
                "correct_minus_incorrect_confidence": condition[
                    "correct_minus_incorrect_confidence"
                ],
                "analysis_role": teacher["analysis_role"],
            }
        )
    return sorted(rows, key=lambda row: row["expert"])


def aggregator_table(teacher: dict[str, Any], robust_soft: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in teacher["committee_results"]:
        if source["k"] != 7:
            continue
        rows.append(
            {
                "method": source["method"],
                "k": source["k"],
                "token_accuracy": source["token_accuracy"],
                "first_token_accuracy": source["first_token_accuracy"],
                "answer_token_accuracy": source["answer_token_accuracy"],
                "eos_accuracy": source["eos_accuracy"],
                "all_gold_token_example_accuracy": source["example_accuracy"],
                "candidate_argmax_certification_rate": source["candidate_argmax_certification_rate"],
                "token_total": source["token_total"],
                "example_total": source["example_total"],
                "expert_token_forward_proxy": source["expert_token_forward_proxy"],
                "analysis_role": source["analysis_role"],
            }
        )
    existing = {row["method"] for row in rows}
    for source in robust_soft["method_summaries"]:
        if source["method"] in existing:
            continue
        rows.append(
            {
                "method": source["method"],
                "k": 7,
                "token_accuracy": source["token_accuracy"],
                "first_token_accuracy": source["first_token_accuracy"],
                "answer_token_accuracy": source["answer_token_accuracy"],
                "eos_accuracy": source["eos_accuracy"],
                "all_gold_token_example_accuracy": source["example_accuracy"],
                "candidate_argmax_certification_rate": source["candidate_argmax_certification_rate"],
                "token_total": source["token_total"],
                "example_total": source["example_total"],
                "expert_token_forward_proxy": 7 * source["token_total"],
                "analysis_role": source["analysis_role"],
            }
        )
    return rows


def scaling_table(
    teacher: dict[str, Any], committee_diagnostics: dict[str, Any]
) -> list[dict[str, Any]]:
    intervals = {
        (row["method"], int(row["k"])): row
        for row in committee_diagnostics["committee_scaling_intervals"]
    }
    balanced = {
        (row["method"], int(row["k"])): row
        for row in committee_diagnostics["balanced_committee_scaling"]
    }
    rows = []
    for source in teacher["committee_scaling"]:
        interval = intervals[(source["method"], int(source["k"]))]
        balanced_row = balanced[(source["method"], int(source["k"]))]
        rows.append(
            {
                "method": source["method"],
                "k": source["k"],
                "committee_count": source["committee_count"],
                "mean_example_accuracy": source["mean_example_accuracy"],
                "mean_example_accuracy_ci95_low": interval[
                    "mean_example_accuracy_ci95_low"
                ],
                "mean_example_accuracy_ci95_high": interval[
                    "mean_example_accuracy_ci95_high"
                ],
                "min_example_accuracy": source["min_example_accuracy"],
                "max_example_accuracy": source["max_example_accuracy"],
                "best_example_accuracy": source["best_example_accuracy"],
                "best_example_accuracy_ci95_low": interval[
                    "best_example_accuracy_ci95_low"
                ],
                "best_example_accuracy_ci95_high": interval[
                    "best_example_accuracy_ci95_high"
                ],
                "mean_balanced_answer_class_accuracy": balanced_row[
                    "mean_balanced_answer_class_accuracy"
                ],
                "best_balanced_answer_class_accuracy": balanced_row[
                    "best_balanced_answer_class_accuracy"
                ],
                "best_balanced_members": ", ".join(
                    balanced_row["best_balanced_members"]
                ),
                "best_token_accuracy": source["best_token_accuracy"],
                "best_members": ", ".join(source["best_members"]),
                "forward_proxy_per_example": source["k"],
                "analysis_role": source["analysis_role"],
            }
        )
    return rows


def loo_shapley_table(
    teacher: dict[str, Any], selected_committee: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    loo = {row["removed_expert"]: row for row in teacher["leave_one_out"] if row["method"] == "hard_plurality"}
    shapley = teacher["shapley"]["hard_plurality"]
    free_strata = {
        (row["removed_expert"], row["value"]): row["difference_full_minus_loo"]
        for row in (
            selected_committee.get("leave_one_out_family", {}).get("descriptive_strata", [])
            if selected_committee
            else []
        )
        if row["dimension"] == "answer_class"
    }
    rows = []
    for expert in sorted(shapley["shapley"]):
        source = loo[expert]
        rows.append(
            {
                "expert": expert,
                "standalone_example_accuracy": shapley["standalone"][expert],
                "full_k7_example_accuracy": source["full_example_accuracy"],
                "leave_one_out_example_accuracy": source["leave_one_out_example_accuracy"],
                "full_minus_leave_one_out_example": source["full_minus_leave_one_out"],
                "full_minus_leave_one_out_token": source["full_minus_leave_one_out_token"],
                "exact_shapley_value": shapley["shapley"][expert],
                "utility": shapley["utility"],
                "answerable_delta": free_strata.get(
                    (expert, "answer"), "pending free-running analysis"
                ),
                "abstention_delta": free_strata.get(
                    (expert, "abstain"), "pending free-running analysis"
                ),
            }
        )
    return rows


def temperature_table(teacher: dict[str, Any], hard_tie_temp: dict[str, Any]) -> list[dict[str, Any]]:
    interventions = teacher["temperature_interventions"]
    hard_baseline = next(row for row in hard_tie_temp["rows"] if row["condition"] == "baseline_raw")
    rows = [
        {
            "scope": "hard raw reference",
            "perturbed_expert": "all",
            "temperature": 1.0,
            "token_accuracy": hard_baseline["token_accuracy"],
            "all_gold_token_example_accuracy": hard_baseline["example_accuracy"],
            "mean_expert_nll": "not applicable",
            "candidate_certification_rate": 1.0,
            "selected_token_change_rate": 0.0,
            "note": "Raw confidence tie-break; top-1 proposals and unique pluralities are positive-temperature invariant.",
        }
    ]
    for source in hard_tie_temp["rows"]:
        if source["condition"] == "baseline_raw":
            continue
        rows.append(
            {
                "scope": f'hard tie-break: {source["scope"]}',
                "perturbed_expert": source["perturbed_expert"],
                "temperature": source["temperature"] if source["temperature"] is not None else "fitted vector",
                "token_accuracy": source["token_accuracy"],
                "all_gold_token_example_accuracy": source["example_accuracy"],
                "mean_expert_nll": "not applicable",
                "candidate_certification_rate": 1.0,
                "selected_token_change_rate": source["change_rate_all_tokens"],
                "note": "Votes unchanged; only confidence resolution among plurality-tied leaders can move.",
            }
        )
    shared_nll = teacher["calibration"]["shared_mean_nll_by_temperature"]
    for temperature, source in interventions["shared_temperature"].items():
        rows.append(
            {
                "scope": "shared temperature",
                "perturbed_expert": "all",
                "temperature": temperature,
                "token_accuracy": source["token_accuracy"],
                "all_gold_token_example_accuracy": source["example_accuracy"],
                "mean_expert_nll": shared_nll[temperature],
                "candidate_certification_rate": source["certification_rate"],
                "selected_token_change_rate": "not captured",
                "note": "Probability mean; DEV gold-prefix intervention.",
            }
        )
    for expert, curve in interventions["one_expert_perturbed"].items():
        for temperature, source in curve.items():
            rows.append(
                {
                    "scope": "one expert perturbed",
                    "perturbed_expert": expert,
                    "temperature": temperature,
                    "token_accuracy": source["token_accuracy"],
                    "all_gold_token_example_accuracy": source["example_accuracy"],
                    "mean_expert_nll": teacher["calibration"]["experts"][expert]["mean_nll_by_temperature"][temperature],
                    "candidate_certification_rate": source["certification_rate"],
                    "selected_token_change_rate": "not captured",
                    "note": "Other experts remain at T=1; DEV gold-prefix intervention.",
                }
            )
    return rows


def adaptive_selection_table(adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in adaptive["policies"]:
        rows.append(
            {
                "threshold": source["threshold"],
                "selected": source["threshold"] == adaptive["selected_threshold"],
                "core": ", ".join(adaptive["core"]),
                "token_accuracy": source["token_accuracy"],
                "all_gold_token_example_accuracy": source["example_accuracy"],
                "token_delta_vs_full_k7": source["token_accuracy_retention_vs_full"],
                "example_delta_vs_full_k7": source["example_accuracy_retention_vs_full"],
                "escalation_rate": source["escalation_rate"],
                "mean_expert_token_forwards": source["mean_expert_token_forwards"],
                "forward_reduction_vs_full_k7": source["forward_reduction_vs_full"],
                "false_non_escalation_rate": source["false_non_escalation_rate"],
                "analysis_role": adaptive["analysis_role"],
            }
        )
    return rows


def teacher_free_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = {row["method"]: row for row in analysis["method_summaries"]}
    comparisons = {
        row["other_method"]: row for row in analysis["paired_family"]["comparisons"]
    }
    rows = []
    for source in analysis["teacher_forced_vs_free_running"]:
        method = source["method"]
        free = summaries[method]
        comparison = comparisons.get(method)
        rows.append(
            {
                **source,
                "free_running_correct": free["correct"],
                "free_running_n": free["n"],
                "free_running_answerable_accuracy": free["answerable_accuracy"],
                "free_running_abstention_accuracy": free["abstention_accuracy"],
                "mean_latency_seconds": free["mean_latency_seconds"],
                "mean_forward_passes": free["mean_forward_passes"],
                "paired_bootstrap_ci95_low": comparison["bootstrap_ci95_low"] if comparison else 0.0,
                "paired_bootstrap_ci95_high": comparison["bootstrap_ci95_high"] if comparison else 0.0,
                "holm_adjusted_mcnemar_p": comparison["holm_adjusted_mcnemar_p"] if comparison else 1.0,
                "analysis_role": analysis["analysis_role"],
            }
        )
    return rows


def adaptive_free_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    paired = analysis["paired_accuracy"]
    efficiency = analysis["efficiency"]
    rows = []
    for source in analysis["summaries"]:
        is_adaptive = source["condition"] == "adaptive_k3_to_k7"
        rows.append(
            {
                **source,
                "accuracy_delta_vs_hard": paired["difference_adaptive_minus_hard"] if is_adaptive else 0.0,
                "paired_ci95_low": paired["paired_bootstrap_ci95_low"] if is_adaptive else 0.0,
                "paired_ci95_high": paired["paired_bootstrap_ci95_high"] if is_adaptive else 0.0,
                "mcnemar_exact_p": paired["mcnemar_exact_p"] if is_adaptive else 1.0,
                "forward_reduction_vs_hard": efficiency["forward_pass_reduction_vs_hard"] if is_adaptive else 0.0,
                "analysis_role": analysis["analysis_role"],
            }
        )
    return rows


def cross_results_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = {
        row["baseline"]: row for row in analysis["paired_family"]["comparisons"]
    }
    rows = []
    for source in analysis["condition_summaries"]:
        comparison = comparisons.get(source["condition"])
        rows.append(
            {
                **source,
                "hard_minus_condition": comparison["difference_hard_minus_baseline"] if comparison else 0.0,
                "paired_ci95_low": comparison["bootstrap_ci95_low"] if comparison else 0.0,
                "paired_ci95_high": comparison["bootstrap_ci95_high"] if comparison else 0.0,
                "holm_adjusted_mcnemar_p": comparison["holm_adjusted_mcnemar_p"] if comparison else 1.0,
                "analysis_role": analysis["analysis_role"],
            }
        )
    return rows


def selected_committee_scaling_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return list(analysis["selected_best_at_k_scaling"])


def selected_committee_loo_table(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    family = analysis["leave_one_out_family"]
    answer_class = {
        (row["removed_expert"], row["value"]): row["difference_full_minus_loo"]
        for row in family["descriptive_strata"]
        if row["dimension"] == "answer_class"
    }
    return [
        {
            **row,
            "answerable_delta": answer_class[(row["removed_expert"], "answer")],
            "abstention_delta": answer_class[(row["removed_expert"], "abstain")],
        }
        for row in family["comparisons"]
    ]


def make_markdown(
    datasets: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    rescue: list[dict[str, Any]],
    v22_calibrated: list[dict[str, Any]] | None,
    calibration: list[dict[str, Any]],
    aggregators: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
    loo: list[dict[str, Any]],
    temperatures: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
    committee_scaling_free: list[dict[str, Any]] | None,
    committee_loo_free: list[dict[str, Any]] | None,
    free_run: list[dict[str, Any]] | None,
    adaptive_free: list[dict[str, Any]] | None,
    cross_results: list[dict[str, Any]] | None,
) -> str:
    selected_scaling = {
        "hard_plurality",
        "probability_mean",
        "calibrated_probability_mean",
        "reciprocal_rank_top5",
    }
    committee_scaling_free_block = (
        md_table(
            ["K", "Committee", "Exact acc.", "Answerable", "Abstention", "Balanced", "Latency (s)", "Forwards", "1000×acc./forward"],
            (
                [
                    row["k"],
                    row["committee"],
                    pct(row["accuracy"]),
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    pct(row["balanced_answer_class_accuracy"]),
                    num(row["mean_latency_seconds"], 3),
                    num(row["mean_forward_passes"], 2),
                    num(1000 * row["accuracy_per_mean_forward_pass"], 4),
                ]
                for row in committee_scaling_free
            ),
        )
        if committee_scaling_free
        else "PENDING E-COMMITTEE-FREERUN. The 12-condition selected DEV panel is frozen."
    )
    committee_loo_free_block = (
        md_table(
            [
                "Removed",
                "Full minus K6",
                "Answerable delta",
                "Abstention delta",
                "Paired 95% CI",
                "Full-only/K6-only",
                "Holm p",
            ],
            (
                [
                    row["removed_expert"],
                    pct(row["difference_full_minus_loo"]),
                    pct(row["answerable_delta"]),
                    pct(row["abstention_delta"]),
                    f'{pct(row["bootstrap_ci95_low"])} to {pct(row["bootstrap_ci95_high"])}',
                    f'{row["full_only_correct"]}/{row["loo_only_correct"]}',
                    sci(row["holm_adjusted_mcnemar_p"]),
                ]
                for row in committee_loo_free
            ),
        )
        if committee_loo_free
        else "PENDING E-COMMITTEE-FREERUN. Cached local LOO values are complete; paired trajectory and systems deltas await the frozen panel."
    )
    loo_scope_note = (
        "Values use DEV all-gold-token example accuracy. The selected free-running "
        "answerable/abstention sequence deltas and paired trajectory effects are reported below."
        if committee_loo_free
        else "Values use DEV all-gold-token example accuracy; answerable/abstention "
        "sequence deltas remain pending free-running analysis."
    )
    best_primary = max(row["exact_accuracy"] for row in primary)
    free_run_block = (
        md_table(
            ["Method", "Local token", "Local all-token ex.", "Free exact", "Free delta vs Hard", "Paired 95% CI", "Forwards"],
            (
                [
                    row["method"],
                    pct(row["teacher_forced_token_accuracy"]),
                    pct(row["teacher_forced_all_token_example_accuracy"]),
                    pct(row["free_running_exact_accuracy"]),
                    pct(row["free_sequence_advantage_vs_hard"]),
                    f'{pct(row["paired_bootstrap_ci95_low"])} to {pct(row["paired_bootstrap_ci95_high"])}',
                    num(row["mean_forward_passes"], 2),
                ]
                for row in free_run
            ),
        )
        if free_run
        else "PENDING E-FREERUN-DEV. The selected DEV reference-decoder run is in progress."
    )
    adaptive_free_block = (
        "Free-running confirmation:\n\n"
        + md_table(
            ["Condition", "Exact acc.", "Delta vs Hard", "Forwards", "Forwards/token", "Trigger/token", "Latency (s)"],
            (
                [
                    row["condition"],
                    pct(row["accuracy"]),
                    pct(row["accuracy_delta_vs_hard"]),
                    num(row["mean_forward_passes"], 2),
                    num(row["mean_forwards_per_generated_token"], 2),
                    pct(row["adaptive_trigger_token_rate"]),
                    num(row["mean_latency_seconds"], 3),
                ]
                for row in adaptive_free
            ),
        )
        if adaptive_free
        else "The selected K3-to-K7 policy is frozen; free-running exact accuracy and measured latency remain pending."
    )
    cross_block = (
        md_table(
            ["Condition", "Correct", "Exact acc.", "Answerable", "Abstention", "Hard minus condition", "Holm p"],
            (
                [
                    row["condition"],
                    f'{row["correct"]}/{row["n"]}',
                    pct(row["accuracy"]),
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    pct(row["hard_minus_condition"]),
                    sci(row["holm_adjusted_mcnemar_p"]),
                ]
                for row in cross_results
            ),
        )
        + "\n\nFixed-method cross-corpus transfer only. No v2.2 and v2.1.2 accuracy is pooled."
        if cross_results
        else "PENDING fixed-method v2.1.2 transfer. E-CROSSBENCH-GATE passed with caveats; no combined v2.2/v2.1.2 accuracy will be reported."
    )
    v22_calibrated_block = (
        md_table(
            ["Comparison", "Calibrated", "Baseline", "Difference (95% CI)", "Calibrated-only/Baseline-only", "McNemar p", "Holm p"],
            (
                [
                    row["comparison"],
                    pct(row["calibrated_accuracy"]),
                    pct(row["baseline_accuracy"]),
                    f'{pct(row["difference_calibrated_minus_baseline"])} ({pct(row["bootstrap_ci95_low"])} to {pct(row["bootstrap_ci95_high"])})',
                    f'{row["calibrated_only_correct"]}/{row["baseline_only_correct"]}',
                    sci(row["mcnemar_exact_p"]),
                    sci(row["holm_adjusted_mcnemar_p"]),
                ]
                for row in v22_calibrated
            ),
        )
        + "\n\nSeparate selected-confirmation family. Hard/raw anchors were known before the calibrated condition was frozen; the archived n=32 exposure caveat remains."
        if v22_calibrated
        else "PENDING E-V22-CALTEST. One calibrated K=7 condition is frozen before output under a separate two-comparison Holm family."
    )
    sections = [
        "# Machine-Generated Paper Tables",
        "",
        "All populated values in this file are generated by `experiments/phase2_mechanism/make_paper_tables.py`. "
        "Exact source and output hashes are recorded in `paper/tables/table_metadata.json`.",
        "",
        "## Table 1 — Dataset and exposure provenance",
        "",
        md_table(
            ["Dataset", "Version", "Split", "N", "SHA256", "Role", "Selection/prior-access boundary"],
            (
                [row["dataset"], row["version"], row["split"], row["n"], row["sha256"], row["evaluation_role"], f'{row["selection_exposure"]}; {row["prior_access_or_overlap"]}']
                for row in datasets
            ),
        ),
        "",
        "## Table 2 — Model and adapter inventory",
        "",
        md_table(
            ["Model/adapter", "Role", "Parameters", "Type", "Rank", "Alpha", "Weights SHA256"],
            ([row["model_or_adapter"], row["role"], row["parameters"], row["adapter_type"], row["rank"] or "—", row["alpha"] or "—", row["weights_sha256"]] for row in inventory),
        ),
        "",
        "## Table 3 — Primary full-TEST benchmark",
        "",
        md_table(
            ["Method", "N", "Exact acc. (Wilson 95% CI)", "Answerable", "Abstention", "False abstention", "Mean tokens", "Latency (s)", "Expert forwards"],
            (
                [
                    f'**{row["method"]}**' if row["exact_accuracy"] == best_primary else row["method"],
                    row["n"],
                    f'{pct(row["exact_accuracy"])} ({pct(row["ci95_low_wilson"])}–{pct(row["ci95_high_wilson"])})',
                    pct(row["answerable_accuracy"]),
                    pct(row["abstention_accuracy"]),
                    pct(row["false_abstention_rate"]),
                    num(row["mean_output_tokens"], 2),
                    num(row["mean_latency_seconds"], 3),
                    num(row["mean_expert_forwards"], 2),
                ]
                for row in primary
            ),
        ),
        "",
        "Source: E-CORNERSTONE-RAW, N=4,050. Single-proportion intervals are Wilson intervals; paired differences are in Table 4.",
        "",
        "## Table 4 — Paired rescue/harm",
        "",
        md_table(
            ["Comparison", "Both correct", "Rescue", "Harm", "Both wrong", "Net gain (95% CI)", "Exact McNemar p", "Holm p"],
            ([row["comparison"], row["both_correct"], row["rescue"], row["harm"], row["both_wrong"], f'{pct(row["net_gain"])} ({pct(row["ci95_low"])}–{pct(row["ci95_high"])})', sci(row["mcnemar_exact_p"]), sci(row["holm_adjusted_p"])] for row in rescue),
        ),
        "",
        "## Table 4b — Selected calibrated probability on v2.2 TEST",
        "",
        v22_calibrated_block,
        "",
        "## Table 5 — Expert calibration (DEV gold prefixes)",
        "",
        md_table(
            ["Expert", "T", "Top-1 acc.", "Mean conf.", "Conf.−acc.", "Conf.|correct", "Conf.|wrong", "ECE-15", "NLL", "Brier", "Entropy", "Margin"],
            ([row["expert"], row["fitted_temperature"], pct(row["top1_accuracy"]), pct(row["mean_confidence"]), pct(row["confidence_minus_accuracy"]), pct(row["mean_confidence_given_correct"]), pct(row["mean_confidence_given_incorrect"]), num(row["ece_15"], 4), num(row["mean_nll"], 3), num(row["mean_brier"], 3), num(row["mean_entropy"], 3), num(row["mean_margin"], 3)] for row in calibration),
        ),
        "",
        "N=4,050 examples and 18,760 gold-prefix token decisions per expert. Temperatures are DEV-selected.",
        "",
        "## Table 6 — K=7 aggregator comparison under teacher forcing",
        "",
        md_table(
            ["Method", "Token", "First token", "Answer token", "EOS", "All-token example", "Certificate"],
            ([row["method"], pct(row["token_accuracy"]), pct(row["first_token_accuracy"]), pct(row["answer_token_accuracy"]), pct(row["eos_accuracy"]), pct(row["all_gold_token_example_accuracy"]), pct(row["candidate_argmax_certification_rate"], 4)] for row in aggregators),
        ),
        "",
        "Validation-weighted hard, entropy-normalized, trimmed, median, and confidence-clipped rows are bounded post-DEV exploratory robustness controls; none was promoted. The median row is a top-64-union approximation at its two uncertified positions; all other displayed winners are certified.",
        "",
        "## Table 7 — Committee-size scaling (selected method families)",
        "",
        md_table(
            [
                "Method",
                "K",
                "Committees",
                "Mean all-token example (95% CI)",
                "Mean balanced",
                "Min all-token example",
                "Best all-token example (95% CI)",
                "Best all-token members",
                "Best balanced",
                "Best balanced members",
            ],
            ([row["method"], row["k"], row["committee_count"], f'{pct(row["mean_example_accuracy"])} ({pct(row["mean_example_accuracy_ci95_low"])} to {pct(row["mean_example_accuracy_ci95_high"])})', pct(row["mean_balanced_answer_class_accuracy"]), pct(row["min_example_accuracy"]), f'{pct(row["best_example_accuracy"])} ({pct(row["best_example_accuracy_ci95_low"])} to {pct(row["best_example_accuracy_ci95_high"])})', row["best_members"], pct(row["best_balanced_answer_class_accuracy"]), row["best_balanced_members"]] for row in scaling if row["method"] in selected_scaling),
        ),
        "",
        "The CSV contains every fitted aggregator. Mean intervals use per-example mean correctness across the exhaustive committees at fixed K. Best-at-K Wilson intervals are conditional on the same DEV selection and are not selection-adjusted; the best envelope is not a monotonicity claim.",
        "",
        "Selected best-at-K free-running DEV systems curve:",
        "",
        committee_scaling_free_block,
        "",
        "## Table 8 — Hard-plurality leave-one-out and exact Shapley",
        "",
        md_table(
            ["Expert", "Standalone", "K7−K6 example", "K7−K6 token", "Exact Shapley"],
            ([row["expert"], pct(row["standalone_example_accuracy"]), pct(row["full_minus_leave_one_out_example"]), pct(row["full_minus_leave_one_out_token"]), pct(row["exact_shapley_value"])] for row in loo),
        ),
        "",
        loo_scope_note,
        "",
        "Selected free-running leave-one-out family:",
        "",
        committee_loo_free_block,
        "",
        "## Table 9 — Temperature interventions",
        "",
        md_table(
            ["Scope", "Expert", "T", "Token acc.", "All-token example", "Selected change", "NLL", "Certificate"],
            ([row["scope"], row["perturbed_expert"], row["temperature"], pct(row["token_accuracy"]), pct(row["all_gold_token_example_accuracy"]), pct(row["selected_token_change_rate"]) if isinstance(row["selected_token_change_rate"], (float, int)) else row["selected_token_change_rate"], num(row["mean_expert_nll"], 3) if isinstance(row["mean_expert_nll"], (float, int)) else row["mean_expert_nll"], pct(row["candidate_certification_rate"], 4)] for row in temperatures),
        ),
        "",
        "The full intervention grid is intentionally kept in CSV. Hard tie-break change rates are exact cache counterfactuals; soft-fusion selected-token change was not captured and remains marked rather than reconstructed.",
        "",
        "## Table 10 — Teacher-forced versus free-running",
        "",
        free_run_block,
        "",
        "## Table 11 — Adaptive consultation and efficiency",
        "",
        md_table(
            ["Threshold", "Selected", "Token acc.", "All-token example", "Escalation", "Forwards/token", "Forward reduction", "False non-escalation"],
            ([row["threshold"], "yes" if row["selected"] else "no", pct(row["token_accuracy"]), pct(row["all_gold_token_example_accuracy"]), pct(row["escalation_rate"]), num(row["mean_expert_token_forwards"], 2), pct(row["forward_reduction_vs_full_k7"]), pct(row["false_non_escalation_rate"])] for row in adaptive),
        ),
        "",
        "DEV teacher-forced policy selection. " + adaptive_free_block,
        "",
        "## Table 12 — Cross-benchmark confirmation",
        "",
        cross_block,
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cornerstone = load_json(CORNERSTONE_AUDIT)
    cross = load_json(CROSSBENCH_GATE)
    summaries = load_json(TEST_SUMMARY)["summaries"]
    stats = load_json(TEST_STATS)
    mechanism = load_json(MECHANISM)
    teacher = load_json(TEACHER)
    robust_soft = load_json(ROBUST_SOFT)
    confidence_conditioning = load_json(CONFIDENCE_CONDITIONING)
    committee_diagnostics = load_json(COMMITTEE_DIAGNOSTICS)
    coalition_strata = load_json(COALITION_STRATA)
    diversity_extended = load_json(DIVERSITY_EXTENDED)
    hard_tie_temp = load_json(HARD_TIE_TEMP)
    source_manifest = load_json(SOURCE_MANIFEST)
    adaptive = load_json(ADAPTIVE)
    free_run = load_optional_complete(
        FREE_RUN, "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE"
    )
    adaptive_free = load_optional_complete(
        ADAPTIVE_FREE_RUN, "DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE"
    )
    cross_results = load_optional_complete(
        CROSS_RESULTS, "V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE"
    )
    v22_calibrated = load_optional_complete(
        V22_CALIBRATED_RESULTS, "V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE"
    )
    selected_committee = load_optional_complete(
        SELECTED_COMMITTEE_RESULTS, "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE"
    )

    tables: dict[str, list[dict[str, Any]]] = {
        "table01_dataset_provenance": dataset_table(cornerstone, cross),
        "table02_model_adapter_inventory": inventory_table(cornerstone, source_manifest, summaries),
        "table03_primary_test": primary_test_table(summaries),
        "table04_paired_rescue_harm": rescue_table(mechanism, stats),
        "table05_expert_calibration": calibration_table(teacher, confidence_conditioning),
        "table06_teacher_forced_aggregators": aggregator_table(teacher, robust_soft),
        "table07_committee_scaling": scaling_table(teacher, committee_diagnostics),
        "table08_leave_one_out_shapley": loo_shapley_table(teacher, selected_committee),
        "table09_temperature_interventions": temperature_table(teacher, hard_tie_temp),
        "table11_adaptive_teacher_forced_selection": adaptive_selection_table(adaptive),
        "appendix_all_127_committees": teacher["committee_results"],
        "appendix_all_127_committee_diagnostics": committee_diagnostics["records"],
        "appendix_full_test_coalition_strata": coalition_strata["records"],
        "appendix_pairwise_diversity_extended": diversity_extended["pairs"],
    }
    if free_run:
        tables["table10_teacher_forced_vs_free_running"] = teacher_free_table(free_run)
    if adaptive_free:
        tables["table11_adaptive_free_running"] = adaptive_free_table(adaptive_free)
    if cross_results:
        tables["table12_cross_v212"] = cross_results_table(cross_results)
    if v22_calibrated:
        tables["table04b_v22_calibrated_confirmation"] = v22_calibrated_table(v22_calibrated)
    if selected_committee:
        tables["table07b_selected_committee_free_running"] = selected_committee_scaling_table(
            selected_committee
        )
        tables["table08b_selected_leave_one_out_free_running"] = selected_committee_loo_table(
            selected_committee
        )
    for name, rows in tables.items():
        write_csv(TABLE_DIR / f"{name}.csv", rows)

    diversity_rows: list[dict[str, Any]] = []
    with EXPERT_DIVERSITY.open(newline="", encoding="utf-8") as handle:
        diversity_rows.extend(csv.DictReader(handle))
    write_csv(TABLE_DIR / "appendix_expert_diversity.csv", diversity_rows)

    markdown = make_markdown(
        tables["table01_dataset_provenance"],
        tables["table02_model_adapter_inventory"],
        tables["table03_primary_test"],
        tables["table04_paired_rescue_harm"],
        tables.get("table04b_v22_calibrated_confirmation"),
        tables["table05_expert_calibration"],
        tables["table06_teacher_forced_aggregators"],
        tables["table07_committee_scaling"],
        tables["table08_leave_one_out_shapley"],
        tables["table09_temperature_interventions"],
        tables["table11_adaptive_teacher_forced_selection"],
        tables.get("table07b_selected_committee_free_running"),
        tables.get("table08b_selected_leave_one_out_free_running"),
        tables.get("table10_teacher_forced_vs_free_running"),
        tables.get("table11_adaptive_free_running"),
        tables.get("table12_cross_v212"),
    )
    markdown_path = ROOT / "paper" / "TABLES.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    source_paths = [
        CORNERSTONE_AUDIT,
        CROSSBENCH_GATE,
        TEST_SUMMARY,
        TEST_STATS,
        MECHANISM,
        TEACHER,
        ROBUST_SOFT,
        CONFIDENCE_CONDITIONING,
        COMMITTEE_DIAGNOSTICS,
        COALITION_STRATA,
        DIVERSITY_EXTENDED,
        HARD_TIE_TEMP,
        SOURCE_MANIFEST,
        EXPERT_DIVERSITY,
        ADAPTIVE,
    ]
    source_paths.extend(
        path
        for path in (
            FREE_RUN,
            ADAPTIVE_FREE_RUN,
            CROSS_RESULTS,
            V22_CALIBRATED_RESULTS,
            SELECTED_COMMITTEE_RESULTS,
        )
        if path.is_file()
    )
    output_paths = sorted(TABLE_DIR.glob("*.csv")) + [markdown_path]
    metadata = {
        "schema_version": "1.0",
        "status": (
            "ALL_PLANNED_TABLES_COMPLETE"
            if free_run and adaptive_free and cross_results and v22_calibrated and selected_committee
            else "TABLES_PARTIAL_WITH_EXPLICIT_PENDING_BLOCKS"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": rel(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "sources": {rel(path): sha256(path) for path in source_paths},
        "outputs": {rel(path): sha256(path) for path in output_paths},
        "notes": [
            "Every populated table block is generated from frozen, TEST, DEV, or cross-corpus analysis artifacts.",
            "Table 3 single-proportion confidence intervals are Wilson intervals.",
            "Missing long-run outputs remain explicit pending placeholders and are filled conditionally after strict analyzers complete.",
            "No v2.2 and v2.1.2 accuracies are aggregated.",
            "The selected calibrated v2.2 TEST confirmation is a separate two-comparison family from the original cornerstone family.",
        ],
    }
    metadata_path = TABLE_DIR / "table_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "csv_count": len(output_paths) - 1,
                "markdown": str(markdown_path),
                "metadata": str(metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
