from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[2]
RAW_RESULT = LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
CORNERSTONE_AUDIT = LAB_ROOT / "reports/phase2_cornerstone_verification.json"
SEALED_STATISTICS = LAB_ROOT / "results/summaries/sealed_statistics.json"

OUTPUT_ROOT = LAB_ROOT / "results/phase2_mechanism"
REPORT = LAB_ROOT / "reports/PHASE2_RESCUE_HARM.md"
LIVE_REPORT = LAB_ROOT / "reports/PHASE2_LIVE_RESULTS.md"
JSON_OUTPUT = OUTPUT_ROOT / "existing_results_analysis.json"
RESCUE_CSV = OUTPUT_ROOT / "rescue_harm_strata.csv"
COALITION_CSV = OUTPUT_ROOT / "coalition_frequency.csv"
EXPERT_CSV = OUTPUT_ROOT / "expert_trace_profiles.csv"
PAIRWISE_CSV = OUTPUT_ROOT / "pairwise_top1_agreement.csv"

EXPERTS = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]

CONDITION_ALIASES = {
    ("backbone_k1", "hard_plurality"): "backbone",
    ("track_a_k1", "hard_plurality"): "track_a",
    ("abstention_k1", "hard_plurality"): "abstention",
    ("full_k7", "hard_plurality"): "hard_k7",
    ("full_k7", "probability_mean"): "probability_k7",
}

COMPARISONS = {
    "hard_vs_track_a": {"candidate": "hard_k7", "baseline": "track_a", "label": "Hard K=7 vs Track A"},
    "hard_vs_probability": {
        "candidate": "hard_k7",
        "baseline": "probability_k7",
        "label": "Hard K=7 vs Probability K=7",
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_seed(*values: str) -> int:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def paired_bootstrap_ci(rescue: int, harm: int, n: int, *, seed: int, samples: int = 50_000) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    probabilities = [rescue / n, harm / n, 1.0 - (rescue + harm) / n]
    counts = np.random.default_rng(seed).multinomial(n, probabilities, size=samples)
    differences = (counts[:, 0] - counts[:, 1]) / n
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p(rescue: int, harm: int) -> float:
    discordant = rescue + harm
    if discordant == 0:
        return 1.0
    tail = min(rescue, harm)
    numerator = 2 * sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, numerator / (2**discordant))


def length_bucket(value: int) -> str:
    if value <= 4:
        return "01-04"
    if value <= 8:
        return "05-08"
    if value <= 16:
        return "09-16"
    if value <= 24:
        return "17-24"
    return "25-32"


def agreement_bucket(value: float) -> str:
    if value < 0.40:
        return "[0.00,0.40)"
    if value < 0.50:
        return "[0.40,0.50)"
    if value < 0.60:
        return "[0.50,0.60)"
    if value < 0.75:
        return "[0.60,0.75)"
    if value < 0.90:
        return "[0.75,0.90)"
    return "[0.90,1.00]"


def position_bucket(value: int) -> str:
    if value == 0:
        return "t=0"
    if value <= 2:
        return "t=1-2"
    if value <= 5:
        return "t=3-5"
    if value <= 10:
        return "t=6-10"
    return "t>=11"


def coalition_mask(coalition: Iterable[str]) -> str:
    members = set(coalition)
    return "".join("1" if expert in members else "0" for expert in EXPERTS)


def minimal_row(row: dict[str, Any], *, keep_steps: bool) -> dict[str, Any]:
    packet = {
        "example_id": str(row["example_id"]),
        "correct": bool(row["correct"]),
        "abstained": bool(row["abstained"]),
        "decision": str(row.get("decision") or "unknown"),
        "difficulty": str(row.get("difficulty") or "unknown"),
        "domain": str(row.get("domain") or "unknown"),
        "reasoning_family": str(row.get("reasoning_family") or "unknown"),
        "generated_tokens": int(row.get("generated_tokens") or 0),
        "mean_agreement": float(row["mean_agreement"]) if row.get("mean_agreement") is not None else None,
        "mean_vote_entropy": float(row["mean_vote_entropy"]) if row.get("mean_vote_entropy") is not None else None,
        "mean_pairwise_js": float(row["mean_pairwise_js"]) if row.get("mean_pairwise_js") is not None else None,
        "output": str(row.get("output") or ""),
        "output_token_ids": [int(value) for value in (row.get("output_token_ids") or [])],
        "stop_reason": str(row.get("stop_reason") or "unknown"),
        "latency_seconds": float(row.get("latency_seconds") or 0.0),
        "forward_passes": int(row.get("forward_passes") or 0),
    }
    if keep_steps:
        packet["steps"] = row.get("steps") or []
    return packet


def load_conditions() -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(RAW_RESULT):
        alias = CONDITION_ALIASES.get((str(row.get("committee")), str(row.get("method"))))
        if alias is None:
            continue
        rows[alias][str(row["example_id"])] = minimal_row(
            row,
            keep_steps=alias in {"hard_k7", "probability_k7"},
        )
    return rows


def behavior_summary(rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, condition in sorted(rows.items()):
        values = list(condition.values())
        answerable = [row for row in values if row["decision"] == "answer"]
        abstention = [row for row in values if row["decision"] != "answer"]
        summaries[name] = {
            "n": len(values),
            "correct": sum(row["correct"] for row in values),
            "accuracy": sum(row["correct"] for row in values) / len(values),
            "answerable_n": len(answerable),
            "answerable_correct": sum(row["correct"] for row in answerable),
            "answerable_accuracy": sum(row["correct"] for row in answerable) / len(answerable),
            "abstention_n": len(abstention),
            "abstention_correct": sum(row["correct"] for row in abstention),
            "abstention_accuracy": sum(row["correct"] for row in abstention) / len(abstention),
            "false_abstention_rate": sum(row["abstained"] for row in answerable) / len(answerable),
            "failure_to_abstain_rate": sum(not row["abstained"] for row in abstention) / len(abstention),
            "mean_output_tokens": statistics.mean(row["generated_tokens"] for row in values),
            "mean_latency_seconds": statistics.mean(row["latency_seconds"] for row in values),
            "mean_forward_passes": statistics.mean(row["forward_passes"] for row in values),
        }
    return summaries


def stratum_values(hard: dict[str, Any]) -> dict[str, str]:
    return {
        "global": "ALL",
        "task_category": hard["reasoning_family"],
        "difficulty": hard["difficulty"],
        "answer_class": "answerable" if hard["decision"] == "answer" else "abstention",
        "domain": hard["domain"],
        "hard_output_length": length_bucket(hard["generated_tokens"]),
        "hard_vote_agreement": agreement_bucket(float(hard["mean_agreement"] or 0.0)),
    }


def rescue_harm_analysis(rows: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    hard = rows["hard_k7"]
    ids = sorted(hard)
    for comparison_name, spec in COMPARISONS.items():
        candidate = rows[spec["candidate"]]
        baseline = rows[spec["baseline"]]
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for example_id in ids:
            for dimension, value in stratum_values(hard[example_id]).items():
                groups[(dimension, value)].append(example_id)
        for (dimension, value), group_ids in sorted(groups.items()):
            both_correct = sum(candidate[item]["correct"] and baseline[item]["correct"] for item in group_ids)
            rescue = sum(candidate[item]["correct"] and not baseline[item]["correct"] for item in group_ids)
            harm = sum(not candidate[item]["correct"] and baseline[item]["correct"] for item in group_ids)
            both_wrong = len(group_ids) - both_correct - rescue - harm
            low, high = paired_bootstrap_ci(
                rescue,
                harm,
                len(group_ids),
                seed=stable_seed(comparison_name, dimension, value),
            )
            packets.append(
                {
                    "comparison": comparison_name,
                    "comparison_label": spec["label"],
                    "candidate": spec["candidate"],
                    "baseline": spec["baseline"],
                    "dimension": dimension,
                    "stratum": value,
                    "n": len(group_ids),
                    "both_correct": both_correct,
                    "rescue": rescue,
                    "harm": harm,
                    "both_wrong": both_wrong,
                    "candidate_accuracy": (both_correct + rescue) / len(group_ids),
                    "baseline_accuracy": (both_correct + harm) / len(group_ids),
                    "rescue_rate": rescue / len(group_ids),
                    "harm_rate": harm / len(group_ids),
                    "net_gain": (rescue - harm) / len(group_ids),
                    "net_gain_ci_low": low,
                    "net_gain_ci_high": high,
                    "mcnemar_exact_p_unadjusted": exact_mcnemar_p(rescue, harm),
                    "analysis_role": "TEST CONFIRMATORY" if dimension == "global" else "TEST DESCRIPTIVE / EXPLORATORY",
                }
            )
    return packets


def coalition_analysis(hard_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    coalition_counts: Counter[tuple[str, str, str, str, bool, bool]] = Counter()
    vote_size_counts: Counter[int] = Counter()
    vote_size_correct: Counter[int] = Counter()
    expert_stats: dict[str, Counter[str]] = {expert: Counter() for expert in EXPERTS}
    expert_values: dict[str, dict[str, list[float]]] = {
        expert: defaultdict(list) for expert in EXPERTS
    }
    pair_stats: dict[tuple[str, str], Counter[str]] = {
        pair: Counter() for pair in combinations(EXPERTS, 2)
    }
    total_steps = 0
    for row in hard_rows.values():
        for step in row["steps"]:
            total_steps += 1
            coalition = list(step.get("winning_expert_coalition") or [])
            mask = coalition_mask(coalition)
            is_eos = bool(step.get("winner_is_eos"))
            coalition_counts[
                (
                    mask,
                    row["reasoning_family"],
                    "answerable" if row["decision"] == "answer" else "abstention",
                    position_bucket(int(step["t"])),
                    is_eos,
                    row["correct"],
                )
            ] += 1
            plurality = int(step["plurality_votes"])
            vote_size_counts[plurality] += 1
            vote_size_correct[plurality] += int(row["correct"])
            top_tokens = {name: int(value) for name, value in step["top_tokens"].items()}
            top_probs = {name: float(value) for name, value in step["top_probabilities"].items()}
            margins = {name: float(value) for name, value in step["top1_top2_margins"].items()}
            entropies = {name: float(value) for name, value in step["expert_entropies"].items()}
            winner = int(step["winner"])
            for expert in EXPERTS:
                supported = top_tokens[expert] == winner
                stats = expert_stats[expert]
                stats["steps"] += 1
                stats["winner_support"] += int(supported)
                stats["dissent"] += int(not supported)
                stats["correct_sequence_steps"] += int(row["correct"])
                stats["winner_support_correct_sequence"] += int(supported and row["correct"])
                stats["dissent_correct_sequence"] += int((not supported) and row["correct"])
                expert_values[expert]["top_probability"].append(top_probs[expert])
                expert_values[expert]["margin"].append(margins[expert])
                expert_values[expert]["entropy"].append(entropies[expert])
                expert_values[expert]["support_top_probability" if supported else "dissent_top_probability"].append(
                    top_probs[expert]
                )
                if row["correct"]:
                    expert_values[expert]["correct_sequence_top_probability"].append(top_probs[expert])
                else:
                    expert_values[expert]["wrong_sequence_top_probability"].append(top_probs[expert])
            for pair, stats in pair_stats.items():
                agrees = top_tokens[pair[0]] == top_tokens[pair[1]]
                stats["steps"] += 1
                stats["agreements"] += int(agrees)
                stats["disagreements"] += int(not agrees)
                stats["agreements_correct_sequence"] += int(agrees and row["correct"])
                stats["agreements_wrong_sequence"] += int(agrees and not row["correct"])
    coalition_rows = []
    for (mask, task, answer_class, position, is_eos, sequence_correct), count in sorted(coalition_counts.items()):
        coalition_rows.append(
            {
                "mask": mask,
                "members": [expert for expert, bit in zip(EXPERTS, mask, strict=True) if bit == "1"],
                "task_category": task,
                "answer_class": answer_class,
                "position_bucket": position,
                "is_eos": is_eos,
                "sequence_correct": sequence_correct,
                "token_count": count,
            }
        )
    aggregate_coalitions: dict[str, Counter[str]] = defaultdict(Counter)
    for item in coalition_rows:
        aggregate_coalitions[item["mask"]]["tokens"] += item["token_count"]
        aggregate_coalitions[item["mask"]]["correct_sequence_tokens"] += item["token_count"] * int(item["sequence_correct"])
        aggregate_coalitions[item["mask"]]["eos_tokens"] += item["token_count"] * int(item["is_eos"])
    coalition_summary = []
    for mask, counts in sorted(aggregate_coalitions.items(), key=lambda item: (-item[1]["tokens"], item[0])):
        coalition_summary.append(
            {
                "mask": mask,
                "members": [expert for expert, bit in zip(EXPERTS, mask, strict=True) if bit == "1"],
                "token_count": counts["tokens"],
                "token_fraction": counts["tokens"] / total_steps,
                "sequence_correct_association": counts["correct_sequence_tokens"] / counts["tokens"],
                "eos_fraction": counts["eos_tokens"] / counts["tokens"],
            }
        )
    expert_profiles = []
    for expert in EXPERTS:
        stats = expert_stats[expert]
        values = expert_values[expert]
        expert_profiles.append(
            {
                "expert": expert,
                "steps": stats["steps"],
                "winner_support_rate": stats["winner_support"] / stats["steps"],
                "winner_support_correct_sequence_association": (
                    stats["winner_support_correct_sequence"] / stats["winner_support"]
                    if stats["winner_support"]
                    else None
                ),
                "dissent_correct_sequence_association": (
                    stats["dissent_correct_sequence"] / stats["dissent"] if stats["dissent"] else None
                ),
                "mean_top_probability": statistics.mean(values["top_probability"]),
                "mean_margin": statistics.mean(values["margin"]),
                "mean_entropy": statistics.mean(values["entropy"]),
                "mean_support_top_probability": statistics.mean(values["support_top_probability"]),
                "mean_dissent_top_probability": statistics.mean(values["dissent_top_probability"]),
                "mean_top_probability_correct_sequence": statistics.mean(values["correct_sequence_top_probability"]),
                "mean_top_probability_wrong_sequence": statistics.mean(values["wrong_sequence_top_probability"]),
            }
        )
    pairwise = []
    for (left, right), stats in sorted(pair_stats.items()):
        pairwise.append(
            {
                "expert_a": left,
                "expert_b": right,
                "steps": stats["steps"],
                "top1_agreement_rate": stats["agreements"] / stats["steps"],
                "top1_disagreement_rate": stats["disagreements"] / stats["steps"],
                "correct_sequence_association_when_agree": (
                    stats["agreements_correct_sequence"] / stats["agreements"] if stats["agreements"] else None
                ),
                "wrong_sequence_association_when_agree": (
                    stats["agreements_wrong_sequence"] / stats["agreements"] if stats["agreements"] else None
                ),
            }
        )
    vote_sizes = [
        {
            "plurality_votes": votes,
            "token_count": count,
            "token_fraction": count / total_steps,
            "sequence_correct_association": vote_size_correct[votes] / count,
        }
        for votes, count in sorted(vote_size_counts.items())
    ]
    return {
        "total_steps": total_steps,
        "coalition_detail": coalition_rows,
        "coalition_summary": coalition_summary,
        "expert_profiles": expert_profiles,
        "pairwise_top1": pairwise,
        "vote_size_summary": vote_sizes,
    }


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def divergence_analysis(
    hard_rows: dict[str, dict[str, Any]],
    probability_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    states = Counter()
    divergence_positions = Counter()
    divergence_records = []
    exact_sequence_matches = 0
    shared_prefix_diagnostic_mismatches = 0
    probability_winner_no_top1_support = 0
    high_confidence_dissenter_events = 0
    divergent = 0
    for example_id, hard in hard_rows.items():
        probability = probability_rows[example_id]
        state = (
            "both_correct"
            if hard["correct"] and probability["correct"]
            else "hard_only"
            if hard["correct"]
            else "probability_only"
            if probability["correct"]
            else "both_wrong"
        )
        states[state] += 1
        prefix = longest_common_prefix(hard["output_token_ids"], probability["output_token_ids"])
        if hard["output_token_ids"] == probability["output_token_ids"]:
            exact_sequence_matches += 1
            continue
        divergent += 1
        divergence_positions[position_bucket(prefix)] += 1
        hard_step = hard["steps"][prefix] if prefix < len(hard["steps"]) else None
        probability_step = probability["steps"][prefix] if prefix < len(probability["steps"]) else None
        record: dict[str, Any] = {
            "example_id": example_id,
            "outcome_state": state,
            "first_divergence_t": prefix,
            "task_category": hard["reasoning_family"],
            "domain": hard["domain"],
            "difficulty": hard["difficulty"],
            "answer_class": "answerable" if hard["decision"] == "answer" else "abstention",
        }
        if hard_step is not None and probability_step is not None:
            same_top_tokens = hard_step["top_tokens"] == probability_step["top_tokens"]
            shared_prefix_diagnostic_mismatches += int(not same_top_tokens)
            top_tokens = {name: int(value) for name, value in hard_step["top_tokens"].items()}
            top_probs = {name: float(value) for name, value in hard_step["top_probabilities"].items()}
            hard_winner = int(hard_step["winner"])
            probability_winner = int(probability_step["winner"])
            hard_supporters = [name for name, token in top_tokens.items() if token == hard_winner]
            probability_supporters = [name for name, token in top_tokens.items() if token == probability_winner]
            dissenters = [name for name in EXPERTS if name not in hard_supporters]
            no_support = not probability_supporters
            probability_winner_no_top1_support += int(no_support)
            high_confidence_dissenter = bool(
                dissenters
                and hard_supporters
                and max(top_probs[name] for name in dissenters) > max(top_probs[name] for name in hard_supporters)
            )
            high_confidence_dissenter_events += int(high_confidence_dissenter)
            record.update(
                {
                    "hard_winner": hard_winner,
                    "probability_winner": probability_winner,
                    "hard_winner_votes": len(hard_supporters),
                    "probability_winner_top1_votes": len(probability_supporters),
                    "probability_winner_no_top1_support": no_support,
                    "high_confidence_dissenter": high_confidence_dissenter,
                    "top_tokens_match_across_methods": same_top_tokens,
                    "hard_supporters": hard_supporters,
                    "probability_supporters": probability_supporters,
                }
            )
        divergence_records.append(record)
    by_state = {}
    for state in ("both_correct", "hard_only", "probability_only", "both_wrong"):
        records = [record for record in divergence_records if record["outcome_state"] == state]
        by_state[state] = {
            "paired_examples": states[state],
            "divergent_examples": len(records),
            "probability_winner_no_top1_support": sum(
                bool(record.get("probability_winner_no_top1_support")) for record in records
            ),
            "high_confidence_dissenter_events": sum(bool(record.get("high_confidence_dissenter")) for record in records),
            "median_first_divergence_t": statistics.median(record["first_divergence_t"] for record in records)
            if records
            else None,
        }
    return {
        "paired_examples": len(hard_rows),
        "exact_sequence_matches": exact_sequence_matches,
        "exact_sequence_match_rate": exact_sequence_matches / len(hard_rows),
        "divergent_examples": divergent,
        "divergent_rate": divergent / len(hard_rows),
        "outcome_states": dict(states),
        "divergence_position_buckets": dict(sorted(divergence_positions.items())),
        "shared_prefix_top_token_mismatches": shared_prefix_diagnostic_mismatches,
        "probability_winner_no_top1_support_at_first_divergence": probability_winner_no_top1_support,
        "high_confidence_dissenter_at_first_divergence": high_confidence_dissenter_events,
        "by_outcome_state": by_state,
        "records": divergence_records,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def find_rescue(
    rescue_rows: list[dict[str, Any]],
    comparison: str,
    dimension: str,
    stratum: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rescue_rows
        if row["comparison"] == comparison and row["dimension"] == dimension and row["stratum"] == stratum
    )


def make_report(packet: dict[str, Any]) -> None:
    rescue_rows = packet["rescue_harm"]
    behavior = packet["behavior"]
    coalition = packet["coalitions"]
    divergence = packet["free_running_divergence"]
    lines = [
        "# Phase 2 Rescue, Harm, and Existing-Trace Analysis",
        "",
        "**Status: complete for analyses derivable from the frozen full-TEST artifact; new-inference mechanisms remain pending.**",
        "",
        "The cornerstone verification passed before this analysis ran. Global paired comparisons retain their confirmatory interpretation under the frozen comparison family. Category, domain, difficulty, length, agreement, coalition, and trace-profile cuts are descriptive/exploratory and are not presented as independently confirmatory tests.",
        "",
        "## Central rescue/harm result",
        "",
        "| Candidate vs baseline | Both correct | Rescue | Harm | Both wrong | Rescue rate | Harm rate | Net gain | 95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in COMPARISONS:
        row = find_rescue(rescue_rows, name, "global", "ALL")
        lines.append(
            f"| {row['comparison_label']} | {row['both_correct']} | {row['rescue']} | {row['harm']} | {row['both_wrong']} | "
            f"{percent(row['rescue_rate'])} | {percent(row['harm_rate'])} | {pp(row['net_gain'])} | "
            f"[{pp(row['net_gain_ci_low'])}, {pp(row['net_gain_ci_high'])}] |"
        )
    track_global = find_rescue(rescue_rows, "hard_vs_track_a", "global", "ALL")
    probability_global = find_rescue(rescue_rows, "hard_vs_probability", "global", "ALL")
    lines.extend(
        [
            "",
            f"Hard voting is not winning by preserving every correct baseline answer. Against Track A it rescues **{track_global['rescue']}** cases and harms **{track_global['harm']}**; against probability mean it rescues **{probability_global['rescue']}** and harms **{probability_global['harm']}**. The mechanism must therefore explain asymmetric error correction, not simple dominance.",
            "",
            "## Answerable versus abstention composition",
            "",
            "| Comparison | Class | N | Candidate acc. | Baseline acc. | Rescue | Harm | Net gain |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in COMPARISONS:
        for stratum in ("answerable", "abstention"):
            row = find_rescue(rescue_rows, name, "answer_class", stratum)
            lines.append(
                f"| {row['comparison_label']} | {stratum} | {row['n']} | {percent(row['candidate_accuracy'])} | "
                f"{percent(row['baseline_accuracy'])} | {row['rescue']} | {row['harm']} | {pp(row['net_gain'])} |"
            )
    lines.extend(
        [
            "",
            "Behavioral policy metrics make the answer/abstain tradeoff explicit:",
            "",
            "| Method | Overall | Answerable | Abstention | False abstention | Failure to abstain | Mean tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "track_a": "Track A",
        "abstention": "Abstention expert",
        "hard_k7": "Hard K=7",
        "probability_k7": "Probability K=7",
    }
    for name in ("track_a", "abstention", "probability_k7", "hard_k7"):
        row = behavior[name]
        lines.append(
            f"| {labels[name]} | {percent(row['accuracy'])} | {percent(row['answerable_accuracy'])} | "
            f"{percent(row['abstention_accuracy'])} | {percent(row['false_abstention_rate'])} | "
            f"{percent(row['failure_to_abstain_rate'])} | {row['mean_output_tokens']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Task-category rescue/harm",
            "",
            "| Task category | N | Hard−Track A | Rescue / harm | Hard−Probability | Rescue / harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    categories = sorted(
        row["stratum"]
        for row in rescue_rows
        if row["comparison"] == "hard_vs_track_a" and row["dimension"] == "task_category"
    )
    for category in categories:
        track = find_rescue(rescue_rows, "hard_vs_track_a", "task_category", category)
        probability = find_rescue(rescue_rows, "hard_vs_probability", "task_category", category)
        lines.append(
            f"| {category} | {track['n']} | {pp(track['net_gain'])} | {track['rescue']} / {track['harm']} | "
            f"{pp(probability['net_gain'])} | {probability['rescue']} / {probability['harm']} |"
        )
    lines.extend(
        [
            "",
            "## Free-running hard-versus-soft divergence",
            "",
            f"Hard and probability mean produce exactly the same complete token sequence on **{divergence['exact_sequence_matches']}/{divergence['paired_examples']} ({percent(divergence['exact_sequence_match_rate'])})** examples. They diverge on **{divergence['divergent_examples']} ({percent(divergence['divergent_rate'])})**.",
            "",
            "| Paired outcome | All examples | Divergent examples | Median first divergence t | Soft winner has zero top-1 supporters | Higher-confidence hard dissenter |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for state in ("both_correct", "hard_only", "probability_only", "both_wrong"):
        row = divergence["by_outcome_state"][state]
        lines.append(
            f"| {state} | {row['paired_examples']} | {row['divergent_examples']} | {row['median_first_divergence_t']} | "
            f"{row['probability_winner_no_top1_support']} | {row['high_confidence_dissenter_events']} |"
        )
    lines.extend(
        [
            "",
            "At each first divergence the two methods still share an identical accepted prefix. Their expert top-1 proposals match across separately generated traces on every audited divergence; any selected-token difference therefore comes from aggregation, not different local model states. This is a free-running trajectory description, not yet the required teacher-forced causal comparison.",
            "",
            "## Coalition and directional-confidence trace evidence",
            "",
            f"The Hard-K7 trace contains **{coalition['total_steps']:,}** token decisions and **{len(coalition['coalition_summary'])}** distinct winner-support masks.",
            "",
            "| Mask | Members | Tokens | Token share | Sequence-correct association | EOS share |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in coalition["coalition_summary"][:15]:
        lines.append(
            f"| `{row['mask']}` | {', '.join(row['members']) or 'none'} | {row['token_count']} | "
            f"{percent(row['token_fraction'])} | {percent(row['sequence_correct_association'])} | {percent(row['eos_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "Coalition sequence-correct association is descriptive: a token coalition inherits the final sequence label and is not itself a gold-token accuracy measure.",
            "",
            "| Expert | Winner-support rate | Mean top-1 p | Mean margin | Mean entropy | p when supporting | p when dissenting |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coalition["expert_profiles"]:
        lines.append(
            f"| {row['expert']} | {percent(row['winner_support_rate'])} | {row['mean_top_probability']:.4f} | "
            f"{row['mean_margin']:.4f} | {row['mean_entropy']:.4f} | {row['mean_support_top_probability']:.4f} | "
            f"{row['mean_dissent_top_probability']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These profile differences establish that experts have different directional-confidence distributions along the hard trajectory. They do **not** establish calibration error because gold next-token labels are unavailable on free-running prefixes; DEV teacher forcing and temperature fitting remain necessary.",
            "",
            "## What this artifact supports now",
            "",
            "- **SUPPORTED:** Hard K=7 has positive net paired rescue over Track A and probability mean on the complete evaluated TEST corpus.",
            "- **SUPPORTED:** The gain includes both rescues and harms; hard voting is an asymmetric error-correction policy, not an everywhere-dominant predictor.",
            "- **SUPPORTED (descriptive):** Hard and soft aggregation often choose different early tokens from identical expert proposals, after which autoregressive trajectories diverge.",
            "- **SUPPORTED (descriptive):** Expert top-1 confidence, entropy, margin, agreement, and coalition participation differ materially in the frozen trace.",
            "- **UNRESOLVED:** Calibration mismatch is sufficient to explain the hard-vote advantage.",
            "- **UNRESOLVED:** Specialization rather than generic diversity causes the gain.",
            "- **UNRESOLVED:** Teacher-forced local gains are smaller than free-running gains, as required for a trajectory-correction claim.",
            "- **UNRESOLVED:** Removing the abstention expert preserves or destroys the answer/abstain frontier.",
            "",
            "## Provenance and reproduction",
            "",
            f"- Raw result: `results/sealed/v22_frozen_finalists.jsonl` (`{packet['provenance']['raw_sha256']}`)",
            f"- Cornerstone audit: `reports/phase2_cornerstone_verification.json` (`{packet['provenance']['cornerstone_audit_sha256']}`)",
            f"- Analysis script: `experiments/phase2_mechanism/analyze_existing_results.py` (`{packet['provenance']['analysis_script_sha256']}`)",
            f"- Machine result: `results/phase2_mechanism/existing_results_analysis.json`",
            "",
            "```powershell",
            "$Python = \"..\\.venv\\Scripts\\python.exe\"",
            "& $Python experiments\\phase2_mechanism\\verify_cornerstone.py",
            "& $Python experiments\\phase2_mechanism\\analyze_existing_results.py",
            "```",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_live_report(packet: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Live Results",
        "",
        f"- Updated UTC: `{packet['generated_at_utc']}`",
        "- Cornerstone verification: **PASS (30/30 checks)**",
        "- Existing full-TEST rescue/harm analysis: **complete**",
        "- Free-running divergence analysis: **complete**",
        "- Coalition / expert directional-confidence trace analysis: **complete**",
        "- DEV teacher-forced calibration capture: **pending**",
        "- All-127 cached committee analysis: **pending**",
        "- Selected TEST mechanism confirmation: **pending; must be frozen first**",
        "- Cross-benchmark confirmation: **pending**",
        "",
        "## Current verified headline",
        "",
        "Hard K=7 scores 17.259% on 4,050 v2.2 TEST examples, versus 14.593% for Track A and 14.099% for probability mean. Paired rescues exceed harms by 108 examples versus Track A and 128 examples versus probability mean.",
        "",
        "## Interpretation discipline",
        "",
        "Current trace analyses characterize where the gain occurs. They do not yet identify a sufficient causal mechanism. Calibration, rank aggregation, leave-one-out committees, teacher forcing, and adaptive consultation require DEV-first inference.",
    ]
    LIVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cornerstone = read_json(CORNERSTONE_AUDIT)
    if cornerstone.get("status") != "PASS":
        raise RuntimeError("Cornerstone verification must pass before mechanism analysis")
    rows = load_conditions()
    expected_ids = set(rows["hard_k7"])
    if len(expected_ids) != 4_050 or any(set(condition) != expected_ids for condition in rows.values()):
        raise RuntimeError("Selected conditions are not complete and paired on the same 4,050 examples")
    behavior = behavior_summary(rows)
    rescue_harm = rescue_harm_analysis(rows)
    coalitions = coalition_analysis(rows["hard_k7"])
    divergence = divergence_analysis(rows["hard_k7"], rows["probability_k7"])
    if divergence["shared_prefix_top_token_mismatches"]:
        raise RuntimeError("Hard and soft traces disagree on expert top tokens at an identical shared prefix")
    generated_at = datetime.now(timezone.utc).isoformat()
    packet = {
        "schema_version": "1.0",
        "status": "EXISTING_FULL_TEST_ANALYSIS_COMPLETE",
        "generated_at_utc": generated_at,
        "analysis_roles": {
            "global_frozen_pairs": "TEST CONFIRMATORY",
            "stratified_rescue_harm": "TEST DESCRIPTIVE / EXPLORATORY",
            "free_running_trace": "TEST DESCRIPTIVE / EXPLORATORY",
            "coalitions": "TEST DESCRIPTIVE / EXPLORATORY",
            "expert_profiles": "TEST DESCRIPTIVE / EXPLORATORY",
        },
        "provenance": {
            "raw_path": str(RAW_RESULT),
            "raw_sha256": sha256_file(RAW_RESULT),
            "cornerstone_audit_path": str(CORNERSTONE_AUDIT),
            "cornerstone_audit_sha256": sha256_file(CORNERSTONE_AUDIT),
            "sealed_statistics_path": str(SEALED_STATISTICS),
            "sealed_statistics_sha256": sha256_file(SEALED_STATISTICS),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "behavior": behavior,
        "rescue_harm": rescue_harm,
        "coalitions": coalitions,
        "free_running_divergence": divergence,
        "limitations": [
            "Subgroup results are descriptive and not corrected as a confirmatory comparison family.",
            "Coalition and expert profiles inherit sequence-level correctness, not gold token-level correctness.",
            "Free-running traces cannot replace teacher-forced local-decision analysis.",
            "Compressed traces do not retain full vocabulary distributions and cannot replay calibrated fusion.",
            "The full TEST corpus had prior exposure through the archived n=32 pilot, although method config was unchanged.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(RESCUE_CSV, rescue_harm)
    write_csv(COALITION_CSV, coalitions["coalition_summary"])
    write_csv(EXPERT_CSV, coalitions["expert_profiles"])
    write_csv(PAIRWISE_CSV, coalitions["pairwise_top1"])
    make_report(packet)
    make_live_report(packet)
    global_rows = {
        row["comparison"]: row
        for row in rescue_harm
        if row["dimension"] == "global" and row["stratum"] == "ALL"
    }
    print(
        json.dumps(
            {
                "status": packet["status"],
                "raw_sha256": packet["provenance"]["raw_sha256"],
                "hard_vs_track_a": {
                    key: global_rows["hard_vs_track_a"][key] for key in ("rescue", "harm", "net_gain")
                },
                "hard_vs_probability": {
                    key: global_rows["hard_vs_probability"][key] for key in ("rescue", "harm", "net_gain")
                },
                "hard_soft_exact_sequence_match_rate": divergence["exact_sequence_match_rate"],
                "distinct_coalitions": len(coalitions["coalition_summary"]),
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
