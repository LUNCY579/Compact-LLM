from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_cache_topk64.jsonl"
TEACHER = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_analysis.json"
SELECTION = ROOT / "artifacts" / "phase2_mechanism" / "selected_dev_free_run_manifest.json"
OUTPUT = ROOT / "results" / "phase2_mechanism" / "tie_eos_counterfactuals_dev.json"
CSV_PATH = ROOT / "results" / "phase2_mechanism" / "tie_eos_counterfactuals_dev.csv"
REPORT = ROOT / "reports" / "PHASE2_TIE_EOS_COUNTERFACTUALS.md"

EOS = 151645
EXPECTED_EXAMPLES = 4050
TIE_POLICIES = ["confidence_sum", "smallest_token_id", "track_a_priority"]
EOS_POLICIES = ["ordinary", "strict_majority", "confidence_threshold_0.5", "minimum_tokens_3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expert_proposals(token: dict[str, Any], *, exclude_eos: bool) -> list[int]:
    proposals = []
    for metrics in token["expert_metrics"]:
        if not exclude_eos:
            proposals.append(int(metrics["top1_token_id"]))
            continue
        replacement = next(int(token_id) for token_id in metrics["topk_token_ids"] if int(token_id) != EOS)
        proposals.append(replacement)
    return proposals


def hard_winner(token: dict[str, Any], *, tie_policy: str, exclude_eos: bool = False) -> tuple[int, dict[str, Any]]:
    proposals = expert_proposals(token, exclude_eos=exclude_eos)
    counts = Counter(proposals)
    winning_votes = max(counts.values())
    tied = sorted(token_id for token_id, count in counts.items() if count == winning_votes)
    if len(tied) == 1:
        winner = tied[0]
    elif tie_policy == "smallest_token_id":
        winner = tied[0]
    elif tie_policy == "track_a_priority" and proposals[0] in tied:
        winner = proposals[0]
    else:
        candidate_ids = [int(value) for value in token["candidate_token_ids"]]
        candidate_index = {token_id: index for index, token_id in enumerate(candidate_ids)}
        logits = token["candidate_logits_by_expert"]
        scores = {
            token_id: sum(float(expert_logits[candidate_index[token_id]]) for expert_logits in logits)
            for token_id in tied
        }
        best = max(scores.values())
        winner = min(token_id for token_id in tied if scores[token_id] == best)
    return winner, {
        "proposals": proposals,
        "winning_votes": winning_votes,
        "tied": tied,
        "eos_vote_fraction": counts[EOS] / len(proposals),
    }


def mean_eos_probability(
    row: dict[str, Any], token: dict[str, Any], temperatures: dict[str, float]
) -> float:
    candidate_ids = [int(value) for value in token["candidate_token_ids"]]
    eos_index = candidate_ids.index(EOS)
    values = []
    for expert_index, expert in enumerate(row["expert_order"]):
        temperature = float(temperatures[expert])
        temperature_index = row["temperature_grid"].index(temperature)
        eos_logit = float(token["candidate_logits_by_expert"][expert_index][eos_index])
        logsumexp = float(token["expert_metrics"][expert_index]["logsumexp_by_temperature"][temperature_index])
        values.append(math.exp(eos_logit / temperature - logsumexp))
    return sum(values) / len(values)


def main() -> None:
    teacher = load_json(TEACHER)
    selection = load_json(SELECTION)
    temperatures = selection["fitted_temperatures"]
    tie_state = {
        policy: {"token_total": 0, "token_correct": 0, "example_total": 0, "example_correct": 0, "affected_tokens": 0, "rescues_vs_current": 0, "harms_vs_current": 0}
        for policy in TIE_POLICIES
    }
    eos_state = {
        policy: {"token_total": 0, "token_correct": 0, "example_total": 0, "example_correct": 0, "overrides": 0, "rescues_vs_ordinary": 0, "harms_vs_ordinary": 0}
        for policy in EOS_POLICIES
    }
    tied_tokens = 0
    eos_winner_tokens = 0
    example_count = 0

    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_count += 1
            tie_example_correct = {policy: True for policy in TIE_POLICIES}
            eos_example_correct = {policy: True for policy in EOS_POLICIES}
            for token in row["tokens"]:
                gold = int(token["gold_token_id"])
                current, current_diag = hard_winner(token, tie_policy="confidence_sum")
                tied = len(current_diag["tied"]) > 1
                tied_tokens += int(tied)
                for policy in TIE_POLICIES:
                    winner, _ = hard_winner(token, tie_policy=policy)
                    correct = winner == gold
                    state = tie_state[policy]
                    state["token_total"] += 1
                    state["token_correct"] += int(correct)
                    state["affected_tokens"] += int(winner != current)
                    state["rescues_vs_current"] += int(winner == gold and current != gold)
                    state["harms_vs_current"] += int(winner != gold and current == gold)
                    tie_example_correct[policy] &= correct

                eos_winner = current == EOS
                eos_winner_tokens += int(eos_winner)
                for policy in EOS_POLICIES:
                    winner = current
                    override = False
                    if eos_winner and policy != "ordinary":
                        if policy == "strict_majority":
                            allow = current_diag["eos_vote_fraction"] > 0.5
                        elif policy == "confidence_threshold_0.5":
                            allow = mean_eos_probability(row, token, temperatures) >= 0.5
                        elif policy == "minimum_tokens_3":
                            allow = int(token["t"]) >= 3
                        else:
                            raise AssertionError(policy)
                        if not allow:
                            winner, _ = hard_winner(token, tie_policy="confidence_sum", exclude_eos=True)
                            override = True
                    correct = winner == gold
                    state = eos_state[policy]
                    state["token_total"] += 1
                    state["token_correct"] += int(correct)
                    state["overrides"] += int(override)
                    state["rescues_vs_ordinary"] += int(winner == gold and current != gold)
                    state["harms_vs_ordinary"] += int(winner != gold and current == gold)
                    eos_example_correct[policy] &= correct
            for policy in TIE_POLICIES:
                tie_state[policy]["example_total"] += 1
                tie_state[policy]["example_correct"] += int(tie_example_correct[policy])
            for policy in EOS_POLICIES:
                eos_state[policy]["example_total"] += 1
                eos_state[policy]["example_correct"] += int(eos_example_correct[policy])

    if example_count != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXAMPLES} examples, found {example_count}")
    expected_hard = next(
        row for row in teacher["committee_results"] if row["mask"] == 127 and row["method"] == "hard_plurality"
    )
    current_token_accuracy = tie_state["confidence_sum"]["token_correct"] / tie_state["confidence_sum"]["token_total"]
    current_example_accuracy = tie_state["confidence_sum"]["example_correct"] / tie_state["confidence_sum"]["example_total"]
    if current_token_accuracy != expected_hard["token_accuracy"] or current_example_accuracy != expected_hard["example_accuracy"]:
        raise RuntimeError("Counterfactual reconstruction does not reproduce the frozen hard K7 anchor")

    rows = []
    for family, states in (("tie", tie_state), ("eos", eos_state)):
        for policy, state in states.items():
            rows.append(
                {
                    "family": family,
                    "policy": policy,
                    "token_accuracy": state["token_correct"] / state["token_total"],
                    "example_accuracy": state["example_correct"] / state["example_total"],
                    "token_delta_vs_current": state["token_correct"] / state["token_total"] - current_token_accuracy,
                    "example_delta_vs_current": state["example_correct"] / state["example_total"] - current_example_accuracy,
                    **state,
                }
            )
    analysis = {
        "schema_version": "1.0",
        "status": "DEV_TIE_EOS_COUNTERFACTUALS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV GOLD-PREFIX COUNTERFACTUAL / EXPLORATORY",
        "example_count": example_count,
        "token_count": tie_state["confidence_sum"]["token_total"],
        "tied_token_count": tied_tokens,
        "tied_token_rate": tied_tokens / tie_state["confidence_sum"]["token_total"],
        "ordinary_eos_winner_count": eos_winner_tokens,
        "current_anchor": {"token_accuracy": current_token_accuracy, "example_accuracy": current_example_accuracy},
        "tie_policies": tie_state,
        "eos_policies": eos_state,
        "rows": rows,
        "provenance": {
            "cache": str(CACHE),
            "cache_sha256": sha256(CACHE),
            "teacher_analysis": str(TEACHER),
            "teacher_analysis_sha256": sha256(TEACHER),
            "selection_manifest": str(SELECTION),
            "selection_manifest_sha256": sha256(SELECTION),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "Counterfactuals are local decisions on gold prefixes, not free-running sequence outcomes.",
            "Tie alternatives change only the deterministic resolution among plurality-tied tokens.",
            "EOS alternatives are applied only when ordinary hard plurality selects EOS; replacement masks EOS and recomputes expert proposals.",
            "Temperature-specific EOS probabilities use the frozen fitted DEV temperatures and captured exact log-sum-exp grid.",
        ],
    }
    OUTPUT.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tie_lines = []
    for policy in TIE_POLICIES:
        state = tie_state[policy]
        tie_lines.append(
            f'| {policy} | {100*state["token_correct"]/state["token_total"]:.2f}% | '
            f'{100*state["example_correct"]/state["example_total"]:.2f}% | {state["affected_tokens"]} | '
            f'{state["rescues_vs_current"]}/{state["harms_vs_current"]} |'
        )
    eos_lines = []
    for policy in EOS_POLICIES:
        state = eos_state[policy]
        eos_lines.append(
            f'| {policy} | {100*state["token_correct"]/state["token_total"]:.2f}% | '
            f'{100*state["example_correct"]/state["example_total"]:.2f}% | {state["overrides"]} | '
            f'{state["rescues_vs_ordinary"]}/{state["harms_vs_ordinary"]} |'
        )
    report = "\n".join(
        [
            "# Phase 2 Tie and EOS Counterfactuals",
            "",
            "Role: **DEV GOLD-PREFIX COUNTERFACTUAL / EXPLORATORY**. The current confidence tie-break exactly reproduces the frozen hard K=7 local anchor.",
            "",
            f'Tied decisions: {tied_tokens:,}/{analysis["token_count"]:,} token positions ({100*analysis["tied_token_rate"]:.2f}%).',
            "",
            "## Tie policy",
            "",
            "| Policy | Token acc. | All-token example acc. | Changed tokens | Rescue/harm vs current |",
            "|---|---:|---:|---:|---:|",
            *tie_lines,
            "",
            "## EOS policy",
            "",
            "| Policy | Token acc. | All-token example acc. | EOS overrides | Rescue/harm vs ordinary |",
            "|---|---:|---:|---:|---:|",
            *eos_lines,
            "",
            "These counterfactuals isolate local decision effects. They cannot establish sequence-level gains because an EOS override changes every later prefix under free running.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
