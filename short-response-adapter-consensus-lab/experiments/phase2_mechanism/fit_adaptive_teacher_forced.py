from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_cache_topk64.jsonl"
ANALYSIS = ROOT / "results" / "phase2_mechanism" / "adaptive_teacher_forced_dev.json"
CSV_PATH = ROOT / "results" / "phase2_mechanism" / "adaptive_teacher_forced_dev.csv"
REPORT = ROOT / "reports" / "PHASE2_ADAPTIVE_SELECTION.md"

CORE = ["track_a", "track_b", "general"]
THRESHOLDS = [0.0, 0.34, 0.67]
EXPECTED_EXAMPLES = 4050


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hard_winner(token: dict[str, Any], active_indices: list[int]) -> tuple[int, float]:
    metrics = token["expert_metrics"]
    proposals = [int(metrics[index]["top1_token_id"]) for index in active_indices]
    counts = Counter(proposals)
    winning_count = max(counts.values())
    tied = sorted(token_id for token_id, count in counts.items() if count == winning_count)
    if len(tied) == 1:
        winner = tied[0]
    else:
        candidate_ids = [int(value) for value in token["candidate_token_ids"]]
        candidate_index = {token_id: index for index, token_id in enumerate(candidate_ids)}
        logits = token["candidate_logits_by_expert"]
        scores = {
            token_id: sum(float(logits[expert_index][candidate_index[token_id]]) for expert_index in active_indices)
            for token_id in tied
        }
        best_score = max(scores.values())
        winner = min(token_id for token_id in tied if scores[token_id] == best_score)
    return winner, winning_count / len(active_indices)


def main() -> None:
    counters = {
        threshold: {
            "token_total": 0,
            "token_correct": 0,
            "example_total": 0,
            "example_correct": 0,
            "escalated_tokens": 0,
            "winner_changed_on_escalation": 0,
            "full_correct_core_wrong_not_escalated": 0,
            "core_correct_full_wrong_on_escalation": 0,
            "forward_proxy": 0,
        }
        for threshold in THRESHOLDS
    }
    core_anchor = {"token_total": 0, "token_correct": 0, "example_total": 0, "example_correct": 0}
    full_anchor = {"token_total": 0, "token_correct": 0, "example_total": 0, "example_correct": 0}
    example_count = 0
    capture_hashes: set[str] = set()

    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_count += 1
            capture_hashes.add(row["capture_config_sha256"])
            order = row["expert_order"]
            core_indices = [order.index(name) for name in CORE]
            full_indices = list(range(len(order)))
            core_all_correct = True
            full_all_correct = True
            adaptive_all_correct = {threshold: True for threshold in THRESHOLDS}
            for token in row["tokens"]:
                gold = int(token["gold_token_id"])
                core_winner, agreement = hard_winner(token, core_indices)
                full_winner, _ = hard_winner(token, full_indices)
                core_correct = core_winner == gold
                full_correct = full_winner == gold
                core_anchor["token_total"] += 1
                core_anchor["token_correct"] += int(core_correct)
                full_anchor["token_total"] += 1
                full_anchor["token_correct"] += int(full_correct)
                core_all_correct &= core_correct
                full_all_correct &= full_correct
                for threshold in THRESHOLDS:
                    target = counters[threshold]
                    escalated = agreement < threshold
                    winner = full_winner if escalated else core_winner
                    correct = winner == gold
                    target["token_total"] += 1
                    target["token_correct"] += int(correct)
                    target["escalated_tokens"] += int(escalated)
                    target["forward_proxy"] += 7 if escalated else 3
                    target["winner_changed_on_escalation"] += int(escalated and core_winner != full_winner)
                    target["full_correct_core_wrong_not_escalated"] += int(
                        not escalated and full_correct and not core_correct
                    )
                    target["core_correct_full_wrong_on_escalation"] += int(
                        escalated and core_correct and not full_correct
                    )
                    adaptive_all_correct[threshold] &= correct
            core_anchor["example_total"] += 1
            core_anchor["example_correct"] += int(core_all_correct)
            full_anchor["example_total"] += 1
            full_anchor["example_correct"] += int(full_all_correct)
            for threshold in THRESHOLDS:
                counters[threshold]["example_total"] += 1
                counters[threshold]["example_correct"] += int(adaptive_all_correct[threshold])

    if example_count != EXPECTED_EXAMPLES or len(capture_hashes) != 1:
        raise RuntimeError(
            f"Cache gate failed: examples={example_count}, capture_hashes={sorted(capture_hashes)}"
        )

    full_token_accuracy = full_anchor["token_correct"] / full_anchor["token_total"]
    full_example_accuracy = full_anchor["example_correct"] / full_anchor["example_total"]
    rows = []
    for threshold in THRESHOLDS:
        source = counters[threshold]
        rows.append(
            {
                "threshold": threshold,
                "core": CORE,
                "token_accuracy": source["token_correct"] / source["token_total"],
                "example_accuracy": source["example_correct"] / source["example_total"],
                "token_accuracy_retention_vs_full": source["token_correct"] / source["token_total"] - full_token_accuracy,
                "example_accuracy_retention_vs_full": source["example_correct"] / source["example_total"] - full_example_accuracy,
                "escalation_rate": source["escalated_tokens"] / source["token_total"],
                "mean_expert_token_forwards": source["forward_proxy"] / source["token_total"],
                "forward_reduction_vs_full": 1 - source["forward_proxy"] / (7 * source["token_total"]),
                "winner_change_rate_given_escalation": source["winner_changed_on_escalation"] / source["escalated_tokens"] if source["escalated_tokens"] else 0.0,
                "false_non_escalation_rate": source["full_correct_core_wrong_not_escalated"] / source["token_total"],
                "core_correct_full_wrong_on_escalation_rate": source["core_correct_full_wrong_on_escalation"] / source["token_total"],
                **source,
            }
        )

    eligible = [
        row
        for row in rows
        if row["token_accuracy_retention_vs_full"] >= -0.005
        and row["example_accuracy_retention_vs_full"] >= -0.01
    ]
    selection_pool = eligible or rows
    selected = min(
        selection_pool,
        key=lambda row: (
            row["mean_expert_token_forwards"],
            -row["example_accuracy"],
            -row["token_accuracy"],
            row["threshold"],
        ),
    )

    analysis = {
        "schema_version": "1.0",
        "status": "DEV_ADAPTIVE_THRESHOLD_SELECTED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MODEL-SELECTION / CALIBRATION",
        "cache": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "capture_config_sha256": next(iter(capture_hashes)),
        "example_count": example_count,
        "token_count": full_anchor["token_total"],
        "core": CORE,
        "threshold_semantics": "escalate from K3 to K7 when core plurality agreement is strictly below threshold",
        "selection_rule": "minimum mean forwards among policies no worse than full hard by 0.5 token pp and 1.0 all-token-example pp; otherwise minimum-forward lexicographic fallback",
        "threshold_grid": THRESHOLDS,
        "anchors": {
            "core_k3": {
                **core_anchor,
                "token_accuracy": core_anchor["token_correct"] / core_anchor["token_total"],
                "example_accuracy": core_anchor["example_correct"] / core_anchor["example_total"],
                "mean_expert_token_forwards": 3.0,
            },
            "full_k7": {
                **full_anchor,
                "token_accuracy": full_token_accuracy,
                "example_accuracy": full_example_accuracy,
                "mean_expert_token_forwards": 7.0,
            },
        },
        "policies": rows,
        "selected_threshold": selected["threshold"],
        "selected_policy": selected,
        "eligible_policy_count": len(eligible),
        "limitations": [
            "Selection uses teacher-forced DEV gold prefixes, not free-running trajectories.",
            "For a K3 core, agreement has only three levels; the threshold grid represents the distinct escalation policies.",
            "Latency must be measured under free-running reference decoding.",
            "No TEST statistic is used for selection.",
        ],
    }
    ANALYSIS.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "core"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "core"})

    report_rows = []
    for row in rows:
        selected_mark = " **SELECTED**" if row["threshold"] == selected["threshold"] else ""
        report_rows.append(
            f'| {row["threshold"]:.2f}{selected_mark} | {100*row["token_accuracy"]:.2f}% | '
            f'{100*row["example_accuracy"]:.2f}% | {100*row["escalation_rate"]:.2f}% | '
            f'{row["mean_expert_token_forwards"]:.2f} | {100*row["forward_reduction_vs_full"]:.2f}% | '
            f'{100*row["false_non_escalation_rate"]:.2f}% |'
        )
    report = "\n".join(
        [
            "# Phase 2 Adaptive Consultation Selection",
            "",
            "Role: **DEV MODEL-SELECTION / CALIBRATION**. The predeclared core is Track A + Track B + General. "
            "The policy escalates to all seven experts when K3 agreement is strictly below the threshold.",
            "",
            "| Threshold | Token acc. | All-token example acc. | Escalation | Mean forwards/token | Forward reduction | False non-escalation |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *report_rows,
            "",
            f'Full K7 anchor: {100*full_token_accuracy:.2f}% token, {100*full_example_accuracy:.2f}% all-token example, 7.00 forwards/token.',
            f'Core K3 anchor: {100*analysis["anchors"]["core_k3"]["token_accuracy"]:.2f}% token, {100*analysis["anchors"]["core_k3"]["example_accuracy"]:.2f}% all-token example, 3.00 forwards/token.',
            "",
            f'Selected threshold: **{selected["threshold"]:.2f}** under the frozen rule: {analysis["selection_rule"]}.',
            "",
            "This cached study chooses a DEV policy; it does not establish free-running accuracy retention or measured latency. "
            "Those require a separately frozen reference-decoder run.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "selected_threshold": selected["threshold"], "selected_policy": selected}, indent=2))


if __name__ == "__main__":
    main()
