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
CACHE = ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
TEACHER = ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
SELECTION = ROOT / "artifacts/phase2_mechanism/selected_dev_free_run_manifest.json"
OUTPUT = ROOT / "results/phase2_mechanism/hard_temperature_tie_sensitivity_dev.json"
CSV_PATH = ROOT / "results/phase2_mechanism/hard_temperature_tie_sensitivity_dev.csv"
REPORT = ROOT / "reports/PHASE2_HARD_TEMPERATURE_TIE_SENSITIVITY.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_TOKENS = 18760


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def winner(token: dict[str, Any], temperatures: list[float]) -> tuple[int, bool]:
    proposals = [int(item["top1_token_id"]) for item in token["expert_metrics"]]
    counts = Counter(proposals)
    maximum = max(counts.values())
    tied = sorted(token_id for token_id, count in counts.items() if count == maximum)
    if len(tied) == 1:
        return tied[0], False
    candidate_ids = [int(value) for value in token["candidate_token_ids"]]
    candidate_index = {token_id: index for index, token_id in enumerate(candidate_ids)}
    logits = token["candidate_logits_by_expert"]
    scores = {
        token_id: sum(
            float(expert_logits[candidate_index[token_id]]) / temperature
            for expert_logits, temperature in zip(logits, temperatures, strict=True)
        )
        for token_id in tied
    }
    best = max(scores.values())
    finalists = sorted(
        token_id
        for token_id, score in scores.items()
        if math.isclose(score, best, rel_tol=1e-8, abs_tol=1e-8)
    )
    return finalists[0], True


def main() -> None:
    teacher = json.loads(TEACHER.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    expert_order: list[str] | None = None
    grid: list[float] | None = None
    condition_temperatures: dict[str, list[float]] = {}
    token_records: list[tuple[dict[str, Any], int, int]] = []
    example_count = 0
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_count += 1
            if expert_order is None:
                expert_order = list(row["expert_order"])
                grid = [float(value) for value in row["temperature_grid"]]
                condition_temperatures["baseline_raw"] = [1.0] * len(expert_order)
                for temperature in grid:
                    condition_temperatures[f"shared_T={temperature:g}"] = [temperature] * len(expert_order)
                for expert_index, expert in enumerate(expert_order):
                    for temperature in grid:
                        values = [1.0] * len(expert_order)
                        values[expert_index] = temperature
                        condition_temperatures[f"one:{expert}:T={temperature:g}"] = values
                condition_temperatures["all:fitted_per_expert"] = [
                    float(selection["fitted_temperatures"][expert]) for expert in expert_order
                ]
            elif list(row["expert_order"]) != expert_order or [float(value) for value in row["temperature_grid"]] != grid:
                raise RuntimeError("Frozen cache expert order or temperature grid changed")
            for token in row["tokens"]:
                token_records.append((token, int(token["gold_token_id"]), example_count - 1))
    if example_count != EXPECTED_EXAMPLES or len(token_records) != EXPECTED_TOKENS:
        raise RuntimeError(
            f"Unexpected cache size: {example_count} examples / {len(token_records)} tokens"
        )
    assert expert_order is not None and grid is not None

    baseline_winners: list[int] = []
    baseline_correct: list[bool] = []
    tied_count = 0
    for token, gold, _ in token_records:
        selected, tied = winner(token, condition_temperatures["baseline_raw"])
        baseline_winners.append(selected)
        baseline_correct.append(selected == gold)
        tied_count += int(tied)

    rows: list[dict[str, Any]] = []
    condition_example_vectors: dict[str, list[bool]] = {}
    for condition, temperatures in condition_temperatures.items():
        token_correct = 0
        changed_tokens = 0
        rescues = 0
        harms = 0
        example_ok = [True] * EXPECTED_EXAMPLES
        for token_index, (token, gold, example_index) in enumerate(token_records):
            selected, _ = winner(token, temperatures)
            correct = selected == gold
            baseline_ok = baseline_correct[token_index]
            token_correct += int(correct)
            changed_tokens += int(selected != baseline_winners[token_index])
            rescues += int(correct and not baseline_ok)
            harms += int(not correct and baseline_ok)
            example_ok[example_index] &= correct
        condition_example_vectors[condition] = example_ok
        if condition == "baseline_raw":
            scope, expert, temperature = "baseline", "all", 1.0
        elif condition == "all:fitted_per_expert":
            scope, expert, temperature = "all fitted", "all", None
        elif condition.startswith("shared_T="):
            scope, expert, temperature = "shared", "all", float(condition.split("=")[1])
        else:
            _, expert, raw_temperature = condition.split(":")
            scope, temperature = "one expert", float(raw_temperature.split("=")[1])
        rows.append(
            {
                "condition": condition,
                "scope": scope,
                "perturbed_expert": expert,
                "temperature": temperature,
                "token_correct": token_correct,
                "token_total": EXPECTED_TOKENS,
                "token_accuracy": token_correct / EXPECTED_TOKENS,
                "example_correct": sum(example_ok),
                "example_total": EXPECTED_EXAMPLES,
                "example_accuracy": sum(example_ok) / EXPECTED_EXAMPLES,
                "changed_tokens_vs_raw_tie_break": changed_tokens,
                "change_rate_all_tokens": changed_tokens / EXPECTED_TOKENS,
                "change_rate_tied_tokens": changed_tokens / tied_count,
                "rescues_vs_raw_tie_break": rescues,
                "harms_vs_raw_tie_break": harms,
            }
        )

    baseline = rows[0]
    expected = next(
        row
        for row in teacher["committee_results"]
        if int(row["mask"]) == 127 and row["method"] == "hard_plurality"
    )
    if baseline["token_correct"] != expected["token_correct"] or baseline["example_correct"] != expected["example_correct"]:
        raise RuntimeError("Raw hard tie-break reconstruction failed to reproduce the frozen anchor")
    if tied_count != 1097:
        raise RuntimeError(f"Expected 1,097 plurality-tied tokens, observed {tied_count}")
    shared = [row for row in rows if row["scope"] == "shared"]
    if any(row["changed_tokens_vs_raw_tie_break"] for row in shared):
        raise RuntimeError("A shared positive temperature unexpectedly changed a hard tie decision")
    one_at_one = [
        row for row in rows if row["scope"] == "one expert" and row["temperature"] == 1.0
    ]
    if any(row["changed_tokens_vs_raw_tie_break"] for row in one_at_one):
        raise RuntimeError("T=1 one-expert controls unexpectedly changed hard decisions")

    changed_rows = [row for row in rows if row["scope"] == "one expert" and row["changed_tokens_vs_raw_tie_break"]]
    maximum_change = max(changed_rows, key=lambda row: row["changed_tokens_vs_raw_tie_break"])
    fitted = next(row for row in rows if row["condition"] == "all:fitted_per_expert")
    packet = {
        "schema_version": 1,
        "status": "DEV_HARD_TEMPERATURE_TIE_SENSITIVITY_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV GOLD-PREFIX IMPLEMENTATION-BOUNDARY COUNTERFACTUAL / EXPLORATORY",
        "example_count": EXPECTED_EXAMPLES,
        "token_count": EXPECTED_TOKENS,
        "plurality_tied_token_count": tied_count,
        "plurality_tied_token_rate": tied_count / EXPECTED_TOKENS,
        "expert_order": expert_order,
        "temperature_grid": grid,
        "fitted_temperatures": selection["fitted_temperatures"],
        "baseline_reproduced": True,
        "shared_temperature_invariant": True,
        "one_expert_temperature_can_change_tie_decision": bool(changed_rows),
        "fitted_per_expert_counterfactual": fitted,
        "maximum_one_expert_change": maximum_change,
        "rows": rows,
        "provenance": {
            "cache_sha256": sha256(CACHE),
            "teacher_analysis_sha256": sha256(TEACHER),
            "selection_manifest_sha256": sha256(SELECTION),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "interpretation": (
            "Positive temperature preserves every expert argmax and therefore every vote count. "
            "The selected token is invariant when plurality is unique. The deployed confidence-sum "
            "tie-break is invariant to a shared temperature but can move when experts receive different "
            "temperatures; this is a tie-boundary qualification, not a failure of argmax invariance."
        ),
        "limitations": [
            "Gold-prefix local counterfactuals do not measure free-running sequence effects.",
            "Only the frozen positive-temperature grid and fitted temperature vector are evaluated.",
            "Changes are possible only at the 1,097 plurality-tied token positions.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Hard-vote temperature invariance at the tie boundary",
        "",
        "Status: **DEV_HARD_TEMPERATURE_TIE_SENSITIVITY_COMPLETE**",
        "",
        "Positive temperature scaling preserves each expert's top-1 proposal and all plurality vote counts. The deployed selected token is therefore invariant whenever the plurality winner is unique. Its confidence-sum tie-break is a narrower exception.",
        "",
        f"- Plurality-tied positions: **{tied_count:,}/{EXPECTED_TOKENS:,} ({100*tied_count/EXPECTED_TOKENS:.2f}%)**.",
        "- Every shared-temperature intervention leaves every hard decision unchanged.",
        f"- The fitted per-expert temperature vector changes **{fitted['changed_tokens_vs_raw_tie_break']:,}** local decisions, with token/example accuracy {100*fitted['token_accuracy']:.2f}%/{100*fitted['example_accuracy']:.2f}% versus raw {100*baseline['token_accuracy']:.2f}%/{100*baseline['example_accuracy']:.2f}%.",
        f"- Largest single-expert grid effect: **{maximum_change['perturbed_expert']} at T={maximum_change['temperature']:g}**, changing {maximum_change['changed_tokens_vs_raw_tie_break']:,} token decisions ({100*maximum_change['change_rate_tied_tokens']:.2f}% of tied positions).",
        "",
        "This qualifies the invariance claim: votes and unique-plurality winners are exactly scale-invariant, while confidence-based resolution among tied vote leaders is not invariant to expert-specific scaling. The counterfactual is DEV-local and does not establish a free-running accuracy effect.",
        "",
        f"Frozen cache SHA-256: `{packet['provenance']['cache_sha256']}`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "fitted": fitted, "maximum": maximum_change}, indent=2))


if __name__ == "__main__":
    main()
