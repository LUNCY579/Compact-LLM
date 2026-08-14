from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
TEACHER = ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json"
OUTPUT = ROOT / "results/phase2_mechanism/confidence_conditioned_correctness_dev.json"
CSV = ROOT / "results/phase2_mechanism/confidence_conditioned_correctness_dev.csv"
REPORT = ROOT / "reports/PHASE2_CONFIDENCE_CONDITIONED_DEV.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_TOKENS = 18760


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    teacher = json.loads(TEACHER.read_text(encoding="utf-8-sig"))
    if teacher.get("example_count") != EXPECTED_EXAMPLES or teacher.get("token_count") != EXPECTED_TOKENS:
        raise RuntimeError("Frozen teacher analysis is not complete")

    expert_order: list[str] | None = None
    confidence: dict[str, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    entropy: dict[str, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    margin: dict[str, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    examples = 0
    tokens = 0
    for row in iter_jsonl(CACHE):
        examples += 1
        if expert_order is None:
            expert_order = list(row["expert_order"])
        elif list(row["expert_order"]) != expert_order:
            raise RuntimeError(f"Expert order changed at {row.get('example_id')}")
        for token in row["tokens"]:
            tokens += 1
            gold = int(token["gold_token_id"])
            for expert, metrics in zip(expert_order, token["expert_metrics"], strict=True):
                correct = int(metrics["top1_token_id"]) == gold
                confidence[expert][correct].append(float(metrics["top1_probability"]))
                entropy[expert][correct].append(float(metrics["entropy"]))
                margin[expert][correct].append(
                    float(metrics["top1_probability"]) - float(metrics["top2_probability"])
                )
    if examples != EXPECTED_EXAMPLES or tokens != EXPECTED_TOKENS or expert_order is None:
        raise RuntimeError(f"Unexpected cache extent: examples={examples}, tokens={tokens}")

    rows = []
    for expert in expert_order:
        correct_confidence = confidence[expert][True]
        incorrect_confidence = confidence[expert][False]
        if not correct_confidence or not incorrect_confidence:
            raise RuntimeError(f"Both correctness strata are required for {expert}")
        rows.append(
            {
                "expert": expert,
                "token_count": len(correct_confidence) + len(incorrect_confidence),
                "correct_token_count": len(correct_confidence),
                "incorrect_token_count": len(incorrect_confidence),
                "mean_confidence_given_correct": sum(correct_confidence) / len(correct_confidence),
                "mean_confidence_given_incorrect": sum(incorrect_confidence) / len(incorrect_confidence),
                "correct_minus_incorrect_confidence": (
                    sum(correct_confidence) / len(correct_confidence)
                    - sum(incorrect_confidence) / len(incorrect_confidence)
                ),
                "mean_entropy_given_correct": sum(entropy[expert][True]) / len(entropy[expert][True]),
                "mean_entropy_given_incorrect": sum(entropy[expert][False]) / len(entropy[expert][False]),
                "mean_margin_given_correct": sum(margin[expert][True]) / len(margin[expert][True]),
                "mean_margin_given_incorrect": sum(margin[expert][False]) / len(margin[expert][False]),
                "analysis_role": "DEV MECHANISTIC / DESCRIPTIVE CALIBRATION",
            }
        )

    packet = {
        "schema_version": "1.0",
        "status": "DEV_CONFIDENCE_CONDITIONING_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV MECHANISTIC / DESCRIPTIVE CALIBRATION",
        "example_count": examples,
        "token_count": tokens,
        "expert_count": len(rows),
        "expert_summaries": rows,
        "provenance": {
            "cache_path": str(CACHE.relative_to(ROOT)).replace("\\", "/"),
            "cache_sha256": sha256(CACHE),
            "teacher_analysis_path": str(TEACHER.relative_to(ROOT)).replace("\\", "/"),
            "teacher_analysis_sha256": sha256(TEACHER),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "cache_reused_without_new_model_forwards": True,
        },
        "limitations": [
            "Conditioning on observed correctness is descriptive and not a calibration intervention.",
            "Token decisions share examples and are not independent sequence-level observations.",
            "All values use DEV gold prefixes rather than free-running prefixes.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report_rows = [
        f'| {row["expert"]} | {row["correct_token_count"]} | {row["incorrect_token_count"]} | '
        f'{100*row["mean_confidence_given_correct"]:.2f}% | '
        f'{100*row["mean_confidence_given_incorrect"]:.2f}% | '
        f'{100*row["correct_minus_incorrect_confidence"]:+.2f} pp |'
        for row in rows
    ]
    REPORT.write_text(
        "\n".join(
            [
                "# DEV confidence conditioned on expert token correctness",
                "",
                "Status: **DEV_CONFIDENCE_CONDITIONING_COMPLETE**",
                "",
                "This cache-only supplement reports each expert's top-1 confidence separately for correct and incorrect gold-prefix token decisions. It does not change the frozen calibration fit or selected runs.",
                "",
                "| Expert | Correct n | Incorrect n | Confidence if correct | Confidence if incorrect | Difference |",
                "|---|---:|---:|---:|---:|---:|",
                *report_rows,
                "",
                f"Cache SHA-256: `{packet['provenance']['cache_sha256']}`.",
                "",
                "Correctness-conditioned values are descriptive DEV token statistics, not independent sequence-level effects.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": packet["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
