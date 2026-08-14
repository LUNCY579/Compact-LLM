from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/sealed/v22_frozen_finalists.jsonl"
OUTPUT = ROOT / "results/phase2_mechanism/coalition_strata_full_test.json"
CSV = ROOT / "results/phase2_mechanism/coalition_strata_full_test.csv"
REPORT = ROOT / "reports/PHASE2_COALITION_STRATA.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_RAW_SHA256 = "d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d"
EXPERTS = ["track_a", "track_b", "general", "grounding", "procedure", "abstention", "social"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def position_bucket(value: int) -> str:
    if value == 0:
        return "t=0"
    if value <= 2:
        return "t=1-2"
    if value <= 5:
        return "t=3-5"
    if value <= 10:
        return "t=6-10"
    return "t>10"


def mask(members: list[str]) -> str:
    selected = set(members)
    return "".join("1" if expert in selected else "0" for expert in EXPERTS)


def main() -> None:
    if sha256(RAW) != EXPECTED_RAW_SHA256:
        raise RuntimeError("Cornerstone hash differs from the verified raw artifact")
    counts: Counter[tuple[str, str, str, bool, bool]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    coalition_members: dict[str, list[str]] = {}
    examples = 0
    tokens = 0
    for row in iter_jsonl(RAW):
        if row["committee"] != "full_k7" or row["method"] != "hard_plurality":
            continue
        examples += 1
        dimensions = {
            "global": "ALL",
            "reasoning_family": str(row["reasoning_family"]),
            "difficulty": str(row["difficulty"]),
            "answer_class": "answerable" if row["decision"] == "answer" else "abstention",
            "domain": str(row["domain"]),
        }
        for step in row["steps"]:
            tokens += 1
            step_dimensions = {
                **dimensions,
                "position_bucket": position_bucket(int(step["t"])),
            }
            coalition = list(step.get("winning_expert_coalition") or [])
            coalition_mask = mask(coalition)
            coalition_members[coalition_mask] = coalition
            is_eos = bool(step.get("winner_is_eos"))
            sequence_correct = bool(row["correct"])
            for dimension, value in step_dimensions.items():
                counts[(dimension, value, coalition_mask, is_eos, sequence_correct)] += 1
                totals[(dimension, value)] += 1
    if examples != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXAMPLES} full-K7 hard examples, observed {examples}")

    rows = []
    aggregate: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for (dimension, value, coalition_mask, is_eos, sequence_correct), count in counts.items():
        item = aggregate[(dimension, value, coalition_mask)]
        item["tokens"] += count
        item["correct_sequence_tokens"] += count * int(sequence_correct)
        item["wrong_sequence_tokens"] += count * int(not sequence_correct)
        item["eos_tokens"] += count * int(is_eos)
        item["successful_eos_tokens"] += count * int(is_eos and sequence_correct)
        item["failed_eos_tokens"] += count * int(is_eos and not sequence_correct)
    for (dimension, value, coalition_mask), item in sorted(aggregate.items()):
        token_count = item["tokens"]
        rows.append(
            {
                "dimension": dimension,
                "stratum": value,
                "mask": coalition_mask,
                "members": coalition_members[coalition_mask],
                "token_count": token_count,
                "token_fraction_within_stratum": token_count / totals[(dimension, value)],
                "correct_sequence_token_count": item["correct_sequence_tokens"],
                "wrong_sequence_token_count": item["wrong_sequence_tokens"],
                "sequence_correct_association": item["correct_sequence_tokens"] / token_count,
                "eos_token_count": item["eos_tokens"],
                "successful_eos_token_count": item["successful_eos_tokens"],
                "failed_eos_token_count": item["failed_eos_tokens"],
                "eos_fraction": item["eos_tokens"] / token_count,
                "analysis_role": "TEST DESCRIPTIVE / NONCAUSAL COALITION ASSOCIATION",
            }
        )

    dimensions = sorted({row["dimension"] for row in rows})
    packet = {
        "schema_version": "1.0",
        "status": "FULL_TEST_COALITION_STRATA_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "TEST DESCRIPTIVE / NONCAUSAL COALITION ASSOCIATION",
        "example_count": examples,
        "token_count": tokens,
        "coalition_count": len(coalition_members),
        "dimensions": dimensions,
        "record_count": len(rows),
        "records": rows,
        "provenance": {
            "raw_path": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "raw_sha256": sha256(RAW),
            "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "new_model_inference": False,
        },
        "limitations": [
            "Sequence-correct association labels every token by the final sequence outcome and is not local token correctness.",
            "Coalition frequencies and associations are descriptive; they do not identify causal expert contributions.",
            "Strata are not independent confirmatory families and receive no subgroup p-values.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat = [{**row, "members": ";".join(row["members"])} for row in rows]
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    difficulty_rows = [row for row in rows if row["dimension"] == "difficulty"]
    leading = sorted(
        difficulty_rows,
        key=lambda row: (row["stratum"], -row["token_count"], row["mask"]),
    )
    report_rows = [
        f'| {row["stratum"]} | {row["mask"]} | {row["token_count"]} | '
        f'{100*row["token_fraction_within_stratum"]:.2f}% | '
        f'{100*row["sequence_correct_association"]:.2f}% | {row["successful_eos_token_count"]} |'
        for row in leading
        if row["token_fraction_within_stratum"] >= 0.02
    ]
    REPORT.write_text(
        "\n".join(
            [
                "# Full-TEST coalition strata",
                "",
                "Status: **FULL_TEST_COALITION_STRATA_COMPLETE**",
                "",
                "This frozen-trace supplement covers global, reasoning-family, difficulty, answer-class, domain, and token-position coalition strata. The table below shows coalitions with at least 2% share within each difficulty stratum; the machine artifact retains every observed coalition.",
                "",
                "| Difficulty | Mask | Tokens | Within-stratum share | Sequence-correct association | Successful EOS tokens |",
                "|---|---|---:|---:|---:|---:|",
                *report_rows,
                "",
                "Sequence-correct association is not local coalition accuracy, and no coalition-frequency pattern is interpreted causally.",
                "",
                f"Raw SHA-256: `{EXPECTED_RAW_SHA256}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": packet["status"], "records": len(rows), "dimensions": dimensions}, indent=2))


if __name__ == "__main__":
    main()
