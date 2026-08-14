from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CACHE = ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl"
DATASET = (
    WORKSPACE
    / "short-response-model-phase-0-1-2/data/phase_9_student_distillation/"
    "multidomain_reasoning_v22/reviewed_splits/dev.reviewed.jsonl"
)
TOKENIZER = (
    WORKSPACE
    / "short-response-model-phase-0-1-2/artifacts/"
    "phase_9_capability_composition_v22_lr1e4_s1200/frozen_anchor"
)
OUTPUT = ROOT / "results/phase2_mechanism/qualitative_teacher_forced_selection.json"
REPORT = ROOT / "reports/PHASE2_TEACHER_FORCED_CASES.md"
EXPECTED_EXAMPLES = 4050
EXPECTED_TOKENS = 18760
EOS = 151645
PER_BUCKET = 3
BUCKETS = (
    "minority_expert_correct_but_outvoted",
    "gold_tied_but_tie_break_failed",
    "tie_break_selected_gold",
    "overconfident_wrong_dissenter_suppressed",
    "majority_wrong",
    "premature_eos",
    "failed_to_select_gold_eos",
    "probability_rescued_hard",
    "hard_rescued_probability",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_hash(bucket: str, example_id: str) -> str:
    return hashlib.sha256(
        f"phase2-tf-qualitative:{bucket}:{example_id}".encode("utf-8")
    ).hexdigest()


def hard_winner(token: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    proposals = [int(item["top1_token_id"]) for item in token["expert_metrics"]]
    counts = Counter(proposals)
    maximum = max(counts.values())
    tied = sorted(token_id for token_id, count in counts.items() if count == maximum)
    if len(tied) == 1:
        selected = tied[0]
    else:
        candidate_ids = [int(value) for value in token["candidate_token_ids"]]
        index = {token_id: position for position, token_id in enumerate(candidate_ids)}
        logits = token["candidate_logits_by_expert"]
        scores = {
            token_id: sum(float(expert_logits[index[token_id]]) for expert_logits in logits)
            for token_id in tied
        }
        best = max(scores.values())
        selected = min(
            token_id
            for token_id, score in scores.items()
            if math.isclose(score, best, rel_tol=1e-8, abs_tol=1e-8)
        )
    return selected, {
        "proposals": proposals,
        "vote_counts": {str(token_id): count for token_id, count in sorted(counts.items())},
        "maximum_vote_count": maximum,
        "tied_leaders": tied,
        "tie": len(tied) > 1,
    }


def probability_winner(row: dict[str, Any], token: dict[str, Any]) -> int:
    raw_index = [float(value) for value in row["temperature_grid"]].index(1.0)
    candidate_ids = np.asarray(token["candidate_token_ids"], dtype=np.int64)
    logits = np.asarray(token["candidate_logits_by_expert"], dtype=np.float64)
    lse = np.asarray(
        [item["logsumexp_by_temperature"][raw_index] for item in token["expert_metrics"]],
        dtype=np.float64,
    )
    scores = np.exp(logits - lse[:, None]).mean(axis=0)
    maximum = float(scores.max())
    finalists = np.flatnonzero(scores == maximum)
    return int(candidate_ids[finalists[np.argmin(candidate_ids[finalists])]])


def decode(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def main() -> None:
    dataset_rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = {row["id"]: row for row in dataset_rows}
    if len(dataset_rows) != EXPECTED_EXAMPLES or len(dataset) != EXPECTED_EXAMPLES:
        raise RuntimeError("DEV dataset completeness check failed")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_example_bucket: set[tuple[str, str]] = set()
    example_count = 0
    token_count = 0
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_count += 1
            example_id = str(row["example_id"])
            source = dataset[example_id]
            prompt = next(message["content"] for message in source["messages"] if message["role"] == "user")
            for token in row["tokens"]:
                token_count += 1
                gold = int(token["gold_token_id"])
                hard, diagnostics = hard_winner(token)
                probability = probability_winner(row, token)
                proposals = diagnostics["proposals"]
                gold_votes = proposals.count(gold)
                wrong_confidences = [
                    float(item["top1_probability"])
                    for item in token["expert_metrics"]
                    if int(item["top1_token_id"]) != gold
                ]
                predicates = {
                    "minority_expert_correct_but_outvoted": (
                        hard != gold and 0 < gold_votes < diagnostics["maximum_vote_count"]
                    ),
                    "gold_tied_but_tie_break_failed": (
                        diagnostics["tie"] and gold in diagnostics["tied_leaders"] and hard != gold
                    ),
                    "tie_break_selected_gold": diagnostics["tie"] and hard == gold,
                    "overconfident_wrong_dissenter_suppressed": (
                        hard == gold and bool(wrong_confidences) and max(wrong_confidences) >= 0.9
                    ),
                    "majority_wrong": hard != gold and diagnostics["maximum_vote_count"] >= 4,
                    "premature_eos": hard == EOS and gold != EOS,
                    "failed_to_select_gold_eos": gold == EOS and hard != EOS,
                    "probability_rescued_hard": probability == gold and hard != gold,
                    "hard_rescued_probability": hard == gold and probability != gold,
                }
                for bucket, include in predicates.items():
                    key = (bucket, example_id)
                    if not include or key in seen_example_bucket:
                        continue
                    seen_example_bucket.add(key)
                    expert_details = []
                    for expert, item in zip(row["expert_order"], token["expert_metrics"], strict=True):
                        token_id = int(item["top1_token_id"])
                        expert_details.append(
                            {
                                "expert": expert,
                                "top1_token_id": token_id,
                                "top1_token_text": decode(tokenizer, token_id),
                                "top1_probability": float(item["top1_probability"]),
                                "margin": float(item["margin"]),
                                "is_gold": token_id == gold,
                            }
                        )
                    eligible[bucket].append(
                        {
                            "bucket": bucket,
                            "example_id": example_id,
                            "t": int(token["t"]),
                            "reasoning_family": row["reasoning_family"],
                            "domain": row["domain"],
                            "difficulty": row["difficulty"],
                            "decision": row["decision"],
                            "prompt": prompt,
                            "gold_answer": source["approved_answer"],
                            "gold_token_id": gold,
                            "gold_token_text": token["gold_token_text"],
                            "hard_token_id": hard,
                            "hard_token_text": decode(tokenizer, hard),
                            "probability_token_id": probability,
                            "probability_token_text": decode(tokenizer, probability),
                            "gold_votes": gold_votes,
                            **diagnostics,
                            "experts": expert_details,
                            "maximum_wrong_expert_confidence": max(wrong_confidences) if wrong_confidences else None,
                        }
                    )
    if example_count != EXPECTED_EXAMPLES or token_count != EXPECTED_TOKENS:
        raise RuntimeError(f"Unexpected cache size: {example_count} examples / {token_count} tokens")

    counts = {bucket: len(eligible[bucket]) for bucket in BUCKETS}
    empty = [bucket for bucket, count in counts.items() if count == 0]
    selected: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        ordered = sorted(
            eligible[bucket], key=lambda item: (selection_hash(bucket, item["example_id"]), item["example_id"])
        )
        for rank, item in enumerate(ordered[:PER_BUCKET], 1):
            item["selection_rank"] = rank
            item["selection_hash"] = selection_hash(bucket, item["example_id"])
            selected.append(item)

    packet = {
        "schema_version": 1,
        "status": "DEV_TEACHER_FORCED_QUALITATIVE_SELECTION_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "DEV GOLD-PREFIX QUALITATIVE / DETERMINISTIC PRE-NARRATIVE SELECTION",
        "selection_rule": (
            "For each bucket, retain the first qualifying token per example, sort examples by "
            "SHA256('phase2-tf-qualitative:<bucket>:<example_id>'), and take the first three."
        ),
        "example_count": example_count,
        "token_count": token_count,
        "eligible_counts": counts,
        "empty_buckets": empty,
        "selected_count": len(selected),
        "selected": selected,
        "provenance": {
            "cache_sha256": sha256(CACHE),
            "dataset_sha256": sha256(DATASET),
            "tokenizer_config_sha256": sha256(TOKENIZER / "tokenizer_config.json"),
            "tokenizer_json_sha256": sha256(TOKENIZER / "tokenizer.json"),
            "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "Cases illustrate machine-counted buckets but do not estimate prevalence beyond eligible_counts.",
            "All decisions use gold prefixes and are local, not free-running trajectories.",
            "An expert's top-1 confidence is descriptive and not a causal attribution of the selected token.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Deterministically selected teacher-forced DEV cases",
        "",
        "Status: **DEV_TEACHER_FORCED_QUALITATIVE_SELECTION_COMPLETE**",
        "",
        "Cases were selected before narrative interpretation by hashing every eligible example ID within each machine-defined bucket. Counts use all 4,050 examples and 18,760 gold-prefix decisions.",
        "",
        "## Bucket inventory",
        "",
        "| Bucket | Eligible examples | Selected |",
        "|---|---:|---:|",
    ]
    for bucket in BUCKETS:
        lines.append(f"| {bucket} | {counts[bucket]:,} | {min(PER_BUCKET, counts[bucket])} |")
    for bucket in BUCKETS:
        lines.extend(["", f"## {bucket}", ""])
        bucket_items = [item for item in selected if item["bucket"] == bucket]
        if not bucket_items:
            lines.append("No eligible examples.")
            continue
        for item in bucket_items:
            gold_experts = [expert["expert"] for expert in item["experts"] if expert["is_gold"]]
            top = sorted(item["experts"], key=lambda expert: expert["top1_probability"], reverse=True)[:3]
            lines.extend(
                [
                    f"### {item['selection_rank']}. `{item['example_id']}` at t={item['t']}",
                    "",
                    f"- Selection hash: `{item['selection_hash']}`",
                    f"- Prompt: {item['prompt']}",
                    f"- Gold answer/token: `{item['gold_answer']}` / `{item['gold_token_text']}` ({item['gold_token_id']})",
                    f"- Hard / probability token: `{item['hard_token_text']}` ({item['hard_token_id']}) / `{item['probability_token_text']}` ({item['probability_token_id']})",
                    f"- Vote counts: `{json.dumps(item['vote_counts'], sort_keys=True)}`; tied leaders: `{item['tied_leaders']}`; gold-voting experts: `{gold_experts}`",
                    "- Highest-confidence experts: "
                    + "; ".join(
                        f"{expert['expert']}=`{expert['top1_token_text']}` p={expert['top1_probability']:.3f} margin={expert['margin']:.3f}"
                        for expert in top
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These are local gold-prefix examples. They support case-level mechanism inspection but do not identify free-running counterfactual trajectories or causal expert effects.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "eligible_counts": counts, "empty_buckets": empty, "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
