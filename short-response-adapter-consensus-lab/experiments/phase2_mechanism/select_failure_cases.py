from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


LAB_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = LAB_ROOT.parent / "short-response-model-phase-0-1-2"
RAW_RESULT = LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
TEST_DATASET = OFFICIAL_ROOT / "data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits/test.reviewed.jsonl"
BACKBONE = OFFICIAL_ROOT / "artifacts/phase_9_capability_composition_v22_lr1e4_s1200/frozen_anchor"
CORNERSTONE = LAB_ROOT / "reports/phase2_cornerstone_verification.json"
OUTPUT = LAB_ROOT / "results/phase2_mechanism/qualitative_selection.json"
REPORT = LAB_ROOT / "reports/PHASE2_FAILURE_ANALYSIS.md"

CONDITIONS = {
    ("track_a_k1", "hard_plurality"): "track_a",
    ("abstention_k1", "hard_plurality"): "abstention",
    ("full_k7", "hard_plurality"): "hard_k7",
    ("full_k7", "probability_mean"): "probability_k7",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> dict[str, dict[str, dict[str, Any]]]:
    conditions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(RAW_RESULT):
        alias = CONDITIONS.get((str(row.get("committee")), str(row.get("method"))))
        if alias:
            conditions[alias][str(row["example_id"])] = row
    return conditions


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def vote_shape(step: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted((int(value) for value in step.get("vote_counts", {}).values()), reverse=True))


def bucket_memberships(
    example_id: str,
    rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, int | None]:
    hard = rows["hard_k7"][example_id]
    track = rows["track_a"][example_id]
    probability = rows["probability_k7"][example_id]
    abstention = rows["abstention"][example_id]
    steps = hard["steps"]
    divergence = first_divergence(hard["output_token_ids"], probability["output_token_ids"])
    memberships: dict[str, int | None] = {}
    if hard["correct"] and not track["correct"]:
        memberships["hard_rescues_track_a"] = 0
    if not hard["correct"] and track["correct"]:
        memberships["hard_harms_track_a"] = 0
    if hard["correct"] and not probability["correct"]:
        memberships["hard_beats_probability"] = divergence
    if probability["correct"] and not hard["correct"]:
        memberships["probability_beats_hard"] = divergence
    if steps and all(int(step["plurality_votes"]) == 7 for step in steps):
        memberships["all_experts_agree_correct" if hard["correct"] else "all_experts_agree_wrong"] = 0
    for index, step in enumerate(steps):
        shape = vote_shape(step)
        if shape == (4, 3) and "vote_4_3" not in memberships:
            memberships["vote_4_3"] = index
        if shape == (3, 2, 2) and "vote_3_2_2" not in memberships:
            memberships["vote_3_2_2"] = index
        if bool(step.get("tie")):
            key = "tie_success" if hard["correct"] else "tie_failure"
            memberships.setdefault(key, index)
    if hard.get("decision") != "answer" and hard["correct"]:
        memberships["abstention_success"] = max(0, len(steps) - 1)
    if hard.get("decision") == "answer" and hard.get("abstained"):
        memberships["false_abstention"] = max(0, len(steps) - 1)
    if hard.get("decision") != "answer" and not hard.get("abstained"):
        memberships["failure_to_abstain"] = max(0, len(steps) - 1)
    if divergence is not None and hard["correct"] and not probability["correct"]:
        memberships["hard_trajectory_rescue_proxy"] = divergence
    if divergence is not None and probability["correct"] and not hard["correct"]:
        memberships["soft_trajectory_rescue_proxy"] = divergence
    if not hard["correct"] and float(hard.get("mean_agreement") or 0.0) >= 0.75:
        memberships["high_agreement_wrong"] = 0
    if not hard["correct"] and any(bool(step.get("tie")) for step in steps):
        memberships["tie_resolution_failure"] = next(
            index for index, step in enumerate(steps) if bool(step.get("tie"))
        )
    if not hard["correct"] and track["correct"]:
        memberships["correct_track_a_broken_by_committee"] = 0
    if probability["correct"] and not hard["correct"]:
        memberships["soft_fusion_rescues_hard"] = divergence
    if hard["correct"] and not probability["correct"]:
        memberships["hard_rescues_soft_fusion"] = divergence
    if abstention["correct"] and not hard["correct"]:
        memberships["abstention_expert_rescues_hard"] = 0
    return memberships


def deterministic_diverse_selection(
    bucket: str,
    candidates: list[str],
    dataset: dict[str, dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    ranked = sorted(
        candidates,
        key=lambda example_id: hashlib.sha256(
            f"phase2-qualitative:{bucket}:{example_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[str] = []
    used_domains: set[str] = set()
    for example_id in ranked:
        domain = str(dataset[example_id].get("domain") or "unknown")
        if domain not in used_domains:
            selected.append(example_id)
            used_domains.add(domain)
            if len(selected) == limit:
                break
    for example_id in ranked:
        if len(selected) == limit:
            break
        if example_id not in selected:
            selected.append(example_id)
    return [
        {
            "example_id": example_id,
            "selection_sha256": hashlib.sha256(
                f"phase2-qualitative:{bucket}:{example_id}".encode("utf-8")
            ).hexdigest(),
        }
        for example_id in selected
    ]


def decode_step(
    tokenizer: Any,
    step: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if step is None:
        return None
    proposals = {
        name: {
            "token_id": int(token_id),
            "text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "top1_probability": float(step["top_probabilities"][name]),
            "top2_probability": float(step["top2_probabilities"][name]),
            "margin": float(step["top1_top2_margins"][name]),
            "entropy": float(step["expert_entropies"][name]),
        }
        for name, token_id in step["top_tokens"].items()
    }
    return {
        "t": int(step["t"]),
        "winner_token_id": int(step["winner"]),
        "winner_text": tokenizer.decode([int(step["winner"])], skip_special_tokens=False),
        "vote_counts": step["vote_counts"],
        "vote_shape": list(vote_shape(step)),
        "tie": bool(step.get("tie")),
        "tied_tokens": [int(value) for value in step.get("tied_tokens") or []],
        "tie_break_rule": step.get("tie_break_rule"),
        "winner_coalition": list(step.get("winning_expert_coalition") or []),
        "agreement": float(step["agreement"]),
        "vote_entropy": float(step["vote_entropy"]),
        "winner_is_eos": bool(step.get("winner_is_eos")),
        "proposals": proposals,
    }


def make_case(
    tokenizer: Any,
    bucket: str,
    selection: dict[str, str],
    trigger: int | None,
    dataset: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    example_id = selection["example_id"]
    gold = dataset[example_id]
    hard = rows["hard_k7"][example_id]
    probability = rows["probability_k7"][example_id]
    track = rows["track_a"][example_id]
    abstention = rows["abstention"][example_id]
    hard_step = hard["steps"][trigger] if trigger is not None and trigger < len(hard["steps"]) else None
    probability_step = (
        probability["steps"][trigger]
        if trigger is not None and trigger < len(probability["steps"])
        else None
    )
    return {
        "bucket": bucket,
        "example_id": example_id,
        "selection_sha256": selection["selection_sha256"],
        "dataset": "v2.2 TEST",
        "analysis_role": "TEST QUALITATIVE / DESCRIPTIVE",
        "domain": gold.get("domain"),
        "reasoning_family": gold.get("reasoning_family"),
        "difficulty": gold.get("difficulty"),
        "decision": gold.get("decision"),
        "messages": gold.get("messages"),
        "gold_answer": gold.get("approved_answer") or gold.get("answer"),
        "trigger_step": trigger,
        "hard_trigger_diagnostics": decode_step(tokenizer, hard_step),
        "probability_trigger_diagnostics": decode_step(tokenizer, probability_step),
        "outputs": {
            "track_a": {"text": track["output"], "correct": bool(track["correct"])},
            "abstention": {"text": abstention["output"], "correct": bool(abstention["correct"])},
            "probability_k7": {"text": probability["output"], "correct": bool(probability["correct"])},
            "hard_k7": {"text": hard["output"], "correct": bool(hard["correct"])},
        },
        "objective_interpretation": (
            "This case satisfies the machine-defined bucket. It illustrates the vote/trajectory pattern but does not estimate frequency or establish causality."
        ),
    }


def make_report(packet: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Failure and Qualitative-Case Analysis",
        "",
        "**Status: complete for frozen full-TEST sequence-level buckets; teacher-forced minority-gold cases remain pending.**",
        "",
        "Cases are selected before prompt inspection by SHA256 ranking within machine-defined buckets, with domain diversity where possible. The JSON artifact preserves the selection hashes, complete prompts, outputs, expert proposals at the trigger step, vote structure, confidence diagnostics, tie/EOS state, and scoring labels.",
        "",
        "## Bucket inventory",
        "",
        "| Bucket | Eligible examples | Selected IDs |",
        "|---|---:|---|",
    ]
    for bucket, values in sorted(packet["bucket_counts"].items()):
        selected = ", ".join(item["example_id"] for item in packet["selections"].get(bucket, []))
        lines.append(f"| `{bucket}` | {values} | {selected or 'none'} |")
    lines.extend(
        [
            "",
            "## Failure taxonomy counts",
            "",
            "| Failure pattern | Count | Interpretation boundary |",
            "|---|---:|---|",
        ]
    )
    explanations = {
        "high_agreement_wrong": "High mean vote agreement with wrong exact output; not necessarily high probability confidence.",
        "tie_resolution_failure": "At least one hard plurality tie occurs in a wrong sequence; the tie is not proven causal.",
        "false_abstention": "Hard emits an abstention on an answerable item.",
        "failure_to_abstain": "Hard does not abstain on an abstention-class item.",
        "correct_track_a_broken_by_committee": "Track A is correct and Hard K=7 is wrong.",
        "soft_fusion_rescues_hard": "Probability mean is correct and Hard K=7 is wrong.",
        "hard_rescues_soft_fusion": "Hard K=7 is correct and probability mean is wrong.",
        "abstention_expert_rescues_hard": "Abstention K=1 is correct and Hard K=7 is wrong.",
    }
    for bucket in explanations:
        lines.append(
            f"| `{bucket}` | {packet['bucket_counts'].get(bucket, 0)} | {explanations[bucket]} |"
        )
    lines.extend(
        [
            "",
            "## Required pending qualitative evidence",
            "",
            "- Minority expert proposes the gold next token but is outvoted: requires full teacher-forced DEV cache.",
            "- Semantically correct but exact-match wrong: requires the preselected secondary semantic audit.",
            "- Causal trajectory correction: requires same-split teacher-forced/free-running DEV comparison.",
            "- Premature EOS causality: requires counterfactual EOS-policy or gold-prefix evidence; short wrong output alone is insufficient.",
            "",
            "## Provenance",
            "",
            f"- Raw result SHA256: `{packet['provenance']['raw_sha256']}`",
            f"- TEST dataset SHA256: `{packet['provenance']['test_sha256']}`",
            f"- Tokenizer SHA256: `{packet['provenance']['tokenizer_sha256']}`",
            f"- Script SHA256: `{packet['provenance']['script_sha256']}`",
            f"- Full selected case records: `results/phase2_mechanism/qualitative_selection.json`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if read_json(CORNERSTONE).get("status") != "PASS":
        raise RuntimeError("Cornerstone verification must pass first")
    dataset = {str(row["id"]): row for row in iter_jsonl(TEST_DATASET)}
    rows = load_rows()
    ids = sorted(rows["hard_k7"])
    if len(ids) != 4_050 or any(set(condition) != set(ids) for condition in rows.values()):
        raise RuntimeError("Required conditions are not fully paired")
    bucket_candidates: dict[str, list[str]] = defaultdict(list)
    triggers: dict[tuple[str, str], int | None] = {}
    for example_id in ids:
        for bucket, trigger in bucket_memberships(example_id, rows).items():
            bucket_candidates[bucket].append(example_id)
            triggers[(bucket, example_id)] = trigger
    selections = {
        bucket: deterministic_diverse_selection(bucket, candidates, dataset)
        for bucket, candidates in sorted(bucket_candidates.items())
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(BACKBONE), trust_remote_code=True, local_files_only=True
    )
    cases = []
    for bucket, selected in selections.items():
        for item in selected:
            cases.append(
                make_case(
                    tokenizer,
                    bucket,
                    item,
                    triggers[(bucket, item["example_id"])],
                    dataset,
                    rows,
                )
            )
    packet = {
        "schema_version": "1.0",
        "status": "QUALITATIVE_SELECTION_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "SHA256 phase2-qualitative:<bucket>:<example_id>; first distinct domains, then remaining rank; limit 3",
        "analysis_role": "TEST QUALITATIVE / DESCRIPTIVE",
        "bucket_counts": {bucket: len(values) for bucket, values in sorted(bucket_candidates.items())},
        "selections": selections,
        "cases": cases,
        "provenance": {
            "raw_sha256": sha256_file(RAW_RESULT),
            "test_sha256": sha256_file(TEST_DATASET),
            "tokenizer_sha256": sha256_file(BACKBONE / "tokenizer.json"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            "Selected cases illustrate pre-counted buckets but do not estimate prevalence.",
            "A tie or early divergence in a failed sequence is not necessarily causal.",
            "Exact-match labels remain primary; semantic correctness is not adjudicated here.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(packet)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "bucket_counts": packet["bucket_counts"],
                "selected_case_records": len(cases),
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
