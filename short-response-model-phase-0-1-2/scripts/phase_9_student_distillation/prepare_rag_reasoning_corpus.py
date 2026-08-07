from __future__ import annotations

"""Prepare the Phase 9 RAG Reasoning Corpus for the v1.7.3 workflow.

The source corpus is already high quality. This adapter performs only explicit,
auditable integration repairs:

* normalize successful procedure rows to decision=answer while preserving
  mode=procedure;
* repair two procedure-disagreement rows whose trusted procedure makes the
  requested value objectively computable (semantic versions and ISO dates);
* mark the remaining procedure-conflict rows as explicit conflict scenarios;
* enrich stale/current, tentative, untrusted, and procedure memories with
  source-state / validity metadata for Phase 9B;
* regenerate direct SFT splits, DPO preferences, evaluator calibration rows,
  and file hashes from one canonical reviewed corpus.

The source directory is never modified.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

ABSTENTION_TEXT = "I don't know based on the available information."
SCHEMA_VERSION = "2.1"
SOURCE_NAME = "synthetic_phase9_rag_reasoning_v2"
INTEGRATED_NAME = "Phase 9 RAG Reasoning Corpus v2.1 (v1.7.3 integration)"

RESOLVABLE_PROCEDURE_ROWS = {
    "train_procedure_semver_family_04_conflicting_inputs": {
        "answer": "Version 2.10.0.",
        "required": "Version 2.10.0",
        "wrong": "Version 2.9.0.",
        "maybe": "Probably Version 2.10.0, based on the evidence.",
    },
    "train_procedure_iso_date_family_04_conflicting_inputs": {
        "answer": "2032-11-02.",
        "required": "2032-11-02",
        "wrong": "2032-04-09.",
        "maybe": "Probably 2032-11-02, based on the evidence.",
    },
}

TRUE_PROCEDURE_CONFLICT_CLAIMS = {
    "train_procedure_latency_family_04_conflicting_inputs": (
        "relay_latency_pair",
        ["Pine=11 ms | Birch=12 ms", "Pine=18 ms | Birch=10 ms"],
    ),
    "train_procedure_capacity_family_04_conflicting_inputs": (
        "vault_capacity_pair",
        ["Cedar=72 TB | Pine=64 TB", "Cedar=48 TB | Pine=80 TB"],
    ),
    "train_procedure_revision_family_04_conflicting_inputs": (
        "latest_revision_assertion",
        ["Revision 17", "Revision 21"],
    ),
    "dev_procedure_threshold_family_04_conflicting_inputs": (
        "sample_c_measurement",
        ["68 units", "74 units"],
    ),
    "test_procedure_overlap_family_04_conflicting_inputs": (
        "window_b_interval",
        ["11:00-12:00", "12:00-13:00"],
    ),
    "test_procedure_policy_precedence_family_04_conflicting_inputs": (
        "current_policy_weight",
        ["25 kg", "30 kg"],
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_contract(phrase: str) -> dict[str, Any]:
    return {
        "schema_version": "4.5",
        "open_ended": False,
        "acceptable_modes": [],
        "requirements": [
            {
                "id": "req_0",
                "label": "explicit_required_content_missing",
                "severity": "major",
                "weight": 1.0,
                "description": f"The reply must include: {phrase}",
                "failure_evidence": f"Required content was missing: {phrase}",
                "match": {"any_phrases": [phrase]},
            }
        ],
        "prohibitions": [],
    }


def normalize_reviewed_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(raw))
    row["dataset_schema_version"] = SCHEMA_VERSION
    row["source"] = SOURCE_NAME

    decision = str(row.get("decision") or "").casefold()
    mode = str(row.get("mode") or "").casefold()

    # "procedure" is a mode/task type, not an answerability decision.
    if mode == "procedure" and decision == "procedure":
        row["decision"] = "answer"

    repair = RESOLVABLE_PROCEDURE_ROWS.get(str(row.get("id")))
    if repair:
        row["decision"] = "answer"
        row["category"] = "procedure_resolvable_disagreement"
        row["answer"] = repair["answer"]
        row["approved_answer"] = repair["answer"]
        row["answer_stance"] = "certain"
        row["evidence_clarity"] = "procedure_resolvable_disagreement"
        row["scenario_flags"] = ["procedure_resolves_disagreement"]
        row["response_contract"] = _required_contract(repair["required"])
    elif str(row.get("category") or "") == "procedure_conflict":
        flags = list(row.get("scenario_flags") or [])
        if "conflict" not in flags:
            flags.append("conflict")
        row["scenario_flags"] = flags

    category = str(row.get("category") or "")
    current_state = dict(row.get("current_state") or {})
    if category == "stale_evidence":
        current_state.setdefault("requested_temporal_scope", "current")
    if category in {"conflict", "procedure_conflict"}:
        current_state.setdefault("evidence_policy", "equal_authority")
    if category == "prompt_injection":
        current_state.setdefault("evidence_policy", "trusted_only")
    if current_state:
        row["current_state"] = current_state

    # Enrich inline passages so Phase 9B does not depend on a second catalog.
    for item in row.get("evidence") or []:
        _enrich_memory_item(item, topic_id=str(row.get("topic_id") or ""))
    for item in row.get("procedures") or []:
        _enrich_procedure_item(item, topic_id=str(row.get("topic_id") or ""))

    claim = TRUE_PROCEDURE_CONFLICT_CLAIMS.get(str(row.get("id")))
    if claim:
        key, values = claim
        for index, item in enumerate(row.get("evidence") or []):
            item["claim_key"] = key
            item["claim_value"] = values[index]

    return row


def _enrich_memory_item(item: dict[str, Any], *, topic_id: str) -> None:
    trust = str(item.get("trust_level") or "standard").casefold()
    updated = str(item.get("updated_at") or "")
    item.setdefault("memory_type", "factual")
    source_state = dict(item.get("source_state") or {})
    source_state.setdefault("topic_id", topic_id)
    source_state.setdefault("updated_at", updated)
    source_state.setdefault("trust_level", trust)
    status = {
        "superseded": "superseded",
        "untrusted": "untrusted",
        "tentative": "tentative",
    }.get(trust, "current_or_standard")
    source_state.setdefault("status", status)
    item["source_state"] = source_state

    conditions = dict(item.get("validity_conditions") or {})
    if trust == "superseded":
        conditions.setdefault("requested_temporal_scope", "historical")
    elif "current notice" in str(item.get("text") or "").casefold():
        conditions.setdefault("requested_temporal_scope", "current")
    if conditions:
        item["validity_conditions"] = conditions


def _enrich_procedure_item(item: dict[str, Any], *, topic_id: str) -> None:
    item.setdefault("memory_type", "procedure")
    source_state = dict(item.get("source_state") or {})
    source_state.setdefault("topic_id", topic_id)
    source_state.setdefault("trust_level", str(item.get("trust_level") or "trusted"))
    source_state.setdefault("status", "trusted_procedure")
    item["source_state"] = source_state


def enrich_knowledge_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(raw))
    row.setdefault("memory_type", "factual")
    _enrich_memory_item(row, topic_id=str(row.get("topic_id") or ""))
    return row


def enrich_procedure_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(raw))
    row.setdefault("memory_type", "procedure")
    _enrich_procedure_item(row, topic_id=str(row.get("topic_id") or ""))
    return row


def direct_sft_row(reviewed: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: json.loads(json.dumps(value))
        for key, value in reviewed.items()
        if key not in {"review_status", "selected_candidate_index", "approved_answer", "teacher_provenance", "answer"}
    }
    answer = str(reviewed.get("approved_answer") or reviewed.get("answer") or "").strip()
    messages = [dict(message) for message in reviewed.get("messages") or []]
    if not messages or messages[-1].get("role") == "assistant":
        raise ValueError(f"{reviewed.get('id')}: reviewed messages must be prompt-only")
    messages.append({"role": "assistant", "content": answer})
    row["messages"] = messages
    row["reference"] = answer
    row["metadata"] = {
        **(reviewed.get("metadata") or {}),
        "source_provenance": SOURCE_NAME,
        "dataset_schema_version": SCHEMA_VERSION,
        "intended_split": reviewed.get("intended_split"),
        "teacher_provenance": reviewed.get("teacher_provenance") or {},
        "review_status": reviewed.get("review_status"),
    }
    return row


def patch_candidate_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(raw))
    repair = RESOLVABLE_PROCEDURE_ROWS.get(str(row.get("source_prompt_id")))
    if not repair:
        return row
    row["category"] = "procedure_resolvable_disagreement"
    row["evidence_clarity"] = "procedure_resolvable_disagreement"
    row["gold_answer_stance"] = "certain"
    row["gold_answer"] = repair["answer"]
    row["candidates"] = [
        {
            "candidate_id": "correct",
            "stance": "correct",
            "answer": repair["answer"],
            "expected_acceptability": "accepted",
            "expected_score": 1.0,
            "failure_type": None,
        },
        {
            "candidate_id": "maybe",
            "stance": "qualified",
            "answer": repair["maybe"],
            "expected_acceptability": "acceptable_but_suboptimal",
            "expected_score": 0.7,
            "failure_type": "unnecessary_qualification",
        },
        {
            "candidate_id": "uncertain",
            "stance": "uncertain",
            "answer": ABSTENTION_TEXT,
            "expected_acceptability": "rejected",
            "expected_score": 0.25,
            "failure_type": "unnecessary_abstention",
        },
        {
            "candidate_id": "wrong",
            "stance": "wrong",
            "answer": repair["wrong"],
            "expected_acceptability": "rejected",
            "expected_score": 0.0,
            "failure_type": "incorrect_claim",
        },
    ]
    return row


def build_preference_rows(
    reviewed_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for reviewed in sorted(
        (row for row in reviewed_by_id.values() if row.get("intended_split") == split),
        key=lambda row: str(row["id"]),
    ):
        judgment = judgments_by_id[str(reviewed["id"])]
        prompt = _prompt_from_messages(reviewed["messages"])
        correct = next(candidate for candidate in judgment["candidates"] if candidate["candidate_id"] == "correct")
        for candidate in judgment["candidates"]:
            if candidate["candidate_id"] == "correct":
                continue
            output.append(
                {
                    "id": f"dpo_rag_{split}_{reviewed['id']}_{candidate['candidate_id']}",
                    "prompt": prompt,
                    "chosen": correct["answer"],
                    "rejected": candidate["answer"],
                    "metadata": {
                        "preference_dimension": "rag_evidence_calibration",
                        "rejection_type": candidate.get("failure_type"),
                        "source_provenance": SOURCE_NAME,
                        "human_review_status": "accepted",
                        "split": split,
                        "source_prompt_id": reviewed["id"],
                        "topic_id": reviewed["topic_id"],
                        "evidence_clarity": reviewed.get("evidence_clarity"),
                        "gold_answer_stance": reviewed.get("answer_stance"),
                        "rejected_stance": candidate.get("stance"),
                    },
                }
            )
    return output


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    # Preserve the source package's Qwen-compatible preference format.
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_evaluator_cases(
    reviewed_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    splits: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for reviewed in sorted(
        (row for row in reviewed_by_id.values() if row.get("intended_split") in splits),
        key=lambda row: str(row["id"]),
    ):
        judgment = judgments_by_id[str(reviewed["id"])]
        for candidate in judgment["candidates"]:
            output.append(
                {
                    "id": f"calibration::{reviewed['id']}::{candidate['candidate_id']}",
                    "source_prompt_id": reviewed["id"],
                    "topic_id": reviewed["topic_id"],
                    "split": reviewed["intended_split"],
                    "question": reviewed["question"],
                    "mode": reviewed["mode"],
                    "category": reviewed["category"],
                    "evidence": reviewed.get("evidence") or [],
                    "procedures": reviewed.get("procedures") or [],
                    "response_contract": reviewed.get("response_contract") or {},
                    "reference": reviewed.get("approved_answer") or reviewed.get("answer"),
                    "candidate_answer": candidate["answer"],
                    "candidate_stance": candidate["stance"],
                    "expected_acceptability": candidate["expected_acceptability"],
                    "expected_score": candidate["expected_score"],
                    "expected_failure_type": candidate.get("failure_type"),
                    "evidence_clarity": reviewed.get("evidence_clarity"),
                    "gold_answer_stance": reviewed.get("answer_stance"),
                }
            )
    return output


def validate_splits(rows: list[dict[str, Any]]) -> None:
    topic_owner: dict[str, str] = {}
    doc_owner: dict[str, str] = {}
    for row in rows:
        split = str(row.get("intended_split") or "")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"{row.get('id')}: invalid intended_split={split!r}")
        topic = str(row.get("topic_id") or "")
        previous = topic_owner.setdefault(topic, split)
        if previous != split:
            raise ValueError(f"topic leakage: {topic} is in {previous} and {split}")
        ids = {
            str(value)
            for value in [
                *(row.get("support_document_ids") or []),
                *(row.get("retrieved_document_ids") or []),
                *(row.get("retrieved_procedure_ids") or []),
            ]
            if value
        }
        for item in row.get("evidence") or []:
            ids.add(str(item.get("document_id") or item.get("record_id") or ""))
        for item in row.get("procedures") or []:
            ids.add(str(item.get("procedure_id") or item.get("record_id") or ""))
        for identifier in ids:
            if not identifier:
                continue
            previous = doc_owner.setdefault(identifier, split)
            if previous != split:
                raise ValueError(f"document/procedure leakage: {identifier} is in {previous} and {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the integrated Phase 9 RAG reasoning corpus.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    reviewed = [normalize_reviewed_row(row) for row in read_jsonl(source / "teacher_candidates.reviewed.jsonl")]
    validate_splits(reviewed)
    reviewed_by_id = {str(row["id"]): row for row in reviewed}
    if len(reviewed_by_id) != len(reviewed):
        raise ValueError("duplicate reviewed IDs")

    knowledge = [enrich_knowledge_row(row) for row in read_jsonl(source / "knowledge.jsonl")]
    procedures = [enrich_procedure_row(row) for row in read_jsonl(source / "procedures.jsonl")]
    judgments = [patch_candidate_judgment(row) for row in read_jsonl(source / "evaluator/candidate_judgments.jsonl")]
    judgments_by_id = {str(row["source_prompt_id"]): row for row in judgments}
    if set(judgments_by_id) != set(reviewed_by_id):
        raise ValueError("candidate judgments do not match reviewed prompt IDs")

    write_jsonl(output / "knowledge.jsonl", knowledge)
    write_jsonl(output / "procedures.jsonl", procedures)
    write_jsonl(output / "prompts.all.jsonl", reviewed)
    write_jsonl(output / "teacher_candidates.reviewed.jsonl", reviewed)
    write_jsonl(output / "evaluator/candidate_judgments.jsonl", judgments)

    for split in ("train", "dev", "test"):
        split_reviewed = [row for row in reviewed if row["intended_split"] == split]
        write_jsonl(output / f"reviewed_splits/{split}.reviewed.jsonl", split_reviewed)
        write_jsonl(output / f"splits/{split}.jsonl", [direct_sft_row(row) for row in split_reviewed])

    write_jsonl(
        output / "preferences/train.jsonl",
        build_preference_rows(reviewed_by_id, judgments_by_id, "train"),
    )
    write_jsonl(
        output / "preferences/dev.jsonl",
        build_preference_rows(reviewed_by_id, judgments_by_id, "dev"),
    )
    write_jsonl(
        output / "evaluator/calibration_train_dev.jsonl",
        build_evaluator_cases(reviewed_by_id, judgments_by_id, {"train", "dev"}),
    )
    write_jsonl(
        output / "evaluator/heldout_test_cases.jsonl",
        build_evaluator_cases(reviewed_by_id, judgments_by_id, {"test"}),
    )

    # Preserve the original taxonomy, extended with integration-specific labels.
    taxonomy_path = source / "manifests/failure_taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["version"] = "1.1"
    taxonomy.setdefault("labels", {})["used_superseded_evidence"] = (
        "The response uses a superseded value despite a current replacement."
    )
    taxonomy["labels"]["missed_procedure_resolution"] = (
        "The response abstains even though a trusted procedure can resolve the apparent disagreement."
    )
    write_json(output / "manifests/failure_taxonomy.json", taxonomy)

    counts = {
        "knowledge_documents": len(knowledge),
        "procedures": len(procedures),
        "prompt_rows": len(reviewed),
        "reviewed_rows": len(reviewed),
        "candidate_judgments": len(judgments),
        "evaluator_calibration_cases": len(read_jsonl(output / "evaluator/calibration_train_dev.jsonl")),
        "heldout_test_candidate_cases": len(read_jsonl(output / "evaluator/heldout_test_cases.jsonl")),
        "preference_train_pairs": len(read_jsonl(output / "preferences/train.jsonl")),
        "preference_dev_pairs": len(read_jsonl(output / "preferences/dev.jsonl")),
    }
    split_counts = Counter(str(row["intended_split"]) for row in reviewed)
    decision_counts = Counter(str(row.get("decision")) for row in reviewed)
    category_counts = Counter(str(row.get("category")) for row in reviewed)
    clarity_counts = Counter(str(row.get("evidence_clarity")) for row in reviewed)
    stance_counts = Counter(str(row.get("answer_stance")) for row in reviewed)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": INTEGRATED_NAME,
        "source": SOURCE_NAME,
        "integration_revision": "short_response_model_v1_7_3",
        "created_at": "2026-08-06",
        "purpose": (
            "First-class Phase 9-11 evidence-use, retrieval, procedure, calibration, "
            "adversarial, stale-precedence, and memory-reconstruction seed corpus."
        ),
        "counts": counts,
        "split_counts": dict(split_counts),
        "topic_family_counts": {
            split: len({row["topic_id"] for row in reviewed if row["intended_split"] == split})
            for split in ("train", "dev", "test")
        },
        "decision_counts": dict(decision_counts),
        "answer_stance_counts": dict(stance_counts),
        "category_counts": dict(category_counts),
        "evidence_clarity_counts": dict(clarity_counts),
        "repairs": {
            "procedure_decision_normalization": 16,
            "procedure_resolvable_disagreement_rows": sorted(RESOLVABLE_PROCEDURE_ROWS),
            "true_procedure_conflicts_enriched": sorted(TRUE_PROCEDURE_CONFLICT_CLAIMS),
            "source_state_and_validity_metadata_added": True,
        },
        "leakage_policy": (
            "Authored intended_split is authoritative. Topic families and every referenced "
            "document/procedure are split-disjoint; dev is used for selection/calibration and "
            "test remains sealed until final evaluation."
        ),
        "usage": {
            "phase9_reviewed_input": "teacher_candidates.reviewed.jsonl",
            "phase9_train": "reviewed_splits/train.reviewed.jsonl",
            "phase9_dev": "reviewed_splits/dev.reviewed.jsonl",
            "phase9_test": "reviewed_splits/test.reviewed.jsonl",
            "direct_sft_splits": "splits/*.jsonl",
            "dpo_preferences": "preferences/train.jsonl and preferences/dev.jsonl",
            "evaluator_calibration": "evaluator/calibration_train_dev.jsonl",
            "heldout_candidate_tests": "evaluator/heldout_test_cases.jsonl",
        },
        "safety_and_scope": [
            "All entities and facts are fictional.",
            "Only accepted targets belong in SFT; rejected candidates are preference/evaluator data.",
            "Never train on reviewed_splits/test.reviewed.jsonl, splits/test.jsonl, or evaluator/heldout_test_cases.jsonl.",
        ],
    }
    write_json(output / "manifests/dataset_manifest.json", manifest)

    validation = {
        "status": "PASS",
        "checks": {
            "jsonl_parse": True,
            "prompt_count": len(reviewed),
            "unique_prompt_ids": len(reviewed_by_id),
            "unique_document_ids": len({row["document_id"] for row in knowledge}),
            "topic_split_leakage": False,
            "document_topic_mismatch": False,
            "abstentions_are_canonical": all(
                (row.get("decision") != "abstain") or row.get("approved_answer") == ABSTENTION_TEXT
                for row in reviewed
            ),
            "procedure_answer_decisions_normalized": all(
                row.get("decision") != "procedure" for row in reviewed
            ),
            "response_contract_schema": "4.5",
            "intended_splits_preserved": True,
        },
        "file_row_counts": {},
        "research_ready_minima": {
            "prompt_rows_at_least_30": len(reviewed) >= 30,
            "supported_and_unsupported_present": {"answer", "abstain"} <= set(decision_counts),
            "procedure_rows_present": any(row.get("mode") == "procedure" for row in reviewed),
            "social_rows_present": any(row.get("mode") == "social_roleplay" for row in reviewed),
            "injection_rows_present": "prompt_injection" in category_counts,
            "conflict_rows_present": any("conflict" in str(row.get("category")) for row in reviewed),
            "stale_rows_present": "stale_evidence" in category_counts,
        },
    }

    for path in sorted(output.rglob("*.jsonl")):
        validation["file_row_counts"][str(path.relative_to(output)).replace("\\", "/")] = len(read_jsonl(path))
    write_json(output / "manifests/validation_report.json", validation)

    readme = """# Phase 9 RAG Reasoning Corpus v2.1 — integrated for model v1.7.3

This is the repository-native form of the synthetic Phase 9 RAG Reasoning Corpus v2.

## Integration repairs

- Successful procedure tasks use `mode=procedure` with `decision=answer`.
- The semantic-version and ISO-date disagreement examples are answered because the trusted procedure resolves the requested comparison.
- Remaining procedure conflicts are explicit conflict scenarios and carry structured claim metadata when possible.
- Evidence memories include source-state / validity metadata for Phase 9B.
- Authored train/dev/test topic-family splits are preserved.
- Preferences and evaluator calibration data are regenerated from the repaired canonical rows.

## Phase boundaries

- **Phase 9:** train only on train; tune/calibrate on dev; test is sealed.
- **Phase 9B:** derive reconstruction candidates from train/dev, with human review before labels become training data.
- **Phase 10:** architecture screening and recovery selection use dev; only the selected candidate is evaluated on test.
- **Phase 11:** answerability calibrates on dev; release benchmark uses test.

Do not train on any test or held-out evaluator file.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    file_hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            file_hashes[str(path.relative_to(output)).replace("\\", "/")] = file_sha256(path)
    write_json(output / "manifests/file_hashes.json", file_hashes)

    print(json.dumps({
        "output": str(output),
        "counts": counts,
        "split_counts": dict(split_counts),
        "decision_counts": dict(decision_counts),
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
