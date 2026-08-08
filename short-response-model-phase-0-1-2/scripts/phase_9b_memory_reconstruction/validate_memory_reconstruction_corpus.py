from __future__ import annotations

"""Strict structural/provenance validation for the integrated Phase 9B corpus."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    DECISIONS,
    compact_reconstruction_json,
    evaluate_validity_conditions,
    validate_memory_record,
    validate_reconstruction_record,
)

KNOWN_PREDICATES = {
    "candidate_price <= max_price",
    "all_required_attributes_match",
    "required_vram_gb > available_vram_gb",
    "required_vram_gb <= available_vram_gb",
    "stock_count < reorder_point",
    "stock_count >= reorder_point",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual_state_diffs(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def validate(root: Path) -> dict[str, Any]:
    problems: list[str] = []
    manifest_path = root / "manifests" / "dataset_manifest.json"
    hash_path = root / "manifests" / "file_hashes.json"
    if not manifest_path.exists() or not hash_path.exists():
        return {"status": "FAIL", "problems": ["missing manifest/hash manifest"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    for rel, expected in hashes.items():
        path = root / rel
        if not path.exists():
            problems.append(f"hash manifest path missing: {rel}")
            continue
        if len(path.read_bytes()) != int(expected.get("bytes", -1)):
            problems.append(f"byte count mismatch: {rel}")
        if file_sha256(path) != expected.get("sha256"):
            problems.append(f"sha256 mismatch: {rel}")

    memories = read_jsonl(root / "memories.jsonl")
    reviewed_raw = read_jsonl(root / "reviewed_memory_reconstruction.jsonl")
    memory_ids = [str(row.get("memory_id") or "") for row in memories]
    reviewed_ids = [str(row.get("id") or "") for row in reviewed_raw]
    if len(memory_ids) != len(set(memory_ids)):
        problems.append("duplicate memory_id")
    if len(reviewed_ids) != len(set(reviewed_ids)):
        problems.append("duplicate reviewed id")
    memory_by_id: dict[str, dict[str, Any]] = {}
    for raw in memories:
        try:
            normalized = validate_memory_record(raw)
        except Exception as exc:
            problems.append(f"invalid memory {raw.get('memory_id')}: {exc}")
            continue
        memory_by_id[normalized["memory_id"]] = normalized
        conditions = normalized.get("validity_conditions") or {}
        predicate = conditions.get("predicate") if isinstance(conditions, dict) else None
        if predicate and str(predicate) not in KNOWN_PREDICATES:
            problems.append(f"unsupported predicate in {normalized['memory_id']}: {predicate}")

    reviewed: list[dict[str, Any]] = []
    for raw in reviewed_raw:
        try:
            row = validate_reconstruction_record(raw, require_reviewed=True)
        except Exception as exc:
            problems.append(f"invalid reviewed row {raw.get('id')}: {exc}")
            continue
        reviewed.append(row)
        intended = str(raw.get("intended_split") or "").lower()
        metadata_split = str((raw.get("metadata") or {}).get("split") or "").lower()
        if intended not in {"train", "dev", "test"} or intended != metadata_split:
            problems.append(f"split metadata mismatch: {row['id']}")
        if not row.get("current_state"):
            problems.append(f"empty current_state: {row['id']}")
        input_ids = {item["memory_id"] for item in row["memories"]}
        for memory in row["memories"]:
            master = memory_by_id.get(memory["memory_id"])
            if master is None:
                problems.append(f"row references unknown memory: {row['id']}->{memory['memory_id']}")
        if not set(row["approved_reconstruction"]["source_ids"]).issubset(input_ids):
            problems.append(f"approved source_ids outside input memories: {row['id']}")
        if not set(row.get("support_document_ids") or []).issubset(input_ids):
            problems.append(f"support ids outside input memories: {row['id']}")

    decision_counts = Counter(row["approved_reconstruction"]["decision"] for row in reviewed)
    if set(decision_counts) != set(DECISIONS):
        problems.append(f"missing decision classes: {sorted(set(DECISIONS)-set(decision_counts))}")

    topic_by_split: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for raw in reviewed_raw:
        split = str(raw.get("intended_split") or "").lower()
        split_counts[split] += 1
        topic_by_split[split].add(str(raw.get("topic_id") or ""))
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = topic_by_split[left] & topic_by_split[right]
        if overlap:
            problems.append(f"topic split leakage {left}/{right}: {sorted(overlap)[:5]}")

    # Split files must be exact slices of the canonical file.
    by_id = {row["id"]: row for row in reviewed_raw}
    for split in ("train", "dev", "test"):
        split_rows = read_jsonl(root / "splits" / f"{split}.reviewed.jsonl")
        expected_ids = {row["id"] for row in reviewed_raw if row.get("intended_split") == split}
        actual_ids = {row["id"] for row in split_rows}
        if actual_ids != expected_ids:
            problems.append(f"split membership mismatch: {split}")
        for row in split_rows:
            if row != by_id.get(row["id"]):
                problems.append(f"split row differs from canonical: {row['id']}")

    # Counterfactual references and exact edit metadata.
    counterfactuals = read_jsonl(root / "counterfactuals" / "counterfactual_pairs.jsonl")
    for pair in counterfactuals:
        left = by_id.get(str(pair.get("positive_record_id") or ""))
        right = by_id.get(str(pair.get("edited_record_id") or ""))
        if left is None or right is None:
            problems.append(f"counterfactual missing row: {pair.get('pair_id')}")
            continue
        actual = _actual_state_diffs(left.get("current_state") or {}, right.get("current_state") or {})
        if actual != sorted(pair.get("changed_fields") or []):
            problems.append(f"counterfactual changed_fields mismatch: {pair.get('pair_id')}")
        decisive = set(pair.get("decisive_fields") or [])
        if not decisive or not decisive.issubset(set(actual)):
            problems.append(f"counterfactual decisive_fields invalid: {pair.get('pair_id')}")
        if left.get("intended_split") != right.get("intended_split") or left.get("intended_split") != pair.get("split"):
            problems.append(f"counterfactual split mismatch: {pair.get('pair_id')}")

    # Curated preference assets: exactly three deliberate negatives per canonical row.
    preference_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        pref_rows = read_jsonl(root / "preferences" / f"{split}.jsonl")
        preference_counts[split] = len(pref_rows)
        source_counts: Counter[str] = Counter()
        for pref in pref_rows:
            source_id = str(pref.get("source_record_id") or "")
            source = by_id.get(source_id)
            if source is None or source.get("intended_split") != split:
                problems.append(f"preference source split mismatch: {pref.get('id')}")
                continue
            source_counts[source_id] += 1
            try:
                chosen = json.loads(str(pref.get("chosen") or ""))
                rejected = json.loads(str(pref.get("rejected") or ""))
            except json.JSONDecodeError:
                problems.append(f"invalid preference JSON: {pref.get('id')}")
                continue
            if compact_reconstruction_json(chosen) != compact_reconstruction_json(source["approved_reconstruction"]):
                problems.append(f"preference chosen != approved target: {pref.get('id')}")
            if compact_reconstruction_json(rejected) == compact_reconstruction_json(chosen):
                problems.append(f"preference rejected == chosen: {pref.get('id')}")
        expected_source_ids = {row["id"] for row in reviewed_raw if row.get("intended_split") == split}
        if set(source_counts) != expected_source_ids or any(count != 3 for count in source_counts.values()):
            problems.append(f"preference coverage is not 3-per-row: {split}")

    # Evaluator candidates: approved + three known failure modes per row.
    eval_all = read_jsonl(root / "evaluator" / "candidate_reconstructions.all.jsonl")
    eval_counts = Counter(str(row.get("source_record_id") or "") for row in eval_all)
    if set(eval_counts) != set(by_id) or any(count != 4 for count in eval_counts.values()):
        problems.append("evaluator coverage is not 4-per-row")
    for candidate in eval_all:
        source = by_id.get(str(candidate.get("source_record_id") or ""))
        if source is None:
            problems.append(f"evaluator references missing source: {candidate.get('id')}")
            continue
        score = candidate.get("expected_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            problems.append(f"invalid expected evaluator score: {candidate.get('id')}")
        if candidate.get("expected_failure_type") is None:
            if compact_reconstruction_json(candidate.get("candidate_reconstruction") or {}) != compact_reconstruction_json(source["approved_reconstruction"]):
                problems.append(f"approved evaluator candidate differs from target: {candidate.get('id')}")

    result = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "memory_records": len(memories),
        "reviewed_rows": len(reviewed),
        "topic_families": len({row["topic_id"] for row in reviewed}),
        "split_counts": dict(split_counts),
        "decision_counts": dict(sorted(decision_counts.items())),
        "counterfactual_pairs": len(counterfactuals),
        "preference_counts": preference_counts,
        "evaluator_candidate_rows": len(eval_all),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the integrated Phase 9B reconstruction corpus.")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / "data" / "phase_9b_memory_reconstruction" / "reconstruction_corpus_v2_large",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    packet = validate(args.corpus_root)
    text = json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if packet["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
