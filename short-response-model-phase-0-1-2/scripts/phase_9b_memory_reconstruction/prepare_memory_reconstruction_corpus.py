from __future__ import annotations

"""Normalize an external Phase 9B memory-reconstruction corpus for repository use.

The normalizer preserves the source semantics while making split intent explicit,
recording exact counterfactual edits, and regenerating provenance hashes. It never
turns preference/evaluator negatives into supervised targets.
"""

import argparse
from collections import Counter
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INTEGRATION_REVISION = "short_response_model_v1_7_4"
PIPELINE_REVISION = "phase9b_memory_reconstruction_v1_7_4"


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
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256_bytes(data)}


def _split_of(row: dict[str, Any]) -> str:
    value = str(row.get("intended_split") or (row.get("metadata") or {}).get("split") or "").strip().lower()
    if value not in {"train", "dev", "test"}:
        raise ValueError(f"{row.get('id')}: missing train/dev/test split")
    return value


def normalize_reviewed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        split = _split_of(row)
        row["intended_split"] = split
        metadata = dict(row.get("metadata") or {})
        metadata["split"] = split
        metadata["integration_revision"] = INTEGRATION_REVISION
        metadata.setdefault("reference_is_example", True)
        row["metadata"] = metadata
        normalized.append(row)
    return normalized


def normalize_counterfactuals(rows: list[dict[str, Any]], reviewed_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        positive = reviewed_by_id.get(str(row.get("positive_record_id") or ""))
        edited = reviewed_by_id.get(str(row.get("edited_record_id") or ""))
        if positive is None or edited is None:
            raise ValueError(f"{row.get('pair_id')}: missing positive or edited record")
        left = positive.get("current_state") or {}
        right = edited.get("current_state") or {}
        keys = sorted(set(left) | set(right))
        actual = [key for key in keys if left.get(key) != right.get(key)]
        decisive = [str(item) for item in (row.get("changed_fields") or [])]
        row["decisive_fields"] = decisive
        row["changed_fields"] = actual
        row["supporting_changed_fields"] = [key for key in actual if key not in decisive]
        row["minimality_contract"] = (
            "single_decisive_edit" if set(actual) == set(decisive)
            else "minimal_decisive_edit_plus_state_coherence_fields"
        )
        normalized.append(row)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the dedicated Phase 9B reconstruction corpus.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Preserve source provenance before normalization.
    source_hashes_path = source / "manifests" / "file_hashes.json"
    if source_hashes_path.exists():
        (output / "manifests").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_hashes_path, output / "manifests" / "source_file_hashes.json")

    memories = read_jsonl(source / "memories.jsonl")
    reviewed = normalize_reviewed(read_jsonl(source / "reviewed_memory_reconstruction.jsonl"))
    reviewed_by_id = {row["id"]: row for row in reviewed}
    counterfactuals = normalize_counterfactuals(
        read_jsonl(source / "counterfactuals" / "counterfactual_pairs.jsonl"), reviewed_by_id
    )

    write_jsonl(output / "memories.jsonl", memories)
    write_jsonl(output / "reviewed_memory_reconstruction.jsonl", reviewed)
    write_jsonl(output / "counterfactuals" / "counterfactual_pairs.jsonl", counterfactuals)

    split_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = [row for row in reviewed if row["intended_split"] == split]
        split_counts[split] = len(rows)
        write_jsonl(output / "splits" / f"{split}.reviewed.jsonl", rows)

        # Preferences and evaluator candidates remain non-SFT assets.
        for directory, source_name, output_name in (
            ("preferences", f"{split}.jsonl", f"{split}.jsonl"),
            ("evaluator", f"{split}.candidate_reconstructions.jsonl", f"{split}.candidate_reconstructions.jsonl"),
        ):
            src = source / directory / source_name
            if src.exists():
                write_jsonl(output / directory / output_name, read_jsonl(src))

    evaluator_all = source / "evaluator" / "candidate_reconstructions.all.jsonl"
    if evaluator_all.exists():
        write_jsonl(output / "evaluator" / evaluator_all.name, read_jsonl(evaluator_all))

    decisions = Counter(row["approved_reconstruction"]["decision"] for row in reviewed)
    scenarios = Counter((row.get("metadata") or {}).get("scenario_type") for row in reviewed)
    topic_ids = sorted({str(row.get("topic_id") or "") for row in reviewed})
    manifest: dict[str, Any] = {
        "schema_version": "1.1",
        "name": "Phase 9B Memory Reconstruction Corpus v1.1 (v1.7.4 integration)",
        "source": "synthetic_phase9b_memharness_v1",
        "integration_revision": INTEGRATION_REVISION,
        "pipeline_revision": PIPELINE_REVISION,
        "source_root_name": source.name,
        "counts": {
            "memory_records": len(memories),
            "reviewed_rows": len(reviewed),
            "topic_families": len(topic_ids),
            "counterfactual_pairs": len(counterfactuals),
            "preference_pairs": sum(1 for split in ("train", "dev", "test") for _ in read_jsonl(output / "preferences" / f"{split}.jsonl")),
            "evaluator_candidate_rows": len(read_jsonl(output / "evaluator" / "candidate_reconstructions.all.jsonl")),
        },
        "split_counts": split_counts,
        "decision_counts": dict(sorted(decisions.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "split_policy": "authored topic-family split preserved via top-level intended_split",
        "training_boundary": {
            "sft_target": "approved_reconstruction and approved_answer only",
            "preference_negatives_are_sft_targets": False,
            "evaluator_candidates_are_sft_targets": False,
            "test_preferences_are_training_data": False,
            "test_evaluator_candidates_are_training_data": False,
        },
        "counterfactual_contract": (
            "decisive_fields identifies the causal edit; changed_fields records every concrete state field that changed. "
            "Some pairs add supporting coherence fields, so they are minimal in decisive cause rather than always one-field literal edits."
        ),
        "important_limit": (
            "This is supervised/cold-start reconstruction data. It does not reproduce the end-to-end GRPO result in MemHarness."
        ),
    }
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    (output / "manifests" / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# Phase 9B Memory Reconstruction Corpus v1.1 — integrated for model v1.7.4

This is the repository-native dedicated memory-reconstruction curriculum.

## Scale

- {len(memories)} external memory records
- {len(reviewed)} accepted canonical reconstruction rows
- {len(topic_ids)} topic families
- {len(counterfactuals)} counterfactual pairs
- decisions: `{dict(sorted(decisions.items()))}`
- authored splits: `{split_counts}`

## Canonical training boundary

Only `approved_reconstruction` and `approved_answer` are supervised targets. The
`preferences/` and `evaluator/` directories contain deliberate wrong/naive
alternatives and must never be concatenated into ordinary SFT.

## Split policy

The authored topic-family split is preserved with a top-level `intended_split`.
Topic families do not cross train/dev/test. Dev is used for reconstruction-policy
selection; test remains sealed until a single strategy is frozen.

## Counterfactual note

The source package described its counterfactuals as minimal edits. v1.7.4 makes
that claim precise: `decisive_fields` records the causal premise/goal edit, while
`changed_fields` records all concrete fields that changed. Some examples add a
small number of supporting state-coherence fields so the resulting state and
answer remain well-defined.

## Runtime role

Phase 9B is the primary consumer. Phase 10 uses the approved Phase 9B checkpoint
and can run a reconstruction-retention probe. Phase 11 can enable prompted or
trained two-pass reconstruction and benchmark it separately from ordinary RAG.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    # Hash every normalized asset except the hash file itself.
    hashes: dict[str, Any] = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "file_hashes.json"):
        rel = path.relative_to(output).as_posix()
        hashes[rel] = hash_file(path)
    (output / "manifests" / "file_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
