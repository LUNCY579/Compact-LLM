"""Freeze the prospectively selected D64 exposure and masks without model imports."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import receipts as r

SALT = "cross-scale-composable-001/stage-d/seed-1729"
TRAIN_SHA = "0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1"
ID_SHA = "8b6f332821ff5f10f4b13b61577930eb5320c205be4c0005d46b01ed22550612"
META_SHA = "38942750da1650f96cad940697e51d86c601365f7514229234835babd2913ad8"
RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
TRAIN = RECORDS / "rrp015-20260917/control-prepared/microfit-train.jsonl"
EVALUATIONS = {
    RECORDS / "select-views-001/SELECT_VIEWS.jsonl": "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3",
    RECORDS / "process-transition-probe-001/CANDIDATE_VIEWS.jsonl": "45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e"}
CALIBRATION = Path(__file__).parent / "evidence/CALIBRATION_BINDING.v2.json"


def read_rows(path):
    return [json.loads(line, object_pairs_hook=r._unique_json_object) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_row(row):
    r.require(row.get("source_split") == "train" and row.get("source_release") == "forge_v2",
              "pilot requires existing open Forge train rows")
    r.require(row.get("truncated") is False and row.get("mode") == "direct", "truncated/non-direct training row")
    r.require(row.get("reasoning_content_tokens") == 0 and row.get("post_eos_tokens_supervised") is False,
              "training signal differs")
    r.require(isinstance(row.get("id"), str) and row["id"] and isinstance(row.get("parent_id"), str)
              and row["parent_id"], "training identity missing")
    ids, labels, prompt = row.get("ids"), row.get("labels"), row.get("prompt_ids")
    r.require(isinstance(ids, list) and 1 <= len(ids) <= 256
              and all(type(token) is int and 0 <= token < 151936 for token in ids), "invalid/overflowing training token IDs")
    r.require(isinstance(labels, list) and len(labels) == len(ids)
              and all(type(token) is int for token in labels), "invalid labels")
    start, end = row.get("answer_start"), row.get("answer_end")
    r.require(type(start) is int and type(end) is int and 0 < start <= end < len(ids), "invalid answer boundaries")
    r.require(prompt == ids[:start], "prompt is not causal training prefix")
    r.require(ids[end] == 151645, "registered EOS differs")
    expected_labels = [token if start <= i <= end else -100 for i, token in enumerate(ids)]
    r.require(labels == expected_labels, "completion mask differs or supervises prompt/post-EOS")
    r.require(row.get("processed_tokens") == len(ids) and row.get("supervised_tokens") == end - start + 1,
              "training token counters differ")
    r.require(row.get("final_content_tokens") == end - start, "completion counter differs")
    r.require(row.get("family") in ("ledger_replay", "register_machine", "prerequisite_plan"), "unknown task family")


def select(rows, excluded_parents):
    r.require(len(rows) == 384, "source must have exactly384 rows")
    r.require(len({row.get("id") for row in rows}) == 384 and
              len({row.get("parent_id") for row in rows}) == 384, "source duplicate row/parent")
    for row in rows:
        validate_row(row)
    ordered = sorted(rows, key=lambda row: (hashlib.sha256((SALT + ":" + row["id"]).encode()).hexdigest(), row["id"]))
    chosen = ordered[:64]
    r.require(not {row["parent_id"] for row in chosen} & set(excluded_parents), "training/calibration/evaluation parent overlap")
    metadata = [{"id": row["id"], "parent_id": row["parent_id"], "family": row["family"],
                 "prompt_ids_sha256": r.digest(row["prompt_ids"]), "ids_sha256": r.digest(row["ids"]),
                 "labels_sha256": r.digest(row["labels"]), "processed_tokens": len(row["ids"]),
                 "supervised_tokens": sum(value != -100 for value in row["labels"])} for row in chosen]
    return chosen, metadata, [row["id"] for row in ordered[64:]]


def prepare(output):
    output = Path(output)
    r.require(not output.exists(), "pilot manifest output already exists")
    inputs = {TRAIN: TRAIN_SHA, **EVALUATIONS, CALIBRATION: r.file_digest(CALIBRATION)}
    for path, sha in inputs.items():
        r.verify_file(path, sha)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    r.validate_calibration(calibration)
    parents = {row["parent_id"] for row in calibration["rows"]}
    exclusion_counts = {"calibration": len(parents)}
    for path in EVALUATIONS:
        # Metadata-only use; no reference, prompt, generated text or score is used.
        current = {row["parent_id"] for row in read_rows(path)}
        r.require(len(current) == 384, "evaluation parent inventory differs")
        exclusion_counts[path.name] = len(current)
        parents.update(current)
    chosen, metadata, unused = select(read_rows(TRAIN), parents)
    ordered_id_sha = r.digest([row["id"] for row in chosen])
    r.require(ordered_id_sha == ID_SHA and r.digest(metadata) == META_SHA, "D64 prospective selection differs")
    for path, sha in inputs.items():
        r.verify_file(path, sha)
    # Fixed before donor targets exist. Every target is used once, never on its own prompt.
    shuffle_indices = [(index + 31) % 64 for index in range(64)]
    shuffle_rows = [chosen[index]["id"] for index in shuffle_indices]
    manifest = {"schema": 2, "kind": "CSC_001_D64_OPEN_TRAINING_MANIFEST", "status": "OFFLINE_ROWS_FROZEN",
        "source_file": str(TRAIN), "source_sha256": TRAIN_SHA, "selection_salt": SALT,
        "input_files": {str(path.resolve()): sha for path, sha in inputs.items()},
        "preparation_source_sha256": r.file_digest(__file__),
        "source_rows": 384, "student_exposure_rows": 64, "student_optimizer_steps_per_arm": 64,
        "ordered_ids_sha256": ordered_id_sha, "metadata_sha256": r.digest(metadata),
        "training_rows_sha256": r.digest(chosen), "rows": chosen, "metadata": metadata,
        "unused_source_ids": unused, "unused_source_rows": 320,
        "family_counts": dict(Counter(row["family"] for row in chosen)),
        "processed_tokens_per_arm": sum(row["processed_tokens"] for row in chosen),
        "supervised_tokens_per_arm": sum(row["supervised_tokens"] for row in chosen),
        "max_complete_tokens": max(len(row["ids"]) for row in chosen),
        "max_prompt_tokens": max(len(row["prompt_ids"]) for row in chosen),
        "excluded_parent_counts": exclusion_counts, "excluded_parent_union_sha256": r.digest(sorted(parents)),
        "selected_parent_overlap": 0, "target_capture_forwards_proposed": 192,
        "training_arms_proposed": ["S0", "X0", "X_shuffle"], "optimizer_steps_all_training_arms_proposed": 192,
        "donor_specificity_control": {"kind": "fixed_target_derangement", "cyclic_offset": 31,
            "training_order_unchanged": True, "target_index_for_training_position": shuffle_indices,
            "target_row_ids": shuffle_rows, "target_row_ids_sha256": r.digest(shuffle_rows),
            "target_multiset_preserved": True, "fixed_points": 0,
            "loss_parameterization_steps_identical_to_X0": True},
        "primary_scalar_control_alpha": 0.5, "native_model_calls": 0, "optimizer_steps_executed": 0,
        "native_execution_ready": False,
        "pending": ["qualified Stage C final-pair map", "native training/gradient/merge qualification",
                    "fresh execution registration/source/runtime binding", "resource gates"]}
    sha = r.write_once(output, manifest)
    return {"status": manifest["status"], "path": str(output.resolve()), "sha256": sha,
            "training_rows_sha256": manifest["training_rows_sha256"], "ordered_ids_sha256": ordered_id_sha,
            "student_rows": 64, "processed_tokens": manifest["processed_tokens_per_arm"],
            "supervised_tokens": manifest["supervised_tokens_per_arm"], "native_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out)))


if __name__ == "__main__":
    main()
