"""Read-back release audits; no tokenizer, model inference, or optimizer is invoked."""
from __future__ import annotations

import argparse
import gzip
import io
import unicodedata
import zipfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .common import SPLITS, digest, dumps, loads, read_jsonl, sha_file, write_json
from .oracle import audit_answer, verify_solution
from .rewards import score_response
from .text import LAYOUTS, canonical_case, parse_prompt
from .validate import require, verify_manifest


def verify_probes(root: Path) -> dict[str, Any]:
    """Reparse every stored diagnostic and validate dependent-parent relationships."""
    checked = verify_manifest(root, "PROBE_MANIFEST.json")
    parents = {r["id"]: r for r in read_jsonl(root / "metadata/test_iid.jsonl.gz")}
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    seen: set[str] = set()
    rows = zip_longest(read_jsonl(root / "probes/views.jsonl.gz"),
                       read_jsonl(root / "probes/metadata.jsonl.gz"))
    count = 0
    for row, meta in rows:
        require(row is not None and meta is not None, "Unaligned probe data and references")
        require(set(row) == {"id", "group_id", "family", "split", "variant", "prompt"}, "Probe prompt has unexpected fields")
        require(row["id"] == meta["id"] and row["id"] not in seen, "Duplicate or misaligned probe ID")
        seen.add(row["id"])
        parent_id = row["group_id"]
        require(parent_id in parents, "Probe has no IID-test parent")
        parent = parents[parent_id]
        require(row["split"] == meta["split"] == parent["split"] == "test_iid", "Probe split leak")
        require(row["family"] == meta["family"] == parent["family"] and parent_id == meta["group_id"], "Probe family/group mismatch")
        variant = row["variant"]
        require(variant == meta["variant"] and variant in ("short", *LAYOUTS, "evidence_edit"), "Unknown probe variant")
        require(variant not in groups[parent_id], "Repeated parent/variant")
        world = parse_prompt(row["prompt"])
        require(digest(row["prompt"]) == meta["prompt_sha256"], "Probe prompt hash mismatch")
        require(canonical_case(world) == canonical_case(meta["world"]), "Probe parsed world mismatch")
        require(meta["certificate"] == audit_answer(world), "Probe oracle disagreement")
        require(verify_solution(world, meta["reference"], meta["certificate"])["correct"], "Invalid probe reference")
        require(score_response(world, dumps(meta["reference"]), certificate=meta["certificate"])["reward"] == 1, "Probe reward mismatch")
        old_still_correct = verify_solution(world, parent["reference"], meta["certificate"])["correct"]
        if variant == "evidence_edit":
            require(not old_still_correct, "Edited world still accepts original reference")
        else:
            require(old_still_correct and canonical_case(world) == canonical_case(parent["world"]), "Layout changed the formal task")
        groups[parent_id][variant] = row
        count += 1
    for views in groups.values():
        require(set(views) == {"short", *LAYOUTS, "evidence_edit"}, "Incomplete diagnostic group")
        multisets = [sorted(line for line in views[v]["prompt"][1]["content"].splitlines()
                            if line.startswith("Record ")) for v in LAYOUTS]
        require(all(records == multisets[0] for records in multisets), "Layout diagnostic changed record multiset")
    require(count > 0, "No probes to verify")
    return {"status": "PASS", "manifest_files_verified": checked, "views_checked": count,
            "parent_worlds": len(groups), "old_answers_invalidated_by_all_edits": True,
            "all_four_long_layouts_record_multiset_matched": True, "model_calls": 0}


def _normalized_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def old_prompt_hashes(archive: Path) -> tuple[set[str], int]:
    """Read v1 canonical user prompts only, without extracting archive members."""
    hashes: set[str] = set()
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for split in ("train", "validation", "test_iid", "test_transfer", "test_length"):
            suffix = f"/data/{split}.jsonl.gz"
            names = [name for name in bundle.namelist() if name.endswith(suffix)]
            require(len(names) == 1, f"Expected one v1 canonical {split} file")
            with bundle.open(names[0]) as compressed:
                with gzip.GzipFile(fileobj=compressed) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8") as text:
                        for line in text:
                            row = loads(line)
                            user = [m["content"] for m in row["messages"] if m["role"] == "user"]
                            require(len(user) == 1, "Unexpected v1 conversation shape")
                            hashes.add(digest(_normalized_prompt(user[0])))
                            count += 1
    return hashes, count


def audit_release(root: Path, original: Path | None = None) -> dict[str, Any]:
    verify_manifest(root)
    old, old_count = old_prompt_hashes(original) if original else (set(), 0)
    normalized: set[str] = set()
    overlap = 0
    totals = 0
    maxima: dict[str, Any] = {}
    for split in SPLITS:
        maximum = {"reason_response_characters": 0, "final_response_characters": 0, "maximum_reason_row_id": None}
        for row, meta in zip_longest(read_jsonl(root / "data/rl" / f"{split}.jsonl.gz"),
                                     read_jsonl(root / "metadata" / f"{split}.jsonl.gz")):
            require(row is not None and meta is not None and row["id"] == meta["id"], "Audit row misalignment")
            key = digest(_normalized_prompt(row["prompt"][1]["content"]))
            require(key not in normalized, "Duplicate whitespace/Unicode-normalized user prompt")
            normalized.add(key)
            overlap += key in old
            text = "<think>\n" + meta["teacher_trace"] + "\n</think>\n<final>" + meta["reference_text"] + "</final>"
            result = score_response(meta["world"], text, mode="reason_then_final", certificate=meta["certificate"])
            require(result["reward"] == 1, "Full-split reason-mode gold scoring failed")
            if len(text) > maximum["reason_response_characters"]:
                maximum["reason_response_characters"] = len(text)
                maximum["maximum_reason_row_id"] = row["id"]
            maximum["final_response_characters"] = max(maximum["final_response_characters"], len(meta["reference_text"]))
            totals += 1
        maxima[split] = maximum
    labels = Counter(str(r["valid"]).lower() for r in read_jsonl(root / "process/train_transitions.jsonl.gz"))
    categories = Counter(r["category"] for r in read_jsonl(root / "preferences/train_pairs.jsonl.gz"))
    require(labels["true"] == labels["false"] and labels["true"] > 0, "Process pair balance failure")
    if original:
        require(overlap == 0, "Original/new normalized exact prompt overlap")
    return {"status": "PASS", "canonical_reason_mode_references_scored": totals,
            "gold_scoring_is_software_test_not_model_evaluation": True,
            "normalized_unique_user_prompts": len(normalized), "response_character_maxima": maxima,
            "process_labels": dict(labels), "preference_categories": dict(categories),
            "source_overlap": {"performed": original is not None, "original_rows": old_count if original else None,
                "original_sha256": sha_file(original) if original else None,
                "matching_normalized_prompts": overlap if original else None,
                "normalization": "NFKC then collapse whitespace; preserve case",
                "semantic_overlap_tested": False},
            "probes": verify_probes(root), "model_calls": 0, "optimizer_steps": 0,
            "native_tokenizer_run": False, "human_review": "NOT_PERFORMED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Refusing to overwrite audit receipt")
    result = audit_release(args.root, args.original)
    if args.output:
        write_json(args.output, result)
    print(dumps(result))


if __name__ == "__main__":
    main()
