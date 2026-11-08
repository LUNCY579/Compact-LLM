"""Read-only checks on selected TRAIN worlds against open historical instruments."""
from __future__ import annotations
import argparse
from collections import Counter
import re
import unicodedata
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, read_json, write_new
from .prepare import read_rows
from .sources import stable_hash


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def keys(row):
    messages = row.get("prompt", row.get("messages"))
    text = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    normal = normalized(text)
    return {"exact_text": digest(text), "normalized_text": digest(normal),
            "digit_blinded_text": digest(re.sub(r"\d", "#", normal))}


def audit(prepared: Path, source_v1: Path, historical_root: Path) -> dict:
    contract = read_json(prepared / "DATA_CONTRACT.json")
    for name, sha in contract["files"].items():
        if stable_hash(prepared / name) != sha:
            raise ValueError("changed prepared data")
    selected = list(read_rows(prepared / "microfit-train.jsonl"))
    if len(selected) != 32 or any(r["source_split"] != "train" for r in selected):
        raise ValueError("audit requires the exact TRAIN-only microfit")
    ids = {r["parent_id"] for r in selected}
    cases = {r["lineage"]["case_sha256"] for r in selected}
    text_keys = {name: {keys(r)[name] for r in selected} for name in keys(selected[0])}
    results = []
    for split in ("validation", "test_iid", "test_transfer", "test_length"):
        p = source_v1 / "data" / (split + ".jsonl.gz")
        meta_path = source_v1 / "metadata" / (split + ".jsonl.gz")
        counts = Counter(); count = 0
        from itertools import zip_longest
        for row, meta in zip_longest(read_rows(p), read_rows(meta_path)):
            if row is None or meta is None or row["id"] != meta["id"] or meta["split"] != split:
                raise ValueError("heldout source alignment failed")
            count += 1
            counts["parent_id"] += meta["group_id"] in ids
            counts["canonical_case"] += meta["case_sha256"] in cases
            for name, value in keys(row).items(): counts[name] += value in text_keys[name]
        results.append({"bank": "forge_v1/" + split, "rows": count, "input_sha256": stable_hash(p),
                        "metadata_sha256": stable_hash(meta_path), "collisions": dict(counts)})
    historical = [historical_root / "experiments/architecture_benchmark_001/data/monitor_probe.jsonl",
                  historical_root.parent / "mq-data-003/dev.jsonl"]
    for p in historical:
        counts = Counter(); count = 0
        for row in read_rows(p):
            count += 1
            counts["parent_id"] += row["id"] in ids
            for name, value in keys(row).items(): counts[name] += value in text_keys[name]
        results.append({"bank": str(p), "rows": count, "input_sha256": stable_hash(p), "collisions": dict(counts)})
    return {"experiment": "RRP-001", "scope": "Selected32 TRAIN versus declared open historical banks",
            "data_contract_sha256": stable_hash(prepared / "DATA_CONTRACT.json"), "banks": results,
            "all_listed_keys_disjoint": all(not any(r["collisions"].values()) for r in results),
            "generalization_axis": "Microfit training memorization only; Forge holdouts share grammar/family",
            "near_duplicate_semantic_audit": "PENDING", "structural_skeleton_disjointness": "NOT_CLAIMED",
            "independent_confirmation": "NOT_ESTABLISHED", "test_answers_exposed_in_report": False,
            "training_permission": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "source-v1", "historical-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args(argv)
    result = audit(args.prepared, args.source_v1, args.historical_root)
    write_new(args.out, result)
    print(result)


if __name__ == "__main__": main()
