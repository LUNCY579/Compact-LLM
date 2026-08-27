#!/usr/bin/env python3
"""Freeze outcome-blind real Phase-C samples and their exact token lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from rineya_pcsm.data import (
    encode_chat_example,
    stable_sample_ids,
    stream_selected_sft_rows,
)
from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, utc_now


def component_rows(components: list[dict[str, Any]]) -> list[str]:
    return [str(row) for component in components for row in component["row_ids"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=5)
    parser.add_argument("--maximum-tokens", type=int, default=40960)
    parser.add_argument("--salt", default="pcsm-phase-c-v1-sample")
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train = split["train_assignment"]
    selected = {
        "meta_train": stable_sample_ids(
            component_rows(train["meta_train"]),
            args.per_split,
            f"{args.salt}:meta",
        ),
        "mask_cal": stable_sample_ids(
            component_rows(train["mask_cal"]),
            args.per_split,
            f"{args.salt}:mask",
        ),
        "one_step_dev": stable_sample_ids(
            component_rows(split["one_step_dev"]["components"]),
            args.per_split,
            f"{args.salt}:dev",
        ),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train_shards = list((args.corpus_root / "data" / "sft" / "train").glob("*.jsonl"))
    validation_shards = list(
        (args.corpus_root / "data" / "sft" / "validation").glob("*.jsonl")
    )
    if not train_shards or not validation_shards:
        raise FileNotFoundError("Expected train and validation SFT shards")

    rows: dict[str, dict[str, Any]] = {}
    train_ids = set(selected["meta_train"]) | set(selected["mask_cal"])
    for row in stream_selected_sft_rows(train_shards, train_ids):
        rows[str(row["id"])] = row
    for row in stream_selected_sft_rows(
        validation_shards, set(selected["one_step_dev"])
    ):
        rows[str(row["id"])] = row

    sections: dict[str, list[dict[str, Any]]] = {}
    for section, row_ids in selected.items():
        records: list[dict[str, Any]] = []
        for row_id in row_ids:
            row = rows[row_id]
            input_ids, labels = encode_chat_example(
                tokenizer,
                row["messages"],
                maximum_tokens=args.maximum_tokens,
                example_id=row_id,
            )
            records.append(
                {
                    "row_id": row_id,
                    "token_count": len(input_ids),
                    "assistant_token_count": sum(label != -100 for label in labels),
                    "content_sha256": canonical_json_hash(row),
                }
            )
        sections[section] = records

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "selection_policy": "stable hash of pre-frozen row IDs; outcomes never read",
        "salt": args.salt,
        "split_manifest": {
            "path": str(args.split_manifest.resolve()),
            "sha256": sha256_file(args.split_manifest),
        },
        "model": str(args.model.resolve()),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "maximum_tokens": args.maximum_tokens,
        "sections": sections,
        "sealed_test_read": False,
    }
    manifest["sample_manifest_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "lengths": {
                    key: [row["token_count"] for row in value]
                    for key, value in sections.items()
                },
                "manifest_sha256": manifest["sample_manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
