#!/usr/bin/env python3
"""Freeze leakage-resistant META_TRAIN, MASK_CAL, and ONE_STEP_DEV IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rineya_pcsm.data import (
    assign_components,
    assignment_manifest,
    connected_components,
    load_lineage,
    select_outcome_blind_components,
)
from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-components", type=int, default=20)
    parser.add_argument("--mask-components", type=int, default=5)
    parser.add_argument("--dev-components", type=int, default=1)
    parser.add_argument("--salt", default="pcsm-phase-c-v1-frozen")
    args = parser.parse_args()

    lineage_root = args.corpus_root.resolve() / "reports" / "main_lineage"
    train_path = lineage_root / "train.jsonl"
    validation_path = lineage_root / "validation.jsonl"
    # Intentionally do not resolve or open test.jsonl here. It is sealed.
    train = load_lineage(train_path, "train")
    assignment = assign_components(
        connected_components(train),
        meta_train_count=args.meta_components,
        mask_cal_count=args.mask_components,
        split_salt=args.salt,
    )
    validation = load_lineage(validation_path, "validation")
    dev = select_outcome_blind_components(
        validation,
        component_count=args.dev_components,
        salt=f"{args.salt}:one-step-dev",
    )
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "selection_policy": (
            "semantic connected components; stable hash; no outcomes or sealed data"
        ),
        "train_assignment": assignment_manifest(assignment, train_path, args.salt),
        "one_step_dev": {
            "source": "validation",
            "lineage_path": str(validation_path),
            "lineage_sha256": sha256_file(validation_path),
            "components": [component.__dict__ for component in dev],
        },
        "sealed": {
            "source": "test",
            "read_during_split_build": False,
            "policy": "unavailable until every model and hyperparameter choice freezes",
        },
    }
    manifest["split_manifest_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "meta_components": len(assignment.meta_train),
                "mask_components": len(assignment.mask_cal),
                "unused_train_components": len(assignment.unused),
                "dev_components": len(dev),
                "meta_rows": sum(len(item.row_ids) for item in assignment.meta_train),
                "mask_rows": sum(len(item.row_ids) for item in assignment.mask_cal),
                "dev_rows": sum(len(item.row_ids) for item in dev),
                "manifest_sha256": manifest["split_manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
