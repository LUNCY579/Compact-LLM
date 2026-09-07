#!/usr/bin/env python3
"""Filter CC0 SFT shards into train-8K + full-stratum val/test views."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
HALO_SCRIPTS = REPO_ROOT / "rineya_halo_long_context_sft_v1" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import atomic_write_json, load_yaml, resolve_repo_path, write_jsonl  # noqa: E402

_spec = importlib.util.spec_from_file_location("halo_common", HALO_SCRIPTS / "common.py")
assert _spec is not None and _spec.loader is not None
halo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(halo)


def load_split(
    corpus_root: Path,
    split: str,
    *,
    keep_strata: set[str] | None,
    lineage: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Counter]:
    shard_dir = corpus_root / "data" / "sft" / split
    kept: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in halo.iter_shards(shard_dir):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        if messages[-1].get("role") != "assistant":
            continue
        stratum = halo.infer_stratum(row, lineage)
        if stratum is None:
            continue
        counts[stratum] += 1
        if keep_strata is not None and stratum not in keep_strata:
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["context_stratum"] = stratum
        out = dict(row)
        out["metadata"] = metadata
        if "id" not in out:
            raise ValueError(f"{split} row missing id")
        kept.append(out)
    return kept, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_qlora_cc0_8k.yaml",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    corpus_root = resolve_repo_path(data_cfg["corpus_root"])
    output_dir = resolve_repo_path(train_cfg["output_dir"]) / "views"
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = corpus_root / "reports" / "lineage.jsonl"
    lineage = halo.load_lineage(lineage_path) if lineage_path.exists() else None

    train_strata = {str(s).lower() for s in data_cfg["strata"]}
    train_rows, train_all_counts = load_split(
        corpus_root, "train", keep_strata=train_strata, lineage=lineage
    )
    if not train_rows:
        raise RuntimeError(
            f"zero train rows for strata {sorted(train_strata)}; "
            f"raw stratum counts={dict(train_all_counts)}"
        )

    val_keep = None if data_cfg.get("val_include_all_strata", True) else train_strata
    test_keep = None if data_cfg.get("test_include_all_strata", True) else train_strata
    val_rows, val_all_counts = load_split(
        corpus_root, "validation", keep_strata=val_keep, lineage=lineage
    )
    test_rows, test_all_counts = load_split(
        corpus_root, "test", keep_strata=test_keep, lineage=lineage
    )

    train_path = output_dir / "train_8k.jsonl"
    val_path = output_dir / "val_all_strata.jsonl"
    test_path = output_dir / "test_all_strata.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    summary = {
        "corpus_root": str(corpus_root),
        "train_strata": sorted(train_strata),
        "train_rows": len(train_rows),
        "train_raw_stratum_counts": dict(train_all_counts),
        "val_rows": len(val_rows),
        "val_raw_stratum_counts": dict(val_all_counts),
        "test_rows": len(test_rows),
        "test_raw_stratum_counts": dict(test_all_counts),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "excluded_from_train": list(data_cfg.get("exclude_strata") or []),
    }
    atomic_write_json(output_dir / "view_manifest.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
