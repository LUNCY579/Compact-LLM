#!/usr/bin/env python3
"""Build Stage A Halo SFT mixes: CC0 2k/4k shards plus short-response replay."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (
    LAB_ROOT,
    atomic_write_json,
    infer_stratum,
    iter_jsonl,
    iter_shards,
    load_config,
    load_lineage,
    resolve_repo_path,
    stage_paths,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Halo long-context Stage A training mixes.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "stage_a_qlora.yaml")
    parser.add_argument("--max-long-rows", type=int, default=0, help="Cap long-context rows per split (smoke).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override paths.output_dir (smoke).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_split_rows(
    corpus_root: Path,
    split: str,
    strata: set[str],
    lineage: dict[str, dict[str, Any]] | None,
    cap: int = 0,
) -> list[dict[str, Any]]:
    shard_dir = corpus_root / "data" / "sft" / split
    kept: list[dict[str, Any]] = []
    for row in iter_shards(shard_dir):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            continue
        stratum = infer_stratum(row, lineage)
        if stratum not in strata:
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["context_stratum"] = stratum
        metadata["mix_source"] = "long_context"
        out = dict(row)
        out["metadata"] = metadata
        kept.append(out)
        if cap > 0 and len(kept) >= cap:
            break
    return kept


def load_short_replay(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            continue
        metadata = dict(row.get("metadata") or {})
        metadata["context_stratum"] = "short"
        metadata["mix_source"] = "short_replay"
        out = dict(row)
        out["metadata"] = metadata
        rows.append(out)
    return rows


def sample_replay(short_rows: list[dict[str, Any]], n_long: int, fraction: float, rng: random.Random) -> list[dict[str, Any]]:
    if fraction <= 0 or n_long == 0 or not short_rows:
        return []
    if fraction >= 1:
        raise ValueError("replay_fraction must be in [0, 1)")
    target = int(round((fraction / (1.0 - fraction)) * n_long))
    target = max(0, min(target, len(short_rows)))
    if target == 0:
        return []
    if target == len(short_rows):
        chosen = list(short_rows)
        rng.shuffle(chosen)
        return chosen
    return rng.sample(short_rows, target)


def write_split(
    path: Path,
    rows: list[dict[str, Any]],
    rng: random.Random,
    *,
    force: bool,
) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return write_jsonl(path, shuffled)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config = dict(config)
        config["paths"] = dict(config["paths"])
        config["paths"]["output_dir"] = str(args.output_dir)
    paths = stage_paths(config)
    corpus_root = resolve_repo_path(config["paths"]["corpus_root"])
    replay_path = resolve_repo_path(config["paths"]["short_replay"])
    strata = {str(item).lower() for item in config["strata"]}
    seed = int(config["seed"])
    replay_fraction = float(config["replay_fraction"])
    trainer_eval_examples = int(config.get("trainer_eval_examples") or 0)

    lineage_dir = corpus_root / "reports" / "main_lineage"
    lineages: dict[str, dict[str, dict[str, Any]]] = {}
    cap = int(args.max_long_rows or 0)
    for split in ("train", "validation", "test"):
        lineage_path = lineage_dir / f"{split}.jsonl"
        if cap > 0:
            lineages[split] = {}
        else:
            lineages[split] = load_lineage(lineage_path) if lineage_path.exists() else {}

    rng = random.Random(seed)
    train_long = load_split_rows(corpus_root, "train", strata, lineages["train"], cap=cap)
    val_long = load_split_rows(corpus_root, "validation", strata, lineages["validation"], cap=cap)
    test_long = load_split_rows(corpus_root, "test", strata, lineages["test"], cap=cap)

    short_rows = load_short_replay(replay_path)
    replay = sample_replay(short_rows, len(train_long), replay_fraction, random.Random(seed + 3))
    train_mix = train_long + replay

    force = bool(args.force)
    n_train = write_split(paths["train_mix"], train_mix, random.Random(seed + 4), force=force)
    n_val = write_split(paths["val_mix"], val_long, random.Random(seed + 5), force=force)
    n_test = write_split(paths["test_mix"], test_long, random.Random(seed + 6), force=force)

    trainer_val = list(val_long)
    random.Random(seed + 7).shuffle(trainer_val)
    if trainer_eval_examples > 0:
        trainer_val = trainer_val[:trainer_eval_examples]
    if paths["val_trainer"].exists() and not force:
        raise FileExistsError(f"{paths['val_trainer']} exists; pass --force to overwrite")
    write_jsonl(paths["val_trainer"], trainer_val)

    def stratum_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for row in rows:
            metadata = row.get("metadata") or {}
            counter[str(metadata.get("context_stratum", "unknown"))] += 1
        return dict(sorted(counter.items()))

    actual_replay_fraction = (len(replay) / n_train) if n_train else 0.0
    manifest = {
        "stage": config.get("stage"),
        "seed": seed,
        "requested_replay_fraction": replay_fraction,
        "actual_replay_fraction": actual_replay_fraction,
        "strata": sorted(strata),
        "max_long_rows": args.max_long_rows,
        "sources": {
            "corpus_root": str(corpus_root),
            "short_replay": str(replay_path),
            "lineage_used": {split: bool(lineages[split]) for split in lineages},
        },
        "counts": {
            "train_long": len(train_long),
            "train_replay": len(replay),
            "train_total": n_train,
            "validation_long": n_val,
            "validation_trainer": len(trainer_val),
            "test_long": n_test,
            "short_replay_pool": len(short_rows),
        },
        "by_stratum": {
            "train": stratum_counts(train_mix),
            "validation": stratum_counts(val_long),
            "test": stratum_counts(test_long),
        },
        "outputs": {
            "train": str(paths["train_mix"]),
            "validation": str(paths["val_mix"]),
            "validation_trainer": str(paths["val_trainer"]),
            "test": str(paths["test_mix"]),
        },
        "note": "Never includes sealed_eval. Halo vault weights are not written.",
    }
    atomic_write_json(paths["mix_manifest"], manifest)
    if args.output_dir is None:
        report_copy = LAB_ROOT / "reports" / "stage_a_mix_summary.json"
        atomic_write_json(report_copy, {
            "counts": manifest["counts"],
            "by_stratum": manifest["by_stratum"],
            "actual_replay_fraction": actual_replay_fraction,
            "seed": seed,
            "strata": sorted(strata),
        })
    print(
        f"Stage A mix: train={n_train} (long={len(train_long)} replay={len(replay)} "
        f"frac={actual_replay_fraction:.3f}) val={n_val} test={n_test}"
    )
    print(f"Manifest: {paths['mix_manifest']}")


if __name__ == "__main__":
    main()
