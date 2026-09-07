#!/usr/bin/env python3
"""Build easy-only Puzzle think_then_short views for the layer-8 smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import load_yaml, resolve_repo_path, write_jsonl  # noqa: E402

PUZZLE_SCRIPTS = REPO_ROOT / "Rineya-Puzzle-Reasoning-51k" / "scripts"
if str(PUZZLE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUZZLE_SCRIPTS))

import build_training_views as views  # noqa: E402


def filter_rows(
    path: Path,
    *,
    difficulties: set[str],
    cap: int | None,
    seed: int,
) -> list[dict]:
    import random

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("difficulty") not in difficulties:
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["id"])
    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        rows = rng.sample(rows, cap)
        rows.sort(key=lambda r: r["id"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_lora_puzzle_smoke.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    puzzle_root = resolve_repo_path(data_cfg["puzzle_root"])
    out_dir = args.output_dir or (
        resolve_repo_path(train_cfg["output_dir"]) / "views"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    difficulties = set(data_cfg["difficulty"])
    view = data_cfg["view"]
    tags = data_cfg["reasoning_tags"]
    seed = int(train_cfg["seed"])

    train_rows = filter_rows(
        puzzle_root / "data" / "train.jsonl",
        difficulties=difficulties,
        cap=int(data_cfg["train_smoke_cap"]),
        seed=seed,
    )
    val_rows = filter_rows(
        puzzle_root / "data" / "validation.jsonl",
        difficulties=difficulties,
        cap=int(data_cfg["val_eval_cap"]),
        seed=seed + 1,
    )

    train_out = [
        views.convert(row, view, tags) for row in train_rows
    ]
    val_out = [
        views.convert(row, view, tags) for row in val_rows
    ]
    # Gold subset for exact-match eval (base schema, not chat view).
    gold_out = val_rows

    train_path = out_dir / "train_easy_think_then_short.jsonl"
    val_path = out_dir / "val_easy_think_then_short.jsonl"
    gold_path = out_dir / "val_easy_gold.jsonl"
    write_jsonl(train_path, train_out)
    write_jsonl(val_path, val_out)
    write_jsonl(gold_path, gold_out)

    summary = {
        "view": view,
        "difficulties": sorted(difficulties),
        "train_rows": len(train_out),
        "val_rows": len(val_out),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "gold_path": str(gold_path),
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "view_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
