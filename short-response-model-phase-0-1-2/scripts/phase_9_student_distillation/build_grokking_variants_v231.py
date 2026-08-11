from __future__ import annotations

"""Create deterministic v2.3.1 training variants and non-sealed probes.

The script never touches any sealed split.  It operates only on artifacts
created by the already-completed v2.3 DEV/OOD run plus packaged DEV/OOD data.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    balanced_sample,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def assistant_ready(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    messages = [dict(message) for message in (out.get("messages") or [])]
    if messages and messages[-1].get("role") == "assistant":
        return out
    answer = str(out.get("approved_answer") or out.get("answer") or out.get("reference") or "").strip()
    if not messages or not answer:
        raise ValueError(f"{out.get('id')}: cannot make training target")
    messages.append({"role": "assistant", "content": answer})
    out["messages"] = messages
    out.setdefault("reference", answer)
    return out


def repeat_exact(pool: list[dict[str, Any]], n: int, *, seed: int, max_repeat: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    if not pool:
        raise ValueError("cannot sample from empty pool")
    if n > len(pool) * max_repeat:
        raise ValueError(f"requested {n} rows from {len(pool)} rows exceeds repeat cap {max_repeat}")
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    base = list(pool)
    while len(out) < n:
        shuffled = list(base)
        rng.shuffle(shuffled)
        out.extend(shuffled[: min(len(shuffled), n - len(out))])
    return out


def stamp(rows: list[dict[str, Any]], *, variant: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = copy.deepcopy(row)
        meta = dict(item.get("metadata") or {})
        meta["v231_grokking_variant"] = variant
        meta["v231_exposure_index"] = index
        item["metadata"] = meta
        out.append(item)
    return out


def build_mix(
    residual: list[dict[str, Any]],
    preservation: list[dict[str, Any]],
    *,
    total: int,
    residual_fraction: float,
    seed: int,
    max_repeat: int,
    variant: str,
    residual_unique_limit: int | None = None,
    preservation_unique_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rpool = list(residual)
    ppool = list(preservation)
    if residual_unique_limit is not None and residual_unique_limit < len(rpool):
        rng.shuffle(rpool)
        rpool = rpool[:residual_unique_limit]
    if preservation_unique_limit is not None and preservation_unique_limit < len(ppool):
        rng.shuffle(ppool)
        ppool = ppool[:preservation_unique_limit]
    nres = round(total * residual_fraction)
    npres = total - nres
    nres = min(nres, len(rpool) * max_repeat)
    npres = min(npres, len(ppool) * max_repeat)
    total_realized = nres + npres
    rows = repeat_exact(rpool, nres, seed=seed + 1, max_repeat=max_repeat)
    rows += repeat_exact(ppool, npres, seed=seed + 2, max_repeat=max_repeat)
    rng.shuffle(rows)
    rows = stamp(rows, variant=variant)
    return rows, {
        "variant": variant,
        "rows": len(rows),
        "requested_rows": total,
        "residual_exposures": nres,
        "preservation_exposures": npres,
        "residual_unique_pool": len(rpool),
        "preservation_unique_pool": len(ppool),
        "residual_fraction_realized": (nres / total_realized) if total_realized else 0.0,
        "max_repeat": max_repeat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reasoning-residual", type=Path, required=True)
    ap.add_argument("--preservation", type=Path, required=True)
    ap.add_argument("--original-v23-mix", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--ood-dev", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=23131)
    args = ap.parse_args()

    cfg = read_json(args.config)
    residual = [assistant_ready(row) for row in read_jsonl(args.reasoning_residual)]
    preservation = [assistant_ready(row) for row in read_jsonl(args.preservation)]
    original_mix = [assistant_ready(row) for row in read_jsonl(args.original_v23_mix)]
    dev = read_jsonl(args.dev)
    ood = read_jsonl(args.ood_dev)
    if not residual or not preservation:
        raise ValueError("v2.3 residual/preservation pools are required")

    root = args.output_root
    variants_root = root / "training_variants"
    probes_root = root / "probes"
    variants_root.mkdir(parents=True, exist_ok=True)
    probes_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "2.3.1",
        "sealed_data_used": False,
        "source_counts": {
            "reasoning_residual": len(residual),
            "preservation": len(preservation),
            "original_v23_mix": len(original_mix),
            "dev": len(dev),
            "ood_dev": len(ood),
        },
        "variants": {},
        "probes": {},
    }

    # Exact v2.3 mixed view is a first-class reference.
    baseline_path = variants_root / "baseline_original_v23.reviewed.jsonl"
    write_jsonl(baseline_path, stamp(original_mix, variant="baseline_original_v23"))
    baseline_res = sum(str((row.get("metadata") or {}).get("v23_role") or "").startswith("residual") for row in original_mix)
    manifest["variants"]["baseline_original_v23"] = {
        "path": str(baseline_path),
        "rows": len(original_mix),
        "residual_exposures": baseline_res,
        "preservation_exposures": len(original_mix) - baseline_res,
        "kind": "source_exact",
    }

    base_total = len(original_mix)
    sweep = cfg["factor_sweep"]
    max_repeat = int(cfg["exposure_control"]["maximum_repeat_factor"])

    for fraction in sweep["data_fraction"]:
        fraction = float(fraction)
        name = f"data_fraction_{fraction:.2f}".replace(".", "p")
        rows, info = build_mix(
            residual,
            preservation,
            total=max(32, round(base_total * fraction)),
            residual_fraction=float(cfg["core_training"]["residual_fraction"]),
            seed=args.seed + round(fraction * 1000),
            max_repeat=max_repeat,
            variant=name,
        )
        path = variants_root / f"{name}.reviewed.jsonl"
        write_jsonl(path, rows)
        manifest["variants"][name] = {"path": str(path), "kind": "data_fraction", **info}

    for fraction in sweep["residual_fraction"]:
        fraction = float(fraction)
        name = f"residual_fraction_{fraction:.2f}".replace(".", "p")
        rows, info = build_mix(
            residual,
            preservation,
            total=base_total,
            residual_fraction=fraction,
            seed=args.seed + 10000 + round(fraction * 1000),
            max_repeat=max_repeat,
            variant=name,
        )
        path = variants_root / f"{name}.reviewed.jsonl"
        write_jsonl(path, rows)
        manifest["variants"][name] = {"path": str(path), "kind": "residual_fraction", **info}

    # Exposure-matched variants: residual exposure count is held fixed while the
    # number of unique residual cases changes. Preservation exposure changes only
    # by the configured fixed ratio.
    fixed_res = min(
        int(cfg["exposure_control"]["fixed_residual_exposures"]),
        len(residual) * max_repeat,
    )
    preservation_ratio = float(cfg["exposure_control"]["preservation_to_residual_ratio"])
    fixed_pres = round(fixed_res * preservation_ratio)
    for raw in sweep["unique_residual_cases"]:
        unique = len(residual) if raw == "all" else min(int(raw), len(residual))
        possible = unique * max_repeat
        residual_exposures = min(fixed_res, possible)
        # Keep the exposure ratio fixed after respecting the uniqueness cap.
        preservation_exposures = round(residual_exposures * preservation_ratio)
        total = residual_exposures + preservation_exposures
        residual_fraction = residual_exposures / total if total else 0.0
        name = f"unique_residual_{unique}"
        rows, info = build_mix(
            residual,
            preservation,
            total=total,
            residual_fraction=residual_fraction,
            seed=args.seed + 20000 + unique,
            max_repeat=max_repeat,
            variant=name,
            residual_unique_limit=unique,
        )
        path = variants_root / f"{name}.reviewed.jsonl"
        write_jsonl(path, rows)
        manifest["variants"][name] = {
            "path": str(path),
            "kind": "unique_exposure_control",
            "fixed_residual_exposure_target": fixed_res,
            "fixed_preservation_exposure_target": fixed_pres,
            **info,
        }

    probe_cfg = cfg["probes"]
    train_answerable = [row for row in residual if str(row.get("decision") or "answer") == "answer"]
    train_probe = balanced_sample(
        train_answerable,
        min(int(probe_cfg["train_answerable_cases"]), len(train_answerable)),
        seed=args.seed + 30001,
        group_fields=("reasoning_family", "domain", "difficulty"),
    )

    def split_probe(rows: list[dict[str, Any]], *, ans_n: int, abst_n: int, seed: int) -> list[dict[str, Any]]:
        answerable = [row for row in rows if str(row.get("decision") or "answer") == "answer"]
        abst = [row for row in rows if str(row.get("decision") or "answer") != "answer"]
        chosen = balanced_sample(answerable, min(ans_n, len(answerable)), seed=seed, group_fields=("reasoning_family", "domain", "difficulty"))
        chosen += balanced_sample(abst, min(abst_n, len(abst)), seed=seed + 1, group_fields=("reasoning_family", "domain"))
        random.Random(seed + 2).shuffle(chosen)
        return chosen

    dev_probe = split_probe(
        dev,
        ans_n=int(probe_cfg["dev_answerable_cases"]),
        abst_n=int(probe_cfg["dev_unanswerable_cases"]),
        seed=args.seed + 31000,
    )
    ood_probe = split_probe(
        ood,
        ans_n=int(probe_cfg["ood_answerable_cases"]),
        abst_n=int(probe_cfg["ood_unanswerable_cases"]),
        seed=args.seed + 32000,
    )
    rep_train = balanced_sample(
        train_answerable,
        min(int(probe_cfg["representation_cases_per_split"]), len(train_answerable)),
        seed=args.seed + 33000,
        group_fields=("reasoning_family",),
    )
    ood_answerable = [row for row in ood if str(row.get("decision") or "answer") == "answer"]
    rep_ood = balanced_sample(
        ood_answerable,
        min(int(probe_cfg["representation_cases_per_split"]), len(ood_answerable)),
        seed=args.seed + 34000,
        group_fields=("reasoning_family",),
    )

    for name, rows in {
        "train_answerable": train_probe,
        "dev": dev_probe,
        "ood": ood_probe,
        "representation_train": rep_train,
        "representation_ood": rep_ood,
    }.items():
        path = probes_root / f"{name}.reviewed.jsonl"
        write_jsonl(path, rows)
        manifest["probes"][name] = {
            "path": str(path),
            "rows": len(rows),
            "answerable": sum(str(row.get("decision") or "answer") == "answer" for row in rows),
            "unanswerable": sum(str(row.get("decision") or "answer") != "answer" for row in rows),
            "surface_formats": sorted({str(row.get("surface_format") or "") for row in rows}),
            "reasoning_families": sorted({str(row.get("reasoning_family") or "") for row in rows}),
        }

    # Explicitly prove no fresh sealed path was passed into this script.
    all_source_strings = [str(args.reasoning_residual), str(args.preservation), str(args.original_v23_mix), str(args.dev), str(args.ood_dev)]
    if any("sealed" in value.casefold() for value in all_source_strings):
        raise ValueError("v2.3.1 variant builder refuses sealed inputs")

    write_json(root / "grokking_data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
