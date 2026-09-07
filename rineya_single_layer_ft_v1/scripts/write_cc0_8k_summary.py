#!/usr/bin/env python3
"""Combine CC0 / ablation / puzzle dual metrics into layer8_cc0_8k_summary.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import atomic_write_json, load_yaml, resolve_repo_path  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_delta_ci(
    base_flags: list[bool],
    other_flags: list[bool],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    assert len(base_flags) == len(other_flags)
    n = len(base_flags)
    base = np.asarray(base_flags, dtype=np.float64)
    other = np.asarray(other_flags, dtype=np.float64)
    observed = float(other.mean() - base.mean()) if n else 0.0
    rng = np.random.default_rng(seed)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(other[idx].mean() - base[idx].mean()))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "positive_excludes_zero": bool(lo > 0),
    }


def stratum_acc(report: dict[str, Any], stratum: str) -> float | None:
    cell = (report.get("exact_by_stratum") or {}).get(stratum)
    if not cell:
        return None
    return float(cell["accuracy"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_qlora_cc0_8k.yaml",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_repo_path(cfg["train"]["output_dir"])
    eval_cfg = cfg["eval"]

    val_base = load_json(root / "eval_cc0" / "val" / "baseline" / "report.json")
    val_ada = load_json(root / "eval_cc0" / "val" / "layer8_qlora" / "report.json")
    test_base = load_json(root / "eval_cc0" / "test" / "baseline" / "report.json")
    test_ada = load_json(root / "eval_cc0" / "test" / "layer8_qlora" / "report.json")

    base_items = {row["id"]: row for row in val_base["per_item"]}
    ada_items = {row["id"]: row for row in val_ada["per_item"]}
    shared = sorted(set(base_items) & set(ada_items))
    ci = bootstrap_delta_ci(
        [bool(base_items[i]["exact"]) for i in shared],
        [bool(ada_items[i]["exact"]) for i in shared],
        n_boot=int(eval_cfg.get("bootstrap_samples", 2000)),
        seed=int(eval_cfg.get("bootstrap_seed", 17)),
    )

    abl_base = load_json(root / "ablation" / "baseline" / "layer8_skip.json")
    abl_ada = load_json(root / "ablation" / "layer8_qlora" / "layer8_skip.json")
    cos_delta = float(abl_ada["mean_logit_cosine_vs_intact"]) - float(
        abl_base["mean_logit_cosine_vs_intact"]
    )
    score_delta = float(abl_ada["catastrophe_score"]) - float(abl_base["catastrophe_score"])
    ablation_activated = (
        cos_delta <= -0.02
        or score_delta >= 0.05
        or abl_ada.get("bucket") in {"soft", "critical"}
    )

    puzzle_path = root / "eval" / "layer8_qlora" / "report.json"
    puzzle_base_acc = float(eval_cfg.get("puzzle_baseline_accuracy", 0.24))
    if puzzle_path.exists():
        puzzle = load_json(puzzle_path)
        puzzle_acc = float(puzzle["accuracy"])
    else:
        puzzle = None
        puzzle_acc = None

    no_short_regress = (
        puzzle_acc is not None and puzzle_acc >= puzzle_base_acc - 0.01
    )
    test_8k_base = stratum_acc(test_base, "8k")
    test_8k_ada = stratum_acc(test_ada, "8k")
    test_8k_nonworse = (
        test_8k_base is None
        or test_8k_ada is None
        or test_8k_ada >= test_8k_base - 1e-9
    )
    cc0_improved = bool(ci.get("positive_excludes_zero")) and test_8k_nonworse

    sealed_path = root / "eval_cc0" / "sealed" / "summary.json"
    sealed = load_json(sealed_path) if sealed_path.exists() else {"status": "SKIPPED_SEALED"}

    claim = bool(cc0_improved and ablation_activated and no_short_regress)
    summary = {
        "schema_version": "layer8_cc0_8k_v1",
        "experiment": cfg.get("experiment", {}),
        "cc0_val": {
            "baseline_exact": val_base["exact_accuracy"],
            "adapter_exact": val_ada["exact_accuracy"],
            "baseline_by_stratum": val_base["exact_by_stratum"],
            "adapter_by_stratum": val_ada["exact_by_stratum"],
            "bootstrap_delta": ci,
        },
        "cc0_test": {
            "baseline_exact": test_base["exact_accuracy"],
            "adapter_exact": test_ada["exact_accuracy"],
            "baseline_by_stratum": test_base["exact_by_stratum"],
            "adapter_by_stratum": test_ada["exact_by_stratum"],
            "test_8k_baseline": test_8k_base,
            "test_8k_adapter": test_8k_ada,
        },
        "ablation_layer8_skip": {
            "baseline": {
                "catastrophe_score": abl_base["catastrophe_score"],
                "mean_logit_cosine_vs_intact": abl_base["mean_logit_cosine_vs_intact"],
                "bucket": abl_base["bucket"],
            },
            "adapter": {
                "catastrophe_score": abl_ada["catastrophe_score"],
                "mean_logit_cosine_vs_intact": abl_ada["mean_logit_cosine_vs_intact"],
                "bucket": abl_ada["bucket"],
            },
            "cosine_delta": cos_delta,
            "catastrophe_score_delta": score_delta,
        },
        "puzzle_transfer": {
            "baseline_accuracy_reference": puzzle_base_acc,
            "adapter_accuracy": puzzle_acc,
            "report": puzzle,
        },
        "sealed": sealed,
        "gates": {
            "cc0_improved": cc0_improved,
            "layer8_activated": ablation_activated,
            "no_short_regress": no_short_regress,
            "claim_eligible": claim,
        },
        "caveat": (
            "Train excluded CC0 2K/4K (already in Supa-A1.1 lineage). "
            "QLoRA train / FP16+Peft eval. No all-layer control."
        ),
    }
    atomic_write_json(LAB_ROOT / "reports" / "layer8_cc0_8k_summary.json", summary)
    atomic_write_json(root / "layer8_cc0_8k_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("gates", "cc0_val", "cc0_test", "ablation_layer8_skip", "puzzle_transfer", "sealed") if k in summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
