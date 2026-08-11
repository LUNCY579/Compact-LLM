from __future__ import annotations

"""Build preservation-heavy residual reasoning/procedure training views."""
import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def repeat_sample(rows: list[dict[str, Any]], n: int, rng: random.Random, max_repeat: int) -> list[dict[str, Any]]:
    if not rows or n <= 0:
        return []
    if n > len(rows) * max_repeat:
        raise ValueError(
            f"requested {n} samples from {len(rows)} rows exceeds max repeat factor {max_repeat}"
        )
    out: list[dict[str, Any]] = []
    pool = list(rows)
    while len(out) < n:
        rng.shuffle(pool)
        out.extend(pool[: min(len(pool), n - len(out))])
    return out


def plan_mix_counts(
    *,
    total: int,
    residual_fraction: float,
    residual_pool: int,
    preservation_pool: int,
    max_repeat: int,
) -> dict[str, Any]:
    """Target residual fraction, but never exceed the per-pool repeat cap.

    When residual rows are scarce (but still above the readiness minimum), cap
    residual draws at pool*max_repeat and backfill with preservation so the mix
    stays preservation-heavy instead of aborting the pipeline.
    """
    desired_res = round(total * residual_fraction)
    max_res = residual_pool * max_repeat
    max_pres = preservation_pool * max_repeat
    nres = min(desired_res, max_res)
    residual_capped = nres < desired_res
    npres = total - nres
    preservation_capped = False
    if npres > max_pres:
        npres = max_pres
        preservation_capped = True
        # Prefer keeping residual exposure; shrink total if preservation cannot fill.
        if nres + npres < total:
            total = nres + npres
    return {
        "total": total,
        "residual_rows_sampled": nres,
        "preservation_rows_sampled": npres,
        "desired_residual_rows": desired_res,
        "residual_capped_by_repeat": residual_capped,
        "preservation_capped_by_repeat": preservation_capped,
        "residual_fraction_realized": (nres / total) if total else 0.0,
    }


def build_view(
    *,
    name: str,
    residual: list[dict[str, Any]],
    preservation: list[dict[str, Any]],
    total: int,
    residual_fraction: float,
    max_repeat: int,
    rng: random.Random,
    output_root: Path,
) -> dict[str, Any]:
    plan = plan_mix_counts(
        total=total,
        residual_fraction=residual_fraction,
        residual_pool=len(residual),
        preservation_pool=len(preservation),
        max_repeat=max_repeat,
    )
    mix = repeat_sample(residual, plan["residual_rows_sampled"], rng, max_repeat) + repeat_sample(
        preservation, plan["preservation_rows_sampled"], rng, max_repeat
    )
    rng.shuffle(mix)
    write_jsonl(output_root / f"{name}_residual_mixed.reviewed.jsonl", mix)
    return {"rows": len(mix), **plan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning-residual", type=Path, required=True)
    ap.add_argument("--procedure-residual", type=Path, required=True)
    ap.add_argument("--preservation", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--reasoning-total", type=int, default=6000)
    ap.add_argument("--reasoning-residual-fraction", type=float, default=0.60)
    ap.add_argument("--procedure-total", type=int, default=5000)
    ap.add_argument("--procedure-residual-fraction", type=float, default=0.48)
    ap.add_argument("--minimum-reasoning-residual", type=int, default=250)
    ap.add_argument("--minimum-procedure-residual", type=int, default=60)
    ap.add_argument("--maximum-repeat-factor", type=int, default=8)
    ap.add_argument("--seed", type=int, default=23071)
    a = ap.parse_args()

    rr = read_jsonl(a.reasoning_residual)
    pr = read_jsonl(a.procedure_residual)
    pres = read_jsonl(a.preservation)
    status = {
        "reasoning_ready": len(rr) >= a.minimum_reasoning_residual,
        "procedure_ready": len(pr) >= a.minimum_procedure_residual,
    }
    rng = random.Random(a.seed)
    manifest: dict[str, Any] = {
        "schema_version": "2.3",
        "source_counts": {
            "reasoning_residual": len(rr),
            "procedure_residual": len(pr),
            "preservation": len(pres),
        },
        "readiness": status,
        "views": {},
    }
    a.output_root.mkdir(parents=True, exist_ok=True)
    if status["reasoning_ready"]:
        if not pres:
            raise ValueError("reasoning residual is ready but preservation pool is empty")
        manifest["views"]["reasoning"] = build_view(
            name="reasoning",
            residual=rr,
            preservation=pres,
            total=a.reasoning_total,
            residual_fraction=a.reasoning_residual_fraction,
            max_repeat=a.maximum_repeat_factor,
            rng=rng,
            output_root=a.output_root,
        )
    if status["procedure_ready"]:
        if not pres:
            raise ValueError("procedure residual is ready but preservation pool is empty")
        manifest["views"]["procedure"] = build_view(
            name="procedure",
            residual=pr,
            preservation=pres,
            total=a.procedure_total,
            residual_fraction=a.procedure_residual_fraction,
            max_repeat=a.maximum_repeat_factor,
            rng=rng,
            output_root=a.output_root,
        )
    (a.output_root / "training_mix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
