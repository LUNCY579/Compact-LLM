#!/usr/bin/env python3
"""Phase C: how far can the KV budget fall before quality breaks?

The confirmed operating point holds quality at budget 256 (`N-071`, `N-072`):
86/192 on held-out stability, identical to full cache, at 0.293x persistent
bytes. Nobody has asked the compression question. This screen maps the frontier.

BUDGET-SCALING IS NOT A RESCALE
-------------------------------
`sinks=48` is not a proportion. It was chosen because the abstention
instruction ("reply UNKNOWN") sits at token position 38, so any sink count
below ~48 loses it and compressed abstention collapses to 0 (`N-068`). That is
a property of the PROMPT, not of the budget.

Consequently the sink share grows as the budget shrinks:

    budget 256 -> sinks 48 = 19% of budget
    budget 128 -> sinks 48 = 38%
    budget  96 -> sinks 48 = 50%
    budget  64 -> sinks 48 = 75%, leaving 16 slots for everything else

So the interesting question at low budget is a genuine trade-off: is protecting
the system prompt still worth it once it consumes most of the cache? Both
policies are therefore tested at every budget:

    FIX48   sinks fixed at 48, preserving abstention, squeezing the heavy pool
    SCALED  sinks proportional to budget, abandoning full prompt coverage

Pool width is also swept, because the optimum plausibly tracks the size of the
eligible heavy pool rather than staying at 13.

Screening runs on showcase-64. The knee configuration is confirmed separately
on held-out stability-192; nothing here is a claim.

EXPLICIT CELL COVERAGE
----------------------
At small budgets `sinks + recent` can meet or exceed the budget, which makes
`SignificancePolicy.allocation` raise when `require_heavy` is set. Every cell is
validated BEFORE any model work, and the run refuses to start if any declared
cell is infeasible. Cells are never silently skipped -- a screen that quietly
drops its hardest cells would report a flattering frontier.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SCRIPTS = LAB_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_published_core_eval import (  # noqa: E402
    Condition,
    _atomic_write_json,
    _atomic_write_jsonl,
    _check_geometry,
    _create_output_dir,
    _load_mapping,
    _normalize_answer,
    _prepare_case,
    _repo_path,
    _write_resolved_config,
    execute_condition,
)
from run_showcase64_eval import read_answer_key, read_showcase_rows  # noqa: E402

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402
from rineya_significance_lab.runtime.policy import SignificancePolicy  # noqa: E402

SINK_FLOOR = 48  # covers the system prompt; abstention instruction is at token 38


def build_cells() -> list[dict[str, Any]]:
    """Declare the frontier grid. Budget 256 is the confirmed reference point."""

    cells: list[dict[str, Any]] = [
        {"id": "B256_FIX48_P13", "budget": 256, "sinks": 48, "recent": 64, "pool": 13,
         "policy_family": "FIX48", "note": "confirmed reference configuration"},
    ]
    # `fix_recent` is separate from `scaled_recent`: holding sinks at 48 forces the
    # recent window down as the budget shrinks, or the heavy pool hits zero. At
    # budget 64 the grid guard caught exactly that -- 48 sinks + 16 recent leaves
    # no content-selected slot at all, so FIX48 must drop to an 8-token window.
    for budget, fix_recent, scaled_recent, pools, scaled_sink in (
        (192, 48, 48, (13, 9), 38),
        (128, 32, 32, (13, 7), 25),
        (96, 24, 24, (13, 5), 19),
        (64, 8, 16, (7, 5), 12),
    ):
        for pool in pools:
            heavy = budget - SINK_FLOOR - fix_recent
            cells.append({
                "id": f"B{budget}_FIX48_P{pool}", "budget": budget, "sinks": SINK_FLOOR,
                "recent": fix_recent, "pool": pool, "policy_family": "FIX48",
                "note": f"sinks {SINK_FLOOR/budget:.0%} of budget, {heavy} heavy slots left",
            })
        cells.append({
            "id": f"B{budget}_SCALED_P{pools[0]}", "budget": budget, "sinks": scaled_sink,
            "recent": scaled_recent, "pool": pools[0], "policy_family": "SCALED",
            "note": "proportional sinks; full prompt coverage abandoned",
        })
    return cells


def validate_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject the whole run if any declared cell cannot allocate a heavy pool."""

    report = []
    infeasible = []
    for c in cells:
        heavy = c["budget"] - c["sinks"] - c["recent"]
        ok = heavy > 0
        if ok:
            try:
                SignificancePolicy(
                    budget=c["budget"], sinks=c["sinks"], recent=c["recent"],
                    count_exponent=1.0, pool_width=c["pool"], require_heavy=True,
                ).allocation(1300)
            except Exception as exc:  # noqa: BLE001
                ok = False
                c["error"] = str(exc)
        if not ok:
            infeasible.append(c["id"])
        report.append({**c, "heavy_slots": heavy, "feasible": ok})
    if infeasible:
        raise RuntimeError(
            f"infeasible cells declared, refusing to run a partial frontier: {infeasible}"
        )
    return report


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config = _load_mapping(config_path.expanduser().resolve())
    config["_source_config_path"] = _repo_path(config_path)
    cells = build_cells()
    grid = validate_cells(cells)

    model_path = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
    prompt_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl"
    key_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_answer_key.jsonl"
    output = (
        output_override.expanduser().resolve() if output_override
        else LAB_ROOT / "results" / "runs" / "halo_compression_frontier_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)
    _atomic_write_json(output / "declared_grid.json", {
        "declared_at": started_at,
        "sink_floor_rationale": "abstention instruction sits at token 38; sinks below ~48 lose it (N-068)",
        "cells": grid,
        "cell_count": len(grid),
        "coverage_rule": "every declared cell must complete; infeasible cells abort the run before model load",
        "screening_split": "showcase (64 cases, 16 packs); the knee is confirmed separately on stability-192",
    })
    print(f"declared {len(grid)} feasible cells", flush=True)

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = (AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False, low_cpu_mem_usage=True).to("cpu").eval())
    _check_geometry(model.config, config["model"])

    rows = read_showcase_rows(prompt_path)
    ids = tuple(str(r["id"]) for r in rows)
    prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in rows]
    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    unscored: list[dict[str, Any]] = []
    total = len(cells) * len(prepared)
    done = 0
    started = time.perf_counter()
    with (output / "predictions.partial.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for cell in cells:
            cache_config = dict(config["cache"])
            cache_config.update(
                retained_history_budget=cell["budget"], sink_tokens=cell["sinks"],
                recent_tokens=cell["recent"], score_pool_width=cell["pool"],
            )
            condition = Condition(condition_id=cell["id"], policy="significance",
                                  exponent=1.0, query_conditioned=True)
            for case in prepared:
                result = execute_condition(
                    model, case, condition, cache_config=cache_config,
                    runtime_config=config["runtime"], eos_token_ids=eos_ids,
                )
                pred = tokenizer.decode(result.pop("generated_token_ids"),
                                        skip_special_tokens=True).strip()
                unscored.append({
                    "case_id": case.case_id, "scenario_id": case.scenario_id,
                    "arm": cell["id"], "budget": cell["budget"], "sinks": cell["sinks"],
                    "recent": cell["recent"], "pool_width": cell["pool"],
                    "policy_family": cell["policy_family"], "prediction": pred,
                    "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                    "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                })
                h.write(json.dumps(unscored[-1], ensure_ascii=False, sort_keys=True) + "\n"); h.flush()
                done += 1
                if done % 32 == 0 or done == total:
                    rate = (time.perf_counter() - started) / done
                    print(f"  {done}/{total} cell={cell['id']} eta={(total-done)*rate/60:.1f} min", flush=True)
                gc.collect()

    # Coverage assertion: no cell may be missing or short.
    seen = {}
    for r in unscored:
        seen[r["arm"]] = seen.get(r["arm"], 0) + 1
    short = {k: v for k, v in seen.items() if v != len(prepared)}
    missing = [c["id"] for c in cells if c["id"] not in seen]
    if short or missing:
        raise RuntimeError(f"incomplete frontier: short={short} missing={missing}")

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)
    key = read_answer_key(key_path, ids)
    for r in unscored:
        kr = key[r["case_id"]]
        acc = {_normalize_answer(v) for v in kr["accepted_answers"]}
        r["correct"] = _normalize_answer(str(r["prediction"])) in acc
        r["variant"] = kr.get("variant"); r["answerable"] = kr.get("answerable")
        r["reference_answer"] = kr["answer"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    import statistics as st
    by_cell = {}
    for cell in cells:
        sel = [r for r in unscored if r["arm"] == cell["id"]]
        un = [r for r in sel if not r.get("answerable")]
        corr = sum(bool(r["correct"]) for r in sel)
        by_cell[cell["id"]] = {
            "budget": cell["budget"], "sinks": cell["sinks"], "recent": cell["recent"],
            "pool_width": cell["pool"], "policy_family": cell["policy_family"],
            "heavy_slots": cell["budget"] - cell["sinks"] - cell["recent"],
            "cases": len(sel), "correct_cases": corr,
            "exact_accuracy": corr / len(sel) if sel else None,
            "unanswerable": {"cases": len(un), "correct_cases": sum(bool(r["correct"]) for r in un)},
            "median_resident": st.median(r["resident_after_prefill"] for r in sel),
            "median_persistent_mb": st.median(r["total_persistent_bytes"] for r in sel) / 1e6,
        }
    summary = {
        "schema_version": "1.0", "status": "PASS",
        "classification": "EXPLORATORY_FRONTIER_SCREEN",
        "split": "showcase", "cases": len(ids), "packs": 16,
        "reference": {"B0_full_cache_showcase": "30/64", "B0_full_cache_MB": 155.1,
                      "shipped_B3": "19/64",
                      "confirmed_B256_stability": "86/192 ties full cache"},
        "cells": by_cell,
        "sink_floor_rationale": "abstention instruction at token 38 (N-068)",
        "interpretation_limits": [
            "Screening split only; the knee must be confirmed on held-out stability-192.",
            "Scenario packs are correlated; 64 rows are 16 packs.",
            "Many cells, no multiplicity control; this maps a curve, it does not test a hypothesis.",
            "Ranking is S/max(C,1) throughout. If the pooling specificity control nulls, the same frontier should hold with a random ranking and that must be re-tested at the knee.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(output / "manifest.json", {
        "schema_version": "1.0", "run_id": output.name,
        "created_at": started_at, "completed_at": utc_now(),
        "classification": "EXPLORATORY_FRONTIER_SCREEN",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE", "seed": seed,
        "benchmark": {"split": "showcase", "case_count": len(ids),
                      "prompt_sha256": sha256_file(prompt_path),
                      "answer_key_sha256_after_inference": sha256_file(key_path)},
        "cells": grid,
        "notes": ["Maps accuracy against KV budget from 256 down to 64.",
                  "FIX48 vs SCALED tests whether full prompt coverage stays worth its share.",
                  "All declared cells validated feasible before model load; coverage asserted after."],
    })
    print(json.dumps({"output": str(output), "cells": len(cells)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_eval.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
