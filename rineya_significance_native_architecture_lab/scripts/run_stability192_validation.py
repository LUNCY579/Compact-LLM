#!/usr/bin/env python3
"""Held-out validation of the pooling+sink configuration on stability-192.

WHY THIS RUN EXISTS
-------------------
The span screens selected a winner on the same 64 showcase cases used
throughout the program, across 14 arms, with no multiplicity control. That is
textbook winner's-curse exposure. On showcase-64:

    POOL1  (= shipped B3)   19/64
    POOL7_SINK48            28/64
    POOL13_SINK48           29/64   <- best
    B0 full cache           30/64
    ORACLE_SPAN             32/64

No contrast was cluster-significant (best p=0.125), and the width trend was
noisy rather than monotone (POOL9 dipped to 23 between POOL7's 26 and POOL13's
27). So the result is a candidate, not a finding.

The stability split is a proper held-out set: 192 cases in 48 scenario packs,
with ZERO scenario overlap against showcase, giving three times the cluster
power.

PRE-DECLARED ANALYSIS PLAN -- fixed before this script was first executed
-------------------------------------------------------------------------
Independence unit: the scenario pack (48 of them). Case-level counts are
reported but are NOT the basis of any claim.

PRIMARY endpoint, one contrast only:
    POOL13_SINK48 versus POOL1, exact two-sided cluster sign test over the 48
    packs. CONFIRMED if p < 0.05 with POOL13_SINK48 ahead. Anything else is
    NOT CONFIRMED.

SECONDARY, descriptive only, never promoted to primary if the primary fails:
    - POOL7_SINK48 versus POOL1
    - B0 versus POOL13_SINK48, to ask whether the compression gap stays
      undetectable at triple the power
    - unanswerable-variant accuracy per arm

Declaring one primary contrast in advance is what makes this a validation
rather than a sixth screen.

Halo only. Answer key opened after all inference and after the answer-blind
artifact is durably written.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from math import comb
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
    _assert_prompt_only,
    _atomic_write_json,
    _atomic_write_jsonl,
    _check_geometry,
    _create_output_dir,
    _load_jsonl,
    _load_mapping,
    _normalize_answer,
    _prepare_case,
    _repo_path,
    _write_resolved_config,
    execute_condition,
)

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402

ARMS = (
    {"id": "B0", "policy": "full_cache", "pool": 1, "sinks": 4, "label": "full_cache_reference"},
    {"id": "POOL1", "policy": "significance", "pool": 1, "sinks": 4, "label": "shipped_b3_baseline"},
    {"id": "POOL13_SINK48", "policy": "significance", "pool": 13, "sinks": 48, "label": "primary_candidate"},
    {"id": "POOL7_SINK48", "policy": "significance", "pool": 7, "sinks": 48, "label": "secondary_candidate"},
)
PRIMARY = ("POOL13_SINK48", "POOL1")


def read_split_rows(path: Path, split: str, expected: int) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        rid = str(row.get("id", ""))
        if not rid or rid in seen:
            raise RuntimeError(f"bad or duplicate prompt id: {rid!r}")
        seen.add(rid)
        _assert_prompt_only(row)
        if str(row.get("split")) != split:
            raise RuntimeError(f"row {rid} is not in split {split!r}")
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows in {split}, found {len(rows)}")
    return rows


def read_key(path: Path, ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    sel: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        rid = str(row.get("id", ""))
        if rid not in ids:
            continue
        if not isinstance(row.get("answer"), str):
            raise RuntimeError(f"answer-key row {rid} has no string answer")
        sel[rid] = row
    missing = [i for i in ids if i not in sel]
    if missing:
        raise RuntimeError(f"missing answer-key ids: {missing[:5]}")
    return sel


def cluster_sign_test(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Exact two-sided sign test over scenario packs."""

    by = {(r["arm"], r["case_id"]): bool(r["correct"]) for r in rows}
    packs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cases = sorted({r["case_id"] for r in rows})
    scen = {r["case_id"]: r["scenario_id"] for r in rows}
    for c in cases:
        packs[scen[c]][0] += by.get((a, c), False)
        packs[scen[c]][1] += by.get((b, c), False)
    better = sum(1 for x, y in packs.values() if x > y)
    worse = sum(1 for x, y in packs.values() if x < y)
    n = better + worse
    k = max(better, worse)
    p = min(1.0, sum(comb(n, j) for j in range(k, n + 1)) / 2**n * 2) if n else 1.0
    cw = sum(1 for c in cases if by.get((a, c), False) and not by.get((b, c), False))
    cl = sum(1 for c in cases if by.get((b, c), False) and not by.get((a, c), False))
    return {
        "comparison": f"{a} vs {b}",
        "packs_better": better, "packs_worse": worse,
        "packs_tied": len(packs) - n, "packs_total": len(packs),
        "cluster_exact_two_sided_p": p,
        "case_wins": cw, "case_losses": cl,
        "direction": "favours_" + (a if better > worse else b if worse > better else "neither"),
    }


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config = _load_mapping(config_path.expanduser().resolve())
    config["_source_config_path"] = _repo_path(config_path)
    model_path = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
    prompt_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "stability_prompts.jsonl"
    key_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "stability_answer_key.jsonl"
    output = (
        output_override.expanduser().resolve() if output_override
        else LAB_ROOT / "results" / "runs" / "halo_stability192_validation_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)
    # The plan is written to disk BEFORE any inference.
    _atomic_write_json(output / "preregistered_plan.json", {
        "declared_at": started_at,
        "independence_unit": "scenario_pack",
        "packs_expected": 48,
        "primary_contrast": f"{PRIMARY[0]} vs {PRIMARY[1]}",
        "primary_test": "exact two-sided cluster sign test over 48 scenario packs",
        "confirm_if": "p < 0.05 with POOL13_SINK48 ahead",
        "secondary_descriptive_only": [
            "POOL7_SINK48 vs POOL1", "B0 vs POOL13_SINK48", "unanswerable accuracy per arm",
        ],
        "selection_provenance": "arms chosen on showcase-64 across 14 arms without multiplicity control",
        "showcase_reference": {"POOL1": "19/64", "POOL7_SINK48": "28/64",
                               "POOL13_SINK48": "29/64", "B0": "30/64"},
    })

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = (AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False, low_cpu_mem_usage=True).to("cpu").eval())
    _check_geometry(model.config, config["model"])

    rows = read_split_rows(prompt_path, "stability_test", 192)
    ids = tuple(str(r["id"]) for r in rows)
    prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in rows]
    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    unscored: list[dict[str, Any]] = []
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()
    with (output / "predictions.partial.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for arm in ARMS:
            cache_config = dict(config["cache"])
            cache_config["sink_tokens"] = int(arm["sinks"])
            cache_config["score_pool_width"] = int(arm["pool"])
            condition = Condition(
                condition_id=arm["id"], policy=str(arm["policy"]),
                exponent=None if arm["policy"] == "full_cache" else 1.0, query_conditioned=True,
            )
            for case in prepared:
                result = execute_condition(
                    model, case, condition, cache_config=cache_config,
                    runtime_config=config["runtime"], eos_token_ids=eos_ids,
                )
                pred = tokenizer.decode(result.pop("generated_token_ids"), skip_special_tokens=True).strip()
                unscored.append({
                    "case_id": case.case_id, "scenario_id": case.scenario_id,
                    "arm": arm["id"], "arm_label": arm["label"],
                    "pool_width": arm["pool"], "sink_tokens": arm["sinks"],
                    "prediction": pred,
                    "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                    "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                })
                h.write(json.dumps(unscored[-1], ensure_ascii=False, sort_keys=True) + "\n"); h.flush()
                done += 1
                if done % 32 == 0 or done == total:
                    rate = (time.perf_counter() - started) / done
                    print(f"  {done}/{total} arm={arm['id']} eta={(total-done)*rate/60:.1f} min", flush=True)
                gc.collect()

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)
    key = read_key(key_path, ids)
    for r in unscored:
        kr = key[r["case_id"]]
        acc = {_normalize_answer(v) for v in kr.get("accepted_answers", [kr["answer"]])}
        r["correct"] = _normalize_answer(str(r["prediction"])) in acc
        r["variant"] = kr.get("variant"); r["answerable"] = kr.get("answerable")
        r["reference_answer"] = kr["answer"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    by_arm = {}
    for arm in ARMS:
        sel = [r for r in unscored if r["arm"] == arm["id"]]
        un = [r for r in sel if not r.get("answerable")]
        by_arm[arm["id"]] = {
            "label": arm["label"], "pool_width": arm["pool"], "sink_tokens": arm["sinks"],
            "cases": len(sel), "correct_cases": sum(bool(r["correct"]) for r in sel),
            "exact_accuracy": sum(bool(r["correct"]) for r in sel) / len(sel) if sel else None,
            "unanswerable": {"cases": len(un), "correct_cases": sum(bool(r["correct"]) for r in un)},
        }
    primary = cluster_sign_test(unscored, *PRIMARY)
    confirmed = primary["cluster_exact_two_sided_p"] < 0.05 and primary["packs_better"] > primary["packs_worse"]
    secondary = [cluster_sign_test(unscored, a, b) for a, b in
                 (("POOL7_SINK48", "POOL1"), ("B0", "POOL13_SINK48"), ("B0", "POOL1"))]

    summary = {
        "schema_version": "1.0", "status": "PASS",
        "classification": "HELD_OUT_VALIDATION_PREREGISTERED_PLAN",
        "split": "stability_test", "cases": len(ids), "packs": 48,
        "scenario_overlap_with_showcase": 0,
        "arms": by_arm,
        "primary_result": primary,
        "primary_verdict": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
        "secondary_descriptive": secondary,
        "showcase_reference": {"POOL1": "19/64", "POOL7_SINK48": "28/64",
                               "POOL13_SINK48": "29/64", "B0": "30/64", "ORACLE_SPAN": "32/64"},
        "interpretation_limits": [
            "One primary contrast declared before inference; everything else is descriptive.",
            "Held-out split with zero scenario overlap, but a single core, schedule, budget and seed.",
            "A confirmed result licenses a configuration change, not a claim about the significance matrix: pooling is a Cornerstone-2 span factor.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(output / "manifest.json", {
        "schema_version": "1.0", "run_id": output.name,
        "created_at": started_at, "completed_at": utc_now(),
        "classification": "HELD_OUT_VALIDATION_PREREGISTERED_PLAN",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE", "seed": seed,
        "benchmark": {"split": "stability_test", "case_count": len(ids),
                      "prompt_sha256": sha256_file(prompt_path),
                      "answer_key_sha256_after_inference": sha256_file(key_path)},
        "arms": [{k: a[k] for k in ("id", "label", "policy", "pool", "sinks")} for a in ARMS],
        "notes": ["Plan written to preregistered_plan.json before any inference.",
                  "Primary contrast POOL13_SINK48 vs POOL1 at the scenario-pack unit."],
    })
    print(json.dumps({"output": str(output), "primary": primary,
                      "verdict": summary["primary_verdict"]}, indent=2))
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
