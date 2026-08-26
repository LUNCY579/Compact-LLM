#!/usr/bin/env python3
"""Does the significance score contribute anything under pooling? (N-073)

WHY THIS GATES EVERYTHING DOWNSTREAM
------------------------------------
`POOL13_SINK48` was CONFIRMED on held-out stability-192: 86/192 against the
shipped 59/192, cluster p=0.00088, tying full cache at 0.293x the bytes
(`N-071`, `N-072`).

But that arm still ranks by `S/max(C,1)` underneath the pooling, and the
program has already established the raw score is inert:

    N-046  the score is indistinguishable from a position counter
    N-054  SUM_SHUF -- shuffling S among eligible entries -- scored 21/64
           against the shipped 19/64. Shuffling did not hurt.

So the confirmed gain may be entirely structural. If pooling a RANDOM ranking
matches pooling the significance ranking, then "Cornerstone-1 x Cornerstone-2
integration" collapses to "Cornerstone-2 alone", and any pipeline built on the
premise that the significance score contributes is built on sand.

This is the control `N-073` records as missing. It must run before any
integration work, not after.

ARMS -- all at the confirmed configuration (budget 256, pool_width 13,
sinks 48, recent 64), differing only in what is ranked:

    POOL13_RANDOM   uniform random ranking, re-drawn per boundary
    POOL13_SHUF     the real S/C values, permuted among eligible entries,
                    preserving the per-row multiset exactly while destroying
                    entry identity. This is the strictest control: identical
                    score distribution, zero entry information.
    POOL13_POS      absolute position as the ranking

The reference arm `POOL13_SINK48` (86/192) already exists in
`halo_stability192_validation_v1` and is not re-run.

PRE-DECLARED ANALYSIS PLAN -- fixed before first execution
----------------------------------------------------------
Independence unit: scenario pack (48).

PRIMARY, one contrast:
    POOL13_SINK48 versus POOL13_SHUF, exact two-sided cluster sign test.
    The significance score is CREDITED only if p < 0.05 with the real score
    ahead. Otherwise the confirmed gain is attributed to pooling and sink
    structure alone, and `N-071`/`N-072` must be reworded accordingly.

SECONDARY, descriptive: POOL13_RANDOM and POOL13_POS versus POOL13_SINK48.

Note the asymmetry deliberately: this test can only ever REMOVE credit from the
significance matrix. A null does not weaken the confirmed configuration result,
which stands on its own comparison against the shipped baseline.
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
from run_showcase64_gated_exposure import _shuffle_eligible  # noqa: E402
from run_stability192_validation import (  # noqa: E402
    cluster_sign_test,
    read_key,
    read_split_rows,
)

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402

POOL, SINKS, RECENT = 13, 48, 64
ARMS = (
    {"id": "POOL13_SHUF", "mode": "shuffle", "label": "eligible_pool_shuffled_scores_strictest_control"},
    {"id": "POOL13_RANDOM", "mode": "random", "label": "uniform_random_ranking"},
    {"id": "POOL13_POS", "mode": "position", "label": "absolute_position_ranking"},
)
PRIMARY = ("POOL13_SINK48", "POOL13_SHUF")


def make_callback(mode: str, generator):
    def callback(cache, policy, chunk_index):
        from rineya_significance_lab.runtime.cache import CompressionReport
        from rineya_significance_lab.runtime.policy import select_keep_indices

        before = sum(l.stored_length() for l in cache.layers if l.is_initialized)
        evicted = compressed = 0
        for layer in cache.layers:
            if not layer.is_initialized:
                continue
            old = layer.stored_length()
            if old <= policy.budget:
                continue
            base = layer.state.scores / layer.state.counts.clamp_min(1.0)
            if mode == "shuffle":
                ranking = _shuffle_eligible(base, sinks=SINKS, recent=RECENT, generator=generator)
            elif mode == "random":
                ranking = torch.rand(base.shape, generator=generator, dtype=base.dtype)
            elif mode == "position":
                ranking = layer.state.positions.to(base.dtype)
            else:
                raise ValueError(f"unsupported mode: {mode}")
            if not torch.isfinite(ranking).all():
                raise RuntimeError(f"layer {layer.layer_idx}: non-finite ranking")
            indices = select_keep_indices(
                ranking, policy, positions=layer.state.positions, entry_ids=layer.state.entry_ids
            )
            gi = indices.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
            layer.keys = layer.keys.gather(2, gi).contiguous()
            layer.values = layer.values.gather(2, gi).contiguous()
            layer.state.gather(indices, cache_length=old)
            layer.evictions += 1
            layer.assert_aligned()
            if layer.stored_length() < old:
                compressed += 1
                evicted += old - layer.stored_length()
        cache.assert_aligned()
        after = sum(l.stored_length() for l in cache.layers if l.is_initialized)
        return CompressionReport(compressed, before, after, evicted)

    return callback


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
        else LAB_ROOT / "results" / "runs" / "halo_stability192_pooling_specificity_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)
    _atomic_write_json(output / "preregistered_plan.json", {
        "declared_at": started_at,
        "question": "Does the significance score contribute under pooling, or is the confirmed gain purely structural?",
        "independence_unit": "scenario_pack", "packs_expected": 48,
        "primary_contrast": f"{PRIMARY[0]} vs {PRIMARY[1]}",
        "primary_test": "exact two-sided cluster sign test over 48 scenario packs",
        "credit_score_if": "p < 0.05 with the real S/C ranking ahead",
        "otherwise": "attribute the confirmed gain to pooling and sink structure alone; reword N-071/N-072",
        "secondary_descriptive_only": ["POOL13_RANDOM vs POOL13_SINK48", "POOL13_POS vs POOL13_SINK48"],
        "reference_arm_not_rerun": {"POOL13_SINK48": "86/192 from halo_stability192_validation_v1"},
        "asymmetry_note": "This test can only remove credit from the significance matrix; it cannot weaken the configuration result.",
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

    cache_config = dict(config["cache"])
    cache_config["sink_tokens"] = SINKS
    cache_config["recent_tokens"] = RECENT
    cache_config["score_pool_width"] = POOL

    unscored: list[dict[str, Any]] = []
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()
    with (output / "predictions.partial.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for arm in ARMS:
            generator = torch.Generator().manual_seed(seed)
            condition = Condition(condition_id=arm["id"], policy="significance",
                                  exponent=1.0, query_conditioned=True)
            for case in prepared:
                result = execute_condition(
                    model, case, condition, cache_config=cache_config,
                    runtime_config=config["runtime"], eos_token_ids=eos_ids,
                    compression_callback=make_callback(arm["mode"], generator),
                )
                pred = tokenizer.decode(result.pop("generated_token_ids"), skip_special_tokens=True).strip()
                unscored.append({
                    "case_id": case.case_id, "scenario_id": case.scenario_id,
                    "arm": arm["id"], "arm_label": arm["label"], "ranking_mode": arm["mode"],
                    "pool_width": POOL, "sink_tokens": SINKS, "prediction": pred,
                    "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
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

    # Pull the confirmed reference arm in for the paired comparison.
    ref_path = LAB_ROOT / "results" / "runs" / "halo_stability192_validation_v1" / "cases.jsonl"
    ref = [json.loads(l) for l in open(ref_path, encoding="utf-8")]
    joined = unscored + [r for r in ref if r["arm"] == "POOL13_SINK48"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    by_arm = {}
    for arm_id in [a["id"] for a in ARMS] + ["POOL13_SINK48"]:
        sel = [r for r in joined if r["arm"] == arm_id]
        un = [r for r in sel if not r.get("answerable")]
        by_arm[arm_id] = {
            "cases": len(sel), "correct_cases": sum(bool(r["correct"]) for r in sel),
            "exact_accuracy": sum(bool(r["correct"]) for r in sel) / len(sel) if sel else None,
            "unanswerable": {"cases": len(un), "correct_cases": sum(bool(r["correct"]) for r in un)},
            "from_reference_run": arm_id == "POOL13_SINK48",
        }
    primary = cluster_sign_test(joined, *PRIMARY)
    credited = primary["cluster_exact_two_sided_p"] < 0.05 and primary["packs_better"] > primary["packs_worse"]
    secondary = [cluster_sign_test(joined, "POOL13_SINK48", a) for a in ("POOL13_RANDOM", "POOL13_POS")]

    summary = {
        "schema_version": "1.0", "status": "PASS",
        "classification": "SPECIFICITY_CONTROL_PREREGISTERED_PLAN",
        "split": "stability_test", "cases": len(ids), "packs": 48,
        "configuration": {"budget": 256, "pool_width": POOL, "sinks": SINKS, "recent": RECENT},
        "arms": by_arm,
        "primary_result": primary,
        "significance_score_credited": credited,
        "verdict": ("SCORE_CONTRIBUTES" if credited else "GAIN_IS_STRUCTURAL_SCORE_NOT_CREDITED"),
        "secondary_descriptive": secondary,
        "interpretation_limits": [
            "One primary contrast declared before inference; everything else descriptive.",
            "A null here does not weaken N-071/N-072, which compare against the shipped baseline.",
            "It does determine whether the gain may be attributed to the significance matrix at all.",
            "Reference arm POOL13_SINK48 comes from the validation run, same split, same protocol, different process.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(output / "manifest.json", {
        "schema_version": "1.0", "run_id": output.name,
        "created_at": started_at, "completed_at": utc_now(),
        "classification": "SPECIFICITY_CONTROL_PREREGISTERED_PLAN",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE", "seed": seed,
        "benchmark": {"split": "stability_test", "case_count": len(ids),
                      "prompt_sha256": sha256_file(prompt_path),
                      "answer_key_sha256_after_inference": sha256_file(key_path)},
        "arms": [{k: a[k] for k in ("id", "label", "mode")} for a in ARMS],
        "notes": ["Addresses N-073.", "Plan written before inference.",
                  "POOL13_SHUF preserves the per-row score multiset exactly."],
    })
    print(json.dumps({"output": str(output), "primary": primary, "verdict": summary["verdict"]}, indent=2))
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
