#!/usr/bin/env python3
"""P4: change the S aggregator, and run the missing shuffled-S control.

Two questions, one screen.

P4 -- does the numerator's aggregation explain the failures?
------------------------------------------------------------
``S`` is a plain sum over scoring events.  Retrieval evidence looks like a
single strong spike at the boundary where the query matches it, and a sum
dilutes that spike against entries receiving steady background attention.  The
audit's P4 proposes spike-preserving aggregators:

    MAX  S_i <- max(S_i, contribution_i)
    EMA  S_i <- alpha * contribution_i + (1 - alpha) * S_i

Both are applied through the audited ``update()`` path by passing a computed
delta, so every invariant, finiteness check, and scoring-event counter behaves
exactly as shipped:

    MAX  delta = clamp(contribution - S, min=0)
    EMA  delta = alpha * (contribution - S)

These arms use ``e=0``, so the ranking is ``S`` alone.  That isolates the
aggregator from the exposure denominator, which the gated-exposure screen
already retired.

The missing foundational control -- does S carry any signal at all?
-------------------------------------------------------------------
Every shuffled control this lab has run permuted the *counts* (`N-021`,
`N-028`, `N-033`) or an allocation statistic.  None permuted the significance
score itself.  So the shipped mechanism has never been tested against a random
permutation of its own numerator.

``SUM_SHUF`` is exactly the shipped B3 configuration -- summed ``S``, ``e=1`` --
with ``S`` permuted among eligible entries at each boundary, preserving the
per-row multiset and the sink/recent denominators while destroying entry
identity.  If ``SUM_SHUF`` matches B3, then the significance score carries no
usable retention information and the mechanism's behaviour is entirely due to
its sink-plus-recent structure.

``MAX_SHUF`` is the same control for the MAX arm.

Implementation note
-------------------
``SignificanceCache.record_attention`` is monkeypatched for the duration of the
run.  No file under ``src/`` is modified; every one is content-hashed by the
completed Experiment F provenance record.

Exploratory and unregistered.  Halo only.  The answer key is opened after all
inference and after the answer-blind artifact is written.
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
    _checkpoint_hashes,
    _create_output_dir,
    _declared_base,
    _load_mapping,
    _normalize_answer,
    _prepare_case,
    _repo_path,
    _resolve_path,
    _validate_halo_core,
    _write_resolved_config,
    execute_condition,
)
from run_showcase64_eval import read_answer_key, read_showcase_rows  # noqa: E402
from run_showcase64_gated_exposure import _shuffle_eligible  # noqa: E402

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402
from rineya_significance_lab.runtime.cache import SignificanceCache  # noqa: E402

EMA_ALPHA = 0.5

ARMS = (
    {"id": "MAX_E0", "aggregator": "max", "exponent": 0.0, "shuffle_scores": False,
     "label": "running_max_spike_preserving"},
    {"id": "EMA_E0", "aggregator": "ema", "exponent": 0.0, "shuffle_scores": False,
     "label": f"ema_alpha_{EMA_ALPHA}"},
    {"id": "MAX_SHUF", "aggregator": "max", "exponent": 0.0, "shuffle_scores": True,
     "label": "running_max_eligible_pool_shuffled_control"},
    {"id": "SUM_SHUF", "aggregator": "sum", "exponent": 1.0, "shuffle_scores": True,
     "label": "shipped_B3_with_eligible_pool_shuffled_scores"},
)


def install_aggregator(mode: str):
    """Patch record_attention to change how S accumulates. Returns the original."""

    original = SignificanceCache.record_attention

    def patched(self, layer_idx: int, contribution: torch.Tensor) -> None:
        if not self.recording_enabled:
            raise RuntimeError("attention recorder called outside a recording context")
        layer = self.layers[layer_idx]
        if mode == "sum":
            delta = contribution
        else:
            current = layer.state.scores.to(contribution.dtype)
            if mode == "max":
                delta = torch.clamp(contribution - current, min=0.0)
            elif mode == "ema":
                delta = EMA_ALPHA * (contribution - current)
            else:
                raise ValueError(f"unsupported aggregator: {mode}")
        layer.record(delta)

    SignificanceCache.record_attention = patched
    return original


def make_callback(exponent: float, *, shuffle_scores: bool, cache_config: dict, generator):
    sinks = int(cache_config["sink_tokens"])
    recent = int(cache_config["recent_tokens"])

    def callback(cache, policy, chunk_index):
        from rineya_significance_lab.runtime.cache import CompressionReport
        from rineya_significance_lab.runtime.policy import select_keep_indices

        before = sum(l.stored_length() for l in cache.layers if l.is_initialized)
        evicted = 0
        compressed = 0
        for layer in cache.layers:
            if not layer.is_initialized:
                continue
            old = layer.stored_length()
            if old <= policy.budget:
                continue
            scores = layer.state.scores
            if shuffle_scores:
                # Permute S only within the eligible pool, leaving the sink and
                # recent partitions untouched, so the multiset and both
                # denominators are preserved exactly.
                scores = _shuffle_eligible(scores, sinks=sinks, recent=recent, generator=generator)
            ranking = (
                scores
                if exponent == 0.0
                else scores / layer.state.counts.clamp_min(1.0).pow(exponent)
            )
            if not torch.isfinite(ranking).all():
                raise RuntimeError(f"layer {layer.layer_idx}: non-finite ranking")
            indices = select_keep_indices(
                ranking, policy, positions=layer.state.positions, entry_ids=layer.state.entry_ids
            )
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
            layer.keys = layer.keys.gather(2, gather_idx).contiguous()
            layer.values = layer.values.gather(2, gather_idx).contiguous()
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
    base = _declared_base(config)
    paths = config["paths"]
    model_path = _resolve_path(paths["model"], base=base)
    prompt_path = _resolve_path(paths["prompts"], base=base)
    answer_key_path = _resolve_path(paths["answer_key"], base=base)
    output = (
        output_override.expanduser().resolve()
        if output_override
        else REPO_ROOT
        / "rineya_significance_native_architecture_lab"
        / "results"
        / "runs"
        / "halo_showcase64_aggregator_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)

    core = json.loads((model_path / "CORE.json").read_text(encoding="utf-8"))
    _validate_halo_core(core, config["model"])
    checkpoint = _checkpoint_hashes(model_path, config["model"])

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        .to("cpu")
        .eval()
    )
    _check_geometry(model.config, config["model"])

    rows = read_showcase_rows(prompt_path)
    case_ids = tuple(str(r["id"]) for r in rows)
    prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in rows]
    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    unscored: list[dict[str, Any]] = []
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()
    partial = output / "predictions.partial.jsonl"

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in ARMS:
            generator = torch.Generator().manual_seed(seed)
            original = install_aggregator(arm["aggregator"])
            try:
                condition = Condition(
                    condition_id=arm["id"],
                    policy="significance",
                    exponent=float(arm["exponent"]),
                    query_conditioned=True,
                )
                callback = make_callback(
                    float(arm["exponent"]),
                    shuffle_scores=bool(arm["shuffle_scores"]),
                    cache_config=config["cache"],
                    generator=generator,
                )
                for case in prepared:
                    result = execute_condition(
                        model,
                        case,
                        condition,
                        cache_config=config["cache"],
                        runtime_config=config["runtime"],
                        eos_token_ids=eos_ids,
                        compression_callback=callback,
                    )
                    prediction = tokenizer.decode(
                        result.pop("generated_token_ids"), skip_special_tokens=True
                    ).strip()
                    unscored.append(
                        {
                            "case_id": case.case_id,
                            "scenario_id": case.scenario_id,
                            "arm": arm["id"],
                            "arm_label": arm["label"],
                            "aggregator": arm["aggregator"],
                            "exponent": arm["exponent"],
                            "shuffled_scores": arm["shuffle_scores"],
                            "prediction": prediction,
                            "input_tokens": int(case.input_ids.shape[-1]),
                            "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                            "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                            "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                        }
                    )
                    handle.write(json.dumps(unscored[-1], ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    done += 1
                    if done % 16 == 0 or done == total:
                        rate = (time.perf_counter() - started) / done
                        print(f"  {done}/{total} arm={arm['id']} eta={(total-done)*rate/60:.1f} min", flush=True)
                    gc.collect()
            finally:
                SignificanceCache.record_attention = original

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)
    answer_key = read_answer_key(answer_key_path, case_ids)
    for row in unscored:
        key_row = answer_key[row["case_id"]]
        accepted = {_normalize_answer(v) for v in key_row["accepted_answers"]}
        row["correct"] = _normalize_answer(str(row["prediction"])) in accepted
        row["variant"] = key_row.get("variant")
        row["needle_profile"] = key_row.get("needle_profile")
        row["reference_answer"] = key_row["answer"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        selected = [r for r in unscored if r["arm"] == arm["id"]]
        correct = sum(bool(r["correct"]) for r in selected)
        needle = {}
        for profile in sorted({str(r["needle_profile"]) for r in selected}):
            group = [r for r in selected if str(r["needle_profile"]) == profile]
            needle[profile] = {"cases": len(group), "correct_cases": sum(bool(r["correct"]) for r in group)}
        by_arm[arm["id"]] = {
            "label": arm["label"],
            "aggregator": arm["aggregator"],
            "exponent": arm["exponent"],
            "shuffled_scores": arm["shuffle_scores"],
            "cases": len(selected),
            "correct_cases": correct,
            "exact_accuracy": correct / len(selected) if selected else None,
            "by_needle_profile": needle,
            "total_inference_wall_seconds": sum(float(r["inference_wall_seconds"]) for r in selected),
        }

    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_SCREEN",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only",
        "benchmark_scope": "complete 64-case showcase split",
        "questions_tested": [
            "P4: does a spike-preserving S aggregator beat the shipped sum?",
            "Foundational control: does S carry any usable signal, or is it all sink-plus-recent structure?",
        ],
        "reference_arms": {
            "B0_full_cache": "30/64",
            "B2_sum_e0": "17/64",
            "B3_sum_e1": "19/64",
            "B4_POS_position": "22/64",
            "R192_positional_heavy": "25/64",
            "SG10_E1_gated_exposure": "16/64",
        },
        "arms": by_arm,
        "decision_rules": {
            "P4": "MAX_E0 must exceed B2 (17/64), which shares e=0 and differs only in aggregation.",
            "P4_specificity": "MAX_E0 must exceed MAX_SHUF.",
            "foundational": (
                "If SUM_SHUF matches B3 (19/64), the significance score carries no usable "
                "retention information and the mechanism's behaviour is structural."
            ),
        },
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exploratory screen; no frozen endpoint and no advance gate.",
            "Scenario packs are correlated; 64 rows are not 64 independent observations.",
            "Reference arms come from separate runs and are compared descriptively.",
            "record_attention was monkeypatched for the run; no source file changed.",
            "MAX and EMA deltas pass through the audited update() path unchanged.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": "1.0",
            "run_id": output.name,
            "created_at": started_at,
            "completed_at": utc_now(),
            "classification": "EXPLORATORY_SCREEN",
            "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
            "seed": seed,
            "ema_alpha": EMA_ALPHA,
            "model": {
                "status": "VERIFIED",
                "id": str(core["core_id"]),
                "checkpoint_aggregate_sha256": checkpoint["aggregate_sha256"],
            },
            "benchmark": {
                "split": "showcase",
                "case_count": len(case_ids),
                "prompt_sha256": sha256_file(prompt_path),
                "answer_key_sha256_after_inference": sha256_file(answer_key_path),
            },
            "arms": [
                {k: a[k] for k in ("id", "label", "aggregator", "exponent", "shuffle_scores")}
                for a in ARMS
            ],
            "notes": [
                "Implements audit repair P4: spike-preserving S aggregation.",
                "Adds the first shuffled-S control in the program; prior shuffles permuted counts only.",
                "MAX and EMA are applied as deltas through the audited update() path.",
                "Shuffles preserve the eligible-pool multiset and both protected denominators.",
                "No file under src/ was modified; record_attention was patched in-process.",
            ],
        },
    )
    print(json.dumps({"output": str(output)}, indent=2))
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
