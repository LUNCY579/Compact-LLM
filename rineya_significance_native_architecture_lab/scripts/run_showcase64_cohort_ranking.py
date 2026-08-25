#!/usr/bin/env python3
"""Test age-neutral cohort ranking against the S/C^e normalization.

Diagnosis this screen acts on
-----------------------------
In the live runtime ``TorchSignificanceState.update`` adds a uniform scalar
exposure of 1.0 to every resident entry at every scoring event, and there is
exactly one scoring event per non-final chunk boundary.  Therefore

    C_i == number of scoring events entry i has survived

which is exactly the entry's arrival-chunk age.  Measured on real Halo runs, C
takes one distinct value per chunk cohort -- four distinct values across 512
resident entries -- so ``S / max(C,1)^e`` is a per-cohort constant rescaling.
Within a cohort it cannot reorder anything; across cohorts it applies an
age discount.  Measured heavy-slot composition confirms the direction: moving
from ``e=0`` to ``e=1`` shifts retention toward the newest cohort by +12.0
percentage points and away from every older cohort.  Observation normalization
as implemented is a recency prior.

That is the wrong correction for long-context evidence, because it penalises
exactly the early evidence that compression is supposed to protect.

What this screen tests
----------------------
``C`` is still useful -- but as a *cohort label*, not a divisor.  Both arms
here rank entries against others of the same age, which removes the age
confound without introducing a recency prior:

    CSS    z_i = (S_i - mean(S | cohort_i)) / std(S | cohort_i)
    CMEAN  r_i = S_i / mean(S | cohort_i)

Persistent S and C are never mutated; the override affects one boundary's
selection only and is gathered with K/V exactly as before.

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

from rineya_significance_lab.manifest import sha256_file, utc_now  # noqa: E402

EPS = 1e-6


def cohort_statistics(
    scores: torch.Tensor, counts: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Rank each entry against entries of the same realized-exposure cohort."""

    values = scores.to(torch.float32)
    cohorts = counts.to(torch.float32)
    ranking = torch.zeros_like(values)
    for cohort in torch.unique(cohorts):
        mask = cohorts == cohort
        if not bool(mask.any()):
            continue
        # Per [batch, head] row statistics over this cohort's slots only.
        masked = torch.where(mask, values, torch.zeros_like(values))
        n = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
        mean = masked.sum(dim=-1, keepdim=True) / n
        if mode == "CMEAN":
            local = values / mean.clamp_min(EPS)
        elif mode == "CSS":
            variance = (
                torch.where(mask, (values - mean) ** 2, torch.zeros_like(values)).sum(
                    dim=-1, keepdim=True
                )
                / n
            )
            local = (values - mean) / variance.sqrt().clamp_min(EPS)
        else:
            raise ValueError(f"unsupported cohort mode: {mode}")
        ranking = torch.where(mask, local, ranking)
    if not torch.isfinite(ranking).all():
        raise RuntimeError("cohort ranking produced non-finite values")
    return ranking


def _compress_layer_with_ranking(layer: Any, policy: Any, ranking: torch.Tensor) -> None:
    """Replicate ``SignificanceDynamicLayer.compress`` with a supplied ranking.

    The selection and gather steps mirror the audited implementation exactly.
    They are reproduced here, rather than added as a parameter to the runtime,
    because ``runtime/cache.py`` is a content-hashed source file pinned by the
    completed Experiment F provenance record and must not drift for an
    exploratory screen.
    """

    from rineya_significance_lab.runtime.policy import select_keep_indices

    layer.assert_aligned()
    stored = layer.stored_length()
    if stored <= policy.budget:
        return
    if bool((layer.state.counts <= 0).any()):
        raise RuntimeError(
            f"layer {layer.layer_idx}: compression requested with unscored entries"
        )
    if ranking.shape != layer.state.scores.shape:
        raise RuntimeError(
            f"layer {layer.layer_idx}: ranking {tuple(ranking.shape)} does not match "
            f"state {tuple(layer.state.scores.shape)}"
        )
    indices = select_keep_indices(
        ranking.to(device=layer.state.scores.device, dtype=layer.state.scores.dtype),
        policy,
        positions=layer.state.positions,
        entry_ids=layer.state.entry_ids,
    )
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
    layer.keys = layer.keys.gather(2, gather_idx).contiguous()
    layer.values = layer.values.gather(2, gather_idx).contiguous()
    layer.state.gather(indices, cache_length=stored)
    layer.evictions += 1
    layer.assert_aligned()


def make_callback(mode: str):
    def callback(cache, policy, chunk_index):
        from rineya_significance_lab.runtime.cache import CompressionReport

        before = sum(layer.stored_length() for layer in cache.layers if layer.is_initialized)
        evicted = 0
        compressed = 0
        for layer in cache.layers:
            if not layer.is_initialized:
                continue
            old = layer.stored_length()
            override = cohort_statistics(layer.state.scores, layer.state.counts, mode=mode)
            _compress_layer_with_ranking(layer, policy, override)
            if layer.stored_length() < old:
                compressed += 1
                evicted += old - layer.stored_length()
        cache.assert_aligned()
        after = sum(layer.stored_length() for layer in cache.layers if layer.is_initialized)
        return CompressionReport(compressed, before, after, evicted)

    return callback


ARMS = (
    {"id": "CSS", "mode": "CSS", "label": "cohort_standardized_significance"},
    {"id": "CMEAN", "mode": "CMEAN", "label": "cohort_mean_ratio"},
)

# Hybrid arms: cohort-relative content selection combined with a wider recency
# window. Motivated by the measured complementarity -- cohort ranking recovers
# early/middle needles while the recency window carries late/distributed ones,
# and their correct-case overlap was only 15 of a 29-case union.
HYBRID_ARMS = (
    {"id": "HYB192", "mode": "CMEAN", "label": "cohort_mean_recent192",
     "budget": 256, "sinks": 4, "recent": 192},
    {"id": "HYB128", "mode": "CMEAN", "label": "cohort_mean_recent128",
     "budget": 256, "sinks": 4, "recent": 128},
)


def run(config_path: Path, output_override: Path | None = None, arms=ARMS) -> dict[str, Any]:
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
        / "halo_showcase64_cohort_ranking_v1"
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
    total = len(arms) * len(prepared)
    done = 0
    started = time.perf_counter()
    partial = output / "predictions.partial.jsonl"

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in arms:
            # exponent is irrelevant under an override, but stays declared.
            condition = Condition(
                condition_id=arm["id"], policy="significance", exponent=1.0, query_conditioned=True
            )
            callback = make_callback(arm["mode"])
            for case in prepared:
                arm_cache = dict(config["cache"])
                if "recent" in arm:
                    arm_cache.update(
                        retained_history_budget=arm["budget"],
                        sink_tokens=arm["sinks"],
                        recent_tokens=arm["recent"],
                    )
                result = execute_condition(
                    model,
                    case,
                    condition,
                    cache_config=arm_cache,
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_ids,
                    compression_callback=callback,
                )
                prediction = tokenizer.decode(
                    result.pop("generated_token_ids"), skip_special_tokens=True
                ).strip()
                row = {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "arm": arm["id"],
                    "arm_label": arm["label"],
                    "ranking_mode": arm["mode"],
                    "prediction": prediction,
                    "input_tokens": int(case.input_ids.shape[-1]),
                    "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                    "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                }
                unscored.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                done += 1
                if done % 16 == 0 or done == total:
                    rate = (time.perf_counter() - started) / done
                    print(f"  {done}/{total} arm={arm['id']} eta={(total-done)*rate/60:.1f} min", flush=True)
                gc.collect()

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
    for arm in arms:
        selected = [r for r in unscored if r["arm"] == arm["id"]]
        correct = sum(bool(r["correct"]) for r in selected)
        needle = {}
        for profile in sorted({str(r["needle_profile"]) for r in selected}):
            group = [r for r in selected if str(r["needle_profile"]) == profile]
            needle[profile] = {
                "cases": len(group),
                "correct_cases": sum(bool(r["correct"]) for r in group),
            }
        by_arm[arm["id"]] = {
            "label": arm["label"],
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
        "reference_arms_from_showcase64_eval": {"B2_raw_S": "17/64", "B3_S_over_C": "19/64", "B0_full_cache": "30/64"},
        "arms": by_arm,
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exploratory screen; no frozen endpoint and no advance gate.",
            "Scenario packs are correlated; 64 rows are not 64 independent observations.",
            "Reference arms come from a separate run and are compared descriptively.",
            "Persistent S and C are unmodified; only per-boundary selection changed.",
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
            "arms": [{"id": a["id"], "mode": a["mode"], "label": a["label"]} for a in arms],
            "notes": [
                "Ranks entries against same-cohort entries instead of dividing by C.",
                "Motivated by the measured identity C == arrival-chunk age.",
                "Persistent S and C are never mutated by the override.",
            ],
        },
    )
    print(json.dumps({"output": str(output)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_eval.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hybrid", action="store_true", help="run the wider-recency hybrid arms")
    args = parser.parse_args(argv)
    run(args.config, args.output, HYBRID_ARMS if args.hybrid else ARMS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
