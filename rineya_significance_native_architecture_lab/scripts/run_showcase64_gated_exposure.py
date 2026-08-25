#!/usr/bin/env python3
"""P1: make realized exposure non-uniform, then test whether C earns its keep.

Why
---
The implementation-theory audit established that the live path records a
uniform scalar exposure of 1.0 for every resident entry, so ``C`` equals
arrival-chunk age, ``S/max(C,1)^e`` is a per-cohort age discount, and
hypothesis H2 ("realized exposure contains information beyond age") is vacuous
by construction.

This screen implements the audit's P1 repair.  Exposure becomes *salience
gated*: an entry accrues exposure at a scoring event only if it drew more than
``tau`` times the uniform attention share at that event.

    exposure_i = 1  if  contribution_i > tau * mean_j(contribution_j)
    exposure_i = 0  otherwise

``C`` then counts the events at which the entry was actually salient, and
``S/max(C,1)^e`` becomes mean attention *per salient event* -- a spike
magnitude, which is age-neutral.  An early entry that spiked once and a recent
entry that spiked once receive the same score, which is precisely the property
the uniform-exposure implementation destroyed.

A premise check on real Halo confirmed the mechanism now varies per KV head for
the same absolute position, which never happened under uniform exposure.

Specificity control
-------------------
The lab has repeatedly found shuffled controls matching or beating the real
signal.  ``SG_SHUF`` therefore permutes the gate among entries that are
eligible at that event -- outside the protected sink and recent partitions --
preserving the per-row gate multiset exactly while destroying entry identity.
If ``SG_SHUF`` matches ``SG10``, the gate carries no entry-specific
information and P1 fails its specificity gate.

Implementation note
-------------------
``SignificanceCache.record_attention`` is monkeypatched for the duration of the
run rather than edited.  Every file under ``src/`` is content-hashed by the
completed Experiment F provenance record and must stay byte-identical.

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
from rineya_significance_lab.runtime.cache import SignificanceCache  # noqa: E402

ARMS = (
    {"id": "SG10_E1", "tau": 1.0, "exponent": 1.0, "shuffle": False,
     "label": "salience_gated_tau1_e1"},
    {"id": "SG10_E05", "tau": 1.0, "exponent": 0.5, "shuffle": False,
     "label": "salience_gated_tau1_e0.5"},
    {"id": "SG20_E1", "tau": 2.0, "exponent": 1.0, "shuffle": False,
     "label": "salience_gated_tau2_e1"},
    {"id": "SG_SHUF", "tau": 1.0, "exponent": 1.0, "shuffle": True,
     "label": "eligible_pool_shuffled_gate_control"},
)


def _gate(contribution: torch.Tensor, *, tau: float) -> torch.Tensor:
    mean = contribution.mean(dim=-1, keepdim=True)
    return (contribution > tau * mean).to(contribution.dtype)


def _shuffle_eligible(gate: torch.Tensor, *, sinks: int, recent: int, generator) -> torch.Tensor:
    """Permute the gate only within the eligible pool, per [batch, head] row.

    Sink and recent denominators stay fixed, and the eligible multiset is
    preserved exactly, so this is a clean identity-breaking control rather than
    the globally-permuting kind the lab flagged as eligible-pool confounded.
    """

    out = gate.clone()
    n = gate.shape[-1]
    lo, hi = sinks, n - recent
    if hi - lo <= 1:
        return out
    flat = out.reshape(-1, n)
    for row in range(flat.shape[0]):
        segment = flat[row, lo:hi]
        perm = torch.randperm(segment.numel(), generator=generator)
        flat[row, lo:hi] = segment[perm]
    return flat.reshape(gate.shape)


def install_gated_recorder(arm: dict[str, Any], cache_config: dict[str, Any], generator):
    """Replace record_attention for this arm; returns the original for restore."""

    original = SignificanceCache.record_attention
    tau = float(arm["tau"])
    shuffle = bool(arm["shuffle"])
    sinks = int(cache_config["sink_tokens"])
    recent = int(cache_config["recent_tokens"])

    def gated(self, layer_idx: int, contribution: torch.Tensor) -> None:
        if not self.recording_enabled:
            raise RuntimeError("attention recorder called outside a recording context")
        gate = _gate(contribution, tau=tau)
        if shuffle:
            gate = _shuffle_eligible(gate, sinks=sinks, recent=recent, generator=generator)
        self.layers[layer_idx].record(contribution, exposure=gate)

    SignificanceCache.record_attention = gated
    return original


def make_callback(exponent: float, stats: dict[str, Any]):
    """Compress with S/max(C,1)^e, tolerating the C==0 the gate makes legitimate."""

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
            counts = layer.state.counts
            if layer.layer_idx == 0:
                stats["distinct_c"].append(int(torch.unique(counts).numel()))
                stats["zero_c_fraction"].append(float((counts == 0).float().mean()))
                stats["head_varies"].append(bool((counts.std(dim=1) > 0).any()))
            ranking = layer.state.scores / counts.clamp_min(1.0).pow(exponent)
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
        / "halo_showcase64_gated_exposure_v1"
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
    arm_stats: dict[str, Any] = {}
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()
    partial = output / "predictions.partial.jsonl"

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in ARMS:
            generator = torch.Generator().manual_seed(seed)
            original = install_gated_recorder(arm, config["cache"], generator)
            stats = {"distinct_c": [], "zero_c_fraction": [], "head_varies": []}
            try:
                condition = Condition(
                    condition_id=arm["id"],
                    policy="significance",
                    exponent=float(arm["exponent"]),
                    query_conditioned=True,
                )
                callback = make_callback(float(arm["exponent"]), stats)
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
                            "tau": arm["tau"],
                            "exponent": arm["exponent"],
                            "shuffled_gate": arm["shuffle"],
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
            arm_stats[arm["id"]] = {
                "mean_distinct_c_values": sum(stats["distinct_c"]) / max(1, len(stats["distinct_c"])),
                "mean_zero_c_fraction": sum(stats["zero_c_fraction"]) / max(1, len(stats["zero_c_fraction"])),
                "c_varies_across_kv_heads": bool(all(stats["head_varies"])) if stats["head_varies"] else None,
                "boundaries_observed": len(stats["distinct_c"]),
            }

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
            "tau": arm["tau"],
            "exponent": arm["exponent"],
            "shuffled_gate": arm["shuffle"],
            "cases": len(selected),
            "correct_cases": correct,
            "exact_accuracy": correct / len(selected) if selected else None,
            "by_needle_profile": needle,
            "exposure_diagnostics": arm_stats[arm["id"]],
            "total_inference_wall_seconds": sum(float(r["inference_wall_seconds"]) for r in selected),
        }

    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_SCREEN",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only",
        "benchmark_scope": "complete 64-case showcase split",
        "repair_tested": "P1 non-uniform salience-gated realized exposure",
        "reference_arms": {
            "B0_full_cache": "30/64",
            "B2_raw_S": "17/64",
            "B3_uniform_exposure_S_over_C": "19/64",
            "B4_POS_position": "22/64",
            "CMEAN_cohort_relative": "22/64",
        },
        "arms": by_arm,
        "specificity_rule": (
            "SG10_E1 must exceed SG_SHUF. If the eligible-pool shuffled gate matches or "
            "exceeds the true gate, the exposure signal carries no entry-specific "
            "information and P1 fails."
        ),
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exploratory screen; no frozen endpoint and no advance gate.",
            "Scenario packs are correlated; 64 rows are not 64 independent observations.",
            "Reference arms come from separate runs and are compared descriptively.",
            "record_attention was monkeypatched for the run; no source file changed.",
            "The C==0 tolerance differs from the shipped compress guard and is stated explicitly.",
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
            "arms": [
                {k: a[k] for k in ("id", "label", "tau", "exponent", "shuffle")} for a in ARMS
            ],
            "notes": [
                "Implements audit repair P1: exposure is salience-gated, not a uniform scalar.",
                "C now counts salient scoring events and varies across KV heads.",
                "SG_SHUF permutes the gate within the eligible pool only, preserving sink and recent denominators.",
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
