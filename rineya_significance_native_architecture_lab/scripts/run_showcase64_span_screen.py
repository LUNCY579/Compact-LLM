#!/usr/bin/env python3
"""GAP-A screen: span integrity and sink coverage.

The corrected oracle (`halo_showcase64_oracle_bounds_v2`) establishes that the
entire recoverable compression deficit is selection error: ORACLE_SPAN 32/64 vs
shipped B3 19/64, 13W/0L, cluster p=0.0156, with no residual gap against full
cache (30/64). Thirteen answers are reachable by choosing better entries.

Two measured failure modes point at concrete, answer-blind fixes.

1. SPAN FRAGMENTATION. Four of eleven compression losses are partial multi-token
   entity corruptions -- 'Lucan Tern' -> 'Lucan T. T', 'Jasper Upton' ->
   'Jasper Ullman'. Selection is an independent per-position, per-KV-head top-k
   with no span-integrity constraint, so the leading name token survives and the
   trailing one is evicted. Verified against the tokenizer, these entities split
   exactly where the failures appear: ' Luc'/'an', ' T'/'ern', ' Far'/'is'.

   Two remedies, both budget-neutral at fixed keep-count:
     POOLn      max-pool the eligible scores over a width-n window. Already
                implemented at runtime/policy.py:121-133 and, before this run,
                never enabled in any executed configuration.
     SPANLINK   word-aware forward propagation: if entry p continues the word
                begun at p-1 and p-1 is still resident, lift score[p] to
                max(score[p], score[p-1]). Iterated so multi-token words
                propagate end to end. Uses only tokenizer word boundaries --
                no gold answer, no benchmark metadata.

2. SINK COVERAGE. The abstention instruction ("If the context does not determine
   the answer, reply UNKNOWN") sits at token position 38, outside the four
   protected sink slots, and the S/C^e recency prior evicts it. Compression drove
   unanswerable accuracy from 2/16 to 0/16. `SINK48` protects the whole system
   prompt. One integer.

CONTROLS
  POOL1        must reproduce B3's 19/64 exactly, or the harness is wrong.
  SPANLINK_RAND propagates along randomly chosen boundaries, matched in count to
               SPANLINK's real ones, so a gain cannot be attributed to "keeping
               more contiguous stuff" independent of word structure.

Exploratory and unregistered. Halo only. The answer key is opened only after all
inference and after the answer-blind artifact is durably written.
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

SPACE_MARKER = chr(0x120)  # GPT-2/Qwen BPE leading-space marker
LINK_ITERATIONS = 4  # propagates through words of up to five tokens

ARMS = (
    {"id": "POOL1", "pool": 1, "sinks": 4, "link": "none", "label": "baseline_identity_gate"},
    {"id": "POOL3", "pool": 3, "sinks": 4, "link": "none", "label": "max_pool_width_3"},
    {"id": "POOL5", "pool": 5, "sinks": 4, "link": "none", "label": "max_pool_width_5"},
    {"id": "POOL7", "pool": 7, "sinks": 4, "link": "none", "label": "max_pool_width_7"},
    {"id": "SPANLINK", "pool": 1, "sinks": 4, "link": "word", "label": "word_aware_forward_propagation"},
    {"id": "SPANLINK_RAND", "pool": 1, "sinks": 4, "link": "random", "label": "count_matched_random_link_control"},
    {"id": "SINK48", "pool": 1, "sinks": 48, "link": "none", "label": "sinks_cover_system_prompt"},
    {"id": "SPAN_SINK48", "pool": 1, "sinks": 48, "link": "word", "label": "spanlink_plus_sink_coverage"},
)


# Pooling was still climbing at width 7 (19/21/25/26 for widths 1/3/5/7), so the
# sweep is extended to locate the peak, plus the best pooling arm combined with
# sink coverage. POOL1 repeats as the identity gate.
WIDE_ARMS = (
    {"id": "POOL1", "pool": 1, "sinks": 4, "link": "none", "label": "baseline_identity_gate"},
    {"id": "POOL9", "pool": 9, "sinks": 4, "link": "none", "label": "max_pool_width_9"},
    {"id": "POOL13", "pool": 13, "sinks": 4, "link": "none", "label": "max_pool_width_13"},
    {"id": "POOL21", "pool": 21, "sinks": 4, "link": "none", "label": "max_pool_width_21"},
    {"id": "POOL7_SINK48", "pool": 7, "sinks": 48, "link": "none", "label": "pool7_plus_sink_coverage"},
    {"id": "POOL13_SINK48", "pool": 13, "sinks": 48, "link": "none", "label": "pool13_plus_sink_coverage"},
)


def continuation_flags(tokenizer, input_ids: list[int]) -> torch.Tensor:
    """True where a token continues the word begun by its predecessor."""

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return torch.tensor(
        [False] + [not str(t).startswith(SPACE_MARKER) for t in tokens[1:]], dtype=torch.bool
    )


def apply_span_link(
    scores: torch.Tensor, positions: torch.Tensor, cont: torch.Tensor, *, mode: str, generator
) -> torch.Tensor:
    """Lift a continuation entry's score to its predecessor's, budget-neutral."""

    if mode == "none":
        return scores
    out = scores.clone()
    # Slot j's position-predecessor is slot j-1 only when positions are adjacent.
    adjacent = positions[..., 1:] == positions[..., :-1] + 1
    idx = positions[..., 1:].clamp_min(0).clamp_max(cft_len := cont.numel() - 1)
    is_cont = cont.to(positions.device)[idx]
    link = adjacent & is_cont
    if mode == "random":
        # Same number of links per row, placed at random eligible slots.
        flat = link.reshape(-1, link.shape[-1])
        for row in range(flat.shape[0]):
            n = int(flat[row].sum())
            perm = torch.randperm(flat.shape[-1], generator=generator)
            new = torch.zeros_like(flat[row])
            new[perm[:n]] = True
            flat[row] = new
        link = flat.reshape(link.shape)
    elif mode != "word":
        raise ValueError(f"unsupported link mode: {mode}")
    for _ in range(LINK_ITERATIONS):
        out[..., 1:] = torch.where(link, torch.maximum(out[..., 1:], out[..., :-1]), out[..., 1:])
    return out


def make_callback(*, exponent: float, cont: torch.Tensor, link_mode: str, generator):
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
            ranking = layer.state.scores / layer.state.counts.clamp_min(1.0).pow(exponent)
            ranking = apply_span_link(
                ranking, layer.state.positions, cont, mode=link_mode, generator=generator
            )
            if not torch.isfinite(ranking).all():
                raise RuntimeError(f"layer {layer.layer_idx}: non-finite ranking")
            # policy carries pool_width from cache_config, applied inside selection
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


def run(config_path: Path, output_override: Path | None = None, arms=ARMS) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config = _load_mapping(config_path.expanduser().resolve())
    config["_source_config_path"] = _repo_path(config_path)
    model_path = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
    prompt_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl"
    answer_key_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_answer_key.jsonl"
    output = (
        output_override.expanduser().resolve()
        if output_override
        else LAB_ROOT / "results" / "runs" / "halo_showcase64_span_screen_v1"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=False, low_cpu_mem_usage=True,
        ).to("cpu").eval()
    )
    _check_geometry(model.config, config["model"])

    rows = read_showcase_rows(prompt_path)
    case_ids = tuple(str(r["id"]) for r in rows)
    prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in rows]
    conts = {c.case_id: continuation_flags(tokenizer, c.input_ids[0].tolist()) for c in prepared}
    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    unscored: list[dict[str, Any]] = []
    total = len(arms) * len(prepared)
    done = 0
    started = time.perf_counter()
    partial = output / "predictions.partial.jsonl"

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in arms:
            generator = torch.Generator().manual_seed(seed)
            cache_config = dict(config["cache"])
            cache_config["sink_tokens"] = int(arm["sinks"])
            cache_config["score_pool_width"] = int(arm["pool"])
            condition = Condition(
                condition_id=arm["id"], policy="significance", exponent=1.0, query_conditioned=True
            )
            for case in prepared:
                result = execute_condition(
                    model, case, condition,
                    cache_config=cache_config,
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_ids,
                    compression_callback=make_callback(
                        exponent=1.0, cont=conts[case.case_id],
                        link_mode=str(arm["link"]), generator=generator,
                    ),
                )
                prediction = tokenizer.decode(
                    result.pop("generated_token_ids"), skip_special_tokens=True
                ).strip()
                unscored.append({
                    "case_id": case.case_id, "scenario_id": case.scenario_id,
                    "arm": arm["id"], "arm_label": arm["label"],
                    "pool_width": arm["pool"], "sink_tokens": arm["sinks"], "link_mode": arm["link"],
                    "prediction": prediction,
                    "resident_after_prefill": result["accounting"]["resident_entries_after_prefill"],
                    "total_persistent_bytes": result["efficiency"]["architecture_total_persistent_bytes"],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
                })
                handle.write(json.dumps(unscored[-1], ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                done += 1
                if done % 32 == 0 or done == total:
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
        row["answerable"] = key_row.get("answerable")
        row["needle_profile"] = key_row.get("needle_profile")
        row["reference_answer"] = key_row["answer"]
    _atomic_write_jsonl(output / "cases.jsonl", unscored)

    by_arm: dict[str, Any] = {}
    for arm in arms:
        sel = [r for r in unscored if r["arm"] == arm["id"]]
        corr = sum(bool(r["correct"]) for r in sel)
        unans = [r for r in sel if not r.get("answerable")]
        needle = {}
        for profile in sorted({str(r["needle_profile"]) for r in sel}):
            grp = [r for r in sel if str(r["needle_profile"]) == profile]
            needle[profile] = {"cases": len(grp), "correct_cases": sum(bool(r["correct"]) for r in grp)}
        by_arm[arm["id"]] = {
            "label": arm["label"], "pool_width": arm["pool"],
            "sink_tokens": arm["sinks"], "link_mode": arm["link"],
            "cases": len(sel), "correct_cases": corr,
            "exact_accuracy": corr / len(sel) if sel else None,
            "unanswerable": {"cases": len(unans), "correct_cases": sum(bool(r["correct"]) for r in unans)},
            "by_needle_profile": needle,
        }

    identity_ok = by_arm["POOL1"]["correct_cases"] == 19
    summary = {
        "schema_version": "1.0", "status": "PASS",
        "classification": "EXPLORATORY_SCREEN",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only",
        "targets": "GAP A selection error, 13 answers, per corrected oracle N-062..N-064",
        "identity_gate": {
            "rule": "POOL1 must reproduce the recorded B3 result of 19/64",
            "observed": by_arm["POOL1"]["correct_cases"],
            "passed": identity_ok,
        },
        "reference_arms": {
            "ORACLE_SPAN_corrected": "32/64", "B0_full_cache": "30/64",
            "R192": "25/64", "B3_shipped": "19/64",
            "B3_unanswerable": "0/16", "B0_unanswerable": "2/16",
        },
        "arms": by_arm,
        "specificity_rule": "SPANLINK must exceed SPANLINK_RAND, which is count-matched but word-boundary-blind.",
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exploratory screen; no frozen endpoint and no advance gate.",
            "Scenario packs are correlated; 64 rows are not 64 independent observations.",
            "Eight arms with no multiplicity control; treat any winner as a candidate needing independent confirmation.",
            "All interventions are answer-blind: tokenizer word boundaries and a sink count only.",
            "record_attention untouched; no file under src/ modified.",
        ],
    }
    _atomic_write_json(output / "summary.json", summary)
    _atomic_write_json(output / "manifest.json", {
        "schema_version": "1.0", "run_id": output.name,
        "created_at": started_at, "completed_at": utc_now(),
        "classification": "EXPLORATORY_SCREEN",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE", "seed": seed,
        "benchmark": {
            "split": "showcase", "case_count": len(case_ids),
            "prompt_sha256": sha256_file(prompt_path),
            "answer_key_sha256_after_inference": sha256_file(answer_key_path),
            "answer_key_join_order": "opened_after_unscored_predictions_were_written",
        },
        "arms": [{k: a[k] for k in ("id", "label", "pool", "sinks", "link")} for a in arms],
        "notes": [
            "First execution of score pooling in this lab; pool_width had never been enabled.",
            "SPANLINK uses tokenizer word boundaries only and is budget-neutral.",
            "SINK48 covers the system prompt containing the UNKNOWN abstention instruction at token 38.",
            "POOL1 is an identity gate against the recorded B3 result.",
        ],
    })
    print(json.dumps({"output": str(output), "identity_gate_passed": identity_ok}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_eval.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wide", action="store_true", help="run the extended pooling sweep")
    args = parser.parse_args(argv)
    run(args.config, args.output, WIDE_ARMS if args.wide else ARMS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
