#!/usr/bin/env python3
"""Corrected oracle bounds. Supersedes run_showcase64_oracle_bounds.py.

WHY v2 EXISTS
-------------
v1 located answer positions by matching individual answer sub-tokens and then
discarding any sub-token occurring more than ``RARE_MAX = 3`` times in the
prompt.  That filter was imported from an AUC diagnostic, where it correctly
prevented common sub-words from contaminating a rank statistic.  Applying it to
the oracle was wrong.

Measured consequence: the v1 oracle protected nothing on 20 of 64 cases and ran
as plain B3 there.  All 20 of those cases contain the gold answer literally in
the context; their sub-tokens simply recur 4-9 times because these tasks are
built around tracking a single named entity through a long document -- exactly
the tokens the filter removed.

    RLC-023-B  'Faris Dane'  -> ' Far'x7, 'is'x7, ' Dane'x7
    RLC-032-B  'Lucan Tern'  -> ' Luc'x5, 'an'x5, ' T'x5, 'ern'x5

So the v1 upper bound was not an upper bound.  Any conclusion drawn from it
about the ceiling of content selection is invalid and is withdrawn.

THE FIX
-------
Locate answers by CONTIGUOUS TOKEN-SPAN match rather than per-token match with
a rarity filter.  Every literal occurrence of the gold answer's token sequence
is protected in full, with no frequency threshold.  This is both stricter
(no incidental sub-token collisions) and complete (recurring entities included).

Arms, all at the same budget 256 / sinks 4 / recent 64:
    ORACLE_SPAN   answer spans ranked above everything else  (upper bound)
    ANTI_SPAN     answer spans ranked below everything else  (lower bound)

These read the answer key during inference by construction.  They are bounds,
never policies, and may never be cited as model performance.
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

ARMS = (
    {"id": "ORACLE_SPAN", "mode": "oracle", "label": "answer_spans_first_corrected_upper_bound"},
    {"id": "ANTI_SPAN", "mode": "anti", "label": "answer_spans_last_corrected_lower_bound"},
)


def _find_subsequence(haystack: list[int], needle: list[int]) -> list[int]:
    """Return every position covered by a contiguous occurrence of ``needle``."""

    if not needle or len(needle) > len(haystack):
        return []
    covered: list[int] = []
    first = needle[0]
    span = len(needle)
    for start in range(len(haystack) - span + 1):
        if haystack[start] != first:
            continue
        if haystack[start : start + span] == needle:
            covered.extend(range(start, start + span))
    return covered


def answer_span_positions(tokenizer, input_ids: list[int], accepted: list[str]) -> set[int]:
    """Positions covered by any literal occurrence of a gold answer. No frequency filter."""

    positions: set[int] = set()
    for answer in accepted:
        text = str(answer).strip()
        if not text:
            continue
        for form in (text, " " + text, text.lower(), " " + text.lower()):
            needle = tokenizer(form, add_special_tokens=False)["input_ids"]
            positions.update(_find_subsequence(input_ids, needle))
    return positions


def make_callback(mode: str, ans_pos: set[int]):
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
            pos = layer.state.positions
            mask = torch.zeros(scores.shape, dtype=torch.bool)
            if ans_pos:
                flat = pos.reshape(-1)
                hit = torch.tensor([int(p) in ans_pos for p in flat.tolist()], dtype=torch.bool)
                mask = hit.reshape(pos.shape)
            lo = scores.amin(dim=-1, keepdim=True)
            hi = scores.amax(dim=-1, keepdim=True)
            norm = (scores - lo) / (hi - lo).clamp_min(1e-9)
            offset = 10.0 if mode == "oracle" else -10.0
            ranking = norm + torch.where(
                mask, torch.full_like(scores, offset), torch.zeros_like(scores)
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
    model_path = REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Halo-1.2.0-devfreeze"
    prompt_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl"
    answer_key_path = REPO_ROOT / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_answer_key.jsonl"
    output = (
        output_override.expanduser().resolve()
        if output_override
        else LAB_ROOT / "results" / "runs" / "halo_showcase64_oracle_bounds_v2"
    )
    _create_output_dir(output)
    _write_resolved_config(output / "resolved_config.yaml", config)

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
    answer_key = read_answer_key(answer_key_path, case_ids)
    ans_positions = {
        c.case_id: answer_span_positions(
            tokenizer, c.input_ids[0].tolist(), answer_key[c.case_id]["accepted_answers"]
        )
        for c in prepared
    }
    fired = sum(1 for v in ans_positions.values() if v)
    answerable = sum(1 for c in case_ids if answer_key[c].get("answerable"))
    print(f"oracle fires on {fired}/64 cases ({answerable} are answerable); v1 fired on 44/64", flush=True)

    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    scored: list[dict[str, Any]] = []
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()
    for arm in ARMS:
        condition = Condition(
            condition_id=arm["id"], policy="significance", exponent=1.0, query_conditioned=True
        )
        for case in prepared:
            result = execute_condition(
                model,
                case,
                condition,
                cache_config=config["cache"],
                runtime_config=config["runtime"],
                eos_token_ids=eos_ids,
                compression_callback=make_callback(arm["mode"], ans_positions[case.case_id]),
            )
            prediction = tokenizer.decode(
                result.pop("generated_token_ids"), skip_special_tokens=True
            ).strip()
            key_row = answer_key[case.case_id]
            accepted = {_normalize_answer(v) for v in key_row["accepted_answers"]}
            scored.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "arm": arm["id"],
                    "arm_label": arm["label"],
                    "oracle_informed": True,
                    "prediction": prediction,
                    "correct": _normalize_answer(prediction) in accepted,
                    "variant": key_row.get("variant"),
                    "answerable": key_row.get("answerable"),
                    "needle_profile": key_row.get("needle_profile"),
                    "answer_span_positions": len(ans_positions[case.case_id]),
                    "oracle_fired": bool(ans_positions[case.case_id]),
                    "reference_answer": key_row["answer"],
                }
            )
            done += 1
            if done % 16 == 0 or done == total:
                rate = (time.perf_counter() - started) / done
                print(f"  {done}/{total} arm={arm['id']} eta={(total-done)*rate/60:.1f} min", flush=True)
            gc.collect()

    _atomic_write_jsonl(output / "cases.jsonl", scored)
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        sel = [r for r in scored if r["arm"] == arm["id"]]
        corr = sum(bool(r["correct"]) for r in sel)
        fired_rows = [r for r in sel if r["oracle_fired"]]
        ans_rows = [r for r in sel if r.get("answerable")]
        by_arm[arm["id"]] = {
            "label": arm["label"],
            "cases": len(sel),
            "correct_cases": corr,
            "exact_accuracy": corr / len(sel) if sel else None,
            "on_cases_where_oracle_fired": {
                "cases": len(fired_rows),
                "correct_cases": sum(bool(r["correct"]) for r in fired_rows),
            },
            "on_answerable_cases": {
                "cases": len(ans_rows),
                "correct_cases": sum(bool(r["correct"]) for r in ans_rows),
            },
        }
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "ORACLE_BOUND_NOT_A_POLICY",
        "supersedes": "halo_showcase64_oracle_bounds_v1",
        "correction": (
            "v1 filtered answer sub-tokens to those occurring at most 3 times, which "
            "disabled the oracle entirely on 20 of 64 cases, all of which contain the "
            "gold answer literally. v1's bound is withdrawn."
        ),
        "oracle_fired_on_cases": fired,
        "v1_oracle_fired_on_cases": 44,
        "reference_arms": {
            "B0_full_cache": "30/64",
            "R192_best_compressed": "25/64",
            "B3_shipped": "19/64",
            "v1_ORACLE_KEEP_invalid": "25/64",
        },
        "arms": by_arm,
        "interpretation_limits": [
            "Oracle arms read the answer key during inference. Bounds only, never policies.",
            "Span matching finds literal occurrences; paraphrased or inferred evidence is still missed.",
            "Sinks and the recent window remain protected, so the contrast isolates the heavy pool.",
            "Scenario packs are correlated; the 16-pack unit governs any comparison.",
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
            "classification": "ORACLE_BOUND_NOT_A_POLICY",
            "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
            "supersedes": "halo_showcase64_oracle_bounds_v1",
            "seed": seed,
            "benchmark": {
                "split": "showcase",
                "case_count": len(case_ids),
                "prompt_sha256": sha256_file(prompt_path),
                "answer_key_sha256": sha256_file(answer_key_path),
                "answer_key_access": "opened_before_inference_by_design_oracle_arms",
            },
            "notes": [
                "Corrects a rarity filter that silently disabled the v1 oracle on 20/64 cases.",
                "Answer positions located by contiguous token-span match, no frequency threshold.",
                "No file under src/ was modified.",
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
