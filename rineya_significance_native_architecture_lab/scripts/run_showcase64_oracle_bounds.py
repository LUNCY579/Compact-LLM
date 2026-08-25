#!/usr/bin/env python3
"""Oracle bounds: how much can KV selection possibly matter at this budget?

WHAT THESE ARMS ARE
-------------------
These are NOT policies and NOT model performance.  The oracle arms read the
answer key *during* inference by construction, which every other run in this
lab is forbidden to do.  They exist to bound the achievable range of any
content-selection rule, including every Significance Matrix variant.

    ORACLE_KEEP   entries whose token appears in the gold answer are ranked
                  above everything else; the rest of the heavy pool is filled
                  by S.  This is perfect content selection.
    ANTI_ORACLE   the same entries are ranked below everything else.  This is
                  worst-case content selection.
    RANDOM        uniform random ranking, the no-information midpoint.

The spread between ORACLE_KEEP and ANTI_ORACLE is the total dynamic range
available to selection at a fixed budget, sink count, and recent window.  If
that spread is small, no ranking function -- however good its signal -- can buy
much, and the Significance Matrix's ceiling is structural rather than
algorithmic.  If it is large, the signal-to-retention gap is worth attacking.

Motivating measurement: with a frequency-matched control, S separates
answer-bearing tokens from equally-rare non-answer tokens with head-mean AUC
0.5412 and best-head AUC 0.7178.  The signal exists.  These bounds say what it
could be worth.

Sinks and the recent window stay protected in every arm, so the contrast
isolates the heavy pool -- exactly the slots the significance score governs.

Exploratory. Halo only. Oracle arms are answer-informed by design and are
recorded as bounds, never as evidence of policy quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
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

ARMS = (
    {"id": "ORACLE_KEEP", "mode": "oracle", "label": "answer_tokens_ranked_first_upper_bound"},
    {"id": "ANTI_ORACLE", "mode": "anti", "label": "answer_tokens_ranked_last_lower_bound"},
    {"id": "RANDOM", "mode": "random", "label": "uniform_random_ranking_no_information"},
)
RARE_MAX = 3


def answer_positions(tokenizer, input_ids: list[int], accepted: list[str]) -> set[int]:
    """Prompt positions holding a rare sub-token of the gold answer."""

    freq = Counter(input_ids)
    ids: set[int] = set()
    for answer in accepted:
        for form in (answer, " " + answer):
            ids.update(tokenizer(form, add_special_tokens=False)["input_ids"])
    ids = {t for t in ids if 0 < freq.get(t, 0) <= RARE_MAX}
    return {p for p, t in enumerate(input_ids) if t in ids}


def make_callback(mode: str, ans_pos: set[int], generator):
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
            if mode == "random":
                ranking = torch.rand(scores.shape, generator=generator, dtype=scores.dtype)
            else:
                pos = layer.state.positions
                flag = torch.zeros_like(scores)
                mask = torch.tensor(
                    [[[int(p) in ans_pos for p in row.tolist()] for row in head] for head in pos],
                    dtype=torch.bool,
                )
                # Normalise S into (0,1) so the oracle offset dominates cleanly.
                lo = scores.amin(dim=-1, keepdim=True)
                hi = scores.amax(dim=-1, keepdim=True)
                norm = (scores - lo) / (hi - lo).clamp_min(1e-9)
                offset = 10.0 if mode == "oracle" else -10.0
                flag = torch.where(mask, torch.full_like(scores, offset), torch.zeros_like(scores))
                ranking = norm + flag
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
        / "halo_showcase64_oracle_bounds_v1"
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
    # Oracle arms are answer-informed by construction; the key is opened up front
    # and this run is permanently labelled ORACLE_BOUND.
    answer_key = read_answer_key(answer_key_path, case_ids)
    ans_positions = {
        case.case_id: answer_positions(
            tokenizer, case.input_ids[0].tolist(), answer_key[case.case_id]["accepted_answers"]
        )
        for case in prepared
    }

    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())

    scored: list[dict[str, Any]] = []
    total = len(ARMS) * len(prepared)
    done = 0
    started = time.perf_counter()

    for arm in ARMS:
        generator = torch.Generator().manual_seed(seed)
        condition = Condition(
            condition_id=arm["id"], policy="significance", exponent=1.0, query_conditioned=True
        )
        for case in prepared:
            callback = make_callback(arm["mode"], ans_positions[case.case_id], generator)
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
            key_row = answer_key[case.case_id]
            accepted = {_normalize_answer(v) for v in key_row["accepted_answers"]}
            scored.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "arm": arm["id"],
                    "arm_label": arm["label"],
                    "oracle_informed": arm["mode"] in {"oracle", "anti"},
                    "prediction": prediction,
                    "correct": _normalize_answer(prediction) in accepted,
                    "variant": key_row.get("variant"),
                    "needle_profile": key_row.get("needle_profile"),
                    "answer_positions_found": len(ans_positions[case.case_id]),
                    "reference_answer": key_row["answer"],
                    "inference_wall_seconds": result["timing"]["inference_wall_seconds"],
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
        selected = [r for r in scored if r["arm"] == arm["id"]]
        correct = sum(bool(r["correct"]) for r in selected)
        needle = {}
        for profile in sorted({str(r["needle_profile"]) for r in selected}):
            group = [r for r in selected if str(r["needle_profile"]) == profile]
            needle[profile] = {"cases": len(group), "correct_cases": sum(bool(r["correct"]) for r in group)}
        by_arm[arm["id"]] = {
            "label": arm["label"],
            "oracle_informed": arm["mode"] in {"oracle", "anti"},
            "cases": len(selected),
            "correct_cases": correct,
            "exact_accuracy": correct / len(selected) if selected else None,
            "by_needle_profile": needle,
        }

    covered = [r for r in scored if r["arm"] == "ORACLE_KEEP" and r["answer_positions_found"] > 0]
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "ORACLE_BOUND_NOT_A_POLICY",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only",
        "benchmark_scope": "complete 64-case showcase split",
        "reference_arms": {
            "B0_full_cache": "30/64",
            "R192_best_compressed": "25/64",
            "B4_POS": "22/64",
            "B3_shipped": "19/64",
            "SUM_SHUF_shuffled_scores": "21/64",
        },
        "arms": by_arm,
        "cases_with_locatable_answer_tokens": len(covered),
        "measured_signal": {
            "S_vs_frequency_matched_negatives_head_mean_auc": 0.5412,
            "S_vs_frequency_matched_negatives_best_head_auc": 0.7178,
            "note": "The score does carry answer-location signal; these bounds say what it could be worth.",
        },
        "interpretation_limits": [
            "ORACLE_KEEP and ANTI_ORACLE read the answer key during inference. They are bounds, never policies.",
            "Neither may be cited as Halo performance or as evidence for any selection rule.",
            "Answer positions are rare sub-token matches and may miss paraphrased evidence, so ORACLE_KEEP is a lower bound on the true oracle.",
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
                "answer_key_sha256": sha256_file(answer_key_path),
                "answer_key_access": "opened_before_inference_by_design_oracle_arms",
            },
            "arms": [{k: a[k] for k in ("id", "label", "mode")} for a in ARMS],
            "notes": [
                "Bounds the dynamic range available to any KV content-selection rule.",
                "ORACLE_KEEP ranks gold-answer tokens first; ANTI_ORACLE ranks them last.",
                "RANDOM is the no-information midpoint.",
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
