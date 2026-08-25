#!/usr/bin/env python3
"""Screen Cornerstone-1 selection policies on the complete 64-case showcase split.

The showcase-64 evaluation found zero compression loss on late-needle cases and
large loss on early, middle, and distributed needles.  That pattern is the
signature of the positional recency window, not of content-significance
selection.  This screen tests that reading directly.

Arms are declared in the config.  Each arm varies only the selection policy --
ranking mode, normalization exponent, recency width, or total budget -- while
holding prompt tokens, chunk boundaries, query access, and the manual greedy
decoder fixed.  The decisive arm is ``B4_POS``: absolute position substituted
for the significance score at identical budget.

Execution reuses ``run_published_core_eval.execute_condition`` unchanged, so
this screen, the showcase-64 evaluation, and the recorded mini evaluation all
share one audited execution path.  The model is loaded once and reused across
arms.

Exploratory and unregistered.  The answer key is opened only after every arm
has finished and the answer-blind artifact is durably written.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

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


def parse_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("arms")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.arms must be a non-empty list")
    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        arm_id = str(item["id"])
        if arm_id in seen:
            raise ValueError(f"duplicate arm id: {arm_id}")
        seen.add(arm_id)
        policy = str(item["policy"])
        if policy not in {"significance", "position"}:
            raise ValueError(f"{arm_id}: policy must be 'significance' or 'position'")
        budget = int(item["budget"])
        sinks = int(item["sinks"])
        recent = int(item["recent"])
        if sinks < 0 or recent < 0 or sinks + recent >= budget:
            raise ValueError(f"{arm_id}: budget must leave at least one content-selected slot")
        pool_width = int(item.get("pool_width", 1))
        if pool_width <= 0 or pool_width % 2 == 0:
            raise ValueError(f"{arm_id}: pool_width must be a positive odd integer")
        arms.append(
            {
                "id": arm_id,
                "label": str(item.get("label", arm_id)),
                "policy": policy,
                "exponent": float(item["exponent"]),
                "budget": budget,
                "sinks": sinks,
                "recent": recent,
                "heavy": budget - sinks - recent,
                "pool_width": pool_width,
                # Pooling is a Cornerstone-2 span factor, not Cornerstone 1.
                "mechanism_scope": "cornerstone_1" if pool_width == 1 else "cornerstone_2_span_factor",
                "question": str(item.get("question", "")).strip(),
            }
        )
    return arms


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config_path = config_path.expanduser().resolve()
    config = _load_mapping(config_path)
    config["_source_config_path"] = _repo_path(config_path)
    arms = parse_arms(config)

    base = _declared_base(config)
    paths = config["paths"]
    model_path = _resolve_path(paths["model"], base=base)
    prompt_path = _resolve_path(paths["prompts"], base=base)
    answer_key_path = _resolve_path(paths["answer_key"], base=base)
    output = (
        output_override.expanduser().resolve()
        if output_override
        else _resolve_path(paths["output_root"], base=base)
    )
    _create_output_dir(output)
    artifact_paths = {name: output / str(f) for name, f in config["outputs"].items()}
    _write_resolved_config(artifact_paths["resolved_config"], config)

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

    prompt_rows = read_showcase_rows(prompt_path)
    case_ids = tuple(str(r["id"]) for r in prompt_rows)
    prepared = [_prepare_case(tokenizer, r, config["prompt"], config["query"]) for r in prompt_rows]

    eos = tokenizer.eos_token_id
    eos_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())
    chunk = int(config["cache"]["compression_chunk_tokens"])

    unscored: list[dict[str, Any]] = []
    partial = output / "predictions.partial.jsonl"
    total = len(arms) * len(prepared)
    done = 0
    started = time.perf_counter()

    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for arm in arms:
            cache_config = {
                "retained_history_budget": arm["budget"],
                "sink_tokens": arm["sinks"],
                "recent_tokens": arm["recent"],
                "compression_chunk_tokens": chunk,
                "score_pool_width": arm["pool_width"],
            }
            condition = Condition(
                condition_id=arm["id"],
                policy=arm["policy"],
                exponent=arm["exponent"],
                query_conditioned=True,
            )
            for case in prepared:
                result = execute_condition(
                    model,
                    case,
                    condition,
                    cache_config=cache_config,
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_ids,
                )
                prediction = tokenizer.decode(
                    result.pop("generated_token_ids"), skip_special_tokens=True
                ).strip()
                row = {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "arm": arm["id"],
                    "arm_label": arm["label"],
                    "policy": arm["policy"],
                    "exponent": arm["exponent"],
                    "budget": arm["budget"],
                    "sinks": arm["sinks"],
                    "recent": arm["recent"],
                    "heavy": arm["heavy"],
                    "pool_width": arm["pool_width"],
                    "mechanism_scope": arm["mechanism_scope"],
                    "prediction": prediction,
                    "input_tokens": int(case.input_ids.shape[-1]),
                    "input_token_sha256": case.input_token_sha256,
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
                    print(
                        f"  {done}/{total}  arm={arm['id']}  eta={(total - done) * rate / 60:.1f} min",
                        flush=True,
                    )
                gc.collect()

    _atomic_write_jsonl(output / "predictions.unscored.jsonl", unscored)

    # First answer-key access of the run.
    answer_key = read_answer_key(answer_key_path, case_ids)
    for row in unscored:
        accepted = {_normalize_answer(v) for v in answer_key[row["case_id"]]["accepted_answers"]}
        key_row = answer_key[row["case_id"]]
        row["correct"] = _normalize_answer(str(row["prediction"])) in accepted
        row["variant"] = key_row.get("variant")
        row["needle_profile"] = key_row.get("needle_profile")
        row["task_type"] = key_row.get("task_type")
        row["difficulty"] = key_row.get("difficulty")
        row["reference_answer"] = key_row["answer"]
    _atomic_write_jsonl(artifact_paths["cases"], unscored)

    by_arm: dict[str, Any] = {}
    for arm in arms:
        rows = [r for r in unscored if r["arm"] == arm["id"]]
        correct = sum(bool(r["correct"]) for r in rows)
        needle: dict[str, Any] = {}
        for profile in sorted({str(r["needle_profile"]) for r in rows}):
            group = [r for r in rows if str(r["needle_profile"]) == profile]
            needle[profile] = {
                "cases": len(group),
                "correct_cases": sum(bool(r["correct"]) for r in group),
            }
        by_arm[arm["id"]] = {
            "label": arm["label"],
            "question": arm["question"],
            "policy": arm["policy"],
            "exponent": arm["exponent"],
            "budget": arm["budget"],
            "sinks": arm["sinks"],
            "recent": arm["recent"],
            "heavy_slots": arm["heavy"],
            "pool_width": arm["pool_width"],
            "mechanism_scope": arm["mechanism_scope"],
            "cases": len(rows),
            "correct_cases": correct,
            "exact_accuracy": correct / len(rows) if rows else None,
            "median_resident_after_prefill": sorted(r["resident_after_prefill"] for r in rows)[len(rows) // 2],
            "median_total_persistent_bytes": sorted(r["total_persistent_bytes"] for r in rows)[len(rows) // 2],
            "total_inference_wall_seconds": sum(float(r["inference_wall_seconds"]) for r in rows),
            "by_needle_profile": needle,
        }

    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "EXPLORATORY_SCREEN",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only",
        "benchmark_scope": "complete 64-case Rineya Long Context Demo showcase split",
        "answer_key_opened_only_after_all_inference": True,
        "arms": by_arm,
        "inferential_statistics": None,
        "interpretation_limits": [
            "Exploratory screen; no endpoint was frozen and no advance gate was declared.",
            "Scenario packs contain four correlated variants; 64 rows are not 64 independent observations.",
            "Arms are compared descriptively; no multiplicity control is applied across arms.",
            "Results describe Halo 1.2.0 on one suite at one chunk schedule only.",
        ],
    }
    _atomic_write_json(artifact_paths["summary"], summary)

    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "created_at": started_at,
        "completed_at": utc_now(),
        "execution_mode": "closed_loop_model_execution",
        "classification": "EXPLORATORY_SCREEN",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "seed": seed,
        "config": {
            "resolved_path": artifact_paths["resolved_config"].name,
            "sha256": sha256_file(artifact_paths["resolved_config"]),
            "source_path": str(config["_source_config_path"]).replace("\\", "/"),
        },
        "model": {
            "status": "VERIFIED",
            "id": str(core["core_id"]),
            "path": _repo_path(model_path),
            "checkpoint_files": checkpoint["files"],
            "checkpoint_aggregate_sha256": checkpoint["aggregate_sha256"],
        },
        "benchmark": {
            "split": "showcase",
            "case_count": len(case_ids),
            "prompt_sha256": sha256_file(prompt_path),
            "answer_key_sha256_after_inference": sha256_file(answer_key_path),
            "answer_key_join_order": "opened_after_unscored_predictions_were_written",
        },
        "arms": [
            {
                k: arm[k]
                for k in (
                    "id", "label", "policy", "exponent", "budget",
                    "sinks", "recent", "heavy", "pool_width", "mechanism_scope",
                )
            }
            for arm in arms
        ],
        "runtime": {
            "device": "cpu",
            "precision": "bfloat16",
            "manual_greedy": True,
            "attention_implementation": "sdpa",
        },
        "notes": [
            "Selection-policy screen on the complete 64-case showcase split.",
            "B4_POS substitutes absolute position for the significance score at identical budget.",
            "Execution reuses run_published_core_eval.execute_condition unchanged.",
            "Exploratory and unregistered; no confirmatory or published-family claim.",
        ],
    }
    _atomic_write_json(artifact_paths["manifest"], manifest)
    print(json.dumps({"output": str(output), "arms": len(arms)}, indent=2))
    return {"output": str(output), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_policy_screen.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
