#!/usr/bin/env python3
"""Run the complete 64-case showcase Cornerstone-1 evaluation on Halo.

This is a sibling of ``run_published_core_eval.py``, not a replacement.  That
script remains hard-locked to its three recorded cases so the completed
``halo_mini_eval_20260823_v2`` artifact keeps an unambiguous provenance.  This
script evaluates the *complete* 64-case showcase split of the Rineya Long
Context Demonstration Suite -- 16 scenario packs by four variants -- under the
same instrumented full-cache control (B0), raw accumulated significance (B2),
and observation-normalized significance (B3).

The compression semantics are not reimplemented here.  ``execute_condition``
and the case-preparation helpers are imported unchanged from
``run_published_core_eval`` so this run and the recorded mini evaluation share
one audited execution path.

Scope limits, deliberately:

* Halo only.  The core gate from the sibling script is retained.
* Exploratory and unregistered.  No endpoint is frozen, no advance gate is
  declared, and no inferential statistics are computed.
* The answer key is not opened until every generation has completed and the
  answer-blind prediction artifact has been durably written.

Per-condition prediction files are additionally emitted in the demonstration
suite's own ``{"id", "prediction"}`` format so that suite's own scorer can
produce metrics directly comparable with the recorded qwen3_8b, qwen3_4b,
qwen3_06b, aegis, and greedy_soup_v2 showcase baselines.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SCRIPTS = LAB_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_published_core_eval import (  # noqa: E402
    _assert_prompt_only,
    _atomic_write_json,
    _atomic_write_jsonl,
    _check_geometry,
    _checkpoint_hashes,
    _conditions_from_config,
    _create_output_dir,
    _declared_base,
    _load_jsonl,
    _load_mapping,
    _normalize_answer,
    _prepare_case,
    _repo_path,
    _resolve_path,
    _schema_validate,
    _validate_halo_core,
    _write_resolved_config,
    execute_condition,
)

from rineya_significance_lab.manifest import (  # noqa: E402
    build_manifest,
    sha256_file,
    utc_now,
)

CONDITION_IDS = ("B0", "B2", "B3")
EXPECTED_CASES = 64
# `answerable` and `variant` describe the oracle design of a scenario pack.
# They never reach the model -- prompts render from `messages` only -- but the
# evaluator refuses to place them in any answer-blind prediction row either.
ORACLE_ADJACENT_FIELDS = frozenset({"answerable", "variant"})


def validate_config(config: dict[str, Any]) -> tuple[Any, ...]:
    """Reject scope drift before any model is loaded."""

    benchmark = config.get("benchmark", {})
    if str(benchmark.get("split")) != "showcase":
        raise ValueError("benchmark.split must be 'showcase'")
    if int(benchmark.get("expected_case_count", 0)) != EXPECTED_CASES:
        raise ValueError(f"benchmark.expected_case_count must be {EXPECTED_CASES}")
    if str(benchmark.get("case_selection")) != "complete_split_no_subsetting":
        raise ValueError("benchmark.case_selection must be complete_split_no_subsetting")

    conditions = _conditions_from_config(config)
    if tuple(c.condition_id for c in conditions) != CONDITION_IDS:
        raise ValueError(f"conditions must be ordered exactly as {list(CONDITION_IDS)}")
    expected = {"B0": ("full_cache", None), "B2": ("significance", 0.0), "B3": ("significance", 1.0)}
    for condition in conditions:
        policy, exponent = expected[condition.condition_id]
        if condition.policy != policy or condition.exponent != exponent:
            raise ValueError(
                f"{condition.condition_id} must use policy={policy!r}, exponent={exponent!r}"
            )
        if not condition.query_conditioned:
            raise ValueError(
                f"{condition.condition_id} must receive the common query probe for access fairness"
            )

    runtime = config.get("runtime", {})
    if str(runtime.get("device", "")).lower() != "cpu":
        raise ValueError("this evaluation is CPU-only")
    if str(runtime.get("dtype", "")).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("this evaluation requires bfloat16")
    if str(runtime.get("decoding", "")) != "manual_greedy":
        raise ValueError("runtime.decoding must be manual_greedy")
    if str(runtime.get("attention_implementation", "")) != "sdpa":
        raise ValueError("runtime.attention_implementation must be sdpa")
    if not bool(runtime.get("deterministic", False)):
        raise ValueError("runtime.deterministic must remain true")
    if int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("runtime.batch_size must be 1")
    if int(runtime.get("max_new_tokens", 0)) <= 0:
        raise ValueError("runtime.max_new_tokens must be positive")

    model = config.get("model", {})
    allowed = tuple(str(v) for v in model.get("allowed_core_ids", []))
    if allowed != (str(model.get("core_id")),):
        raise ValueError("model.allowed_core_ids must contain only model.core_id")
    if ".halo." not in allowed[0].lower():
        raise ValueError("the only permitted published core is Halo")
    if str(model.get("expected_source_base")) != "Qwen3-0.6B":
        raise ValueError("Halo must declare the Qwen3-0.6B source base")
    if bool(model.get("trust_remote_code", True)):
        raise ValueError("trust_remote_code must remain false")

    cache = config.get("cache", {})
    chunk_size = int(cache.get("compression_chunk_tokens", 0))
    history_budget = int(cache.get("retained_history_budget", 0))
    sinks = int(cache.get("sink_tokens", -1))
    recent = int(cache.get("recent_tokens", -1))
    if chunk_size <= 0 or history_budget <= 0:
        raise ValueError("chunk size and retained-history budget must be positive")
    if sinks < 0 or recent < 0 or sinks + recent >= history_budget:
        raise ValueError("the retained-history budget must leave at least one heavy slot")
    pooling = config.get("significance", {}).get("score_pooling", {})
    if bool(pooling.get("enabled", False)) or int(pooling.get("width", 1)) != 1:
        raise ValueError("score pooling must remain disabled for Cornerstone-1 isolation")
    query = config.get("query", {})
    if str(query.get("mode")) != "conditioned":
        raise ValueError("query.mode must be conditioned")
    if str(query.get("rendering")) != "raw_question_no_special_tokens":
        raise ValueError("query.rendering must be raw_question_no_special_tokens")
    if str(query.get("probe_schedule")) != "every_nonfinal_chunk_boundary":
        raise ValueError("query.probe_schedule must be every_nonfinal_chunk_boundary")
    return conditions


def read_showcase_rows(path: Path) -> list[dict[str, Any]]:
    """Load the complete showcase split without subsetting or reordering."""

    rows = _load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise RuntimeError("showcase prompt row has no id")
        if row_id in seen:
            raise RuntimeError(f"duplicate showcase prompt id: {row_id}")
        seen.add(row_id)
        _assert_prompt_only(row)
        if str(row.get("split")) != "showcase":
            raise RuntimeError(f"row {row_id} is not in the showcase split")
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} showcase rows, found {len(rows)}")
    return rows


def _assert_unscored_predictions(rows: list[dict[str, Any]], case_ids: tuple[str, ...]) -> None:
    expected_pairs = [(c, k) for c in case_ids for k in CONDITION_IDS]
    observed_pairs = [(str(r.get("case_id")), str(r.get("condition"))) for r in rows]
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            f"unscored prediction order/coverage mismatch: "
            f"expected {len(expected_pairs)} rows, observed {len(observed_pairs)}"
        )
    forbidden = ORACLE_ADJACENT_FIELDS | {
        "answer",
        "answers",
        "accepted_answers",
        "correct",
        "full_cache_correct",
        "compression_conditioned_eligible",
        "reference_answer",
    }
    for row in rows:
        exposed = forbidden.intersection(row)
        if exposed:
            raise RuntimeError(f"unscored prediction contains answer-derived fields: {sorted(exposed)}")


def read_answer_key(path: Path, case_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        row_id = str(row.get("id", ""))
        if row_id not in case_ids:
            continue
        if row_id in selected:
            raise RuntimeError(f"duplicate answer-key id: {row_id}")
        if not isinstance(row.get("answer"), str):
            raise RuntimeError(f"answer-key row {row_id} has no string answer")
        accepted = row.get("accepted_answers", [row["answer"]])
        if not isinstance(accepted, list) or not accepted or not all(isinstance(v, str) for v in accepted):
            raise RuntimeError(f"answer-key row {row_id} has invalid accepted_answers")
        selected[row_id] = row
    missing = [c for c in case_ids if c not in selected]
    if missing:
        raise RuntimeError(f"missing answer-key ids: {missing}")
    return selected


def score_predictions(
    predictions: list[dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    *,
    case_ids: tuple[str, ...],
    run_id: str,
    seed: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Join answer-derived fields only after blind inference is complete."""

    _assert_unscored_predictions(predictions, case_ids)
    correctness: dict[tuple[str, str], bool] = {}
    for row in predictions:
        case_id = str(row["case_id"])
        accepted = {_normalize_answer(v) for v in answer_key[case_id]["accepted_answers"]}
        correctness[(case_id, str(row["condition"]))] = _normalize_answer(str(row["prediction"])) in accepted

    scored: list[dict[str, Any]] = []
    for source in predictions:
        row = dict(source)
        case_id = str(row["case_id"])
        condition = str(row["condition"])
        key_row = answer_key[case_id]
        correct = correctness[(case_id, condition)]
        full_correct = correctness[(case_id, "B0")]
        eligible = full_correct if condition != "B0" else None
        failure_labels: list[str] = []
        if not full_correct:
            failure_labels.append("FULL_CACHE_CAPABILITY_FAILURE")
        elif not correct:
            failure_labels.append("UNKNOWN")
        row.update(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "scenario_id": key_row.get("scenario_id"),
                "task_family": "R0",
                "generator_seed": seed,
                "prompt_sha256": row["input_token_sha256"],
                "chunk_size": chunk_size,
                "status": "OK",
                "correct": correct,
                "full_cache_correct": full_correct,
                "compression_conditioned_eligible": eligible,
                "full_cache_conditioned_correct": correct if eligible else None,
                "variant": key_row.get("variant"),
                "answerable": key_row.get("answerable"),
                "reference_answer": key_row["answer"],
                "accepted_answers": key_row["accepted_answers"],
                "failure_labels": failure_labels,
                "trace_ids": [],
                "error": None,
            }
        )
        scored.append(row)
    return scored


def _descriptive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in CONDITION_IDS:
        selected = [r for r in rows if r["condition"] == condition]
        correct = sum(bool(r["correct"]) for r in selected)
        entry: dict[str, Any] = {
            "cases": len(selected),
            "correct_cases": correct,
            "exact_accuracy": correct / len(selected) if selected else None,
            "total_inference_wall_seconds": sum(
                float(r["timing"]["inference_wall_seconds"]) for r in selected
            ),
        }
        by_variant: dict[str, Any] = {}
        for variant in sorted({str(r.get("variant")) for r in selected}):
            group = [r for r in selected if str(r.get("variant")) == variant]
            hits = sum(bool(r["correct"]) for r in group)
            by_variant[variant] = {
                "cases": len(group),
                "correct_cases": hits,
                "exact_accuracy": hits / len(group) if group else None,
            }
        entry["by_variant"] = by_variant
        if condition != "B0":
            eligible = [r for r in selected if r["compression_conditioned_eligible"]]
            hits = sum(bool(r["correct"]) for r in eligible)
            entry["full_cache_conditioned_correctness"] = {
                "eligible_cases": len(eligible),
                "correct_cases": hits,
                "accuracy": hits / len(eligible) if eligible else None,
            }
        else:
            entry["full_cache_conditioned_correctness"] = {
                "applicable": False,
                "reason": "B0 is the full-cache capability reference",
            }
        by_condition[condition] = entry

    paired = {}
    for left, right in (("B3", "B2"),):
        wins = sum(
            1
            for case_id in {str(r["case_id"]) for r in rows}
            if _lookup(rows, case_id, left) and not _lookup(rows, case_id, right)
        )
        losses = sum(
            1
            for case_id in {str(r["case_id"]) for r in rows}
            if _lookup(rows, case_id, right) and not _lookup(rows, case_id, left)
        )
        paired[f"{left}_vs_{right}"] = {"wins": wins, "losses": losses, "net": wins - losses}

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "BOUNDED_EXPLORATORY",
        "registration": "NONE_EXPLORATORY_UNREGISTERED",
        "execution_mode": "CLOSED_LOOP_MODEL_EXECUTION",
        "model_scope": "Rineya Halo 1.2.0 devfreeze only (Qwen3-0.6B source base)",
        "benchmark_scope": "complete 64-case Rineya Long Context Demo showcase split",
        "conditions": list(CONDITION_IDS),
        "answer_key_opened_only_after_all_inference": True,
        "manual_greedy_decode": True,
        "descriptive_results": by_condition,
        "paired_answer_counts": paired,
        "inferential_statistics": None,
        "interpretation_limits": [
            "The complete showcase split was evaluated, but this run is exploratory and unregistered.",
            "No endpoint was frozen, no advance gate was declared, and no confidence interval or hypothesis test is reported.",
            "Scenario packs contain four correlated variants; the 64 rows are not 64 independent observations.",
            "B0 is an instrumented full-cache behavioral control; its temporary query probes are rolled back and do not select cache entries.",
            "CPU timing is local-run accounting, not a deployment or GPU latency claim.",
            "Results describe Halo 1.2.0 only and support no published-family claim.",
        ],
    }


def _lookup(rows: list[dict[str, Any]], case_id: str, condition: str) -> bool:
    for row in rows:
        if str(row["case_id"]) == case_id and str(row["condition"]) == condition:
            return bool(row["correct"])
    return False


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config_path = config_path.expanduser().resolve()
    config = _load_mapping(config_path)
    conditions = validate_config(config)
    config["_source_config_path"] = _repo_path(config_path)
    base = _declared_base(config)
    paths = config["paths"]
    model_path = _resolve_path(paths["model"], base=base)
    tokenizer_path = _resolve_path(paths.get("tokenizer", paths["model"]), base=base)
    prompt_path = _resolve_path(paths["prompts"], base=base)
    # Resolve only: the answer key is neither opened nor hashed until blind
    # inference has completed.
    answer_key_path = _resolve_path(paths["answer_key"], base=base)
    output = (
        output_override.expanduser().resolve()
        if output_override
        else _resolve_path(paths["output_root"], base=base)
    )
    _create_output_dir(output)

    outputs = config["outputs"]
    artifact_paths = {name: output / str(filename) for name, filename in outputs.items()}
    _write_resolved_config(artifact_paths["resolved_config"], config)

    core_path = model_path / "CORE.json"
    if tokenizer_path != model_path:
        raise RuntimeError("this evaluation requires the tokenizer packaged with the exact core")
    if not core_path.is_file():
        raise FileNotFoundError(f"published core metadata is missing: {core_path}")
    core = json.loads(core_path.read_text(encoding="utf-8"))
    _validate_halo_core(core, config["model"])
    checkpoint = _checkpoint_hashes(model_path, config["model"])

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
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
    prepared_cases = [
        _prepare_case(tokenizer, row, config["prompt"], config["query"]) for row in prompt_rows
    ]
    max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    max_new_tokens = int(config["runtime"]["max_new_tokens"])
    for prepared in prepared_cases:
        if max_positions and prepared.input_ids.shape[-1] + max_new_tokens > max_positions:
            raise RuntimeError(f"{prepared.case_id} exceeds model context limit; truncation is refused")

    eos = tokenizer.eos_token_id
    eos_token_ids = {int(v) for v in eos} if isinstance(eos, list) else ({int(eos)} if eos is not None else set())
    run_id = output.name
    partial_path = output / "predictions.partial.jsonl"

    unscored: list[dict[str, Any]] = []
    total = len(prepared_cases) * len(conditions)
    done = 0
    with partial_path.open("w", encoding="utf-8", newline="\n") as partial:
        for prepared in prepared_cases:
            invariant = (
                prepared.input_token_sha256,
                prepared.context_token_sha256,
                prepared.query_token_sha256,
            )
            for condition in conditions:
                result = execute_condition(
                    model,
                    prepared,
                    condition,
                    cache_config=config["cache"],
                    runtime_config=config["runtime"],
                    eos_token_ids=eos_token_ids,
                )
                if invariant != (
                    prepared.input_token_sha256,
                    prepared.context_token_sha256,
                    prepared.query_token_sha256,
                ):
                    raise RuntimeError("shared token tensors changed between conditions")
                prediction = tokenizer.decode(
                    result.pop("generated_token_ids"), skip_special_tokens=True
                ).strip()
                row = {
                    "case_id": prepared.case_id,
                    "scenario_id": prepared.scenario_id,
                    "condition": condition.condition_id,
                    "prediction": prediction,
                    "prompt_text_sha256": prepared.prompt_text_sha256,
                    "input_token_sha256": prepared.input_token_sha256,
                    "context_token_sha256": prepared.context_token_sha256,
                    "query_token_sha256": prepared.query_token_sha256,
                    "input_tokens": int(prepared.input_ids.shape[-1]),
                    "context_tokens": int(prepared.context_ids.shape[-1]),
                    "query_tokens": int(prepared.query_ids.shape[-1]),
                    "benchmark_metadata": prepared.metadata,
                    **result,
                }
                unscored.append(row)
                partial.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
                done += 1
                if done % 12 == 0 or done == total:
                    print(f"  {done}/{total} case-conditions complete", flush=True)
                gc.collect()

    _assert_unscored_predictions(unscored, case_ids)
    _atomic_write_jsonl(artifact_paths["unscored_predictions"], unscored)

    # Emit the demonstration suite's own prediction format, one file per
    # condition, so its scorer can compare these against the recorded
    # showcase baselines. Written before the key is opened.
    condition_files: dict[str, str] = {}
    for condition_id in CONDITION_IDS:
        name = f"showcase64_halo_{condition_id}.jsonl"
        _atomic_write_jsonl(
            output / name,
            [
                {"id": r["case_id"], "prediction": r["prediction"]}
                for r in unscored
                if r["condition"] == condition_id
            ],
        )
        condition_files[condition_id] = name

    # Deliberately the first answer-key access in this run.
    answer_key = read_answer_key(answer_key_path, case_ids)
    scored = score_predictions(
        unscored,
        answer_key,
        case_ids=case_ids,
        run_id=run_id,
        seed=seed,
        chunk_size=int(config["cache"]["compression_chunk_tokens"]),
    )
    case_schema = LAB_ROOT / "schemas" / "case.schema.json"
    for row in scored:
        _schema_validate(row, case_schema)
    _atomic_write_jsonl(artifact_paths["cases"], scored)

    summary = _descriptive_summary(scored)
    _atomic_write_json(artifact_paths["summary"], summary)

    artifacts = {name: path.name for name, path in artifact_paths.items()}
    artifacts.update({f"predictions_{k}": v for k, v in condition_files.items()})
    artifacts["predictions_partial"] = partial_path.name

    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id=run_id,
        execution_mode="closed_loop_model_execution",
        architecture_variant="B0 full cache / B2 e=0 / B3 e=1",
        seed=seed,
        compute_tier="B",
        status="COMPLETE",
        evidence_class="NEW_CHATBUBBLE_EVIDENCE",
        resolved_config_path=artifacts["resolved_config"],
        benchmark={"name": str(config["benchmark"]["name"])},
        model={
            "status": "VERIFIED",
            "id": str(core["core_id"]),
            "path": _repo_path(model_path),
            "architecture": str(config["model"]["expected_architecture"]),
            "hashes": {
                target: checkpoint["files"][source]["sha256"] if source in checkpoint["files"] else None
                for target, source in (
                    ("weights", "weights"),
                    ("tokenizer", "tokenizer"),
                    ("config", "config"),
                    ("chat_template", "chat_template"),
                )
            },
            "checkpoint_files": checkpoint["files"],
            "checkpoint_aggregate_sha256": checkpoint["aggregate_sha256"],
        },
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["config"] = {
        "resolved_path": artifacts["resolved_config"],
        "sha256": sha256_file(artifact_paths["resolved_config"]),
        "source_path": str(config["_source_config_path"]).replace("\\", "/"),
    }
    manifest["runtime"].update(
        {
            "device": "cpu",
            "precision": "bfloat16",
            "manual_greedy": True,
            "attention_implementation": "sdpa",
        }
    )
    manifest["budget"] = {
        "resident_entries": None,
        "kv_only_bytes": None,
        "total_persistent_bytes": None,
        "matched": False,
        "comparison_note": "B2 and B3 share the same resident-entry budget; B0 is full cache.",
        "by_condition": {
            condition_id: {
                "resident_entries_after_prefill": sorted(
                    {r["accounting"]["resident_entries_after_prefill"] for r in scored if r["condition"] == condition_id}
                ),
                "compression_boundaries": sorted(
                    {r["accounting"]["compression_boundaries"] for r in scored if r["condition"] == condition_id}
                ),
            }
            for condition_id in CONDITION_IDS
        },
    }
    manifest["benchmark"] = {
        "split": "showcase",
        "case_count": len(case_ids),
        "prompt_sha256": sha256_file(prompt_path),
        "answer_key_sha256_after_inference": sha256_file(answer_key_path),
        "answer_key_join_order": "opened_after_unscored_predictions_were_written",
    }
    manifest["artifacts"] = artifacts
    manifest["notes"] = [
        "Complete 64-case showcase split: 16 scenario packs x 4 variants.",
        "Exploratory and unregistered; no frozen endpoint, advance gate, or inferential statistics.",
        "Scenario-pack variants are correlated; 64 rows are not 64 independent observations.",
        "All conditions share identical prompt tokens, chunking, query tokens, probe boundaries, and manual greedy decoding.",
        "B0 query probes are instrumentation-only because no entries are evicted.",
        "Halo 1.2.0 only; no Halo Uncensored, Lyra, Nimbus, Pulse, R2, or Aegis result is implied.",
        "Compression semantics are imported unchanged from run_published_core_eval.execute_condition.",
    ]
    _schema_validate(manifest, LAB_ROOT / "schemas" / "manifest.schema.json")
    _atomic_write_json(artifact_paths["manifest"], manifest)

    print(json.dumps({"output": str(output), "status": summary["status"]}, indent=2))
    return {"output": str(output), "manifest": manifest, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_showcase64_eval.yaml")
    parser.add_argument("--output", type=Path, help="Fresh output directory; existing paths are rejected.")
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
