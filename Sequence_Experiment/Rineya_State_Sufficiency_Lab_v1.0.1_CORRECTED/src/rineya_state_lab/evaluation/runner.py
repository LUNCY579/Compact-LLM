from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..config import load_config, require
from ..data.layout import DatasetLayout, discover_dataset_root
from ..inference.base import GenerationRequest, GenerationResponse
from ..inference.factory import create_backend
from ..utils import (
    append_jsonl,
    atomic_write_json,
    environment_manifest,
    load_completed_ids,
    normalize_casefold,
    seed_everything,
    sha256_file,
    sha256_text,
    stable_int_hash,
    utc_now_iso,
)
from .parser import parse_response, parse_stateful_response
from .protocols import (
    EvaluationExample,
    build_ablation_examples,
    build_direct_examples,
    build_oracle_transition_examples,
    deterministic_limit,
    group_transitions_by_problem,
    load_full_records,
    load_transition_records,
    stateful_prompt,
)
from .scorers import normalized_exact, score_answer, state_equivalence, token_f1


def _score_direct(example: EvaluationExample, response: GenerationResponse, scoring_cfg: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_response(response.text)
    result = score_answer(
        example.expected["answer"],
        parsed.strict_final,
        parsed.lenient_final,
        example.expected["answer_type"],
        abs_tol=float(scoring_cfg.get("absolute_tolerance", 1e-6)),
        rel_tol=float(scoring_cfg.get("relative_tolerance", 1e-6)),
    )
    return {
        **result.to_dict(),
        "parse_status": parsed.status,
        "parse": parsed.to_dict(),
        "backend_error": response.error,
        "correct_strict": bool(result.correct_strict and response.error is None),
        "correct_lenient": bool(result.correct_lenient and response.error is None),
    }


def _score_transition(example: EvaluationExample, response: GenerationResponse, scoring_cfg: dict[str, Any]) -> dict[str, Any]:
    target_style = str(example.metadata.get("target_style", "stateful"))
    if target_style != "stateful":
        raw = response.text.strip()
        exact = normalized_exact(example.expected["target_step"], raw)
        f1 = token_f1(example.expected["target_step"], raw)
        return {
            "correct_strict": bool(exact and response.error is None),
            "correct_lenient": bool((exact or f1 >= float(scoring_cfg.get("step_f1_threshold", 0.85))) and response.error is None),
            "step_exact": exact,
            "step_token_f1": f1,
            "parse_status": "RAW_NEXT_STEP",
            "backend_error": response.error,
        }

    parsed = parse_stateful_response(response.text)
    structured = parsed.structured or {}
    expected_done = bool(example.expected["done"])
    predicted_done = structured.get("done")
    done_match = predicted_done is expected_done
    state_metrics = state_equivalence(
        example.expected["next_state"],
        structured.get("updated_compact_state"),
        f1_threshold=float(scoring_cfg.get("state_f1_threshold", 0.90)),
    )
    step_metrics = state_equivalence(
        example.expected["target_step"],
        structured.get("next_reasoning_step"),
        f1_threshold=float(scoring_cfg.get("step_f1_threshold", 0.85)),
    )
    if expected_done:
        final_result = score_answer(
            example.expected["final_answer"],
            structured.get("final_answer") if parsed.json_object is not None else parsed.strict_final,
            structured.get("final_answer") or parsed.lenient_final,
            example.expected["answer_type"],
            abs_tol=float(scoring_cfg.get("absolute_tolerance", 1e-6)),
            rel_tol=float(scoring_cfg.get("relative_tolerance", 1e-6)),
        ).to_dict()
    else:
        final_result = {
            "correct_strict": structured.get("final_answer") is None,
            "correct_lenient": True,
            "scorer": "not_applicable",
        }
    required_keys = {"next_reasoning_step", "updated_compact_state", "done", "final_answer"}
    contract_valid = parsed.json_object is not None and required_keys.issubset(parsed.json_object) and predicted_done is not None
    strict = (
        response.error is None
        and contract_valid
        and done_match
        and state_metrics["exact"]
        and step_metrics["exact"]
        and bool(final_result["correct_strict"])
    )
    lenient = (
        response.error is None
        and done_match
        and state_metrics["lenient_equivalent"]
        and step_metrics["lenient_equivalent"]
        and bool(final_result["correct_lenient"])
    )
    return {
        "correct_strict": strict,
        "correct_lenient": lenient,
        "contract_valid": contract_valid,
        "done_match": done_match,
        "predicted_done": predicted_done,
        "state": state_metrics,
        "step": step_metrics,
        "final": final_result,
        "parse_status": parsed.status,
        "parse": parsed.to_dict(),
        "backend_error": response.error,
    }


def _scored_record(example: EvaluationExample, response: GenerationResponse, scoring_cfg: dict[str, Any]) -> dict[str, Any]:
    if example.transition_id is None:
        score = _score_direct(example, response, scoring_cfg)
    else:
        score = _score_transition(example, response, scoring_cfg)
    return {
        "schema_version": "1.0",
        "recorded_at": utc_now_iso(),
        "request_id": example.request.request_id,
        "protocol": example.protocol,
        "problem_id": example.problem_id,
        "transition_id": example.transition_id,
        "category": example.category,
        "domain": example.domain,
        "difficulty": example.difficulty,
        "metadata": example.metadata,
        "expected": example.expected,
        "raw_output": response.text,
        "response": response.to_dict(),
        "score": score,
    }


def _evaluate_examples(
    examples: list[EvaluationExample],
    *,
    backend: Any,
    raw_path: Path,
    scored_path: Path,
    completed: set[str],
    scoring_cfg: dict[str, Any],
    generation_batch_size: int,
    progress_path: Path,
) -> int:
    pending = [example for example in examples if example.request.request_id not in completed]
    processed = 0
    for start in range(0, len(pending), generation_batch_size):
        batch = pending[start : start + generation_batch_size]
        responses = backend.generate([example.request for example in batch])
        by_id = {response.request_id: response for response in responses}
        for example in batch:
            response = by_id.get(example.request.request_id)
            if response is None:
                response = GenerationResponse(
                    request_id=example.request.request_id,
                    text="",
                    latency_ms=0.0,
                    backend=getattr(backend, "name", "unknown"),
                    error="Backend returned no response for request",
                )
            append_jsonl(
                raw_path,
                {
                    "schema_version": "1.0",
                    "request_id": response.request_id,
                    "protocol": example.protocol,
                    "problem_id": example.problem_id,
                    "transition_id": example.transition_id,
                    "messages": example.request.messages,
                    "reference_output": example.request.reference_output,
                    "response": response.to_dict(),
                },
            )
            append_jsonl(scored_path, _scored_record(example, response, scoring_cfg))
            completed.add(example.request.request_id)
            processed += 1
        atomic_write_json(
            progress_path,
            {
                "updated_at": utc_now_iso(),
                "protocol": batch[-1].protocol if batch else None,
                "completed_total": len(completed),
                "processed_this_run": processed,
                "remaining_in_current_protocol": max(0, len(pending) - processed),
            },
        )
    return processed


def _run_rollout_protocol(
    *,
    layout: DatasetLayout,
    split: str,
    protocol_cfg: dict[str, Any],
    backend: Any,
    raw_path: Path,
    scored_path: Path,
    completed: set[str],
    scoring_cfg: dict[str, Any],
    seed: int,
    progress_path: Path,
) -> int:
    protocol_name = str(protocol_cfg.get("name", "generated_rollout"))
    full_records = deterministic_limit(
        load_full_records(layout, split),
        protocol_cfg.get("limit"),
        seed,
        "problem_id",
    )
    chains = group_transitions_by_problem(load_transition_records(layout, split))
    max_steps_override = protocol_cfg.get("max_steps")
    max_state_chars = int(protocol_cfg.get("max_state_chars", 4096))
    allow_implicit_final = bool(protocol_cfg.get("allow_implicit_final", True))
    processed = 0

    for full in full_records:
        pid = str(full["problem_id"])
        aggregate_id = f"{protocol_name}::{pid}"
        if aggregate_id in completed:
            continue
        chain = chains[pid]
        current_state = str(chain[0]["current_state"])
        max_steps = int(max_steps_override or (len(chain) + 3))
        trajectory: list[dict[str, Any]] = []
        seen_states = {normalize_casefold(current_state)}
        seen_outputs: set[str] = set()
        final_candidate_strict: Any = None
        final_candidate_lenient: Any = None
        termination_reason = "MAX_STEPS"
        all_contract_valid = True
        first_exact_divergence: int | None = None
        first_lenient_divergence: int | None = None

        for turn in range(1, max_steps + 1):
            canonical = chain[turn - 1] if turn <= len(chain) else chain[-1]
            request_id = f"{aggregate_id}::turn_{turn:02d}"
            request = GenerationRequest(
                request_id=request_id,
                messages=stateful_prompt(str(full["problem"]), current_state),
                reference_output=str(canonical.get("stateful_rollout_target")),
                metadata={"problem_id": pid, "turn": turn},
            )
            response = backend.generate([request])[0]
            parsed = parse_stateful_response(response.text)
            structured = parsed.structured or {}
            contract_valid = (
                parsed.json_object is not None
                and all(key in parsed.json_object for key in ["next_reasoning_step", "updated_compact_state", "done", "final_answer"])
                and structured.get("done") is not None
            )
            all_contract_valid = all_contract_valid and contract_valid
            predicted_state = structured.get("updated_compact_state")
            state_metrics = state_equivalence(
                canonical["next_state"],
                predicted_state,
                f1_threshold=float(scoring_cfg.get("state_f1_threshold", 0.90)),
            )
            if first_exact_divergence is None and not state_metrics["exact"]:
                first_exact_divergence = turn
            if first_lenient_divergence is None and not state_metrics["lenient_equivalent"]:
                first_lenient_divergence = turn
            turn_record = {
                "turn": turn,
                "request_id": request_id,
                "input_state": current_state,
                "canonical_step_index": int(canonical["step_index"]),
                "canonical_next_state": canonical["next_state"],
                "raw_output": response.text,
                "response": response.to_dict(),
                "parse": parsed.to_dict(),
                "state_comparison": state_metrics,
                "contract_valid": contract_valid,
            }
            trajectory.append(turn_record)
            append_jsonl(
                raw_path,
                {
                    "schema_version": "1.0",
                    "request_id": request_id,
                    "aggregate_request_id": aggregate_id,
                    "protocol": protocol_name,
                    "problem_id": pid,
                    "messages": request.messages,
                    "reference_output": request.reference_output,
                    "response": response.to_dict(),
                },
            )
            if response.error is not None:
                termination_reason = "BACKEND_ERROR"
                break
            normalized_output = normalize_casefold(response.text)
            if normalized_output in seen_outputs:
                termination_reason = "REPEATED_OUTPUT_LOOP"
                break
            seen_outputs.add(normalized_output)

            if structured.get("final_answer") is not None:
                final_candidate_lenient = structured["final_answer"]
                if parsed.json_object is not None:
                    final_candidate_strict = structured["final_answer"]
            elif parsed.json_object is None and parsed.lenient_final is not None:
                final_candidate_lenient = parsed.lenient_final
                final_candidate_strict = parsed.strict_final

            if structured.get("done") is True:
                termination_reason = "COMPLETED" if final_candidate_lenient is not None else "DONE_WITHOUT_FINAL"
                break
            if final_candidate_lenient is not None and allow_implicit_final:
                termination_reason = "IMPLICIT_FINAL"
                break
            if predicted_state is None or not str(predicted_state).strip():
                termination_reason = "MISSING_UPDATED_STATE"
                break
            if len(str(predicted_state)) > max_state_chars:
                termination_reason = "STATE_TOO_LARGE"
                break
            normalized_state = normalize_casefold(predicted_state)
            if normalized_state in seen_states:
                termination_reason = "REPEATED_STATE_LOOP"
                break
            seen_states.add(normalized_state)
            current_state = str(predicted_state)

        final_score = score_answer(
            full["answer"],
            final_candidate_strict,
            final_candidate_lenient,
            full["answer_type"],
            abs_tol=float(scoring_cfg.get("absolute_tolerance", 1e-6)),
            rel_tol=float(scoring_cfg.get("relative_tolerance", 1e-6)),
        ).to_dict()
        strict_correct = (
            termination_reason == "COMPLETED"
            and all_contract_valid
            and bool(final_score["correct_strict"])
        )
        lenient_correct = (
            termination_reason in {"COMPLETED", "IMPLICIT_FINAL"}
            and bool(final_score["correct_lenient"])
        )
        scored = {
            "schema_version": "1.0",
            "recorded_at": utc_now_iso(),
            "request_id": aggregate_id,
            "protocol": protocol_name,
            "problem_id": pid,
            "transition_id": None,
            "category": full["category"],
            "domain": full["domain"],
            "difficulty": int(full["difficulty"]),
            "metadata": {"step_count": len(chain), "turns_used": len(trajectory)},
            "expected": {"answer": full["answer"], "answer_type": full["answer_type"]},
            "raw_output": trajectory[-1]["raw_output"] if trajectory else "",
            "response": trajectory[-1]["response"] if trajectory else None,
            "score": {
                "correct_strict": strict_correct,
                "correct_lenient": lenient_correct,
                "final": final_score,
                "termination_reason": termination_reason,
                "all_turns_contract_valid": all_contract_valid,
                "first_exact_state_divergence": first_exact_divergence,
                "first_lenient_state_divergence": first_lenient_divergence,
                "turns_used": len(trajectory),
            },
            "trajectory": trajectory,
        }
        append_jsonl(scored_path, scored)
        completed.add(aggregate_id)
        processed += 1
        atomic_write_json(
            progress_path,
            {
                "updated_at": utc_now_iso(),
                "protocol": protocol_name,
                "completed_total": len(completed),
                "processed_this_run": processed,
                "last_problem_id": pid,
            },
        )
    return processed


def run_evaluation(config_path: str | Path, overrides: list[str] | None = None, *, force: bool = False) -> dict[str, Any]:
    config = load_config(config_path, overrides)
    run_cfg = require(config, "run")
    data_cfg = require(config, "data")
    backend_cfg = require(config, "backend")
    protocols_cfg = require(config, "protocols")
    scoring_cfg = config.get("scoring", {}) or {}
    seed = int(run_cfg.get("seed", 42))
    seed_everything(seed)
    split = str(data_cfg.get("split", "dev"))
    if split == "test" and bool(data_cfg.get("require_test_unlock", True)):
        expected_unlock = str(data_cfg.get("test_unlock_value", "I_UNDERSTAND_TEST_IS_SEALED"))
        if os.environ.get("RINEYA_ALLOW_TEST") != expected_unlock:
            raise PermissionError(
                "TEST is sealed. Set RINEYA_ALLOW_TEST=I_UNDERSTAND_TEST_IS_SEALED only after the DEV protocol is frozen."
            )

    dataset_root = discover_dataset_root(data_cfg["root"])
    layout = DatasetLayout(dataset_root)
    output_dir = Path(str(run_cfg["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_predictions.jsonl"
    scored_path = output_dir / "scored_predictions.jsonl"
    progress_path = output_dir / "LIVE_PROGRESS.json"
    manifest_path = output_dir / "run_manifest.json"
    resolved_text = __import__("yaml").safe_dump(config, sort_keys=False, allow_unicode=True)
    config_hash = sha256_text(resolved_text)
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") not in {None, config_hash}:
            raise RuntimeError(
                f"Output directory {output_dir} contains results from a different config. Use a new directory or --force."
            )
    if force:
        for path in [raw_path, scored_path, progress_path, output_dir / "metrics.json", output_dir / "report.md", output_dir / "report.html"]:
            if path.exists():
                path.unlink()
    completed = load_completed_ids(scored_path)
    manifest = {
        "schema_version": "1.0",
        "run_name": run_cfg["name"],
        "started_at": utc_now_iso(),
        "status": "RUNNING",
        "config_sha256": config_hash,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "split": split,
        "dataset_hashes": {
            "full": sha256_file(layout.full_file(split)),
            "transitions": sha256_file(layout.transition_file(split)),
        },
        "environment": environment_manifest(),
        "completed_at_start": len(completed),
    }
    atomic_write_json(manifest_path, manifest)
    (output_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")

    backend = create_backend(backend_cfg)
    processed = 0
    try:
        for protocol_cfg in protocols_cfg:
            protocol_type = str(protocol_cfg["type"]).casefold()
            protocol_name = str(protocol_cfg.get("name", protocol_type))
            if protocol_type in {"direct", "direct_solve"}:
                examples = build_direct_examples(
                    layout,
                    split=split,
                    protocol_name=protocol_name,
                    limit=protocol_cfg.get("limit"),
                    seed=seed,
                    instruction=protocol_cfg.get("instruction"),
                )
                processed += _evaluate_examples(
                    examples,
                    backend=backend,
                    raw_path=raw_path,
                    scored_path=scored_path,
                    completed=completed,
                    scoring_cfg=scoring_cfg,
                    generation_batch_size=int(protocol_cfg.get("generation_batch_size", 1)),
                    progress_path=progress_path,
                )
            elif protocol_type in {"oracle_transition", "transition"}:
                examples = build_oracle_transition_examples(
                    layout,
                    split=split,
                    protocol_name=protocol_name,
                    limit=protocol_cfg.get("limit"),
                    seed=seed,
                    target_style=str(protocol_cfg.get("target_style", "stateful")),
                    one_per_problem=bool(protocol_cfg.get("one_per_problem", False)),
                )
                processed += _evaluate_examples(
                    examples,
                    backend=backend,
                    raw_path=raw_path,
                    scored_path=scored_path,
                    completed=completed,
                    scoring_cfg=scoring_cfg,
                    generation_batch_size=int(protocol_cfg.get("generation_batch_size", 1)),
                    progress_path=progress_path,
                )
            elif protocol_type in {"state_ablation", "ablation"}:
                examples = build_ablation_examples(
                    layout,
                    split=split,
                    protocol_name=protocol_name,
                    variants=list(protocol_cfg.get("variants", ["compact_state", "full_history", "last_step_only", "no_state", "shuffled_state", "distractor_state"])),
                    limit_problems=protocol_cfg.get("limit_problems"),
                    seed=seed,
                )
                processed += _evaluate_examples(
                    examples,
                    backend=backend,
                    raw_path=raw_path,
                    scored_path=scored_path,
                    completed=completed,
                    scoring_cfg=scoring_cfg,
                    generation_batch_size=int(protocol_cfg.get("generation_batch_size", 1)),
                    progress_path=progress_path,
                )
            elif protocol_type in {"generated_rollout", "rollout"}:
                processed += _run_rollout_protocol(
                    layout=layout,
                    split=split,
                    protocol_cfg=protocol_cfg,
                    backend=backend,
                    raw_path=raw_path,
                    scored_path=scored_path,
                    completed=completed,
                    scoring_cfg=scoring_cfg,
                    seed=seed,
                    progress_path=progress_path,
                )
            else:
                raise ValueError(f"Unsupported evaluation protocol type: {protocol_type}")

        from .reporting import build_evaluation_reports

        metrics = build_evaluation_reports(scored_path, output_dir)
        manifest.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now_iso(),
                "processed_this_run": processed,
                "completed_total": len(completed),
                "metrics_file": str(output_dir / "metrics.json"),
            }
        )
        atomic_write_json(manifest_path, manifest)
        return {"manifest": manifest, "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "FAILED", "failed_at": utc_now_iso(), "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        backend.close()
