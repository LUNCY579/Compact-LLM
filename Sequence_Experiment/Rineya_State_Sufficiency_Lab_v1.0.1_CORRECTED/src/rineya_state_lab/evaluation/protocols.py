from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..data.layout import DatasetLayout
from ..inference.base import GenerationRequest
from ..utils import iter_jsonl, normalize_casefold, stable_int_hash


@dataclass
class EvaluationExample:
    request: GenerationRequest
    protocol: str
    problem_id: str
    transition_id: str | None
    category: str
    domain: str
    difficulty: int
    expected: dict[str, Any]
    metadata: dict[str, Any]


def stateful_prompt(problem: str, current_state: str, *, extra_context: str | None = None) -> list[dict[str, str]]:
    content = (
        f"Original problem:\n{problem}\n\n"
        f"Current compact state:\n{current_state}\n\n"
    )
    if extra_context:
        content += f"Additional context:\n{extra_context}\n\n"
    content += (
        "Give exactly the next correct reasoning step. Do not replay omitted reasoning history.\n"
        "Return a JSON object with next_reasoning_step, updated_compact_state, done, and final_answer."
    )
    return [{"role": "user", "content": content}]


def next_step_prompt(problem: str, current_state: str, *, extra_context: str | None = None) -> list[dict[str, str]]:
    content = f"Original problem:\n{problem}\n\nCurrent compact state:\n{current_state}\n\n"
    if extra_context:
        content += f"Additional context:\n{extra_context}\n\n"
    content += "Give exactly the next correct reasoning step. Do not replay omitted reasoning history."
    return [{"role": "user", "content": content}]


def direct_prompt(record: dict[str, Any], instruction: str | None = None) -> list[dict[str, str]]:
    content = str(record["problem"])
    if instruction:
        content = f"{content}\n\n{instruction}"
    return [{"role": "user", "content": content}]


def deterministic_limit(records: list[dict[str, Any]], limit: int | None, seed: int, key: str) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    return sorted(records, key=lambda row: stable_int_hash(f"{seed}:{row[key]}"))[: int(limit)]


def load_full_records(layout: DatasetLayout, split: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(layout.full_file(split)))


def load_transition_records(layout: DatasetLayout, split: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(layout.transition_file(split)))


def build_direct_examples(
    layout: DatasetLayout,
    *,
    split: str,
    protocol_name: str,
    limit: int | None,
    seed: int,
    instruction: str | None = None,
) -> list[EvaluationExample]:
    records = deterministic_limit(load_full_records(layout, split), limit, seed, "problem_id")
    examples: list[EvaluationExample] = []
    for record in records:
        request_id = f"{protocol_name}::{record['problem_id']}"
        examples.append(
            EvaluationExample(
                request=GenerationRequest(
                    request_id=request_id,
                    messages=direct_prompt(record, instruction),
                    reference_output=str(record["full_response"]),
                    metadata={"problem_id": record["problem_id"]},
                ),
                protocol=protocol_name,
                problem_id=str(record["problem_id"]),
                transition_id=None,
                category=str(record["category"]),
                domain=str(record["domain"]),
                difficulty=int(record["difficulty"]),
                expected={
                    "answer": record["answer"],
                    "answer_type": record["answer_type"],
                    "full_response": record["full_response"],
                },
                metadata={"step_count": int(record["step_count"]), "generator": record.get("generator")},
            )
        )
    return examples


def build_oracle_transition_examples(
    layout: DatasetLayout,
    *,
    split: str,
    protocol_name: str,
    limit: int | None,
    seed: int,
    target_style: str = "stateful",
    one_per_problem: bool = False,
) -> list[EvaluationExample]:
    rows = load_transition_records(layout, split)
    if one_per_problem:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["problem_id"]), []).append(row)
        chosen: list[dict[str, Any]] = []
        for pid, options in groups.items():
            ordered = sorted(options, key=lambda item: int(item["step_index"]))
            selected = stable_int_hash(f"{seed}:{pid}") % len(ordered)
            chosen.append(ordered[selected])
        rows = chosen
    rows = deterministic_limit(rows, limit, seed, "transition_id")
    examples: list[EvaluationExample] = []
    for row in rows:
        stateful = target_style == "stateful"
        messages = stateful_prompt(row["problem"], row["current_state"]) if stateful else next_step_prompt(
            row["problem"], row["current_state"]
        )
        reference = row["stateful_rollout_target"] if stateful else row["target_step"]
        request_id = f"{protocol_name}::{row['transition_id']}"
        examples.append(
            EvaluationExample(
                request=GenerationRequest(
                    request_id=request_id,
                    messages=messages,
                    reference_output=str(reference),
                    metadata={"problem_id": row["problem_id"], "transition_id": row["transition_id"]},
                ),
                protocol=protocol_name,
                problem_id=str(row["problem_id"]),
                transition_id=str(row["transition_id"]),
                category=str(row["category"]),
                domain=str(row["domain"]),
                difficulty=int(row["difficulty"]),
                expected={
                    "answer_type": row["answer_type"],
                    "target_step": row["target_step"],
                    "next_state": row["next_state"],
                    "done": bool(row["done"]),
                    "final_answer": row.get("final_answer"),
                    "stateful_target": row.get("stateful_rollout_target"),
                },
                metadata={
                    "step_index": int(row["step_index"]),
                    "total_steps": int(row["total_steps"]),
                    "target_style": target_style,
                },
            )
        )
    return examples


def build_ablation_examples(
    layout: DatasetLayout,
    *,
    split: str,
    protocol_name: str,
    variants: list[str],
    limit_problems: int | None,
    seed: int,
) -> list[EvaluationExample]:
    full_by_id = {str(row["problem_id"]): row for row in load_full_records(layout, split)}
    rows = load_transition_records(layout, split)
    problem_ids = sorted({str(row["problem_id"]) for row in rows}, key=lambda pid: stable_int_hash(f"{seed}:{pid}"))
    if limit_problems is not None:
        problem_ids = problem_ids[: int(limit_problems)]
    allowed = set(problem_ids)
    rows = [row for row in rows if str(row["problem_id"]) in allowed]
    rows.sort(key=lambda row: (str(row["problem_id"]), int(row["step_index"])))
    alternative_states = [str(row["current_state"]) for row in rows]
    examples: list[EvaluationExample] = []
    for row_index, row in enumerate(rows):
        source = full_by_id[str(row["problem_id"])]
        step_index = int(row["step_index"])
        previous_steps = source["reasoning_steps"][: step_index - 1]
        for variant in variants:
            current_state = str(row["current_state"])
            extra_context: str | None = None
            if variant == "compact_state":
                pass
            elif variant == "full_history":
                current_state = "The complete prior reasoning history is supplied below."
                extra_context = "Prior reasoning:\n" + ("\n".join(previous_steps) if previous_steps else "No prior steps.")
            elif variant == "last_step_only":
                current_state = previous_steps[-1] if previous_steps else "No derived facts yet."
            elif variant == "no_state":
                current_state = f"No intermediate state supplied. The target is transition {step_index} of {row['total_steps']}."
            elif variant == "shuffled_state":
                candidate = alternative_states[(row_index + 7) % len(alternative_states)]
                if normalize_casefold(candidate) == normalize_casefold(row["current_state"]):
                    candidate = alternative_states[(row_index + 13) % len(alternative_states)]
                current_state = candidate
            elif variant == "distractor_state":
                distractor = alternative_states[(row_index + 11) % len(alternative_states)]
                current_state = f"{row['current_state']}\nIrrelevant note from another task: {distractor}"
            else:
                raise ValueError(f"Unknown state ablation variant: {variant}")
            request_id = f"{protocol_name}::{variant}::{row['transition_id']}"
            examples.append(
                EvaluationExample(
                    request=GenerationRequest(
                        request_id=request_id,
                        messages=stateful_prompt(row["problem"], current_state, extra_context=extra_context),
                        reference_output=str(row["stateful_rollout_target"]),
                    ),
                    protocol=protocol_name,
                    problem_id=str(row["problem_id"]),
                    transition_id=str(row["transition_id"]),
                    category=str(row["category"]),
                    domain=str(row["domain"]),
                    difficulty=int(row["difficulty"]),
                    expected={
                        "answer_type": row["answer_type"],
                        "target_step": row["target_step"],
                        "next_state": row["next_state"],
                        "done": bool(row["done"]),
                        "final_answer": row.get("final_answer"),
                    },
                    metadata={
                        "variant": variant,
                        "step_index": step_index,
                        "total_steps": int(row["total_steps"]),
                    },
                )
            )
    return examples


def group_transitions_by_problem(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["problem_id"]), []).append(row)
    for pid in groups:
        groups[pid].sort(key=lambda item: int(item["step_index"]))
    return groups
