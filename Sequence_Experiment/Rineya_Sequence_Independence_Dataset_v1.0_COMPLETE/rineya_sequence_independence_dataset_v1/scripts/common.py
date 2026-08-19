from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"
DATASET_NAME = "Rineya Sequence-Independence Reasoning Dataset"
DATASET_VERSION = "1.0"
BASE_SEED = 20260818


@dataclass(frozen=True)
class ReasoningStep:
    state_before: str
    target_step: str
    state_after: str
    done: bool = False


@dataclass(frozen=True)
class GeneratedProblem:
    problem: str
    answer: str
    answer_type: str
    steps: tuple[ReasoningStep, ...]
    difficulty: int
    generator: str
    tags: tuple[str, ...] = ()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_key(text: str, salt: str = "") -> str:
    return sha256_text(f"{salt}|{text}")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json_dumps(row))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fmt_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def fmt_decimal(value: Decimal, places: int = 2, trim: bool = True) -> str:
    quant = Decimal(1).scaleb(-places)
    s = str(value.quantize(quant, rounding=ROUND_HALF_UP))
    if trim and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_money_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"


def fmt_time_24(total_minutes: int) -> str:
    day_offset, minute_of_day = divmod(total_minutes, 24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    base = f"{hour:02d}:{minute:02d}"
    if day_offset == 0:
        return base
    if day_offset == 1:
        return f"{base} the next day"
    return f"{base} ({day_offset} days later)"


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def make_steps(
    actions: list[str],
    states_after: list[str],
    answer: str,
    *,
    initial_state: str = "No derived facts yet.",
    final_template: str = "Therefore, the answer is {answer}.",
) -> tuple[ReasoningStep, ...]:
    if len(actions) != len(states_after):
        raise ValueError("actions and states_after must have equal length")
    result: list[ReasoningStep] = []
    state = initial_state
    for action, next_state in zip(actions, states_after):
        result.append(ReasoningStep(state, action, next_state, False))
        state = next_state
    final_step = final_template.format(answer=answer)
    result.append(ReasoningStep(state, final_step, f"Final answer: {answer}", True))
    return tuple(result)


def full_response(steps: tuple[ReasoningStep, ...]) -> str:
    lines = ["Reasoning:"]
    for idx, step in enumerate(steps, 1):
        lines.append(f"{idx}. {step.target_step}")
    return "\n".join(lines)


def infer_split(local_index: int) -> str:
    # Exactly 90/5/5 within every 20 examples in each category.
    mod = local_index % 20
    if mod == 0:
        return "dev"
    if mod == 1:
        return "test"
    return "train"


def repair_adjacent_problem_ids(rows: list[dict[str, Any]]) -> None:
    """Deterministically remove adjacent rows sharing a problem_id after sorting."""
    for i in range(1, len(rows)):
        if rows[i]["problem_id"] != rows[i - 1]["problem_id"]:
            continue
        swap_idx = None
        for j in range(i + 1, min(len(rows), i + 256)):
            candidate = rows[j]["problem_id"]
            if candidate != rows[i - 1]["problem_id"]:
                if j + 1 >= len(rows) or candidate != rows[j + 1]["problem_id"]:
                    swap_idx = j
                    break
        if swap_idx is not None:
            rows[i], rows[swap_idx] = rows[swap_idx], rows[i]
