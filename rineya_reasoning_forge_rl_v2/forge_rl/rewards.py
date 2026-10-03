"""Correctness-first outcome rewards and strict, channel-aware output parsing."""
from __future__ import annotations
import re
from typing import Any
from .common import loads, dumps, integer
from .oracle import verify_solution

MAX_RESPONSE_CHARACTERS = 100_000
MAX_FINAL_CHARACTERS = 8192


def parse_response(text: str, mode: str = "final_only") -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in ("final_only", "reason_then_final"):
        raise ValueError("Unknown output mode")
    if not isinstance(text, str) or len(text) > MAX_RESPONSE_CHARACTERS:
        raise ValueError("Response must be bounded text")
    trace = ""
    final = text.strip()
    if mode == "reason_then_final":
        for tag in ("<think>", "</think>", "<final>", "</final>"):
            if text.count(tag) != 1:
                raise ValueError("Exactly one reasoning and one final block are required")
        match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<final>(.*?)</final>\s*", text, flags=re.S)
        if not match:
            raise ValueError("Malformed or contaminated channel envelope")
        trace, final = match[1].strip(), match[2].strip()
        if not trace:
            raise ValueError("Reasoning-supervision mode requires a nonempty trace")
    if len(final) > MAX_FINAL_CHARACTERS:
        raise ValueError("Final response exceeds the bounded parser limit")
    try:
        obj = loads(final)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Final response is not one strict JSON object") from exc
    if not isinstance(obj, dict):
        raise ValueError("Final response must be a JSON object")
    return obj, {"final_characters": len(final), "trace_characters": len(trace),
                 "response_characters": len(text), "final_words": len(final.split()),
                 "trace_words": len(trace.split())}


def score_response(world: dict[str, Any], response: str, *, mode: str = "final_only",
                   certificate: dict[str, Any] | None = None,
                   efficiency_weight: float = 0.0, trace_tokens: int | None = None,
                   trace_budget: int | None = None) -> dict[str, Any]:
    """Default is binary correctness. Optional length shaping is bounded and gated.

    Correct answers retain >=0.95, incorrect ones get 0. The optional penalty is
    based on redundant final serialization and, ONLY when measured, trace tokens.
    No reward is granted just for producing a short string or plausible prose.
    A valid final answer does not certify the truth/faithfulness of a free trace.
    """
    if not 0 <= efficiency_weight <= .05:
        raise ValueError("Efficiency weight must be between 0 and 0.05")
    if (trace_tokens is None) != (trace_budget is None):
        raise ValueError("Supply measured trace tokens and a token budget together")
    if trace_tokens is not None and (not integer(trace_tokens) or trace_tokens < 0 or not integer(trace_budget) or trace_budget <= 0):
        raise ValueError("Token counters must be nonnegative integers and budget positive")
    out = {"reward": 0.0, "correct": False, "format_valid": False,
           "schema_valid": False, "feasible": False, "reason": "parse_failure",
           "final_characters": 0, "trace_characters": 0,
           "response_characters": len(response) if isinstance(response, str) else 0,
           "trace_verified": False}
    try:
        obj, sizes = parse_response(response, mode)
    except ValueError:
        return out
    out.update(sizes)
    out["format_valid"] = True
    out.update(verify_solution(world, obj, certificate))
    if out["correct"]:
        # Canonical serialization length is computed from this accepted witness,
        # not the generator's witness, so equally optimal plans are not penalized.
        budget = max(64, len(dumps(obj)) + 16)
        final_excess = max(0.0, (out["final_characters"] - budget) / budget)
        trace_excess = 0.0 if trace_tokens is None else max(0.0, (trace_tokens - trace_budget) / trace_budget)
        penalty = efficiency_weight * min(1.0, max(final_excess, trace_excess))
        out["reward"] = 1.0 - penalty
    return out
