"""Titan-matched training recipe pieces, VENDORED (not re-derived) from the native lane.

Source checkout: ChatBubble-retention-pilot-20260917 @ 5b4c22dae78f587c41575269f419087b0c7997f6
(branch codex/dataset-retention-audit-v1; files last changed in 2cd46c16):

* ``rate_at``        <- experiments/model_quality_gpu_v1/config.py  (WSD 20/60/20, sqrt decay to 0.1x)
* ``group_updates``  <- experiments/model_quality_gpu_v1/data.py    (order-preserving <=N-token groups)
* ``compile_row``    <- experiments/reasoning_posttraining_v1/serialization.py (direct mode only)

The native lane itself is pinned to torch 2.10.0 / transformers 4.57.1 and cannot host a masked
trainer (its engine hardcodes the tail, the LR and has no mask hook), so the pieces are vendored
into this experiment's own runner. The single adaptation: transformers 5.x returns a
``BatchEncoding`` from ``apply_chat_template(tokenize=True)``; ``_native_ids`` normalises that.
Every contract check of the original is kept (prefix ends with the closed think stub, full ==
prefix + final + EOS + "\\n", tokenisation agrees with the native template, exactly one EOS,
no truncation).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

VENDORED_FROM = {"checkout": "ChatBubble-retention-pilot-20260917",
                 "commit": "5b4c22dae78f587c41575269f419087b0c7997f6",
                 "files_last_changed": "2cd46c16195a65e57819c69e3a994eb4c5642934"}


@dataclass(frozen=True)
class Recipe:
    """Optimiser/schedule knobs. Defaults are Titan's (CORE.json), except ``peak_lr``, which is a
    registered study parameter (Titan used 1e-5 on 384 rows)."""
    peak_lr: float = 1e-5
    warmup_fraction: float = 0.20
    decay_fraction: float = 0.20
    minimum_lr_fraction: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    update_tokens: int = 2048
    max_sequence: int = 2048
    completion_reserve: int = 128

    def __post_init__(self):
        if not (math.isfinite(self.peak_lr) and self.peak_lr > 0):
            raise ValueError("peak_lr must be positive")
        if self.weight_decay != 0.0:
            # Non-zero decoupled decay would move frozen-at-zero coordinates only if they were
            # non-zero; it is kept at 0 so that the masked-coordinate invariant is exact.
            raise ValueError("weight_decay must be 0 in CLARE-RINEYA-001")

    def as_dict(self) -> dict:
        return asdict(self)


def rate_at(peak: float, completed_tokens: int, total_tokens: int, recipe: Recipe) -> float:
    """Vendored verbatim (model_quality_gpu_v1/config.py::rate_at), Design -> Recipe."""
    if not math.isfinite(peak) or peak <= 0 or not 0 < completed_tokens <= total_tokens:
        raise ValueError("invalid learning-rate position")
    progress = completed_tokens / total_tokens
    if progress < recipe.warmup_fraction:
        return peak * progress / recipe.warmup_fraction
    start = 1.0 - recipe.decay_fraction
    if progress <= start:
        return peak
    phase = min(1.0, (progress - start) / recipe.decay_fraction)
    return peak * (recipe.minimum_lr_fraction
                   + (1.0 - recipe.minimum_lr_fraction) * (1.0 - math.sqrt(phase)))


def group_updates(rows: list[dict], token_limit: int) -> list[list[int]]:
    """Vendored verbatim (model_quality_gpu_v1/data.py::group_updates)."""
    groups, current, used = [], [], 0
    for i, row in enumerate(rows):
        n = len(row["ids"])
        if n > token_limit:
            raise ValueError("sequence cannot fit update group")
        if current and used + n > token_limit:
            groups.append(current); current, used = [], 0
        current.append(i); used += n
    if current:
        groups.append(current)
    return groups


def _native_ids(encoded) -> list[int]:
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one sequence")
        encoded = encoded[0]
    return [int(t) for t in encoded]


def compile_row(prompt: list[dict], final: str, tokenizer, *, max_sequence: int = 2048,
                completion_reserve: int = 128) -> dict:
    """Vendored from reasoning_posttraining_v1/serialization.py (direct mode): assistant-only
    labels covering the answer and its EOS; the trailing newline after EOS is unsupervised."""
    if ([m.get("role") for m in prompt] != ["system", "user"]
            or any(set(m) != {"role", "content"} or not isinstance(m["content"], str)
                   or not m["content"].strip() or "<|" in m["content"] for m in prompt)):
        raise ValueError("only explicit system/user evidence is allowed")
    if not isinstance(final, str) or not final.strip() or final != final.strip():
        raise ValueError("final target must be nonempty canonical text")
    if any(x in final for x in ("<|", "<think>", "</think>", "<final>", "</final>")):
        raise ValueError("embedded control delimiter in final target")
    if type(max_sequence) is not int or max_sequence < 3 or type(completion_reserve) is not int or completion_reserve < 1:
        raise ValueError("invalid context budget")
    assistant = {"role": "assistant", "content": final}
    kwargs = {"enable_thinking": False}
    prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prompt + [assistant], tokenize=False, add_generation_prompt=False, **kwargs)
    if not prefix.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("unqualified native direct prefix")
    if full != prefix + final + tokenizer.eos_token + "\n":
        raise ValueError("native training/generation serialization differs")
    ids = tokenizer.encode(full, add_special_tokens=False)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    native = _native_ids(tokenizer.apply_chat_template(prompt + [assistant], tokenize=True,
                                                       add_generation_prompt=False, **kwargs))
    if ids != native or ids[:len(before)] != before:
        raise ValueError("native token prefix mismatch")
    eos_positions = [i for i in range(len(before), len(ids)) if ids[i] == tokenizer.eos_token_id]
    if len(eos_positions) != 1 or eos_positions[0] <= len(before):
        raise ValueError("exactly one post-answer EOS required")
    end = eos_positions[0]
    if tokenizer.decode(ids[end + 1:], skip_special_tokens=False) != "\n":
        raise ValueError("unqualified post-EOS suffix")
    if len(ids) > max_sequence or len(before) + completion_reserve > max_sequence:
        raise ValueError("OVERLENGTH: no truncation, silent skip or evidence removal")
    labels = [token if len(before) <= i <= end else -100 for i, token in enumerate(ids)]
    return {"ids": ids, "prompt_ids": before, "labels": labels,
            "answer_start": len(before), "answer_end": end, "mode": "direct",
            "processed_tokens": len(ids), "supervised_tokens": end - len(before) + 1,
            "final_content_tokens": len(tokenizer.encode(final, add_special_tokens=False)),
            "completion_reserve": completion_reserve, "max_sequence": max_sequence,
            "post_eos_tokens_supervised": False, "truncated": False}
