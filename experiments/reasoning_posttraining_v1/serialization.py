"""Two explicit native chat contracts; no model loading or invented thinking channel."""
from __future__ import annotations
import re


def assistant_record(final: str, trace: str | None, mode: str) -> dict:
    if mode not in ("direct", "reasoning"):
        raise ValueError("unknown response contract")
    if not isinstance(final, str) or not final.strip() or final != final.strip():
        raise ValueError("final target must be nonempty canonical text")
    if any(x in final for x in ("<|", "<think>", "</think>", "<final>", "</final>")):
        raise ValueError("embedded control delimiter in final target")
    if mode == "direct":
        if trace is not None:
            raise ValueError("direct target cannot hide an active reasoning trace")
        return {"role": "assistant", "content": final}
    if not isinstance(trace, str) or not trace.strip() or trace != trace.strip():
        raise ValueError("reasoning target requires a separately verified bounded trace")
    if any(x in trace for x in ("<|", "<think>", "</think>", "<final>", "</final>")):
        raise ValueError("embedded control delimiter in reasoning target")
    return {"role": "assistant", "content": "<final>" + final + "</final>",
            "reasoning_content": trace}


def compile_row(prompt: list[dict], final: str, tokenizer, *, mode="direct",
                trace: str | None = None, max_sequence=2048, completion_reserve=128) -> dict:
    if ([m.get("role") for m in prompt] != ["system", "user"]
            or any(set(m) != {"role", "content"} or not isinstance(m["content"], str)
                   or not m["content"].strip() or "<|" in m["content"] for m in prompt)):
        raise ValueError("only explicit system/user evidence is allowed")
    return _compile(prompt, final, tokenizer, mode=mode, trace=trace,
                    max_sequence=max_sequence, completion_reserve=completion_reserve)


def compile_dialogue(prompt: list[dict], final: str, tokenizer, *,
                     max_sequence=2048, completion_reserve=128) -> dict:
    """Final-turn-only supervision while preserving every previous conversation turn.

    A new explicitly requested serialization mode. Historical compile_row remains
    single-turn. Native prefix/token/EOS checks are shared, not reimplemented.
    """
    if (not isinstance(prompt, list) or len(prompt) < 2 or len(prompt) % 2 or
            any(not isinstance(m, dict) or set(m) != {"role", "content"} or
                not isinstance(m["content"], str) or not m["content"].strip() or
                "<|" in m["content"] for m in prompt)):
        raise ValueError("invalid complete dialogue prompt")
    expected = ["system"] + ["user" if i % 2 else "assistant" for i in range(1, len(prompt))]
    if [m["role"] for m in prompt] != expected:
        raise ValueError("dialogue must alternate user/assistant and end with user")
    return _compile(prompt, final, tokenizer, mode="direct", trace=None,
                    max_sequence=max_sequence, completion_reserve=completion_reserve)


def _compile(prompt, final, tokenizer, *, mode, trace, max_sequence, completion_reserve):
    if type(max_sequence) is not int or max_sequence < 3 or type(completion_reserve) is not int or completion_reserve < 1:
        raise ValueError("invalid context budget")
    assistant = assistant_record(final, trace, mode)
    kwargs = {"enable_thinking": mode == "reasoning"}
    prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prompt + [assistant], tokenize=False, add_generation_prompt=False, **kwargs)
    if mode == "direct":
        suffix = final
        if not prefix.endswith("<think>\n\n</think>\n\n"):
            raise ValueError("unqualified native direct prefix")
        reasoning_tokens = 0
    else:
        if not prefix.endswith("<|im_start|>assistant\n"):
            raise ValueError("active trace would follow an already closed thinking stub")
        suffix = "<think>\n" + trace + "\n</think>\n\n<final>" + final + "</final>"
        reasoning_tokens = len(tokenizer.encode(trace, add_special_tokens=False))
    if full != prefix + suffix + tokenizer.eos_token + "\n":
        raise ValueError("native training/generation serialization differs")
    ids = tokenizer.encode(full, add_special_tokens=False)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    native = tokenizer.apply_chat_template(prompt + [assistant], tokenize=True, add_generation_prompt=False, **kwargs)
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
            "answer_start": len(before), "answer_end": end, "mode": mode,
            "processed_tokens": len(ids), "supervised_tokens": end - len(before) + 1,
            "reasoning_content_tokens": reasoning_tokens,
            "final_content_tokens": len(tokenizer.encode(final, add_special_tokens=False)),
            "completion_reserve": completion_reserve, "max_sequence": max_sequence,
            "post_eos_tokens_supervised": False, "truncated": False}
