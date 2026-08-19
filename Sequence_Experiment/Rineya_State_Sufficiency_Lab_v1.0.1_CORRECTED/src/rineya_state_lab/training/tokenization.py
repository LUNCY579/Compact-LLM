from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizedConversation:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    target_tokens: int
    truncated_prompt_tokens: int
    truncated_target_tokens: int


def fallback_chat_render(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"]
        parts.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except Exception:
            pass
    return fallback_chat_render(messages, add_generation_prompt=add_generation_prompt)


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    length = min(len(left), len(right))
    index = 0
    while index < length and left[index] == right[index]:
        index += 1
    return index


def tokenize_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int,
    max_target_tokens: int | None = None,
    train_on_prompt: bool = False,
    append_eos: bool = True,
) -> TokenizedConversation:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("A training conversation must end with an assistant message")
    prompt_messages = messages[:-1]
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids = _tokenize_text(tokenizer, full_text)
    prompt_ids = _tokenize_text(tokenizer, prompt_text)
    assistant_start = _longest_common_prefix(full_ids, prompt_ids)
    if assistant_start == 0:
        raise ValueError("Could not identify an assistant-token boundary from the tokenizer chat template")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_id is not None and (not full_ids or full_ids[-1] != int(eos_id)):
        full_ids.append(int(eos_id))

    prompt_part = full_ids[:assistant_start]
    target_part = full_ids[assistant_start:]
    if not target_part:
        raise ValueError("The assistant target tokenized to zero tokens")

    truncated_target = 0
    if max_target_tokens is not None and len(target_part) > max_target_tokens:
        truncated_target = len(target_part) - max_target_tokens
        target_part = target_part[:max_target_tokens]
        if append_eos and eos_id is not None:
            target_part[-1] = int(eos_id)

    if len(target_part) >= max_length:
        # Preserve the beginning of the answer and a terminal EOS; this is preferable to silently training on no prompt.
        truncated_target += len(target_part) - (max_length - 1)
        target_part = target_part[: max_length - 1]
        if append_eos and eos_id is not None and target_part:
            target_part[-1] = int(eos_id)
        prompt_part = prompt_part[-1:]

    prompt_budget = max(1, max_length - len(target_part))
    truncated_prompt = max(0, len(prompt_part) - prompt_budget)
    if truncated_prompt:
        prompt_part = prompt_part[-prompt_budget:]

    input_ids = prompt_part + target_part
    if train_on_prompt:
        labels = list(input_ids)
    else:
        labels = [-100] * len(prompt_part) + list(target_part)
    attention_mask = [1] * len(input_ids)
    return TokenizedConversation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_tokens=len(prompt_part),
        target_tokens=len(target_part),
        truncated_prompt_tokens=truncated_prompt,
        truncated_target_tokens=truncated_target,
    )
