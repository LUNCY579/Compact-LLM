from __future__ import annotations

"""Token-aware, auditable prompt construction for Phase 9.

Phase 9 is a grounding experiment. Silent tokenizer truncation is therefore a
technical invalidity: it can remove the evidence or the final question while
leaving an apparently valid generation. This module budgets retrieved passages
*before* rendering the chat prompt and records exactly what the model received.
"""

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

from scripts.phase_9_student_distillation.phase9_common import build_grounded_messages, passage_id


@dataclass(frozen=True)
class PromptPlan:
    messages: list[dict[str, Any]]
    prompt_text: str
    prompt_token_ids: list[int]
    full_prompt_tokens_before_budget: int
    selected_evidence: list[dict[str, Any]]
    selected_procedures: list[dict[str, Any]]
    dropped_evidence_ids: list[str]
    dropped_procedure_ids: list[str]
    prompt_sha256: str
    question_present: bool
    grounding_contract_present: bool
    evidence_section_present: bool
    procedure_section_present: bool
    hidden_truncation_used: bool
    integrity_issues: list[str]

    def diagnostics(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("messages", None)
        payload.pop("prompt_text", None)
        payload.pop("prompt_token_ids", None)
        payload["prompt_tokens"] = len(self.prompt_token_ids)
        payload["selected_evidence_ids"] = [passage_id(item) for item in self.selected_evidence]
        payload["selected_procedure_ids"] = [passage_id(item) for item in self.selected_procedures]
        payload["integrity_ok"] = not self.integrity_issues and not self.hidden_truncation_used
        return payload


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool = True) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _truncate_text_to_tokens(tokenizer: Any, text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    ids = token_ids(tokenizer, text)
    if len(ids) <= limit:
        return text
    try:
        return str(tokenizer.decode(ids[:limit], skip_special_tokens=True)).strip()
    except TypeError:
        return str(tokenizer.decode(ids[:limit])).strip()


def _budget_passages(
    tokenizer: Any,
    passages: Sequence[dict[str, Any]],
    *,
    token_budget: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    dropped: list[str] = []
    remaining = max(0, int(token_budget))
    for original in passages:
        item = dict(original)
        text = str(item.get("text") or "").strip()
        item_id = passage_id(item)
        if not text or remaining <= 0:
            dropped.append(item_id)
            continue
        ids = token_ids(tokenizer, text)
        if len(ids) <= remaining:
            selected.append(item)
            remaining -= len(ids)
            continue
        # Preserve a useful prefix only when there is enough room to avoid a
        # misleading one- or two-token fragment. The diagnostic records that
        # the source passage was shortened.
        if remaining >= 8:
            shortened = _truncate_text_to_tokens(tokenizer, text, remaining)
            if shortened:
                item["text"] = shortened
                item["prompt_truncated"] = True
                item["original_text_token_count"] = len(ids)
                selected.append(item)
            else:
                dropped.append(item_id)
        else:
            dropped.append(item_id)
        remaining = 0
    return selected, dropped


def _render(
    tokenizer: Any,
    *,
    question: str,
    evidence: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    mode: str,
    system_prompt: str | None,
) -> tuple[list[dict[str, Any]], str, list[int]]:
    messages = build_grounded_messages(
        question=question,
        evidence=evidence,
        procedures=procedures,
        mode=mode,
        system_prompt=system_prompt,
    )
    prompt = apply_chat_template(tokenizer, messages, generation=True)
    return messages, prompt, token_ids(tokenizer, prompt)


def plan_grounded_prompt(
    tokenizer: Any,
    *,
    question: str,
    evidence: Sequence[dict[str, Any]],
    procedures: Sequence[dict[str, Any]],
    mode: str,
    system_prompt: str | None,
    max_sequence_length: int,
    maximum_evidence_tokens: int,
) -> PromptPlan:
    """Build a prompt that preserves the grounding contract and final question.

    Candidate passages are first limited by a shared evidence budget. If the
    rendered chat prompt still exceeds the model context, lowest-priority
    passages are removed from the end. The system contract and question are
    never tail-truncated. If those fixed fields alone do not fit, an integrity
    issue is returned and the caller must stop rather than silently continue.
    """

    raw_evidence = [dict(item) for item in evidence]
    raw_procedures = [dict(item) for item in procedures]
    # Trusted procedures receive at most one quarter of the evidence budget;
    # unused procedure capacity remains available to factual evidence.
    procedure_budget = min(maximum_evidence_tokens // 4, 96) if raw_procedures else 0
    selected_procedures, dropped_procedures = _budget_passages(
        tokenizer, raw_procedures, token_budget=procedure_budget
    )
    used_procedure_tokens = sum(len(token_ids(tokenizer, str(item.get("text") or ""))) for item in selected_procedures)
    evidence_budget = max(0, maximum_evidence_tokens - used_procedure_tokens)
    selected_evidence, dropped_evidence = _budget_passages(
        tokenizer, raw_evidence, token_budget=evidence_budget
    )

    # Social prompts intentionally omit evidence/procedure sections. Real RAG may
    # still retrieve passages; drop them from the plan so diagnostics match what
    # the model actually receives and integrity checks stay honest.
    if mode == "social_roleplay":
        dropped_evidence.extend(passage_id(item) for item in selected_evidence)
        dropped_procedures.extend(passage_id(item) for item in selected_procedures)
        selected_evidence = []
        selected_procedures = []

    full_messages, full_prompt, full_ids = _render(
        tokenizer,
        question=question,
        evidence=raw_evidence,
        procedures=raw_procedures,
        mode=mode,
        system_prompt=system_prompt,
    )
    del full_messages, full_prompt

    messages, prompt, ids = _render(
        tokenizer,
        question=question,
        evidence=selected_evidence,
        procedures=selected_procedures,
        mode=mode,
        system_prompt=system_prompt,
    )
    # Remove lowest-priority passages until the entire prompt fits. Retrieval
    # order is assumed to be highest priority first.
    while len(ids) > max_sequence_length and (selected_evidence or selected_procedures):
        if selected_evidence:
            removed = selected_evidence.pop()
            dropped_evidence.append(passage_id(removed))
        else:
            removed = selected_procedures.pop()
            dropped_procedures.append(passage_id(removed))
        messages, prompt, ids = _render(
            tokenizer,
            question=question,
            evidence=selected_evidence,
            procedures=selected_procedures,
            mode=mode,
            system_prompt=system_prompt,
        )

    issues: list[str] = []
    question_present = bool(question) and question in prompt
    grounding_contract_present = "Use only the supplied trusted procedures" in prompt or mode == "social_roleplay"
    evidence_section_present = "<FACTUAL_EVIDENCE>" in prompt
    procedure_section_present = "<TRUSTED_PROCEDURES>" in prompt
    if len(ids) > max_sequence_length:
        issues.append(
            f"fixed system/question prompt requires {len(ids)} tokens, exceeding max_sequence_length={max_sequence_length}"
        )
    if not question_present:
        issues.append("final question is missing from rendered prompt")
    if mode != "social_roleplay" and not grounding_contract_present:
        issues.append("grounding contract is missing from rendered prompt")
    if mode != "social_roleplay" and selected_evidence and not evidence_section_present:
        issues.append("selected factual evidence is missing from rendered prompt")
    if mode != "social_roleplay" and selected_procedures and not procedure_section_present:
        issues.append("selected trusted procedures are missing from rendered prompt")

    return PromptPlan(
        messages=messages,
        prompt_text=prompt,
        prompt_token_ids=ids,
        full_prompt_tokens_before_budget=len(full_ids),
        selected_evidence=selected_evidence,
        selected_procedures=selected_procedures,
        dropped_evidence_ids=sorted(set(dropped_evidence)),
        dropped_procedure_ids=sorted(set(dropped_procedures)),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        question_present=question_present,
        grounding_contract_present=grounding_contract_present,
        evidence_section_present=evidence_section_present,
        procedure_section_present=procedure_section_present,
        hidden_truncation_used=False,
        integrity_issues=issues,
    )
