"""Single-turn native chat compilation with proven prefix/EOS boundaries.

No truncation, packing, answer-dependent selection, or silent long-row exclusion.
Loss targets live at input-token positions and are shifted exactly once in losses.py.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import re
import unicodedata
from .config import Design
from .integrity import canonical, decode, digest, file_hash, regular, seal, unseal


BLOCKED_TEXT = ("<|", "<think>", "</think>", "<tool_call>", "<tool_response>")


def normal_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_rows(path: Path) -> list[dict]:
    path = regular(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("raw corpus exceeds 64 MiB v1 limit")
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(line) > 200000:
                raise ValueError("oversized corpus row")
            row = decode(line)
            if not isinstance(row, dict):
                raise ValueError("corpus row must be an object")
            for key in ("id", "template_family"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"row requires explicit {key}")
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
            if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
                raise ValueError("v1 accepts one user/assistant exchange, optional leading system only")
            for message in messages:
                if set(message) != {"role", "content"} or not isinstance(message["content"], str):
                    raise ValueError("only string role/content messages are supported")
                text = message["content"]
                if not text.strip() or any(s in text for s in BLOCKED_TEXT):
                    raise ValueError("empty content or embedded chat/control delimiters")
            if messages[-1]["content"].startswith("\n"):
                raise ValueError("native template would strip leading assistant newlines")
            # Preserve caller metadata, but never select by labels or model outcomes.
            result.append(row)
            if len(result) > 20000:
                raise ValueError("too many corpus rows")
    if not result:
        raise ValueError("empty corpus")
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate row IDs")
    return result


def overlap_audit(banks: dict[str, list[dict]]) -> dict:
    seen = {}
    counts = {}
    for split, rows in banks.items():
        counts[split] = len(rows)
        for row in rows:
            text = next(m["content"] for m in row["messages"] if m["role"] == "user")
            prompt = normal_prompt(text)
            keys = (("id", row["id"]), ("family", row["template_family"]),
                    ("prompt", prompt), ("digit_blind_prompt", re.sub(r"\d", "#", prompt)))
            for key in keys:
                previous = seen.get(key)
                if previous is not None and previous != split:
                    raise ValueError(f"cross-split {key[0]} overlap: {previous}/{split}")
                seen[key] = split
    return {"counts": counts, "exact_id_family_prompt_digit_blind": "PASS",
            "semantic_and_generator_leakage": "REQUIRES_INDEPENDENT_REVIEW"}


def encode_row(row: dict, tokenizer, design: Design) -> dict:
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
    eos = tokenizer.eos_token
    if not isinstance(eos, str) or not eos:
        raise ValueError("explicit native EOS required")
    # Native training and generation contracts must be identical up to the answer.
    if full != prefix + messages[-1]["content"] + eos + "\n":
        raise ValueError("native template does not satisfy the registered single-turn contract")
    ids = tokenizer.encode(full, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    eos_ids = tokenizer.encode(eos, add_special_tokens=False)
    if len(eos_ids) != 1 or eos_ids[0] != tokenizer.eos_token_id:
        raise ValueError("native EOS is not exactly one token")
    if ids[:len(prompt_ids)] != prompt_ids or not prompt_ids:
        raise ValueError("tokenization merges across the assistant boundary")
    native_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                               add_generation_prompt=False, enable_thinking=False)
    if ids != native_ids:
        raise ValueError("rendered and native tokenization disagree")
    start = len(prompt_ids)
    ends = [i for i in range(start, len(ids)) if ids[i] == eos_ids[0]]
    if len(ends) != 1 or ends[0] == start:
        raise ValueError("assistant must contain content followed by exactly one EOS")
    end = ends[0]
    if tokenizer.decode(ids[end + 1:], skip_special_tokens=False) != "\n":
        raise ValueError("unexpected post-EOS suffix")
    if len(ids) > design.max_sequence:
        raise ValueError(f"row {row['id']} exceeds sequence limit; no truncation or silent skipping")
    return {"id": row["id"], "family": row["template_family"],
            "ids": ids, "answer_start": start, "answer_end": end,
            "prompt_ids": prompt_ids, "reference": messages[-1]["content"],
            "raw_sha256": digest(row), "evaluation": row.get("evaluation")}


def targets(row: dict, objective: str) -> list[int]:
    if objective not in ("whole", "response"):
        raise ValueError("unknown objective")
    ids = row["ids"]
    if objective == "whole":
        return [-100] + ids[1:]
    return [token if row["answer_start"] <= i <= row["answer_end"] else -100
            for i, token in enumerate(ids)]


def group_updates(rows: list[dict], token_limit: int) -> list[list[int]]:
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


def compile_plan(train: list[dict], dev: list[dict], excluded: list[dict], tokenizer,
                 design: Design, inputs: dict, tokenizer_identity: dict, *, synthetic=False) -> dict:
    audit = overlap_audit({"train": train, "dev": dev, "excluded": excluded})
    if len(dev) > design.max_dev_rows:
        raise ValueError("explicitly select a bounded dev file before preparation")
    # Compile EVERY candidate row before selection. Invalid/long rows cannot be silently dropped.
    encoded = [encode_row(r, tokenizer, design) for r in sorted(train, key=lambda r: r["id"])]
    random.Random(design.selection_seed).shuffle(encoded)
    selected, count = [], 0
    for row in encoded:
        if count + len(row["ids"]) > design.processed_budget:
            break  # Frozen prefix, never search later rows for convenient lengths.
        selected.append(row); count += len(row["ids"])
    if count < math.ceil(design.min_budget_fraction * design.processed_budget):
        raise ValueError("insufficient full-row token coverage for the registered budget")
    random.Random(design.order_seed).shuffle(selected)
    body = {"schema": 1, "experiment": "MQ-001", "kind": "synthetic" if synthetic else "native_plan",
            "design": design.as_dict(), "inputs": inputs, "tokenizer": tokenizer_identity,
            "audit": audit, "processed_tokens": count, "train": selected,
            "dev": [encode_row(r, tokenizer, design) for r in dev],
            "groups": group_updates(selected, design.update_tokens)}
    return seal(body)


def validate_plan(value: dict) -> dict:
    plan = unseal(value)
    if set(plan) != {"schema", "experiment", "kind", "design", "inputs", "tokenizer", "audit",
                     "processed_tokens", "train", "dev", "groups"}:
        raise ValueError("unexpected plan fields")
    if plan["schema"] != 1 or plan["experiment"] != "MQ-001" or plan["kind"] not in ("synthetic", "native_plan"):
        raise ValueError("wrong plan scope")
    design = Design(**plan["design"])
    for bank in (plan["train"], plan["dev"]):
        if not isinstance(bank, list) or not bank:
            raise ValueError("empty plan bank")
        if len({r["id"] for r in bank}) != len(bank):
            raise ValueError("duplicate plan IDs")
        for r in bank:
            ids = r["ids"]
            if (not isinstance(ids, list) or len(ids) > design.max_sequence
                    or any(type(t) is not int or not 0 <= t < 151936 for t in ids)
                    or type(r["answer_start"]) is not int or type(r["answer_end"]) is not int
                    or not 0 < r["answer_start"] < r["answer_end"] < len(ids)
                    or r["prompt_ids"] != ids[:r["answer_start"]]):
                raise ValueError("invalid compiled token/mask boundary")
    count = sum(len(r["ids"]) for r in plan["train"])
    if (count != plan["processed_tokens"] or not math.ceil(design.processed_budget * design.min_budget_fraction)
            <= count <= design.processed_budget or len(plan["dev"]) > design.max_dev_rows):
        raise ValueError("plan budget mismatch")
    if plan["groups"] != group_updates(plan["train"], design.update_tokens):
        raise ValueError("plan update grouping mismatch")
    return plan


def review_template(plan: dict) -> dict:
    return {"schema": 1, "plan_sha256": plan["sha256"], "decision": "PENDING",
            "reviewer": "", "rationale": "", "source_rights_checked": False,
            "labels_and_response_contracts_checked": False,
            "scenario_and_generator_leakage_checked": False,
            "dev_only_no_confirmation_outputs": False}


def validate_review(review: dict, plan_sha: str) -> None:
    expected = review_template({"sha256": plan_sha})
    if set(review) != set(expected) or review["schema"] != 1 or review["plan_sha256"] != plan_sha:
        raise ValueError("review does not bind this plan")
    if review["decision"] != "APPROVED" or not review["reviewer"].strip() or not review["rationale"].strip():
        raise ValueError("independent data review remains pending")
    for key, value in expected.items():
        if type(value) is bool and review[key] is not True:
            raise ValueError(f"review missing: {key}")
