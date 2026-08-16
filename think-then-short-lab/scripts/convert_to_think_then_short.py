#!/usr/bin/env python3
"""Convert instruction/reasoning/output JSONL into think-then-short records.

Foreign rows with Alpaca-style columns are rewritten into the lab schema:
  messages  = [system, user(instruction)]
  thinking  = reasoning (or thinking)
  answer    = output only

Rows whose tokenized answer or thinking exceed configured budgets are dropped
(never truncated). Official sealed Phase datasets are not imported here.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WHITESPACE_TOKEN = re.compile(r"\S+")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return len(WHITESPACE_TOKEN.findall(text))


def load_tokenizer(model_path: str | None) -> Any | None:
    if not model_path:
        return None
    path = Path(model_path)
    if not path.exists():
        print(f"warning: tokenizer path missing ({model_path}); using whitespace fallback")
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("warning: transformers not installed; using whitespace fallback")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        return tokenizer
    except Exception as exc:  # noqa: BLE001 - conversion should continue with fallback
        print(f"warning: failed to load tokenizer ({exc}); using whitespace fallback")
        return None


def validate_record(row: dict[str, Any]) -> None:
    for key in ("id", "messages", "thinking", "answer", "source"):
        if key not in row:
            raise ValueError(f"converted row missing required field: {key}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{row['id']}: messages must contain system and user")
    if messages[0].get("role") != "system" or messages[-1].get("role") != "user":
        raise ValueError(f"{row['id']}: messages must start with system and end with user")
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            raise ValueError(f"{row['id']}: messages may only contain system/user roles")
        if not str(message.get("content") or "").strip():
            raise ValueError(f"{row['id']}: empty message content")
    if not str(row["answer"]).strip():
        raise ValueError(f"{row['id']}: answer must be non-empty")
    if row.get("thinking") is None:
        raise ValueError(f"{row['id']}: thinking must be a string (may be empty)")


def convert_row(
    raw: dict[str, Any],
    *,
    index: int,
    system_prompt: str,
    source_name: str,
    max_answer_tokens: int,
    max_thinking_tokens: int,
    allow_empty_thinking: bool,
    drop_overlong_answer: bool,
    drop_overlong_thinking: bool,
    tokenizer: Any | None,
) -> tuple[dict[str, Any] | None, str | None]:
    instruction = str(raw.get("instruction") or "").strip()
    output = str(raw.get("output") or "").strip()
    thinking = str(raw.get("thinking") if raw.get("thinking") is not None else raw.get("reasoning") or "").strip()

    if not instruction:
        return None, "missing_instruction"
    if not output:
        return None, "missing_output"
    if not thinking and not allow_empty_thinking:
        return None, "empty_thinking_disallowed"

    answer_tokens = count_tokens(output, tokenizer)
    thinking_tokens = count_tokens(thinking, tokenizer) if thinking else 0

    if drop_overlong_answer and answer_tokens > max_answer_tokens:
        return None, "answer_exceeds_max_answer_tokens"
    if drop_overlong_thinking and thinking_tokens > max_thinking_tokens:
        return None, "thinking_exceeds_max_thinking_tokens"

    row_id = str(raw.get("id") or f"{source_name}_{index:06d}")
    record = {
        "id": row_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        "thinking": thinking,
        "answer": output,
        "source": source_name,
        "empty_thinking": not bool(thinking),
        "thinking_token_count": thinking_tokens,
        "answer_token_count": answer_tokens,
        "provenance": {
            "converter": "think_then_short_lab.convert_to_think_then_short",
            "source_fields": {
                "instruction": True,
                "reasoning_or_thinking": bool(raw.get("reasoning") is not None or raw.get("thinking") is not None),
                "output": True,
            },
        },
    }
    validate_record(record)
    return record, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Foreign JSONL with instruction/output (+ optional reasoning)")
    parser.add_argument("--output", type=Path, required=True, help="Converted think-then-short JSONL")
    parser.add_argument("--config", type=Path, default=Path("configs/think_then_short.json"))
    parser.add_argument("--source-name", type=str, default=None, help="Override source label written into each row")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path; defaults to config student.init")
    parser.add_argument("--audit", type=Path, default=None, help="Optional audit JSON path")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    cfg = read_json(args.config)
    thinking_cfg = cfg.get("thinking") or {}
    conversion_cfg = cfg.get("conversion") or {}
    student_cfg = cfg.get("student") or {}

    system_prompt = str(thinking_cfg.get("system_prompt") or "").strip()
    if not system_prompt:
        raise ValueError("config.thinking.system_prompt is required")
    max_answer_tokens = int(thinking_cfg.get("max_answer_tokens", 32))
    max_thinking_tokens = int(thinking_cfg.get("max_thinking_tokens", 512))
    allow_empty_thinking = bool(conversion_cfg.get("allow_empty_thinking", True))
    drop_overlong_answer = bool(conversion_cfg.get("drop_overlong_answer", True))
    drop_overlong_thinking = bool(conversion_cfg.get("drop_overlong_thinking", True))

    tokenizer_path = args.tokenizer or student_cfg.get("default_init")
    tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)
    source_name = args.source_name or args.input.stem
    audit_path = args.audit or args.output.with_suffix(".conversion_audit.json")

    kept: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    total = 0
    # utf-8-sig strips a Windows BOM if present without affecting ordinary UTF-8 files.
    with args.input.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            raw = json.loads(line)
            if not isinstance(raw, dict):
                drop_reasons["non_object_row"] += 1
                continue
            record, reason = convert_row(
                raw,
                index=index,
                system_prompt=system_prompt,
                source_name=source_name,
                max_answer_tokens=max_answer_tokens,
                max_thinking_tokens=max_thinking_tokens,
                allow_empty_thinking=allow_empty_thinking,
                drop_overlong_answer=drop_overlong_answer,
                drop_overlong_thinking=drop_overlong_thinking,
                tokenizer=tokenizer,
            )
            if reason is not None:
                drop_reasons[reason] += 1
                continue
            assert record is not None
            kept.append(record)

    write_jsonl(args.output, kept)
    audit = {
        "schema_version": "0.1.0",
        "input": str(args.input),
        "output": str(args.output),
        "config": str(args.config),
        "source_name": source_name,
        "tokenizer": str(tokenizer_path) if tokenizer_path else None,
        "tokenizer_mode": "pretrained" if tokenizer is not None else "whitespace_fallback",
        "max_answer_tokens": max_answer_tokens,
        "max_thinking_tokens": max_thinking_tokens,
        "total_rows": total,
        "kept_rows": len(kept),
        "dropped_rows": total - len(kept),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "empty_thinking_kept": sum(1 for row in kept if row.get("empty_thinking")),
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Wrote {len(kept)} rows -> {args.output}")
    print(f"Audit -> {audit_path}")


if __name__ == "__main__":
    main()
