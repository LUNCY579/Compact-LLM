from __future__ import annotations

"""Create mixed teacher-reference/student-trajectory data for on-policy KD.

Student trajectories receive ``hard_label_weight=0`` so subsequent KD does not
reinforce their mistakes through hard cross-entropy. Teacher-reference rows keep
the configured hard-label weight.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json, write_jsonl
from scripts.common.response_control import build_stopping_criteria, finalize_response

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate student trajectories for Phase 9 on-policy KD.")
    parser.add_argument("--student", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="eos_only")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def choose_student(record_id: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.student_fraction <= 1.0:
        raise ValueError("--student-fraction must be in [0,1]")
    if args.adapter is not None and PeftModel is None:
        raise RuntimeError("PEFT is required when --adapter is supplied")
    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.student, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    output: list[dict[str, Any]] = []
    student_count = 0
    for row in rows:
        record_id = str(row.get("id") or "")
        messages = [dict(message) for message in (row.get("messages") or [])]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{record_id} must end with an assistant reference")
        if not choose_student(record_id, seed=args.seed, fraction=args.student_fraction):
            teacher_row = dict(row)
            teacher_row.setdefault("metadata", {})
            teacher_row["metadata"] = {
                **teacher_row["metadata"],
                "trajectory_source": "teacher_reference",
                "hard_label_weight": teacher_row["metadata"].get("hard_label_weight"),
            }
            output.append(teacher_row)
            continue

        prompt_messages = messages[:-1]
        prompt = apply_chat_template(tokenizer, prompt_messages)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])
        stopping, _ = build_stopping_criteria(tokenizer, prompt_length, args.response_control)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping,
        }
        if args.do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        raw = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)
        answer = finalize_response(raw, args.response_control).text
        if not answer:
            answer = "..."
        trajectory = dict(row)
        trajectory["messages"] = [*prompt_messages, {"role": "assistant", "content": answer}]
        trajectory["hard_label_weight"] = 0.0
        trajectory["metadata"] = {
            **(row.get("metadata") or {}),
            "trajectory_source": "student_generated",
            "hard_label_weight": 0.0,
            "reference_answer": messages[-1].get("content"),
            "student_model": args.student,
            "student_adapter": str(args.adapter) if args.adapter else None,
        }
        output.append(trajectory)
        student_count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "source": str(args.input),
            "source_sha256": sha256(args.input),
            "student": args.student,
            "adapter": str(args.adapter) if args.adapter else None,
            "student_fraction_requested": args.student_fraction,
            "student_trajectory_count": student_count,
            "teacher_reference_count": len(output) - student_count,
            "hard_label_rule": "student-generated rows use hard_label_weight=0; teacher references retain configured weight",
        },
    )
    print(f"Wrote {len(output)} mixed trajectories to {args.output}")


if __name__ == "__main__":
    main()
