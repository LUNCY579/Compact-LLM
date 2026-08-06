from __future__ import annotations

"""Generate teacher reconstruction candidates for human review.

Teacher output is never accepted automatically. The script adds a parsed teacher
candidate and preserves the raw text so reviewers can reject malformed,
unsupported, or overlong reconstructions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import append_jsonl_durable, read_jsonl, read_jsonl_recover, sha256_tree, write_json
from scripts.phase_9_student_distillation.evaluate_rag import generate_answer
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    PIPELINE_REVISION,
    build_reconstruction_messages,
    parse_reconstruction_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 9B teacher reconstruction candidates.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 9B teacher-reconstruction review packet",
        "",
        "Teacher reconstructions are candidates, not labels. Verify source-state applicability, provenance, "
        "the final answer, and any rejected assumptions before setting `review_status=accepted`.",
        "",
    ]
    for row in rows:
        teacher = row.get("teacher_reconstruction") or {}
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- Question: {row['question']}",
                f"- Teacher parse valid: `{row.get('teacher_parse_valid')}`",
                f"- Teacher parse error: `{row.get('teacher_parse_error')}`",
                f"- Teacher decision: `{teacher.get('decision')}`",
                f"- Teacher guidance: {teacher.get('guidance') or '(empty)' }",
                f"- Provisional decision: `{(row.get('provisional_reconstruction') or {}).get('decision')}`",
                f"- Current approved answer candidate: {row.get('approved_answer') or '(missing)' }",
                "",
                "Raw teacher output:",
                "",
                "```text",
                str(row.get("teacher_raw_output") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    rows = read_jsonl(args.candidates)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing, _ = read_jsonl_recover(args.output, id_field="id")
        completed = {str(row["id"]): row for row in existing}
    elif args.output.exists():
        raise FileExistsError(f"Output exists; use --resume: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires CUDA and bitsandbytes")
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(args.teacher, **model_kwargs)
    model.eval()

    for row in rows:
        record_id = str(row.get("id"))
        if record_id in completed:
            continue
        messages = build_reconstruction_messages(
            question=str(row.get("question") or ""),
            current_state=dict(row.get("current_state") or {}),
            memories=list(row.get("memories") or []),
            mode=row.get("mode"),
        )
        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_sequence_length=args.max_sequence_length,
            max_new_tokens=args.max_new_tokens,
            response_control="eos_only",
        )
        parsed = parse_reconstruction_output(str(generated["answer"]))
        enriched = {
            **row,
            "teacher_reconstruction": parsed.payload,
            "teacher_parse_valid": parsed.valid,
            "teacher_parse_error": parsed.error,
            "teacher_raw_output": parsed.raw_text,
            "teacher_generation": {
                key: generated.get(key)
                for key in (
                    "generated_tokens",
                    "visible_tokens",
                    "generation_seconds",
                    "prompt_tokens",
                    "stop_reason",
                )
            },
            "review_status": "pending",
            "selected_reconstruction_source": None,
            "approved_reconstruction": None,
        }
        append_jsonl_durable(args.output, enriched)
        completed[record_id] = enriched

    ordered = [completed[str(row["id"])] for row in rows if str(row["id"]) in completed]
    packet = args.review_packet or args.output.with_suffix(".review.md")
    packet.write_text(review_markdown(ordered), encoding="utf-8")
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "schema_version": "1.0",
            "pipeline_revision": PIPELINE_REVISION,
            "teacher": args.teacher,
            "teacher_tree_sha256": sha256_tree(Path(args.teacher)) if Path(args.teacher).exists() else None,
            "candidate_count": len(rows),
            "completed_count": len(ordered),
            "output": str(args.output),
            "review_packet": str(packet),
        },
    )
    print(args.output)
    print(packet)


if __name__ == "__main__":
    main()
