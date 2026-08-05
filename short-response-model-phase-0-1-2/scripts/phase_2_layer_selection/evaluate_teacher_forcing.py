from __future__ import annotations

"""Measure assistant-target NLL with durable, resumable per-case evidence.

Generation quality remains the primary human-facing evidence.  Teacher-forced
negative log likelihood supplies a continuous sensitivity signal when generation
scores are close or saturated.  Every completed case is appended durably so a
Windows interruption or process failure does not force the whole NLL pass to be
repeated.
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import (
    approved_content_only_migration,
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_json_best_effort,
)
from scripts.common.incremental_results import DurableJsonlStore


def apply_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate assistant-target NLL on an SFT-style JSONL probe.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse durable completed rows by stable input ID.")
    parser.add_argument(
        "--legacy-input-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _canonical_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    candidate = Path(text)
    resolved = candidate if candidate.is_absolute() else ROOT / candidate
    if resolved.exists():
        try:
            return str(resolved.resolve()).casefold()
        except OSError:
            return str(resolved.absolute()).casefold()
    return text


def build_manifest(args: argparse.Namespace, record_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": str(args.model_id),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "requested_limit": args.limit,
        "input_case_count": len(record_ids),
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest(),
        "max_sequence_length": int(args.max_sequence_length),
        "evaluator_script_sha256": sha256(Path(__file__)),
    }


def manifest_mismatches(existing: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    path_fields = {"model_id", "adapter_path", "input"}
    # Artifacts from v1.4.6 predate this manifest. Missing fields are accepted
    # only for migration; conflicting fields remain fatal.
    optional_legacy = {
        "input_sha256",
        "requested_limit",
        "input_case_count",
        "record_ids_sha256",
        "evaluator_script_sha256",
    }
    mismatches: list[str] = []
    for key, requested_value in requested.items():
        if key == "schema_version":
            continue
        if key not in existing:
            if key in optional_legacy:
                continue
            mismatches.append(f"{key}: missing from existing manifest")
            continue
        existing_value = existing.get(key)
        equal = (
            _canonical_identifier(existing_value) == _canonical_identifier(requested_value)
            if key in path_fields
            else existing_value == requested_value
        )
        if not equal:
            mismatches.append(f"{key}: existing={existing_value!r}, requested={requested_value!r}")
    return mismatches


def finalize(
    args: argparse.Namespace,
    store: DurableJsonlStore,
    record_ids: list[str],
) -> dict[str, Any]:
    ordered = store.finalize_order(record_ids)
    evaluated = [row for row in ordered if int(row.get("target_tokens", 0)) > 0]
    total_tokens = sum(int(row["target_tokens"]) for row in evaluated)
    if total_tokens == 0:
        raise RuntimeError("No assistant target tokens were evaluated.")
    total_loss = sum(float(row["mean_target_nll"]) * int(row["target_tokens"]) for row in evaluated)
    mean_nll = total_loss / total_tokens
    summary = {
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "input": str(args.input),
        "input_case_count": len(ordered),
        "case_count": len(evaluated),
        "skipped_case_count": len(ordered) - len(evaluated),
        "target_token_count": total_tokens,
        "mean_target_nll": mean_nll,
        "target_perplexity": math.exp(min(20.0, mean_nll)),
        "interpretation_boundary": (
            "NLL is a continuous sensitivity signal for saturated generation scores. "
            "It does not replace generated-output, category, safety, or human review."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json_best_effort(
        args.output_dir / "evaluation_progress.json",
        {
            "status": "complete",
            "examples_completed": len(ordered),
            "examples_total": len(record_ids),
            "evaluated_case_count": len(evaluated),
            "skipped_case_count": len(ordered) - len(evaluated),
        },
        label="teacher-forcing progress",
    )
    print(args.output_dir / "summary.json")
    return summary


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None  # type: ignore[assignment]

    args = parse_args()
    if args.max_sequence_length <= 1:
        raise ValueError("--max-sequence-length must be greater than 1.")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable and --require-cuda was requested.")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("The selected teacher-forcing input slice contains no records.")

    normalized_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        stable_id = str(row.get("id", f"row_{index}"))
        row["id"] = stable_id
        normalized_ids.append(stable_id)
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Teacher-forcing input contains duplicate stable IDs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "evaluation_manifest.json"
    expected_manifest = build_manifest(args, normalized_ids)
    if args.resume and manifest_path.exists():
        existing = read_json(manifest_path)
        mismatches = manifest_mismatches(existing, expected_manifest)
        if mismatches:
            migration = approved_content_only_migration(
                args.input,
                previous_sha256=existing.get("input_sha256"),
                current_sha256=expected_manifest.get("input_sha256"),
            )
            remaining = [item for item in mismatches if not item.startswith("input_sha256:")]
            if migration is not None and not remaining:
                print(
                    "[approved input migration] IDs/messages/targets are unchanged; "
                    "durable NLL rows remain valid"
                )
                write_json(
                    args.output_dir / "INPUT_CONTRACT_MIGRATION.json",
                    {
                        "schema_version": 1,
                        "migration_type": "approved_contract_only_dataset_migration",
                        "previous_input_sha256": migration["previous_sha256"],
                        "current_input_sha256": migration["current_sha256"],
                        "dataset_manifest": migration["manifest_path"],
                        "stable_ids_messages_targets_unchanged": True,
                        "nll_was_not_recomputed": True,
                    },
                )
            else:
                raise RuntimeError(
                    "Teacher-forcing artifacts are incompatible with the requested model, input, or policy.\n  - "
                    + "\n  - ".join(mismatches)
                )
    elif args.resume and (args.output_dir / "results.jsonl").exists():
        migration = approved_content_only_migration(
            args.input,
            previous_sha256=args.legacy_input_sha256,
            current_sha256=expected_manifest.get("input_sha256"),
        )
        if migration is None:
            raise RuntimeError(
                "Legacy teacher-forcing rows have no provenance manifest and cannot be reused safely. "
                "Restart this NLL directory, or resume through the Phase 2C v1.4.7 wrapper so the "
                "authorized contract-only migration can be verified."
            )
        legacy_summary_path = args.output_dir / "summary.json"
        if legacy_summary_path.exists():
            legacy_summary = read_json(legacy_summary_path)
            checks = {
                "model_id": (legacy_summary.get("model_id"), args.model_id),
                "adapter_path": (legacy_summary.get("adapter_path"), str(args.adapter_path) if args.adapter_path else None),
                "input": (legacy_summary.get("input"), str(args.input)),
            }
            conflicts = [
                f"{key}: existing={existing!r}, requested={requested!r}"
                for key, (existing, requested) in checks.items()
                if _canonical_identifier(existing) != _canonical_identifier(requested)
            ]
            if conflicts:
                raise RuntimeError(
                    "Legacy teacher-forcing summary conflicts with the requested evaluation.\n  - "
                    + "\n  - ".join(conflicts)
                )
        write_json(
            args.output_dir / "INPUT_CONTRACT_MIGRATION.json",
            {
                "schema_version": 1,
                "migration_type": "approved_legacy_phase2c_contract_only_dataset_migration",
                "previous_input_sha256": migration["previous_sha256"],
                "current_input_sha256": migration["current_sha256"],
                "dataset_manifest": migration["manifest_path"],
                "stable_ids_messages_targets_unchanged": True,
                "legacy_rows_validated_by_stable_id": True,
                "nll_was_not_recomputed": True,
            },
        )
        print(
            "[approved legacy NLL migration] manifestless v1.4.6 rows will be "
            "validated by stable ID and reused"
        )
    write_json(manifest_path, expected_manifest)

    store = DurableJsonlStore.open(args.output_dir / "results.jsonl", resume=args.resume)
    unknown_ids = sorted(set(store.rows_by_id) - set(normalized_ids))
    if unknown_ids:
        raise RuntimeError(f"Existing NLL results contain IDs not present in this input: {unknown_ids[:8]}")
    for warning in store.recovery_warnings:
        print(f"[jsonl recovery] {warning}")

    completed = store.rows_in_order(normalized_ids)
    if len(completed) == len(rows):
        finalize(args, store, normalized_ids)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    if args.adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("Install peft before loading an adapter.")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    for index, row in enumerate(rows, 1):
        stable_id = str(row["id"])
        if store.contains(stable_id):
            print(f"[{index:03d}/{len(rows):03d}] {stable_id}: [resume reuse]")
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{args.input}: record {stable_id} must end with an assistant target.")
        prompt_messages = messages[:-1]
        prompt_text = apply_template(tokenizer, prompt_messages, generation=True)
        full_text = apply_template(tokenizer, messages, generation=False)
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_sequence_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_sequence_length,
        )["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            store.append(
                {
                    "id": stable_id,
                    "target_tokens": 0,
                    "mean_target_nll": None,
                    "target_perplexity": None,
                    "skipped": True,
                    "skip_reason": "assistant target was fully removed by truncation",
                }
            )
            print(f"[{index:03d}/{len(rows):03d}] {stable_id}: [skipped target truncated]")
        else:
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits
            shifted_logits = logits[:, :-1, :].float()
            shifted_targets = input_ids[:, 1:]
            start = max(0, len(prompt_ids) - 1)
            target_logits = shifted_logits[:, start:, :]
            target_ids = shifted_targets[:, start:]
            token_losses = F.cross_entropy(
                target_logits.reshape(-1, target_logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            )
            case_tokens = int(token_losses.numel())
            case_loss_sum = float(token_losses.sum().item())
            case_mean = case_loss_sum / max(1, case_tokens)
            store.append(
                {
                    "id": stable_id,
                    "target_tokens": case_tokens,
                    "mean_target_nll": case_mean,
                    "target_perplexity": math.exp(min(20.0, case_mean)),
                    "skipped": False,
                }
            )
            print(f"[{index:03d}/{len(rows):03d}] {stable_id}: nll={case_mean:.5f}")

        write_json_best_effort(
            args.output_dir / "evaluation_progress.json",
            {
                "status": "running",
                "examples_completed": len(store.rows_in_order(normalized_ids)),
                "examples_total": len(rows),
            },
            label="teacher-forcing progress",
        )

    finalize(args, store, normalized_ids)


if __name__ == "__main__":
    main()
