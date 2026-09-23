"""Collect native predictions for one SOUP-MINI-001 arm.

This is a gated collector, not a scorer or an authorization mechanism.  It loads
one locally materialized checkpoint at a time and emits raw predictions.  The
current registered protocol deliberately refuses because ``native_authorized``
is false and the runtime, suite, and review receipts are unbound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from soup import ARMS, EXPERIMENT, PINS, read_json, require, sha256


DOMAINS = (
    "transition",
    "arithmetic",
    "short_format",
    "answerable_retrieval",
    "unanswerable_abstention",
    "long_context",
)
NEW_ARMS = tuple(arm for arm in ARMS if not arm.startswith("local_"))
ALL_ARMS = tuple(PINS) + NEW_ARMS
MAX_ROWS = 256
MAX_TOTAL_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 64
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def hash_value(value: Any, field: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"unbound SHA-256: {field}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: row must be an object")
            rows.append(value)
    require(bool(rows), f"{path}: empty suite")
    require(len(rows) <= MAX_ROWS, f"suite exceeds {MAX_ROWS} rows")
    return rows


def validate_suite(path: Path, authorization: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    actual_sha256 = sha256(path)
    require(actual_sha256 == hash_value(authorization.get("dataset_sha256"), "dataset_sha256"),
            "suite hash does not match authorization")
    rows = read_jsonl(path)
    ids: set[str] = set()
    parents: set[str] = set()
    for row in rows:
        require(set(row) == {"id", "parent_id", "domain", "answerable", "prompt"},
                "suite rows must contain only identity, role, and rendered prompt")
        for key in ("id", "parent_id", "prompt"):
            require(isinstance(row[key], str) and row[key].strip(), f"missing suite field: {key}")
        require(row["id"] not in ids, "duplicate suite row id")
        require(row["parent_id"] not in parents, "collector requires one canonical view per parent")
        require(row["domain"] in DOMAINS, "unknown suite domain")
        require(type(row["answerable"]) is bool, "answerable must be boolean")
        ids.add(row["id"])
        parents.add(row["parent_id"])
    return rows, actual_sha256


def validate_admission(
    *,
    protocol_path: Path,
    authorization_path: Path,
    arm: str,
    split: str,
    model_dir: Path,
    suite_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    protocol_sha256 = sha256(protocol_path)
    protocol = read_json(protocol_path)
    authorization = read_json(authorization_path)
    require(protocol.get("experiment") == EXPERIMENT, "wrong protocol experiment")
    require(authorization.get("experiment") == EXPERIMENT, "wrong authorization experiment")
    require(authorization.get("protocol_sha256") == protocol_sha256,
            "authorization is bound to a different protocol")
    require(protocol.get("native_authorized") is True,
            "registered protocol has native_authorized=false")
    require(authorization.get("native_authorized") is True,
            "native authorization receipt is not active")
    require(split in ("CAL", "CHECK"), "split must be CAL or CHECK")
    require(arm in ALL_ARMS, "unknown or unregistered arm")
    require(authorization.get("arm_id") == arm, "authorization arm does not match request")
    require(authorization.get("split") == split, "authorization split does not match request")
    for field in (
        "authorization_id",
        "suite_id",
        "runtime_sha256",
        "scorer_sha256",
        "decoding_sha256",
        "tokenizer_sha256",
        "qualification_receipt_sha256",
        "ancestry_audit_sha256",
        "dataset_sha256",
        "freeze_sha256",
    ):
        if field.endswith("_sha256"):
            hash_value(authorization.get(field), field)
        else:
            require(isinstance(authorization.get(field), str) and authorization[field].strip(),
                    f"unbound authorization field: {field}")
    require(authorization.get("evaluation_dtype") == "float32",
            "native collector requires evaluation_dtype=float32")
    require(model_dir.is_dir() and not model_dir.is_symlink(),
            "model directory must be a real local directory")
    weights = model_dir / "model.safetensors"
    tokenizer = model_dir / "tokenizer.json"
    require(weights.is_file() and not weights.is_symlink(),
            "unsharded model.safetensors is required")
    require(tokenizer.is_file() and not tokenizer.is_symlink(),
            "tokenizer.json is required")
    expected_checkpoint = authorization.get("checkpoint_sha256", {}).get(arm)
    require(sha256(weights) == hash_value(expected_checkpoint, f"checkpoint_sha256.{arm}"),
            "checkpoint hash does not match authorization")
    require(sha256(tokenizer) == authorization["tokenizer_sha256"],
            "tokenizer hash does not match authorization")
    rows, dataset_sha256 = validate_suite(suite_path, authorization)
    return authorization, rows, dataset_sha256, protocol_sha256


def render_prediction_run(
    *,
    authorization: dict[str, Any],
    arm: str,
    split: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    dataset_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    metadata = {
        "experiment": EXPERIMENT,
        "arm_id": arm,
        "split": split,
        "suite_id": authorization["suite_id"],
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": authorization["checkpoint_sha256"][arm],
        "tokenizer_sha256": authorization["tokenizer_sha256"],
        "runtime_sha256": authorization["runtime_sha256"],
        "scorer_sha256": authorization["scorer_sha256"],
        "decoding_sha256": authorization["decoding_sha256"],
        "qualification_receipt_sha256": authorization["qualification_receipt_sha256"],
        "ancestry_audit_sha256": authorization["ancestry_audit_sha256"],
        "freeze_sha256": authorization["freeze_sha256"],
        "protocol_sha256": protocol_sha256,
        "evaluation_dtype": "float32",
    }
    require(len(rows) == len(predictions), "prediction count mismatch")
    return {
        "schema": 1,
        "status": "RAW_PREDICTIONS_NOT_SCORED",
        "metadata": metadata,
        "generation": {
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "batch_size": 1,
            "do_sample": False,
            "model_dir": str(model_dir),
            "authorization_id": authorization["authorization_id"],
        },
        "rows": predictions,
    }


def collect(
    *,
    authorization: dict[str, Any],
    arm: str,
    split: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    dataset_sha256: str,
    protocol_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    require(max_new_tokens == DEFAULT_MAX_NEW_TOKENS,
            "SOUP-MINI-001 requires max_new_tokens=64")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("qualified torch/transformers runtime is unavailable") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    require(all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for parameter in model.parameters()
    ), "loaded model contains non-FP32 floating-point parameters")
    if tokenizer.pad_token_id is None:
        require(tokenizer.eos_token_id is not None, "tokenizer has neither pad nor EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_device = model.get_input_embeddings().weight.device
    predictions: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded["input_ids"].shape[-1])
        require(input_tokens + max_new_tokens <= MAX_TOTAL_TOKENS,
                f"{row['id']}: prompt exceeds the 4096-token budget")
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, input_tokens:]
        predictions.append({
            "id": row["id"],
            "parent_id": row["parent_id"],
            "domain": row["domain"],
            "answerable": row["answerable"],
            "status": "ok",
            "prediction": tokenizer.decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip(),
            "input_tokens": input_tokens,
            "generated_tokens": int(continuation.shape[-1]),
        })
    return render_prediction_run(
        authorization=authorization,
        arm=arm,
        split=split,
        model_dir=model_dir,
        rows=rows,
        predictions=predictions,
        dataset_sha256=dataset_sha256,
        protocol_sha256=protocol_sha256,
    )


def write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), "refusing to overwrite an existing prediction artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ALL_ARMS)
    parser.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()
    try:
        authorization, rows, dataset_sha256, protocol_sha256 = validate_admission(
            protocol_path=args.protocol,
            authorization_path=args.authorization,
            arm=args.arm,
            split=args.split,
            model_dir=args.model,
            suite_path=args.suite,
        )
        result = collect(
            authorization=authorization,
            arm=args.arm,
            split=args.split,
            model_dir=args.model,
            rows=rows,
            dataset_sha256=dataset_sha256,
            protocol_sha256=protocol_sha256,
            max_new_tokens=args.max_new_tokens,
        )
        write_new(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "arm": args.arm,
            "split": args.split,
            "rows": len(result["rows"]),
            "output_sha256": sha256(args.output),
        }))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
