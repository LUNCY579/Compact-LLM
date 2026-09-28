"""Tokenize only open bridge roles, with no model import, downloads or truncation.

Original data are immutable. Derived IDs/masks and measured counts are written
once to a new output directory. Neither CHECK nor FINAL files are opened.
"""
from __future__ import annotations

import os

# These must be set before importing transformers, including indirect imports.
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as bridge

OPEN_ROLES = ("TRAIN", "TRAIN_SELECTION", "DEV")
EOS_ID = 151645
MAX_TOTAL = 1024
MAX_COMPLETION = 128
TEMPLATE_KWARGS = {"enable_thinking": False, "add_generation_prompt": True}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=bridge._unique_object,
                      parse_constant=bridge._reject_constant)


def file_identity(path):
    raw = path.read_bytes()
    return {"sha256": bridge.sha256(raw), "bytes": len(raw)}


def load_tokenizer(model_dir: Path):
    require("torch" not in sys.modules, "tokenizer process must not already contain torch")
    # Validate before importing: newer unrelated runtimes may ignore USE_TORCH.
    for name, expected in (("transformers", "4.57.1"), ("tokenizers", "0.22.2")):
        require(importlib.metadata.version(name) == expected,
                f"wrong runtime: {name} must be {expected}; use the pinned E0 interpreter")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    require("torch" not in sys.modules, "unexpected torch import in tokenizer-only process")
    return tokenizer


def _ids(ids):
    require(type(ids) is list and all(type(x) is int and x >= 0 for x in ids), "invalid token IDs")
    return ids


def assemble_example(prompt_ids, completion_ids, pad_to=None):
    """Mask by positions, never token value: PAD=EOS must preserve completion EOS."""
    _ids(prompt_ids)
    _ids(completion_ids)
    require(bool(prompt_ids), "empty prompt tokenization")
    require(bool(completion_ids) and completion_ids[-1] == EOS_ID, "completion EOS mismatch")
    require(len(completion_ids) <= MAX_COMPLETION, "completion exceeds 128 tokens; truncation forbidden")
    length = len(prompt_ids) + len(completion_ids)
    require(length <= MAX_TOTAL, "example exceeds 1024 tokens; truncation forbidden")
    if pad_to is None:
        pad_to = length
    require(type(pad_to) is int and length <= pad_to <= MAX_TOTAL, "invalid padded sequence length")
    padding = pad_to - length
    return {
        "input_ids": prompt_ids + completion_ids + [EOS_ID] * padding,
        "attention_mask": [1] * length + [0] * padding,
        "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids) + [0] * padding,
        "labels": [-100] * len(prompt_ids) + completion_ids + [-100] * padding,
        "prompt_tokens": len(prompt_ids), "target_json_tokens": len(completion_ids) - 1,
        "completion_tokens": len(completion_ids), "nonpadding_tokens": length,
        "padding_tokens": padding,
    }


def tokenize_row(row: dict, tokenizer) -> dict:
    require(row.get("role") in OPEN_ROLES, "only open roles may be tokenized")
    require(tokenizer.eos_token_id == EOS_ID, "Titan EOS ID mismatch")
    require(tokenizer.eos_token == "<|im_end|>", "Titan EOS string mismatch")
    parent = bridge.allocate_parent(row["spec"], row["generator_seed"], row["candidate_index"])
    require(parent.role == row["role"] and parent.parent_id == row["parent_id"], "row parent/role drift")
    require(parent.canonical_parent_sha256 == row["canonical_parent_sha256"], "canonical parent drift")
    require(bridge.sha256(row["prompt"].encode("utf-8")) == row["prompt_sha256"], "prompt hash mismatch")
    require(bridge.sha256(row["target"].encode("utf-8")) == row["target_sha256"], "target hash mismatch")
    require(bridge.render_prompt(parent, row["variant_id"]) == row["prompt"], "prompt differs from bound parent")
    require(bridge.verify_answer(parent, row["target"]), "source target failed independent verifier")
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
    prompt_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True, **TEMPLATE_KWARGS))
    require(prompt_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "non-thinking assistant generation prefix mismatch")
    require(prompt_ids == tokenizer.encode(prompt_text, add_special_tokens=False), "rendered prompt tokenization differs")
    target_ids = _ids(tokenizer.encode(row["target"], add_special_tokens=False))
    require(EOS_ID not in target_ids, "target contains an unexpected EOS token")
    completion_ids = target_ids + [EOS_ID]
    require(tokenizer.decode(completion_ids, skip_special_tokens=False) == row["target"] + tokenizer.eos_token,
            "completion roundtrip mismatch")
    full_messages = messages + [{"role": "assistant", "content": row["target"]}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False,
                                            add_generation_prompt=False, enable_thinking=False)
    expected_full = prompt_text + row["target"] + tokenizer.eos_token + "\n"
    require(full_text == expected_full, "assistant template is not the exact prompt + target + EOS + newline")
    full_ids = _ids(tokenizer.apply_chat_template(full_messages, tokenize=True,
                                                add_generation_prompt=False, enable_thinking=False))
    require(full_ids == tokenizer.encode(full_text, add_special_tokens=False), "full template tokenization differs")
    require(full_ids[:len(prompt_ids)] == prompt_ids, "full conversation changes the prompt token prefix")
    require(full_ids[len(prompt_ids):len(prompt_ids) + len(completion_ids)] == completion_ids,
            "completion token boundary differs from separate target encoding")
    require(full_ids[len(prompt_ids) + len(completion_ids):] == tokenizer.encode("\n", add_special_tokens=False),
            "unexpected template tail after EOS")
    result = assemble_example(prompt_ids, completion_ids)
    result.update({
        "schema_version": 1, "row_id": row["row_id"], "parent_id": row["parent_id"],
        "family": row["family"], "role": row["role"], "variant_id": row["variant_id"],
        "source_row_sha256": bridge.sha256(bridge.json_bytes(row)),
        "source_prompt_sha256": row["prompt_sha256"], "source_target_sha256": row["target_sha256"],
        "tokenized_prompt_sha256": bridge.sha256(bridge.json_bytes(prompt_ids)),
        "completion_token_ids_sha256": bridge.sha256(bridge.json_bytes(completion_ids)),
        "input_token_ids_sha256": bridge.sha256(bridge.json_bytes(result["input_ids"])),
        "token_hash_encoding": "UTF8 canonical compact JSON integer list; SHA256",
        "rendered_prompt_sha256": bridge.sha256(prompt_text.encode("utf-8")),
        "template_kwargs": dict(TEMPLATE_KWARGS), "completion_eos_id": EOS_ID,
        "mask_contract": "prompt=-100; completion including EOS=token ID; right padding=-100",
        "training_role_eligible": row["role"] == "TRAIN", "training_eligibility": False,
        "training_eligibility_reasons": ["HISTORICAL_EXCLUSION_NOT_QUALIFIED", "RETENTION_NOT_QUALIFIED",
                                         "NATIVE_AND_COMPLETE_E1_LAUNCH_GATES_NOT_QUALIFIED"],
    })
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "sum": 0, "min": None, "max": None, "mean": None,
                "p50": None, "p90": None, "p95": None, "p99": None}
    result = {"count": len(values), "sum": sum(values), "min": values[0], "max": values[-1],
              "mean": sum(values) / len(values)}
    result.update({f"p{q}": values[math.ceil(q / 100 * len(values)) - 1] for q in (50, 90, 95, 99)})
    return result


def summarize(rows):
    return {field: distribution([row[field] for row in rows])
            for field in ("prompt_tokens", "target_json_tokens", "completion_tokens", "nonpadding_tokens")}


def tokenize_dataset(data_dir: Path, tokenizer_receipt_path: Path, output: Path) -> dict:
    require(not output.exists(), "output is write-once and must not exist")
    started = time.perf_counter()
    receipt_path = data_dir / "DATA_RECEIPT.json"
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "MEASURED_DETERMINISTIC_DATA_MECHANICS_ONLY", "unqualified source data status")
    require(receipt["generator_file_sha256"] == file_identity(Path(bridge.__file__))["sha256"], "bridge generator source drift")
    registration_path = data_dir / "REGISTRATION.json"
    require(file_identity(registration_path) == receipt["files"]["REGISTRATION.json"], "source registration drift")
    registration = read_json(registration_path)
    counts = bridge.validate_registration(registration)
    require(registration["context_contract"]["max_total_tokens"] == MAX_TOTAL
            and registration["context_contract"]["max_target_tokens"] == MAX_COMPLETION
            and registration["context_contract"]["truncation"].startswith("forbidden"), "context contract drift")
    source_files, source_rows = {}, {}
    for role in OPEN_ROLES:
        name = role + ".jsonl"
        raw = (data_dir / name).read_bytes()
        identity = {"sha256": bridge.sha256(raw), "bytes": len(raw)}
        require(identity == receipt["files"][name], "source file drift: " + name)
        source_files[name] = identity
        source_rows[role] = [json.loads(line, object_pairs_hook=bridge._unique_object,
                                      parse_constant=bridge._reject_constant) for line in raw.splitlines()]
        require(len(source_rows[role]) == receipt["open_row_counts"][role], "open row count drift")
        require(len({row["parent_id"] for row in source_rows[role]}) == counts[role], "parent count drift")
        require(all(row["role"] == role for row in source_rows[role]), "role/file mismatch")
    tokenizer_receipt = read_json(tokenizer_receipt_path)
    model_dir = Path(tokenizer_receipt["checkpoint_path"])
    tokenizer_files = {name: file_identity(model_dir / name) for name in TOKENIZER_FILES}
    require(tokenizer_files == tokenizer_receipt["tokenizer_files"], "bound tokenizer file drift")
    template = read_json(model_dir / "tokenizer_config.json")["chat_template"]
    template_hash = bridge.sha256(template.encode("utf-8"))
    require(template_hash == tokenizer_receipt["chat_template_sha256"], "bound chat template drift")
    tokenizer = load_tokenizer(model_dir)
    require(tokenizer.chat_template == template, "runtime tokenizer chat template drift")
    rows = {role: [tokenize_row(row, tokenizer) for row in source_rows[role]] for role in OPEN_ROLES}
    all_rows = [row for role in OPEN_ROLES for row in rows[role]]
    roles_by_parent = {}
    for row in all_rows:
        require(roles_by_parent.setdefault(row["parent_id"], row["role"]) == row["role"], "parent crosses roles")
    require({name: file_identity(model_dir / name) for name in TOKENIZER_FILES} == tokenizer_files,
            "tokenizer files changed during measurement")
    output.mkdir(parents=True, exist_ok=False)
    output_files = {}
    for role in OPEN_ROLES:
        path = output / (role + ".tokenized.jsonl")
        with path.open("xb") as handle:
            for row in rows[role]:
                handle.write(bridge.json_bytes(row) + b"\n")
        output_files[path.name] = file_identity(path)
    package_versions = {}
    for name in ("transformers", "tokenizers"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = "UNAVAILABLE"
    result = {
        "schema_version": 1, "status": "MEASURED_OPEN_ROLE_TOKENIZATION_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_versions, "script_sha256": file_identity(Path(__file__))["sha256"],
        "source_data_receipt_sha256": file_identity(receipt_path)["sha256"],
        "source_registration_sha256": file_identity(registration_path)["sha256"],
        "source_open_files_verified": source_files, "source_role_parent_counts": {role: counts[role] for role in OPEN_ROLES},
        "source_row_counts": {role: len(rows[role]) for role in OPEN_ROLES},
        "sealed_access": {"opened": False, "roles": ["CHECK", "FINAL"],
                          "reservation_file_hash_from_source_receipt_only": receipt["files"]["SEALED_RESERVATIONS.jsonl"]["sha256"],
                          "independently_hashed": False},
        "tokenizer_receipt_sha256": file_identity(tokenizer_receipt_path)["sha256"],
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_files": tokenizer_files,
        "chat_template_sha256": template_hash, "template_kwargs": dict(TEMPLATE_KWARGS),
        "completion_policy": "exact target JSON encoding, no special tokens added, then EOS151645; template trailing newline excluded",
        "prompt_hash_encoding": "SHA256 of UTF8 canonical compact JSON integer list",
        "original_pad_id": tokenizer.pad_token_id, "padding_policy": "no stored padding; positional helper supports PAD=EOS151645",
        "limits": {"max_total_tokens": MAX_TOTAL, "max_completion_tokens_including_eos": MAX_COMPLETION},
        "truncation": False, "oversize_rows": 0, "files": output_files,
        "distributions": {"all_open": summarize(all_rows),
                          "by_role": {role: summarize(rows[role]) for role in OPEN_ROLES},
                          "by_role_family": {role: {family: summarize([r for r in rows[role] if r["family"] == family])
                                                     for family in bridge.FAMILIES} for role in OPEN_ROLES}},
        "training_role_nonpadding_tokens_per_pass": sum(row["nonpadding_tokens"] for row in rows["TRAIN"]),
        "training_role_supervised_tokens_per_pass": sum(row["completion_tokens"] for row in rows["TRAIN"]),
        "cost": {"wall_seconds": time.perf_counter() - started, "model_loads": 0, "model_calls": 0,
                 "teacher_calls": 0, "gpu_seconds": 0, "actual_training_tokens": 0},
        "torch_imported": "torch" in sys.modules, "environment": {name: os.environ[name] for name in
                           ("USE_TORCH", "USE_TF", "USE_FLAX", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM")},
        "tokenizer_context_qualified": True, "historical_exclusion_qualified": False,
        "retention_dataset_qualified": False, "native_qualification_inferred": False,
        "training_eligibility": False, "e1_launch_ready": False,
    }
    (output / "TOKENIZATION_RECEIPT.json").write_bytes(bridge.json_bytes(result) + b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = tokenize_dataset(args.data, args.tokenizer_receipt, args.output)
    receipt_path = args.output / "TOKENIZATION_RECEIPT.json"
    print(json.dumps({"receipt": str(receipt_path.resolve()), "receipt_sha256": file_identity(receipt_path)["sha256"],
                      "source_row_counts": result["source_row_counts"], "maximum_tokens": result["distributions"]["all_open"]["nonpadding_tokens"]["max"],
                      "training_nonpadding_tokens_per_pass": result["training_role_nonpadding_tokens_per_pass"],
                      "torch_imported": result["torch_imported"], "e1_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
