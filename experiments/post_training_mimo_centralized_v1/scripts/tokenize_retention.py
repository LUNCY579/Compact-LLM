"""Measure native token fit of open retention roles; never admit native training.

The unchanged bridge tokenizer loader and masking helper supply the exact Titan
template/token contract. Only TRAIN/DEV payloads are read; sealed reservations,
CHECK and FINAL are never opened. Outputs are write-once and never truncated.
"""
from __future__ import annotations

# Import this first: it disables model frameworks before any tokenizer import.
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_bridge as bridge_tokens
from experiments.post_training_mimo_centralized_v1.scripts import retention_data as retention

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

OPEN_ROLES = ("TRAIN", "DEV")
EOS_ID = bridge_tokens.EOS_ID
MAX_TOTAL = bridge_tokens.MAX_TOTAL
MAX_COMPLETION = bridge_tokens.MAX_COMPLETION
TEMPLATE_KWARGS = bridge_tokens.TEMPLATE_KWARGS
TOKENIZER_FILES = bridge_tokens.TOKENIZER_FILES
load_tokenizer = bridge_tokens.load_tokenizer
assemble_example = bridge_tokens.assemble_example
distribution = bridge_tokens.distribution
require = bridge_tokens.require
file_identity = bridge_tokens.file_identity
read_json = bridge_tokens.read_json


def _validate_source_row(row: dict):
    require(type(row) is dict, "source row must be an object")
    require(row.get("role") in OPEN_ROLES, "only TRAIN and DEV may be tokenized")
    require(type(row.get("variant_id")) is str, "retention variant_id must be a string")
    parent = retention.allocate_parent(row["spec"], row["generator_seed"], row["candidate_index"])
    require(parent.role == row["role"] and parent.parent_id == row["parent_id"], "row parent/role drift")
    require(parent.canonical_parent_sha256 == row["canonical_parent_sha256"], "canonical parent drift")
    require(parent.split_sha256 == row["split_sha256"] and parent.stratum == row["stratum"], "parent metadata drift")
    require(row["spec"] == json.loads(parent.canonical_json), "noncanonical source specification")
    require(row["domain"] == row["family"] == row["spec"]["domain"], "source domain/family drift")
    require(row["row_id"] == parent.parent_id + "-" + row["variant_id"], "retention row ID drift")
    require(retention.sha256(row["prompt"].encode("utf-8")) == row["prompt_sha256"], "prompt hash mismatch")
    require(retention.sha256(row["target"].encode("utf-8")) == row["target_sha256"], "target hash mismatch")
    require(retention.sha256(" ".join(row["prompt"].split()).encode("utf-8")) == row["normalized_prompt_sha256"],
            "normalized prompt hash mismatch")
    require(retention.render_prompt(parent, row["variant_id"]) == row["prompt"], "prompt differs from bound parent")
    require(retention.verify_answer(parent, row["target"], row["variant_id"]), "source target failed independent verifier")
    return parent


def tokenize_row(row: dict, tokenizer) -> dict:
    _validate_source_row(row)
    require(tokenizer.eos_token_id == EOS_ID, "Titan EOS ID mismatch")
    require(tokenizer.eos_token == "<|im_end|>", "Titan EOS string mismatch")
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
    prompt_ids = bridge_tokens._ids(tokenizer.apply_chat_template(messages, tokenize=True, **TEMPLATE_KWARGS))
    require(prompt_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "non-thinking assistant generation prefix mismatch")
    require(prompt_ids == tokenizer.encode(prompt_text, add_special_tokens=False), "rendered prompt tokenization differs")
    target_ids = bridge_tokens._ids(tokenizer.encode(row["target"], add_special_tokens=False))
    require(EOS_ID not in target_ids, "target contains an unexpected EOS token")
    completion_ids = target_ids + [EOS_ID]
    require(tokenizer.decode(completion_ids, skip_special_tokens=False) == row["target"] + tokenizer.eos_token,
            "completion roundtrip mismatch")
    full_messages = messages + [{"role": "assistant", "content": row["target"]}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False,
                                            add_generation_prompt=False, enable_thinking=False)
    require(full_text == prompt_text + row["target"] + tokenizer.eos_token + "\n",
            "assistant template is not the exact prompt + target + EOS + newline")
    full_ids = bridge_tokens._ids(tokenizer.apply_chat_template(full_messages, tokenize=True,
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
        "domain": row["domain"], "family": row["family"], "role": row["role"],
        "stratum": row["stratum"], "variant_id": row["variant_id"],
        "canonical_parent_sha256": row["canonical_parent_sha256"], "split_sha256": row["split_sha256"],
        "source_revision": row["source_revision"], "source_row_sha256": retention.sha256(retention.json_bytes(row)),
        "source_prompt_sha256": row["prompt_sha256"], "source_target_sha256": row["target_sha256"],
        "tokenized_prompt_sha256": retention.sha256(retention.json_bytes(prompt_ids)),
        "completion_token_ids_sha256": retention.sha256(retention.json_bytes(completion_ids)),
        "input_token_ids_sha256": retention.sha256(retention.json_bytes(result["input_ids"])),
        "token_hash_encoding": "UTF8 canonical compact JSON integer list; SHA256",
        "rendered_prompt_sha256": retention.sha256(prompt_text.encode("utf-8")),
        "template_kwargs": dict(TEMPLATE_KWARGS), "completion_eos_id": EOS_ID,
        "mask_contract": "prompt=-100; completion including EOS=token ID; right padding=-100",
        "training_role_eligible": row["role"] == "TRAIN", "training_eligibility": False,
        "source_correct_membership": None, "native_qualification_inferred": False,
        "training_eligibility_reasons": ["HISTORICAL_EXCLUSION_NOT_QUALIFIED", "SOURCE_CORRECT_ENROLLMENT_NOT_RUN",
                                         "BROAD_RETENTION_NOT_QUALIFIED", "NATIVE_AND_E1_LAUNCH_GATES_NOT_QUALIFIED"],
    })
    return result


def _validate_groups(source_rows: dict, counts: dict, receipt: dict, registration: dict):
    groups, row_ids, prompt_hashes = defaultdict(list), set(), set()
    for role in OPEN_ROLES:
        rows = source_rows[role]
        require(all(row.get("role") == role for row in rows), "role/file mismatch")
        require(len(rows) == receipt["open_row_counts"][role], "open row count drift")
        for row in rows:
            parent = _validate_source_row(row)
            require(row["source_id"] == "original-rineya-retention", "source identity drift")
            require(row["source_revision"] == receipt["source_revision"] == row["environment_revision"], "source revision drift")
            require(row["generator_seed"] == registration["selection_seed"], "source selection seed drift")
            require(row["environment_identity"] == retention.VERSION and row["verifier_version"] == retention.VERIFIER_VERSION,
                    "source generator/verifier version drift")
            require(row["row_id"] not in row_ids, "duplicate retention row ID")
            require(row["normalized_prompt_sha256"] not in prompt_hashes, "duplicate normalized source prompt")
            row_ids.add(row["row_id"])
            prompt_hashes.add(row["normalized_prompt_sha256"])
            groups[parent.parent_id].append(row)
        for domain in retention.DOMAINS:
            selected = [row for row in rows if row["domain"] == domain]
            parent_count = len({row["parent_id"] for row in selected})
            expected_rows = counts[role][domain] * (2 if domain == "answerability_abstention" else 1)
            require(parent_count == counts[role][domain], "domain parent count drift")
            require(len(selected) == expected_rows == receipt["domain_row_counts"][role].get(domain, 0),
                    "domain row count drift")
            strata = ["format_" + str(i) for i in range(4)] if domain == "instruction_following" else retention.POSITIONS
            for stratum in strata:
                actual = len({row["parent_id"] for row in selected if row["stratum"] == stratum})
                require(actual == counts[role][domain] // 4, "parent stratum count drift")
        require(len({row["parent_id"] for row in rows}) == receipt["parent_counts"][role] == sum(counts[role].values()),
                "role parent count drift")
    for rows in groups.values():
        first = rows[0]
        require(all(row["role"] == first["role"] for row in rows), "parent crosses roles")
        require(all(row["domain"] == first["domain"] and row["spec"] == first["spec"] for row in rows),
                "parent has conflicting domain/specification derivatives")
        expected = {"answerable", "missing_evidence"} if first["domain"] == "answerability_abstention" else {"standard"}
        require(len(rows) == len(expected) and {row["variant_id"] for row in rows} == expected,
                "missing or duplicate retention twin/variant")


def summarize(rows):
    return {field: distribution([row[field] for row in rows])
            for field in ("prompt_tokens", "target_json_tokens", "completion_tokens", "nonpadding_tokens")}


def tokenize_dataset(data_dir: Path, tokenizer_receipt_path: Path, output: Path) -> dict:
    require(not output.exists(), "output is write-once and must not exist")
    started = time.perf_counter()
    require(OPEN_ROLES == retention.OPEN_ROLES, "retention open-role contract drift")
    require("torch" not in sys.modules, "tokenizer process must not already contain torch")
    dependencies = {"retention_data.py": file_identity(Path(retention.__file__)),
                    "tokenize_bridge.py": file_identity(Path(bridge_tokens.__file__)),
                    "bridge_data.py": file_identity(Path(bridge_tokens.bridge.__file__))}
    receipt_path, registration_path = data_dir / "DATA_RECEIPT.json", data_dir / "REGISTRATION.json"
    bound_files = {str(receipt_path): file_identity(receipt_path), str(registration_path): file_identity(registration_path),
                   str(tokenizer_receipt_path): file_identity(tokenizer_receipt_path)}
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "MEASURED_DETERMINISTIC_RETENTION_MECHANICS_ONLY", "unqualified source data status")
    require(receipt["generator_file_sha256"] == dependencies["retention_data.py"]["sha256"], "retention generator source drift")
    require(receipt["generator_version"] == retention.VERSION and receipt["verifier_version"] == retention.VERIFIER_VERSION,
            "source generator/verifier version drift")
    require(bound_files[str(registration_path)] == receipt["files"]["REGISTRATION.json"], "source registration drift")
    registration = read_json(registration_path)
    counts = retention.validate_registration(registration)
    require(sum(sum(counts[role].values()) for role in OPEN_ROLES) > 0, "no open retention parents to tokenize")
    require(receipt["domain_parent_counts"] == counts and receipt["selection_seed"] == registration["selection_seed"],
            "registered allocation drift")
    require(registration.get("context_ceiling") == MAX_TOTAL and registration.get("completion_ceiling_including_eos") == MAX_COMPLETION,
            "context contract drift")
    source_files, source_rows = {}, {}
    for role in OPEN_ROLES:
        name = role + ".jsonl"
        raw = (data_dir / name).read_bytes()
        identity = {"sha256": retention.sha256(raw), "bytes": len(raw)}
        require(identity == receipt["files"][name], "source file drift: " + name)
        source_files[name] = identity
        bound_files[str(data_dir / name)] = identity
        source_rows[role] = [json.loads(line, object_pairs_hook=retention._unique_object,
                                      parse_constant=retention._reject_constant) for line in raw.splitlines()]
    _validate_groups(source_rows, counts, receipt, registration)
    tokenizer_receipt = read_json(tokenizer_receipt_path)
    model_dir = Path(tokenizer_receipt["checkpoint_path"])
    tokenizer_files = {name: file_identity(model_dir / name) for name in TOKENIZER_FILES}
    require(tokenizer_files == tokenizer_receipt["tokenizer_files"], "bound tokenizer file drift")
    template = read_json(model_dir / "tokenizer_config.json")["chat_template"]
    template_hash = retention.sha256(template.encode("utf-8"))
    require(template_hash == tokenizer_receipt["chat_template_sha256"], "bound chat template drift")
    tokenizer = load_tokenizer(model_dir)
    require(tokenizer.chat_template == template, "runtime tokenizer chat template drift")
    rows = {role: [tokenize_row(row, tokenizer) for row in source_rows[role]] for role in OPEN_ROLES}
    all_rows = [row for role in OPEN_ROLES for row in rows[role]]
    require(len({row["tokenized_prompt_sha256"] for row in all_rows}) == len(all_rows), "duplicate tokenized prompt")
    require("torch" not in sys.modules, "unexpected torch import in tokenizer-only process")
    require({name: file_identity(model_dir / name) for name in TOKENIZER_FILES} == tokenizer_files,
            "tokenizer files changed during measurement")
    require(all(file_identity(Path(path)) == identity for path, identity in bound_files.items()),
            "source or receipt changed during measurement")
    require(dependencies == {"retention_data.py": file_identity(Path(retention.__file__)),
                             "tokenize_bridge.py": file_identity(Path(bridge_tokens.__file__)),
                             "bridge_data.py": file_identity(Path(bridge_tokens.bridge.__file__))},
            "tokenization dependency changed during measurement")
    output.mkdir(parents=True, exist_ok=False)
    output_files = {}
    for role in OPEN_ROLES:
        path = output / (role + ".tokenized.jsonl")
        with path.open("xb") as handle:
            for row in rows[role]:
                handle.write(retention.json_bytes(row) + b"\n")
        output_files[path.name] = file_identity(path)
    packages = {}
    for name in ("transformers", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "UNAVAILABLE"
    result = {
        "schema_version": 1, "status": "MEASURED_OPEN_RETENTION_ROLE_TOKENIZATION_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "python_executable": sys.executable,
        "packages": packages, "script_sha256": file_identity(Path(__file__))["sha256"], "dependency_files": dependencies,
        "source_data_receipt_sha256": bound_files[str(receipt_path)]["sha256"],
        "source_registration_sha256": bound_files[str(registration_path)]["sha256"],
        "source_generator_sha256": dependencies["retention_data.py"]["sha256"],
        "source_open_files_verified": source_files,
        "source_role_parent_counts": {role: sum(counts[role].values()) for role in OPEN_ROLES},
        "source_domain_parent_counts": {role: counts[role] for role in OPEN_ROLES},
        "source_row_counts": {role: len(rows[role]) for role in OPEN_ROLES},
        "source_domain_row_counts": {role: dict(Counter(row["domain"] for row in rows[role])) for role in OPEN_ROLES},
        "twins": {"preserved": True, "grouping": "canonical semantic parent", "independent_unit": "parent; twins count once",
                  "parent_counts": {role: counts[role]["answerability_abstention"] for role in OPEN_ROLES}},
        "sealed_access": {"opened": False, "roles": ["CHECK", "FINAL"], "reservation_payload_opened": False,
                          "reservation_file_hash_from_source_receipt_only": receipt["files"]["SEALED_RESERVATIONS.jsonl"]["sha256"],
                          "independently_hashed": False},
        "tokenizer_receipt_sha256": bound_files[str(tokenizer_receipt_path)]["sha256"],
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_files": tokenizer_files,
        "chat_template_sha256": template_hash, "template_kwargs": dict(TEMPLATE_KWARGS),
        "completion_policy": "exact target JSON encoding then EOS151645; template trailing newline excluded",
        "prompt_hash_encoding": "SHA256 of UTF8 canonical compact JSON integer list",
        "original_pad_id": tokenizer.pad_token_id, "padding_policy": "no stored padding; unchanged helper masks PAD=EOS positionally",
        "limits": {"max_total_tokens": MAX_TOTAL, "max_completion_tokens_including_eos": MAX_COMPLETION},
        "truncation": False, "oversize_rows": 0, "files": output_files,
        "distributions": {"all_open": summarize(all_rows),
                          "by_role": {role: summarize(rows[role]) for role in OPEN_ROLES},
                          "by_role_domain": {role: {domain: summarize([r for r in rows[role] if r["domain"] == domain])
                                                     for domain in retention.DOMAINS} for role in OPEN_ROLES}},
        "training_role_nonpadding_tokens_per_pass": sum(row["nonpadding_tokens"] for row in rows["TRAIN"]),
        "training_role_supervised_tokens_per_pass": sum(row["completion_tokens"] for row in rows["TRAIN"]),
        "cost": {"wall_seconds": time.perf_counter() - started, "model_loads": 0, "model_calls": 0,
                 "teacher_calls": 0, "gpu_seconds": 0, "actual_training_tokens": 0},
        "torch_imported": "torch" in sys.modules,
        "environment": {name: os.environ[name] for name in
                        ("USE_TORCH", "USE_TF", "USE_FLAX", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM")},
        "tokenizer_context_qualified": True, "historical_exclusion_qualified": False,
        "source_correct_enrollment_qualified": False, "retention_dataset_qualified": False,
        "language_preservation_qualified": False, "long_context_preservation_qualified": False,
        "broad_preservation_coverage_qualified": False, "native_qualification_inferred": False,
        "training_eligibility": False, "e1_launch_ready": False,
    }
    with (output / "TOKENIZATION_RECEIPT.json").open("xb") as handle:
        handle.write(retention.json_bytes(result) + b"\n")
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
                      "source_row_counts": result["source_row_counts"], "source_role_parent_counts": result["source_role_parent_counts"],
                      "maximum_tokens": result["distributions"]["all_open"]["nonpadding_tokens"]["max"],
                      "training_nonpadding_tokens_per_pass": result["training_role_nonpadding_tokens_per_pass"],
                      "torch_imported": result["torch_imported"], "e1_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
