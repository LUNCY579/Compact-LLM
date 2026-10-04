"""Staged CPU-only long prompt adapter; no model/framework/scorer imports.

Policy validation/retokenization completes for the entire ordered source before
output creation. Rows are streamed on a second checked pass; a receipt is only
published after complete source, tokenizer, row-order and output verification.
Partial output is retained as failed CPU preparation and grants no admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

WORK = Path(__file__).resolve().parent
PACKAGE = WORK.parents[1]
ROOT = PACKAGE.parents[1]
SOURCE = PACKAGE / "evidence" / "long_context_guards_root1_20261004"
VERSION = "synth-staged-streaming-long-inputs-1"
ROW_FIELDS = {"row_id", "messages", "input_ids", "attention_mask", "prompt_tokens", "prompt_token_sha256"}
PROMPT_ENVELOPES = {992: 1024, 2016: 2048, 4064: 4096}
RESERVE, SOURCE_ROWS = 32, 2304
MAX_LINE, MAX_METADATA, MAX_PACKET, MAX_OUTPUT = 1024**2, 4 * 1024**2, 64 * 1024**2, 128 * 1024**2
MAX_RSS = 512 * 1024**2
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
FRAMEWORKS = {"torch", "transformers", "tensorflow", "jax", "flax"}
SOURCE_PINS = {
    "policy_inputs.jsonl": {"bytes": 57864922, "sha256": "0a4c590ea0c886571a936e19cebebc4c00888ce6ae310c4157f727aad26e39b5"},
    "receipt.json": {"bytes": 3345, "sha256": "9b10166fdd3a7a46a0deac1e481956da9038cae1e87d563555035b6a70774c4b"},
    "catalog.json": {"bytes": 1651621, "sha256": "0fa2743f8236ba73c2926697591cd02616164930d928ae7bbc92a268e7edfa82"}}
CPU_TOKENIZER_PIN = "268d47037581e7d022cb4d4d53dd9b7e4454909c74b910572d2ba153e463b70f"


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()
def hash_value(value): return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def no_frameworks():
    require(not any(name.split(".", 1)[0] in FRAMEWORKS for name in sys.modules), "CPU process cannot contain model frameworks")


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def strict_json(raw):
    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON numeric exponent")
        return value
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_float=finite_float,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("malformed bounded UTF-8 JSON") from error


def checked_root(path, *, exists=True):
    requested = Path(path).absolute()
    for item in (requested, *requested.parents):
        require(not item.is_symlink(), "symlink/reparse paths forbidden")
        if not item.exists(): continue
        details = item.lstat()
        require(not item.is_symlink() and not (getattr(details, "st_file_attributes", 0) & 0x400),
                "symlink/reparse paths forbidden")
    require(not exists or requested.is_dir(), "existing owned directory required")
    return requested


def safe_path(root, relative, *, exists=True):
    require(type(relative) is str and relative and "\\" not in relative and ":" not in relative, "relative owned path required")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") and re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts), "path traversal or malformed leaf")
    base = checked_root(root)
    requested = base.joinpath(*parts)
    checked_root(requested.parent, exists=exists)
    require(not requested.is_symlink(), "symlink/reparse file forbidden")
    if requested.exists():
        details = requested.lstat()
        require(not requested.is_symlink() and not (getattr(details, "st_file_attributes", 0) & 0x400), "symlink/reparse file forbidden")
    require(requested.resolve().is_relative_to(base.resolve()), "owned path escaped root")
    require(not exists or requested.is_file(), "existing owned file required")
    return requested


def file_binding(path, maximum=MAX_PACKET):
    value, size = hashlib.sha256(), 0
    with Path(path).open("rb", buffering=0) as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            size += len(chunk)
            require(size <= maximum, "bounded file byte ceiling")
            value.update(chunk)
    return {"sha256": value.hexdigest(), "bytes": size}


def metadata(path):
    with Path(path).open("rb") as stream: raw = stream.read(MAX_METADATA + 1)
    require(len(raw) <= MAX_METADATA, "metadata byte ceiling")
    value = strict_json(raw)
    require(type(value) is dict, "metadata object required")
    return value


def lines(path):
    """Bound each binary read before UTF-8/JSON parsing; no whole packet load."""
    with Path(path).open("rb", buffering=0) as stream:
        count, total = 0, 0
        while True:
            raw = stream.readline(MAX_LINE + 1)
            if not raw: break
            count += 1; total += len(raw)
            require(count <= SOURCE_ROWS and total <= MAX_PACKET and len(raw) <= MAX_LINE and raw.endswith(b"\n"),
                    "row/packet/line bound or unterminated record")
            require(raw.strip(), "blank JSONL line forbidden")
            yield count - 1, strict_json(raw), hashlib.sha256(raw).hexdigest()


def validate_row(row):
    require(type(row) is dict and set(row) == ROW_FIELDS, "prompt-only row allowlist; private/unknown fields forbidden")
    require(hash_value(row["row_id"]) and hash_value(row["prompt_token_sha256"]), "typed source/token hashes")
    messages, ids, mask, length = row["messages"], row["input_ids"], row["attention_mask"], row["prompt_tokens"]
    require(type(messages) is list and len(messages) == 1 and type(messages[0]) is dict and
            set(messages[0]) == {"role", "content"} and messages[0]["role"] == "user" and
            type(messages[0]["content"]) is str and 0 < len(messages[0]["content"].encode("utf-8")) <= MAX_LINE,
            "single prompt-only user message")
    require(type(length) is int and length in PROMPT_ENVELOPES and type(ids) is list and len(ids) == length and
            all(type(token) is int and 0 <= token < 151936 for token in ids) and 151643 not in ids,
            "exact 992/2016/4064 prompt envelope; no padding/truncation")
    require(type(mask) is list and len(mask) == length and all(type(token) is int and token == 1 for token in mask),
            "all-one exact attention mask")
    require(row["prompt_token_sha256"] == digest(ids), "prompt token checksum mismatch")
    require(PROMPT_ENVELOPES[length] - length == RESERVE, "exact32 generation reservation")
    return True


class LocalPromptTokenizer:
    """Prompt-only reuse of cpu_tokenizer.py's pinned tokenizer/template profile.

    Uses the same local tokenizers JSON, immutable Jinja environment, filters,
    exact chat kwargs, EOS/PAD identities and no automatic special tokens. It
    does not import cpu_tokenizer's supervised verifier or encode_example path.
    No model config/weights, oracle sidecar or public task solver is opened.
    """
    def __init__(self, binding):
        self._binding = deepcopy(binding)
        self.verify_binding()
        local = checked_root(binding["model_dir"])
        configuration = metadata(safe_path(local, "tokenizer_config.json"))
        text = configuration.get("chat_template")
        require(type(text) is str and hashlib.sha256(text.encode("utf-8")).hexdigest() == binding["chat_template_sha256"], "chat template binding drift")
        require(configuration.get("eos_token") == "<|im_end|>" and configuration.get("pad_token") == "<|endoftext|>" and
                configuration.get("add_bos_token") is False and configuration.get("clean_up_tokenization_spaces") is False,
                "tokenizer special token configuration drift")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self._engine = Tokenizer.from_file(str(safe_path(local, "tokenizer.json")))
        require(self._engine.padding is None and self._engine.truncation is None and
                self._engine.token_to_id("<|im_end|>") == 151645 and self._engine.token_to_id("<|endoftext|>") == 151643,
                "no padding/truncation and exact EOS/PAD required")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self._template = environment.from_string(text)
        no_frameworks()

    def verify_binding(self):
        no_frameworks()
        binding = self._binding
        require(binding["dependency_versions"] == {"jinja2": "3.1.6", "tokenizers": "0.22.2"} and
                binding["template_kwargs"] == TEMPLATE_KWARGS and binding["native_qualification"] is False and
                binding["training_admitted"] is False and binding["framework_modules_imported"] == [], "closed pinned CPU tokenizer profile")
        require(set(binding["assets"]) == {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}, "exact tokenizer asset inventory")
        directory = checked_root(binding["model_dir"])
        for filename, pin in binding["assets"].items():
            require(file_binding(safe_path(directory, filename), maximum=16 * 1024**2) == pin, "tokenizer asset binding drift")
        for filename, expected in binding["binding_sources"].items():
            path = Path(filename).absolute()
            require(path in ((PACKAGE / "PILOT_DECISIONS.json").absolute(),
                    (PACKAGE.parent / "post_training_mimo_centralized_v1/receipts/titan_static_v1/TOKENIZER_RECEIPT.json").absolute()),
                    "unregistered tokenizer binding source path")
            checked_root(path.parent)
            require(not path.is_symlink() and file_binding(path, MAX_METADATA)["sha256"] == expected, "CPU tokenizer source binding drift")
        for name, expected in binding["dependency_versions"].items():
            require(importlib.metadata.version(name) == expected, "CPU tokenizer dependency drift; no install/fallback")
        return deepcopy(binding)

    def encode_prompt(self, messages):
        no_frameworks()
        rendered = self._template.render(messages=deepcopy(messages), tools=None, **TEMPLATE_KWARGS)
        require(type(rendered) is str and rendered.endswith(ASSISTANT_PREFIX), "non-thinking assistant prefix drift")
        return self._engine.encode(rendered, add_special_tokens=False).ids


@dataclass(frozen=True)
class Admission:
    source: Path
    source_pins: dict
    receipt_sha256: str
    tokenizer_binding: dict
    row_ids: tuple
    line_hashes: tuple
    row_order_sha256: str
    length_counts: dict


def source_check(source, pins):
    root = checked_root(source)
    require(set(pins) == {"policy_inputs.jsonl", "receipt.json", "catalog.json"}, "exact public source pin allowlist")
    for filename, expected in pins.items():
        require(set(expected) == {"sha256", "bytes"} and hash_value(expected["sha256"]) and
                type(expected["bytes"]) is int and 0 < expected["bytes"] <= MAX_PACKET,
                "typed public source binding")
        require(file_binding(safe_path(root, filename)) == expected, "public source file checksum/size mismatch")
    return root


def validate_packet(source=SOURCE, *, tokenizer=None, _fixture_pins=None, resource_guard=lambda: None):
    """Validate all source rows and visible-message retokenization before outputs."""
    no_frameworks()
    require(_fixture_pins is not None or Path(source).absolute() == SOURCE.absolute(), "registered source path required")
    pins = deepcopy(SOURCE_PINS if _fixture_pins is None else _fixture_pins)
    source = source_check(source, pins)
    receipt = metadata(safe_path(source, "receipt.json"))
    require(receipt.get("constructed_rows") == SOURCE_ROWS and receipt.get("planned_rows") == SOURCE_ROWS and
            receipt.get("blocked_variant_rows") == 0 and receipt.get("blockers") == [] and receipt.get("all_worlds_retained") is True and
            receipt.get("policy_file_sha256") == pins["policy_inputs.jsonl"]["sha256"] and receipt.get("generation_reserve_tokens") == 32,
            "complete2304 source denominator/packet binding")
    for flag in ("native_long_context_profile_qualified", "source_correct_membership_enrolled",
                 "statistical_preservation_qualified", "confirmation_eligible", "native_resource_qualification_inferred"):
        require(receipt.get(flag) is False, "native/enrollment/statistical qualification must remain closed")
    if _fixture_pins is None:
        require(file_binding(safe_path(PACKAGE, "cpu_tokenizer.py"))["sha256"] == CPU_TOKENIZER_PIN, "pinned CPU tokenizer implementation drift")
    tokenizer = LocalPromptTokenizer(receipt["tokenizer_binding"]) if tokenizer is None else tokenizer
    binding = tokenizer.verify_binding()
    require(binding == receipt["tokenizer_binding"], "tokenizer/source receipt binding mismatch")
    seen, identities, hashes, counts = set(), [], [], Counter()
    for index, row, line_hash in lines(safe_path(source, "policy_inputs.jsonl")):
        resource_guard(); validate_row(row)
        require(row["row_id"] not in seen, "duplicate source row identity")
        seen.add(row["row_id"]); identities.append(row["row_id"]); hashes.append(line_hash)
        ids = tokenizer.encode_prompt(row["messages"])
        require(type(ids) is list and ids == row["input_ids"] and all(type(x) is int for x in ids), "visible prompt/token binding drift")
        counts[row["prompt_tokens"]] += 1
    require(len(identities) == SOURCE_ROWS and dict(counts) == {length: SOURCE_ROWS // 3 for length in PROMPT_ENVELOPES}, "complete2304 exact envelope inventory")
    require(tokenizer.verify_binding() == binding, "tokenizer drift during whole-packet validation")
    source_check(source, pins); no_frameworks()
    return Admission(source, pins, pins["receipt.json"]["sha256"], deepcopy(binding), tuple(identities), tuple(hashes), digest(identities), dict(counts)), tokenizer


def materialize(admission, tokenizer, output, *, resource_guard=lambda: None):
    """Publish only complete CPU arrays; an interrupted/failed leaf stays unadmitted."""
    no_frameworks()
    require(isinstance(admission, Admission), "validated whole-packet admission required")
    source_check(admission.source, admission.source_pins)
    require(tokenizer.verify_binding() == admission.tokenizer_binding, "tokenizer drift before output processing")
    checked_root(WORK)
    target = Path(output).absolute()
    require(target.parent == WORK and target.name not in ("", ".", "..") and not target.exists(), "new direct staging output leaf required")
    checked_root(target, exists=False)
    target.mkdir()
    output_file = safe_path(target, "prompt_arrays.jsonl", exists=False)
    started, count, byte_count = time.monotonic(), 0, 0
    output_hash = hashlib.sha256()
    try:
        with output_file.open("xb", buffering=0) as stream:
            for index, row, line_hash in lines(safe_path(admission.source, "policy_inputs.jsonl")):
                resource_guard(); validate_row(row)
                require(index < len(admission.row_ids) and row["row_id"] == admission.row_ids[index] and
                        line_hash == admission.line_hashes[index], "bound source row order/content drift during output")
                ids = tokenizer.encode_prompt(row["messages"])
                require(ids == row["input_ids"], "prompt token binding drift during output")
                record = {"row_id": row["row_id"], "source_line_index": index, "prompt_sha256": digest(row["messages"]),
                          "prompt_ids": list(ids), "prompt_tokens": len(ids), "attention_mask": list(row["attention_mask"]),
                          "prompt_ids_sha256": row["prompt_token_sha256"], "total_sequence_envelope": PROMPT_ENVELOPES[len(ids)],
                          "maximum_generated_tokens": 32, "scope": "CPU_LONG_PROMPT_ARRAYS_NATIVE_UNQUALIFIED",
                          "native_execution_authorized": False, "native_long_context_profile_qualified": False,
                          "source_correct_membership_enrolled": False, "confirmation_eligible": False}
                raw = (canonical(record) + "\n").encode("ascii")
                byte_count += len(raw)
                require(len(raw) <= MAX_LINE and byte_count <= MAX_OUTPUT and stream.write(raw) == len(raw), "bounded complete array write")
                output_hash.update(raw); count += 1
            os.fsync(stream.fileno())
        require(count == SOURCE_ROWS, "complete source denominator changed")
        source_check(admission.source, admission.source_pins)
        require(tokenizer.verify_binding() == admission.tokenizer_binding, "tokenizer drift during output")
        expected = {"bytes": byte_count, "sha256": output_hash.hexdigest()}
        require(file_binding(output_file, MAX_OUTPUT) == expected, "written arrays checksum mismatch")
        no_frameworks()
        receipt = {"schema_version": 1, "implementation": VERSION, "status": "CPU_LONG_PROMPT_ARRAYS_PREPARED_NATIVE_UNADMITTED",
                   "scope": "CPU_ONLY_STAGED_ADAPTER_NOT_GRADER_OR_MODEL_AUTHORITY", "source_pins": admission.source_pins,
                   "source_receipt_sha256": admission.receipt_sha256, "source_row_order_sha256": admission.row_order_sha256,
                   "tokenizer_binding": admission.tokenizer_binding, "rows": count, "arrays": expected,
                   "input_positions": sum(length * rows for length, rows in admission.length_counts.items()),
                   "prompt_token_counts": admission.length_counts, "total_sequence_envelopes": [1024, 2048, 4096],
                   "generation_reserve_tokens": 32, "rows_dropped": 0, "rows_truncated": 0, "padding": False, "packing": False,
                   "whole_packet_validated_before_output": True, "oracle_inputs_read": False, "model_or_GPU_calls": 0,
                   "native_execution_authorized": False, "native_long_context_profile_qualified": False,
                   "source_correct_membership_enrolled": False, "statistical_preservation_qualified": False,
                   "confirmation_eligible": False, "human_language_qualification_established": False,
                   "tokenizer_context_cap512_is_historical_helper_metadata_not_native_scope": True,
                   "adapter_source_sha256": file_binding(Path(__file__), MAX_METADATA)["sha256"],
                   "output_pass_wall_seconds": time.monotonic() - started}
        receipt["sha256"] = digest(receipt)
        raw = (canonical(receipt) + "\n").encode("ascii")
        with safe_path(target, "receipt.json", exists=False).open("xb", buffering=0) as stream:
            require(len(raw) <= MAX_METADATA and stream.write(raw) == len(raw), "complete bounded receipt write")
            os.fsync(stream.fileno())
        return receipt
    except BaseException as error:
        failure = {"status": "CPU_PREPARATION_FAILED_NO_ADMISSION", "rows_written": count,
                   "error": {"type": type(error).__name__, "message": str(error)},
                   "native_execution_authorized": False, "oracle_inputs_read": False, "model_or_GPU_calls": 0}
        with safe_path(target, "failure.json", exists=False).open("xb", buffering=0) as stream:
            raw = (canonical(failure) + "\n").encode("ascii")
            require(stream.write(raw) == len(raw), "complete failure receipt write"); os.fsync(stream.fileno())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import psutil
    peak = [0]
    def guard():
        details = psutil.Process().memory_info()
        peak[0] = max(peak[0], details.rss, getattr(details, "peak_wset", details.rss))
        require(peak[0] <= MAX_RSS, "CPU512MiB RSS/process lifetime peak bound")
    started = time.monotonic()
    admitted, tokenizer = validate_packet(resource_guard=guard)
    result = materialize(admitted, tokenizer, args.output_dir, resource_guard=guard)
    print(canonical({"status": result["status"], "rows": result["rows"], "CPU_peak_bytes": peak[0],
                     "whole_adapter_wall_seconds": time.monotonic() - started, "model_or_GPU_calls": 0}))


if __name__ == "__main__": main()
