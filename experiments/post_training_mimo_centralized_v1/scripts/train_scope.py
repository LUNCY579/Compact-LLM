"""Stream actual TRAIN source-to-array joins for the registered CE512 interface.

This is byte/schema/reference compatibility, not new tokenization, native model
evidence, language grading or training authorization. No tokenizer preparation
module, model framework, checkpoint weights or closed-bank payload is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import time
from types import SimpleNamespace


MANIFEST_RELATIVE = "manifests/E1_CE512_TRAIN_SCOPE_V1.json"
MANIFEST_SHA256 = "951f8278bfe107bad23c927c87db64c75d9b47a83eab0ba3252cad758add3e70"
PREFIX = "experiments/post_training_mimo_centralized_v1/"
RUNTIME_SCOPE = "UNPADDED_SINGLE_SEQUENCE_CE512_V1"
E1_CONFIG = PREFIX + "configs/E1_PLAN_V1.json"
RUNTIME_CONFIG = PREFIX + "configs/E0_LAPTOP_STREAM_UNPADDED_CE512.json"
CONSUMER = {"microbatch": 1, "maximum_sequence_tokens": 512, "maximum_completion_tokens": 128,
            "use_cache": False, "attention_mask": "validateallones thenomit", "logits_to_keep": 0,
            "packing": False, "custom_positions": False,
            "loss": "mean completion-token CE per sequence; then mean of20sequences",
            "padding_tokens": 0, "completion_eos_id": 151645, "target_truncation": False}
LIMITS = {"metadata_bytes": 2097152, "line_bytes": 65536, "RSS_bytes": 134217728, "wall_seconds": 600}
COMPONENTS = ("acquisition", "deterministic_replay", "language_replay")
CODE_PATHS = {name: PREFIX + "scripts/" + name + ".py" for name in ("e1_data", "bridge_data", "retention_data", "laptop_profile")}
FILE_INVENTORY = {E1_CONFIG, RUNTIME_CONFIG, PREFIX + "manifests/E1_CANDIDATE_DOSE.json", "experiments/__init__.py",
    *(PREFIX + "receipts/" + name + ".json" for name in ("PRIMARY_STUDENT_RECEIPT", "TOKENIZER_RECEIPT", "BRIDGE_TOKENIZATION_RECEIPT",
        "RETENTION_TOKENIZATION_RECEIPT", "LANGUAGE_TRAIN_ARRAYS_RECEIPT", "LANGUAGE_CURATED_TOKENIZATION_RECEIPT")),
    *CODE_PATHS.values(), PREFIX + "scripts/qualify_native.py",
    *(PREFIX + "datasets/bridge_v1/" + name for name in ("DATA_RECEIPT.json", "REGISTRATION.json", "TRAIN.jsonl")),
    "artifacts/mimo_central/bridge_tokenized_v1/TRAIN.tokenized.jsonl",
    *("artifacts/mimo_central/retention_v1/" + name for name in ("DATA_RECEIPT.json", "REGISTRATION.json", "TRAIN.jsonl")),
    "artifacts/mimo_central/retention_tokenized_v1/TRAIN.tokenized.jsonl",
    *("artifacts/mimo_central/language_curated_v1/" + name for name in ("DATA_RECEIPT.json", "REGISTRATION.json", "LINEAGE.json", "TRAIN_CANDIDATES.jsonl")),
    "artifacts/mimo_central/language_train_arrays_v1/TRAIN.tokenized.jsonl"}
FORBIDDEN_ARRAY_FIELDS = {"packing", "packed", "cu_seqlens", "position_ids", "past_key_values", "cache_position", "inputs_embeds"}


class ScopeError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(condition, code, detail):
    if not condition:
        raise ScopeError(code, detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(actual, expected):
    return canonical(actual) == canonical(expected)


def parse(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "DUPLICATE_JSON_KEY", name)
            result[name] = value
        return result
    def invalid(value):
        raise ScopeError("NONFINITE_JSON", value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def safe_path(root, relative):
    require(type(relative) is str and relative and not PureWindowsPath(relative).drive,
            "PATH_ESCAPE", "relative path required")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "PATH_ESCAPE", relative)
    require(not any(re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed)([^a-z]|$)", part.lower()) for part in path.parts),
            "CLOSED_PATH", relative)
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), "PATH_ESCAPE", relative)
    require(not any(re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed)([^a-z]|$)", part.lower())
                    for part in resolved.relative_to(root.resolve()).parts), "CLOSED_PATH", relative)
    return resolved


def rss_bytes():
    """Use an OS high-water observation so transient allocation remains visible."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        record = Counters(); record.cb = ctypes.sizeof(record)
        require(bool(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(record), record.cb)),
                "RSS_OBSERVATION_FAILED", "Windows working-set query failed")
        return max(record.WorkingSetSize, record.PeakWorkingSetSize)
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.peak_rss_bytes = 0
    def check(self):
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes())
        require(self.peak_rss_bytes <= LIMITS["RSS_bytes"], "CPU_RSS_LIMIT", "128 MiB CPU compatibility ceiling")
        require(time.monotonic() - self.started <= LIMITS["wall_seconds"], "CPU_WALL_LIMIT", "600-second compatibility ceiling")


def identity(path, budget):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block); budget.check()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "FILE_CHANGED", str(path))
    return {"sha256": h.hexdigest(), "bytes": after.st_size}


def read_metadata(path):
    require(path.stat().st_size <= LIMITS["metadata_bytes"], "METADATA_SIZE_LIMIT", str(path))
    return parse(path.read_bytes())


def production_snapshot(root, manifest_path):
    """Require the producer and registration to be exactly in the current HEAD."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip()
    require(re.fullmatch("[0-9a-f]{40}", head), "PRODUCER_HEAD", "full committed source identity required")
    script = Path(__file__).resolve()
    require(script.is_relative_to(root) and script.relative_to(root).as_posix() == PREFIX + "scripts/train_scope.py",
            "PRODUCER_PATH", "registered producer source path required")
    observed = {}
    for path in (script, manifest_path):
        require(path.stat().st_size <= LIMITS["metadata_bytes"], "METADATA_SIZE_LIMIT", str(path))
        relative, raw = path.relative_to(root).as_posix(), path.read_bytes()
        committed = subprocess.run(["git", "show", head + ":" + relative], cwd=root, capture_output=True, check=False, timeout=15)
        require(committed.returncode == 0, "PRODUCER_SOURCE_UNTRACKED", relative)
        require(committed.stdout == raw, "PRODUCER_SOURCE_DIRTY", relative)
        observed[relative] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {"code_sha": head, "files": observed}


def recheck_production(root, snapshot):
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip()
    require(current == snapshot["code_sha"], "PRODUCER_HEAD_CHANGED", "HEAD changed during proof")
    for relative, expected in snapshot["files"].items():
        path = safe_path(root, relative); raw = path.read_bytes()
        require({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == expected,
                "PRODUCER_SOURCE_CHANGED", relative)


def validate_manifest(manifest):
    fixed = {"schema_version": 1, "status": "FROZEN_CE512_TRAIN_COMPATIBILITY_DATA_ONLY", "runtime_scope": RUNTIME_SCOPE,
             "runtime_config": RUNTIME_CONFIG, "historical_binding_config": E1_CONFIG,
             "model_weight_sha256": "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7",
             "consumer": CONSUMER, "limits": LIMITS,
             "expected_totals": {"rows": 5120, "nonpadding_tokens": 680532, "supervised_tokens": 63546, "maximum_sequence_tokens": 398},
             "native_qualified": False, "training_authorized": False, "e1_authorized": False,
             "CHECK_FINAL_access": False, "native_executor": False}
    require(type(manifest) is dict and set(manifest) == set(fixed) | {"recorded_at_utc", "files", "components", "output_claim", "language_origin_scope"},
            "MANIFEST_FIELDS", "registered manifest fields required")
    for name, expected in fixed.items():
        require(same(manifest[name], expected), "MANIFEST_CONTRACT", name)
    require(type(manifest["files"]) is dict and set(manifest["files"]) == FILE_INVENTORY, "FILE_INVENTORY", "exact 28-file graph required")
    for relative, pin in manifest["files"].items():
        require(type(pin) is dict and set(pin) == {"sha256", "bytes"} and type(pin["sha256"]) is str
                and re.fullmatch("[0-9a-f]{64}", pin["sha256"]) and type(pin["bytes"]) is int and pin["bytes"] > 0,
                "FILE_PIN", relative)
    require(type(manifest["components"]) is dict and set(manifest["components"]) == set(COMPONENTS), "COMPONENT_INVENTORY", "three components required")
    for component, count in zip(COMPONENTS, (4096, 768, 256)):
        spec = manifest["components"][component]
        require(type(spec) is dict and type(spec.get("count")) is int and spec["count"] == count, "COMPONENT_QUOTA", component)
        for field in ("source", "arrays", "token_receipt", "source_receipt", "registration"):
            require(spec.get(field) in FILE_INVENTORY, "COMPONENT_PATH", component + ":" + field)
    require(manifest["files"][RUNTIME_CONFIG]["sha256"] == "42a32cfa2fe4dc1250a288bacec2a60d150c5498d8dced9bba23b0c5037f5929"
            and manifest["files"][E1_CONFIG]["sha256"] == "1cbe40eebff71dfdf39b21207fab7952b78af06dd4f89b3047db585e0c94f5a2",
            "CONFIG_IDENTITY", "frozen runtime/historical bindings differ")


def load_verified_module(root, relative, pin):
    """Execute only already byte-pinned stdlib helpers, under a fresh private name."""
    require(relative in set(CODE_PATHS.values()), "HELPER_ALLOWLIST", relative)
    path = safe_path(root, relative)
    raw = path.read_bytes()
    require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "HELPER_DRIFT", relative)
    name = "_ce512_scope_" + path.stem + "_" + pin["sha256"][:12] + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    cached = sys.modules.get(name)
    if cached is not None:
        require(getattr(cached, "_ce512_source_bytes", None) == pin["sha256"] and Path(cached.__file__).resolve() == path,
                "HELPER_CACHE_DRIFT", relative)
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Compile the exact bytes just checked, not a later reopening or stale pyc.
        exec(compile(raw, str(path), "exec"), module.__dict__)
        module._ce512_source_bytes = pin["sha256"]
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def jsonl_rows(path, budget):
    with path.open("rb") as stream:
        line_number, offset = 0, 0
        while raw := stream.readline(LIMITS["line_bytes"] + 1):
            require(len(raw) <= LIMITS["line_bytes"], "LINE_SIZE_LIMIT", str(path))
            require(raw.strip(), "BLANK_JSONL_LINE", str(path))
            row = parse(raw)
            require(type(row) is dict, "ROW_SCHEMA", str(path))
            line_number += 1
            if line_number % 32 == 0: budget.check()
            yield line_number, offset, raw, row
            offset += len(raw)


def source_index(path, component, count, budget):
    result = {}
    for line, offset, raw, row in jsonl_rows(path, budget):
        row_id = row.get("id" if component == "language_replay" else "row_id")
        require(type(row_id) is str and row_id and row_id not in result, "SOURCE_IDENTITY", "missing/duplicate source ID")
        result[row_id] = (line, offset, len(raw), hashlib.sha256(raw).hexdigest())
        require(len(result) <= count, "SOURCE_QUOTA", component)
    require(len(result) == count, "SOURCE_QUOTA", component)
    return result


def validate_source_join(source, array, component, spec, verifier, e1_data, *, lineage=None, profile=None):
    """Validate one actual source body, target and stored completion-mask chain."""
    require(type(source) is dict and type(array) is dict and component in COMPONENTS, "ROW_SCHEMA", component)
    require(not FORBIDDEN_ARRAY_FIELDS.intersection(array), "UNSUPPORTED_FORWARD_FIELDS", "packing/cache/custom inputs")
    e1_data.validate_array(array, component)
    require(array["nonpadding_tokens"] <= 512, "FORWARD_SCOPE", "sequence exceeds CE512")
    require(array["source_row_sha256"] == digest(source), "SOURCE_BODY_JOIN", "canonical source body differs")
    if component != "language_replay":
        for field in ("row_id", "parent_id", "family", "variant_id", "role"):
            require(same(source.get(field), array.get(field)), "SOURCE_IDENTITY", field)
        require(source["role"] == "TRAIN" and source["training_role_eligible"] is True
                and source["source_revision"] == spec["source_revision"], "SOURCE_TRAIN_ROLE", component)
        parent = verifier.allocate_parent(source["spec"], source["generator_seed"], source["candidate_index"])
        require(parent.role == "TRAIN" and parent.parent_id == source["parent_id"]
                and parent.canonical_parent_sha256 == source["canonical_parent_sha256"]
                and parent.split_sha256 == source["split_sha256"] and same(parse(parent.canonical_json), source["spec"]),
                "SOURCE_CANONICAL_PARENT", component)
        if component == "acquisition":
            require(source["family"] == source["spec"]["family"] and source["structure_id"] == parent.structure_id,
                    "SOURCE_CANONICAL_PARENT", "acquisition family/structure differs")
        require(verifier.render_prompt(parent, source["variant_id"]) == source["prompt"], "SOURCE_RENDER", "prompt differs from canonical parent")
        answer = verifier.reference_answer(parent, source["variant_id"]) if component == "deterministic_replay" else verifier.reference_answer(parent)
        require(source["target"] == canonical({"answer": answer}).decode("utf-8"), "SOURCE_TARGET", "target differs from deterministic reference")
        correct = verifier.verify_answer(parent, source["target"], source["variant_id"]) if component == "deterministic_replay" else verifier.verify_answer(parent, source["target"])
        require(correct is True, "SOURCE_VERIFIER", "reference rejected")
        prompt_hash, target_hash = hashlib.sha256(source["prompt"].encode()).hexdigest(), hashlib.sha256(source["target"].encode()).hexdigest()
        require(source["prompt_sha256"] == array["source_prompt_sha256"] == prompt_hash
                and source["target_sha256"] == array["source_target_sha256"] == target_hash, "SOURCE_TEXT_HASH", component)
        require(source["normalized_prompt_sha256"] == hashlib.sha256(" ".join(source["prompt"].split()).encode()).hexdigest(),
                "SOURCE_TEXT_HASH", "normalized prompt differs")
        require(array["template_kwargs"] == {"add_generation_prompt": True, "enable_thinking": False}
                and array["token_hash_encoding"] == "UTF8 canonical compact JSON integer list; SHA256", "ARRAY_TEMPLATE_CONTRACT", component)
        if component == "deterministic_replay":
            require(source["stratum"] == parent.stratum and source["domain"] == source["family"] == source["spec"]["domain"],
                    "SOURCE_CANONICAL_PARENT", "retention strata/domain differ")
            for field in ("stratum", "domain", "canonical_parent_sha256", "split_sha256", "source_revision"):
                require(same(source[field], array[field]), "SOURCE_IDENTITY", field)
        prior, target = 0, source["target"]
    else:
        require(set(source) == {"id", "system_policy_version", "messages", "metadata"} and source["id"] == array["row_id"], "LANGUAGE_SCHEMA", "source identity/schema")
        metadata, messages = source["metadata"], source["messages"]
        require(type(metadata) is dict and metadata.get("split") == "train", "LANGUAGE_TRAIN_ROLE", "only TRAIN origins")
        require(type(messages) is list and len(messages) >= 2 and all(type(message) is dict and set(message) == {"role", "content"}
                    and type(message["content"]) is str and message["content"] for message in messages), "LANGUAGE_MESSAGES", "invalid conversation")
        start = 1 if messages[0]["role"] == "system" else 0
        require(all(message["role"] == ("user" if index % 2 == 0 else "assistant") for index, message in enumerate(messages[start:]))
                and messages[-1]["role"] == "assistant" and messages[-2]["role"] == "user", "LANGUAGE_MESSAGES", "final reply/role order")
        cluster = "v2.0:" + metadata["base_template_family"]
        require(type(lineage) is dict and type(profile) is dict and lineage["row_id"] == profile["row_id"] == source["id"]
                and lineage["cluster"] == profile["parent_cluster"] == array["parent_id"] == cluster
                and lineage["reply_class"] == profile["reply_class"] == array["family"] == metadata["reply_class"]
                and lineage["source_record_modified"] is False and lineage["training_eligibility"] is False, "LANGUAGE_LINEAGE", "class/cluster/identity differs")
        original = deepcopy(source)
        require(metadata.get("preservation_replay_for") == "rineya_v3" and metadata.get("preservation_source_version") == "v2.0",
                "LANGUAGE_LINEAGE", "added preservation metadata values differ")
        for field in ("preservation_replay_for", "preservation_source_version"):
            require(field in original["metadata"], "LANGUAGE_LINEAGE", "missing inherited metadata")
            del original["metadata"][field]
        require(digest(original) == lineage["original_v2_canonical_sha256"], "LANGUAGE_ORIGINAL_BODY", "stripped original body differs")
        prior = sum(message["role"] == "assistant" for message in messages[:-1])
        require(type(array["prior_assistant_turns"]) is int and array["prior_assistant_turns"] == profile["prior_assistant_turns"] == prior
                and array["prior_turns_supervised"] is False and profile["all_prior_turns_masked"] is True, "LANGUAGE_PRIOR_SUPERVISION", "prior turns are not targets")
        for field in ("prompt_tokens", "completion_tokens", "nonpadding_tokens"):
            require(same(array[field], profile[field]), "LANGUAGE_PROFILE", field)
        require(array["tokenized_prompt_sha256"] == profile["prompt_token_ids_sha256"]
                and array["completion_token_ids_sha256"] == profile["completion_token_ids_sha256"]
                and profile["supervised_positions"] == array["completion_tokens"], "LANGUAGE_PROFILE", "measured token hash/mask linkage")
        target = messages[-1]["content"]
    require(array["training_eligibility"] is False and type(array["target_json_tokens"]) is int
            and array["target_json_tokens"] == array["completion_tokens"] - 1, "ARRAY_NO_GRANT", "target length/eligibility differs")
    return {"row_id": array["row_id"], "parent_id": array["parent_id"], "family": array["family"],
            "source_canonical_sha256": digest(source), "target_text_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "input_token_ids_sha256": array["input_token_ids_sha256"], "prompt_token_ids_sha256": array["tokenized_prompt_sha256"],
            "completion_token_ids_sha256": array["completion_token_ids_sha256"], "prompt_tokens": array["prompt_tokens"],
            "completion_tokens": array["completion_tokens"], "nonpadding_tokens": array["nonpadding_tokens"], "prior_assistant_turns": prior}


class ShapeOnly:
    def __init__(self, length, values=None):
        self.shape = (1, length); self.values = values
    def __eq__(self, value):
        return SimpleNamespace(all=lambda: all(item == value for item in self.values))


def validate_consumer(array, consumer, laptop):
    require(same(consumer, CONSUMER), "CONSUMER_CONTRACT", "frozen single-sequence interface required")
    result = laptop.prepare_unpadded_forward(ShapeOnly(len(array["input_ids"])), {
        "use_cache": consumer["use_cache"], "logits_to_keep": consumer["logits_to_keep"],
        "attention_mask": ShapeOnly(len(array["attention_mask"]), array["attention_mask"])}, runtime_scope=RUNTIME_SCOPE)
    require(same(result, {"use_cache": False, "logits_to_keep": 0}), "CONSUMER_CONTRACT", "validated mask must be omitted")


def receipt_links(manifest, metadata):
    files = manifest["files"]
    for component, spec in manifest["components"].items():
        source, tokens = metadata[spec["source_receipt"]], metadata[spec["token_receipt"]]
        require(same(source["files"][Path(spec["source"]).name], files[spec["source"]])
                and same(source["files"]["REGISTRATION.json"], files[spec["registration"]])
                and same(tokens["files"]["TRAIN.tokenized.jsonl"], files[spec["arrays"]]), "RECEIPT_GRAPH", component)
        if component != "language_replay":
            require(same(tokens["source_open_files_verified"]["TRAIN.jsonl"], files[spec["source"]])
                    and tokens["source_data_receipt_sha256"] == files[spec["source_receipt"]]["sha256"]
                    and tokens["source_registration_sha256"] == files[spec["registration"]]["sha256"]
                    and source["source_revision"] == spec["source_revision"], "RECEIPT_GRAPH", component)
        else:
            profile = metadata[spec["profile"]]
            require(same(tokens["source_primary_file"], files[spec["source"]])
                    and same(profile["source_primary_file"], files[spec["source"]])
                    and tokens["profile_sha256"] == files[spec["profile"]]["sha256"]
                    and profile["candidate_receipt_sha256"] == files[spec["source_receipt"]]["sha256"]
                    and same(source["files"]["LINEAGE.json"], files[spec["lineage"]]), "RECEIPT_GRAPH", component)


def validate_train_scope(package_root, *, manifest_path=None, output_directory=None):
    package = Path(package_root).resolve(); root = package.parents[1]
    budget = Budget()
    report = {"status": "BLOCKED", "reasons": [], "runtime_scope": RUNTIME_SCOPE, "native_qualified": False,
              "native_launch_authorized": False, "training_authorized": False, "e1_authorized": False,
              "native_executor": False, "CHECK_FINAL_access": False, "model_calls": 0, "training_updates": 0,
              "resource_enforcement": "cooperative checks with OS high-water RSS observations; historical binder checked on return; no hard process containment"}
    try:
        expected_manifest = package / MANIFEST_RELATIVE
        path = Path(manifest_path).resolve() if manifest_path is not None else expected_manifest
        require(path == expected_manifest.resolve(), "MANIFEST_PATH", "only the registered manifest may be opened")
        budget.check()
        require(path.stat().st_size <= LIMITS["metadata_bytes"], "METADATA_SIZE_LIMIT", "manifest")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, "MANIFEST_IDENTITY", "frozen manifest bytes changed")
        manifest = parse(raw); validate_manifest(manifest)
        production = production_snapshot(root, path) if output_directory is not None else None
        bound, metadata = {}, {}
        for relative, pin in manifest["files"].items():
            target = safe_path(root, relative)
            require(identity(target, budget) == pin, "FILE_PIN_DRIFT", relative)
            bound[relative] = pin
            if target.suffix == ".json": metadata[relative] = read_metadata(target)
        receipt_links(manifest, metadata)
        modules = {name: load_verified_module(root, relative, bound[relative]) for name, relative in CODE_PATHS.items()}
        data = modules["e1_data"]; historical = data.bind(package, metadata[E1_CONFIG]); budget.check()
        require(historical["native_qualified"] is False and historical["training_authorized"] is False,
                "HISTORICAL_NO_GRANT", "historical binder cannot authorize training")
        language = manifest["components"]["language_replay"]
        lineage_rows = metadata[language["lineage"]]; profile_rows = metadata[language["profile"]]["rows"]
        require(type(lineage_rows) is list and len(lineage_rows) == 256 and len(profile_rows) == 256,
                "LANGUAGE_QUOTA", "256 curated rows required")
        lineage = {row["row_id"]: row for row in lineage_rows}; profile = {row["row_id"]: row for row in profile_rows}
        require(len(lineage) == len(profile) == 256 and set(lineage) == set(profile), "LANGUAGE_IDENTITY", "complete unique lineage/profile")
        summaries, total, total_target, row_count, prior_count = {}, 0, 0, 0, 0
        joined_hash = hashlib.sha256(); all_ids = set()
        for component in COMPONENTS:
            spec = manifest["components"][component]
            source_path, arrays_path = safe_path(root, spec["source"]), safe_path(root, spec["arrays"])
            index = source_index(source_path, component, spec["count"], budget)
            used, component_hash, counts = set(), hashlib.sha256(), Counter()
            nonpadding, supervised, maximum, prior = 0, 0, 0, 0
            verifier = modules["bridge_data" if component == "acquisition" else "retention_data"] if component != "language_replay" else None
            with source_path.open("rb") as source_stream:
                for line, offset, array_raw, array in jsonl_rows(arrays_path, budget):
                    row_id = array.get("row_id")
                    require(row_id in index and row_id not in used and row_id not in all_ids, "ROW_ID_JOIN", "unmatched/repeated TRAIN identity")
                    source_line, source_offset, length, source_raw_sha = index[row_id]
                    source_stream.seek(source_offset); source_raw = source_stream.read(length)
                    require(hashlib.sha256(source_raw).hexdigest() == source_raw_sha, "SOURCE_CHANGED", row_id)
                    source = parse(source_raw)
                    row = validate_source_join(source, array, component, spec, verifier, data,
                                               lineage=lineage.get(row_id), profile=profile.get(row_id))
                    location = historical["locations"][component + ":" + row_id]
                    require(location["line"] == line and location["byte_offset"] == offset
                            and location["raw_line_sha256"] == hashlib.sha256(array_raw).hexdigest()
                            and location["source_row_sha256"] == row["source_canonical_sha256"], "HISTORICAL_ROW_JOIN", row_id)
                    validate_consumer(array, manifest["consumer"], modules["laptop_profile"])
                    row.update(component=component, source_file=spec["source"], source_line=source_line,
                               source_byte_offset=source_offset, source_raw_line_sha256=source_raw_sha,
                               array_file=spec["arrays"], array_line=line, array_byte_offset=offset,
                               array_raw_line_sha256=location["raw_line_sha256"])
                    commitment = canonical(row) + b"\n"; component_hash.update(commitment); joined_hash.update(commitment)
                    used.add(row_id); all_ids.add(row_id); counts[array["family"]] += 1
                    nonpadding += array["nonpadding_tokens"]; supervised += array["completion_tokens"]
                    maximum = max(maximum, array["nonpadding_tokens"]); prior += row["prior_assistant_turns"]
            require(used == set(index) and len(used) == spec["count"] and maximum == spec["expected_max_tokens"]
                    and nonpadding == spec["nonpadding_tokens"] and supervised == spec["supervised_tokens"], "COMPONENT_TOTALS", component)
            if component == "language_replay": require(prior == spec["prior_assistant_turns"] == 81, "LANGUAGE_PRIOR_TOTAL", "all prior turns masked")
            summaries[component] = {"rows": len(used), "nonpadding_tokens": nonpadding, "supervised_tokens": supervised,
                "maximum_sequence_tokens": maximum, "prior_assistant_turns_masked": prior,
                "family_counts": dict(counts), "source_array_join_sha256": component_hash.hexdigest()}
            row_count += len(used); total += nonpadding; total_target += supervised; prior_count += prior
        totals = {"rows": row_count, "nonpadding_tokens": total, "supervised_tokens": total_target,
                  "maximum_sequence_tokens": max(value["maximum_sequence_tokens"] for value in summaries.values())}
        require(same(totals, manifest["expected_totals"]), "TOTAL_DOSE", "full registered dose required")
        for relative, pin in bound.items():
            require(identity(safe_path(root, relative), budget) == pin, "FILE_CHANGED", relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST_SHA256, "MANIFEST_CHANGED", "registration changed during binding")
        budget.check()
        report.update(status="PASS_CE512_TRAIN_COMPATIBILITY_DATA_ONLY", manifest_sha256=MANIFEST_SHA256,
                      files=bound, components=summaries, totals=totals, source_array_join_sha256=joined_hash.hexdigest(),
                      historical_data_sha256=historical["data_sha256"], language_prior_assistant_turns_masked=prior_count,
                      consumer=CONSUMER, claim=manifest["output_claim"], language_origin_scope=manifest["language_origin_scope"],
                      new_tokenization_performed=False, original_preservation_corpus_reaudited=False)
        if output_directory is not None:
            output = Path(output_directory).resolve()
            require(not output.exists() and output.is_relative_to((root / "artifacts/mimo_central").resolve()),
                    "OUTPUT_PATH", "new artifact directory required")
            recheck_production(root, production)
            budget.check()
            output.mkdir(parents=True)
            report["producer_code_sha"] = production["code_sha"]
            report["producer_source_files"] = production["files"]
            report["script_sha256"] = production["files"][PREFIX + "scripts/train_scope.py"]["sha256"]
            report["dirty_producer_source"] = False
            report["wall_seconds"] = time.monotonic() - budget.started
            report["peak_observed_RSS_bytes"] = budget.peak_rss_bytes
            with (output / "TRAIN_SCOPE_RECEIPT.json").open("xb") as stream: stream.write(canonical(report) + b"\n")
            report["output_receipt"] = str(output / "TRAIN_SCOPE_RECEIPT.json")
    except (ScopeError, OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, RecursionError, subprocess.SubprocessError) as error:
        report["status"] = "BLOCKED"
        report["reasons"].append({"code": getattr(error, "code", "TRAIN_SCOPE_INVALID"), "detail": str(error)[:400]})
    report["wall_seconds"] = time.monotonic() - budget.started
    report["peak_observed_RSS_bytes"] = budget.peak_rss_bytes
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_train_scope(args.experiment_root, output_directory=args.output)
    print(json.dumps({name: result.get(name) for name in ("status", "reasons", "totals", "source_array_join_sha256",
        "peak_observed_RSS_bytes", "wall_seconds", "native_qualified", "training_authorized", "e1_authorized", "output_receipt")}, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
