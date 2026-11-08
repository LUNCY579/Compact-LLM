"""Bounded random access to the frozen E1 TRAIN arrays; no native admission.

The production factory binds the immutable historical schedule and data receipt,
then retains byte offsets and immutable schedule metadata rather than token rows.
It never imports tokenization preparation, model frameworks, or model weights.
The explicit fixture factory is data-only and cannot produce a production reader.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
from types import MappingProxyType


PREFIX = "experiments/post_training_mimo_centralized_v1/"
SCOPE_RELATIVE = PREFIX + "manifests/E1_CE512_TRAIN_SCOPE_V1.json"
SCOPE_SHA256 = "951f8278bfe107bad23c927c87db64c75d9b47a83eab0ba3252cad758add3e70"
PLAN_RELATIVE = "artifacts/mimo_central/e1_plan_seed101_v2/PLAN.json"
BINDING_RELATIVE = "artifacts/mimo_central/e1_plan_seed101_v2/DATA_BINDING.json"
RECEIPT_RELATIVE = PREFIX + "receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json"
HELPER_PINS = {
    "e1_data": {"sha256": "c91f170bf1f3230b62adfd496c7d6a6172de966b9bf21becc7bec16c90d2aeb7", "bytes": 10752},
    "train_scope": {"sha256": "7d24b694571a65855021ad5e68ff6c1c66a4e8c278077dc9f8d5241653fbc3ef", "bytes": 33290},
    "e1_schedule": {"sha256": "44051045d5738c9fb5da28c96afba0242b306506e60da9f8a936b8f78a5044eb", "bytes": 16412},
}
COMPONENTS = ("acquisition", "deterministic_replay", "language_replay")
EXPECTED_COUNTS = {"acquisition": 4096, "deterministic_replay": 768, "language_replay": 256}
EXPECTED_DOSES = {"acquisition": (414554, 49272), "deterministic_replay": (243336, 11814), "language_replay": (22642, 2460)}
LINE_BYTES = 65536
METADATA_FILE_BYTES = 8 * 1024 * 1024
COMBINED_METADATA_BYTES = 16 * 1024 * 1024
RETAINED_METADATA_BYTES = 32 * 1024 * 1024
FORBIDDEN_ARRAY_FIELDS = {"packing", "packed", "cu_seqlens", "position_ids", "past_key_values", "cache_position", "inputs_embeds"}
_CONSTRUCTION_TOKEN = object()
_WINDOWS_FILE_API = None


class ReaderError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(condition, code, detail):
    if not condition:
        raise ReaderError(code, detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY", key)
            result[key] = value
        return result
    def invalid(value):
        raise ReaderError("NONFINITE_JSON", value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def safe_path(root, relative):
    require(type(relative) is str and relative and not PureWindowsPath(relative).drive,
            "PATH_ESCAPE", "relative path required")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and ".." not in PureWindowsPath(relative).parts,
            "PATH_ESCAPE", relative)
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), "PATH_ESCAPE", relative)
    for part in (*path.parts, *resolved.relative_to(root).parts):
        require(not re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed)([^a-z]|$)", part.lower()), "CLOSED_PATH", relative)
    return resolved


def _windows_change_time(path):
    global _WINDOWS_FILE_API
    if _WINDOWS_FILE_API is None:
        import ctypes
        import msvcrt
        class BasicInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")] + [
                ("FileAttributes", ctypes.c_ulong)]
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        query.restype = ctypes.c_int
        _WINDOWS_FILE_API = (ctypes, msvcrt, BasicInfo, query)
    ctypes, msvcrt, BasicInfo, query = _WINDOWS_FILE_API
    record = BasicInfo()
    with path.open("rb") as stream:
        require(bool(query(msvcrt.get_osfhandle(stream.fileno()), 0, ctypes.byref(record), ctypes.sizeof(record))),
                "FILE_IDENTITY_QUERY", str(path))
    require(record.ChangeTime > 0, "FILE_IDENTITY_QUERY", "positive Windows ChangeTime required")
    return record.ChangeTime


def _stat(path):
    value = path.stat()
    require(path.is_file(), "NOT_A_FILE", str(path))
    # Windows st_ctime is creation time. NTFS ChangeTime also catches same-size
    # writes followed by restoration of LastWriteTime (os.utime).
    change = _windows_change_time(path) if os.name == "nt" else value.st_ctime_ns
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, change)


def _pin(pin):
    require(type(pin) is dict and set(pin) == {"sha256", "bytes"}
            and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
            and type(pin["bytes"]) is int and pin["bytes"] > 0, "FILE_PIN", "exact positive byte/hash pin required")


def _hash(path):
    before = _stat(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    require(_stat(path) == before, "FILE_CHANGED", str(path))
    return {"sha256": h.hexdigest(), "bytes": before[2]}, before


def _read_metadata(root, relative, pin, snapshots, budget):
    _pin(pin)
    require(pin["bytes"] <= METADATA_FILE_BYTES, "METADATA_SIZE_LIMIT", relative)
    budget[0] += pin["bytes"]
    require(budget[0] <= COMBINED_METADATA_BYTES, "METADATA_SIZE_LIMIT", "combined metadata ceiling")
    path = safe_path(root, relative); before = _stat(path)
    require(before[2] == pin["bytes"], "FILE_PIN_DRIFT", relative)
    with path.open("rb") as stream:
        raw = stream.read(METADATA_FILE_BYTES + 1)
    require(_stat(path) == before and {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == pin,
            "FILE_PIN_DRIFT", relative)
    snapshots[relative] = (before, dict(pin))
    return parse(raw)


def _load_helper(root, name, snapshots):
    relative = PREFIX + "scripts/" + name + ".py"
    pin = HELPER_PINS[name]; path = safe_path(root, relative); before = _stat(path)
    raw = path.read_bytes()
    require(_stat(path) == before and {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == pin,
            "HELPER_DRIFT", name)
    module_name = "_e1_reader_" + name + "_" + pin["sha256"][:12]
    # Fresh source execution avoids stale bytecode or a mutated module cache.
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    snapshots[relative] = (before, dict(pin))
    return module


def _lines(path):
    before = _stat(path)
    with path.open("rb") as stream:
        number, offset = 0, 0
        while raw := stream.readline(LINE_BYTES + 1):
            require(len(raw) <= LINE_BYTES, "LINE_SIZE_LIMIT", str(path))
            require(raw.strip(), "BLANK_JSONL_LINE", str(path))
            number += 1
            yield number, offset, raw, parse(raw)
            offset += len(raw)
    require(_stat(path) == before, "FILE_CHANGED", str(path))


def _deep_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value)); total = sys.getsizeof(value)
    if isinstance(value, (dict, MappingProxyType)):
        if isinstance(value, MappingProxyType):
            total += sys.getsizeof(dict(value))
        total += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        total += sum(_deep_size(item, seen) for item in value)
    elif isinstance(value, Entry):
        total += sum(_deep_size(getattr(value, field), seen) for field in value.__slots__)
    return total


@dataclass(frozen=True, slots=True)
class Entry:
    component: str
    file: str
    line: int
    offset: int
    length: int
    raw_line_sha256: str
    source_row_sha256: str
    input_token_ids_sha256: str
    parent_id: str
    nonpadding_tokens: int
    completion_tokens: int


def _normalized(row, component, data):
    require(type(row) is dict and not FORBIDDEN_ARRAY_FIELDS.intersection(row), "UNSUPPORTED_FORWARD_FIELDS", component)
    try:
        normalized = data.validate_array(row, component)
    except (ValueError, KeyError, TypeError) as error:
        raise ReaderError("ARRAY_CONTRACT", str(error)) from error
    require(row["nonpadding_tokens"] <= 512, "FORWARD_SCOPE", "CE512 maximum exceeded")
    require(row.get("component", component) == component, "WRONG_COMPONENT", component)
    require(row.get("training_eligibility") is False and type(row.get("target_json_tokens")) is int
            and row["target_json_tokens"] == row["completion_tokens"] - 1, "ARRAY_NO_GRANT", "complete targets and data-only eligibility required")
    if component == "language_replay":
        require(type(row.get("prior_assistant_turns")) is int and row["prior_assistant_turns"] >= 0
                and row.get("prior_turns_supervised") is False, "LANGUAGE_PRIOR_SUPERVISION", "prior replies must be masked")
    return normalized


class BoundTrainReader:
    """One row per lookup; no resident token populations.

    At most two files are open while indexing a language array/source join;
    random lookup opens at most one. No file handles persist between queries.

    Pin arguments must be supplied from the caller's frozen registration. This
    reader does not perform source-current/Git admission for a native executor.
    The root must measure standalone peak RSS; metadata limits here are explicit
    byte/object bounds, not a claim of hard OS process containment.
    """
    def __init__(self, token):
        require(token is _CONSTRUCTION_TOKEN, "FACTORY_REQUIRED", "use the production or fixture factory")

    @classmethod
    def open_production(cls, repository_root, *, plan_relative, plan_pin, binding_relative, binding_pin,
                        scope_receipt_relative, scope_receipt_pin):
        root = Path(repository_root).resolve()
        require((plan_relative, binding_relative, scope_receipt_relative) == (PLAN_RELATIVE, BINDING_RELATIVE, RECEIPT_RELATIVE),
                "PRODUCTION_METADATA_PATH", "only the frozen historical plan/binding and actual scope receipt")
        snapshots, budget = {}, [0]
        scope_path = safe_path(root, SCOPE_RELATIVE)
        require(scope_path.stat().st_size <= METADATA_FILE_BYTES, "METADATA_SIZE_LIMIT", SCOPE_RELATIVE)
        manifest_pin = {"sha256": SCOPE_SHA256, "bytes": scope_path.stat().st_size}
        manifest = _read_metadata(root, SCOPE_RELATIVE, manifest_pin, snapshots, budget)
        helpers = {name: _load_helper(root, name, snapshots) for name in HELPER_PINS}
        helpers["train_scope"].validate_manifest(manifest)
        for relative, pin in manifest["files"].items():
            observed, stat = _hash(safe_path(root, relative))
            require(observed == pin, "FILE_PIN_DRIFT", relative)
            snapshots[relative] = (stat, dict(pin))
        plan = _read_metadata(root, plan_relative, plan_pin, snapshots, budget)
        binding = _read_metadata(root, binding_relative, binding_pin, snapshots, budget)
        receipt = _read_metadata(root, scope_receipt_relative, scope_receipt_pin, snapshots, budget)
        require(receipt.get("status") == "PASS_CE512_TRAIN_COMPATIBILITY_DATA_ONLY"
                and receipt.get("manifest_sha256") == SCOPE_SHA256 and receipt.get("files") == manifest["files"]
                and receipt.get("source_array_join_sha256") == "555776a347ea7a82e7a111b90ab041f5947bb0e54bd32c8b306e9064fbd5fc68"
                and receipt.get("language_prior_assistant_turns_masked") == 81, "SCOPE_RECEIPT", "actual frozen source join required")
        for name in ("native_qualified", "native_launch_authorized", "training_authorized", "e1_authorized", "native_executor", "CHECK_FINAL_access"):
            require(receipt.get(name) is False, "SCOPE_NO_GRANT", name)
        require(binding.get("status") == "BOUND_TRAIN_INPUTS_NOT_TRAINING_ADMISSION"
                and binding.get("native_qualified") is False and binding.get("training_authorized") is False
                and binding.get("data_sha256") == receipt["historical_data_sha256"]
                == "51b87fe4fd70d2fb23d16f270d1023716172dbbe1c88efd79dff8c3548623d6d", "HISTORICAL_BINDING", "frozen historical TRAIN binding required")
        require(binding.get("parent_sha256") == manifest["model_weight_sha256"]
                and plan.get("identities", {}).get("parent_sha256") == binding["parent_sha256"]
                and plan["identities"].get("model_sha256") == binding.get("model_sha256")
                and plan["identities"].get("data_sha256") == binding["data_sha256"], "PLAN_IDENTITY", "parent/model/data bindings differ")
        helpers["e1_schedule"].validate_plan(plan, binding["rows"], identities=plan["identities"])
        spec = manifest["components"]["language_replay"]
        lineage = _read_metadata(root, spec["lineage"], manifest["files"][spec["lineage"]], snapshots, budget)
        profile = _read_metadata(root, spec["profile"], manifest["files"][spec["profile"]], snapshots, budget)["rows"]
        reader = cls._build(root, manifest["components"], manifest["files"], plan, binding, helpers,
                            snapshots, lineage, profile, fixture=False,
                            expected_counts=EXPECTED_COUNTS, expected_doses=EXPECTED_DOSES,
                            expected_maximum=398, expected_prior=81)
        reader._identities = MappingProxyType(dict(plan["identities"]))
        reader._plan_sha256 = plan["plan_sha256"]
        return reader

    @classmethod
    def for_test(cls, repository_root, *, components, files, plan, binding, lineage=(), profile=(), expected_prior=0):
        """Tiny pure fixtures only; cannot bind the production repository or paths."""
        root = Path(repository_root).resolve()
        require(not (root / ".git").exists() and not (root / SCOPE_RELATIVE).exists(), "FIXTURE_ROOT", "fixture directory must be outside the production checkout")
        require(type(binding.get("rows")) is list and 0 < len(binding["rows"]) <= 32, "FIXTURE_QUOTA", "at most 32 fixture rows")
        require(not any(relative.startswith(("artifacts/mimo_central/", PREFIX)) for relative in files), "FIXTURE_PATH", "production bindings forbidden")
        # Import only exact pinned stdlib helpers from this source's own checkout.
        source_root = Path(__file__).resolve().parents[3]
        helpers = {name: _load_helper(source_root, name, {}) for name in ("e1_data", "train_scope")}
        counts = dict(Counter(row["component"] for row in binding["rows"]))
        doses = {component: (sum(row["nonpadding_tokens"] for row in binding["rows"] if row["component"] == component),
                             sum(row["completion_tokens"] for row in binding["rows"] if row["component"] == component)) for component in counts}
        return cls._build(root, components, files, plan, binding, helpers, {}, list(lineage), list(profile), fixture=True,
                          expected_counts=counts, expected_doses=doses,
                          expected_maximum=max(row["nonpadding_tokens"] for row in binding["rows"]), expected_prior=expected_prior)

    @classmethod
    def _build(cls, root, components, files, plan, binding, helpers, snapshots, lineage, profile, *, fixture,
               expected_counts, expected_doses, expected_maximum, expected_prior):
        reader = cls(_CONSTRUCTION_TOKEN)
        reader._root, reader._data, reader._scope = root, helpers["e1_data"], helpers["train_scope"]
        reader._fixture_only, reader._closed = fixture, False
        require(type(components) is dict and set(components) == set(expected_counts)
                and set(components) <= set(COMPONENTS), "COMPONENT_INVENTORY", "exact component inventory required")
        require(type(binding.get("rows")) is list and type(binding.get("locations")) is dict
                and type(plan.get("ordered_rows")) is list and canonical(plan["ordered_rows"]) == canonical(binding["rows"]),
                "ORDERED_ROWS", "plan must preserve the exact bound file-order rows")
        reader._snapshots = dict(snapshots)
        reader._language_spec = dict(components.get("language_replay", {}))
        reader._lineage = {item["row_id"]: deepcopy(item) for item in lineage}
        reader._profile = {item["row_id"]: deepcopy(item) for item in profile}
        reader._language_source = {}
        if "language_replay" in components:
            require(len(reader._lineage) == len(lineage) == len(reader._profile) == len(profile) == expected_counts["language_replay"],
                    "LANGUAGE_QUOTA", "unique complete lineage/profile required")
            relative = components["language_replay"]["source"]; _pin(files[relative])
            observed, stat = _hash(safe_path(root, relative)); require(observed == files[relative], "FILE_PIN_DRIFT", relative)
            reader._snapshots[relative] = (stat, dict(files[relative]))
            for number, offset, raw, source in _lines(safe_path(root, relative)):
                row_id = source.get("id")
                require(type(row_id) is str and row_id not in reader._language_source, "LANGUAGE_IDENTITY", "unique source IDs required")
                reader._language_source[row_id] = (number, offset, len(raw), hashlib.sha256(raw).hexdigest())
                require(len(reader._language_source) <= expected_counts["language_replay"], "LANGUAGE_QUOTA", "source row quota exceeded")
            require(set(reader._language_source) == set(reader._lineage) == set(reader._profile), "LANGUAGE_IDENTITY", "source/lineage/profile must match")
        entries, rows, all_ids, maximum, prior = {}, [], set(), 0, 0
        for component in COMPONENTS:
            if component not in components:
                continue
            relative = components[component]["arrays"]
            require(Path(relative).name == "TRAIN.tokenized.jsonl", "TRAIN_PATH", relative)
            _pin(files[relative]); path = safe_path(root, relative); stat = _stat(path)
            h, size, count, n_total, c_total = hashlib.sha256(), 0, 0, 0, 0
            population = []
            for number, offset, raw, row in _lines(path):
                h.update(raw); size += len(raw); normalized = _normalized(row, component, reader._data)
                key = normalized["key"]
                require(row["row_id"] not in all_ids and key not in entries, "DUPLICATE_ROW", key)
                all_ids.add(row["row_id"])
                entry = Entry(component, relative, number, offset, len(raw), hashlib.sha256(raw).hexdigest(),
                              row["source_row_sha256"], row["input_token_ids_sha256"], row["parent_id"],
                              row["nonpadding_tokens"], row["completion_tokens"])
                require(canonical(binding["locations"].get(key)) == canonical({"file": relative, "line": number, "byte_offset": offset,
                    "raw_line_sha256": entry.raw_line_sha256, "source_row_sha256": entry.source_row_sha256,
                    "input_token_ids_sha256": entry.input_token_ids_sha256}), "LOCATION_BINDING", key)
                if component == "language_replay":
                    reader._validate_language(row)
                    prior += row["prior_assistant_turns"]
                entries[key] = entry; rows.append(normalized); count += 1
                n_total += row["nonpadding_tokens"]; c_total += row["completion_tokens"]
                maximum = max(maximum, row["nonpadding_tokens"])
                population.append({name: row[name] for name in ("row_id", "parent_id", "family", "nonpadding_tokens", "completion_tokens")}
                                  | {"variant_id": row.get("variant_id")})
                require(count <= expected_counts[component], "COMPONENT_QUOTA", component)
            require(_stat(path) == stat and {"sha256": h.hexdigest(), "bytes": size} == files[relative], "FILE_PIN_DRIFT", relative)
            require(count == expected_counts[component] and (n_total, c_total) == expected_doses[component], "COMPONENT_DOSE", component)
            if not fixture:
                reader._data.validate_population(population, component)
            reader._snapshots[relative] = (stat, dict(files[relative]))
        require(canonical(rows) == canonical(binding["rows"]) and len(entries) == len(binding["locations"]), "ORDERED_ROWS", "actual indexed row order differs")
        require(maximum == expected_maximum and prior == expected_prior, "DOSE_TOTALS", "maximum sequence or prior-turn dose differs")
        reader._index = MappingProxyType(entries)
        require(type(plan.get("steps")) is list and plan["steps"], "PLAN_STEPS", "nonempty immutable schedule required")
        steps, scheduled = [], []
        for ordinal, step in enumerate(plan["steps"], 1):
            require(type(step) is dict and type(step.get("optimizer_step")) is int and step["optimizer_step"] == ordinal
                    and type(step.get("row_keys")) is list and step["row_keys"], "PLAN_STEPS", "ordered nonempty optimizer steps required")
            keys = tuple(step["row_keys"])
            require(all(type(key) is str and key in entries for key in keys), "PLAN_KEY", "unknown or wrong-component row key")
            require(type(step.get("learning_rate")) in (int, float) and 0 <= step["learning_rate"] <= 0.0001,
                    "PLAN_LEARNING_RATE", "finite bounded learning rate required")
            require(type(step.get("checkpoint_after")) is bool and type(step.get("dev_evaluation_after")) is bool, "PLAN_STEPS", "boundary booleans required")
            count = Counter(entries[key].component for key in keys)
            component_rows = {component: count[component] for component in components}
            require(canonical(step.get("component_rows")) == canonical(component_rows)
                    and type(step.get("nonpadding_tokens")) is int
                    and type(step.get("completion_tokens")) is int
                    and step["nonpadding_tokens"] == sum(entries[key].nonpadding_tokens for key in keys)
                    and step["completion_tokens"] == sum(entries[key].completion_tokens for key in keys), "PLAN_STEP_DOSE", "step accounting differs")
            steps.append(MappingProxyType({"optimizer_step": ordinal, "row_keys": keys,
                "learning_rate": step["learning_rate"], "checkpoint_after": step["checkpoint_after"],
                "dev_evaluation_after": step["dev_evaluation_after"], "nonpadding_tokens": step["nonpadding_tokens"],
                "completion_tokens": step["completion_tokens"], "component_rows": MappingProxyType(component_rows)}))
            scheduled.extend(keys)
        require(len(scheduled) == len(entries) and len(set(scheduled)) == len(scheduled) and set(scheduled) == set(entries),
                "PLAN_COVERAGE", "schedule must consume each TRAIN row exactly once")
        reader._steps = tuple(steps)
        reader._totals = MappingProxyType({"rows": len(entries), "nonpadding_tokens": sum(row["nonpadding_tokens"] for row in rows),
            "supervised_tokens": sum(row["completion_tokens"] for row in rows), "maximum_sequence_tokens": maximum,
            "language_prior_assistant_turns_masked": prior})
        reader._identities = MappingProxyType({}); reader._plan_sha256 = digest(plan)
        reader._metadata_bytes = _deep_size((reader._index, reader._steps, reader._snapshots,
                                            reader._language_source, reader._lineage, reader._profile))
        require(reader._metadata_bytes <= RETAINED_METADATA_BYTES, "RETAINED_METADATA_LIMIT", "32 MiB object ceiling")
        reader.verify_unchanged(full_hash=True)
        return reader

    def _read_location(self, relative, offset, length, raw_sha):
        require(type(offset) is int and offset >= 0 and type(length) is int and 0 < length <= LINE_BYTES,
                "LOCATION_RANGE", relative)
        path = safe_path(self._root, relative); before = _stat(path)
        require(relative in self._snapshots and before == self._snapshots[relative][0]
                and offset + length <= before[2], "FILE_CHANGED", relative)
        with path.open("rb") as stream:
            if offset:
                stream.seek(offset - 1); require(stream.read(1) == b"\n", "LOCATION_BOUNDARY", relative)
            stream.seek(offset); raw = stream.read(length)
            require(len(raw) == length and hashlib.sha256(raw).hexdigest() == raw_sha, "LINE_HASH_DRIFT", relative)
            require(raw.endswith(b"\n") or offset + length == before[2], "LOCATION_BOUNDARY", relative)
        require(_stat(path) == before, "FILE_CHANGED", relative)
        return parse(raw)

    def _validate_language(self, row):
        row_id = row["row_id"]
        require(row_id in self._language_source, "LANGUAGE_IDENTITY", row_id)
        _, offset, length, raw_sha = self._language_source[row_id]
        source = self._read_location(self._language_spec["source"], offset, length, raw_sha)
        try:
            self._scope.validate_source_join(source, row, "language_replay", self._language_spec, None, self._data,
                                             lineage=self._lineage.get(row_id), profile=self._profile.get(row_id))
        except (ValueError, KeyError, TypeError) as error:
            raise ReaderError("LANGUAGE_SOURCE_JOIN", str(error)) from error

    def verify_unchanged(self, *, full_hash=False):
        require(not self._closed, "READER_CLOSED", "reader closed")
        for relative, (stat, pin) in self._snapshots.items():
            path = safe_path(self._root, relative)
            require(_stat(path) == stat, "FILE_CHANGED", relative)
            if full_hash:
                observed, after = _hash(path)
                require(observed == pin and after == stat, "FILE_PIN_DRIFT", relative)
        return True

    def get(self, key):
        self.verify_unchanged()
        require(type(key) is str and key in self._index, "PLAN_KEY", "unknown TRAIN key")
        entry = self._index[key]
        row = self._read_location(entry.file, entry.offset, entry.length, entry.raw_line_sha256)
        normalized = _normalized(row, entry.component, self._data)
        require(normalized == {"key": key, "component": entry.component, "role": "TRAIN", "parent_id": entry.parent_id,
                              "nonpadding_tokens": entry.nonpadding_tokens, "completion_tokens": entry.completion_tokens}
                and row["source_row_sha256"] == entry.source_row_sha256
                and row["input_token_ids_sha256"] == entry.input_token_ids_sha256, "INDEX_ROW_DRIFT", key)
        if entry.component == "language_replay":
            self._validate_language(row)
        self.verify_unchanged()
        return row

    def iter_plan(self, *, start_optimizer_step=1):
        require(type(start_optimizer_step) is int and 1 <= start_optimizer_step <= len(self._steps) + 1,
                "START_STEP", "1-based complete optimizer boundary required")
        self.verify_unchanged()
        sequence_index = sum(len(step["row_keys"]) for step in self._steps[:start_optimizer_step - 1])
        for step in self._steps[start_optimizer_step - 1:]:
            for microbatch, key in enumerate(step["row_keys"], 1):
                row = self.get(key)
                schedule = {name: step[name] for name in ("optimizer_step", "learning_rate", "checkpoint_after", "dev_evaluation_after",
                                                           "nonpadding_tokens", "completion_tokens")}
                schedule.update(row_key=key, component=self._index[key].component, sequence_index=sequence_index,
                                microbatch_index=microbatch, sequences_in_step=len(step["row_keys"]),
                                component_rows=dict(step["component_rows"]), fixture_only=self._fixture_only,
                                native_launch_authorized=False, training_authorized=False, e1_authorized=False)
                yield {"row": row, "schedule": schedule}
                sequence_index += 1
        self.verify_unchanged(full_hash=True)

    def receipt(self):
        self.verify_unchanged()
        return {"status": "BOUND_RANDOM_ACCESS_TRAIN_DATA_ONLY", "fixture_only": self._fixture_only,
                "runtime_scope": "UNPADDED_SINGLE_SEQUENCE_CE512_V1", "totals": dict(self._totals),
                "retained_metadata_bytes": self._metadata_bytes, "retained_token_rows": 0,
                "metadata_measurement": "recursive sys.getsizeof including mapping storage; allocator/interpreter overhead excluded",
                "maximum_line_bytes": LINE_BYTES, "maximum_simultaneous_file_handles": 2,
                "maximum_lookup_file_handles": 1, "persistent_file_handles": 0,
                "metadata_file_bytes_limit": METADATA_FILE_BYTES, "combined_metadata_bytes_limit": COMBINED_METADATA_BYTES,
                "plan_sha256": self._plan_sha256, "identities": dict(self._identities),
                "native_qualified": False, "native_launch_authorized": False, "training_authorized": False,
                "e1_authorized": False, "native_executor": False, "CHECK_FINAL_access": False,
                "model_calls": 0, "training_updates": 0, "new_tokenization_performed": False}

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
