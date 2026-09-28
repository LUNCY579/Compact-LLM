"""Strict replay records, content binding, and split admission. Standard library only.

Hashes establish byte identity, not truth: historical admission additionally needs
an accountable curator to verify proposal-time context and original receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class Invalid(ValueError):
    """An input fails the frozen study contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    def bad_constant(value: str) -> None:
        raise Invalid(f"nonfinite JSON constant: {value}")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=strict_object,
                          parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Invalid(f"invalid JSON in {path.name}: {exc}") from exc


def write_new(path: Path, value: Any) -> None:
    """Never overwrite an evidence artifact, including on a retry."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def fields(obj: Any, expected: str) -> None:
    require(type(obj) is dict, "expected JSON object")
    require(set(obj) == set(expected.split()), f"unexpected/missing keys: {set(obj) ^ set(expected.split())}")


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 512,
            f"invalid {label}")
    return value


def number(value: Any, label: str, low: float = 0, high: float = math.inf) -> float:
    require(type(value) in (int, float) and math.isfinite(value)
            and low <= value <= high, f"invalid {label}")
    return float(value)


def integer(value: Any, label: str, low: int = 0, high: int = 10**9) -> int:
    require(type(value) is int and low <= value <= high, f"invalid {label}")
    return value


def hash_string(value: Any, size: int = 64) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None,
            "invalid content hash")
    return value


def safe_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative,
            "path must be relative POSIX")
    pure = PurePosixPath(relative)
    require(bool(relative) and not pure.is_absolute() and ".." not in pure.parts,
            "path traversal rejected")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "symlink escape rejected")
    return path


def pointer(value: Any, location: str) -> Any:
    require(isinstance(location, str) and (location == "" or location.startswith("/")),
            "invalid JSON pointer")
    if not location:
        return value
    try:
        for raw in location[1:].split("/"):
            require(re.search(r"~(?![01])", raw) is None, "invalid pointer escape")
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                require(re.fullmatch(r"0|[1-9][0-9]*", key) is not None, "invalid array index")
                value = value[int(key)]
            else:
                value = value[key]
        return value
    except (KeyError, IndexError, TypeError) as exc:
        raise Invalid("JSON pointer does not resolve") from exc


@dataclass(frozen=True)
class Outcome:
    status: str
    qualified: bool
    protocol_valid: bool
    capability: float | None
    retention: float | None
    gpu_seconds: float | None
    wall_seconds: float | None
    peak_host_gib: float | None
    peak_vram_gib: float | None
    parameter_delta: int | None
    latency_ratio: float | None
    failure: str

    @classmethod
    def parse(cls, obj: dict) -> Outcome:
        fields(obj, "status qualified protocol_valid capability retention gpu_seconds wall_seconds "
                    "peak_host_gib peak_vram_gib parameter_delta latency_ratio failure")
        require(obj["status"] in ("measured", "failed", "blocked", "not_run"), "unknown status")
        require(type(obj["qualified"]) is bool and type(obj["protocol_valid"]) is bool,
                "gates must be booleans")
        for key in ("capability", "retention"):
            if obj[key] is not None:
                number(obj[key], key, 0, 1)
        for key in ("gpu_seconds", "wall_seconds", "peak_host_gib", "peak_vram_gib", "latency_ratio"):
            if obj[key] is not None:
                number(obj[key], key)
        if obj["parameter_delta"] is not None:
            integer(obj["parameter_delta"], "parameter_delta", -10**12, 10**12)
        require(obj["failure"] in ("none", "implementation", "numerical", "resource", "environment"),
                "unknown failure category")
        if obj["status"] == "measured":
            require(obj["capability"] is not None and obj["retention"] is not None,
                    "measured outcome needs both metrics")
            require(obj["failure"] == "none", "measured outcome cannot be an execution failure")
        else:
            require(obj["capability"] is None and obj["retention"] is None,
                    "missing/failed work cannot contain imputed scores")
            require(not obj["qualified"], "unmeasured work cannot be qualified")
            require(obj["failure"] != "none", "unmeasured work needs failure category")
        return cls(**obj)


@dataclass(frozen=True)
class Node:
    key: str
    parent: str
    order: int
    context: tuple[str, ...]
    outcome: Outcome


@dataclass(frozen=True)
class World:
    key: str
    kind: str
    leakage_keys: tuple[str, ...]
    baseline: Outcome
    nodes: tuple[Node, ...]
    content_sha256: str


def validate_source_binding(raw: dict, root: Path) -> None:
    """Exact JSON-field extraction from byte-verified receipts; no fuzzy text parsing.

    The manifest binds every baseline/node field. This detects transcription or
    stale-source errors; it does NOT independently authenticate the Git commit.
    """
    admission = raw["admission"]
    fields(admission, "curator review_id context_review complete_failure_log sources bindings")
    text(admission["curator"], "curator")
    text(admission["review_id"], "review id")
    require(admission["context_review"] is True and admission["complete_failure_log"] is True,
            "historical causal-context/failure-log review required")
    require(type(admission["sources"]) is list and bool(admission["sources"]), "missing source receipts")
    sources = []
    for source in admission["sources"]:
        fields(source, "path sha256 git_blob git_commit repository_path")
        hash_string(source["sha256"])
        hash_string(source["git_blob"], 40)
        hash_string(source["git_commit"], 40)
        text(source["repository_path"], "repository source path")
        path = safe_path(root, source["path"])
        data = path.read_bytes()
        require(bytes_hash(data) == source["sha256"], "source SHA256 mismatch")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == source["git_blob"], "source Git blob mismatch")
        sources.append(read_json(path))
    targets = {"/identity/" + key for key in raw["identity"]}
    targets.update("/baseline/" + key for key in raw["baseline"])
    for i, node in enumerate(raw["nodes"]):
        for key in node:
            if key == "outcome":
                targets.update(f"/nodes/{i}/outcome/{k}" for k in node[key])
            else:
                targets.add(f"/nodes/{i}/{key}")
    require(type(admission["bindings"]) is list, "bindings must be a list")
    seen = set()
    for binding in admission["bindings"]:
        fields(binding, "target source pointer")
        target = binding["target"]
        require(target in targets and target not in seen, "unexpected/duplicate binding")
        index = integer(binding["source"], "source index", 0, len(sources) - 1)
        require(canonical(pointer(raw, target)) == canonical(pointer(sources[index], binding["pointer"])),
                f"receipt field mismatch: {target}")
        seen.add(target)
    require(seen == targets, "every replay field needs a source binding")


def identity_keys(raw: dict) -> tuple[str, ...]:
    fields(raw, "checkpoint_sha256 protocol_sha256 dataset_sha256 study_family task_family lineage_family")
    for name in ("checkpoint_sha256", "protocol_sha256", "dataset_sha256"):
        hash_string(raw[name])
    for name in ("study_family", "task_family", "lineage_family"):
        text(raw[name], name)
    return ("model:" + raw["checkpoint_sha256"], "data:" + raw["dataset_sha256"],
            "study:" + raw["study_family"], "task:" + raw["task_family"],
            "lineage:" + raw["lineage_family"])


def parse_world(raw: dict, root: Path | None = None) -> World:
    fields(raw, "schema id kind closed sealed identity leakage_keys baseline nodes admission")
    require(raw["schema"] == "rineya-dream-world-v1", "wrong world schema")
    key = text(raw["id"], "world id")
    require(raw["kind"] in ("synthetic", "historical"), "unknown evidence kind")
    require(raw["closed"] is True and raw["sealed"] is False, "active/sealed worlds forbidden")
    require(type(raw["leakage_keys"]) is list and bool(raw["leakage_keys"]), "missing leakage keys")
    keys = tuple(text(k, "leakage key") for k in raw["leakage_keys"])
    require(len(set(keys)) == len(keys), "duplicate leakage keys")
    require(set(identity_keys(raw["identity"])) <= set(keys), "required identity leakage keys missing")
    baseline = Outcome.parse(raw["baseline"])
    require(baseline.status == "measured" and baseline.qualified and baseline.protocol_valid,
            "baseline must be a complete qualified measurement")
    require(type(raw["nodes"]) is list and bool(raw["nodes"]), "empty replay world")
    nodes: list[Node] = []
    parents = {"root": None}
    ancestors: dict[str, set[str]] = {"root": {"root"}}
    orders: set[int] = set()
    children: set[str] = set()
    for obj in raw["nodes"]:
        fields(obj, "id parent order context outcome")
        node_id = text(obj["id"], "node id")
        parent = text(obj["parent"], "parent")
        require(node_id not in parents and parent in parents, "duplicate, cyclic, or unordered parent")
        order = integer(obj["order"], "creation order", 1)
        require(order not in orders and (not nodes or order > nodes[-1].order), "ambiguous creation order")
        require(parent == "root" or parent not in children, "non-root branching/DAG unsupported; do not flatten")
        require(type(obj["context"]) is list and all(isinstance(c, str) for c in obj["context"]),
                "invalid proposal context")
        require(len(set(obj["context"])) == len(obj["context"]), "duplicate proposal context")
        require(set(obj["context"]) <= ancestors[parent], "unmodeled sibling/global-history dependency")
        outcome = Outcome.parse(obj["outcome"])
        require(outcome.status != "not_run", "unexecuted plans are not replay attempts")
        nodes.append(Node(node_id, parent, order, tuple(obj["context"]), outcome))
        parents[node_id] = parent
        ancestors[node_id] = ancestors[parent] | {node_id}
        children.add(parent)
        orders.add(order)
    if raw["kind"] == "historical":
        require(root is not None, "historical receipt root required")
        validate_source_binding(raw, root)
    else:
        require(raw["admission"] is None, "synthetic world must not carry historical admission")
    return World(key, raw["kind"], keys, baseline, tuple(nodes), digest(raw))


def validate_catalog(raw: dict) -> dict:
    """Validate split metadata WITHOUT opening world files, especially holdout."""
    fields(raw, "schema kind worlds")
    require(raw["schema"] == "rineya-dream-catalog-v1", "wrong catalog schema")
    require(raw["kind"] in ("synthetic", "historical"), "unknown catalog kind")
    require(type(raw["worlds"]) is list and bool(raw["worlds"]), "empty catalog")
    ids, paths, hashes = set(), set(), set()
    key_splits: dict[str, str] = {}
    splits = set()
    for entry in raw["worlds"]:
        fields(entry, "id path sha256 split leakage_keys")
        text(entry["id"], "catalog world id")
        text(entry["path"], "catalog path")
        hash_string(entry["sha256"])
        require(entry["split"] in ("train", "validation", "test"), "unknown split")
        require(entry["id"] not in ids and entry["path"] not in paths and entry["sha256"] not in hashes,
                "duplicate world/path/hash")
        require(type(entry["leakage_keys"]) is list and bool(entry["leakage_keys"]), "missing leakage keys")
        require(all(isinstance(k, str) for k in entry["leakage_keys"]), "invalid leakage key type")
        require(len(set(entry["leakage_keys"])) == len(entry["leakage_keys"]), "duplicate leakage keys")
        for key in entry["leakage_keys"]:
            text(key, "leakage key")
            require(key not in key_splits or key_splits[key] == entry["split"], "cross-split lineage/task/data leakage")
            key_splits[key] = entry["split"]
        ids.add(entry["id"])
        paths.add(entry["path"])
        hashes.add(entry["sha256"])
        splits.add(entry["split"])
    require(splits == {"train", "validation", "test"}, "all three nonempty splits required")
    return raw


def load_partition(root: Path, catalog: dict, split: str) -> tuple[World, ...]:
    validate_catalog(catalog)
    require(split in ("train", "validation", "test"), "unknown requested split")
    worlds = []
    for entry in catalog["worlds"]:
        if entry["split"] != split:
            continue
        raw = read_json(safe_path(root, entry["path"]))
        require(digest(raw) == entry["sha256"], "world commitment mismatch")
        world = parse_world(raw, root)
        require(world.key == entry["id"] and world.kind == catalog["kind"] and
                tuple(entry["leakage_keys"]) == world.leakage_keys, "world metadata mismatch")
        worlds.append(world)
    return tuple(worlds)
