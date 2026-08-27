"""Leakage-resistant split construction and fail-closed chat encoding."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .hashing import canonical_json_hash, sha256_file
from .manifest import atomic_write_json, utc_now


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any: ...

    def __call__(
        self, text: str, *, add_special_tokens: bool
    ) -> dict[str, list[int]]: ...


@dataclass(frozen=True)
class LineageRecord:
    sft_id: str
    split: str
    source_family_id: str
    origin_group_ids: tuple[str, ...]
    domain: str
    task_key: str
    stratum: str
    evidence_position: str
    prompt_sha256: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> LineageRecord:
        required = {
            "sft_id",
            "split",
            "source_family_id",
            "origin_group_ids",
            "domain",
            "task_key",
            "stratum",
            "evidence_position",
            "prompt_sha256",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"Lineage row is missing fields: {missing}")
        origins = value["origin_group_ids"]
        if not isinstance(origins, list) or not origins:
            raise ValueError("origin_group_ids must be a non-empty list")
        return cls(
            sft_id=str(value["sft_id"]),
            split=str(value["split"]),
            source_family_id=str(value["source_family_id"]),
            origin_group_ids=tuple(str(item) for item in origins),
            domain=str(value["domain"]),
            task_key=str(value["task_key"]),
            stratum=str(value["stratum"]),
            evidence_position=str(value["evidence_position"]),
            prompt_sha256=str(value["prompt_sha256"]),
        )


@dataclass(frozen=True)
class Component:
    component_id: str
    origin_group_ids: tuple[str, ...]
    source_family_ids: tuple[str, ...]
    row_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitAssignment:
    meta_train: tuple[Component, ...]
    mask_cal: tuple[Component, ...]
    unused: tuple[Component, ...]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            yield value


def load_lineage(path: Path, expected_split: str) -> list[LineageRecord]:
    records = [LineageRecord.from_mapping(value) for value in iter_jsonl(path)]
    if not records:
        raise RuntimeError(f"Lineage file is empty: {path}")
    if any(record.split != expected_split for record in records):
        raise RuntimeError(f"Lineage file mixes splits: {path}")
    row_ids = [record.sft_id for record in records]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"Duplicate sft_id in {path}")
    return records


def connected_components(records: Sequence[LineageRecord]) -> list[Component]:
    """Connect rows sharing an origin or family to prevent semantic leakage."""

    graph = UnionFind()
    row_nodes: dict[str, str] = {}
    for record in records:
        row_node = f"row:{record.sft_id}"
        row_nodes[record.sft_id] = row_node
        graph.add(row_node)
        family_node = f"family:{record.source_family_id}"
        graph.add(family_node)
        graph.union(row_node, family_node)
        for origin in record.origin_group_ids:
            origin_node = f"origin:{origin}"
            graph.add(origin_node)
            graph.union(row_node, origin_node)

    grouped: dict[str, list[LineageRecord]] = defaultdict(list)
    for record in records:
        grouped[graph.find(row_nodes[record.sft_id])].append(record)

    components: list[Component] = []
    for rows in grouped.values():
        origins = sorted({origin for row in rows for origin in row.origin_group_ids})
        families = sorted({row.source_family_id for row in rows})
        row_ids = sorted(row.sft_id for row in rows)
        component_id = canonical_json_hash(
            {"origins": origins, "families": families, "rows": row_ids}
        )
        components.append(
            Component(
                component_id=component_id,
                origin_group_ids=tuple(origins),
                source_family_ids=tuple(families),
                row_ids=tuple(row_ids),
            )
        )
    return sorted(components, key=lambda item: item.component_id)


def assign_components(
    components: Sequence[Component],
    *,
    meta_train_count: int,
    mask_cal_count: int,
    split_salt: str,
) -> SplitAssignment:
    required = meta_train_count + mask_cal_count
    if len(components) < required:
        raise RuntimeError(
            f"Need at least {required} semantic components, found {len(components)}"
        )
    ordered = sorted(
        components,
        key=lambda item: canonical_json_hash(
            {"salt": split_salt, "component": item.component_id}
        ),
    )
    assignment = SplitAssignment(
        meta_train=tuple(ordered[:meta_train_count]),
        mask_cal=tuple(ordered[meta_train_count:required]),
        unused=tuple(ordered[required:]),
    )
    assert_disjoint_assignment(assignment)
    return assignment


def assert_disjoint_assignment(assignment: SplitAssignment) -> None:
    def values(components: Iterable[Component], attribute: str) -> set[str]:
        return {
            item for component in components for item in getattr(component, attribute)
        }

    meta_rows = values(assignment.meta_train, "row_ids")
    cal_rows = values(assignment.mask_cal, "row_ids")
    meta_origins = values(assignment.meta_train, "origin_group_ids")
    cal_origins = values(assignment.mask_cal, "origin_group_ids")
    meta_families = values(assignment.meta_train, "source_family_ids")
    cal_families = values(assignment.mask_cal, "source_family_ids")
    if meta_rows & cal_rows:
        raise RuntimeError("META_TRAIN and MASK_CAL share row IDs")
    if meta_origins & cal_origins:
        raise RuntimeError("META_TRAIN and MASK_CAL share origin groups")
    if meta_families & cal_families:
        raise RuntimeError("META_TRAIN and MASK_CAL share source families")


def assignment_manifest(
    assignment: SplitAssignment, lineage_path: Path, split_salt: str
) -> dict[str, Any]:
    def serialize(components: Sequence[Component]) -> list[dict[str, Any]]:
        return [asdict(component) for component in components]

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "policy": "connected_components_of_origin_groups_and_source_families",
        "split_salt": split_salt,
        "lineage": {
            "path": str(lineage_path.resolve()),
            "sha256": sha256_file(lineage_path),
            "size": lineage_path.stat().st_size,
        },
        "meta_train": serialize(assignment.meta_train),
        "mask_cal": serialize(assignment.mask_cal),
        "unused": serialize(assignment.unused),
    }
    manifest["assignment_sha256"] = canonical_json_hash(manifest)
    return manifest


def select_outcome_blind_components(
    records: Sequence[LineageRecord], *, component_count: int, salt: str
) -> tuple[Component, ...]:
    """Select whole validation components using IDs only, never model outcomes."""

    components = connected_components(records)
    if component_count <= 0 or component_count > len(components):
        raise ValueError(
            f"component_count must be in [1, {len(components)}], got {component_count}"
        )
    return tuple(
        sorted(
            components,
            key=lambda item: canonical_json_hash(
                {"salt": salt, "component": item.component_id}
            ),
        )[:component_count]
    )


def write_assignment(
    output_path: Path,
    assignment: SplitAssignment,
    lineage_path: Path,
    split_salt: str,
) -> None:
    atomic_write_json(
        output_path, assignment_manifest(assignment, lineage_path, split_salt)
    )


class CorpusAccessPolicy:
    """Audit every data read and prohibit sealed access until choices are frozen."""

    def __init__(self, sealed_paths: Iterable[Path]) -> None:
        self._sealed = {path.resolve() for path in sealed_paths}
        self._choices_frozen = False
        self.access_log: list[dict[str, Any]] = []

    def freeze_choices(self, decision_hash: str) -> None:
        if len(decision_hash) != 64:
            raise ValueError("decision_hash must be a SHA-256 digest")
        self._choices_frozen = True
        self.access_log.append(
            {"event": "choices_frozen", "decision_sha256": decision_hash}
        )

    def authorize(self, path: Path, purpose: str) -> Path:
        resolved = path.resolve()
        is_sealed = any(
            resolved == sealed or sealed in resolved.parents for sealed in self._sealed
        )
        if is_sealed and not self._choices_frozen:
            raise PermissionError(
                f"Sealed data read blocked before all choices freeze: {resolved}"
            )
        self.access_log.append(
            {
                "event": "data_access",
                "path": str(resolved),
                "purpose": purpose,
                "sealed": is_sealed,
            }
        )
        return resolved


def encode_chat_example(
    tokenizer: ChatTokenizer,
    messages: Sequence[dict[str, str]],
    *,
    maximum_tokens: int,
    example_id: str,
) -> tuple[list[int], list[int]]:
    """Build assistant-only labels and reject non-prefix templates or truncation."""

    if not messages:
        raise ValueError("Conversation is empty")
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"Unsupported message role at {example_id}:{index}: {role}"
            )
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"Example {example_id} must end with an assistant target")

    def template_ids(
        conversation: Sequence[dict[str, str]], *, add_generation_prompt: bool
    ) -> list[int]:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            encoded = tokenizer.apply_chat_template(
                conversation, enable_thinking=False, **kwargs
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(conversation, **kwargs)
        if isinstance(encoded, str):
            encoded = tokenizer(encoded, add_special_tokens=False)
        if hasattr(encoded, "input_ids"):
            return [int(token) for token in encoded.input_ids]
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            return [int(token) for token in encoded["input_ids"]]
        elif isinstance(encoded, list):
            return [int(token) for token in encoded]
        else:
            raise TypeError(
                f"Unsupported chat-template result type: {type(encoded).__name__}"
            )

    prompt_ids = template_ids(messages[:-1], add_generation_prompt=True)
    input_ids = template_ids(messages, add_generation_prompt=False)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat template is not prefix-stable; labels would be ambiguous"
        )
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if len(input_ids) > maximum_tokens:
        raise RuntimeError(
            f"Example {example_id} requires {len(input_ids)} tokens, exceeding "
            f"{maximum_tokens}; PCSM forbids silent truncation"
        )
    if len(input_ids) <= len(prompt_ids) or not any(label != -100 for label in labels):
        raise RuntimeError(f"Example {example_id} has zero assistant tokens")
    if len(input_ids) != len(labels):
        raise AssertionError("Input/label lengths differ")
    return input_ids, labels


def stream_selected_sft_rows(
    shard_paths: Sequence[Path], selected_ids: set[str]
) -> Iterator[dict[str, Any]]:
    found: set[str] = set()
    for shard_path in sorted(shard_paths):
        for row in iter_jsonl(shard_path):
            row_id = str(row.get("id", ""))
            if row_id in selected_ids:
                if row_id in found:
                    raise RuntimeError(f"Duplicate selected SFT row: {row_id}")
                found.add(row_id)
                yield row
                if found == selected_ids:
                    return
    missing = sorted(selected_ids - found)
    if missing:
        raise RuntimeError(f"Selected SFT rows are missing: {missing[:5]}")


def stable_sample_ids(row_ids: Iterable[str], count: int, salt: str) -> tuple[str, ...]:
    unique = set(row_ids)
    if count <= 0 or count > len(unique):
        raise ValueError(f"count must be in [1, {len(unique)}], got {count}")
    return tuple(
        sorted(
            unique,
            key=lambda row_id: canonical_json_hash({"salt": salt, "row_id": row_id}),
        )[:count]
    )
