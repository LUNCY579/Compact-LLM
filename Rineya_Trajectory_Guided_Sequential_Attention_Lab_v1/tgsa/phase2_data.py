"""Deterministic sequence data for the TGSA x MTP complementarity pipeline.

The Phase-2 smoke study needs a task on which more than one future token is
predictable from the current causal state.  It therefore uses three compact,
programmatically scored recurrence families.  The data are mechanism-validation
inputs, not a substitute for a sealed natural-language Rineya evaluation.

Every split is materialized before training, hashed, and latent-signature
disjoint.  Evaluation splits are generated before train by default so a large
training split cannot exhaust a finite subspace and bias the held-out band.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

PHASE2_DATA_SCHEMA = "tgsa.mtp-sequence-example.v1"
PHASE2_MANIFEST_SCHEMA = "tgsa.mtp-sequence-dataset.v1"
PHASE2_GENERATOR_VERSION = "tgsa.mtp-recurrence-generator.v1"
EOS_TOKEN_ID = 2


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(master_seed: int, *parts: object) -> int:
    payload = canonical_json_bytes([int(master_seed), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


@dataclass(frozen=True, slots=True)
class Phase2Vocabulary:
    domain_size: int = 16

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    SEP: int = 3
    TASK_AFFINE: int = 4
    TASK_SECOND_ORDER: int = 5
    TASK_SWITCH: int = 6

    def __post_init__(self) -> None:
        if self.domain_size < 8:
            raise ValueError("domain_size must be at least 8")

    @property
    def value_base(self) -> int:
        return 16

    @property
    def vocab_size(self) -> int:
        return self.value_base + self.domain_size

    def value(self, number: int) -> int:
        if not 0 <= number < self.domain_size:
            raise ValueError(f"value {number} outside domain [0, {self.domain_size})")
        return self.value_base + number

    def decode_value(self, token: int) -> int:
        value = token - self.value_base
        if not 0 <= value < self.domain_size:
            raise ValueError(f"token {token} is not a value token")
        return value

    def to_dict(self) -> dict[str, int]:
        return {
            "domain_size": self.domain_size,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.PAD,
            "bos_token_id": self.BOS,
            "eos_token_id": self.EOS,
            "separator_token_id": self.SEP,
            "value_base": self.value_base,
        }


@dataclass(frozen=True, slots=True)
class Phase2SequenceExample:
    schema_version: str
    id: str
    split: str
    family: str
    tokens: tuple[int, ...]
    prompt_length: int
    latent_signature: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != PHASE2_DATA_SCHEMA:
            raise ValueError("unsupported Phase-2 example schema")
        if len(self.tokens) < 6:
            raise ValueError("sequence example is too short")
        if not 2 <= self.prompt_length < len(self.tokens):
            raise ValueError("prompt_length must leave at least one target token")
        if self.tokens[-1] != EOS_TOKEN_ID:
            raise ValueError("every sequence must terminate in EOS")

    @property
    def target_tokens(self) -> tuple[int, ...]:
        return self.tokens[self.prompt_length :]

    def to_json_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["tokens"] = list(self.tokens)
        value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_json_dict(cls, value: Mapping[str, object]) -> "Phase2SequenceExample":
        return cls(
            schema_version=str(value["schema_version"]),
            id=str(value["id"]),
            split=str(value["split"]),
            family=str(value["family"]),
            tokens=tuple(int(token) for token in value["tokens"]),  # type: ignore[arg-type]
            prompt_length=int(value["prompt_length"]),
            latent_signature=str(value["latent_signature"]),
            metadata=dict(value.get("metadata", {})),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class Phase2Batch:
    input_ids: Tensor
    attention_mask: Tensor
    target_mask: Tensor
    key_padding_mask: Tensor
    example_ids: tuple[str, ...]
    families: tuple[str, ...]
    prompt_lengths: Tensor

    def to(self, device: torch.device | str) -> "Phase2Batch":
        return Phase2Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            target_mask=self.target_mask.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
            example_ids=self.example_ids,
            families=self.families,
            prompt_lengths=self.prompt_lengths.to(device),
        )


def _stable_signature(payload: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _example_id(split: str, family: str, signature: str) -> str:
    return f"tgsa_mtp__{split}__{family}__{signature[:20]}"


def _affine_episode(
    rng: random.Random,
    *,
    split: str,
    vocab: Phase2Vocabulary,
    continuation_length: int,
) -> Phase2SequenceExample:
    modulus = vocab.domain_size
    # Non-degenerate coefficients.  Odd a avoids collapsing onto a small even
    # subgroup for power-of-two domains.
    a = rng.randrange(1, modulus, 2)
    b = rng.randrange(1, modulus)
    start = rng.randrange(modulus)
    values = [start]
    for _ in range(continuation_length):
        values.append((a * values[-1] + b) % modulus)
    prefix = [vocab.BOS, vocab.TASK_AFFINE, vocab.value(a), vocab.value(b), vocab.value(start), vocab.SEP]
    continuation = [vocab.value(value) for value in values[1:]] + [vocab.EOS]
    payload = {
        "family": "affine",
        "a": a,
        "b": b,
        "start": start,
        "continuation_length": continuation_length,
    }
    signature = _stable_signature(payload)
    return Phase2SequenceExample(
        schema_version=PHASE2_DATA_SCHEMA,
        id=_example_id(split, "affine", signature),
        split=split,
        family="affine",
        tokens=tuple(prefix + continuation),
        prompt_length=len(prefix),
        latent_signature=signature,
        metadata=payload,
    )


def _second_order_episode(
    rng: random.Random,
    *,
    split: str,
    vocab: Phase2Vocabulary,
    continuation_length: int,
) -> Phase2SequenceExample:
    modulus = vocab.domain_size
    c = rng.randrange(1, modulus)
    first = rng.randrange(modulus)
    second = rng.randrange(modulus)
    values = [first, second]
    for _ in range(continuation_length):
        values.append((values[-1] + values[-2] + c) % modulus)
    prefix = [
        vocab.BOS,
        vocab.TASK_SECOND_ORDER,
        vocab.value(c),
        vocab.value(first),
        vocab.value(second),
        vocab.SEP,
    ]
    continuation = [vocab.value(value) for value in values[2:]] + [vocab.EOS]
    payload = {
        "family": "second_order",
        "c": c,
        "first": first,
        "second": second,
        "continuation_length": continuation_length,
    }
    signature = _stable_signature(payload)
    return Phase2SequenceExample(
        schema_version=PHASE2_DATA_SCHEMA,
        id=_example_id(split, "second_order", signature),
        split=split,
        family="second_order",
        tokens=tuple(prefix + continuation),
        prompt_length=len(prefix),
        latent_signature=signature,
        metadata=payload,
    )


def _switch_episode(
    rng: random.Random,
    *,
    split: str,
    vocab: Phase2Vocabulary,
    continuation_length: int,
) -> Phase2SequenceExample:
    modulus = vocab.domain_size
    add_even = rng.randrange(1, modulus)
    add_odd = rng.randrange(1, modulus)
    drift = rng.randrange(1, modulus)
    start = rng.randrange(modulus)
    values = [start]
    for step in range(continuation_length):
        increment = add_even if step % 2 == 0 else add_odd
        values.append((values[-1] + increment + step * drift) % modulus)
    prefix = [
        vocab.BOS,
        vocab.TASK_SWITCH,
        vocab.value(add_even),
        vocab.value(add_odd),
        vocab.value(drift),
        vocab.value(start),
        vocab.SEP,
    ]
    continuation = [vocab.value(value) for value in values[1:]] + [vocab.EOS]
    payload = {
        "family": "switch",
        "add_even": add_even,
        "add_odd": add_odd,
        "drift": drift,
        "start": start,
        "continuation_length": continuation_length,
    }
    signature = _stable_signature(payload)
    return Phase2SequenceExample(
        schema_version=PHASE2_DATA_SCHEMA,
        id=_example_id(split, "switch", signature),
        split=split,
        family="switch",
        tokens=tuple(prefix + continuation),
        prompt_length=len(prefix),
        latent_signature=signature,
        metadata=payload,
    )


_GENERATORS = {
    "affine": _affine_episode,
    "second_order": _second_order_episode,
    "switch": _switch_episode,
}


def generate_split(
    *,
    split: str,
    count_by_family: Mapping[str, int],
    dataset_seed: int,
    vocab: Phase2Vocabulary,
    continuation_lengths: Sequence[int],
    forbidden_signatures: set[str] | None = None,
) -> list[Phase2SequenceExample]:
    if not continuation_lengths or any(length < 3 for length in continuation_lengths):
        raise ValueError("continuation_lengths must contain values >= 3")
    forbidden = set(forbidden_signatures or ())
    generated: list[Phase2SequenceExample] = []
    local_signatures: set[str] = set()
    for family in sorted(count_by_family):
        if family not in _GENERATORS:
            raise ValueError(f"unsupported Phase-2 family: {family}")
        count = int(count_by_family[family])
        if count <= 0:
            raise ValueError(f"family count must be positive: {family}")
        rng = random.Random(derive_seed(dataset_seed, split, family))
        attempts = 0
        family_generated = 0
        while family_generated < count:
            attempts += 1
            if attempts > max(10_000, count * 500):
                raise RuntimeError(
                    f"could not generate {count} unique {family} examples; "
                    "increase the domain or reduce the requested count"
                )
            length = continuation_lengths[rng.randrange(len(continuation_lengths))]
            example = _GENERATORS[family](
                rng,
                split=split,
                vocab=vocab,
                continuation_length=length,
            )
            if example.latent_signature in forbidden or example.latent_signature in local_signatures:
                continue
            local_signatures.add(example.latent_signature)
            generated.append(example)
            family_generated += 1
    random.Random(derive_seed(dataset_seed, split, "shuffle")).shuffle(generated)
    return generated


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def materialize_phase2_dataset(
    *,
    dataset_config: Mapping[str, object],
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "dataset_manifest.json"
    config_hash = sha256_bytes(canonical_json_bytes(dataset_config))
    if manifest_path.is_file() and not force:
        manifest = validate_phase2_dataset(manifest_path)
        if manifest.get("dataset_config_sha256") != config_hash:
            raise ValueError("existing Phase-2 dataset was created from a different config")
        return manifest

    vocabulary_cfg = dataset_config.get("vocabulary", {})
    if not isinstance(vocabulary_cfg, Mapping):
        raise TypeError("dataset.vocabulary must be an object")
    vocab = Phase2Vocabulary(domain_size=int(vocabulary_cfg.get("domain_size", 16)))
    split_cfg = dataset_config.get("splits")
    if not isinstance(split_cfg, Mapping):
        raise TypeError("dataset.splits must be an object")
    lengths = tuple(int(value) for value in dataset_config.get("continuation_lengths", [6, 8]))  # type: ignore[arg-type]
    order = [str(value) for value in dataset_config.get("split_generation_order", ["test", "dev", "train"])]  # type: ignore[arg-type]
    if sorted(order) != ["dev", "test", "train"]:
        raise ValueError("split_generation_order must be a permutation of train/dev/test")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_signatures: set[str] = set()
    entries: dict[str, object] = {}
    dataset_seed = int(dataset_config.get("seed", 20260830))
    for split in order:
        counts = split_cfg.get(split)
        if not isinstance(counts, Mapping):
            raise TypeError(f"dataset.splits.{split} must be an object")
        examples = generate_split(
            split=split,
            count_by_family={str(key): int(value) for key, value in counts.items()},
            dataset_seed=dataset_seed,
            vocab=vocab,
            continuation_lengths=lengths,
            forbidden_signatures=all_signatures,
        )
        signatures = {example.latent_signature for example in examples}
        if signatures & all_signatures:
            raise AssertionError("cross-split latent-signature overlap")
        all_signatures.update(signatures)
        path = output_dir / f"{split}.jsonl"
        _atomic_text(
            path,
            "".join(
                json.dumps(example.to_json_dict(), sort_keys=True, ensure_ascii=True) + "\n"
                for example in examples
            ),
        )
        family_counts: dict[str, int] = {}
        for example in examples:
            family_counts[example.family] = family_counts.get(example.family, 0) + 1
        entries[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "examples": len(examples),
            "family_counts": dict(sorted(family_counts.items())),
            "max_tokens": max(len(example.tokens) for example in examples),
            "instance_ids_sha256": sha256_bytes(
                canonical_json_bytes(sorted(example.id for example in examples))
            ),
        }

    manifest: dict[str, object] = {
        "schema_version": PHASE2_MANIFEST_SCHEMA,
        "generator_version": PHASE2_GENERATOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_seed": dataset_seed,
        "dataset_config_sha256": config_hash,
        "vocabulary": vocab.to_dict(),
        "continuation_lengths": list(lengths),
        "split_generation_order": order,
        "unbiased_split": order[0],
        "cross_split_unique_latent_signatures": True,
        "splits": entries,
        "claim_scope": (
            "deterministic recurrence tasks for pipeline and mechanism validation; "
            "not a sealed natural-language or broad-capability evaluation"
        ),
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_phase2_dataset(manifest_path)
    return manifest


def load_phase2_examples(path: Path) -> list[Phase2SequenceExample]:
    examples: list[Phase2SequenceExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError("row is not an object")
                examples.append(Phase2SequenceExample.from_json_dict(value))
            except Exception as error:
                raise ValueError(f"invalid Phase-2 row {path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"empty Phase-2 split: {path}")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"duplicate example IDs in {path}")
    return examples


def validate_phase2_dataset(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PHASE2_MANIFEST_SCHEMA:
        raise ValueError("unsupported Phase-2 dataset manifest schema")
    entries = manifest.get("splits")
    if not isinstance(entries, Mapping):
        raise TypeError("manifest.splits must be an object")
    signatures: set[str] = set()
    for split in ("train", "dev", "test"):
        entry = entries.get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing split manifest: {split}")
        path = manifest_path.parent / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"split hash mismatch: {split}")
        examples = load_phase2_examples(path)
        if len(examples) != int(entry["examples"]):
            raise ValueError(f"split count mismatch: {split}")
        if any(example.split != split for example in examples):
            raise ValueError(f"split label mismatch: {split}")
        current = {example.latent_signature for example in examples}
        if current & signatures:
            raise ValueError("cross-split latent-signature overlap")
        signatures.update(current)
    return manifest


def collate_phase2_examples(
    examples: Sequence[Phase2SequenceExample],
    *,
    pad_token_id: int = 0,
) -> Phase2Batch:
    if not examples:
        raise ValueError("cannot collate an empty Phase-2 batch")
    batch = len(examples)
    sequence = max(len(example.tokens) for example in examples)
    input_ids = torch.full((batch, sequence), pad_token_id, dtype=torch.long)
    attention = torch.zeros((batch, sequence), dtype=torch.bool)
    target = torch.zeros((batch, sequence), dtype=torch.bool)
    prompt_lengths = torch.empty(batch, dtype=torch.long)
    for row, example in enumerate(examples):
        tokens = torch.tensor(example.tokens, dtype=torch.long)
        length = len(example.tokens)
        input_ids[row, :length] = tokens
        attention[row, :length] = True
        target[row, example.prompt_length : length] = True
        prompt_lengths[row] = example.prompt_length
    return Phase2Batch(
        input_ids=input_ids,
        attention_mask=attention,
        target_mask=target,
        key_padding_mask=~attention,
        example_ids=tuple(example.id for example in examples),
        families=tuple(example.family for example in examples),
        prompt_lengths=prompt_lengths,
    )


def deterministic_batch_indices(
    example_count: int,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Materialize one arm-independent batch schedule for a paired seed."""

    if example_count <= 0 or batch_size <= 0 or steps <= 0:
        raise ValueError("example_count, batch_size, and steps must be positive")
    if batch_size > example_count:
        raise ValueError("batch_size may not exceed example_count")
    rng = random.Random(seed)
    order = list(range(example_count))
    cursor = example_count
    schedule: list[tuple[int, ...]] = []
    while len(schedule) < steps:
        if cursor + batch_size > example_count:
            rng.shuffle(order)
            cursor = 0
        schedule.append(tuple(order[cursor : cursor + batch_size]))
        cursor += batch_size
    return schedule


def batches_from_schedule(
    examples: Sequence[Phase2SequenceExample],
    schedule: Sequence[Sequence[int]],
    *,
    pad_token_id: int = 0,
) -> Iterator[Phase2Batch]:
    for indices in schedule:
        yield collate_phase2_examples(
            [examples[int(index)] for index in indices],
            pad_token_id=pad_token_id,
        )
