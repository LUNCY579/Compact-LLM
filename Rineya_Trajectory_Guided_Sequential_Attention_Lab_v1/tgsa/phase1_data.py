"""Deterministic, materialized Phase-1 capability tasks for TGSA.

The tasks are deliberately synthetic and programmatically scored. They are not
presented as a replacement for a sealed language-model evaluation. Their role
is to make the mechanism-identification question falsifiable before a costly
Qwen/GQA retrofit:

* multi-hop pointer chasing tests iterative retrieval;
* modular state updates test sequential state refinement;
* direct lookup is a short-horizon guardrail.

Every generated episode is materialized to JSONL before any arm is trained. A
manifest hashes the exact files so all arms and seeds consume identical data.
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

GENERATOR_VERSION = "tgsa.phase1.synthetic.v1"
DATASET_MANIFEST_VERSION = "tgsa.phase1-dataset-manifest.v1"
EXAMPLE_SCHEMA_VERSION = "tgsa.phase1-example.v1"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


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
class SyntheticVocabulary:
    """Integer-token vocabulary with disjoint semantic ranges."""

    domain_size: int = 32
    max_hops: int = 8
    max_abs_step: int = 8

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    PAIR: int = 3
    OP: int = 4
    QUERY: int = 5
    ANSWER: int = 6
    TASK_POINTER: int = 7
    TASK_STATE: int = 8
    TASK_LOOKUP: int = 9

    def __post_init__(self) -> None:
        if self.domain_size < 4:
            raise ValueError("domain_size must be at least 4")
        if self.max_hops < 1:
            raise ValueError("max_hops must be positive")
        if self.max_abs_step < 1:
            raise ValueError("max_abs_step must be positive")

    @property
    def special_count(self) -> int:
        return 16

    @property
    def key_base(self) -> int:
        return self.special_count

    @property
    def value_base(self) -> int:
        return self.key_base + self.domain_size

    @property
    def op_base(self) -> int:
        return self.value_base + self.domain_size

    @property
    def hop_base(self) -> int:
        return self.op_base + (2 * self.max_abs_step + 1)

    @property
    def vocab_size(self) -> int:
        return self.hop_base + self.max_hops + 1

    def key(self, value: int) -> int:
        self._check_domain(value)
        return self.key_base + value

    def value(self, value: int) -> int:
        self._check_domain(value)
        return self.value_base + value

    def operation(self, step: int) -> int:
        if not -self.max_abs_step <= step <= self.max_abs_step:
            raise ValueError("operation is outside configured range")
        return self.op_base + step + self.max_abs_step

    def hop(self, count: int) -> int:
        if not 1 <= count <= self.max_hops:
            raise ValueError("hop count is outside configured range")
        return self.hop_base + count

    def _check_domain(self, value: int) -> None:
        if not 0 <= value < self.domain_size:
            raise ValueError("domain value is outside configured range")

    def to_dict(self) -> dict[str, int]:
        return {
            "domain_size": self.domain_size,
            "max_hops": self.max_hops,
            "max_abs_step": self.max_abs_step,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.PAD,
            "bos_token_id": self.BOS,
            "eos_token_id": self.EOS,
        }


@dataclass(frozen=True, slots=True)
class Phase1Example:
    schema_version: str
    id: str
    split: str
    family: str
    stratum: str
    tokens: tuple[int, ...]
    answer_token: int
    answer_input_index: int
    latent_signature: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != EXAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported example schema")
        if len(self.tokens) < 3:
            raise ValueError("example is too short")
        if not 0 <= self.answer_input_index < len(self.tokens) - 1:
            raise ValueError("answer_input_index must index an input position")
        if self.tokens[self.answer_input_index + 1] != self.answer_token:
            raise ValueError("answer token must immediately follow answer_input_index")

    def to_json_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["tokens"] = list(self.tokens)
        result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_json_dict(cls, value: Mapping[str, object]) -> "Phase1Example":
        return cls(
            schema_version=str(value["schema_version"]),
            id=str(value["id"]),
            split=str(value["split"]),
            family=str(value["family"]),
            stratum=str(value["stratum"]),
            tokens=tuple(int(token) for token in value["tokens"]),  # type: ignore[arg-type]
            answer_token=int(value["answer_token"]),
            answer_input_index=int(value["answer_input_index"]),
            latent_signature=str(value["latent_signature"]),
            metadata=dict(value.get("metadata", {})),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class Phase1Batch:
    input_ids: Tensor
    labels: Tensor
    loss_mask: Tensor
    key_padding_mask: Tensor
    example_ids: tuple[str, ...]
    families: tuple[str, ...]
    strata: tuple[str, ...]
    answer_input_indices: Tensor
    answer_tokens: Tensor

    def to(self, device: torch.device | str) -> "Phase1Batch":
        return Phase1Batch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            loss_mask=self.loss_mask.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
            example_ids=self.example_ids,
            families=self.families,
            strata=self.strata,
            answer_input_indices=self.answer_input_indices.to(device),
            answer_tokens=self.answer_tokens.to(device),
        )


def _stable_signature(payload: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _example_id(split: str, family: str, signature: str) -> str:
    return f"tgsa_p1__{split}__{family}__{signature[:20]}"


def _context_bucket(length: int) -> str:
    if length <= 32:
        return "short"
    if length <= 64:
        return "medium"
    if length <= 96:
        return "long"
    return "xlong"


def _make_pointer_example(
    *,
    rng: random.Random,
    vocab: SyntheticVocabulary,
    split: str,
    min_nodes: int,
    max_nodes: int,
    min_hops: int,
    max_hops: int,
    family: str,
) -> Phase1Example:
    node_count = rng.randint(min_nodes, max_nodes)
    if node_count > vocab.domain_size:
        raise ValueError("pointer node count exceeds vocabulary domain")
    nodes = rng.sample(range(vocab.domain_size), node_count)
    destinations = nodes[:]
    rng.shuffle(destinations)
    mapping = dict(zip(nodes, destinations, strict=True))
    start = rng.choice(nodes)
    hops = 1 if family == "direct_lookup" else rng.randint(min_hops, max_hops)
    current = start
    for _ in range(hops):
        current = mapping[current]
    target = current

    pairs = list(mapping.items())
    rng.shuffle(pairs)
    task_token = vocab.TASK_LOOKUP if family == "direct_lookup" else vocab.TASK_POINTER
    tokens: list[int] = [vocab.BOS, task_token]
    for key, value in pairs:
        tokens.extend((vocab.PAIR, vocab.key(key), vocab.value(value)))
    tokens.extend((vocab.QUERY, vocab.key(start), vocab.hop(hops), vocab.ANSWER))
    answer_input_index = len(tokens) - 1
    answer_token = vocab.value(target)
    tokens.extend((answer_token, vocab.EOS))

    latent = {
        "family": family,
        "mapping": sorted(mapping.items()),
        "start": start,
        "hops": hops,
    }
    signature = _stable_signature(latent)
    stratum = f"{family}:h{hops}:{_context_bucket(len(tokens))}"
    # Difficulty diagnostics. A random permutation has fixed points and short
    # cycles, so a nominal multi-hop item can be reachable in fewer distinct
    # transitions than its hop label claims, or can resolve to the start node
    # itself. These items are valid but dilute the iterative signal, so they are
    # measured and frozen rather than assumed away
    # (``PHASE1_LAUNCH_AUDIT.md`` finding P1-5).
    cycle_length = 1
    walker = mapping[start]
    while walker != start:
        walker = mapping[walker]
        cycle_length += 1
    effective_hops = hops
    probe = start
    for step in range(1, hops + 1):
        probe = mapping[probe]
        if probe == target:
            effective_hops = step
            break
    return Phase1Example(
        schema_version=EXAMPLE_SCHEMA_VERSION,
        id=_example_id(split, family, signature),
        split=split,
        family=family,
        stratum=stratum,
        tokens=tuple(tokens),
        answer_token=answer_token,
        answer_input_index=answer_input_index,
        latent_signature=signature,
        metadata={
            "node_count": node_count,
            "hops": hops,
            "token_length": len(tokens),
            "effective_hops": effective_hops,
            "start_cycle_length": cycle_length,
            "start_is_fixed_point": mapping[start] == start,
            "answer_equals_start": target == start,
            "trivial": target == start or effective_hops < hops,
        },
    )


def _make_state_example(
    *,
    rng: random.Random,
    vocab: SyntheticVocabulary,
    split: str,
    min_steps: int,
    max_steps: int,
) -> Phase1Example:
    step_count = rng.randint(min_steps, max_steps)
    start = rng.randrange(vocab.domain_size)
    steps: list[int] = []
    current = start
    for _ in range(step_count):
        step = 0
        while step == 0:
            step = rng.randint(-vocab.max_abs_step, vocab.max_abs_step)
        steps.append(step)
        current = (current + step) % vocab.domain_size

    tokens: list[int] = [vocab.BOS, vocab.TASK_STATE, vocab.value(start)]
    for step in steps:
        tokens.extend((vocab.OP, vocab.operation(step)))
    tokens.extend((vocab.QUERY, vocab.ANSWER))
    answer_input_index = len(tokens) - 1
    answer_token = vocab.value(current)
    tokens.extend((answer_token, vocab.EOS))

    latent = {"family": "state_update", "start": start, "steps": steps}
    signature = _stable_signature(latent)
    stratum = f"state_update:s{step_count}:{_context_bucket(len(tokens))}"
    # A sequence of individually non-zero steps can still cancel modulo the
    # domain, leaving the answer equal to the start value (finding P1-5).
    net_shift = sum(steps) % vocab.domain_size
    prefix = start
    first_return = 0
    for index, step in enumerate(steps[:-1], start=1):
        prefix = (prefix + step) % vocab.domain_size
        if prefix == current and first_return == 0:
            first_return = index
    return Phase1Example(
        schema_version=EXAMPLE_SCHEMA_VERSION,
        id=_example_id(split, "state_update", signature),
        split=split,
        family="state_update",
        stratum=stratum,
        tokens=tuple(tokens),
        answer_token=answer_token,
        answer_input_index=answer_input_index,
        latent_signature=signature,
        metadata={
            "step_count": step_count,
            "token_length": len(tokens),
            "net_shift": net_shift,
            "cancels_to_start": net_shift == 0,
            "answer_equals_start": current == start,
            "effective_steps": first_return if first_return else step_count,
            "trivial": current == start or bool(first_return),
        },
    )


def _family_spec(
    spec: Mapping[str, object], family: str, *, split: str | None = None
) -> Mapping[str, object]:
    families = spec.get("families")
    if not isinstance(families, Mapping) or family not in families:
        raise ValueError(f"missing dataset family specification: {family}")
    value = families[family]
    if not isinstance(value, Mapping):
        raise TypeError(f"family specification must be an object: {family}")
    resolved = dict(value)
    overrides = spec.get("split_overrides", {})
    if split is not None and isinstance(overrides, Mapping):
        split_overrides = overrides.get(split, {})
        if isinstance(split_overrides, Mapping):
            family_override = split_overrides.get(family, {})
            if family_override:
                if not isinstance(family_override, Mapping):
                    raise TypeError(f"split override must be an object: {split}.{family}")
                resolved.update(family_override)
    return resolved


def generate_split(
    *,
    split: str,
    count_by_family: Mapping[str, int],
    dataset_seed: int,
    vocab: SyntheticVocabulary,
    generation_spec: Mapping[str, object],
    forbidden_signatures: set[str] | None = None,
) -> list[Phase1Example]:
    """Generate one split, rejecting cross-split and within-split duplicates."""

    forbidden = set() if forbidden_signatures is None else set(forbidden_signatures)
    generated: list[Phase1Example] = []
    seen = set(forbidden)

    for family in sorted(count_by_family):
        requested = int(count_by_family[family])
        if requested < 0:
            raise ValueError("family counts must be non-negative")
        family_cfg = _family_spec(generation_spec, family, split=split)
        rng = random.Random(derive_seed(dataset_seed, split, family))
        accepted = 0
        attempts = 0
        max_attempts = max(100, requested * 100)
        while accepted < requested:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"could not generate enough unique {family} examples")
            if family in {"pointer_chase", "direct_lookup"}:
                example = _make_pointer_example(
                    rng=rng,
                    vocab=vocab,
                    split=split,
                    min_nodes=int(family_cfg["min_nodes"]),
                    max_nodes=int(family_cfg["max_nodes"]),
                    min_hops=int(family_cfg.get("min_hops", 1)),
                    max_hops=int(family_cfg.get("max_hops", 1)),
                    family=family,
                )
            elif family == "state_update":
                example = _make_state_example(
                    rng=rng,
                    vocab=vocab,
                    split=split,
                    min_steps=int(family_cfg["min_steps"]),
                    max_steps=int(family_cfg["max_steps"]),
                )
            else:
                raise ValueError(f"unsupported task family: {family}")
            if example.latent_signature in seen:
                continue
            # Registered, outcome-blind difficulty control. A random permutation
            # produces fixed points and short cycles, and a modular step sequence
            # can cancel, so a nominal multi-hop item may resolve to the start
            # node or need fewer distinct transitions than its label claims.
            # Retaining those items dilutes the iterative mechanism signal and
            # costs power (``PHASE1_LAUNCH_AUDIT.md`` finding P1-5). Rejection is
            # applied at generation time, before any model exists, and is
            # recorded in the dataset manifest.
            if bool(family_cfg.get("reject_trivial", False)) and bool(
                example.metadata.get("trivial", False)
            ):
                continue
            if max(example.tokens) >= vocab.vocab_size or min(example.tokens) < 0:
                raise AssertionError("generator emitted a token outside the vocabulary")
            seen.add(example.latent_signature)
            generated.append(example)
            accepted += 1

    order_rng = random.Random(derive_seed(dataset_seed, split, "final-order"))
    order_rng.shuffle(generated)
    return generated


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
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
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
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


def materialize_dataset(
    *,
    dataset_config: Mapping[str, object],
    output_dir: Path,
    force: bool = False,
) -> dict[str, object]:
    """Materialize all splits and return a hash-verifiable manifest."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "dataset_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dataset_manifest(manifest_path)
        requested_hash = sha256_bytes(canonical_json_bytes(dataset_config))
        if manifest.get("dataset_config_sha256") != requested_hash:
            raise ValueError("existing dataset manifest was created from a different config")
        return manifest

    vocab_cfg = dataset_config.get("vocabulary", {})
    if not isinstance(vocab_cfg, Mapping):
        raise TypeError("dataset.vocabulary must be an object")
    vocab = SyntheticVocabulary(
        domain_size=int(vocab_cfg.get("domain_size", 32)),
        max_hops=int(vocab_cfg.get("max_hops", 8)),
        max_abs_step=int(vocab_cfg.get("max_abs_step", 8)),
    )
    dataset_seed = int(dataset_config["seed"])
    splits_cfg = dataset_config.get("splits")
    if not isinstance(splits_cfg, Mapping):
        raise TypeError("dataset.splits must be an object")

    all_signatures: set[str] = set()
    split_entries: dict[str, object] = {}
    generated_families: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        counts = splits_cfg.get(split)
        if not isinstance(counts, Mapping):
            raise TypeError(f"dataset split must be an object: {split}")
        generated_families.update(str(key) for key in counts)
        examples = generate_split(
            split=split,
            count_by_family={str(k): int(v) for k, v in counts.items()},
            dataset_seed=dataset_seed,
            vocab=vocab,
            generation_spec=dataset_config,
            forbidden_signatures=all_signatures,
        )
        signatures = {example.latent_signature for example in examples}
        if signatures & all_signatures:
            raise AssertionError("cross-split latent leakage detected")
        all_signatures.update(signatures)
        split_path = output_dir / f"{split}.jsonl"
        _write_jsonl_atomic(split_path, (example.to_json_dict() for example in examples))
        family_counts: dict[str, int] = {}
        for example in examples:
            family_counts[example.family] = family_counts.get(example.family, 0) + 1
        split_entries[split] = {
            "path": split_path.name,
            "sha256": sha256_file(split_path),
            "examples": len(examples),
            "family_counts": dict(sorted(family_counts.items())),
            "max_tokens": max((len(example.tokens) for example in examples), default=0),
            "instance_ids_sha256": sha256_bytes(canonical_json_bytes(sorted(example.id for example in examples))),
        }

    manifest: dict[str, object] = {
        "schema_version": DATASET_MANIFEST_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_seed": dataset_seed,
        "dataset_config_sha256": sha256_bytes(canonical_json_bytes(dataset_config)),
        "vocabulary": vocab.to_dict(),
        "splits": split_entries,
        "cross_split_unique_latent_signatures": True,
        "difficulty_policy": {
            family: bool(
                _family_spec(dataset_config, family, split="train").get("reject_trivial", False)
            )
            for family in sorted(generated_families)
        },
        "claim_scope": (
            "controlled synthetic mechanism-identification data; not an independently authored "
            "sealed language-model evaluation"
        ),
    }
    _write_json_atomic(manifest_path, manifest)
    validate_dataset_manifest(manifest_path)
    return manifest


def load_examples(path: Path) -> list[Phase1Example]:
    examples: list[Phase1Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError("row is not an object")
                examples.append(Phase1Example.from_json_dict(value))
            except Exception as exc:
                raise ValueError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"dataset split is empty: {path}")
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs in {path}")
    return examples


def validate_dataset_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_MANIFEST_VERSION:
        raise ValueError("unsupported dataset manifest schema")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise TypeError("manifest.splits must be an object")
    all_signatures: set[str] = set()
    for split in ("train", "dev", "test"):
        entry = splits.get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing split manifest: {split}")
        path = manifest_path.parent / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"split hash mismatch: {split}")
        examples = load_examples(path)
        if len(examples) != int(entry["examples"]):
            raise ValueError(f"split count mismatch: {split}")
        if any(example.split != split for example in examples):
            raise ValueError(f"split label mismatch: {split}")
        signatures = {example.latent_signature for example in examples}
        if signatures & all_signatures:
            raise ValueError("cross-split latent signature overlap")
        all_signatures.update(signatures)
    return manifest


def collate_examples(examples: Sequence[Phase1Example], pad_token_id: int = 0) -> Phase1Batch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_input_length = max(len(example.tokens) - 1 for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_input_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_input_length), -100, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_input_length), dtype=torch.bool)
    key_padding_mask = torch.ones((batch_size, max_input_length), dtype=torch.bool)
    answer_indices = torch.empty(batch_size, dtype=torch.long)
    answer_tokens = torch.empty(batch_size, dtype=torch.long)

    for row, example in enumerate(examples):
        tokens = torch.tensor(example.tokens, dtype=torch.long)
        length = len(example.tokens) - 1
        input_ids[row, :length] = tokens[:-1]
        labels[row, :length] = tokens[1:]
        key_padding_mask[row, :length] = False
        loss_mask[row, example.answer_input_index] = True
        answer_indices[row] = example.answer_input_index
        answer_tokens[row] = example.answer_token

    return Phase1Batch(
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        key_padding_mask=key_padding_mask,
        example_ids=tuple(example.id for example in examples),
        families=tuple(example.family for example in examples),
        strata=tuple(example.stratum for example in examples),
        answer_input_indices=answer_indices,
        answer_tokens=answer_tokens,
    )


def deterministic_batch_stream(
    examples: Sequence[Phase1Example],
    *,
    batch_size: int,
    seed: int,
    start_epoch: int = 0,
) -> Iterator[tuple[int, Phase1Batch]]:
    """Yield an infinite, arm-independent sequence of shuffled batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("examples must be non-empty")
    epoch = start_epoch
    while True:
        indices = list(range(len(examples)))
        random.Random(derive_seed(seed, "epoch", epoch)).shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            selection = [examples[index] for index in indices[offset : offset + batch_size]]
            yield epoch, collate_examples(selection)
        epoch += 1


def fixed_batches(examples: Sequence[Phase1Example], *, batch_size: int) -> Iterator[Phase1Batch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for offset in range(0, len(examples), batch_size):
        yield collate_examples(examples[offset : offset + batch_size])
