"""Explicit identities and leakage-resistant activation contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Iterable

import torch

ROLES = frozenset({"FIT", "MAP_CHECK", "SELECT", "CHECK", "RETENTION"})
SITES = frozenset({"block_input_raw", "post_attention_raw", "block_output_raw"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha256(value: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "expected a lowercase SHA-256")


def positive_int(value: int, label: str) -> None:
    require(type(value) is int and value > 0, f"{label} must be a positive integer")


def finite(value: float, label: str) -> None:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")


@dataclass(frozen=True)
class Identity:
    name: str
    revision: str
    weights_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    runtime_sha256: str
    precision: str
    logit_policy: str
    fixture: bool = False

    def __post_init__(self) -> None:
        require(isinstance(self.name, str) and bool(self.name.strip()), "model name missing")
        require(isinstance(self.revision, str) and
                re.fullmatch(r"[0-9a-f]{40}", self.revision) is not None, "immutable revision required")
        for field in (self.weights_sha256, self.tokenizer_sha256,
                      self.template_sha256, self.runtime_sha256):
            sha256(field)
        require(self.precision in {"float16", "bfloat16", "float32", "float64"}, "unknown precision")
        require(self.logit_policy in {"dense_unmodified", "custom_a11_unqualified"}, "unknown logit policy")
        require(type(self.fixture) is bool, "fixture must be explicit bool")

    @property
    def key(self) -> str:
        return digest(asdict(self))


def same_token_grid(a: Identity, b: Identity) -> None:
    require(a.tokenizer_sha256 == b.tokenizer_sha256, "tokenizer mismatch")
    require(a.template_sha256 == b.template_sha256, "template mismatch")
    require(a.fixture == b.fixture, "cannot mix fixture and native evidence")


@dataclass(frozen=True)
class Site:
    block_index: int
    kind: str
    width: int

    def __post_init__(self) -> None:
        require(type(self.block_index) is int and self.block_index >= 0, "invalid block index")
        require(self.kind in SITES, "capture must be an explicit raw residual site")
        positive_int(self.width, "width")


@dataclass(frozen=True)
class RowKey:
    example_id: str
    group_id: str
    prompt_sha256: str
    input_ids_sha256: str
    position: int
    token_id: int

    def __post_init__(self) -> None:
        require(isinstance(self.example_id, str) and bool(self.example_id), "example ID missing")
        require(isinstance(self.group_id, str) and bool(self.group_id), "group ID missing")
        sha256(self.prompt_sha256)
        sha256(self.input_ids_sha256)
        for label, value in (("position", self.position), ("token ID", self.token_id)):
            require(type(value) is int and value >= 0, f"invalid {label}")


@dataclass(frozen=True)
class Capture:
    identity: Identity
    site: Site
    role: str
    registration_sha256: str
    rows: tuple[RowKey, ...]
    values: torch.Tensor

    def validate(self) -> None:
        require(self.role in ROLES, "sealed/test2/unrecognized role refused")
        sha256(self.registration_sha256)
        require(type(self.rows) is tuple and len(self.rows) > 0, "empty/non-tuple row keys")
        x = self.values
        require(isinstance(x, torch.Tensor) and x.device.type == "cpu", "capture must be CPU tensor")
        require(x.ndim == 2 and tuple(x.shape) == (len(self.rows), self.site.width), "capture shape mismatch")
        require(x.is_floating_point() and not x.requires_grad, "capture must be detached floating point")
        require(torch.isfinite(x).all().item(), "nonfinite activations")
        require(len(set(self.rows)) == len(self.rows), "duplicate row key")
        examples: dict[str, tuple] = {}
        coordinates = set()
        for row in self.rows:
            require(isinstance(row, RowKey), "invalid row key type")
            value = (row.group_id, row.prompt_sha256, row.input_ids_sha256)
            require(row.example_id not in examples or examples[row.example_id] == value,
                    "inconsistent example metadata")
            examples[row.example_id] = value
            coordinate = (row.example_id, row.position)
            require(coordinate not in coordinates, "duplicate example/token position")
            coordinates.add(coordinate)

    @property
    def contract(self) -> dict:
        return {"identity": asdict(self.identity), "site": asdict(self.site),
                "role": self.role, "registration_sha256": self.registration_sha256,
                "rows": [asdict(r) for r in self.rows]}


def paired(a: Capture, b: Capture) -> None:
    a.validate()
    b.validate()
    same_token_grid(a.identity, b.identity)
    require(a.role == b.role and a.registration_sha256 == b.registration_sha256,
            "paired capture role/registration mismatch")
    require(a.rows == b.rows, "token grids/order differ; never pair by row number alone")


def disjoint(banks: Iterable[Capture]) -> None:
    """Enforce split isolation by group, example and normalized-prompt digest.

    Multiple captures of the same examples within a role are allowed.
    Producer review must establish group IDs and prompt normalization honestly.
    """
    seen: dict[tuple[str, str], str] = {}
    registration = None
    for bank in banks:
        bank.validate()
        registration = registration or bank.registration_sha256
        require(bank.registration_sha256 == registration, "registration mismatch between splits")
        for row in bank.rows:
            for label, value in (("group", row.group_id), ("example", row.example_id),
                                 ("prompt", row.prompt_sha256)):
                key = (label, value)
                require(key not in seen or seen[key] == bank.role, f"{label} leakage across roles")
                seen[key] = bank.role


def capture_digest(bank: Capture) -> str:
    """Hash row identity, site and raw tensor bytes without a whole-buffer bytes copy."""
    bank.validate()
    h = hashlib.sha256()
    h.update(digest(bank.contract).encode())
    h.update(str(bank.values.dtype).encode())
    data = memoryview(bank.values.contiguous().reshape(-1).view(torch.uint8).numpy()).cast("B")
    for start in range(0, len(data), 1048576):
        h.update(data[start:start + 1048576])
    return h.hexdigest()
