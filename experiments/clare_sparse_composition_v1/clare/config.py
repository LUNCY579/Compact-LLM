"""The registered CLARE-RINEYA-001 configuration: one JSON file, validated and hashed.

Every run artifact records ``config_sha256`` so a result can never be joined to a different
registration. Changing any registered value is an AMENDMENT (new file name, dated), not an edit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .allocation import ARMS
from .data import ALL_SIGNATURES, SKILLS
from .footprint import Footprint, qwen3_mlp_footprint
from .io import canonical_json, read_json
from .recipe import Recipe

EXP = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXP / "configs" / "clare_rineya_001.json"


@dataclass(frozen=True)
class StudyConfig:
    raw: dict
    sha256: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "StudyConfig":
        raw = read_json(path)
        cfg = cls(raw, hashlib.sha256(canonical_json(raw).encode("ascii")).hexdigest())
        cfg.validate()
        return cfg

    # -- accessors ----------------------------------------------------------------------
    @property
    def experiment(self) -> str:
        return self.raw["experiment"]

    def footprint(self) -> Footprint:
        f = self.raw["footprint"]
        if f["kind"] != "qwen3_mlp":
            raise ValueError(f"unsupported footprint kind {f['kind']!r}")
        fp = qwen3_mlp_footprint(range(f["first_layer"], f["last_layer"] + 1), f["hidden"], f["intermediate"],
                                 tuple(f["modules"]))
        if fp.total != f["expected_total_coordinates"]:
            raise ValueError(f"footprint total {fp.total} != registered {f['expected_total_coordinates']}")
        return fp

    def recipe(self) -> Recipe:
        return Recipe(**self.raw["recipe"])

    def order(self, name: str) -> tuple[str, ...]:
        return tuple(self.raw["orders"][name])

    # -- validation -----------------------------------------------------------------------
    def validate(self) -> None:
        r = self.raw
        if r.get("experiment") != "CLARE-RINEYA-001":
            raise ValueError("wrong experiment id")
        if set(r["skills"]) != set(SKILLS):
            raise ValueError("skills must be exactly R/U/S/P")
        for name, order in r["orders"].items():
            if sorted(order) != sorted(r["skills"]):
                raise ValueError(f"order {name} is not a permutation of the skills")
        for arm in r["arms"]["primary"] + r["arms"]["secondary"] + r["arms"]["config_support_only"]:
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm}")
        ev = r["evaluation"]
        if ev["role"] != "dev":
            raise ValueError("CLARE-RINEYA-001 evaluates on DEV only (TEST/CHALLENGE are not readable)")
        if set(ev["signatures"]) != set(ALL_SIGNATURES):
            raise ValueError("evaluation must cover all 15 signatures")
        from .data import MANIFEST_SHA256
        if r["dataset"]["manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("dataset manifest pin differs from clare.data.MANIFEST_SHA256")
        if r["train"]["role"] != "train":
            raise ValueError("training reads the TRAIN role only")
        self.footprint()
        self.recipe()
