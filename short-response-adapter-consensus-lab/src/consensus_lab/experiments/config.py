from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from consensus_lab.io import canonical_json_sha256
from consensus_lab.models.committee import ExpertSpec
from consensus_lab.paths import LabPaths


@dataclass(frozen=True)
class CommitteeSpec:
    name: str
    experts: list[str]
    role: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    backbone: Path
    experts: list[ExpertSpec]
    committees: list[CommitteeSpec]
    methods: list[str]
    datasets: dict[str, Path]
    max_sequence_length: int
    max_new_tokens: int
    cache_mode: str
    eos_policy: str
    eos_confidence_threshold: float
    minimum_tokens: int
    adaptive_core_experts: list[str]
    adaptive_agreement_threshold: float
    seed: int
    raw: dict[str, Any]
    config_sha256: str


def load_experiment_config(path: Path, paths: LabPaths) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cache_mode = str(raw.get("cache_mode", "reference"))
    if cache_mode not in {"reference", "separate_cache"}:
        raise ValueError(f"Unknown cache_mode {cache_mode!r}")
    backbone = paths.assert_official_source(paths.official_root / Path(raw["backbone"]))
    experts: list[ExpertSpec] = []
    for packet in raw.get("experts") or []:
        adapter_value = packet.get("adapter")
        adapter = None if adapter_value is None else paths.assert_official_source(paths.official_root / adapter_value)
        experts.append(
            ExpertSpec(
                name=str(packet["name"]),
                adapter_path=adapter,
                validation_weight=float(packet.get("validation_weight", 1.0)),
                temperature=float(packet.get("temperature", 1.0)),
                logit_scale=float(packet.get("logit_scale", 1.0)),
            )
        )
    known = {expert.name for expert in experts}
    committees: list[CommitteeSpec] = []
    for packet in raw.get("committees") or []:
        names = [str(value) for value in packet["experts"]]
        missing = sorted(set(names) - known)
        if missing:
            raise ValueError(f"Committee {packet['name']} references missing experts: {missing}")
        committees.append(
            CommitteeSpec(name=str(packet["name"]), experts=names, role=str(packet.get("role") or "experimental"))
        )
    datasets = {
        name: paths.assert_official_source(paths.official_root / relative)
        for name, relative in (raw.get("datasets") or {}).items()
    }
    adaptive_core = [str(value) for value in raw.get("adaptive_core_experts") or []]
    if "adaptive_hard_plurality" in (raw.get("methods") or []) and not adaptive_core:
        raise ValueError("adaptive_hard_plurality requires adaptive_core_experts")
    missing_adaptive = sorted(set(adaptive_core) - known)
    if missing_adaptive:
        raise ValueError(f"Adaptive core references missing experts: {missing_adaptive}")
    for committee in committees:
        if "adaptive_hard_plurality" in (raw.get("methods") or []) and not set(adaptive_core) <= set(committee.experts):
            raise ValueError(f"Committee {committee.name} does not contain every adaptive core expert")
    return ExperimentConfig(
        name=str(raw["name"]),
        backbone=backbone,
        experts=experts,
        committees=committees,
        methods=[str(value) for value in raw.get("methods") or ["hard_plurality"]],
        datasets=datasets,
        max_sequence_length=int(raw.get("max_sequence_length", 512)),
        max_new_tokens=int(raw.get("max_new_tokens", 32)),
        cache_mode=cache_mode,
        eos_policy=str(raw.get("eos_policy", "ordinary")),
        eos_confidence_threshold=float(raw.get("eos_confidence_threshold", 0.5)),
        minimum_tokens=int(raw.get("minimum_tokens", 1)),
        adaptive_core_experts=adaptive_core,
        adaptive_agreement_threshold=float(raw.get("adaptive_agreement_threshold", 2 / 3 + 1e-6)),
        seed=int(raw.get("seed", 20260811)),
        raw=raw,
        config_sha256=canonical_json_sha256(raw),
    )
