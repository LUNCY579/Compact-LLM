from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from consensus_lab.io import canonical_json_sha256
from consensus_lab.models.committee import ExpertSpec
from consensus_lab.paths import LabPaths
from consensus_lab.voting.rules import METHODS


SPECIAL_DECODING_METHODS = {"adaptive_hard_plurality", "sequence_medoid"}
EOS_POLICIES = {"ordinary", "strict_majority", "confidence_threshold", "minimum_tokens"}


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
    expert_names = [expert.name for expert in experts]
    if len(expert_names) != len(set(expert_names)):
        raise ValueError("Expert names must be unique")
    known = {expert.name for expert in experts}
    committees: list[CommitteeSpec] = []
    for packet in raw.get("committees") or []:
        names = [str(value) for value in packet["experts"]]
        if not names:
            raise ValueError(f"Committee {packet['name']} is empty")
        if len(names) != len(set(names)):
            raise ValueError(f"Committee {packet['name']} contains duplicate experts")
        missing = sorted(set(names) - known)
        if missing:
            raise ValueError(f"Committee {packet['name']} references missing experts: {missing}")
        committees.append(
            CommitteeSpec(name=str(packet["name"]), experts=names, role=str(packet.get("role") or "experimental"))
        )
    if not committees:
        raise ValueError("At least one committee is required")
    committee_names = [committee.name for committee in committees]
    if len(committee_names) != len(set(committee_names)):
        raise ValueError("Committee names must be unique")
    datasets = {
        name: paths.assert_official_source(paths.official_root / relative)
        for name, relative in (raw.get("datasets") or {}).items()
    }
    if not datasets:
        raise ValueError("At least one dataset is required")
    methods = [str(value) for value in raw.get("methods") or ["hard_plurality"]]
    if len(methods) != len(set(methods)):
        raise ValueError("Experiment methods must be unique")
    unknown_methods = sorted(set(methods) - METHODS - SPECIAL_DECODING_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown experiment methods: {unknown_methods}")
    eos_policy = str(raw.get("eos_policy", "ordinary"))
    if eos_policy not in EOS_POLICIES:
        raise ValueError(f"Unknown EOS policy: {eos_policy!r}")
    max_sequence_length = int(raw.get("max_sequence_length", 512))
    max_new_tokens = int(raw.get("max_new_tokens", 32))
    if max_sequence_length <= 0 or max_new_tokens <= 0:
        raise ValueError("max_sequence_length and max_new_tokens must be positive")
    adaptive_core = [str(value) for value in raw.get("adaptive_core_experts") or []]
    if "adaptive_hard_plurality" in methods and not adaptive_core:
        raise ValueError("adaptive_hard_plurality requires adaptive_core_experts")
    missing_adaptive = sorted(set(adaptive_core) - known)
    if missing_adaptive:
        raise ValueError(f"Adaptive core references missing experts: {missing_adaptive}")
    for committee in committees:
        if "adaptive_hard_plurality" in methods and not set(adaptive_core) <= set(committee.experts):
            raise ValueError(f"Committee {committee.name} does not contain every adaptive core expert")
    return ExperimentConfig(
        name=str(raw["name"]),
        backbone=backbone,
        experts=experts,
        committees=committees,
        methods=methods,
        datasets=datasets,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
        cache_mode=cache_mode,
        eos_policy=eos_policy,
        eos_confidence_threshold=float(raw.get("eos_confidence_threshold", 0.5)),
        minimum_tokens=int(raw.get("minimum_tokens", 1)),
        adaptive_core_experts=adaptive_core,
        adaptive_agreement_threshold=float(raw.get("adaptive_agreement_threshold", 2 / 3 + 1e-6)),
        seed=int(raw.get("seed", 20260811)),
        raw=raw,
        config_sha256=canonical_json_sha256(raw),
    )


def filter_committees_by_size(config: ExperimentConfig, committee_size: int) -> ExperimentConfig:
    """Return a scientifically distinct config for a CLI committee-size filter."""
    if committee_size <= 0:
        raise ValueError("committee_size must be positive")
    selected = [committee for committee in config.committees if len(committee.experts) == committee_size]
    if not selected:
        raise ValueError(f"No configured committee has K={committee_size}")
    selected_names = {committee.name for committee in selected}
    raw = copy.deepcopy(config.raw)
    raw["committees"] = [
        packet for packet in (raw.get("committees") or []) if str(packet.get("name")) in selected_names
    ]
    return replace(
        config,
        committees=selected,
        raw=raw,
        config_sha256=canonical_json_sha256(raw),
    )
