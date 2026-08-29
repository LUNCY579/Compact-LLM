#!/usr/bin/env python3
"""Transactionally publish the 48/46/45/42 Rineya family systems."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VAULT = Path(__file__).resolve().parent
SYSTEMS_ROOT = VAULT / "systems"
CATALOG = VAULT / "CATALOG.json"
SYSTEM_CATALOG = SYSTEMS_ROOT / "SYSTEM_CATALOG.json"
SYSTEMS_README = SYSTEMS_ROOT / "README.md"
RUNTIME_SOURCE = VAULT / "_family_system_runtime.py"
VERSION = "2.0.0-devfreeze"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def relative(path: str) -> Path:
    return REPO / Path(path.replace("/", os.sep))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "source_path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class EvidenceSpec:
    hits: int
    items: int
    inputs: dict[str, str]
    output: str
    report: str


@dataclass(frozen=True)
class SystemSpec:
    codename: str
    display_name: str
    version: str
    directory: str
    system_id: str
    kind: str
    arm_count: int
    results: dict[str, tuple[int, int]]
    routing: dict[str, Any]
    arms: dict[str, dict[str, Any]]
    evidence: dict[str, EvidenceSpec]
    source_config: str | None
    assets: dict[str, str]
    caveat: str


SPECS = (
    SystemSpec(
        codename="Supa",
        display_name="Rineya Supa 48",
        version="2.0.0-system",
        directory="Rineya-Supa-2.0.0-system48",
        system_id="rineya.system.supa.v2.0.0.system48.lcr-nine-arm-consensus",
        kind="multi_model_prediction_cascade",
        arm_count=9,
        results={
            "showcase64": (48, 64),
            "stability192": (128, 192),
            "all256": (176, 256),
        },
        routing={
            "initial_arm": "nimbus_low",
            "steps": [
                {"op": "select_if_arm_unknown", "arm": "nimbus_high"},
                {"op": "select_if_arm_unknown", "arm": "pulse_strong"},
                {
                    "op": "consensus_replace",
                    "left": "halo_uncensored",
                    "right": "nimbus_mid",
                },
                {
                    "op": "consensus_replace",
                    "left": "pulse_base",
                    "right": "pulse_step250",
                },
                {"op": "consensus_replace", "left": "halo_low", "right": "halo_high"},
            ],
        },
        arms={
            "nimbus_low": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 0.8125,
            },
            "nimbus_high": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 1.25,
            },
            "pulse_strong": {
                "base": "Rineya-Pulse-1.6.0-devfreeze",
                "adapter": "pulse_negative_strong",
                "scale": 1.0,
            },
            "halo_uncensored": {"base": "Rineya-Halo-Uncensored-1.5.0-devfreeze"},
            "nimbus_mid": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 1.0625,
            },
            "pulse_base": {"base": "Rineya-Pulse-1.6.0-devfreeze"},
            "pulse_step250": {
                "base": "Rineya-Pulse-1.6.0-devfreeze",
                "adapter": "step250",
                "scale": 0.75,
            },
            "halo_low": {
                "base": "Rineya-Halo-1.2.0-devfreeze",
                "adapter": "step200",
                "scale": 0.25,
            },
            "halo_high": {
                "base": "Rineya-Halo-1.2.0-devfreeze",
                "adapter": "step200",
                "scale": 0.75,
            },
        },
        evidence={
            "showcase64": EvidenceSpec(
                48,
                64,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale0p8125_showcase64.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale1p25_showcase64.jsonl",
                    "pulse_strong": "artifacts/rineya_published_core_search_v1/negative_vectors/pulse_neg_200_250_1p5_m0p5_showcase64.jsonl",
                    "halo_uncensored": "artifacts/rineya_published_core_search_v1/baselines/halo_uncensored_showcase64.jsonl",
                    "nimbus_mid": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale1p0625_showcase64.jsonl",
                    "pulse_base": "artifacts/rineya_published_core_search_v1/baselines/pulse_showcase64.jsonl",
                    "pulse_step250": "artifacts/rineya_published_core_search_v1/checkpoint_triage/pulse_step250/pulse_step250_scale0p75.jsonl",
                    "halo_low": "artifacts/rineya_published_core_search_v1/scale_grids/halo/halo_scale0p25.jsonl",
                    "halo_high": "artifacts/rineya_published_core_search_v1/scale_grids/halo/halo_scale0p75.jsonl",
                },
                "artifacts/rineya_published_core_search_v1/finalists/nimbus_pulse_consensus_cascade_showcase48.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/consensus_cascade_showcase_report.json",
            ),
            "stability192": EvidenceSpec(
                128,
                192,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale0p8125_stability192.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale1p25_stability192.jsonl",
                    "pulse_strong": "artifacts/rineya_published_core_search_v1/consensus48_stability/pulse_strong_negative_stability192.jsonl",
                    "halo_uncensored": "artifacts/rineya_published_core_search_v1/baselines/halo_uncensored_stability192.jsonl",
                    "nimbus_mid": "artifacts/rineya_published_core_search_v1/consensus48_stability/nimbus_scale1p0625_stability192.jsonl",
                    "pulse_base": "artifacts/rineya_published_core_search_v1/baselines/pulse_stability192.jsonl",
                    "pulse_step250": "artifacts/rineya_published_core_search_v1/consensus48_stability/pulse_step250_scale0p75_stability192.jsonl",
                    "halo_low": "artifacts/rineya_published_core_search_v1/consensus48_stability/halo_scale0p25.jsonl",
                    "halo_high": "artifacts/rineya_published_core_search_v1/consensus48_stability/halo_scale0p75.jsonl",
                },
                "artifacts/rineya_published_core_search_v1/consensus48_stability/cascade_stability192.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/consensus_cascade_stability_report.json",
            ),
            "all256": EvidenceSpec(
                176,
                256,
                {},
                "artifacts/rineya_published_core_search_v1/finalists/nimbus_pulse_consensus_cascade_all256.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/nimbus_pulse_consensus_cascade_all256_report.json",
            ),
        },
        source_config="Rineya_Long_Context_Weight_Geometry_Lab_v1/configs/final/nimbus_pulse_consensus_cascade_showcase48.json",
        assets={
            "step200": "artifacts/rineya_long_context_weight_geometry_v1/selected/halo_stage_a_step200_adapter",
            "step250": "artifacts/halo_long_context_sft_v1/stage_a/checkpoint-250",
            "pulse_negative_strong": "artifacts/rineya_published_core_search_v1/negative_vectors/neg_200_250_1p5_m0p5",
        },
        caveat="Showcase selected the nine-arm ordering; Stability was frozen validation. This is a routed system, not one dense checkpoint.",
    ),
    SystemSpec(
        codename="Baya",
        display_name="Rineya Baya 46",
        version="2.0.0-system",
        directory="Rineya-Baya-2.0.0-system46",
        system_id="rineya.system.baya.v2.0.0.system46.lcr-three-arm-unknown",
        kind="multi_model_prediction_cascade",
        arm_count=3,
        results={
            "showcase64": (46, 64),
            "stability192": (125, 192),
            "all256": (171, 256),
        },
        routing={
            "initial_arm": "nimbus_low",
            "steps": [
                {"op": "select_if_arm_unknown", "arm": "nimbus_high"},
                {"op": "select_if_arm_unknown", "arm": "pulse_strong"},
            ],
        },
        arms={
            "nimbus_low": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 0.8125,
            },
            "nimbus_high": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 1.25,
            },
            "pulse_strong": {
                "base": "Rineya-Pulse-1.6.0-devfreeze",
                "adapter": "pulse_negative_strong",
                "scale": 1.0,
            },
        },
        evidence={
            "showcase64": EvidenceSpec(
                46,
                64,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale0p8125_showcase64.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale1p25_showcase64.jsonl",
                    "pulse_strong": "artifacts/rineya_published_core_search_v1/negative_vectors/pulse_neg_200_250_1p5_m0p5_showcase64.jsonl",
                },
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_showcase64.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_showcase_report.json",
            ),
            "stability192": EvidenceSpec(
                125,
                192,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale0p8125_stability192.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale1p25_stability192.jsonl",
                    "pulse_strong": "artifacts/rineya_published_core_search_v1/consensus48_stability/pulse_strong_negative_stability192.jsonl",
                },
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_stability192.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_stability_report.json",
            ),
            "all256": EvidenceSpec(
                171,
                256,
                {},
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_all256.jsonl",
                "artifacts/rineya_published_core_search_v1/finalists/pulse_strong_over_nimbus_gate_all256_report.json",
            ),
        },
        source_config="Rineya_Long_Context_Weight_Geometry_Lab_v1/configs/final/pulse_strong_over_nimbus_gate46.json",
        assets={
            "step200": "artifacts/rineya_long_context_weight_geometry_v1/selected/halo_stage_a_step200_adapter",
            "pulse_negative_strong": "artifacts/rineya_published_core_search_v1/negative_vectors/neg_200_250_1p5_m0p5",
        },
        caveat="Showcase selected the three-arm cascade; Stability was frozen validation. This is a routed system, not one dense checkpoint.",
    ),
    SystemSpec(
        codename="Nimbus DualScale PEFT Gate45",
        display_name="Rineya Nimbus DualScale PEFT Gate 45",
        version="1.0.0-system",
        directory="Rineya-Nimbus-DualScale-1.0.0-system45",
        system_id="rineya.system.nimbus-dualscale.v1.0.0.system45.peft-unknown-gate",
        kind="two_arm_peft_unknown_gate",
        arm_count=2,
        results={
            "showcase64": (45, 64),
            "stability192": (124, 192),
            "all256": (169, 256),
        },
        routing={
            "initial_arm": "nimbus_low",
            "steps": [{"op": "select_if_arm_unknown", "arm": "nimbus_high"}],
        },
        arms={
            "nimbus_low": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 0.8125,
            },
            "nimbus_high": {
                "base": "Rineya-Nimbus-1.6.0-devfreeze",
                "adapter": "step200",
                "scale": 1.25,
            },
        },
        evidence={
            "showcase64": EvidenceSpec(
                45,
                64,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale0p8125_showcase64.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/nimbus_scale/nimbus_step200_scale1p25_showcase64.jsonl",
                },
                "artifacts/rineya_long_context_weight_geometry_v1/ensembles/nimbus_scale0p8125_1p25_unknown_gate_showcase64.jsonl",
                "artifacts/rineya_long_context_weight_geometry_v1/ensembles/nimbus_scale0p8125_1p25_unknown_gate_showcase64_report.json",
            ),
            "stability192": EvidenceSpec(
                124,
                192,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale0p8125_stability192.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_step200_scale1p25_stability192.jsonl",
                },
                "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_scale0p8125_1p25_unknown_gate_stability192.jsonl",
                "artifacts/rineya_long_context_weight_geometry_v1/stability_finalists/nimbus_scale0p8125_1p25_unknown_gate_stability192_report.json",
            ),
            "all256": EvidenceSpec(
                169,
                256,
                {
                    "nimbus_low": "artifacts/rineya_long_context_weight_geometry_v1/final/nimbus_step200_scale0p8125_all256.jsonl",
                    "nimbus_high": "artifacts/rineya_long_context_weight_geometry_v1/final/nimbus_step200_scale1p25_all256.jsonl",
                },
                "artifacts/rineya_long_context_weight_geometry_v1/final/nimbus_scale0p8125_1p25_unknown_gate_all256.jsonl",
                "artifacts/rineya_long_context_weight_geometry_v1/final/nimbus_scale0p8125_1p25_unknown_gate_all256_report.json",
            ),
        },
        source_config="Rineya_Long_Context_Weight_Geometry_Lab_v1/configs/final/nimbus_two_scale_unknown_gate.json",
        assets={
            "step200": "artifacts/rineya_long_context_weight_geometry_v1/selected/halo_stage_a_step200_adapter",
        },
        caveat="The 45 score requires two PEFT executions. Dense materialization changes predictions and does not inherit this result.",
    ),
    SystemSpec(
        codename="Supa-Baya Dense Gate42",
        display_name="Rineya Supa-Baya Dense Gate 42",
        version="1.0.0-system",
        directory="Rineya-Supa-Baya-DenseGate-1.0.0-system42",
        system_id="rineya.system.supa-baya.v1.0.0.system42.dense-unknown-gate",
        kind="two_dense_core_unknown_gate",
        arm_count=2,
        results={
            "showcase64": (42, 64),
            "stability192": (124, 192),
            "all256": (166, 256),
        },
        routing={
            "initial_arm": "supa_dense_low",
            "steps": [{"op": "select_if_arm_unknown", "arm": "baya_dense_high"}],
        },
        arms={
            "supa_dense_low": {"base": "Rineya-Supa-1.7.0-devfreeze"},
            "baya_dense_high": {"base": "Rineya-Baya-1.7.0-devfreeze"},
        },
        evidence={
            "all256": EvidenceSpec(
                166,
                256,
                {
                    "supa_dense_low": "artifacts/rineya_long_context_weight_geometry_v1/published_dense_parity_v2/taria/taria_dense_all256.jsonl",
                    "baya_dense_high": "artifacts/rineya_long_context_weight_geometry_v1/published_dense_parity_v2/lumina/lumina_dense_all256.jsonl",
                },
                "artifacts/rineya_published_core_search_v1/published_pair/taria_lumina_unknown_gate_all256.jsonl",
                "artifacts/rineya_published_core_search_v1/published_pair/taria_lumina_unknown_gate_all256_report.json",
            ),
        },
        source_config=None,
        assets={},
        caveat="This 42 result is the answer-blind gate over the two preserved dense 1.7 cores. It is distinct from the coincidental 42-point single PEFT arm.",
    ),
)


def load_runtime_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rineya_family_runtime", RUNTIME_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load family runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_adapter(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "soup_manifest.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    if (
        not (destination / "adapter_config.json").is_file()
        or not (destination / "adapter_model.safetensors").is_file()
    ):
        raise RuntimeError(f"Incomplete adapter source: {source}")


def base_record(directory: str) -> dict[str, Any]:
    path = VAULT / directory
    core = load_json(path / "CORE.json")
    weight = path / "model.safetensors"
    return {
        "directory": directory,
        "core_id": core["core_id"],
        "weight_sha256": sha256_file(weight),
        "weight_bytes": weight.stat().st_size,
    }


def render_readme(system: dict[str, Any]) -> str:
    scores = system["results"]
    return f"""# {system["display_name"]}

Status: **Showcase-selected, Stability-validated family system**.

This directory is a runnable multi-model system bundle. It is **not** a single
`model.safetensors` checkpoint. The measured score depends on the exact
answer-blind routing rule across {system["arm_count"]} inference arms.

- Showcase (selection): **{scores["showcase64"]["hits"]}/{scores["showcase64"]["items"]}**
- Stability (frozen validation): **{scores["stability192"]["hits"]}/{scores["stability192"]["items"]}**
- Complete public Demo: **{scores["all256"]["hits"]}/{scores["all256"]["items"]}**

Run `python run_family_system.py --replay <split>` to replay bundled prediction
arms and verify the canonical output hash. See `SYSTEM.json` for live arm base,
adapter, scale, hashes, and routing order.

{system["caveat"]}
"""


def build_system(spec: SystemSpec, staging: Path, created_at: str) -> dict[str, Any]:
    runtime = load_runtime_module()
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copy2(RUNTIME_SOURCE, staging / "run_family_system.py")
    shutil.copy2(
        VAULT / "Rineya-Nimbus-1.6.0-devfreeze" / "LICENSE", staging / "LICENSE"
    )
    (staging / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )

    assets: dict[str, Any] = {}
    for label, source_name in spec.assets.items():
        source = relative(source_name)
        destination = staging / "assets" / label
        copy_adapter(source, destination)
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(destination.iterdir())
            if path.is_file()
        }
        assets[label] = {
            "source": source_name,
            "bundled_path": f"assets/{label}",
            "files": inventory,
        }

    arms: dict[str, Any] = {}
    for label, arm in spec.arms.items():
        record = copy.deepcopy(arm)
        record["base_model"] = base_record(str(record.pop("base")))
        if "adapter" in record:
            asset_label = str(record["adapter"])
            record["adapter"] = {
                "asset": asset_label,
                "bundled_path": assets[asset_label]["bundled_path"],
                "weight_sha256": assets[asset_label]["files"][
                    "adapter_model.safetensors"
                ]["sha256"],
            }
        arms[label] = record

    evidence: dict[str, Any] = {}
    system_stub = {"system_id": spec.system_id, "routing": spec.routing}
    for split, evidence_spec in spec.evidence.items():
        input_records: dict[str, Any] = {}
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        for arm, source_name in evidence_spec.inputs.items():
            source = relative(source_name)
            destination = staging / "evidence" / "inputs" / split / f"{arm}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            input_records[arm] = {
                **file_record(source),
                "bundled_path": destination.relative_to(staging).as_posix(),
            }
            arm_rows[arm] = runtime.load_jsonl(source)
        expected_source = relative(evidence_spec.output)
        expected_rows = runtime.load_jsonl(expected_source)
        expected_hash = runtime.canonical_prediction_hash(expected_rows)
        replayable = bool(arm_rows)
        if replayable:
            routed = runtime.route_predictions(system_stub, arm_rows)
            routed_hash = runtime.canonical_prediction_hash(routed)
            if routed_hash != expected_hash:
                raise RuntimeError(f"{spec.codename}/{split}: replay mismatch")
        expected_destination = staging / "evidence" / "expected" / f"{split}.jsonl"
        expected_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expected_source, expected_destination)
        report_source = relative(evidence_spec.report)
        report_destination = staging / "evidence" / "expected" / f"{split}_report.json"
        shutil.copy2(report_source, report_destination)
        evidence[split] = {
            "hits": evidence_spec.hits,
            "items": evidence_spec.items,
            "inputs": input_records,
            "expected_output": {
                **file_record(expected_source),
                "bundled_path": expected_destination.relative_to(staging).as_posix(),
            },
            "report": {
                **file_record(report_source),
                "bundled_path": report_destination.relative_to(staging).as_posix(),
            },
            "canonical_prediction_sha256": expected_hash,
            "replayable": replayable,
            "replay_verified": replayable,
        }

    source_config: dict[str, Any] | None = None
    if spec.source_config is not None:
        source = relative(spec.source_config)
        destination = staging / "recipe" / "source_config.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_config = {
            **file_record(source),
            "bundled_path": destination.relative_to(staging).as_posix(),
        }

    results = {
        split: {"hits": hits, "items": items, "strict_exact": hits / items}
        for split, (hits, items) in spec.results.items()
    }
    system = {
        "schema_version": "2.0",
        "system_id": spec.system_id,
        "codename": spec.codename,
        "display_name": spec.display_name,
        "directory_name": spec.directory,
        "version": spec.version,
        "kind": spec.kind,
        "classification": "showcase_selected_stability_validated",
        "created_at_utc": created_at,
        "single_checkpoint": False,
        "arm_count": spec.arm_count,
        "answer_key_access_at_inference": False,
        "routing": spec.routing,
        "arms": arms,
        "assets": assets,
        "results": results,
        "evidence": evidence,
        "source_config": source_config,
        "protocol": {
            "dtype": "bfloat16",
            "decoding": "greedy",
            "seed": 17,
            "thinking": False,
            "max_new_tokens": 24,
            "max_context": 4096,
            "truncation": "forbidden",
        },
        "caveat": spec.caveat,
    }
    system["system_manifest_sha256"] = canonical_json_hash(system)
    atomic_json(staging / "SYSTEM.json", system)
    core = {
        "schema_version": "2.0",
        "core_id": spec.system_id,
        "codename": spec.codename,
        "directory_name": spec.directory,
        "version": spec.version,
        "kind": spec.kind,
        "status": "showcase_selected_stability_validated",
        "public_claim": (
            f"{spec.display_name}: {results['showcase64']['hits']}/64 Showcase "
            f"(selection), {results['stability192']['hits']}/192 Stability "
            f"(frozen validation), {results['all256']['hits']}/256 overall. "
            "Routed family system; not one dense checkpoint."
        ),
        "system_manifest_sha256": system["system_manifest_sha256"],
        "results": results,
        "arm_count": spec.arm_count,
        "single_checkpoint": False,
        "created_at_utc": created_at,
    }
    atomic_json(staging / "SYSTEM_CARD.json", core)
    atomic_write(staging / "README.md", render_readme(system))
    inventory = {
        path.relative_to(staging).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "BUNDLE.json"
    }
    bundle = {
        "schema_version": "1.0",
        "system_id": spec.system_id,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(record["bytes"] for record in inventory.values()),
    }
    bundle["bundle_sha256"] = canonical_json_hash(bundle)
    atomic_json(staging / "BUNDLE.json", bundle)
    return core


def verify_directory(path: Path, spec: SystemSpec) -> None:
    runtime = load_runtime_module()
    system = load_json(path / "SYSTEM.json")
    declared = system.pop("system_manifest_sha256")
    if declared != canonical_json_hash(system):
        raise RuntimeError(f"{spec.codename}: SYSTEM self-hash mismatch")
    system["system_manifest_sha256"] = declared
    if system["system_id"] != spec.system_id or system["arm_count"] != spec.arm_count:
        raise RuntimeError(f"{spec.codename}: system identity mismatch")
    for split, record in system["evidence"].items():
        if record.get("replayable"):
            runtime.replay(path, split, None)
    bundle = load_json(path / "BUNDLE.json")
    declared_bundle = bundle.pop("bundle_sha256")
    if declared_bundle != canonical_json_hash(bundle):
        raise RuntimeError(f"{spec.codename}: bundle self-hash mismatch")
    for name, record in bundle["files"].items():
        target = path / name
        if (
            target.stat().st_size != record["bytes"]
            or sha256_file(target) != record["sha256"]
        ):
            raise RuntimeError(f"{spec.codename}: bundle file mismatch: {name}")


def transform_catalog(
    catalog: dict[str, Any], cores: dict[str, dict[str, Any]], now: str
) -> dict[str, Any]:
    result = copy.deepcopy(catalog)
    active = result.get("cores")
    if not isinstance(active, list):
        raise TypeError("Catalog cores must be a list")
    retired_directories = {
        "Rineya-Supa-1.7.0-devfreeze": cores["Supa"]["core_id"],
        "Rineya-Baya-1.7.0-devfreeze": cores["Baya"]["core_id"],
    }
    retained: list[dict[str, Any]] = []
    superseded = list(result.get("superseded") or [])
    for entry in active:
        if entry.get("directory") in retired_directories:
            archived = copy.deepcopy(entry)
            archived["superseded_at_utc"] = now
            archived["superseded_by"] = retired_directories[entry["directory"]]
            archived["retained_present"] = True
            archived["retained_role"] = "dense dependency anchor for the 42 system"
            superseded.append(archived)
        else:
            retained.append(entry)
    systems = list(result.get("systems") or [])
    existing_ids = {entry.get("system_id") for entry in systems}
    for spec in SPECS:
        if spec.system_id in existing_ids:
            raise RuntimeError(f"System already in catalog: {spec.system_id}")
        systems.append(
            {
                "system_id": spec.system_id,
                "codename": spec.codename,
                "display_name": spec.display_name,
                "directory": spec.directory,
                "present": True,
                "kind": spec.kind,
                "core": cores[spec.codename],
            }
        )
    result["cores"] = retained
    result["systems"] = systems
    result["superseded"] = superseded
    result["updated_at_utc"] = now
    result["last_written"] = [spec.directory for spec in SPECS]
    return result


def system_catalog(cards: dict[str, dict[str, Any]], now: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "vault": "Rineya_Published_Cores/systems",
        "dense_core_catalog": "../CATALOG.json",
        "updated_at_utc": now,
        "systems": [
            {
                "system_id": spec.system_id,
                "codename": spec.codename,
                "display_name": spec.display_name,
                "directory": spec.directory,
                "kind": spec.kind,
                "present": True,
                "system_card": cards[spec.codename],
            }
            for spec in SPECS
        ],
        "naming": {
            "Supa": "48/64 nine-arm system",
            "Baya": "46/64 three-arm system",
            "lower_tiers": [
                "Nimbus DualScale PEFT Gate45",
                "Supa-Baya Dense Gate42",
            ],
        },
        "legacy_dense_anchors": [
            {
                "directory": "../Rineya-Supa-1.7.0-devfreeze",
                "weight_sha256": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
                "actual_showcase_hits": 40,
            },
            {
                "directory": "../Rineya-Baya-1.7.0-devfreeze",
                "weight_sha256": "1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba",
                "actual_showcase_hits": 40,
            },
        ],
    }


def render_systems_readme() -> str:
    return """# Rineya published family systems

These are versioned, runnable inference systems backed up alongside the dense
core vault. They are not single checkpoints, because their measured scores
depend on answer-blind routing across multiple model predictions.

| System | Form | Showcase | Stability | All 256 |
|---|---|---:|---:|---:|
| **Supa 2.0 System48** | 9-arm ordered consensus cascade | **48/64** | **128/192** | **176/256** |
| **Baya 2.0 System46** | 3-arm UNKNOWN cascade | **46/64** | **125/192** | **171/256** |
| Nimbus DualScale PEFT Gate45 | 2 PEFT executions | **45/64** | **124/192** | **169/256** |
| Supa-Baya Dense Gate42 | 2 dense-core executions | **42/64** | **124/192** | **166/256** |

Supa Core 1.7 and Baya Core 1.7 remain immutable 40/64 dense dependency
anchors. Their weights are not relabeled with system scores. Each system bundle
contains `SYSTEM.json`, adapter assets where needed, frozen evidence, hashes,
and `run_family_system.py` for prediction-only replay.

Showcase64 was selection data. Stability192 was frozen validation. Sealed TEST
was not run.
"""


def apply_publication() -> None:
    if len(SPECS) != 4:
        raise AssertionError("Family specification must contain four systems")
    SYSTEMS_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        if (SYSTEMS_ROOT / spec.directory).exists():
            raise FileExistsError(SYSTEMS_ROOT / spec.directory)
    catalog_original = (
        SYSTEM_CATALOG.read_text(encoding="utf-8") if SYSTEM_CATALOG.is_file() else None
    )
    readme_original = (
        SYSTEMS_README.read_text(encoding="utf-8") if SYSTEMS_README.is_file() else None
    )
    root_catalog_original = CATALOG.read_text(encoding="utf-8")
    now = utc_now()
    staging_paths: dict[str, Path] = {}
    promoted: list[SystemSpec] = []
    try:
        cards: dict[str, dict[str, Any]] = {}
        for spec in SPECS:
            staging = SYSTEMS_ROOT / f".{spec.directory}.staging-{uuid.uuid4().hex}"
            staging_paths[spec.codename] = staging
            cards[spec.codename] = build_system(spec, staging, now)
            verify_directory(staging, spec)
        for spec in SPECS:
            os.replace(staging_paths[spec.codename], SYSTEMS_ROOT / spec.directory)
            promoted.append(spec)
        atomic_json(SYSTEM_CATALOG, system_catalog(cards, now))
        atomic_write(SYSTEMS_README, render_systems_readme())
        root_catalog = json.loads(root_catalog_original)
        root_catalog["system_catalog"] = "systems/SYSTEM_CATALOG.json"
        root_catalog["frontier_systems"] = {
            "Supa": SPECS[0].system_id,
            "Baya": SPECS[1].system_id,
            "showcase_hits": [48, 46, 45, 42],
        }
        root_catalog["updated_at_utc"] = now
        atomic_json(CATALOG, root_catalog)
        for spec in SPECS:
            verify_directory(SYSTEMS_ROOT / spec.directory, spec)
    except Exception:
        atomic_write(CATALOG, root_catalog_original)
        if catalog_original is None:
            if SYSTEM_CATALOG.exists():
                SYSTEM_CATALOG.unlink()
        else:
            atomic_write(SYSTEM_CATALOG, catalog_original)
        if readme_original is None:
            if SYSTEMS_README.exists():
                SYSTEMS_README.unlink()
        else:
            atomic_write(SYSTEMS_README, readme_original)
        for spec in reversed(promoted):
            final = SYSTEMS_ROOT / spec.directory
            if final.exists():
                os.replace(final, staging_paths[spec.codename])
        for staging in staging_paths.values():
            if staging.exists():
                shutil.rmtree(staging)
        raise


def status() -> dict[str, Any]:
    return {
        spec.codename: {
            "directory": spec.directory,
            "exists": (SYSTEMS_ROOT / spec.directory).is_dir(),
            "system_id": spec.system_id,
        }
        for spec in SPECS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "apply", "verify", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.command == "preflight":
        for spec in SPECS:
            if (SYSTEMS_ROOT / spec.directory).exists():
                raise FileExistsError(SYSTEMS_ROOT / spec.directory)
            for asset in spec.assets.values():
                if not relative(asset).is_dir():
                    raise FileNotFoundError(relative(asset))
            for evidence in spec.evidence.values():
                for source in [
                    *evidence.inputs.values(),
                    evidence.output,
                    evidence.report,
                ]:
                    if not relative(source).is_file():
                        raise FileNotFoundError(relative(source))
        print(json.dumps({"status": "preflight_pass", "systems": status()}, indent=2))
        return 0
    if args.command == "verify":
        for spec in SPECS:
            verify_directory(SYSTEMS_ROOT / spec.directory, spec)
        print(json.dumps({"status": "verified", "systems": status()}, indent=2))
        return 0
    apply_publication()
    print(json.dumps({"status": "published", "systems": status()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
