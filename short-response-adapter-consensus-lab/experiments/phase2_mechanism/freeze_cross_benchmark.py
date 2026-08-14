from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs" / "phase2_selected_dev_free_run.yaml"
SELECTION = ROOT / "artifacts" / "phase2_mechanism" / "selected_dev_free_run_manifest.json"
GATE = ROOT / "results" / "phase2_mechanism" / "cross_benchmark_verification.json"
FULL_CONFIG = ROOT / "configs" / "phase2_cross_v212_full_k7.yaml"
CONTROL_CONFIG = ROOT / "configs" / "phase2_cross_v212_controls.yaml"
MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "cross_v212_confirmation_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    selection = load_json(SELECTION)
    gate = load_json(GATE)
    if gate["status"] != "PASS_WITH_CAVEATS" or gate["checks_passed"] != gate["checks_total"]:
        raise RuntimeError(f"Cross-benchmark provenance gate is not usable: {gate['status']}")
    if selection["test_data_read_for_selection"]:
        raise RuntimeError("DEV-selected comparator manifest unexpectedly read TEST data")
    selected = set(selection["selected_methods"])
    required = {"hard_plurality", "probability_mean", "calibrated_probability_mean"}
    if not required <= selected:
        raise RuntimeError(f"Required fixed methods absent from DEV selection: {sorted(required - selected)}")

    dataset = {
        "v212_test": "data/phase_9_student_distillation/general_reasoning_v212/reviewed_splits/test.reviewed.jsonl"
    }
    full = {
        "name": "phase2_cross_v212_full_k7",
        "backbone": source["backbone"],
        "experts": source["experts"],
        "committees": [
            {
                "name": "full_k7",
                "experts": [expert["name"] for expert in source["experts"]],
                "role": "fixed_cross_corpus_transfer",
            }
        ],
        "methods": ["hard_plurality", "probability_mean", "calibrated_probability_mean"],
        "datasets": dataset,
        "max_sequence_length": source["max_sequence_length"],
        "max_new_tokens": source["max_new_tokens"],
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": source["seed"],
    }
    source_experts = {expert["name"]: expert for expert in source["experts"]}
    controls = {
        "name": "phase2_cross_v212_controls",
        "backbone": source["backbone"],
        "experts": [
            {"name": "backbone", "adapter": None, "validation_weight": 1.0, "temperature": 1.0},
            source_experts["track_a"],
            source_experts["abstention"],
        ],
        "committees": [
            {"name": "backbone_k1", "experts": ["backbone"], "role": "fixed_cross_corpus_control"},
            {"name": "track_a_k1", "experts": ["track_a"], "role": "fixed_cross_corpus_control"},
            {"name": "abstention_k1", "experts": ["abstention"], "role": "fixed_cross_corpus_control"},
        ],
        "methods": ["hard_plurality"],
        "datasets": dataset,
        "max_sequence_length": source["max_sequence_length"],
        "max_new_tokens": source["max_new_tokens"],
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": source["seed"],
    }
    FULL_CONFIG.write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    CONTROL_CONFIG.write_text(yaml.safe_dump(controls, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "status": "V212_CROSS_CORPUS_CONFIRMATION_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "FIXED-METHOD CROSS-CORPUS TRANSFER WITH PROVENANCE CAVEATS",
        "selection_manifest": str(SELECTION),
        "selection_manifest_sha256": sha256(SELECTION),
        "cross_gate": str(GATE),
        "cross_gate_sha256": sha256(GATE),
        "v212_test_sha256": gate["splits"]["test"]["sha256"],
        "v212_test_rows": gate["splits"]["test"]["rows"],
        "configs": {
            "full_k7": {
                "path": str(FULL_CONFIG),
                "file_sha256": sha256(FULL_CONFIG),
                "canonical_sha256": canonical_hash(full),
                "expected_rows": 1320 * 3,
            },
            "controls": {
                "path": str(CONTROL_CONFIG),
                "file_sha256": sha256(CONTROL_CONFIG),
                "canonical_sha256": canonical_hash(controls),
                "expected_rows": 1320 * 3,
            },
        },
        "planned_conditions": [
            "backbone_k1/hard_plurality",
            "track_a_k1/hard_plurality",
            "abstention_k1/hard_plurality",
            "full_k7/hard_plurality",
            "full_k7/probability_mean",
            "full_k7/calibrated_probability_mean",
        ],
        "planned_paired_family": [
            "hard_k7 vs backbone",
            "hard_k7 vs Track A",
            "hard_k7 vs abstention",
            "hard_k7 vs raw probability K7",
            "hard_k7 vs calibrated probability K7",
        ],
        "statistics": "paired bootstrap difference CI; exact McNemar; Holm correction over the five planned hard-K7 comparisons",
        "no_switching_rule": "Do not change methods, temperatures, controls, EOS, decoder, or comparison family after v2.1.2 outputs are inspected.",
        "outcome_data_read_before_freeze": False,
        "caveats": gate["caveats"],
        "claim_boundary": "Fixed-method cross-corpus transfer only; not globally untouched or template-independent holdout evidence; do not aggregate with v2.2.",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(MANIFEST), "configs": manifest["configs"]}, indent=2))


if __name__ == "__main__":
    main()
