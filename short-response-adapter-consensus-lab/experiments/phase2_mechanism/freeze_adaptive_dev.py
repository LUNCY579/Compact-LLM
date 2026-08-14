from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs" / "phase2_selected_dev_free_run.yaml"
ANALYSIS = ROOT / "results" / "phase2_mechanism" / "adaptive_teacher_forced_dev.json"
OUTPUT_CONFIG = ROOT / "configs" / "phase2_selected_adaptive_dev_free_run.yaml"
MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "selected_adaptive_dev_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    analysis = load_json(ANALYSIS)
    if analysis["status"] != "DEV_ADAPTIVE_THRESHOLD_SELECTED":
        raise RuntimeError(f"Adaptive selection is not complete: {analysis['status']}")
    if analysis["analysis_role"] != "DEV MODEL-SELECTION / CALIBRATION":
        raise RuntimeError("Adaptive threshold must be selected on DEV")
    threshold = float(analysis["selected_threshold"])
    core = list(analysis["core"])

    config = {
        "name": "phase2_selected_adaptive_dev_free_run",
        "backbone": source["backbone"],
        "experts": source["experts"],
        "committees": [
            {
                "name": "adaptive_k3_to_k7",
                "experts": [expert["name"] for expert in source["experts"]],
                "role": "dev_selected_adaptive_efficiency_confirmation",
            }
        ],
        "methods": ["adaptive_hard_plurality"],
        "adaptive_core_experts": core,
        "adaptive_agreement_threshold": threshold,
        "datasets": {"v22_dev": source["datasets"]["v22_dev"]},
        "max_sequence_length": source["max_sequence_length"],
        "max_new_tokens": source["max_new_tokens"],
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": source["seed"],
    }
    OUTPUT_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "DEV_ADAPTIVE_POLICY_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": analysis["analysis_role"],
        "analysis_path": str(ANALYSIS),
        "analysis_sha256": sha256(ANALYSIS),
        "cache_sha256": analysis["cache_sha256"],
        "capture_config_sha256": analysis["capture_config_sha256"],
        "source_config_path": str(SOURCE_CONFIG),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "output_config_path": str(OUTPUT_CONFIG),
        "output_config_sha256": sha256(OUTPUT_CONFIG),
        "output_config_canonical_sha256": canonical_json_sha256(config),
        "core": core,
        "selected_threshold": threshold,
        "threshold_semantics": analysis["threshold_semantics"],
        "selection_rule": analysis["selection_rule"],
        "selected_teacher_forced_policy": analysis["selected_policy"],
        "test_data_read_for_selection": False,
        "decoder_choice": "reference adaptive full-prefix recomputation",
        "no_switching_rule": "Do not change core, threshold, EOS policy, or committee after free-running output is inspected.",
        "limitations": analysis["limitations"],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "config": str(OUTPUT_CONFIG),
                "canonical_config_sha256": manifest["output_config_canonical_sha256"],
                "manifest": str(MANIFEST),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
