from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs/phase2_selected_dev_free_run.yaml"
SELECTION = ROOT / "artifacts/phase2_mechanism/selected_dev_free_run_manifest.json"
CORNERSTONE = ROOT / "results/sealed/v22_frozen_finalists.jsonl"
CORNERSTONE_AUDIT = ROOT / "reports/phase2_cornerstone_verification.json"
CONFIG = ROOT / "configs/phase2_v22_calibrated_test_confirmation.yaml"
MANIFEST = ROOT / "artifacts/phase2_mechanism/v22_calibrated_test_confirmation_manifest.json"
OUTPUT = ROOT / "results/runs/phase2_v22_calibrated_test_confirmation_dev_full.jsonl"


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
    if OUTPUT.is_file() and OUTPUT.stat().st_size:
        raise RuntimeError(
            "Calibrated v2.2 TEST output already exists; refusing to refreeze after outcomes"
        )
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    selection = load_json(SELECTION)
    audit = load_json(CORNERSTONE_AUDIT)
    if audit.get("status") != "PASS":
        raise RuntimeError("Cornerstone audit must pass before freezing confirmation")
    if selection.get("test_data_read_for_selection") is not False:
        raise RuntimeError("Selected calibration comparator was not DEV-only")
    if "calibrated_probability_mean" not in selection.get("selected_methods", []):
        raise RuntimeError("DEV selection did not include calibrated probability mean")
    expected_cornerstone_sha = "d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d"
    if sha256(CORNERSTONE) != expected_cornerstone_sha:
        raise RuntimeError("Cornerstone hash changed before confirmation freeze")

    config = {
        "name": "phase2_v22_calibrated_test_confirmation",
        "backbone": source["backbone"],
        "experts": source["experts"],
        "committees": [
            {
                "name": "full_k7",
                "experts": [expert["name"] for expert in source["experts"]],
                "role": "dev_selected_primary_test_calibration_confirmation",
            }
        ],
        "methods": ["calibrated_probability_mean"],
        "datasets": {
            "v22_sealed": (
                "data/phase_9_student_distillation/multidomain_reasoning_v22/"
                "reviewed_splits/test.reviewed.jsonl"
            )
        },
        "max_sequence_length": source["max_sequence_length"],
        "max_new_tokens": source["max_new_tokens"],
        "cache_mode": "reference",
        "eos_policy": "ordinary",
        "seed": source["seed"],
    }
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    selected_rows = selection["selected_teacher_forced_rows"]
    manifest = {
        "schema_version": "1.0",
        "status": "V22_CALIBRATED_TEST_CONFIRMATION_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "analysis_role": "V2.2 TEST SELECTED CALIBRATION CONFIRMATION WITH PRIOR N=32 EXPOSURE CAVEAT",
        "selection_manifest": str(SELECTION),
        "selection_manifest_sha256": sha256(SELECTION),
        "teacher_analysis_sha256": selection["analysis_sha256"],
        "selection_used_test_outcomes": False,
        "existing_cornerstone_outcomes_known_before_freeze": True,
        "calibrated_confirmation_outcomes_read_before_freeze": False,
        "config": {
            "path": str(CONFIG),
            "file_sha256": sha256(CONFIG),
            "canonical_sha256": canonical_hash(config),
            "expected_rows": 4050,
        },
        "output": str(OUTPUT),
        "dataset": {
            "name": "v22_sealed",
            "rows": 4050,
            "sha256": audit["dataset_splits"]["test"]["sha256"],
        },
        "cornerstone": {
            "path": str(CORNERSTONE),
            "sha256": expected_cornerstone_sha,
            "hard_condition": "full_k7/hard_plurality",
            "raw_probability_condition": "full_k7/probability_mean",
        },
        "fitted_temperatures": selection["fitted_temperatures"],
        "selected_dev_rows": {
            method: selected_rows[method]
            for method in ("hard_plurality", "probability_mean", "calibrated_probability_mean")
        },
        "planned_condition": "full_k7/calibrated_probability_mean",
        "planned_paired_family": [
            "calibrated probability K7 vs hard K7",
            "calibrated probability K7 vs raw probability K7",
        ],
        "statistics": (
            "paired example bootstrap difference CI; exact McNemar; Holm correction over the two "
            "predeclared calibrated-fusion comparisons"
        ),
        "no_switching_rule": (
            "Do not change temperatures, method, committee, EOS, decoder, comparison family, or "
            "reporting direction after calibrated v2.2 TEST output is inspected."
        ),
        "caveats": [
            "An archived n=32 pilot previously exposed examples from the same v2.2 TEST split.",
            "Hard and raw probability full-TEST outcomes were known before this confirmation was frozen.",
            "The calibrated method and all seven temperatures were selected exclusively from the frozen DEV cache.",
            "This adds one predeclared calibrated condition; it does not refit or replace the cornerstone family.",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(MANIFEST),
                "config": manifest["config"],
                "calibrated_output_exists": OUTPUT.exists(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
