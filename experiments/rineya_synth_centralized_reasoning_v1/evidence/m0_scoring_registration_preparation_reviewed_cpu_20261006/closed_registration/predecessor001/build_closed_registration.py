"""CPU metadata-only registration draft builder. Never imports a scorer or capture loader."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
SCORER_STAGE = PACKAGE / "artifacts/m0_scoring_preparation_work"
AUDIT_NAMES = (
    "short_m0_atomic_batch01_attempt1_independent_audit_20261005.json",
    "short_m0_atomic_batch02_attempt1_independent_audit_20261005.json",
    "short_m0_atomic_batch03_attempt1_independent_audit_20261005.json",
    "short_m0_atomic_batch04_phase3_attempt1_20261005_independent_audit_20261005.json",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def metadata(path):
    raw = Path(path).read_bytes()
    assert 0 < len(raw) <= 1024**2, "bounded selected source/metadata file"
    return {"path": str(Path(path).resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def document(path):
    ref = metadata(path)
    value = json.loads(Path(path).read_bytes())
    if "sha256" in value:
        assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    return value, ref


def anchor(path):
    value, ref = document(path)
    return {"path": ref["path"], "file_sha256": ref["sha256"], "body_sha256": value["sha256"]}


def build():
    finish, finish_ref = document(PACKAGE / "DEV_FINISH_POLICY_001.json")
    frozen, frozen_ref = document(SCORER_STAGE / "SOURCE_FROZEN_002.json")
    qualification, qualification_ref = document(SCORER_STAGE / "AUTHORED_SEMANTIC_QUALIFICATION_001.json")
    assert qualification["checks_count"] == 62 and qualification["native_scoring_authorized"] is False
    for name, ref in frozen["source_files"].items():
        assert metadata(SCORER_STAGE / name) == ref
    scorer_refs = finish["population"]["scorer_sources"]
    for name, ref in scorer_refs.items():
        assert metadata(Path(ref["path"])) == ref
    tree = ast.parse((PACKAGE / "native_loader.py").read_text(encoding="utf-8"))
    asset_pins = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ASSET_PINS" for t in node.targets))
    batches, known = [], []
    for ordinal, name in enumerate(AUDIT_NAMES, 1):
        audit_path = PACKAGE / "evidence" / name
        audit, ref = document(audit_path)
        assert audit["status"] == "PASS_SCOPED_SHORT_M0_RAW_COLLECTION_MECHANICS_UNSCORED"
        assert audit["rows"] == 64 and audit["batch_selector"] == {"ordinal": ordinal, "source_kind": "atomic_RS_DEV"}
        assert audit["scoring_authorized"] is False and audit["correctness_grade_performed"] is False
        batches.append({"preserved_directory": audit["preserved_path"], "audit_path": ref["path"],
            "expected_audit_sha256": ref["sha256"], "expected_config_sha256": audit["config_file_sha256"]})
        known.append({"ordinal": ordinal, "audit": ref, "audit_body_sha256": audit["sha256"],
            "config_path_recorded_by_audit": audit["config_path"],
            "config_body_sha256_recorded_by_audit": audit["config_body_sha256"],
            "config_file_sha256_recorded_by_audit": audit["config_file_sha256"],
            "source_commit_recorded_by_audit": audit["source_commit"],
            "rows_recorded_by_audit": 64, "status_recorded_by_audit": audit["status"],
            "capture_config_and_response_files_opened": False})
    batches.extend([None, None])
    implementation = {name: frozen["source_files"][name]["sha256"] for name in ("m0_scoring.py", "m0_score_audit.py")}
    future_output = ROOT.parent / "rineya-synth-m0-scoring-20261006" / "attempt001" / "COMPLETE_M0_SCORE_PACKET.json"
    result = {
        "kind": "REGISTERED_M0_ONLY_SCORING_V1", "scope": "M0_FIXED384_DEV_WHOLE_RESPONSE_SCORING",
        "status": "CLOSED_METADATA_TEMPLATE_UNRESOLVED_NOT_DISPATCH", "native_scoring_authorized": False,
        "owner_native_launch_authorized": False, "training_authorized": False,
        "protected_bank_access": False, "confirmation_authorized": False, "promotion_authorized": False,
        "source_snapshot_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_snapshot_is_current_native_qualification": False,
        "implementation_files": implementation,
        "expected_public_source_directory": str(PACKAGE / "m0_scoring_v1"),
        "staged_source_freeze": frozen_ref, "scorer_sources": {name: ref["sha256"] for name, ref in scorer_refs.items()},
        "scorer_source_descriptors": scorer_refs,
        "finish_policy_file_sha256": finish_ref["sha256"], "finish_policy_anchor": anchor(PACKAGE / "DEV_FINISH_POLICY_001.json"),
        "importer_file_sha256": metadata(PACKAGE / "score_import.py")["sha256"],
        "importer_descriptor": metadata(PACKAGE / "score_import.py"),
        "baseline_first_amendment_anchor": anchor(PACKAGE / "BASELINE_FIRST_MEASUREMENT_AMENDMENT_001.json"),
        "qualification_anchor": anchor(SCORER_STAGE / "AUTHORED_SEMANTIC_QUALIFICATION_001.json"),
        "qualification_scope": "62 authored CPU semantic checks only; separate trusted Root-qualified review required. This receipt is not native scoring permission.",
        "review_anchor": None, "approval_anchor": None,
        "panel_registration_sha256": None,
        "panel_registration_unresolved_reason": "Requires complete frozen-panel registration reconstruction by the existing importer; full DEV prompt/token inputs were not opened for this metadata template.",
        "row_order_sha256": finish["population"]["row_order_sha256"],
        "panel_metadata_sha256": finish["population"]["panel_metadata_sha256"],
        "public_panel_file_descriptors": finish["population"]["panel_files"],
        "statistics_descriptors": finish["population"]["statistics_sources"],
        "model_pins": asset_pins,
        "model_pins_provenance": "Declared ASSET_PINS literal from hash-selected native_loader.py source; no asset bytes read and no current model qualification claimed.",
        "model_pin_source": metadata(PACKAGE / "native_loader.py"),
        "batches": batches, "known_audit_metadata": known,
        "missing_batch_ordinals": [5, 6], "known_audited_raw_rows": 256,
        "complete_rows_required": 384, "complete_joined_parents_required": 96,
        "known_four_audit_hashes_are_not_complete_population_or_fresh_anchor_qualification": True,
        "trusted_audit_anchor_selection": None,
        "requires_treatment_checkpoints": False,
        "output_path": str(future_output), "output_path_status": "PROSPECTIVE_ONLY_NO_DIRECTORY_OR_FILE_CREATED",
        "correctness": finish["correctness"], "generation": finish["generation"],
        "timing_disclosure": finish["timing_disclosure"],
        "retention_enrollment": "Separate complete qualified domain-specific M0 guard enrollment must be frozen before treatment; this DEV scorer supplies no guard grant.",
        "unresolved": ["Reviewed exact source promotion", "Trusted independent qualification review", "Complete M0 batch5 audit", "Complete M0 batch6 audit", "Trusted complete six-audit anchor selection", "Frozen panel registration body hash", "Explicit scope-matching M0-only scoring permission", "Separate guard support and qualified enrollment before treatment"],
        "activity": {"capture_response_reads": 0, "capture_config_reads": 0, "DEV_prompt_or_token_or_oracle_reads": 0,
            "model_or_tokenizer_reads": 0, "scorer_calls": 0, "native_calls": 0, "output_publications": 0},
        "quality_gain_established": False,
    }
    result["sha256"] = digest(result)
    return result


if __name__ == "__main__":
    output = HERE / "M0_SCORING_REGISTRATION_CLOSED_TEMPLATE_001.json"
    value = build()
    with output.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
    print(json.dumps(metadata(output), sort_keys=True))
