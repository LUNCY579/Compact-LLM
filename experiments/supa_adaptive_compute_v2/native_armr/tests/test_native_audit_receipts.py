"""Exercise real receipt/file reconciliation using synthetic temporary evidence."""
import json

import pytest

from native_armr.src import audit
from native_armr.src.protocol import DEFAULT_CONFIG, load_protocol
from shared import evaluation
from shared.datasets import generate_rows
from shared.hashing import canonical_sha256, sha256_file, write_json
from test_native_analysis import evidence


@pytest.mark.parametrize("corruption", [None, "case_file", "logits", "gradient", "missing_arm"])
def test_independent_audit_receipt_success_and_preserved_failure(tmp_path, monkeypatch, corruption):
    config = load_protocol(DEFAULT_CONFIG)
    config["analysis"]["bootstrap_replicates"] = 50
    cases = generate_rows("native_armr", "architecture_dev")
    rows = evidence(cases, lambda arm, i, c: True)
    manifest_path = tmp_path / "dataset_manifest.json"
    write_json(manifest_path, {"fixture": True})
    registration = {"registration_sha256": "fixture-registration", "config": config,
                    "sources": {"fixture.py": "fixture-source"},
                    "dataset_manifests": {manifest_path.relative_to(tmp_path).as_posix(): sha256_file(manifest_path)}}
    runs = []
    for arm, values in rows.items():
        path = tmp_path / "native_armr/evidence/01_architecture_dev" / arm
        (path / "logits").mkdir(parents=True)
        (path / "cases").mkdir()
        hashes = {}
        for row in values:
            name = canonical_sha256(row["example_id"])
            write_json(path / "cases" / (name + ".json"), row)
            file = path / "logits" / (name + ".pt")
            # Audit verifies these artifacts' hashes; it never deserializes or
            # claims to rerun inference on this intentionally non-model fixture.
            file.write_bytes(b"synthetic first-logit artifact")
            hashes[file.name] = sha256_file(file)
        write_json(path / "results.json", values)
        extra = 31457280 if arm == "AM" else 0
        work = 0 if arm == "A0" else 62914560 if arm in {"A3", "AM"} else 31457280
        checkpoint = {"model.safetensors": config["checkpoint_sha256"]}
        final = {"status": "COMPLETE", "track": "native_armr", "arm": arm, "role": "architecture_dev",
                 "canonical_unchanged": True, "example_ids": [r["example_id"] for r in values],
                 "results_sha256": sha256_file(path / "results.json"), "logits_sha256": hashes,
                 "registration_sha256": registration["registration_sha256"], "sources": registration["sources"],
                 "config_sha256": canonical_sha256(config), "dataset_manifest_sha256": sha256_file(manifest_path),
                 **{k: config[k] for k in ("seed", "threads", "dtype", "device", "max_prompt_seconds", "model_class")},
                 "cache_mode": "full_prefix_no_cache", "torch": "2.12.1+cu126", "transformers": "5.14.1",
                 "parameters": 1, "trainable_parameters": 0, "backbone_gradients": 0,
                 "state_sha256_before": "fixture-state", "state_sha256_after": "fixture-state",
                 "checkpoint_files_before": checkpoint, "checkpoint_files_after": checkpoint,
                 "qwen_implementation_sha256": "fixture-implementation", "peak_memory_bytes": 2048,
                 "operator": {"added_parameter_count": extra, "added_active_parameters": extra,
                              "extra_matmul_flops_per_token": work},
                 "operator_state_sha256": "fixture-control", "operator_trainable_parameters": 0,
                 "operator_gradients": 0, "operator_trainable_parameters_after": 0, "operator_gradients_after": 0}
        write_json(path / "run_manifest.json", final)
        runs.append(path)
    if corruption == "case_file":
        file = next((runs[-1] / "cases").glob("*.json"))
        file.write_text(json.dumps({"tampered": True}))
    elif corruption == "logits":
        next((runs[-1] / "logits").glob("*.pt")).write_bytes(b"changed")
    elif corruption == "gradient":
        file = runs[-1] / "run_manifest.json"
        value = json.loads(file.read_text())
        value["operator_gradients_after"] = 1
        file.write_text(json.dumps(value))
    elif corruption == "missing_arm":
        runs.pop()
    output = tmp_path / "audit_fixture"
    def new_run(track, stage, details):
        output.mkdir()
        return output, {**details, "track": track, "stage": stage, "fixture": True}
    # Only provenance/freezing boundaries are injected; file hashes, raw-row
    # integrity, shared pairing and independent score reconstruction run normally.
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "REPO", tmp_path)
    monkeypatch.setattr(audit, "new_run", new_run)
    monkeypatch.setattr(audit, "load_protocol", lambda path: config)
    monkeypatch.setattr(audit, "verify_registration", lambda *args: registration)
    monkeypatch.setattr(evaluation, "verify_registration", lambda *args: registration)
    monkeypatch.setattr(audit, "load_frozen_dataset", lambda *args, **kwargs: cases)
    monkeypatch.setattr(evaluation, "load_frozen_dataset", lambda *args, **kwargs: cases)
    if corruption:
        with pytest.raises((ValueError, RuntimeError)):
            audit.audit_runs(config_path=DEFAULT_CONFIG, dataset_manifest=manifest_path,
                             role="architecture_dev", run_dirs=runs)
        receipt = json.loads((output / "run_manifest.json").read_text())
        assert receipt["status"] == "FAILED" and receipt["failure_receipt"]["message"]
        assert not (output / "analysis.json").exists()
    else:
        path, result = audit.audit_runs(config_path=DEFAULT_CONFIG, dataset_manifest=manifest_path,
                                       role="architecture_dev", run_dirs=runs)
        assert path == output
        assert result["decision"] == "MECHANICS_PASS_NO_EFFICACY_CLAIM"
        receipt = json.loads((output / "run_manifest.json").read_text())
        assert receipt["status"] == "COMPLETE"
        assert receipt["analysis_sha256"] == sha256_file(output / "analysis.json")
        assert receipt["checked_runs"] == 5 and receipt["peak_memory_bytes"] > 0
