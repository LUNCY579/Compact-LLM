from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "_publish_taria_lumina.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("publish_taria_lumina", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_release_identities_and_encoded_scales_are_frozen() -> None:
    taria = MODULE.CORE_SPECS["taria"]
    lumina = MODULE.CORE_SPECS["lumina"]

    assert taria.directory == "Rineya-Taria-1.7.0-devfreeze"
    assert lumina.directory == "Rineya-Lumina-1.7.0-devfreeze"
    assert taria.version == lumina.version == "1.7.0-devfreeze"
    assert taria.core_id == (
        "rineya.core.taria.v1.7.0-devfreeze.28L."
        "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
    )
    assert lumina.core_id == (
        "rineya.core.lumina.v1.7.0-devfreeze.28L."
        "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
    )
    assert taria.encoded_alpha / 32 == taria.scale == 0.8125
    assert lumina.encoded_alpha / 32 == lumina.scale == 1.25
    assert taria.showcase_hits == 42
    assert taria.stability_hits == 121
    assert taria.all_hits == 163
    assert lumina.showcase_hits == 40
    assert lumina.stability_hits == 113
    assert lumina.all_hits == 153
    assert taria.reference_predictions_sha256 != lumina.reference_predictions_sha256
    assert MODULE.RENAMED_SUCCESSORS == {
        "taria": {
            "codename": "Supa",
            "directory": "Rineya-Supa-1.7.0-devfreeze",
            "core_id": (
                "rineya.core.supa.v1.7.0-devfreeze.28L."
                "qwen3-nimbus-lcr-stage-a-step200-scale-0p8125"
            ),
        },
        "lumina": {
            "codename": "Baya",
            "directory": "Rineya-Baya-1.7.0-devfreeze",
            "core_id": (
                "rineya.core.baya.v1.7.0-devfreeze.28L."
                "qwen3-nimbus-lcr-stage-a-step200-scale-1p25"
            ),
        },
    }


@pytest.mark.parametrize("core", ["taria", "lumina"])
def test_selected_adapter_configs_encode_the_frozen_scale(core: str) -> None:
    release = MODULE.CORE_SPECS[core]
    path = release.adapter_path / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))

    assert MODULE.sha256_file(path) == release.adapter_config_sha256
    assert config["r"] == 16
    assert config["lora_alpha"] == release.encoded_alpha
    assert set(config["target_modules"]) == MODULE.EXPECTED_TARGET_MODULES
    assert MODULE._resolve_adapter_base(config["base_model_name_or_path"]) == (
        MODULE.BASE_MODEL.resolve()
    )


def test_nimbus_tensor_inventory_pin_allows_only_tied_head_omission() -> None:
    inventory = MODULE.safetensor_inventory(MODULE.BASE_MODEL / "model.safetensors")
    by_key = {row["key"]: row for row in inventory}

    assert len(inventory) == 311
    assert (
        MODULE.canonical_json_sha256(inventory) == MODULE.BASE_TENSOR_INVENTORY_SHA256
    )
    assert by_key["lm_head.weight"]["shape"] == [151936, 1024]
    assert by_key["model.embed_tokens.weight"]["shape"] == [151936, 1024]
    assert by_key["lm_head.weight"]["dtype"] == "BF16"


def test_copy_preserved_assets_is_byte_exact_and_hash_pinned(tmp_path: Path) -> None:
    base = tmp_path / "base"
    output = tmp_path / "output"
    base.mkdir()
    output.mkdir()
    expected: dict[str, str] = {}
    for name, content in {
        "config.json": b'{"model_type":"qwen3"}\n',
        "tokenizer.json": b'{"tokenizer":"fixed"}\n',
    }.items():
        (base / name).write_bytes(content)
        expected[name] = MODULE.sha256_file(base / name)

    observed = MODULE.copy_preserved_assets(base, output, expected)

    assert observed == expected
    for name in expected:
        assert (output / name).read_bytes() == (base / name).read_bytes()

    (base / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MODULE.copy_preserved_assets(base, tmp_path / "other", expected)


def test_refuse_publish_paths_never_overwrites_final_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    spec = MODULE.CORE_SPECS["taria"]

    spec.final_path.mkdir()
    with pytest.raises(FileExistsError, match="published core"):
        MODULE.refuse_publish_paths([spec], require_staged=False)

    spec.final_path.rmdir()
    spec.staging_path.mkdir()
    with pytest.raises(FileExistsError, match="staging core"):
        MODULE.refuse_publish_paths([spec], require_staged=False)


def test_renamed_successor_retires_all_legacy_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(MODULE, "VAULT", vault)
    successor = MODULE.RENAMED_SUCCESSORS["taria"]
    (vault / successor["directory"]).mkdir()

    with pytest.raises(RuntimeError, match="publisher is retired.*Taria->Supa"):
        MODULE.refuse_publish_paths([MODULE.CORE_SPECS["lumina"]], require_staged=False)
    with pytest.raises(RuntimeError, match="publisher is retired"):
        MODULE.parity_commands(MODULE.CORE_SPECS["taria"])

    assert not (vault / MODULE.CORE_SPECS["taria"].directory).exists()
    assert not (vault / MODULE.CORE_SPECS["lumina"].directory).exists()


def test_paired_stage_rolls_back_first_stage_if_second_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    specs = [MODULE.CORE_SPECS["taria"], MODULE.CORE_SPECS["lumina"]]
    monkeypatch.setattr(MODULE, "validate_pinned_sources", lambda _spec: {})

    def fake_stage(spec: object) -> Path:
        if spec.key == "lumina":
            raise RuntimeError("simulated second-stage failure")
        spec.staging_path.mkdir()
        return spec.staging_path

    monkeypatch.setattr(MODULE, "stage_core", fake_stage)

    with pytest.raises(RuntimeError, match="second-stage failure"):
        MODULE.stage_many(specs)

    assert not specs[0].staging_path.exists()
    assert not specs[1].staging_path.exists()


def test_catalog_upsert_preserves_active_archived_and_unknown_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    catalog_path = tmp_path / "CATALOG.json"
    original_active = {
        "core_id": "rineya.core.nimbus.existing",
        "codename": "Nimbus",
        "directory": "Rineya-Nimbus-existing",
        "present": True,
        "core": {"core_id": "rineya.core.nimbus.existing"},
    }
    superseded = [{"directory": "_superseded/Rineya-Pulse-old"}]
    _write_json(
        catalog_path,
        {
            "vault": "Rineya_Published_Cores",
            "custom_future_field": {"keep": True},
            "cores": [original_active],
            "superseded": superseded,
            "last_written": ["old"],
        },
    )
    published: list[tuple[object, Path]] = []
    for key in ("taria", "lumina"):
        spec = MODULE.CORE_SPECS[key]
        directory = tmp_path / spec.directory
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"weights")
        _write_json(
            directory / "CORE.json",
            {
                "core_id": spec.core_id,
                "codename": spec.codename,
                "directory_name": spec.directory,
                "provisional": False,
            },
        )
        published.append((spec, directory))

    updated = MODULE.update_catalog_atomically(catalog_path, published)

    assert updated["custom_future_field"] == {"keep": True}
    assert updated["superseded"] == superseded
    assert updated["cores"][0] == original_active
    assert [row["codename"] for row in updated["cores"]] == [
        "Nimbus",
        "Taria",
        "Lumina",
    ]
    assert updated["last_written"] == [
        "Rineya-Taria-1.7.0-devfreeze",
        "Rineya-Lumina-1.7.0-devfreeze",
    ]

    # Exact identities are upserts, not duplicate appends.
    updated["cores"][1]["future_entry_field"] = {"keep": True}
    _write_json(catalog_path, updated)
    again = MODULE.update_catalog_atomically(catalog_path, published)
    assert len(again["cores"]) == 3
    assert again["superseded"] == superseded
    assert again["cores"][1]["future_entry_field"] == {"keep": True}


def test_catalog_rejects_codename_reassignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    catalog_path = tmp_path / "CATALOG.json"
    _write_json(
        catalog_path,
        {
            "cores": [
                {
                    "core_id": "different.taria.id",
                    "codename": "Taria",
                    "directory": "different-directory",
                }
            ],
            "superseded": [],
        },
    )
    spec = MODULE.CORE_SPECS["taria"]
    directory = tmp_path / spec.directory
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"weights")
    _write_json(
        directory / "CORE.json",
        {"core_id": spec.core_id, "directory_name": spec.directory},
    )

    with pytest.raises(RuntimeError, match="already assigns codename"):
        MODULE.update_catalog_atomically(catalog_path, [(spec, directory)])


def test_catalog_successor_blocks_legacy_upsert_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(MODULE, "VAULT", vault)
    catalog_path = tmp_path / "CATALOG.json"
    successor = MODULE.RENAMED_SUCCESSORS["taria"]
    _write_json(
        catalog_path,
        {
            "cores": [
                {
                    "core_id": successor["core_id"],
                    "codename": successor["codename"],
                    "directory": successor["directory"],
                }
            ],
            "superseded": [],
            "future_field": {"keep": True},
        },
    )
    before = catalog_path.read_bytes()

    with pytest.raises(RuntimeError, match="catalog:Supa"):
        MODULE.update_catalog_atomically(
            catalog_path,
            [(MODULE.CORE_SPECS["taria"], tmp_path / "never-read")],
        )

    assert catalog_path.read_bytes() == before


def _parity_fixture(
    tmp_path: Path, *, drift: bool
) -> tuple[Path, Path, dict[str, Path]]:
    reference_rows: list[dict[str, object]] = []
    dense_rows: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    for index in range(256):
        row_id = f"case-{index:03d}"
        answer = f"answer-{index}"
        reference_rows.append(
            {
                "id": row_id,
                "status": "ok",
                "prediction": answer,
                "adapter_scale": 0.8125,
            }
        )
        dense_prediction = "Answer: answer-0" if drift and index == 0 else answer
        dense_rows.append(
            {"id": row_id, "status": "ok", "prediction": dense_prediction}
        )
        keys.append({"id": row_id, "accepted_answers": [answer]})
    reference = tmp_path / "reference.jsonl"
    dense = tmp_path / "dense.jsonl"
    _write_jsonl(reference, reference_rows)
    _write_jsonl(dense, dense_rows)
    answer_keys = {
        "showcase64": tmp_path / "showcase.jsonl",
        "stability192": tmp_path / "stability.jsonl",
        "all256": tmp_path / "all.jsonl",
    }
    _write_jsonl(answer_keys["showcase64"], keys[:64])
    _write_jsonl(answer_keys["stability192"], keys[64:])
    _write_jsonl(answer_keys["all256"], keys)
    return reference, dense, answer_keys


def test_parity_evidence_scores_dense_predictions_independently(
    tmp_path: Path,
) -> None:
    reference, dense, answer_keys = _parity_fixture(tmp_path, drift=True)

    evidence = MODULE.build_parity_evidence(
        MODULE.CORE_SPECS["taria"],
        model_checks={"weight_sha256": "dense-hash", "dtype_counts": {"BF16": 3}},
        predictions_path=dense,
        reference_path=reference,
        answer_key_paths=answer_keys,
        enforce_pins=False,
    )

    # The raw string differs once, but the canonical scorer still accepts it.
    assert evidence["parity"]["raw_prediction_matches"] == 255
    assert evidence["parity"]["normalized_prediction_matches"] == 256
    assert evidence["parity"]["differing_ids"] == ["case-000"]
    assert evidence["dense_results"]["showcase64"]["hits"] == 64
    assert evidence["dense_results"]["stability192"]["hits"] == 192
    assert evidence["dense_results"]["all256"]["hits"] == 256
    assert (
        evidence["candidate"]["canonical_prediction_sha256"]
        != evidence["peft_reference"]["canonical_prediction_sha256"]
    )


def test_parity_evidence_rejects_partial_coverage(tmp_path: Path) -> None:
    reference, dense, answer_keys = _parity_fixture(tmp_path, drift=False)
    lines = dense.read_text(encoding="utf-8").splitlines()
    dense.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="256 rows"):
        MODULE.build_parity_evidence(
            MODULE.CORE_SPECS["taria"],
            model_checks={"weight_sha256": "x", "dtype_counts": {"BF16": 1}},
            predictions_path=dense,
            reference_path=reference,
            answer_key_paths=answer_keys,
            enforce_pins=False,
        )


def _valid_publishable_evidence() -> dict[str, object]:
    spec = MODULE.CORE_SPECS["taria"]
    dense_results = {
        "showcase64": {"hits": 42, "items": 64, "strict_exact": 42 / 64},
        "stability192": {"hits": 121, "items": 192, "strict_exact": 121 / 192},
        "all256": {"hits": 163, "items": 256, "strict_exact": 163 / 256},
    }
    candidate = {
        "path": "artifacts/dense.jsonl",
        "file_sha256": "candidate-file",
        "canonical_prediction_sha256": "candidate-canonical",
        "rows": 256,
    }
    report = {"path": "artifacts/report.json", "sha256": "report", "run": {}}
    run_manifest = {
        "mode": MODULE.CONTROLLED_PARITY_MODE,
        "core_id": spec.core_id,
        "runner": {
            "path": MODULE.rel_repo(MODULE.RUNNER),
            "sha256": MODULE.RUNNER_SHA256,
        },
        "evaluator": {
            "path": MODULE.rel_repo(MODULE.EVALUATOR),
            "sha256": MODULE.EVALUATOR_SHA256,
        },
        "model_weight_sha256_before": "weight-hash",
        "model_weight_sha256_after": "weight-hash",
        "invocation": {
            "model": spec.staging_path.name,
            "prompts": MODULE.rel_repo(MODULE.DEMO_DATA / "all_prompts.jsonl"),
            "predictions": candidate["path"],
            "report": report["path"],
            "dtype": "bfloat16",
            "decoding": "greedy",
            "seed": 17,
            "thinking": False,
            "max_new_tokens": 24,
            "max_context": 4096,
            "truncation": "forbidden",
            "max_gpu_memory": "2GiB",
            "max_cpu_memory": "32GiB",
        },
    }
    return {
        "schema_version": "1.0",
        "kind": "dense_vs_peft_prediction_parity",
        "core_id": spec.core_id,
        "publishable_provenance": True,
        "model": {
            "weight_sha256": "weight-hash",
            "dtype_counts": {"BF16": 310},
        },
        "protocol": {
            "dtype": "bfloat16",
            "decoding": "greedy",
            "seed": 17,
            "thinking": False,
            "max_new_tokens": 24,
            "max_context": 4096,
            "truncation": "forbidden",
            "prompts_sha256": MODULE.DEMO_FILE_SHA256["all_prompts.jsonl"],
        },
        "candidate": candidate,
        "peft_reference": {
            "path": "artifacts/reference.jsonl",
            "file_sha256": "reference-file",
            "canonical_prediction_sha256": "reference-canonical",
            "rows": 256,
        },
        "parity": {
            "raw_prediction_matches": 255,
            "normalized_prediction_matches": 256,
            "items": 256,
            "exact_raw_parity": False,
            "exact_normalized_parity": True,
            "differing_ids": ["case-000"],
        },
        "dense_results": dense_results,
        "dense_report": report,
        "run_manifest": run_manifest,
        "selection_caveat": "caveat",
    }


def test_finalize_evidence_rejects_string_boolean_and_fractional_hits() -> None:
    checks = {"weight_sha256": "weight-hash", "dtype_counts": {"BF16": 310}}
    evidence = _valid_publishable_evidence()
    evidence["parity"]["exact_raw_parity"] = "false"
    with pytest.raises(TypeError, match="must be a boolean"):
        MODULE.validate_parity_evidence(
            MODULE.CORE_SPECS["taria"], evidence, checks, require_exact=False
        )

    evidence = _valid_publishable_evidence()
    evidence["dense_results"]["all256"]["hits"] = -0.5
    with pytest.raises(TypeError, match="hits must be an integer"):
        MODULE.validate_parity_evidence(
            MODULE.CORE_SPECS["taria"], evidence, checks, require_exact=False
        )


def test_finalize_rebuild_detects_tampered_parity_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = {"weight_sha256": "weight-hash", "dtype_counts": {"BF16": 310}}
    pristine = _valid_publishable_evidence()
    tampered = json.loads(json.dumps(pristine))
    tampered["candidate"]["file_sha256"] = "tampered"
    monkeypatch.setattr(MODULE, "resolve_repo_relative", lambda *_args: tmp_path)
    monkeypatch.setattr(MODULE, "require_sha256", lambda *_args: "ignored")
    monkeypatch.setattr(
        MODULE,
        "build_parity_evidence",
        lambda *_args, **_kwargs: pristine,
    )

    with pytest.raises(ValueError, match="altered or is stale: candidate"):
        MODULE.validate_parity_evidence(
            MODULE.CORE_SPECS["taria"], tampered, checks, require_exact=False
        )


def test_finalize_rejects_uncontrolled_record_parity_packet() -> None:
    evidence = _valid_publishable_evidence()
    evidence["publishable_provenance"] = False
    evidence["run_manifest"] = None

    with pytest.raises(ValueError, match="controlled run-parity"):
        MODULE.validate_parity_evidence(
            MODULE.CORE_SPECS["taria"],
            evidence,
            {"weight_sha256": "weight-hash", "dtype_counts": {"BF16": 310}},
            require_exact=False,
        )


def test_controlled_run_manifest_uses_relative_paths_and_binds_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    vault = repo / "Rineya_Published_Cores"
    data = repo / "demo" / "data"
    runner = repo / "demo" / "run_local_hf.py"
    evaluator = repo / "demo" / "evaluate_predictions.py"
    data.mkdir(parents=True)
    vault.mkdir()
    runner.write_text("# runner\n", encoding="utf-8")
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    (data / "all_prompts.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(MODULE, "REPO", repo)
    monkeypatch.setattr(MODULE, "VAULT", vault)
    monkeypatch.setattr(MODULE, "DEMO_DATA", data)
    monkeypatch.setattr(MODULE, "RUNNER", runner)
    monkeypatch.setattr(MODULE, "EVALUATOR", evaluator)
    spec = MODULE.CORE_SPECS["taria"]
    spec.staging_path.mkdir()
    checks = {"weight_sha256": "weight-hash", "dtype_counts": {"BF16": 310}}
    monkeypatch.setattr(
        MODULE, "verify_dense_checkpoint", lambda *_args, **_kwargs: checks
    )
    monkeypatch.setattr(MODULE, "require_sha256", lambda *_args: "pinned")

    def fake_subprocess(command: list[str], **_kwargs: object) -> None:
        if "--output" in command:
            Path(command[command.index("--output") + 1]).write_text(
                "predictions\n", encoding="utf-8"
            )
        if "--json-report" in command:
            Path(command[command.index("--json-report") + 1]).write_text(
                "{}\n", encoding="utf-8"
            )
            Path(command[command.index("--markdown-report") + 1]).write_text(
                "report\n", encoding="utf-8"
            )

    monkeypatch.setattr(MODULE.subprocess, "run", fake_subprocess)
    captured: dict[str, object] = {}

    def fake_record(
        _spec: object,
        predictions: Path,
        report: Path,
        *,
        require_exact: bool,
        run_manifest: dict[str, object],
    ) -> Path:
        captured.update(run_manifest)
        assert predictions.is_file()
        assert report.is_file()
        assert require_exact is False
        return spec.parity_path

    monkeypatch.setattr(MODULE, "record_parity", fake_record)

    result = MODULE.run_parity(
        spec,
        output_root=repo / "artifacts" / "parity-v2",
        max_gpu_memory="2GiB",
        max_cpu_memory="32GiB",
        require_exact=False,
    )

    assert result == spec.parity_path
    assert captured["mode"] == MODULE.CONTROLLED_PARITY_MODE
    assert captured["core_id"] == spec.core_id
    assert captured["model_weight_sha256_before"] == "weight-hash"
    serialized = json.dumps(captured)
    assert str(tmp_path) not in serialized
    assert "runner_command" not in captured
    assert "evaluator_command" not in captured


def test_finalized_metadata_uses_dense_scores_not_reference_scores() -> None:
    spec = MODULE.CORE_SPECS["taria"]
    evidence = {
        "dense_results": {
            "showcase64": {"hits": 41, "items": 64},
            "stability192": {"hits": 120, "items": 192},
            "all256": {"hits": 161, "items": 256},
        },
        "parity": {"exact_raw_parity": False},
    }
    core, manifest = MODULE.finalized_metadata(
        spec,
        {"core_id": spec.core_id, "provisional": True},
        {"core_id": spec.core_id, "status": "staged_pending_dense_parity"},
        evidence,
    )

    assert core["provisional"] is False
    assert core["status"] == "showcase_selected_stability_validated"
    assert "41/64" in core["public_claim"]
    assert "120/192" in core["public_claim"]
    assert "161/256" in core["public_claim"]
    assert manifest["parity_classification"] == "measured_dense_prediction_drift"


def test_failed_catalog_commit_rolls_back_core_and_provisional_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    monkeypatch.setattr(MODULE, "CATALOG", tmp_path / "CATALOG.json")
    monkeypatch.setattr(
        MODULE, "LEGACY_PARITY_ARCHIVE_ROOT", tmp_path / "legacy-archive"
    )
    specs = [MODULE.CORE_SPECS["taria"], MODULE.CORE_SPECS["lumina"]]
    originals: dict[str, tuple[dict[str, object], dict[str, object], str]] = {}
    for spec in specs:
        spec.staging_path.mkdir()
        original_core = {
            "core_id": spec.core_id,
            "directory_name": spec.directory,
            "provisional": True,
        }
        original_manifest = {
            "core_id": spec.core_id,
            "status": "staged_pending_dense_parity",
        }
        original_readme = f"provisional {spec.key} readme\n"
        originals[spec.key] = (original_core, original_manifest, original_readme)
        _write_json(spec.staging_path / "CORE.json", original_core)
        _write_json(spec.staging_path / "merge_manifest.json", original_manifest)
        (spec.staging_path / "README.md").write_text(original_readme, encoding="utf-8")
        _write_json(
            spec.parity_path,
            {
                "schema_version": "1.0",
                "core_id": spec.core_id,
                "model": {"weight_sha256": "weight-hash"},
                "parity": {"items": 256, "exact_raw_parity": False},
                "dense_results": {
                    "showcase64": {"hits": 41, "items": 64},
                    "stability192": {"hits": 120, "items": 192},
                    "all256": {"hits": 161, "items": 256},
                },
            },
        )
        _write_json(spec.legacy_parity_path, {"legacy": spec.key})
    monkeypatch.setattr(
        MODULE,
        "verify_dense_checkpoint",
        lambda _path, reload_model: {
            "weight_sha256": "weight-hash",
            "dtype_counts": {"BF16": 310},
        },
    )
    monkeypatch.setattr(
        MODULE, "validate_parity_evidence", lambda *args, **kwargs: None
    )

    def fail_catalog(_path: Path, _published: object) -> None:
        raise RuntimeError("simulated catalog failure")

    monkeypatch.setattr(MODULE, "update_catalog_atomically", fail_catalog)

    with pytest.raises(RuntimeError, match="simulated catalog failure"):
        MODULE.finalize_many(specs, require_exact=False)

    for spec in specs:
        original_core, original_manifest, original_readme = originals[spec.key]
        assert spec.staging_path.is_dir()
        assert not spec.final_path.exists()
        assert (
            json.loads((spec.staging_path / "CORE.json").read_text()) == original_core
        )
        assert (
            json.loads((spec.staging_path / "merge_manifest.json").read_text())
            == original_manifest
        )
        assert (spec.staging_path / "README.md").read_text() == original_readme
        assert spec.legacy_parity_path.is_file()
        assert not spec.archived_parity_path(spec.legacy_parity_path).exists()


def test_cli_exposes_nonexecuting_commands_and_two_phase_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "VAULT", tmp_path)
    parser = MODULE.build_parser()
    assert parser.parse_args(["stage", "taria"]).command == "stage"
    assert parser.parse_args(["commands", "all"]).command == "commands"
    assert (
        parser.parse_args(
            ["record-parity", "lumina", "--predictions", "predictions.jsonl"]
        ).command
        == "record-parity"
    )
    assert parser.parse_args(["run-parity", "taria"]).command == "run-parity"
    assert parser.parse_args(["finalize", "taria", "lumina"]).command == "finalize"

    commands = MODULE.parity_commands(MODULE.CORE_SPECS["taria"])
    assert len(commands) == 1
    assert "run-parity taria" in commands[0]
    assert "--max-gpu-memory 2GiB" in commands[0]
    with pytest.raises(ValueError, match="all.*itself"):
        MODULE.selected_specs(["all", "taria"])
    with pytest.raises(ValueError, match="paired-release"):
        MODULE.require_release_pair([MODULE.CORE_SPECS["taria"]], "stage")


def test_status_reports_archival_successors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    successor = MODULE.RENAMED_SUCCESSORS["lumina"]
    (vault / successor["directory"]).mkdir()
    catalog = vault / "CATALOG.json"
    _write_json(catalog, {"cores": [], "superseded": []})
    monkeypatch.setattr(MODULE, "VAULT", vault)
    monkeypatch.setattr(MODULE, "CATALOG", catalog)

    assert MODULE.main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["_publisher"]["archival"] is True
    assert status["_publisher"]["retired"] is True
    assert status["_publisher"]["canonical_successors"]["lumina"] == {
        **successor,
        "present": True,
    }
