"""Small scalar receipts and tensors; no real model, GPU or efficacy input."""
from copy import deepcopy
import json

import pytest

from native_armr.src import control_calibration as earlier
from native_armr.src import learned_control_calibration as calibration
from shared.hashing import canonical_sha256, sha256_file, write_json


def test_reuses_existing_prompt_arithmetic_and_streaming_boundaries():
    for name in ("prompt_projection", "calibration_prompts", "ordinary_mask", "scenario_energies",
                 "aggregate_partition", "fit_alpha", "replay_native_delta", "_stream_prefix",
                 "_module_receipt", "_buffer_hashes", "tensor_sha256"):
        assert getattr(calibration, name) is getattr(earlier, name)
    config = calibration.load_config()
    assert config == calibration.expected_config()
    assert config["control"]["source_layers"] == [12, 13, 14]
    assert config["parameter_counts"]["extra_matmul_flops_per_token"] / 62914560 == .9
    assert config["reused_engineering_calibration"] is True
    assert config["alpha_search"] is config["holdout_selects_alpha"] is False


def test_prompt_projection_never_consults_answer_fields():
    class PoisonLabels(dict):
        def __getitem__(self, key):
            if key in {"answer", "scorer", "facts", "question", "latent_sha256"}:
                raise AssertionError("calibration read an answer field")
            return super().__getitem__(key)
    row = PoisonLabels(track="native_armr", split="calibration", variant="original",
        example_id="native_armr_calibration_tiny_0003_original", scenario_id="tiny_0003",
        family="tiny", template_id="tiny", prompt="Prompt only", alignment_partition="holdout",
        prompt_sha256=canonical_sha256("Prompt only"))
    assert set(calibration.prompt_projection(row)) == set(calibration.PROMPT_FIELDS)
    for change in ({"track": "cross_scale"}, {"split": "confirmatory_dev"},
                   {"split": "architecture_dev"}, {"variant": "paraphrase"}):
        with pytest.raises(ValueError):
            calibration.prompt_projection(PoisonLabels({**row, **change}))


def test_old_calibration_config_cannot_be_relabelled():
    with pytest.raises(ValueError, match="only the prospective v2.4"):
        calibration.load_config(earlier.DEFAULT_CONFIG)


def test_new_artifact_root_is_exclusive_and_rejects_failed_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    path = tmp_path / "native_armr/evidence" / calibration.STAGE / "fake/report.json"
    write_json(path, {"status": "FAILED"})
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "COMPLETE"})
    with pytest.raises(RuntimeError, match="hash/path"):
        calibration.verify_calibration_artifact(path, "0" * 64, require_committed=False)
    with pytest.raises(RuntimeError, match="science/freeze"):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


def _state(digest="a" * 64, parameters=2):
    return {"state_sha256_before": digest, "state_sha256_after": digest,
            "trainable_parameters": 0, "backbone_gradients": 0, "parameters": parameters,
            "buffer_sha256_before": {}, "buffer_sha256_after": {}}


def _metric(tokens):
    return {"token_count": tokens, "native_energy": 4., "control_energy": 16.,
            "hidden_rms": 1., "native_delta_rms": 2., "control_delta_rms": 4.}


@pytest.fixture
def synthetic_artifact(tmp_path, monkeypatch):
    """128 prompt-only synthetic receipts exercise every provenance boundary."""
    config = calibration.load_config()
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "REPO", tmp_path)
    monkeypatch.setattr(calibration, "ARMR", tmp_path / "historical")
    config_path = tmp_path / "native_armr/configs/learned_control_calibration_v2_4.json"
    monkeypatch.setattr(calibration, "DEFAULT_CONFIG", config_path)
    write_json(config_path, config)
    source = tmp_path / "source.py"
    source.write_text("# scalar-only fixture\n", encoding="utf-8")
    sources = {"source.py": sha256_file(source)}
    monkeypatch.setattr(calibration, "calibration_sources", lambda: sources)
    data = {}
    for kind in ("dataset_manifest", "calibration_data"):
        target = tmp_path / "data" / (kind + ".json")
        write_json(target, {"synthetic": True})
        data[kind + "_path"], data[kind + "_sha256"] = target.relative_to(tmp_path).as_posix(), sha256_file(target)
    rows, cases = [], []
    for index in range(128):
        row = {"example_id": f"native_armr_calibration_tiny_{index:04d}_original",
               "scenario_id": f"native_armr_calibration_tiny_{index:04d}", "family": "tiny",
               "template_id": "tiny.synthetic", "prompt": f"Tiny prompt {index}",
               "alignment_partition": "holdout" if index % 4 == 3 else "fit"}
        row["prompt_sha256"] = canonical_sha256(row["prompt"])
        rows.append(row)
        case = {key: value for key, value in row.items() if key != "prompt"}
        case.update(token_ids=[1, 3], full_prompt_tokens=2, retained_prompt_tokens=2,
                    metrics={"all": _metric(2), "ordinary": _metric(1),
                             "special_control": _metric(1), "final_prompt": _metric(1)},
                    original_rope_sha256=["b" * 64, "c" * 64])
        case.update({key: "d" * 64 for key in ("embedding_rows_sha256", "hidden_sha256", "native_delta_sha256",
                    "control_delta_sha256", "original_positions_sha256", "original_mask_sha256")})
        cases.append(case)
    monkeypatch.setattr(calibration, "calibration_prompts", lambda path: (rows, data))
    contract = {"schema_version": calibration.SCHEMA, "track": "native_armr", "config": config,
                "source_files": sources, "calibration_config_path": config_path.relative_to(tmp_path).as_posix(),
                "calibration_config_sha256": sha256_file(config_path), **data,
                **{partition + "_example_ids": [r["example_id"] for r in rows
                                               if r["alignment_partition"] == partition]
                   for partition in ("fit", "holdout")}}
    digest = canonical_sha256(contract)
    registration = {"contract": contract, "status": "FROZEN", "registration_sha256": digest}
    registration_path = tmp_path / "native_armr/evidence" / calibration.REGISTRATION_STAGE / digest / "registration.json"
    write_json(registration_path, registration)
    directory = tmp_path / "native_armr/evidence" / calibration.STAGE / "synthetic"
    for case in cases:
        write_json(directory / "cases" / (canonical_sha256(case["example_id"]) + ".json"), case)
    fit, holdout = [calibration.aggregate_partition([c for c in cases if c["alignment_partition"] == partition])
                    for partition in ("fit", "holdout")]
    alpha = calibration.fit_alpha(fit)
    write_json(directory / "fit_scale.json", {"alpha": alpha, "fit": fit, "formula": calibration.FORMULA})
    for partition in (fit, holdout):
        partition["scaled_control_energy"] = alpha ** 2 * partition["control_energy"]
    provenance, hashes, branches = {}, {}, {}
    for index in (12, 13, 14):
        mlp, norm = canonical_sha256([index, "mlp"]), canonical_sha256([index, "norm"])
        provenance[str(index)] = {"source_layer": index, "source_mlp_sha256": mlp, "copy_mlp_sha256": mlp,
                                  "source_norm_sha256": norm, "copy_norm_sha256": norm,
                                  "norm_epsilon": 1e-6, "hidden_size": 1024, "intermediate_size": 3072,
                                  "mlp_class": "Qwen3MLP", "norm_class": "Qwen3RMSNorm",
                                  "activation_class": "SiLUActivation"}
        hashes[str(index)] = {"mlp": mlp, "post_attention_layernorm": norm}
        branches[str(index)] = {"source_layer": index, "mlp": _state(mlp, 9437184),
                                "post_attention_layernorm": _state(norm, 1024)}
    checkpoints = {"model.safetensors": calibration.CHECKPOINT_SHA256,
                   "config.json": calibration.MODEL_CONFIG_SHA256, "tokenizer.json": "e" * 64}
    write_json(calibration.ARMR / "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json",
               {"checkpoint_files_before": checkpoints})
    report = {**calibration._required_report(), "alpha": alpha, "fit": fit, "holdout": holdout,
              "registration_path": registration_path.relative_to(tmp_path).as_posix(), "registration_sha256": digest,
              **{key: value for key, value in contract.items()
                 if key not in {"config", "fit_example_ids", "holdout_example_ids"}},
              "case_files": {p.name: sha256_file(p) for p in sorted((directory / "cases").glob("*.json"))},
              "special_control_token_ids": [1], "norm_epsilon": 1e-6,
              "layer_state_receipts": [{"stage": stage, "layer": i, **_state()} for stage, i in
                  [("ordinary_prefix", i) for i in range(14)] + [("replay", 12), ("replay", 13), ("control_source", 14)]],
              "rotary_state_receipt": _state(), "control_state_receipt": _state(parameters=28314624),
              "control_state_sha256": "a" * 64, "branch_provenance": provenance,
              "source_hashes": deepcopy(hashes), "copy_hashes": deepcopy(hashes),
              "source_branch_receipts": deepcopy(branches), "copy_branch_receipts": deepcopy(branches),
              "parameter_inventory": dict(calibration.PARAMETER_COUNTS),
              "checkpoint_files_before": deepcopy(checkpoints), "checkpoint_files_after": deepcopy(checkpoints),
              "qwen_implementation_sha256": calibration.HF_SHA256, "model_config_sha256": calibration.MODEL_CONFIG_SHA256,
              "runtime_identity": dict(calibration.RUNTIME), "fit_scale_sha256": sha256_file(directory / "fit_scale.json")}
    path = directory / "report.json"
    write_json(path, report)
    start = {**contract, "status": "STARTED", "registration_sha256": digest,
             "registration_path": report["registration_path"], "track": "native_armr", "stage": calibration.STAGE,
             "branch": "codex/synthetic", "commit_sha": "f" * 40, "start_utc": "2026-01-01T00:00:00+00:00",
             "failure_receipt": None}
    write_json(directory / "started_manifest.json", start)
    write_json(directory / "run_manifest.json", {**start, "status": "COMPLETE", "report_sha256": sha256_file(path)})
    return path, report


def _rewrite(path, report):
    path.write_text(json.dumps(report), encoding="utf-8")
    run_path = path.parent / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["report_sha256"] = sha256_file(path)
    run_path.write_text(json.dumps(run), encoding="utf-8")


def test_complete_artifact_recomputes_scale_and_enforces_commit_boundary(synthetic_artifact, monkeypatch):
    path, report = synthetic_artifact
    checked = []
    monkeypatch.setattr(calibration, "_require_committed", lambda p: checked.append(p))
    assert calibration.verify_calibration_artifact(path, sha256_file(path))["alpha"] == .5
    assert len(checked) == 137  # 128 cases, source/config/data (4), five evidence files.
    assert path.parent / "fit_scale.json" in checked
    assert path.parent / "started_manifest.json" in checked


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(alpha=.75), "alpha differs"),
    (lambda r: r.update(holdout_selects_alpha=True), "science/freeze"),
    (lambda r: r.update(reused_engineering_calibration=False), "science/freeze"),
    (lambda r: r["holdout"].update(native_energy=1.), "aggregation mismatch"),
    (lambda r: r["layer_state_receipts"][0].update(backbone_gradients=1), "has gradients"),
    (lambda r: r["layer_state_receipts"].pop(), "layer coverage"),
    (lambda r: r["copy_branch_receipts"]["12"]["mlp"].update(state_sha256_after="0" * 64), "mutated"),
    (lambda r: r["copy_branch_receipts"]["14"]["post_attention_layernorm"].update(trainable_parameters=1), "has gradients"),
    (lambda r: r["branch_provenance"]["13"].update(source_mlp_sha256="0" * 64), "source/copy"),
    (lambda r: r["branch_provenance"]["13"].update(source_layer=16), "geometry"),
    (lambda r: r["branch_provenance"]["14"].update(norm_epsilon=.1), "geometry"),
    (lambda r: r["branch_provenance"]["12"].update(activation_class="SiLU"), "geometry"),
    (lambda r: r["parameter_inventory"].update(extra_matmul_flops_per_token=62914560), "parameter/FLOP"),
    (lambda r: r["parameter_inventory"].update(newly_learned_parameters=28314624), "parameter/FLOP"),
    (lambda r: r["checkpoint_files_after"].update(extra="changed"), "checkpoint mutation"),
    (lambda r: r["runtime_identity"].update(device="cuda:0"), "runtime/model"),
    (lambda r: r.update(qwen_implementation_sha256="0" * 64), "runtime/model"),
])
def test_artifact_rejects_science_runtime_and_source_copy_tampering(synthetic_artifact, mutation, match):
    path, report = synthetic_artifact
    mutation(report)
    _rewrite(path, report)
    with pytest.raises(RuntimeError, match=match):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update(retained_prompt_tokens=1), "length mismatch"),
    (lambda c: c["metrics"]["ordinary"].update(token_count=2), "token-stratum"),
    (lambda c: c["metrics"]["final_prompt"].update(native_energy=float("inf")), "token-stratum"),
    (lambda c: c.update(hidden_sha256="missing"), "tensor identity"),
    (lambda c: c.update(original_rope_sha256=[]), "rotary identity"),
    (lambda c: c.update(alignment_partition="holdout"), "frozen calibration identity"),
])
def test_case_verifier_checks_tokens_strata_and_original_inputs(synthetic_artifact, mutation, match):
    path, report = synthetic_artifact
    filename = sorted(report["case_files"])[0]
    # Choose a fit row so the partition mutation above is always a change.
    for filename in report["case_files"]:
        candidate = path.parent / "cases" / filename
        case = json.loads(candidate.read_text(encoding="utf-8"))
        if case["alignment_partition"] == "fit":
            break
    mutation(case)
    candidate.write_text(json.dumps(case), encoding="utf-8")
    report["case_files"][filename] = sha256_file(candidate)
    _rewrite(path, report)
    with pytest.raises(RuntimeError, match=match):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


def test_unlisted_case_files_are_not_silently_ignored(synthetic_artifact):
    path, _ = synthetic_artifact
    write_json(path.parent / "cases/extra.json", {"synthetic": True})
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


def test_fit_receipt_cannot_be_changed_after_holdout(synthetic_artifact):
    path, report = synthetic_artifact
    fit_path = path.parent / "fit_scale.json"
    value = json.loads(fit_path.read_text(encoding="utf-8"))
    value["fit"] = {k: v for k, v in report["holdout"].items() if k != "scaled_control_energy"}
    fit_path.write_text(json.dumps(value), encoding="utf-8")
    report["fit_scale_sha256"] = sha256_file(fit_path)
    _rewrite(path, report)
    with pytest.raises(RuntimeError, match="fit scale was not frozen"):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)


@pytest.mark.parametrize("field,value", [("status", "FAILED"), ("commit_sha", "a" * 40),
                                         ("branch", "codex/changed"), ("source_files", {})])
def test_run_must_match_started_source_and_branch_identity(synthetic_artifact, field, value):
    path, _ = synthetic_artifact
    run_path = path.parent / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run[field] = value
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest required|binding mismatch|identity or completion"):
        calibration.verify_calibration_artifact(path, sha256_file(path), require_committed=False)
