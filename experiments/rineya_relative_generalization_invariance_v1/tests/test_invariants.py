import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
from rgi import data, metrics, native, protocol
from rgi.io import (align, artifact, assert_disjoint, digest, file_hash, load, score_rows, seal,
                    validate_artifact, validate_bank, verify, write_new)
from rgi.__main__ import main
from conftest import make_bank, make_artifact


def reseal(obj):
    return seal({k: v for k, v in obj.items() if k != "content_sha256"})


def test_canonical_hash_order():
    assert digest({"x": 1, "y": 2}) == digest({"y": 2, "x": 1})


def test_write_new_refuses_overwrite(tmp_path):
    p = tmp_path / "receipt.json"
    write_new(p, {"x": 1})
    original = file_hash(p)
    with pytest.raises(FileExistsError):
        write_new(p, {"x": 2})
    assert file_hash(p) == original
    assert not list(tmp_path.glob(".rgi-*"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_io_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        write_new(tmp_path / "x.json", {"x": value})


def test_load_nonfinite_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):
        load(p)


def test_tamper_hash(bank):
    bank["items"][0]["input_ids"][2] = 123
    with pytest.raises(ValueError, match="hash"):
        validate_bank(bank)


@pytest.mark.parametrize("field,value", [
    ("input_ids", [1]), ("input_ids", [True, 1, 2, 3]), ("input_ids", [-1, 1, 2, 3]),
    ("roles", ["content"] * 4), ("roles", ["prompt"] * 4),
    ("roles", ["prompt", "content", "prompt", "format"]), ("roles", ["prompt", "prompt", "x", "format"]),
    ("roles", ["prompt"]), ("split", "test"), ("target_rule", 1), ("horizon", 0),
    ("horizon", True), ("payload_sha256", "not-a-hash"), ("parent_id", ""), ("family", None)])
def test_bank_schema_negative(bank, field, value):
    bank["items"][0][field] = value
    with pytest.raises((ValueError, TypeError)):
        validate_bank(reseal(bank))


def test_duplicate_items(bank):
    bank["items"].append(copy.deepcopy(bank["items"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_bank(reseal(bank))


@pytest.mark.parametrize("identity", ["payload_sha256", "lineage_id"])
def test_cross_split_leakage(bank, identity):
    bank["items"][1][identity] = bank["items"][0][identity]
    bank["items"][1]["split"] = "train"
    with pytest.raises(ValueError, match="leakage"):
        validate_bank(reseal(bank))


def test_parent_consistency(bank):
    bank["items"][1]["parent_id"] = bank["items"][0]["parent_id"]
    with pytest.raises(ValueError, match="Parent"):
        validate_bank(reseal(bank))


def test_bank_disjointness(bank):
    assert_disjoint(bank, make_bank(split="train"))
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(bank, bank)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01])
def test_bad_losses(bank, value):
    with pytest.raises(ValueError):
        score_rows(bank, [[value, 0.1] for _ in bank["items"]])


def test_missing_item_or_token(bank):
    for losses in ([], [[1] for _ in bank["items"]]):
        with pytest.raises(ValueError):
            score_rows(bank, losses)


def test_alignment_joins_by_key(reference):
    other = copy.deepcopy(reference)
    other["rows"].reverse()
    rows, values = align(reference, reseal(other))
    assert values[0] == values[1]
    assert len(rows) == len(reference["rows"])


@pytest.mark.parametrize("field,value", [
    ("tokenizer_sha256", digest("other")), ("bank_sha256", digest("other")),
    ("evidence_class", "native_model"), ("precision", "bfloat16"),
    ("context_policy", "self_rollout"), ("scoring_mode", "generation"), ("completed", False)])
def test_alignment_meta_refusal(reference, field, value):
    other = copy.deepcopy(reference)
    other[field] = value
    with pytest.raises(ValueError):
        align(reference, reseal(other))


@pytest.mark.parametrize("field,value", [("target_id", 99), ("context_sha256", digest("different")),
    ("role", "mixed"), ("rule_case", "different"), ("payload_sha256", digest("wrong"))])
def test_alignment_row_refusal(reference, field, value):
    other = copy.deepcopy(reference)
    other["rows"][0][field] = value
    with pytest.raises(ValueError):
        align(reference, reseal(other))


def test_incomplete_and_duplicate_artifact(reference):
    for mode in ("missing", "duplicate"):
        other = copy.deepcopy(reference)
        if mode == "missing":
            other["rows"].pop()
        else:
            other["rows"][1] = copy.deepcopy(other["rows"][0])
        with pytest.raises(ValueError):
            validate_artifact(reseal(other))


@pytest.mark.parametrize("offset", [-0.7, 0, 0.3, 10])
def test_uniform_shift_is_rgi(offset):
    x = np.array([1., 2., 4., 6.])
    p = metrics.profile(x, x + offset)
    assert p["offset"] == pytest.approx(offset)
    assert p["residual_rms"] == pytest.approx(0, abs=1e-12)
    assert p["centered_ccc"] == pytest.approx(1)


def test_perfect_correlation_not_invariance():
    p = metrics.profile([1, 2, 3], [2, 4, 6])
    assert p["pearson"] == pytest.approx(1)
    assert p["centered_ccc"] == pytest.approx(0.8)
    assert p["weighted_single_run_rdiff"] == pytest.approx(0.625)
    assert p["residual_rms"] > 0


def test_rdiff_matches_explicit_pairs():
    x, y = np.array([1., 2., 4.]), np.array([2., 2., 5.])
    dx, dy = x[:, None] - x, y[:, None] - y
    s = ((dx - dy)**2).sum()
    expected = 0.5 * s * (1 / (dx**2).sum() + 1 / (dy**2).sum())
    assert metrics.profile(x, y)["weighted_single_run_rdiff"] == pytest.approx(expected)


def test_degenerate_metrics_are_not_perfect_correlations():
    p = metrics.profile([1, 1], [2, 2])
    assert p["degenerate_difficulty"] and p["centered_ccc"] is None
    assert p["pearson"] is None and p["weighted_single_run_rdiff"] is None
    assert p["residual_rms"] == 0


@pytest.mark.parametrize("x,y,w", [([], [], None), ([1], [1, 2], None), ([math.nan], [1], None),
    ([1, 2], [1, 2], [0, 0]), ([1, 2], [1, 2], [-1, 2]), ([1, 2], [1, 2], [1])])
def test_profile_invalid(x, y, w):
    with pytest.raises(ValueError):
        metrics.profile(x, y, w)


def test_family_parent_weights():
    rows = [{"family": "a", "parent_id": "a1"}] * 100 + [{"family": "b", "parent_id": "b1"}]
    w = metrics.weights(rows)
    assert w[:100].sum() == pytest.approx(0.5)
    assert w[-1] == pytest.approx(0.5)


def test_bootstrap_seed_and_clusters(reference):
    rows = reference["rows"]
    x = np.array([r["nll"] for r in rows])
    y = x + np.arange(len(x)) / 100
    a = metrics.parent_bootstrap(rows, x, y, 200, 42)
    b = metrics.parent_bootstrap(rows, x, y, 200, 42)
    assert a == b
    assert sum(a["parent_counts"].values()) == 24  # not 48 target tokens
    assert a["intervals"]["offset"][0] < a["intervals"]["offset"][1]


def test_tiny_sample_not_certified(reference):
    rows = reference["rows"][:2]
    assert metrics.parent_bootstrap(rows, [1, 2], [2, 3], 100)["status"] == "UNDERPOWERED"


@pytest.mark.parametrize("repeats", [0, 99, 10001, 100.5])
def test_bootstrap_bounds(reference, repeats):
    with pytest.raises(ValueError):
        metrics.parent_bootstrap(reference["rows"], [1]*48, [1]*48, repeats)


def test_slice_keeps_global_offset(bank, reference):
    other = make_artifact(bank)
    for r in other["rows"]:
        if r["family"] == "f0":
            r["nll"] -= 0.2
    result = metrics.compare(reference, reseal(other), 100)
    assert result["slices"]["family=f0"]["mean_residual_from_global_offset"] < -0.1
    assert result["slices"]["family=f1"]["mean_residual_from_global_offset"] > 0
    assert set(result["role_profiles"]) == {"content", "format"}
    assert result["quality_improvement_established"] is False


def composition_models(bank):
    a, b = digest("a"), digest("b")
    return [make_artifact(bank), make_artifact(bank, .1, components={a: 1}),
            make_artifact(bank, .2, components={b: .25}), make_artifact(bank, .3, components={a: 1, b: .25})]


def test_additive_composition_zero_interaction(bank):
    result = metrics.interaction(*composition_models(bank), repeats=100)
    assert result["interaction_rms"] == pytest.approx(0, abs=1e-12)
    assert result["centered_effect_cosine"] is None


@pytest.mark.parametrize("error", ["host", "coefficients", "shared_component", "base_not_off"])
def test_composition_provenance(bank, error):
    models = composition_models(bank)
    if error == "host":
        models[3]["host_sha256"] = digest("wrong")
    elif error == "coefficients":
        models[3]["components"][digest("b")] = 1
    elif error == "shared_component":
        models[2]["components"] = models[1]["components"]
    else:
        models[0]["components"] = models[1]["components"]
    models = [reseal(x) for x in models]
    with pytest.raises(ValueError):
        metrics.interaction(*models, repeats=100)


def test_batch_decomposition():
    r = metrics.batch_decomposition(2.0, 3.0, 2.98)
    assert r["total"] == pytest.approx(r["batch_change"] + r["parameter_update"])
    assert r["absolute_data_to_update"] == pytest.approx(50)
    assert metrics.batch_decomposition(1, 2, 2)["absolute_data_to_update"] is None


def test_oracle_rules():
    assert data.transition("ledger_mini", [10, 20], {"op": "transfer", "posted": False, "amount": 3}) == [10, 20]
    assert data.transition("ledger_mini", [10, 20], {"op": "transfer", "posted": True, "amount": 3}) == [7, 23]
    assert data.transition("prerequisite_mini", 1, {"need": 2, "add": 4}) == 1
    assert data.transition("prerequisite_mini", 3, {"need": 1, "add": 3}) == 3  # not integer addition
    assert data.transition("register_mini", [90, 3], {"op": "mix", "amount": 4}) == [5, 3]
    with pytest.raises(ValueError):
        data.transition("unknown", 0, {})


def test_mini_bank_determinism_and_encoding():
    raw = data.mini_bank(8, 123)
    assert raw == data.mini_bank(8, 123)
    enc = data.encode_bank(raw, data.FixtureTokenizer(), digest("fixture"))
    validate_bank(enc)
    assert len(enc["items"]) > 40
    assert all("content" in i["roles"] for i in enc["items"])


def test_boundary_and_truncation_refusal():
    raw = data.mini_bank(1)
    with pytest.raises(ValueError, match="truncation"):
        data.encode_bank(raw, data.FixtureTokenizer(), digest("x"), max_length=2)
    class Crossing(data.FixtureTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
    with pytest.raises(ValueError, match="boundary"):
        data.encode_bank(raw, Crossing(), digest("x"))


def matching_bank():
    b = make_bank(n=120, split="train")
    for i, x in enumerate(b["items"]):
        x["family"] = "matched_family"
        x["target_rule"] = i % 3 == 0
    return reseal(b)


def test_exact_dose_control_matching():
    b = matching_bank()
    output = data.match_controls(b, 15)
    assert output == data.match_controls(b, 15)
    t, c = output["targeted"], output["control"]
    assert output["receipt"]["coverage_gap"] >= .15
    assert output["receipt"]["processed_tokens_each"] == 60
    assert output["receipt"]["supervised_tokens_each"] == 30
    assert not ({x["payload_sha256"] for x in t["items"]} & {x["payload_sha256"] for x in c["items"]})
    validate_bank(t)
    validate_bank(c)


def test_infeasible_and_eval_control_refusal(bank):
    with pytest.raises(ValueError, match="training"):
        data.match_controls(bank, 1)
    with pytest.raises(ValueError, match="Infeasible"):
        data.match_controls(matching_bank(), 1000)
    b = matching_bank()
    for x in b["items"]:
        x["target_rule"] = True
    with pytest.raises(ValueError, match="coverage"):
        data.match_controls(reseal(b), 10)


def spec_for(bank):
    return {"evidence_class": bank["evidence_class"], "tokenizer_sha256": bank["tokenizer_sha256"],
        "max_length": 512, "max_scored_tokens": 10000, "adapters": [], "model_path": "NOT_LOADED_BY_PLAN",
        "training": {"optimizer": "adamw", "updates": 4, "gradient_accumulation": 3, "rank": 4,
                     "alpha": 8, "learning_rate": .001, "max_grad_norm": 1, "target_modules": ["q_proj"], "seed": 17}}


def test_matched_training_plans_share_dose():
    pair = data.match_controls(matching_bank(), 15)
    plans = [native.training_plan(pair[k], spec_for(pair[k])) for k in ("targeted", "control")]
    assert plans[0]["order"] == plans[1]["order"]
    assert plans[0]["processed_tokens"] == plans[1]["processed_tokens"] == 48
    assert plans[0]["supervised_tokens"] == plans[1]["supervised_tokens"] == 24


@pytest.mark.parametrize("field,value", [("updates", 0), ("updates", 129), ("rank", 100),
    ("gradient_accumulation", 0), ("optimizer", "muon"), ("learning_rate", .1),
    ("target_modules", ["embed_tokens"]), ("max_grad_norm", -1)])
def test_training_bounds(field, value):
    bank = make_bank(split="train")
    s = spec_for(bank)
    s["training"][field] = value
    with pytest.raises(ValueError):
        native.training_plan(bank, s)


def test_training_needs_opt_in_and_no_eval(bank, tmp_path):
    with pytest.raises(ValueError, match="Training cannot"):
        native.training_plan(bank, spec_for(bank))
    with pytest.raises(ValueError, match="execute"):
        native.train({}, {}, {}, tmp_path / "out")


def test_artifact_and_length_native_boundaries(bank):
    s = spec_for(bank)
    s["evidence_class"] = "native_model"
    with pytest.raises(ValueError, match="boundary"):
        native.check_bank_spec(bank, s)
    s = spec_for(bank)
    s["max_scored_tokens"] = 1
    with pytest.raises(ValueError, match="cap"):
        native.check_bank_spec(bank, s)


def test_local_fingerprints_and_protected_outputs(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "config.json").write_text('{"model_type":"qwen3"}')
    (host / "model.safetensors").write_bytes(b"fingerprint-test-not-a-model")
    before = native.fingerprint(host, "model")
    (host / "model.safetensors").write_bytes(b"changed")
    assert native.fingerprint(host, "model") != before
    with pytest.raises(ValueError, match="Output"):
        native.safe_output_directory(host / "out", {"model_path": str(host), "tokenizer_path": str(host)})


def outcomes(bank, model_id, delta=0.0, vary=True):
    rows = []
    for endpoint in ("skill", "retention"):
        for i, x in enumerate(bank["items"]):
            gain = delta if endpoint == "skill" else 0
            if vary:
                gain += (i % 5 - 2) / 1000
            rows.append({**{k: x[k] for k in ("item_id", "parent_id", "family", "split", "payload_sha256")},
                         "endpoint": endpoint, "value": .5 + gain})
    return {"kind": "behavior_outcomes", "complete": True, "invalid_policy": "count_as_failure",
            "bank_sha256": bank["content_sha256"], "evidence_class": bank["evidence_class"],
            "model_id": model_id, "rows": rows}


def test_behavior_and_degenerate_guard(bank):
    a, b = outcomes(bank, "a", vary=False), outcomes(bank, "b", delta=.2)
    result = metrics.paired_outcomes(a, b, "skill", .1, "higher", repeats=200)
    assert result["verdict"] == "PASS"
    constant = metrics.paired_outcomes(a, outcomes(bank, "b", delta=.2, vary=False), "skill", .1, "higher", repeats=200)
    assert constant["verdict"] == "INCONCLUSIVE_DEGENERATE"
    lower = metrics.paired_outcomes(b, a, "skill", .1, "lower", repeats=200)
    assert lower["verdict"] == "PASS"


def test_behavior_dropping_failures_forbidden(bank):
    a, b = outcomes(bank, "a"), outcomes(bank, "b")
    b["invalid_policy"] = "drop"
    with pytest.raises(ValueError):
        metrics.paired_outcomes(a, b, "skill", .1, "higher", repeats=100)


def freeze_fixture(tmp_path):
    bank = make_bank(split="confirmation")
    inputs = {}
    specs = {"reference_spec": {"evidence_class": "synthetic_fixture", "id": "reference"},
             "candidate_spec": {"evidence_class": "synthetic_fixture", "id": "candidate"}}
    for key, value in {"confirmation_bank": bank, "training_bank": make_bank(split="train"),
                       "development_bank": make_bank(), **specs}.items():
        path = tmp_path / (key + ".json")
        write_new(path, value)
        inputs[key] = str(path)
    settings = {"approved": True, "claim": "Fixture pipeline only", "bootstrap_repeats": 2000, "seed": 17,
        "behavior_gates": [{"endpoint": "skill", "direction": "higher", "mode": "superiority", "margin": .1, "expected_items": 24},
                           {"endpoint": "retention", "direction": "higher", "mode": "noninferiority", "margin": .02, "expected_items": 24}]}
    manifest = protocol.freeze(inputs, settings, str(tmp_path / "ledger"))
    a = make_artifact(bank, model_id=digest(specs["reference_spec"]))
    b = make_artifact(bank, -.1, model_id=digest(specs["candidate_spec"]))
    for name, value in (("ref", a), ("cand", b), ("ref_behavior", outcomes(bank, a["model_id"], vary=False)),
                        ("cand_behavior", outcomes(bank, b["model_id"], delta=.2))):
        if "behavior" in name:
            value["freeze_sha256"] = manifest["content_sha256"]
        write_new(tmp_path / (name + ".json"), value)
    return manifest


def test_one_read_and_no_synthetic_confirmation(tmp_path):
    manifest = freeze_fixture(tmp_path)
    result = protocol.confirm(manifest, str(tmp_path / "ref.json"), str(tmp_path / "cand.json"),
                              str(tmp_path / "ref_behavior.json"), str(tmp_path / "cand_behavior.json"))
    assert result["verdict"] == "SYNTHETIC_OR_TINY_VALIDATION_ONLY"
    assert all(g["verdict"] == "PASS" for g in result["behavior_gates"])
    assert not result["quality_improvement_established"]
    with pytest.raises(FileExistsError):
        protocol.confirm(manifest, "unread", "unread", None, None)


def test_frozen_input_mutation(tmp_path):
    manifest = freeze_fixture(tmp_path)
    (tmp_path / "candidate_spec.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        protocol.consume(manifest)
    assert not (tmp_path / "ledger").exists()


def test_failed_analysis_consumes_read(tmp_path):
    manifest = freeze_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        protocol.confirm(manifest, "not-a-file", "not-a-file", None, None)
    assert (tmp_path / "ledger" / "FAILED.json").exists()
    with pytest.raises(FileExistsError):
        protocol.consume(manifest)


def test_missing_behavior_cannot_establish_quality(tmp_path):
    manifest = freeze_fixture(tmp_path)
    result = protocol.confirm(manifest, str(tmp_path / "ref.json"), str(tmp_path / "cand.json"), None, None)
    assert not result["behavior_gates"] and not result["quality_improvement_established"]


def test_cli_confirmation_bypass_refused(tmp_path):
    freeze_fixture(tmp_path)
    assert main(["compare", "--reference", str(tmp_path / "ref.json"), "--candidate", str(tmp_path / "cand.json"),
                 "--out", str(tmp_path / "forbidden.json")]) == 2
    assert not (tmp_path / "forbidden.json").exists()


def test_cli_native_no_implicit_execution(tmp_path):
    b = make_bank()
    write_new(tmp_path / "b.json", b)
    write_new(tmp_path / "s.json", spec_for(b))
    assert main(["collect", "--spec", str(tmp_path / "s.json"), "--bank", str(tmp_path / "b.json"),
                 "--out", str(tmp_path / "no.json")]) == 2


def test_cli_demo(tmp_path):
    root = tmp_path / "demo"
    assert main(["demo", "--out", str(root)]) == 0
    result = load(root / "demo.json")
    assert all(result["checks"].values()) and result["native_execution"] is False
    assert main(["demo", "--out", str(root)]) == 2


@pytest.mark.parametrize("fault", [None, "context", "target", "duplicate", "missing", "checksum", "rollout"])
def test_external_import_requires_aligned_token_evidence(tmp_path, bank, reference, fault):
    from rgi.importer import import_losses
    rows = copy.deepcopy(reference["rows"])
    if fault == "context":
        rows[0]["context_sha256"] = digest("wrong")
    elif fault == "target":
        rows[0]["target_id"] += 1
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing":
        rows.pop()
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    receipt = {"kind": "external_token_loss_receipt", "records_sha256": file_hash(path),
               **{k: reference[k] for k in ("bank_sha256", "tokenizer_sha256", "evidence_class",
                  "scoring_mode", "context_policy", "model_id", "host_sha256", "components")}}
    if fault == "checksum":
        receipt["records_sha256"] = digest("wrong")
    elif fault == "rollout":
        receipt["context_policy"] = "self_rollout"
    receipt = seal(receipt)
    if fault:
        with pytest.raises(ValueError):
            import_losses(bank, str(path), receipt)
    else:
        converted = import_losses(bank, str(path), receipt)
        assert converted["rows"] == reference["rows"]
        assert converted["source_provenance"]["records_sha256"] == file_hash(path)


def test_three_seed_control_plan_matrix():
    pair = data.match_controls(matching_bank(), 15)
    matrix = native.control_plan_matrix(pair["targeted"], pair["control"], spec_for(pair["targeted"]), [17, 23, 29])
    assert len(matrix["arms"]) == 6 and not matrix["training_executed"]
    for seed in (17, 23, 29):
        a, b = (matrix["arms"][f"{name}_s{seed}"] for name in ("targeted", "control"))
        assert a["plan"]["order"] == b["plan"]["order"]
        assert a["spec"]["training"]["seed"] == b["spec"]["training"]["seed"] == seed


@pytest.mark.parametrize("fault", ["seeds", "length", "frame", "labels"])
def test_control_plan_matrix_guards(fault):
    pair = data.match_controls(matching_bank(), 15)
    t, c, seeds = pair["targeted"], pair["control"], [17, 23]
    if fault == "seeds":
        seeds = [17, 17]
    elif fault == "length":
        c["items"][0]["input_ids"].append(1)
        c["items"][0]["roles"].append("format")
    elif fault == "frame":
        c["items"][0]["format_id"] = "wrong"
    else:
        c["items"][0]["roles"][1] = "content"
    c = reseal(c)
    with pytest.raises(ValueError):
        native.control_plan_matrix(t, c, spec_for(t), seeds)


def fingerprint_spec(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3"}')
    (root / "model.safetensors").write_bytes(b"guard-fixture-not-a-model")
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    return {"model_path": str(root), "tokenizer_path": str(root),
        "model_sha256": native.fingerprint(root, "model"), "tokenizer_sha256": native.fingerprint(root, "tokenizer"),
        "evidence_class": "native_model", "precision": "float32", "device": "cpu",
        "max_length": 512, "max_scored_tokens": 1000, "adapters": []}


@pytest.mark.parametrize("fault", [None, "host_hash", "tokenizer_hash", "precision", "device", "length", "cap", "architecture", "quantized", "remote_code"])
def test_native_spec_guard_without_loading_dependencies(tmp_path, fault):
    spec = fingerprint_spec(tmp_path)
    mapping = {"host_hash": ("model_sha256", digest("bad")), "tokenizer_hash": ("tokenizer_sha256", digest("bad")),
               "precision": ("precision", "bfloat16"), "device": ("device", "auto"), "length": ("max_length", 9000),
               "cap": ("max_scored_tokens", 0)}
    if fault in mapping:
        key, value = mapping[fault]
        spec[key] = value
    if fault in {"architecture", "quantized", "remote_code"}:
        config = {"model_type": "qwen3"}
        if fault == "architecture":
            config["model_type"] = "qwen3_moe"
        elif fault == "quantized":
            config["quantization_config"] = {"bits": 4}
        else:
            config["auto_map"] = {"AutoModel": "custom.py"}
        (Path(spec["model_path"]) / "config.json").write_text(json.dumps(config))
        spec["model_sha256"] = native.fingerprint(spec["model_path"], "model")
    if fault:
        with pytest.raises(ValueError):
            native.inspect_spec(spec)
    else:
        assert native.inspect_spec(spec)["host_sha256"] == spec["model_sha256"]


@pytest.mark.parametrize("fault", [None, "dora", "modules_to_save", "bias", "duplicate", "coefficient", "hash"])
def test_registered_adapter_contract(tmp_path, fault):
    spec = fingerprint_spec(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {"peft_type": "LORA", "bias": "none"}
    if fault == "dora":
        config["use_dora"] = True
    elif fault == "modules_to_save":
        config["modules_to_save"] = ["embed_tokens"]
    elif fault == "bias":
        config["bias"] = "all"
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapter_model.safetensors").write_bytes(b"guard-only-fixture")
    entry = {"path": str(adapter), "sha256": native.fingerprint(adapter, "adapter"), "coefficient": .25}
    if fault == "coefficient":
        entry["coefficient"] = 0
    elif fault == "hash":
        entry["sha256"] = digest("bad")
    spec["adapters"] = [entry, entry] if fault == "duplicate" else [entry]
    if fault:
        with pytest.raises(ValueError):
            native.inspect_spec(spec)
    else:
        assert native.inspect_spec(spec)["components"] == {entry["sha256"]: .25}


def test_execution_contract_alignment(reference):
    other = copy.deepcopy(reference)
    other["execution_contract"] = {"collector": "different-runtime"}
    with pytest.raises(ValueError, match="execution_contract"):
        align(reference, reseal(other))
