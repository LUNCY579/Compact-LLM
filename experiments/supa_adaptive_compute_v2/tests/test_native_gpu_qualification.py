"""Independent native qualification report audit; no model execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from shared.datasets import load_frozen_dataset
from shared.qualifications import validate_native_qualification_report as validate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_gpu_qualifier_test", ROOT / "scripts/qualify_native_gpu.py")
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


@pytest.fixture(scope="module")
def frozen_inputs():
    config = json.loads((ROOT / "native_armr/configs/protocol_gpu_v2_2.json").read_text())
    rows = load_frozen_dataset(ROOT / "data/generated/v2/native_armr/manifest.json",
                               track="native_armr", role="calibration")
    return config, rows


@pytest.fixture
def valid_report(frozen_inputs):
    config, rows = frozen_inputs
    chosen = qualifier.select_fixtures(rows)
    metric = dict(max_abs=0., relative_l2=0., cosine=1.)
    comparisons = [dict(arm=arm, example_id=row["example_id"], logits=copy.deepcopy(metric),
        hidden={str(site): copy.deepcopy(metric) for site in (8, 12, 13, 16, 19)},
        absolute_nll_error=0., cpu_token_ids=[11, 19], gpu_token_ids=[11, 19],
        gpu_repeat_identity=True, **{"pass": True}) for arm in config["arms"] for row in chosen]
    files = {"model.safetensors": config["checkpoint_sha256"], "config.json": "a" * 64}
    identity = dict(model_class="Qwen3AbstainForCausalLM", canonical_unchanged=True,
        trainable_parameters=0, backbone_gradients=0, state_sha256_before="b" * 64,
        state_sha256_after="b" * 64, checkpoint_files_before=files,
        checkpoint_files_after=copy.deepcopy(files), operator_hashes={"AM": "c" * 64})
    return dict(qualification_pass=True, fixture_ids=[r["example_id"] for r in chosen],
        fixture_prompt_sha256={r["example_id"]: r["prompt_sha256"] for r in chosen},
        thresholds=copy.deepcopy(config["qualification"]), comparisons=comparisons,
        cpu_identity=identity, gpu_identity=copy.deepcopy(identity),
        same_device_disabled_restored_identity=True)


def test_complete_report_and_fixture_selection_are_order_independent(valid_report, frozen_inputs):
    config, rows = frozen_inputs
    assert len(valid_report["comparisons"]) == 40
    assert qualifier.select_fixtures(rows) == qualifier.select_fixtures(rows[::-1])
    assert validate(valid_report, config, rows)
    valid_report["comparisons"].reverse()
    assert validate(valid_report, config, rows[::-1])


@pytest.mark.parametrize("change", ["empty", "missing_arm", "duplicate", "changed_fixture", "wrong_prompt", "loosened_threshold"])
def test_complete_claim_does_not_replace_frozen_coverage(valid_report, frozen_inputs, change):
    config, rows = frozen_inputs
    if change == "empty":
        valid_report["comparisons"] = []
    elif change == "missing_arm":
        valid_report["comparisons"] = [r for r in valid_report["comparisons"] if r["arm"] != "AM"]
    elif change == "duplicate":
        valid_report["comparisons"][-1] = copy.deepcopy(valid_report["comparisons"][0])
    elif change == "changed_fixture":
        valid_report["fixture_ids"][0] = "unregistered"
    elif change == "wrong_prompt":
        valid_report["fixture_prompt_sha256"][valid_report["fixture_ids"][0]] = "changed"
    else:
        valid_report["thresholds"]["max_abs_nll_error"] = 1.
    with pytest.raises(RuntimeError):
        validate(valid_report, config, rows)


@pytest.mark.parametrize("path,value", [
    ("comparisons.0.logits.relative_l2", float("nan")),
    ("comparisons.0.logits.max_abs", float("inf")),
    ("comparisons.0.logits.max_abs", .003),
    ("comparisons.0.hidden.12.relative_l2", .001),
    ("comparisons.0.hidden.19.cosine", .9),
    ("comparisons.0.absolute_nll_error", .003),
    ("comparisons.0.gpu_token_ids", [11, 20]),
    ("comparisons.0.gpu_repeat_identity", False),
    ("gpu_identity.operator_hashes.AM", "changed"),
    ("gpu_identity.state_sha256_after", "changed"),
    ("gpu_identity.trainable_parameters", 1),
    ("cpu_identity.backbone_gradients", 1),
    ("same_device_disabled_restored_identity", False)])
def test_numerical_and_frozen_identity_gates_are_recomputed(valid_report, frozen_inputs, path, value):
    target = valid_report
    parts = path.split(".")
    for key in parts[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[parts[-1]] = value
    with pytest.raises(RuntimeError):
        validate(valid_report, *frozen_inputs)


def test_preregistered_threshold_boundaries_are_inclusive(valid_report, frozen_inputs):
    config, rows = frozen_inputs
    q, row = config["qualification"], valid_report["comparisons"][0]
    row["logits"] = dict(max_abs=q["max_abs_logit_error"], relative_l2=q["max_relative_logit_l2"], cosine=q["minimum_logit_cosine"])
    row["absolute_nll_error"] = q["max_abs_nll_error"]
    row["hidden"]["12"].update(relative_l2=q["max_relative_hidden_l2"], cosine=q["minimum_hidden_cosine"])
    assert validate(valid_report, config, rows)


def test_numeric_agreement_exposes_low_norm_token_drift():
    reference = torch.tensor([[1000., 1000.], [1., 1.]], dtype=torch.float64)
    actual = reference.clone()
    actual[1] += .01
    global_metric = qualifier.numeric_agreement(reference, actual)
    token_metric = qualifier.numeric_agreement(reference, actual, tokenwise=True)
    assert token_metric["relative_l2"] == pytest.approx(.01)
    assert token_metric["relative_l2"] > 100 * global_metric["relative_l2"]
    assert qualifier.numeric_agreement(reference, reference)["max_abs"] == 0
    with pytest.raises(FloatingPointError):
        qualifier.numeric_agreement(reference, actual.fill_(float("nan")))
