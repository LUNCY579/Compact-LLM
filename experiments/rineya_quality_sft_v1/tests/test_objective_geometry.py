"""Tiny autograd/algebra fixtures; no model, owner bank or native capture."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from experiments.rineya_quality_sft_v1 import objective_geometry as G
from experiments.rineya_quality_sft_v1 import integrity as I
from experiments.rineya_quality_sft_v1.compiler import targets, validate_plan
from experiments.rineya_quality_sft_v1.tests.test_bank_compiler import fixture_plan, fixture_rows


def test_complete_group_partition_includes_history_header_and_post_eos_newline(tmp_path):
    rows = fixture_rows()
    rows["train"][0]["messages"] = [{"role": "system", "content": "Fixture contract"},
        {"role": "user", "content": "Original earlier query"}, {"role": "assistant", "content": "Earlier history answer"},
        {"role": "user", "content": "Final original query"}, {"role": "assistant", "content": "A"}]
    rows["train"][2]["messages"][-1]["content"] = "Longer verified fixture final answer"
    plan = fixture_plan(tmp_path, rows)
    body = validate_plan(plan)
    report = G.describe_plan_geometry(plan)
    assert report["all_worlds_preserved"] and report["order_and_groups_preserved"]
    assert [i for group in report["groups"] for i in group["indices"]] == list(range(200))
    assert report["groups_sha256"] == I.digest(body["groups"])
    for group in report["groups"]:
        whole = response = complement = 0
        for index in group["indices"]:
            row = body["train"][index]
            w, r = targets(row, "whole"), targets(row, "response")
            whole += sum(label != -100 for label in w[1:])
            response += sum(label != -100 for label in r[1:])
            complement += sum(a != -100 and b == -100 for a, b in zip(w[1:], r[1:]))
            assert all(r[i] == -100 for i in range(row["answer_end"] + 1, len(r)))
        assert group["whole_targets"] == whole
        assert group["response_targets"] == response
        assert group["prompt_targets"] == complement == whole - response
        assert group["wr"] == response / whole and group["wp"] == complement / whole
        assert group["wr"] + group["wp"] == pytest.approx(1.)
    assert not report["native_gain"] and not report["gradient_direction_observed"]


def _autograd_case(prompt_target):
    parameter = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    logits = torch.stack((parameter.expand(24672), torch.zeros(24672, dtype=torch.float64)), dim=-1)
    labels = torch.cat((torch.zeros(1944, dtype=torch.long), torch.full((22728,), prompt_target, dtype=torch.long)))
    response_loss = F.cross_entropy(logits[:1944], labels[:1944])
    prompt_loss = F.cross_entropy(logits[1944:], labels[1944:])
    whole_loss = F.cross_entropy(logits, labels)
    r = torch.autograd.grad(response_loss, parameter, retain_graph=True)[0]
    p = torch.autograd.grad(prompt_loss, parameter, retain_graph=True)[0]
    w = torch.autograd.grad(whole_loss, parameter)[0]
    return G.gradient_geometry(iter([{"name": "tiny_scalar", "response": r, "prompt": p, "whole": w}]), 1944, 22728)


def test_opposed_prompt_term_reverses_whole_gradient_despite_response_supervision():
    report = _autograd_case(1)
    assert report["wr"] == pytest.approx(1944 / 24672)
    assert report["wr"] == pytest.approx(.0788, abs=.00001)
    assert report["mixture_verified"]
    assert report["cosines"]["response_prompt"] == pytest.approx(-1.)
    assert report["cosines"]["response_whole"] == pytest.approx(-1.)
    assert report["weighted_signal_cancellation_ratio"] < 1.
    assert not report["native_gain"] and not report["actual_parameter_update_inferred"]


def test_aligned_prompt_has_same_whole_direction_with_identical_weights():
    report = _autograd_case(0)
    assert report["cosines"]["response_prompt"] == pytest.approx(1.)
    assert report["cosines"]["response_whole"] == pytest.approx(1.)
    assert report["weighted_signal_cancellation_ratio"] == pytest.approx(1.)


def test_streamed_blocks_equal_independent_flattened_dot_norms_and_clipping():
    r = torch.tensor([1., -2., 3., 4.], dtype=torch.float64)
    p = torch.tensor([2., 3., -1., 2.], dtype=torch.float64)
    w = .25 * r + .75 * p
    blocks = ({"name": str(i), "response": r[start:end], "prompt": p[start:end], "whole": w[start:end]}
              for i, (start, end) in enumerate(((0, 1), (1, 4))))
    report = G.gradient_geometry(blocks, 1, 3)
    assert report["blocks"] == 2 and report["elements"] == 4
    assert report["squared_norms_and_dots"]["rp"] == float(torch.dot(r, p))
    assert report["norms"]["whole"] == pytest.approx(float(torch.linalg.vector_norm(w)))
    assert report["gram_expected_whole_squared_norm"] == pytest.approx(float(w.square().sum()))
    assert report["clipping"]["whole"]["clipped_norm"] <= 1.
    assert report["actual_parameter_update_l2"] is None


@pytest.mark.parametrize("mutation", ["mixture", "nan", "dtype", "shape", "names", "empty", "integer", "oversize"])
def test_gradient_capture_validation_is_fail_closed(mutation):
    r = torch.tensor([1., 2.], dtype=torch.float64)
    p = torch.tensor([2., 1.], dtype=torch.float64)
    block = {"name": "a", "response": r, "prompt": p, "whole": (r + p) / 2}
    blocks = [block]
    if mutation == "mixture": block["whole"] += 1.
    if mutation == "nan": block["response"][0] = float("nan")
    if mutation == "dtype": block["prompt"] = p.float()
    if mutation == "shape": block["prompt"] = p.reshape(1, 2)
    if mutation == "names": blocks.append(copy.copy(block))
    if mutation == "empty": blocks = []
    if mutation == "integer": block["response"] = torch.tensor([1, 2])
    if mutation == "oversize": block["response"] = torch.empty(G.MAX_BLOCK_ELEMENTS + 1, dtype=torch.float64)
    with pytest.raises(ValueError): G.gradient_geometry(iter(blocks), 1, 1)


def test_zero_mixture_has_undefined_cosine_not_false_perfect_alignment():
    report = G.gradient_geometry(iter([{"name": "cancel", "response": torch.ones(2),
        "prompt": -torch.ones(2), "whole": torch.zeros(2)}]), 1, 1)
    assert report["norms"]["whole"] == 0 and report["cosines"]["response_whole"] is None
    assert report["weighted_signal_cancellation_ratio"] == 0


def test_fp32_near_cancellation_uses_signal_scale_without_accepting_bad_tiny_mixtures():
    r = torch.tensor([1., -2.], dtype=torch.float32)
    report = G.gradient_geometry(iter([{"name": "rounding", "response": r, "prompt": -r,
        "whole": torch.tensor([1e-7, -2e-7])}]), 1, 1)
    assert report["mixture_verified"] and report["mixture_relative_error"] < 1e-5
    tiny = torch.tensor([1e-12], dtype=torch.float32)
    with pytest.raises(ValueError, match="global signal-scaled"):
        G.gradient_geometry(iter([{"name": "bad_tiny", "response": tiny, "prompt": tiny,
            "whole": torch.tensor([1e-8])}]), 1, 1)
    zero = torch.zeros(2, dtype=torch.float32)
    exact = G.gradient_geometry(iter([{"name": "allzero", "response": zero, "prompt": zero, "whole": zero}]), 1, 1)
    assert exact["mixture_relative_error"] is None and exact["mixture_error_l2"] == 0
    assert exact["cosines"]["response_whole"] is None
    with pytest.raises(ValueError, match="both means"):
        G.gradient_geometry(iter([{"name": "undefined_prompt", "response": r, "prompt": zero, "whole": r}]), 1, 0)


def test_tensor_device_materialization_and_float64_overflow_fail_closed():
    with pytest.raises(ValueError):
        G.gradient_geometry(iter([{"name": "meta", "response": torch.empty(2, device="meta"),
            "prompt": torch.ones(2), "whole": torch.ones(2)}]), 1, 1)
    enormous = torch.tensor([1e200], dtype=torch.float64)
    with pytest.raises(ValueError, match="overflow"):
        G.gradient_geometry(iter([{"name": "overflow", "response": enormous, "prompt": enormous, "whole": enormous}]), 1, 1)
    with pytest.raises(ValueError, match="gradient norm"):
        G.first_step_adam_delta(torch.full((4,), 1e308, dtype=torch.float64), 1e-5, 1e-8)


def test_positive_gradient_scale_does_not_scale_first_adam_delta_after_clipping():
    gradient = torch.tensor([2., -3., .5], dtype=torch.float64)
    scale = 24672 / 1944
    ordinary = G.first_step_adam_delta(gradient, 1e-5, 1e-8)
    amplified = G.first_step_adam_delta(gradient * scale, 1e-5, 1e-8)
    ratio = float(torch.linalg.vector_norm(amplified) / torch.linalg.vector_norm(ordinary))
    assert ratio == pytest.approx(1., abs=1e-7)
    assert ratio != pytest.approx(scale)
    assert not torch.allclose(amplified, scale * ordinary, rtol=1e-3, atol=1e-12)


def test_first_adam_equation_is_epsilon_aware_and_matches_independent_bias_correction():
    gradient = torch.tensor([1e-12, -2e-12], dtype=torch.float64)
    lr, eps, beta1, beta2 = 1e-5, 1e-8, .9, .95
    m = (1 - beta1) * gradient
    v = (1 - beta2) * gradient.square()
    expected = -lr * (m / (1 - beta1)) / ((v / (1 - beta2)).sqrt() + eps)
    torch.testing.assert_close(G.first_step_adam_delta(gradient, lr, eps), expected, rtol=1e-12, atol=1e-20)
    scaled = G.first_step_adam_delta(gradient * 12.69, lr, eps)
    assert float(torch.linalg.vector_norm(scaled) / torch.linalg.vector_norm(expected)) > 10
    # Near epsilon is precisely why the helper does not claim universal scale invariance.


def test_plan_cli_is_metadata_only_fixture_optin_and_create_only(tmp_path, monkeypatch):
    folder = tmp_path / "fixture"
    folder.mkdir()
    sealed = fixture_plan(folder)
    path = tmp_path / "plan.json"
    I.write_new(path, sealed)
    out = tmp_path / "geometry.json"
    monkeypatch.setattr(G, "external_output", lambda p: p)
    args = ["--plan", str(path), "--output", str(out)]
    with pytest.raises(ValueError, match="explicit --fixture-only"):
        G.main(args)
    assert not out.exists()
    assert G.main(args + ["--fixture-only"]) == 0
    report = I.read_json(out)
    assert report["evidence"] == "PLAN_METADATA_ONLY" and report["native_model_calls"] == 0
    assert not report["training_authorized"] and not report["native_rqsft_recipe_changed"]
    assert "messages" not in json.dumps(report)
    with pytest.raises(FileExistsError): G.main(args + ["--fixture-only"])
