"""Analytic trajectories and gate boundary tests, independent of model outcomes."""
from pathlib import Path
import json

import pytest
import torch

from armr.metrics import canonical_answer, hidden_metrics
from armr.reporting import summarize


def test_known_contracting_trajectory():
    states = [torch.tensor([4., 2.]), torch.tensor([2., 1.]), torch.tensor([1., .5])]
    first, delta = hidden_metrics(states[1], states[1], states[0], None, 1)
    second, _ = hidden_metrics(states[2], states[1], states[1], delta, 2)
    assert first["rho"] is None
    assert second["rho"] == pytest.approx(.5)
    assert second["rms_ratio_to_normal"] == pytest.approx(.5)
    assert second["delta_cosine"] == pytest.approx(1.)


def test_known_oscillating_trajectory():
    a = torch.tensor([2., 1.])
    first, delta = hidden_metrics(-a, -a, a, None, 1)
    second, _ = hidden_metrics(a, -a, -a, delta, 2)
    assert second["rho"] == pytest.approx(1.)
    assert second["delta_cosine"] == pytest.approx(-1.)
    assert second["cosine_to_normal"] == pytest.approx(-1.)


def test_gold_does_not_misuse_keyword_as_exact_answer():
    assert canonical_answer({"scorer":{"type":"keywords","phrase":"ampere"}}) is None
    assert canonical_answer({"scorer":{"type":"numeric","value":803}}) == "803"
    assert canonical_answer({"scorer":{"type":"exact","answers":["yes","yeah"]}}) is None


def row():
    return {"hidden": [], "trajectory_min_rms_ratio":1., "trajectory_max_rms_ratio":1.,
            "invalid":False, "logit_cosine":.95, "kl_base_to_arm":.1, "top1_agreement":1,
            "topk_overlap":.9, "score":1., "format_score":1., "safety_stopped":False,
            "eos_emitted":True, "wall_seconds":1.}


@pytest.mark.parametrize("key,value,failed_check", [
    ("logit_cosine", .8999, "logit_preservation"),
    ("kl_base_to_arm", .5001, "kl_preservation"),
    ("score", 0., "score_preservation"),
    ("invalid", True, "validity_preservation"),
    ("safety_stopped", True, "no_safety_stops"),
    ("trajectory_max_rms_ratio", 4.01, "stable_norms"),
    ("trajectory_min_rms_ratio", .249, "stable_norms"),
])
def test_each_frozen_gate_independently_blocks(key, value, failed_check):
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/reapplication_screen.json").read_text())
    baseline=[row() for _ in range(8)]
    passing=summarize([row() for _ in range(8)],baseline,config)
    assert passing["go_fixed_smoke"]
    bad=[{**row(),key:value} for _ in range(8)]
    result=summarize(bad,baseline,config)
    assert not result["go_fixed_smoke"]
    assert not result["gate_checks"][failed_check]


def test_incomplete_stage_cannot_pass():
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/reapplication_screen.json").read_text())
    result=summarize([row()],[row(),row()],config)
    assert not result["go_fixed_smoke"]
