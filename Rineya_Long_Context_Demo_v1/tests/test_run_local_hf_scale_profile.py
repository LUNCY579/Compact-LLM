from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_hf.py"
SPEC = importlib.util.spec_from_file_location("rineya_run_local_hf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_constant_token_bias_changes_only_the_selected_logit() -> None:
    scores = torch.zeros(2, 8)
    result = MODULE.ConstantTokenBiasProcessor(token_id=3, bias=5.25)(None, scores)
    assert result is scores
    assert torch.count_nonzero(result) == 2
    torch.testing.assert_close(result[:, 3], torch.full((2,), 5.25))


def test_profile_resolution_combines_global_layer_and_module_scales(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "label": "targeted",
                "default_multiplier": 0.5,
                "layer_multipliers": {"7": 2.0},
                "module_multipliers": {"q_proj": 0.25},
            }
        ),
        encoding="utf-8",
    )
    profile, digest = MODULE.load_adapter_scale_profile(profile_path)

    targeted, layer, projection = MODULE.adapter_scale_for_module(
        "base_model.model.model.layers.7.self_attn.q_proj", 0.8, profile
    )
    default, _, _ = MODULE.adapter_scale_for_module(
        "base_model.model.model.layers.8.self_attn.o_proj", 0.8, profile
    )

    assert targeted == pytest.approx(0.4)
    assert default == pytest.approx(0.4)
    assert layer == 7
    assert projection == "q_proj"
    assert len(digest) == 64


@pytest.mark.parametrize(
    "profile, message",
    [
        ({"module_multipliers": {"bogus": 1.0}}, "Unknown module"),
        ({"layer_multipliers": {"-1": 1.0}}, "Invalid layer"),
        ({"default_multiplier": float("inf")}, "finite number"),
        ({"unexpected": 1}, "Unknown adapter scale profile fields"),
    ],
)
def test_profile_validation_rejects_ambiguous_recipes(
    tmp_path: Path, profile: dict[str, object], message: str
) -> None:
    profile_path = tmp_path / "bad.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        MODULE.load_adapter_scale_profile(profile_path)
