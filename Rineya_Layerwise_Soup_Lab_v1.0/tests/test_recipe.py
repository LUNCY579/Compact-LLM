import pytest

from rineya_soup_lab.errors import ConfigurationError
from rineya_soup_lab.model.layer_map import MergeRecipe, ParameterClassifier


def test_recipe_identity_ignores_label_and_metadata() -> None:
    first = MergeRecipe(global_alpha=0.5, layer_alphas={2: 0.75}, label="one")
    second = MergeRecipe(
        global_alpha=0.5,
        layer_alphas={2: 0.75},
        label="two",
        metadata={"note": "same weights"},
    )
    assert first.id == second.id
    assert first.identity_dict() == second.identity_dict()


def test_parameter_classifier() -> None:
    classifier = ParameterClassifier(r"(?:^|\.)layers\.(\d+)\.")
    assert classifier.classify("model.layers.7.mlp.down_proj.weight").layer_index == 7
    assert classifier.classify("model.embed_tokens.weight").group == "embedding"
    assert classifier.classify("lm_head.weight").group == "lm_head"


def test_endpoint_recipe_cannot_hide_interpolation_overrides() -> None:
    with pytest.raises(ConfigurationError):
        MergeRecipe(kind="endpoint", endpoint="anchor", global_alpha=0.5)
    with pytest.raises(ConfigurationError):
        MergeRecipe(
            kind="endpoint",
            endpoint="greedy",
            global_alpha=1.0,
            layer_alphas={0: 0.0},
        )
