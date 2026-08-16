from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from rineya_soup_lab.model.layer_map import MergeRecipe


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_release_json_schemas_are_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(_schema("coefficient_recipe.schema.json"))
    Draft202012Validator.check_schema(_schema("run_manifest.schema.json"))


def test_documented_layerwise_recipe_matches_schema() -> None:
    schema = _schema("coefficient_recipe.schema.json")
    payload = json.loads(
        (ROOT / "examples" / "recipe.layerwise.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(payload)


def test_runtime_endpoint_and_merge_recipes_match_schema() -> None:
    validator = Draft202012Validator(_schema("coefficient_recipe.schema.json"))
    validator.validate(MergeRecipe.endpoint_recipe("anchor").to_dict())
    validator.validate(MergeRecipe.endpoint_recipe("greedy").to_dict())
    validator.validate(MergeRecipe(global_alpha=0.6, layer_alphas={4: 0.8}).to_dict())
