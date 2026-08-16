from pathlib import Path

from rineya_soup_lab.state import RunState


def test_candidate_alias_registration_preserves_retention(tmp_path: Path) -> None:
    state = RunState.open(tmp_path / "state.json", run_id="run", config_digest="cfg")
    recipe = {
        "schema_version": "1.0",
        "kind": "merge",
        "endpoint": None,
        "global_alpha": 0.5,
        "layer_alphas": {},
        "group_alphas": {},
        "label": "uniform",
    }
    state.register_candidate(
        "cand",
        {
            "label": "uniform",
            "recipe": recipe,
            "retained": True,
            "materialized": True,
            "model_path": "candidate",
            "evaluations": {"dev:p": {"objective": 0.8}},
        },
    )
    state.register_candidate(
        "cand",
        {
            "label": "global_alpha_0.5000",
            "recipe": {**recipe, "label": "global_alpha_0.5000"},
            "retained": False,
            "materialized": True,
            "model_path": "candidate",
            "evaluations": {},
        },
    )
    record = state.candidate("cand")
    assert record["retained"] is True
    assert record["label"] == "uniform"
    assert record["evaluations"]["dev:p"]["objective"] == 0.8
    assert set(record["labels"]) == {"uniform", "global_alpha_0.5000"}
