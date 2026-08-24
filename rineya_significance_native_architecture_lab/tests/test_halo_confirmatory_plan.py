from __future__ import annotations

import importlib.util
from pathlib import Path


def test_confirmatory_plan_is_model_free_and_cluster_based() -> None:
    lab = Path(__file__).resolve().parents[1]
    script = lab / "scripts" / "plan_halo_confirmatory.py"
    spec = importlib.util.spec_from_file_location("halo_plan", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_plan(lab / "configs" / "confirmatory.yaml")
    assert plan["status"] == "PLANNING_ONLY_NOT_REGISTERED"
    assert plan["calibration_stage"]["independent_clusters"] == 32
    assert plan["candidate_confirmatory_stage"]["inference_unit"] == "cluster_id"
    assert len(plan["mde_sensitivity"]) == 12
