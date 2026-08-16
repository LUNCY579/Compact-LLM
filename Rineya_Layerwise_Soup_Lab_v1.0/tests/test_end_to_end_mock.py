from __future__ import annotations

from pathlib import Path

import yaml

from conftest import write_dataset
from rineya_soup_lab.config import load_config
from rineya_soup_lab.pipeline import ExperimentPipeline
from rineya_soup_lab.run_context import RunContext


def test_complete_mock_pipeline(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=3)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=3)
    dev = write_dataset(tmp_path / "dev.jsonl", count=14, prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", count=14, prefix="test")
    repo_root = Path(__file__).resolve().parents[1]
    config_payload = {
        "project": {"run_root": str(tmp_path / "runs"), "run_name": "mock_run"},
        "models": {
            "anchor_path": str(anchor),
            "greedy_path": str(greedy),
            "tokenizer_path": str(anchor),
            "expected_layers": 3,
            "lineage_acknowledged": True,
            "max_shard_size_gb": 0.000001,
        },
        "datasets": {"dev": str(dev), "test": str(test)},
        "evaluation": {
            "backend": "mock",
            "mock_allowed": True,
            "quality_config": str(repo_root / "configs" / "official_v45" / "quality_evaluation.json"),
        },
        "statistics": {"bootstrap_samples": 100, "minimum_accept_probability": 0.50},
        "experiments": {
            "global_sweep": {
                "alphas": [0.0, 0.5, 1.0],
                "refine_around_best": False,
            },
            "probability_profile": {"enabled": False},
            "layer_influence": {
                "layers": "all",
                "bidirectional_swaps": True,
                "centered_alpha_grid": [0.0, 0.5, 1.0],
                "smoothing_penalty": 0.05,
                "global_shrinkage_penalty": 0.1,
            },
            "layer_search": {
                "max_passes": 1,
                "alpha_grid": [0.0, 0.5, 1.0],
                "max_candidate_evaluations": 8,
                "pairwise_top_k": 0,
            },
        },
        "reporting": {"make_plots": False},
        "storage": {"delete_ephemeral_after_evaluation": True, "max_disk_gb": 2},
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    context = RunContext.create(config)
    ExperimentPipeline(context).run()

    assert context.path("manifests", "compatibility.json").exists()
    assert context.path("analysis", "global", "interpolation_curve.csv").exists()
    assert context.path("analysis", "layers", "per_response_layer_influence.jsonl").exists()
    assert context.path("selection", "selection_freeze.json").exists()
    assert context.path("sealed", "sealed_test_result.json").exists()
    assert context.path("reports", "RESEARCH_REPORT.md").exists()
    assert context.state.stage_status("report") == "complete"
