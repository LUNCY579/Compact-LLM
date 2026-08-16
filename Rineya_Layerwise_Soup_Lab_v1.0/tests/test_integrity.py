from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import write_dataset
from rineya_soup_lab.candidates import CandidateManager
from rineya_soup_lab.config import load_config
from rineya_soup_lab.errors import ConfigurationError, SealedTestError
from rineya_soup_lab.experiments.global_soup import evaluate_global_alpha
from rineya_soup_lab.experiments.preflight import run_preflight
from rineya_soup_lab.model.checkpoint import (
    checkpoint_fingerprint,
    checkpoint_fingerprint_digest,
)
from rineya_soup_lab.model.compatibility import compare_checkpoints
from rineya_soup_lab.model.layer_map import MergeRecipe
from rineya_soup_lab.model.merge import merge_checkpoints
from rineya_soup_lab.pipeline import ExperimentPipeline
from rineya_soup_lab.run_context import RunContext


def _config_payload(tmp_path: Path, anchor: Path, greedy: Path, dev: Path, test: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "project": {"run_root": str(tmp_path / "runs"), "run_name": "integrity_run"},
        "models": {
            "anchor_path": str(anchor),
            "greedy_path": str(greedy),
            "tokenizer_path": str(anchor),
            "expected_layers": 2,
            "lineage_acknowledged": True,
            "max_shard_size_gb": 0.000001,
        },
        "datasets": {"dev": str(dev), "test": str(test)},
        "evaluation": {
            "backend": "mock",
            "mock_allowed": True,
            "quality_config": str(
                repo_root / "configs" / "official_v45" / "quality_evaluation.json"
            ),
        },
        "statistics": {"bootstrap_samples": 50, "minimum_accept_probability": 0.50},
        "experiments": {
            "global_sweep": {"alphas": [0.0, 0.5, 1.0], "refine_around_best": False},
            "probability_profile": {"enabled": False},
            "layer_influence": {
                "layers": "all",
                "bidirectional_swaps": True,
                "centered_alpha_grid": [0.0, 0.5, 1.0],
            },
            "layer_search": {
                "max_passes": 1,
                "alpha_grid": [0.0, 0.5, 1.0],
                "max_candidate_evaluations": 4,
                "pairwise_top_k": 0,
            },
        },
        "reporting": {"make_plots": False},
        "storage": {"delete_ephemeral_after_evaluation": True, "max_disk_gb": 2},
    }


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_config_requires_explicit_shared_lineage_acknowledgement(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["models"]["lineage_acknowledged"] = False
    with pytest.raises(ConfigurationError, match="lineage_acknowledged"):
        load_config(_write_config(tmp_path / "config.yaml", payload))


def test_merge_manifest_fingerprints_materialized_checkpoint(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    destination = tmp_path / "merged"
    merge_checkpoints(
        anchor,
        greedy,
        destination,
        MergeRecipe(global_alpha=0.5, label="uniform"),
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        max_shard_size_gb=0.000001,
    )
    manifest = json.loads((destination / "merge_manifest.json").read_text(encoding="utf-8"))
    current = checkpoint_fingerprint_digest(checkpoint_fingerprint(destination))
    assert manifest["output_checkpoint_digest"] == current
    assert manifest["output_checkpoint_fingerprint"]["path"] == str(destination.resolve())


def test_configured_tokenizer_mismatch_is_rejected(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tokenizer = tmp_path / "other_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 999, "chat_template": "different"}),
        encoding="utf-8",
    )
    report = compare_checkpoints(
        anchor,
        greedy,
        layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
        expected_layers=2,
        allow_quantized_sources=False,
        require_non_floating_equal=True,
        lineage_acknowledged=True,
        tokenizer_path=tokenizer,
    )
    assert report["compatible"] is False
    assert any("Configured tokenizer" in error for error in report["errors"])


def test_run_context_refuses_silent_reuse_of_named_nonempty_directory(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            _config_payload(tmp_path, anchor, greedy, dev, test),
        )
    )
    first = RunContext.create(config)
    assert first.root.exists()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        RunContext.create(config)


def test_uniform_endpoint_still_receives_uniform_alias(tmp_path: Path, tiny_checkpoint_factory) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            _config_payload(tmp_path, anchor, greedy, dev, test),
        )
    )
    context = RunContext.create(config)
    sources = run_preflight(context)
    manager = CandidateManager(context, sources)
    manager.evaluate(MergeRecipe.endpoint_recipe("anchor"), alias="anchor", retain=True)
    manager.evaluate(MergeRecipe.endpoint_recipe("greedy"), alias="greedy", retain=True)
    result = evaluate_global_alpha(context, manager, 0.0, alias="uniform")
    assert context.state.resolve_alias("uniform") == result.candidate_id
    assert result.recipe.endpoint == "anchor"


def test_sealed_dataset_mutation_after_freeze_is_rejected(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", count=10, prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", count=10, prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    config = load_config(_write_config(tmp_path / "config.yaml", payload))
    context = RunContext.create(config)
    pipeline = ExperimentPipeline(context)
    pipeline.run(through="freeze")
    with test.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "test_mutated",
                    "category": "casual",
                    "messages": [{"role": "user", "content": "Mutation?"}],
                    "reference": "No.",
                }
            )
            + "\n"
        )
    with pytest.raises(SealedTestError, match="changed after selection"):
        pipeline.run_stage("sealed")


def test_candidate_cache_must_remain_inside_run_directory(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["storage"]["candidate_cache_dir"] = str(tmp_path / "outside_cache")
    with pytest.raises(ConfigurationError, match="candidate_cache_dir"):
        load_config(_write_config(tmp_path / "config.yaml", payload))


def test_run_context_rejects_artifact_path_escape(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            _config_payload(tmp_path, anchor, greedy, dev, test),
        )
    )
    context = RunContext.create(config)
    with pytest.raises(ValueError, match="escapes the run directory"):
        context.path("..", "outside")


def test_merge_refuses_output_that_overlaps_source_checkpoint(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    recipe = MergeRecipe(global_alpha=0.5, label="unsafe")
    with pytest.raises(ValueError, match="overlaps source checkpoint"):
        merge_checkpoints(
            anchor,
            greedy,
            anchor,
            recipe,
            layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
            overwrite=True,
        )
    with pytest.raises(ValueError, match="overlaps source checkpoint"):
        merge_checkpoints(
            anchor,
            greedy,
            tmp_path,
            recipe,
            layer_key_regex=r"(?:^|\.)layers\.(\d+)\.",
            overwrite=True,
        )


def test_removed_existing_backend_is_rejected(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["evaluation"]["backend"] = "existing"
    with pytest.raises(ConfigurationError, match="evaluation.backend"):
        load_config(_write_config(tmp_path / "config.yaml", payload))


def test_manual_cli_merge_materializes_full_checkpoint(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    from rineya_soup_lab.cli import main

    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=10.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(tmp_path, anchor, greedy, dev, test),
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(MergeRecipe(global_alpha=0.25, label="manual").to_dict()),
        encoding="utf-8",
    )
    output = tmp_path / "manual_output"
    assert (
        main(
            [
                "merge",
                "--config",
                str(config_path),
                "--recipe",
                str(recipe_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "model.safetensors").exists() or (
        output / "model.safetensors.index.json"
    ).exists()
    manifest = json.loads((output / "merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recipe"]["global_alpha"] == 0.25
    assert manifest["source_fingerprints"]["anchor"]["kind"] == "full_checkpoint"


def test_run_directory_cannot_overlap_model_or_dataset_inputs(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["project"] = {"run_root": str(anchor), "run_name": "nested_run"}
    config = load_config(_write_config(tmp_path / "overlap.yaml", payload))
    with pytest.raises(ValueError, match="overlaps protected input"):
        RunContext.create(config)


def test_candidate_load_rejects_modified_evaluation_evidence(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    config = load_config(
        _write_config(
            tmp_path / "config.yaml",
            _config_payload(tmp_path, anchor, greedy, dev, test),
        )
    )
    context = RunContext.create(config)
    sources = run_preflight(context)
    manager = CandidateManager(context, sources)
    result = manager.evaluate(MergeRecipe.endpoint_recipe("anchor"), alias="anchor")
    summary_path = result.evidence.output_dir / "summary.json"
    summary_path.write_text(summary_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        manager.load_result("anchor")


def test_resolved_tokenizer_is_recorded_for_endpoint_evaluation(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        (anchor / "tokenizer_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["models"]["tokenizer_path"] = str(tokenizer)
    config = load_config(_write_config(tmp_path / "config.yaml", payload))
    context = RunContext.create(config)
    sources = run_preflight(context)
    manager = CandidateManager(context, sources)
    result = manager.evaluate(MergeRecipe.endpoint_recipe("anchor"), alias="anchor")
    manifest = json.loads(
        (result.evidence.output_dir / "lab_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tokenizer_path"] == str(tokenizer.resolve())


def test_manual_cli_refuses_to_overwrite_configured_tokenizer(
    tmp_path: Path, tiny_checkpoint_factory
) -> None:
    from rineya_soup_lab.cli import main

    anchor = tiny_checkpoint_factory(tmp_path / "anchor", offset=0.0, layers=2)
    greedy = tiny_checkpoint_factory(tmp_path / "greedy", offset=1.0, layers=2)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    token_file = tokenizer / "tokenizer_config.json"
    token_file.write_text(
        (anchor / "tokenizer_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    payload = _config_payload(tmp_path, anchor, greedy, dev, test)
    payload["models"]["tokenizer_path"] = str(tokenizer)
    config_path = _write_config(tmp_path / "config.yaml", payload)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(MergeRecipe(global_alpha=0.5, label="unsafe").to_dict()),
        encoding="utf-8",
    )
    assert main(
        [
            "merge",
            "--config",
            str(config_path),
            "--recipe",
            str(recipe_path),
            "--output",
            str(tokenizer),
            "--overwrite",
        ]
    ) == 1
    assert token_file.exists()
