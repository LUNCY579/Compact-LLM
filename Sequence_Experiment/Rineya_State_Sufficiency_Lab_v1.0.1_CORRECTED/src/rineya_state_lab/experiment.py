from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .data.prepare import prepare_dataset
from .data.validate import validate_dataset
from .evaluation.compare import compare_runs
from .evaluation.runner import run_evaluation
from .training.run import run_training
from .utils import atomic_write_json, sha256_file, utc_now_iso


ALL_STAGES = [
    "prepare",
    "validate",
    "train_full",
    "train_split",
    "evaluate_full",
    "evaluate_split",
    "compare",
]


def run_experiment(
    config_path: str | Path,
    *,
    stages: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    selected = stages or ALL_STAGES
    unknown = sorted(set(selected) - set(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown experiment stages: {unknown}")
    output_dir = (root / config.get("output_dir", "outputs/experiment/main")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "experiment_state.json"
    state = {
        "schema_version": "1.0",
        "experiment": config.get("name", "state_sufficiency"),
        "config": str(Path(config_path).expanduser().resolve()),
        "started_at": utc_now_iso(),
        "status": "RUNNING",
        "stages": {},
    }
    if state_path.exists() and not force:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["stages"].update(previous.get("stages", {}))
    atomic_write_json(state_path, state)

    def stage(name: str, function: Any) -> Any:
        if not force and state["stages"].get(name, {}).get("status") == "COMPLETED":
            return state["stages"][name].get("result")
        state["stages"][name] = {"status": "RUNNING", "started_at": utc_now_iso()}
        atomic_write_json(state_path, state)
        try:
            result = function()
            state["stages"][name] = {"status": "COMPLETED", "completed_at": utc_now_iso(), "result": result}
            atomic_write_json(state_path, state)
            return result
        except Exception as exc:
            state["stages"][name] = {
                "status": "FAILED",
                "failed_at": utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            state["status"] = "FAILED"
            atomic_write_json(state_path, state)
            raise

    paths = config["paths"]
    if "prepare" in selected:
        stage(
            "prepare",
            lambda: prepare_dataset(
                root / paths["dataset_archive"],
                root / paths["prepared_data"],
                overwrite=force,
            ),
        )
    if "validate" in selected:
        stage(
            "validate",
            lambda: validate_dataset(
                root / paths["prepared_data"],
                strict=True,
                output=output_dir / "dataset_validation.json",
            ),
        )
    if "train_full" in selected:
        stage("train_full", lambda: run_training(root / paths["train_full_config"]))
    if "train_split" in selected:
        stage("train_split", lambda: run_training(root / paths["train_split_config"]))
    if "evaluate_full" in selected:
        stage("evaluate_full", lambda: run_evaluation(root / paths["eval_full_config"], force=force))
    if "evaluate_split" in selected:
        stage("evaluate_split", lambda: run_evaluation(root / paths["eval_split_config"], force=force))
    if "compare" in selected:
        comparison = config["comparison"]
        stage(
            "compare",
            lambda: compare_runs(
                root / comparison["baseline_predictions"],
                root / comparison["candidate_predictions"],
                root / comparison["output_dir"],
                baseline_protocol=comparison.get("baseline_protocol"),
                candidate_protocol=comparison.get("candidate_protocol"),
                metric=comparison.get("metric", "score.correct_strict"),
                noninferiority_margin=float(comparison.get("noninferiority_margin", 0.02)),
                bootstrap_samples=int(comparison.get("bootstrap_samples", 10000)),
                randomization_samples=int(comparison.get("randomization_samples", 20000)),
                seed=int(config.get("seed", 42)),
            ),
        )
    state["status"] = "COMPLETED"
    state["completed_at"] = utc_now_iso()
    atomic_write_json(state_path, state)
    return state
