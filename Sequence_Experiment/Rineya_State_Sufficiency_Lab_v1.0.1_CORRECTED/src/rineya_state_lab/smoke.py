from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .data.prepare import prepare_dataset
from .data.validate import validate_dataset
from .evaluation.compare import compare_runs
from .evaluation.runner import run_evaluation
from .utils import atomic_write_json, atomic_write_text, sha256_file, utc_now_iso


def run_smoke_test(
    *,
    work_dir: str | Path,
    archive: str | Path,
    limit: int = 24,
    force: bool = False,
) -> dict[str, Any]:
    work = Path(work_dir).expanduser().resolve()
    if force and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / "data"
    prepare = prepare_dataset(archive, prepared, overwrite=force)
    validation = validate_dataset(prepared, strict=True, output=work / "validation.json")
    eval_output = work / "eval_reference"
    config = {
        "run": {"name": "reference_smoke", "seed": 17, "output_dir": str(eval_output)},
        "data": {"root": str(prepared), "split": "dev", "require_test_unlock": False},
        "backend": {"type": "reference"},
        "scoring": {"absolute_tolerance": 1e-6, "relative_tolerance": 1e-6},
        "protocols": [
            {"type": "direct_solve", "name": "direct_solve", "limit": limit, "generation_batch_size": 8},
            {"type": "oracle_transition", "name": "oracle_transition", "limit": limit, "target_style": "stateful", "generation_batch_size": 8},
            {"type": "generated_rollout", "name": "generated_rollout", "limit": limit},
            {
                "type": "state_ablation",
                "name": "state_ablation",
                "limit_problems": 2,
                "variants": ["compact_state", "full_history"],
                "generation_batch_size": 4,
            },
        ],
    }
    config_path = work / "smoke_eval.yaml"
    atomic_write_text(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    evaluation = run_evaluation(config_path, force=force)
    comparison = compare_runs(
        eval_output / "scored_predictions.jsonl",
        eval_output / "scored_predictions.jsonl",
        work / "comparison",
        baseline_protocol="direct_solve",
        candidate_protocol="generated_rollout",
        metric="score.correct_strict",
        noninferiority_margin=0.02,
        bootstrap_samples=1000,
        randomization_samples=2000,
        seed=17,
    )
    metrics = evaluation["metrics"]["protocols"]
    required = {
        "direct_solve": metrics["direct_solve"]["overall"]["strict_accuracy"],
        "oracle_transition": metrics["oracle_transition"]["overall"]["strict_accuracy"],
        "generated_rollout": metrics["generated_rollout"]["overall"]["strict_accuracy"],
    }
    passed = all(value == 1.0 for value in required.values()) and validation["passed"]
    report = {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "passed": passed,
        "archive_sha256": sha256_file(archive),
        "prepare": prepare,
        "validation_summary": validation["issue_summary"],
        "reference_strict_accuracies": required,
        "comparison_verdict": comparison["decision"]["verdict"],
        "work_dir": str(work),
    }
    atomic_write_json(work / "SMOKE_TEST_REPORT.json", report)
    atomic_write_text(
        work / "SMOKE_TEST_REPORT.md",
        "# Smoke-Test Report\n\n"
        + f"- Passed: **{passed}**\n"
        + "\n".join(f"- {name}: {100*value:.2f}% strict" for name, value in required.items())
        + f"\n- Paired comparison verdict: {comparison['decision']['verdict']}\n",
    )
    if not passed:
        raise RuntimeError(f"Smoke test failed: {json.dumps(report, indent=2)}")
    return report
