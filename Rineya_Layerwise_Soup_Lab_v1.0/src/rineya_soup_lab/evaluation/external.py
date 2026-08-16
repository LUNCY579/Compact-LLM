from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import EvaluationError
from ..utils import read_json, read_jsonl, run_command


def run_official_phase_v45(
    *,
    official_repo_root: Path,
    python_executable: Path | None,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    protocol: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    script = official_repo_root / "scripts" / "phase_1_dense_sft" / "evaluate_model.py"
    quality_config = official_repo_root / "configs" / "quality_evaluation.json"
    if not script.exists():
        raise EvaluationError(f"Official evaluator not found: {script}")
    if not quality_config.exists():
        raise EvaluationError(f"Official v4.5 config not found: {quality_config}")
    if protocol.get("enable_thinking"):
        raise EvaluationError(
            "The supplied official phase evaluator hard-codes enable_thinking=False. "
            "Use bundled_v45 for the separately labeled thinking protocol."
        )
    command = [
        str(python_executable or Path(sys.executable)),
        str(script),
        "--model-id",
        str(model_path),
        "--benchmark",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--quality-config",
        str(quality_config),
        "--max-new-tokens",
        str(int(protocol.get("max_new_tokens", 32))),
        "--response-control",
        str(protocol.get("response_control", "eos_only")),
        "--min-new-tokens-before-boundary",
        str(int(protocol.get("min_new_tokens_before_boundary", 1))),
        "--system-prompt",
        str(protocol.get("system_prompt", "")),
    ]
    if protocol.get("do_sample"):
        command.extend(
            [
                "--do-sample",
                "--temperature",
                str(float(protocol.get("temperature", 0.6))),
                "--top-p",
                str(float(protocol.get("top_p", 0.9))),
            ]
        )
    if protocol.get("require_cuda"):
        command.append("--require-cuda")
    if resume:
        command.append("--resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(command, cwd=official_repo_root, log_path=output_dir / "official_evaluator.log")
    return load_external_evidence(output_dir / "summary.json", output_dir / "results.jsonl")


def render_external_command(template: str | Sequence[str], values: Mapping[str, Any]) -> list[str]:
    if isinstance(template, str):
        parts = shlex.split(template, posix=os.name != "nt")
    else:
        parts = [str(item) for item in template]
    try:
        return [part.format(**values) for part in parts]
    except KeyError as exc:
        raise EvaluationError(f"Unknown external evaluator placeholder: {exc}") from exc


def run_external_evaluator(
    *,
    template: str | Sequence[str],
    values: Mapping[str, Any],
    cwd: Path,
    output_dir: Path,
    summary_path_template: str,
    results_path_template: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = render_external_command(template, values)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(command, cwd=cwd, log_path=output_dir / "external_evaluator.log")
    summary_path = Path(summary_path_template.format(**values))
    results_path = Path(results_path_template.format(**values))
    return load_external_evidence(summary_path, results_path)


def load_external_evidence(
    summary_path: Path, results_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not summary_path.exists():
        raise EvaluationError(f"Evaluator did not produce summary: {summary_path}")
    if not results_path.exists():
        raise EvaluationError(f"Evaluator did not produce results: {results_path}")
    summary = read_json(summary_path)
    rows = read_jsonl(results_path)
    if str(summary.get("quality_evaluation_version")) != "4.5":
        raise EvaluationError(
            f"Expected Quality Evaluator v4.5, got {summary.get('quality_evaluation_version')!r}"
        )
    if int(summary.get("case_count", -1)) != len(rows):
        raise EvaluationError(
            f"Summary case_count {summary.get('case_count')} does not match {len(rows)} rows."
        )
    return summary, rows
