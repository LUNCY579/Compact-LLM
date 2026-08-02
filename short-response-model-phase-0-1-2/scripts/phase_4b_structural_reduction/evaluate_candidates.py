# Purpose: Evaluates every Phase 4B candidate with one benchmark and produces the candidate registry.
from __future__ import annotations

"""Evaluate Phase 4B candidates with the repository's existing evaluator.

This script is the integration point between structural experiments and the
mature phase pipeline. It does not invent a second scoring system: every model
is passed to ``phase_1_dense_sft/evaluate_model.py`` with identical benchmark
and decoding arguments, then measured by ``measure_checkpoint.py``.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION, require_quality_v4

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "phase_4b_structural_reduction"


def parse_candidate_spec(spec: str) -> tuple[str, str]:
    """Parse ``NAME=MODEL_PATH_OR_ID`` and reject unsafe output names."""

    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Candidate must use NAME=MODEL format, received: {spec!r}"
        )
    name, model = spec.split("=", 1)
    name = name.strip()
    model = model.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(
            "Candidate names may contain letters, numbers, dot, underscore, and dash."
        )
    if not model:
        raise argparse.ArgumentTypeError("Candidate model path/ID cannot be empty.")
    return name, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense and structurally reduced Phase 4B candidates."
    )
    parser.add_argument(
        "--dense-reference",
        required=True,
        help="Accepted Phase 4 checkpoint used as the untouched control.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        type=parse_candidate_spec,
        metavar="NAME=MODEL",
        help="Reduced candidate. Repeat this argument for multiple candidates.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "benchmark" / "shortchatbench_pilot.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse complete summary/footprint files instead of rerunning them.",
    )
    return parser.parse_args()


def run_checked(command: list[str]) -> None:
    """Print and execute one subprocess, failing immediately on errors."""

    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_one(
    *,
    name: str,
    model: str,
    benchmark: Path,
    output_root: Path,
    max_new_tokens: int,
    limit: int | None,
    reuse_existing: bool,
) -> dict[str, Any]:
    """Run behavioral evaluation and real checkpoint-size measurement."""

    evaluation_dir = output_root / "evaluations" / name
    footprint_path = output_root / "footprints" / f"{name}.json"
    summary_path = evaluation_dir / "summary.json"

    if not (reuse_existing and summary_path.exists()):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "phase_1_dense_sft" / "evaluate_model.py"),
            "--model-id",
            model,
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(evaluation_dir),
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        run_checked(command)

    if not (reuse_existing and footprint_path.exists()):
        run_checked(
            [
                sys.executable,
                str(Path(__file__).with_name("measure_checkpoint.py")),
                "--model",
                model,
                "--out",
                str(footprint_path),
            ]
        )

    # Read the generated files now so a partial or malformed run fails before
    # the candidate registry is presented as complete.
    summary = read_json(summary_path)
    require_quality_v4(summary, f"Phase 4B evaluation {name}")
    footprint = read_json(footprint_path)
    return {
        "name": name,
        "model": model,
        "evaluation_dir": str(evaluation_dir),
        "summary": str(summary_path),
        "results": str(evaluation_dir / "results.jsonl"),
        "footprint": str(footprint_path),
        "measured": {
            "mean_quality_score": summary.get("mean_quality_score"),
            "mean_heuristic_score": summary.get("mean_heuristic_score"),
            "quality_pass_rate": summary.get("quality_pass_rate"),
            "intelligibility_pass_rate": summary.get("intelligibility_pass_rate"),
            "worst_decile_mean_quality_score": summary.get("worst_decile_mean_quality_score"),
            "hard_failure_case_count": summary.get("hard_failure_case_count"),
            "critical_failure_count": summary.get("critical_failure_count"),
            "major_case_count": summary.get("major_case_count"),
            "parameters_total": footprint.get("parameters", {}).get("total"),
            "local_directory_bytes": footprint.get("local_directory_bytes"),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    # The untouched Phase 4 checkpoint is always candidate zero and remains
    # available even when a smaller checkpoint is later selected.
    specs = [("dense_reference", args.dense_reference), *args.candidate]
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise SystemExit("Candidate names must be unique.")

    candidates = [
        evaluate_one(
            name=name,
            model=model,
            benchmark=args.benchmark,
            output_root=args.output_root,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            reuse_existing=args.reuse_existing,
        )
        for name, model in specs
    ]

    registry_path = args.output_root / "candidate_registry.json"
    write_json(
        registry_path,
        {
            "registry_version": 3,
            "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
            "phase": "4B",
            "dense_reference_name": "dense_reference",
            "dense_reference_model": args.dense_reference,
            "benchmark": str(args.benchmark),
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "candidates": candidates,
            "rule": (
                "All candidates were evaluated through the same repository evaluator. "
                "Human review of raw replies remains required."
            ),
        },
    )
    print(registry_path)


if __name__ == "__main__":
    main()
