from __future__ import annotations

"""Student-specific contextual-sparsity simulation and measured runtime stage.

This wrapper deliberately bypasses teacher Phase-5/6 gate numbering while reusing
the same proven Phase-6/7 implementation primitives.  It never promotes a model
checkpoint.  It returns a DEV-only runtime recommendation (dense runtime or
sparse feature flag) that remains subordinate to the final Phase-11 human gate.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_gate import assert_valid_reference


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run student contextual-sparsity simulation and runtime measurement.")
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--oracle-label-ratio", type=float, default=0.65)
    p.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.8, 0.65, 0.5, 0.35])
    p.add_argument("--max-calibration-records", type=int, default=512)
    p.add_argument("--max-benchmark-cases", type=int, default=150)
    p.add_argument("--router-epochs", type=int, default=20)
    p.add_argument("--router-width", type=int, default=128)
    p.add_argument("--runtime-repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="one_sentence")
    p.add_argument("--reuse-existing", action="store_true")
    return p.parse_args()


def choose_ratio(frontier: dict[str, Any]) -> tuple[float | None, list[dict[str, Any]]]:
    dense = frontier.get("dense_quality")
    comparisons = frontier.get("masked_generation_comparison") or []
    if dense is None:
        raise ValueError("Sparse frontier contains no dense quality reference")
    assert_valid_reference(dense, label="student sparse dense reference")
    assessed: list[dict[str, Any]] = []
    for row in comparisons:
        ratio = float(row["active_ratio"])
        if ratio >= 1.0:
            continue
        assessment = evaluate_opportunity(
            dense,
            row,
            resource_gain=max(0.0, 1.0 - ratio),
            resource_name="active_compute_reduction",
        )
        assessed.append({**row, "assessment": assessment, "eligible": bool(assessment.get("eligible"))})
    eligible = [row for row in assessed if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["assessment"].get("opportunity_margin", float("-inf"))),
            1.0 - float(row["active_ratio"]),
        ),
        default=None,
    )
    return (float(selected["active_ratio"]) if selected else None), assessed


def main() -> None:
    args = parse_args()
    for path in (args.calibration, args.benchmark):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test" in args.benchmark.name.casefold() or "sealed" in str(args.benchmark).casefold():
        raise ValueError("Sparse selection must use DEV/non-sealed benchmark data")

    args.output_root.mkdir(parents=True, exist_ok=True)
    router_dataset = args.output_root / "router_dataset.pt"
    frontier_file = args.output_root / "frontier.json"
    simulation_report = args.output_root / "SPARSITY_SIMULATION_REPORT.md"
    simulation_gate = args.output_root / "SPARSITY_SIMULATION_DIAGNOSTIC_GATE.md"
    comparison_file = args.output_root / "runtime_comparison.json"
    runtime_report = args.output_root / "SPARSE_RUNTIME_REPORT.md"
    runtime_gate = args.output_root / "SPARSE_RUNTIME_DIAGNOSTIC_GATE.md"

    if not (args.reuse_existing and router_dataset.exists()):
        run([
            sys.executable,
            str(ROOT / "scripts/phase_6_sparse_simulation/profile_activations.py"),
            "--model", args.model,
            "--calibration", str(args.calibration),
            "--out", str(router_dataset),
            "--block-size", str(args.block_size),
            "--fixed-active-ratio", str(args.oracle_label_ratio),
            "--max-records", str(args.max_calibration_records),
            "--positions", "last",
        ])

    if not (args.reuse_existing and frontier_file.exists()):
        run([
            sys.executable,
            str(ROOT / "scripts/phase_6_sparse_simulation/simulate_sparsity.py"),
            "--traces", str(router_dataset),
            "--model", args.model,
            "--benchmark", str(args.benchmark),
            "--max-cases", str(args.max_benchmark_cases),
            "--max-new-tokens", str(args.max_new_tokens),
            "--response-control", args.response_control,
            "--ratios", *[str(value) for value in args.ratios],
            "--out", str(frontier_file),
        ])

    # Produce the familiar Phase-6-style diagnostic report, but keep it local to
    # this cross-phase workflow so no teacher gate is overwritten.
    run([
        sys.executable,
        str(ROOT / "scripts/phase_6_sparse_simulation/check_phase_6_gate.py"),
        "--frontier", str(frontier_file),
        "--report", str(simulation_report),
        "--gate", str(simulation_gate),
    ])

    frontier = read_json(frontier_file)
    selected_ratio, assessed = choose_ratio(frontier)
    routers = args.output_root / "routers.pt"
    if selected_ratio is None:
        if not (args.reuse_existing and comparison_file.exists()):
            run([
                sys.executable,
                str(ROOT / "scripts/phase_7_sparse_runtime/benchmark_runtime.py"),
                "--model", args.model,
                "--benchmark", str(args.benchmark),
                "--dense-only",
                "--max-cases", str(args.max_benchmark_cases),
                "--max-new-tokens", str(args.max_new_tokens),
                "--response-control", args.response_control,
                "--repeats", str(args.runtime_repeats),
                "--out", str(comparison_file),
            ])
    else:
        if not (args.reuse_existing and routers.exists()):
            run([
                sys.executable,
                str(ROOT / "scripts/phase_7_sparse_runtime/train_router.py"),
                "--dataset", str(router_dataset),
                "--target-ratio", str(selected_ratio),
                "--epochs", str(args.router_epochs),
                "--width", str(args.router_width),
                "--out", str(routers),
            ])
        if not (args.reuse_existing and comparison_file.exists()):
            run([
                sys.executable,
                str(ROOT / "scripts/phase_7_sparse_runtime/benchmark_runtime.py"),
                "--model", args.model,
                "--routers", str(routers),
                "--benchmark", str(args.benchmark),
                "--keep-ratio", str(selected_ratio),
                "--max-cases", str(args.max_benchmark_cases),
                "--max-new-tokens", str(args.max_new_tokens),
                "--response-control", args.response_control,
                "--repeats", str(args.runtime_repeats),
                "--out", str(comparison_file),
            ])

    run([
        sys.executable,
        str(ROOT / "scripts/phase_7_sparse_runtime/check_phase_7_gate.py"),
        "--comparison", str(comparison_file),
        "--report", str(runtime_report),
        "--gate", str(runtime_gate),
    ])
    comparison = read_json(comparison_file)
    response_ok = bool(comparison.get("passes_response_control_target"))
    opportunity = comparison.get("opportunity_assessment") or comparison.get("quality_comparison") or {}
    sparse_attempted = bool(comparison.get("sparse_attempted", selected_ratio is not None))
    if response_ok and sparse_attempted and opportunity.get("eligible"):
        recommendation = "SPARSE_FEATURE_FLAG"
    else:
        recommendation = "DENSE_RUNTIME"

    selection = {
        "schema_version": "2.0",
        "stage": "student_sparse_runtime",
        "model": args.model,
        "calibration": str(args.calibration),
        "benchmark": str(args.benchmark),
        "sealed_test_used": False,
        "simulation_selected_ratio": selected_ratio,
        "simulation_assessments": assessed,
        "router_dataset": str(router_dataset),
        "routers": str(routers) if routers.exists() else None,
        "runtime_comparison": str(comparison_file),
        "runtime_recommendation": recommendation,
        "runtime_opportunity": opportunity,
        "decode_speedup": comparison.get("decode_speedup"),
        "peak_vram_reduction_fraction": comparison.get("peak_vram_reduction_fraction"),
        "integration_boundary": (
            "SPARSE_FEATURE_FLAG is research/runtime evidence only in v2.0. Phase 11 packaging remains dense/quantized unless a future "
            "production runtime explicitly integrates the router with a dense fallback."
        ),
    }
    write_json(args.output_root / "sparse_selection.json", selection)
    print(args.output_root / "sparse_selection.json")


if __name__ == "__main__":
    main()
