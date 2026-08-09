from __future__ import annotations

"""DEV-only context/KV budget sweep for an accepted Phase 10 student.

The script deterministically splits the Phase 9 DEV set by group so threshold
calibration and context-profile selection do not use the same examples.  Each
profile receives its own answerability calibration, then the production runtime
benchmark measures actual prompt token counts and quality on the selection half.
The shared relative-opportunity engine chooses the best worthwhile context
reduction; the full reference remains the fallback.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, read_jsonl, write_json, write_jsonl
from scripts.common.opportunity_evaluation import evaluate_opportunity


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep context budgets for the accepted student on non-sealed DEV data.")
    p.add_argument("--model", required=True)
    p.add_argument("--dev-benchmark", type=Path, required=True)
    p.add_argument("--knowledge-db", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/student_optimization_v2.json")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="one_sentence")
    p.add_argument("--memory-reconstruction-mode", choices=("off", "prompted", "trained"), default="off")
    p.add_argument("--reconstruction-max-new-tokens", type=int, default=128)
    p.add_argument("--reuse-existing", action="store_true")
    return p.parse_args()


def group_key(row: dict[str, Any]) -> str:
    for key in ("topic_family", "group_id", "document_group", "document_id", "source_group"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    record_id = str(row.get("id") or "")
    # IDs often encode a family prefix; retain that before the final numeric suffix.
    parts = record_id.rsplit("_", 1)
    return f"id:{parts[0] if len(parts) == 2 else record_id}"


def split_dev(rows: list[dict[str, Any]], calibration_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    calibration: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    threshold = int(max(0.05, min(0.95, calibration_fraction)) * 10_000)
    for key in sorted(groups):
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        (calibration if bucket < threshold else selection).extend(groups[key])
    # Degenerate tiny fixtures are handled deterministically without row overlap.
    if not calibration or not selection:
        ordered_groups = sorted(groups)
        split = max(1, min(len(ordered_groups) - 1, len(ordered_groups) // 2))
        calibration = [row for key in ordered_groups[:split] for row in groups[key]]
        selection = [row for key in ordered_groups[split:] for row in groups[key]]
    if not calibration or not selection:
        raise ValueError("DEV corpus is too small to create disjoint calibration and selection subsets")
    return calibration, selection


def prompt_stats(results_path: Path) -> dict[str, float]:
    rows = read_jsonl(results_path)
    values = [int(row.get("prompt_tokens") or 0) for row in rows if row.get("prompt_tokens") is not None]
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError(f"No prompt token measurements found in {results_path}")
    return {
        "mean_prompt_tokens": statistics.fmean(values),
        "median_prompt_tokens": statistics.median(values),
        "max_prompt_tokens": max(values),
    }


def main() -> None:
    args = parse_args()
    if not args.dev_benchmark.exists() or not args.knowledge_db.exists() or not args.config.exists():
        missing = [str(path) for path in (args.dev_benchmark, args.knowledge_db, args.config) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))
    if "test" in args.dev_benchmark.name.casefold() or "sealed" in str(args.dev_benchmark).casefold():
        raise ValueError("Context tuning must not use TEST/sealed data")

    config = read_json(args.config)
    section = config.get("context_sweep") or {}
    profiles = section.get("profiles") or []
    if not profiles:
        raise ValueError("No context profiles configured")
    names = [str(row["name"]) for row in profiles]
    if len(set(names)) != len(names):
        raise ValueError("Context profile names must be unique")
    reference_profile = next((row for row in profiles if row.get("name") == "reference"), profiles[0])

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.dev_benchmark)
    calibration_rows, selection_rows = split_dev(rows, float(section.get("calibration_fraction", 0.5)))
    calibration_file = args.output_root / "dev_context_calibration.jsonl"
    selection_file = args.output_root / "dev_context_selection.jsonl"
    write_jsonl(calibration_file, calibration_rows)
    write_jsonl(selection_file, selection_rows)
    split_manifest = {
        "source": str(args.dev_benchmark),
        "calibration_rows": len(calibration_rows),
        "selection_rows": len(selection_rows),
        "calibration_groups": len({group_key(row) for row in calibration_rows}),
        "selection_groups": len({group_key(row) for row in selection_rows}),
        "group_overlap": sorted({group_key(row) for row in calibration_rows} & {group_key(row) for row in selection_rows}),
    }
    if split_manifest["group_overlap"]:
        raise RuntimeError("Context DEV split contains group leakage")
    write_json(args.output_root / "dev_split_manifest.json", split_manifest)

    packets: list[dict[str, Any]] = []
    for profile in profiles:
        name = str(profile["name"])
        root = args.output_root / "profiles" / name
        root.mkdir(parents=True, exist_ok=True)
        calibration = root / "answerability_calibration.json"
        benchmark_root = root / "runtime_dev"
        summary_path = benchmark_root / "summary.json"
        if not (args.reuse_existing and calibration.exists()):
            run([
                sys.executable,
                str(ROOT / "scripts/phase_11_production_rag/calibrate_answerability.py"),
                "--benchmark", str(calibration_file),
                "--knowledge-db", str(args.knowledge_db),
                "--output", str(calibration),
                "--retrieval-top-k", str(int(profile["retrieval_top_k"])),
                "--selected-top-k", str(int(profile["selected_top_k"])),
                "--false-answer-cost", "3.0",
                "--false-abstention-cost", "1.0",
            ])
        if not (args.reuse_existing and summary_path.exists()):
            command = [
                sys.executable,
                str(ROOT / "scripts/phase_11_production_rag/benchmark_production_runtime.py"),
                "--model", args.model,
                "--runtime-backend", "transformers",
                "--benchmark", str(selection_file),
                "--knowledge-db", str(args.knowledge_db),
                "--answerability-calibration", str(calibration),
                "--output-dir", str(benchmark_root),
                "--retrieval-top-k", str(int(profile["retrieval_top_k"])),
                "--selected-top-k", str(int(profile["selected_top_k"])),
                "--maximum-evidence-tokens", str(int(profile["maximum_evidence_tokens"])),
                "--max-sequence-length", str(args.max_sequence_length),
                "--max-new-tokens", str(args.max_new_tokens),
                "--response-control", args.response_control,
                "--memory-reconstruction-mode", args.memory_reconstruction_mode,
                "--reconstruction-max-new-tokens", str(args.reconstruction_max_new_tokens),
            ]
            if args.reuse_existing:
                command.append("--resume")
            run(command)
        summary = read_json(summary_path)
        stats = prompt_stats(benchmark_root / "results.jsonl")
        packets.append({
            "name": name,
            "profile": profile,
            "calibration": str(calibration),
            "summary": str(summary_path),
            "prompt_stats": stats,
            "quality": summary,
        })

    reference = next(packet for packet in packets if packet["name"] == str(reference_profile["name"]))
    reference_tokens = float(reference["prompt_stats"]["mean_prompt_tokens"])
    ranked: list[dict[str, Any]] = []
    for packet in packets:
        candidate_tokens = float(packet["prompt_stats"]["mean_prompt_tokens"])
        gain = max(0.0, 1.0 - candidate_tokens / max(reference_tokens, 1e-9))
        assessment = evaluate_opportunity(
            reference["quality"],
            packet["quality"],
            resource_gain=gain,
            resource_name="mean_prompt_tokens",
        )
        packet["assessment"] = assessment
        packet["eligible"] = packet["name"] == reference["name"] or bool(assessment.get("eligible"))
        ranked.append(packet)

    worthwhile = [packet for packet in ranked if packet["name"] != reference["name"] and packet["assessment"].get("eligible")]
    selected = max(
        worthwhile,
        key=lambda packet: (
            float(packet["assessment"].get("opportunity_margin", float("-inf"))),
            -float(packet["prompt_stats"]["mean_prompt_tokens"]),
        ),
        default=reference,
    )
    selection = {
        "schema_version": "2.0",
        "stage": "student_context_budget_optimization",
        "reference_profile": reference["name"],
        "selected_profile": selected["name"],
        "selected": {
            **selected["profile"],
            "calibration": selected["calibration"],
            "summary": selected["summary"],
            **selected["prompt_stats"],
            "assessment": selected["assessment"],
        },
        "dev_split": str(args.output_root / "dev_split_manifest.json"),
        "selection_benchmark": str(selection_file),
        "calibration_benchmark": str(calibration_file),
        "profiles": [
            {
                "name": packet["name"],
                **packet["profile"],
                "calibration": packet["calibration"],
                "summary": packet["summary"],
                **packet["prompt_stats"],
                "assessment": packet["assessment"],
                "eligible": packet["eligible"],
            }
            for packet in ranked
        ],
        "sealed_test_used": False,
    }
    write_json(args.output_root / "context_selection.json", selection)

    lines = [
        "# Student context / KV-budget optimization",
        "",
        f"- Selected profile: `{selected['name']}`",
        f"- Calibration rows: `{len(calibration_rows)}`",
        f"- Selection rows: `{len(selection_rows)}`",
        "- Sealed TEST used: `NO`",
        "",
        "| Profile | Retrieval K | Selected K | Evidence cap | Mean prompt tokens | Token reduction | Quality loss | Opportunity margin | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for packet in sorted(ranked, key=lambda x: float(x["assessment"].get("opportunity_margin", -999)), reverse=True):
        a = packet["assessment"]
        token_reduction = max(0.0, 1.0 - float(packet["prompt_stats"]["mean_prompt_tokens"]) / max(reference_tokens, 1e-9))
        lines.append(
            f"| {packet['name']} | {packet['profile']['retrieval_top_k']} | {packet['profile']['selected_top_k']} | "
            f"{packet['profile']['maximum_evidence_tokens']} | {packet['prompt_stats']['mean_prompt_tokens']:.2f} | "
            f"{token_reduction:.3%} | {float(a.get('quality_opportunity_loss_fraction') or 0):.3%} | "
            f"{float(a.get('opportunity_margin') or 0):+.3%} | {packet['eligible']} |"
        )
    lines.extend([
        "",
        "The winning profile is selected only on the disjoint DEV selection subset. The final Phase 11 run recalibrates that frozen profile on the full DEV set before opening sealed TEST.",
    ])
    (args.output_root / "CONTEXT_BUDGET_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(args.output_root / "context_selection.json")


if __name__ == "__main__":
    main()
