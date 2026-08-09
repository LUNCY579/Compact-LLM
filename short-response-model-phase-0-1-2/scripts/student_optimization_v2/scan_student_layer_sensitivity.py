from __future__ import annotations

"""Empirically scan single-layer physical deletions for an accepted student.

This is the student analogue of the teacher Phase 2B every-layer deletion study.
Each layer is physically removed, fresh-reloaded, evaluated on a non-sealed DEV
subset under grounded RAG conditions, and compared against the intact student
using the shared relative-opportunity engine.  Temporary checkpoints may be
removed after their evidence is recorded.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction


def run(command: list[str]) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan physical single-layer deletions for the accepted Phase 9 student.")
    p.add_argument("--model", required=True)
    p.add_argument("--profile-data", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--knowledge-db", type=Path, required=True)
    p.add_argument("--gate-calibration", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--max-profile-records", type=int, default=256)
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--conditions", default="oracle_rag,real_rag,adversarial_rag")
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    return p.parse_args()


def decoder_layer_count(model: str) -> int:
    # Keep heavy Transformers imports lazy so --help/static repository checks do not
    # require the CUDA/model stack to be installed.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not resolve decoder layer count from model config")


def evaluate(model: str, benchmark: Path, knowledge_db: Path, calibration: Path, output: Path, args: argparse.Namespace) -> None:
    summary = output / "summary.json"
    if args.reuse_existing and summary.exists():
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/phase_9_student_distillation/evaluate_rag.py"),
        "--model", model,
        "--benchmark", str(benchmark),
        "--output-dir", str(output),
        "--knowledge-db", str(knowledge_db),
        "--gate-calibration", str(calibration),
        "--conditions", args.conditions,
        "--response-control", "one_sentence",
        "--max-new-tokens", str(args.max_new_tokens),
        "--limit", str(args.limit),
    ]
    if args.reuse_existing:
        command.append("--resume")
    run(command)


def main() -> None:
    args = parse_args()
    for path in (args.profile_data, args.benchmark, args.knowledge_db, args.gate_calibration):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test" in args.benchmark.name.casefold() or "sealed" in str(args.benchmark).casefold():
        raise ValueError("Layer-sensitivity selection must use DEV/non-sealed data, not TEST/sealed data")

    args.output_root.mkdir(parents=True, exist_ok=True)
    profile = args.output_root / "student_structure_profile.pt"
    profile_summary = args.output_root / "student_structure_profile.json"
    if not (args.reuse_existing and profile.exists() and profile_summary.exists()):
        run([
            sys.executable,
            str(ROOT / "scripts/phase_10_student_architecture/profile_student_structure.py"),
            "--model", args.model,
            "--data", str(args.profile_data),
            "--output", str(profile),
            "--summary", str(profile_summary),
            "--max-records", str(args.max_profile_records),
            "--max-sequence-length", str(args.max_sequence_length),
        ])

    reference_eval = args.output_root / "reference_eval"
    evaluate(args.model, args.benchmark, args.knowledge_db, args.gate_calibration, reference_eval, args)
    reference_summary = read_json(reference_eval / "summary.json")
    layer_count = decoder_layer_count(args.model)

    rows: list[dict[str, Any]] = []
    evidence_root = args.output_root / "layers"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for layer_id in range(layer_count):
        layer_root = evidence_root / f"layer_{layer_id:02d}"
        candidate = layer_root / "candidate"
        evaluation = layer_root / "evaluation"
        manifest = candidate / "phase_10_candidate_manifest.json"
        if not (args.reuse_existing and manifest.exists()):
            if candidate.exists():
                shutil.rmtree(candidate)
            run([
                sys.executable,
                str(ROOT / "scripts/phase_10_student_architecture/create_structural_candidate.py"),
                "--model", args.model,
                "--profile", str(profile),
                "--output", str(candidate),
                "--name", f"single_drop_layer_{layer_id}",
                "--description", f"Student sensitivity control deleting original decoder layer {layer_id}.",
                "--depth-drop", "1",
                "--depth-mode", "explicit-drop",
                "--drop-layer-ids", str(layer_id),
                "--overwrite",
            ])
        evaluate(str(candidate), args.benchmark, args.knowledge_db, args.gate_calibration, evaluation, args)
        summary = read_json(evaluation / "summary.json")
        gain = resource_reduction(reference_summary.get("model_weight_bytes"), summary.get("model_weight_bytes"))
        assessment = evaluate_opportunity(
            reference_summary,
            summary,
            resource_gain=gain,
            resource_name="deployed_model_weight_bytes",
        )
        row = {
            "layer_id": layer_id,
            "candidate_manifest": str(manifest),
            "summary": str(evaluation / "summary.json"),
            "model_weight_bytes": summary.get("model_weight_bytes"),
            "assessment": assessment,
            "eligible": bool(assessment.get("eligible")),
        }
        rows.append(row)
        write_json(args.output_root / "layer_sensitivity.partial.json", {"layers": rows})
        if not args.keep_candidates and candidate.exists():
            # The evaluation summary + candidate manifest are enough to preserve the scientific evidence.
            # Keep the manifest outside the deleted checkpoint first.
            saved_manifest = layer_root / "candidate_manifest.json"
            shutil.copy2(manifest, saved_manifest)
            row["candidate_manifest"] = str(saved_manifest)
            shutil.rmtree(candidate)

    ranked = sorted(
        rows,
        key=lambda row: (
            -float((row["assessment"] or {}).get("opportunity_margin", float("-inf"))),
            float((row["assessment"] or {}).get("quality_opportunity_loss_fraction", float("inf"))),
            int(row["layer_id"]),
        ),
    )
    order = [int(row["layer_id"]) for row in ranked]
    payload = {
        "schema_version": "2.0",
        "stage": "student_layer_sensitivity",
        "model": args.model,
        "reference_summary": str(reference_eval / "summary.json"),
        "profile": str(profile),
        "benchmark": str(args.benchmark),
        "benchmark_limit": args.limit,
        "conditions": args.conditions,
        "layer_count": layer_count,
        "recommended_drop_order": order,
        "layers": rows,
        "interpretation": (
            "The order ranks one-layer physical deletion evidence only. Multi-layer interactions are not assumed additive; "
            "every importance-guided multi-layer candidate is rebuilt, reloaded, evaluated, and recovery-trained in Phase 10."
        ),
    }
    write_json(args.output_root / "layer_sensitivity.json", payload)

    lines = [
        "# Student layer-sensitivity scan",
        "",
        f"- Model: `{args.model}`",
        f"- Layers scanned: `{layer_count}`",
        f"- DEV limit: `{args.limit}`",
        f"- Recommended single-layer deletion order: `{order}`",
        "",
        "| Layer | Resource gain | Quality loss | Opportunity margin | Eligible |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        a = row["assessment"]
        lines.append(
            f"| {row['layer_id']} | {float(a.get('resource_gain_fraction') or 0):.3%} | "
            f"{float(a.get('quality_opportunity_loss_fraction') or 0):.3%} | "
            f"{float(a.get('opportunity_margin') or 0):+.3%} | {row['eligible']} |"
        )
    lines.extend(["", payload["interpretation"]])
    (args.output_root / "LAYER_SENSITIVITY_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(args.output_root / "layer_sensitivity.json")


if __name__ == "__main__":
    main()
