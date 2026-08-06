from __future__ import annotations

"""Materialize and register the human-approved Phase 9B checkpoint/runtime policy."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.check_human_gate import has_human_pass
from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the Phase 9B memory-reconstruction selection.")
    parser.add_argument("--assessment", type=Path, default=ROOT / "reports" / "PHASE_9B_ASSESSMENT.json")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports" / "PHASE_9B_STOP_GATE.md")
    parser.add_argument("--candidate", required=True, choices=("prompted_reconstruction", "trained_reconstruction", "direct_candidate"))
    parser.add_argument(
        "--role",
        required=True,
        choices=("phase10_candidate", "runtime_only", "research_control"),
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--phase9-reference", type=Path, default=ROOT / "artifacts" / "references" / "phase_9_student_reference.json")
    parser.add_argument("--materialized-output", type=Path, default=ROOT / "artifacts" / "phase_9b_memory_reconstruction" / "accepted_student")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "references" / "phase_9b_student_reference.json")
    parser.add_argument("--allow-two-pass-only-handoff", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _materialize(model_path: Path, adapter_path: Path | None, output: Path, *, overwrite: bool) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Phase 9B materialized output exists; use --overwrite: {output}")
        shutil.rmtree(output)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map=None,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map=None,
    )
    return output


def main() -> None:
    args = parse_args()
    assessment_path = _resolve(args.assessment)
    gate_path = _resolve(args.gate)
    phase9_reference_path = _resolve(args.phase9_reference)
    output = _resolve(args.materialized_output)
    manifest_path = _resolve(args.manifest)
    if not has_human_pass(gate_path):
        raise SystemExit(f"Phase 9B gate lacks explicit human PASS: {gate_path}")
    assessment = read_json(assessment_path)
    strategies = assessment.get("strategies") or {}
    if args.candidate not in strategies:
        raise ValueError(f"Candidate {args.candidate!r} is not in the Phase 9B assessment")
    candidate = strategies[args.candidate]
    if args.candidate == "prompted_reconstruction" and args.role == "phase10_candidate":
        raise ValueError(
            "Prompted reconstruction is a runtime policy, not a model-native Phase 10 checkpoint. "
            "Select role=runtime_only or approve a trained/direct candidate."
        )
    if args.role == "phase10_candidate":
        direct = assessment.get("phase_10_direct_compatibility") or {}
        if not direct.get("eligible") and not args.allow_two_pass_only_handoff:
            raise ValueError(
                "Phase 10 consumes a one-pass checkpoint. direct_candidate was not proportionally eligible. "
                "Use --allow-two-pass-only-handoff only after explicit human acceptance of that limitation."
            )
    artifact = candidate.get("artifact") or {}
    model_value = str(artifact.get("model") or "").strip()
    adapter_value = str(artifact.get("adapter") or "").strip() or None
    if not model_value:
        # Prompted reconstruction may intentionally use the already materialized
        # Phase 9 checkpoint without a new adapter.
        phase9_reference = read_json(phase9_reference_path)
        model_value = str(phase9_reference.get("selected_model") or "").strip()
    model_path = _resolve(model_value)
    adapter_path = _resolve(adapter_value) if adapter_value else None
    materialized = _materialize(model_path, adapter_path, output, overwrite=args.overwrite)
    phase9_reference = read_json(phase9_reference_path) if phase9_reference_path.exists() else {}
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": "9b_memory_reconstruction",
        "candidate": args.candidate,
        "role": args.role,
        "reason": args.reason,
        "inference_mode": (
            "two_pass_reconstruct_then_answer"
            if args.candidate != "direct_candidate"
            else "single_pass_reconstruction_distilled"
        ),
        "phase_10_direct_compatibility": assessment.get("phase_10_direct_compatibility"),
        "allow_two_pass_only_handoff": bool(args.allow_two_pass_only_handoff),
        "source_assessment": str(assessment_path),
        "source_assessment_sha256": sha256(assessment_path),
        "source_gate": str(gate_path),
        "source_model": model_value,
        "source_adapter": adapter_value,
        "selected_model": str(materialized),
        "selected_model_tree_sha256": sha256_tree(materialized),
        "phase_9_reference": str(phase9_reference_path),
        "phase_9_selected_model": phase9_reference.get("selected_model"),
        "rollback": {
            "phase_9_reference": str(phase9_reference_path),
            "phase_9_selected_model": phase9_reference.get("selected_model"),
            "source_model": model_value,
            "source_adapter": adapter_value,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, payload)
    print(manifest_path)


if __name__ == "__main__":
    main()
