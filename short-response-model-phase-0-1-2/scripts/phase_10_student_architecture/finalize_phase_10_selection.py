from __future__ import annotations

"""Materialize and register the human-approved Phase 10 student."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.check_human_gate import has_human_pass
from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the Phase 10 student selection.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "artifacts" / "phase_10_student_architecture" / "candidate_registry.json",
    )
    parser.add_argument("--gate", type=Path, default=ROOT / "reports" / "PHASE_10_STOP_GATE.md")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--role", choices=("deployment_candidate", "intermediate_teacher", "research_control"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--materialized-output",
        type=Path,
        default=ROOT / "artifacts" / "phase_10_student_architecture" / "accepted_student",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "references" / "phase_10_student_reference.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not has_human_pass(args.gate):
        raise SystemExit(f"Phase 10 gate lacks explicit human PASS: {args.gate}")
    registry = read_json(args.registry)
    packet = (registry.get("candidates") or {}).get(args.candidate)
    if not packet:
        raise ValueError(f"Unknown Phase 10 candidate: {args.candidate}")
    source_model = str(packet.get("model") or "").strip()
    source_adapter = str(packet.get("adapter") or "").strip() or None
    if not source_model:
        raise ValueError("Selected candidate has no model path")
    if args.materialized_output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Accepted Phase 10 student exists; use --overwrite intentionally: {args.materialized_output}"
            )
        shutil.rmtree(args.materialized_output)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    if source_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, source_adapter)
        model = model.merge_and_unload(safe_merge=True)
    args.materialized_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.materialized_output, safe_serialization=True)
    tokenizer.save_pretrained(args.materialized_output)
    AutoModelForCausalLM.from_pretrained(
        args.materialized_output,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    write_json(
        args.manifest,
        {
            "schema_version": "1.0",
            "phase": "10_student_architecture",
            "candidate": args.candidate,
            "role": args.role,
            "reason": args.reason,
            "source_registry": str(args.registry),
            "source_registry_sha256": sha256(args.registry),
            "source_gate": str(args.gate),
            "source_model": source_model,
            "source_adapter": source_adapter,
            "selected_model": str(args.materialized_output),
            "selected_model_tree_sha256": sha256_tree(args.materialized_output),
            "assessment": packet.get("assessment"),
            "rollback": {"source_model": source_model, "source_adapter": source_adapter},
        },
    )
    print(args.manifest)


if __name__ == "__main__":
    main()
