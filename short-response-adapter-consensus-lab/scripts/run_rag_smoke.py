from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.decoding.decoder import generate_reference
from consensus_lab.evaluation.sampling import deterministic_stratified_sample
from consensus_lab.evaluation.scoring import repetition_failure
from consensus_lab.evaluation.summary import write_summary_files
from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import append_jsonl_durable, read_jsonl
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths
from consensus_lab.run_lock import exclusive_run_lock


def _prompt_hash(messages: list[dict[str, Any]]) -> str:
    value = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded oracle-RAG adapter-consensus pilot.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/rag_oracle.yaml")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(args.config, paths)

    # Reuse the official evaluator's definitions directly from the read-only source.
    sys.path.insert(0, str(paths.official_root))
    from scripts.phase_9_student_distillation.evaluate_rag import reference_answer
    from scripts.phase_9_student_distillation.phase9_common import grounding_check, is_abstention
    from scripts.phase_9_student_distillation.phase9_metrics import (
        decision_class,
        functional_answer_correct,
        is_safe_conflict_response,
    )

    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=config.experts,
        device=args.device,
        precision=args.precision,
    )
    source_rows = read_jsonl(config.datasets["rag_dev"])
    eligible = [row for row in source_rows if decision_class(row) in {"supported", "unsupported", "conflict"}]
    rows = deterministic_stratified_sample(
        eligible,
        limit=args.limit,
        seed=config.seed,
        strata=("decision", "difficulty", "category", "mode"),
    )
    output = paths.assert_output_path(LAB_ROOT / "results/runs/v22_oracle_rag_consensus_smoke.jsonl")
    with exclusive_run_lock(output.with_suffix(output.suffix + ".lock")):
        existing = read_jsonl(output) if output.is_file() else []
        for row in existing:
            if row.get("config_sha256") != config.config_sha256:
                raise RuntimeError("Existing RAG rows use a different config")
        done = {(row["committee"], row["example_id"]) for row in existing}
        all_rows = list(existing)
        for committee_spec in config.committees:
            view = committee.subset(committee_spec.experts)
            metadata = view.metadata()
            for row in rows:
                example_id = str(row["id"])
                if (committee_spec.name, example_id) in done:
                    continue
                messages = [dict(value) for value in row.get("messages") or []]
                input_ids = view.encode_messages(messages, max_sequence_length=config.max_sequence_length)
                result = generate_reference(
                    view,
                    input_ids,
                    method="hard_plurality",
                    max_new_tokens=config.max_new_tokens,
                    eos_policy="ordinary",
                )
                answer = result.output_text
                evidence = [dict(value) for value in row.get("evidence") or []]
                grounding = grounding_check(answer, evidence)
                abstained = is_abstention(answer)
                klass = decision_class(row)
                supported_correct, functional = functional_answer_correct(
                    row,
                    answer,
                    reference=reference_answer(row),
                    abstained=abstained,
                    grounding_supported=grounding.supported,
                )
                if klass == "supported":
                    correct = supported_correct
                elif klass == "unsupported":
                    correct = abstained
                else:
                    correct = is_safe_conflict_response(answer)
                packet = {
                    "schema_version": "1.0",
                    "evaluation_protocol_version": "1.0-official-oracle-rag-prompt-and-functional-metrics",
                    "config_sha256": config.config_sha256,
                    "run_name": config.name,
                    "dataset": "rag_dev_oracle",
                    "benchmark_path": str(config.datasets["rag_dev"]),
                    "example_id": example_id,
                    "prompt_hash": _prompt_hash(messages),
                    "committee": committee_spec.name,
                    "committee_role": committee_spec.role,
                    "experts": committee_spec.experts,
                    "method": "hard_plurality",
                    "domain": row.get("category"),
                    "reasoning_family": row.get("mode"),
                    "difficulty": row.get("difficulty"),
                    "decision": "answer" if klass == "supported" else klass,
                    "case_class": klass,
                    "gold": reference_answer(row),
                    "correct": bool(correct),
                    "supported_answer_correct": bool(supported_correct),
                    "functional_correctness": functional,
                    "abstained": abstained,
                    "grounding_supported": grounding.supported,
                    "grounding_hard_failures": list(grounding.hard_failures),
                    "grounding_lexical_support_fraction": grounding.lexical_support_fraction,
                    "repetition_failure": repetition_failure(result.output_token_ids),
                    "mean_agreement": sum(float(step["agreement"]) for step in result.steps) / len(result.steps),
                    "mean_vote_entropy": sum(float(step["vote_entropy"]) for step in result.steps) / len(result.steps),
                    "mean_pairwise_js": (
                        sum(float(step["mean_pairwise_js"]) for step in result.steps if step["mean_pairwise_js"] is not None)
                        / max(1, sum(step["mean_pairwise_js"] is not None for step in result.steps))
                    ),
                    "tie_count": sum(bool(step["tie"]) for step in result.steps),
                    "adapter_storage_bytes": metadata["adapter_storage_bytes"],
                    "backbone_parameter_count": metadata["backbone_parameter_count"],
                    "active_parameter_count": metadata["active_parameter_count_per_expert_forward"],
                    "committee_adapter_parameter_count": metadata["committee_adapter_parameter_count"],
                    **result.to_dict(),
                }
                append_jsonl_durable(output, packet)
                all_rows.append(packet)
                done.add((committee_spec.name, example_id))
        summaries = write_summary_files(
            all_rows,
            LAB_ROOT / "results/summaries/v22_oracle_rag_consensus_smoke.json",
            LAB_ROOT / "results/summaries/v22_oracle_rag_consensus_smoke.csv",
        )
    print(json.dumps({"status": "RAG_SMOKE_COMPLETE", "rows": len(all_rows), "summaries": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
