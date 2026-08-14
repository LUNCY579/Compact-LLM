from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.decoding.decoder import generate_reference, generate_separate_cache
from consensus_lab.evaluation.sampling import deterministic_stratified_sample
from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths


def _decision_signature(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fields that must remain identical for the cache optimization to be valid."""
    return [
        {
            "winner": step["winner"],
            "top_tokens": step["top_tokens"],
            "vote_counts": step["vote_counts"],
            "tie": step["tie"],
            "tied_tokens": step["tied_tokens"],
            "winning_expert_coalition": step["winning_expert_coalition"],
            "dissenting_experts": step["dissenting_experts"],
            "eos_policy": step["eos_policy"],
        }
        for step in steps
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate the optimized per-expert KV cache against exact full-prefix recomputation."
    )
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/hard_vote.yaml")
    parser.add_argument("--dataset", default="v22_dev")
    parser.add_argument("--committee", action="append", dest="committees")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()

    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(args.config, paths)
    selected_names = args.committees or ["backbone_k1", "specialist_k3"]
    by_name = {committee.name: committee for committee in config.committees}
    missing = sorted(set(selected_names) - set(by_name))
    if missing:
        raise KeyError(f"Unknown committees: {missing}")
    selected = [by_name[name] for name in selected_names]
    selected_methods = args.methods or config.methods
    unsupported = sorted(set(selected_methods) & {"sequence_medoid", "adaptive_hard_plurality"})
    if unsupported:
        raise ValueError(f"Cache parity supports token aggregation methods only: {unsupported}")
    needed_names = {name for committee in selected for name in committee.experts}
    needed_experts = [expert for expert in config.experts if expert.name in needed_names]
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=needed_experts,
        device=args.device,
        precision=args.precision,
    )
    rows = deterministic_stratified_sample(
        read_jsonl(config.datasets[args.dataset]), limit=args.limit, seed=config.seed
    )
    max_new_tokens = args.max_new_tokens or config.max_new_tokens
    comparisons: list[dict[str, Any]] = []
    for committee_spec in selected:
        view = committee.subset(committee_spec.experts)
        for method in selected_methods:
            for row in rows:
                messages = [dict(message) for message in (row.get("messages") or [])]
                input_ids = view.encode_messages(messages[:-1], max_sequence_length=config.max_sequence_length)
                reference = generate_reference(
                    view,
                    input_ids,
                    method=method,
                    max_new_tokens=max_new_tokens,
                    eos_policy="ordinary",
                )
                cached = generate_separate_cache(
                    view,
                    input_ids,
                    method=method,
                    max_new_tokens=max_new_tokens,
                    eos_policy="ordinary",
                )
                token_match = reference.output_token_ids == cached.output_token_ids
                trace_match = _decision_signature(reference.steps) == _decision_signature(cached.steps)
                comparisons.append(
                    {
                        "dataset": args.dataset,
                        "example_id": str(row.get("id")),
                        "committee": committee_spec.name,
                        "experts": committee_spec.experts,
                        "method": method,
                        "token_match": token_match,
                        "decision_trace_match": trace_match,
                        "reference_tokens": reference.output_token_ids,
                        "cached_tokens": cached.output_token_ids,
                        "reference_output": reference.output_text,
                        "cached_output": cached.output_text,
                        "reference_latency_seconds": reference.latency_seconds,
                        "cached_latency_seconds": cached.latency_seconds,
                    }
                )
    passed = all(row["token_match"] and row["decision_trace_match"] for row in comparisons)
    packet = {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "claim": "separate_cache is decision-equivalent to full-prefix reference for every tested aggregation rule",
        "config": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "precision": args.precision,
        "device": str(committee.device),
        "dataset": args.dataset,
        "committee_names": selected_names,
        "methods": selected_methods,
        "example_count_per_committee": len(rows),
        "comparison_count": len(comparisons),
        "token_match_rate": sum(row["token_match"] for row in comparisons) / len(comparisons),
        "decision_trace_match_rate": sum(row["decision_trace_match"] for row in comparisons) / len(comparisons),
        "comparisons": comparisons,
    }
    output = paths.assert_output_path(
        LAB_ROOT / f"results/summaries/cache_parity_{args.config.stem}.json"
    )
    write_json_atomic(output, packet)
    print(json.dumps({**{key: value for key, value in packet.items() if key != "comparisons"}, "output": str(output)}, indent=2))
    if not passed:
        raise SystemExit("Cache parity failed; use cache_mode: reference for experiments")


if __name__ == "__main__":
    main()
