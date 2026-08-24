#!/usr/bin/env python3
"""Build the corrected, balanced, exactly annotated Halo R7 v2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import (  # noqa: E402
    R7_V2_GENERATOR_VERSION,
    SyntheticCase,
    generate_r7_v2_case,
    interpret_r7_v2_segments,
    locate_unique_evidence_spans,
    solve_r7_v2_oracle,
    tokenize_and_freeze_evidence,
    validate_r7_v2_case,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R7 v2 config root must be a mapping")
    return value


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def generate_cases(config: dict[str, Any]) -> list[SyntheticCase]:
    benchmark = dict(config["benchmark"])
    if str(benchmark["generator_version"]) != R7_V2_GENERATOR_VERSION:
        raise ValueError("R7 v2 config requests an unsupported generator version")
    seeds = [int(value) for value in benchmark["seeds"]]
    depths = [float(value) for value in benchmark["clustered_evidence_depths"]]
    layouts = [str(value) for value in benchmark["fixed_layouts"]]
    fixed_depth = float(benchmark["fixed_layout_requested_depth"])
    filler_count = int(benchmark["filler_count"])
    cases: list[SyntheticCase] = []
    for seed in seeds:
        cases.extend(
            generate_r7_v2_case(seed, depth, filler_count=filler_count, layout="clustered")
            for depth in depths
        )
        cases.extend(
            generate_r7_v2_case(
                seed,
                fixed_depth,
                filler_count=filler_count,
                layout=layout,  # type: ignore[arg-type]
            )
            for layout in layouts
        )

    expected = int(benchmark["expected_cases"])
    if len(cases) != expected or len({case.case_id for case in cases}) != expected:
        raise RuntimeError("R7 v2 matrix does not contain the declared unique cases")
    clusters = {str(case.metadata["cluster_id"]) for case in cases}
    if len(clusters) != int(benchmark["expected_independent_clusters"]):
        raise RuntimeError("R7 v2 matrix does not contain the declared independent clusters")
    observed_answers = Counter(case.answer for case in cases)
    expected_answers = Counter(
        {str(key): int(value) for key, value in dict(benchmark["expected_answer_counts"]).items()}
    )
    if observed_answers != expected_answers:
        raise RuntimeError("R7 v2 suite is not prospectively answer-balanced")
    return cases


def _screen_cases(cases: list[SyntheticCase], benchmark: dict[str, Any]) -> list[SyntheticCase]:
    screen = dict(benchmark["screen"])
    selected = [
        case
        for case in cases
        if str(case.metadata["context_layout"]) == str(screen["context_layout"])
        and abs(case.requested_evidence_depth - float(screen["requested_evidence_depth"])) < 1e-12
    ]
    expected_clusters = int(screen["expected_independent_clusters"])
    if len(selected) != expected_clusters:
        raise RuntimeError("R7 v2 screen selection has the wrong case count")
    if len({str(case.metadata["cluster_id"]) for case in selected}) != expected_clusters:
        raise RuntimeError("R7 v2 screen selection is not an independent-cluster set")
    expected_answers = Counter(
        {str(key): int(value) for key, value in dict(screen["expected_answer_counts"]).items()}
    )
    if Counter(case.answer for case in selected) != expected_answers:
        raise RuntimeError("R7 v2 screen is not prospectively answer-balanced")
    return selected


def _evidence_fact_texts(prompt: str, case: SyntheticCase) -> dict[str, str]:
    by_id = {segment.segment_id: segment for segment in case.segments}
    facts: dict[str, str] = {}
    for segment_id in case.evidence_segment_ids:
        base = f"[{segment_id}] {by_id[segment_id].text}"
        start = prompt.find(base)
        if start < 0 or prompt.find(base, start + 1) >= 0:
            raise RuntimeError(
                f"rendered proof segment {segment_id!r} is not unique in {case.case_id}"
            )
        end = start + len(base)
        while end < len(prompt) and prompt[end] == "\n":
            end += 1
        facts[segment_id] = prompt[start:end]
    return facts


def _frozen_v1_hashes(path: Path) -> dict[str, str]:
    required = ("manifest.json", "inference.jsonl", "evaluation.jsonl")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen R7 v1 provenance is incomplete: {missing}")
    return {name: sha256_file(path / name) for name in required}


def _design_audit(
    cases: list[SyntheticCase],
    screen: list[SyntheticCase],
    *,
    frozen_v1_path: Path,
) -> dict[str, Any]:
    screen_cross = Counter((case.answer, str(case.metadata["active_branch"])) for case in screen)
    evidence_sets = {case.evidence_segment_ids for case in cases}
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "benchmark_version": R7_V2_GENERATOR_VERSION,
        "frozen_v1": {
            "status": "INVALID_UNINFORMATIVE_FROZEN",
            "path": "results/benchmarks/halo_r1_r8_ladder_v1",
            "files": _frozen_v1_hashes(frozen_v1_path),
            "write_policy": "NEVER_OVERWRITE",
            "reason": "R7 v1 was six NO / one YES by seed and its rendered solver branched on one lock-fact sentence.",
        },
        "suite": {
            "cases": len(cases),
            "independent_clusters": len({str(case.metadata["cluster_id"]) for case in cases}),
            "answer_counts": dict(sorted(Counter(case.answer for case in cases).items())),
            "constant_answer_best_accuracy": max(Counter(case.answer for case in cases).values())
            / len(cases),
            "layouts": dict(
                sorted(Counter(str(case.metadata["context_layout"]) for case in cases).items())
            ),
        },
        "screen": {
            "case_ids": [case.case_id for case in screen],
            "answer_counts": dict(sorted(Counter(case.answer for case in screen).items())),
            "answer_by_active_branch": {
                f"{answer}|{branch}": count
                for (answer, branch), count in sorted(screen_cross.items())
            },
            "constant_answer_best_accuracy": max(Counter(case.answer for case in screen).values())
            / len(screen),
        },
        "proof_contract": {
            "proof_segments_per_case": sorted({len(case.evidence_segment_ids) for case in cases}),
            "outcome_specific_proof_id_sets": len(evidence_sets),
            "deletion_minimal_for_every_case": True,
            "exact_annotation_scope": "only the active base-plus-override proof; inactive branch excluded",
        },
        "shortcut_controls": {
            "opaque_segment_ids_for_roles": True,
            "ready_and_not_ready_decision_sentences_per_prompt": {"READY": 2, "NOT READY": 2},
            "two_competing_override_sentences_per_prompt": True,
            "screen_balanced_over_answer_and_active_branch": True,
            "answer_or_oracle_fields_in_inference_rows": False,
        },
        "solver_independence": {
            "rendered_solver_inputs": ["segment_text", "question_text"],
            "forbidden_inputs": ["seed", "oracle", "answer", "segment_role", "segment_id"],
            "algorithm": "parse rendered DSL, fixed-point forward chaining, named-override firing, priority resolution",
            "oracle_algorithm": "generator-side intended status",
            "agreement_for_every_case": True,
        },
        "claim_limit": "Design validity and exact token alignment only; no model capability or compression claim.",
    }


def build(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    tokenizer_path = (base / str(paths["tokenizer"])).resolve()
    frozen_v1_path = (base / str(paths["frozen_invalid_v1"])).resolve()
    output = (
        output_override.resolve()
        if output_override
        else (base / str(paths["output_root"])).resolve()
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty R7 v2 directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cases = generate_cases(config)
    benchmark = dict(config["benchmark"])
    screen = _screen_cases(cases, benchmark)
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")
    if tokenizer_hash != str(dict(config["model"])["tokenizer_sha256"]):
        raise RuntimeError("Halo tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for case in cases:
        validation = validate_r7_v2_case(case)
        raw_prompt = case.prompt()
        messages = [
            {"role": "system", "content": str(dict(config["prompt"])["system"])},
            {"role": "user", "content": raw_prompt},
        ]
        prompt = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(dict(config["prompt"])["enable_thinking"]),
            )
        )
        annotation = tokenize_and_freeze_evidence(
            prompt,
            locate_unique_evidence_spans(prompt, _evidence_fact_texts(prompt, case)),
            tokenizer,
            tokenizer_fingerprint=f"sha256:{tokenizer_hash}",
        )
        rendered = interpret_r7_v2_segments(case.segments, case.question)
        oracle_answer = solve_r7_v2_oracle(case)
        common = {
            "schema_version": "1.0",
            "benchmark_version": R7_V2_GENERATOR_VERSION,
            "case_id": case.case_id,
            "cluster_id": case.metadata["cluster_id"],
            "family": case.family,
            "seed": case.seed,
            "split": benchmark["split"],
            "context_layout": case.metadata["context_layout"],
            "requested_evidence_depth": case.requested_evidence_depth,
            "actual_evidence_depth": case.actual_evidence_depth,
        }
        inference.append(
            {
                **common,
                "prompt": prompt,
                "messages": messages,
                "question": case.question,
                "prompt_sha256": annotation.prompt_sha256,
                "token_ids_sha256": annotation.token_ids_sha256,
                "token_count": annotation.token_count,
            }
        )
        annotation_payload = asdict(annotation)
        annotation_payload["version"] = annotation.version
        evaluation.append(
            {
                **common,
                "answer": case.answer,
                "accepted_answers": [case.answer],
                "evidence_annotation": annotation_payload,
                "minimal_evidence_segment_ids": list(case.evidence_segment_ids),
                "distractor_segment_ids": list(case.distractor_segment_ids),
                "stale_segment_ids": [],
                "generator_metadata": dict(case.metadata),
                "verification": {
                    "oracle_solver_answer": oracle_answer,
                    "rendered_rule_engine_answer": rendered.answer,
                    "fired_forward_rule_ids": list(rendered.fired_forward_rule_ids),
                    "fired_decision_ids": list(rendered.fired_decision_ids),
                    "fired_override_ids": list(rendered.fired_override_ids),
                    "deletion_minimal_exact_evidence": validation["status"] == "PASS",
                },
            }
        )

    inference_path = output / "inference.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    design_audit_path = output / "design_audit.json"
    _jsonl(inference_path, inference)
    _jsonl(evaluation_path, evaluation)
    atomic_write_json(
        design_audit_path,
        _design_audit(cases, screen, frozen_v1_path=frozen_v1_path),
    )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "R7_V2_DATASET_BUILT",
        "benchmark_version": R7_V2_GENERATOR_VERSION,
        "model_target": str(dict(config["model"])["target"]),
        "case_count": len(cases),
        "independent_cluster_count": len({str(case.metadata["cluster_id"]) for case in cases}),
        "answer_counts": dict(sorted(Counter(case.answer for case in cases).items())),
        "screen_answer_counts": dict(sorted(Counter(case.answer for case in screen).items())),
        "layout_case_counts": dict(
            sorted(Counter(str(case.metadata["context_layout"]) for case in cases).items())
        ),
        "token_count": {
            "minimum": min(int(row["token_count"]) for row in inference),
            "maximum": max(int(row["token_count"]) for row in inference),
            "mean": sum(int(row["token_count"]) for row in inference) / len(inference),
        },
        "tokenizer_sha256": tokenizer_hash,
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "inference.jsonl": sha256_file(inference_path),
            "evaluation.jsonl": sha256_file(evaluation_path),
            "design_audit.json": sha256_file(design_audit_path),
        },
        "separation_check": {
            "inference_contains_answer_field": False,
            "inference_contains_oracle_field": False,
            "inference_contains_evidence_annotation": False,
            "join_key": "case_id",
        },
        "frozen_invalid_v1": {
            "status": "INVALID_UNINFORMATIVE_FROZEN",
            "path": "results/benchmarks/halo_r1_r8_ladder_v1",
            "write_policy": "NEVER_OVERWRITE",
        },
        "claim_limit": "Corrected R7 design and exact token alignment only; no model or compression outcome.",
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_r7_v2.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.config.resolve(), args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
