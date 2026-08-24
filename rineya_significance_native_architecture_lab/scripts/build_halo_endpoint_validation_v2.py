#!/usr/bin/env python3
"""Build the easier, fresh-cluster prospective endpoint-validation v2 set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import (  # noqa: E402
    SyntheticCase,
    generate_case,
    generate_r1_cardinality_case,
    locate_unique_evidence_spans,
    solve_case,
    solve_case_from_segments,
    tokenize_and_freeze_evidence,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("endpoint-validation v2 config must be a mapping")
    return value


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def generate_cases(config: dict[str, Any]) -> list[SyntheticCase]:
    benchmark = dict(config["benchmark"])
    seeds = [int(value) for value in benchmark["seeds"]]
    depth = float(benchmark["evidence_depth"])
    filler = int(benchmark["filler_count"])
    cases: list[SyntheticCase] = []
    for seed in seeds:
        cases.append(
            generate_r1_cardinality_case(
                seed,
                depth,
                key_count=int(benchmark["r1_required_fact_count"]),
                filler_count=filler,
                layout="clustered",
            )
        )
    for family in ("R3", "R4"):
        for seed in seeds:
            case = generate_case(family, seed, depth, filler_count=filler)
            if family == "R3" and len(case.oracle["updates"]) != int(
                benchmark["r3_update_count"]
            ):
                raise RuntimeError(f"seed {seed} does not realize the frozen R3 update count")
            cases.append(case)
    expected = int(benchmark["expected_independent_clusters"])
    if len(cases) != expected or len({str(case.metadata["cluster_id"]) for case in cases}) != expected:
        raise RuntimeError("endpoint-validation v2 matrix is not the declared cluster set")
    return cases


def _fact_texts(prompt: str, case: SyntheticCase) -> dict[str, str]:
    by_id = {segment.segment_id: segment for segment in case.segments}
    result: dict[str, str] = {}
    for segment_id in case.evidence_segment_ids:
        base = f"[{segment_id}] {by_id[segment_id].text}"
        start = prompt.find(base)
        if start < 0 or prompt.find(base, start + 1) >= 0:
            raise RuntimeError(f"rendered fact {segment_id!r} is not unique")
        end = start + len(base)
        while end < len(prompt) and prompt[end] == "\n":
            end += 1
        result[segment_id] = prompt[start:end]
    return result


def build(config_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config_path = config_path.resolve(strict=True)
    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    tokenizer_path = (base / str(paths["tokenizer"])).resolve(strict=True)
    output = (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite endpoint-validation v2 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")
    if tokenizer_hash != str(config["model"]["tokenizer_sha256"]):
        raise RuntimeError("Halo tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    cases = generate_cases(config)
    benchmark = dict(config["benchmark"])
    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for case in cases:
        messages = [
            {"role": "system", "content": str(config["prompt"]["system"])},
            {"role": "user", "content": case.prompt()},
        ]
        prompt = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(config["prompt"]["enable_thinking"]),
            )
        )
        annotation = tokenize_and_freeze_evidence(
            prompt,
            locate_unique_evidence_spans(prompt, _fact_texts(prompt, case)),
            tokenizer,
            tokenizer_fingerprint=f"sha256:{tokenizer_hash}",
        )
        oracle = solve_case(case)
        rendered = solve_case_from_segments(case)
        if oracle != rendered or oracle != case.answer:
            raise RuntimeError(f"independent solver mismatch for {case.case_id}")
        common = {
            "schema_version": "1.0",
            "case_id": case.case_id,
            "cluster_id": case.metadata["cluster_id"],
            "family": case.family,
            "seed": case.seed,
            "split": benchmark["split"],
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
                "distractor_segment_ids": list(case.distractor_segment_ids),
                "stale_segment_ids": list(case.stale_segment_ids),
                "generator_metadata": dict(case.metadata),
                "verification": {
                    "oracle_solver_answer": oracle,
                    "rendered_segment_solver_answer": rendered,
                },
            }
        )
    inference_path = output / "inference.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    _jsonl(inference_path, inference)
    _jsonl(evaluation_path, evaluation)
    manifest = {
        "schema_version": "1.0",
        "status": "PROSPECTIVE_ENDPOINT_VALIDATION_V2_DATASET_BUILT",
        "classification": "DIFFICULTY_REDESIGN_AFTER_V1_B0_FAILURE_BEFORE_COMPRESSION",
        "model_target": config["model"]["target"],
        "case_count": len(cases),
        "independent_cluster_count": len({str(case.metadata["cluster_id"]) for case in cases}),
        "families": ["R1", "R3", "R4"],
        "difficulty": {
            "R1_required_facts": benchmark["r1_required_fact_count"],
            "R3_updates": benchmark["r3_update_count"],
            "R4": "unchanged",
        },
        "seeds": benchmark["seeds"],
        "tokenizer_sha256": tokenizer_hash,
        "config_sha256": sha256_file(config_path),
        "artifacts": {
            "inference.jsonl": sha256_file(inference_path),
            "evaluation.jsonl": sha256_file(evaluation_path),
        },
        "separation_check": {
            "inference_contains_answer": False,
            "inference_contains_evidence_annotation": False,
            "join_key": "case_id",
        },
        "claim_limit": "Dataset construction only; no B0 or compression outcome.",
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_endpoint_validation_v2_dataset.yaml",
    )
    args = parser.parse_args()
    result = build(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
