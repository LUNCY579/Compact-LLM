#!/usr/bin/env python3
"""Build the frozen, exactly annotated Halo R1-R8 Tier-A ladder."""

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
    SyntheticCase,
    generate_advanced_case,
    generate_case,
    generate_r1_cardinality_case,
    locate_unique_evidence_spans,
    relayout_case,
    solve_advanced_case,
    solve_advanced_case_from_segments,
    solve_case,
    solve_case_from_segments,
    tokenize_and_freeze_evidence,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ladder config root must be a mapping")
    return value


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def generate_ladder_cases(config: dict[str, Any]) -> list[SyntheticCase]:
    benchmark = dict(config["benchmark"])
    seeds = [int(value) for value in benchmark["seeds"]]
    depths = [float(value) for value in benchmark["clustered_evidence_depths"]]
    fixed_layouts = [str(value) for value in benchmark["fixed_layouts"]]
    fixed_depth = float(benchmark["fixed_layout_requested_depth"])
    filler_count = int(benchmark["filler_count"])
    cases: list[SyntheticCase] = []

    for family_value in benchmark["families"]:
        family = str(family_value)
        for seed in seeds:
            if family == "R1":
                for count_value in benchmark["r1_required_fact_counts"]:
                    key_count = int(count_value)
                    cases.extend(
                        generate_r1_cardinality_case(
                            seed,
                            depth,
                            key_count=key_count,
                            filler_count=filler_count,
                            layout="clustered",
                        )
                        for depth in depths
                    )
                    cases.extend(
                        generate_r1_cardinality_case(
                            seed,
                            fixed_depth,
                            key_count=key_count,
                            filler_count=filler_count,
                            layout=layout,  # type: ignore[arg-type]
                        )
                        for layout in fixed_layouts
                    )
            elif family in {"R2", "R3", "R4"}:
                base = generate_case(family, seed, 0.5, filler_count=filler_count)
                cases.extend(relayout_case(base, layout="clustered", evidence_depth=depth) for depth in depths)
                cases.extend(
                    relayout_case(base, layout=layout, evidence_depth=fixed_depth)  # type: ignore[arg-type]
                    for layout in fixed_layouts
                )
            elif family in {"R5", "R6", "R7", "R8"}:
                cases.extend(
                    generate_advanced_case(
                        family,
                        seed,
                        depth,
                        filler_count=filler_count,
                        layout="clustered",
                    )
                    for depth in depths
                )
                cases.extend(
                    generate_advanced_case(
                        family,
                        seed,
                        fixed_depth,
                        filler_count=filler_count,
                        layout=layout,  # type: ignore[arg-type]
                    )
                    for layout in fixed_layouts
                )
            else:
                raise ValueError(f"unsupported ladder family: {family}")

    expected = int(benchmark["expected_cases"])
    if len(cases) != expected or len({case.case_id for case in cases}) != expected:
        raise RuntimeError("ladder matrix does not contain the declared unique cases")
    cluster_ids = {str(case.metadata["cluster_id"]) for case in cases}
    if len(cluster_ids) != int(benchmark["expected_independent_clusters"]):
        raise RuntimeError("ladder matrix does not contain the declared independent clusters")
    return cases


def _verified_answer(case: SyntheticCase) -> tuple[str, str]:
    if case.family in {"R1", "R2", "R3", "R4"}:
        return solve_case(case), solve_case_from_segments(case)
    return solve_advanced_case(case), solve_advanced_case_from_segments(case)


def _evidence_fact_texts(prompt: str, case: SyntheticCase) -> dict[str, str]:
    """Return unique rendered facts with their complete trailing newline run.

    Halo's byte-level tokenizer can encode the blank line before ``QUESTION``
    as one token. Including the full local newline run keeps fact boundaries on
    exact token offsets without absorbing any question text.
    """

    by_id = {segment.segment_id: segment for segment in case.segments}
    facts: dict[str, str] = {}
    for segment_id in case.evidence_segment_ids:
        base = f"[{segment_id}] {by_id[segment_id].text}"
        start = prompt.find(base)
        if start < 0 or prompt.find(base, start + 1) >= 0:
            raise RuntimeError(f"rendered evidence {segment_id!r} is not unique in {case.case_id}")
        end = start + len(base)
        while end < len(prompt) and prompt[end] == "\n":
            end += 1
        facts[segment_id] = prompt[start:end]
    return facts


def build(config_path: Path, output_override: Path | None = None) -> dict[str, object]:
    from transformers import AutoTokenizer

    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    tokenizer_path = (base / str(paths["tokenizer"])).resolve()
    output = output_override.resolve() if output_override else (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty ladder directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cases = generate_ladder_cases(config)
    benchmark = dict(config["benchmark"])
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")
    expected_hash = str(dict(config["model"])["tokenizer_sha256"])
    if tokenizer_hash != expected_hash:
        raise RuntimeError("Halo tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for case in cases:
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
        fact_texts = _evidence_fact_texts(prompt, case)
        annotation = tokenize_and_freeze_evidence(
            prompt,
            locate_unique_evidence_spans(prompt, fact_texts),
            tokenizer,
            tokenizer_fingerprint=f"sha256:{tokenizer_hash}",
        )
        oracle_answer, rendered_answer = _verified_answer(case)
        if oracle_answer != rendered_answer or oracle_answer != case.answer:
            raise RuntimeError(f"independent answer verification failed for {case.case_id}")
        common = {
            "schema_version": "1.0",
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
                "distractor_segment_ids": list(case.distractor_segment_ids),
                "stale_segment_ids": list(case.stale_segment_ids),
                "generator_metadata": dict(case.metadata),
                "verification": {
                    "oracle_solver_answer": oracle_answer,
                    "rendered_segment_solver_answer": rendered_answer,
                },
            }
        )

    inference_path = output / "inference.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    _jsonl(inference_path, inference)
    _jsonl(evaluation_path, evaluation)
    families = Counter(case.family for case in cases)
    layouts = Counter(str(case.metadata["context_layout"]) for case in cases)
    r1_counts = Counter(int(case.metadata["required_fact_count"]) for case in cases if case.family == "R1")
    r2_hops = Counter(int(case.oracle["hops"]) for case in cases if case.family == "R2")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "TIER_A_LADDER_BUILT",
        "model_target": str(dict(config["model"])["target"]),
        "case_count": len(cases),
        "independent_cluster_count": len({str(case.metadata["cluster_id"]) for case in cases}),
        "family_case_counts": dict(sorted(families.items())),
        "layout_case_counts": dict(sorted(layouts.items())),
        "r1_required_fact_count_cases": {str(key): value for key, value in sorted(r1_counts.items())},
        "r2_hop_count_cases": {str(key): value for key, value in sorted(r2_hops.items())},
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
        },
        "separation_check": {
            "inference_contains_answer_field": False,
            "inference_contains_oracle_field": False,
            "inference_contains_evidence_annotation": False,
            "join_key": "case_id",
        },
        "claim_limit": "Tier-A construction and exact evidence-token alignment only; no model outcome.",
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_r1_r8_ladder.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.config.resolve(), args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
