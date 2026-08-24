#!/usr/bin/env python3
"""Build separate prompt/evaluation views for Halo capability calibration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.benchmarks import (  # noqa: E402
    generate_suite,
    locate_unique_evidence_spans,
    solve_case,
    solve_case_from_segments,
    tokenize_and_freeze_evidence,
)
from rineya_significance_lab.manifest import atomic_write_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("calibration config root must be a mapping")
    return value


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def build(config_path: Path, output_override: Path | None = None) -> dict[str, object]:
    from transformers import AutoTokenizer

    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    tokenizer_path = (base / str(paths["tokenizer"])).resolve()
    output = output_override.resolve() if output_override else (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty calibration directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    benchmark = dict(config["benchmark"])
    cases = generate_suite(
        benchmark["seeds"],
        benchmark["evidence_depths"],
        families=benchmark["families"],
        filler_count=int(benchmark["filler_count"]),
    )
    expected = int(benchmark["expected_independent_clusters"])
    cluster_ids = {str(case.metadata["cluster_id"]) for case in cases}
    if len(cases) != expected or len(cluster_ids) != expected:
        raise RuntimeError("calibration matrix does not contain the declared independent clusters")
    tokenizer_hash = sha256_file(tokenizer_path / "tokenizer.json")
    declared_hash = str(dict(config["model"])["tokenizer_sha256"])
    if tokenizer_hash != declared_hash:
        raise RuntimeError("Halo tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    inference: list[dict[str, object]] = []
    evaluation: list[dict[str, object]] = []
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
        by_id = {segment.segment_id: segment for segment in case.segments}
        # Include the rendered segment label and trailing newline so evidence
        # boundaries coincide exactly with Halo byte-BPE token boundaries.
        fact_texts = {
            segment_id: f"[{segment_id}] {by_id[segment_id].text}\n"
            for segment_id in case.evidence_segment_ids
        }
        spans = locate_unique_evidence_spans(prompt, fact_texts)
        annotation = tokenize_and_freeze_evidence(
            prompt,
            spans,
            tokenizer,
            tokenizer_fingerprint=f"sha256:{tokenizer_hash}",
        )
        rendered_answer = solve_case_from_segments(case)
        oracle_answer = solve_case(case)
        if rendered_answer != oracle_answer or rendered_answer != case.answer:
            raise RuntimeError(f"independent answer verification failed for {case.case_id}")
        inference.append(
            {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "cluster_id": case.metadata["cluster_id"],
                "family": case.family,
                "seed": case.seed,
                "split": benchmark["split"],
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
                "schema_version": "1.0",
                "case_id": case.case_id,
                "cluster_id": case.metadata["cluster_id"],
                "family": case.family,
                "answer": case.answer,
                "accepted_answers": [case.answer],
                "evidence_annotation": annotation_payload,
                "distractor_segment_ids": list(case.distractor_segment_ids),
                "stale_segment_ids": list(case.stale_segment_ids),
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
    manifest = {
        "schema_version": "1.0",
        "status": "CALIBRATION_DATASET_BUILT",
        "model_target": "Rineya-Halo-1.2.0-devfreeze",
        "case_count": len(cases),
        "independent_cluster_count": len(cluster_ids),
        "families": list(benchmark["families"]),
        "seeds": list(benchmark["seeds"]),
        "evidence_depths": list(benchmark["evidence_depths"]),
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
        "claim_limit": "Dataset construction and evidence-token alignment only; no model outcome.",
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_calibration.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.config.resolve(), args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
