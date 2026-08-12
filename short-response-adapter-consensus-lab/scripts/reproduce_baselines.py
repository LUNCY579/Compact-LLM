from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.decoding.decoder import generate_reference, generate_separate_cache
from consensus_lab.evaluation.scoring import answer_correct
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee, ExpertSpec
from consensus_lab.paths import LabPaths


def run_candidate(
    *,
    paths: LabPaths,
    name: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    official_by_id: dict[str, dict[str, Any]],
    device: str,
    precision: str,
    max_sequence_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=model_path,
        experts=[ExpertSpec(name="backbone", adapter_path=None)],
        device=device,
        precision=precision,
    )
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        messages = [dict(value) for value in row["messages"]]
        input_ids = committee.encode_messages(messages[:-1], max_sequence_length=max_sequence_length)
        result = generate_reference(
            committee,
            input_ids,
            method="hard_plurality",
            max_new_tokens=max_new_tokens,
            eos_policy="ordinary",
        )
        recorded = official_by_id[str(row["id"])]
        comparison = {
                "id": row["id"],
                "output": result.output_text,
                "official_output": recorded["answer"],
                "exact_output_match": result.output_text == recorded["answer"],
                "generated_tokens": len(result.output_token_ids),
                "official_generated_tokens": recorded["generated_tokens"],
                "token_count_match": len(result.output_token_ids) == int(recorded["generated_tokens"]),
                "correct": answer_correct(result.output_text, row),
                "official_correct": bool(recorded["correct"]),
                "correctness_match": answer_correct(result.output_text, row) == bool(recorded["correct"]),
                "latency_seconds": result.latency_seconds,
            }
        if not comparison["exact_output_match"]:
            cached = generate_separate_cache(
                committee,
                input_ids,
                method="hard_plurality",
                max_new_tokens=max_new_tokens,
                eos_policy="ordinary",
            )
            comparison.update(
                {
                    "separate_cache_output": cached.output_text,
                    "separate_cache_matches_official": cached.output_text == recorded["answer"],
                    "cached_matches_no_cache": cached.output_token_ids == result.output_token_ids,
                }
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                generated = committee.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=committee.tokenizer.pad_token_id,
                    eos_token_id=committee.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            hf_ids = generated.sequences[0, input_ids.shape[1] :]
            hf_output = committee.tokenizer.decode(hf_ids, skip_special_tokens=True).strip()
            raw_logits, _ = committee.forward(
                committee.experts[0], input_ids, attention_mask=attention_mask, use_cache=True
            )
            hf_first_scores = generated.scores[0][0].float()
            hf_top = torch.topk(hf_first_scores, k=5)
            raw_top = torch.topk(raw_logits, k=5)
            comparison.update(
                {
                    "same_loaded_model_hf_generate_output": hf_output,
                    "same_loaded_model_hf_matches_official": hf_output == recorded["answer"],
                    "same_loaded_model_hf_matches_reference": hf_ids.tolist() == result.output_token_ids,
                    "hf_first_token_top5": [
                        {"token_id": int(token), "score": float(score)}
                        for token, score in zip(hf_top.indices.tolist(), hf_top.values.tolist(), strict=True)
                    ],
                    "raw_first_token_top5": [
                        {"token_id": int(token), "score": float(score)}
                        for token, score in zip(raw_top.indices.tolist(), raw_top.values.tolist(), strict=True)
                    ],
                    "hf_raw_first_score_max_abs_diff": float(torch.max(torch.abs(hf_first_scores - raw_logits)).item()),
                }
            )
        comparisons.append(comparison)
    packet = {
        "name": name,
        "model": str(model_path),
        "case_count": len(comparisons),
        "exact_output_match_rate": sum(row["exact_output_match"] for row in comparisons) / len(comparisons),
        "token_count_match_rate": sum(row["token_count_match"] for row in comparisons) / len(comparisons),
        "correctness_match_rate": sum(row["correctness_match"] for row in comparisons) / len(comparisons),
        "reproduced_correctness": sum(row["correct"] for row in comparisons) / len(comparisons),
        "official_correctness_on_packet": sum(row["official_correct"] for row in comparisons) / len(comparisons),
        "comparisons": comparisons,
    }
    del committee
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce deterministic recorded outputs for frozen and D_greedy anchors.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/baseline.yaml")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--candidate", choices=("frozen_anchor", "greedy_soup"), action="append")
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark = paths.assert_official_source(paths.official_root / raw["benchmark"])
    rows = read_jsonl(benchmark)[: args.limit]
    candidates = []
    for name, model_key, results_key in (
        ("frozen_anchor", "frozen_anchor", "official_frozen_results"),
        ("greedy_soup", "greedy_soup", "official_greedy_results"),
    ):
        if args.candidate and name not in args.candidate:
            continue
        model_path = paths.assert_official_source(paths.official_root / raw[model_key])
        official = {
            str(row["id"]): row
            for row in read_jsonl(paths.assert_official_source(paths.official_root / raw[results_key]))
        }
        candidates.append(
            run_candidate(
                paths=paths,
                name=name,
                model_path=model_path,
                rows=rows,
                official_by_id=official,
                device=args.device,
                precision=args.precision,
                max_sequence_length=int(raw["max_sequence_length"]),
                max_new_tokens=int(raw["max_new_tokens"]),
            )
        )
    packet = {"schema_version": "1.0", "benchmark": str(benchmark), "candidates": candidates}
    output = paths.assert_output_path(LAB_ROOT / "results/summaries/baseline_reproduction.json")
    write_json_atomic(output, packet)
    lines = ["# Baseline Reproduction", ""]
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate['name']}",
                "",
                f"- Cases: **{candidate['case_count']}**",
                f"- Exact output match: **{candidate['exact_output_match_rate']:.4f}**",
                f"- Token-count match: **{candidate['token_count_match_rate']:.4f}**",
                f"- Correctness match: **{candidate['correctness_match_rate']:.4f}**",
                "",
            ]
        )
    report = paths.assert_output_path(LAB_ROOT / "reports/BASELINE_REPRODUCTION.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
