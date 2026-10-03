"""Offline prediction evaluation. Never starts inference or training."""
from __future__ import annotations
import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from .common import read_jsonl, loads, sha_file, write_json, dumps, integer
from .rewards import score_response


def _wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total; d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0., center-radius), min(1., center+radius)]


def evaluate(root: Path, predictions: Path, split: str, mode: str = "final_only", samples_per_prompt: int = 1) -> dict[str, Any]:
    if not integer(samples_per_prompt) or samples_per_prompt < 1 or samples_per_prompt > 128:
        raise ValueError("samples_per_prompt must be in [1,128]")
    metadata_path = root / "metadata" / f"{split}.jsonl.gz"
    manifest = loads((root / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    rel = f"metadata/{split}.jsonl.gz"
    if rel not in manifest["files"] or sha_file(metadata_path) != manifest["files"][rel]["sha256"]:
        raise ValueError("Reference checksum does not match frozen manifest")
    refs = {r["id"]: r for r in read_jsonl(metadata_path)}
    supplied: dict[tuple[str, int], dict[str, Any]] = {}
    for p in read_jsonl(predictions):
        if not {"id", "prediction"} <= set(p) or set(p) - {"id", "prediction", "sample_id", "finish_reason", "completion_tokens", "reasoning_tokens"}:
            raise ValueError("Unexpected prediction schema")
        i = p.get("sample_id", 0)
        if not isinstance(p["id"], str) or p["id"] not in refs or not integer(i) or not 0 <= i < samples_per_prompt or not isinstance(p["prediction"], str):
            raise ValueError("Unknown ID, invalid sample index, or nontext prediction")
        key = (p["id"], i)
        if key in supplied:
            raise ValueError("Duplicate prediction key")
        for name in ("completion_tokens", "reasoning_tokens"):
            if name in p and (not integer(p[name]) or p[name] < 0):
                raise ValueError("Invalid measured token count")
        if "finish_reason" in p and p["finish_reason"] is not None and not isinstance(p["finish_reason"], str):
            raise ValueError("finish_reason must be text or null")
        supplied[key] = p
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_row = []; detail = []; nonzero = 0; complete_groups = 0
    for rid, ref in refs.items():
        scores = []
        for i in range(samples_per_prompt):
            p = supplied.get((rid, i))
            result = score_response(ref["world"], p["prediction"] if p else "", mode=mode, certificate=ref["certificate"])
            result["id"] = rid; result["sample_id"] = i; result["missing"] = p is None
            result["truncated"] = p is not None and p.get("finish_reason") in ("length", "max_tokens")
            # A truncated sample cannot be counted as a completed solution even if
            # a syntactically complete prefix happened to appear before truncation.
            if result["truncated"]:
                result["correct"] = False; result["reward"] = 0.; result["reason"] = "truncated_generation"
            if p:
                for token_field in ("completion_tokens", "reasoning_tokens"):
                    if token_field in p:
                        result[token_field] = p[token_field]
            scores.append(result); detail.append(result)
            groups["family:" + ref["family"]].append(result)
            groups["layout:" + ref["layout"]["layout"]].append(result)
            groups["words:" + str(ref["layout"]["target_words"])].append(result)
            groups["status:" + ref["reference"].get("status", "ANSWER")].append(result)
        complete = all(not s["missing"] for s in scores)
        if complete:
            complete_groups += 1
            nonzero += int(len({s["reward"] for s in scores}) > 1)
        per_row.append({"id": rid, "correct": sum(int(s["correct"]) for s in scores), "complete": complete})
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows); c = sum(bool(s["correct"]) for s in rows)
        tokens = [s["completion_tokens"] for s in rows if "completion_tokens" in s]
        reasoning_tokens = [s["reasoning_tokens"] for s in rows if "reasoning_tokens" in s]
        present = [s for s in rows if not s["missing"]]
        return {"samples_expected": n, "samples_received": len(present), "coverage": len(present)/n if n else None,
                "correct": c, "accuracy_missing_as_wrong": c/n if n else None,
                "wilson_95_descriptive": _wilson(c, n) if samples_per_prompt == 1 else None,
                "format_valid": sum(bool(s["format_valid"]) for s in rows),
                "schema_valid": sum(bool(s["schema_valid"]) for s in rows),
                "feasible": sum(bool(s["feasible"]) for s in rows),
                "truncated": sum(bool(s["truncated"]) for s in rows),
                "mean_response_characters_received": statistics.mean(s["response_characters"] for s in present) if present else None,
                "mean_final_characters_received": statistics.mean(s["final_characters"] for s in present) if present else None,
                "measured_token_coverage": len(tokens)/n if n else None,
                "mean_measured_completion_tokens": statistics.mean(tokens) if tokens else None,
                "mean_trace_characters_received": statistics.mean(s["trace_characters"] for s in present) if present else None,
                "measured_reasoning_token_coverage": len(reasoning_tokens)/n if n else None,
                "mean_measured_reasoning_tokens": statistics.mean(reasoning_tokens) if reasoning_tokens else None}
    metrics = {k: aggregate(v) for k, v in sorted(groups.items())}
    all_metrics = aggregate(detail)
    pass_k = {}
    for k in (1, 2, 4, 8):
        if k <= samples_per_prompt:
            eligible = [r for r in per_row if r["complete"]]
            estimates = [1 - (math.comb(samples_per_prompt-r["correct"], k)/math.comb(samples_per_prompt,k) if samples_per_prompt-r["correct"] >= k else 0) for r in eligible]
            pass_k[str(k)] = {"complete_groups": len(eligible), "expected_groups": len(refs), "estimate_complete_groups_only": statistics.mean(estimates) if estimates else None}
    family_scores = [m["accuracy_missing_as_wrong"] for key,m in metrics.items() if key.startswith("family:")]
    return {"split": split, "mode": mode, "independent_worlds": len(refs), "samples_per_prompt": samples_per_prompt,
            "overall": all_metrics, "macro_family_accuracy_missing_as_wrong": statistics.mean(family_scores),
            "strata": metrics, "pass_at_k": pass_k,
            "complete_prompt_groups": complete_groups, "nonzero_reward_variance_groups": nonzero,
            "nonzero_reward_variance_fraction": nonzero/complete_groups if complete_groups else None,
            "prediction_file_sha256": sha_file(predictions), "reference_file_sha256": sha_file(metadata_path),
            "full_coverage": len(supplied) == len(refs)*samples_per_prompt,
            "notes": ["No inference was run by this evaluator.", "Published holdouts are developer-visible, not cryptographically sealed.",
                      "Free-form reasoning correctness and faithfulness are NOT scored.", "Wilson intervals are descriptive; shared templates and related tasks limit IID interpretation.",
                      "pass@k uses complete groups only; missing-group coverage must be reported alongside it."],
            "per_sample": detail}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "test_iid", "test_transfer", "test_length"], required=True)
    p.add_argument("--mode", choices=["final_only", "reason_then_final"], default="final_only")
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Refusing to overwrite evaluation output")
    result = evaluate(a.root, a.predictions, a.split, a.mode, a.samples_per_prompt)
    write_json(a.output, result)
    print(dumps({"overall": result["overall"], "full_coverage": result["full_coverage"]}))


if __name__ == "__main__":
    main()
