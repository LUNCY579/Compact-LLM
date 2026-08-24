#!/usr/bin/env python3
"""Run the 32-cluster full-cache Halo capability calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    git_snapshot,
    sha256_file,
    utc_now,
)
from rineya_significance_lab.runtime import SignificanceCache, chunked_prefill  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("capability config must be a mapping")
    return value


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(text: str) -> str:
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text.strip().lower())
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s,-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip().rstrip(".,")


def _task_correct(family: str, prediction: str, accepted_answers: list[str]) -> bool:
    accepted = {_normalize(value) for value in accepted_answers}
    if _normalize(prediction) in accepted:
        return True
    if family in {"R1", "R3", "R4"}:
        values = re.findall(r"(?<![A-Za-z])[-+]?\d+", prediction)
        return bool(values) and str(int(values[-1])) in accepted
    if family == "R2":
        pairs = re.findall(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", prediction)
        return bool(pairs) and f"{int(pairs[-1][0])},{int(pairs[-1][1])}" in accepted
    return False


def _token_hash(ids) -> str:
    import torch

    value = ids.detach().to("cpu", dtype=torch.int64).contiguous()
    canonical = json.dumps(tuple(int(token) for token in value.flatten().tolist()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _generate(model, tokenizer, input_ids, *, chunk_size: int, max_new_tokens: int) -> tuple[str, int, float]:
    import torch

    cache = SignificanceCache(config=model.config)
    started = time.perf_counter()
    prefill = chunked_prefill(
        model,
        input_ids,
        cache,
        chunk_size=chunk_size,
        policy=None,
        collect_logits=False,
        logits_to_keep=1,
    )
    logits = prefill.output.logits
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    for index in range(max_new_tokens):
        token = int(logits[0, -1].float().argmax().item())
        generated.append(token)
        if token in eos_ids or index == max_new_tokens - 1:
            break
        step_ids = torch.tensor([[token]], dtype=torch.long)
        step = chunked_prefill(
            model,
            step_ids,
            cache,
            chunk_size=1,
            policy=None,
            collect_logits=False,
            logits_to_keep=1,
        )
        logits = step.output.logits
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), cache.stored_length(), time.perf_counter() - started


def run(config_path: Path) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = utc_now()
    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    model_path = (base / str(paths["model"])).resolve()
    inference_path = (base / str(paths["inference"])).resolve()
    evaluation_path = (base / str(paths["evaluation"])).resolve()
    output = (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty capability run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expected = dict(dict(config["model"])["expected_hashes"])
    observed = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed != expected:
        raise RuntimeError(f"Halo checkpoint hash mismatch: {observed}")
    inference = _rows(inference_path)
    selection = dict(config.get("selection", {}))
    expected_clusters = int(selection.get("expected_independent_clusters", 32))
    if len(inference) != expected_clusters or len({row["cluster_id"] for row in inference}) != expected_clusters:
        raise RuntimeError(
            f"capability input must contain {expected_clusters} independent clusters"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    torch.manual_seed(int(dict(config["runtime"])["seed"]))
    unscored: list[dict[str, object]] = []
    for row in inference:
        ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        if _token_hash(ids) != row["token_ids_sha256"]:
            raise RuntimeError(f"token hash mismatch for {row['case_id']}")
        prediction, resident, seconds = _generate(
            model,
            tokenizer,
            ids,
            chunk_size=int(dict(config["runtime"])["chunk_size"]),
            max_new_tokens=int(dict(config["runtime"])["max_new_tokens"]),
        )
        unscored.append(
            {
                "case_id": row["case_id"],
                "cluster_id": row["cluster_id"],
                "family": row["family"],
                "prediction": prediction,
                "input_tokens": int(ids.shape[-1]),
                "token_ids_sha256": row["token_ids_sha256"],
                "resident_entries": resident,
                "inference_wall_seconds": seconds,
            }
        )
    predictions_path = output / "predictions.unscored.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in unscored),
        encoding="utf-8",
    )
    # First answer/evidence view access occurs only after predictions are durable.
    evaluation = {str(row["case_id"]): row for row in _rows(evaluation_path)}
    scored: list[dict[str, object]] = []
    for row in unscored:
        key = evaluation[str(row["case_id"])]
        correct = _task_correct(
            str(row["family"]),
            str(row["prediction"]),
            [str(value) for value in key["accepted_answers"]],
        )
        scored.append(
            {
                "schema_version": "1.0",
                "run_id": output.name,
                "case_id": row["case_id"],
                "scenario_id": row["cluster_id"],
                "condition": "B0",
                "task_family": row["family"],
                "generator_seed": None,
                "prompt_sha256": row["token_ids_sha256"],
                "context_tokens": row["input_tokens"],
                "evidence_depth": 0.5,
                "chunk_size": int(dict(config["runtime"])["chunk_size"]),
                "status": "OK",
                "prediction": row["prediction"],
                "correct": correct,
                "full_cache_correct": correct,
                "compression_conditioned_eligible": correct,
                "evidence": None,
                "efficiency": {"wall_seconds": row["inference_wall_seconds"]},
                "failure_labels": [] if correct else ["FULL_CACHE_CAPABILITY_FAILURE"],
                "trace_ids": [],
                "error": None,
                "cluster_id": row["cluster_id"],
                "accepted_answers": key["accepted_answers"],
            }
        )
    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    families = sorted({str(row["task_family"]) for row in scored})
    by_family = {
        family: {
            "cases": sum(row["task_family"] == family for row in scored),
            "correct": sum(row["task_family"] == family and row["correct"] for row in scored),
        }
        for family in families
    }
    correct_count = sum(bool(row["correct"]) for row in scored)
    minimum_overall = int(selection.get("minimum_correct_overall", 12))
    minimum_per_family = int(selection.get("minimum_correct_per_family", 2))
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "FULL_CACHE_CAPABILITY_CALIBRATION",
        "independent_clusters": expected_clusters,
        "correct_clusters": correct_count,
        "capability_rate": correct_count / expected_clusters,
        "by_family": by_family,
        "advance_rule": (
            f"at least {minimum_overall} correct overall and at least "
            f"{minimum_per_family} per family"
        ),
        "advance_gate_passed": correct_count >= minimum_overall
        and all(row["correct"] >= minimum_per_family for row in by_family.values()),
        "inferential_statistics": None,
    }
    atomic_write_json(output / "summary.json", summary)
    repo = git_snapshot(REPO_ROOT)
    manifest = {
        "schema_version": "1.0",
        "run_id": output.name,
        "status": "COMPLETE",
        "evidence_class": "NEW_CHATBUBBLE_EVIDENCE",
        "created_at": started_at,
        "completed_at": utc_now(),
        "repositories": {
            "chatbubble": {"remote": repo["remote"], "commit": repo["commit"], "dirty": repo["dirty"]},
            "source": {
                "remote": "https://github.com/LUNCY579/RineyaBeta.git",
                "implementation": {"ref": "refs/heads/main", "commit": "9e00be74098878290dc2b0d54cbffee4652c6088", "path": "scripts/kv/significance.py", "blob": "30ee29724dce52de5a288f6421123fb718240541"},
                "paper": {"ref": "refs/heads/r2-healing-harness", "commit": "1664f54de0bbdfde45e311b75bbf3038dbfffe42", "path": "paper/significance_matrix/main.tex", "blob": "b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3"},
            },
        },
        "experiment": {"name": config["experiment"]["name"], "compute_tier": "B", "architecture_variant": "B0 full cache", "benchmark": selection.get("benchmark_id", inference_path.parent.name), "seed": config["runtime"]["seed"], "query_conditioned": False},
        "config": {"resolved_path": f"configs/{config_path.name}", "sha256": sha256_file(config_path)},
        "model": {"status": "VERIFIED", "id": config["model"]["core_id"], "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze", "architecture": "Qwen3ForCausalLM", "hashes": {**observed, "chat_template": None}},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu", "precision": "bfloat16", "hardware": None},
        "budget": None,
        "artifacts": {"predictions": "predictions.unscored.jsonl", "cases": "cases.jsonl", "summary": "summary.json"},
        "notes": ["Full-cache capability calibration only; no compression-policy outcome."],
    }
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "halo_capability.yaml")
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
