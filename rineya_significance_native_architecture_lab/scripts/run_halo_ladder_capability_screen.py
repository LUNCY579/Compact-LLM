#!/usr/bin/env python3
"""Run a bounded full-cache Halo capability screen on new R5-R8 stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

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


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner helper: {path}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("capability-screen config must be a mapping")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(text: str) -> str:
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text.strip().lower())
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s,+-]", " ", value)
    return re.sub(r"\s+", " ", value).strip().rstrip(".,")


def _task_correct(family: str, prediction: str, accepted_answers: list[str]) -> bool:
    if family not in {"R5", "R6", "R7", "R8"}:
        raise ValueError(f"unsupported capability-screen family: {family}")
    accepted = {_normalize(value) for value in accepted_answers}
    if _normalize(prediction) in accepted:
        return True
    if family in {"R5", "R6"}:
        values = re.findall(r"(?<![A-Za-z])[-+]?\d+", prediction)
        return bool(values) and str(int(values[-1])) in accepted
    if family == "R7":
        values = re.findall(r"\b(?:yes|no)\b", prediction, flags=re.IGNORECASE)
        return bool(values) and values[-1].lower() in accepted
    if family == "R8":
        pairs = re.findall(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", prediction)
        return bool(pairs) and f"{int(pairs[-1][0])},{int(pairs[-1][1])}" in accepted
    return False


def _select(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    families = {str(value) for value in selection["families"]}
    seeds = {int(value) for value in selection["seeds"]}
    layout = str(selection["context_layout"])
    depth = float(selection["requested_evidence_depth"])
    selected = [
        row
        for row in rows
        if str(row["family"]) in families
        and int(row["seed"]) in seeds
        and str(row["context_layout"]) == layout
        and abs(float(row["requested_evidence_depth"]) - depth) < 1e-12
    ]
    expected = int(selection["expected_independent_clusters"])
    if len(selected) != expected or len({str(row["cluster_id"]) for row in selected}) != expected:
        raise RuntimeError("capability-screen selection is not the declared independent-cluster set")
    return selected


def run(config_path: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    helper = _module("halo_capability_helpers_for_ladder", LAB_ROOT / "scripts" / "run_halo_capability.py")
    started_at = utc_now()
    config = _load(config_path)
    paths = dict(config["paths"])
    base = REPO_ROOT if paths.get("base") == "repository_root" else LAB_ROOT
    model_path = (base / str(paths["model"])).resolve()
    inference_path = (base / str(paths["inference"])).resolve()
    evaluation_path = (base / str(paths["evaluation"])).resolve()
    output = (base / str(paths["output_root"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty capability-screen run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    expected = dict(config["model"]["expected_hashes"])
    observed = {
        "weights": sha256_file(model_path / "model.safetensors"),
        "tokenizer": sha256_file(model_path / "tokenizer.json"),
        "config": sha256_file(model_path / "config.json"),
    }
    if observed != expected:
        raise RuntimeError("Halo checkpoint hash mismatch")
    inference = _select(_rows(inference_path), dict(config["selection"]))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = model.to("cpu").eval()
    torch.manual_seed(int(config["runtime"]["seed"]))

    unscored: list[dict[str, Any]] = []
    for row in inference:
        ids = tokenizer(str(row["prompt"]), add_special_tokens=False, return_tensors="pt")["input_ids"]
        if helper._token_hash(ids) != row["token_ids_sha256"]:
            raise RuntimeError(f"token hash mismatch for {row['case_id']}")
        prediction, resident, seconds = helper._generate(
            model,
            tokenizer,
            ids,
            chunk_size=int(config["runtime"]["chunk_size"]),
            max_new_tokens=int(config["runtime"]["max_new_tokens"]),
        )
        unscored.append(
            {
                "case_id": row["case_id"],
                "cluster_id": row["cluster_id"],
                "family": row["family"],
                "seed": row["seed"],
                "context_layout": row["context_layout"],
                "requested_evidence_depth": row["requested_evidence_depth"],
                "actual_evidence_depth": row["actual_evidence_depth"],
                "prediction": prediction,
                "input_tokens": int(ids.shape[-1]),
                "prompt_sha256": row["prompt_sha256"],
                "token_ids_sha256": row["token_ids_sha256"],
                "resident_entries": resident,
                "inference_wall_seconds": seconds,
            }
        )
    predictions_path = output / "predictions.unscored.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in unscored),
        encoding="utf-8",
        newline="\n",
    )

    # The answer/evidence view is opened only after every prediction is durable.
    evaluation = {str(row["case_id"]): row for row in _rows(evaluation_path)}
    scored: list[dict[str, Any]] = []
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
                "generator_seed": row["seed"],
                "prompt_sha256": row["prompt_sha256"],
                "context_tokens": row["input_tokens"],
                "evidence_depth": row["actual_evidence_depth"],
                "chunk_size": int(config["runtime"]["chunk_size"]),
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
                "context_layout": row["context_layout"],
                "accepted_answers": key["accepted_answers"],
            }
        )
    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
        newline="\n",
    )
    by_family = {
        family: {
            "clusters": sum(row["task_family"] == family for row in scored),
            "correct": sum(row["task_family"] == family and row["correct"] for row in scored),
        }
        for family in sorted({str(row["task_family"]) for row in scored})
    }
    correct_count = sum(bool(row["correct"]) for row in scored)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "classification": "BOUNDED_FULL_CACHE_CAPABILITY_SCREEN",
        "independent_clusters": len(scored),
        "correct_clusters": correct_count,
        "capability_rate": correct_count / len(scored),
        "by_family": by_family,
        "selection": dict(config["selection"]),
        "compression_policy_evaluated": False,
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
        "experiment": {
            "name": config["experiment"]["name"],
            "compute_tier": "B",
            "architecture_variant": "B0 full cache",
            "benchmark": "halo_r1_r8_ladder_v1:R5-R8 bounded screen",
            "seed": config["runtime"]["seed"],
            "query_conditioned": False,
        },
        "config": {"resolved_path": "configs/halo_r5_r8_capability_screen.yaml", "sha256": sha256_file(config_path)},
        "model": {
            "status": "VERIFIED",
            "id": config["model"]["core_id"],
            "path": "Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze",
            "architecture": "Qwen3ForCausalLM",
            "hashes": {**observed, "chat_template": None},
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "precision": "bfloat16",
            "hardware": None,
        },
        "budget": None,
        "artifacts": {"predictions": "predictions.unscored.jsonl", "cases": "cases.jsonl", "summary": "summary.json"},
        "notes": [
            "Bounded full-cache capability screen only; no compression policy was evaluated.",
            "The answer/evidence view was opened only after all predictions were written.",
        ],
    }
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_r5_r8_capability_screen.yaml",
    )
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
