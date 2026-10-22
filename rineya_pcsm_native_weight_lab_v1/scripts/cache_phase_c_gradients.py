#!/usr/bin/env python3
"""Cache real Nimbus calibration gradients without taking an optimizer step."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from rineya_pcsm.data import stream_selected_sft_rows
from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, environment_manifest, utc_now
from rineya_pcsm.phase_c import (
    aggregate_target_gradient,
    encode_row,
    load_native_model,
    resolve_parameter,
    save_gradient_cache,
)
from rineya_pcsm.randomness import seed_everything


def rows_by_id(corpus_root: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    shards = list((corpus_root / "data" / "sft" / "train").glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError("No train SFT shards found")
    return {
        str(row["id"]): row for row in stream_selected_sft_rows(shards, selected_ids)
    }


def assert_unchanged(before: torch.Tensor, after: torch.Tensor) -> None:
    if not torch.equal(before, after):
        raise RuntimeError(
            "Calibration changed native weights without an optimizer step"
        )


def existing_cache_manifest(
    cache: Path,
    *,
    section: str,
    model_path: Path,
    target_name: str,
    expected_row_ids: list[str],
) -> dict[str, Any] | None:
    manifest_path = cache / f"{section}.json"
    tensor_path = cache / f"{section}.safetensors"
    if not manifest_path.exists() and not tensor_path.exists():
        return None
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise RuntimeError(f"Partial existing gradient cache: {section}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("cache_sha256")
    payload = dict(manifest)
    payload.pop("cache_sha256", None)
    if declared != canonical_json_hash(payload):
        raise RuntimeError(f"Existing cache manifest self-hash mismatch: {section}")
    if manifest.get("name") != section or manifest.get("target") != target_name:
        raise RuntimeError(f"Existing cache identity mismatch: {section}")
    if manifest.get("model", {}).get("weight_sha256") != sha256_file(
        model_path / "model.safetensors"
    ) or manifest.get("model", {}).get("config_sha256") != sha256_file(
        model_path / "config.json"
    ):
        raise RuntimeError(f"Existing cache base mismatch: {section}")
    if sha256_file(tensor_path) != manifest.get("tensor", {}).get("sha256"):
        raise RuntimeError(f"Existing cache tensor hash mismatch: {section}")
    recorded_rows = [row["row_id"] for row in manifest["report"]["examples"]]
    if recorded_rows != expected_row_ids:
        raise RuntimeError(f"Existing cache row IDs mismatch: {section}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--maximum-tokens", type=int, default=40960)
    args = parser.parse_args()

    budget = json.loads(args.budget_manifest.read_text(encoding="utf-8"))
    expected_weight = budget["base_model"]["weight_sha256"]
    actual_weight = sha256_file(args.model / "model.safetensors")
    if expected_weight != actual_weight:
        raise RuntimeError("Base weight hash differs from frozen budget manifest")
    target_name = str(budget["target_parameter"])
    selected = {
        section: [str(row["row_id"]) for row in budget["calibration"][section]]
        for section in ("meta_train", "mask_cal")
    }
    selected_ids = set(selected["meta_train"]) | set(selected["mask_cal"])
    source_rows = rows_by_id(args.corpus_root.resolve(), selected_ids)
    for row_id in selected_ids:
        expected = next(
            row["content_sha256"]
            for section in budget["calibration"].values()
            for row in section
            if row["row_id"] == row_id
        )
        if canonical_json_hash(source_rows[row_id]) != expected:
            raise RuntimeError(f"Frozen SFT content hash mismatch: {row_id}")

    device = torch.device(args.device)
    if device.type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 6 * 1024**3:
            free_gib = free_bytes / 1024**3
            raise RuntimeError(
                f"CUDA pilot requires 6 GiB free by policy; found {free_gib:.2f}"
            )
    seed_everything(17)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "started_at_utc": utc_now(),
        "status": "running",
        "device": str(device),
        "target": target_name,
        "base_weight_sha256": actual_weight,
        "budget_manifest_sha256": sha256_file(args.budget_manifest),
        "environment": environment_manifest(),
    }
    atomic_write_json(args.result, summary)
    try:
        cache_manifests: dict[str, Any] = {}
        existing = {
            section: existing_cache_manifest(
                args.cache,
                section=section,
                model_path=args.model,
                target_name=target_name,
                expected_row_ids=selected[section],
            )
            for section in ("meta_train", "mask_cal")
        }
        pending_sections = [
            section for section, value in existing.items() if value is None
        ]
        model, tokenizer = load_native_model(args.model, device=device)
        target = resolve_parameter(model, target_name)
        before = target.detach().cpu().clone()
        for section in ("meta_train", "mask_cal"):
            if existing[section] is not None:
                cache_manifests[section] = existing[section]
                continue
            examples = [
                encode_row(tokenizer, source_rows[row_id], args.maximum_tokens)
                for row_id in selected[section]
            ]
            gradient, report = aggregate_target_gradient(
                model, target_name, examples, device=device
            )
            cache_manifests[section] = save_gradient_cache(
                args.cache,
                name=section,
                gradient=gradient,
                report=report,
                model_path=args.model,
                target_name=target_name,
            )
        assert_unchanged(before, target.detach().cpu())
        summary.update(
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "native_weight_delta": "bitwise_zero",
                "resumed_sections": [
                    section for section, value in existing.items() if value is not None
                ],
                "computed_sections": pending_sections,
                "cache": {
                    name: {
                        "manifest_sha256": value["cache_sha256"],
                        "tensor_sha256": value["tensor"]["sha256"],
                        "report": value["report"],
                    }
                    for name, value in cache_manifests.items()
                },
                "sealed_test_read": False,
            }
        )
    except Exception as error:
        summary.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "sealed_test_read": False,
            }
        )
        atomic_write_json(args.result, summary)
        raise
    summary["summary_sha256"] = canonical_json_hash(summary)
    atomic_write_json(args.result, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
