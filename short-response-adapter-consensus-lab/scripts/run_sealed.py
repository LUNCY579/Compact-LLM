from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.direct import run_direct_experiment
from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import read_json, sha256_file, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths
from consensus_lab.run_lock import exclusive_run_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one frozen sealed packet; selection is immutable.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-vram-gb", type=float)
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    manifest_path = LAB_ROOT / "artifacts/frozen_configs/frozen_final_config.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_SEALED":
        raise RuntimeError("Finalist manifest is not frozen")
    config_path = Path(manifest["frozen_config"])
    if sha256_file(config_path) != manifest["frozen_config_sha256"]:
        raise RuntimeError("Frozen config hash mismatch")
    config = load_experiment_config(config_path, paths)
    if config.config_sha256 != manifest["parsed_config_sha256"]:
        raise RuntimeError("Parsed frozen config hash mismatch")
    for name, packet in manifest["sealed_datasets"].items():
        if sha256_file(Path(packet["path"])) != packet["sha256"]:
            raise RuntimeError(f"Frozen sealed dataset hash mismatch: {name}")
    output = paths.assert_output_path(LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl")
    run_manifest_path = paths.assert_output_path(LAB_ROOT / "results/sealed/sealed_run_manifest.json")
    if not run_manifest_path.exists():
        write_json_atomic(
            run_manifest_path,
            {
                "schema_version": "1.0",
                "status": "SEALED_RUN_STARTED",
                "frozen_manifest": str(manifest_path),
                "frozen_manifest_sha256": sha256_file(manifest_path),
                "limit": int(manifest["sealed_limit"]),
                "config_sha256": config.config_sha256,
                "output": str(output),
            },
        )
    run_manifest = read_json(run_manifest_path)
    if int(run_manifest["limit"]) != int(manifest["sealed_limit"]):
        raise RuntimeError("Sealed limit changed after run start")
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=config.experts,
        device=args.device,
        precision=args.precision,
    )
    if args.max_vram_gb is not None and committee.device.type == "cuda":
        import torch

        allocated = torch.cuda.memory_allocated(committee.device) / (1024**3)
        if allocated > args.max_vram_gb:
            raise RuntimeError(f"Loaded committee uses {allocated:.2f} GiB > {args.max_vram_gb}")
    with exclusive_run_lock(output.with_suffix(output.suffix + ".lock")):
        rows = run_direct_experiment(
            paths=paths,
            config=config,
            committee=committee,
            output_path=output,
            summary_json=LAB_ROOT / "results/summaries/v22_frozen_finalists_sealed.json",
            summary_csv=LAB_ROOT / "results/summaries/v22_frozen_finalists_sealed.csv",
            live_report=LAB_ROOT / "reports/LIVE_RESULTS.md",
            dataset_names=["v22_sealed"],
            limit=int(manifest["sealed_limit"]),
            resume=True,
        )
    run_manifest["status"] = "SEALED_RUN_COMPLETE"
    run_manifest["rows"] = len(rows)
    write_json_atomic(run_manifest_path, run_manifest)
    print(json.dumps({"status": "SEALED_RUN_COMPLETE", "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
