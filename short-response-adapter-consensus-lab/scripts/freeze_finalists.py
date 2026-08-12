from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import sha256_file, write_json_atomic
from consensus_lab.paths import LabPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze DEV-selected configs before any lab sealed run.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/sealed_finalists.yaml")
    parser.add_argument("--sealed-limit", type=int, default=32)
    args = parser.parse_args()
    if args.sealed_limit <= 0:
        raise ValueError("sealed-limit must be positive")
    paths = LabPaths.discover(LAB_ROOT)
    if any((LAB_ROOT / "results/sealed").glob("*.jsonl")):
        raise RuntimeError("Sealed output already exists; finalist configuration cannot be refrozen")
    config = load_experiment_config(args.config, paths)
    frozen_dir = paths.assert_output_path(LAB_ROOT / "artifacts/frozen_configs")
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = frozen_dir / "sealed_finalists.yaml"
    if frozen_config.exists():
        raise FileExistsError(f"Frozen config already exists: {frozen_config}")
    shutil.copy2(args.config, frozen_config)
    dataset_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)} for name, path in config.datasets.items()
    }
    source_manifest = LAB_ROOT / "artifacts/manifests/official_sources_before.json"
    manifest = {
        "schema_version": "1.0",
        "status": "FROZEN_BEFORE_SEALED",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(args.config.resolve()),
        "frozen_config": str(frozen_config),
        "frozen_config_sha256": sha256_file(frozen_config),
        "parsed_config_sha256": config.config_sha256,
        "sealed_limit": args.sealed_limit,
        "sealed_sample_seed": config.seed,
        "sealed_datasets": dataset_hashes,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "selection_basis": {
            "primary": "full_k7 hard plurality: best balanced capability/brevity DEV pilot committee",
            "soft_comparator": "full_k7 probability mean tied primary on the 8-case paired fusion pilot",
            "controls": [
                "frozen backbone",
                "Track A: official best v2.2 multidomain individual",
                "abstention specialist: best K=1 on balanced 16-case lab pilot",
            ],
            "excluded_from_sealed": [
                "weighted/logit/confidence fusion: no DEV advantage",
                "sequence medoid: lower DEV pilot correctness",
                "EOS ablations: no DEV advantage",
                "mixed PEFT and random-seed diversity: no matched trained adapters",
                "adaptive consultation: retained as separate DEV extension, not core finalist",
            ],
        },
        "post_sealed_rule": "No candidate, threshold, method, or committee changes based on sealed outcomes.",
    }
    manifest_path = frozen_dir / "frozen_final_config.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
