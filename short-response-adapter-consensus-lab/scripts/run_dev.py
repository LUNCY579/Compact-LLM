from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.direct import run_direct_experiment
from consensus_lab.experiments.config import filter_committees_by_size, load_experiment_config
from consensus_lab.models.committee import ExpertCommittee, resolve_device, resolve_dtype
from consensus_lab.paths import LabPaths
from consensus_lab.run_lock import exclusive_run_lock
from consensus_lab.run_provenance import (
    begin_runtime_invocation,
    build_runtime_identity,
    finish_runtime_invocation,
)


def _jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable DEV experiment; omit --limit for full DEV.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/hard_vote.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--committee-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--max-loaded-vram-gb",
        "--max-vram-gb",
        dest="max_loaded_vram_gb",
        type=float,
        help=(
            "Post-load CUDA allocation guard, not a GPU utilization or hard allocator limit. "
            "The legacy --max-vram-gb spelling remains accepted."
        ),
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=25,
        help="Rebuild summaries/live report every N newly durable rows (always rebuilt at completion).",
    )
    parser.add_argument(
        "--max-new-rows",
        type=int,
        help="Exit cleanly after this many newly durable rows; rerun to resume the same frozen condition.",
    )
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    config_path = args.config.resolve()
    config = load_experiment_config(config_path, paths)
    if args.committee_size is not None:
        config = filter_committees_by_size(config, args.committee_size)
    selected_datasets = args.datasets or list(config.datasets)
    unknown_datasets = sorted(set(selected_datasets) - set(config.datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown selected datasets: {unknown_datasets}")
    resolved_device = resolve_device(args.device)
    resolved_dtype = resolve_dtype(args.precision, resolved_device)
    suffix = f"dev_n{args.limit}" if args.limit else "dev_full"
    result_path = LAB_ROOT / f"results/runs/{config.name}_{suffix}.jsonl"
    runtime_manifest_path = result_path.with_suffix(result_path.suffix + ".runtime.json")
    expected_rows = (
        sum(
            min(args.limit, _jsonl_rows(config.datasets[name]))
            if args.limit is not None
            else _jsonl_rows(config.datasets[name])
            for name in selected_datasets
        )
        * len(config.committees)
        * len(config.methods)
    )
    # Acquire the single-writer lock before provenance hashing or model load.
    # A second accidental launch therefore cannot allocate another committee.
    with exclusive_run_lock(result_path.with_suffix(result_path.suffix + ".lock")):
        identity = build_runtime_identity(
            paths=paths,
            config_path=config_path,
            config=config,
            selected_datasets=selected_datasets,
            limit=args.limit,
            committee_size=args.committee_size,
            requested_device=args.device,
            requested_precision=args.precision,
            resolved_device=str(resolved_device),
            resolved_dtype=str(resolved_dtype),
        )
        invocation_id, _ = begin_runtime_invocation(
            manifest_path=runtime_manifest_path,
            output_path=result_path,
            identity=identity,
            runtime_controls={
                "summary_every": args.summary_every,
                "max_new_rows": args.max_new_rows,
                "max_loaded_vram_gb": args.max_loaded_vram_gb,
            },
        )
        committee = ExpertCommittee.load(
            paths=paths,
            backbone=config.backbone,
            experts=config.experts,
            device=args.device,
            precision=args.precision,
        )
        if args.max_loaded_vram_gb is not None and committee.device.type == "cuda":
            allocated = __import__("torch").cuda.memory_allocated(committee.device) / (1024**3)
            if allocated > args.max_loaded_vram_gb:
                raise RuntimeError(
                    f"Loaded committee uses {allocated:.2f} GiB > "
                    f"--max-loaded-vram-gb={args.max_loaded_vram_gb}"
                )
        rows = run_direct_experiment(
            paths=paths,
            config=config,
            committee=committee,
            output_path=result_path,
            summary_json=LAB_ROOT / f"results/summaries/{config.name}_{suffix}.json",
            summary_csv=LAB_ROOT / f"results/summaries/{config.name}_{suffix}.csv",
            live_report=LAB_ROOT / "reports/LIVE_RESULTS.md",
            dataset_names=selected_datasets,
            limit=args.limit,
            resume=True,
            summary_every=args.summary_every,
            max_new_rows=args.max_new_rows,
            runtime_identity_sha256=identity["identity_sha256"],
        )
        runtime_manifest = finish_runtime_invocation(
            manifest_path=runtime_manifest_path,
            invocation_id=invocation_id,
            output_path=result_path,
            expected_rows=expected_rows,
        )
    print(
        json.dumps(
            {
                "status": (
                    "DEV_COMPLETE"
                    if runtime_manifest["status"] == "COMPLETE"
                    else "DEV_BATCH_RETURNED_CLEANLY"
                ),
                "rows": len(rows),
                "expected_rows": expected_rows,
                "results": str(result_path),
                "runtime_manifest": str(runtime_manifest_path),
                "runtime_identity_sha256": identity["identity_sha256"],
                "max_new_rows": args.max_new_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
