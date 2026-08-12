from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.direct import run_direct_experiment
from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths
from consensus_lab.run_lock import exclusive_run_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded, non-research-ready DEV smoke experiment.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/hard_vote.yaml")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--committee-size", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-vram-gb", type=float)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(args.config, paths)
    if args.committee_size is not None:
        object.__setattr__(config, "committees", [c for c in config.committees if len(c.experts) == args.committee_size])
        if not config.committees:
            raise ValueError(f"No configured committee has K={args.committee_size}")
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=config.experts,
        device=args.device,
        precision=args.precision,
    )
    if args.max_vram_gb is not None and committee.device.type == "cuda":
        allocated = __import__("torch").cuda.memory_allocated(committee.device) / (1024**3)
        if allocated > args.max_vram_gb:
            raise RuntimeError(f"Loaded committee uses {allocated:.2f} GiB > --max-vram-gb={args.max_vram_gb}")
    result_path = LAB_ROOT / f"results/runs/{config.name}_smoke.jsonl"
    with exclusive_run_lock(result_path.with_suffix(result_path.suffix + ".lock")):
        rows = run_direct_experiment(
            paths=paths,
            config=config,
            committee=committee,
            output_path=result_path,
            summary_json=LAB_ROOT / f"results/summaries/{config.name}_smoke.json",
            summary_csv=LAB_ROOT / f"results/summaries/{config.name}_smoke.csv",
            live_report=LAB_ROOT / "reports/LIVE_RESULTS.md",
            dataset_names=args.datasets,
            limit=args.limit,
            resume=not args.no_resume,
        )
    print(json.dumps({"status": "SMOKE_COMPLETE", "rows": len(rows), "results": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
