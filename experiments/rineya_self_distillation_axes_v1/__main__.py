"""All output paths are create-only. Plans and synthetic commands never load models."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .controlled import CONTROLLED_RATES, ControlledTask, Geometry
from .data import synthetic_rows
from .plan import Arm, arms, checkpoint_manifest, make_plan, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="DSD-001 independent self-distillation axes")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "synthetic-data", "controlled", "manifest", "qualify", "train"):
        sub = subs.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        if command == "controlled":
            sub.add_argument("--steps", type=int, default=20)
            sub.add_argument("--seeds", type=int, nargs="+", default=[17])
            sub.add_argument("--rates", type=float, nargs="+")
            sub.add_argument("--paper-grid", action="store_true", help="Appendix F eighteen-rate grid, not the LLM five-rate grid")
            sub.add_argument("--concentration", type=float, default=2.5)
            sub.add_argument("--context-strength", type=float, default=.6)
            sub.add_argument("--feature-overlap", type=float, default=.5)
            sub.add_argument("--readout-compatibility", type=float, default=0.)
        if command == "synthetic-data":
            sub.add_argument("--seed", type=int, default=17)
            sub.add_argument("--parents-per-split", type=int, default=12)
        if command == "manifest":
            sub.add_argument("--checkpoint", type=Path, required=True)
        if command in {"qualify", "train"}:
            sub.add_argument("--config", type=Path, required=True)
        if command == "train":
            sub.add_argument("--qualification", type=Path, required=True)
            sub.add_argument("--arm", required=True, choices=[a.name for a in arms((0., .02, .05, .1, .25, 1.))])
            sub.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = make_plan()
    elif args.command == "manifest":
        result = checkpoint_manifest(args.checkpoint)
    elif args.command == "synthetic-data":
        rows = synthetic_rows(args.seed, args.parents_per_split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"status": "SYNTHETIC_DATA_ONLY", "rows": len(rows)}))
        return 0
    elif args.command == "controlled":
        if args.paper_grid and args.rates is not None:
            raise ValueError("Choose --paper-grid or --rates, not both")
        rates = CONTROLLED_RATES if args.paper_grid else (args.rates or (0., .0025, .02))
        if not 1 <= len(args.seeds) <= 16 or len(set(args.seeds)) != len(args.seeds) or len(rates) > 18:
            raise ValueError("CLI sweep exceeds bounded seed/rate budget")
        result = {"status": "CONTROLLED_MODEL_ONLY", "native_rineya_executed": False, "runs": []}
        for seed in args.seeds:
            task = ControlledTask(Geometry(concentration=args.concentration, context_strength=args.context_strength,
                                           feature_overlap=args.feature_overlap, readout_compatibility=args.readout_compatibility), seed)
            for arm in arms(tuple(rates), False):
                result["runs"].append(task.run(arm.rollout_source, arm.kl_direction, arm.ema_rate, args.steps))
    else:
        from .native import execute
        config = json.loads(args.config.read_text())
        if args.command == "train":
            arm = next(a for a in arms((0., .02, .05, .1, .25, 1.)) if a.name == args.arm)
            qualification = json.loads(args.qualification.read_text())
            result = execute(config, args.output, "train", arm, qualification, args.execute_training)
        else:
            result = execute(config, args.output)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0
    write_json(args.output, result)
    print(json.dumps({"status": result.get("status", "WRITTEN"), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
