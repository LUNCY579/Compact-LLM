"""Run `python -m idr --help` from the experiment directory."""
import argparse
from pathlib import Path
import sys
from .artifacts import load_json, write_json, freeze, verify_freeze, tree_manifest, sha256_file


def main(argv=None):
    p = argparse.ArgumentParser(description="IDR-001 guarded routing research")
    commands = p.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="8 routing arms, 15 tiny training arms, 5 controller traces")
    demo.add_argument("--out", required=True)
    for command in ("calibrate", "replay", "execute-native"):
        c = commands.add_parser(command)
        c.add_argument("--spec", required=True)
        c.add_argument("--rows", required=True, help="JSON list, one split only")
        c.add_argument("--out", required=True)
        if command != "calibrate":
            c.add_argument("--state", required=True)
            c.add_argument("--split", choices=["dev", "confirm"], default="dev")
            c.add_argument("--freeze")
        if command == "replay":
            c.add_argument("--evidence", choices=["synthetic", "native_replay"], default="synthetic")
        if command == "execute-native":
            c.add_argument("--config", required=True)
    c = commands.add_parser("compare")
    c.add_argument("--candidate", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--objective", choices=["quality", "efficiency"], required=True)
    c.add_argument("--out", required=True)
    c = commands.add_parser("freeze")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c = commands.add_parser("verify")
    c.add_argument("--freeze", required=True)
    c = commands.add_parser("manifest")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c = commands.add_parser("preflight")
    c.add_argument("--config", required=True)
    c.add_argument("--out", required=True)
    c = commands.add_parser("train-native")
    c.add_argument("--config", required=True)
    c.add_argument("--rows", required=True)
    c.add_argument("--arm", choices=["none", "sign", "integral", "id", "auxiliary"], required=True)
    c.add_argument("--steps", type=int, default=20)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c = commands.add_parser("generate-ledger")
    c.add_argument("--split", choices=["train", "calibration", "dev", "confirm"], required=True)
    c.add_argument("--parents", type=int, default=40)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c = commands.add_parser("capture-native")
    c.add_argument("--config", required=True)
    c.add_argument("--spec", required=True)
    c.add_argument("--rows", required=True)
    c.add_argument("--out", required=True)
    for command in ("fit-quality", "predict-quality"):
        c = commands.add_parser(command)
        c.add_argument("--spec", required=True)
        c.add_argument("--rows", required=True)
        c.add_argument("--out", required=True)
        if command == "predict-quality":
            c.add_argument("--head", required=True)
            c.add_argument("--head-sha256", required=True)
    args = p.parse_args(argv)
    try:
        if hasattr(args, "out") and Path(args.out).exists():
            raise FileExistsError("output already exists; refusing to execute or overwrite")
        if args.command == "demo":
            import torch
            from .experiment import routing_demo, tiny_moe, paper_demo
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=False)
            torch.set_num_threads(1)
            routing = routing_demo(out)
            tiny = [tiny_moe(method, auxiliary=method == "auxiliary", seed=seed)
                    for seed in range(3) for method in ("none", "sign", "integral", "id", "auxiliary")]
            write_json(out / "tiny_moe.json", tiny)
            write_json(out / "paper_control.json", paper_demo())
            write_json(out / "summary.json", {"routing": routing, "tiny_optimizer_runs": len(tiny),
                                              "native_executed": False})
        elif args.command in {"calibrate", "replay", "execute-native"}:
            from .experiment import calibrate, evaluate
            required = [args.spec, args.rows]
            if args.command != "calibrate":
                required.append(args.state)
                if args.command == "execute-native":
                    required.append(args.config)
                if args.split == "confirm":
                    if not args.freeze:
                        raise ValueError("confirmation requires a frozen inputs manifest")
                    frozen = verify_freeze(args.freeze)
                    if any(str(Path(path).resolve()) not in frozen["files"] for path in required):
                        raise ValueError("confirmation manifest does not bind every input")
                    write_json(str(args.freeze) + ".consumed.json",
                               {"freeze_sha256": sha256_file(args.freeze), "output": str(Path(args.out).resolve()),
                                "status": "claimed_before_opening_confirmation_data"})
            spec, rows = load_json(args.spec), load_json(args.rows)
            if args.command == "calibrate":
                report = calibrate(spec, rows)
            else:
                backend = None
                if args.command == "execute-native":
                    from .native import LocalAdapterBackend
                    config = load_json(args.config)
                    # Prevent fixture identities or wrong adapters from being dispatched.
                    for expert in spec["experts"]:
                        artifact = config["base"] if expert["name"] == "core" else config["adapters"][expert["name"]]
                        if expert["artifact_sha256"] != artifact["manifest"]["tree_sha256"]:
                            raise ValueError("registry/backend artifact identity mismatch")
                    backend = LocalAdapterBackend(config)
                report = evaluate(spec, rows, load_json(args.state), args.split, backend,
                                  getattr(args, "evidence", "synthetic"))
                if backend is not None:
                    report["model_load_seconds"] = backend.load_seconds
            write_json(args.out, report)
        elif args.command == "compare":
            from .evaluation import compare
            write_json(args.out, compare(load_json(args.candidate), load_json(args.baseline), args.objective))
        elif args.command == "freeze":
            freeze(args.inputs, args.out)
        elif args.command == "verify":
            verify_freeze(args.freeze)
            print("Frozen input hashes verified.")
        elif args.command == "manifest":
            write_json(args.out, tree_manifest(args.root))
        elif args.command == "preflight":
            from .native import preflight
            report = preflight(load_json(args.config))
            write_json(args.out, report)
            return 0 if report["ready"] else 2
        elif args.command == "generate-ledger":
            from .worlds import ledger_rows
            write_json(args.out, ledger_rows(args.split, args.parents, args.seed))
        elif args.command == "capture-native":
            from .native import LocalAdapterBackend
            from .dataflow import capture
            spec, config = load_json(args.spec), load_json(args.config)
            for expert in spec["experts"]:
                artifact = config["base"] if expert["name"] == "core" else config["adapters"][expert["name"]]
                if expert["artifact_sha256"] != artifact["manifest"]["tree_sha256"]:
                    raise ValueError("registry/backend artifact identity mismatch")
            rows = load_json(args.rows)
            if any(r["split"] == "confirm" for r in rows):
                raise ValueError("cannot capture confirmation outcomes for counterfactual tuning")
            write_json(args.out, capture(spec, rows, LocalAdapterBackend(config)))
        elif args.command == "fit-quality":
            from .dataflow import save_quality
            report = save_quality(load_json(args.spec), load_json(args.rows), args.out)
            write_json(args.out + ".json", report)
        elif args.command == "predict-quality":
            from .dataflow import predict_quality
            write_json(args.out, predict_quality(load_json(args.spec), load_json(args.rows),
                                                args.head, args.head_sha256))
        elif args.command == "train-native":
            from .native import train_native
            from .moe import MoEConfig
            from .control import ControlConfig
            train_native(load_json(args.config), load_json(args.rows), args.out,
                         MoEConfig(auxiliary_weight=0.02 if args.arm == "auxiliary" else 0),
                         ControlConfig(method="none" if args.arm == "auxiliary" else args.arm),
                         steps=args.steps, seed=args.seed)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"IDR-001 stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
