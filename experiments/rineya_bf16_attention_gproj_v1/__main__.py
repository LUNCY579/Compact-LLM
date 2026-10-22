"""Run from repository root: python -m experiments.rineya_bf16_attention_gproj_v1."""
import argparse
import json
from pathlib import Path

from .audit import audit_case, synthetic_report
from .capture import load_capture
from .protocol import checkpoint_manifest, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BAG-001: bounded BF16 attention correctness experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("synthetic", help="CPU witness, offset sweep and arithmetic audit; no checkpoint")
    smoke.add_argument("--output", type=Path, required=True)
    manifest = commands.add_parser("manifest", help="Hash an exact local safetensors checkpoint")
    manifest.add_argument("--checkpoint", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit", help="Replay an existing hash-verified tensor capture")
    audit.add_argument("--capture", type=Path, required=True)
    audit.add_argument("--backend", choices=["math", "flash", "efficient", "cudnn"], default="math")
    audit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    audit.add_argument("--output", type=Path, required=True)
    for name in ("qualify", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "train":
            sub.add_argument("--qualification", type=Path, required=True)
            sub.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = synthetic_report()
        write_json(args.output, result)
    elif args.command == "manifest":
        result = checkpoint_manifest(args.checkpoint)
        write_json(args.output, result)
    elif args.command == "audit":
        result = audit_case(load_capture(args.capture, args.device), args.backend)
        write_json(args.output, result)
    else:
        from .runner import matched_train, qualify
        config = json.loads(args.config.read_text())
        if args.command == "qualify":
            result = qualify(config, args.output)
        else:
            qualification = json.loads(args.qualification.read_text())
            result = matched_train(config, qualification, args.output, args.execute_training)
    print(json.dumps({"command": args.command, "output": str(args.output),
                      "status": result.get("status", "WRITTEN")}, sort_keys=True))
    return 2 if result.get("status") == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
