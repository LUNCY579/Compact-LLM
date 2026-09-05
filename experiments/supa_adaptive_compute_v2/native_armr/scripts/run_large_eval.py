"""Run one exclusive, deterministic Track A arm/shard using shared machinery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROGRAM / "native_armr/configs/protocol.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=["architecture_dev", "confirmatory_dev", "confirmatory_diagnostics"], default="confirmatory_dev")
    parser.add_argument("--arm", choices=["A0", "A1", "A3", "A16", "AM"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--preflight", action="store_true", help="validate registration and inputs without model execution or writes")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from native_armr.src.operators import operator_factory
    from native_armr.src.protocol import DEFAULT_CONFIG, load_protocol, preflight
    if args.config.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("the shared runner requires the canonical native configs/protocol.json")
    config = load_protocol(args.config)
    count = config["default_shard_count"] if args.shard_count is None else args.shard_count
    if args.preflight:
        print(json.dumps(preflight(args.config, args.manifest, args.role, args.arm,
                                   args.shard_index, count), indent=2))
        return 0
    from shared.evaluation import run_shard
    # Source/config registration is verified by run_shard, not bypassed here.
    path = run_shard(track="native_armr", arm=args.arm, config=config,
                     dataset_manifest=args.manifest, role=args.role,
                     shard_index=args.shard_index, shard_count=count,
                     operator_factory=operator_factory, model_path=args.model_path)
    print(f"RUN_DIR={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
