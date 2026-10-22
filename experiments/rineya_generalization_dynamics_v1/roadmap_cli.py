"""Offline quality-card, admission and owner-packet commands; never model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import Blocked, read, require
from .protected_paths import reject_protected_path
from .roadmap import prepare_roadmap, verify_roadmap_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    card = commands.add_parser("check-card")
    card.add_argument("--input", type=Path, required=True)
    data = commands.add_parser("check-data")
    data.add_argument("--packet", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-card":
            from .quality_card import validate_quality_card
            path = reject_protected_path(args.input, "Quality card input", include_instruments=True)
            require(args.input.is_absolute() and path.is_file() and not args.input.is_symlink()
                    and path.suffix.lower() == ".json" and path.stat().st_size <= 2 * 1024 * 1024,
                    "Quality card input needs a bounded absolute regular JSON metadata file")
            result = validate_quality_card(read(path))
        elif args.command == "check-data":
            from .data_admission import assess_data_admission
            result = assess_data_admission(str(args.packet))
        elif args.command == "prepare":
            result = prepare_roadmap(args.spec, args.output)
        else:
            result = verify_roadmap_packet(args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("status", "").startswith("BLOCKED") else 0
    except (Blocked, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
