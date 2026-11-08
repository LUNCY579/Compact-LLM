"""Model-free commands. Even a complete approved-looking request cannot launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
from .contract import (NATIVE_BLOCK, PACKAGE, Refusal, draft, loads, refuse_native,
                       safe_path, validate_request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("draft")
    validate = sub.add_parser("validate-request")
    validate.add_argument("--request", required=True, type=Path)
    launch = sub.add_parser("launch")
    launch.add_argument("--request", type=Path)
    preflight = sub.add_parser("preflight")
    for name in ("source", "partial", "subject", "out"):
        preflight.add_argument("--" + name, required=True, type=Path)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--out", required=True, type=Path)
    verify = sub.add_parser("verify-fixture")
    verify.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "launch":
            refuse_native()  # BEFORE request reads, native imports, journal or child creation.
        if args.command == "status":
            result = {"native_status": NATIVE_BLOCK, "native_execution_permitted": False,
                      "training_permission": False, "confirmation_permission": False,
                      "native_model_or_tokenizer_calls": 0, "scope": "OFFLINE_BOUNDARY_IMPLEMENTATION"}
        elif args.command == "draft":
            result = draft()
        elif args.command == "validate-request":
            result = validate_request(loads(safe_path(args.request).read_text("utf-8")))
        elif args.command == "preflight":
            from .preflight import prepare
            result = prepare(PACKAGE.parents[1], args.source, args.partial, args.subject, args.out)
        elif args.command == "fixture":
            from .adapter import HostAdapter, ScriptedTransport, FixtureTokenizer, Reservations
            from .evidence import write_fixture_bundle
            from .contract import PROBE
            charges, events = [], []
            reservations = Reservations(charges.append)
            reservations.qualification()  # Accounting simulation; NOT six native forwards.
            host = HostAdapter(ScriptedTransport(((0., 5., 0., 0.), (0., 0., 0., 5.))),
                               FixtureTokenizer(), [2, 3], reservations, lambda: None, events.append)
            messages = [{"role": "system", "content": PROBE["system"]},
                        {"role": "user", "content": PROBE["user"]}]
            caps = [64, 288] + [64, 64, 64, 64, 32] * 2
            results = [host.generate(messages, cap) for cap in caps]
            result = write_fixture_bundle(args.out, PACKAGE.parents[1], events, results)
        elif args.command == "verify-fixture":
            from .evidence import verify_bundle
            result = verify_bundle(args.directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Refusal, OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc), "native_execution_permitted": False,
                          "permit_consumed": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
