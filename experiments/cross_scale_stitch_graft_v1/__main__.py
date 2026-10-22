"""Offline commands only. See docs/RUNBOOK.md for authority and limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import fit_bridge
from .artifacts import load_capture, new_directory, save_bridge, strict_json, write_json, preflight_fit_request
from .contracts import Identity, require
from .evaluation import Outcome, Panel, paired_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("verify-source")
    for name in ("demo", "qualify", "fit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True, type=Path)
        if name in {"fit", "score"}:
            p.add_argument("--request", required=True, type=Path)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        from .planning import plan
        result = plan()
    elif args.command == "verify-source":
        from .qualification import verify_source
        result = verify_source()
    elif args.command == "qualify":
        from .qualification import qualify
        result = qualify(args.output)
    elif args.command == "_worker":
        from .qualification import worker
        worker(args.path)
        return
    elif args.command == "demo":
        from .demo import run
        result = run(args.output)
    elif args.command == "fit":
        spec = strict_json(args.request)
        require(set(spec) == {"source", "target", "check_source", "check_target"}, "unexpected fitting request fields")
        resource_preflight = preflight_fit_request(args.request, spec)
        def capture(name):
            item = spec[name]
            require(set(item) == {"directory", "manifest_sha256"}, "invalid capture reference")
            return load_capture(args.request.parent / item["directory"], item["manifest_sha256"])
        source, target, cs, ct = [capture(n) for n in ("source", "target", "check_source", "check_target")]
        fit, report = fit_bridge(source, target, cs, ct)
        report["resource_preflight"] = resource_preflight
        manifest_sha = save_bridge(fit, report, source, target, args.output)
        result = {"manifest_sha256": manifest_sha, "report": report, "native_qualification": False}
    else:
        spec = strict_json(args.request)
        require(set(spec) == {"receiver", "candidate"}, "unexpected scoring request fields")
        def panel(name):
            item = spec[name]
            return Panel(model=Identity(**item["model"]), registration_sha256=item["registration_sha256"],
                         scorer_sha256=item["scorer_sha256"], decoding_sha256=item["decoding_sha256"],
                         role=item["role"], expected_ids=tuple(item["expected_ids"]),
                         outcomes=tuple(Outcome(**row) for row in item["outcomes"]))
        result = paired_report(panel("receiver"), panel("candidate"))
        write_json(new_directory(args.output) / "SCORE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
